#!/usr/bin/env python3
"""Two-Tower cross-architecture check (manuscript Table 8): retraining and evaluation.

The model, training and evaluation code is copied from the notebook
03_實驗與結果_experiments_results/05_第二模型驗證/source_programs/P15_fashionclip_text_conditioned_retrieval_baseline.ipynb
(cells 2-13): CP binary compatibility pretraining, then OR retrieval
fine-tuning that reuses the CP item and text encoders, 100 epochs each, the
checkpoint with the best validation FITB accuracy, 5 seeds x 2 text conditions
(original text, context-aware description). Only these parts differ from P15:
paths are options, the outputs go to a new folder (P15 deleted earlier outputs
in place), epoch times use a monotonic clock, and progress bars are off.

Modes:
- --evaluate-preserved: evaluate the 20 preserved Two-Tower checkpoints
  (02_模型訓練和驗證_model_training_validation/second_model_two_tower/models/)
  with this code and compare with the archived seed summary A30. This checks
  the port and the inputs; it runs on the CPU in a few minutes.
- --train: retrain the 10 models (GPU, about 1.5-3 hours) and write the A30-A32
  tables, Table 8 with paired 95% confidence intervals, and a comparison with
  the archived 2025-2026 values and the manuscript.
- --smoke: the --train code path on a small subset with one epoch (CPU).
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import pickle
import random
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import stats
from torch.utils.data import DataLoader, Dataset

from _ext import FEATURES, TWO_TOWER_DIR, TWO_TOWER_MODELS, local_path, output_dir, sha256, write_text

try:
    from sklearn.metrics import roc_auc_score as sklearn_roc_auc_score
except Exception:
    sklearn_roc_auc_score = None


def tqdm(iterable=None, **kwargs):  # progress bars off
    return iterable


ARCHIVED_A30 = TWO_TOWER_DIR / "圖表_figures_tables" / "tables" / "A30_second_model_two_tower_seed_summary.csv"
TABLE8_METRICS = ("cp_test_auc", "cp_test_fitb_acc", "recall@10", "recall@30", "recall@50", "mean_rank", "median_rank")
MANUSCRIPT_TABLE8_DIFF = {"cp_test_auc": "+0.0378", "cp_test_fitb_acc": "-0.0077", "recall@10": "+0.0102",
                          "recall@30": "+0.0153", "recall@50": "+0.0170", "mean_rank": "-28.89",
                          "median_rank": "-31.4"}

# ---------------------------------------------------------------- P15 cell 2 (settings)
SEEDS = [1, 2, 3, 4, 5]
TEXT_VARIANTS = {
    'original_text': 'encoded_outfitUrlTitle_en_fashionClip.pkl',
    'context_aware_description': 'encoded_NewoutfitUrlTitle_en_fashionClip.pkl',
}
MAX_ITEM_LEN = 16
ITEM_DIM = 1024       # FashionCLIP image 512 + item text 512
TEXT_DIM = 512        # outfit-level FashionCLIP text embedding
CATEGORY_DIM = 512    # category text embedding
CP_EPOCHS = 100
OR_EPOCHS = 100
BATCH_SIZE = 256
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
DROPOUT = 0.15
HIDDEN_DIM = 512
EMBED_DIM = 256
NUM_WORKERS = 0
MIN_CANDIDATES_PER_FG = 3000
K_LIST = [1, 3, 5, 10, 30, 50]
FORCE_RETRAIN = True
TRAIN_LIMIT = None
VALID_LIMIT = None
TEST_LIMIT = None
DEVICE = 'cpu'
PIN_MEMORY = False

# Set by load_data(); the module-level names of the notebook.
FASHIONCLIP_DIR = FEATURES
DISJOINT_DIR: Path
MODEL_ROOT: Path
img_feats: dict
item_text_feats: dict
category_feats: dict
metadata: dict
split_outfits: dict
fitb_raw: dict
ref_to_item_by_split: dict
compat_examples_by_split: dict
fitb_examples_by_split: dict
candidate_pools: dict


# ---------------------------------------------------------------- P15 cell 3
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_json(path: Path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def load_pickle(path: Path):
    with open(path, 'rb') as f:
        return pickle.load(f)


def as_key(x) -> str:
    return str(x)


def vector_to_float32(x) -> np.ndarray:
    return np.asarray(x, dtype=np.float32)


def write_csv(path: Path, rows: List[dict], fieldnames: Optional[List[str]] = None):
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = sorted({key for row in rows for key in row.keys()}) if rows else []
    with open(path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def append_csv(path: Path, row: dict, fieldnames: List[str]):
    path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = path.exists()
    with open(path, 'a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def mean_std_text(values: List[float], digits: int = 4) -> str:
    arr = np.asarray(values, dtype=np.float64)
    if len(arr) == 0:
        return ''
    if len(arr) == 1:
        return f'{arr.mean():.{digits}f} ± 0.0000'
    return f'{arr.mean():.{digits}f} ± {arr.std(ddof=1):.{digits}f}'


def binary_auc(labels, scores) -> float:
    labels = np.asarray(labels, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    if sklearn_roc_auc_score is not None:
        return float(sklearn_roc_auc_score(labels, scores))

    order = np.argsort(scores)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(scores) + 1)
    pos = labels == 1
    n_pos = pos.sum()
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float('nan')
    rank_sum_pos = ranks[pos].sum()
    auc = (rank_sum_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
    return float(auc)


# ---------------------------------------------------------------- P15 cell 5 (functions)
def build_ref_to_item(outfits: List[dict]) -> Dict[str, str]:
    ref_to_item = {}
    for outfit in outfits:
        set_id = as_key(outfit['set_id'])
        for item in outfit['items']:
            ref_to_item[f'{set_id}_{item["index"]}'] = as_key(item['item_id'])
    return ref_to_item


def parse_compatibility_examples(split: str, limit: Optional[int] = None) -> List[dict]:
    path = DISJOINT_DIR / f'compatibility_{split}.txt'
    ref_to_item = ref_to_item_by_split[split]
    examples = []
    skipped = 0
    with open(path, 'r', encoding='utf-8') as f:
        for line_idx, line in enumerate(f):
            parts = line.strip().split()
            if len(parts) < 2:
                skipped += 1
                continue
            label = int(parts[0])
            refs = parts[1:]
            set_id = as_key(refs[0].split('_')[0])
            try:
                item_ids = [ref_to_item[ref] for ref in refs]
            except KeyError:
                skipped += 1
                continue
            examples.append({
                'question_index': line_idx,
                'set_id': set_id,
                'item_ids': item_ids,
                'label': label,
            })
            if limit is not None and len(examples) >= limit:
                break
    print(f'compatibility {split}: parsed={len(examples):,}, skipped={skipped:,}')
    return examples


def parse_fitb_examples(split: str, limit: Optional[int] = None) -> List[dict]:
    ref_to_item = ref_to_item_by_split[split]
    examples = []
    skipped = 0
    for q_idx, row in enumerate(fitb_raw[split]):
        question_refs = row['question']
        answer_refs = row['answers']
        if not question_refs or not answer_refs:
            skipped += 1
            continue
        set_id = as_key(question_refs[0].split('_')[0])
        try:
            partial_ids = [ref_to_item[ref] for ref in question_refs]
            answer_ids = [ref_to_item[ref] for ref in answer_refs]
        except KeyError:
            skipped += 1
            continue
        correct_indices = [i for i, ref in enumerate(answer_refs) if as_key(ref.split('_')[0]) == set_id]
        if len(correct_indices) != 1:
            skipped += 1
            continue
        correct_idx = correct_indices[0]
        target_item_id = answer_ids[correct_idx]
        if target_item_id not in metadata:
            skipped += 1
            continue
        examples.append({
            'question_index': q_idx,
            'set_id': set_id,
            'blank_position': int(row.get('blank_position', -1)),
            'partial_item_ids': partial_ids,
            'answer_item_ids': answer_ids,
            'correct_index': correct_idx,
            'target_item_id': target_item_id,
            'target_fg': as_key(metadata[target_item_id]['category_id']),
            'target_semantic_category': as_key(metadata[target_item_id].get('semantic_category', '')),
        })
        if limit is not None and len(examples) >= limit:
            break
    print(f'FITB {split}: parsed={len(examples):,}, skipped={skipped:,}')
    return examples


# ---------------------------------------------------------------- P15 cell 6
_item_vec_cache: Dict[str, np.ndarray] = {}


def get_item_vec(item_id: str) -> np.ndarray:
    item_id = as_key(item_id)
    if item_id not in _item_vec_cache:
        img = vector_to_float32(img_feats[item_id])
        txt = vector_to_float32(item_text_feats[item_id])
        _item_vec_cache[item_id] = np.concatenate([img, txt]).astype(np.float32)
    return _item_vec_cache[item_id]


def get_category_vec(item_id: str) -> np.ndarray:
    return vector_to_float32(category_feats[as_key(item_id)])


def pad_items(item_ids: List[str]) -> Tuple[np.ndarray, np.ndarray]:
    arr = np.zeros((MAX_ITEM_LEN, ITEM_DIM), dtype=np.float32)
    mask = np.zeros((MAX_ITEM_LEN,), dtype=np.float32)
    keep = item_ids[:MAX_ITEM_LEN]
    for i, item_id in enumerate(keep):
        arr[i] = get_item_vec(item_id)
        mask[i] = 1.0
    return arr, mask


class CPCompatibilityDataset(Dataset):
    def __init__(self, examples: List[dict], outfit_feats: Dict[str, np.ndarray]):
        self.examples = [ex for ex in examples if ex['set_id'] in outfit_feats]
        self.outfit_feats = outfit_feats

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index: int):
        ex = self.examples[index]
        items, mask = pad_items(ex['item_ids'])
        outfit_text = vector_to_float32(self.outfit_feats[ex['set_id']])
        return {
            'items': torch.from_numpy(items),
            'mask': torch.from_numpy(mask),
            'outfit_text': torch.from_numpy(outfit_text),
            'label': torch.tensor(float(ex['label']), dtype=torch.float32),
        }


class ORFITBDataset(Dataset):
    def __init__(self, examples: List[dict], outfit_feats: Dict[str, np.ndarray]):
        self.examples = [ex for ex in examples if ex['set_id'] in outfit_feats]
        self.outfit_feats = outfit_feats

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index: int):
        ex = self.examples[index]
        partial_items, partial_mask = pad_items(ex['partial_item_ids'])
        candidates = np.stack([get_item_vec(item_id) for item_id in ex['answer_item_ids']], axis=0).astype(np.float32)
        outfit_text = vector_to_float32(self.outfit_feats[ex['set_id']])
        category = get_category_vec(ex['target_item_id'])
        return {
            'partial_items': torch.from_numpy(partial_items),
            'partial_mask': torch.from_numpy(partial_mask),
            'outfit_text': torch.from_numpy(outfit_text),
            'category': torch.from_numpy(category),
            'candidates': torch.from_numpy(candidates),
            'label': torch.tensor(int(ex['correct_index']), dtype=torch.long),
        }


# ---------------------------------------------------------------- P15 cell 7 (functions)
def build_fg_to_unique_items(outfits: List[dict]) -> Dict[str, List[str]]:
    fg_to_items = defaultdict(list)
    seen_by_fg = defaultdict(set)
    for outfit in outfits:
        for item in outfit['items']:
            item_id = as_key(item['item_id'])
            if item_id not in metadata:
                continue
            fg = as_key(metadata[item_id]['category_id'])
            if item_id not in seen_by_fg[fg]:
                seen_by_fg[fg].add(item_id)
                fg_to_items[fg].append(item_id)
    return dict(fg_to_items)


def build_or_candidate_pools(min_candidates: int = 3000) -> Dict[str, List[str]]:
    test_fg = build_fg_to_unique_items(split_outfits['test'])
    train_fg = build_fg_to_unique_items(split_outfits['train'])
    pools = {}
    for fg, test_items in test_fg.items():
        seen = set()
        pool = []
        for item_id in test_items:
            if item_id not in seen:
                seen.add(item_id)
                pool.append(item_id)
                if len(pool) >= min_candidates:
                    break
        if len(pool) < min_candidates:
            for item_id in train_fg.get(fg, []):
                if item_id not in seen:
                    seen.add(item_id)
                    pool.append(item_id)
                    if len(pool) >= min_candidates:
                        break
        if len(pool) >= min_candidates:
            pools[fg] = pool[:min_candidates]
    print(f'OR/CIR evaluable fine-grained categories: {len(pools):,} / test categories: {len(test_fg):,}')
    return pools


# ---------------------------------------------------------------- P15 cell 8
class MLPEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, embed_dim: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def masked_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask = mask.unsqueeze(-1)
    denom = mask.sum(dim=1).clamp_min(1.0)
    return (x * mask).sum(dim=1) / denom


class CPCompatibilityModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.item_encoder = MLPEncoder(ITEM_DIM, HIDDEN_DIM, EMBED_DIM, DROPOUT)
        self.outfit_text_encoder = MLPEncoder(TEXT_DIM, HIDDEN_DIM, EMBED_DIM, DROPOUT)
        self.classifier = nn.Sequential(
            nn.Linear(EMBED_DIM * 2, HIDDEN_DIM),
            nn.GELU(),
            nn.Dropout(DROPOUT),
            nn.Linear(HIDDEN_DIM, 1),
        )

    def encode_outfit(self, items: torch.Tensor, mask: torch.Tensor, outfit_text: torch.Tensor) -> torch.Tensor:
        item_z = self.item_encoder(items)
        outfit_z = masked_mean(item_z, mask)
        text_z = self.outfit_text_encoder(outfit_text)
        return torch.cat([outfit_z, text_z], dim=-1)

    def forward(self, items: torch.Tensor, mask: torch.Tensor, outfit_text: torch.Tensor) -> torch.Tensor:
        z = self.encode_outfit(items, mask, outfit_text)
        return self.classifier(z).squeeze(-1)


class ORRetrievalModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.item_encoder = MLPEncoder(ITEM_DIM, HIDDEN_DIM, EMBED_DIM, DROPOUT)
        self.outfit_text_encoder = MLPEncoder(TEXT_DIM, HIDDEN_DIM, EMBED_DIM, DROPOUT)
        self.category_encoder = MLPEncoder(CATEGORY_DIM, HIDDEN_DIM, EMBED_DIM, DROPOUT)
        self.query_net = nn.Sequential(
            nn.Linear(EMBED_DIM * 3, HIDDEN_DIM),
            nn.LayerNorm(HIDDEN_DIM),
            nn.GELU(),
            nn.Dropout(DROPOUT),
            nn.Linear(HIDDEN_DIM, EMBED_DIM),
        )
        self.logit_scale = nn.Parameter(torch.tensor(math.log(10.0), dtype=torch.float32))

    def encode_query(self, partial_items: torch.Tensor, partial_mask: torch.Tensor, outfit_text: torch.Tensor,
                     category: torch.Tensor) -> torch.Tensor:
        partial_item_z = self.item_encoder(partial_items)
        partial_z = masked_mean(partial_item_z, partial_mask)
        text_z = self.outfit_text_encoder(outfit_text)
        category_z = self.category_encoder(category)
        query = self.query_net(torch.cat([partial_z, text_z, category_z], dim=-1))
        return F.normalize(query, dim=-1)

    def encode_candidate(self, candidates: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.item_encoder(candidates), dim=-1)

    def forward(self, partial_items: torch.Tensor, partial_mask: torch.Tensor, outfit_text: torch.Tensor,
                category: torch.Tensor, candidates: torch.Tensor) -> torch.Tensor:
        query = self.encode_query(partial_items, partial_mask, outfit_text, category)
        cand = self.encode_candidate(candidates)
        scale = self.logit_scale.exp().clamp(max=100.0)
        return torch.einsum('bd,bkd->bk', query, cand) * scale


def make_loader(dataset: Dataset, batch_size: int, shuffle: bool, seed: int) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        generator=generator if shuffle else None,
    )


# ---------------------------------------------------------------- P15 cell 9
def evaluate_cp_auc(model: CPCompatibilityModel, loader: DataLoader) -> float:
    model.eval()
    labels = []
    scores = []
    with torch.no_grad():
        for batch in loader:
            items = batch['items'].to(DEVICE).float()
            mask = batch['mask'].to(DEVICE).float()
            outfit_text = batch['outfit_text'].to(DEVICE).float()
            label = batch['label'].detach().cpu().numpy()
            logit = model(items, mask, outfit_text).detach().cpu().numpy()
            labels.extend(label.tolist())
            scores.extend(logit.tolist())
    return binary_auc(labels, scores)


def evaluate_cp_fitb(model: CPCompatibilityModel, examples: List[dict], outfit_feats: Dict[str, np.ndarray]) -> float:
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for ex in tqdm(examples, desc='CP FITB'):
            if ex['set_id'] not in outfit_feats:
                continue
            batch_items = []
            batch_mask = []
            for answer_id in ex['answer_item_ids']:
                items, mask = pad_items(ex['partial_item_ids'] + [answer_id])
                batch_items.append(items)
                batch_mask.append(mask)
            items_t = torch.from_numpy(np.stack(batch_items)).to(DEVICE).float()
            mask_t = torch.from_numpy(np.stack(batch_mask)).to(DEVICE).float()
            outfit_text = vector_to_float32(outfit_feats[ex['set_id']])
            outfit_text_t = torch.from_numpy(np.stack([outfit_text] * len(batch_items))).to(DEVICE).float()
            scores = model(items_t, mask_t, outfit_text_t)
            pred = int(torch.argmax(scores).item())
            correct += int(pred == ex['correct_index'])
            total += 1
    return correct / total if total else 0.0


def train_cp_one_seed(variant_key: str, outfit_feats: Dict[str, np.ndarray], seed: int
                      ) -> Tuple[Path, CPCompatibilityModel, dict]:
    set_seed(seed)
    model_dir = MODEL_ROOT / variant_key / f'seed_{seed}'
    model_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = model_dir / 'cp_best_model.pt'
    log_path = model_dir / 'cp_train_log.csv'

    if ckpt_path.exists() and not FORCE_RETRAIN:
        checkpoint = torch.load(ckpt_path, map_location=DEVICE)
        model = CPCompatibilityModel().to(DEVICE)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        return ckpt_path, model, checkpoint

    if log_path.exists():
        log_path.unlink()

    train_dataset = CPCompatibilityDataset(compat_examples_by_split['train'], outfit_feats)
    valid_dataset = CPCompatibilityDataset(compat_examples_by_split['valid'], outfit_feats)
    train_loader = make_loader(train_dataset, BATCH_SIZE, shuffle=True, seed=seed)
    valid_loader = make_loader(valid_dataset, BATCH_SIZE, shuffle=False, seed=seed)

    model = CPCompatibilityModel().to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2)

    best_fitb = -1.0
    best_epoch = -1
    log_fields = ['variant', 'seed', 'stage', 'epoch', 'train_loss', 'valid_auc', 'valid_fitb_acc', 'lr', 'seconds']
    start_all = time.monotonic()

    print(f'\n[CP train] variant={variant_key}, seed={seed}, train={len(train_dataset):,}, valid={len(valid_dataset):,}')
    for epoch in range(1, CP_EPOCHS + 1):
        model.train()
        loss_sum = 0.0
        total = 0
        epoch_start = time.monotonic()
        for batch in tqdm(train_loader, desc=f'CP train {variant_key} seed={seed} epoch={epoch}'):
            items = batch['items'].to(DEVICE).float()
            mask = batch['mask'].to(DEVICE).float()
            outfit_text = batch['outfit_text'].to(DEVICE).float()
            label = batch['label'].to(DEVICE).float()

            optimizer.zero_grad(set_to_none=True)
            logit = model(items, mask, outfit_text)
            loss = F.binary_cross_entropy_with_logits(logit, label)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            batch_n = int(label.numel())
            loss_sum += float(loss.item()) * batch_n
            total += batch_n

        train_loss = loss_sum / total if total else 0.0
        valid_auc = evaluate_cp_auc(model, valid_loader)
        valid_fitb = evaluate_cp_fitb(model, fitb_examples_by_split['valid'], outfit_feats)
        scheduler.step(valid_fitb)
        lr_now = optimizer.param_groups[0]['lr']
        seconds = time.monotonic() - epoch_start

        append_csv(log_path, {
            'variant': variant_key, 'seed': seed, 'stage': 'CP', 'epoch': epoch,
            'train_loss': f'{train_loss:.6f}', 'valid_auc': f'{valid_auc:.6f}',
            'valid_fitb_acc': f'{valid_fitb:.6f}',
            'lr': f'{lr_now:.8f}', 'seconds': f'{seconds:.2f}',
        }, log_fields)
        print(f'CP epoch={epoch:02d} loss={train_loss:.4f} valid_AUC={valid_auc:.4f} valid_FITB={valid_fitb:.4f} '
              f'lr={lr_now:.2e}', flush=True)

        if valid_fitb > best_fitb:
            best_fitb = valid_fitb
            best_epoch = epoch
            torch.save({
                'variant': variant_key,
                'seed': seed,
                'stage': 'CP',
                'epoch': epoch,
                'best_valid_fitb_acc': best_fitb,
                'valid_auc_at_best_fitb': valid_auc,
                'model_state_dict': model.state_dict(),
                'selection_rule': 'best checkpoint selected by validation FITB accuracy',
            }, ckpt_path)

    checkpoint = torch.load(ckpt_path, map_location=DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    checkpoint['total_training_seconds'] = time.monotonic() - start_all
    checkpoint['best_epoch'] = best_epoch
    return ckpt_path, model, checkpoint


# ---------------------------------------------------------------- P15 cell 10
def evaluate_or_fitb(model: ORRetrievalModel, loader: DataLoader) -> float:
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in loader:
            partial_items = batch['partial_items'].to(DEVICE).float()
            partial_mask = batch['partial_mask'].to(DEVICE).float()
            outfit_text = batch['outfit_text'].to(DEVICE).float()
            category = batch['category'].to(DEVICE).float()
            candidates = batch['candidates'].to(DEVICE).float()
            label = batch['label'].to(DEVICE)
            scores = model(partial_items, partial_mask, outfit_text, category, candidates)
            pred = torch.argmax(scores, dim=1)
            correct += int((pred == label).sum().item())
            total += int(label.numel())
    return correct / total if total else 0.0


def train_or_one_seed(variant_key: str, outfit_feats: Dict[str, np.ndarray], seed: int, cp_model: CPCompatibilityModel
                      ) -> Tuple[Path, ORRetrievalModel, dict]:
    set_seed(seed)
    model_dir = MODEL_ROOT / variant_key / f'seed_{seed}'
    model_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = model_dir / 'best_model.pt'
    log_path = model_dir / 'or_train_log.csv'

    if ckpt_path.exists() and not FORCE_RETRAIN:
        checkpoint = torch.load(ckpt_path, map_location=DEVICE)
        model = ORRetrievalModel().to(DEVICE)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        return ckpt_path, model, checkpoint

    if log_path.exists():
        log_path.unlink()

    train_dataset = ORFITBDataset(fitb_examples_by_split['train'], outfit_feats)
    valid_dataset = ORFITBDataset(fitb_examples_by_split['valid'], outfit_feats)
    train_loader = make_loader(train_dataset, BATCH_SIZE, shuffle=True, seed=seed)
    valid_loader = make_loader(valid_dataset, BATCH_SIZE, shuffle=False, seed=seed)

    model = ORRetrievalModel().to(DEVICE)
    model.item_encoder.load_state_dict(cp_model.item_encoder.state_dict())
    model.outfit_text_encoder.load_state_dict(cp_model.outfit_text_encoder.state_dict())

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2)

    best_fitb = -1.0
    best_epoch = -1
    log_fields = ['variant', 'seed', 'stage', 'epoch', 'train_loss', 'valid_fitb_acc', 'lr', 'seconds']
    start_all = time.monotonic()

    print(f'\n[OR train] variant={variant_key}, seed={seed}, train={len(train_dataset):,}, valid={len(valid_dataset):,}')
    for epoch in range(1, OR_EPOCHS + 1):
        model.train()
        loss_sum = 0.0
        total = 0
        epoch_start = time.monotonic()
        for batch in tqdm(train_loader, desc=f'OR train {variant_key} seed={seed} epoch={epoch}'):
            partial_items = batch['partial_items'].to(DEVICE).float()
            partial_mask = batch['partial_mask'].to(DEVICE).float()
            outfit_text = batch['outfit_text'].to(DEVICE).float()
            category = batch['category'].to(DEVICE).float()
            candidates = batch['candidates'].to(DEVICE).float()
            label = batch['label'].to(DEVICE)

            optimizer.zero_grad(set_to_none=True)
            scores = model(partial_items, partial_mask, outfit_text, category, candidates)
            loss = F.cross_entropy(scores, label)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            batch_n = int(label.numel())
            loss_sum += float(loss.item()) * batch_n
            total += batch_n

        train_loss = loss_sum / total if total else 0.0
        valid_fitb = evaluate_or_fitb(model, valid_loader)
        scheduler.step(valid_fitb)
        lr_now = optimizer.param_groups[0]['lr']
        seconds = time.monotonic() - epoch_start

        append_csv(log_path, {
            'variant': variant_key, 'seed': seed, 'stage': 'OR', 'epoch': epoch,
            'train_loss': f'{train_loss:.6f}', 'valid_fitb_acc': f'{valid_fitb:.6f}',
            'lr': f'{lr_now:.8f}', 'seconds': f'{seconds:.2f}',
        }, log_fields)
        print(f'OR epoch={epoch:02d} loss={train_loss:.4f} valid_FITB={valid_fitb:.4f} lr={lr_now:.2e}', flush=True)

        if valid_fitb > best_fitb:
            best_fitb = valid_fitb
            best_epoch = epoch
            torch.save({
                'variant': variant_key,
                'seed': seed,
                'stage': 'OR',
                'epoch': epoch,
                'best_valid_fitb_acc': best_fitb,
                'model_state_dict': model.state_dict(),
                'selection_rule': 'best checkpoint selected by validation FITB accuracy',
                'transfer_note': 'item_encoder and outfit_text_encoder initialized from CP model',
            }, ckpt_path)

    checkpoint = torch.load(ckpt_path, map_location=DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    checkpoint['total_training_seconds'] = time.monotonic() - start_all
    checkpoint['best_epoch'] = best_epoch
    return ckpt_path, model, checkpoint


# ---------------------------------------------------------------- P15 cell 11
def encode_candidates(model: ORRetrievalModel, item_ids: List[str], batch_size: int = 1024) -> torch.Tensor:
    encoded_batches = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(item_ids), batch_size):
            batch_ids = item_ids[start:start + batch_size]
            raw = np.stack([get_item_vec(item_id) for item_id in batch_ids], axis=0).astype(np.float32)
            raw_t = torch.from_numpy(raw).to(DEVICE).float()
            encoded_batches.append(model.encode_candidate(raw_t))
    return torch.cat(encoded_batches, dim=0)


def encode_one_query(model: ORRetrievalModel, ex: dict, outfit_feats: Dict[str, np.ndarray]) -> torch.Tensor:
    partial_items, partial_mask = pad_items(ex['partial_item_ids'])
    outfit_text = vector_to_float32(outfit_feats[ex['set_id']])
    category = get_category_vec(ex['target_item_id'])
    with torch.no_grad():
        return model.encode_query(
            torch.from_numpy(partial_items).unsqueeze(0).to(DEVICE).float(),
            torch.from_numpy(partial_mask).unsqueeze(0).to(DEVICE).float(),
            torch.from_numpy(outfit_text).unsqueeze(0).to(DEVICE).float(),
            torch.from_numpy(category).unsqueeze(0).to(DEVICE).float(),
        ).squeeze(0)


def evaluate_or_recall(model: ORRetrievalModel, outfit_feats: Dict[str, np.ndarray], variant_key: str, seed: int
                       ) -> Tuple[dict, List[dict]]:
    model.eval()
    pool_emb_cache = {}
    pool_id_cache = {}
    for fg, ids in tqdm(candidate_pools.items(), desc=f'encode candidate pools [{variant_key} seed={seed}]'):
        pool_id_cache[fg] = list(ids)
        pool_emb_cache[fg] = encode_candidates(model, ids)

    target_emb_cache: Dict[str, torch.Tensor] = {}
    rows = []
    hits = {k: 0 for k in K_LIST}
    ranks = []
    skipped = 0

    for ex in tqdm(fitb_examples_by_split['test'], desc=f'OR/CIR recall [{variant_key} seed={seed}]'):
        if ex['set_id'] not in outfit_feats:
            skipped += 1
            continue
        fg = ex['target_fg']
        target_item_id = ex['target_item_id']
        if fg not in candidate_pools:
            skipped += 1
            continue

        ids = list(pool_id_cache[fg])
        cand_z = pool_emb_cache[fg]
        if target_item_id in ids:
            target_pos = ids.index(target_item_id)
        else:
            ids = ids[:-1] + [target_item_id]
            if target_item_id not in target_emb_cache:
                target_emb_cache[target_item_id] = encode_candidates(model, [target_item_id])
            cand_z = torch.cat([cand_z[:-1], target_emb_cache[target_item_id]], dim=0)
            target_pos = len(ids) - 1

        q_z = encode_one_query(model, ex, outfit_feats)
        scores = torch.matmul(cand_z, q_z)
        target_score = scores[target_pos]
        rank = int((scores > target_score).sum().item()) + 1
        ranks.append(rank)

        top_k = min(max(K_LIST), len(ids))
        _, top_idx = torch.topk(scores, k=top_k, largest=True)
        top_ids = [ids[int(i)] for i in top_idx.detach().cpu().tolist()]

        row = {
            'variant': variant_key,
            'seed': seed,
            'question_index': ex['question_index'],
            'set_id': ex['set_id'],
            'target_item_id': target_item_id,
            'target_fg': fg,
            'target_semantic_category': ex['target_semantic_category'],
            'rank': rank,
            'partial_item_ids': json.dumps(ex['partial_item_ids'], ensure_ascii=False),
            'answer_item_ids': json.dumps(ex['answer_item_ids'], ensure_ascii=False),
            'top10_ids': json.dumps(top_ids[:10], ensure_ascii=False),
        }
        for k in K_LIST:
            hit = int(rank <= k)
            hits[k] += hit
            row[f'recall@{k}'] = hit
        rows.append(row)

    n = len(rows)
    summary = {
        'variant': variant_key,
        'seed': seed,
        'test_fitb_questions': len(fitb_examples_by_split['test']),
        'or_recall_evaluable_pairs': n,
        'or_recall_skipped_questions': skipped,
        'mean_rank': float(np.mean(ranks)) if ranks else float('nan'),
        'median_rank': float(np.median(ranks)) if ranks else float('nan'),
    }
    for k in K_LIST:
        summary[f'recall@{k}'] = hits[k] / n if n else 0.0
    return summary, rows


# ---------------------------------------------------------------- data loading (P15 cells 4, 5 and 7, module level)
def load_data(polyvore: Path) -> None:
    global DISJOINT_DIR, img_feats, item_text_feats, category_feats, metadata, split_outfits, fitb_raw
    global ref_to_item_by_split, compat_examples_by_split, fitb_examples_by_split, candidate_pools
    DISJOINT_DIR = polyvore / 'disjoint'
    img_feats = load_pickle(FASHIONCLIP_DIR / 'img_feats_fashionClip.pkl')
    item_text_feats = load_pickle(FASHIONCLIP_DIR / 'encoded_title_description_distiluse-base-multilingual-cased-v2.pkl')
    category_feats = load_pickle(FASHIONCLIP_DIR / 'encoded_category_distiluse-base-multilingual-cased-v2.pkl')
    metadata = load_json(polyvore / 'polyvore_item_metadata.json')
    split_outfits = {split: load_json(DISJOINT_DIR / f'{split}.json') for split in ('train', 'valid', 'test')}
    fitb_raw = {split: load_json(DISJOINT_DIR / f'fill_in_blank_{split}.json') for split in ('train', 'valid', 'test')}
    ref_to_item_by_split = {split: build_ref_to_item(outfits) for split, outfits in split_outfits.items()}
    compat_examples_by_split = {
        'train': parse_compatibility_examples('train', TRAIN_LIMIT),
        'valid': parse_compatibility_examples('valid', VALID_LIMIT),
        'test': parse_compatibility_examples('test', TEST_LIMIT),
    }
    fitb_examples_by_split = {
        'train': parse_fitb_examples('train', TRAIN_LIMIT),
        'valid': parse_fitb_examples('valid', VALID_LIMIT),
        'test': parse_fitb_examples('test', TEST_LIMIT),
    }
    candidate_pools = build_or_candidate_pools(MIN_CANDIDATES_PER_FG)


def evaluate_seed(variant_key: str, seed: int, outfit_feats, cp_model, or_model) -> Tuple[dict, List[dict]]:
    """The test evaluation of P15 cell 12."""
    cp_test_dataset = CPCompatibilityDataset(compat_examples_by_split['test'], outfit_feats)
    cp_test_loader = make_loader(cp_test_dataset, BATCH_SIZE, shuffle=False, seed=seed)
    cp_test_auc = evaluate_cp_auc(cp_model, cp_test_loader)
    cp_test_fitb = evaluate_cp_fitb(cp_model, fitb_examples_by_split['test'], outfit_feats)
    or_test_dataset = ORFITBDataset(fitb_examples_by_split['test'], outfit_feats)
    or_test_loader = make_loader(or_test_dataset, BATCH_SIZE, shuffle=False, seed=seed)
    or_test_fitb = evaluate_or_fitb(or_model, or_test_loader)
    or_summary, or_rows = evaluate_or_recall(or_model, outfit_feats, variant_key, seed)
    return {'cp_test_auc': float(cp_test_auc), 'cp_test_fitb_acc': float(cp_test_fitb),
            'or_test_fitb_acc': float(or_test_fitb), **or_summary}, or_rows


def load_model(cls, path: Path):
    model = cls().to(DEVICE)
    checkpoint = torch.load(path, map_location=DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model, checkpoint


def table8(rows: List[dict]) -> List[List[str]]:
    """Mean ± SD per condition, paired difference and paired t 95% CI over the seeds (manuscript Table 8)."""
    by = defaultdict(dict)
    for r in rows:
        by[r['variant']][int(r['seed'])] = r
    seeds = sorted(set(by['original_text']) & set(by['context_aware_description']))
    out = []
    for metric in TABLE8_METRICS:
        o = np.array([float(by['original_text'][s][metric]) for s in seeds])
        c = np.array([float(by['context_aware_description'][s][metric]) for s in seeds])
        d = c - o
        half = stats.t.ppf(0.975, len(seeds) - 1) * d.std(ddof=1) / math.sqrt(len(seeds)) if len(seeds) > 1 else 0.0
        out.append([metric, f'{o.mean():.4f} ± {o.std(ddof=1) if len(seeds) > 1 else 0:.4f}',
                    f'{c.mean():.4f} ± {c.std(ddof=1) if len(seeds) > 1 else 0:.4f}', f'{d.mean():+.4f}',
                    f'[{d.mean() - half:.4f}, {d.mean() + half:.4f}]', MANUSCRIPT_TABLE8_DIFF[metric]])
    return out


def run_evaluate_preserved(out: Path, seeds: List[int]) -> List[str]:
    archived = {(r['variant'], int(r['seed'])): r for r in csv.DictReader(open(ARCHIVED_A30, encoding='utf-8'))}
    compare = ('cp_test_auc', 'cp_test_fitb_acc', 'or_test_fitb_acc', 'or_recall_evaluable_pairs', 'mean_rank',
               'median_rank') + tuple(f'recall@{k}' for k in K_LIST)
    rows, worst = [], defaultdict(float)
    for variant_key, feature_file in TEXT_VARIANTS.items():
        outfit_feats = load_pickle(FASHIONCLIP_DIR / feature_file)
        for seed in seeds:
            model_dir = TWO_TOWER_MODELS / variant_key / f'seed_{seed}'
            cp_model, _ = load_model(CPCompatibilityModel, model_dir / 'cp_best_model.pt')
            or_model, _ = load_model(ORRetrievalModel, model_dir / 'best_model.pt')
            summary, _ = evaluate_seed(variant_key, seed, outfit_feats, cp_model, or_model)
            old = archived[(variant_key, seed)]
            row = {'variant': variant_key, 'seed': seed}
            for metric in compare:
                diff = abs(float(summary[metric]) - float(old[metric]))
                worst[metric] = max(worst[metric], diff)
                row[metric] = summary[metric]
                row[f'{metric}_archived'] = old[metric]
            rows.append(row)
            print(f'[TWO-TOWER] preserved {variant_key} seed {seed}: AUC {summary["cp_test_auc"]:.6f} '
                  f'(archived {float(old["cp_test_auc"]):.6f}), R@10 {summary["recall@10"]:.6f}', flush=True)
    write_csv(out / 'validation_preserved_checkpoints.csv', rows, list(rows[0]))
    lines = ['## Preserved checkpoints evaluated with this code', '',
             f'{len(rows)} runs (2 text conditions x seeds {", ".join(map(str, seeds))}) against the archived A30. '
             'Largest absolute difference per metric:', '', '| Metric | Largest difference |', '|---|---:|']
    lines += [f'| {m} | {worst[m]:.3g} |' for m in compare]
    return lines + ['']


def run_train(out: Path, seeds: List[int]) -> List[str]:
    global MODEL_ROOT
    MODEL_ROOT = out / 'models'
    result_root = out / 'results'
    all_seed_rows, all_or_rows = [], []
    for variant_key, feature_file in TEXT_VARIANTS.items():
        outfit_feats = load_pickle(FASHIONCLIP_DIR / feature_file)
        for seed in seeds:
            model_dir = MODEL_ROOT / variant_key / f'seed_{seed}'
            result_dir = result_root / variant_key / f'seed_{seed}'
            model_dir.mkdir(parents=True, exist_ok=True)
            result_dir.mkdir(parents=True, exist_ok=True)
            cp_ckpt_path, cp_model, cp_checkpoint = train_cp_one_seed(variant_key, outfit_feats, seed)
            or_ckpt_path, or_model, or_checkpoint = None, None, None
            # P15 evaluates the CP model on the test split before OR training; keep that order.
            cp_test_dataset = CPCompatibilityDataset(compat_examples_by_split['test'], outfit_feats)
            cp_test_loader = make_loader(cp_test_dataset, BATCH_SIZE, shuffle=False, seed=seed)
            cp_test_auc = evaluate_cp_auc(cp_model, cp_test_loader)
            cp_test_fitb = evaluate_cp_fitb(cp_model, fitb_examples_by_split['test'], outfit_feats)
            or_ckpt_path, or_model, or_checkpoint = train_or_one_seed(variant_key, outfit_feats, seed, cp_model)
            or_test_dataset = ORFITBDataset(fitb_examples_by_split['test'], outfit_feats)
            or_test_loader = make_loader(or_test_dataset, BATCH_SIZE, shuffle=False, seed=seed)
            or_test_fitb = evaluate_or_fitb(or_model, or_test_loader)
            or_summary, or_rows = evaluate_or_recall(or_model, outfit_feats, variant_key, seed)
            seed_summary = {
                'variant': variant_key,
                'seed': seed,
                'cp_best_epoch': int(cp_checkpoint.get('epoch', cp_checkpoint.get('best_epoch', -1))),
                'cp_best_valid_fitb_acc': float(cp_checkpoint.get('best_valid_fitb_acc', float('nan'))),
                'cp_valid_auc_at_best_fitb': float(cp_checkpoint.get('valid_auc_at_best_fitb', float('nan'))),
                'cp_test_auc': float(cp_test_auc),
                'cp_test_fitb_acc': float(cp_test_fitb),
                'or_best_epoch': int(or_checkpoint.get('epoch', or_checkpoint.get('best_epoch', -1))),
                'or_best_valid_fitb_acc': float(or_checkpoint.get('best_valid_fitb_acc', float('nan'))),
                'or_test_fitb_acc': float(or_test_fitb),
                'cp_checkpoint_sha256': sha256(cp_ckpt_path),
                'or_checkpoint_sha256': sha256(or_ckpt_path),
                'training_seconds': round(float(cp_checkpoint['total_training_seconds'])
                                          + float(or_checkpoint['total_training_seconds']), 1),
            }
            seed_summary.update(or_summary)
            all_seed_rows.append(seed_summary)
            all_or_rows.extend(or_rows)
            write_csv(result_dir / 'seed_summary.csv', [seed_summary])
            write_csv(result_dir / 'or_recall_rowlevel.csv', or_rows)
            (model_dir / 'config.json').write_text(json.dumps({
                'variant': variant_key, 'seed': seed,
                'pipeline': 'CP binary compatibility pretraining -> OR retrieval fine-tuning',
                'checkpoint_selection': 'CP and OR checkpoints selected by validation FITB accuracy; '
                                        'train up to 100 epochs without early stopping',
                'text_feature_file': feature_file, 'seed_summary': seed_summary}, ensure_ascii=False, indent=2),
                encoding='utf-8')
    write_csv(out / 'A30_second_model_two_tower_seed_summary.csv', all_seed_rows)
    write_csv(out / 'A31_second_model_two_tower_or_rowlevel.csv', all_or_rows)
    metrics = ['cp_best_valid_fitb_acc', 'cp_valid_auc_at_best_fitb', 'cp_test_auc', 'cp_test_fitb_acc',
               'or_best_valid_fitb_acc', 'or_test_fitb_acc', 'mean_rank', 'median_rank'] + [f'recall@{k}' for k in K_LIST]
    aggregate_rows = []
    for variant_key in TEXT_VARIANTS:
        rows = [row for row in all_seed_rows if row['variant'] == variant_key]
        agg = {'variant': variant_key, 'n_seeds': len(rows), 'seeds': ','.join(str(row['seed']) for row in rows),
               'test_fitb_questions': rows[0]['test_fitb_questions'],
               'or_recall_evaluable_pairs': rows[0]['or_recall_evaluable_pairs']}
        for metric in metrics:
            values = [float(row[metric]) for row in rows]
            agg[f'{metric}_mean'] = float(np.mean(values))
            agg[f'{metric}_std'] = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
            agg[f'{metric}_mean±std'] = mean_std_text(values)
        aggregate_rows.append(agg)
    write_csv(out / 'A32_second_model_two_tower_mean_std.csv', aggregate_rows)
    new = table8(all_seed_rows)
    old = table8(list(csv.DictReader(open(ARCHIVED_A30, encoding='utf-8'))))
    write_csv(out / 'table8_two_tower.csv', [dict(zip(['metric', 'original_mean_sd', 'context_aware_mean_sd',
                                                       'mean_difference', 'paired_95_ci', 'manuscript_difference'], r))
                                              for r in new])
    lines = ['## Retrained models (Table 8)', '',
             '| Metric | Original | Context-aware | Difference [95% CI] | Archived run | Manuscript |',
             '|---|---|---|---|---|---:|']
    lines += [f'| {n[0]} | {n[1]} | {n[2]} | {n[3]} {n[4]} | {o[3]} {o[4]} | {n[5]} |' for n, o in zip(new, old)]
    return lines + ['']


def main() -> None:
    global DEVICE, PIN_MEMORY, TRAIN_LIMIT, VALID_LIMIT, TEST_LIMIT, CP_EPOCHS, OR_EPOCHS
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--evaluate-preserved', action='store_true')
    mode.add_argument('--train', action='store_true')
    mode.add_argument('--smoke', action='store_true')
    parser.add_argument('--out-dir', type=Path, required=True)
    parser.add_argument('--polyvore-root', default=None)
    parser.add_argument('--device', choices=('cuda', 'cpu'), default=None,
                        help='default: cuda for --train, cpu otherwise')
    parser.add_argument('--seeds', default='1,2,3,4,5')
    parser.add_argument('--threads', type=int, default=4)
    args = parser.parse_args()

    DEVICE = args.device or ('cuda' if args.train else 'cpu')
    if DEVICE == 'cuda' and not torch.cuda.is_available():
        raise SystemExit('[TWO-TOWER BLOCKED] CUDA is not available; --train needs the GPU (or pass --device cpu)')
    PIN_MEMORY = DEVICE == 'cuda'
    torch.set_num_threads(args.threads)
    polyvore = local_path('polyvore_root', args.polyvore_root)
    if polyvore is None or not (polyvore / 'disjoint' / 'compatibility_valid.txt').is_file():
        raise SystemExit('[TWO-TOWER BLOCKED] Polyvore root not found; pass --polyvore-root or run bootstrap_data.sh')
    out = output_dir(args.out_dir)
    if (args.train or args.smoke) and (out / 'models').exists():
        raise SystemExit(f'[TWO-TOWER BLOCKED] {out}/models exists; choose a new output folder')
    seeds = [int(s) for s in args.seeds.split(',')]
    if args.smoke:
        TRAIN_LIMIT, VALID_LIMIT, TEST_LIMIT, CP_EPOCHS, OR_EPOCHS = 512, 256, 256, 1, 1
        seeds = seeds[:1]
    started = time.monotonic()
    load_data(polyvore)
    if args.evaluate_preserved:
        lines = run_evaluate_preserved(out, seeds)
    else:
        lines = run_train(out, seeds)
    header = ['# Two-Tower cross-architecture check (manuscript Table 8)', '',
              f'Code copied from notebook P15. Mode: {"evaluate preserved checkpoints" if args.evaluate_preserved else "smoke test (1 epoch, subset)" if args.smoke else "retraining"}; '
              f'device {DEVICE}; {round(time.monotonic() - started)} s.', '']
    write_text(out / 'summary.md', header + lines)
    (out / 'manifest.json').write_text(json.dumps({
        'mode': 'evaluate_preserved' if args.evaluate_preserved else 'smoke' if args.smoke else 'train',
        'device': DEVICE, 'torch': torch.__version__, 'seeds': seeds,
        'cuda_device': torch.cuda.get_device_name(0) if DEVICE == 'cuda' else None,
        'features': {name: sha256(FASHIONCLIP_DIR / name) for name in
                     ['img_feats_fashionClip.pkl', 'encoded_title_description_distiluse-base-multilingual-cased-v2.pkl',
                      'encoded_category_distiluse-base-multilingual-cased-v2.pkl', *TEXT_VARIANTS.values()]},
    }, indent=1) + '\n', encoding='utf-8')
    print('\n'.join(header + lines))


if __name__ == '__main__':
    main()
