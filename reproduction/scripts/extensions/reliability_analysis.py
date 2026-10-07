#!/usr/bin/env python3
"""Reliability analysis of the seed-1 Original and Context CIR models (2025 table T18, figures F12-F16, A16).

Ported from the notebook
03_實驗與結果_experiments_results/00_控制檢查與附加稽核/source_programs/P04_reliability_xai_analysis.ipynb
(cell 0, "01_reliability_analysis_figures.py"). P04 evaluates the two seed-1 CIR models again with its own
loop. The candidate pool of a fine-grained category holds the unique test items, then train items, up to
3,000 (P04 skips the train categories that the test split lacks, where the archived evaluator stops at the
first one; both rules select the same categories and queries, see summary.md). For each query it records the
rank of the target, Hit@k, the top-10 items and a confidence proxy, sigmoid(50 x (top-1 score - top-2 score));
a query with Hit@10 = 0 and confidence >= 0.85 is a high-confidence error. Each query is labelled from the
input data (the weather, occasion and style fragments, the generated and original titles, categories.csv
with its first label of a duplicated ID):

- temperature_c: the first temperature in the weather fragments, else the generated description, else the
  original title (Fahrenheit converted to Celsius);
- weather_group: cold up to the median temperature of the evaluated queries, warm above it;
- occasion_group: formal or casual by P04's keyword lists, else unknown;
- style_group: high-style with two or more style fragments, else low-style;
- category_group: clothing-led or accessory-led by the target's major category.

Outputs: the per-query table in the T18 layout (reliability_required_fields.csv, about 40 MB, not committed),
the performance table (F12: Hit@1, Hit@10, median and mean rank, mean confidence, error rates, ECE over 10
bins, Brier score), the calibration bins and reliability diagram (F13), the error types (F14), the subgroup
Hit@10 (F15) and its difference heatmap (F16), the confidence bands, the top high-confidence errors and the
range checks of the labels (A16, from P12 cell 6). Every mode also compares the labels with the archived T18
and A16.

--labels-only builds the labels without a model. --run-root RUN evaluates RUN/main/original_seed1 and
RUN/main/context_seed1. --validate-2025 evaluates the preserved 2025 cir_old_seed1 and cir_new_seed1
checkpoints and compares each query with the archived T18 and the performance table printed by P04. P04 ran
on CUDA with float16 autocast, the default here; --device cpu runs in float32, which moves some scores and
ranks slightly, and is marked as a check run. --max-queries N stops after N evaluated queries (plumbing test).
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import shutil
import sys
import tempfile
import textwrap
import time
from collections import Counter, defaultdict
from contextlib import nullcontext
from pathlib import Path
from statistics import mean

import numpy as np

from _ext import (CHECKPOINTS_2025, D01, D03, FEATURES, MODEL_CODE, REPO, REPRO, SCRIPTS, WOS_JSONL, local_path,
                  output_dir, passed_run, read_json, sha256, write_csv, write_json, write_text)
from _svg import PALETTE, VIRIDIS, Chart, grouped_vbar_chart, heatmap_chart, table_chart

sys.path.insert(0, str(SCRIPTS))
import standard_decoder  # noqa: E402

P04 = D03 / "00_控制檢查與附加稽核" / "source_programs" / "P04_reliability_xai_analysis.ipynb"
ARCHIVED_T18 = (D03 / "00_控制檢查與附加稽核" / "圖表_figures_tables" / "tables"
                / "T18_reliability_required_fields_compare.csv")
ARCHIVED_A16 = D01 / "clo_met_temperature" / "A16_environment_proxy_range_error_handling.csv"

SEED = 1
POOL_SIZE = 3000
HIGH_CONF_THRESHOLD = 0.85
SOFTMAX_TEMPERATURE = 0.05
GAP_SCALE = 50.0
N_BINS = 10
K_LIST = (1, 3, 5, 10, 30, 50)
BASELINE, PROPOSED = "original", "proposed"
MODEL_NAMES = {BASELINE: "Original CIR / outfitUrlTitle", PROPOSED: "Proposed CIR / NewoutfitUrlTitle"}
OUTFIT_FEATURES = {BASELINE: "encoded_outfitUrlTitle_en_fashionClip.pkl",
                   PROPOSED: "encoded_NewoutfitUrlTitle_en_fashionClip.pkl"}
RUN_UNITS = {BASELINE: "original_seed1", PROPOSED: "context_seed1"}
CHECKPOINTS_2025_SEED1 = {BASELINE: "cir_old_seed1", PROPOSED: "cir_new_seed1"}
COLORS = {BASELINE: PALETTE[0], PROPOSED: PALETTE[1]}
P04_PRINTED = {  # P04's printed performance table (2025 seed-1 models on CUDA), six decimals
    BASELINE: {"n": 9311, "Hit@1": 0.013855, "Hit@10": 0.077972, "Median Rank": 230.0, "Mean Rank": 463.324777,
               "Avg Confidence": 0.720271, "Error Rate": 0.922028, "High-conf Error Rate": 0.219740,
               "ECE": 0.642299, "Brier Score": 0.504966},
    PROPOSED: {"n": 9311, "Hit@1": 0.015251, "Hit@10": 0.087101, "Median Rank": 205.0, "Mean Rank": 441.759102,
               "Avg Confidence": 0.718550, "Error Rate": 0.912899, "High-conf Error Rate": 0.206315,
               "ECE": 0.631449, "Brier Score": 0.497345},
}
METRICS = ("Hit@1", "Hit@10", "Median Rank", "Mean Rank", "Avg Confidence", "Error Rate", "High-conf Error Rate",
           "ECE", "Brier Score")
REQUIRED_COLUMNS = [  # P04 main(): needed_cols, the T18 layout
    "run_tag", "query_key", "metadata_key", "seed", "set_id", "model_name",
    "model_output_top1_id", "model_output_top10_ids", "confidence", "confidence_band", "confidence_source",
    "correct_answer", "rank", "hit@1", "hit@3", "hit@5", "hit@10", "hit@30", "hit@50", "is_error",
    "is_high_conf_error", "error_type", "possible_source", "observable_log", "risk_level",
    "target_item_id", "target_item_fg", "fine_category", "major_category", "temperature_c", "weather_group",
    "occasion_group", "style_count", "style_group", "category_group",
    "top1_score", "top2_score", "target_score", "top1_top2_gap",
    "weather_text", "occasion_text", "style_text", "title_full_text", "original_text",
]
LABEL_FIELDS = ("set_id", "target_item_id", "target_item_fg", "fine_category", "major_category", "temperature_c",
                "weather_group", "occasion_group", "style_count", "style_group", "category_group", "weather_text",
                "occasion_text", "style_text", "title_full_text", "original_text")
SUBGROUP_COLUMNS = ("weather_group", "occasion_group", "style_group", "category_group")

# ---------------------------------------------------------------- P04 sections 2-3: text helpers and labels


def normalize_text(x) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return ""
    if isinstance(x, str):
        return x.strip()
    if isinstance(x, list):
        return " | ".join(str(i).strip() for i in x if str(i).strip())
    return str(x).strip()


def normalize_token_text(x) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return ""
    x = str(x).strip().lower()
    x = re.sub(r"[_/|,-]+", " ", x)
    return re.sub(r"\s+", " ", x).strip()


def sigmoid(x: float) -> float:
    if x >= 0:
        return 1 / (1 + math.exp(-x))
    z = math.exp(x)
    return z / (1 + z)


def confidence_band(x: float) -> str:
    if x is None or math.isnan(x):
        return "unknown"
    for upper, band in ((0.60, "0.50–0.60"), (0.70, "0.60–0.70"), (0.80, "0.70–0.80"), (0.90, "0.80–0.90")):
        if x < upper:
            return band
    return "0.90–1.00"


def load_original_titles(path: Path) -> dict[str, str]:
    out = {}
    for sid, value in read_json(path).items():
        if isinstance(value, str):
            text = value.strip()
        elif isinstance(value, dict):
            url = normalize_text(value.get("url_name", value.get("url", "")))
            title = normalize_text(value.get("title", ""))
            text = " ".join(p for p in [url, title] if p).strip()
        else:
            text = normalize_text(value)
        out[str(sid)] = text
    return out


def load_fragments(path: Path) -> dict[str, dict]:
    out = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if obj.get("valid") is not True:
                continue
            frags = obj.get("fragments", {})
            terms = {k: frags.get(k, []) if isinstance(frags.get(k, []), list) else []
                     for k in ("weather", "occasion", "style")}
            set_id = str(obj.get("id", ""))
            if set_id in out:  # P04 merges every valid row; the archived file has one per ID
                raise SystemExit(f"[RELIABILITY BLOCKED] {path.name}: more than one valid row for {set_id}")
            out[set_id] = {"title_full_text": normalize_text(obj.get("title", "")), "style_terms": terms["style"],
                           **{f"{k}_text": normalize_text(v) for k, v in terms.items()}}
    return out


def load_category_map(path: Path) -> dict[str, tuple[str, str]]:
    """categories.csv: the first label of a duplicated ID, normalized (P04 load_category_map)."""
    out = {}
    with path.open(encoding="utf-8", newline="") as f:
        for row in csv.reader(f):
            if not row:
                continue
            fine = normalize_token_text(row[1]) if len(row) >= 2 else ""
            major = normalize_token_text(row[2]) if len(row) >= 3 else ""
            out.setdefault(row[0].strip(), (fine, major))
    return out


TEMP_PATTERN = re.compile(r"(-?\d+(?:\.\d+)?)\s*°?\s*([CFcf])")
FORMAL_KEYWORDS = (r"\b(formal|office|business|work|professional|interview|meeting|wedding|party|cocktail|elegant|"
                   r"tailored|blazer|suit)\b")
CASUAL_KEYWORDS = (r"\b(casual|weekend|streetwear|street|relaxed|everyday|daily|chill|laid-back|lounge|vacation|"
                   r"travel)\b")
CLOTHING_MAJOR = {"all body", "bottoms", "tops", "outerwear"}
ACCESSORY_MAJOR = {"bags", "shoes", "accessories", "hats", "jewellery", "jewelry", "scarves", "sunglasses"}


def temperature_celsius(text: str) -> float | None:
    if not isinstance(text, str) or not text.strip():
        return None
    m = TEMP_PATTERN.search(text)
    if not m:
        return None
    value = float(m.group(1))
    return (value - 32.0) * 5.0 / 9.0 if m.group(2).upper() == "F" else value


def occasion_group(row: dict) -> str:
    text = " ".join(normalize_text(row[k]) for k in ("occasion_text", "title_full_text", "original_text")).lower()
    if re.search(FORMAL_KEYWORDS, text):
        return "formal"
    if re.search(CASUAL_KEYWORDS, text):
        return "casual"
    return "unknown"


def category_group(major: str) -> str:
    major = normalize_token_text(major)
    if major in CLOTHING_MAJOR:
        return "clothing-led"
    if major in ACCESSORY_MAJOR:
        return "accessory-led"
    return "unknown"


def query_labels(queries: list[tuple[str, str, str]], polyvore: Path) -> dict[str, dict]:
    """P04 enrich_metadata: metadata_key -> labels, for (set_id, target_item_id, target_item_fg) queries."""
    titles = load_original_titles(polyvore / "polyvore_outfit_titles.json")
    fragments = load_fragments(WOS_JSONL)
    categories = load_category_map(polyvore / "categories.csv")
    labels = {}
    for set_id, target, fg in queries:
        frag = fragments.get(set_id, {})
        fine, major = categories.get(str(fg).strip(), ("", ""))
        row = {"set_id": set_id, "target_item_id": target, "target_item_fg": str(fg), "fine_category": fine,
               "major_category": major, "original_text": titles.get(set_id, ""),
               **{k: frag.get(k, "") for k in ("title_full_text", "weather_text", "occasion_text", "style_text")}}
        row["temperature_c"] = next((t for t in (temperature_celsius(row[k]) for k in
                                                 ("weather_text", "title_full_text", "original_text"))
                                     if t is not None), None)
        row["occasion_group"] = occasion_group(row)
        row["style_count"] = len([t for t in frag.get("style_terms", []) if str(t).strip()])
        row["style_group"] = "high-style" if row["style_count"] >= 2 else "low-style"
        row["category_group"] = category_group(major)
        labels[f"{SEED}||{set_id}||{target}"] = row
    temps = [r["temperature_c"] for r in labels.values() if r["temperature_c"] is not None]
    threshold = float(np.median(temps)) if temps else None
    for r in labels.values():
        t = r["temperature_c"]
        r["weather_group"] = "unknown" if t is None or threshold is None else ("cold" if t <= threshold else "warm")
    return labels


def pool_queries(polyvore: Path, rule: str) -> tuple[list[tuple[str, str, str]], dict[str, int]]:
    """The queries whose target category gets a full pool, without a model.

    rule "p04": P04 build_distractors (train categories absent from the test split are skipped);
    rule "evaluator": the archived evaluate_cir.py (the train loop stops at the first such category).
    """
    meta = read_json(polyvore / "polyvore_item_metadata.json")
    fg2ims, id2im = {}, {}
    for split in ("test", "train"):
        fg2ims[split] = {}
        for outfit in read_json(polyvore / "disjoint" / f"{split}.json"):
            for item in outfit["items"]:
                fg = str(meta[item["item_id"]]["category_id"])
                fg2ims[split].setdefault(fg, {}).setdefault(outfit["set_id"], []).append(item["item_id"])
                if split == "test":
                    id2im[f"{outfit['set_id']}_{item['index']}"] = item["item_id"]
    ids, seen = {}, {}
    for k, v in fg2ims["test"].items():
        ids[k], seen[k] = [], set()
        for item_lst in v.values():
            for item_id in item_lst:
                if item_id not in seen[k]:
                    seen[k].add(item_id)
                    ids[k].append(item_id)
                if len(ids[k]) >= POOL_SIZE:
                    break
            if len(ids[k]) >= POOL_SIZE:
                break
    for k, v in fg2ims["train"].items():
        if k not in ids:
            if rule == "evaluator":
                break
            continue
        for item_lst in v.values():
            for item_id in item_lst:
                if item_id not in seen[k]:
                    seen[k].add(item_id)
                    ids[k].append(item_id)
                if len(ids[k]) >= POOL_SIZE:
                    break
            if len(ids[k]) >= POOL_SIZE:
                break
    sizes = {k: len(v) for k, v in ids.items()}
    queries = []
    for q in read_json(polyvore / "disjoint" / "fill_in_blank_test.json"):
        gt = q["question"][0].split("_")[0]
        target = next(id2im[a] for a in q["answers"] if a.split("_")[0] == gt)
        fg = str(meta[target]["category_id"])
        if sizes.get(fg, 0) >= POOL_SIZE:
            queries.append((gt, target, fg))
    return queries, sizes


def proxy_checks(labels: list[dict]) -> list[list[str]]:
    """P12 cell 6 (archived A16): range and label checks of the query labels."""
    n = len(labels)
    temps = [r["temperature_c"] for r in labels if r["temperature_c"] is not None]
    weather = Counter(r["weather_group"] for r in labels)
    occasion = Counter(r["occasion_group"] for r in labels)
    style_groups = Counter(r["style_group"] for r in labels)
    styles = [r["style_count"] for r in labels]

    def groups(counts: Counter, allowed: tuple[str, ...]) -> tuple[int, str]:
        valid = sum(counts[g] for g in allowed)
        return valid, "; ".join(f"{k}:{v}" for k, v in sorted(counts.items()))

    w_valid, w_range = groups(weather, ("cold", "warm"))
    o_valid, o_range = groups(occasion, ("formal", "casual", "unknown"))
    return [
        ["temperature_c", n, len(temps), n - len(temps), 0,
         f"{min(temps):.2f} to {max(temps):.2f}" if temps else "", f"{mean(temps):.4f}" if temps else "",
         sum(1 for t in temps if t < -30 or t > 50), "-30 to 50 Celsius sanity range.",
         "Non-numeric or missing values are counted, not silently imputed."],
        ["weather_group", n, w_valid, 0, n - w_valid, w_range, "", 0, "Allowed groups: cold, warm.",
         "Unexpected labels are counted as invalid."],
        ["occasion_group", n, o_valid, 0, n - o_valid, o_range, "", 0, "Allowed groups: formal, casual, unknown.",
         "Unknown is retained as an explicit category rather than guessed from images."],
        ["style_count/style_group", n, len(styles), 0, 0,
         f"style_count {min(styles)} to {max(styles)}; " + "; ".join(f"{k}:{v}" for k, v in sorted(style_groups.items())),
         f"{mean(styles):.4f}", sum(1 for s in styles if s < 0 or s > 20), "Style term count must be non-negative.",
         "Numeric parsing failures are counted as invalid."],
    ]


A16_HEADER = ["proxy", "n_rows", "valid_count", "missing_count", "invalid_count", "observed_range", "mean",
              "out_of_reasonable_range_count", "reasonable_range_rule", "error_handling_note"]


def label_text(row: dict, field: str) -> str:
    value = row[field]
    if field == "temperature_c":
        return "" if value is None else repr(float(value))
    return str(value)


def read_archived_t18() -> list[dict]:
    with ARCHIVED_T18.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def compare_labels(labels: dict[str, dict], archived: list[dict]) -> tuple[list[list], bool]:
    """Field-by-field comparison with the labels of the archived T18 (its original-model rows)."""
    old = [r for r in archived if r["run_tag"] == BASELINE]
    old_keys = [r["metadata_key"] for r in old]
    rows = [["queries (metadata_key)", len(labels), len(old), sum(1 for k in old_keys if k in labels),
             "yes" if list(labels) == old_keys else "NO", ""]]
    all_equal = list(labels) == old_keys
    for field in LABEL_FIELDS:
        equal, first = 0, ""
        for r in old:
            mine = labels.get(r["metadata_key"])
            now = label_text(mine, field) if mine else ""
            same = now == r[field] or (field == "temperature_c" and now and r[field]
                                       and abs(float(now) - float(r[field])) < 1e-9)
            equal += same
            if not same and not first:
                first = f"{r['metadata_key']}: {now!r} vs {r[field]!r}"
        rows.append([field, len(old), len(old), equal, "yes" if equal == len(old) else "NO", first])
        all_equal &= equal == len(old)
    return rows, all_equal


# ---------------------------------------------------------------- P04 sections 4-5: evaluation


class Evaluator:
    """The archived dataset and model classes from a working copy of the archived code (as counterfactual.py)."""

    def __init__(self, polyvore: Path, device: str, work: Path):
        src = work / "main_hybrid_attention_code"
        shutil.copytree(MODEL_CODE, src)
        standard_decoder.apply(src / "outfit_transformer.py")  # changes the CP decoder only, as in the full run
        if device == "cpu":
            cfg = src / "config" / "base_config.py"
            text = cfg.read_text(encoding="utf-8")
            switched = text.replace("device_type = 'cuda'", "device_type = 'cpu'", 1)
            if switched == text:
                raise SystemExit("[RELIABILITY BLOCKED] could not switch the working-copy config to CPU")
            cfg.write_text(switched, encoding="utf-8")
        (work / "polyvore_data").mkdir()
        (work / "polyvore_data" / "polyvore_outfits").symlink_to(polyvore, target_is_directory=True)
        sys.path.insert(0, str(src))
        previous = os.getcwd()
        os.chdir(work)  # the archived config creates runs/<name> relative to the working directory
        try:
            import torch
            import torchvision
            from dataset import UIUCPolyvoreRetrievalDataset
            from outfit_transformer import LinearImageEncoder, LinearTextEncoder, OutfitTransformerRetrieval
            import config.cir_cond_hardneg as cfg
        finally:
            os.chdir(previous)
        self.torch, self.cfg, self.device, self.work = torch, cfg, device, work
        self.dataset_class = UIUCPolyvoreRetrievalDataset
        ptdtype = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[cfg.dtype]
        self.ctx = nullcontext() if device == "cpu" else torch.amp.autocast(device_type="cuda", dtype=ptdtype)
        self.precision = "float32" if device == "cpu" else f"{cfg.dtype} autocast"
        normalize = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.transform = torchvision.transforms.Compose([  # P04 build_transform; the features are precomputed
            torchvision.transforms.Resize((cfg.img_size, cfg.img_size)), torchvision.transforms.ToTensor(), normalize])
        self.make_model = lambda: OutfitTransformerRetrieval(
            img_encoder=LinearImageEncoder(img_inp_size=cfg.img_inp_size, img_emb_size=cfg.img_emb_size),
            text_encoder=LinearTextEncoder(txt_inp_size=cfg.txt_inp_size, txt_emb_size=cfg.txt_emb_size),
            outfit_txt_encoder=LinearTextEncoder(txt_inp_size=cfg.outfit_txt_inp_size,
                                                 txt_emb_size=cfg.txt_emb_size + cfg.img_emb_size),
            nhead=cfg.nhead, num_layers=cfg.num_layers, margin=cfg.margin, target_item_info="category",
            use_outfit_txt=cfg.use_outfit_txt)

    def datasets(self, outfit_feature: str, splits: tuple[str, ...] = ("train", "test")):
        common = dict(datadir=str(self.work / "polyvore_data"),
                      txt_feat_path=str(FEATURES / "encoded_title_description_distiluse-base-multilingual-cased-v2.pkl"),
                      img_feat_path=str(FEATURES / "img_feats_fashionClip.pkl"),
                      category_feat_path=str(FEATURES / "encoded_category_distiluse-base-multilingual-cased-v2.pkl"),
                      outfit_feat_path=str(FEATURES / outfit_feature), polyvore_split="disjoint",
                      max_item_len=self.cfg.max_item_len, transform=self.transform, sample_hard_negative=False)
        return tuple(self.dataset_class(split=split, **common) for split in splits)

    def distractors(self, train_dat, test_dat, model):
        """P04 build_distractors."""
        torch = self.torch
        ids, feats, seen = {}, {}, {}

        def add(k, item_id):
            seen[k].add(item_id)
            ids[k].append(item_id)
            feats[k].append(test_dat.extract_emb({"item_id": item_id}, model, self.device, self.ctx, to_numpy=True))

        for k, v in test_dat.fg2ims.items():
            ids[k], feats[k], seen[k] = [], [], set()
            for item_lst in v.values():
                for item_id in item_lst:
                    if item_id not in seen[k]:
                        add(k, item_id)
                    if len(ids[k]) >= POOL_SIZE:
                        break
                if len(ids[k]) >= POOL_SIZE:
                    break
        for k, v in train_dat.fg2ims.items():
            if k not in ids:
                continue
            for item_lst in v.values():
                for item_id in item_lst:
                    if item_id not in seen[k]:
                        add(k, item_id)
                    if len(ids[k]) >= POOL_SIZE:
                        break
                if len(ids[k]) >= POOL_SIZE:
                    break
        return ({k: np.array(v) for k, v in ids.items() if len(v) >= POOL_SIZE},
                {k: torch.tensor(np.concatenate(v)).float() for k, v in feats.items() if len(v) >= POOL_SIZE})

    def evaluate(self, run_tag: str, checkpoint: Path, max_queries: int | None) -> list[dict]:
        """P04 run_model_evaluation for one model."""
        torch = self.torch
        torch.manual_seed(SEED)
        np.random.seed(SEED)
        train_dat, test_dat = self.datasets(OUTFIT_FEATURES[run_tag])
        model = self.make_model()
        model.load_state_dict(torch.load(str(checkpoint), map_location=self.device)["model"])
        model = model.to(self.device).eval()
        dist_ids, dist_feats = self.distractors(train_dat, test_dat, model)
        print(f"[RELIABILITY] {run_tag}: {len(dist_feats)} categories with {POOL_SIZE} candidates", flush=True)
        rows = []
        for outfits, is_correct, set_id in test_dat.fitb_questions:
            correct = np.where(np.array(is_correct))[0]
            if len(correct) == 0:
                continue
            target = outfits[int(correct[0])]["items"][-1]["item_id"]
            fg = test_dat.im2fg[target]
            if fg not in dist_feats:
                continue
            partial = outfits[0]["items"][:-1]
            imgs, txts, _ = test_dat.load_outfit({"items": partial})
            imgs, txts, mask = test_dat.pad_imgs_and_txts(imgs, txts)
            category = torch.tensor(test_dat.category_feats[target]).float().to(self.device)
            outfit_txt = torch.tensor(test_dat.outfit_feats[set_id]).float().to(self.device)
            with torch.no_grad(), self.ctx:
                out = model(imgs.to(self.device).unsqueeze(0), txts.to(self.device).unsqueeze(0),
                            mask.to(self.device).unsqueeze(0), positive_category=category.unsqueeze(0),
                            outfit_txt=outfit_txt.unsqueeze(0))
                logits = out["logits"].detach().cpu().float()
            cur_ids, cur_feats = dist_ids[fg], dist_feats[fg]
            if not np.any(cur_ids == target):
                cur_ids = np.concatenate([cur_ids[:-1], np.array([target])])
                emb = test_dat.extract_emb({"item_id": target}, model, self.device, self.ctx,
                                           to_numpy=False).detach().cpu().float()
                cur_feats = torch.cat([cur_feats[:-1, :], emb], dim=0)
            score = torch.cosine_similarity(logits.float(), cur_feats.float())  # P04 retrieve_sorted
            values, indices = torch.sort(score, descending=True)
            sorted_ids = cur_ids[indices.numpy()]
            sorted_scores = values.detach().cpu().numpy()
            pos = np.where(sorted_ids == target)[0]
            if len(pos) == 0:
                continue
            rank = int(pos[0]) + 1
            top1, top2 = float(sorted_scores[0]), float(sorted_scores[1])
            gap = float(top1 - top2)
            conf = sigmoid(GAP_SCALE * gap)  # P04 compute_confidence, CONFIDENCE_METHOD = "gap_sigmoid"
            hits = {k: int(rank <= k) for k in K_LIST}
            key = f"{SEED}||{set_id}||{target}"
            rows.append({
                "run_tag": run_tag, "query_key": f"{run_tag}||{key}", "metadata_key": key, "seed": str(SEED),
                "set_id": str(set_id), "model_name": MODEL_NAMES[run_tag],
                "model_output_top1_id": str(sorted_ids[0]),
                "model_output_top10_ids": json.dumps([str(x) for x in sorted_ids[:10]], ensure_ascii=False),
                "confidence": conf, "confidence_band": confidence_band(conf),
                "confidence_source": "gap_sigmoid: retrieval score proxy", "correct_answer": str(target),
                "rank": rank, **{f"hit@{k}": hits[k] for k in K_LIST}, "is_error": int(hits[10] == 0),
                "is_high_conf_error": int(conf >= HIGH_CONF_THRESHOLD and hits[10] == 0),
                "target_item_id": str(target), "target_item_fg": str(fg), "top1_score": top1, "top2_score": top2,
                "target_score": float(sorted_scores[pos[0]]), "top1_top2_gap": gap})
            if max_queries and len(rows) >= max_queries:
                break
        return rows


# ---------------------------------------------------------------- P04 section 7: error type, source, log


def error_type(r: dict) -> str:
    if r["hit@10"] == 1:
        return "correct"
    if r["is_high_conf_error"] == 1:
        return "high-confidence semantic mismatch"
    if r["category_group"] == "accessory-led":
        return "accessory-led retrieval failure"
    if r["style_group"] == "low-style":
        return "low-style ambiguity"
    if r["weather_group"] == "unknown" or r["occasion_group"] == "unknown":
        return "context metadata incomplete"
    return "general retrieval error"


POSSIBLE_SOURCE = {
    "correct": "none",
    "high-confidence semantic mismatch": "模型信心高，但正確答案未進入 Top-10；可能代表 confidence calibration 不佳、語意 "
                                         "shortcut，或候選池中存在高度相似干擾項。",
    "accessory-led retrieval failure": "配件受天氣與場合約束較弱，模型可能較依賴視覺或風格相似度，導致配件類單品排序錯誤。",
    "low-style ambiguity": "輸入描述中的 style fragment 較少，語意條件不足，模型可能用少量 token 做過度推論。",
    "context metadata incomplete": "天氣或場合資訊未被解析，導致模型失敗來源較難診斷。",
    "general retrieval error": "候選單品相似度高，或模型未充分捕捉 partial outfit 與 target item 的細節相容性。",
}


def risk_level(r: dict) -> str:
    if r["hit@10"] == 1:
        return "low"
    if r["is_high_conf_error"] == 1:
        return "medium-high" if r["occasion_group"] == "formal" else "medium"
    return "low-medium"


def observable_log(r: dict) -> str:
    def f(x):
        return float("nan") if x is None else float(x)
    log = {k: str(r[k]) for k in ("run_tag", "model_name", "query_key", "metadata_key", "set_id", "target_item_id",
                                  "correct_answer", "model_output_top1_id")}
    log.update({"model_output_top10_ids": json.loads(r["model_output_top10_ids"]), "rank": f(r["rank"]),
                "hit_at_10": int(r["hit@10"]), "confidence": f(r["confidence"]),
                "confidence_band": str(r["confidence_band"]),
                **{k: f(r[k]) for k in ("top1_score", "top2_score", "target_score", "top1_top2_gap")},
                **{k: str(r[k]) for k in ("target_item_fg", "fine_category", "major_category")},
                "temperature_c": f(r["temperature_c"]), "weather_group": str(r["weather_group"]),
                "occasion_group": str(r["occasion_group"]), "style_count": f(r["style_count"]),
                **{k: str(r[k]) for k in ("style_group", "category_group", "weather_text", "occasion_text",
                                          "style_text", "error_type", "possible_source", "risk_level")}})
    return json.dumps(log, ensure_ascii=False)


def enrich(rows: list[dict], polyvore: Path) -> list[dict]:
    """P04 enrich_metadata plus the error columns; the labels use the queries the models evaluated."""
    queries = list(dict.fromkeys((r["set_id"], r["target_item_id"], r["target_item_fg"]) for r in rows))
    labels = query_labels(queries, polyvore)
    for r in rows:
        r.update({k: v for k, v in labels[r["metadata_key"]].items() if k not in r})
        r["error_type"] = error_type(r)
        r["possible_source"] = POSSIBLE_SOURCE[r["error_type"]]
        r["risk_level"] = risk_level(r)
        r["observable_log"] = observable_log(r)
    return rows


def cell(value) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return "" if math.isnan(value) else repr(value)
    return str(value)


def write_required_fields(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:  # P04 wrote utf-8-sig with pandas
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow(REQUIRED_COLUMNS)
        writer.writerows([[cell(r[c]) for c in REQUIRED_COLUMNS] for r in rows])


# ---------------------------------------------------------------- P04 section 8: metrics and tables


def calibration(rows: list[dict]) -> tuple[float, list[dict]]:
    """P04 compute_ece: 10 equal-width bins of the confidence (right-closed, the first includes 0)."""
    edges = np.linspace(0, 1, N_BINS + 1)
    conf = np.clip(np.array([r["confidence"] for r in rows], dtype=float), 0, 1)
    hit = np.array([r["hit@10"] for r in rows], dtype=float)
    idx = np.maximum(np.searchsorted(edges, conf, side="left") - 1, 0)
    ece, bins = 0.0, []
    for b in range(N_BINS):
        sel = idx == b
        if not sel.any():
            continue
        avg_conf, acc = float(conf[sel].mean()), float(hit[sel].mean())
        ece += sel.sum() / len(conf) * abs(avg_conf - acc)
        bins.append({"bin": f"({edges[b]:.1f}, {edges[b + 1]:.1f}]", "n": int(sel.sum()), "avg_confidence": avg_conf,
                     "hit10_accuracy": acc, "gap": abs(avg_conf - acc)})
    return float(ece), bins


def summarize(rows: list[dict]) -> dict:
    """P04 summarize_one_run."""
    def avg(key):
        return float(np.mean(np.array([r[key] for r in rows], dtype=float)))
    conf = np.clip(np.array([r["confidence"] for r in rows], dtype=float), 0, 1)
    hit = np.array([r["hit@10"] for r in rows], dtype=float)
    return {"n": len(rows), "Hit@1": avg("hit@1"), "Hit@10": avg("hit@10"),
            "Median Rank": float(np.median([r["rank"] for r in rows])), "Mean Rank": avg("rank"),
            "Avg Confidence": avg("confidence"), "Error Rate": avg("is_error"),
            "High-conf Error Rate": avg("is_high_conf_error"), "ECE": calibration(rows)[0],
            "Brier Score": float(np.mean((conf - hit) ** 2))}


def as_number(r: dict) -> dict:
    """Archived T18 rows as numbers."""
    out = dict(r)
    for k in ("rank", "hit@1", "hit@3", "hit@5", "hit@10", "hit@30", "hit@50", "is_error", "is_high_conf_error",
              "style_count"):
        out[k] = int(r[k])
    for k in ("confidence", "top1_score", "top2_score", "target_score", "top1_top2_gap"):
        out[k] = float(r[k])
    out["temperature_c"] = float(r["temperature_c"]) if r["temperature_c"] else None
    return out


def subgroup_metrics(by_run: dict[str, list[dict]]) -> list[dict]:
    """P04 make_comparison_plots: subgroup table (unknown dropped when a known group exists)."""
    out = []
    for run_tag in sorted(by_run):
        for col in SUBGROUP_COLUMNS:
            groups = defaultdict(list)
            for r in by_run[run_tag]:
                groups[str(r[col])].append(r)
            keys = sorted(groups)
            if any(k != "unknown" for k in keys):
                keys = [k for k in keys if k != "unknown"]
            for k in keys:
                g = groups[k]
                out.append({"run_tag": run_tag, "subgroup_type": col, "subgroup": k, "n": len(g),
                            "hit10": mean(r["hit@10"] for r in g), "avg_confidence": float(np.mean([r["confidence"] for r in g])),
                            "high_conf_error_rate": mean(r["is_high_conf_error"] for r in g),
                            "median_rank": float(np.median([r["rank"] for r in g]))})
    return out


def performance_rows(perf: dict[str, dict]) -> list[list]:
    rows = [[tag, MODEL_NAMES[tag], perf[tag]["n"]] + [f"{perf[tag][m]:.6f}" for m in METRICS] for tag in perf]
    if BASELINE in perf and PROPOSED in perf:
        rows.append(["Δ proposed - original", "", ""] + [f"{perf[PROPOSED][m] - perf[BASELINE][m]:+.6f}"
                                                         for m in METRICS])
    return rows


def pct(x: float) -> str:
    return f"{x * 100:.2f}%"


# ---------------------------------------------------------------- figures (P04 plot_01, 03, 05, 06, 08)


def reliability_diagram(path: Path, title: str, subtitle: str, series: list[tuple[str, list, str]]) -> None:
    c = Chart(600, 600, title, subtitle)
    p = c.panel(90, 70, 440, 440)
    p.set_x(0, 1)
    p.set_y(0, 1)
    ticks = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
    p.axes(xticks=ticks, yticks=ticks, xlabel="Average confidence", ylabel="Empirical Hit@10 accuracy", grid="both")
    c.line(p.xp(0), p.yp(0), p.xp(1), p.yp(1), "#555555", 1.5, "6,4")
    for _, pts, color in series:
        p.line([x for x, _ in pts], [y for _, y in pts], color, 2)
        for x, y in pts:
            c.circle(p.xp(x), p.yp(y), 4, color)
    c.legend(p.left + 16, p.top + 18, [(name, color, "line") for name, _, color in series]
             + [("Perfect calibration", "#555555", "dash")])
    c.save(path)


def draw_figures(out: Path, run_label: str, perf: dict, bins: dict, errors: dict, subgroups: list[dict]) -> list[str]:
    figs = out / "figures"
    tags = [t for t in (BASELINE, PROPOSED) if t in perf]
    subtitle = f"{run_label}, seed {SEED}"
    names = []
    header = ["n", "Hit@1", "Hit@10", "Median Rank", "Mean Rank", "Avg Confidence", "Error Rate",
              "High-conf Error Rate", "ECE", "Brier Score", "run_tag"]
    rows = []
    for tag in tags:
        p = perf[tag]
        rows.append([str(p["n"]), pct(p["Hit@1"]), pct(p["Hit@10"]), f"{p['Median Rank']:.3f}", f"{p['Mean Rank']:.3f}",
                     f"{p['Avg Confidence']:.3f}", pct(p["Error Rate"]), pct(p["High-conf Error Rate"]),
                     f"{p['ECE']:.3f}", f"{p['Brier Score']:.3f}", tag])
    if len(tags) == 2:
        d = {m: perf[PROPOSED][m] - perf[BASELINE][m] for m in METRICS}
        rows.append(["", pct(d["Hit@1"]), pct(d["Hit@10"]), f"{d['Median Rank']:.3f}", f"{d['Mean Rank']:.3f}",
                     f"{d['Avg Confidence']:.3f}", pct(d["Error Rate"]), pct(d["High-conf Error Rate"]),
                     f"{d['ECE']:.3f}", f"{d['Brier Score']:.3f}", "Δ proposed - original"])
    table_chart(figs / "figure_F12_performance_comparison.svg", "Performance Comparison: Original vs Proposed",
                header, rows, subtitle=subtitle)
    names.append("figure_F12_performance_comparison.svg")
    reliability_diagram(figs / "figure_F13_reliability_diagram.svg", "Reliability Diagram: Original vs Proposed",
                        subtitle, [(f"{t}, ECE={perf[t]['ECE']:.3f}",
                                    [(b["avg_confidence"], b["hit10_accuracy"]) for b in bins[t]], COLORS[t])
                                   for t in tags])
    names.append("figure_F13_reliability_diagram.svg")
    types = sorted({k for t in tags for k in errors[t]})
    grouped_vbar_chart(figs / "figure_F14_error_types.svg", "Error Slicing by Error Type: Original vs Proposed",
                       ["\n".join(textwrap.wrap(k, 18)) for k in types],
                       [(t, [errors[t].get(k, 0) for k in types], COLORS[t]) for t in tags], "Number of errors",
                       value_fmt="{:.0f}", subtitle=subtitle, legend_at="upper right")
    names.append("figure_F14_error_types.svg")
    labels = sorted({(s["subgroup_type"], s["subgroup"]) for s in subgroups})
    value = {(s["run_tag"], s["subgroup_type"], s["subgroup"]): s["hit10"] for s in subgroups}
    grouped_vbar_chart(figs / "figure_F15_subgroup_hit10.svg", "Subgroup Metrics: Hit@10, Original vs Proposed",
                       [f"{a}\n{b}" for a, b in labels],
                       [(t, [value.get((t, a, b), 0.0) for a, b in labels], COLORS[t]) for t in tags], "Hit@10",
                       value_fmt="{:.3f}", subtitle=subtitle)
    names.append("figure_F15_subgroup_hit10.svg")
    if len(tags) == 2:
        both = [(a, b) for a, b in labels if (BASELINE, a, b) in value and (PROPOSED, a, b) in value]
        columns = list(dict.fromkeys(b for _, b in both))
        order = [c for c in SUBGROUP_COLUMNS if any(a == c for a, _ in both)]
        delta = {(a, b): value[(PROPOSED, a, b)] - value[(BASELINE, a, b)] for a, b in both}
        matrix = [[delta.get((a, b)) for b in columns] for a in order]
        vals = list(delta.values())
        heatmap_chart(figs / "figure_F16_subgroup_delta_hit10.svg", "Subgroup ΔHit@10 Heatmap: Proposed - Original",
                      order, columns, matrix, min(vals), max(vals), value_fmt="{:+.3f}", cmap=VIRIDIS,
                      colorbar_label="ΔHit@10 = proposed - original", subtitle=subtitle)
        names.append("figure_F16_subgroup_delta_hit10.svg")
    return names


# ---------------------------------------------------------------- main


def analyse(out: Path, rows: list[dict], run_label: str) -> dict:
    """Writes the T18-layout file and the tables and figures of P04 make_comparison_plots."""
    write_required_fields(out / "reliability_required_fields.csv", rows)
    by_run = defaultdict(list)
    for r in rows:
        by_run[r["run_tag"]].append(r)
    perf = {t: summarize(by_run[t]) for t in (BASELINE, PROPOSED) if t in by_run}
    write_csv(out / "performance.csv", ["run_tag", "model_name", "n", *METRICS], performance_rows(perf))
    bins = {t: calibration(by_run[t])[1] for t in perf}
    write_csv(out / "calibration_bins.csv", ["run_tag", "bin", "n", "avg_confidence", "hit10_accuracy", "gap"],
              [[t, b["bin"], b["n"], f"{b['avg_confidence']:.6f}", f"{b['hit10_accuracy']:.6f}", f"{b['gap']:.6f}"]
               for t in perf for b in bins[t]])
    bands = defaultdict(dict)
    for t in perf:
        groups = defaultdict(list)
        for r in by_run[t]:
            groups[r["confidence_band"]].append(r)
        for band, g in groups.items():
            bands[band][t] = g
    write_csv(out / "confidence_bands.csv", ["confidence_band", "run_tag", "n", "hit10", "error_rate",
                                             "high_conf_error_rate", "median_rank"],
              [[band, t, len(g), f"{mean(r['hit@10'] for r in g):.6f}", f"{mean(r['is_error'] for r in g):.6f}",
                f"{mean(r['is_high_conf_error'] for r in g):.6f}", f"{float(np.median([r['rank'] for r in g])):.1f}"]
               for band in sorted(bands) for t, g in sorted(bands[band].items())])
    errors = {t: Counter(r["error_type"] for r in by_run[t] if r["is_error"] == 1) for t in perf}
    write_csv(out / "error_types.csv", ["error_type", *perf],
              [[k, *[errors[t].get(k, 0) for t in perf]] for k in sorted({k for t in perf for k in errors[t]})])
    subgroups = subgroup_metrics({t: by_run[t] for t in perf})
    write_csv(out / "subgroup_metrics.csv", ["run_tag", "subgroup_type", "subgroup", "n", "hit10", "avg_confidence",
                                             "high_conf_error_rate", "median_rank"],
              [[s["run_tag"], s["subgroup_type"], s["subgroup"], s["n"], f"{s['hit10']:.6f}",
                f"{s['avg_confidence']:.6f}", f"{s['high_conf_error_rate']:.6f}", f"{s['median_rank']:.1f}"]
               for s in subgroups])
    examples = []
    if len(perf) == 2:  # P04 plot_09: the Context model's high-confidence errors, highest confidence first
        base = {r["metadata_key"]: r for r in by_run[BASELINE]}
        hce = sorted((r for r in by_run[PROPOSED] if r["is_high_conf_error"] == 1),
                     key=lambda r: (-r["confidence"], -r["rank"]))
        for r in hce[:3]:
            b = base.get(r["metadata_key"])
            examples.append([r["set_id"], r["target_item_id"], r["model_output_top1_id"],
                             b["model_output_top1_id"] if b else "", r["rank"], b["rank"] if b else "",
                             r["rank"] - b["rank"] if b else "", f"{r['confidence']:.6f}",
                             f"{b['confidence']:.6f}" if b else "", r["fine_category"], r["weather_group"],
                             r["occasion_group"], r["style_group"], r["category_group"], r["error_type"]])
        write_csv(out / "high_confidence_error_examples.csv",
                  ["set_id", "target_item_id", "model_output_top1_id", "original_top1_id", "rank", "original_rank",
                   "rank_change_vs_original", "confidence", "original_confidence", "fine_category", "weather_group",
                   "occasion_group", "style_group", "category_group", "error_type"], examples)
    figures = draw_figures(out, run_label, perf, bins, errors, subgroups)
    return {"perf": perf, "errors": errors, "subgroups": subgroups, "figures": figures, "by_run": by_run}


def per_query_agreement(rows: list[dict], archived: list[dict]) -> list[list]:
    """Re-evaluated 2025 models against the archived T18, per model."""
    old = {r["query_key"]: as_number(r) for r in archived}
    out = []
    for tag in (BASELINE, PROPOSED):
        mine = [r for r in rows if r["run_tag"] == tag]
        pairs = [(r, old.get(r["query_key"])) for r in mine]
        found = [(a, b) for a, b in pairs if b is not None]
        diffs = [abs(a["rank"] - b["rank"]) for a, b in found]
        conf = [abs(a["confidence"] - b["confidence"]) for a, b in found]
        out.append([tag, len(mine), sum(1 for r in archived if r["run_tag"] == tag), len(found),
                    sum(1 for d in diffs if d == 0), max(diffs, default=0),
                    sum(1 for a, b in found if a["hit@10"] == b["hit@10"]),
                    sum(1 for a, b in found if a["model_output_top1_id"] == b["model_output_top1_id"]),
                    sum(1 for a, b in found if a["model_output_top10_ids"] == b["model_output_top10_ids"]),
                    f"{max(conf, default=0):.6f}",
                    sum(1 for a, b in found if a["is_high_conf_error"] == b["is_high_conf_error"]),
                    sum(1 for a, b in found if a["error_type"] == b["error_type"])])
    return out


AGREEMENT_HEADER = ["run_tag", "queries", "archived_queries", "matched", "rank_equal", "largest_rank_difference",
                    "hit10_equal", "top1_equal", "top10_list_equal", "largest_confidence_difference",
                    "high_conf_error_flag_equal", "error_type_equal"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--labels-only", action="store_true", help="query labels and their checks only; no model")
    mode.add_argument("--run-root", type=Path, help="completed full run folder with the checkpoints")
    mode.add_argument("--validate-2025", action="store_true",
                      help="evaluate the preserved 2025 seed-1 checkpoints and compare with the archived T18")
    parser.add_argument("--out-dir", type=Path, default=None,
                        help="default: <run>/extensions/reliability; required for the other modes")
    parser.add_argument("--polyvore-root", default=None)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--max-queries", type=int, default=None, help="plumbing test: stop after N queries per model")
    args = parser.parse_args()

    run_root = passed_run(args.run_root) if args.run_root else None
    if args.out_dir is None and run_root is None:
        raise SystemExit("[RELIABILITY BLOCKED] pass --out-dir")
    out = output_dir(args.out_dir or run_root / "extensions" / "reliability", run_root)
    polyvore = local_path("polyvore_root", args.polyvore_root)
    if polyvore is None or not (polyvore / "disjoint" / "test.json").is_file():
        raise SystemExit("[RELIABILITY BLOCKED] Polyvore root not found; pass --polyvore-root or run bootstrap_data.sh")
    archived = read_archived_t18()

    # Query labels from the input data and the pool rule, without a model.
    queries, p04_sizes = pool_queries(polyvore, "p04")
    eval_queries, eval_sizes = pool_queries(polyvore, "evaluator")
    labels = query_labels(queries, polyvore)
    label_rows, labels_equal = compare_labels(labels, archived)
    write_csv(out / "label_check.csv", ["field", "queries", "archived_queries", "equal", "all_equal",
                                        "first_difference"], label_rows)
    a16 = proxy_checks(list(labels.values()))
    write_csv(out / "environment_proxy_checks.csv", A16_HEADER, a16)
    with ARCHIVED_A16.open(encoding="utf-8-sig", newline="") as f:
        a16_archived = [[r[k] for k in A16_HEADER] for r in csv.DictReader(f)]
    a16_equal = [[str(x) for x in r] for r in a16] == a16_archived
    full = lambda sizes: sorted(k for k, n in sizes.items() if n >= POOL_SIZE)  # noqa: E731
    lines = ["# Reliability analysis (2025 notebook P04: table T18, figures F12-F16; A16 from P12)", "",
             "Ported from `03_實驗與結果_experiments_results/00_控制檢查與附加稽核/source_programs/"
             "P04_reliability_xai_analysis.ipynb` (cell 0). Settings as in P04: seed 1, pools of 3,000 candidates,",
             "confidence = sigmoid(50 × (top-1 score − top-2 score)), high-confidence error = Hit@10 of 0 with",
             "confidence ≥ 0.85, ECE over 10 equal-width bins against Hit@10.", "",
             "## Queries and labels (no model)", "",
             f"- P04's pool rule fills {len(full(p04_sizes))} of {len(p04_sizes)} test categories "
             f"({len(queries):,} queries); the archived evaluator's rule fills {len(full(eval_sizes))} "
             f"({len(eval_queries):,} queries). Same categories: "
             f"{'yes' if full(p04_sizes) == full(eval_sizes) else 'NO'}; same queries in the same order: "
             f"{'yes' if queries == eval_queries else 'NO'}.",
             f"- Labels against the archived T18 (its {sum(1 for r in archived if r['run_tag'] == BASELINE):,} "
             f"original-model rows), field by field (`label_check.csv`): "
             f"{'all equal' if labels_equal else 'NOT all equal'}.", ""]
    lines += ["| Field | Equal | All equal |", "|---|---:|---|"]
    lines += [f"| {r[0]} | {r[3]:,} of {r[2]:,} | {r[4]} |" for r in label_rows]
    lines += ["", f"Range checks of the labels (`environment_proxy_checks.csv`, layout of the archived A16): equal to "
                  f"the archived A16: {'yes' if a16_equal else 'NO'}.", "",
              "| Proxy | Valid | Observed range | Mean |", "|---|---:|---|---:|"]
    lines += [f"| {r[0]} | {r[2]:,} of {r[1]:,} | {r[5]} | {r[6]} |" for r in a16]
    lines.append("")
    record = {"mode": "labels_only" if args.labels_only else "validate_2025" if args.validate_2025 else "run",
              "polyvore_queries": len(queries), "labels_equal_archived_t18": labels_equal,
              "a16_equal_archived": a16_equal, "archived_t18_sha256": sha256(ARCHIVED_T18)}

    if not args.labels_only:
        import torch
        torch.set_num_threads(args.threads)
        if args.device == "cuda" and not torch.cuda.is_available():
            raise SystemExit("[RELIABILITY BLOCKED] CUDA is not available; P04 ran on the GPU (or pass --device cpu)")
        if run_root:
            models, run_label = {}, f"Run {run_root.name}"
            for tag, unit in RUN_UNITS.items():
                manifest = json.loads((run_root / "main" / unit / "full_train_manifest.json").read_text(encoding="utf-8"))
                ckpt = run_root / "main" / unit / "cir_best_ckpt.pt"
                if manifest.get("status") != "passed" or sha256(ckpt) != manifest.get("cir_checkpoint_sha256"):
                    raise SystemExit(f"[RELIABILITY BLOCKED] {unit}: not a passed unit or its checkpoint SHA-256 differs")
                models[tag] = ckpt
        else:
            models = {tag: CHECKPOINTS_2025 / name / "ckpt.pt" for tag, name in CHECKPOINTS_2025_SEED1.items()}
            run_label = "Preserved 2025 checkpoints"
        work_root = REPRO / ".work"
        work_root.mkdir(exist_ok=True)
        work = Path(tempfile.mkdtemp(prefix="reliability_", dir=work_root))
        started = time.time()
        try:
            evaluator = Evaluator(polyvore, args.device, work)
            rows = []
            for tag, ckpt in models.items():
                print(f"[RELIABILITY] evaluating {tag}: {ckpt}", flush=True)
                rows += evaluator.evaluate(tag, ckpt, args.max_queries)
            precision = evaluator.precision
        finally:
            shutil.rmtree(work, ignore_errors=True)
        rows = enrich(rows, polyvore)
        result = analyse(out, rows, run_label)
        perf = result["perf"]
        check = bool(args.max_queries) or args.device == "cpu"
        lines += [f"## Models ({run_label}; {args.device}, {precision}{'; check run' if check else ''})", ""]
        if run_root:
            lines += [f"- original = `main/{RUN_UNITS[BASELINE]}`, proposed = `main/{RUN_UNITS[PROPOSED]}` "
                      "(the run tags of P04 and T18 are kept).", ""]
        else:
            lines += ["- original = `cir_old_seed1`, proposed = `cir_new_seed1` from "
                      "`02_模型訓練和驗證_model_training_validation/main_hybrid_attention_checkpoints/`.", ""]
        lines += ["| Model | n | Hit@1 | Hit@10 | Median rank | Mean rank | Mean confidence | High-conf error | ECE | Brier |",
                  "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
        lines += [f"| {t} | {p['n']:,} | {p['Hit@1']:.4f} | {p['Hit@10']:.4f} | {p['Median Rank']:.0f} | "
                  f"{p['Mean Rank']:.2f} | {p['Avg Confidence']:.4f} | {p['High-conf Error Rate']:.4f} | "
                  f"{p['ECE']:.4f} | {p['Brier Score']:.4f} |" for t, p in perf.items()]
        lines.append("")
        errors = result["errors"]
        lines += ["Errors (Hit@10 = 0) by type (`error_types.csv`, F14):", "",
                  "| Error type | " + " | ".join(perf) + " |", "|---|" + "---:|" * len(perf)]
        lines += [f"| {k} | " + " | ".join(f"{errors[t].get(k, 0):,}" for t in perf) + " |"
                  for k in sorted({k for t in perf for k in errors[t]})]
        lines.append("")
        if len(perf) == 2:
            value = {(s["run_tag"], s["subgroup_type"], s["subgroup"]): s for s in result["subgroups"]}
            keys = sorted({(s["subgroup_type"], s["subgroup"]) for s in result["subgroups"]})
            lines += ["Subgroup Hit@10 (`subgroup_metrics.csv`, F15/F16):", "",
                      "| Subgroup | n | Original | Proposed | Difference |", "|---|---:|---:|---:|---:|"]
            for a, b in keys:
                o, p = value.get((BASELINE, a, b)), value.get((PROPOSED, a, b))
                if o and p:
                    lines.append(f"| {a} = {b} | {o['n']:,} | {o['hit10']:.4f} | {p['hit10']:.4f} | "
                                 f"{p['hit10'] - o['hit10']:+.4f} |")
            lines.append("")
        if args.validate_2025:
            agreement = per_query_agreement(rows, archived)
            write_csv(out / "agreement_with_archived_t18.csv", AGREEMENT_HEADER, agreement)
            old_perf = {t: summarize([as_number(r) for r in archived if r["run_tag"] == t]) for t in (BASELINE, PROPOSED)}
            lines += ["## Against the 2025 outputs", "",
                      "Per query against the archived T18 (`agreement_with_archived_t18.csv`):", "",
                      "| Model | Queries | Ranks equal | Largest rank difference | Hit@10 equal | Top-10 lists equal | "
                      "Largest confidence difference | Error type equal |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
            lines += [f"| {a[0]} | {a[1]:,} | {a[4]:,} | {a[5]} | {a[6]:,} | {a[8]:,} | {a[9]} | {a[11]:,} |"
                      for a in agreement]
            lines += ["", "Performance table: P04's printed values, the same metrics computed here from the archived "
                          "T18 rows, and from this re-evaluation:", "",
                      "| Model | Metric | P04 printed | From archived T18 | Re-evaluated |", "|---|---|---:|---:|---:|"]
            for t in (BASELINE, PROPOSED):
                for m in METRICS:
                    lines.append(f"| {t} | {m} | {P04_PRINTED[t][m]:.6f} | {old_perf[t][m]:.6f} | {perf[t][m]:.6f} |")
            lines.append("")
            printed_ok = all(abs(old_perf[t][m] - P04_PRINTED[t][m]) < 5e-7 for t in P04_PRINTED for m in METRICS)
            lines += [f"Metric code against P04's printed table (from the archived T18 rows, six decimals): "
                      f"{'equal' if printed_ok else 'NOT equal'}.", ""]
            record.update({"agreement": [dict(zip(AGREEMENT_HEADER, a)) for a in agreement],
                           "metric_code_equals_p04_printed": printed_ok})
        lines += ["Files: `reliability_required_fields.csv` (T18 layout, not committed), `performance.csv` (F12), "
                  "`calibration_bins.csv` (F13), `confidence_bands.csv`, `error_types.csv` (F14), "
                  "`subgroup_metrics.csv` (F15, F16), `high_confidence_error_examples.csv`, "
                  f"`figures/` ({', '.join(result['figures'])}).", ""]
        def where(path: Path) -> str:
            if run_root and path.is_relative_to(run_root):
                return f"<run>/{path.relative_to(run_root)}"
            return str(path.relative_to(REPO)) if path.is_relative_to(REPO) else path.name
        record.update({"device": args.device, "precision": precision, "threads": args.threads,
                       "max_queries": args.max_queries, "status": "check_only" if check else "complete",
                       "run": run_root.name if run_root else None, "seconds": round(time.time() - started),
                       "torch": torch.__version__,
                       "cuda_device": torch.cuda.get_device_name(0) if args.device == "cuda" else None,
                       "models": {t: {"checkpoint": where(p), "sha256": sha256(p)} for t, p in models.items()},
                       "queries": {t: len(v) for t, v in result["by_run"].items()},
                       "performance": {t: {m: v for m, v in p.items()} for t, p in perf.items()}})
    write_text(out / "summary.md", lines)
    write_json(out / "manifest.json", record)
    print("\n".join(lines))


if __name__ == "__main__":
    main()
