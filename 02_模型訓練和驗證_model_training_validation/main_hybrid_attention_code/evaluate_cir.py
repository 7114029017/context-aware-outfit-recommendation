import torch
from dataset import UIUCPolyvoreRetrievalDataset
from outfit_transformer import (
    OutfitTransformerRetrieval, 
    LinearImageEncoder,
    LinearTextEncoder,
    MlpImageEncoder,
    MlpTextEncoder
)
import argparse
import torchvision
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
from sklearn.metrics import roc_auc_score
import numpy as np
from tqdm import tqdm
import time
import os
import random
import math
from torchvision.ops.focal_loss import sigmoid_focal_loss
from pathlib import Path
import torch.nn.functional as F
import pickle
from contextlib import nullcontext
from copy import deepcopy
import inspect
import json 

# Change to other config for different settings
from config.cir_cond_hardneg import *
import config.cir_cond_hardneg as cfg
import argparse
import csv

parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=1)
parser.add_argument("--exp_root", type=str, default="./experiments")

# 新增：對齊 train_cir.py
parser.add_argument(
    "--variant",
    type=str,
    default="outfitUrlTitle",
    choices=["outfitUrlTitle", "NewoutfitUrlTitle", "no_weather", "no_occasion", "no_style"]
)
parser.add_argument(
    "--outfit_feat_name",
    type=str,
    default="encoded_outfitUrlTitle_en_fashionClip.pkl"
)
parser.add_argument(
    "--subset_ids_path",
    type=str,
    default=None,
    help="txt 檔，一行一個 set_id；若不提供則使用完整 split"
)

parser.add_argument("--save_detail", action="store_true")
parser.add_argument("--detail_file", type=str, default=None)  # 不給就用預設命名
parser.add_argument("--run_tag", type=str, default="eval")    # 可自行指定，例如 ablation

args = parser.parse_args()

seed = args.seed
exp_root = args.exp_root
variant = args.variant
outfit_feat_name = args.outfit_feat_name
subset_ids_path = args.subset_ids_path

subset_tag = "subset" if subset_ids_path else "all"

exp_dir = os.path.join(exp_root, f"cir_{variant}_{subset_tag}_seed{seed}")
print(f"[CIR EVAL] seed={seed}, variant={variant}, subset_tag={subset_tag}, exp_dir={exp_dir}")

def load_subset_ids(path):
    if path is None:
        return None
    ids = set()
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            x = line.strip()
            if x:
                ids.add(x)
    print(f"[subset] loaded {len(ids)} ids from {path}")
    return ids


def apply_subset_filter_inplace(dat, subset_ids, filter_fg2ims=False):
    """
    最小侵入式過濾（對齊 train_*）
    1) dat.data 只留 subset set_id
    2) compatibility_questions / fitb_questions 同步過濾（若存在）
    3) CIR 可選擇過濾 fg2ims（讓 recall@k 的 distractor pool 也對齊 subset）
    """
    if subset_ids is None:
        return dat

    before_n = len(dat.data)
    dat.data = [o for o in dat.data if str(o["set_id"]) in subset_ids]
    after_n = len(dat.data)

    if hasattr(dat, "compatibility_questions"):
        before_q = len(dat.compatibility_questions)
        dat.compatibility_questions = [
            q for q in dat.compatibility_questions if str(q[2]) in subset_ids
        ]
        after_q = len(dat.compatibility_questions)
    else:
        before_q = after_q = None

    if hasattr(dat, "fitb_questions"):
        before_f = len(dat.fitb_questions)
        dat.fitb_questions = [
            q for q in dat.fitb_questions if str(q[2]) in subset_ids
        ]
        after_f = len(dat.fitb_questions)
    else:
        before_f = after_f = None

    msg = f"[subset] split={getattr(dat, 'split', 'unknown')} data: {before_n} -> {after_n}"
    if before_q is not None:
        msg += f", compatibility_questions: {before_q} -> {after_q}"
    if before_f is not None:
        msg += f", fitb_questions: {before_f} -> {after_f}"

    # CIR recall@k 會用 fg2ims 建 distractor pool，建議同步對齊 subset
    if filter_fg2ims and hasattr(dat, "fg2ims") and isinstance(dat.fg2ims, dict):
        before_fg_pairs = sum(len(v) for v in dat.fg2ims.values())
        new_fg2ims = {}
        for fg, sid2items in dat.fg2ims.items():
            if not isinstance(sid2items, dict):
                continue
            kept = {sid: items for sid, items in sid2items.items() if str(sid) in subset_ids}
            if len(kept) > 0:
                new_fg2ims[fg] = kept
        dat.fg2ims = new_fg2ims
        after_fg_pairs = sum(len(v) for v in dat.fg2ims.values())
        msg += f", fg2ims-sets: {before_fg_pairs} -> {after_fg_pairs}"

    print(msg)
    return dat

# print config parameters
for x in dir(cfg):
    if x.startswith('__'):
        continue
    if inspect.ismodule(cfg.__dict__.get(x)):
        continue
    if isinstance(cfg.__dict__.get(x), type):
        continue
    print('{:<40}{:<40}'.format(x, repr(cfg.__dict__.get(x))))

# note: float16 data type will automatically use a GradScaler
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# +
# data path setting
current_dir = os.getcwd()

# 設定上一層目錄 (workspace)
parent_dir = os.path.dirname(current_dir)

# A. 設定特徵檔資料夾 (fashionclip_data)
feat_dir = os.path.join(parent_dir, 'fashionclip_data')

# B. 設定資料集資料夾 (polyvore_data/polyvore_outfit)
# 注意：程式碼通常會去讀取 {datadir}/{split}/train.json
# 所以 datadir 應該指向含有 'disjoint' 資料夾的那個父目錄
data_root = os.path.join(parent_dir, 'polyvore_data', 'polyvore_outfits')

# C. 設定具體的檔案路徑
img_feat_path = os.path.join(feat_dir, 'img_feats_fashionClip.pkl')
txt_feat_path = os.path.join(feat_dir, 'encoded_title_description_distiluse-base-multilingual-cased-v2.pkl')
category_feat_path = os.path.join(feat_dir, 'encoded_category_distiluse-base-multilingual-cased-v2.pkl')

# 新生成的特徵
outfit_feat_path = os.path.join(feat_dir, outfit_feat_name)
print(f"[eval-cir] variant={variant}")
print(f"[eval-cir] outfit_feat_name={outfit_feat_name}")
print(f"[eval-cir] outfit_feat_path={outfit_feat_path}")
print(f"[eval-cir] subset_ids_path={subset_ids_path}")

# path checking
print(f"檢查資料路徑結構...")

# 檢查 Split 設定
polyvore_split = 'disjoint'  # 您的目標
target_split_dir = os.path.join(data_root, polyvore_split)

if os.path.exists(target_split_dir):
    print(f"成功找到 Disjoint 資料夾: {target_split_dir}")
    final_datadir = os.path.join(parent_dir, 'polyvore_data')
    print(final_datadir)

# 檢查特徵檔是否存在
required_files = [img_feat_path, txt_feat_path, outfit_feat_path]
for f in required_files:
    if os.path.exists(f):
        print(f"特徵檔存在: {os.path.basename(f)}")
    else:
        print(f"錯誤：找不到特徵檔 {f}")

# dataloader
normalize = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                std=[0.229, 0.224, 0.225])
transform = torchvision.transforms.Compose([
    torchvision.transforms.Resize((img_size, img_size)),
    torchvision.transforms.ToTensor(),
    normalize
])
train_dat = UIUCPolyvoreRetrievalDataset(
        datadir=final_datadir,
        txt_feat_path=txt_feat_path,
        img_feat_path=img_feat_path,
        category_feat_path=category_feat_path,
        outfit_feat_path=outfit_feat_path,
        polyvore_split=polyvore_split,
        split='train',
        max_item_len=max_item_len,
        transform=transform,
        sample_hard_negative=False
    )
train_loader = DataLoader(train_dat, batch_size=batch_size, shuffle=False)
test_dat = UIUCPolyvoreRetrievalDataset(
        datadir=final_datadir,
        txt_feat_path=txt_feat_path,
        img_feat_path=img_feat_path,
        category_feat_path=category_feat_path,
        outfit_feat_path=outfit_feat_path,
        polyvore_split=polyvore_split,
        split='test',
        max_item_len=max_item_len,
        transform=transform,
        sample_hard_negative=False,
    )
test_loader = DataLoader(test_dat, batch_size=batch_size, shuffle=False)
print('data load complete')

subset_ids = load_subset_ids(subset_ids_path)
if subset_ids is not None:
    # CIR 的 recall@k 會用到 fg2ims，建議 filter_fg2ims=True
    apply_subset_filter_inplace(train_dat, subset_ids, filter_fg2ims=True)
    apply_subset_filter_inplace(test_dat, subset_ids, filter_fg2ims=True)
else:
    print("[subset] 未啟用 subset filter（使用完整 split）")
    
model = OutfitTransformerRetrieval(
    img_encoder=LinearImageEncoder(img_inp_size=img_inp_size, img_emb_size=img_emb_size),
    text_encoder=LinearTextEncoder(txt_inp_size=txt_inp_size, txt_emb_size=txt_emb_size),
    outfit_txt_encoder=LinearTextEncoder(txt_inp_size=outfit_txt_inp_size, txt_emb_size=txt_emb_size+img_emb_size) if use_outfit_txt else None,
    nhead=nhead,
    num_layers=num_layers,
    margin=margin,
    target_item_info='category',
    use_outfit_txt=use_outfit_txt
)
'''
ckpt_path = os.path.join(out_dir, "ckpt.pt")
checkpoint = torch.load(ckpt_path, map_location=device)
'''
ckpt_path = os.path.join(exp_dir, "ckpt.pt")
print(f"Loading checkpoint from: {ckpt_path}")
checkpoint = torch.load(ckpt_path, map_location=device)

state_dict = checkpoint['model']
model.load_state_dict(state_dict)
start_epoch = checkpoint['epoch']
best_acc = checkpoint.get('best_acc', -1)
model = model.to(device)
print(f"Loaded checkpoint from {exp_dir}, best_acc: {best_acc:.4f}, epoch: {start_epoch}")
model.eval()

# test recall@topk
# generate database for each fine-grained category
distractors_id = dict()
distractors_feats = dict()
distractors_idSet = dict()
for k, v in tqdm(test_dat.fg2ims.items()):
    if k not in distractors_id:
        distractors_id[k] = list()
        distractors_feats[k] = list()
        distractors_idSet[k] = set()
    for set_id, item_lst in v.items():
        for item_id in item_lst:
            if item_id not in distractors_idSet[k]:
                distractors_idSet[k].add(item_id)
                distractors_id[k].append(item_id)
                distractors_feats[k].append(
                    test_dat.extract_emb({'item_id':item_id}, model, device, ctx, to_numpy=True)
                )
            if len(distractors_id[k]) >= 3000:
                break
        if len(distractors_id[k]) >= 3000:
            break
for k, v in tqdm(train_dat.fg2ims.items()):
    if k not in distractors_id:
        break
    for set_id, item_lst in v.items():
        for item_id in item_lst:
            if item_id not in distractors_idSet[k]:
                distractors_idSet[k].add(item_id)
                distractors_id[k].append(item_id)
                distractors_feats[k].append(
                    test_dat.extract_emb({'item_id':item_id}, model, device, ctx, to_numpy=True)
                )
            if len(distractors_id[k]) >= 3000:
                break
        if len(distractors_id[k]) >= 3000:
            break
# filter out fine-grained categories with less than 3000 distractors
distractors_id = {k:v for k, v in distractors_id.items() if len(v) >= 3000}
distractors_feats = {k:v for k, v in distractors_feats.items() if len(v) >= 3000}
distractors_id = {k:np.array(v) for k, v in distractors_id.items()}
distractors_feats = {k:torch.tensor(np.concatenate(v)) for k, v in distractors_feats.items()}
print(f'[{len(distractors_feats)}/{len(test_dat.fg2ims)}] fine-grained categories are selected.')

def retrieval_top_k(
        logits: torch.tensor, 
        distractors_feats: torch.tensor,
        distractors_id: np.ndarray,
        k: int
    ):
    score = torch.cosine_similarity(logits, distractors_feats)
    values, indices = torch.sort(score, descending=True)
    topIdx = distractors_id[indices.numpy()]
    return topIdx[:k]

count = 0
k = [1, 3, 5, 10, 30, 50]
correct = [0] * 6
detail_writer = None
detail_f = None
if args.save_detail:
    if args.detail_file is None:
        args.detail_file = os.path.join(
            exp_root,
            f"detail_cir_{variant}_{subset_tag}_{args.run_tag}_seed{seed}.csv"
        )
    detail_f = open(args.detail_file, "w", newline="", encoding="utf-8")
    detail_writer = csv.writer(detail_f)
    detail_writer.writerow([
        "run_tag","seed","set_id","target_item_id","target_item_fg",
        "rank","hit@1","hit@3","hit@5","hit@10","hit@30","hit@50",
        "top10_ids"
    ])
for outfits, is_correct, set_id in tqdm(test_dat.fitb_questions, desc='compute recall@topk'):
    target_item_id = outfits[np.where(is_correct)[0][0]]['items'][-1]['item_id']
    target_item_fg = test_dat.im2fg[target_item_id]
    if target_item_fg not in distractors_feats:
        continue

    partial_imgs, partial_txts, _ = test_dat.load_outfit({'items': outfits[0]['items'][:-1]})
    partial_imgs, partial_txts, mask = test_dat.pad_imgs_and_txts(partial_imgs, partial_txts)
    partial_imgs, partial_txts, mask = partial_imgs.to(device), partial_txts.to(device), mask.to(device)
    target_item_category = torch.tensor(test_dat.category_feats[target_item_id])
    target_item_category = target_item_category.to(device)
    outfit_txt = torch.tensor(test_dat.outfit_feats[set_id]).to(device)

    count += 1
    with torch.no_grad(), ctx:
        out = model(
            partial_imgs.unsqueeze(0), partial_txts.unsqueeze(0), mask.unsqueeze(0), 
            positive_category=target_item_category.unsqueeze(0),
            outfit_txt=outfit_txt.unsqueeze(0)
        )
        logits = out['logits'].detach().cpu().float()
        cur_dist_id = deepcopy(distractors_id[target_item_fg])
        cur_dist_feats = deepcopy(distractors_feats[target_item_fg])
        if target_item_id not in cur_dist_id:
            cur_dist_id = np.concatenate([cur_dist_id[:-1], np.array([target_item_id])])
            cur_dist_feats = torch.cat([cur_dist_feats[:-1, :], 
                                        test_dat.extract_emb({'item_id':target_item_id}, model, device, ctx, to_numpy=False).detach().cpu()], dim=0)
        topIdx = retrieval_top_k(logits, cur_dist_feats, cur_dist_id, k[-1])
        # ===== 逐題 rank 計算（用同一個 score 排序）=====
        score = torch.cosine_similarity(logits, cur_dist_feats)  # logits 已經是 cpu float
        values, indices = torch.sort(score, descending=True)
        sorted_ids = cur_dist_id[indices.numpy()]  # 完整排序（長度約 3000）

        rank = int(np.where(sorted_ids == target_item_id)[0][0]) + 1
        hit1  = int(rank <= 1)
        hit3  = int(rank <= 3)
        hit5  = int(rank <= 5)
        hit10 = int(rank <= 10)
        hit30 = int(rank <= 30)
        hit50 = int(rank <= 50)

        if detail_writer is not None:
            top10 = topIdx[:10].tolist()
            detail_writer.writerow([
                args.run_tag, seed, int(set_id), int(target_item_id), str(target_item_fg),
                rank, hit1, hit3, hit5, hit10, hit30, hit50,
                json.dumps(top10, ensure_ascii=False)
            ])

        for i, k_ in enumerate(k):
            if target_item_id in topIdx[:k_]:
                correct[i] += 1

recall =  [c / count for c in correct]
recall_at_1, recall_at_3, recall_at_5, recall_at_10, recall_at_30, recall_at_50 = recall

for k_, recall_ in zip(k, recall):
    print(f'Recall@top{k_}: {recall_*100:.4f}%')

results_file = os.path.join(exp_root, "results_cir.csv")
file_exists = os.path.exists(results_file)

with open(results_file, "a", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    if not file_exists:
        writer.writerow([
            "seed", "variant", "subset_tag", "outfit_feat_name", "run_tag",
            "recall_at_1", "recall_at_3", "recall_at_5", "recall_at_10", "recall_at_30", "recall_at_50"
        ])
    writer.writerow([
        seed, variant, subset_tag, outfit_feat_name, args.run_tag,
        float(recall_at_1),
        float(recall_at_3),
        float(recall_at_5),
        float(recall_at_10),
        float(recall_at_30),
        float(recall_at_50),
    ])

print(f"[CIR EVAL] 結果已寫入 {results_file}")

if detail_f is not None:
    detail_f.close()
    print(f"[CIR {args.run_tag}] detail 已寫入 {args.detail_file}")
