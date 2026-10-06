import torch
from dataset import UIUCPolyvoreRetrievalDataset, UIUCPolyvorePredictionDataset
from outfit_transformer import (
    OutfitTransformerRetrieval, 
    LinearImageEncoder,
    LinearTextEncoder,
    MlpImageEncoder,
    MlpTextEncoder,
    OutfitTransformerPrediction
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
from contextlib import nullcontext
import inspect
from copy import deepcopy
import csv

# Change to other config for different settings
from config.cp_cond_hardneg import *
import config.cp_cond_hardneg as cfg

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


def apply_subset_filter_inplace(dat, subset_ids):
    """
    最小侵入式過濾（對齊 train_*）
    1) dat.data 只留 subset set_id
    2) compatibility_questions / fitb_questions 同步過濾（若存在）
    注意：fg2ims/category2ims 等索引不重建（CP eval 不影響主指標）
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
    print(msg)

    return dat

# 解析 CLI 參數
parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=1)
parser.add_argument("--exp_root", type=str, default="./experiments")

# 新增：對齊 train_cp.py 的 5 種 variant 與 pkl
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

# 新增：subset（none ids）
parser.add_argument(
    "--subset_ids_path",
    type=str,
    default=None,
    help="txt 檔，一行一個 set_id；若不提供則使用完整 split"
)

args = parser.parse_args()

seed = args.seed
exp_root = args.exp_root
variant = args.variant
outfit_feat_name = args.outfit_feat_name
subset_ids_path = args.subset_ids_path

subset_tag = "subset" if subset_ids_path else "all"

# 跟 train_cp.py 對齊
exp_dir = os.path.join(exp_root, f"cp_{variant}_{subset_tag}_seed{seed}")
print(f"[CP EVAL] seed={seed}, variant={variant}, subset_tag={subset_tag}, exp_dir={exp_dir}")

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
print(f"[eval-cp] variant={variant}")
print(f"[eval-cp] outfit_feat_name={outfit_feat_name}")
print(f"[eval-cp] outfit_feat_path={outfit_feat_path}")
print(f"[eval-cp] subset_ids_path={subset_ids_path}")

# path checking
print(f"檢查資料路徑結構...")

# 檢查 Split 設定
polyvore_split = 'disjoint'  # 您的目標
target_split_dir = os.path.join(data_root, polyvore_split)

if os.path.exists(target_split_dir):
    print(f"✅ 成功找到 Disjoint 資料夾: {target_split_dir}")
    final_datadir = os.path.join(parent_dir, 'polyvore_data')
    print(final_datadir)

# 檢查特徵檔是否存在
required_files = [img_feat_path, txt_feat_path, outfit_feat_path]
for f in required_files:
    if os.path.exists(f):
        print(f"✅ 特徵檔存在: {os.path.basename(f)}")
    else:
        print(f"❌ 錯誤：找不到特徵檔 {f}")
# -

# dataloader
normalize = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                std=[0.229, 0.224, 0.225])
transform = torchvision.transforms.Compose([
    torchvision.transforms.Resize((img_size, img_size)),
    torchvision.transforms.ToTensor(),
    normalize
])
train_dat = UIUCPolyvorePredictionDataset(
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
train_loader = DataLoader(train_dat, batch_size, shuffle=True, num_workers=num_workers)
test_dat = UIUCPolyvorePredictionDataset(
        datadir=final_datadir,
        txt_feat_path=txt_feat_path,
        img_feat_path=img_feat_path,
        category_feat_path=category_feat_path,
        outfit_feat_path=outfit_feat_path,
        polyvore_split=polyvore_split,
        split='test',
        max_item_len=max_item_len,
        transform=transform,
        sample_hard_negative=False
    )
test_loader = DataLoader(test_dat, batch_size, shuffle=False, num_workers=num_workers)
print("Load data complete")

subset_ids = load_subset_ids(subset_ids_path)
if subset_ids is not None:
    apply_subset_filter_inplace(train_dat, subset_ids)
    apply_subset_filter_inplace(test_dat, subset_ids)
else:
    print("[subset] 未啟用 subset filter（使用完整 split）")
    
# model init
best_acc = -1
best_auc = -1
start_epoch = 0
model = OutfitTransformerPrediction(
    img_encoder=LinearImageEncoder(img_inp_size=img_inp_size, img_emb_size=img_emb_size),
    text_encoder=LinearTextEncoder(txt_inp_size=txt_inp_size, txt_emb_size=txt_emb_size),
    outfit_txt_encoder=LinearTextEncoder(txt_inp_size=outfit_txt_inp_size, txt_emb_size=txt_emb_size+img_emb_size),
    nhead=nhead,
    num_layers=num_layers,
    use_outfit_txt=use_outfit_txt
)

# +
ckpt_path = os.path.join(exp_dir, "ckpt.pt")
print(f"Loading checkpoint from: {ckpt_path}")
checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)

state_dict = checkpoint['model']
model.load_state_dict(state_dict)
best_acc = checkpoint.get('best_acc', -1)
start_epoch = checkpoint['epoch']
print(f"Loaded checkpoint from {exp_dir}, best_acc: {best_acc}, epoch: {start_epoch}")
model = model.to(device)
# -

# compile the model
if compile_model:
    print("compiling the model ...")
    unoptimized_model = model
    model = torch.compile(model)

# evaluate model
model.eval()

# test CP auc
target_lst = []
output_lst = []
for outfit, target, set_id in tqdm(test_dat.compatibility_questions):
    outfit_txt = torch.tensor(test_dat.outfit_feats[set_id]).to(device)
    imgs, txts = test_dat.load_outfit(outfit)[:2]
    imgs, txts, mask = test_dat.pad_imgs_and_txts(imgs, txts)
    imgs, txts, mask = imgs.to(device), txts.to(device), mask.to(device)
    with torch.no_grad(), ctx:
        output = model(imgs.unsqueeze(0), txts.unsqueeze(0), mask.unsqueeze(0), outfit_txt=outfit_txt.unsqueeze(0))
    output_lst.append(output['logits'].item())
    target_lst.append(target)
output_lst = np.array(output_lst)
target_lst = np.array(target_lst)
auc = roc_auc_score(target_lst, output_lst)
print(f'AUC: {auc:.3f}')        

# test fitb       
n_questions = 0
correct = 0
for outfits, is_correct, set_id in tqdm(test_dat.fitb_questions, desc="compute FITB"):
    answer_score = []
    outfit_txt = torch.tensor(test_dat.outfit_feats[set_id]).to(device)
    for outfit in outfits:
        imgs, txts = test_dat.load_outfit(outfit)[:2]
        imgs, txts, mask = test_dat.pad_imgs_and_txts(imgs, txts)
        imgs, txts, mask = imgs.to(device), txts.to(device), mask.to(device)
        with torch.no_grad(), ctx:
            output = model(imgs.unsqueeze(0), txts.unsqueeze(0), mask.unsqueeze(0), outfit_txt=outfit_txt.unsqueeze(0))
        answer_score.append(output['logits'].item())
    answer_score = np.array(answer_score)
    correct += is_correct[np.argmax(answer_score)]
    n_questions += 1
acc = correct / n_questions
print(f'FITB ACC: {acc:.4f}')

# 統一結果檔（含 variant / subset 資訊）
results_file = os.path.join(exp_root, "results_cp.csv")
file_exists = os.path.exists(results_file)

with open(results_file, "a", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    if not file_exists:
        writer.writerow(["seed", "variant", "subset_tag", "outfit_feat_name", "auc", "fitb_acc"])
    writer.writerow([seed, variant, subset_tag, outfit_feat_name, float(auc), float(acc)])

print(f"[CP EVAL] 結果已寫入 {results_file}")