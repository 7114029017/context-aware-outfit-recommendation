# !pip install ipdb

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

from ipdb import iex, launch_ipdb_on_exception
os.environ['PYTHONBREAKPOINT'] = 'ipdb.set_trace'

from config.cp_cond_hardneg import *
import config.cp_cond_hardneg as cfg

# 解析命令列參數
parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=1)
parser.add_argument("--subset_ids_path", type=str, default=None, help="每行一個 set_id；若提供則只使用這些 outfit IDs")

# 檔案命名
parser.add_argument(
    "--variant",
    type=str,
    default="outfitUrlTitle",
    choices=["outfitUrlTitle", "NewoutfitUrlTitle", "no_weather", "no_occasion", "no_style"],
    help="實驗條件名稱（用於命名與紀錄）"
)
parser.add_argument(
    "--outfit_feat_name",
    type=str,
    default="encoded_outfitUrlTitle_en_fashionClip.pkl",
    help="fashionclip_data 底下的 outfit text embedding 檔名"
)

parser.add_argument("--exp_root", type=str, default="./experiments") 
args = parser.parse_args()
seed = args.seed
subset_ids_path = args.subset_ids_path
variant = args.variant
outfit_feat_name = args.outfit_feat_name
exp_root = args.exp_root

def set_seed(seed: int):
    print(f"==== 使用亂數種子 seed = {seed} ====")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # 讓 cudnn 較 deterministic（會稍微慢一點，但比較可重現）
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(seed)

def load_subset_ids(path):
    """讀取每行一個 set_id 的 txt，回傳 set[str]"""
    if path is None:
        return None
    with open(path, "r", encoding="utf-8") as f:
        ids = {line.strip() for line in f if line.strip()}
    print(f"[subset] loaded {len(ids)} ids from: {path}")
    return ids


def apply_subset_filter_inplace(dat, subset_ids, logger=None):
    """
    最小侵入式過濾：
    1) dat.data 只留 subset set_id
    2) 相依的 evaluation question list 也同步過濾（若存在）
    注意：dataset 內部 category2ims/fg2ims 等索引仍維持原始 split（先求穩定可跑）
    """
    if subset_ids is None:
        return dat

    before_n = len(dat.data)
    dat.data = [o for o in dat.data if str(o["set_id"]) in subset_ids]
    after_n = len(dat.data)

    # prediction/retrieval dataset 都有 compatibility_questions / fitb_questions（若 split 有）
    if hasattr(dat, "compatibility_questions"):
        before_q = len(dat.compatibility_questions)
        dat.compatibility_questions = [
            q for q in dat.compatibility_questions
            if str(q[2]) in subset_ids   # q = (outfit, target, set_id)
        ]
        after_q = len(dat.compatibility_questions)
    else:
        before_q = after_q = None

    if hasattr(dat, "fitb_questions"):
        before_f = len(dat.fitb_questions)
        dat.fitb_questions = [
            q for q in dat.fitb_questions
            if str(q[2]) in subset_ids   # q = (outfits, is_correct, set_id)
        ]
        after_f = len(dat.fitb_questions)
    else:
        before_f = after_f = None

    msg = (
        f"[subset] split={getattr(dat, 'split', 'unknown')} "
        f"data: {before_n} -> {after_n}"
    )
    if before_q is not None:
        msg += f", compatibility_questions: {before_q} -> {after_q}"
    if before_f is not None:
        msg += f", fitb_questions: {before_f} -> {after_f}"

    print(msg)
    if logger is not None:
        logger.info(msg)

    return dat

import os
import sys
from datetime import datetime
from pathlib import Path
sys.path.append(os.getcwd())
from utils import set_logger

base_output_dir = exp_root
subset_tag = "subset" if subset_ids_path else "all"
experiment_name = f'cp_{variant}_{subset_tag}_seed{seed}'

out_dir = os.path.join(base_output_dir, experiment_name)

if not os.path.exists(out_dir):
    os.makedirs(out_dir)
    print(f"✅ 已建立輸出資料夾: {out_dir}")

out_dir = Path(out_dir)
logger = set_logger(out_dir)
logger.info(f"Log 系統啟動成功！所有結果將儲存於: {out_dir}")

subset_ids = load_subset_ids(subset_ids_path)
if subset_ids is not None:
    logger.info(f"[subset] 啟用 subset_ids_path={subset_ids_path}, n_ids={len(subset_ids)}")
else:
    logger.info("[subset] 未啟用 subset filter（使用完整 split）")

# print config parameters
for x in dir(cfg):
    if x.startswith('__'):
        continue
    if inspect.ismodule(cfg.__dict__.get(x)):
        continue
    if isinstance(cfg.__dict__.get(x), type):
        continue
    logger.info('{:<40}{:<40}'.format(x, repr(cfg.__dict__.get(x))))

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
logger.info(f"[exp] variant={variant}")
logger.info(f"[exp] seed={seed}")
logger.info(f"[exp] exp_root={exp_root}")
logger.info(f"[exp] subset_ids_path={subset_ids_path}")
logger.info(f"[exp] outfit_feat_name={outfit_feat_name}")
logger.info(f"[exp] outfit_feat_path={outfit_feat_path}")

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
        polyvore_split=polyvore_split, # disjoint
        split='train',
        max_item_len=max_item_len,
        transform=transform,
        sample_hard_negative=False
    )
# 建立 dataset 後先做 subset filter（若有）
train_dat = apply_subset_filter_inplace(train_dat, subset_ids, logger)
train_loader = DataLoader(train_dat, batch_size, shuffle=True, num_workers=num_workers)
valid_dat = UIUCPolyvorePredictionDataset(
        datadir=final_datadir,
        txt_feat_path=txt_feat_path,
        img_feat_path=img_feat_path,
        category_feat_path=category_feat_path,
        outfit_feat_path=outfit_feat_path,
        polyvore_split=polyvore_split, # disjoint
        split='valid',
        max_item_len=max_item_len,
        transform=transform,
        sample_hard_negative=False
    )
valid_dat = apply_subset_filter_inplace(valid_dat, subset_ids, logger)
valid_loader = DataLoader(valid_dat, batch_size, shuffle=False, num_workers=num_workers)
logger.info("Load data complete")
logger.info(f"[subset] final train size = {len(train_dat)}, valid size = {len(valid_dat)}")

# model init
best_acc = -1
best_auc = -1
start_epoch = 0
model = OutfitTransformerPrediction(
    img_encoder=LinearImageEncoder(img_inp_size=img_inp_size, img_emb_size=img_emb_size),
    text_encoder=LinearTextEncoder(txt_inp_size=txt_inp_size, txt_emb_size=txt_emb_size),
    outfit_txt_encoder=LinearTextEncoder(txt_inp_size=outfit_txt_inp_size, txt_emb_size=txt_emb_size+img_emb_size) if use_outfit_txt else None,
    nhead=nhead,
    num_layers=num_layers,
    use_outfit_txt=use_outfit_txt
)
if init_from == 'scratch':
    logger.info("Initializing a new model from scratch")
elif init_from == "resume":
    ckpt_path = os.path.join(out_dir, "ckpt.pt")
    checkpoint = torch.load(ckpt_path, map_location=device)
    state_dict = checkpoint['model']
    model.load_state_dict(state_dict)
    best_acc = checkpoint.get('best_acc', -1)
    start_epoch = checkpoint['epoch']
    logger.info(f"Resuming training from {out_dir}, best_acc: {best_acc}, epoch: {start_epoch}")
model = model.to(device)

# initialize a GradScaler. If enabled=False scaler is a no-op
scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'float16'))
if init_from == 'resume':
    scaler.load_state_dict(checkpoint["scaler"])

# optimizer
parameters = model.parameters()
optimizer = optim.Adam(parameters, lr=learning_rate)
if init_from == 'resume':
    optimizer.load_state_dict(checkpoint['optimizer'])
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.5)

# compile the model
if compile_model:
    logger.info("compiling the model ...")
    unoptimized_model = model
    model = torch.compile(model)

# +
# check using cuda
first_param = next(model.parameters())

print(f"🔍 模型參數所在的裝置: {first_param.device}")

if first_param.is_cuda:
    print("✅ 確認：模型正在使用 CUDA (GPU)")
else:
    print("❌ 警告：模型目前在 CPU 上！(請確認是否有執行 model.to('cuda'))")
# -

t0 = time.time()
global_iter_num = 1
for epoch in range(start_epoch+1, epochs+1):
    # train one epoch
    model.train()
    if epoch >= hard_negatives_start_epoch:
        train_loader.dataset.sample_hard_negative = True
        logger.info('Sampling hard negatives')
    else:
        train_loader.dataset.sample_hard_negative = False
        logger.info('Not sampling hard negatives')
    for iter_num, (outfit_txt,
                   imgs, txts, mask,
                   negative_imgs, negative_txts, negative_mask,
                   hard_negative_imgs, hard_negative_txts, hard_negative_mask) in enumerate(train_loader, start=1):
                
        imgs, txts, mask = imgs.to(device), txts.to(device), mask.to(device)
        negative_imgs, negative_txts, negative_mask = negative_imgs.to(device), negative_txts.to(device), negative_mask.to(device)
        outfit_txt = outfit_txt.to(device)
        with ctx:
            if epoch >= hard_negatives_start_epoch:
                hard_negative_imgs, hard_negative_txts, hard_negative_mask = \
                    hard_negative_imgs.to(device), hard_negative_txts.to(device), hard_negative_mask.to(device)
                out_hard_neg = model.forward(
                    hard_negative_imgs, hard_negative_txts, hard_negative_mask, 
                    outfit_txt=outfit_txt,
                    target=torch.zeros((hard_negative_mask.shape[0], 1)).to(device)
                )
            out = model.forward(
                imgs, txts, mask, 
                outfit_txt=outfit_txt,
                target=torch.ones((imgs.shape[0], 1)).to(device)
            )
            out_neg = model.forward(
                negative_imgs, negative_txts, negative_mask, 
                outfit_txt=outfit_txt,
                target=torch.zeros((negative_imgs.shape[0], 1)).to(device)
            )
        loss = out['loss'] + out_neg['loss']
        if epoch >= hard_negatives_start_epoch:
            loss += out_hard_neg['loss']
                                
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

        global_iter_num += 1
        t1 = time.time()
        dt = t1 - t0
        t0 = t1
        if iter_num % log_interval_steps == 0 or iter_num == 1:
            logger.info(f"epoch [{epoch: 3}/{epochs}] "
                         f"iter {iter_num: 5} [{iter_num*batch_size: 6}/{len(train_dat)}]: "
                         f"lr {scheduler.get_last_lr()[0]:.4e} "
                         f"loss {loss.item():.4f}, "
                         f"time {dt*1000:.2f}ms")
            
    # decay learning rate if need
    if epoch % 10 == 0:
        scheduler.step()
    
    if epoch % eval_interval_epochs == 0:
        # evaluate model
        model.eval()

        # test CP auc
        target_lst = []
        output_lst = []
        for outfit, target, set_id in tqdm(valid_dat.compatibility_questions):
            outfit_txt = torch.tensor(valid_dat.outfit_feats[set_id]).to(device)
            imgs, txts = valid_dat.load_outfit(outfit)[:2]
            imgs, txts, mask = valid_dat.pad_imgs_and_txts(imgs, txts)
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
        for outfits, is_correct, set_id in tqdm(valid_dat.fitb_questions, desc="compute FITB"):
            answer_score = []
            outfit_txt = torch.tensor(valid_dat.outfit_feats[set_id]).to(device)
            for outfit in outfits:
                imgs, txts = valid_dat.load_outfit(outfit)[:2]
                imgs, txts, mask = valid_dat.pad_imgs_and_txts(imgs, txts)
                imgs, txts, mask = imgs.to(device), txts.to(device), mask.to(device)
                with torch.no_grad(), ctx:
                    output = model(imgs.unsqueeze(0), txts.unsqueeze(0), mask.unsqueeze(0), outfit_txt=outfit_txt.unsqueeze(0))
                answer_score.append(output['logits'].item())
            answer_score = np.array(answer_score)
            correct += is_correct[np.argmax(answer_score)]
            n_questions += 1
        acc = correct / n_questions
        logger.info(f'FITB ACC: {acc:.4f}')
        
        if acc > best_acc:
            best_acc = acc
            checkpoint = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scaler': scaler.state_dict(),
                'best_acc': best_acc,
                'epoch': epoch
            }
            logger.info(f"saving best checkpoint to {out_dir}")
            torch.save(checkpoint, os.path.join(out_dir, f'ckpt.pt'))

        if always_save_checkpoint:
            checkpoint = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scaler': scaler.state_dict(),
                'best_acc': best_acc,
                'epoch': epoch
            }
            logger.info(f"saving current epoch checkpoint to {out_dir}")
            torch.save(checkpoint, os.path.join(out_dir, f'ckpt_{epoch}.pt'))
