# !pip install ipdb

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
from contextlib import nullcontext
import inspect
from copy import deepcopy

# 解析命令列參數
parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=1)

# 子集控制（你已加好，保留）
parser.add_argument(
    "--subset_ids_path",
    type=str,
    default=None,
    help="每行一個 set_id；若提供則只使用這些 outfit IDs"
)

# 實驗條件控制（新增）
parser.add_argument(
    "--variant",
    type=str,
    choices=["outfitUrlTitle", "NewoutfitUrlTitle", "no_weather", "no_occasion", "no_style"],
    default="outfitUrlTitle"
)
parser.add_argument(
    "--outfit_feat_name",
    type=str,
    default="encoded_outfitUrlTitle_en_fashionClip.pkl",
    help="fashionclip_data 底下的 outfit title embedding 檔名"
)
parser.add_argument(
    "--exp_root",
    type=str,
    default="./experiments",
    help="實驗輸出根目錄"
)

parser.add_argument(
    "--pretrained_cp_ckpt",
    type=str,
    default=None,
    help="要載入的 CP checkpoint 路徑（若不提供則用預設規則推導）"
)

args = parser.parse_args()
pretrained_cp_ckpt = args.pretrained_cp_ckpt
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

from config.cir_cond_hardneg import *
import config.cir_cond_hardneg as cfg

# +
import sys
from datetime import datetime
from pathlib import Path
sys.path.append(os.getcwd())
from utils import set_logger

base_output_dir = exp_root
subset_tag = "subset" if subset_ids_path else "all"
experiment_name = f'cir_{variant}_{subset_tag}_seed{seed}'

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
train_dat = apply_subset_filter_inplace(train_dat, subset_ids, logger)

train_loader = DataLoader(train_dat, batch_size, shuffle=True, num_workers=num_workers)
valid_dat = UIUCPolyvoreRetrievalDataset(
        datadir=final_datadir,
        txt_feat_path=txt_feat_path,
        img_feat_path=img_feat_path,
        category_feat_path=category_feat_path,
        outfit_feat_path=outfit_feat_path,
        polyvore_split=polyvore_split,
        split='valid',
        max_item_len=max_item_len,
        transform=transform,
        sample_hard_negative=False
    )
valid_dat = apply_subset_filter_inplace(valid_dat, subset_ids, logger)
valid_loader = DataLoader(valid_dat, batch_size, shuffle=False, num_workers=num_workers)
logger.info("Load data complete")

if pretrained_cp_ckpt is not None:
    pretrained_cp_model = pretrained_cp_ckpt
else:
    subset_tag = "subset" if subset_ids_path else "all"
    pretrained_cp_model = os.path.join(
        exp_root,
        f"cp_{variant}_{subset_tag}_seed{seed}",
        "ckpt.pt"
    )
print(f"預計載入 CP 舊版模型: {pretrained_cp_model}")
logger.info(f"預計載入 CP 舊版模型: {pretrained_cp_model}")

best_acc = -1
best_auc = -1
start_epoch = 0

# 初始化模型結構
model = OutfitTransformerRetrieval(
    img_encoder=LinearImageEncoder(img_inp_size=img_inp_size, img_emb_size=img_emb_size),
    text_encoder=LinearTextEncoder(txt_inp_size=txt_inp_size, txt_emb_size=txt_emb_size),
    outfit_txt_encoder=LinearTextEncoder(txt_inp_size=outfit_txt_inp_size, txt_emb_size=txt_emb_size+img_emb_size) if use_outfit_txt else None,
    nhead=nhead,
    num_layers=num_layers,
    margin=margin,  
    target_item_info='category',
    use_outfit_txt=use_outfit_txt,
)
if init_from == 'scratch':
    logger.info("Initializing a new model from scratch")
    
    if pretrained_cp_model is not None:
        if os.path.exists(pretrained_cp_model):
            logger.info(f"Loading pretrained CP model from: {pretrained_cp_model}")
            try:
                # 1. 讀取檔案 (加入 weights_only=False)
                cp_checkpoint = torch.load(pretrained_cp_model, map_location='cpu', weights_only=False)
                pretrained_dict = cp_checkpoint['model']
                
                # 2. 取得當前模型的 state_dict
                model_dict = model.state_dict()
                
                # 3. ★★★ 關鍵步驟：過濾掉形狀不匹配的層 ★★★
                # 我們只保留那些「名字存在」且「形狀完全一樣」的權重
                filtered_dict = {k: v for k, v in pretrained_dict.items() 
                                 if k in model_dict and v.shape == model_dict[k].shape}
                
                # (選擇性) 印出被過濾掉的層，讓您安心
                skipped_layers = [k for k in pretrained_dict.keys() if k not in filtered_dict]
                if len(skipped_layers) > 0:
                    print(f"⚠️ 自動跳過 {len(skipped_layers)} 個形狀不符的層 (例如: {skipped_layers[:3]}...)")

                # 4. 更新模型權重
                model_dict.update(filtered_dict)
                
                # 5. 載入更新後的權重
                model.load_state_dict(model_dict)
                
                print(f"✅ 成功載入 CP 模型權重！(共移植了 {len(filtered_dict)} 層)")
                
            except Exception as e:
                print(f"❌ 載入失敗: {e}")
                print("將使用隨機初始化的權重繼續訓練...")
        else:
            print(f"❌ 找不到檔案: {pretrained_cp_model}")

elif init_from == "resume":
    # ... (原本的 resume 邏輯保持不變) ...
    ckpt_path = os.path.join(out_dir, "ckpt.pt")
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    state_dict = checkpoint['model']
    model.load_state_dict(state_dict)
    best_acc = checkpoint.get('best_acc', -1)
    start_epoch = checkpoint['epoch']
    logger.info(f"Resuming training from {out_dir}, best_acc: {best_acc}, epoch: {start_epoch}")

# 轉移到 GPU
model = model.to(device)
print("模型準備完成！")
# -

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
    for iter_num, (partial_imgs,
                   partial_txts,
                   mask,
                   positive_img,
                   positive_txt,
                   positive_category,
                   negative_imgs,
                   negative_txts,
                   hard_negative_imgs,
                   hard_negative_txts,
                   outfit_txt) in enumerate(train_loader, start=1):
                
        partial_imgs, partial_txts, mask = partial_imgs.to(device), partial_txts.to(device), mask.to(device)
        positive_img, positive_txt = positive_img.to(device), positive_txt.to(device)
        positive_category = positive_category.to(device)
        negative_imgs, negative_txts = negative_imgs.to(device), negative_txts.to(device)
        outfit_txt = outfit_txt.to(device)
        with ctx:
            if epoch >= hard_negatives_start_epoch:
                hard_negative_imgs, hard_negative_txts = hard_negative_imgs.to(device), hard_negative_txts.to(device)
                out = model.forward(
                    partial_imgs, partial_txts, mask, 
                    positive_img, positive_txt, positive_category, 
                    negative_imgs, negative_txts,
                    hard_negative_imgs, hard_negative_txts,
                    outfit_txt=outfit_txt
                )
            else:
                out = model.forward(
                    partial_imgs, partial_txts, mask, 
                    positive_img, positive_txt, positive_category, 
                    negative_imgs, negative_txts,
                    outfit_txt=outfit_txt
                )
        loss = out['loss']
                                
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

        # test fitb       
        n_questions = 0
        correct = 0

        for outfits, is_correct, set_id in tqdm(valid_loader.dataset.fitb_questions, desc="compute FITB"):
            partial_imgs, partial_txts, _ = valid_loader.dataset.load_outfit({'items': outfits[0]['items'][:-1]})
            partial_imgs, partial_txts, mask = valid_loader.dataset.pad_imgs_and_txts(partial_imgs, partial_txts)

            ans_imgs = []
            ans_txts = []
            ans_categories = []
            for outfit in outfits:
                ans_img, ans_txt, ans_category = valid_loader.dataset.load_item(outfit['items'][-1])
                ans_imgs.append(ans_img)
                ans_txts.append(ans_txt)
                ans_categories.append(ans_category)
            ans_imgs = torch.stack(ans_imgs).to(device)
            ans_txts = torch.stack(ans_txts).to(device)
            ans_categories = torch.stack(ans_categories).to(device)

            correct_ind = np.where(is_correct)[0][0]
            positive_txt = ans_txts[correct_ind]
            positive_category = ans_categories[correct_ind]

            positive_txt = positive_txt.to(device)
            positive_category = positive_category.to(device)
            partial_imgs, partial_txts, mask = partial_imgs.to(device), partial_txts.to(device), mask.to(device)

            outfit_txt = torch.tensor(valid_dat.outfit_feats[set_id]).to(device)

            with torch.no_grad():
                with ctx:
                    out = model(
                        partial_imgs.unsqueeze(0), partial_txts.unsqueeze(0), mask.unsqueeze(0), 
                        positive_category=positive_category.unsqueeze(0),
                        outfit_txt=outfit_txt.unsqueeze(0)
                    )
                    ans_imgs_emb = model.img_encoder(ans_imgs)
                    ans_txts_emb = model.text_encoder(ans_txts)
                    ans_emb = torch.cat([ans_imgs_emb, ans_txts_emb], dim=-1)
                    dist = F.cosine_similarity(out['logits'], ans_emb, dim=-1)
                    ans = torch.argmax(dist)

            if is_correct[ans]:
                correct += 1
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

        # evaluate recall@topk
        # generate database for each fine-grained category
        distractors_id = dict()
        distractors_feats = dict()
        distractors_idSet = dict()
        for k, v in tqdm(valid_dat.fg2ims.items()):
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
                            valid_dat.extract_emb({'item_id':item_id}, model, device, ctx, to_numpy=True)
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
                            valid_dat.extract_emb({'item_id':item_id}, model, device, ctx, to_numpy=True)
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
        logger.info(f'[{len(distractors_feats)}/{len(valid_dat.fg2ims)}] fine-grained categories are selected.')

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
        k = [10, 30, 50]
        correct = [0] * 3
        for outfits, is_correct, set_id in tqdm(valid_dat.fitb_questions, desc='compute recall@topk'):
            target_item_id = outfits[np.where(is_correct)[0][0]]['items'][-1]['item_id']
            target_item_fg = valid_dat.im2fg[target_item_id]
            if target_item_fg not in distractors_feats:
                continue

            partial_imgs, partial_txts, _ = valid_dat.load_outfit({'items': outfits[0]['items'][:-1]})
            partial_imgs, partial_txts, mask = valid_dat.pad_imgs_and_txts(partial_imgs, partial_txts)
            partial_imgs, partial_txts, mask = partial_imgs.to(device), partial_txts.to(device), mask.to(device)
            target_item_category = torch.tensor(valid_dat.category_feats[target_item_id])
            target_item_category = target_item_category.to(device)
            outfit_txt = torch.tensor(valid_dat.outfit_feats[set_id])
            outfit_txt = outfit_txt.to(device)

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
                                                valid_dat.extract_emb({'item_id':target_item_id}, model, device, ctx, to_numpy=False).detach().cpu()], dim=0)
                topIdx = retrieval_top_k(logits, cur_dist_feats, cur_dist_id, k[-1])
                for i, k_ in enumerate(k):
                    if target_item_id in topIdx[:k_]:
                        correct[i] += 1

        recall =  [c / count for c in correct]
        for k_, recall_ in zip(k, recall):
            logger.info(f'Recall@top{k_}: {recall_*100:.4f}%')


