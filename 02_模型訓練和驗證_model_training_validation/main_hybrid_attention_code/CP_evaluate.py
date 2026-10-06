import torch
from dataset import UIUCPolyvoreRetrievalDataset, UIUCPolyvorePredictionDataset
from outfit_transformer import (
    LinearImageEncoder,
    LinearTextEncoder,
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
import csv
import random
import math
from torchvision.ops.focal_loss import sigmoid_focal_loss
from pathlib import Path
import torch.nn.functional as F
from contextlib import nullcontext
import inspect
from copy import deepcopy

# Change to other config for different settings
from config.cp_cond_hardneg import *
import config.cp_cond_hardneg as cfg

def resolve_paths():
    current_dir = os.getcwd()
    parent_dir = os.path.dirname(current_dir)

    feat_dir = os.path.join(parent_dir, "fashionclip_data")
    data_root = os.path.join(parent_dir, "polyvore_data", "polyvore_outfits")

    img_feat_path = os.path.join(feat_dir, "img_feats_fashionClip.pkl")
    txt_feat_path = os.path.join(feat_dir, "encoded_title_description_distiluse-base-multilingual-cased-v2.pkl")
    category_feat_path = os.path.join(feat_dir, "encoded_category_distiluse-base-multilingual-cased-v2.pkl")

    return parent_dir, feat_dir, data_root, img_feat_path, txt_feat_path, category_feat_path


def get_outfit_feat_path(feat_dir, feat_tag: str):
    if feat_tag == "new":
        return os.path.join(feat_dir, "encoded_NewoutfitUrlTitle_en_fashionClip.pkl")
    elif feat_tag == "old":
        return os.path.join(feat_dir, "encoded_outfitUrlTitle_en_fashionClip.pkl")
    else:
        raise ValueError(f"Unknown feat_tag: {feat_tag}")


def print_cfg():
    for x in dir(cfg):
        if x.startswith("__"):
            continue
        if inspect.ismodule(cfg.__dict__.get(x)):
            continue
        if isinstance(cfg.__dict__.get(x), type):
            continue
        print("{:<40}{:<40}".format(x, repr(cfg.__dict__.get(x))))


def build_dataset(final_datadir, polyvore_split, split, txt_feat_path, img_feat_path, category_feat_path, outfit_feat_path):
    normalize = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                 std=[0.229, 0.224, 0.225])
    transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize((img_size, img_size)),
        torchvision.transforms.ToTensor(),
        normalize
    ])

    return UIUCPolyvorePredictionDataset(
        datadir=final_datadir,
        txt_feat_path=txt_feat_path,
        img_feat_path=img_feat_path,
        category_feat_path=category_feat_path,
        outfit_feat_path=outfit_feat_path,
        polyvore_split=polyvore_split,
        split=split,
        max_item_len=max_item_len,
        transform=transform,
        sample_hard_negative=False
    )


def load_model(ckpt_path, device):
    model = OutfitTransformerPrediction(
        img_encoder=LinearImageEncoder(img_inp_size=img_inp_size, img_emb_size=img_emb_size),
        text_encoder=LinearTextEncoder(txt_inp_size=txt_inp_size, txt_emb_size=txt_emb_size),
        outfit_txt_encoder=LinearTextEncoder(txt_inp_size=outfit_txt_inp_size, txt_emb_size=txt_emb_size + img_emb_size),
        nhead=nhead,
        num_layers=num_layers,
        use_outfit_txt=use_outfit_txt
    )
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    model = model.to(device)
    model.eval()
    return model


def eval_cp(model, test_dat, device, ctx):
    # AUC
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

    # FITB
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
    return float(auc), float(acc)


def append_csv(results_file, row_dict):
    file_exists = os.path.exists(results_file)
    with open(results_file, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row_dict.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(row_dict)


def run_once(exp_root, seed, model_tag, feat_tag, compile_eval=False):
    # device / autocast
    device = "cuda" if torch.cuda.is_available() else "cpu"
    device_type = "cuda" if torch.cuda.is_available() else "cpu"
    ptdtype = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[dtype]
    ctx = nullcontext() if device_type == "cpu" else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

    parent_dir, feat_dir, data_root, img_feat_path, txt_feat_path, category_feat_path = resolve_paths()
    outfit_feat_path = get_outfit_feat_path(feat_dir, feat_tag)

    # dataset root（沿用你原本做法：final_datadir 指到 polyvore_data）
    polyvore_split = "disjoint"
    target_split_dir = os.path.join(data_root, polyvore_split)
    if not os.path.exists(target_split_dir):
        raise FileNotFoundError(f"找不到 split 資料夾：{target_split_dir}")
    final_datadir = os.path.join(parent_dir, "polyvore_data")

    for f in [img_feat_path, txt_feat_path, outfit_feat_path]:
        if not os.path.exists(f):
            raise FileNotFoundError(f"找不到特徵檔：{f}")

    test_dat = build_dataset(final_datadir, polyvore_split, "test",
                             txt_feat_path, img_feat_path, category_feat_path, outfit_feat_path)

    # ckpt
    exp_dir = os.path.join(exp_root, f"{model_tag}_seed{seed}")
    ckpt_path = os.path.join(exp_dir, "ckpt.pt")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"找不到 checkpoint：{ckpt_path}")

    model = load_model(ckpt_path, device)

    if compile_eval:
        model = torch.compile(model)

    auc, fitb_acc = eval_cp(model, test_dat, device, ctx)

    return auc, fitb_acc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--exp_root", type=str, default="./experiments")

    parser.add_argument("--model_tag", choices=["cp_old", "cp_new"], default="cp_new")
    parser.add_argument("--feat_tag", choices=["old", "new"], default="new")

    parser.add_argument("--sweep", action="store_true")

    parser.add_argument("--compile_eval", action="store_true")

    args = parser.parse_args()

    print_cfg()

    results_file = os.path.join(args.exp_root, "results_cp_sweep.csv")

    if not args.sweep:
        auc, fitb = run_once(args.exp_root, args.seed, args.model_tag, args.feat_tag, compile_eval=args.compile_eval)
        print(f"[CP] seed={args.seed} model={args.model_tag} feat={args.feat_tag} | AUC={auc:.4f} FITB={fitb:.4f}")

        append_csv(results_file, {
            "seed": args.seed,
            "model_tag": args.model_tag,
            "feat_tag": args.feat_tag,
            "auc": auc,
            "fitb_acc": fitb
        })
        print(f"結果已寫入：{results_file}")
        return

    combos = [
        ("cp_old", "old"),
        ("cp_old", "new"),
        ("cp_new", "old"),
        ("cp_new", "new"),
    ]
    for mtag, ftag in combos:
        auc, fitb = run_once(args.exp_root, args.seed, mtag, ftag, compile_eval=args.compile_eval)
        print(f"[CP] seed={args.seed} model={mtag} feat={ftag} | AUC={auc:.4f} FITB={fitb:.4f}")

        append_csv(results_file, {
            "seed": args.seed,
            "model_tag": mtag,
            "feat_tag": ftag,
            "auc": auc,
            "fitb_acc": fitb
        })

    print(f"完成，結果已寫入：{results_file}")


if __name__ == "__main__":
    main()