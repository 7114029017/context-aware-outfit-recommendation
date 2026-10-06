import torch
from dataset import UIUCPolyvoreRetrievalDataset
from outfit_transformer import (
    OutfitTransformerRetrieval, 
    LinearImageEncoder,
    LinearTextEncoder,
)
import argparse
import torchvision
import numpy as np
from tqdm import tqdm
import os
from contextlib import nullcontext
from copy import deepcopy
import inspect
import csv
import json

# Change to other config for different settings
from config.cir_cond_hardneg import *
import config.cir_cond_hardneg as cfg


def resolve_paths():
    current_dir = os.getcwd()
    parent_dir = os.path.dirname(current_dir)

    feat_dir = os.path.join(parent_dir, "fashionclip_data")
    data_root = os.path.join(parent_dir, "polyvore_data", "polyvore_outfits")

    img_feat_path = os.path.join(feat_dir, "img_feats_fashionClip.pkl")
    txt_feat_path = os.path.join(feat_dir, "encoded_title_description_distiluse-base-multilingual-cased-v2.pkl")
    category_feat_path = os.path.join(feat_dir, "encoded_category_distiluse-base-multilingual-cased-v2.pkl")

    # datadir should point to parent of "disjoint"
    final_datadir = os.path.join(parent_dir, "polyvore_data")

    return parent_dir, feat_dir, data_root, final_datadir, img_feat_path, txt_feat_path, category_feat_path


def get_outfit_feat_path(feat_dir, feat_tag: str):
    if feat_tag == "new":
        return os.path.join(feat_dir, "encoded_NewoutfitUrlTitle_en_fashionClip.pkl")
    elif feat_tag == "old":
        return os.path.join(feat_dir, "encoded_outfitUrlTitle_en_fashionClip.pkl")
    else:
        raise ValueError(f"Unknown feat_tag: {feat_tag}")


def print_cfg():
    for x in dir(cfg):
        if x.startswith('__'):
            continue
        if inspect.ismodule(cfg.__dict__.get(x)):
            continue
        if isinstance(cfg.__dict__.get(x), type):
            continue
        print('{:<40}{:<40}'.format(x, repr(cfg.__dict__.get(x))))


def run_once(seed, exp_root, model_tag, feat_tag, save_detail=False, detail_file=None, run_tag=None):
    # run_tag 預設：用 model+feat 自動命名（避免 sweep 時覆蓋）
    if run_tag is None:
        run_tag = f"{model_tag}_feat{feat_tag}"

    # print config parameters
    print_cfg()

    # note: float16 data type will automatically use a GradScaler
    ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
    ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

    # paths
    parent_dir, feat_dir, data_root, final_datadir, img_feat_path, txt_feat_path, category_feat_path = resolve_paths()
    outfit_feat_path = get_outfit_feat_path(feat_dir, feat_tag)

    polyvore_split = "disjoint"
    target_split_dir = os.path.join(data_root, polyvore_split)
    if not os.path.exists(target_split_dir):
        raise FileNotFoundError(f"找不到 split 資料夾：{target_split_dir}")

    # ckpt
    exp_dir = os.path.join(exp_root, f"{model_tag}_seed{seed}")
    ckpt_path = os.path.join(exp_dir, "ckpt.pt")
    print(f"[CIR] seed={seed}, model={model_tag}, feat={feat_tag}, exp_dir={exp_dir}")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"找不到 checkpoint：{ckpt_path}")

    # dataset
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
        split="train",
        max_item_len=max_item_len,
        transform=transform,
        sample_hard_negative=False
    )

    test_dat = UIUCPolyvoreRetrievalDataset(
        datadir=final_datadir,
        txt_feat_path=txt_feat_path,
        img_feat_path=img_feat_path,
        category_feat_path=category_feat_path,
        outfit_feat_path=outfit_feat_path,
        polyvore_split=polyvore_split,
        split="test",
        max_item_len=max_item_len,
        transform=transform,
        sample_hard_negative=False
    )

    # device
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # model
    model = OutfitTransformerRetrieval(
        img_encoder=LinearImageEncoder(img_inp_size=img_inp_size, img_emb_size=img_emb_size),
        text_encoder=LinearTextEncoder(txt_inp_size=txt_inp_size, txt_emb_size=txt_emb_size),
        outfit_txt_encoder=LinearTextEncoder(
            txt_inp_size=outfit_txt_inp_size,
            txt_emb_size=txt_emb_size + img_emb_size
        ),
        nhead=nhead,
        num_layers=num_layers,
        margin=margin,
        target_item_info="category",
        use_outfit_txt=use_outfit_txt
    )

    checkpoint = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(checkpoint["model"])
    model = model.to(device)
    model.eval()

    # === build distractors (same style as your original) ===
    distractors_id = dict()
    distractors_feats = dict()
    distractors_idSet = dict()

    for k_fg, v in tqdm(test_dat.fg2ims.items(), desc=f"build distractors (test) [{run_tag}]"):
        if k_fg not in distractors_id:
            distractors_id[k_fg] = list()
            distractors_feats[k_fg] = list()
            distractors_idSet[k_fg] = set()
        for set_id, item_lst in v.items():
            for item_id in item_lst:
                if item_id not in distractors_idSet[k_fg]:
                    distractors_idSet[k_fg].add(item_id)
                    distractors_id[k_fg].append(item_id)
                    distractors_feats[k_fg].append(
                        test_dat.extract_emb({'item_id': item_id}, model, device, ctx, to_numpy=True)
                    )
                if len(distractors_id[k_fg]) >= 3000:
                    break
            if len(distractors_id[k_fg]) >= 3000:
                break

    for k_fg, v in tqdm(train_dat.fg2ims.items(), desc=f"build distractors (train fill) [{run_tag}]"):
        if k_fg not in distractors_id:
            break
        for set_id, item_lst in v.items():
            for item_id in item_lst:
                if item_id not in distractors_idSet[k_fg]:
                    distractors_idSet[k_fg].add(item_id)
                    distractors_id[k_fg].append(item_id)
                    distractors_feats[k_fg].append(
                        test_dat.extract_emb({'item_id': item_id}, model, device, ctx, to_numpy=True)
                    )
                if len(distractors_id[k_fg]) >= 3000:
                    break
            if len(distractors_id[k_fg]) >= 3000:
                break

    distractors_id = {k_fg: v for k_fg, v in distractors_id.items() if len(v) >= 3000}
    distractors_feats = {k_fg: v for k_fg, v in distractors_feats.items() if len(v) >= 3000}

    distractors_id = {k_fg: np.array(v) for k_fg, v in distractors_id.items()}
    distractors_feats = {k_fg: torch.tensor(np.concatenate(v)) for k_fg, v in distractors_feats.items()}

    print(f'[{len(distractors_feats)}/{len(test_dat.fg2ims)}] fine-grained categories are selected.')

    def retrieval_top_k(logits: torch.tensor, dist_feats: torch.tensor, dist_id: np.ndarray, kk: int):
        score = torch.cosine_similarity(logits, dist_feats)
        values, indices = torch.sort(score, descending=True)
        topIdx = dist_id[indices.numpy()]
        return topIdx[:kk]

    # === detail writer ===
    detail_writer = None
    detail_f = None
    if save_detail:
        if detail_file is None:
            detail_file = os.path.join(exp_root, f"detail_cir_{run_tag}_seed{seed}.csv")
        detail_f = open(detail_file, "w", newline="", encoding="utf-8")
        detail_writer = csv.writer(detail_f)
        detail_writer.writerow([
            "run_tag", "seed", "set_id", "target_item_id", "target_item_fg",
            "rank", "hit@1", "hit@3", "hit@5", "hit@10", "hit@30", "hit@50",
            "partial_item_ids",
            "model_top1_item_id",
            "recommended_outfit_item_ids",
            "top10_ids"
        ])

    # === evaluate recall@topk ===
    count = 0
    k_list = [1, 3, 5, 10, 30, 50]
    correct = [0] * len(k_list)

    for outfits, is_correct, set_id in tqdm(test_dat.fitb_questions, desc=f'compute recall@topk [{run_tag}]'):
        target_item_id = outfits[np.where(is_correct)[0][0]]['items'][-1]['item_id']
        target_item_fg = test_dat.im2fg[target_item_id]
        partial_item_ids = [
            int(item["item_id"]) for item in outfits[0]["items"][:-1]
        ]
        if target_item_fg not in distractors_feats:
            continue

        partial_imgs, partial_txts, _ = test_dat.load_outfit({'items': outfits[0]['items'][:-1]})
        partial_imgs, partial_txts, mask = test_dat.pad_imgs_and_txts(partial_imgs, partial_txts)
        partial_imgs, partial_txts, mask = partial_imgs.to(device), partial_txts.to(device), mask.to(device)

        target_item_category = torch.tensor(test_dat.category_feats[target_item_id]).to(device)
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
            cur_dist_feats = torch.cat([
                cur_dist_feats[:-1, :],
                test_dat.extract_emb({'item_id': target_item_id}, model, device, ctx, to_numpy=False).detach().cpu()
            ], dim=0)

        topIdx = retrieval_top_k(logits, cur_dist_feats, cur_dist_id, k_list[-1])

        # rank (full sorting)
        score = torch.cosine_similarity(logits, cur_dist_feats)
        values, indices = torch.sort(score, descending=True)
        sorted_ids = cur_dist_id[indices.numpy()]
        rank = int(np.where(sorted_ids == target_item_id)[0][0]) + 1

        hit1 = int(rank <= 1)
        hit3 = int(rank <= 3)
        hit5 = int(rank <= 5)
        hit10 = int(rank <= 10)
        hit30 = int(rank <= 30)
        hit50 = int(rank <= 50)

        if detail_writer is not None:
    top10 = [int(x) for x in topIdx[:10].tolist()]
    model_top1_item_id = int(top10[0])
    recommended_outfit_item_ids = partial_item_ids + [model_top1_item_id]

    detail_writer.writerow([
            args.run_tag,
            seed,
            int(set_id),
            int(target_item_id),
            str(target_item_fg),
            rank,
            hit1,
            hit3,
            hit5,
            hit10,
            hit30,
            hit50,
            json.dumps(partial_item_ids, ensure_ascii=False),
            model_top1_item_id,
            json.dumps(recommended_outfit_item_ids, ensure_ascii=False),
            json.dumps(top10, ensure_ascii=False)
        ])
        for i, kk in enumerate(k_list):
            if target_item_id in topIdx[:kk]:
                correct[i] += 1

    recall = [c / count for c in correct]
    recall_at_1, recall_at_3, recall_at_5, recall_at_10, recall_at_30, recall_at_50 = recall

    for kk, recall_ in zip(k_list, recall):
        print(f'[{run_tag}] Recall@top{kk}: {recall_ * 100:.4f}%')

    if detail_f is not None:
        detail_f.close()
        print(f"[{run_tag}] detail 已寫入 {detail_file}")

    return {
        "seed": seed,
        "model_tag": model_tag,
        "feat_tag": feat_tag,
        "recall_at_1": float(recall_at_1),
        "recall_at_3": float(recall_at_3),
        "recall_at_5": float(recall_at_5),
        "recall_at_10": float(recall_at_10),
        "recall_at_30": float(recall_at_30),
        "recall_at_50": float(recall_at_50),
    }


def append_results(results_file, row):
    file_exists = os.path.exists(results_file)
    with open(results_file, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow([
                "seed", "model_tag", "feat_tag",
                "recall_at_1", "recall_at_3", "recall_at_5", "recall_at_10", "recall_at_30", "recall_at_50"
            ])
        writer.writerow([
            row["seed"], row["model_tag"], row["feat_tag"],
            row["recall_at_1"], row["recall_at_3"], row["recall_at_5"],
            row["recall_at_10"], row["recall_at_30"], row["recall_at_50"]
        ])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--exp_root", type=str, default="./experiments")

    parser.add_argument("--model_tag", choices=["cir_old", "cir_new"], default="cir_new")
    parser.add_argument("--feat_tag", choices=["old", "new"], default="new")
    parser.add_argument("--sweep", action="store_true")

    parser.add_argument("--save_detail", action="store_true")
    parser.add_argument("--detail_file", type=str, default=None)
    parser.add_argument("--run_tag", type=str, default=None)

    # 避免 notebook/ipykernel 注入 -f kernel.json 造成 argparse 爆掉
    args, _ = parser.parse_known_args()

    results_file = os.path.join(args.exp_root, "results_cir_sweep.csv")

    if not args.sweep:
        out = run_once(
            seed=args.seed,
            exp_root=args.exp_root,
            model_tag=args.model_tag,
            feat_tag=args.feat_tag,
            save_detail=args.save_detail,
            detail_file=args.detail_file,
            run_tag=args.run_tag
        )
        append_results(results_file, out)
        print(f"完成，結果已寫入：{results_file}")
        return

    combos = [
        ("cir_old", "old"),
        ("cir_old", "new"),
        ("cir_new", "old"),
        ("cir_new", "new"),
    ]
    for mtag, ftag in combos:
        out = run_once(
            seed=args.seed,
            exp_root=args.exp_root,
            model_tag=mtag,
            feat_tag=ftag,
            save_detail=args.save_detail,
            detail_file=None,   # sweep 不共用 detail_file，避免覆蓋
            run_tag=None        # sweep 自動用 cir_xxx_featyyy 命名
        )
        append_results(results_file, out)

    print(f"完成，結果已寫入：{results_file}")


if __name__ == "__main__":
    main()
