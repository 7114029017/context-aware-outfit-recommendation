#!/usr/bin/env python3
"""Counterfactual context sensitivity (manuscript Tables 4 and 9) with a run's own models.

Ported from the notebook
03_實驗與結果_experiments_results/06_反事實情境敏感度/source_programs/P16_counterfactual_context_consistency_check.ipynb
(cells 5 and 7). The 24 counterfactual pairs are the archived A44 file; they do
not depend on a model. For each pair the Context CIR model ranks the target item
among 3,000 candidates of its fine-grained category (the archived evaluator's
pool rule) under three conditions:

- context_aware: the stored context-aware text feature, as in P16 and the manuscript;
- counterfactual: the counterfactual description encoded with FashionCLIP
  (patrickjohncyh/fashion-clip, the revision recorded in P16's log), as in P16;
- context_aware_live: the context-aware description encoded with the same
  encoder, a symmetric control that P16 did not have.

Table 9 is the mean rank change (counterfactual minus context_aware), the share
of pairs whose top-1 item changes and the mean top-5 Jaccard, overall and by
factor; it is also given against context_aware_live.

Models: --run-root RUN uses RUN/main/context_seed<k>/cir_best_ckpt.pt for
--seeds (default 1-5); --validate-2025 uses the preserved 2025 seed-1
checkpoint and compares with the archived A45 output. The model code is a
working copy of the archived code with the same standardized decoder as the
full run's evaluation. CPU (float32) by default; P16 ran on CUDA with float16
autocast, which moves some ranks by one.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import sys
import tempfile
from contextlib import nullcontext
from copy import deepcopy
from pathlib import Path
from statistics import mean, stdev

import numpy as np

from _ext import (CHECKPOINTS_2025, COUNTERFACTUAL_DIR, FEATURES, MODEL_CODE, REPO, REPRO, SCRIPTS, local_path,
                  output_dir, passed_run, sha256, write_csv, write_json, write_text)

sys.path.insert(0, str(SCRIPTS))
import standard_decoder  # noqa: E402

PAIRS = COUNTERFACTUAL_DIR / "A44_counterfactual_context_pairs.csv"
ARCHIVED = COUNTERFACTUAL_DIR / "A45_counterfactual_context_retrieval_seed1.csv"
POOL_SIZE = 3000
FACTORS = ("weather", "occasion", "style")
MANUSCRIPT_TABLE9 = {  # manuscript Table 9: mean rank change, top-1 changed share, top-5 Jaccard
    "Overall": ("+17.0", "0.500", "0.351"), "Weather": ("-146.4", "0.125", "0.508"),
    "Occasion": ("-11.9", "0.375", "0.416"), "Style": ("+209.1", "1.000", "0.130"),
}


def read_rows(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


class FashionClipText:
    """FashionCLIP text features as fashion-clip 0.2.2 computes them (encode_text, batch size 16)."""

    def __init__(self, model_dir: Path):
        import torch
        from transformers import CLIPModel, CLIPProcessor
        self.torch = torch
        self.model = CLIPModel.from_pretrained(model_dir, local_files_only=True).eval()
        self.processor = CLIPProcessor.from_pretrained(model_dir, local_files_only=True)

    def encode(self, texts: list[str], batch_size: int = 16) -> np.ndarray:
        out = []
        with self.torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch = self.processor(text=texts[i:i + batch_size], return_tensors="pt", max_length=77,
                                       padding="max_length", truncation=True)
                out.extend(self.model.get_text_features(**batch).numpy())
        return np.stack(out).astype(np.float32)


class Retrieval:
    """P16 cell 5: datasets, candidate pools and ranking, for one model at a time."""

    def __init__(self, polyvore: Path, device: str, work: Path):
        src = work / "main_hybrid_attention_code"
        shutil.copytree(MODEL_CODE, src)
        standard_decoder.apply(src / "outfit_transformer.py")
        if device == "cpu":
            cfg = src / "config" / "base_config.py"
            text = cfg.read_text(encoding="utf-8")
            switched = text.replace("device_type = 'cuda'", "device_type = 'cpu'", 1)
            if switched == text:
                raise SystemExit("[COUNTERFACTUAL BLOCKED] could not switch the working-copy config to CPU")
            cfg.write_text(switched, encoding="utf-8")
        (work / "polyvore_data").mkdir()
        (work / "polyvore_data" / "polyvore_outfits").symlink_to(polyvore, target_is_directory=True)
        sys.path.insert(0, str(src))
        previous = os.getcwd()
        os.chdir(work)  # the archived config creates runs/<name> relative to the working directory
        try:
            import torch
            from dataset import UIUCPolyvoreRetrievalDataset
            from outfit_transformer import LinearImageEncoder, LinearTextEncoder, OutfitTransformerRetrieval
            from config.cir_cond_hardneg import (img_emb_size, img_inp_size, margin, max_item_len, nhead, num_layers,
                                                outfit_txt_inp_size, txt_emb_size, txt_inp_size, use_outfit_txt)
        finally:
            os.chdir(previous)
        self.torch, self.device = torch, device
        self.ctx = nullcontext() if device == "cpu" else torch.amp.autocast(device_type="cuda", dtype=torch.float16)
        common = dict(datadir=str(work / "polyvore_data"),
                      txt_feat_path=str(FEATURES / "encoded_title_description_distiluse-base-multilingual-cased-v2.pkl"),
                      img_feat_path=str(FEATURES / "img_feats_fashionClip.pkl"),
                      category_feat_path=str(FEATURES / "encoded_category_distiluse-base-multilingual-cased-v2.pkl"),
                      polyvore_split="disjoint", max_item_len=max_item_len, transform=None, sample_hard_negative=False)
        outfit_feat = str(FEATURES / "encoded_NewoutfitUrlTitle_en_fashionClip.pkl")
        self.train_dat = UIUCPolyvoreRetrievalDataset(split="train", outfit_feat_path=outfit_feat, **common)
        self.test_dat = UIUCPolyvoreRetrievalDataset(split="test", outfit_feat_path=outfit_feat, **common)
        self.make_model = lambda: OutfitTransformerRetrieval(
            img_encoder=LinearImageEncoder(img_inp_size, img_emb_size),
            text_encoder=LinearTextEncoder(txt_inp_size, txt_emb_size),
            outfit_txt_encoder=LinearTextEncoder(outfit_txt_inp_size, txt_emb_size + img_emb_size),
            nhead=nhead, num_layers=num_layers, margin=margin, target_item_info="category",
            use_outfit_txt=use_outfit_txt)
        self.model = None
        self.pool_ids = {}

    def stored_vector(self, set_id: str) -> np.ndarray:
        return np.asarray(self.test_dat.outfit_feats[str(set_id)], dtype=np.float32)

    def build_pool_for_fg(self, fg: str) -> list[str]:
        ids, seen = [], set()
        for dat in (self.test_dat, self.train_dat):
            for _, item_list in dat.fg2ims.get(str(fg), {}).items():
                for item_id in item_list:
                    item_id = str(item_id)
                    if item_id not in seen:
                        seen.add(item_id)
                        ids.append(item_id)
                    if len(ids) >= POOL_SIZE:
                        return ids[:POOL_SIZE]
        return ids

    def load(self, checkpoint: Path, fgs: list[str]) -> None:
        torch = self.torch
        model = self.make_model()
        model.load_state_dict(torch.load(str(checkpoint), map_location=self.device)["model"])
        self.model = model.to(self.device).eval()
        self.cache = {}
        self.pool_feats = {}
        for fg in fgs:
            if fg not in self.pool_ids:
                ids = self.build_pool_for_fg(fg)
                if len(ids) < POOL_SIZE:
                    raise SystemExit(f"[COUNTERFACTUAL BLOCKED] category {fg} has {len(ids)} candidates, not {POOL_SIZE}")
                self.pool_ids[fg] = np.array(ids, dtype=str)
            self.pool_feats[fg] = torch.from_numpy(
                np.concatenate([self.item_emb(x) for x in self.pool_ids[fg]], axis=0)).float()

    def item_emb(self, item_id: str) -> np.ndarray:
        if item_id not in self.cache:
            emb = self.test_dat.extract_emb({"item_id": item_id}, self.model, self.device, self.ctx, to_numpy=True)
            self.cache[item_id] = np.asarray(emb, dtype=np.float32)
        return self.cache[item_id]

    def rank(self, partial_ids: list[str], target_item_id: str, fg: str, outfit_vec: np.ndarray):
        torch, dat = self.torch, self.test_dat
        stack = (lambda x: torch.stack(x) if isinstance(x, list) else x)
        imgs, txts, _ = dat.load_outfit({"items": [{"item_id": str(x)} for x in partial_ids]})
        imgs, txts, mask = dat.pad_imgs_and_txts(stack(imgs), stack(txts))
        category = torch.tensor(dat.category_feats[str(target_item_id)]).to(self.device)
        outfit_txt = torch.tensor(np.asarray(outfit_vec, dtype=np.float32)).to(self.device)
        with torch.no_grad(), self.ctx:
            out = self.model(imgs.to(self.device).unsqueeze(0), txts.to(self.device).unsqueeze(0),
                             mask.to(self.device).unsqueeze(0), positive_category=category.unsqueeze(0),
                             outfit_txt=outfit_txt.unsqueeze(0))
        logits = out["logits"].detach().cpu().float()
        cur_ids = deepcopy(self.pool_ids[fg])
        cur_feats = self.pool_feats[fg].clone()
        if target_item_id not in set(cur_ids.tolist()):
            cur_ids = np.concatenate([cur_ids[:-1], np.array([target_item_id], dtype=str)])
            cur_feats = torch.cat([cur_feats[:-1, :], torch.from_numpy(self.item_emb(target_item_id)).float()], dim=0)
        scores = torch.cosine_similarity(logits, cur_feats)
        values, indices = torch.sort(scores, descending=True)
        sorted_ids = cur_ids[indices.numpy()].astype(str)
        rank = int(np.where(sorted_ids == target_item_id)[0][0]) + 1
        return rank, sorted_ids, values.numpy().astype(float)


def score_model(retrieval: Retrieval, label: str, checkpoint: Path, pairs: list[dict], vectors: dict) -> list[dict]:
    retrieval.load(checkpoint, sorted({p["target_item_fg"] for p in pairs}))
    rows = []
    for p in pairs:
        partial = [x.strip() for x in p["partial_item_ids"].split("|") if x.strip()]
        for condition, text, vec in (
                ("context_aware", p["context_aware_description"], retrieval.stored_vector(p["set_id"])),
                ("counterfactual", p["counterfactual_description"], vectors["counterfactual"][p["pair_id"]]),
                ("context_aware_live", p["context_aware_description"], vectors["context_aware_live"][p["pair_id"]])):
            rank, ids, scores = retrieval.rank(partial, p["target_item_id"], p["target_item_fg"], vec)
            rows.append({"model": label, "pair_id": p["pair_id"], "condition": condition, "set_id": p["set_id"],
                         "factor": p["factor"], "direction": p["direction"], "query_text": text,
                         "target_item_id": p["target_item_id"], "target_item_fg": p["target_item_fg"], "rank": rank,
                         "hit@1": int(rank <= 1), "hit@5": int(rank <= 5), "hit@10": int(rank <= 10),
                         "top1_id": ids[0], "top5_ids": " | ".join(ids[:5]), "top10_ids": " | ".join(ids[:10]),
                         "top10_scores": " | ".join(f"{x:.6f}" for x in scores[:10])})
    return rows


def table9(rows: list[dict], baseline: str) -> list[list]:
    """P16 cell 7 / manuscript Table 9: counterfactual against the given baseline condition."""
    base = {r["pair_id"]: r for r in rows if r["condition"] == baseline}
    cf = {r["pair_id"]: r for r in rows if r["condition"] == "counterfactual"}
    cases = []
    for pid, b in base.items():
        c = cf[pid]
        tb, tc = set(b["top5_ids"].split(" | ")), set(c["top5_ids"].split(" | "))
        cases.append({"factor": b["factor"], "r0": int(b["rank"]), "r1": int(c["rank"]),
                      "top1": int(str(b["top1_id"]) != str(c["top1_id"])), "jaccard": len(tb & tc) / len(tb | tc)})
    out = []
    for label, group in [("Overall", cases)] + [(f.capitalize(), [c for c in cases if c["factor"] == f]) for f in FACTORS]:
        out.append([label, len(group), mean(g["r0"] for g in group), mean(g["r1"] for g in group),
                    mean(g["r1"] - g["r0"] for g in group), mean(g["top1"] for g in group),
                    mean(g["jaccard"] for g in group)])
    return out


TABLE9_HEADER = ["scope", "n", "mean_rank_before", "mean_rank_after", "mean_rank_change", "top1_changed_share",
                 "top5_jaccard"]


def table9_text(t: list[list]) -> list[list]:
    return [[r[0], r[1], f"{r[2]:.1f}", f"{r[3]:.1f}", f"{r[4]:+.1f}", f"{r[5]:.3f}", f"{r[6]:.3f}"] for r in t]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-root", type=Path, default=None, help="completed full run folder with the checkpoints")
    parser.add_argument("--seeds", default="1,2,3,4,5")
    parser.add_argument("--validate-2025", action="store_true",
                        help="also score the preserved 2025 seed-1 checkpoint and compare with A45")
    parser.add_argument("--out-dir", type=Path, default=None, help="default: <run>/extensions/counterfactual")
    parser.add_argument("--polyvore-root", default=None)
    parser.add_argument("--fashionclip", default=None, help="FashionCLIP folder (bootstrap_data.sh --with-fashionclip)")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    if not args.run_root and not args.validate_2025:
        raise SystemExit("[COUNTERFACTUAL BLOCKED] pass --run-root and/or --validate-2025")

    import torch
    torch.set_num_threads(args.threads)
    run_root = passed_run(args.run_root) if args.run_root else None
    if args.out_dir is None and run_root is None:
        raise SystemExit("[COUNTERFACTUAL BLOCKED] pass --out-dir")
    out = output_dir(args.out_dir or run_root / "extensions" / "counterfactual", run_root)
    polyvore = local_path("polyvore_root", args.polyvore_root)
    fclip_dir = local_path("fashionclip_root", args.fashionclip)
    if polyvore is None or not (polyvore / "disjoint" / "test.json").is_file():
        raise SystemExit("[COUNTERFACTUAL BLOCKED] Polyvore root not found; pass --polyvore-root or run bootstrap_data.sh")
    if fclip_dir is None or not (fclip_dir / "model.safetensors").is_file():
        raise SystemExit("[COUNTERFACTUAL BLOCKED] FashionCLIP not found; run bootstrap_data.sh --with-fashionclip")

    pairs = read_rows(PAIRS)
    encoder = FashionClipText(fclip_dir)
    vectors = {"counterfactual": dict(zip([p["pair_id"] for p in pairs],
                                          encoder.encode([p["counterfactual_description"] for p in pairs]))),
               "context_aware_live": dict(zip([p["pair_id"] for p in pairs],
                                              encoder.encode([p["context_aware_description"] for p in pairs])))}
    models = []
    if args.validate_2025:
        models.append(("2025_seed1", CHECKPOINTS_2025 / "cir_new_seed1" / "ckpt.pt"))
    if run_root:
        for seed in [int(s) for s in args.seeds.split(",")]:
            unit = run_root / "main" / f"context_seed{seed}"
            manifest = json.loads((unit / "full_train_manifest.json").read_text(encoding="utf-8"))
            ckpt = unit / "cir_best_ckpt.pt"
            if sha256(ckpt) != manifest.get("cir_checkpoint_sha256"):
                raise SystemExit(f"[COUNTERFACTUAL BLOCKED] {ckpt}: SHA-256 differs from its training manifest")
            models.append((f"seed{seed}", ckpt))

    work_root = REPRO / ".work"
    work_root.mkdir(exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix="counterfactual_", dir=work_root))
    try:
        retrieval = Retrieval(polyvore, args.device, work)
        encoder_rows = []
        for p in pairs:
            stored = retrieval.stored_vector(p["set_id"])
            live = vectors["context_aware_live"][p["pair_id"]]
            cos = float(np.dot(stored, live) / (np.linalg.norm(stored) * np.linalg.norm(live)))
            encoder_rows.append([p["pair_id"], p["set_id"], f"{cos:.6f}", f"{float(np.abs(stored - live).max()):.6f}"])
        write_csv(out / "encoder_check.csv", ["pair_id", "set_id", "cosine_stored_vs_live", "max_abs_difference"],
                  encoder_rows)
        results = {}
        for label, ckpt in models:
            print(f"[COUNTERFACTUAL] scoring {label}: {ckpt}", flush=True)
            rows = score_model(retrieval, label, ckpt, pairs, vectors)
            results[label] = rows
            write_csv(out / f"cases_{label}.csv", list(rows[0]), [list(r.values()) for r in rows])
            write_csv(out / f"table9_{label}.csv", TABLE9_HEADER, table9_text(table9(rows, "context_aware")))
            write_csv(out / f"table9_live_baseline_{label}.csv", TABLE9_HEADER,
                      table9_text(table9(rows, "context_aware_live")))
    finally:
        shutil.rmtree(work, ignore_errors=True)

    lines = ["# Counterfactual context sensitivity (manuscript Tables 4 and 9)", "",
             "Ported from notebook P16. Pairs: the archived A44 file (24 pairs). Baseline condition as in P16 and the",
             f"manuscript: the stored context-aware feature. Device: {args.device}.", ""]
    cos = [float(r[2]) for r in encoder_rows]
    lines += [f"Encoder check (24 context-aware descriptions): cosine between the stored feature and the live FashionCLIP "
              f"encoding, min {min(cos):.6f}, mean {mean(cos):.6f}.", ""]
    seeds = [k for k in results if k.startswith("seed")]
    if seeds:
        tables = {k: table9(results[k], "context_aware") for k in seeds}
        summary = []
        for i, scope in enumerate(t[0] for t in tables[seeds[0]]):
            cells = [[tables[k][i][j] for k in seeds] for j in range(2, 7)]
            summary.append([scope, tables[seeds[0]][i][1]] +
                           [f"{mean(c):.3f} ± {stdev(c):.3f}" if len(c) > 1 else f"{c[0]:.3f}" for c in cells])
        write_csv(out / "table9_seeds_mean_sd.csv", TABLE9_HEADER, summary)
        lines += [f"## Run `{run_root.name}`, Context models of seeds {', '.join(s[4:] for s in seeds)}", "",
                  "| Scope | Seed | Mean rank before → after | Mean rank change | Top-1 changed | Top-5 Jaccard |",
                  "|---|---|---|---:|---:|---:|"]
        for k in seeds:
            lines += [f"| {r[0]} | {k[4:]} | {r[2]} → {r[3]} | {r[4]} | {r[5]} | {r[6]} |"
                      for r in table9_text(tables[k])]
        lines += ["", "Mean ± SD over the seeds: `table9_seeds_mean_sd.csv`. Against the live-encoded baseline: "
                      "`table9_live_baseline_<seed>.csv`.", ""]
    if "2025_seed1" in results:
        archived = {(r["pair_id"], r["condition"]): r for r in read_rows(ARCHIVED)}
        rows = results["2025_seed1"]
        diffs = {c: [abs(int(r["rank"]) - int(archived[(r["pair_id"], c)]["rank"])) for r in rows if r["condition"] == c]
                 for c in ("context_aware", "counterfactual")}
        same_top = {c: sum(1 for r in rows if r["condition"] == c and
                           r["top5_ids"] == archived[(r["pair_id"], c)]["top5_ids"]) for c in diffs}
        t = table9_text(table9(rows, "context_aware"))
        lines += ["## Validation with the preserved 2025 seed-1 model", "",
                  "| Condition | Ranks equal to A45 | Largest rank difference | Top-5 lists equal to A45 |",
                  "|---|---:|---:|---:|"]
        lines += [f"| {c} | {sum(1 for d in v if d == 0)}/24 | {max(v)} | {same_top[c]}/24 |" for c, v in diffs.items()]
        lines += ["", "| Scope | Manuscript (change, top-1, Jaccard) | Recomputed |", "|---|---|---|"]
        lines += [f"| {r[0]} | {', '.join(MANUSCRIPT_TABLE9[r[0]])} | {r[4]}, {r[5]}, {r[6]} |" for r in t]
        lines.append("")
    write_text(out / "summary.md", lines)
    def where(path: Path) -> str:
        if run_root and path.is_relative_to(run_root):
            return f"<run>/{path.relative_to(run_root)}"
        return str(path.relative_to(REPO)) if path.is_relative_to(REPO) else path.name
    write_json(out / "manifest.json", {
        "device": args.device, "threads": args.threads, "run": run_root.name if run_root else None,
        "pairs": str(PAIRS.relative_to(REPO)), "pairs_sha256": sha256(PAIRS),
        "fashionclip": "patrickjohncyh/fashion-clip@7e3ba62ce16b379a1ab479346b66f192e76f51b7",
        "fashionclip_weights_sha256": sha256(fclip_dir / "model.safetensors"),
        "models": {k: {"checkpoint": where(v), "sha256": sha256(v)} for k, v in models}})
    print("\n".join(lines))


if __name__ == "__main__":
    main()
