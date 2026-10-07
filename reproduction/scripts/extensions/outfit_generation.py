#!/usr/bin/env python3
"""Step-by-step outfit generation for one test outfit (the 2025 try-on demo P02, without the try-on).

Ported from 03_實驗與結果_experiments_results/04_情境子集與三因子分析/source_programs/P02_subset_robustness_analysis.ipynb
(cells 1 and 2). For test outfit 224188768 the seed-1 Context CIR model fills the outfit's FastFit slots
(upper, lower, overall, shoe, bag; this outfit has overall, shoe and bag) one at a time from the
context-aware description, which is encoded live with FashionCLIP as in P02: Keep0 generates every slot
from an empty outfit, Keep1 keeps the original item of the first slot, Keep2 those of the first two. Each
step conditions the model on the mean category feature of the slot's first 512 candidates, ranks the
candidate bank (the test items with an image whose major category maps to the slot) by cosine similarity
and takes the top item that is not already in the outfit.

P02 then dressed a person photo in each outfit with FastFit (FastFit-MR-1024, 30 steps, guidance 2.5, seed
42). The person photo (model_image.png) was not preserved and FastFit is not part of the repository, so this
script stops before the try-on. It writes the generated items and scores (generation.csv), summary.md and,
outside the repository, P02's item grid (Polyvore photos, never committed).

--run-root RUN uses RUN/main/context_seed1; --validate-2025 uses the preserved 2025 cir_new_seed1 checkpoint
and compares the picks and scores with those printed by P02. P02 ran on CUDA with float16 autocast; the
default here is the CPU (float32), which changes the scores slightly.
"""
from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from collections import defaultdict
from pathlib import Path

from PIL import Image, ImageDraw

from _ext import (CHECKPOINTS_2025, GENERATED, OFFICIAL_RUN, REPO, REPRO, local_path, output_dir, passed_run,
                  read_json, sha256, write_csv, write_json, write_text)
from case_figures import font, safe_out_dir
from counterfactual import FashionClipText
from reliability_analysis import Evaluator

CASE_SET_ID = "224188768"
DISPLAY_ORDER = ("upper", "lower", "overall", "shoe", "bag")
MAJOR_TO_SLOT = {"tops": "upper", "outerwear": "upper", "bottoms": "lower", "all-body": "overall", "shoes": "shoe",
                 "bags": "bag"}
TOPK_PER_STEP = 20
CONDITION_ITEMS = 512
P02_PRINTED = {  # P02 cell 2 output, 2025 cir_new_seed1 on CUDA: (slot, item, score) per step
    "keep0": [("overall", "56998165", 0.7139), ("shoe", "121507083", 0.8288), ("bag", "161194648", 0.7800)],
    "keep1": [("shoe", "82253359", 0.7508), ("bag", "112382686", 0.7622)],
    "keep2": [("bag", "112382686", 0.6873)],
}
P02_MEAN_SCORES = {"keep0": 0.7742259502410889, "keep1": 0.7565205991268158, "keep2": 0.6873121857643127}
TILE, GAP = 150, 10


def normalize_text_value(v) -> str:
    if v is None:
        return ""
    if isinstance(v, str):
        return v.strip()
    if isinstance(v, list):
        return " | ".join(str(x).strip() for x in v if str(x).strip())
    if isinstance(v, dict):
        return " ".join(str(v[k]).strip() for k in ("url_name", "url", "title") if k in v and str(v[k]).strip())
    return str(v).strip()


def canon_category_id(x) -> str | None:
    s = str(x).strip() if x is not None else ""
    if s == "" or s.lower() == "nan":
        return None
    try:
        f = float(s)
        return str(int(f)) if f.is_integer() else s
    except ValueError:
        return s


def category_slots(path: Path) -> dict[str, str | None]:
    """P02 load_category_major_map: category ID (first row) -> FastFit slot of its major category."""
    import csv
    out = {}
    with path.open(encoding="utf-8", newline="") as f:
        for row in csv.reader(f):
            cid = canon_category_id(row[0]) if row else None
            if cid is None or cid in out:
                continue
            major = row[2].strip().lower() if len(row) > 2 else ""
            out[cid] = MAJOR_TO_SLOT.get(major)
    return out


class Generator:
    """P02 sections 6-10 for one checkpoint."""

    def __init__(self, evaluator: Evaluator, checkpoint: Path, polyvore: Path, encoder: FashionClipText):
        torch = evaluator.torch
        self.torch, self.ev, self.encoder = torch, evaluator, encoder
        (self.dat,) = evaluator.datasets("encoded_NewoutfitUrlTitle_en_fashionClip.pkl", ("test",))
        model = evaluator.make_model()
        model.load_state_dict(torch.load(str(checkpoint), map_location=evaluator.device)["model"], strict=True)
        self.model = model.to(evaluator.device).eval()
        slots = category_slots(polyvore / "categories.csv")
        images = polyvore / "images"
        self.slot, self.image = {}, {}
        for item_id in self.dat.category_feats.keys():  # P02 build_item_meta_from_fg
            item_id = str(item_id)
            self.slot[item_id] = slots.get(canon_category_id(self.dat.im2fg.get(item_id)))
            path = next((images / f"{item_id}{ext}" for ext in (".jpg", ".jpeg", ".png", ".webp")
                         if (images / f"{item_id}{ext}").exists()), None)
            self.image[item_id] = path
        ids = defaultdict(list)  # P02 build_candidate_bank
        for item_id, slot in self.slot.items():
            if slot is not None and self.image[item_id] is not None:
                ids[slot].append(item_id)
        self.bank = {}
        for slot, slot_ids in ids.items():
            embs = []
            for item_id in slot_ids:
                with torch.no_grad(), evaluator.ctx:
                    emb = self.dat.extract_emb({"item_id": item_id}, self.model, evaluator.device, evaluator.ctx,
                                               to_numpy=False)
                embs.append(emb.detach().cpu().float())
            self.bank[slot] = {"ids": slot_ids, "embs": torch.cat(embs, dim=0)}
        self.condition = {slot: torch.stack([torch.tensor(self.dat.category_feats[i], dtype=torch.float32)
                                             for i in pack["ids"][:CONDITION_ITEMS]]).mean(dim=0).to(evaluator.device)
                          for slot, pack in self.bank.items()}
        any_id = next(iter(self.dat.category_feats.keys()))
        imgs, txts, _ = self.dat.load_outfit({"items": [{"item_id": str(any_id)}]})
        imgs, txts, _ = self.dat.pad_imgs_and_txts(imgs, txts)
        self.img_dim, self.txt_dim = imgs.shape[-1], txts.shape[-1]

    def partial(self, item_ids: list[str]):
        torch, device, n = self.torch, self.ev.device, self.ev.cfg.max_item_len
        if not item_ids:
            return (torch.zeros((n, self.img_dim), dtype=torch.float32, device=device),
                    torch.zeros((n, self.txt_dim), dtype=torch.float32, device=device),
                    torch.ones((n,), dtype=torch.bool, device=device))
        imgs, txts, _ = self.dat.load_outfit({"items": [{"item_id": i} for i in item_ids]})
        imgs, txts, mask = self.dat.pad_imgs_and_txts(imgs, txts)
        return imgs.to(device), txts.to(device), mask.to(device)

    def generate(self, text: str, schema: list[str], kept: list[str]) -> list[dict]:
        """P02 generate_outfit_keepk with SAMPLING_TOPK = None (the top item each step)."""
        torch = self.torch
        outfit_txt = torch.tensor(self.encoder.encode([text])[0], dtype=torch.float32, device=self.ev.device)
        partial_ids, steps = list(kept), []
        for step, slot in enumerate(schema, start=1):
            imgs, txts, mask = self.partial(partial_ids)
            with torch.no_grad(), self.ev.ctx:
                out = self.model(imgs.unsqueeze(0), txts.unsqueeze(0), mask.unsqueeze(0),
                                 positive_category=self.condition[slot].unsqueeze(0), outfit_txt=outfit_txt.unsqueeze(0))
                logits = out["logits"].detach().cpu().float()
            pack = self.bank[slot]
            sims = torch.nn.functional.cosine_similarity(logits.cpu(), pack["embs"], dim=1)
            ranked = [(pack["ids"][i], float(sims[i].item())) for i in torch.argsort(sims, descending=True).tolist()
                      if pack["ids"][i] not in set(partial_ids)][:TOPK_PER_STEP]
            item_id, score = ranked[0]
            partial_ids.append(item_id)
            steps.append({"step": step, "slot": slot, "item_id": item_id, "score": score,
                          "runner_up": ranked[1][0], "runner_up_score": ranked[1][1]})
        return steps


def case_setup(outfit: dict, slot_of: dict) -> tuple[dict, list[str]]:
    """P02 extract_fastfit_item_ids_from_outfit and infer_schema_from_item_ids."""
    gt = {s: None for s in DISPLAY_ORDER}
    for item in outfit["items"]:
        slot = slot_of.get(str(item["item_id"]))
        if slot in gt and gt[slot] is None:
            gt[slot] = str(item["item_id"])
    schema = [s for s in DISPLAY_ORDER if gt[s] is not None]
    if "overall" in schema and ("upper" in schema or "lower" in schema):
        schema = [s for s in schema if s != "overall"]
    return gt, schema


def item_grid(path: Path, title: str, subtitle: str, rows: list[tuple[str, dict, set, set]], images: dict) -> None:
    """P02 plot_paper_style_grid: Original / Keep2 / Keep1 / Keep0; kept items solid, generated dashed."""
    label_w, top = 110, 76
    width = label_w + len(DISPLAY_ORDER) * (TILE + GAP) + 20
    height = top + 24 + len(rows) * (TILE + GAP) + 16
    canvas = Image.new("RGB", (width, height), "white")
    d = ImageDraw.Draw(canvas)
    d.text((width // 2, 18), title, fill=(17, 17, 17), font=font(18, True), anchor="mm")
    d.text((width // 2, 44), subtitle, fill=(85, 85, 85), font=font(13), anchor="mm")
    for c, slot in enumerate(DISPLAY_ORDER):
        d.text((label_w + c * (TILE + GAP) + TILE // 2, top + 6), slot, fill=(17, 17, 17), font=font(13), anchor="mm")
    for r, (name, refs, kept, generated) in enumerate(rows):
        y = top + 24 + r * (TILE + GAP)
        d.text((label_w - 12, y + TILE // 2), name, fill=(17, 17, 17), font=font(15, True), anchor="rm")
        for c, slot in enumerate(DISPLAY_ORDER):
            x = label_w + c * (TILE + GAP)
            item_id = refs.get(slot)
            path_img = images.get(item_id) if item_id else None
            if path_img:
                img = Image.open(path_img).convert("RGB")
                img.thumbnail((TILE - 12, TILE - 12), Image.Resampling.LANCZOS)
                canvas.paste(img, (x + (TILE - img.width) // 2, y + (TILE - img.height) // 2))
            else:
                d.rectangle([x, y, x + TILE - 1, y + TILE - 1], fill=(238, 238, 238))
                d.text((x + TILE // 2, y + TILE // 2), "N/A", fill=(120, 120, 120), font=font(12), anchor="mm")
            if name == "Original":
                d.rectangle([x, y, x + TILE - 1, y + TILE - 1], outline=(17, 17, 17), width=2)
            elif slot in kept or slot in generated:
                if slot in kept:
                    d.rectangle([x, y, x + TILE - 1, y + TILE - 1], outline=(17, 17, 17), width=3)
                else:
                    for k in range(0, TILE, 12):  # dashed border
                        for a, b in (((x + k, y), (x + min(k + 6, TILE - 1), y)),
                                     ((x + k, y + TILE - 1), (x + min(k + 6, TILE - 1), y + TILE - 1)),
                                     ((x, y + k), (x, y + min(k + 6, TILE - 1))),
                                     ((x + TILE - 1, y + k), (x + TILE - 1, y + min(k + 6, TILE - 1)))):
                            d.line([a, b], fill=(17, 17, 17), width=3)
                tag = "keep" if slot in kept else "gen"
                d.rectangle([x + 4, y + TILE - 22, x + 44, y + TILE - 5], fill="white", outline=(17, 17, 17))
                d.text((x + 24, y + TILE - 13), tag, fill=(17, 17, 17), font=font(11), anchor="mm")
            else:
                d.rectangle([x, y, x + TILE - 1, y + TILE - 1], outline=(187, 187, 187), width=1)
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--run-root", type=Path, help="completed full run folder with the checkpoints")
    source.add_argument("--validate-2025", action="store_true", help="use the preserved 2025 cir_new_seed1 checkpoint")
    parser.add_argument("--out-dir", type=Path, default=None, help="default: <run>/extensions/outfit_generation")
    parser.add_argument("--figures-dir", type=Path, default=None,
                        help="item grid with Polyvore photos; outside the repository, under reproduction/runs/ or "
                             "_external/ (default: _external/figures_official for the official run, else none)")
    parser.add_argument("--polyvore-root", default=None)
    parser.add_argument("--fashionclip", default=None, help="FashionCLIP folder (bootstrap_data.sh --with-fashionclip)")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()

    import torch
    torch.set_num_threads(args.threads)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("[OUTFIT GENERATION BLOCKED] CUDA is not available")
    run_root = passed_run(args.run_root) if args.run_root else None
    if args.out_dir is None and run_root is None:
        raise SystemExit("[OUTFIT GENERATION BLOCKED] pass --out-dir")
    out = output_dir(args.out_dir or run_root / "extensions" / "outfit_generation", run_root)
    polyvore = local_path("polyvore_root", args.polyvore_root)
    fclip_dir = local_path("fashionclip_root", args.fashionclip)
    if polyvore is None or not (polyvore / "images").is_dir():
        raise SystemExit("[OUTFIT GENERATION BLOCKED] Polyvore images not found; run bootstrap_data.sh --with-images")
    if fclip_dir is None or not (fclip_dir / "model.safetensors").is_file():
        raise SystemExit("[OUTFIT GENERATION BLOCKED] FashionCLIP not found; run bootstrap_data.sh --with-fashionclip")
    if run_root:
        unit = run_root / "main" / "context_seed1"
        manifest = json.loads((unit / "full_train_manifest.json").read_text(encoding="utf-8"))
        checkpoint = unit / "cir_best_ckpt.pt"
        if sha256(checkpoint) != manifest.get("cir_checkpoint_sha256"):
            raise SystemExit("[OUTFIT GENERATION BLOCKED] context_seed1: CIR checkpoint SHA-256 differs")
        label = f"run {run_root.name}, main/context_seed1"
    else:
        checkpoint = CHECKPOINTS_2025 / "cir_new_seed1" / "ckpt.pt"
        label = "preserved 2025 cir_new_seed1"

    text = normalize_text_value(read_json(GENERATED)[CASE_SET_ID])
    outfit = next(o for o in read_json(polyvore / "disjoint" / "test.json") if str(o["set_id"]) == CASE_SET_ID)
    encoder = FashionClipText(fclip_dir)
    work_root = REPRO / ".work"
    work_root.mkdir(exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix="outfit_generation_", dir=work_root))
    try:
        evaluator = Evaluator(polyvore, args.device, work)
        gen = Generator(evaluator, checkpoint, polyvore, encoder)
        gt, schema = case_setup(outfit, gen.slot)
        results = {}
        for keep_n in (0, 1, 2):
            kept_slots = schema[:keep_n]  # P02 KEEP_MODE_IF_NOT_SPECIFIED = "prefix"
            results[f"keep{keep_n}"] = {"kept": kept_slots,
                                        "steps": gen.generate(text, [s for s in schema if s not in kept_slots],
                                                              [gt[s] for s in kept_slots])}
        bank_sizes = {slot: len(pack["ids"]) for slot, pack in gen.bank.items()}
        images = gen.image
    finally:
        shutil.rmtree(work, ignore_errors=True)

    rows = []
    for keep, r in results.items():
        for s in r["steps"]:
            rows.append([keep, " ".join(r["kept"]), s["step"], s["slot"], s["item_id"], f"{s['score']:.6f}",
                         s["runner_up"], f"{s['runner_up_score']:.6f}"])
    write_csv(out / "generation.csv", ["setting", "kept_slots", "step", "slot", "item_id", "score", "runner_up_id",
                                       "runner_up_score"], rows)
    lines = ["# Step-by-step outfit generation (2025 try-on demo P02, without the try-on)", "",
             f"Case {CASE_SET_ID}: \"{text}\". Model: {label}; device {args.device}.", "",
             f"Original outfit by slot: " + ", ".join(f"{s} {gt[s]}" for s in DISPLAY_ORDER if gt[s]) +
             f"; generation order {' → '.join(schema)}. Candidate bank: " +
             ", ".join(f"{s} {n:,}" for s, n in sorted(bank_sizes.items())) + " test items.", "",
             "| Setting | Kept | Step | Slot | Item | Score | Runner-up (score) |", "|---|---|---:|---|---|---:|---|"]
    lines += [f"| {r[0]} | {r[1] or '—'} | {r[2]} | {r[3]} | {r[4]} | {float(r[5]):.4f} | {r[6]} ({float(r[7]):.4f}) |"
              for r in rows]
    lines.append("")
    record = {"case_set_id": CASE_SET_ID, "query_text": text, "device": args.device, "model": label,
              "checkpoint_sha256": sha256(checkpoint), "fashionclip": "patrickjohncyh/fashion-clip@7e3ba62ce16b379a1ab479346b66f192e76f51b7",
              "bank_sizes": bank_sizes, "gt_by_slot": gt, "schema": schema,
              "results": {k: v for k, v in results.items()}}
    if args.validate_2025:
        same, diffs = 0, []
        for keep, printed in P02_PRINTED.items():
            for (slot, item, score), s in zip(printed, results[keep]["steps"]):
                same += s["slot"] == slot and s["item_id"] == item
                diffs.append(abs(round(s["score"], 4) - score))
        total = sum(len(v) for v in P02_PRINTED.values())
        lines += [f"Against P02's printed picks (2025 model on CUDA): {same} of {total} picks equal; largest "
                  f"difference of the four-decimal scores {max(diffs):.4f}.", ""]
        record["p02_picks_equal"] = f"{same}/{total}"
    figures = Path(args.figures_dir) if args.figures_dir else (
        REPO / "_external" / "figures_official" if run_root and run_root.name == OFFICIAL_RUN else None)
    if figures:
        fig_dir = safe_out_dir(figures)
        name = f"figure_P02_outfit_generation_{CASE_SET_ID}_{'2025' if args.validate_2025 else 'seed1'}.png"
        grid_rows = [("Original", gt, set(), set())]
        for keep in ("keep2", "keep1", "keep0"):
            refs = {s: gt[s] for s in results[keep]["kept"]}
            refs.update({s["slot"]: s["item_id"] for s in results[keep]["steps"]})
            grid_rows.append((keep.capitalize(), refs, set(results[keep]["kept"]),
                              {s["slot"] for s in results[keep]["steps"]}))
        item_grid(fig_dir / name, f"Case {CASE_SET_ID} | Original / Keep2 / Keep1 / Keep0", text, grid_rows, images)
        lines += [f"Item grid (Polyvore photos, not committed): `{name}` in the figures folder.", ""]
        record["figure"] = name
    lines += ["The try-on images of P02 (FastFit on the person photo model_image.png) are not reproduced: the person "
              "photo was not preserved and FastFit is not part of the repository.", ""]
    write_text(out / "summary.md", lines)
    write_json(out / "manifest.json", record)
    print("\n".join(lines))


if __name__ == "__main__":
    main()
