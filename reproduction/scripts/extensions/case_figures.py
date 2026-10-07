#!/usr/bin/env python3
"""Case figures of the manuscript appendix (Figures A1-A3) drawn from a run's fair-subset results.

The figures show, for one (set, target) case, the target item and the top-5
items retrieved under Original, No-Weather, No-Occasion, No-Style and Full,
with the target's rank and a green box on a hit. As in the 2025 notebook
03_實驗與結果_experiments_results/04_情境子集與三因子分析/source_programs/P03_qualitative_case_analysis.ipynb
(cell 4, show_full_ablation_case), the seed shown is the one whose Full rank
is closest to the five-seed mean Full rank (the lowest seed on ties). The
notebook drew with matplotlib; this script draws the same panels with Pillow,
and adds a top row with the query outfit (the target boxed), which Figures A2
and A3 have but whose drawing code was not preserved.

The figures contain Polyvore product photos: they are written only outside the
repository or into a Git-ignored folder (reproduction/runs/, _external/), never
committed or redistributed. The panel contents (item IDs and ranks) are also
written to case_figures_manifest.csv, which contains no images.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean

from PIL import Image, ImageDraw, ImageFont

from _ext import OFFICIAL_RUN, REPO, REPRO, SEEDS, local_path, passed_run, write_csv

CASES = (  # manuscript Figures A1-A3
    ("A1", "purse", "224499261"),
    ("A2", "dress", "94771580"),
    ("A3", "sunglasses", "200099867"),
)
CONDITIONS = (  # run directory prefix, label, header color
    ("original", "Original (Baseline)", (85, 85, 85)),
    ("no_weather", "No-Weather (ablation)", (192, 57, 43)),
    ("no_occasion", "No-Occasion (ablation)", (230, 126, 34)),
    ("no_style", "No-Style (ablation)", (142, 68, 173)),
    ("context", "Full (Proposed)", (33, 102, 172)),
)
TOPK = 5
TILE = 150
GAP = 8
HEADER = 30
GREEN, BLACK, GREY = (39, 174, 96), (17, 17, 17), (204, 204, 204)


def font(size: int, bold: bool = False):
    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    for folder in ("/usr/share/fonts/truetype/dejavu", "/usr/share/fonts/dejavu", "/Library/Fonts"):
        path = Path(folder) / name
        if path.is_file():
            return ImageFont.truetype(str(path), size)
    return ImageFont.load_default()


def safe_out_dir(path: Path) -> Path:
    out = path.expanduser().resolve()
    allowed = (REPO / "reproduction" / "runs", REPO / "_external")
    if out.is_relative_to(REPO) and not any(out.is_relative_to(a) for a in allowed):
        raise SystemExit(f"[FIGURES BLOCKED] {out} is inside the repository; the figures contain Polyvore photos. "
                         "Use a folder outside the repository, under reproduction/runs/ or under _external/.")
    out.mkdir(parents=True, exist_ok=True)
    return out


def tile(image_root: Path, item_id: str, border: tuple, width: int) -> Image.Image:
    canvas = Image.new("RGB", (TILE, TILE), "white")
    path = image_root / f"{item_id}.jpg"
    if path.is_file():
        img = Image.open(path).convert("RGB")
        img.thumbnail((TILE - 12, TILE - 12), Image.Resampling.LANCZOS)
        canvas.paste(img, ((TILE - img.width) // 2, (TILE - img.height) // 2))
    else:
        ImageDraw.Draw(canvas).text((TILE // 2 - 12, TILE // 2 - 6), "N/A", fill=(153, 153, 153), font=font(12))
    ImageDraw.Draw(canvas).rectangle([0, 0, TILE - 1, TILE - 1], outline=border, width=width)
    return canvas


def draw_case(fig_id: str, name: str, set_id: str, target: str, category: str, outfit: list[str], seed: int,
              rows: dict, means: dict, image_root: Path) -> Image.Image:
    cols = max(TOPK, len(outfit))
    width = GAP + cols * (TILE + GAP)
    bands = 2 + 1 + len(CONDITIONS)  # title, outfit, target, results
    height = HEADER + bands * (HEADER + TILE + GAP) - TILE + 20
    fig = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(fig)
    y = 6
    draw.text((GAP, y), f"Figure {fig_id} ({name})  |  set {set_id}  |  {category}  |  seed {seed}", fill=BLACK,
              font=font(15, True))
    y += HEADER

    def header(text: str, color: tuple) -> None:
        nonlocal y
        draw.text((GAP, y + 6), text, fill=color, font=font(13, True))
        y += HEADER

    header("Query outfit (target boxed)", (51, 51, 51))
    for i, item in enumerate(outfit):
        fig.paste(tile(image_root, item, BLACK if item == target else GREY, 4 if item == target else 1),
                  (GAP + i * (TILE + GAP), y))
    y += TILE + GAP
    header(f"Query target  —  {category}", (51, 51, 51))
    fig.paste(tile(image_root, target, BLACK, 4), (GAP + (TOPK // 2) * (TILE + GAP), y))
    y += TILE + GAP
    for prefix, label, color in CONDITIONS:
        row = rows[prefix]
        header(f"{label}   (rank = {row['rank']}; five-seed mean {means[prefix]:.1f})", color)
        for i, item in enumerate(json.loads(row["top10_ids"])[:TOPK]):
            hit = str(item) == target
            t = tile(image_root, str(item), GREEN if hit else GREY, 6 if hit else 1)
            d = ImageDraw.Draw(t)
            d.rectangle([4, 4, 30, 20], fill="white")
            d.text((6, 5), f"#{i + 1}", fill=(68, 68, 68), font=font(11))
            if hit:
                d.text((TILE - 46, TILE - 22), "✓ Hit", fill=GREEN, font=font(12, True))
            fig.paste(t, (GAP + i * (TILE + GAP), y))
        y += TILE + GAP
    return fig


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--run-root", type=Path, help="completed full run folder")
    source.add_argument("--official", action="store_true", help="the official run's committed per-query files")
    parser.add_argument("--out-dir", type=Path, required=True, help="for the PNG files; never a tracked folder")
    parser.add_argument("--manifest-dir", type=Path, default=None,
                        help="folder for case_figures_manifest.csv (no images); default: --out-dir")
    parser.add_argument("--polyvore-root", default=None)
    args = parser.parse_args()

    if args.official:
        run_id, detail_root = OFFICIAL_RUN, REPRO / "results" / "raw" / OFFICIAL_RUN / "ablation"
    else:
        run_root = passed_run(args.run_root)
        run_id, detail_root = run_root.name, run_root / "ablation" / "runs"
    out = safe_out_dir(args.out_dir)
    polyvore = local_path("polyvore_root", args.polyvore_root)
    if polyvore is None or not (polyvore / "images").is_dir():
        raise SystemExit("[FIGURES BLOCKED] Polyvore images not found; run bootstrap_data.sh --with-images")
    test = {str(o["set_id"]): [str(i["item_id"]) for i in o["items"]]
            for o in json.loads((polyvore / "disjoint" / "test.json").read_text(encoding="utf-8"))}
    labels = {}
    with (polyvore / "categories.csv").open(encoding="utf-8-sig", newline="") as f:
        for r in csv.reader(f):
            if len(r) >= 3:
                labels.setdefault(r[0].strip(), f"{r[1].strip()} ({r[2].strip()})")  # first occurrence, as in P03

    case_rows = {}
    for prefix, _, _ in CONDITIONS:
        for seed in SEEDS:
            with (detail_root / f"{prefix}_seed{seed}" / "detail_cir_fresh_subset.csv").open(encoding="utf-8") as f:
                for r in csv.DictReader(f):
                    if r["set_id"] in {c[2] for c in CASES}:
                        case_rows[(r["set_id"], prefix, seed)] = r

    manifest = []
    for fig_id, name, set_id in CASES:
        full = {s: int(case_rows[(set_id, "context", s)]["rank"]) for s in SEEDS}
        mean_full = mean(full.values())
        seed = min(SEEDS, key=lambda s: (abs(full[s] - mean_full), s))
        rows = {p: case_rows[(set_id, p, seed)] for p, _, _ in CONDITIONS}
        means = {p: mean(int(case_rows[(set_id, p, s)]["rank"]) for s in SEEDS) for p, _, _ in CONDITIONS}
        target = rows["context"]["target_item_id"]
        category = labels.get(rows["context"]["target_item_fg"], rows["context"]["target_item_fg"])
        image = draw_case(fig_id, name, set_id, target, category, test.get(set_id, [target]), seed, rows, means,
                          polyvore / "images")
        image.save(out / f"figure_{fig_id}_{name}_set{set_id}_seed{seed}.png")
        for p, label, _ in CONDITIONS:
            manifest.append([fig_id, name, set_id, target, seed, label, rows[p]["rank"], f"{means[p]:.1f}",
                             " | ".join(json.loads(rows[p]["top10_ids"])[:TOPK])])
        print(f"[FIGURES] {fig_id} {name}: seed {seed}; ranks " +
              ", ".join(f"{label.split(' ')[0]} {rows[p]['rank']}" for p, label, _ in CONDITIONS))
    write_csv((args.manifest_dir or out) / "case_figures_manifest.csv",
              ["figure", "case", "set_id", "target_item_id", "seed", "condition", "rank", "five_seed_mean_rank",
               "top5_item_ids"], manifest)
    print(f"[FIGURES] run {run_id}: figures in {out} (local only; they contain Polyvore photos)")


if __name__ == "__main__":
    main()
