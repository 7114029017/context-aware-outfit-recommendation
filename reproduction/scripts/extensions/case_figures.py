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

The script also draws the color-shift case of thesis Figure 4-14 (set
174710752, query color gray), as the same notebook's cell 7 (show_case) drew
it: the query outfit, then the top-5 items under Original and under Full, each
labeled with its dominant image color; green boxes the query color, red the
case's most frequent wrong color, gold the target. The seed shown is the
representative row of the notebook (the largest color problem under Original,
then the best Original rank); the colors come from color_analysis.py.

The figures contain Polyvore product photos: they are written only outside the
repository or into a Git-ignored folder (reproduction/runs/, _external/), never
committed or redistributed. The panel contents (item IDs, ranks and colors) are
also written to case_figures_manifest.csv and color_case_figure_manifest.csv,
which contain no images.
"""
from __future__ import annotations

import argparse
import csv
import json
import textwrap
from pathlib import Path
from statistics import mean

from PIL import Image, ImageDraw, ImageFont

import color_analysis as color
from _ext import OFFICIAL_RUN, REPO, REPRO, SEEDS, WOS_JSONL, local_path, passed_run, write_csv

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
COLOR_CASE = ("4-14", "gray", "174710752")  # thesis Figure 4-14
TOPK = 5
TILE = 150
GAP = 8
HEADER = 30
GREEN, BLACK, GREY = (39, 174, 96), (17, 17, 17), (204, 204, 204)
QUERY_COLOR, WRONG_COLOR, TARGET = (46, 139, 87), (192, 57, 43), (212, 172, 13)  # P03 show_case borders


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


def color_case_rows(case_rows: dict, set_id: str, original_text: str, full_text: str) -> dict:
    """P03 cell 7 for one case: per-seed color analysis, representative seed and wrong-color modes."""
    colors_o, colors_f = color.extract_query_colors(original_text), color.extract_query_colors(full_text)
    if len(colors_o) != 1 or colors_o != colors_f:
        raise SystemExit(f"[FIGURES BLOCKED] set {set_id}: query colors {colors_o} / {colors_f} are not one shared color")
    per_seed = []
    for seed in SEEDS:
        o, f = case_rows[(set_id, "original", seed)], case_rows[(set_id, "context", seed)]
        row = {"query_color_original": colors_o[0], "query_color_full": colors_o[0], "target_item_id": o["target_item_id"],
               "top10_ids_list_original": color.parse_top_ids(o["top10_ids"]),
               "top10_ids_list_full": color.parse_top_ids(f["top10_ids"])}
        result = {"seed": seed, "rank_original": int(o["rank"]), "rank_full": int(f["rank"]), **row}
        for variant in ("original", "full"):
            result.update(color.analyze_variant_row(row, variant).to_dict())
        per_seed.append(result)
    # P03: sort by alt_advantage_original (descending) and rank_original; the first row of the case is shown
    rep = sorted(per_seed, key=lambda r: (-r["alt_advantage_original"], r["rank_original"]))[0]
    return {"query_color": colors_o[0], "rows": per_seed, "rep": rep,
            "wrong_original": color.safe_mode_non_none([r["wrong_color_original"] for r in per_seed]),
            "wrong_full": color.safe_mode_non_none([r["wrong_color_full"] for r in per_seed])}


def draw_color_case(fig_id: str, set_id: str, target: str, category: str, outfit: list[str], case: dict,
                    original_text: str, full_text: str, image_root: Path) -> Image.Image:
    rep = case["rep"]
    cols = max(TOPK, len(outfit), 3)
    label_h = 18
    title = (f"Figure {fig_id} (color shift)  |  set {set_id}  |  target {target}  |  {category}  |  "
             f"query color {case['query_color']}  |  seed {rep['seed']}")
    width = max(GAP + cols * (TILE + GAP), int(font(14, True).getlength(title)) + 2 * GAP)
    wrapped = [textwrap.wrap(f"{name} query: {text}", width=max(60, width // 8)) for name, text in
               (("Original", original_text), ("Full", full_text))]
    text_h = 18 * sum(len(w) for w in wrapped) + 8
    height = HEADER + text_h + 3 * (HEADER + TILE + label_h + GAP) + 12
    fig = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(fig)
    draw.text((GAP, 6), title, fill=BLACK, font=font(14, True))
    y = HEADER
    for lines in wrapped:
        for line in lines:
            draw.text((GAP, y), line, fill=(51, 51, 51), font=font(12))
            y += 18
    y += 8

    def band(title: str, color_rgb: tuple, items: list, labels: list, borders: list) -> None:
        nonlocal y
        draw.text((GAP, y + 6), title, fill=color_rgb, font=font(13, True))
        y += HEADER
        for i, (item, label, border) in enumerate(zip(items, labels, borders)):
            x = GAP + i * (TILE + GAP)
            fig.paste(tile(image_root, item, border, 1 if border == GREY else 5), (x, y))
            draw.text((x + 4, y + TILE + 2), label, fill=(68, 68, 68), font=font(11))
        y += TILE + label_h + GAP

    band("Full outfit (gold: target slot)", (51, 51, 51), outfit,
         ["TARGET SLOT" if item == target else f"Item {i + 1}" for i, item in enumerate(outfit)],
         [TARGET if item == target else GREY for item in outfit])
    for variant, title, rgb, rank in (("original", "Original retrieval", (68, 68, 68), rep["rank_original"]),
                                      ("full", "Full retrieval", (33, 102, 172), rep["rank_full"])):
        items = [str(i) for i in rep[f"top10_ids_list_{variant}"][:TOPK]]
        colors = list(rep[f"topk_colors_{variant}"])[:TOPK]
        wrong = case[f"wrong_{variant}"]
        borders = [TARGET if item == target else QUERY_COLOR if c == case["query_color"] else
                   WRONG_COLOR if c == wrong else GREY for item, c in zip(items, colors)]
        band(f"{title}   (correct answer rank #{rank}; green: {case['query_color']}, red: {wrong})", rgb, items,
             [f"#{i + 1}  {c}" for i, c in enumerate(colors)], borders)
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
                    if r["set_id"] in {c[2] for c in CASES} | {COLOR_CASE[2]}:
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

    # thesis Figure 4-14: the color-shift case
    fig_id, name, set_id = COLOR_CASE
    color.IMAGE_ROOTS.append(str(polyvore / "images"))
    titles = json.loads((polyvore / "polyvore_outfit_titles.json").read_text(encoding="utf-8"))
    value = titles.get(set_id, "")
    original_text = (" ".join(p for p in (color.normalize_text(value.get("url_name", value.get("url", ""))),
                                         color.normalize_text(value.get("title", ""))) if p).strip()
                     if isinstance(value, dict) else color.normalize_text(value))
    full_text = ""
    for line in WOS_JSONL.read_text(encoding="utf-8").splitlines():
        record = json.loads(line) if line.strip() else {}
        if record.get("valid") is True and str(record.get("id", "")) == set_id:
            full_text = color.normalize_text(record.get("title", ""))
    case = color_case_rows(case_rows, set_id, original_text, full_text)
    rep = case["rep"]
    target = rep["target_item_id"]
    target_fg = case_rows[(set_id, "context", rep["seed"])]["target_item_fg"]
    category = labels.get(target_fg, target_fg)
    image = draw_color_case(fig_id, set_id, target, category, test.get(set_id, [target]), case, original_text,
                            full_text, polyvore / "images")
    image.save(out / f"figure_{fig_id.replace('-', '_')}_{name}_set{set_id}_seed{rep['seed']}.png")
    color_manifest = []
    for r in case["rows"]:
        for variant, label in (("original", "Original"), ("full", "Full")):
            color_manifest.append([fig_id, set_id, target, category, case["query_color"], r["seed"],
                                   "yes" if r is rep else "", label, r[f"rank_{variant}"],
                                   f"{mean(x[f'rank_{variant}'] for x in case['rows']):.1f}", r[f"issue_{variant}"],
                                   r[f"alt_advantage_{variant}"], case[f"wrong_{variant}"],
                                   " | ".join(str(i) for i in r[f"top10_ids_list_{variant}"][:TOPK]),
                                   " | ".join(r[f"topk_colors_{variant}"][:TOPK])])
    write_csv((args.manifest_dir or out) / "color_case_figure_manifest.csv",
              ["figure", "set_id", "target_item_id", "category", "query_color", "seed", "shown_seed", "condition",
               "rank", "five_seed_mean_rank", "color_problem", "alt_advantage", "case_wrong_color_mode",
               "top5_item_ids", "top5_dominant_colors"], color_manifest)
    print(f"[FIGURES] {fig_id} {name}: seed {rep['seed']}; rank Original {rep['rank_original']}, Full "
          f"{rep['rank_full']}; top-5 colors Original {', '.join(rep['topk_colors_original'][:TOPK])}; "
          f"Full {', '.join(rep['topk_colors_full'][:TOPK])}")
    print(f"[FIGURES] run {run_id}: figures in {out} (local only; they contain Polyvore photos)")


if __name__ == "__main__":
    main()
