#!/usr/bin/env python3
"""Color-shift analysis of the fair-subset CIR results (manuscript Section 5.5: "39 of 89 row-level
problems improve, 5 of 15 case-level problems"; thesis Table 4-19).

The computation is copied from the 2025 notebook
03_實驗與結果_experiments_results/04_情境子集與三因子分析/source_programs/P03_qualitative_case_analysis.ipynb
(cell 7). It is automatic, not a human judgment: the color words of the
Original and the Full query text are found by regular expressions, the dominant
colors of the target and of the top-5 retrieved items come from the pixels of
the Polyvore item images (HSV thresholds on 120-pixel thumbnails), and a row
has a color problem when, among the top 5, the most frequent other color
outnumbers the query color by at least two while the target itself has the
query color. A problem is solved when it disappears under Full. Cases are
(set, target) pairs over the five seeds.

Inputs: the Original and Full fair-subset per-query files of a run
(ablation/runs/<variant>_seed<k>/detail_cir_fresh_subset.csv, or the official
run's committed copies with --official), the W/O/S fragments file, and the
Polyvore item images (bootstrap_data.sh --with-images). CPU only. The images
are only read; nothing derived from them except the counts is written.
"""
from __future__ import annotations

import argparse
import ast
import json
import re
from collections import Counter
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from _ext import OFFICIAL_RUN, REPRO, SEEDS, WOS_JSONL, local_path, output_dir, passed_run, write_text

# ---------------------------------------------------------------- P03 cell 7, settings
TOPK = 5
COMPARE_SAME_SINGLE_COLOR_ONLY = True
REQUIRE_TARGET_MATCH_QUERY = True
MIN_ALT_ADVANTAGE = 2
VARIANTS = {"original": "original", "full": "context"}  # P03 name -> run directory prefix
COLOR_PATTERNS = {
    "black": [r"\bblack\b", r"\bjet black\b", r"\bcharcoal\b"],
    "white": [r"\bwhite\b", r"\bivory\b", r"\bcream\b", r"\boff[- ]white\b"],
    "gray": [r"\bgray\b", r"\bgrey\b", r"\bsilver\b", r"\bslate\b"],
    "brown": [r"\bbrown\b", r"\bbeige\b", r"\btan\b", r"\bkhaki\b", r"\bcamel\b", r"\bnude\b", r"\bmocha\b",
              r"\bcognac\b"],
    "red": [r"\bred\b", r"\bcrimson\b", r"\bburgundy\b", r"\bmaroon\b", r"\bwine\b"],
    "orange": [r"\borange\b", r"\bcoral\b", r"\bpeach\b", r"\brust\b"],
    "yellow": [r"\byellow\b", r"\bgold\b", r"\bmustard\b"],
    "green": [r"\bgreen\b", r"\bolive\b", r"\bemerald\b", r"\bmint\b", r"\blime\b"],
    "blue": [r"\bblue\b", r"\bnavy\b", r"\bnavy blue\b", r"\bsky blue\b", r"\blight blue\b", r"\bbaby blue\b",
             r"\bcobalt\b", r"\broyal blue\b", r"\bteal\b", r"\bturquoise\b", r"\bcyan\b"],
    "purple": [r"\bpurple\b", r"\blavender\b", r"\bviolet\b", r"\blilac\b"],
    "pink": [r"\bpink\b", r"\brose\b", r"\bfuchsia\b", r"\bmagenta\b", r"\bblush\b", r"\bhot pink\b"],
}
CANONICAL_COLOR_ORDER = ["black", "white", "gray", "brown", "red", "orange", "yellow", "green", "blue", "purple", "pink"]
IMAGE_ROOTS: list[str] = []


# ---------------------------------------------------------------- P03 cell 7, utilities
def normalize_text(x):
    if x is None:
        return ""
    if isinstance(x, str):
        return x.strip()
    if isinstance(x, list):
        return " | ".join(str(i).strip() for i in x if str(i).strip())
    return str(x).strip()


def parse_top_ids(x):
    if pd.isna(x):
        return []
    if isinstance(x, list):
        return [str(i) for i in x]
    try:
        v = json.loads(x)
        if isinstance(v, list):
            return [str(i) for i in v]
    except Exception:
        pass
    try:
        v = ast.literal_eval(x)
        if isinstance(v, list):
            return [str(i) for i in v]
    except Exception:
        pass
    return []


def ensure_list(x):
    if isinstance(x, list):
        return x
    return []


def extract_query_colors(text):
    text = normalize_text(text).lower()
    found = []
    for canonical in CANONICAL_COLOR_ORDER:
        for pat in COLOR_PATTERNS[canonical]:
            if re.search(pat, text):
                found.append(canonical)
                break
    return list(dict.fromkeys(found))


def find_image_by_id(img_id):
    img_id = str(img_id)
    for root in IMAGE_ROOTS:
        for ext in (".jpg", ".jpeg", ".png", ".webp"):
            p = Path(root) / f"{img_id}{ext}"
            if p.exists():
                return str(p)
    return None


def short_profile_text(profile, topn=2):
    if not profile:
        return "unknown"
    return ", ".join(f"{c}({p:.2f})" for c, p in profile[:topn])


def safe_mode_non_none(series, default="none"):
    vals = [x for x in series if pd.notna(x) and x != "none"]
    if not vals:
        return default
    return Counter(vals).most_common(1)[0][0]


@lru_cache(maxsize=None)
def get_image_color_profile(img_id):
    fp = find_image_by_id(img_id)
    if fp is None:
        return {"dominant": "unknown", "profile": [], "path": None}
    try:
        img_rgba = Image.open(fp).convert("RGBA")
    except Exception:
        return {"dominant": "unknown", "profile": [], "path": fp}
    try:
        resample = Image.Resampling.LANCZOS
    except Exception:
        resample = Image.LANCZOS
    img_rgba.thumbnail((120, 120), resample)
    arr = np.array(img_rgba)
    rgb = arr[..., :3]
    alpha = arr[..., 3].astype(np.float32) / 255.0
    hsv = np.array(Image.fromarray(rgb).convert("HSV")).astype(np.float32)
    h = hsv[..., 0] * (360.0 / 255.0)
    s = hsv[..., 1] / 255.0
    v = hsv[..., 2] / 255.0
    mask = alpha > 0.20
    mask &= ~((v > 0.97) & (s < 0.08))
    if mask.sum() < 50:
        mask = alpha > 0.20
    h = h[mask]
    s = s[mask]
    v = v[mask]
    if len(h) == 0:
        return {"dominant": "unknown", "profile": [], "path": fp}
    labels = np.array(["unknown"] * len(h), dtype=object)
    black_mask = v < 0.18
    white_mask = (v > 0.88) & (s < 0.12)
    gray_mask = (s < 0.18) & (~black_mask) & (~white_mask)
    labels[black_mask] = "black"
    labels[white_mask] = "white"
    labels[gray_mask] = "gray"
    rest = labels == "unknown"
    brown_mask = rest & (h >= 10) & (h < 40) & (s >= 0.20) & (v < 0.65)
    labels[brown_mask] = "brown"
    rest = labels == "unknown"
    labels[rest & ((h < 15) | (h >= 345))] = "red"
    rest = labels == "unknown"
    labels[rest & (h >= 15) & (h < 40)] = "orange"
    rest = labels == "unknown"
    labels[rest & (h >= 40) & (h < 65)] = "yellow"
    rest = labels == "unknown"
    labels[rest & (h >= 65) & (h < 170)] = "green"
    rest = labels == "unknown"
    labels[rest & (h >= 170) & (h < 255)] = "blue"
    rest = labels == "unknown"
    labels[rest & (h >= 255) & (h < 290)] = "purple"
    rest = labels == "unknown"
    labels[rest & (h >= 290) & (h < 345)] = "pink"
    cnt = Counter(labels.tolist())
    cnt.pop("unknown", None)
    if len(cnt) == 0:
        return {"dominant": "unknown", "profile": [], "path": fp}
    total = sum(cnt.values())
    profile = sorted([(k, v / total) for k, v in cnt.items()], key=lambda x: x[1], reverse=True)
    return {"dominant": profile[0][0], "profile": profile, "path": fp}


def load_one_variant_detail(detail_root: Path, prefix: str, short_name: str) -> pd.DataFrame:
    dfs = []
    for seed in SEEDS:
        df = pd.read_csv(detail_root / f"{prefix}_seed{seed}" / "detail_cir_fresh_subset.csv")
        df["variant"] = short_name
        dfs.append(df)
    out = pd.concat(dfs, ignore_index=True)
    out["seed"] = out["seed"].astype(int)
    out["set_id"] = out["set_id"].astype(str)
    out["target_item_id"] = out["target_item_id"].astype(str)
    out["target_item_fg"] = out["target_item_fg"].astype(str)
    out["rank"] = pd.to_numeric(out["rank"], errors="coerce")
    out["top10_ids_list"] = out["top10_ids"].apply(parse_top_ids)
    out["query_key"] = out["seed"].astype(str) + "||" + out["set_id"] + "||" + out["target_item_id"]
    return out


def load_original_titles_json(path: Path) -> pd.DataFrame:
    data = json.loads(path.read_text(encoding="utf-8"))
    rows = []
    for sid, value in data.items():
        if isinstance(value, str):
            original_text = value.strip()
        elif isinstance(value, dict):
            title_part = normalize_text(value.get("title", ""))
            url_part = normalize_text(value.get("url_name", value.get("url", "")))
            original_text = " ".join(p for p in [url_part, title_part] if p).strip()
        else:
            original_text = normalize_text(value)
        rows.append({"set_id": str(sid), "original_text": original_text})
    return pd.DataFrame(rows)


def load_fragments_jsonl(path: Path) -> pd.DataFrame:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        obj = json.loads(line)
        if obj.get("valid") is not True:
            continue
        title_ablation = obj.get("title_ablation", {})
        rows.append({
            "set_id": str(obj.get("id", "")),
            "title_full_text": normalize_text(obj.get("title", "")),
            "title_no_weather_text": normalize_text(title_ablation.get("no_weather", "")),
            "title_no_occasion_text": normalize_text(title_ablation.get("no_occasion", "")),
            "title_no_style_text": normalize_text(title_ablation.get("no_style", "")),
        })
    return pd.DataFrame(rows)


def analyze_variant_row(row, variant_name):
    query_color = row[f"query_color_{variant_name}"]
    target_id = str(row["target_item_id"])
    top_ids = ensure_list(row[f"top10_ids_list_{variant_name}"])[:TOPK]
    target_info = get_image_color_profile(target_id)
    target_top2 = [c for c, _ in target_info["profile"][:2]]
    target_match = int(query_color in target_top2)
    top_infos = [get_image_color_profile(str(i)) for i in top_ids]
    top_colors = [x["dominant"] for x in top_infos]
    cnt = Counter([c for c in top_colors if c != "unknown"])
    query_count = cnt.get(query_color, 0)
    non_query = {k: v for k, v in cnt.items() if k != query_color}
    if non_query:
        wrong_color, wrong_count = sorted(non_query.items(), key=lambda x: (-x[1], x[0]))[0]
    else:
        wrong_color, wrong_count = "none", 0
    alt_advantage = wrong_count - query_count
    issue = int(alt_advantage >= MIN_ALT_ADVANTAGE)
    if REQUIRE_TARGET_MATCH_QUERY:
        issue = int(issue and (target_match == 1))
    return pd.Series({
        f"target_match_{variant_name}": target_match,
        f"target_profile_text_{variant_name}": short_profile_text(target_info["profile"], topn=2),
        f"query_count_{variant_name}": query_count,
        f"wrong_color_{variant_name}": wrong_color,
        f"wrong_count_{variant_name}": wrong_count,
        f"alt_advantage_{variant_name}": alt_advantage,
        f"topk_colors_{variant_name}": top_colors,
        f"topk_color_count_text_{variant_name}": ", ".join([f"{k}:{v}" for k, v in cnt.most_common()]) if cnt else "none",
        f"issue_{variant_name}": issue,
    })


def analyse(detail_root: Path, polyvore: Path):
    detail_orig = load_one_variant_detail(detail_root, VARIANTS["original"], "original")
    detail_full = load_one_variant_detail(detail_root, VARIANTS["full"], "full")
    orig_title_df = load_original_titles_json(polyvore / "polyvore_outfit_titles.json")
    frag_df = load_fragments_jsonl(WOS_JSONL)
    base = (
        detail_orig[["query_key", "seed", "set_id", "target_item_id", "target_item_fg", "rank", "top10_ids_list"]]
        .rename(columns={"rank": "rank_original", "top10_ids_list": "top10_ids_list_original"})
        .merge(detail_full[["query_key", "rank", "top10_ids_list"]]
               .rename(columns={"rank": "rank_full", "top10_ids_list": "top10_ids_list_full"}),
               on="query_key", how="inner")
        .merge(orig_title_df, on="set_id", how="left")
        .merge(frag_df, on="set_id", how="left")
    )
    base["query_text_original"] = base["original_text"].fillna("").astype(str)
    base["query_text_full"] = base["title_full_text"].fillna("").astype(str)
    base["query_colors_original"] = base["query_text_original"].apply(extract_query_colors)
    base["query_colors_full"] = base["query_text_full"].apply(extract_query_colors)
    base = base[
        (base["query_colors_original"].apply(lambda x: len(x) == 1)) &
        (base["query_colors_full"].apply(lambda x: len(x) == 1)) &
        (base["query_colors_original"].str[0] == base["query_colors_full"].str[0])
    ].copy()
    base["query_color"] = base["query_colors_original"].str[0]
    base["query_color_original"] = base["query_color"]
    base["query_color_full"] = base["query_color"]
    needed_ids = set(base["target_item_id"].astype(str).tolist())
    for col in ["top10_ids_list_original", "top10_ids_list_full"]:
        for ids in base[col]:
            needed_ids.update([str(x) for x in ensure_list(ids)[:TOPK]])
    missing = sorted(i for i in needed_ids if find_image_by_id(i) is None)
    if missing:
        raise SystemExit(f"[COLOR BLOCKED] {len(missing)} of {len(needed_ids)} item images are missing, "
                         f"e.g. {missing[:3]}; run bootstrap_data.sh --with-images")
    base = base.reset_index(drop=True)
    ana_orig = base.apply(analyze_variant_row, axis=1, variant_name="original").reset_index(drop=True)
    ana_full = base.apply(analyze_variant_row, axis=1, variant_name="full").reset_index(drop=True)
    row_df = pd.concat([base, ana_orig, ana_full], axis=1)
    row_df["solved_row"] = ((row_df["issue_original"] == 1) & (row_df["issue_full"] == 0)).astype(int)
    row_df["unsolved_row"] = ((row_df["issue_original"] == 1) & (row_df["issue_full"] == 1)).astype(int)
    row_df["case_key"] = row_df["set_id"].astype(str) + "||" + row_df["target_item_id"].astype(str)
    case_df = row_df.groupby("case_key").agg(
        set_id=("set_id", "first"),
        target_item_id=("target_item_id", "first"),
        target_item_fg=("target_item_fg", "first"),
        query_color=("query_color", "first"),
        n_seeds=("seed", "size"),
        mean_rank_original=("rank_original", "mean"),
        mean_rank_full=("rank_full", "mean"),
        orig_issue_rate=("issue_original", "mean"),
        full_issue_rate=("issue_full", "mean"),
        orig_mean_alt_advantage=("alt_advantage_original", "mean"),
        full_mean_alt_advantage=("alt_advantage_full", "mean"),
        orig_target_match_rate=("target_match_original", "mean"),
        full_target_match_rate=("target_match_full", "mean"),
        orig_mean_query_count=("query_count_original", "mean"),
        full_mean_query_count=("query_count_full", "mean"),
        solved_seed_rate=("solved_row", "mean"),
        unsolved_seed_rate=("unsolved_row", "mean"),
        wrong_color_mode_original=("wrong_color_original", lambda s: safe_mode_non_none(s, default="none")),
        wrong_color_mode_full=("wrong_color_full", lambda s: safe_mode_non_none(s, default="none")),
    ).reset_index()
    if REQUIRE_TARGET_MATCH_QUERY:
        case_df = case_df[(case_df["orig_target_match_rate"] >= 0.5) & (case_df["full_target_match_rate"] >= 0.5)].copy()
    orig_problem_cases = case_df[case_df["orig_issue_rate"] >= 0.5].copy()
    solved_cases = orig_problem_cases[orig_problem_cases["full_issue_rate"] < 0.5]
    orig_problem_rows = row_df[row_df["issue_original"] == 1]
    solved_rows = orig_problem_rows[orig_problem_rows["issue_full"] == 0]
    overall = pd.DataFrame([
        {"level": "case", "denominator": len(orig_problem_cases), "solved_n": len(solved_cases),
         "solved_ratio": len(solved_cases) / len(orig_problem_cases) if len(orig_problem_cases) else 0.0},
        {"level": "row", "denominator": len(orig_problem_rows), "solved_n": len(solved_rows),
         "solved_ratio": len(solved_rows) / len(orig_problem_rows) if len(orig_problem_rows) else 0.0},
    ])
    return overall, row_df, case_df, len(needed_ids)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--run-root", type=Path, help="completed full run folder")
    source.add_argument("--official", action="store_true", help="the official run's committed per-query files")
    parser.add_argument("--out-dir", type=Path, default=None, help="default: <run>/extensions/color_analysis")
    parser.add_argument("--polyvore-root", default=None)
    args = parser.parse_args()

    if args.official:
        run_root, run_id = None, OFFICIAL_RUN
        detail_root = REPRO / "results" / "raw" / OFFICIAL_RUN / "ablation"
        if args.out_dir is None:
            raise SystemExit("[COLOR BLOCKED] pass --out-dir with --official")
    else:
        run_root = passed_run(args.run_root)
        run_id, detail_root = run_root.name, run_root / "ablation" / "runs"
    out = output_dir(args.out_dir or run_root / "extensions" / "color_analysis", run_root)
    polyvore = local_path("polyvore_root", args.polyvore_root)
    if polyvore is None or not (polyvore / "images").is_dir():
        raise SystemExit("[COLOR BLOCKED] Polyvore images not found; run bootstrap_data.sh --with-images")
    IMAGE_ROOTS.append(str(polyvore / "images"))

    overall, row_df, case_df, n_images = analyse(detail_root, polyvore)
    overall.to_csv(out / "color_shift_summary.csv", index=False, lineterminator="\n")
    keep = ["seed", "set_id", "target_item_id", "target_item_fg", "query_color", "rank_original", "rank_full",
            "target_profile_text_original", "topk_color_count_text_original", "topk_color_count_text_full",
            "alt_advantage_original", "alt_advantage_full", "issue_original", "issue_full", "solved_row"]
    row_df[keep].to_csv(out / "color_shift_rows.csv", index=False, lineterminator="\n")
    case_df.drop(columns=["case_key"]).to_csv(out / "color_shift_cases.csv", index=False, lineterminator="\n")
    case, row = overall.iloc[0], overall.iloc[1]
    lines = [
        f"# Color-shift analysis of run `{run_id}`",
        "",
        "Copied from notebook P03 (cell 7). Automatic: color words in the query texts and the dominant colors of",
        f"the item images ({n_images} images of targets and top-5 items). Comparable rows (the same single color",
        f"named in the Original and the Full text): {len(row_df)}, over {row_df['case_key'].nunique()} cases.",
        "",
        "| Level | Problems under Original | Solved under Full | Share | Manuscript (2025 models) |",
        "|---|---:|---:|---:|---:|",
        f"| Row (seed, set, target) | {row['denominator']} | {row['solved_n']} | {row['solved_ratio']:.4f} | 39 / 89 = 0.4382 |",
        f"| Case (set, target) | {case['denominator']} | {case['solved_n']} | {case['solved_ratio']:.4f} | 5 / 15 = 0.3333 |",
        "",
        "The 2025 per-query files are not preserved, so the 2025 counts cannot be recomputed; the comparable rows",
        "(745 over 149 cases in 2025) depend only on the texts and the fair subset.",
    ]
    write_text(out / "summary.md", lines)
    print("\n".join(lines))


if __name__ == "__main__":
    main()
