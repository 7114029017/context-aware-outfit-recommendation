#!/usr/bin/env python3
"""Regenerate the 24 counterfactual pairs (manuscript Table 4) and compare them with the archived A44.

The pairs were drawn by the 2025 notebook
03_實驗與結果_experiments_results/06_反事實情境敏感度/source_programs/P16_counterfactual_context_consistency_check.ipynb
(cells 1 and 3): from the 9,311 main CIR queries, four queries for each of six directions (warm to cold,
cold to warm, formal to casual, casual to formal, high to low style, low to high style), drawn with
numpy's default_rng(42) and balanced over clothing-led and accessory-led targets; the counterfactual
description is the description without the factor (the W/O/S annotation's title_ablation) joined with a
fixed phrase. The code below is copied from those cells. P16 took the queries from the 2025 seed-1
per-query file, which is not preserved; the same 9,311 queries are taken from the seed-1 rows of the
archived A07 table (their ranks are not used by the selection).

Writes under --out-dir (default reproduction/results/supplementary/counterfactual_pairs_check/)
regenerated_pairs.csv and summary.md (pairs identical to A44 in id, set, factor, direction, target and
counterfactual text). CPU only, a few seconds.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from _common import COUNTERFACTUAL, D03, SUPPLEMENTARY, WOS_JSONL, polyvore_root, read_csv, write_csv, write_text

A07 = D03 / "01_文字長度影響分析" / "A07_length_performance_rowlevel_cir.csv"
A44 = COUNTERFACTUAL / "A44_counterfactual_context_pairs.csv"
RANDOM_SEED, N_PER_DIRECTION = 42, 4  # P16 cell 1
TEMP_RE = re.compile(r"(-?\d+(?:\.\d+)?)\s*(?:°\s*)?[cC]\b")
FORMAL_RE = re.compile(r"\b(formal|office|business|work|professional|interview|meeting|wedding|party|cocktail|"
                       r"elegant|tailored|blazer|suit)\b", re.I)
CASUAL_RE = re.compile(r"\b(casual|weekend|streetwear|street|relaxed|everyday|daily|chill|laid-back|lounge|vacation|"
                       r"travel)\b", re.I)
COMPARED = ("pair_id", "set_id", "factor", "direction", "target_item_id", "counterfactual_description")


def load_json(path: Path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def clean_text(v) -> str:
    if isinstance(v, list):
        v = " ".join(str(x) for x in v if str(x).strip())
    return re.sub(r"\s+", " ", str(v or "")).strip()


def parse_temperature(t) -> float:
    m = TEMP_RE.search(str(t))
    return float(m.group(1)) if m else np.nan


def read_categories(path: Path) -> dict:
    out = {}
    with open(path, "r", encoding="utf-8") as f:
        for row in csv.reader(f):
            if len(row) >= 3:
                out[str(row[0])] = {"fine_category": row[1].strip(), "main_category": row[2].strip()}
    return out


def normalize_main_category(t) -> str:
    return str(t or "").strip().lower().replace("-", " ")


def target_group(m) -> str:
    m = normalize_main_category(m)
    if m in {"all body", "bottoms", "tops", "outerwear"}:
        return "clothing-led"
    if m in {"bags", "shoes", "accessories", "hats", "jewellery", "scarves", "sunglasses"}:
        return "accessory-led"
    return "other"


def item_category(meta: dict, cat_map: dict, i) -> tuple[str, str, str]:
    info = meta.get(str(i), {})
    cid = str(info.get("category_id", ""))
    mapped = cat_map.get(cid, {})
    return cid, mapped.get("fine_category", ""), mapped.get("main_category", info.get("semantic_category", ""))


def load_fragments(path: Path) -> pd.DataFrame:
    rows = []
    for line in open(path, encoding="utf-8"):
        if not line.strip():
            continue
        obj = json.loads(line)
        if obj.get("valid") is not True:
            continue
        frags, abl = obj.get("fragments", {}), obj.get("title_ablation", {})
        w, o, s = frags.get("weather", []), frags.get("occasion", []), frags.get("style", [])
        title_full, wt = clean_text(obj.get("title", "")), clean_text(w)
        temp = parse_temperature(wt)
        if pd.isna(temp):
            temp = parse_temperature(title_full)
        rows.append({"set_id": str(obj.get("id", "")), "title_full_text": title_full, "weather_text": wt,
                     "occasion_text": clean_text(o), "style_text": clean_text(s),
                     "style_richness": len([x for x in s if str(x).strip()]) if isinstance(s, list) else 0,
                     "title_no_weather_text": clean_text(abl.get("no_weather", "")),
                     "title_no_occasion_text": clean_text(abl.get("no_occasion", "")),
                     "title_no_style_text": clean_text(abl.get("no_style", "")), "temperature_c": temp})
    return pd.DataFrame(rows)


def controlled_counterfactual_text(row, factor: str, direction: str) -> str:
    if factor == "weather":
        core = clean_text(row.get("title_no_weather_text", ""))
        rep = "10°C, cool layered weather" if direction == "warm_to_cold" else "30°C, warm breathable weather"
        return clean_text(f"{rep}, {core}")
    if factor == "occasion":
        core = clean_text(row.get("title_no_occasion_text", ""))
        rep = "for a casual daily outing" if direction == "formal_to_casual" else "for a formal work meeting"
        return clean_text(f"{core}, {rep}")
    core = clean_text(row.get("title_no_style_text", ""))
    rep = ("simple minimal look" if direction == "high_to_low_style"
           else "polished layered elegant style with coordinated details")
    return clean_text(f"{core}, {rep}")


def regenerate(poly: Path) -> tuple[list[dict], int, float]:
    fragments = load_fragments(WOS_JSONL)
    a07 = pd.read_csv(A07, dtype=str)
    detail = a07[a07["seed"] == "1"][["set_id", "target_item_id", "target_item_fg"]].copy()
    test_data = load_json(poly / "disjoint" / "test.json")
    fitb_data = load_json(poly / "disjoint" / "fill_in_blank_test.json")
    meta, cat_map = load_json(poly / "polyvore_item_metadata.json"), read_categories(poly / "categories.csv")
    ref_to_item = {}
    for outfit in test_data:
        sid = str(outfit["set_id"])
        for item in outfit["items"]:
            ref_to_item[f"{sid}_{int(item['index'])}"] = str(item["item_id"])
    official = set(detail["set_id"])
    fitb_rows = []
    for q in fitb_data:
        if not q.get("question"):
            continue
        sid = str(q["question"][0]).split("_")[0]
        if sid not in official:
            continue
        correct = [x for x in q["answers"] if str(x).split("_")[0] == sid]
        if not correct:
            continue
        tid = ref_to_item[correct[0]]
        cid, fine, main = item_category(meta, cat_map, tid)
        fitb_rows.append({"set_id": sid, "target_item_id_from_fitb": tid, "target_category_id": cid,
                          "target_fine_category": fine, "target_main_category": main,
                          "target_group": target_group(main)})
    fitb_df = pd.DataFrame(fitb_rows)
    work = (fragments.merge(detail, on="set_id", how="inner").merge(fitb_df, on="set_id", how="inner"))
    work = work[work["target_item_id"] == work["target_item_id_from_fitb"]].copy()
    thr = float(work["temperature_c"].dropna().median())
    src = work["occasion_text"].fillna("") + " " + work["title_full_text"].fillna("")
    work["is_formal"] = src.map(lambda x: bool(FORMAL_RE.search(str(x))))
    work["is_casual"] = src.map(lambda x: bool(CASUAL_RE.search(str(x))))
    work["is_high_style"] = work["style_richness"].astype(int) >= 2
    work["is_low_style"] = work["style_richness"].astype(int) <= 1
    specs = [("weather", "warm_to_cold", lambda d: d["temperature_c"] > thr + 2),
             ("weather", "cold_to_warm", lambda d: d["temperature_c"] <= thr - 2),
             ("occasion", "formal_to_casual", lambda d: d["is_formal"] & ~d["is_casual"]),
             ("occasion", "casual_to_formal", lambda d: d["is_casual"] & ~d["is_formal"]),
             ("style", "high_to_low_style", lambda d: d["is_high_style"]),
             ("style", "low_to_high_style", lambda d: d["is_low_style"])]
    rng = np.random.default_rng(RANDOM_SEED)

    def pick_balanced(pool: pd.DataFrame, n: int, used: set) -> pd.DataFrame:
        pool = pool[~pool["set_id"].isin(used)].copy()
        pool["_rand"] = rng.random(len(pool))
        pool = pool.sort_values("_rand")
        picked, used_main = [], set()
        for group_name, quota in [("clothing-led", 2), ("accessory-led", 2)]:
            sub = pool[pool["target_group"] == group_name]
            for _, r in sub.iterrows():
                if len([i for i in picked if pool.loc[i, "target_group"] == group_name]) >= quota:
                    break
                if r["target_main_category"] in used_main:
                    continue
                picked.append(r.name)
                used_main.add(r["target_main_category"])
            if len([i for i in picked if pool.loc[i, "target_group"] == group_name]) < quota:
                for _, r in sub.iterrows():
                    if r.name in picked:
                        continue
                    picked.append(r.name)
                    if len([i for i in picked if pool.loc[i, "target_group"] == group_name]) >= quota:
                        break
        if len(picked) < n:
            for _, r in pool.iterrows():
                if r.name not in picked:
                    picked.append(r.name)
                if len(picked) >= n:
                    break
        return pool.loc[picked[:n]].drop(columns=["_rand"]).sort_values(
            ["target_group", "target_main_category", "set_id"])

    used, selected = set(), []
    for factor, direction, filt in specs:
        for _, row in pick_balanced(work[filt(work)].copy(), N_PER_DIRECTION, used).iterrows():
            used.add(row["set_id"])
            selected.append({"pair_id": f"CF{len(selected) + 1:02d}", "set_id": row["set_id"], "factor": factor,
                             "direction": direction, "target_item_id": row["target_item_id"],
                             "counterfactual_description": controlled_counterfactual_text(row, factor, direction)})
    return selected, len(detail), thr


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--polyvore-root", default=None)
    parser.add_argument("--out-dir", type=Path, default=SUPPLEMENTARY / "counterfactual_pairs_check")
    args = parser.parse_args()
    poly = polyvore_root(args.polyvore_root)
    selected, frame, thr = regenerate(poly)
    archived = read_csv(A44)
    write_csv(args.out_dir / "regenerated_pairs.csv", list(COMPARED), [[r[k] for k in COMPARED] for r in selected])
    same = [all(a[k] == b[k] for k in COMPARED) for a, b in zip(archived, selected)]
    lines = ["# Counterfactual pairs regenerated (manuscript Table 4; 2025 notebook P16, cells 1 and 3)", "",
             f"Sampling frame: {frame:,} main CIR queries (seed-1 rows of the archived A07); temperature median "
             f"{thr:.2f} °C; numpy default_rng({RANDOM_SEED}), {N_PER_DIRECTION} pairs per direction.", "",
             f"Pairs identical to the archived A44 (pair id, set, factor, direction, target item and counterfactual "
             f"description): {sum(same)}/{len(archived)}.", ""]
    differ = [f"- {a['pair_id']}: archived set {a['set_id']}, regenerated {b['set_id']}"
              for a, b, ok in zip(archived, selected, same) if not ok]
    if differ or len(selected) != len(archived):
        lines += ["Differences:", "", *differ, ""]
    write_text(args.out_dir / "summary.md", lines)
    print("\n".join(lines))
    print(f"[SUPPLEMENTARY] counterfactual pairs written to {args.out_dir}")


if __name__ == "__main__":
    main()
