#!/usr/bin/env python3
"""Audits of the input data that the reproduction uses as fixed inputs.

The generated descriptions, the CLO / MET / temperature records, the W/O/S
annotations and the counterfactual pairs were built in 2025 and are used by the
official run unchanged. This script checks them and writes, under --out-dir
(default reproduction/results/supplementary/input_data_audit/):

- temperature_prefix_audit.csv and temperature_prefix_mismatches.csv: the
  temperature at the start of each generated description against the computed
  reference TSUB_target_C;
- feature_text_identity.json: whether the context-aware text features used by
  the official run encode the stored descriptions (descriptions with identical
  text but different temperature references must have identical vectors);
- thermal_mapping_checks.json: equation (2) recomputed for every outfit, and the
  outfits outside McIntyre's stated range (M < 150 W/m2, Icl < 1.5 clo);
- split_item_overlap.csv: item IDs shared between the disjoint split files;
- counterfactual_pair_check.csv: non-target words removed by each
  counterfactual replacement, and occasion pairs without an occasion fragment;
- wos_formal_fragments.csv and met_candidate_check.csv: two annotation counts;
- clo_distribution_summary.csv and figures/figure_D_4_clo_distribution.svg: the
  distribution of the CLO estimates (thesis Table D-3 and Figure D-4; notebook
  01_資料建構_data_construction/clo_met_temperature/CLO_reference/Adding_CLO.ipynb, cell 17);
- category_threshold_check.csv: for every fine-grained category, its items per
  split and its CIR candidate pool under the rule of the archived evaluator
  (evaluate_cir.py: at most 3,000 items, from the test split and then from the
  train split; categories below 3,000 are not evaluated), with the male-labelled
  categories marked (thesis Section 5.2);
- summary.md.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import pickle
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from _common import (CLO_RESULTS, CONTEXT_FEATURES, COUNTERFACTUAL, GENERATED, MET_CANDIDATES, SUPPLEMENTARY,
                     WOS_JSONL, fmt, load_temperatures, polyvore_root, read_csv, read_json, write_csv, write_json,
                     write_text)
from _svg import Chart, draw_boxplot, draw_histogram, nice_ticks, text_width

PREFIX = re.compile(r"^\s*(-?\d+(?:\.\d+)?)\s*°\s*C", re.IGNORECASE)
WORD = re.compile(r"[a-z][a-z'-]+")
CLO_SPLITS = ("train", "test", "valid")  # Adding_CLO cell 17 reads the files in this order
TABLE_D3 = {"count": 35140, "mean": 1.0066, "std": 0.5586, "min": 0.0, "q1": 0.63, "median": 0.93, "q3": 1.2125,
            "max": 5.2, "iqr": 0.5825, "iqr_lower_bound": -0.2437, "iqr_upper_bound": 2.0862, "n_outliers": 1726}
POOL_SIZE = 3000  # evaluate_cir.py and train_cir.py


class NumpyOnlyUnpickler(pickle.Unpickler):
    """Loads the feature dictionary; refuses any class other than numpy arrays."""
    ALLOWED = {("numpy._core.multiarray", "_reconstruct"), ("numpy.core.multiarray", "_reconstruct"),
               ("numpy", "ndarray"), ("numpy", "dtype")}

    def find_class(self, module, name):
        if (module, name) in self.ALLOWED:
            return super().find_class(module, name)
        raise pickle.UnpicklingError(f"blocked {module}.{name}")


def matches_reference(stated: float, tsub: float) -> bool:
    """The prompt asks for the rounded temperature; one decimal place is also accepted."""
    half_up = int(tsub + 0.5) if tsub >= 0 else -int(-tsub + 0.5)
    return abs(stated - tsub) < 0.051 or stated in (round(tsub), int(tsub), half_up)


def mismatch_pattern(stated: float, tsub: float) -> str:
    if tsub < 0 and stated in (abs(round(tsub)), abs(int(tsub)), abs(int(-tsub + 0.5))):
        return "minus sign dropped"
    if stated == round(abs(tsub) * 10) % 10:
        return "only the decimal digit kept"
    return "no pattern"


def norm(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "").lower()).strip()


def clo_values(split: str) -> list[float]:
    """Adding_CLO cell 17: total_clo of every record whose value is a finite number."""
    values = []
    for line in (CLO_RESULTS / f"{split}_clo_results.jsonl").read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            v = json.loads(line).get("total_clo")
        except Exception:
            continue
        if isinstance(v, (int, float)) and math.isfinite(v):
            values.append(float(v))
    return values


def clo_statistics(arr: np.ndarray) -> dict:
    """Adding_CLO cell 17, statistics (numpy percentiles, sample SD)."""
    q1, median, q3 = (float(np.percentile(arr, q)) for q in (25, 50, 75))
    iqr = q3 - q1
    lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    return {"count": len(arr), "mean": float(np.mean(arr)), "std": float(np.std(arr, ddof=1)), "min": float(np.min(arr)),
            "q1": q1, "median": median, "q3": q3, "max": float(np.max(arr)), "iqr": iqr, "iqr_lower_bound": lower,
            "iqr_upper_bound": upper, "n_outliers": int(np.sum((arr < lower) | (arr > upper)))}


def clo_figure(path: Path, arr: np.ndarray, per_split: dict, st: dict) -> float:
    """Thesis Figure D-4 as drawn by Adding_CLO cell 17: zoomed histogram, full range by split, box plot."""
    p99 = float(np.percentile(arr, 99))
    x_zoom = max(min(p99, st["iqr_upper_bound"] * 2.5), st["q3"] + st["iqr"] * 2)
    share = 100 * st["n_outliers"] / st["count"]
    c = Chart(1060, 520, f"CLO distribution summary, train + test + valid (n={st['count']}, outliers={st['n_outliers']} / "
              f"{share:.1f}%) (thesis Figure D-4)")
    a = c.panel(80, 76, 520, 360, "CLO histogram: " + " + ".join(f"{s}({len(per_split[s])})" for s in CLO_SPLITS))
    lo = float(arr.min())
    # cell 17 clips to [0, x_zoom]: values above x_zoom are counted in the last bin
    draw_histogram(a, [("all", np.clip(arr, 0, x_zoom), "#4C72B0")], 40, lo, x_zoom, "total_clo", "count",
                   vlines=[(v, color, dash) for v, color, dash in ((st["mean"], "#DD8452", "6,3"),
                                                                    (st["median"], "#55A868", "2,2"),
                                                                    (st["iqr_lower_bound"], "#C44E52", "6,2,2,2"),
                                                                    (st["iqr_upper_bound"], "#C44E52", "6,2,2,2"))
                           if lo <= v <= x_zoom], opacity=0.85)
    c.legend(a.left + a.width - 205, a.top + 16,
             [(f"mean = {st['mean']:.3f}", "#DD8452", "dash"), (f"median = {st['median']:.3f}", "#55A868", "dash"),
              (f"IQR bounds [{st['iqr_lower_bound']:.2f}, {st['iqr_upper_bound']:.2f}]", "#C44E52", "dash")])
    clipped = int(np.sum(arr > x_zoom))
    if clipped:
        note = f"{clipped} values > {x_zoom:.1f} clipped into the last bin"
        c.rect(a.left + a.width - 211, a.top + 60, text_width(note, 9) + 12, 15, "#ffffff", stroke="#C44E52", opacity=0.9)
        c.text(a.left + a.width - 205, a.top + 71, note, size=9, color="#C44E52")
    b = c.panel(690, 76, 330, 130, "Full range (by split)")
    edges = np.linspace(lo, float(arr.max()), 80)
    counts = {s: np.histogram(np.asarray(per_split[s]), bins=edges)[0] for s in CLO_SPLITS}
    top = max(int(v.max()) for v in counts.values()) * 1.08
    b.set_x(lo, float(arr.max()))
    b.set_y(0, top)
    b.axes(xticks=[t for t in nice_ticks(lo, float(arr.max()), 5) if t <= arr.max()],
           yticks=[t for t in nice_ticks(0, top, 3) if t <= top], size=8.5)
    colors = {"train": "#4C72B0", "test": "#55A868", "valid": "#DD8452"}
    for s in CLO_SPLITS:
        for i, n in enumerate(counts[s]):
            if n:
                b.bar(edges[i], edges[i + 1], 0, int(n), colors[s], opacity=0.55, stroke=None)
    b.vline(st["iqr_upper_bound"], "#C44E52", "6,2,2,2")
    c.legend(b.left + b.width - 120, b.top + 14, [(f"{s} ({len(per_split[s])})", colors[s], "box") for s in CLO_SPLITS],
             size=9)
    d = c.panel(760, 266, 120, 200, "CLO box plot (zoomed)")
    inside = arr[(arr >= st["iqr_lower_bound"]) & (arr <= st["iqr_upper_bound"])]
    outliers = arr[(arr < inside.min()) | (arr > inside.max())]
    draw_boxplot(d, st["q1"], st["median"], st["q3"], float(inside.min()), float(inside.max()), outliers.tolist(),
                 max(lo - 0.1, -0.15), x_zoom, "total_clo")
    above = int(np.sum(arr > x_zoom))
    if above:
        c.text(d.left + d.width + 8, d.top + 10, f"{above} points > {x_zoom:.1f}", size=9, color="#C44E52")
    c.save(path)
    return x_zoom


def evaluator_pools(primary: dict, extra: dict) -> tuple[dict, object]:
    """evaluate_cir.py / train_cir.py: per fine-grained category, unique items of the evaluated split, then of the
    train split, up to 3,000; the train loop stops (break) at the first category the evaluated split lacks."""
    ids, seen = {}, {}
    for k, v in primary.items():
        ids.setdefault(k, [])
        seen.setdefault(k, set())
        for item_lst in v.values():
            for item_id in item_lst:
                if item_id not in seen[k]:
                    seen[k].add(item_id)
                    ids[k].append(item_id)
                if len(ids[k]) >= POOL_SIZE:
                    break
            if len(ids[k]) >= POOL_SIZE:
                break
    stopped = None
    for k, v in extra.items():
        if k not in ids:
            stopped = k
            break
        for item_lst in v.values():
            for item_id in item_lst:
                if item_id not in seen[k]:
                    seen[k].add(item_id)
                    ids[k].append(item_id)
                if len(ids[k]) >= POOL_SIZE:
                    break
            if len(ids[k]) >= POOL_SIZE:
                break
    return {k: len(v) for k, v in ids.items()}, stopped


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--polyvore-root", default=None)
    parser.add_argument("--out-dir", type=Path, default=SUPPLEMENTARY / "input_data_audit")
    args = parser.parse_args()
    out = args.out_dir
    poly = polyvore_root(args.polyvore_root)

    generated = read_json(GENERATED)
    temperatures = load_temperatures()

    # ---------------------------------------------------------------- temperature prefixes
    per_split = defaultdict(Counter)
    mismatches = []
    for set_id, value in generated.items():
        split, record = temperatures[set_id]
        tsub = float(record["TSUB_target_C"])
        text = value["title"] if isinstance(value, dict) else str(value)
        match = PREFIX.match(text)
        per_split[split]["descriptions"] += 1
        if not match:
            per_split[split]["no temperature prefix"] += 1
            mismatches.append([set_id, split, fmt(tsub, 1), "", "no temperature prefix", text[:60]])
            continue
        stated = float(match.group(1))
        per_split[split]["integer format" if "." not in match.group(1) else "one-decimal format"] += 1
        if matches_reference(stated, tsub):
            per_split[split]["matching"] += 1
        else:
            pattern = mismatch_pattern(stated, tsub)
            per_split[split]["mismatching"] += 1
            per_split[split][pattern] += 1
            mismatches.append([set_id, split, fmt(tsub, 1), match.group(1), pattern, text[:60]])
    columns = ["descriptions", "matching", "mismatching", "only the decimal digit kept", "minus sign dropped",
               "no pattern", "no temperature prefix", "integer format", "one-decimal format"]
    split_rows = [[split, *[per_split[split][c] for c in columns]] for split in ("train", "valid", "test")]
    split_rows.append(["all", *[sum(per_split[s][c] for s in per_split) for c in columns]])
    write_csv(out / "temperature_prefix_audit.csv", ["split", *columns], split_rows)
    mismatches.sort(key=lambda r: (r[1], r[0]))
    write_csv(out / "temperature_prefix_mismatches.csv",
              ["set_id", "split", "tsub_target_c", "stated_temperature", "pattern", "description_start"], mismatches)

    # ---------------------------------------------------------------- feature identity
    with CONTEXT_FEATURES.open("rb") as f:
        features = NumpyOnlyUnpickler(f).load()
    groups = defaultdict(list)
    for set_id, value in generated.items():
        groups[(value["title"] if isinstance(value, dict) else str(value)).strip()].append(set_id)
    pairs = identical = 0
    for members in groups.values():
        for i, a in enumerate(members):
            for b in members[i + 1:]:
                if abs(float(temperatures[a][1]["TSUB_target_C"]) - float(temperatures[b][1]["TSUB_target_C"])) > 0.05:
                    pairs += 1
                    identical += int(np.allclose(features[a], features[b], atol=1e-6))
    identity = {"feature_file": CONTEXT_FEATURES.name,
                "pairs_with_identical_text_and_different_reference": pairs,
                "pairs_with_identical_vectors": identical,
                "conclusion": ("the features encode the stored descriptions as written" if pairs and identical == pairs
                               else "inconclusive")}
    write_json(out / "feature_text_identity.json", identity)

    # ---------------------------------------------------------------- equation (2) and McIntyre range
    worst = 0.0
    matching = outside_m = outside_clo = outside_any = 0
    for _, record in temperatures.values():
        clo, met = float(record["total_clo"]), float(record["met_value"])
        m = met * 58.2
        tsub = 33.5 - 3 * clo - (0.08 + 0.05 * clo) * m
        diff = abs(round(tsub, 1) - float(record["TSUB_target_C"]))
        worst = max(worst, diff)
        matching += int(diff <= 0.051)
        outside_m += int(m >= 150)
        outside_clo += int(clo >= 1.5)
        outside_any += int(m >= 150 or clo >= 1.5)
    n = len(temperatures)
    thermal = {"outfits": n,
               "equation_2_matching_stored_reference": matching,
               "largest_difference_after_rounding": round(worst, 3),
               "metabolic_rate_w_m2": "MET x 58.2",
               "outside_mcintyre_range": {"M_at_least_150": outside_m, "Icl_at_least_1_5": outside_clo,
                                          "either": outside_any, "share": round(outside_any / n, 4)}}
    write_json(out / "thermal_mapping_checks.json", thermal)

    # ---------------------------------------------------------------- split overlap
    items = {}
    split_outfits = {}
    for split in ("train", "valid", "test"):
        outfits = split_outfits[split] = read_json(poly / "disjoint" / f"{split}.json")
        items[split] = {str(item["item_id"]) for outfit in outfits for item in outfit["items"]}
    overlap_rows = [[a, b, len(items[a] & items[b])] for a, b in (("train", "test"), ("train", "valid"), ("valid", "test"))]
    write_csv(out / "split_item_overlap.csv", ["split_a", "split_b", "shared_item_ids"], overlap_rows)

    # ---------------------------------------------------------------- counterfactual pairs
    pair_rows = []
    for record in read_csv(COUNTERFACTUAL / "A44_counterfactual_context_pairs.csv"):
        factor, cf = record["factor"], norm(record["counterfactual_description"])
        removed = []
        for other, column in (("weather", "weather_text"), ("occasion", "occasion_text"), ("style", "style_text")):
            if other == factor:
                continue
            lost = [w for w in WORD.findall(norm(record[column])) if w not in cf]
            if lost:
                removed.append(f"{other}: {' / '.join(lost)}")
        no_fragment = factor == "occasion" and not record["occasion_text"].strip()
        pair_rows.append([record["pair_id"], factor, record["direction"], "; ".join(removed),
                          "yes" if no_fragment else ""])
    write_csv(out / "counterfactual_pair_check.csv",
              ["pair_id", "factor", "direction", "non_target_words_removed", "occasion_pair_without_occasion_fragment"],
              pair_rows)

    # ---------------------------------------------------------------- annotation counts
    formal = Counter()
    for line in WOS_JSONL.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        if record.get("valid") is not True:
            continue
        for factor, values in (record.get("fragments") or {}).items():
            for value in values if isinstance(values, list) else [values]:
                if str(value).strip().lower() == "formal":
                    formal[factor] += 1
    write_csv(out / "wos_formal_fragments.csv", ["factor", "standalone_formal_fragments"],
              [[k, formal[k]] for k in ("weather", "occasion", "style")])

    candidate_rows = read_json(MET_CANDIDATES)
    candidates = {str(c["ActivityDescription"]).strip() for c in candidate_rows}
    outside = Counter(str(r["activity"]).strip() for _, r in temperatures.values()
                      if str(r["activity"]).strip() not in candidates)
    write_csv(out / "met_candidate_check.csv", ["activity_not_in_candidate_list", "outfits"],
              sorted(([a, c] for a, c in outside.items()), key=lambda r: (-r[1], r[0])))

    # ---------------------------------------------------------------- CLO distribution (thesis Table D-3, Figure D-4)
    per_split = {split: clo_values(split) for split in CLO_SPLITS}
    clo = np.array([v for split in CLO_SPLITS for v in per_split[split]])
    clo_stats = clo_statistics(clo)
    x_zoom = clo_figure(out / "figures" / "figure_D_4_clo_distribution.svg", clo, per_split, clo_stats)

    def shown(key: str, value: float) -> str:
        return f"{value:.0f}" if key in ("count", "n_outliers") else f"{value:.4f}"

    clo_rows = [[k, shown(k, v), shown(k, TABLE_D3[k]), "yes" if shown(k, v) == shown(k, TABLE_D3[k]) else "NO"]
                for k, v in clo_stats.items()]
    clo_rows += [[f"count_{split}", len(per_split[split]), "", ""] for split in CLO_SPLITS]
    write_csv(out / "clo_distribution_summary.csv", ["statistic", "value", "thesis_table_d3", "same_at_4_decimals"],
              clo_rows)

    # ---------------------------------------------------------------- category threshold (thesis Section 5.2)
    meta = read_json(poly / "polyvore_item_metadata.json")
    first_label, all_labels = {}, defaultdict(list)
    with (poly / "categories.csv").open(encoding="utf-8-sig", newline="") as f:
        for r in csv.reader(f):
            if len(r) >= 3 and r[0].strip():
                first_label.setdefault(r[0].strip(), (r[1].strip(), r[2].strip()))
                all_labels[r[0].strip()].append(r[1].strip())
    fg2ims = {}  # dataset.py: category_id -> {set_id: [item ids]} in order of first appearance
    id2im = {}
    for split, outfits in split_outfits.items():
        fg2ims[split] = {}
        for outfit in outfits:
            for item in outfit["items"]:
                fg = str(meta[item["item_id"]]["category_id"])
                fg2ims[split].setdefault(fg, {}).setdefault(outfit["set_id"], []).append(item["item_id"])
                if split == "test":
                    id2im[f"{outfit['set_id']}_{item['index']}"] = item["item_id"]
    test_pool, _ = evaluator_pools(fg2ims["test"], fg2ims["train"])
    valid_pool, valid_stop = evaluator_pools(fg2ims["valid"], fg2ims["train"])
    evaluated = {k for k, n in test_pool.items() if n >= POOL_SIZE}
    queries = Counter()
    for q in read_json(poly / "disjoint" / "fill_in_blank_test.json"):  # evaluate_cir.py: test FITB targets
        gt = q["question"][0].split("_")[0]
        target = next(id2im[a] for a in q["answers"] if a.split("_")[0] == gt)
        queries[str(meta[target]["category_id"])] += 1
    male_any = {k for k, labels in all_labels.items() if any(l.lower().startswith("male") for l in labels)}
    male_first = {k for k, (name, _) in first_label.items() if name.lower().startswith("male")}
    unique = {split: {k: len({i for lst in v.values() for i in lst}) for k, v in fg2ims[split].items()}
              for split in fg2ims}
    present = sorted(set().union(*(set(u) for u in unique.values())), key=lambda k: (-test_pool.get(k, 0), int(k)))
    category_rows = [[k, first_label.get(k, ("", ""))[0], " | ".join(all_labels.get(k, [])), first_label.get(k, ("", ""))[1],
                      "yes" if k in male_first else "", "yes" if k in male_any else "",
                      *[unique[split].get(k, 0) for split in ("train", "valid", "test")], test_pool.get(k, ""),
                      "yes" if k in evaluated else "", queries.get(k, 0), valid_pool.get(k, ""),
                      "yes" if valid_pool.get(k, 0) >= POOL_SIZE else ""] for k in present]
    write_csv(out / "category_threshold_check.csv",
              ["category_id", "first_label", "all_labels", "main_category", "male_first_label", "male_any_label",
               "unique_items_train", "unique_items_valid", "unique_items_test", "cir_test_pool", "cir_evaluated",
               "cir_test_questions", "validation_pool", "validation_evaluated"], category_rows)
    male_present = [k for k in present if k in male_any]
    biggest_male = max(male_present, key=lambda k: test_pool.get(k, 0))
    male_outfits = {split: sum(1 for o in outfits if any(str(meta[i["item_id"]]["category_id"]) in male_any
                                                         for i in o["items"])) for split, outfits in split_outfits.items()}
    male_items = {split: sum(unique[split].get(k, 0) for k in male_any) for split in unique}

    # ---------------------------------------------------------------- summary
    total = split_rows[-1]
    lost_pairs = [r for r in pair_rows if r[3]]
    lines = [
        "# Input data audit",
        "",
        "Computed by `reproduction/scripts/supplementary/input_data_audit.py`. The audited files are the 2025",
        "inputs that the official run uses unchanged; nothing was regenerated.",
        "",
        "## Temperature at the start of the generated descriptions",
        "",
        "| Split | Descriptions | Matching | Mismatching | Decimal digit only | Minus sign dropped | No pattern |",
        "|---|---:|---:|---:|---:|---:|---:|",
        *[f"| {r[0]} | {r[1]} | {r[2]} | {r[3]} | {r[4]} | {r[5]} | {r[6]} |" for r in split_rows],
        "",
        f"{total[3]} of {total[1]} descriptions ({100 * total[3] / total[1]:.1f}%) state a temperature that differs",
        "from `TSUB_target_C`. The train descriptions use an integer format and the valid and test descriptions one",
        "decimal place; the preserved post-processing writes only the latter.",
        "",
        f"Feature check: {identical} of {pairs} description pairs with identical text but different temperature",
        f"references have identical context-aware text features ({identity['conclusion']}).",
        "",
        "## Thermal mapping",
        "",
        f"- Equation (2) reproduces the stored temperature reference for {matching} of {n} outfits.",
        f"- Outside McIntyre's stated range (M < 150 W/m², Icl < 1.5 clo): {outside_any} outfits "
        f"({100 * outside_any / n:.1f}%); Icl ≥ 1.5: {outside_clo}; M ≥ 150: {outside_m}.",
        "",
        "## Other checks",
        "",
        "- Item IDs shared between the disjoint split files: "
        + "; ".join(f"{a} and {b}: {c}" for a, b, c in overlap_rows) + ".",
        f"- Counterfactual pairs whose replacement also removed words of a non-target factor: "
        + (", ".join(f"{r[0]} ({r[3]})" for r in lost_pairs) or "none") + ".",
        f"- Occasion pairs without an occasion fragment: "
        + (", ".join(r[0] for r in pair_rows if r[4]) or "none") + ".",
        f"- Standalone \"formal\" fragments: occasion {formal['occasion']}, style {formal['style']}, "
        f"weather {formal['weather']}.",
        f"- Outfits whose activity is not in the {len(candidate_rows)}-entry MET candidate list "
        f"({len(candidates)} distinct descriptions): {sum(outside.values())} ({len(outside)} activities).",
        "",
        "## CLO distribution (thesis Table D-3 and Figure D-4)",
        "",
        "Adding_CLO cell 17 recomputed on the preserved CLO estimates (`clo_distribution_summary.csv`,",
        "`figures/figure_D_4_clo_distribution.svg`).",
        "",
        "| Statistic | Recomputed | Thesis Table D-3 |",
        "|---|---:|---:|",
        *[f"| {r[0]} | {r[1]} | {r[2]} |" for r in clo_rows if not r[0].startswith("count_")],
        "",
        f"All {len(clo_stats)} statistics equal Table D-3 at four decimals: "
        f"{'yes' if all(r[3] == 'yes' for r in clo_rows[:len(clo_stats)]) else 'NO'}. Records per file: "
        + ", ".join(f"{split} {len(per_split[split])}" for split in CLO_SPLITS) + f"; the zoomed panels end at {x_zoom:.2f}.",
        "",
        "## Category threshold (thesis Section 5.2)",
        "",
        "The archived evaluator builds, for every fine-grained category of the test split, a candidate pool of at",
        f"most {POOL_SIZE:,} items (test items first, then train items) and evaluates only the categories whose pool",
        f"reaches {POOL_SIZE:,} (`category_threshold_check.csv`).",
        "",
        f"- CIR test evaluation: {len(evaluated)} of {len(test_pool)} test categories; "
        f"{sum(queries[k] for k in evaluated)} of {sum(queries.values())} test FITB questions have their target in "
        f"them (the official run evaluates 9,311 queries: {'same' if sum(queries[k] for k in evaluated) == 9311 else 'DIFFERENT'}).",
        f"- Male-labelled categories in the data: {len(male_present)} ({len(male_first & set(male_present))} of them by the "
        "first label of a duplicated ID in categories.csv, as the 2025 programs read it; category 21 is also listed "
        "as 'tshirt'). The largest pool among them is category "
        f"{biggest_male} ({' / '.join(all_labels[biggest_male])}) with {test_pool.get(biggest_male, 0):,} items; none "
        f"reaches {POOL_SIZE:,}, so no male-labelled item is a CIR target "
        f"({sum(queries[k] for k in male_present)} test questions with a male-labelled target are skipped).",
        f"- Male-labelled items are nevertheless part of the training and test outfits: {male_items['train']:,} items "
        f"in {male_outfits['train']:,} train outfits, {male_items['test']:,} items in {male_outfits['test']:,} test "
        "outfits. The 3,000 rule selects the CIR evaluation categories only; it does not filter the data used for "
        "training or for the compatibility and FITB evaluation, whereas Section 5.2 of the thesis says that only "
        "categories with more than 3,000 items were used in training.",
        f"- Validation during CIR training (train_cir.py, the same rule on the validation split): "
        f"{sum(1 for n in valid_pool.values() if n >= POOL_SIZE)} of {len(valid_pool)} categories, because the "
        f"train loop stops at category {valid_stop} ({' / '.join(all_labels.get(str(valid_stop), ['?']))}), the "
        "first train category without validation items. Only the logged validation recall is affected; checkpoints "
        "are selected by validation FITB accuracy.",
    ]
    write_text(out / "summary.md", lines)
    print(f"[SUPPLEMENTARY] input data audit written to {out}")


if __name__ == "__main__":
    main()
