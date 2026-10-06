#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import argparse
import csv
import json
import math
import statistics
import sys

PAPER = {
    "cp_auc_original": 0.9292,
    "cp_auc_context": 0.9454,
    "cp_fitb_original": 0.6374,
    "cp_fitb_context": 0.6478,
    "or_r1_original": 0.0139,
    "or_r1_context": 0.0151,
    "or_r3_original": 0.0333,
    "or_r3_context": 0.0369,
    "or_r5_original": 0.0476,
    "or_r5_context": 0.0557,
    "or_r10_original": 0.0780,
    "or_r10_context": 0.0921,
    "or_r30_original": 0.1677,
    "or_r30_context": 0.1874,
    "or_r50_original": 0.2303,
    "or_r50_context": 0.2518,
}

VARIANT_TO_CONDITION = {
    "outfitUrlTitle": "original",
    "NewoutfitUrlTitle": "context",
}

def read_csv(path: Path):
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))

def unique_exact(rows):
    seen = set()
    out = []
    for row in rows:
        key = tuple(sorted(row.items()))
        if key not in seen:
            seen.add(key)
            out.append(row)
    return out

def mean(values):
    return statistics.fmean(values)

def sd(values):
    return statistics.stdev(values) if len(values) >= 2 else float("nan")

def status(value, target, mode):
    diff = abs(value - target)
    if round(value, 4) == round(target, 4):
        return "exact"
    tol = 0.0005 if mode == "checkpoint" else 0.005
    return "numerically_close" if diff <= tol else "outside_numeric_tolerance"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cp", required=True, help="results_cp.csv")
    ap.add_argument("--cir", required=True, help="results_cir.csv")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--mode", choices=["checkpoint", "retrain"], default="checkpoint")
    args = ap.parse_args()

    cp_rows = unique_exact(read_csv(Path(args.cp)))
    cir_rows = unique_exact(read_csv(Path(args.cir)))

    cp_rows = [
        r for r in cp_rows
        if r.get("variant") in VARIANT_TO_CONDITION
        and r.get("subset_tag") == "all"
        and int(r["seed"]) in range(1, 6)
    ]
    cir_rows = [
        r for r in cir_rows
        if r.get("variant") in VARIANT_TO_CONDITION
        and r.get("subset_tag") == "all"
        and int(r["seed"]) in range(1, 6)
    ]

    by_cp = {}
    for r in cp_rows:
        key = (VARIANT_TO_CONDITION[r["variant"]], int(r["seed"]))
        by_cp[key] = r

    by_cir = {}
    for r in cir_rows:
        key = (VARIANT_TO_CONDITION[r["variant"]], int(r["seed"]))
        by_cir[key] = r

    expected_keys = {(c, s) for c in ("original", "context") for s in range(1, 6)}
    missing_cp = sorted(expected_keys - set(by_cp))
    missing_cir = sorted(expected_keys - set(by_cir))

    if missing_cp or missing_cir:
        print("[ERROR] Missing required seed-condition rows")
        print("  CP:", missing_cp)
        print("  CIR:", missing_cir)
        sys.exit(2)

    specs = [
        ("cp_auc", "auc", by_cp),
        ("cp_fitb", "fitb_acc", by_cp),
        ("or_r1", "recall_at_1", by_cir),
        ("or_r3", "recall_at_3", by_cir),
        ("or_r5", "recall_at_5", by_cir),
        ("or_r10", "recall_at_10", by_cir),
        ("or_r30", "recall_at_30", by_cir),
        ("or_r50", "recall_at_50", by_cir),
    ]

    summary_rows = []
    all_positive = True
    all_close = True

    for metric, column, source in specs:
        condition_means = {}
        condition_sds = {}
        seed_values = {}
        for condition in ("original", "context"):
            vals = [float(source[(condition, seed)][column]) for seed in range(1, 6)]
            seed_values[condition] = vals
            condition_means[condition] = mean(vals)
            condition_sds[condition] = sd(vals)

        delta = condition_means["context"] - condition_means["original"]
        all_positive = all_positive and delta > 0

        for condition in ("original", "context"):
            key = f"{metric}_{condition}"
            target = PAPER[key]
            st = status(condition_means[condition], target, args.mode)
            all_close = all_close and st in ("exact", "numerically_close")
            summary_rows.append({
                "metric": metric,
                "condition": condition,
                "n_seeds": 5,
                "seeds": "1,2,3,4,5",
                "reproduced_mean": condition_means[condition],
                "reproduced_sd": condition_sds[condition],
                "paper_mean": target,
                "absolute_difference": abs(condition_means[condition] - target),
                "status": st,
                "seed_values": json.dumps(seed_values[condition]),
            })

        summary_rows.append({
            "metric": metric,
            "condition": "context_minus_original",
            "n_seeds": 5,
            "seeds": "1,2,3,4,5",
            "reproduced_mean": delta,
            "reproduced_sd": "",
            "paper_mean": PAPER[f"{metric}_context"] - PAPER[f"{metric}_original"],
            "absolute_difference": abs(delta - (PAPER[f"{metric}_context"] - PAPER[f"{metric}_original"])),
            "status": "positive_direction" if delta > 0 else "direction_not_reproduced",
            "seed_values": json.dumps([
                seed_values["context"][i] - seed_values["original"][i] for i in range(5)
            ]),
        })

    if all_close:
        overall = "numerically_close_or_exact"
    elif args.mode == "retrain" and all_positive:
        overall = "conclusion_level_reproduced"
    else:
        overall = "partial_or_failed_review_required"

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / "main_reproduction_summary.csv"
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        w.writeheader()
        w.writerows(summary_rows)

    result = {
        "mode": args.mode,
        "overall": overall,
        "all_primary_context_deltas_positive": all_positive,
        "all_condition_means_within_numeric_threshold": all_close,
        "cp_input_rows_after_exact_dedup": len(cp_rows),
        "cir_input_rows_after_exact_dedup": len(cir_rows),
    }
    (out_dir / "main_reproduction_acceptance.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(json.dumps(result, ensure_ascii=False, indent=2))
    print("[OK] wrote", out_csv)

if __name__ == "__main__":
    main()
