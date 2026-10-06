#!/usr/bin/env python3
"""Post-training main significance and fair-subset factor-direction audit.

No training, checkpoint loading, CUDA use, or modification of the source run.
Uses 5 paired seed values from the FULL main experiment, NOT the fair
subset's historical T02 significance table. BH family: all 8 prespecified
main CP/OR metrics; two-sided paired t-tests, df=4, alpha=.05.

Also writes thesis Tables 4-11/4-12 in the column layout of the fair-subset
T03/T04 tables (Tables 4-13/4-14) produced by summarize_fair_subset_5seed.py.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics

from scipy.stats import t as student_t

SEEDS = (1, 2, 3, 4, 5)
MAIN_METRICS = ("cp_auc", "cp_fitb", "or_r1", "or_r3", "or_r5", "or_r10", "or_r30", "or_r50")
FACTORS = {"weather": "no_weather", "occasion": "no_occasion", "style": "no_style"}
FAIR_METRICS = (
    ("CP", "AUC", "auc"), ("CP", "FITB Acc", "fitb_acc"),
    ("CIR", "Recall@1", "recall_at_1"), ("CIR", "Recall@3", "recall_at_3"),
    ("CIR", "Recall@5", "recall_at_5"), ("CIR", "Recall@10", "recall_at_10"),
    ("CIR", "Recall@30", "recall_at_30"), ("CIR", "Recall@50", "recall_at_50"),
)
MAIN_TABLE_LABELS = {
    "cp_auc": ("CP", "AUC"), "cp_fitb": ("CP", "FITB Acc"),
    "or_r1": ("OR", "Recall@1"), "or_r3": ("OR", "Recall@3"), "or_r5": ("OR", "Recall@5"),
    "or_r10": ("OR", "Recall@10"), "or_r30": ("OR", "Recall@30"), "or_r50": ("OR", "Recall@50"),
}
# Same columns as the fair-subset T03/T04 tables (thesis Tables 4-13/4-14).
THESIS_TABLE_FIELDS = ("Task", "Metric", "Original (mean±std)", "Context-aware (mean±std)",
                       "Δ", "95% CI of Δ", "BH-adjusted p", "Sig. (BH)", "Cohen’s dz", "n", "Raw p")
# Thesis Table -> output file, relative to the run root.
THESIS_TABLE_FILES = {
    "4-11": "statistics/table_4_11_cp_main.csv",
    "4-12": "statistics/table_4_12_or_main.csv",
    "4-13": "ablation/summary/T03_stage1_cp_original_vs_full.csv",
    "4-14": "ablation/summary/T04_stage1_cir_original_vs_full.csv",
    "4-15": "ablation/summary/T05_stage2_cp_ablation.csv",
    "4-16": "ablation/summary/T06_stage2_cir_ablation.csv",
}


def fail(message: str) -> None:
    raise ValueError(message)


def rows(path: Path) -> list[dict]:
    if not path.is_file():
        fail(f"Missing required source: {path}")
    with path.open(encoding="utf-8-sig", newline="") as stream:
        result = list(csv.DictReader(stream))
    if not result:
        fail(f"Empty CSV: {path}")
    return result


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def finite_float(value, label: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Non-numeric {label}: {value!r}") from exc
    if not math.isfinite(number):
        fail(f"Non-finite {label}: {value!r}")
    return number


def parse_seed_values(row: dict) -> list[float]:
    metric, condition = row["metric"], row["condition"]
    if int(row["n_seeds"]) != 5 or row["seeds"] != "1,2,3,4,5":
        fail(f"Expected seeds 1..5 for {metric}/{condition}")
    values = json.loads(row["seed_values"])
    if not isinstance(values, list) or len(values) != 5:
        fail(f"Expected exactly five seed values for {metric}/{condition}")
    result = [finite_float(x, f"{metric}/{condition}/seed{i}") for i, x in zip(SEEDS, values)]
    if not math.isclose(statistics.mean(result), finite_float(row["reproduced_mean"], "reported mean"), abs_tol=1e-10):
        fail(f"Seed values do not match reported mean for {metric}/{condition}")
    return result


def bh_adjust(p_values: list[float]) -> list[float]:
    """Benjamini-Hochberg step-up adjusted p-values in original row order."""
    n = len(p_values)
    if not n or any(not (0 <= p <= 1 and math.isfinite(p)) for p in p_values):
        fail("BH adjustment requires finite p-values between 0 and 1")
    order = sorted(range(n), key=lambda i: p_values[i])
    result = [0.0] * n
    running = 1.0
    for rank_index in range(n - 1, -1, -1):
        index = order[rank_index]
        rank = rank_index + 1
        running = min(running, p_values[index] * n / rank)
        result[index] = running
    return result


def paired_stats(original: list[float], context: list[float]) -> dict:
    diff = [c - o for o, c in zip(original, context)]
    n = len(diff)
    avg = statistics.mean(diff)
    stdev = statistics.stdev(diff)
    if stdev == 0:
        # Deterministic data are unusual; represent the degenerate case
        # without NaN/Infinity in JSON or CSV.
        p = 1.0 if avg == 0 else 0.0
        tstat = 0.0 if avg == 0 else None
        low = high = avg
        dz = 0.0 if avg == 0 else None
    else:
        standard_error = stdev / math.sqrt(n)
        tstat = avg / standard_error
        p = float(2 * student_t.sf(abs(tstat), n - 1))
        margin = float(student_t.ppf(0.975, n - 1) * standard_error)
        low, high = avg - margin, avg + margin
        dz = avg / stdev
    return {"delta_mean": avg, "delta_sd": stdev, "ci95_low": low,
            "ci95_high": high, "t_statistic": tstat, "p_value": p,
            "cohen_dz": dz, "paired_differences_by_seed": json.dumps(diff)}


def sign(value: float) -> str:
    # Treat sub-picounit subtraction roundoff as exact zero, not a sign flip.
    if abs(value) <= 1e-12:
        return "zero"
    if value > 0:
        return "positive"
    if value < 0:
        return "negative"
    return "zero"


def make_main(path: Path) -> list[dict]:
    source = rows(path)
    keyed = {}
    for row in source:
        key = (row["metric"], row["condition"])
        if key in keyed:
            fail(f"Duplicate main row {key}")
        keyed[key] = row
    expected = {(m, c) for m in MAIN_METRICS for c in ("original", "context", "context_minus_original")}
    if set(keyed) != expected:
        fail(f"Main rows missing={sorted(expected - set(keyed))} unexpected={sorted(set(keyed) - expected)}")
    result = []
    for metric in MAIN_METRICS:
        original = parse_seed_values(keyed[metric, "original"])
        context = parse_seed_values(keyed[metric, "context"])
        derived = parse_seed_values(keyed[metric, "context_minus_original"])
        for seed, (o, c, d) in enumerate(zip(original, context, derived), 1):
            if not math.isclose(c - o, d, abs_tol=1e-10):
                fail(f"Main paired delta mismatch: {metric} seed {seed}")
        stats = paired_stats(original, context)
        result.append({"metric": metric, "task": "CP" if metric.startswith("cp_") else "OR/CIR",
                       "n_seeds": 5, "seeds": "1,2,3,4,5",
                       "original_mean": statistics.mean(original),
                       "context_mean": statistics.mean(context), **stats})
    adjusted = bh_adjust([row["p_value"] for row in result])
    for row, q in zip(result, adjusted):
        row["bh_adjusted_p_8_main"] = q
        row["significant_bh_0_05"] = q < 0.05
    return result


def p_display(p: float) -> str:
    return "< .001" if p < 0.001 else f"{p:.4f}"


def stars(p: float) -> str:
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."


def make_thesis_main_tables(path: Path, main_rows: list[dict]) -> list[dict]:
    """Tables 4-11/4-12 in the column layout and number formats of T03/T04."""
    keyed = {(row["metric"], row["condition"]): row for row in rows(path)}
    result = []
    for row in main_rows:
        metric = row["metric"]
        original_sd = statistics.stdev(parse_seed_values(keyed[metric, "original"]))
        context_sd = statistics.stdev(parse_seed_values(keyed[metric, "context"]))
        task, label = MAIN_TABLE_LABELS[metric]
        dz = row["cohen_dz"]
        result.append({
            "Task": task, "Metric": label,
            "Original (mean±std)": f"{row['original_mean']:.4f} ± {original_sd:.4f}",
            "Context-aware (mean±std)": f"{row['context_mean']:.4f} ± {context_sd:.4f}",
            "Δ": f"{row['delta_mean']:+.4f}",
            "95% CI of Δ": f"[{row['ci95_low']:+.5f}, {row['ci95_high']:+.5f}]",
            "BH-adjusted p": p_display(row["bh_adjusted_p_8_main"]),
            "Sig. (BH)": stars(row["bh_adjusted_p_8_main"]),
            "Cohen’s dz": f"{dz:.3f}" if dz is not None else "n/a",
            "n": row["n_seeds"],
            "Raw p": p_display(row["p_value"]),
        })
    return result


def make_factors(fresh_path: Path, archived_path: Path) -> list[dict]:
    fresh = {}
    for row in rows(fresh_path):
        key = (row["variant"], row["metric_key"])
        if key in fresh:
            fail(f"Duplicate fresh ablation row: {key}")
        if int(row["n_seeds"]) != 5 or row["paired_seeds"] != "1,2,3,4,5":
            fail(f"Missing 5-seed ablation row: {key}")
        fresh[key] = finite_float(row["mean"], str(key))
    expected_fresh = {(v, m) for v in ("original", "context", *FACTORS.values())
                      for _task, _label, m in FAIR_METRICS}
    if set(fresh) != expected_fresh:
        fail(f"Ablation rows missing={sorted(expected_fresh - set(fresh))} unexpected={sorted(set(fresh) - expected_fresh)}")
    archived = {}
    for row in rows(archived_path):
        key = (row["variant"], row["metric"])
        if key in archived:
            fail(f"Duplicate archived T01 comparator: {key}")
        archived[key] = finite_float(row["archived_mean"], str(key))
    result = []
    for factor, excluded in FACTORS.items():
        for task, metric_label, metric_key in FAIR_METRICS:
            full = fresh["context", metric_key]
            without = fresh[excluded, metric_key]
            delta = full - without  # Explicit sign convention, not NoFactor - Full.
            archived_delta = None
            if ("context", metric_label) in archived and (excluded, metric_label) in archived:
                archived_delta = archived["context", metric_label] - archived[excluded, metric_label]
            result.append({"factor": factor, "task": task, "metric": metric_label,
                           "delta_definition": "full_context_minus_without_factor",
                           "full_fresh_mean": full, "without_factor_fresh_mean": without,
                           "fresh_delta": delta, "fresh_sign": sign(delta),
                           "archived_T01_delta": archived_delta,
                           "archived_T01_sign": sign(archived_delta) if archived_delta is not None else "not_available",
                           "sign_comparison_with_archived_T01":
                           ("not_available" if archived_delta is None else
                            "same" if sign(delta) == sign(archived_delta) else "different"),
                           "scope_note": "reconstructed fair subset vs archived T01 aggregate; not identical memberwise historical subset"})
    return result


def write_csv(path: Path, data: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(data[0]))
        writer.writeheader()
        writer.writerows(data)


def write_thesis_table(path: Path, data: list[dict]) -> None:
    """Same encoding and line endings as the T03/T04 thesis-format tables."""
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=THESIS_TABLE_FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(data)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    args = parser.parse_args()
    run = args.run_root.expanduser().resolve()
    out = args.out_dir.expanduser().resolve()
    main_csv = run / "main/summary/main_reproduction_summary.csv"
    fresh_csv = run / "ablation/summary/fresh_5variant_8metric_summary.csv"
    archived_csv = run / "ablation/summary/archived_vs_fresh_T01_numeric_comparison.csv"
    if out == run or out in (main_csv.parent, fresh_csv.parent):
        parser.error("Choose a dedicated --out-dir, not a source/result-summary directory")
    main_rows = make_main(main_csv)
    thesis_rows = make_thesis_main_tables(main_csv, main_rows)
    factor_rows = make_factors(fresh_csv, archived_csv)
    out.mkdir(parents=True, exist_ok=True)
    write_csv(out / "main_paired_bh_8metrics.csv", main_rows)
    write_csv(out / "factor_direction_candidate_vs_archived_T01.csv", factor_rows)
    write_thesis_table(out / "table_4_11_cp_main.csv", [r for r in thesis_rows if r["Task"] == "CP"])
    write_thesis_table(out / "table_4_12_or_main.csv", [r for r in thesis_rows if r["Task"] == "OR"])
    evidence = {"classification": "post-training statistical analysis; no models trained",
                "main_scope": "full-data main CP/OR Original vs Context seeds 1..5",
                "test": "two-sided paired t-test on context-original per seed, df=4",
                "confidence_interval": "95% two-sided t confidence interval on mean paired difference",
                "bh_family": list(MAIN_METRICS), "alpha": 0.05,
                "ablation_scope": "fresh 5-seed reconstructed fair subset",
                "ablation_delta_definition": "full_context_minus_without_factor",
                "ablation_reference": "archived T01 aggregate; not paper Tables 4-13 through 4-16 cellwise proof",
                "thesis_tables": THESIS_TABLE_FILES,
                "thesis_tables_note": "Paths are relative to the run root. Tables 4-13/4-14 BH-adjust the five reported fair-subset metrics as one family (archived convention), computed by summarize_fair_subset_5seed.py.",
                "input_sha256": {str(p.relative_to(run)): sha256(p) for p in (main_csv, fresh_csv, archived_csv)},
                "source_run_git_commit": (run / "git_commit.txt").read_text(encoding="utf-8").strip() if (run / "git_commit.txt").is_file() else None,
                "analysis_script_sha256": sha256(Path(__file__).resolve()),
                "full_main_R1_R3": {r["metric"]: {"delta_mean": r["delta_mean"],
                    "p_value": r["p_value"], "bh_adjusted_p_8_main": r["bh_adjusted_p_8_main"],
                    "significant_bh_0_05": r["significant_bh_0_05"]}
                    for r in main_rows if r["metric"] in ("or_r1", "or_r3")},
                "note": "Mean-positive is not synonymous with statistically significant; no source-exact historical ablation claim."}
    (out / "statistical_evidence.json").write_text(json.dumps(evidence, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    markdown = ["# Post-training statistical evidence", "",
                "Main: original vs context, full-data five paired seeds, two-sided paired t-test;",
                "BH adjusted over the eight prespecified main metrics (2 CP + 6 OR).",
                "95% CI covers the paired mean difference. These are new calculations, not archived thesis p-values.", "",
                "| Main metric | Mean context−original | raw p | BH p (8 tests) | BH significance at .05 |",
                "|---|---:|---:|---:|---|",]
    for row in main_rows:
        markdown.append(f"| {row['metric']} | {row['delta_mean']:+.8f} | {row['p_value']:.6g} | {row['bh_adjusted_p_8_main']:.6g} | {row['significant_bh_0_05']} |")
    markdown += ["", "## Fair-subset ablation factors", "",
                 "Definition: full context − without factor. Negative means removing the factor had a higher mean for that metric.",
                 "Archived comparison uses T01 aggregates, NOT proven-identical historical memberwise IDs.",
                 "Signs treat absolute mean differences <= 1e-12 as zero (floating point roundoff only).", "",
                 "| Factor | Metric | Fresh delta | Fresh sign | Archived T01 delta | Archived sign |",
                 "|---|---|---:|---|---:|---|",]
    for row in factor_rows:
        a = row["archived_T01_delta"]
        markdown.append(f"| {row['factor']} | {row['metric']} | {row['fresh_delta']:+.8f} | {row['fresh_sign']} | {a if a is not None else 'N/A'} | {row['archived_T01_sign']} |")
    (out / "statistical_evidence.md").write_text("\n".join(markdown) + "\n", encoding="utf-8")
    print(f"[PASS] eight full-data main paired tests; 24 fair-subset factor directions; Tables 4-11/4-12 -> {out}")
    for metric in ("or_r1", "or_r3"):
        row = next(r for r in main_rows if r["metric"] == metric)
        print(f"[MAIN] {metric}: delta={row['delta_mean']:+.8f}, p={row['p_value']:.8g}, "
              f"BH(8)={row['bh_adjusted_p_8_main']:.8g}, significant={row['significant_bh_0_05']}")


if __name__ == "__main__":
    main()
