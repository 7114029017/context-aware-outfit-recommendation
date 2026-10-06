#!/usr/bin/env python3
"""Recompute thesis ablation five-seed statistics/tables from 25 fresh subset runs.

This script NEVER trains a model and NEVER overwrites archived thesis tables.
It consumes only verified fresh fair-subset ablation outputs supplied by --input-dir,
cross-checks the 25-unit batch report, then writes a separate fresh summary dir.

Scope caveat: the runs use the reproducibly reconstructed fair subset
and the standardized decoder implementation. Numeric agreement is evidence,
not proof of source-exact reproduction of the historical 2025 experiment.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
REPRO_ROOT = ROOT / "reproduction"
REPRO_SCRIPTS = REPRO_ROOT / "scripts"
VARIANTS = ("original", "context", "no_weather", "no_occasion", "no_style")
SEEDS = (1, 2, 3, 4, 5)
EXPECTED_SCOPE = 3432
CORE_METRICS = (
    ("CP", "AUC", "auc"),
    ("CP", "FITB Acc", "fitb_acc"),
    ("CIR", "Recall@10", "recall_at_10"),
    ("CIR", "Recall@30", "recall_at_30"),
    ("CIR", "Recall@50", "recall_at_50"),
)
ALL_METRICS = (
    ("CP", "AUC", "auc"),
    ("CP", "FITB Acc", "fitb_acc"),
    ("CIR", "Recall@1", "recall_at_1"),
    ("CIR", "Recall@3", "recall_at_3"),
    ("CIR", "Recall@5", "recall_at_5"),
    ("CIR", "Recall@10", "recall_at_10"),
    ("CIR", "Recall@30", "recall_at_30"),
    ("CIR", "Recall@50", "recall_at_50"),
)
ARCHIVED_TABLE_DIR = ROOT / (
    "03_實驗與結果_experiments_results/03_主推薦任務結果/"
    "圖表_figures_tables/tables"
)

METHOD_ROWS = {
    "original": (
        "Original description",
        "Original baseline",
        "使用原始 Polyvore title，未加入天氣、場合與風格條件。",
        "作為未建構環境語意的原始基準。",
    ),
    "context": (
        "Full contextual rewrite",
        "Proposed full method",
        "同時加入天氣、場合與風格條件，形成完整環境語意描述。",
        "檢驗完整環境語意建構後的整體提升。",
    ),
    "no_weather": (
        "Simplified w/o weather",
        "Simplified condition baseline",
        "保留場合與風格，但移除天氣條件。",
        "排除只有 prompt 變長造成提升；檢查天氣語意是否必要。",
    ),
    "no_occasion": (
        "Simplified w/o occasion",
        "Simplified condition baseline",
        "保留天氣與風格，但移除場合條件。",
        "檢查場合語意是否對推薦任務有額外貢獻。",
    ),
    "no_style": (
        "Simplified w/o style",
        "Simplified condition baseline",
        "保留天氣與場合，但移除風格條件。",
        "檢查風格語意是否對推薦任務有額外貢獻。",
    ),
}
FACTOR_TO_VARIANT = {
    "weather": "no_weather",
    "occasion": "no_occasion",
    "style": "no_style",
}


def abort(msg: str) -> None:
    raise SystemExit("[ABORT] " + msg)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def portable_path(path: Path) -> str:
    q = path.expanduser().resolve()
    try:
        return str(q.relative_to(ROOT))
    except ValueError:
        return q.name


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".writing")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                   encoding="utf-8")
    tmp.replace(path)


def write_csv(path: Path, fieldnames, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n")
        w.writeheader()
        w.writerows(rows)


def one_csv(path: Path) -> dict:
    with path.open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    if len(rows) != 1:
        abort(f"Expected exactly one metric row: {path}; found {len(rows)}")
    return rows[0]


def sample_stats(values) -> tuple[float, float]:
    a = np.asarray(values, dtype=float)
    if len(a) != 5 or not np.all(np.isfinite(a)):
        abort(f"Need five finite seed values, got {a}")
    return float(a.mean()), float(a.std(ddof=1))


def paired_test(a, b) -> tuple[float, float, float]:
    """Return mean(a-b), two-sided paired t p, Cohen's dz for a-b."""
    x = np.asarray(a, dtype=float)
    y = np.asarray(b, dtype=float)
    d = x - y
    if len(d) != 5 or not np.all(np.isfinite(d)):
        abort("Paired test requires five finite paired values")
    mean_d = float(d.mean())
    sd_d = float(d.std(ddof=1))
    if sd_d == 0:
        dz = math.inf if mean_d > 0 else (-math.inf if mean_d < 0 else 0.0)
    else:
        dz = mean_d / sd_d
    p = float(stats.ttest_rel(x, y).pvalue)
    return mean_d, p, float(dz)


def signif(p: float) -> str:
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "n.s."


def p_display(p: float) -> str:
    return "< .001" if p < 0.001 else f"{p:.4f}"


def bh_adjust(pvals: list[float]) -> list[float]:
    """Benjamini-Hochberg adjusted p-values, matching archived T03/T04 convention."""
    p = np.asarray(pvals, dtype=float)
    n = len(p)
    order = np.argsort(p)
    ranked = p[order]
    adjusted_ranked = np.empty(n, dtype=float)
    running = 1.0
    for i in range(n - 1, -1, -1):
        rank = i + 1
        running = min(running, ranked[i] * n / rank)
        adjusted_ranked[i] = min(1.0, running)
    adjusted = np.empty(n, dtype=float)
    adjusted[order] = adjusted_ranked
    return adjusted.tolist()


def paired_ci95(target, baseline) -> tuple[float, float]:
    """Unadjusted 95% Student-t CI for target-baseline paired seed differences."""
    x = np.asarray(target, dtype=float)
    y = np.asarray(baseline, dtype=float)
    d = x - y
    if len(d) != 5 or not np.all(np.isfinite(d)):
        abort("CI requires five finite paired values")
    mean_d = float(d.mean())
    sem = float(stats.sem(d))
    half = float(stats.t.ppf(0.975, df=len(d)-1) * sem)
    return mean_d - half, mean_d + half


def pm(mean: float, std: float) -> str:
    return f"{mean:.4f} ± {std:.4f}"


def signed(x: float) -> str:
    return f"{x:+.4f}"


def query_membership_sha(detail_path: Path) -> tuple[int, str]:
    seen = set()
    with detail_path.open(encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            key = (str(row["set_id"]), str(row["target_item_id"]), str(row["target_item_fg"]))
            if key in seen:
                abort(f"Duplicate CIR evaluation question in {detail_path}: {key}")
            seen.add(key)
    canonical = "\n".join("\t".join(v) for v in sorted(seen))
    return len(seen), hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def metric_values(data, variant: str, key: str) -> list[float]:
    return [float(data[(variant, seed)][key]) for seed in SEEDS]


def parse_pm(value: str) -> tuple[float, float]:
    m = re.fullmatch(r"\s*([+-]?\d+(?:\.\d+)?)\s*±\s*([+-]?\d+(?:\.\d+)?)\s*", value)
    if not m:
        abort(f"Cannot parse archived mean±std value: {value!r}")
    return float(m.group(1)), float(m.group(2))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input-dir", default="reproduction/runs/fair_subset_batch")
    ap.add_argument("--batch-report",
                    default="reproduction/runs/fair_subset_batch_status.json")
    ap.add_argument("--out-dir", default="reproduction/runs/fair_subset_summary")
    args = ap.parse_args()

    input_dir = (ROOT / args.input_dir).resolve() if not Path(args.input_dir).is_absolute() else Path(args.input_dir)
    batch_path = (ROOT / args.batch_report).resolve() if not Path(args.batch_report).is_absolute() else Path(args.batch_report)
    out_dir = (ROOT / args.out_dir).resolve() if not Path(args.out_dir).is_absolute() else Path(args.out_dir)

    if not batch_path.is_file():
        abort(f"Missing 25-unit batch report: {batch_path}")
    batch = read_json(batch_path)
    comp = batch.get("comparability") or {}
    if not (
        batch.get("status") == "PASSED_CANDIDATE_SOURCE"
        and batch.get("completed_units") == 25
        and batch.get("total_units") == 25
        and comp.get("verified_complete") == 25
        and comp.get("all_completed_shared_membership") is True
        and comp.get("evaluation_query_membership_sha256")
    ):
        abort("Batch report is not a verified 25/25 common-membership result")
    expected_query_sha = comp["evaluation_query_membership_sha256"]

    data = {}
    provenance = None
    combined_rows = []
    for seed in SEEDS:
        for variant in VARIANTS:
            d = input_dir / f"{variant}_seed{seed}"
            mf_path = d / "fair_subset_run_manifest.json"
            cp_path = d / "results_cp_fresh_subset.csv"
            cir_path = d / "results_cir_fresh_subset.csv"
            detail_path = d / "detail_cir_fresh_subset.csv"
            for p in (mf_path, cp_path, cir_path, detail_path):
                if not p.is_file():
                    abort(f"Missing fresh artifact: {p}")
            mf = read_json(mf_path)
            ident = mf.get("identity") or {}
            ev = mf.get("evaluation") or {}
            if (
                mf.get("status") != "passed_candidate_source"
                or ident.get("variant") != variant
                or ident.get("seed") != seed
                or ev.get("cir_evaluable") != EXPECTED_SCOPE
            ):
                abort(f"Manifest identity/status/scope mismatch: {mf_path}")
            current_prov = {
                "candidate_id_sha256": ident.get("candidate_id_sha256"),
                "candidate_reconstruction_manifest_sha256": ident.get("candidate_reconstruction_manifest_sha256"),
                "cir_3432_scope_report_sha256": ident.get("cir_3432_scope_report_sha256"),
                "source_py_sha256": ident.get("source_py_sha256"),
                "candidate_source_label": ident.get("candidate_source_label"),
                "decoder_label": ident.get("decoder_label"),
                "training_semantics": ident.get("training_semantics"),
            }
            if provenance is None:
                provenance = current_prov
            elif current_prov != provenance:
                abort(f"Cross-run provenance differs: {variant}/seed{seed}")

            for p in (cp_path, cir_path, detail_path):
                expected = ev.get("sha256", {}).get(p.name)
                if not expected or sha256(p) != expected:
                    abort(f"Evaluation file hash differs from manifest: {p}")
            n_questions, qsha = query_membership_sha(detail_path)
            if n_questions != EXPECTED_SCOPE or qsha != expected_query_sha:
                abort(f"CIR question membership differs: {variant}/seed{seed}")

            cp = one_csv(cp_path)
            cir = one_csv(cir_path)
            if int(cp["seed"]) != seed or int(cir["seed"]) != seed:
                abort(f"Seed mismatch in result CSV: {variant}/seed{seed}")
            if cp.get("subset_tag") != "subset" or cir.get("subset_tag") != "subset":
                abort(f"Result not labeled subset: {variant}/seed{seed}")
            row = {
                "auc": float(cp["auc"]),
                "fitb_acc": float(cp["fitb_acc"]),
                "recall_at_1": float(cir["recall_at_1"]),
                "recall_at_3": float(cir["recall_at_3"]),
                "recall_at_5": float(cir["recall_at_5"]),
                "recall_at_10": float(cir["recall_at_10"]),
                "recall_at_30": float(cir["recall_at_30"]),
                "recall_at_50": float(cir["recall_at_50"]),
            }
            for task, _label, key in ALL_METRICS:
                recorded = float(ev["cp" if task == "CP" else "cir"][key])
                if abs(row[key] - recorded) > 1e-12:
                    abort(f"CSV vs manifest metric mismatch: {variant}/seed{seed}/{key}")
            data[(variant, seed)] = row
            combined_rows.append({"variant": variant, "seed": seed, **row})

    # Exact 25 units and common seed coverage.
    if len(data) != 25:
        abort(f"Expected 25 units, got {len(data)}")

    out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(
        out_dir / "fresh_25unit_metrics.csv",
        ["variant", "seed"] + [m[2] for m in ALL_METRICS],
        combined_rows,
    )

    metric_summary = {}
    all_summary_rows = []
    for variant in VARIANTS:
        metric_summary[variant] = {}
        for task, label, key in ALL_METRICS:
            mean, std = sample_stats(metric_values(data, variant, key))
            metric_summary[variant][key] = {"task": task, "metric": label, "mean": mean, "std": std}
            all_summary_rows.append({
                "variant": variant, "task": task, "metric": label, "metric_key": key,
                "mean": mean, "std": std, "n_seeds": 5, "paired_seeds": "1,2,3,4,5",
            })
    write_csv(
        out_dir / "fresh_5variant_8metric_summary.csv",
        ["variant", "task", "metric", "metric_key", "mean", "std", "n_seeds", "paired_seeds"],
        all_summary_rows,
    )

    # T01
    t01_fields = [
        "Method", "Type", "Condition design", "Controlled comparison purpose",
        "AUC", "FITB Acc", "Recall@10", "Recall@30", "Recall@50",
        "Δ AUC vs Original", "Δ FITB Acc vs Original", "Δ Recall@10 vs Original",
        "Δ Recall@30 vs Original", "Δ Recall@50 vs Original",
        "Δ AUC vs Full", "Δ FITB Acc vs Full", "Δ Recall@10 vs Full",
        "Δ Recall@30 vs Full", "Δ Recall@50 vs Full",
    ]
    t01_rows = []
    key_order = ["auc", "fitb_acc", "recall_at_10", "recall_at_30", "recall_at_50"]
    display_order = ["AUC", "FITB Acc", "Recall@10", "Recall@30", "Recall@50"]
    for variant in VARIANTS:
        method, type_, design, purpose = METHOD_ROWS[variant]
        r = {
            "Method": method, "Type": type_, "Condition design": design,
            "Controlled comparison purpose": purpose,
        }
        for disp, key in zip(display_order, key_order):
            s = metric_summary[variant][key]
            r[disp] = pm(s["mean"], s["std"])
            r[f"Δ {disp} vs Original"] = signed(s["mean"] - metric_summary["original"][key]["mean"])
            r[f"Δ {disp} vs Full"] = signed(s["mean"] - metric_summary["context"][key]["mean"])
        t01_rows.append(r)
    write_csv(out_dir / "T01_main_overall_baseline_and_ablation.csv", t01_fields, t01_rows)

    # T02: full vs original + each factor-removed version.
    t02_rows = []
    comparison_targets = [
        ("original", "Original description"),
        ("no_weather", "Simplified w/o weather"),
        ("no_occasion", "Simplified w/o occasion"),
        ("no_style", "Simplified w/o style"),
    ]
    for baseline_variant, baseline_name in comparison_targets:
        for task, label, key in CORE_METRICS:
            full = metric_values(data, "context", key)
            base = metric_values(data, baseline_variant, key)
            mean_diff, p, _dz = paired_test(full, base)
            t02_rows.append({
                "Comparison": f"Full contextual rewrite vs {baseline_name}",
                "Task": task, "Metric": label, "Mean difference": signed(mean_diff),
                "p-value": p_display(p), "Paired seeds": "1,2,3,4,5",
            })
    write_csv(
        out_dir / "T02_main_significance_tests.csv",
        ["Comparison", "Task", "Metric", "Mean difference", "p-value", "Paired seeds"],
        t02_rows,
    )

    # T03/T04: thesis Tables 4-13/4-14 shape.
    # Published archived convention: two-sided paired t tests over five seeds,
    # BH correction across the five CP+OR core metrics, unadjusted 95% paired CI.
    stage1_specs = [
        ("CP", "AUC", "auc"),
        ("CP", "FITB Acc", "fitb_acc"),
        ("CIR", "Recall@10", "recall_at_10"),
        ("CIR", "Recall@30", "recall_at_30"),
        ("CIR", "Recall@50", "recall_at_50"),
    ]
    stage1_rows = []
    stage1_raw_p = []
    for task, label, key in stage1_specs:
        original_values = metric_values(data, "original", key)
        context_values = metric_values(data, "context", key)
        original_mean, original_std = sample_stats(original_values)
        context_mean, context_std = sample_stats(context_values)
        delta, p_raw, dz = paired_test(context_values, original_values)
        ci_low, ci_high = paired_ci95(context_values, original_values)
        stage1_rows.append({
            "Task": task,
            "Metric": label,
            "Original (mean±std)": pm(original_mean, original_std),
            "Context-aware (mean±std)": pm(context_mean, context_std),
            "Δ": signed(delta),
            "95% CI of Δ": f"[{ci_low:+.5f}, {ci_high:+.5f}]",
            "BH-adjusted p": None,
            "Sig. (BH)": None,
            "Cohen’s dz": f"{dz:.3f}",
            "n": 5,
            "Raw p": p_display(p_raw),
        })
        stage1_raw_p.append(p_raw)
    for row, p_bh in zip(stage1_rows, bh_adjust(stage1_raw_p)):
        row["BH-adjusted p"] = p_display(float(p_bh))
        row["Sig. (BH)"] = signif(float(p_bh))

    stage1_fields = [
        "Task", "Metric", "Original (mean±std)", "Context-aware (mean±std)",
        "Δ", "95% CI of Δ", "BH-adjusted p", "Sig. (BH)", "Cohen’s dz", "n", "Raw p",
    ]
    write_csv(
        out_dir / "T03_stage1_cp_original_vs_full.csv",
        stage1_fields,
        [r for r in stage1_rows if r["Task"] == "CP"],
    )
    write_csv(
        out_dir / "T04_stage1_cir_original_vs_full.csv",
        stage1_fields,
        [r for r in stage1_rows if r["Task"] == "CIR"],
    )

    # T05/T06
    def ablation_table(task: str, metrics, filename: str):
        rows = []
        for label, key in metrics:
            full = metric_summary["context"][key]
            nw = metric_summary["no_weather"][key]
            no = metric_summary["no_occasion"][key]
            ns = metric_summary["no_style"][key]
            rows.append({
                "Metric": label,
                "Proposed (mean±std)": pm(full["mean"], full["std"]),
                "No-weather (mean±std)": pm(nw["mean"], nw["std"]),
                "Δ_weather": signed(nw["mean"] - full["mean"]),
                "No-occasion (mean±std)": pm(no["mean"], no["std"]),
                "Δ_occasion": signed(no["mean"] - full["mean"]),
                "No-style (mean±std)": pm(ns["mean"], ns["std"]),
                "Δ_style": signed(ns["mean"] - full["mean"]),
            })
        write_csv(
            out_dir / filename,
            ["Metric", "Proposed (mean±std)", "No-weather (mean±std)", "Δ_weather",
             "No-occasion (mean±std)", "Δ_occasion", "No-style (mean±std)", "Δ_style"],
            rows,
        )

    ablation_table("CP", [("AUC", "auc"), ("FITB Acc", "fitb_acc")], "T05_stage2_cp_ablation.csv")
    ablation_table(
        "CIR",
        [("Recall@10", "recall_at_10"), ("Recall@30", "recall_at_30"), ("Recall@50", "recall_at_50")],
        "T06_stage2_cir_ablation.csv",
    )

    # T07
    t07_rows = []
    for task, label, key in CORE_METRICS:
        full_mean = metric_summary["context"][key]["mean"]
        losses = {
            factor: full_mean - metric_summary[variant][key]["mean"]
            for factor, variant in FACTOR_TO_VARIANT.items()
        }
        denom = sum(losses.values())
        ratio_defined = all(v >= 0 for v in losses.values()) and denom > 0
        t07_rows.append({
            "Task": task, "Metric": label,
            "weather_loss": losses["weather"],
            "weather_ratio": losses["weather"] / denom if ratio_defined else "",
            "occasion_loss": losses["occasion"],
            "occasion_ratio": losses["occasion"] / denom if ratio_defined else "",
            "style_loss": losses["style"],
            "style_ratio": losses["style"] / denom if ratio_defined else "",
        })
    write_csv(
        out_dir / "T07_stage2_factor_contribution_ratio.csv",
        ["Task", "Metric", "weather_loss", "weather_ratio", "occasion_loss", "occasion_ratio",
         "style_loss", "style_ratio"],
        t07_rows,
    )

    # T10/T11
    def detail_rows(task: str, metrics):
        rows = []
        for label, key in metrics:
            full_values = metric_values(data, "context", key)
            full_mean, full_std = sample_stats(full_values)
            for factor, variant in FACTOR_TO_VARIANT.items():
                no_values = metric_values(data, variant, key)
                no_mean, no_std = sample_stats(no_values)
                delta, p, dz = paired_test(no_values, full_values)  # historical sign: NoFactor - Proposed
                ci_low, ci_high = paired_ci95(no_values, full_values)
                rows.append({
                    "Task": task, "Metric": label, "Factor": factor,
                    "Proposed_mean": full_mean, "Proposed_std": full_std,
                    "NoFactor_mean": no_mean, "NoFactor_std": no_std,
                    "Delta_mean_NoFactor_minus_Proposed": delta,
                    "ci95_low": ci_low, "ci95_high": ci_high,
                    "p_raw": p, "signif": signif(p), "cohens_dz": dz,
                    "paired_seeds": "1,2,3,4,5",
                })
        return rows

    t10 = detail_rows("CP", [("AUC", "auc"), ("FITB Acc", "fitb_acc")])
    t11 = detail_rows(
        "CIR",
        [("Recall@10", "recall_at_10"), ("Recall@30", "recall_at_30"), ("Recall@50", "recall_at_50")],
    )
    detail_fields = [
        "Task", "Metric", "Factor", "Proposed_mean", "Proposed_std",
        "NoFactor_mean", "NoFactor_std", "Delta_mean_NoFactor_minus_Proposed",
        "ci95_low", "ci95_high", "p_raw", "signif", "cohens_dz", "paired_seeds",
    ]
    write_csv(out_dir / "T10_stage2_cp_seed_detail.csv", detail_fields, t10)
    write_csv(out_dir / "T11_stage2_cir_seed_detail.csv", detail_fields, t11)

    # Numeric comparison against archived T01. No pass/fail tolerance is invented.
    archived_t01 = ARCHIVED_TABLE_DIR / "T01_main_overall_baseline_and_ablation.csv"
    comparison_rows = []
    if archived_t01.is_file():
        with archived_t01.open(encoding="utf-8-sig", newline="") as f:
            archived_rows = {row["Method"]: row for row in csv.DictReader(f)}
        for variant in VARIANTS:
            method = METHOD_ROWS[variant][0]
            if method not in archived_rows:
                abort(f"Archived T01 is missing expected method: {method}")
            for disp, key in zip(display_order, key_order):
                archived_mean, archived_std = parse_pm(archived_rows[method][disp])
                fresh = metric_summary[variant][key]
                comparison_rows.append({
                    "variant": variant, "method": method, "metric": disp,
                    "archived_mean": archived_mean, "fresh_mean": fresh["mean"],
                    "fresh_minus_archived_mean": fresh["mean"] - archived_mean,
                    "abs_mean_difference": abs(fresh["mean"] - archived_mean),
                    "archived_std": archived_std, "fresh_std": fresh["std"],
                    "fresh_minus_archived_std": fresh["std"] - archived_std,
                    "interpretation": "numeric comparison only; reconstructed subset and standardized decoder prevent a source-exact claim",
                })
        write_csv(
            out_dir / "archived_vs_fresh_T01_numeric_comparison.csv",
            ["variant", "method", "metric", "archived_mean", "fresh_mean",
             "fresh_minus_archived_mean", "abs_mean_difference", "archived_std", "fresh_std",
             "fresh_minus_archived_std", "interpretation"],
            comparison_rows,
        )

    summary = {
        "classification": "fresh five-seed ablation on the reconstructed fair subset; not a historical source-exact certificate",
        "source_input_dir": portable_path(input_dir),
        "batch_report": portable_path(batch_path),
        "batch_report_sha256": sha256(batch_path),
        "batch_status": batch["status"],
        "units_verified": 25,
        "seeds": list(SEEDS),
        "variants": list(VARIANTS),
        "cir_evaluable_per_unit": EXPECTED_SCOPE,
        "evaluation_query_membership_sha256": expected_query_sha,
        "provenance": provenance,
        "metrics": metric_summary,
        "outputs": [
            "fresh_25unit_metrics.csv",
            "fresh_5variant_8metric_summary.csv",
            "T01_main_overall_baseline_and_ablation.csv",
            "T02_main_significance_tests.csv",
            "T03_stage1_cp_original_vs_full.csv",
            "T04_stage1_cir_original_vs_full.csv",
            "T05_stage2_cp_ablation.csv",
            "T06_stage2_cir_ablation.csv",
            "T07_stage2_factor_contribution_ratio.csv",
            "T10_stage2_cp_seed_detail.csv",
            "T11_stage2_cir_seed_detail.csv",
            "archived_vs_fresh_T01_numeric_comparison.csv",
        ],
        "historical_comparison": {
            "archived_T01": portable_path(archived_t01),
            "rows_compared": len(comparison_rows),
            "max_abs_mean_difference": (
                max((r["abs_mean_difference"] for r in comparison_rows), default=None)
            ),
            "note": "No acceptance tolerance is imposed here; report numeric differences with provenance caveats.",
        },
        "statistical_method": {
            "mean_std": "five seeds; sample standard deviation ddof=1",
            "paired_test": "two-sided scipy.stats.ttest_rel on seed-paired values",
            "T03_T04_multiple_testing": "Benjamini-Hochberg correction across the five CP+OR core metrics, matching archived published table convention",
            "T03_T04_ci": "unadjusted two-sided 95% Student-t CI on paired seed differences",
            "effect_size": "Cohen's dz = mean(paired differences) / sample SD(paired differences)",
            "T10_T11_difference_sign": "NoFactor - Proposed, matching archived table convention",
            "T10_T11_ci": "unadjusted two-sided 95% Student-t CI on paired NoFactor - Proposed differences; p_raw and signif are not multiplicity-adjusted, as in the archived tables",
            "T02_difference_sign": "Full - comparison baseline",
            "significance_stars": "*** p<.001; ** p<.01; * p<.05; otherwise n.s.",
        },
        "limits": [
            "Historical 2025 fair-subset memberwise ID source/hash has not been recovered.",
            "Training uses the standardized decoder implementation (torch.nn.TransformerDecoderLayer); the historical DecoderLayerWithCrossAttn definition was not preserved.",
            "Therefore the fresh tables reproduce the ablation on the reconstructed fair subset; they are not proof of byte/source-exact historical reproduction.",
        ],
    }
    atomic_json(out_dir / "fresh_fair_subset_5seed_summary.json", summary)

    print("[PASSED] recomputed fresh five-seed fair-subset ablation statistics")
    print("  25/25 units; common CIR queries:", EXPECTED_SCOPE, expected_query_sha)
    print("  output:", out_dir)
    if comparison_rows:
        print("  archived T01 max |fresh mean - archived mean|:",
              summary["historical_comparison"]["max_abs_mean_difference"])


if __name__ == "__main__":
    main()
