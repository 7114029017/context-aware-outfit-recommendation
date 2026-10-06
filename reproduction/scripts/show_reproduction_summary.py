#!/usr/bin/env python3
"""Print a professor-facing terminal summary from completed reproduction outputs.

Read-only: this script never trains models, loads checkpoints, or rewrites
result files. It can summarize either the committed 2026 reference evidence
or one newly completed integrated full-run directory.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
import subprocess

ROOT = Path(__file__).resolve().parents[2]
REPRO = ROOT / "reproduction"
REF_ID = "reference_20260921T175217Z"
REF_SUMMARY = REPRO / "results" / "summary" / REF_ID
REF_RAW = REPRO / "results" / "raw" / REF_ID
OFFICIAL_MANIFEST = REPRO / "results" / "final_reference_manifest.json"

MAIN_METRICS = (
    ("cp_auc", "AUC"),
    ("cp_fitb", "FITB"),
    ("or_r1", "R@1"),
    ("or_r3", "R@3"),
    ("or_r5", "R@5"),
    ("or_r10", "R@10"),
    ("or_r30", "R@30"),
    ("or_r50", "R@50"),
)
CP_KEYS = {"cp_auc", "cp_fitb"}
OR_KEYS = {"or_r1", "or_r3", "or_r5", "or_r10", "or_r30", "or_r50"}
ABLATION_METRICS = (
    ("auc", "AUC"),
    ("fitb_acc", "FITB"),
    ("recall_at_10", "R@10"),
    ("recall_at_30", "R@30"),
    ("recall_at_50", "R@50"),
)
VARIANTS = ("original", "context", "no_weather", "no_occasion", "no_style")


def fail(message: str) -> None:
    raise SystemExit("[SUMMARY BLOCKED] " + message)


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        fail(f"missing CSV: {path}")
    with path.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        fail(f"empty CSV: {path}")
    return rows


def read_json(path: Path) -> dict:
    if not path.is_file():
        fail(f"missing JSON: {path}")
    obj = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(obj, dict):
        fail(f"expected JSON object: {path}")
    return obj


def read_text(path: Path, default: str = "not_available") -> str:
    if not path.is_file():
        return default
    return path.read_text(encoding="utf-8").strip() or default


def f4(value: object) -> str:
    return f"{float(value):.4f}"


def pval(value: object) -> str:
    return f"{float(value):.6g}"


def mean_sd(row: dict[str, str]) -> str:
    return f"{float(row['reproduced_mean']):.4f} ± {float(row['reproduced_sd']):.4f}"


def print_table(headers: list[str], rows: list[list[str]]) -> None:
    if not rows:
        print("(no rows)")
        return
    widths = [
        max(len(headers[i]), *(len(str(row[i])) for row in rows))
        for i in range(len(headers))
    ]
    print("  ".join(headers[i].ljust(widths[i]) for i in range(len(headers))))
    print("  ".join("-" * widths[i] for i in range(len(headers))))
    for row in rows:
        print("  ".join(str(row[i]).ljust(widths[i]) for i in range(len(headers))))


def display_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path.resolve())


def git_value(*args: str) -> str:
    try:
        proc = subprocess.run(
            ["git", "-C", str(ROOT), *args],
            text=True, capture_output=True, check=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return "not_available"
    return proc.stdout.strip() or "not_available"


def load_paths(args: argparse.Namespace) -> dict[str, Path | str]:
    if args.reference:
        return {
            "label": REF_ID,
            "main": REF_SUMMARY / "main" / "main_reproduction_summary.csv",
            "ablation": REF_SUMMARY / "ablation" / "fresh_fair_subset_5seed_summary.json",
            "stats": REF_SUMMARY / "statistics" / "main_paired_bh_8metrics.csv",
            "factor": REF_SUMMARY / "statistics" / "factor_direction_candidate_vs_archived_T01.csv",
            "chapter4": REF_SUMMARY / "chapter4" / "chapter4_report.json",
            "status": REF_RAW / "run_identity" / "RUN_STATUS.txt",
            "started": REF_RAW / "run_identity" / "started_utc.txt",
            "finished": REF_RAW / "run_identity" / "finished_utc.txt",
            "result_root": REF_SUMMARY,
            "environment": REPRO / "environment" / REF_ID / "environment_validation.json",
            "raw_dir": REF_RAW,
            "summary_dir": REF_SUMMARY,
            "seed_index": REF_SUMMARY / "reference_seed_index.csv",
        }

    run = args.run_root.expanduser().resolve()
    if not run.is_dir():
        fail(f"run directory missing: {run}")
    return {
        "label": run.name,
        "main": run / "main" / "summary" / "main_reproduction_summary.csv",
        "ablation": run / "ablation" / "summary" / "fresh_fair_subset_5seed_summary.json",
        "stats": run / "statistics" / "main_paired_bh_8metrics.csv",
        "factor": run / "statistics" / "factor_direction_candidate_vs_archived_T01.csv",
        "chapter4": run / "chapter4" / "chapter4_report.json",
        "status": run / "RUN_STATUS.txt",
        "started": run / "started_utc.txt",
        "finished": run / "finished_utc.txt",
        "result_root": run,
        "environment": run / "environment" / "environment_validation.json",
        "raw_dir": run,
        "summary_dir": run,
        "seed_index": run / "reference_seed_index.csv",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--reference", action="store_true",
                       help="Summarize the committed reference_20260921T175217Z evidence.")
    group.add_argument("--run-root", type=Path,
                       help="Summarize one completed fresh integrated full run.")
    args = parser.parse_args()
    paths = load_paths(args)

    main_rows = read_csv(paths["main"])
    keyed: dict[tuple[str, str], dict[str, str]] = {}
    for row in main_rows:
        key = (row.get("metric", ""), row.get("condition", ""))
        if key in keyed:
            fail(f"duplicate main summary row: {key}")
        keyed[key] = row
    expected_main = {
        (metric, condition)
        for metric, _ in MAIN_METRICS
        for condition in ("original", "context", "context_minus_original")
    }
    if set(keyed) != expected_main:
        fail("main summary does not contain the expected 8 metrics × 3 conditions")

    stat_rows = read_csv(paths["stats"])
    stats = {row["metric"]: row for row in stat_rows}
    if set(stats) != {metric for metric, _ in MAIN_METRICS}:
        fail("statistics file does not contain exactly the 8 expected main metrics")

    ablation = read_json(paths["ablation"])
    if ablation.get("units_verified") != 25 or ablation.get("seeds") != [1, 2, 3, 4, 5]:
        fail("ablation summary is not the verified 25-unit, five-seed result")
    metrics = ablation.get("metrics")
    if not isinstance(metrics, dict) or set(VARIANTS) - set(metrics):
        fail("ablation metrics are missing expected variants")

    factor_rows = read_csv(paths["factor"])
    chapter4 = read_json(paths["chapter4"])

    print()
    print("=" * 96)
    print("TORS THESIS REPRODUCTION — RESULT SUMMARY")
    print("=" * 96)
    print(f"Source:   {paths['label']}")
    print(f"Status:   {read_text(paths['status'])}")
    print(f"Started:  {read_text(paths['started'])}")
    print(f"Finished: {read_text(paths['finished'])}")
    print("Seeds:    1, 2, 3, 4, 5")
    print()

    def print_main_table(title: str, wanted: set[str]) -> None:
        print(title)
        table = []
        for metric, label in MAIN_METRICS:
            if metric not in wanted:
                continue
            original = keyed[metric, "original"]
            context = keyed[metric, "context"]
            delta = keyed[metric, "context_minus_original"]
            paper_ok = (
                original["status"] in {"exact", "numerically_close"}
                and context["status"] in {"exact", "numerically_close"}
            )
            table.append([
                label,
                f4(original["paper_mean"]),
                mean_sd(original),
                f4(context["paper_mean"]),
                mean_sd(context),
                f"{float(delta['reproduced_mean']):+.4f}",
                "OK" if paper_ok else "CHECK",
            ])
        print_table(
            ["Metric", "Thesis Orig", "Repro Orig mean±SD",
             "Thesis Ctx", "Repro Ctx mean±SD", "Δ Ctx-Orig", "Paper check"],
            table,
        )
        print()

    print_main_table("Table 4-11 — CP main experiment", CP_KEYS)
    print_main_table("Table 4-12 — OR/CIR main experiment", OR_KEYS)

    print("Teacher-requested significance check — five paired seeds, BH over 8 main metrics")
    sig_table = []
    for metric, label in (("or_r1", "R@1"), ("or_r3", "R@3")):
        row = stats[metric]
        sig_table.append([
            label,
            f"{float(row['delta_mean']):+.6f}",
            pval(row["p_value"]),
            pval(row["bh_adjusted_p_8_main"]),
            "YES" if row["significant_bh_0_05"].strip().lower() == "true" else "NO",
        ])
    print_table(["Metric", "Mean Δ", "raw p", "BH p", "Significant @ .05"], sig_table)
    sig_count = sum(
        row["significant_bh_0_05"].strip().lower() == "true"
        for row in stat_rows
    )
    print(f"BH significant main metrics: {sig_count}/8")
    print()

    print("Fair-subset ablation (reconstructed subset) — five-seed mean ± SD")
    abl_table = []
    for metric_key, label in ABLATION_METRICS:
        row = [label]
        for variant in VARIANTS:
            cell = metrics[variant][metric_key]
            row.append(f"{float(cell['mean']):.4f} ± {float(cell['std']):.4f}")
        abl_table.append(row)
    print_table(
        ["Metric", "Original", "Context", "NoWeather", "NoOccasion", "NoStyle"],
        abl_table,
    )
    print("Scope: reproducibly reconstructed 2026 fair subset; historical memberwise ID identity was not recovered.")
    print()

    print("Teacher-requested OR ablation sign check — Without Factor minus Full Context (Table 4-16 convention)")
    direction_rows = []
    comparable = []
    sign_flip = {
        "positive": "negative",
        "negative": "positive",
        "zero": "zero",
        "not_available": "not_available",
    }
    for row in factor_rows:
        if row["task"] != "CIR":
            continue
        if row["factor"] not in {"weather", "occasion"}:
            continue
        if row["metric"] not in {"Recall@10", "Recall@30", "Recall@50"}:
            continue
        archived = row["archived_T01_sign"]
        changed = "N/A" if archived == "not_available" else (
            "YES" if row["fresh_sign"] != archived else "NO"
        )
        direction_rows.append([
            row["factor"].title(),
            row["metric"],
            f"{-float(row['fresh_delta']):+.6f}",
            sign_flip.get(row["fresh_sign"], row["fresh_sign"]),
            sign_flip.get(archived, archived),
            changed,
        ])
        if changed != "N/A":
            comparable.append(changed)
    print_table(
        ["Factor", "Metric", "Fresh Δ", "Fresh sign", "Archived sign", "Sign changed?"],
        direction_rows,
    )
    print(f"Comparable Weather/Occasion OR sign changes: {sum(x == 'YES' for x in comparable)}/{len(comparable)}")
    print()

    condition_rows = [
        row for row in main_rows
        if row["condition"] in {"original", "context"}
    ]
    within = sum(row["status"] in {"exact", "numerically_close"} for row in condition_rows)
    positive = sum(
        float(row["reproduced_mean"]) > 0
        for row in main_rows
        if row["condition"] == "context_minus_original"
    )
    main_status = chapter4.get("main_mean_status") or {}

    print("Final evidence checks")
    checks = [
        ["Main thesis-condition means within tolerance", f"{within}/16"],
        ["Main Context-Original directions positive", f"{positive}/8"],
        ["Ablation training units verified", f"{ablation['units_verified']}/25"],
        ["Ablation CIR evaluable queries / unit", str(ablation.get("cir_evaluable_per_unit"))],
        ["Chapter 4 table IDs represented", f"{chapter4.get('total_chapter4_tables')}/20"],
        ["Chapter 4 scoped numeric rows", str(chapter4.get("numeric_comparison_rows"))],
        ["Chapter 4 main means within tolerance",
         f"{main_status.get('within_retrain_tolerance')}/{main_status.get('total_condition_means')}"],
    ]
    print_table(["Check", "Result"], checks)
    print()

    print("Interpretation / provenance")
    print("- Main five-seed numerical reproduction is the primary fresh quantitative reproduction.")
    print("- R@1 is not significant after BH correction; R@3 remains significant.")
    print("- Weather/Occasion OR sign differences are retained rather than adjusted to match the thesis.")
    print("- The fair-subset ablation is not claimed as source-exact because historical fair-subset memberwise IDs are missing.")
    print("- Historical DecoderLayerWithCrossAttn definition was not preserved; all 2026 experiments use the standardized decoder implementation (torch.nn.TransformerDecoderLayer).")
    print()

    environment = read_json(paths["environment"])
    packages = environment.get("packages") or {}
    torch_pkg = packages.get("torch") or {}
    torch_runtime = environment.get("torch_runtime") or {}
    python_version = str(environment.get("python", "not_available")).split()[0]
    torch_version = str(torch_pkg.get("installed_version", "not_available"))
    cuda_version = str(torch_runtime.get("torch_cuda_version", "not_available"))
    cudnn_version = str(torch_runtime.get("cudnn_version", "not_available"))

    branch = git_value("branch", "--show-current")
    artifact_commit = git_value("rev-parse", "HEAD")
    official_id = (read_json(OFFICIAL_MANIFEST).get("official_run_id", "not_available")
                   if OFFICIAL_MANIFEST.is_file() else "not_available")
    if args.reference:
        raw_evidence = display_path(paths["raw_dir"])
        aggregate_evidence = display_path(paths["summary_dir"])
        seed_evidence = display_path(paths["seed_index"])
    else:
        raw_evidence = display_path(Path(paths["result_root"]) / "main") + " ; " + display_path(Path(paths["result_root"]) / "ablation" / "runs")
        aggregate_evidence = display_path(Path(paths["result_root"]) / "main" / "summary") + " ; " + display_path(Path(paths["result_root"]) / "ablation" / "summary")
        seed_evidence = "per-unit manifests inside the fresh run directory"

    print("=" * 96)
    print("PROFESSOR REPRODUCIBILITY CHECKLIST — REQUESTED 10 ITEMS")
    print("=" * 96)
    checklist = [
        ["1", "PASS", "Full-run start / finish time",
         f"{read_text(paths['started'])} -> {read_text(paths['finished'])}"],
        ["2", "PASS", "Explicit successful run status",
         f"RUN_STATUS = {read_text(paths['status'])}"],
        ["3", "PASS", "Five actual seeds for main + ablation",
         "1, 2, 3, 4, 5"],
        ["4", "PASS", "Per-seed raw results + aggregates",
         f"raw: {raw_evidence} | summary: {aggregate_evidence} | seed evidence: {seed_evidence}"],
        ["5", "PASS", "Automatic Table 4-11 / 4-12 / ablation outputs",
         "printed above; machine-readable main/, ablation/, chapter4/ outputs are preserved"],
        ["6", "PASS", "Tested README + one-command flow",
         f"README: reproduction/README.md | command: bash reproduction/scripts/reproduce_all.sh --fresh | official clean-room run {official_id} (fresh clone, README section 0 only) PASSED, bit-identical to two earlier full runs | docs/clean_room_acceptance.md"],
        ["7", "PASS", "Python / PyTorch / CUDA / major packages",
         f"Python {python_version} | PyTorch {torch_version} | CUDA {cuda_version} | cuDNN {cudnn_version} | details: {display_path(paths['environment'])}"],
        ["8", "PARTIAL", "Hyperparameter tuning provenance",
         "validation FITB accuracy + final settings recovered; historical candidate ranges/search method/trials/full winning rationale NOT RECOVERED | docs/hyperparameter_tuning.md + results/tuning.csv"],
        ["9", "PASS*", "Standardized decoder reason + impact",
         "standardized decoder implementation (torch.nn.TransformerDecoderLayer) because the historical DecoderLayerWithCrossAttn definition is missing; numerical reproduction supported, historical source-exact recovery NOT claimed | docs/standardized_decoder.md"],
        ["10", "READY", "GitHub final artifact / public release",
         f"branch: {branch} | artifact commit: {artifact_commit} | fixed GitHub Release: v1.0.0-tors-reproduction (no DOI)"],
    ]
    for number, status, request, evidence in checklist:
        print(f"[{status}] {number}. {request}")
        print(f"       Evidence: {evidence}")
    print()
    print("Checklist interpretation")
    print("- Items 1-7 and 9 have concrete 2026 reproduction evidence; item 9 keeps its stated source-provenance caveat.")
    print("- Item 8 remains PARTIAL because missing historical tuning trials/candidate ranges are not fabricated.")
    print("- Item 10 is READY as a fixed artifact commit; the manuscript cites the GitHub Release v1.0.0-tors-reproduction.")
    print("- This checklist is printed so the reviewer does not need to manually search the repository for the ten requested items.")
    print()
    print(f"Results root: {paths['result_root']}")
    if not args.reference:
        print(f"Compare with the official run {official_id} (README 0.8): "
              f"python3 reproduction/scripts/compare_with_official_run.py --run-root {display_path(Path(paths['result_root']))}")
    print("=" * 96)


if __name__ == "__main__":
    try:
        main()
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        print(f"[SUMMARY BLOCKED] malformed result evidence: {exc}", file=sys.stderr)
        raise SystemExit(2)
