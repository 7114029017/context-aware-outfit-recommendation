#!/usr/bin/env python3
"""Assemble a Chapter-4 evidence table from independently produced run outputs.

This is an evidence aggregator, NOT a fresh model/LLM execution and NOT a
claim of exact historical provenance. Direct thesis-display comparisons,
numeric comparisons against an archived T01 aggregate, and provenance-only
tables are explicitly distinguished.

--source full: consume one integrated --mode full run.
--source existing: consume four explicitly named, independently accepted
prior-run result directories; never label this as a single fresh full run.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
REPRO_ROOT = ROOT / "reproduction"
REPRO_SCRIPTS = REPRO_ROOT / "scripts"
sys.path.insert(0, str(Path(__file__).resolve().parent))
from validate_secondary_matrix import validate as validate_secondary  # noqa: E402

METRICS = ("cp_auc", "cp_fitb", "or_r1", "or_r3", "or_r5", "or_r10", "or_r30", "or_r50")
METRIC_LABELS = ("CP AUC", "CP FITB Acc", "OR R@1", "OR R@3",
                 "OR R@5", "OR R@10", "OR R@30", "OR R@50")
CONDITIONS = ("original", "context", "context_minus_original")
VARIANTS = ("original", "context", "no_weather", "no_occasion", "no_style")
T01_CP = ("AUC", "FITB Acc")
T01_OR = ("Recall@10", "Recall@30", "Recall@50")
FIELDS = ("table", "item", "condition", "reference_kind", "reference_value",
          "recomputed_value", "numeric_difference", "comparison", "evidence", "limitation")
TABLE_NOTES = {
    "4-1": ("environment", "Runtime of this reproduction is recorded separately; exact 2025 environment is not proven."),
    "4-2": ("method", "Archived model code runs with the standardized decoder implementation; historical CP decoder source not recovered."),
    "4-3": ("preserved_numeric", "24 checked PDF-display fields from aligned saved CLO/MET/Tsub proxy outputs; independent MET valid source incomplete."),
    "4-4": ("preserved_numeric", "Length correlations recalculated from archived seed-level rows; module-level archived-table comparison."),
    "4-5": ("preserved_numeric", "Length matched-subset/bucket tables recalculated from archived outputs; module-level archived-table comparison."),
    "4-6": ("partial", "Historical P12 reliability_meta_from_subset.csv missing; present-day category reconstruction differs."),
    "4-7": ("partial", "Saved-output bottom-p intersections differ from thesis; cutoff ties and historical input/runtime identity unresolved."),
    "4-8": ("preserved_numeric", "30 checked PDF-display fields from six saved Judge comparison files; no LLM rerun and historical 150-ID selection not recovered."),
    "4-9": ("preserved_numeric", "30-case coverage recomputed from selected archived records; source label-generation provenance not proven."),
    "4-10": ("preserved_numeric", "750 archived human judgments recomputed and compared to archived T30; no new human judgments."),
    "4-11": ("main_numeric", "Fresh main CP condition means versus paper; paper SD/CI/p/effect sizes are NOT fully verified by this output."),
    "4-12": ("main_numeric", "Fresh main OR condition means versus paper; paper SD/CI/p/significance stars are NOT fully verified."),
    "4-13": ("candidate_numeric", "CP Original/Context metrics compared ONLY against archived T01 aggregate; this is NOT the individual published Table 4-13 cell comparison."),
    "4-14": ("candidate_numeric", "OR Original/Context metrics compared ONLY against archived T01 aggregate; historical fair-subset membership not recovered."),
    "4-15": ("candidate_numeric", "CP factor-removal metrics compared ONLY against archived T01 aggregate; not a published Table 4-15 cell-by-cell test."),
    "4-16": ("candidate_numeric", "OR factor-removal metrics compared ONLY against archived T01 aggregate; some factor directions differ; historical subset ID missing."),
    "4-17": ("archived_only", "Fresh rankings for identical set_id/target/seed/pool not independently matched against published examples."),
    "4-18": ("archived_only", "Fresh rankings for identical set_id/target/seed/pool not independently matched against published examples."),
    "4-19": ("archived_only", "Image-dependent human visual/color judgments not independently re-executed."),
    "4-20": ("archived_only", "Identical-case fresh ranking comparison not independently verified."),
}


def block(message: str) -> None:
    raise SystemExit("[BLOCKED] chapter4 report: " + message)


def resolve(path: str | Path) -> Path:
    path = Path(path).expanduser()
    return (path if path.is_absolute() else ROOT / path).resolve()


def load_json(path: Path) -> dict:
    if not path.is_file():
        block(f"Missing required input JSON: {path}")
    obj = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(obj, dict):
        block(f"Expected JSON object: {path}")
    return obj


def read_csv(path: Path) -> list[dict]:
    if not path.is_file():
        block(f"Missing required input CSV: {path}")
    with path.open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        block(f"Empty required CSV: {path}")
    return rows


def hash_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def real_number(value: object, label: str) -> float:
    try:
        x = float(value)
    except (TypeError, ValueError):
        block(f"Missing or invalid number {label}: {value!r}")
    if not math.isfinite(x):
        block(f"Nonfinite number {label}: {value!r}")
    return x


def require_keys(rows: list[dict], keys: tuple[str, ...], expected: set[tuple]) -> dict:
    found = {}
    for row in rows:
        try:
            key = tuple(row[name] for name in keys)
        except KeyError as e:
            block(f"Required CSV key missing: {e}")
        if key in found:
            block(f"Duplicate input row {key}")
        found[key] = row
    if set(found) != expected:
        block(f"Input keys wrong: missing={sorted(expected - set(found))}; "
              f"unexpected={sorted(set(found) - expected)}")
    return found


def add(rows: list[dict], table: str, item: str, condition: str,
        ref_kind: str, reference: object, observed: object, verdict: str,
        evidence: Path, limitation: str = "", difference: object = "") -> None:
    rows.append({
        "table": table, "item": item, "condition": condition,
        "reference_kind": ref_kind,
        "reference_value": "" if reference is None else str(reference),
        "recomputed_value": "" if observed is None else str(observed),
        "numeric_difference": "" if difference == "" else str(difference),
        "comparison": verdict, "evidence": str(evidence),
        "limitation": limitation or TABLE_NOTES[table][1],
    })


def check_main(path: Path, rows: list[dict]) -> None:
    comparison = read_csv(path)
    expected = {(metric, cond) for metric in METRICS for cond in CONDITIONS}
    indexed = require_keys(comparison, ("metric", "condition"), expected)
    for metric, label in zip(METRICS, METRIC_LABELS):
        table = "4-11" if metric.startswith("cp_") else "4-12"
        for condition in CONDITIONS:
            src = indexed[metric, condition]
            got = real_number(src.get("reproduced_mean"), f"{metric}/{condition}/fresh")
            paper = real_number(src.get("paper_mean"), f"{metric}/{condition}/paper")
            difference = got - paper
            if condition == "context_minus_original":
                # This paper value is derived from the two displayed paper
                # means by the archived comparator, NOT a separately printed
                # paper delta or an independent significance-test result.
                kind = "paper_means_derived_delta"
                verdict = "direction_agrees" if got > 0 and paper > 0 else "direction_differs"
            else:
                kind = "paper_mean"
                if round(got, 4) == round(paper, 4):
                    verdict = "same_4dp"
                elif abs(difference) <= 0.005:
                    verdict = "within_retrain_tolerance_0.005"
                else:
                    verdict = "outside_retrain_tolerance_0.005"
                expected_verdict = {
                    "same_4dp": "exact",
                    "within_retrain_tolerance_0.005": "numerically_close",
                    "outside_retrain_tolerance_0.005": "outside_numeric_tolerance",
                }[verdict]
                if src.get("status") != expected_verdict:
                    block(f"Paper comparison status inconsistent: {metric}/{condition}")
                if int(src.get("n_seeds", 0)) != 5:
                    block(f"Not a five-seed paper comparison: {metric}/{condition}")
                real_number(src.get("reproduced_sd"), f"{metric}/{condition}/std")
            add(rows, table, label, condition, kind, paper, got, verdict,
                path, difference=difference)


def check_ablation(directory: Path, rows: list[dict]) -> None:
    summary_path = directory / "fresh_fair_subset_5seed_summary.json"
    summary = load_json(summary_path)
    if not (summary.get("units_verified") == 25
            and summary.get("cir_evaluable_per_unit") == 3432
            and summary.get("batch_status") == "PASSED_CANDIDATE_SOURCE"
            and summary.get("evaluation_query_membership_sha256")):
        block("Ablation summary has no verified 25-unit / 3432-query fair-subset identity")
    path = directory / "archived_vs_fresh_T01_numeric_comparison.csv"
    comparison = read_csv(path)
    expected = {(v, metric) for v in VARIANTS for metric in T01_CP + T01_OR}
    indexed = require_keys(comparison, ("variant", "metric"), expected)
    for (variant, metric), src in sorted(indexed.items()):
        if variant in ("original", "context"):
            table = "4-13" if metric in T01_CP else "4-14"
        else:
            table = "4-15" if metric in T01_CP else "4-16"
        paper = real_number(src.get("archived_mean"), f"T01 {variant}/{metric}/archived")
        fresh = real_number(src.get("fresh_mean"), f"T01 {variant}/{metric}/fresh")
        delta = fresh - paper
        reported_abs = real_number(src.get("abs_mean_difference"), "abs_mean_difference")
        if not math.isclose(abs(delta), reported_abs, rel_tol=0, abs_tol=1e-12):
            block(f"T01 difference arithmetic mismatch: {variant}/{metric}")
        # T01 is an archived aggregate, not verified as the literal cells
        # of each individually published Chapter 4 table.
        add(rows, table, metric, variant, "archived_T01_mean_not_published_table_cell",
            paper, fresh, "same_number" if abs(delta) <= 1e-12 else "different_from_archived_T01",
            path, difference=delta)


def check_secondary(directory: Path, proxy_dir: Path, rows: list[dict]) -> dict:
    matrix_path = directory / "remaining_reproduction_matrix.json"
    matrix = load_json(matrix_path)
    try:
        validate_secondary(matrix)
    except ValueError as e:
        block(f"Secondary matrix invalid: {e}")
    modules = {m["module"]: m for m in matrix["modules"]}
    if modules["length"]["status"] == "failed":
        block("Length module failed")
    if modules["human_audit"]["status"] == "failed":
        block("Human audit module failed")
    judge = modules["judge_preserved_outputs"]
    proxy_path = proxy_dir / "table_4_3_manifest.json"
    proxy = load_json(proxy_path)
    proxy_rows = read_csv(proxy_dir / "table_4_3_paper_field_comparison.csv")
    expected_proxy = {(str(x["proxy"]), str(x["field"]))
                      for x in proxy.get("paper_field_comparison", [])}
    if len(expected_proxy) != 24:
        block("Table 4-3 manifest does not contain exactly 24 distinct paper fields")
    proxy_index = require_keys(proxy_rows, ("proxy", "field"), expected_proxy)
    for key, src in sorted(proxy_index.items()):
        ref = real_number(src.get("paper"), f"Table 4-3 {key} paper")
        fresh = real_number(src.get("preserved_recomputed"), f"Table 4-3 {key} value")
        matched = str(src.get("paper_display_match", "")).lower() == "true"
        if str(src.get("paper_display_match", "")).lower() not in ("true", "false"):
            block(f"Invalid Table 4-3 comparison flag: {key}")
        add(rows, "4-3", "/".join(key), "unique_outfit_preserved_proxy",
            "paper_display_value", ref, fresh,
            "same_paper_display" if matched else "different_paper_display",
            proxy_dir / "table_4_3_paper_field_comparison.csv",
            difference=fresh - ref)
    if bool(proxy.get("all_listed_paper_fields_match_display_precision")) != all(
            r["comparison"] == "same_paper_display" for r in rows if r["table"] == "4-3"):
        block("Table 4-3 manifest overall result differs from CSV")

    clue = modules["target_clue"].get("details") or {}
    archived = clue.get("archived_counts") or {}
    present = clue.get("recomputed_counts") or {}
    if set(archived) != set(present) or len(archived) != 6:
        block("Target-clue archived/current six-count scope incomplete")
    for metric in sorted(archived):
        old, new = int(archived[metric]), int(present[metric])
        add(rows, "4-6", metric, "N=9311_current_vs_archived",
            "archived_A03_count_not_all_published_cells", old, new,
            "same_number" if old == new else "different_from_archived_A03",
            matrix_path, difference=new - old)

    judge_details = judge.get("details") or {}
    intersections = judge_details.get("table_4_7_paper_intersections")
    if not isinstance(intersections, list) or len(intersections) != 3:
        block("Table 4-7 paper intersection source not present in Judge report")
    bottom_path = directory / "judge/bottomp_recomputed.csv"
    bottom = read_csv(bottom_path)
    indexed_bottom = {}
    for src in bottom:
        p = real_number(src.get("p(bottom)"), "bottom p")
        if p in indexed_bottom:
            block("Duplicate Judge bottom-p row")
        indexed_bottom[p] = src
    if set(indexed_bottom) != {.05, .10, .20}:
        block("Judge bottom-p percentages are missing/unexpected")
    observed_intersections = []
    for p, paper in zip((.05, .10, .20), intersections):
        val = int(real_number(indexed_bottom[p].get("intersection"), "bottom intersection"))
        paper = int(real_number(paper, "paper bottom intersection"))
        observed_intersections.append(val)
        add(rows, "4-7", f"bottom {p:.0%} intersection", "saved_Judge_outputs",
            "paper_reported_intersection", paper, val,
            "same_number" if paper == val else "different_from_paper",
            bottom_path, difference=val - paper)
    if observed_intersections != judge_details.get("table_4_7_recomputed_intersections"):
        block("Judge Table 4-7 rows disagree with matrix detail")

    robustness_path = directory / "judge/table_4_8_paper_field_comparison.csv"
    robustness = read_csv(robustness_path)
    judges = ("Gemma-3", "Qwen3-VL")
    variants = ("P0-R2", "P1", "P2")
    metrics = ("mean_abs_diff", "median_abs_diff", "max_abs_diff",
               "within_0_05_rate", "within_0_10_rate")
    expected_robustness = {(j, v, metric) for j in judges for v in variants for metric in metrics}
    idx = require_keys(robustness, ("judge", "variant", "field"), expected_robustness)
    for key, src in sorted(idx.items()):
        if int(src.get("n_archived", 0)) != 150:
            block(f"Judge robustness group must use 150 archived cases: {key}")
        paper = src["paper_display"]
        fresh = src["preserved_recomputed_display"]
        matched = paper == fresh
        if str(src.get("paper_display_match", "")).lower() != ("true" if matched else "false"):
            block(f"Judge table 4-8 display comparison flag inconsistent: {key}")
        add(rows, "4-8", key[2], f"{key[0]}/{key[1]} archived 150 IDs",
            "paper_display_value", paper, fresh,
            "same_paper_display" if matched else "different_paper_display", robustness_path)
    if bool(judge_details.get("table_4_8_paper_display_match_from_preserved_6x150")) != all(
            r["comparison"] == "same_paper_display" for r in rows if r["table"] == "4-8"):
        block("Table 4-8 matrix overall flag inconsistent with CSV")

    return {"secondary_overall": matrix["overall"],
            "secondary_module_statuses": {name: obj["status"] for name, obj in modules.items()},
            "proxy_primary_unique_outfits": proxy.get("n_unique_primary_rows")}


def build_overview(rows: list[dict], secondary: dict) -> list[dict]:
    grouped = {k: [r for r in rows if r["table"] == k] for k in TABLE_NOTES}
    module = secondary["secondary_module_statuses"]
    overview = []
    for table, (kind, note) in TABLE_NOTES.items():
        rs = grouped[table]
        if table in ("4-3", "4-8"):
            result = "all_checked_display_fields_match" if (
                rs and all(r["comparison"] == "same_paper_display" for r in rs)
            ) else "some_checked_display_fields_differ"
        elif table in ("4-11", "4-12"):
            conditions = [r for r in rs if r["reference_kind"] == "paper_mean"]
            result = ("main_condition_means_within_retrain_tolerance" if
                      conditions and all(r["comparison"] in (
                          "same_4dp", "within_retrain_tolerance_0.005") for r in conditions)
                      else "some_main_condition_means_outside_retrain_tolerance")
        elif table in ("4-13", "4-14", "4-15", "4-16"):
            result = "candidate_ablation_vs_archived_T01_only_not_paper_table_exact"
        elif table in ("4-6", "4-7"):
            result = "all_checked_counts_match" if rs and all(
                r["comparison"] == "same_number" for r in rs
            ) else "checked_counts_differ"
        elif table in ("4-4", "4-5"):
            result = "archived_length_reanalysis_" + module["length"]
        elif table in ("4-9", "4-10"):
            result = "archived_human_audit_reanalysis_" + module["human_audit"]
        else:
            result = "provenance_only_or_not_independently_verified"
        overview.append({
            "table": table, "scope": kind, "checked_numeric_rows": len(rs),
            "result": result, "limitation": note,
        })
    return overview


def save_csv(path: Path, records: list[dict], fieldnames: tuple[str, ...]) -> None:
    if not records:
        block(f"No rows to write: {path}")
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(records)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", choices=("full", "existing"), default="full")
    parser.add_argument("--run-root", default=None)
    parser.add_argument("--main-csv", default=None)
    parser.add_argument("--ablation-dir", default=None)
    parser.add_argument("--secondary-dir", default=None)
    parser.add_argument("--proxy-dir", default=None)
    parser.add_argument("--out-dir", default=None)
    args = parser.parse_args()

    if args.source == "full":
        if not args.run_root or any((
            args.main_csv, args.ablation_dir, args.secondary_dir, args.proxy_dir
        )):
            parser.error("--source full requires only --run-root; no ad hoc input overrides")
        run = resolve(args.run_root)
        main_path = run / "main/summary/main_reproduction_summary.csv"
        ablation = run / "ablation/summary"
        secondary = run / "secondary/results"
        proxy = secondary / "environment_proxy"
        out = resolve(args.out_dir) if args.out_dir else run / "chapter4"
        classification = "new single integrated full-run outputs"
    else:
        if args.run_root or not all((
            args.main_csv, args.ablation_dir, args.secondary_dir, args.proxy_dir, args.out_dir
        )):
            parser.error("--source existing requires explicit --main-csv --ablation-dir "
                         "--secondary-dir --proxy-dir --out-dir; do not pass --run-root")
        main_path = resolve(args.main_csv)
        ablation = resolve(args.ablation_dir)
        secondary = resolve(args.secondary_dir)
        proxy = resolve(args.proxy_dir)
        out = resolve(args.out_dir)
        classification = "previously completed SEPARATE reproduction runs; NO new training/evaluation"

    inputs = (
        main_path, ablation / "fresh_fair_subset_5seed_summary.json",
        ablation / "archived_vs_fresh_T01_numeric_comparison.csv",
        secondary / "remaining_reproduction_matrix.json",
        secondary / "judge/bottomp_recomputed.csv",
        secondary / "judge/table_4_8_paper_field_comparison.csv",
        proxy / "table_4_3_manifest.json",
        proxy / "table_4_3_paper_field_comparison.csv",
    )
    if any(out == p or out in p.parents for p in inputs):
        block("Output directory overlaps an evidence input")
    if out.exists() and any(out.iterdir()):
        block(f"Use a new/empty report output directory: {out}")

    rows = []
    check_main(main_path, rows)
    check_ablation(ablation, rows)
    secondary_status = check_secondary(secondary, proxy, rows)
    if len(rows) != 112:
        block(f"Expected 112 numeric comparisons; got {len(rows)}")
    overview = build_overview(rows, secondary_status)
    if len(overview) != 20:
        block("Not all twenty Chapter-4 table IDs represented")
    out.mkdir(parents=True, exist_ok=True)
    save_csv(out / "chapter4_numeric_comparison.csv", rows, FIELDS)
    save_csv(out / "chapter4_table_overview.csv", overview,
             ("table", "scope", "checked_numeric_rows", "result", "limitation"))
    report = {
        "classification": classification,
        "source_mode": args.source,
        "new_model_training_or_evaluation_performed_by_this_report": False,
        "total_chapter4_tables": 20,
        "numeric_comparison_rows": len(rows),
        "numeric_rows_by_table": dict(sorted(Counter(r["table"] for r in rows).items())),
        "paper_display_match_4_3": sum(r["comparison"] == "same_paper_display"
                                       for r in rows if r["table"] == "4-3"),
        "paper_display_match_4_8": sum(r["comparison"] == "same_paper_display"
                                       for r in rows if r["table"] == "4-8"),
        "main_mean_status": {
            "within_retrain_tolerance": sum(r["comparison"] in
                ("same_4dp", "within_retrain_tolerance_0.005")
                for r in rows if r["reference_kind"] == "paper_mean"),
            "total_condition_means": 16,
        },
        "ablation_reference_is_archived_T01_not_individual_paper_tables": True,
        "secondary": secondary_status,
        "input_evidence_sha256": {str(p): hash_file(p) for p in inputs},
        "overview": overview,
        "limitations": [
            "A full pipeline pass does not prove all 20 published tables match cell by cell.",
            "Table 4-13..4-16 numeric rows refer to the archived T01 aggregate, not each separately printed thesis table.",
            "Table 4-11..4-12 paper SD/CI/p/effect-size/significance cells are NOT individually compared.",
            "Tables without independently verified cell-level sources remain explicitly provenance-only.",
            "The original historical memberwise membership of the fair subset has not been recovered.",
            "Judge robustness uses preserved compare outputs; no Qwen/Gemma inference is rerun.",
        ],
    }
    (out / "chapter4_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    lines = [
        "# Chapter 4 reproduction comparison",
        "",
        f"- Source: **{classification}**",
        "- Numeric comparison rows: **112**, across the explicitly enumerated sources below.",
        "- Table overview covers **4-1 through 4-20**, including unverified tables.",
        "- The report does not assert all printed thesis table cells are exactly reproduced.",
        "",
        "| Thesis table | Scope | Checked rows | Comparison result |",
        "|---|---|---:|---|",
    ]
    for row in overview:
        lines.append(
            f"| {row['table']} | {row['scope']} | {row['checked_numeric_rows']} "
            f"| {row['result']} |"
        )
    lines += [
        "",
        "## Interpretive boundaries",
        "",
    ] + ["- " + item for item in report["limitations"]]
    lines += [
        "",
        "See `chapter4_numeric_comparison.csv` for source-specific references, "
        "recomputed values, differences, and provenance notes. "
        "See `chapter4_report.json` for input SHA-256 evidence.",
        "",
    ]
    (out / "chapter4_report.md").write_text("\n".join(lines), encoding="utf-8")
    print("[PASS] 20/20 Chapter 4 table IDs represented, including provenance-only tables")
    print("[PASS] 112 explicitly sourced numeric comparison rows")
    print("[OK]", out / "chapter4_report.md")
    print("[OK]", out / "chapter4_numeric_comparison.csv")
    print("[NOTE] Archived T01 comparison of the reconstructed fair subset is NOT exact verification of thesis Tables 4-13..4-16")


if __name__ == "__main__":
    main()
