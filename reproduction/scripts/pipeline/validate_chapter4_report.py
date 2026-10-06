#!/usr/bin/env python3
"""Independently validate an assembled Chapter-4 report; no model/data rerun.

Checks the saved CSV/JSON's row membership, arithmetic, explicit reference
scope, per-table classifications, and SHA-256 of *small report input files*.
Does NOT claim all published thesis cells or original training are exact.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter
from pathlib import Path

EXPECTED = {
    "4-3": 24, "4-6": 6, "4-7": 3, "4-8": 30,
    "4-11": 6, "4-12": 18,
    "4-13": 4, "4-14": 6, "4-15": 6, "4-16": 9,
}
EXACT_TABLES = {f"4-{n}" for n in range(1, 21)}
FIELDS = ("table", "item", "condition", "reference_kind", "reference_value",
          "recomputed_value", "numeric_difference", "comparison", "evidence", "limitation")


def fail(message: str) -> None:
    raise SystemExit("[BLOCKED] Chapter 4 output audit: " + message)


def read_rows(path: Path) -> list[dict]:
    if not path.is_file():
        fail(f"Missing report output {path}")
    with path.open(encoding="utf-8-sig", newline="") as f:
        result = list(csv.DictReader(f))
    if not result:
        fail(f"Empty report CSV {path}")
    return result


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def num(value: str, label: str) -> float:
    try:
        x = float(value)
    except (TypeError, ValueError):
        fail(f"{label}: expected numeric value, got {value!r}")
    if not math.isfinite(x):
        fail(f"{label}: nonfinite value")
    return x


def check(out: Path, source: str | None) -> None:
    path = out / "chapter4_report.json"
    if not path.is_file():
        fail(f"Missing {path}")
    report = json.loads(path.read_text(encoding="utf-8"))
    mode = report.get("source_mode")
    if mode not in ("existing", "full") or (source and mode != source):
        fail(f"Source classification mismatch: expected={source} actual={mode}")
    if report.get("total_chapter4_tables") != 20 or report.get("numeric_comparison_rows") != 112:
        fail("JSON does not describe 20 tables and 112 scoped comparison rows")
    if report.get("new_model_training_or_evaluation_performed_by_this_report") is not False:
        fail("The report generator must not claim it retrained/evaluated models")
    if report.get("ablation_reference_is_archived_T01_not_individual_paper_tables") is not True:
        fail("Archived T01 scope caveat absent")

    overview = read_rows(out / "chapter4_table_overview.csv")
    numeric = read_rows(out / "chapter4_numeric_comparison.csv")
    if len(overview) != 20 or len(numeric) != 112:
        fail(f"CSV row-count mismatch: overview={len(overview)} numeric={len(numeric)}")
    if set(numeric[0]) != set(FIELDS):
        fail(f"Unexpected numeric comparison columns: {sorted(numeric[0])}")
    by_table = Counter(row["table"] for row in numeric)
    if dict(by_table) != EXPECTED:
        fail(f"Unexpected numeric coverage by table: {dict(sorted(by_table.items()))}")
    keys = [(r["table"], r["item"], r["condition"], r["reference_kind"]) for r in numeric]
    if len(set(keys)) != len(keys):
        fail("Duplicate comparison key: table/item/condition/reference_kind")
    indexed = {}
    for row in overview:
        key = row["table"]
        if key in indexed:
            fail(f"Duplicate Chapter-4 overview entry: {key}")
        indexed[key] = row
        if int(row["checked_numeric_rows"]) != by_table.get(key, 0):
            fail(f"Overview numeric count mismatch: {key}")
        if not row["scope"] or not row["result"] or not row["limitation"]:
            fail(f"Overview scope/result/limitation incomplete: {key}")
    if set(indexed) != EXACT_TABLES:
        fail(f"Chapter-4 overview coverage wrong: {set(indexed)}")
    manifest_counts = report.get("numeric_rows_by_table")
    if manifest_counts != dict(sorted(by_table.items())):
        fail("JSON reported per-table numeric counts do not match CSV")
    if len(report.get("overview", [])) != 20:
        fail("JSON table overview incomplete")
    for item in report["overview"]:
        key = item["table"]
        if key not in indexed or str(item["checked_numeric_rows"]) != indexed[key]["checked_numeric_rows"]:
            fail(f"JSON overview row mismatch: {key}")
        for field in ("scope", "result", "limitation"):
            if str(item[field]) != indexed[key][field]:
                fail(f"JSON overview {field} mismatch: {key}")

    ref_counts = Counter()
    failures = Counter()
    for row in numeric:
        table = row["table"]
        kind = row["reference_kind"]
        ref_counts[(table, kind)] += 1
        if not row["evidence"] or not row["limitation"]:
            fail(f"Missing evidence or provenance limitation: {row}")
        if kind in ("paper_display_value", "paper_mean", "paper_means_derived_delta",
                    "archived_T01_mean_not_published_table_cell",
                    "archived_A03_count_not_all_published_cells",
                    "paper_reported_intersection"):
            if table != "4-8":
                a = num(row["reference_value"], f"{table}/{row['item']} reference")
                z = num(row["recomputed_value"], f"{table}/{row['item']} actual")
                d = num(row["numeric_difference"], f"{table}/{row['item']} difference")
                if not math.isclose(z - a, d, rel_tol=0, abs_tol=1e-11):
                    fail(f"Numeric difference arithmetic mismatch: {table}/{row['item']}")
        if table in ("4-3", "4-8"):
            if kind != "paper_display_value":
                fail(f"{table} must refer to explicitly checked paper-displayed values")
            # A future run can genuinely disagree with the displayed paper
            # result; this audit does not silently turn that into a success.
            expected = ("same_paper_display" if row["reference_value"] == row["recomputed_value"]
                        else "different_paper_display")
            if row["comparison"] != expected:
                # Table 4-3 reports raw full-precision values, so a display
                # match can exist even if raw floats are different.
                if table == "4-8":
                    fail("Table 4-8 reference/display flag inconsistent")
            if row["comparison"] != "same_paper_display":
                failures[table] += 1
        if table in ("4-11", "4-12"):
            if kind not in ("paper_mean", "paper_means_derived_delta"):
                fail(f"Main comparison incorrectly labeled as {kind}")
            if kind == "paper_mean" and row["comparison"] not in (
                "same_4dp", "within_retrain_tolerance_0.005",
                "outside_retrain_tolerance_0.005"
            ):
                fail("Unrecognized main paper-mean status")
            if kind == "paper_means_derived_delta" and row["comparison"] not in (
                "direction_agrees", "direction_differs"
            ):
                fail("Unrecognized derived-delta direction status")
        if table in ("4-13", "4-14", "4-15", "4-16"):
            if kind != "archived_T01_mean_not_published_table_cell":
                fail("Candidate ablation mistakenly promoted to published-table exactness")
            if row["comparison"] not in ("same_number", "different_from_archived_T01"):
                fail("Unexpected archived-T01 comparison status")
    if sum(ref_counts[t, "paper_mean"] for t in ("4-11", "4-12")) != 16:
        fail("Main comparison does not have 16 printed condition means")
    if sum(ref_counts[t, "paper_means_derived_delta"] for t in ("4-11", "4-12")) != 8:
        fail("Main comparison does not have 8 separately labeled derived deltas")
    for table, required in (("4-3", 24), ("4-8", 30)):
        matched = by_table[table] - failures[table]
        key = f"paper_display_match_{table.replace('-', '_')}"
        if report.get(key) != matched:
            fail(f"JSON {key} count differs from CSV rows")
        verdict = ("all_checked_display_fields_match" if failures[table] == 0
                   else "some_checked_display_fields_differ")
        if indexed[table]["result"] != verdict:
            fail(f"Overview {table} display-match verdict inconsistent")

    inputs = report.get("input_evidence_sha256")
    if not isinstance(inputs, dict) or len(inputs) != 8:
        fail("Expected eight distinct file-hash evidence entries")
    for raw, expected_hash in inputs.items():
        input_path = Path(raw)
        if not input_path.is_file() or sha256(input_path) != expected_hash:
            fail(f"Source input is missing or has changed since report generation: {raw}")

    print(f"[PASS] mode={mode}: 20 table entries, 112 unique source-scoped numeric rows")
    print("[PASS] 16 paper main means + 8 derived deltas; 25 ablation values remain archived-T01-only")
    print(f"[PASS] paper-display matched: Table 4-3 {by_table['4-3']-failures['4-3']}/24; "
          f"Table 4-8 {by_table['4-8']-failures['4-8']}/30")
    for table in ("4-6", "4-7"):
        changed = [f"{r['item']}: {r['reference_value']} -> {r['recomputed_value']}"
                   for r in numeric if r["table"] == table and r["comparison"] != "same_number"]
        print(f"[NOTE] Table {table} differences from archived/paper values: "
              + ("; ".join(changed) if changed else "none"))
    differences = [abs(num(r["numeric_difference"], "archived T01 difference"))
                   for r in numeric if r["reference_kind"] ==
                   "archived_T01_mean_not_published_table_cell"]
    print(f"[NOTE] Max |fresh fair-subset - archived T01 mean|: {max(differences):.12g}; "
          "NOT exact verification of Tables 4-13..4-16")
    print("[PASS] JSON/CSV row counts, arithmetic, overview classifications, and all 8 input SHA-256 hashes")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--report-dir", required=True, type=Path)
    ap.add_argument("--source", choices=("full", "existing"), default=None)
    args = ap.parse_args()
    check(args.report_dir.expanduser().resolve(), args.source)


if __name__ == "__main__":
    main()
