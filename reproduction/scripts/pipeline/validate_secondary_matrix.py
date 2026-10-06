#!/usr/bin/env python3
"""Gate full-run secondary analysis completion without claiming source exactness.

The preserved-analysis runner may catch module exceptions and write
status=failed into the matrix while still returning subprocess exit 0.
The full wrapper must treat any failed or missing module as an execution
failure. Historical-provenance status=partial is permitted and preserved.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

EXPECTED = (
    "length",
    "target_clue",
    "judge_preserved_outputs",
    "human_audit",
    "case_analysis",
    "two_tower",
    "judge_robustness_sampling_provenance",
    "judge_exact_generation",
    "case_visuals",
)


def validate(report: dict) -> tuple[int, int]:
    rows = report.get("modules")
    if not isinstance(rows, list):
        raise ValueError("Secondary analysis matrix has no module list")
    keyed = {}
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get("module"), str):
            raise ValueError("Malformed module entry")
        name = row["module"]
        if name in keyed:
            raise ValueError(f"Duplicate secondary module entry: {name}")
        keyed[name] = row.get("status")
    if set(keyed) != set(EXPECTED):
        raise ValueError(
            f"Secondary modules missing={sorted(set(EXPECTED) - set(keyed))} "
            f"unexpected={sorted(set(keyed) - set(EXPECTED))}"
        )
    failed = {name: status for name, status in keyed.items()
              if status not in ("exact", "partial")}
    if failed:
        raise ValueError(f"Secondary module failure/incomplete: {failed}")
    partial = sum(status == "partial" for status in keyed.values())
    overall = "partial" if partial else "exact"
    if report.get("overall") != overall:
        raise ValueError(
            f"Secondary overall status inconsistent: {report.get('overall')!r} "
            f"(expected {overall!r})"
        )
    return len(keyed), partial


def self_test() -> None:
    valid = {
        "modules": [
            {"module": name, "status": "partial" if name == "target_clue" else "exact"}
            for name in EXPECTED
        ],
        "overall": "partial",
    }
    assert validate(valid) == (9, 1)
    for label, changed in (
        ("failed module", {**valid, "modules": [
            {**r, "status": "failed"} if r["module"] == "two_tower" else r
            for r in valid["modules"]]}),
        ("missing module", {**valid, "modules": valid["modules"][:-1]}),
        ("wrong overall", {**valid, "overall": "exact"}),
        ("duplicate module", {**valid, "modules": valid["modules"] + valid["modules"][:1]}),
    ):
        try:
            validate(changed)
        except ValueError:
            pass
        else:
            raise AssertionError(f"{label} should have been blocked")
    print("[PASS] synthetic valid 9-module matrix with historical partial status")
    print("[PASS] failed, missing, duplicate, and inconsistent secondary matrices rejected")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--matrix", type=Path)
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()
    if args.self_test:
        self_test()
        if args.matrix is None:
            return
    if args.matrix is None:
        ap.error("Pass --matrix PATH or --self-test")
    if not args.matrix.is_file():
        raise SystemExit(f"[ERROR] missing secondary matrix: {args.matrix}")
    report = json.loads(args.matrix.read_text(encoding="utf-8"))
    try:
        count, partial = validate(report)
    except ValueError as error:
        raise SystemExit(f"[ERROR] secondary analysis validation failed: {error}")
    print(f"[PASS] secondary analysis matrix: {count}/{len(EXPECTED)} modules "
          f"completed, {partial} historical-provenance partial; "
          f"overall={report['overall']}")


if __name__ == "__main__":
    main()
