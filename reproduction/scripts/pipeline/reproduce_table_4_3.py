#!/usr/bin/env python3
"""Recompute thesis Table 4-3 from the handoff's preserved CLO/MET/Tsub data.

No proxy estimation, image inference, model training, or data correction occurs.
Primary sample = archived {train,valid,test}_temperature.json rows; the
temperature.ipynb handoff creates these records by set_id merging CLO and MET.
Standalone CLO/MET files are audited independently and *not silently mixed*
with missing rows from another source.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import subprocess
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
REPRO_ROOT = ROOT / "reproduction"
REPRO_SCRIPTS = REPRO_ROOT / "scripts"
BASE = ROOT / "01_資料建構_data_construction" / "clo_met_temperature"
SPLITS = {"train": 16995, "valid": 3000, "test": 15145}
PAPER = {
    "CLO": {"median": .93, "q25": .63, "q75": 1.21, "p05": .25, "p95": 2.07,
            "max": 5.20, "count_gt_4": 20},
    "MET": {"median": 1.00, "q25": 1.00, "q75": 1.30, "p05": 1.00, "p95": 1.50,
            "count_eq_1": 25786, "count_gt_10": 8, "pct_eq_1_1dp": 73.4},
    "Tsub": {"median": 22.9, "q25": 19.9, "q75": 24.5, "p05": 13.5, "p95": 27.1,
             "count_lt_0": 257, "count_lt_minus_20": 51,
             "pct_lt_0_2dp": .73, "pct_lt_minus_20_2dp": .15},
}


def fail(message: str) -> None:
    raise SystemExit("[BLOCKED] " + message)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_json(path: Path):
    if not path.is_file():
        fail(f"Missing archived input: {path}")
    with path.open(encoding="utf-8-sig") as f:
        return json.load(f)


def parse_records(path: Path, *, jsonl: bool) -> list[dict]:
    if not path.is_file():
        fail(f"Missing archived input: {path}")
    if jsonl:
        with path.open(encoding="utf-8-sig") as f:
            records = [json.loads(line) for line in f if line.strip()]
    else:
        records = read_json(path)
    if not isinstance(records, list) or any(not isinstance(r, dict) for r in records):
        fail(f"Expected list of records: {path}")
    return records


def index_records(rows: list[dict], path: Path) -> tuple[dict[str, dict], dict]:
    """Index preserved records by outfit, retaining explicit duplicate provenance.

    A repeated set_id may be collapsed ONLY when every field of the entire
    archived record is identical. Conflicting copies are a blocking error.
    The file itself remains unchanged; raw and unique counts are both reported.
    """
    indexed = {}
    counts = Counter()
    for row in rows:
        sid = str(row.get("set_id", "")).strip()
        if not sid:
            fail(f"Blank set_id in {path}")
        if sid in indexed and indexed[sid] != row:
            fail(f"Conflicting duplicate set_id in {path}: {sid!r}; no row selected")
        indexed.setdefault(sid, row)
        counts[sid] += 1
    duplicates = {sid: n for sid, n in counts.items() if n > 1}
    audit = {
        "raw_records": len(rows),
        "unique_set_ids": len(indexed),
        "identical_duplicate_ids": len(duplicates),
        "extra_identical_records": len(rows) - len(indexed),
        "duplicated_id_multiplicities": duplicates,
        "duplicate_policy": (
            "Only byte-equivalent-in-content JSON record objects with the same "
            "set_id are collapsed for unique-outfit statistics; any conflicting "
            "record is a blocking error. Original files are never rewritten."
        ),
    }
    return indexed, audit


def numeric(row: dict, field: str, path: Path, sid: str) -> float:
    value = row.get(field)
    if isinstance(value, bool) or value is None or value == "":
        fail(f"Missing/non-numeric {field} at {path} set_id={sid}")
    try:
        result = float(value)
    except (TypeError, ValueError):
        fail(f"Invalid {field} at {path} set_id={sid}: {value!r}")
    if not math.isfinite(result):
        fail(f"Non-finite {field} at {path} set_id={sid}")
    return result


def describe(values: list[float], name: str) -> dict:
    a = np.asarray(values, dtype=np.float64)
    if not a.size:
        fail(f"No {name} values")
    quant = np.quantile(a, [.05, .25, .50, .75, .95], method="linear")
    row = {
        "proxy": name, "n": len(values), "p05": float(quant[0]),
        "q25": float(quant[1]), "median": float(quant[2]),
        "q75": float(quant[3]), "p95": float(quant[4]),
        "min": float(a.min()), "max": float(a.max()),
    }
    if name == "CLO":
        row["count_gt_4"] = int(np.count_nonzero(a > 4))
    elif name == "MET":
        row["count_eq_1"] = int(np.count_nonzero(a == 1))
        row["count_gt_10"] = int(np.count_nonzero(a > 10))
        row["pct_eq_1_1dp"] = round(row["count_eq_1"] / len(values) * 100, 1)
    else:
        row["count_lt_0"] = int(np.count_nonzero(a < 0))
        row["count_lt_minus_20"] = int(np.count_nonzero(a < -20))
        row["pct_lt_0_2dp"] = round(row["count_lt_0"] / len(values) * 100, 2)
        row["pct_lt_minus_20_2dp"] = round(row["count_lt_minus_20"] / len(values) * 100, 2)
    return row


def compare_display(row: dict, reference: dict, name: str) -> list[dict]:
    # Paper displays CLO and MET to two decimals, temperature to one.
    digits = 1 if name == "Tsub" else 2
    checks = []
    for field, expected in reference.items():
        got = row[field]
        if field in {"median", "q25", "q75", "p05", "p95", "max"}:
            match = round(got, digits) == round(expected, digits)
        else:
            match = got == expected
        checks.append({
            "proxy": name, "field": field, "paper": expected,
            "preserved_recomputed": got, "paper_display_match": bool(match),
        })
    return checks


def git_sha() -> str:
    p = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                       capture_output=True, text=True)
    return p.stdout.strip() if p.returncode == 0 else ""


def run(out: Path) -> dict:
    temp_data = {}
    raw_values = {"CLO": [], "MET": [], "Tsub": []}
    provenance = []
    primary_scope = {}
    external_audits = {}
    actual_split_ids = {}
    for split, expected in SPLITS.items():
        split_path = REPRO_ROOT / "splits" / ("validation_ids.csv" if split == "valid" else f"{split}_ids.csv")
        if not split_path.is_file():
            fail(f"Missing frozen split manifest: {split_path}")
        with split_path.open(encoding="utf-8-sig", newline="") as stream:
            split_rows = list(csv.DictReader(stream))
        expected_ids = [str(r["set_id"]) for r in split_rows]
        if len(expected_ids) != expected or len(set(expected_ids)) != expected:
            fail(f"Frozen {split} split has unexpected count / duplicate IDs")
        actual_split_ids[split] = set(expected_ids)
        temp_path = BASE / "temperature_results" / f"{split}_temperature.json"
        raw_temp = parse_records(temp_path, jsonl=False)
        temp_rows, temp_audit = index_records(raw_temp, temp_path)
        if set(temp_rows) != actual_split_ids[split]:
            fail(f"{split} Tsub IDs disagree with frozen split: "
                 f"missing={len(actual_split_ids[split] - set(temp_rows))}, "
                 f"unexpected={len(set(temp_rows) - actual_split_ids[split])}")
        temp_data[split] = temp_rows
        primary_scope[split] = {
            "expected_unique_set_ids": expected,
            "temperature_rows": len(temp_rows),
            "temperature_raw_records": len(raw_temp),
            "duplicate_audit": temp_audit,
            "matches_frozen_split_ids": True,
        }
        provenance.append({"path": str(temp_path.relative_to(ROOT)),
                           "sha256": sha256(temp_path),
                           "raw_records": len(raw_temp),
                           "unique_set_ids": len(temp_rows)})
        for row in raw_temp:
            sid = str(row["set_id"])
            raw_values["CLO"].append(numeric(row, "total_clo", temp_path, sid))
            raw_values["MET"].append(numeric(row, "met_value", temp_path, sid))
            raw_values["Tsub"].append(numeric(row, "TSUB_target_C", temp_path, sid))

    if len(set.union(*actual_split_ids.values())) != sum(SPLITS.values()):
        fail("Frozen train/valid/test split IDs overlap")

    values = {"CLO": [], "MET": [], "Tsub": []}
    for split in SPLITS:
        for sid, row in temp_data[split].items():
            path = BASE / "temperature_results" / f"{split}_temperature.json"
            values["CLO"].append(numeric(row, "total_clo", path, sid))
            values["MET"].append(numeric(row, "met_value", path, sid))
            values["Tsub"].append(numeric(row, "TSUB_target_C", path, sid))

    # Independent source audit; missing historical MET records remain reported
    # as missing instead of being secretly supplied by temperature.json.
    for split in SPLITS:
        t = temp_data[split]
        for source, relative, field, jsonl in [
            ("CLO", f"CLO_results/{split}_clo_results.jsonl", "total_clo", True),
            ("MET", f"MET_results/{split}_met.json", "met_value", False),
        ]:
            p = BASE / relative
            rows, source_audit = index_records(parse_records(p, jsonl=jsonl), p)
            matched_ids = set(rows) & set(t)
            mismatches = [
                sid for sid in matched_ids
                if not math.isclose(
                    numeric(rows[sid], field, p, sid),
                    numeric(t[sid], field, BASE / f"temperature_results/{split}_temperature.json", sid),
                    rel_tol=0, abs_tol=1e-9
                )
            ]
            key = f"{source}_{split}"
            external_audits[key] = {
                "file": str(p.relative_to(ROOT)),
                "rows": len(rows),
                "duplicate_audit": source_audit,
                "overlap_with_temperature": len(matched_ids),
                "missing_relative_to_temperature": len(set(t) - set(rows)),
                "extra_relative_to_temperature": len(set(rows) - set(t)),
                "value_mismatch_with_temperature": len(mismatches),
                "first_five_mismatched_ids": sorted(mismatches)[:5],
            }
            provenance.append({"path": str(p.relative_to(ROOT)),
                               "sha256": sha256(p),
                               "raw_records": source_audit["raw_records"],
                               "unique_set_ids": len(rows)})

    summary = [describe(values[proxy], proxy) for proxy in ("CLO", "MET", "Tsub")]
    # Sensitivity check: show what changes if identical duplicate file records
    # are counted as distinct observations. Not an alternative official result.
    raw_summary = [describe(raw_values[proxy], proxy) for proxy in ("CLO", "MET", "Tsub")]
    checks = []
    for row in summary:
        checks.extend(compare_display(row, PAPER[row["proxy"]], row["proxy"]))
    checks_ok = all(r["paper_display_match"] for r in checks)
    raw_checks = []
    for row in raw_summary:
        raw_checks.extend(compare_display(row, PAPER[row["proxy"]], row["proxy"]))
    sources_consistent = all(
        x["value_mismatch_with_temperature"] == 0 and x["extra_relative_to_temperature"] == 0
        for x in external_audits.values()
    )
    payload = {
        "classification": "independent preserved-output Table 4-3 distribution audit",
        "git_commit": git_sha(),
        "historical_source": (
            "01_資料建構_data_construction/clo_met_temperature/"
            "temperature_results/temperature.ipynb, step1_make_tsub.py cell"
        ),
        "primary_sample": (
            "One unique set_id per outfit from the three preserved "
            "*_temperature.json split files; only identical duplicate records "
            "may be collapsed and are explicitly inventoried."
        ),
        "primary_sample_note": (
            "Both raw-record and unique-outfit summaries are preserved to "
            "show duplicate-count sensitivity. The archived temperature rows "
            "contain CLO, MET, Tsub in one aligned set_id record; standalone "
            "CLO/MET files are cross-checked, not used to silently fill or "
            "replace missing source rows."
        ),
        "percentile_method": "numpy.quantile(method='linear'), explicitly declared audit choice",
        "source_files": provenance, "split_scope": primary_scope,
        "standalone_CLO_MET_source_audits": external_audits,
        "n_unique_primary_rows": sum(len(v) for v in temp_data.values()),
        "n_raw_primary_records": len(raw_values["CLO"]),
        "n_extra_identical_primary_records": (
            len(raw_values["CLO"]) - sum(len(v) for v in temp_data.values())
        ),
        "summary": summary, "paper_field_comparison": checks,
        "raw_record_sensitivity_summary": raw_summary,
        "raw_record_sensitivity_paper_comparison": raw_checks,
        "raw_record_sensitivity_not_official": True,
        "all_listed_paper_fields_match_display_precision": checks_ok,
        "standalone_source_values_consistent_on_overlap": sources_consistent,
        "historical_2025_runtime_or_selection_rule_exact": False,
        "interpretation": (
            "A displayed-value match reproduces Table 4-3 from preserved "
            "aligned proxy outputs; it does not prove identical historical "
            "2025 execution or fill gaps in standalone CLO/MET files."
        ),
    }
    out.mkdir(parents=True, exist_ok=True)
    for filename, rows in [
        ("table_4_3_recomputed.csv", summary),
        ("table_4_3_paper_field_comparison.csv", checks),
        ("table_4_3_raw_record_sensitivity.csv", raw_summary),
    ]:
        # CLO/MET/Tsub have different threshold columns.  Infer the union
        # of fields from *all* rows, not only the first (CLO) row.
        # Write via a temporary file to avoid leaving a truncated CSV when
        # serialization fails; replace only after the entire CSV is written.
        fields = list(dict.fromkeys(key for row in rows for key in row))
        if not fields:
            fail(f"No rows to write: {filename}")
        target = out / filename
        temporary = out / (filename + ".tmp")
        try:
            with temporary.open("w", encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=fields)
                writer.writeheader()
                writer.writerows(rows)
            temporary.replace(target)
        finally:
            temporary.unlink(missing_ok=True)
    (out / "table_4_3_manifest.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"[AUDIT] primary unique outfits={payload['n_unique_primary_rows']} / "
          f"{sum(SPLITS.values())}; raw records={payload['n_raw_primary_records']}; "
          f"extra identical records={payload['n_extra_identical_primary_records']}")
    for split, scope in primary_scope.items():
        d = scope["duplicate_audit"]
        print(f"[DUPLICATES] {split} raw={d['raw_records']} "
              f"unique={d['unique_set_ids']} repeated_ids={d['identical_duplicate_ids']} "
              f"extra_identical_records={d['extra_identical_records']}")
    for row in summary:
        print(f"[{row['proxy']}] n={row['n']} median={row['median']:.3f} "
              f"IQR=[{row['q25']:.3f},{row['q75']:.3f}] "
              f"P5-P95=[{row['p05']:.3f},{row['p95']:.3f}]")
    for key, v in external_audits.items():
        print(f"[SOURCE] {key}: rows={v['rows']} missing={v['missing_relative_to_temperature']} "
              f"extra={v['extra_relative_to_temperature']} value_mismatch={v['value_mismatch_with_temperature']}")
    print("[PAPER DISPLAY MATCH]", checks_ok)
    print("[OK]", out / "table_4_3_manifest.json")
    return payload


def self_test() -> None:
    a = describe([0.0, 1.0, 1.0, 2.0, 4.5], "CLO")
    assert a["n"] == 5 and a["median"] == 1 and a["count_gt_4"] == 1
    assert math.isclose(a["p05"], 0.2, rel_tol=0, abs_tol=1e-12), a["p05"]
    assert math.isclose(a["p95"], 4.0, rel_tol=0, abs_tol=1e-12), a["p95"]
    dummy = Path("synthetic_identical_and_conflicting_duplicates.json")
    indexed, duplicate_audit = index_records(
        [{"set_id": "x", "total_clo": 1.0},
         {"set_id": "x", "total_clo": 1.0},
         {"set_id": "y", "total_clo": 2.0}], dummy
    )
    assert len(indexed) == 2 and duplicate_audit["extra_identical_records"] == 1
    try:
        index_records(
            [{"set_id": "x", "total_clo": 1.0},
             {"set_id": "x", "total_clo": 2.0}], dummy
        )
    except SystemExit as exc:
        assert "Conflicting duplicate" in str(exc)
    else:
        raise AssertionError("Conflicting duplicates must block the audit")
    # Reproduce the previous serialization failure in memory: CLO and MET
    # contain different threshold columns; no column may be lost.
    from io import StringIO
    heterogeneous = [describe([0.0, 1.0, 4.5], "CLO"),
                     describe([1.0, 1.3, 12.0], "MET"),
                     describe([-22.0, 15.0, 23.0], "Tsub")]
    fields = list(dict.fromkeys(key for row in heterogeneous for key in row))
    assert {"count_gt_4", "count_eq_1", "count_gt_10",
            "count_lt_0", "count_lt_minus_20"}.issubset(fields)
    stream = StringIO()
    writer = csv.DictWriter(stream, fieldnames=fields)
    writer.writeheader()
    writer.writerows(heterogeneous)
    readback = list(csv.DictReader(StringIO(stream.getvalue())))
    assert len(readback) == 3
    assert readback[1]["count_gt_10"] == "1"
    assert readback[2]["count_lt_minus_20"] == "1"
    print("[PASS] synthetic quantile/threshold, duplicates and heterogeneous CSV self-test")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", help="Fresh audit directory; never overwrite archived table")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
    else:
        if not args.out_dir:
            parser.error("--out-dir is required unless --self-test is supplied")
        p = Path(args.out_dir).expanduser()
        run((p if p.is_absolute() else ROOT / p).resolve())
