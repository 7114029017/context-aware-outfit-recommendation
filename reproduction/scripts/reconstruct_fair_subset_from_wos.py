#!/usr/bin/env python3
"""Audit and reproducibly reconstruct the no-missing fair subset.

This is a new, explicitly labeled reconstruction of ID membership, NOT the
missing original no_missing_none_ids.txt. Never silently substitute count
matching for a historical file hash or an original generation rule.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
REPRO_ROOT = ROOT / "reproduction"
REPRO_SCRIPTS = REPRO_ROOT / "scripts"
DESC = ROOT / "01_資料建構_data_construction" / "generated_descriptions"
WOS = DESC / "02_三因子拆分_wos_factor_split" / "wos_split_results_v5_merged_retry_round3.jsonl"
ABLATION = DESC / "01_生成結果_generation_results" / "new_polyvore_outfit_titles_with_ablation.json"
FACTORS = ("weather", "occasion", "style")
ABLATION_KEYS = ("no_weather", "no_occasion", "no_style")
EXPECTED_SPLITS = {"train": 16995, "valid": 3000, "test": 15145}
EXPECTED_WOS = 35140
EXPECTED_CANDIDATES = 21903
EXPECTED_TEST_OVERLAP = 9930


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fail(reason: str) -> None:
    raise ValueError(reason)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--polyvore-root", required=True, help="Directory containing disjoint/{train,valid,test}.json")
    ap.add_argument(
        "--out-dir", default="reproduction/runs/fair_subset_wos_reconstruction",
        help="Dedicated output directory, never the missing historical ID-file location.",
    )
    args = ap.parse_args()

    polyvore = Path(args.polyvore_root).expanduser().resolve()
    out = Path(args.out_dir).expanduser()
    out = (out if out.is_absolute() else ROOT / out).resolve()
    if out.exists():
        fail(f"Refusing to overwrite existing audit output: {out}. Choose --out-dir for a fresh run.")

    inputs = [WOS, ABLATION] + [polyvore / "disjoint" / f"{split}.json" for split in EXPECTED_SPLITS]
    for source in inputs:
        if not source.is_file():
            fail(f"Missing required input: {source}")

    ablation = json.loads(ABLATION.read_text(encoding="utf-8"))
    if not isinstance(ablation, dict):
        fail("The ablation description artifact must be an ID-to-description mapping.")

    split_ids: dict[str, set[str]] = {}
    split_in_order: dict[str, list[str]] = {}
    for name, expected in EXPECTED_SPLITS.items():
        p = polyvore / "disjoint" / f"{name}.json"
        items = json.loads(p.read_text(encoding="utf-8"))
        if not isinstance(items, list):
            fail(f"Invalid split data shape: {p}")
        ids = [str(item["set_id"]) for item in items]
        if len(ids) != expected or len(set(ids)) != expected:
            fail(f"Unexpected {name} split cardinality or duplicate IDs: n={len(ids)}, unique={len(set(ids))}, expected={expected}")
        split_ids[name] = set(ids)
        split_in_order[name] = ids

    all_split_ids = set().union(*split_ids.values())
    if len(all_split_ids) != EXPECTED_WOS or sum(map(len, split_ids.values())) != EXPECTED_WOS:
        fail("Polyvore disjoint splits overlap or do not cover the expected 35,140 unique outfit IDs.")

    seen: set[str] = set()
    candidate_in_wos_order: list[str] = []
    patterns: Counter[str] = Counter()
    invalid = 0
    ablation_mismatch: Counter[str] = Counter()
    malformed_fragments = 0

    with WOS.open(encoding="utf-8") as f:
        for line_number, line in enumerate(f, 1):
            if not line.strip():
                fail(f"Empty WOS JSONL line at {line_number}")
            row = json.loads(line)
            sid = str(row["id"])
            if sid in seen:
                fail(f"Duplicate WOS ID at line {line_number}: {sid}")
            seen.add(sid)
            if row.get("valid") is not True:
                invalid += 1

            fragments = row.get("fragments")
            if not isinstance(fragments, dict):
                fail(f"Missing fragments mapping for {sid}")

            present = []
            for factor in FACTORS:
                values = fragments.get(factor)
                if not isinstance(values, list) or any(not isinstance(x, str) for x in values):
                    malformed_fragments += 1
                    present.append(False)
                else:
                    present.append(any(x.strip() for x in values))
            key = "".join("1" if p else "0" for p in present)
            patterns[key] += 1

            a = ablation.get(sid)
            if not isinstance(a, dict):
                fail(f"Ablation description absent or malformed for outfit {sid}")
            if row.get("title") != a.get("title"):
                ablation_mismatch["title"] += 1
            a_from_wos = row.get("title_ablation")
            if not isinstance(a_from_wos, dict):
                fail(f"Missing WOS title_ablation for outfit {sid}")
            for label in ABLATION_KEYS:
                if a_from_wos.get(label) != a.get(label):
                    ablation_mismatch[label] += 1
                if not isinstance(a.get(label), str) or not a[label].strip():
                    fail(f"Empty or non-string ablation description: {sid}/{label}")

            if all(present):
                candidate_in_wos_order.append(sid)

    candidate_ids = set(candidate_in_wos_order)
    coverage = {name: len(candidate_ids & ids) for name, ids in split_ids.items()}
    matches_split = seen == all_split_ids
    matches_ablation = set(ablation) == seen
    cardinality_ok = (
        len(seen) == EXPECTED_WOS
        and len(candidate_ids) == EXPECTED_CANDIDATES
        and coverage["test"] == EXPECTED_TEST_OVERLAP
    )
    structure_ok = (
        matches_split and matches_ablation and not invalid
        and not malformed_fragments and not ablation_mismatch
    )
    gates_passed = cardinality_ok and structure_ok

    report = {
        "classification": "reproducibly reconstructed fair-subset membership; NOT the archived original ID file",
        "candidate_rule": (
            "Select WOS v5 merged-retry rows whose weather, occasion, AND style "
            "fragment lists each contain at least one non-blank string."
        ),
        "rule_provenance": (
            "Rule inferred from archived WOS fragment schema and 21,903/9,930 historical "
            "cardinality checks; the original ID-generation source and historical ID-file SHA256 "
            "are not preserved, so exact historical membership is NOT established."
        ),
        "wos_records_unique": len(seen),
        "wos_all_rows_valid": invalid == 0,
        "wos_invalid_rows": invalid,
        "wos_malformed_fragment_rows": malformed_fragments,
        "wos_factor_presence_patterns": dict(sorted(patterns.items())),
        "ablation_description_count": len(ablation),
        "ablation_mismatch_by_field": dict(ablation_mismatch),
        "wos_id_membership_equals_split_union": matches_split,
        "ablation_id_membership_equals_wos": matches_ablation,
        "candidate_unique_ids": len(candidate_ids),
        "candidate_overlap_by_split": coverage,
        "historical_reference_counts": {
            "wos_all": EXPECTED_WOS,
            "fair_subset_total": EXPECTED_CANDIDATES,
            "fair_subset_test_overlap": EXPECTED_TEST_OVERLAP,
            "fair_subset_cir_evaluable_pairs": 3432,
        },
        "cir_evaluable_pairs": "NOT MEASURED: requires original subset-specific CIR candidate-pool/evaluation scope",
        "gates_passed": gates_passed,
        "source_files": [
            {"path": str(p), "bytes": p.stat().st_size, "sha256": sha256(p)}
            for p in inputs
        ],
        "id_file_order": "WOS JSONL source order, not claimed to match the missing historical file order",
        "training_warning": (
            "Do not treat this audit as proof of historical fair-subset identity or train models "
            "until train/validation filtering and CIR subset evaluation semantics are independently checked."
        ),
    }
    print(json.dumps({
        "classification": report["classification"],
        "candidate_rule": report["candidate_rule"],
        "patterns": report["wos_factor_presence_patterns"],
        "candidate_total": report["candidate_unique_ids"],
        "train_valid_test": coverage,
        "wos_equals_split_union": matches_split,
        "ablation_matches_wos": matches_ablation and not ablation_mismatch,
        "gates_passed": gates_passed,
        "output_dir": str(out),
    }, ensure_ascii=False, indent=2), flush=True)

    out.mkdir(parents=True)
    (out / "fair_subset_reconstruction_manifest.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    if not gates_passed:
        print("[BLOCKED] count/coverage/description identity gate failed; NO ID file was emitted.", file=sys.stderr)
        raise SystemExit(3)

    ids_path = out / "fair_subset_ids.txt"
    ids_path.write_text("".join(sid + "\n" for sid in candidate_in_wos_order), encoding="utf-8")
    id_files = [ids_path]
    for split_name in split_ids:
        p = out / f"{split_name}_ids.txt"
        p.write_text(
            "".join(sid + "\n" for sid in split_in_order[split_name] if sid in candidate_ids),
            encoding="utf-8",
        )
        id_files.append(p)
    sums = [f"{sha256(p)}  {p.name}" for p in sorted(id_files)]
    (out / "SHA256SUMS.txt").write_text("\n".join(sums) + "\n", encoding="utf-8")
    print(f"[AUDIT PASSED] reconstructed fair-subset ID file: {ids_path}")
    print("[PROVENANCE] Historical membership remains unverified without original ID-file hash or generation code.")


if __name__ == "__main__":
    main()
