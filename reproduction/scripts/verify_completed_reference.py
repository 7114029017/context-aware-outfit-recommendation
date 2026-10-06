#!/usr/bin/env python3
"""Strictly verify the preserved completed full reproduction before skipping training.

This verifier is intentionally read-only.  It ties the local 35-unit full run
to the committed reference seed index by SHA-256 of all 70 CP/CIR checkpoints,
and independently verifies every file listed in reproduction/results/SHA256SUMS.txt.
A local run is reusable only when every gate passes.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[2]
REPRO = ROOT / "reproduction"
RESULTS = REPRO / "results"
REFERENCE_ID = "reference_20260921T175217Z"
SEED_INDEX = RESULTS / "summary" / REFERENCE_ID / "reference_seed_index.csv"
RESULT_HASHES = RESULTS / "SHA256SUMS.txt"
RAW_STATUS = RESULTS / "raw" / REFERENCE_ID / "run_identity" / "RUN_STATUS.txt"
EXPECTED_TRAINING_COMMIT = "eb081e7a426cf5bd91acd8bb0f92a8398f08f45e"
EXPECTED_CONFIG_BUNDLE_SHA = "136b739400a6744fcb2965be7e51992810e84df4e6ff64d9767f82ee2580a7aa"
EXPECTED_CANDIDATE_SHA = "bd1d24067b8d5021d3dc1e90651855e9821b1df7b7bf7683ff55ee61c5a298e7"
SHA_RE = re.compile(r"^[0-9a-f]{64}$")


def fail(message: str) -> None:
    raise SystemExit("[REUSE BLOCKED] " + message)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_result_manifest() -> int:
    if not RESULT_HASHES.is_file():
        fail(f"missing result hash manifest: {RESULT_HASHES}")

    count = 0
    for line_no, line in enumerate(RESULT_HASHES.read_text(encoding="utf-8").splitlines(), 1):
        if not line:
            continue
        try:
            expected, rel = line.split("  ", 1)
        except ValueError:
            fail(f"malformed result hash line {line_no}")
        if not SHA_RE.fullmatch(expected):
            fail(f"invalid SHA-256 on result hash line {line_no}")
        path = RESULTS / rel
        if not path.is_file():
            fail(f"missing committed reference result: {path}")
        actual = sha256(path)
        if actual != expected:
            fail(f"reference result hash mismatch: {path}")
        count += 1

    if count != 201:
        fail(f"expected 201 committed reference-result hashes, got {count}")
    return count


def expected_keys() -> set[tuple[str, str, int]]:
    keys = {
        ("main", variant, seed)
        for variant in ("original", "context")
        for seed in range(1, 6)
    }
    keys |= {
        ("candidate_ablation", variant, seed)
        for variant in ("original", "context", "no_weather", "no_occasion", "no_style")
        for seed in range(1, 6)
    }
    return keys


def load_seed_index() -> tuple[list[dict[str, str]], set[str]]:
    if not SEED_INDEX.is_file():
        fail(f"missing seed index: {SEED_INDEX}")

    with SEED_INDEX.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != 35:
        fail(f"expected 35 reference training rows, got {len(rows)}")

    seen: set[tuple[str, str, int]] = set()
    checkpoint_shas: set[str] = set()

    for row in rows:
        try:
            key = (row["family"], row["variant"], int(row["seed"]))
        except (KeyError, ValueError):
            fail(f"invalid seed-index identity row: {row}")
        if key in seen:
            fail(f"duplicate seed-index identity: {key}")
        seen.add(key)

        expected_status = "passed" if row["family"] == "main" else "passed_candidate_source"
        if row.get("status") != expected_status:
            fail(f"{key}: expected status {expected_status!r}, got {row.get('status')!r}")
        if row.get("git_commit") != EXPECTED_TRAINING_COMMIT:
            fail(f"{key}: unexpected training commit")
        if row.get("documented_config_bundle_sha256") != EXPECTED_CONFIG_BUNDLE_SHA:
            fail(f"{key}: unexpected documented config bundle hash")
        if row["family"] == "candidate_ablation" and row.get("candidate_id_sha256") != EXPECTED_CANDIDATE_SHA:
            fail(f"{key}: candidate-subset SHA mismatch")

        for field in ("cp_checkpoint_sha256", "cir_checkpoint_sha256"):
            value = row.get(field, "")
            if not SHA_RE.fullmatch(value):
                fail(f"{key}: invalid {field}")
            checkpoint_shas.add(value)

    if seen != expected_keys():
        fail(f"seed-index scope mismatch; missing={sorted(expected_keys() - seen)} extra={sorted(seen - expected_keys())}")
    if len(checkpoint_shas) != 70:
        fail(f"expected 70 unique CP/CIR checkpoint identities, got {len(checkpoint_shas)}")
    return rows, checkpoint_shas


def verify_local_checkpoints(full_run: Path, required_shas: set[str]) -> int:
    checkpoint_paths: list[Path] = []
    for pattern in ("*.pt", "*.pth", "*.ckpt"):
        checkpoint_paths.extend(full_run.rglob(pattern))
    checkpoint_paths = sorted(set(checkpoint_paths))

    if len(checkpoint_paths) != 70:
        fail(f"expected 70 checkpoint-like files in completed full run, got {len(checkpoint_paths)}")

    observed: dict[str, Path] = {}
    for index, path in enumerate(checkpoint_paths, 1):
        digest = sha256(path)
        if digest in observed:
            fail(f"duplicate checkpoint bytes: {observed[digest]} and {path}")
        observed[digest] = path
        if index % 10 == 0 or index == len(checkpoint_paths):
            print(f"[CHECKPOINT HASH] {index}/{len(checkpoint_paths)}", flush=True)

    missing = required_shas - set(observed)
    unexpected = set(observed) - required_shas
    if missing or unexpected:
        fail(
            "local checkpoint identity set differs from reference seed index; "
            f"missing={len(missing)} unexpected={len(unexpected)}"
        )
    return len(observed)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full-run", required=True, type=Path)
    args = parser.parse_args()

    full_run = args.full_run.expanduser().resolve()
    if not full_run.is_dir():
        fail(f"completed full-run directory missing: {full_run}")

    status_path = full_run / "RUN_STATUS.txt"
    if not status_path.is_file() or status_path.read_text(encoding="utf-8").strip() != "PASSED":
        fail(f"completed full run is not PASSED: {status_path}")
    if not RAW_STATUS.is_file() or RAW_STATUS.read_text(encoding="utf-8").strip() != "PASSED":
        fail(f"committed reference raw status is not PASSED: {RAW_STATUS}")

    result_count = verify_result_manifest()
    _rows, checkpoint_shas = load_seed_index()
    checkpoint_count = verify_local_checkpoints(full_run, checkpoint_shas)

    print(f"[PASS] completed full run verified: {full_run}")
    print("[PASS] training units = 35 (10 main + 25 candidate ablation)")
    print(f"[PASS] checkpoint SHA-256 identities = {checkpoint_count}/70")
    print(f"[PASS] committed reference result hashes = {result_count}/201")
    print(f"[PASS] reference training commit = {EXPECTED_TRAINING_COMMIT}")


if __name__ == "__main__":
    main()
