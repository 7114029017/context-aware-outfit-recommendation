#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import argparse
import importlib.util
import json

ROOT = Path(__file__).resolve().parents[3]
REPRO_ROOT = ROOT / "reproduction"
REPRO_SCRIPTS = REPRO_ROOT / "scripts"
SOURCE = REPRO_SCRIPTS / "run_fair_subset_batch.py"


def load_module():
    spec = importlib.util.spec_from_file_location("fair_batch_validation", SOURCE)
    if spec is None or spec.loader is None:
        raise SystemExit(f"Cannot import: {SOURCE}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    ap = argparse.ArgumentParser(
        description="Validate 25 fair-subset outputs using the existing batch validator and emit a portable report."
    )
    ap.add_argument("--input-dir", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    base = Path(args.input_dir).expanduser()
    base = (base if base.is_absolute() else ROOT / base).resolve()
    out = Path(args.out).expanduser()
    out = (out if out.is_absolute() else ROOT / out).resolve()

    mod = load_module()
    mod.OUT_BASE = base

    tasks = [(v, s) for s in range(1, 6) for v in mod.VARIANTS]
    records = [mod.record_summary(v, s, verify_files=True) for v, s in tasks]
    comp = mod.assert_comparable(records)
    complete = sum(r.get("state") == "PASSED_CANDIDATE" for r in records)
    if complete != 25 or comp.get("verified_complete") != 25 or not comp.get("all_completed_shared_membership"):
        raise SystemExit("25-unit fair-subset validation did not pass")

    report = {
        "classification": (
            "reconstructed WOS fair-subset fresh 5-variant x 5-seed CP/CIR batch; "
            "NOT historical source exactness"
        ),
        "status": "PASSED_CANDIDATE_SOURCE",
        "total_units": 25,
        "completed_units": 25,
        "comparability": comp,
        "tasks": records,
        "limits": [
            "Historical fair-subset memberwise ID has not been recovered.",
            "CP decoder uses the standardized decoder implementation; the historical definition was not preserved.",
        ],
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"[OK] wrote {out}")


if __name__ == "__main__":
    main()
