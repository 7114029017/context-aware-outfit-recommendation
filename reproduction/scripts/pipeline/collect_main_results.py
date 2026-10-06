#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import argparse
import csv
import hashlib
import json

ROOT = Path(__file__).resolve().parents[3]
REPRO_ROOT = ROOT / "reproduction"
REPRO_SCRIPTS = REPRO_ROOT / "scripts"
VARIANTS = {
    "original": "outfitUrlTitle",
    "context": "NewoutfitUrlTitle",
}


def read_rows(path: Path):
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_rows(path: Path, rows):
    if not rows:
        raise SystemExit(f"No rows to write: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = []
    seen = set()
    for row in rows:
        for k in row:
            if k not in seen:
                seen.add(k)
                fields.append(k)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
        w.writeheader()
        w.writerows(rows)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def verify_input(path: Path, expected: str, name: str) -> None:
    if not path.is_file() or not isinstance(expected, str) or len(expected) != 64:
        raise SystemExit(f"Missing {name} or expected SHA-256: {path}")
    if sha256(path) != expected:
        raise SystemExit(f"{name} SHA-256 mismatch: {path}")


def portable_run_path(path: Path) -> str:
    """Keep readable evidence paths for repo-local AND external run roots."""
    resolved = path.expanduser().resolve()
    try:
        return str(resolved.relative_to(ROOT))
    except ValueError:
        return str(resolved)


def main():
    ap = argparse.ArgumentParser(description="Collect ten fresh full-retraining evaluations.")
    ap.add_argument("--input-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    base = Path(args.input_dir).expanduser()
    base = (base if base.is_absolute() else ROOT / base).resolve()
    out = Path(args.out_dir).expanduser()
    out = (out if out.is_absolute() else ROOT / out).resolve()

    cp = []
    cir = []
    manifest_rows = []
    for seed in range(1, 6):
        for label, source_variant in VARIANTS.items():
            run = base / f"{label}_seed{seed}"
            ev = run / "evaluation"
            mf = ev / "evaluation_manifest.json"
            if not mf.is_file():
                raise SystemExit(f"Missing evaluation manifest: {mf}")
            obj = json.loads(mf.read_text(encoding="utf-8"))
            if (
                obj.get("status") != "passed"
                or obj.get("variant_label") != label
                or int(obj.get("seed", -1)) != seed
            ):
                raise SystemExit(f"Evaluation identity mismatch: {mf}")

            train_path = run / "full_train_manifest.json"
            if not train_path.is_file():
                raise SystemExit(f"Missing training manifest: {train_path}")
            training = json.loads(train_path.read_text(encoding="utf-8"))
            expected_variant = {
                "original": ("outfitUrlTitle", "encoded_outfitUrlTitle_en_fashionClip.pkl"),
                "context": ("NewoutfitUrlTitle", "encoded_NewoutfitUrlTitle_en_fashionClip.pkl"),
            }
            expected_feature = expected_variant[label][1]
            if not (
                training.get("status") == "passed"
                and training.get("variant_label") == label
                and training.get("variant") == source_variant
                and training.get("outfit_feat_name") == expected_feature
                and int(training.get("seed", -1)) == seed
                and training.get("epochs") == 100
                and training.get("full_train_split") is True
                and training.get("full_valid_split") is True
                and obj.get("source_variant") == source_variant
                and obj.get("outfit_feat_name") == expected_feature
            ):
                raise SystemExit(f"Training/evaluation provenance mismatch: {run}")
            verify_input(train_path, obj.get("training_manifest_sha256"), "training manifest")
            for ckpt, field in (
                ("cp_best_ckpt.pt", "cp_checkpoint_sha256"),
                ("cir_best_ckpt.pt", "cir_checkpoint_sha256"),
            ):
                if obj.get(field) != training.get(field):
                    raise SystemExit(f"Checkpoint SHA differs between manifests: {run / ckpt}")
                verify_input(run / ckpt, obj.get(field), ckpt)
            verify_input(ev / "results_cp.csv", obj.get("results_cp_sha256"), "CP CSV")
            verify_input(ev / "results_cir.csv", obj.get("results_cir_sha256"), "CIR CSV")

            cp_rows = [
                r for r in read_rows(ev / "results_cp.csv")
                if r.get("variant") == source_variant
                and r.get("subset_tag") == "all"
                and int(r.get("seed", -1)) == seed
            ]
            cir_rows = [
                r for r in read_rows(ev / "results_cir.csv")
                if r.get("variant") == source_variant
                and r.get("subset_tag") == "all"
                and int(r.get("seed", -1)) == seed
                and r.get("run_tag") == "full_retrain"
            ]
            if len(cp_rows) != 1 or len(cir_rows) != 1:
                raise SystemExit(f"Expected one CP and CIR row: {run}")
            cp.append(cp_rows[0])
            cir.append(cir_rows[0])
            manifest_rows.append({
                "variant": label,
                "seed": seed,
                "training_manifest": portable_run_path(run / "full_train_manifest.json"),
                "evaluation_manifest": portable_run_path(mf),
            })

    write_rows(out / "results_cp.csv", cp)
    write_rows(out / "results_cir.csv", cir)
    write_rows(out / "run_index.csv", manifest_rows)
    print(f"[OK] collected {len(cp)} CP rows and {len(cir)} CIR rows -> {out}")


if __name__ == "__main__":
    main()
