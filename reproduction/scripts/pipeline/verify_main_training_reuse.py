#!/usr/bin/env python3
"""Verify a NEW integrated main run is safe to skip; no model loads/training.

A passed JSON flag alone is insufficient. Fail closed on identity, mode,
feature or checkpoint-content mismatch. No prior old-format run is
implicitly imported into the new run layout.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

VARIANTS = {
    "original": ("outfitUrlTitle", "encoded_outfitUrlTitle_en_fashionClip.pkl"),
    "context": ("NewoutfitUrlTitle", "encoded_NewoutfitUrlTitle_en_fashionClip.pkl"),
}


def digest(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def blocked(why: str) -> None:
    raise SystemExit("[REUSE BLOCKED] " + why)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--variant", required=True, choices=tuple(VARIANTS))
    ap.add_argument("--seed", type=int, required=True)
    args = ap.parse_args()
    run = Path(args.run_dir).expanduser().resolve()
    path = run / "full_train_manifest.json"
    if not path.is_file():
        blocked(f"Missing full training manifest: {path}")
    obj = json.loads(path.read_text(encoding="utf-8"))
    variant, feature = VARIANTS[args.variant]
    expected = {
        "status": "passed", "variant_label": args.variant,
        "variant": variant, "outfit_feat_name": feature,
        "seed": args.seed, "epochs": 100,
        "full_train_split": True, "full_valid_split": True,
        "source_modified_in_repository": False,
    }
    for k, expected_value in expected.items():
        if obj.get(k) != expected_value:
            blocked(f"{path}: {k} expected {expected_value!r}, got {obj.get(k)!r}")
    if args.seed not in range(1, 6):
        blocked(f"Seed outside main five-seed scope: {args.seed}")
    for ckpt, key in (
        ("cp_best_ckpt.pt", "cp_checkpoint_sha256"),
        ("cir_best_ckpt.pt", "cir_checkpoint_sha256"),
    ):
        p = run / ckpt
        expected_sha = obj.get(key)
        if not p.is_file() or p.stat().st_size == 0:
            blocked(f"Missing/empty checkpoint: {p}")
        if not isinstance(expected_sha, str) or len(expected_sha) != 64:
            blocked(f"Missing SHA-256 provenance for {p}")
        if digest(p) != expected_sha:
            blocked(f"Checkpoint bytes no longer match manifest SHA-256: {p}")
    print(f"[REUSE VERIFIED] {args.variant} seed{args.seed}: "
          "full-data 100 epochs; two checkpoint SHA-256 digests matched")


if __name__ == "__main__":
    main()
