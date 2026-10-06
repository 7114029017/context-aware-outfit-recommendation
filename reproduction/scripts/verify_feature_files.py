#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import argparse
import hashlib
import json
import sys

from repro_paths import REPO_ROOT

ROOT = REPO_ROOT
FEATURE_DIR = ROOT / "02_模型訓練和驗證_model_training_validation" / "fashionclip_data"

EXPECTED = {
    "encoded_NewoutfitUrlTitle_en_fashionClip.pkl": (73590756, "f4d731433ef9182e16518ac7757d3aad530b362a807e0196fd79c4fa6db21ebb"),
    "encoded_category_distiluse-base-multilingual-cased-v2.pkl": (525651371, "03cb9b3d81c09f54867b00ea7c4c7bd1ccbc842555e19f757288a29f9f0537b8"),
    "encoded_no_occasion_outfitUrlTitle_en_fashionClip.pkl": (73590756, "474a6f7f1eceaeaacc8fbd173f13148385ae25380fdea1590fbc02543a1beb08"),
    "encoded_no_style_outfitUrlTitle_en_fashionClip.pkl": (73590756, "e54733fdd6a262d796b7d96749435e8c55a8ca1e2de726b0335d5b0f1ae72fcf"),
    "encoded_no_weather_outfitUrlTitle_en_fashionClip.pkl": (73590756, "f8be5098f0cd8d660522169dbd32989a76db3f6aa670ec1fb4f962ee9999783b"),
    "encoded_outfitUrlTitle_en_fashionClip.pkl": (143049389, "fb74f2c5638c175f897d07dcde5e825359127b42b330fe28783b8f50b21494a6"),
    "encoded_title_description_distiluse-base-multilingual-cased-v2.pkl": (525651371, "9e425f7e62aed5828d39cf64d4647b973d278f4579c39fccc8011c8660b201c4"),
    "img_feats_fashionClip.pkl": (525651371, "aa4859f4440e6c96b11314da30435907cf029460352318ba639a8ab41699efce"),
}

def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

ap = argparse.ArgumentParser()
ap.add_argument("--out", default=None)
args = ap.parse_args()

failed = False
records = []
for name, (expected_size, expected_hash) in EXPECTED.items():
    path = FEATURE_DIR / name
    rec = {"file": name, "expected_bytes": expected_size, "expected_sha256": expected_hash}
    if not path.exists():
        print(f"[MISSING] {name}")
        rec["status"] = "missing"
        records.append(rec)
        failed = True
        continue
    size = path.stat().st_size
    head = path.read_bytes()[:128]
    rec["observed_bytes"] = size
    if b"git-lfs.github.com/spec/v1" in head:
        print(f"[LFS POINTER] {name}: run 'git lfs pull'")
        rec["status"] = "lfs_pointer"
        records.append(rec)
        failed = True
        continue
    if size != expected_size:
        print(f"[SIZE MISMATCH] {name}: got={size} expected={expected_size}")
        rec["status"] = "size_mismatch"
        records.append(rec)
        failed = True
        continue
    digest = sha256(path)
    rec["observed_sha256"] = digest
    if digest != expected_hash:
        print(f"[HASH MISMATCH] {name}: got={digest} expected={expected_hash}")
        rec["status"] = "hash_mismatch"
        records.append(rec)
        failed = True
        continue
    rec["status"] = "ok"
    records.append(rec)
    print(f"[OK] {name}  bytes={size}  sha256={digest}")

report = {
    "all_features_ok": not failed,
    "n_expected": len(EXPECTED),
    "n_ok": sum(r.get("status") == "ok" for r in records),
    "files": records,
}
if args.out:
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
sys.exit(1 if failed else 0)
