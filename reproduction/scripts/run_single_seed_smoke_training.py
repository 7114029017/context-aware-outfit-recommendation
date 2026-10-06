#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import argparse
import datetime as dt
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile

from repro_paths import REPO_ROOT, REPRO_ROOT
import standard_decoder

ROOT = REPO_ROOT
MODEL_ROOT = ROOT / "02_模型訓練和驗證_model_training_validation"
SOURCE = MODEL_ROOT / "main_hybrid_attention_code"
FEATURES = MODEL_ROOT / "fashionclip_data"

VARIANTS = {
    "original": ("outfitUrlTitle", "encoded_outfitUrlTitle_en_fashionClip.pkl"),
    "context": ("NewoutfitUrlTitle", "encoded_NewoutfitUrlTitle_en_fashionClip.pkl"),
}

def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def run(cmd, cwd: Path, log_path: Path):
    print("+", " ".join(str(x) for x in cmd), flush=True)
    with log_path.open("w", encoding="utf-8") as log:
        p = subprocess.run(
            cmd,
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        log.write(p.stdout)
        print(p.stdout, end="")
    if p.returncode != 0:
        raise SystemExit(p.returncode)

def set_ids(polyvore_root: Path, split: str, n: int):
    import json as _json
    data = _json.loads((polyvore_root / "disjoint" / f"{split}.json").read_text(encoding="utf-8"))
    return [str(x["set_id"]) for x in data[:n]]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--polyvore-root", required=True)
    ap.add_argument("--variant", choices=sorted(VARIANTS), default="context")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--train-outfits", type=int, default=200)
    ap.add_argument("--valid-outfits", type=int, default=100)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()

    import torch
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for this smoke run; torch.cuda.is_available() is False.")
    if args.seed not in range(1, 6):
        raise SystemExit("--seed must be 1..5")
    if args.epochs < 1:
        raise SystemExit("--epochs must be >=1")

    polyvore_root = Path(args.polyvore_root).expanduser().resolve()
    required = [
        polyvore_root / "disjoint" / "train.json",
        polyvore_root / "disjoint" / "valid.json",
        polyvore_root / "disjoint" / "compatibility_valid.txt",
        polyvore_root / "disjoint" / "fill_in_blank_valid.json",
        polyvore_root / "polyvore_item_metadata.json",
    ]
    for p in required:
        if not p.exists():
            raise SystemExit(f"Missing required file: {p}")

    variant, outfit_feat = VARIANTS[args.variant]
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = Path(args.out_dir) if args.out_dir else REPRO_ROOT / "runs" / f"smoke_{args.variant}_seed{args.seed}_{stamp}"
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    repro_work = ROOT / ".repro_work"
    repro_work.mkdir(exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix="smoke_train_", dir=repro_work))

    manifest = {
        "classification": "non-scientific smoke training",
        "purpose": "Exercise real PO-D/features/CP->CIR training code paths on GPU before full retraining.",
        "git_head": subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip(),
        "variant_label": args.variant,
        "variant": variant,
        "outfit_feat_name": outfit_feat,
        "seed": args.seed,
        "train_outfits": args.train_outfits,
        "valid_outfits": args.valid_outfits,
        "epochs": args.epochs,
        "num_workers": args.num_workers,
        "torch": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0),
        "gpu_capability": list(torch.cuda.get_device_capability(0)),
        "cp_decoder": standard_decoder.MANIFEST,
        "source_modified_in_repository": False,
        "scientific_metrics_comparable_to_thesis": False,
        "notes": [
            "Only a working copy of the archived source is changed.",
            "Epoch count, subset size, and worker count are smoke-test controls, not thesis settings.",
            "CP uses the standardized decoder implementation (reproduction/docs/standardized_decoder.md).",
            "CIR uses the handoff retrieval architecture and initializes from the smoke CP checkpoint using train_cir.py's existing name/shape filter.",
        ],
    }

    try:
        src = work / "main_hybrid_attention_code"
        shutil.copytree(SOURCE, src)

        standard_decoder.apply(src / "outfit_transformer.py")

        # Explicit smoke-only config controls.
        cfg_path = src / "config" / "base_config.py"
        cfg = cfg_path.read_text(encoding="utf-8")
        cfg2 = cfg
        cfg2 = cfg2.replace("epochs = 100", f"epochs = {args.epochs}", 1)
        cfg2 = cfg2.replace("num_workers = multiprocessing.cpu_count()", f"num_workers = {args.num_workers}", 1)
        if cfg2 == cfg:
            raise SystemExit("Smoke config patch did not change copied base_config.py as expected.")
        cfg_path.write_text(cfg2, encoding="utf-8")

        os.symlink(FEATURES, work / "fashionclip_data", target_is_directory=True)
        (work / "polyvore_data").mkdir()
        os.symlink(polyvore_root, work / "polyvore_data" / "polyvore_outfits", target_is_directory=True)

        train_ids = set_ids(polyvore_root, "train", args.train_outfits)
        valid_ids = set_ids(polyvore_root, "valid", args.valid_outfits)
        subset = work / "smoke_subset_ids.txt"
        subset.write_text("\n".join(train_ids + valid_ids) + "\n", encoding="utf-8")
        shutil.copy2(subset, out_dir / "smoke_subset_ids.txt")

        exp = work / "experiments"
        exp.mkdir()

        py = sys.executable
        cp_log = out_dir / "cp_smoke.log"
        run([
            py, "train_cp.py",
            "--seed", str(args.seed),
            "--subset_ids_path", str(subset),
            "--variant", variant,
            "--outfit_feat_name", outfit_feat,
            "--exp_root", str(exp),
        ], cwd=src, log_path=cp_log)

        cp_dir = exp / f"cp_{variant}_subset_seed{args.seed}"
        cp_ckpt = cp_dir / "ckpt.pt"
        if not cp_ckpt.exists():
            raise SystemExit(f"CP smoke did not produce best checkpoint: {cp_ckpt}")

        cir_log = out_dir / "cir_smoke.log"
        run([
            py, "train_cir.py",
            "--seed", str(args.seed),
            "--subset_ids_path", str(subset),
            "--variant", variant,
            "--outfit_feat_name", outfit_feat,
            "--exp_root", str(exp),
            "--pretrained_cp_ckpt", str(cp_ckpt),
        ], cwd=src, log_path=cir_log)

        cir_dir = exp / f"cir_{variant}_subset_seed{args.seed}"
        cir_ckpt = cir_dir / "ckpt.pt"
        if not cir_ckpt.exists():
            raise SystemExit(f"CIR smoke did not produce best checkpoint: {cir_ckpt}")

        cp_out = out_dir / "cp_smoke_ckpt.pt"
        cir_out = out_dir / "cir_smoke_ckpt.pt"
        shutil.copy2(cp_ckpt, cp_out)
        shutil.copy2(cir_ckpt, cir_out)

        manifest["status"] = "passed"
        manifest["cp_checkpoint_sha256"] = sha256(cp_out)
        manifest["cir_checkpoint_sha256"] = sha256(cir_out)
        manifest["cp_log"] = str(cp_log)
        manifest["cir_log"] = str(cir_log)
        manifest["cp_checkpoint"] = str(cp_out)
        manifest["cir_checkpoint"] = str(cir_out)

        # Preserve training logs from the copied source's experiment directories.
        for d, prefix in [(cp_dir, "cp"), (cir_dir, "cir")]:
            for p in d.glob("*.log"):
                shutil.copy2(p, out_dir / f"{prefix}_{p.name}")

        print(json.dumps({
            "status": "passed",
            "variant": args.variant,
            "seed": args.seed,
            "epochs": args.epochs,
            "train_outfits": args.train_outfits,
            "valid_outfits": args.valid_outfits,
            "gpu": manifest["gpu"],
            "output": str(out_dir),
        }, ensure_ascii=False, indent=2))
    finally:
        (out_dir / "smoke_manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        shutil.rmtree(work, ignore_errors=True)

if __name__ == "__main__":
    main()
