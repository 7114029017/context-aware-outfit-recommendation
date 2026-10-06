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

import standard_decoder

ROOT = Path(__file__).resolve().parents[2]
REPRO_ROOT = ROOT / "reproduction"
REPRO_SCRIPTS = REPRO_ROOT / "scripts"
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

def stream_run(cmd, cwd: Path, log_path: Path):
    print("+", " ".join(str(x) for x in cmd), flush=True)
    with log_path.open("w", encoding="utf-8") as log:
        p = subprocess.Popen(
            cmd, cwd=cwd,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1,
        )
        assert p.stdout is not None
        for line in p.stdout:
            print(line, end="")
            log.write(line)
            log.flush()
        rc = p.wait()
    if rc != 0:
        raise SystemExit(rc)

def copy_logs(src_dir: Path, out_dir: Path, prefix: str):
    for p in src_dir.glob("*.log"):
        shutil.copy2(p, out_dir / f"{prefix}_{p.name}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--polyvore-root", required=True)
    ap.add_argument("--variant", choices=sorted(VARIANTS), default="context")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()

    import torch
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required; torch.cuda.is_available() is False.")
    if args.seed not in range(1, 6):
        raise SystemExit("--seed must be 1..5")

    polyvore_root = Path(args.polyvore_root).expanduser().resolve()
    for p in [
        polyvore_root / "disjoint" / "train.json",
        polyvore_root / "disjoint" / "valid.json",
        polyvore_root / "polyvore_item_metadata.json",
    ]:
        if not p.exists():
            raise SystemExit(f"Missing required file: {p}")

    variant, outfit_feat = VARIANTS[args.variant]
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = Path(args.out_dir) if args.out_dir else (
        REPRO_ROOT / "runs" / f"full_train_{args.variant}_seed{args.seed}_{stamp}"
    )
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    repro_work = REPRO_ROOT / ".work"
    repro_work.mkdir(exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix="full_train_", dir=repro_work))

    manifest = {
        "classification": "full single-seed retraining reproduction run",
        "variant_label": args.variant,
        "variant": variant,
        "outfit_feat_name": outfit_feat,
        "seed": args.seed,
        "full_train_split": True,
        "full_valid_split": True,
        "epochs": 100,
        "torch": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "cuda_available": bool(torch.cuda.is_available()),
        "gpu": torch.cuda.get_device_name(0),
        "gpu_capability": list(torch.cuda.get_device_capability(0)),
        "cp_decoder": standard_decoder.MANIFEST,
        "source_modified_in_repository": False,
        "preserve_handoff_training_behavior": True,
        "provenance_caveat": (
            "The historical DecoderLayerWithCrossAttn definition was not preserved; all 2026 experiments "
            "use the standardized decoder implementation (reproduction/docs/standardized_decoder.md)."
        ),
        "runtime_caveat": (
            "This run uses the current GPU environment and is not claimed to be byte-identical to the 2025 runtime."
        ),
    }

    try:
        src = work / "main_hybrid_attention_code"
        shutil.copytree(SOURCE, src)

        standard_decoder.apply(src / "outfit_transformer.py")

        os.symlink(FEATURES, work / "fashionclip_data", target_is_directory=True)
        (work / "polyvore_data").mkdir()
        os.symlink(polyvore_root, work / "polyvore_data" / "polyvore_outfits", target_is_directory=True)

        exp = work / "experiments"
        exp.mkdir()
        py = sys.executable

        cp_log = out_dir / "cp_full_train.log"
        stream_run([
            py, "train_cp.py",
            "--seed", str(args.seed),
            "--variant", variant,
            "--outfit_feat_name", outfit_feat,
            "--exp_root", str(exp),
        ], cwd=src, log_path=cp_log)

        cp_dir = exp / f"cp_{variant}_all_seed{args.seed}"
        cp_best = cp_dir / "ckpt.pt"
        if not cp_best.exists():
            raise SystemExit(f"CP full train did not produce best checkpoint: {cp_best}")

        cp_saved = out_dir / "cp_best_ckpt.pt"
        shutil.copy2(cp_best, cp_saved)
        copy_logs(cp_dir, out_dir, "cp")

        cir_log = out_dir / "cir_full_train.log"
        stream_run([
            py, "train_cir.py",
            "--seed", str(args.seed),
            "--variant", variant,
            "--outfit_feat_name", outfit_feat,
            "--exp_root", str(exp),
            "--pretrained_cp_ckpt", str(cp_best),
        ], cwd=src, log_path=cir_log)

        cir_dir = exp / f"cir_{variant}_all_seed{args.seed}"
        cir_best = cir_dir / "ckpt.pt"
        if not cir_best.exists():
            raise SystemExit(f"CIR full train did not produce best checkpoint: {cir_best}")

        cir_saved = out_dir / "cir_best_ckpt.pt"
        shutil.copy2(cir_best, cir_saved)
        copy_logs(cir_dir, out_dir, "cir")

        cp_obj = torch.load(cp_saved, map_location="cpu", weights_only=False)
        cir_obj = torch.load(cir_saved, map_location="cpu", weights_only=False)

        manifest.update({
            "status": "passed",
            "cp_best_epoch": cp_obj.get("epoch"),
            "cp_best_acc": cp_obj.get("best_acc"),
            "cir_best_epoch": cir_obj.get("epoch"),
            "cir_best_acc": cir_obj.get("best_acc"),
            "cp_checkpoint_sha256": sha256(cp_saved),
            "cir_checkpoint_sha256": sha256(cir_saved),
            "cp_checkpoint": str(cp_saved),
            "cir_checkpoint": str(cir_saved),
            "cp_stdout_log": str(cp_log),
            "cir_stdout_log": str(cir_log),
        })

        print(json.dumps({
            "status": "passed",
            "variant": args.variant,
            "seed": args.seed,
            "cp_best_epoch": manifest["cp_best_epoch"],
            "cp_best_acc": manifest["cp_best_acc"],
            "cir_best_epoch": manifest["cir_best_epoch"],
            "cir_best_acc": manifest["cir_best_acc"],
            "gpu": manifest["gpu"],
            "output": str(out_dir),
        }, ensure_ascii=False, indent=2))
    finally:
        (out_dir / "full_train_manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        shutil.rmtree(work, ignore_errors=True)

if __name__ == "__main__":
    main()
