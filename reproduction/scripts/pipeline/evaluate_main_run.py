#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import argparse
import csv
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[3]
REPRO_ROOT = ROOT / "reproduction"
REPRO_SCRIPTS = REPRO_ROOT / "scripts"
sys.path.insert(0, str(REPRO_SCRIPTS))
import standard_decoder  # noqa: E402
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


def read_csv(path: Path):
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def run(cmd, cwd: Path, log_path: Path):
    print("+", " ".join(map(str, cmd)), flush=True)
    with log_path.open("w", encoding="utf-8") as log:
        p = subprocess.Popen(
            cmd,
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert p.stdout is not None
        for line in p.stdout:
            print(line, end="")
            log.write(line)
            log.flush()
        rc = p.wait()
    if rc:
        raise SystemExit(rc)


def select_one(rows, variant: str, seed: int, *, cir: bool):
    candidates = [
        row for row in rows
        if row.get("variant") == variant
        and int(row.get("seed", -1)) == seed
        and row.get("subset_tag") == "all"
    ]
    if cir:
        tagged = [r for r in candidates if r.get("run_tag") == "full_retrain"]
        if tagged:
            candidates = tagged
    if len(candidates) != 1:
        raise SystemExit(
            f"Expected exactly one evaluation row for variant={variant}, seed={seed}, "
            f"cir={cir}; got {len(candidates)}"
        )
    return candidates[0]


def main():
    ap = argparse.ArgumentParser(
        description="Evaluate one fresh full-data CP/CIR run by invoking the archived handoff evaluators."
    )
    ap.add_argument("--polyvore-root", required=True)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--eval-device", choices=["auto", "cuda", "cpu"], default="auto")
    args = ap.parse_args()

    import torch

    run_dir = Path(args.run_dir).expanduser()
    run_dir = (run_dir if run_dir.is_absolute() else ROOT / run_dir).resolve()
    manifest_path = run_dir / "full_train_manifest.json"
    if not manifest_path.is_file():
        raise SystemExit(f"Missing training manifest: {manifest_path}")

    training = json.loads(manifest_path.read_text(encoding="utf-8"))
    if training.get("status") != "passed":
        raise SystemExit("Training manifest must have status=passed")

    variant_label = str(training["variant_label"])
    if variant_label not in VARIANTS:
        raise SystemExit(f"Unknown variant_label: {variant_label}")
    variant, outfit_feat = VARIANTS[variant_label]
    seed = int(training["seed"])
    if not (
        training.get("variant") == variant
        and training.get("outfit_feat_name") == outfit_feat
        and seed in range(1, 6)
        and training.get("epochs") == 100
        and training.get("full_train_split") is True
        and training.get("full_valid_split") is True
    ):
        raise SystemExit("Training manifest identity/full-data/feature config mismatch")

    cp_ckpt = run_dir / "cp_best_ckpt.pt"
    cir_ckpt = run_dir / "cir_best_ckpt.pt"
    for p, field in (
        (cp_ckpt, "cp_checkpoint_sha256"),
        (cir_ckpt, "cir_checkpoint_sha256"),
    ):
        if not p.is_file():
            raise SystemExit(f"Missing checkpoint: {p}")
        expected_sha = training.get(field)
        if not isinstance(expected_sha, str) or len(expected_sha) != 64:
            raise SystemExit(f"Training manifest missing valid checkpoint SHA: {p}")
        if sha256(p) != expected_sha:
            raise SystemExit(f"Checkpoint SHA mismatch: {p}")

    polyvore = Path(args.polyvore_root).expanduser().resolve()
    required = [
        polyvore / "disjoint" / "train.json",
        polyvore / "disjoint" / "test.json",
        polyvore / "disjoint" / "fill_in_blank_test.json",
        polyvore / "polyvore_item_metadata.json",
    ]
    for p in required:
        if not p.is_file():
            raise SystemExit(f"Missing PO-D input: {p}")

    out_dir = Path(args.out_dir).expanduser() if args.out_dir else run_dir / "evaluation"
    out_dir = (out_dir if out_dir.is_absolute() else ROOT / out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    done = out_dir / "evaluation_manifest.json"
    if done.is_file():
        old = json.loads(done.read_text(encoding="utf-8"))
        cp_existing = out_dir / "results_cp.csv"
        cir_existing = out_dir / "results_cir.csv"
        def verified_existing_csv(path: Path, hash_key: str) -> bool:
            expected = old.get(hash_key)
            return (
                path.is_file() and path.stat().st_size > 0
                and isinstance(expected, str) and len(expected) == 64
                and sha256(path) == expected
            )

        if (
            old.get("status") == "passed"
            and old.get("variant_label") == variant_label
            and old.get("source_variant") == variant
            and old.get("outfit_feat_name") == outfit_feat
            and int(old.get("seed", -1)) == seed
            and old.get("training_manifest_sha256") == sha256(manifest_path)
            and old.get("cp_checkpoint_sha256") == training["cp_checkpoint_sha256"]
            and old.get("cir_checkpoint_sha256") == training["cir_checkpoint_sha256"]
            and (args.eval_device == "auto" or old.get("eval_device") == args.eval_device)
            and verified_existing_csv(cp_existing, "results_cp_sha256")
            and verified_existing_csv(cir_existing, "results_cir_sha256")
        ):
            # Check that hashed CSV contents still contain the exact
            # seed/variant/full-data scope, not merely arbitrary bytes.
            try:
                select_one(read_csv(cp_existing), variant, seed, cir=False)
                select_one(read_csv(cir_existing), variant, seed, cir=True)
            except (ValueError, KeyError, SystemExit):
                pass
            else:
                print(f"[SKIP VERIFIED] existing evaluation with matching manifest/checkpoint/CSV SHA: {done}")
                return
        print(f"[RE-EVALUATE] existing evaluation is stale or incomplete: {done}", flush=True)

    if args.eval_device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.eval_device
    if device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but torch.cuda.is_available() is False")

    work_root = REPRO_ROOT / ".work"
    work_root.mkdir(exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix="main_eval_", dir=work_root))

    result = {
        "status": "started",
        "variant_label": variant_label,
        "source_variant": variant,
        "outfit_feat_name": outfit_feat,
        "seed": seed,
        "training_manifest": str(manifest_path),
        "training_manifest_sha256": sha256(manifest_path),
        "cp_checkpoint_sha256": sha256(cp_ckpt),
        "cir_checkpoint_sha256": sha256(cir_ckpt),
        "eval_device": device,
        "archived_source_root": str(SOURCE.relative_to(ROOT)),
        "source_modified_in_repository": False,
        "cp_decoder": standard_decoder.MANIFEST,
    }

    try:
        src = work / "main_hybrid_attention_code"
        shutil.copytree(SOURCE, src)

        standard_decoder.apply(src / "outfit_transformer.py")

        if device == "cpu":
            cfg = src / "config" / "base_config.py"
            old = cfg.read_text(encoding="utf-8")
            new = old.replace("device_type = 'cuda'", "device_type = 'cpu'", 1)
            if new == old:
                raise SystemExit("Could not switch the working-copy evaluation config to CPU")
            cfg.write_text(new, encoding="utf-8")

        os.symlink(FEATURES, work / "fashionclip_data", target_is_directory=True)
        (work / "polyvore_data").mkdir()
        os.symlink(polyvore, work / "polyvore_data" / "polyvore_outfits", target_is_directory=True)

        exp = work / "experiments"
        cp_dir = exp / f"cp_{variant}_all_seed{seed}"
        cir_dir = exp / f"cir_{variant}_all_seed{seed}"
        cp_dir.mkdir(parents=True)
        cir_dir.mkdir(parents=True)
        os.symlink(cp_ckpt, cp_dir / "ckpt.pt")
        os.symlink(cir_ckpt, cir_dir / "ckpt.pt")

        py = sys.executable
        run(
            [
                py, "evaluate_cp.py",
                "--seed", str(seed),
                "--exp_root", str(exp),
                "--variant", variant,
                "--outfit_feat_name", outfit_feat,
            ],
            src,
            out_dir / "cp_evaluate.log",
        )
        run(
            [
                py, "evaluate_cir.py",
                "--seed", str(seed),
                "--exp_root", str(exp),
                "--variant", variant,
                "--outfit_feat_name", outfit_feat,
                "--run_tag", "full_retrain",
            ],
            src,
            out_dir / "cir_evaluate.log",
        )

        cp_src = exp / "results_cp.csv"
        cir_src = exp / "results_cir.csv"
        if not cp_src.is_file() or not cir_src.is_file():
            raise SystemExit("Archived evaluators did not produce expected results CSV files")

        cp_rows = read_csv(cp_src)
        cir_rows = read_csv(cir_src)
        cp_row = select_one(cp_rows, variant, seed, cir=False)
        cir_row = select_one(cir_rows, variant, seed, cir=True)

        shutil.copy2(cp_src, out_dir / "results_cp.csv")
        shutil.copy2(cir_src, out_dir / "results_cir.csv")

        result.update({
            "status": "passed",
            "cp": {
                "auc": float(cp_row["auc"]),
                "fitb_acc": float(cp_row["fitb_acc"]),
            },
            "cir": {
                key: float(cir_row[key])
                for key in (
                    "recall_at_1", "recall_at_3", "recall_at_5",
                    "recall_at_10", "recall_at_30", "recall_at_50",
                )
            },
            "results_cp_sha256": sha256(out_dir / "results_cp.csv"),
            "results_cir_sha256": sha256(out_dir / "results_cir.csv"),
        })
        done.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(result, ensure_ascii=False, indent=2))
    finally:
        shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    main()
