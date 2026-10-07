#!/usr/bin/env python3
"""Per-query CIR results of the 10 main units of a completed full run (GPU).

The full run's main evaluation (pipeline/evaluate_main_run.py) saved only
Recall@k for each unit. This step evaluates the 10 main CIR checkpoints again
with the same archived evaluator, the same working-copy setup (standardized
decoder, feature and data links) and the same arguments, and adds
--save_detail, which writes the rank, hits and top-10 items of each of the
9,311 queries. Each unit must reproduce the run's stored results_cir.csv
exactly, and its per-query hits must average to those recalls. The run folder
is only read.

The official run evaluated on CUDA with float16 autocast. A CPU evaluation runs
in float32 and is not expected to match; it is accepted only with --eval-device
cpu and is then marked as a check run. --subset-ids FILE restricts the queries
(and the candidate pools, as the archived evaluator does) for a quick plumbing
test; such a run is never compared with the stored results.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from _ext import MAIN_UNITS, MODEL_CODE, FEATURES, REPRO, SCRIPTS, local_path, output_dir, passed_run, sha256, \
    write_json

sys.path.insert(0, str(SCRIPTS))
import standard_decoder  # noqa: E402

RECALLS = ("recall_at_1", "recall_at_3", "recall_at_5", "recall_at_10", "recall_at_30", "recall_at_50")
HITS = ("hit@1", "hit@3", "hit@5", "hit@10", "hit@30", "hit@50")


def stored_recalls(path: Path, variant: str, seed: int) -> dict[str, str]:
    with path.open(encoding="utf-8-sig", newline="") as f:
        rows = [r for r in csv.DictReader(f) if r["variant"] == variant and int(r["seed"]) == seed
                and r["subset_tag"] == "all" and r.get("run_tag") == "full_retrain"]
    if len(rows) != 1:
        raise SystemExit(f"[EXTENSION BLOCKED] expected one full_retrain row in {path}, found {len(rows)}")
    return {k: rows[0][k] for k in RECALLS}


def evaluate_unit(run_root: Path, unit: str, polyvore: Path, out: Path, device: str, subset_ids: Path | None) -> dict:
    variant, outfit_feat = MAIN_UNITS[unit]
    seed = int(unit.rsplit("seed", 1)[1])
    unit_dir = run_root / "main" / unit
    training = json.loads((unit_dir / "full_train_manifest.json").read_text(encoding="utf-8"))
    ckpt = unit_dir / "cir_best_ckpt.pt"
    if training.get("status") != "passed" or int(training.get("seed", -1)) != seed:
        raise SystemExit(f"[EXTENSION BLOCKED] {unit}: training manifest is not a passed seed-{seed} unit")
    if sha256(ckpt) != training.get("cir_checkpoint_sha256"):
        raise SystemExit(f"[EXTENSION BLOCKED] {unit}: CIR checkpoint SHA-256 differs from its training manifest")
    target = out / unit
    target.mkdir(parents=True, exist_ok=True)
    detail = target / "detail_cir_main.csv"
    work_root = REPRO / ".work"
    work_root.mkdir(exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix="main_cir_per_query_", dir=work_root))
    try:
        # The same working copy as pipeline/evaluate_main_run.py.
        src = work / "main_hybrid_attention_code"
        shutil.copytree(MODEL_CODE, src)
        standard_decoder.apply(src / "outfit_transformer.py")
        if device == "cpu":
            cfg = src / "config" / "base_config.py"
            text = cfg.read_text(encoding="utf-8")
            switched = text.replace("device_type = 'cuda'", "device_type = 'cpu'", 1)
            if switched == text:
                raise SystemExit("[EXTENSION BLOCKED] could not switch the working-copy config to CPU")
            cfg.write_text(switched, encoding="utf-8")
        os.symlink(FEATURES, work / "fashionclip_data", target_is_directory=True)
        (work / "polyvore_data").mkdir()
        os.symlink(polyvore, work / "polyvore_data" / "polyvore_outfits", target_is_directory=True)
        exp = work / "experiments"
        scope = "subset" if subset_ids else "all"
        (exp / f"cir_{variant}_{scope}_seed{seed}").mkdir(parents=True)
        os.symlink(ckpt, exp / f"cir_{variant}_{scope}_seed{seed}" / "ckpt.pt")
        cmd = [sys.executable, "evaluate_cir.py", "--seed", str(seed), "--exp_root", str(exp), "--variant", variant,
               "--outfit_feat_name", outfit_feat, "--run_tag", "full_retrain", "--save_detail",
               "--detail_file", str(detail)]
        if subset_ids:
            cmd += ["--subset_ids_path", str(subset_ids)]
        print(f"[MAIN CIR PER QUERY] {unit} on {device}", flush=True)
        with (target / "cir_evaluate.log").open("w", encoding="utf-8") as log:
            code = subprocess.run(cmd, cwd=src, stdout=log, stderr=subprocess.STDOUT).returncode
        if code:
            raise SystemExit(f"[EXTENSION FAILED] {unit}: evaluate_cir.py exited with {code}; see {target}/cir_evaluate.log")
        shutil.copy2(exp / "results_cir.csv", target / "results_cir.csv")
    finally:
        shutil.rmtree(work, ignore_errors=True)

    fresh = stored_recalls(target / "results_cir.csv", variant, seed) if not subset_ids else None
    with detail.open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    n = len(rows)
    hits_match = bool(fresh) and all(sum(int(r[h]) for r in rows) / n == float(fresh[k]) for h, k in zip(HITS, RECALLS))
    rank_ok = all((int(r["rank"]) <= int(h[4:])) == bool(int(r[h])) for r in rows for h in HITS)
    record = {"unit": unit, "seed": seed, "variant": variant, "device": device, "queries": n,
              "cir_checkpoint_sha256": training["cir_checkpoint_sha256"], "detail_sha256": sha256(detail),
              "hits_equal_ranks": rank_ok, "subset_ids": str(subset_ids) if subset_ids else None}
    if fresh is not None:
        stored = stored_recalls(unit_dir / "evaluation" / "results_cir.csv", variant, seed)
        record.update({"recalls": fresh, "recalls_equal_stored": fresh == stored, "hits_average_to_recalls": hits_match})
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-root", type=Path, required=True, help="completed full run folder with the checkpoints")
    parser.add_argument("--out-dir", type=Path, default=None, help="default: <run>/extensions/main_cir_per_query")
    parser.add_argument("--polyvore-root", default=None)
    parser.add_argument("--units", default=",".join(MAIN_UNITS), help="comma-separated, e.g. original_seed1")
    parser.add_argument("--eval-device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--subset-ids", type=Path, default=None, help="plumbing test only: a file of set IDs")
    args = parser.parse_args()

    import torch
    if args.eval_device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("[EXTENSION BLOCKED] CUDA is not available; this step needs the GPU (or --eval-device cpu)")
    run_root = passed_run(args.run_root)
    out = output_dir(args.out_dir or run_root / "extensions" / "main_cir_per_query", run_root)
    polyvore = local_path("polyvore_root", args.polyvore_root)
    if polyvore is None or not (polyvore / "disjoint" / "test.json").is_file():
        raise SystemExit("[EXTENSION BLOCKED] Polyvore root not found; pass --polyvore-root or run bootstrap_data.sh")
    units = [u.strip() for u in args.units.split(",") if u.strip()]
    unknown = [u for u in units if u not in MAIN_UNITS]
    if unknown:
        raise SystemExit(f"[EXTENSION BLOCKED] unknown units: {unknown}")

    records = [evaluate_unit(run_root, u, polyvore, out, args.eval_device, args.subset_ids) for u in units]
    exact = all(r.get("recalls_equal_stored") and r.get("hits_average_to_recalls") and r["hits_equal_ranks"]
                for r in records)
    status = ("check_only" if args.subset_ids or args.eval_device == "cpu" else "passed" if exact else "failed")
    write_json(out / "manifest.json", {"run": run_root.name, "status": status, "units": records,
                                       "torch": torch.__version__, "cuda_device": (torch.cuda.get_device_name(0)
                                       if args.eval_device == "cuda" else None)})
    for r in records:
        print(f"[MAIN CIR PER QUERY] {r['unit']}: {r['queries']} queries; recalls equal stored: "
              f"{r.get('recalls_equal_stored')}; hits average to recalls: {r.get('hits_average_to_recalls')}")
    print(f"[MAIN CIR PER QUERY] {status}: {out}")
    if status == "failed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
