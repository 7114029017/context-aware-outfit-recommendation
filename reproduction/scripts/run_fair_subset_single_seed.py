#!/usr/bin/env python3
"""FRESH CP -> CIR training/evaluation on the reconstructed fair subset, one variant/seed.

Defaults to read-only --dry-run. Requires explicit --run --acknowledge-candidate-source
for GPU work. Stage-level resume only: a failed/incomplete CP or CIR 100-epoch
stage is restarted from epoch 0, while a *verified completed* CP stage is reused.
No historical checkpoint is used as training initialization.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time

import standard_decoder

ROOT = Path(__file__).resolve().parents[2]
REPRO_ROOT = ROOT / "reproduction"
REPRO_SCRIPTS = REPRO_ROOT / "scripts"
MODEL = ROOT / "02_模型訓練和驗證_model_training_validation"
SOURCE = MODEL / "main_hybrid_attention_code"
FEATURES = MODEL / "fashionclip_data"
SCOPE = REPRO_ROOT / "splits/fair_subset/fair_subset_cir_scope_audit.json"
SUBSET = REPRO_ROOT / "splits/fair_subset/fair_subset_ids.txt"
RECON = REPRO_ROOT / "splits/fair_subset/fair_subset_reconstruction_manifest.json"
EXPECTED_SCOPE = 3432
VARIANTS = {
    "original": ("outfitUrlTitle", "encoded_outfitUrlTitle_en_fashionClip.pkl"),
    "context": ("NewoutfitUrlTitle", "encoded_NewoutfitUrlTitle_en_fashionClip.pkl"),
    "no_weather": ("no_weather", "encoded_no_weather_outfitUrlTitle_en_fashionClip.pkl"),
    "no_occasion": ("no_occasion", "encoded_no_occasion_outfitUrlTitle_en_fashionClip.pkl"),
    "no_style": ("no_style", "encoded_no_style_outfitUrlTitle_en_fashionClip.pkl"),
}
CP_METRICS = ("auc", "fitb_acc")
CIR_METRICS = ("recall_at_1", "recall_at_3", "recall_at_5", "recall_at_10", "recall_at_30", "recall_at_50")


def abort(msg):
    raise SystemExit("[BLOCKED] " + msg)


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for part in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(part)
    return h.hexdigest()


def load(path):
    if not path.is_file():
        abort(f"Required file missing: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def atomic(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    draft = path.with_name(path.name + ".writing")
    draft.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(draft, path)


def assert_sha_entries(records):
    if not isinstance(records, list) or not records:
        abort("Input manifest has no source hash entries")
    for item in records:
        p = Path(item["path"])
        if not p.is_file() or digest(p) != item["sha256"]:
            abort(f"Input source missing or changed: {p}")


def verify_gate(polyvore, subset_path, recon_path, scope_path):
    recon = load(recon_path)
    scope = load(scope_path)
    if not (subset_path.is_file() and recon.get("gates_passed") is True
            and recon.get("candidate_unique_ids") == 21903
            and scope.get("gates_passed") is True
            and scope.get("candidate_sha256") == digest(subset_path)
            and scope.get("observed_preserved_evaluator_semantics", {}).get("evaluable_questions") == EXPECTED_SCOPE
            and scope.get("observed_preserved_evaluator_semantics", {}).get("question_parse_anomaly_count") == 0):
        abort("WOS candidate or CIR 3432-query gate failed")
    assert_sha_entries(recon.get("source_files"))
    assert_sha_entries(scope.get("inputs"))
    expected = {"train": 10225, "valid": 1748, "test": 9930}
    if recon.get("candidate_overlap_by_split") != expected or scope.get("subset_coverage") != expected:
        abort("Train/valid/test fair-subset counts differ from gate")
    required = [polyvore / "disjoint" / (n + ".json") for n in ("train", "valid", "test")]
    required += [polyvore / "polyvore_item_metadata.json"]
    if not all(p.is_file() for p in required):
        abort("Polyvore disjoint data or metadata missing")
    return {
        "candidate_id_sha256": digest(subset_path),
        "candidate_reconstruction_manifest_sha256": digest(recon_path),
        "cir_3432_scope_report_sha256": digest(scope_path),
        "input_paths": {str(p): digest(p) for p in required},
    }


def checked_checkpoint(path):
    if not path.is_file():
        return None
    # Corrupt/mismatched .pt cannot be promoted to a completed stage.
    import torch
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(ckpt, dict) or not isinstance(ckpt.get("model"), dict):
        abort(f"Malformed checkpoint: {path}")
    epoch = ckpt.get("epoch")
    if not isinstance(epoch, int) or not 0 <= epoch < 100:
        abort(f"Unexpected best epoch: {path}: {epoch}")
    return {"sha256": digest(path), "best_epoch": epoch, "best_acc": float(ckpt.get("best_acc"))}


def verify_checkpoint(path, recorded):
    if not isinstance(recorded, dict) or not path.is_file():
        return False
    if digest(path) != recorded.get("sha256"):
        abort(f"Previously completed checkpoint was changed: {path}")
    info = checked_checkpoint(path)
    return info is not None and info["best_epoch"] == recorded.get("best_epoch")


def read_one_result(path, cols, *, variant, feat, seed, cir=False):
    if not path.is_file():
        abort(f"Missing evaluation result: {path}")
    with path.open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    if len(rows) != 1:
        abort(f"Expected exactly one isolated eval result: {path}; got {len(rows)}")
    row = rows[0]
    if (row.get("variant"), row.get("outfit_feat_name"), row.get("subset_tag")) != (variant, feat, "subset"):
        abort(f"Evaluation identity mismatch: {path}")
    if int(row["seed"]) != seed or (cir and row.get("run_tag") != "none_subset"):
        abort(f"Evaluation seed/run_tag mismatch: {path}")
    values = {key: float(row[key]) for key in cols}
    if any(not (0 <= val <= 1) for val in values.values()):
        abort(f"Metrics out of range: {path}")
    return values


def verify_eval(out, state, variant, feat, seed):
    eval_record = state.get("evaluation")
    if not isinstance(eval_record, dict):
        return False
    cp = out / "results_cp_fresh_subset.csv"
    cir = out / "results_cir_fresh_subset.csv"
    detail = out / "detail_cir_fresh_subset.csv"
    for p in (cp, cir, detail):
        if not p.is_file() or digest(p) != eval_record.get("sha256", {}).get(p.name):
            abort(f"Previous completed evaluation artifact was missing/altered: {p}")
    read_one_result(cp, CP_METRICS, variant=variant, feat=feat, seed=seed)
    read_one_result(cir, CIR_METRICS, variant=variant, feat=feat, seed=seed, cir=True)
    with detail.open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    if len(rows) != EXPECTED_SCOPE or len({str(r["set_id"]) for r in rows}) != EXPECTED_SCOPE:
        abort("CIR detailed evaluation denominator/unique ID gate failed")
    return True


def run_stream(cmd, cwd, log_path):
    print("[EXEC]", " ".join(map(str, cmd)), flush=True)
    with log_path.open("w", encoding="utf-8", buffering=1) as f:
        p = subprocess.Popen(cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                             text=True, bufsize=1)
        try:
            for line in p.stdout:
                print(line, end="", flush=True)
                f.write(line)
            rc = p.wait()
        except BaseException:
            if p.poll() is None:
                p.terminate()
                try:
                    p.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    p.kill()
                    p.wait()
            raise
    if rc:
        abort(f"Experiment subprocess failed rc={rc}; check {log_path}")


def stage(state, stage_key, checkpoint_path, func, state_path):
    record = state.get(stage_key)
    if record and verify_checkpoint(checkpoint_path, record):
        print(f"[SKIP VERIFIED] {stage_key} {checkpoint_path}", flush=True)
        return
    if record:
        abort(f"{stage_key} completed marker exists but checkpoint verification failed")
    state["active_stage"] = stage_key
    atomic(state_path, state)
    try:
        func()
        info = checked_checkpoint(checkpoint_path)
        if info is None:
            abort(f"{stage_key} did not save best checkpoint")
        state[stage_key] = info
        state["active_stage"] = None
        atomic(state_path, state)
        print(f"[DONE] {stage_key} best_epoch={info['best_epoch']}, sha256={info['sha256']}", flush=True)
    except BaseException:
        state["active_stage"] = stage_key + "_interrupted_stage_must_restart_from_epoch_0"
        atomic(state_path, state)
        raise


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--polyvore-root", required=True)
    ap.add_argument("--variant", choices=sorted(VARIANTS), required=True)
    ap.add_argument("--seed", type=int, choices=range(1, 6), required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--scope-report", default=str(SCOPE))
    ap.add_argument("--subset-ids", default=str(SUBSET))
    ap.add_argument("--subset-manifest", default=str(RECON))
    ap.add_argument("--dry-run", action="store_true", help="Read-only readiness check; default if --run absent.")
    ap.add_argument("--run", action="store_true", help="Actually execute 100 epoch CP and CIR and both test evaluations.")
    ap.add_argument("--acknowledge-candidate-source", action="store_true",
                    help="Explicitly accept inferred WOS subset membership and the standardized decoder implementation.")
    args = ap.parse_args()
    if args.run and args.dry_run:
        abort("Select --dry-run or --run, not both")
    if args.run and not args.acknowledge_candidate_source:
        abort("Actual training requires --acknowledge-candidate-source")
    import os
    polyvore = Path(args.polyvore_root).expanduser().resolve()
    out = Path(args.out_dir).expanduser()
    out = (out if out.is_absolute() else ROOT / out).resolve()
    # Full handoff runs live in ONE Git-ignored reproduction/runs/<run-id>/
    # directory. Keep run outputs isolated from archived research directories; never
    # allow outputs to overwrite archived original research/source directories.
    in_repository = out == ROOT or ROOT in out.parents
    allowed_inside = (
        str(out).startswith(str(REPRO_ROOT / "runs") + os.sep)

    )
    if in_repository and not allowed_inside:
        abort("--out-dir inside repository must be under reproduction/runs/")
    if not in_repository and (out == Path(out.anchor) or out == ROOT.parent):
        abort("--out-dir must be a dedicated experiment directory")
    scope_path = Path(args.scope_report).expanduser()
    scope_path = (scope_path if scope_path.is_absolute() else ROOT / scope_path).resolve()
    subset_path = Path(args.subset_ids).expanduser()
    subset_path = (subset_path if subset_path.is_absolute() else ROOT / subset_path).resolve()
    recon_path = Path(args.subset_manifest).expanduser()
    recon_path = (recon_path if recon_path.is_absolute() else ROOT / recon_path).resolve()
    if not scope_path.is_file():
        abort(f"Scope report missing: {scope_path}")
    if not subset_path.is_file() or not recon_path.is_file():
        abort("Subset candidate IDs/manifest are missing")
    variant, feat = VARIANTS[args.variant]
    if not (FEATURES / feat).is_file():
        abort(f"Missing feature file: {FEATURES / feat}")
    inputs = verify_gate(polyvore, subset_path, recon_path, scope_path)
    source_paths = sorted(SOURCE.rglob("*.py"))
    if not source_paths:
        abort("Missing archived Python training sources")
    inputs["source_py_sha256"] = {p.relative_to(SOURCE).as_posix(): digest(p) for p in source_paths}
    inputs["feature_sha256"] = {str(FEATURES / feat): digest(FEATURES / feat)}
    inputs["python"] = sys.version
    inputs["candidate_source_label"] = "reproducibly reconstructed fair subset (WOS W/O/S rule); NOT the recovered historical ID file"
    inputs["decoder_label"] = standard_decoder.LABEL
    inputs["training_semantics"] = "100 epochs CP -> pretrained CP best initializes same-seed CIR; full original split indexes retained for negative sampling; fair subset filtered in dat.data/questions"
    inputs["subset_id_file"] = str(subset_path)
    inputs["variant"] = args.variant
    inputs["source_variant"] = variant
    inputs["feature_file"] = feat
    inputs["seed"] = args.seed
    state_file = out / "fair_subset_run_manifest.json"
    if state_file.is_file():
        state = load(state_file)
        if state.get("identity") != inputs:
            abort("Existing run identity changed (source/data/PKL/runtime/seed/variant); choose a fresh --out-dir")
    elif out.exists() and any(out.iterdir()):
        abort(f"Output directory exists without our run manifest; refusing overwrite: {out}")
    else:
        state = {"identity": inputs, "status": "planned", "cp": None, "cir": None,
                 "evaluation": None, "active_stage": None}
    state_info = {"variant": args.variant, "seed": args.seed, "out_dir": str(out),
                  "status": state["status"], "cp_done": isinstance(state["cp"], dict),
                  "cir_done": isinstance(state["cir"], dict),
                  "eval_done": isinstance(state["evaluation"], dict),
                  "scope": EXPECTED_SCOPE, "identity_ok": True}
    print(json.dumps(state_info, ensure_ascii=False, indent=2), flush=True)
    if not args.run:
        return

    # Branch-neutral safety gate: this runner is intended to work after the
    # reproduction wrappers are merged onto main.  Protect the archived
    # research implementation itself rather than requiring a branch name.
    proc = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=no", "--",
         str(SOURCE.relative_to(ROOT))],
        cwd=ROOT, text=True, capture_output=True, check=True,
    )
    source_changes = proc.stdout.splitlines()
    if source_changes:
        abort(
            "Archived main_hybrid_attention_code has tracked modifications; "
            "restore it before reproduction: " + repr(source_changes)
        )
    import torch
    if not torch.cuda.is_available():
        abort("CUDA is required for this source's 100-epoch training")
    print("[ENV]", torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0), flush=True)

    out.mkdir(parents=True, exist_ok=True)
    atomic(state_file, state)
    import fcntl
    with (out / "exclusive_run.lock").open("a+") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            abort("Same task is already running")
        workroot = REPRO_ROOT / ".work"
        workroot.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="fresh_subset_", dir=workroot) as temp:
            work = Path(temp)
            src = work / "main_hybrid_attention_code"
            shutil.copytree(SOURCE, src)
            standard_decoder.apply(src / "outfit_transformer.py")
            os.symlink(FEATURES, work / "fashionclip_data", target_is_directory=True)
            (work / "polyvore_data").mkdir()
            os.symlink(polyvore, work / "polyvore_data" / "polyvore_outfits", target_is_directory=True)
            exp = work / "experiments"
            exp.mkdir()
            cpdir = exp / f"cp_{variant}_subset_seed{args.seed}"
            cirdir = exp / f"cir_{variant}_subset_seed{args.seed}"
            cp_saved = out / "cp_best_ckpt.pt"
            cir_saved = out / "cir_best_ckpt.pt"
            common = ["--seed", str(args.seed), "--variant", variant, "--outfit_feat_name", feat,
                      "--subset_ids_path", str(subset_path), "--exp_root", str(exp)]

            def train_cp():
                run_stream([sys.executable, "train_cp.py", *common], src, out / "cp_train.log")
                best = cpdir / "ckpt.pt"
                if not best.is_file():
                    abort(f"CP finished without best: {best}")
                shutil.copy2(best, cp_saved)
                for p in cpdir.glob("*.log"):
                    shutil.copy2(p, out / ("cp_" + p.name))

            stage(state, "cp", cp_saved, train_cp, state_file)

            def train_cir():
                run_stream([sys.executable, "train_cir.py", *common,
                            "--pretrained_cp_ckpt", str(cp_saved)], src, out / "cir_train.log")
                best = cirdir / "ckpt.pt"
                if not best.is_file():
                    abort(f"CIR finished without best: {best}")
                shutil.copy2(best, cir_saved)
                for p in cirdir.glob("*.log"):
                    shutil.copy2(p, out / ("cir_" + p.name))

            stage(state, "cir", cir_saved, train_cir, state_file)
            if not (verify_checkpoint(cp_saved, state["cp"]) and verify_checkpoint(cir_saved, state["cir"])):
                abort("Checkpoint identity failed before test evaluation")
            if not verify_eval(out, state, variant, feat, args.seed):
                cpdir.mkdir(parents=True, exist_ok=True)
                cirdir.mkdir(parents=True, exist_ok=True)
                for stage_dir, saved_ckpt in ((cpdir, cp_saved), (cirdir, cir_saved)):
                    current = stage_dir / "ckpt.pt"
                    if current.is_file():
                        if digest(current) != digest(saved_ckpt):
                            abort(f"Evaluation workspace checkpoint differs from saved best: {current}")
                    elif current.exists() or current.is_symlink():
                        abort(f"Unexpected broken/invalid checkpoint link: {current}")
                    else:
                        os.symlink(saved_ckpt, current)
                state["active_stage"] = "evaluation"
                atomic(state_file, state)
                try:
                    run_stream([sys.executable, "evaluate_cp.py", *common], src, out / "cp_evaluate.log")
                    run_stream([sys.executable, "evaluate_cir.py", *common,
                                "--run_tag", "none_subset", "--save_detail",
                                "--detail_file", str(out / "detail_cir_fresh_subset.csv")],
                               src, out / "cir_evaluate.log")
                    cp_csv = exp / "results_cp.csv"
                    cir_csv = exp / "results_cir.csv"
                    cp_metrics = read_one_result(cp_csv, CP_METRICS, variant=variant, feat=feat, seed=args.seed)
                    cir_metrics = read_one_result(cir_csv, CIR_METRICS, variant=variant, feat=feat, seed=args.seed, cir=True)
                    detail = out / "detail_cir_fresh_subset.csv"
                    with detail.open(encoding="utf-8-sig", newline="") as f:
                        details = list(csv.DictReader(f))
                    if len(details) != EXPECTED_SCOPE or len({str(r["set_id"]) for r in details}) != EXPECTED_SCOPE:
                        abort("CIR evaluation produced wrong denominator/duplicated FITB IDs")
                    shutil.copy2(cp_csv, out / "results_cp_fresh_subset.csv")
                    shutil.copy2(cir_csv, out / "results_cir_fresh_subset.csv")
                    state["evaluation"] = {
                        "cp": cp_metrics, "cir": cir_metrics, "cir_evaluable": len(details),
                        "sha256": {p.name: digest(p) for p in
                                   (out / "results_cp_fresh_subset.csv", out / "results_cir_fresh_subset.csv", detail)}
                    }
                    state["active_stage"] = None
                    state["status"] = "passed_candidate_source"
                    atomic(state_file, state)
                except BaseException:
                    state["active_stage"] = "evaluation_interrupted_rerun_entire_evaluation"
                    atomic(state_file, state)
                    raise
            else:
                state["status"] = "passed_candidate_source"
                atomic(state_file, state)
    print("[PASSED] fresh subset", args.variant, args.seed, "CP/CIR/eval", out, flush=True)


if __name__ == "__main__":
    main()