#!/usr/bin/env python3
"""Text-swap evaluation of the main models (the 2025 sweep of CP_evaluate.py and CIR_evaluate.py; GPU).

The 2025 code (02_模型訓練和驗證_model_training_validation/main_hybrid_attention_code/CP_evaluate.py and
CIR_evaluate.py with --sweep) evaluated each seed's Original and Context models with both outfit texts, the
original titles (encoded_outfitUrlTitle_en_fashionClip.pkl) and the context-aware descriptions
(encoded_NewoutfitUrlTitle_en_fashionClip.pkl): four combinations per seed. Its result files,
results_cp_sweep.csv and results_cir_sweep.csv, are not preserved; P12 read their matching-text rows for A14.
The archived CIR_evaluate.py does not run as preserved (an indentation error at line 272), so this step runs
the evaluators of the full run, evaluate_cp.py and evaluate_cir.py, in the same working copy as
pipeline/evaluate_main_run.py (standardized decoder, feature and data links), with --outfit_feat_name set to
either text. A matching-text combination reproduces the main evaluation; a swapped one gives the model's
results with the other text.

--run-root RUN evaluates RUN/main/{original,context}_seed<k>: the two swapped combinations of every seed and,
as a check of the setup, the matching-text combinations of --check-seeds (default 1), which must equal the
run's stored results_cp.csv and results_cir.csv. --checkpoints-2025 evaluates the 20 preserved 2025
checkpoints (cp_old, cp_new, cir_old, cir_new; seeds 1-5) in all four combinations, which recomputes the lost
sweep, and compares the matching-text means and standard deviations with the archived A14. --subset-ids FILE
restricts the evaluation to a set of outfits (plumbing test on the CPU); such a run is never compared with
stored results.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from statistics import mean, stdev

from _ext import (CHECKPOINTS_2025, D03, FEATURES, MODEL_CODE, REPO, REPRO, SCRIPTS, SEEDS, local_path, output_dir,
                  passed_run, read_csv, sha256, write_csv, write_json, write_text)

sys.path.insert(0, str(SCRIPTS))
import standard_decoder  # noqa: E402

TEXTS = {"original": "encoded_outfitUrlTitle_en_fashionClip.pkl",
         "context": "encoded_NewoutfitUrlTitle_en_fashionClip.pkl"}
VARIANTS = {"original": "outfitUrlTitle", "context": "NewoutfitUrlTitle"}  # model -> evaluator --variant
TAGS_2025 = {"original": "old", "context": "new"}  # model -> 2025 checkpoint folder suffix (cp_old_seed1, ...)
CP_METRICS = ("auc", "fitb_acc")
CIR_METRICS = ("recall_at_1", "recall_at_3", "recall_at_5", "recall_at_10", "recall_at_30", "recall_at_50")
METRICS = CP_METRICS + CIR_METRICS
ARCHIVED_A14 = (D03 / "03_主推薦任務結果" / "圖表_figures_tables" / "tables"
                / "A14_statistical_rigor_ci_adjusted_p_effect_sizes.csv")
HEADER = ["seed", "model", "text", "matching_text", *METRICS, "equal_to_stored"]


def run(cmd: list[str], cwd: Path, log: Path) -> None:
    with log.open("w", encoding="utf-8") as f:
        code = subprocess.run(cmd, cwd=cwd, stdout=f, stderr=subprocess.STDOUT).returncode
    if code:
        raise SystemExit(f"[TEXT SWAP FAILED] {cmd[1]} exited with {code}; see {log}")


def working_copy(work: Path, polyvore: Path, device: str) -> Path:
    """The working copy of pipeline/evaluate_main_run.py."""
    src = work / "main_hybrid_attention_code"
    shutil.copytree(MODEL_CODE, src)
    standard_decoder.apply(src / "outfit_transformer.py")
    if device == "cpu":
        cfg = src / "config" / "base_config.py"
        text = cfg.read_text(encoding="utf-8")
        switched = text.replace("device_type = 'cuda'", "device_type = 'cpu'", 1)
        if switched == text:
            raise SystemExit("[TEXT SWAP BLOCKED] could not switch the working-copy config to CPU")
        cfg.write_text(switched, encoding="utf-8")
    os.symlink(FEATURES, work / "fashionclip_data", target_is_directory=True)
    (work / "polyvore_data").mkdir()
    os.symlink(polyvore, work / "polyvore_data" / "polyvore_outfits", target_is_directory=True)
    return src


def evaluate(src: Path, work: Path, logs: Path, seed: int, model: str, text: str, cp_ckpt: Path,
             cir_ckpt: Path, subset_ids: Path | None = None) -> dict:
    """evaluate_cp.py and evaluate_cir.py for one model with one text; returns the eight metrics."""
    variant, scope = VARIANTS[model], "subset" if subset_ids else "all"
    exp = work / f"experiments_{model}_seed{seed}_{text}_text"
    for task, ckpt in (("cp", cp_ckpt), ("cir", cir_ckpt)):
        (exp / f"{task}_{variant}_{scope}_seed{seed}").mkdir(parents=True)
        os.symlink(ckpt, exp / f"{task}_{variant}_{scope}_seed{seed}" / "ckpt.pt")
    common = ["--seed", str(seed), "--exp_root", str(exp), "--variant", variant, "--outfit_feat_name", TEXTS[text]]
    if subset_ids:
        common += ["--subset_ids_path", str(subset_ids)]
    name = f"{model}_model_seed{seed}_{text}_text"
    print(f"[TEXT SWAP] {name}", flush=True)
    run([sys.executable, "evaluate_cp.py", *common], src, logs / f"{name}_cp.log")
    run([sys.executable, "evaluate_cir.py", *common, "--run_tag", "text_swap"], src, logs / f"{name}_cir.log")
    cp = read_csv(exp / "results_cp.csv")
    cir = read_csv(exp / "results_cir.csv")
    if len(cp) != 1 or len(cir) != 1:
        raise SystemExit(f"[TEXT SWAP FAILED] {name}: expected one result row per task")
    return {**{k: cp[0][k] for k in CP_METRICS}, **{k: cir[0][k] for k in CIR_METRICS}}


def stored(run_root: Path, model: str, seed: int) -> dict:
    folder = run_root / "main" / f"{model}_seed{seed}" / "evaluation"
    cp, cir = read_csv(folder / "results_cp.csv"), read_csv(folder / "results_cir.csv")
    return {**{k: cp[0][k] for k in CP_METRICS}, **{k: cir[0][k] for k in CIR_METRICS}}


def run_checkpoints(run_root: Path, model: str, seed: int) -> tuple[Path, Path]:
    unit = run_root / "main" / f"{model}_seed{seed}"
    manifest = json.loads((unit / "full_train_manifest.json").read_text(encoding="utf-8"))
    cp, cir = unit / "cp_best_ckpt.pt", unit / "cir_best_ckpt.pt"
    if (manifest.get("status") != "passed" or sha256(cp) != manifest.get("cp_checkpoint_sha256")
            or sha256(cir) != manifest.get("cir_checkpoint_sha256")):
        raise SystemExit(f"[TEXT SWAP BLOCKED] {unit.name}: not a passed unit or a checkpoint SHA-256 differs")
    return cp, cir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--run-root", type=Path, help="completed full run folder with the checkpoints")
    source.add_argument("--checkpoints-2025", action="store_true", help="the 20 preserved 2025 checkpoints")
    parser.add_argument("--out-dir", type=Path, default=None, help="default: <run>/extensions/text_swap")
    parser.add_argument("--polyvore-root", default=None)
    parser.add_argument("--seeds", default=",".join(map(str, SEEDS)))
    parser.add_argument("--check-seeds", default="1",
                        help="with --run-root: seeds whose matching-text combinations are evaluated again")
    parser.add_argument("--eval-device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--subset-ids", type=Path, default=None, help="plumbing test only: a file of set IDs")
    args = parser.parse_args()

    import torch
    if args.eval_device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("[TEXT SWAP BLOCKED] CUDA is not available; this step needs the GPU (or --eval-device cpu)")
    run_root = passed_run(args.run_root) if args.run_root else None
    if args.out_dir is None and run_root is None:
        raise SystemExit("[TEXT SWAP BLOCKED] pass --out-dir")
    out = output_dir(args.out_dir or run_root / "extensions" / "text_swap", run_root)
    polyvore = local_path("polyvore_root", args.polyvore_root)
    if polyvore is None or not (polyvore / "disjoint" / "test.json").is_file():
        raise SystemExit("[TEXT SWAP BLOCKED] Polyvore root not found; pass --polyvore-root or run bootstrap_data.sh")
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    check_seeds = {int(s) for s in args.check_seeds.split(",") if s.strip()} if run_root else set(seeds)

    plan = []  # (seed, model, text)
    for seed in seeds:
        for model in ("original", "context"):
            for text in ("original", "context"):
                if text != model or seed in check_seeds:
                    plan.append((seed, model, text))
    logs = out / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    work_root = REPRO / ".work"
    work_root.mkdir(exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix="text_swap_", dir=work_root))
    rows, models = [], {}
    started = time.time()
    try:
        src = working_copy(work, polyvore, args.eval_device)
        for seed, model, text in plan:
            if run_root:
                cp, cir = run_checkpoints(run_root, model, seed)
            else:
                cp = CHECKPOINTS_2025 / f"cp_{TAGS_2025[model]}_seed{seed}" / "ckpt.pt"
                cir = CHECKPOINTS_2025 / f"cir_{TAGS_2025[model]}_seed{seed}" / "ckpt.pt"
            models[f"{model}_seed{seed}"] = {"cp": cp, "cir": cir}
            values = evaluate(src, work, logs, seed, model, text, cp, cir, args.subset_ids)
            same = ""
            if run_root and text == model and not args.subset_ids:
                same = "yes" if values == stored(run_root, model, seed) else "NO"
            rows.append([seed, model, text, "yes" if text == model else "no", *[values[k] for k in METRICS], same])
            write_csv(out / "text_swap_results.csv", HEADER, rows)  # written after every combination
    finally:
        shutil.rmtree(work, ignore_errors=True)

    def values_of(model: str, text: str, metric: str) -> list[float]:
        return [float(r[HEADER.index(metric)]) for r in rows if r[1] == model and r[2] == text]

    if run_root and not args.subset_ids:  # the matching-text values of every seed are the run's stored results
        for seed in seeds:
            for model in ("original", "context"):
                if not any(r[0] == seed and r[1] == model and r[2] == model for r in rows):
                    s = stored(run_root, model, seed)
                    rows.append([seed, model, model, "yes", *[s[k] for k in METRICS], "stored"])
        rows.sort(key=lambda r: (r[0], r[1] != "original", r[2] != "original"))
        write_csv(out / "text_swap_results.csv", HEADER, rows)
    combos = [(m, t) for m in ("original", "context") for t in ("original", "context")]
    summary = []
    for m, t in combos:
        for metric in METRICS:
            v = values_of(m, t, metric)
            if v:
                summary.append([m, t, metric, len(v), f"{mean(v):.6f}", f"{stdev(v):.6f}" if len(v) > 1 else ""])
    write_csv(out / "text_swap_summary.csv", ["model", "text", "metric", "n_seeds", "mean", "std"], summary)

    label = f"run `{run_root.name}`" if run_root else "the preserved 2025 checkpoints"
    lines = ["# Text-swap evaluation (2025 sweep of CP_evaluate.py and CIR_evaluate.py)", "",
             f"Models of {label}, seeds {', '.join(map(str, seeds))}; device {args.eval_device}. Each model is "
             "evaluated with the original titles and with the context-aware descriptions; the evaluators and the "
             "working copy are those of the full run's main evaluation.", ""]
    if args.subset_ids:
        lines += [f"Plumbing test on the outfits listed in `{args.subset_ids.name}`; not comparable with stored results.",
                  ""]
    elif run_root:
        checks = [r for r in rows if r[-1] in ("yes", "NO")]
        lines += [f"Setup check: the matching-text combinations evaluated again ({len(checks)}) equal the run's "
                  f"stored results: {'yes' if checks and all(r[-1] == 'yes' for r in checks) else 'NO'}. The other "
                  "matching-text rows are the stored results.", ""]
    lines += ["| Model | Text | AUC | FITB | R@1 | R@10 | R@30 | R@50 |", "|---|---|---:|---:|---:|---:|---:|---:|"]
    for m, t in combos:
        cells = []
        for metric in ("auc", "fitb_acc", "recall_at_1", "recall_at_10", "recall_at_30", "recall_at_50"):
            v = values_of(m, t, metric)
            cells.append(f"{mean(v):.4f} ± {stdev(v):.4f}" if len(v) > 1 else (f"{v[0]:.4f}" if v else ""))
        lines.append(f"| {m.capitalize()} | {t} | " + " | ".join(cells) + " |")
    lines += ["", "Mean over the seeds ± SD. Effect of the text for each model (context-aware minus original text, "
                  "mean over seeds; seeds with a positive difference):", "",
              "| Model | AUC | FITB | R@10 | R@30 | R@50 |", "|---|---:|---:|---:|---:|---:|"]
    for m in ("original", "context"):
        cells = []
        for metric in ("auc", "fitb_acc", "recall_at_10", "recall_at_30", "recall_at_50"):
            a, b = values_of(m, "original", metric), values_of(m, "context", metric)
            if len(a) == len(b) and a:
                d = [y - x for x, y in zip(a, b)]
                cells.append(f"{mean(d):+.4f} ({sum(1 for x in d if x > 0)}/{len(d)})")
            else:
                cells.append("")
        lines.append(f"| {m.capitalize()} | " + " | ".join(cells) + " |")
    lines.append("")
    a14_rows = []
    if not run_root:
        a14 = {(r["task"], r["metric"]): r for r in read_csv(ARCHIVED_A14)}
        for (task, metric), r in a14.items():
            for m, side in (("original", "baseline"), ("context", "proposed")):
                v = values_of(m, m, metric)
                if len(v) > 1:
                    now_mean, now_sd = f"{mean(v):.6f}", f"{stdev(v):.6f}"
                    a14_rows.append([task, metric, m, now_mean, r[f"{side}_mean"], now_sd, r[f"{side}_std"],
                                     "yes" if (now_mean, now_sd) == (r[f"{side}_mean"], r[f"{side}_std"]) else "NO"])
        write_csv(out / "a14_check.csv", ["task", "metric", "model", "mean", "a14_mean", "std", "a14_std", "equal"],
                  a14_rows)
        lines += ["Matching-text means and SDs against the archived A14 (`a14_check.csv`):", "",
                  "| Task | Metric | Model | Mean (A14) | SD (A14) | Equal |", "|---|---|---|---|---|---|"]
        lines += [f"| {r[0]} | {r[1]} | {r[2]} | {r[3]} ({r[4]}) | {r[5]} ({r[6]}) | {r[7]} |" for r in a14_rows]
        lines.append("")
    lines += ["Files: `text_swap_results.csv` (one row per seed, model and text), `text_swap_summary.csv`, "
              "`logs/` (evaluator logs, not committed).", ""]
    write_text(out / "summary.md", lines)

    def where(path: Path) -> str:
        if run_root and path.is_relative_to(run_root):
            return f"<run>/{path.relative_to(run_root)}"
        return str(path.relative_to(REPO)) if path.is_relative_to(REPO) else path.name
    write_json(out / "manifest.json", {
        "run": run_root.name if run_root else None, "device": args.eval_device, "seeds": seeds,
        "status": "check_only" if args.subset_ids or args.eval_device == "cpu" else "complete",
        "subset_ids": str(args.subset_ids) if args.subset_ids else None,
        "evaluated": [{"seed": s, "model": m, "text": t} for s, m, t in plan], "seconds": round(time.time() - started),
        "torch": torch.__version__, "cuda_device": torch.cuda.get_device_name(0) if args.eval_device == "cuda" else None,
        "texts": TEXTS, "checkpoints": {k: {t: {"path": where(p), "sha256": sha256(p)} for t, p in v.items()}
                                        for k, v in models.items()},
        "a14_equal": all(r[-1] == "yes" for r in a14_rows) if a14_rows else None})
    print("\n".join(lines))


if __name__ == "__main__":
    main()
