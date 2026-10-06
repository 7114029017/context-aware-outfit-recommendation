# Post-restructure fresh full-run verification

This document records the second complete 35-unit training execution, run
from the restructured repository with the documented one-command entrypoint,
and its comparison against the preserved reference run
`reference_20260921T175217Z`.

## Run identity

- Run ID: `full_20260924T102513Z`
- Command: `bash reproduction/scripts/reproduce_all.sh --fresh`
- Branch: `thesis-full-reproduction`
- Git commit: `2cd8d04694ef314955be48ba47689e98a4b13d35`
- Working tree at start: clean
- Start: 2026-09-24T10:25:13Z
- Finish: 2026-09-26T11:34:44Z
- Wall time: 49:09:31
- Status: `PASSED`
- Training units: 35 = 10 main + 25 candidate ablation; seeds 1, 2, 3, 4, 5
- Runtime: Python 3.12.3; PyTorch 2.9.1+cu130; CUDA 13.0; NVIDIA GB10
  (same machine as the reference run)

Preceding preflight runs from the same session: `check_20260924T102053Z`
and `smoke_20260924T102104Z`.

The run directory is local (`reproduction/runs/` is git-ignored; about
1.9 GB including checkpoints). Its identity is established by the hashes
below, not by committing the checkpoints.

## Fresh-training evidence

- `logs/full_console.log` reports `Initializing a new model from scratch`
  70 times (35 units x CP + CIR).
- Checkpoint modification times fall within this run's execution window.
- No error or traceback lines appear in the console log.

## Comparison against the reference run

Each hash below was compared with
`reproduction/results/summary/reference_20260921T175217Z/reference_seed_index.csv`.

| Evidence | Result |
|---|---|
| CP best-checkpoint SHA-256 | 35 / 35 identical |
| CIR best-checkpoint SHA-256 | 35 / 35 identical |
| CP evaluation result CSV SHA-256 | 35 / 35 identical |
| CIR evaluation result CSV SHA-256 | 35 / 35 identical |
| Fair-subset CIR detail CSV SHA-256 | 25 / 25 identical |

The repository's own strict verifier also accepts this run as the reference:

    python3 reproduction/scripts/verify_completed_reference.py \
      --full-run reproduction/runs/full_20260924T102513Z

reports `training units = 35`, `checkpoint SHA-256 identities = 70/70`, and
`committed reference result hashes = 201/201`.

Summary outputs, compared with
`reproduction/results/summary/reference_20260921T175217Z/`:

| Output | Result |
|---|---|
| `main/main_reproduction_summary.csv`, `main_reproduction_acceptance.json` | byte-identical |
| `statistics/main_paired_bh_8metrics.csv`, `factor_direction_candidate_vs_archived_T01.csv`, `statistical_evidence.md` | byte-identical |
| `ablation/` T01–T11 tables, `fresh_25unit_metrics.csv`, `fresh_5variant_8metric_summary.csv` | byte-identical |
| `secondary/` recomputed CSVs (Tables 4-3 to 4-10) | byte-identical |
| `chapter4/chapter4_numeric_comparison.csv` (112 rows) | identical after normalizing run-directory paths |
| `chapter4/chapter4_table_overview.csv`, `chapter4_report.md` | byte-identical |

The remaining JSON files that differ byte-wise (`statistical_evidence.json`,
`chapter4_report.json`, `fresh_fair_subset_5seed_summary.json`,
`remaining_reproduction_matrix.*`, `table_4_3_manifest.json`) differ only in
embedded run-directory paths, the recorded git commit, and hashes of
path-bearing manifests. No numeric field differs.

## Interpretation

On the same hardware and runtime, the restructured one-command pipeline
retrains all 35 units bit-identically to the reference run. This closes the
earlier caveat that the reference run predated repository restructuring.

It does NOT change the documented historical provenance limits: the
`DecoderLayerWithCrossAttn` implementation, fair-subset memberwise IDs,
tuning trials and Judge generation provenance remain unrecovered. Bit-level
identity is also not expected on a different GPU / CUDA / PyTorch build.

## How to re-check

    RUN=reproduction/runs/full_20260924T102513Z
    cat "$RUN/RUN_STATUS.txt"
    python3 - <<'PY'
    import csv, hashlib, os
    run = "reproduction/runs/full_20260924T102513Z"
    index = "reproduction/results/summary/reference_20260921T175217Z/reference_seed_index.csv"
    sha = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest() if os.path.exists(p) else None
    layout = {
        "main": ("main", {"cp_checkpoint_sha256": "cp_best_ckpt.pt",
                          "cir_checkpoint_sha256": "cir_best_ckpt.pt",
                          "cp_results_sha256": "evaluation/results_cp.csv",
                          "cir_results_sha256": "evaluation/results_cir.csv"}),
        "candidate_ablation": ("ablation/runs", {"cp_checkpoint_sha256": "cp_best_ckpt.pt",
                                       "cir_checkpoint_sha256": "cir_best_ckpt.pt",
                                       "cp_results_sha256": "results_cp_fresh_subset.csv",
                                       "cir_results_sha256": "results_cir_fresh_subset.csv",
                                       "detail_results_sha256": "detail_cir_fresh_subset.csv"}),
    }
    bad = 0
    for r in csv.DictReader(open(index)):
        sub, files = layout[r["family"]]
        for key, name in files.items():
            if r.get(key) and sha(f"{run}/{sub}/{r['variant']}_seed{r['seed']}/{name}") != r[key]:
                bad += 1
                print("MISMATCH", r["family"], r["variant"], r["seed"], key)
    print("mismatches:", bad)
    PY
