# Reproducibility contract

The authoritative reproduction entrypoint is:

`reproduction/README.md`

## Protected original research tree

The following directories are treated as the frozen senior-student research
handoff:

- `01_資料建構_data_construction/`
- `02_模型訓練和驗證_model_training_validation/`
- `03_實驗與結果_experiments_results/`

Baseline:

`7a5cc9cd8f884865e1e27a234c18dd132f949598`

This commit belongs to the original development repository and is not in the
history of this public repository; it is the content the official run used.
For the public release, third-party publications and manuscripts were removed
and the embedded Polyvore images were removed from three notebooks; the
former `04_文件資料_documents/` folder held only manuscripts and the thesis and
is not published. The changes are listed in
`reproduction/docs/public_release_cleanup.md`. The published content of folders
01-03 is recorded in `reproduction/environment/archived_sources_manifest.json`
and checked file by file by `reproduction/scripts/verify_archived_sources.py`
before a full run starts.

Reproduction wrappers must not modify these directories in place.

## New artifact namespace

All new reproducibility material belongs under:

`reproduction/`

New experiment output belongs under:

`reproduction/runs/`

## Official run

The formal results of the revised manuscript come from the clean-room run
`full_20261002T161428Z` (commit `b9bf5aa`): a fresh GitHub clone with a new
environment, following `reproduction/README.md` section 0, finished `PASSED`
and bit-identical to the two runs below. See `docs/clean_room_acceptance.md`
and `results/final_reference_manifest.json`.

## Completed reference run

The completed 35-unit reference run used:

`eb081e7a426cf5bd91acd8bb0f92a8398f08f45e`

and finished with status `PASSED`.

After repository restructuring, a second complete 35-unit execution was run
with `reproduce_all.sh --fresh` (`full_20260924T102513Z`, commit `2cd8d04`).
It finished `PASSED` and is bit-identical to the reference run; see
`docs/post_restructure_fresh_run_verification.md`.

## Scientific provenance rule

Missing historical evidence must remain explicitly unresolved.

In particular, do not silently claim exact recovery of:

- historical fair-subset memberwise IDs;
- historical DecoderLayerWithCrossAttn source/forward behavior;
- historical hyperparameter tuning trials;
- exact 2025 runtime identity;
- complete Judge P0/P1/P2 generation provenance.

## Entry points

Preflight:

    bash reproduction/scripts/check.sh

Optional smoke test:

    bash reproduction/scripts/smoke_test.sh

Full reproduction / verified local reuse:

    bash reproduction/scripts/reproduce_all.sh

If the preserved local completed reference run exists, the entrypoint verifies
all 35 seed identities, all 70 checkpoint SHA-256 values, and the committed
reference-result hash manifest before skipping training.  A clean clone without
that local run starts a new full reproduction.

Force a new 35-unit training execution explicitly with:

    bash reproduction/scripts/reproduce_all.sh --fresh

Reuse mode verifies existing evidence; it does not claim a second training run.
