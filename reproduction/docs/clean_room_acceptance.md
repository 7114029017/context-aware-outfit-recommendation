# Clean-room acceptance record (official run)

Status: **ACCEPTED — run PASSED and is bit-identical to the reference run.**
This record follows section 二 of the 2026-09-30 guideline.

## Clone

| Item | Value |
|---|---|
| Workspace | `~/Documents/course/tors-cleanroom-20261003/` (new directory) |
| Repository URL | `https://github.com/guoyouuu/context-aware-outfit-recommendation.git` |
| Branch | `thesis-full-reproduction` |
| Commit | `b9bf5aac7d07cb39cacc7f73eb28166848c19fe4` |
| Working tree | clean after cloning and at run start (`git status --short` empty) |

## Procedure

Every step is `reproduction/README.md` section 0, run verbatim by
`cleanroom_setup.sh` and `cleanroom_launch.sh` in the workspace. Times are UTC
on 2026-10-02.

| Step | Command | Time | Result |
|---|---|---|---|
| 0.1 | `git clone --branch thesis-full-reproduction --single-branch …` | 16:08:23–16:11:32 | branch and commit as above |
| 0.2 | `git lfs install`, `git lfs pull` | 16:11:33 | feature files present (verified by 0.5) |
| 0.3 | new `.venv-repro`; `torch==2.9.1`, `torchvision==0.24.1` from the cu130 index; `requirements-reproduction-runtime.txt` | 16:11:35–16:12:51 | CUDA available; versions below |
| 0.4 | `bash reproduction/scripts/bootstrap_data.sh` | 16:12:53 | 12 files fetched from `Stylique/Polyvore`; 8 required files present |
| 0.5 | `bash reproduction/scripts/check.sh` | 16:12:58 | passed: 8 feature hashes, 6 split manifests byte-identical (`check_20261002T161258Z`) |
| 0.6 | `bash reproduction/scripts/smoke_test.sh` | 16:13:04–16:13:44 | passed (`smoke_20261002T161304Z`); CP and CIR smoke checkpoints byte-identical to `smoke_20260924T102104Z` from the previous working copy |
| 0.7 | `bash reproduction/scripts/reproduce_all.sh --fresh` in tmux session `tors_official` | 16:14:28 | run `full_20261002T161428Z` started |

## Environment

From `cleanroom_pip_freeze.txt` and the CUDA check:

| Component | Version |
|---|---|
| Python | 3.12.3 |
| PyTorch / torchvision | 2.9.1+cu130 / 0.24.1 |
| CUDA / cuDNN | 13.0 / 9.13.0 |
| GPU | NVIDIA GB10 |
| numpy / scipy | 2.0.2 / 1.18.1 |
| transformers / scikit-learn / pandas | 4.57.1 / 1.6.1 / 2.2.3 |
| einops / huggingface_hub / Pillow | 0.8.1 / 0.36.2 / 11.3.0 |

## Manual steps and deviations

- No file was moved by hand, no program was edited, and no step outside
  README section 0 was used.
- The page-cache flush that README 0.6 recommends on GB10 machines was not
  performed before launch (free memory about 17 GiB, available about
  117 GB). During the run, free memory fell to 5.9 GB, so the operator
  flushed the page cache once at about 2026-10-04T03:08Z
  (`sudo sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'`); free memory
  returned to 100 GiB. This is a system action, not a pipeline step, and it
  did not interrupt or change the run.
- The setup and launch scripts, their logs and the pip freeze are kept in the
  workspace, outside the clone, so the clone's working tree stays clean.

Before the clean-room run, a smoke test in the previous working copy hit
`CUDA error: out of memory` (2026-10-01) because CUDA on the GB10 counts only
free memory, not page cache. The cause and the remedy were added to README
0.6 before this run.

## Run

| Item | Value |
|---|---|
| Run ID | `full_20261002T161428Z` |
| Start | 2026-10-02T16:14:28Z |
| Finish | 2026-10-04T23:22:08Z |
| Total time | 55:07:40 |
| `RUN_STATUS.txt` | `PASSED` (`reproduce_all.sh` exit 0) |
| Final terminal summary | `results/raw/full_20261002T161428Z/run_identity/final_terminal_summary.txt` |
| Failures during the run | none; the console log has no traceback or out-of-memory line |

The run took about 10% longer than the two earlier full runs (49:42:27 and
49:09:31), while CPU-heavy jobs of another project ran on the same machine.
Results are unaffected (see below). The two console lines marked
`BLOCKED/PARTIAL` are the expected status of the `judge_exact_generation` and
`case_visuals` modules (no LLM rerun, no Polyvore images), listed in
`known_limitations.md`.

The run is packaged in `results/raw/full_20261002T161428Z/`,
`results/summary/full_20261002T161428Z/` and
`results/final_reference_manifest.json`; the setup scripts, setup log and pip
freeze are in `results/raw/full_20261002T161428Z/clean_room/`.

## Comparison with the reference run

Against `reference_20260921T175217Z`:

| Evidence | Result |
|---|---|
| CP and CIR checkpoint SHA-256 (re-hashed) | 70 / 70 identical |
| Seed-level CP / CIR result CSVs | 70 / 70 identical |
| Fair-subset per-query detail CSVs | 25 / 25 identical |
| Fair-subset ID SHA-256 | 25 / 25 identical |
| Summary files present in both runs | 35 byte-identical; 15 differ only in wording, paths, hashes or the new T10 / T11 CI columns, with every number identical |

The conclusions are therefore unchanged: 8 / 8 positive main directions,
7 / 8 metrics significant after BH (Recall@1 not, Recall@3 yes), 16 / 16
thesis condition means within 0.005, Style the largest factor in CP and OR,
and 5 / 6 Weather / Occasion OR signs reversed relative to the thesis. This
is the third independent 35-unit training with bit-identical results
(`reference_20260921T175217Z`, `full_20260924T102513Z`,
`full_20261002T161428Z`).
