# TORS reproducibility compliance status

This matrix separates the reproducibility artifact that is already
available from historical evidence or publication steps that remain
unresolved.

| Item | Status | Evidence / remaining work |
|---|---|---|
| A1 Public GitHub repository | PUBLIC; RELEASED | https://github.com/7114029017/context-aware-outfit-recommendation; the release `v1.0.0-tors-reproduction` was published on 2026-10-07 and checked without logging in (release page, source archives, the eight Git LFS files, a clone of the tag); Polyvore images and third-party documents are not redistributed (`docs/public_release_cleanup.md`, `THIRD_PARTY_NOTICES.md`) |
| A2 Fixed Release / preservation / DOI | RELEASED | GitHub Release `v1.0.0-tors-reproduction` (2026-10-07). Following the advisor's decision of 2026-10-08, the data preparation is redone and a new official run made with one command precedes the next release (`docs/handoff_status.md`). No DOI is used |
| A3 End-to-end README | COMPLETE FOR CURRENT ARTIFACT | `reproduction/README.md` documents bootstrap, environment, check, smoke, verified reuse, fresh full run, outputs and provenance limits; one command also runs every analysis after the 35 units and reports each of the 46 analysis items (`ITEMS_STATUS.md`, `FINAL_SUMMARY.txt`) |
| A4 Environment specification | COMPLETE FOR 2026 REPRODUCTION; HISTORICAL PARTIAL | `reproduction/environment/`; exact 2025 runtime identity is not established |
| A5 Experiment configs | COMPLETE WITH DOCUMENTED PROVENANCE LIMITS | `reproduction/configs/` |
| A6 Fixed IDs | COMPLETE FOR 2026 ARTIFACT; HISTORICAL ABLATION IDENTITY PARTIAL | Main IDs are fixed; the fair subset (21,903 IDs, splits and 3,432 OR query IDs) is a reproducibly reconstructed subset in `splits/fair_subset/`; historical memberwise identity not recovered |
| A7 Hyperparameter tuning provenance | PARTIAL | Main-model settings fully sourced to the upstream paper and code (`docs/hyperparameter_tuning.md`, four-category `results/tuning.csv`); no tuning search evidenced; second-model selection rationale not recorded |
| A8 Reproduction scripts | IMPLEMENTED + VERIFIED | Official clean-room run `full_20261002T161428Z` from a fresh clone following the README finished PASSED (`clean_room_acceptance.md`); it is bit-identical to the reference run and to the post-restructure run `full_20260924T102513Z` |
| A9 Seed-level raw and summary results | COMPLETE | Official run in `results/raw/full_20261002T161428Z/` and `results/summary/full_20261002T161428Z/` with `seed_index.csv` and `final_reference_manifest.json`; earlier reference evidence kept |
| A10 Artifact Availability statement | MANUSCRIPT UPDATE | The manuscript statement should cite the GitHub release and its official run ID (now `v1.0.0-tors-reproduction` and `full_20261002T161428Z`), not a commit of an earlier repository |

## Completed reference full reproduction

Reference ID:

`reference_20260921T175217Z`

Training source commit:

`eb081e7a426cf5bd91acd8bb0f92a8398f08f45e`

Execution:

- main units: 10
- candidate-ablation units: 25
- total training units: 35
- status: PASSED
- start: 2026-09-21T17:52:17Z
- finish: 2026-09-23T19:34:44Z
- total wall time: 49:42:27

Repository restructuring did not modify these preserved results. A separate
post-restructure 35-unit `--fresh` execution (`full_20260924T102513Z`) was
later completed and is bit-identical to them; see
`post_restructure_fresh_run_verification.md`.

## Seed-level evidence

The machine-readable reference index is:

`reproduction/results/summary/reference_20260921T175217Z/reference_seed_index.csv`

It contains 35 rows and records the common training commit, variant, seed,
runtime recovered from saved logs, checkpoint hashes, result hashes and
candidate-subset identity where applicable.

## Core scientific result status

The main five-seed numerical reproduction is close to the thesis reference
values under the predefined comparison tolerance.

All eight Context-minus-Original aggregate main metric directions are
positive in the completed reproduction.

The post-run paired five-seed analysis finds seven of eight main metrics
significant after Benjamini-Hochberg correction; Recall@1 is not significant.

These statistical results are preserved under:

`reproduction/results/summary/reference_20260921T175217Z/statistics/`

## Historical provenance limits

The artifact does not claim exact historical recovery of:

1. the original fair-subset memberwise ID file;
2. the original `DecoderLayerWithCrossAttn` implementation;
3. historical hyperparameter-search trials;
4. the exact 2025 runtime identity;
5. all Judge generation parameters and complete historical Judge prompt
   provenance.

Consequently, the appropriate overall provenance classification remains
partial even though the main numerical reproduction and the 2026
reproduction artifact are preserved.

## Professor-requested reproducibility checklist (2026-09-22)

| Requested item | Current status | Evidence / note |
|---|---|---|
| Full-run start and finish time | COMPLETE | 2026-09-21T17:52:17Z to 2026-09-23T19:34:44Z |
| Explicit success status | COMPLETE | Completed reference run has `RUN_STATUS.txt = PASSED` |
| Five actual seeds for main and ablation | COMPLETE | Training seeds are 1, 2, 3, 4, 5 |
| Per-seed raw results and aggregate results | COMPLETE | `results/raw/reference_20260921T175217Z/`, `results/summary/reference_20260921T175217Z/`, 35-row seed index |
| Automatic Table 4-11 / 4-12 / ablation output | COMPLETE FOR AVAILABLE EVIDENCE | Pipeline writes main, ablation and Chapter 4 CSV/JSON/MD outputs; `show_reproduction_summary.py` renders the core results read-only in terminal |
| Tested README and one-command flow | COMPLETE | Reference full run PASSED before restructuring; post-restructure `reproduce_all.sh --fresh` run `full_20260924T102513Z` PASSED and is bit-identical to the reference (70/70 checkpoints, 95/95 result CSVs) |
| Python / PyTorch / CUDA / package versions | COMPLETE FOR 2026 REPRODUCTION | `reproduction/environment/` |
| Hyperparameter tuning explanation | PARTIAL | Every setting classified as fixed by literature/upstream code, validation-selected (checkpoint epoch only), history missing, or supplementary (none); second-model rationale not recorded |
| Standardized decoder explanation | COMPLETE WITH LIMITATION | Standardized decoder implementation (`torch.nn.TransformerDecoderLayer`) is documented in `docs/standardized_decoder.md`; historical `DecoderLayerWithCrossAttn` behavior remains unrecovered |
| Public GitHub final version | COMPLETE | Release `v1.0.0-tors-reproduction` published on 2026-10-07; the next release follows the redone data preparation and a new official run |

The one-command terminal summary now also prints these ten requested items directly,
including evidence paths and the honest `PARTIAL` states,
so a reviewer does not need to manually locate each artifact in the repository.

Special result checks requested by the professor are preserved explicitly:

- OR R@1: positive mean difference but **not significant** after BH correction over the eight main metrics.
- OR R@3: remains **significant** after the same BH correction.
- Weather / Occasion candidate-ablation OR directions include fresh-vs-archived sign differences; these are reported rather than adjusted.
- The current one-command reuse path verifies the completed 35-unit evidence without manual file moving or manual source edits, then prints the result summary.

