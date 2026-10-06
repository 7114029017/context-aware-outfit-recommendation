# Data access and provenance

## Dataset

The reproduction uses the **Polyvore Outfits Disjoint** split.

Expected scope:

| Scope | Count |
|---|---:|
| train outfits | 16,995 |
| validation outfits | 3,000 |
| test outfits | 15,145 |
| CP compatibility test pairs | 30,290 |
| FITB questions | 15,145 |
| OR evaluable pairs | 9,311 |
| reconstructed candidate ablation IDs | 21,903 |
| candidate-ablation CIR evaluable queries | 3,432 |

## Data source and bootstrap

The current reproduction bootstrap is configured to retrieve the required
Polyvore annotations and metadata from:

`Stylique/Polyvore`

For a future clean setup, run:

    bash reproduction/scripts/bootstrap_data.sh

The bootstrap downloads the annotation and metadata files required by the
reproduction pipeline. It does not download or redistribute the original
Polyvore images.

The already-completed 2026 reference reproduction does not need to be
downloaded or rerun during repository restructuring.

## Required annotation and metadata files

The bootstrap expects the Polyvore Disjoint files used by the archived
training and evaluation code, including:

- `disjoint/train.json`
- `disjoint/valid.json`
- `disjoint/test.json`
- `disjoint/compatibility_train.txt`
- `disjoint/compatibility_valid.txt`
- `disjoint/compatibility_test.txt`
- `disjoint/fill_in_blank_train.json`
- `disjoint/fill_in_blank_valid.json`
- `disjoint/fill_in_blank_test.json`
- `categories.csv`
- `polyvore_item_metadata.json`
- `polyvore_outfit_titles.json`

## Precomputed feature artifacts

The model reproduction additionally uses preserved feature artifacts under:

`02_模型訓練和驗證_model_training_validation/fashionclip_data/`

Before a new reproduction run, these files are checked by:

`reproduction/scripts/verify_feature_files.py`

The verifier checks the expected byte size and SHA-256 identity of the eight
required feature files.

## Fixed main split manifests

The 2026 reproducibility artifact fixes the following ordered ID manifests
under:

`reproduction/splits/`

- `train_ids.csv`
- `validation_ids.csv`
- `test_ids.csv`
- `cp_ids.csv`
- `fitb_ids.csv`
- `or_ids.csv`

Their SHA-256 values are stored in:

`reproduction/splits/SHA256SUMS.txt`

The source dataset identity used to regenerate them is recorded in:

`reproduction/splits/source_dataset_manifest.json`

For a clean reproduction, `freeze_splits.py` regenerates the six files from
the source dataset. The preflight compares the regenerated files
byte-for-byte with the preserved references.

For OR evaluation, candidate pools reproduce the archived evaluator's
candidate-pool construction behavior.

## Reproducibly reconstructed fair subset

The 2026 artifact also preserves the fair subset used by Tables 4-13 to 4-16:

`reproduction/splits/fair_subset/`

The reconstruction rule is:

Select WOS v5 merged-retry rows whose weather, occasion, and style fragment
lists each contain at least one non-blank string.

This produces:

| Scope | Count |
|---|---:|
| all fair-subset IDs | 21,903 |
| train | 10,225 |
| validation | 1,748 |
| test | 9,930 |
| CIR-evaluable test queries | 3,432 |

The ID files and the 3,432 query IDs are fixed with SHA-256; see
`reproduction/docs/fair_subset.md`.

Important provenance limitation:

This is a reproducibly reconstructed **2026 subset**, used as the formal
subset of the revised manuscript.

The original historical fair-subset generation program, original memberwise
ID file, and historical ID-file SHA-256 were not recovered. Therefore this
repository does not claim that the reconstructed 21,903 members are
historically identical member-by-member to the original experiment subset.

## Dataset access date

The exact original 2025 dataset download or access date has not been
recovered from the archived handoff evidence and is not guessed here.

The 2026 reproduction uses the already-downloaded local dataset identified
by the source-file hashes recorded in:

`reproduction/splits/source_dataset_manifest.json`

If a future release performs a new clean download, that new access date
should be recorded at release time.

## Licensing and redistribution

Dataset and image rights are upstream third-party rights and are separate
from any license later applied to reproduction code.

This repository must not claim ownership of the original Polyvore images or
silently relicense upstream data.

Before public release, the maintainer should verify the then-current
upstream dataset terms and confirm which dataset and feature artifacts may
be redistributed.

Where third-party data cannot be redistributed, this artifact provides
download instructions, fixed ID manifests, source hashes and validation
scripts instead.

## Completed reference reproduction

The completed full reproduction is preserved under:

`reproduction/results/raw/reference_20260921T175217Z/`

and:

`reproduction/results/summary/reference_20260921T175217Z/`

Training commit:

`eb081e7a426cf5bd91acd8bb0f92a8398f08f45e`

Run status:

`PASSED`

The completed run contains 10 main training units and 25 candidate-ablation
training units.

Repository restructuring after the completed run did not trigger another
35-unit training execution.
