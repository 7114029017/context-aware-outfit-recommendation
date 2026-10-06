# Frozen split manifests

Immutable ID manifests generated from the exact PO-D source used in the rerun.
Do not create these files from paper counts alone: they are derived from the
source dataset and evaluation logic, then hashed.

## Main experiment

- `train_ids.csv`
- `validation_ids.csv`
- `test_ids.csv`
- `cp_ids.csv`
- `fitb_ids.csv`
- `or_ids.csv`
- `SHA256SUMS.txt`
- `source_dataset_manifest.json`

Every run regenerates these files and compares them byte-for-byte with the
copies here.

## Fair subset (Tables 4-13 to 4-16)

`fair_subset/` holds the reproducibly reconstructed fair subset:

- `fair_subset_ids.txt`: 21,903 outfit IDs
- `train_ids.txt`, `valid_ids.txt`, `test_ids.txt`: 10,225 / 1,748 / 9,930
- `or_query_ids.csv`: the 3,432 evaluable OR (CIR) queries
- `fair_subset_reconstruction_manifest.json`, `fair_subset_cir_scope_audit.json`
- `SHA256SUMS.txt`

Every full run rebuilds the subset under `ablation/fair_subset_reconstruction/`
and re-exports the query IDs to `ablation/fair_subset_or_query_ids.csv`.
Construction rule, exclusions and verification are in
`reproduction/docs/fair_subset.md`.
