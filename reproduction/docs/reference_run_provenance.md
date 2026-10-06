# Reference full-run provenance

This document describes the already-completed 2026 reproduction reference
run. Repository restructuring did not modify it. A later post-restructure
`--fresh` 35-unit run (`full_20260924T102513Z`) reproduced it bit-identically;
see `post_restructure_fresh_run_verification.md`.

## Run identity

- Training commit: `eb081e7a426cf5bd91acd8bb0f92a8398f08f45e`
- Main units: 10
- Candidate-ablation units: 25
- Total units: 35
- Run status: `PASSED`
- Start: 2026-09-21T17:52:17Z
- Finish: 2026-09-23T19:34:44Z
- Total run wall time: 49:42:27

The machine-readable seed-level index is:

`reproduction/results/summary/reference_20260921T175217Z/reference_seed_index.csv`

## Per-seed provenance

The index records, for each of the 35 units:

- experiment family, variant and seed;
- completed full-run Git commit;
- training/evaluation manifest SHA-256;
- CP and CIR checkpoint SHA-256;
- saved metric-file SHA-256;
- row-level detail SHA-256 where applicable;
- reconstructed candidate subset SHA-256 where applicable;
- runtime derived from timestamps already preserved in saved execution logs.

Runtime information was recovered for 35/35 units from saved
logs. No model was rerun to create this index.

## Configuration hash

`documented_config_bundle_sha256` identifies the current reproducibility
configuration bundle:

- `main_hybrid_attention.yaml`
- `cp.yaml`
- `fitb.yaml`
- `or.yaml`

This is a post-run traceability artifact. It must NOT be interpreted as an
original 2025 hyperparameter-search trial/config hash.

## Historical limitations

The historical hyperparameter-search candidate space and per-trial results
were not recovered. They must not be invented.

The original historical fair-subset memberwise ID file was not recovered.
The 2026 ablation uses an explicitly labeled reconstructed candidate set.

The historical `DecoderLayerWithCrossAttn` implementation was not recovered.
The completed fresh reproduction used the standardized decoder implementation
(`torch.nn.TransformerDecoderLayer`, see `docs/standardized_decoder.md`); its
recorded run manifests still carry the earlier wording "temporary ...
candidate" and are kept unchanged as historical records.
