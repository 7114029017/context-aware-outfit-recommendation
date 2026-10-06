# Reproduction results

This directory contains preserved outputs from the completed 2026 thesis
reproduction. These are versioned reference results, not newly generated
outputs from every later documentation or repository refactor.

## Reference full run

Directory:

- `raw/reference_20260921T175217Z/`
- `summary/reference_20260921T175217Z/`

Original execution:

- Start: 2026-09-21T17:52:17Z
- Finish: 2026-09-23T19:34:44Z
- Duration: 49:42:27
- Git commit used for training:
  `eb081e7a426cf5bd91acd8bb0f92a8398f08f45e`
- Status: `PASSED`
- Main experiments: 10 = Original / Context x seeds 1–5
- Candidate fair-subset ablations: 25 = 5 variants x seeds 1–5

The repository was reorganized after this completed run. The restructuring
does NOT represent another 35-unit training execution.

## raw/

`raw/reference_20260921T175217Z/` preserves seed-level evaluation outputs
and run manifests needed to inspect or recalculate the reported summaries.

Large model checkpoints are intentionally not duplicated here. Their
identities are recorded by SHA-256 in the corresponding manifests, and the
original complete local run remains preserved by the handoff maintainer.

## summary/

`summary/reference_20260921T175217Z/` contains:

- five-seed main summaries,
- candidate fair-subset ablation summaries,
- Chapter 4 reproduction comparison,
- paired tests and BH-adjusted statistics,
- preserved-output secondary analyses.

## Important provenance limitation

The fair-subset IDs in this reproduction are an independently reconstructed
candidate set. They match the documented counts and evaluation scope, but
they are NOT claimed to be the missing historical memberwise ID file.

The historical `DecoderLayerWithCrossAttn` implementation was also not
recovered. The fresh training reproduction used the documented temporary
standard PyTorch Transformer decoder candidate reconstruction.
