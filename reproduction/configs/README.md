# Reproduction configs

These YAML files are artifact-facing descriptions derived from archived code/log evidence. They do not replace the archived Python implementation.

Canonical task configs:

- `cp.yaml` — CP training + CP AUC evaluation; also defines the CP checkpoint used by FITB and CIR/OR.
- `fitb.yaml` — FITB evaluation contract. FITB is not a separately trained model; it evaluates a CP checkpoint.
- `or.yaml` — CIR training / OR retrieval evaluation and candidate-pool rules.
- `llm_judge.yaml` — preserved Judge models/checklists/outputs, robustness analysis, and explicit missing-generation provenance.
- `main_hybrid_attention.yaml` — earlier consolidated recovery of the shared Hybrid Attention settings.
- `paths.reproduction.example.yaml` — portable path example.

Recovered shared main-model settings include seeds 1–5, 100 epochs, batch size 50, Adam at 5e-5, gradient clipping 0.5, hard negatives from epoch 40, 16 heads, 3 transformer layers, dropout 0.1, float16 + GradScaler, and validation FITB checkpoint selection.

Important provenance constraints:

1. The exact historical `DecoderLayerWithCrossAttn` implementation is not archived. CP candidate reconstruction must not be described as source-exact history.
2. Exact upstream revisions for some precomputed feature encoders are not established by the handoff.
3. Exact Judge generation parameters and P0/P1/P2 generation prompt files are incomplete. `llm_judge.yaml` leaves those fields null rather than inventing values.
4. Historical robustness sample-generation provenance is partial even though preserved Judge numerical outputs are internally consistent and reproduce exactly.

The final scientific status is recorded in `reproduction/docs/tors_compliance.md`.
