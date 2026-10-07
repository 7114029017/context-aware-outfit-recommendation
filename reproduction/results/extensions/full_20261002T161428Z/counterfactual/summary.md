# Counterfactual context sensitivity (manuscript Tables 4 and 9)

Ported from notebook P16. Pairs: the archived A44 file (24 pairs). Baseline condition as in P16 and the
manuscript: the stored context-aware feature. Device: cpu.

Encoder check (24 context-aware descriptions): cosine between the stored feature and the live FashionCLIP encoding, min 1.000000, mean 1.000000.

## Run `full_20261002T161428Z`, Context models of seeds 1, 2, 3, 4, 5

| Scope | Seed | Mean rank before → after | Mean rank change | Top-1 changed | Top-5 Jaccard |
|---|---|---|---:|---:|---:|
| Overall | 1 | 507.0 → 573.7 | +66.7 | 0.708 | 0.386 |
| Weather | 1 | 777.5 → 674.0 | -103.5 | 0.500 | 0.503 |
| Occasion | 1 | 223.0 → 229.4 | +6.4 | 0.625 | 0.569 |
| Style | 1 | 520.5 → 817.8 | +297.2 | 1.000 | 0.085 |
| Overall | 2 | 607.3 → 634.9 | +27.5 | 0.708 | 0.358 |
| Weather | 2 | 850.8 → 738.9 | -111.9 | 0.625 | 0.574 |
| Occasion | 2 | 230.2 → 266.5 | +36.2 | 0.500 | 0.426 |
| Style | 2 | 741.0 → 899.2 | +158.2 | 1.000 | 0.073 |
| Overall | 3 | 610.7 → 750.4 | +139.7 | 0.708 | 0.361 |
| Weather | 3 | 927.8 → 810.0 | -117.8 | 0.750 | 0.473 |
| Occasion | 3 | 341.5 → 384.4 | +42.9 | 0.500 | 0.505 |
| Style | 3 | 562.9 → 1056.8 | +493.9 | 0.875 | 0.104 |
| Overall | 4 | 616.0 → 701.5 | +85.5 | 0.500 | 0.350 |
| Weather | 4 | 882.6 → 868.2 | -14.4 | 0.125 | 0.503 |
| Occasion | 4 | 329.1 → 331.2 | +2.1 | 0.375 | 0.466 |
| Style | 4 | 636.2 → 904.9 | +268.6 | 1.000 | 0.081 |
| Overall | 5 | 589.6 → 649.9 | +60.3 | 0.667 | 0.380 |
| Weather | 5 | 790.1 → 752.0 | -38.1 | 0.375 | 0.557 |
| Occasion | 5 | 286.0 → 298.1 | +12.1 | 0.625 | 0.486 |
| Style | 5 | 692.8 → 899.6 | +206.9 | 1.000 | 0.099 |

Mean ± SD over the seeds: `table9_seeds_mean_sd.csv`. Against the live-encoded baseline: `table9_live_baseline_<seed>.csv`.

## Validation with the preserved 2025 seed-1 model

| Condition | Ranks equal to A45 | Largest rank difference | Top-5 lists equal to A45 |
|---|---:|---:|---:|
| context_aware | 17/24 | 1 | 24/24 |
| counterfactual | 22/24 | 1 | 24/24 |

| Scope | Manuscript (change, top-1, Jaccard) | Recomputed |
|---|---|---|
| Overall | +17.0, 0.500, 0.351 | +17.2, 0.500, 0.351 |
| Weather | -146.4, 0.125, 0.508 | -146.0, 0.125, 0.508 |
| Occasion | -11.9, 0.375, 0.416 | -11.9, 0.375, 0.416 |
| Style | +209.1, 1.000, 0.130 | +209.4, 1.000, 0.130 |

