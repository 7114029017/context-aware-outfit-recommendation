# Judge and audit checks

Computed by `reproduction/scripts/supplementary/judge_audit_checks.py` from the preserved judge scores
and the preserved human audit; no judge or model was run.

## Score distribution (thesis Figure 4-1; P05 cell 3)

35,140 descriptions scored by both judges. `figures/figure_4_1_judge_score_histogram.svg`.

| Judge | Mean | SD | IQR | P(score < 0.8) | P(score < 0.9) | Thesis |
|---|---:|---:|---:|---:|---:|---:|
| A(Qwen) | 0.8279 | 0.1422 | 0.2273 | 0.3284 | 0.7041 | 0.828 ± 0.142 |
| B(Gemma) | 0.9596 | 0.0892 | 0.0606 | 0.0406 | 0.1005 | 0.960 ± 0.089 |

Mean and SD equal the 2025 table T19 to all digits: yes. The 100 bars equal the thesis figure (the archived SVG F17): yes.

## Bottom-p% sensitivity (thesis Figure 4-3; P05 cell 3)

p = 1% to 30%, bootstrap B = 500, seed 123. `bottom_p_sensitivity_band.csv`,
`figures/figure_4_3_bottom_p_overlap_band.svg`, `figures/bottom_p_lift_band.svg`.

| p | Intersection | Thesis Table 4-7 | Jaccard | Thesis | F1 | Thesis |
|---:|---:|---:|---:|---:|---:|---:|
| 0.05 | 669 | 668 | 0.2351 | 0.2347 | 0.3808 | 0.3802 |
| 0.10 | 1338 | 1323 | 0.2351 | 0.2319 | 0.3808 | 0.3765 |
| 0.20 | 2877 | 2833 | 0.2574 | 0.2524 | 0.4094 | 0.4031 |

Many descriptions share a score, so which ones fall in the lowest p% depends on how ties are ordered
(`numpy.argsort`); the intersections differ from the thesis by up to 44 descriptions, as in the
pipeline's recomputation of Table 4-7. The curve and the band are computed with the same ordering.
Largest difference from the thesis figure (the archived SVG F19), over the 30 values of p: jaccard 0.0122, jaccard_lo 0.0024, jaccard_hi 0.0026, f1 0.0156, f1_lo 0.0032, f1_hi 0.0034.

## Item disagreement between judge and human audit (2025 table T31, thesis Figure 4-4; P06 cell 10)

30 audited descriptions (A36), 750 human answers (A38). `item_disagreement.csv`,
`figures/figure_4_4_item_disagreement.svg`.

| Rank | Item | Weight | Disagreement | Weighted | Model − human |
|---:|---|---:|---:|---:|---:|
| 1 | Gemma c15 (grounding) | 4 | 0.3333 | 0.0404 | -0.2667 |
| 2 | Qwen c8 (grounding) | 4 | 0.2000 | 0.0364 | -0.1333 |
| 3 | Qwen c2 (weather_style_occasion) | 3 | 0.2667 | 0.0364 | -0.2667 |
| 4 | Gemma c1 (query_register) | 2 | 0.5667 | 0.0343 | 0.5667 |
| 5 | Qwen c1 (weather_style_occasion) | 3 | 0.2333 | 0.0318 | 0.1000 |
| 6 | Qwen c4 (weather_style_occasion) | 3 | 0.2333 | 0.0318 | -0.1667 |
| 7 | Qwen c3 (query_register) | 2 | 0.3333 | 0.0303 | 0.2000 |
| 8 | Gemma c3 (weather_style_occasion) | 3 | 0.3000 | 0.0273 | 0.0333 |
| 9 | Gemma c8 (weather_style_occasion) | 3 | 0.3000 | 0.0273 | -0.2333 |
| 10 | Gemma c5 (grounding) | 4 | 0.2000 | 0.0242 | -0.1333 |
| 11 | Gemma c2 (weather_style_occasion) | 3 | 0.2667 | 0.0242 | -0.2667 |
| 12 | Gemma c7 (query_register) | 2 | 0.4000 | 0.0242 | 0.0667 |

Equal to the archived T31 in all 11 compared columns and the order of the 25 rows: yes (the Chinese translation of each question, a display aid, is not recomputed).

## Judge outputs in the training and evaluation code (2025 tables A05 and A06)

Search terms (2025): phase3_scores, Qwen3VL32B, phase3_scores_Qwen3VL32B, phase3_scores_Gemma3, checklist_C, LLM-as-a-Judge, judge_score, rewrite_qwen, rewrite_gemma. `judge_reference_scan.csv`, `judge_input_roles.csv`.

- The 2025 list of 13 files: 11 found in the repository and free of judge terms; 2 not in the repository (the two t-test notebooks).
- Files executed for the reproduction's 35 units (29: the archived model code, and the scripts named or imported by the split freezing, the main and ablation stages and the statistics, followed recursively): none references a judge output.
- Other reproduction scripts that read judge outputs (post-hoc analyses, not run by the 35 units): `preflight_remaining_thesis_reproduction.py`, `reproduce_remaining_thesis.py`, `checklist_coverage.py`, `judge_audit_checks.py`.
