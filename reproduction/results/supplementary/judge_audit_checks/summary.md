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

### Ties at the cut-offs of Table 4-7 (added by this reproduction)

`bottom_p_tie_orders.csv`: how many scores are tied at each judge's cut-off, the smallest and largest
intersection over every order of those ties, and the intersections of 2,000 random orders of the ties (seed 123).

| p | Lowest k | Tied at the cut-off (A, B) | numpy argsort | Thesis | Any order of the ties | Random orders: median [2.5%, 97.5%] |
|---:|---:|---:|---:|---:|---:|---:|
| 0.05 | 1,757 | 330, 1,246 | 669 | 668 | 545 to 917 | 668 [653, 682] |
| 0.10 | 3,514 | 2,378, 844 | 1,338 | 1,323 | 1,269 to 1,605 | 1,333 [1,320, 1,346] |
| 0.20 | 7,028 | 4,745, 3,431 | 2,877 | 2,833 | 2,503 to 3,566 | 2,843 [2,817, 2,870] |

Every thesis value lies within the range of the tie orders: yes. So does the F1 curve of the thesis figure (F19) at 30 of 30 values of p
(`bottom_p_curve_tie_range.csv`). The differences from the thesis come from the order of the tied scores
(numpy's default sort is not stable; the order depends on the input), not from the scores.

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

## Human audit scores (2025 tables T30 and A41; figures F30-F32)

Weighted human score per description and checklist against the judge's score (P06 cell 10).
`audit_score_metrics.csv` equals the archived T30: yes; `audit_sample_scores.csv`
equals the archived A41 (30 descriptions): yes. Figures: `figures/figure_F30_audit_sampling_coverage.svg`, `figure_F31_audit_error_metrics.svg`, `figure_F32_audit_score_scatter.svg`.

## Other 2025 judge figures and T20

- `figures/figure_F18_quantile_confusion.svg` and `judge_quantile_confusion.csv`: the decile confusion
  matrix behind the QWK of T19 (P05 cell 3).
- `figures/figure_F20_prompt_robustness.svg`: mean absolute difference of P0-R2, P1 and P2 for each judge
  (Qwen3-VL P0-R2: 0.0048, Qwen3-VL P1: 0.0709, Qwen3-VL P2: 0.0506, Gemma-3 P0-R2: 0.0461, Gemma-3 P1: 0.0713, Gemma-3 P2: 0.0521).
- The archived T20 equals the correlation columns of T19: yes; the pipeline's
  secondary step recomputes T19 (with its 5,000 bootstrap draws).

## Judge outputs in the training and evaluation code (2025 tables A05 and A06)

Search terms (2025): phase3_scores, Qwen3VL32B, phase3_scores_Qwen3VL32B, phase3_scores_Gemma3, checklist_C, LLM-as-a-Judge, judge_score, rewrite_qwen, rewrite_gemma. `judge_reference_scan.csv`, `judge_input_roles.csv`.

- The 2025 list of 13 files: 11 found in the repository and free of judge terms; 2 not in the repository (the two t-test notebooks).
- Files executed for the reproduction's 35 units (29: the archived model code, and the scripts named or imported by the split freezing, the main and ablation stages and the statistics, followed recursively): none references a judge output.
- Other reproduction scripts that read judge outputs (post-hoc analyses, not run by the 35 units): `preflight_remaining_thesis_reproduction.py`, `reproduce_remaining_thesis.py`, `checklist_coverage.py`, `judge_audit_checks.py`.
