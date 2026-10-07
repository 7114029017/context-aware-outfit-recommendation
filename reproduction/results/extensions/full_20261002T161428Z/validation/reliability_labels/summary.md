# Reliability analysis (2025 notebook P04: table T18, figures F12-F16; A16 from P12)

Ported from `03_實驗與結果_experiments_results/00_控制檢查與附加稽核/source_programs/P04_reliability_xai_analysis.ipynb` (cell 0). Settings as in P04: seed 1, pools of 3,000 candidates,
confidence = sigmoid(50 × (top-1 score − top-2 score)), high-confidence error = Hit@10 of 0 with
confidence ≥ 0.85, ECE over 10 equal-width bins against Hit@10.

## Queries and labels (no model)

- P04's pool rule fills 19 of 152 test categories (9,311 queries); the archived evaluator's rule fills 19 (9,311 queries). Same categories: yes; same queries in the same order: yes.
- Labels against the archived T18 (its 9,311 original-model rows), field by field (`label_check.csv`): all equal.

| Field | Equal | All equal |
|---|---:|---|
| queries (metadata_key) | 9,311 of 9,311 | yes |
| set_id | 9,311 of 9,311 | yes |
| target_item_id | 9,311 of 9,311 | yes |
| target_item_fg | 9,311 of 9,311 | yes |
| fine_category | 9,311 of 9,311 | yes |
| major_category | 9,311 of 9,311 | yes |
| temperature_c | 9,311 of 9,311 | yes |
| weather_group | 9,311 of 9,311 | yes |
| occasion_group | 9,311 of 9,311 | yes |
| style_count | 9,311 of 9,311 | yes |
| style_group | 9,311 of 9,311 | yes |
| category_group | 9,311 of 9,311 | yes |
| weather_text | 9,311 of 9,311 | yes |
| occasion_text | 9,311 of 9,311 | yes |
| style_text | 9,311 of 9,311 | yes |
| title_full_text | 9,311 of 9,311 | yes |
| original_text | 9,311 of 9,311 | yes |

Range checks of the labels (`environment_proxy_checks.csv`, layout of the archived A16): equal to the archived A16: yes.

| Proxy | Valid | Observed range | Mean |
|---|---:|---|---:|
| temperature_c | 9,311 of 9,311 | -78.60 to 28.70 | 21.5311 |
| weather_group | 9,311 of 9,311 | cold:5264; warm:4047 |  |
| occasion_group | 9,311 of 9,311 | casual:1444; formal:1121; unknown:6746 |  |
| style_count/style_group | 9,311 of 9,311 | style_count 0 to 7; high-style:7199; low-style:2112 | 2.1697 |

