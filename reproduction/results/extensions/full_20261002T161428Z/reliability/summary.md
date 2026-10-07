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

## Models (Run full_20261002T161428Z; cuda, float16 autocast)

- original = `main/original_seed1`, proposed = `main/context_seed1` (the run tags of P04 and T18 are kept).

| Model | n | Hit@1 | Hit@10 | Median rank | Mean rank | Mean confidence | High-conf error | ECE | Brier |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| original | 9,311 | 0.0142 | 0.0795 | 228 | 463.36 | 0.7239 | 0.2258 | 0.6444 | 0.5089 |
| proposed | 9,311 | 0.0159 | 0.0898 | 194 | 433.30 | 0.7192 | 0.2067 | 0.6294 | 0.4964 |

Errors (Hit@10 = 0) by type (`error_types.csv`, F14):

| Error type | original | proposed |
|---|---:|---:|
| accessory-led retrieval failure | 4,662 | 4,686 |
| context metadata incomplete | 968 | 988 |
| general retrieval error | 412 | 408 |
| high-confidence semantic mismatch | 2,102 | 1,925 |
| low-style ambiguity | 427 | 468 |

Subgroup Hit@10 (`subgroup_metrics.csv`, F15/F16):

| Subgroup | n | Original | Proposed | Difference |
|---|---:|---:|---:|---:|
| category_group = accessory-led | 6,621 | 0.0760 | 0.0844 | +0.0085 |
| category_group = clothing-led | 2,690 | 0.0881 | 0.1030 | +0.0149 |
| occasion_group = casual | 1,444 | 0.0748 | 0.0796 | +0.0048 |
| occasion_group = formal | 1,121 | 0.0919 | 0.0999 | +0.0080 |
| style_group = high-style | 7,199 | 0.0765 | 0.0882 | +0.0117 |
| style_group = low-style | 2,112 | 0.0895 | 0.0952 | +0.0057 |
| weather_group = cold | 5,264 | 0.0798 | 0.0885 | +0.0087 |
| weather_group = warm | 4,047 | 0.0791 | 0.0914 | +0.0124 |

Files: `reliability_required_fields.csv` (T18 layout, not committed), `performance.csv` (F12), `calibration_bins.csv` (F13), `confidence_bands.csv`, `error_types.csv` (F14), `subgroup_metrics.csv` (F15, F16), `high_confidence_error_examples.csv`, `figures/` (figure_F12_performance_comparison.svg, figure_F13_reliability_diagram.svg, figure_F14_error_types.svg, figure_F15_subgroup_hit10.svg, figure_F16_subgroup_delta_hit10.svg).

