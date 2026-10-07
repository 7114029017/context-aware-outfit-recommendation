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

## Models (Preserved 2025 checkpoints; cpu, float32; check run)

- original = `cir_old_seed1`, proposed = `cir_new_seed1` from `02_模型訓練和驗證_model_training_validation/main_hybrid_attention_checkpoints/`.

| Model | n | Hit@1 | Hit@10 | Median rank | Mean rank | Mean confidence | High-conf error | ECE | Brier |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| original | 9,311 | 0.0139 | 0.0779 | 230 | 463.32 | 0.7203 | 0.2197 | 0.6424 | 0.5051 |
| proposed | 9,311 | 0.0153 | 0.0870 | 205 | 441.76 | 0.7185 | 0.2066 | 0.6315 | 0.4974 |

Errors (Hit@10 = 0) by type (`error_types.csv`, F14):

| Error type | original | proposed |
|---|---:|---:|
| accessory-led retrieval failure | 4,700 | 4,748 |
| context metadata incomplete | 1,004 | 982 |
| general retrieval error | 394 | 397 |
| high-confidence semantic mismatch | 2,046 | 1,924 |
| low-style ambiguity | 442 | 450 |

Subgroup Hit@10 (`subgroup_metrics.csv`, F15/F16):

| Subgroup | n | Original | Proposed | Difference |
|---|---:|---:|---:|---:|
| category_group = accessory-led | 6,621 | 0.0739 | 0.0819 | +0.0080 |
| category_group = clothing-led | 2,690 | 0.0877 | 0.0996 | +0.0119 |
| occasion_group = casual | 1,444 | 0.0713 | 0.0769 | +0.0055 |
| occasion_group = formal | 1,121 | 0.0856 | 0.0901 | +0.0045 |
| style_group = high-style | 7,199 | 0.0745 | 0.0861 | +0.0117 |
| style_group = low-style | 2,112 | 0.0895 | 0.0900 | +0.0005 |
| weather_group = cold | 5,264 | 0.0809 | 0.0861 | +0.0051 |
| weather_group = warm | 4,047 | 0.0739 | 0.0882 | +0.0143 |

## Against the 2025 outputs

Per query against the archived T18 (`agreement_with_archived_t18.csv`):

| Model | Queries | Ranks equal | Largest rank difference | Hit@10 equal | Top-10 lists equal | Largest confidence difference | Error type equal |
|---|---:|---:|---:|---:|---:|---:|---:|
| original | 9,311 | 7,342 | 8 | 9,310 | 8,775 | 0.005378 | 9,303 |
| proposed | 9,311 | 7,387 | 5 | 9,310 | 8,746 | 0.005178 | 9,307 |

Performance table: P04's printed values, the same metrics computed here from the archived T18 rows, and from this re-evaluation:

| Model | Metric | P04 printed | From archived T18 | Re-evaluated |
|---|---|---:|---:|---:|
| original | Hit@1 | 0.013855 | 0.013855 | 0.013855 |
| original | Hit@10 | 0.077972 | 0.077972 | 0.077865 |
| original | Median Rank | 230.000000 | 230.000000 | 230.000000 |
| original | Mean Rank | 463.324777 | 463.324777 | 463.322522 |
| original | Avg Confidence | 0.720271 | 0.720271 | 0.720287 |
| original | Error Rate | 0.922028 | 0.922028 | 0.922135 |
| original | High-conf Error Rate | 0.219740 | 0.219740 | 0.219740 |
| original | ECE | 0.642299 | 0.642299 | 0.642422 |
| original | Brier Score | 0.504966 | 0.504966 | 0.505062 |
| proposed | Hit@1 | 0.015251 | 0.015251 | 0.015251 |
| proposed | Hit@10 | 0.087101 | 0.087101 | 0.086994 |
| proposed | Median Rank | 205.000000 | 205.000000 | 205.000000 |
| proposed | Mean Rank | 441.759102 | 441.759102 | 441.762969 |
| proposed | Avg Confidence | 0.718550 | 0.718550 | 0.718544 |
| proposed | Error Rate | 0.912899 | 0.912899 | 0.913006 |
| proposed | High-conf Error Rate | 0.206315 | 0.206315 | 0.206637 |
| proposed | ECE | 0.631449 | 0.631449 | 0.631550 |
| proposed | Brier Score | 0.497345 | 0.497345 | 0.497410 |

Metric code against P04's printed table (from the archived T18 rows, six decimals): equal.

Files: `reliability_required_fields.csv` (T18 layout, not committed), `performance.csv` (F12), `calibration_bins.csv` (F13), `confidence_bands.csv`, `error_types.csv` (F14), `subgroup_metrics.csv` (F15, F16), `high_confidence_error_examples.csv`, `figures/` (figure_F12_performance_comparison.svg, figure_F13_reliability_diagram.svg, figure_F14_error_types.svg, figure_F15_subgroup_hit10.svg, figure_F16_subgroup_delta_hit10.svg).

