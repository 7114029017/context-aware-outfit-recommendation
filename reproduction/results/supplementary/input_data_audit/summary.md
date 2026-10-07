# Input data audit

Computed by `reproduction/scripts/supplementary/input_data_audit.py`. The audited files are the 2025
inputs that the official run uses unchanged; nothing was regenerated.

## Temperature at the start of the generated descriptions

| Split | Descriptions | Matching | Mismatching | Decimal digit only | Minus sign dropped | No pattern |
|---|---:|---:|---:|---:|---:|---:|
| train | 16995 | 14918 | 2077 | 2036 | 41 | 0 |
| valid | 3000 | 3000 | 0 | 0 | 0 | 0 |
| test | 15145 | 15105 | 40 | 0 | 0 | 40 |
| all | 35140 | 33023 | 2117 | 2036 | 41 | 40 |

2117 of 35140 descriptions (6.0%) state a temperature that differs
from `TSUB_target_C`. The train descriptions use an integer format and the valid and test descriptions one
decimal place; the preserved post-processing writes only the latter.

Feature check: 219 of 219 description pairs with identical text but different temperature
references have identical context-aware text features (the features encode the stored descriptions as written).

## Thermal mapping

- Equation (2) reproduces the stored temperature reference for 35140 of 35140 outfits.
- Outside McIntyre's stated range (M < 150 W/m², Icl < 1.5 clo): 6348 outfits (18.1%); Icl ≥ 1.5: 5934; M ≥ 150: 524.

## Other checks

- Item IDs shared between the disjoint split files: train and test: 84; train and valid: 3781; valid and test: 34.
- Counterfactual pairs whose replacement also removed words of a non-target factor: CF16 (style: weekend), CF17 (occasion: evening), CF18 (occasion: evening / date), CF21 (weather: summer).
- Occasion pairs without an occasion fragment: CF12.
- Standalone "formal" fragments: occasion 18, style 1, weather 0.
- Outfits whose activity is not in the 457-entry MET candidate list (455 distinct descriptions): 245 (14 activities).

## CLO distribution (thesis Table D-3 and Figure D-4)

Adding_CLO cell 17 recomputed on the preserved CLO estimates (`clo_distribution_summary.csv`,
`figures/figure_D_4_clo_distribution.svg`).

| Statistic | Recomputed | Thesis Table D-3 |
|---|---:|---:|
| count | 35140 | 35140 |
| mean | 1.0066 | 1.0066 |
| std | 0.5586 | 0.5586 |
| min | 0.0000 | 0.0000 |
| q1 | 0.6300 | 0.6300 |
| median | 0.9300 | 0.9300 |
| q3 | 1.2125 | 1.2125 |
| max | 5.2000 | 5.2000 |
| iqr | 0.5825 | 0.5825 |
| iqr_lower_bound | -0.2437 | -0.2437 |
| iqr_upper_bound | 2.0862 | 2.0862 |
| n_outliers | 1726 | 1726 |

All 12 statistics equal Table D-3 at four decimals: yes. Records per file: train 16995, test 15145, valid 3000; the zoomed panels end at 2.73.

## Category threshold (thesis Section 5.2)

The archived evaluator builds, for every fine-grained category of the test split, a candidate pool of at
most 3,000 items (test items first, then train items) and evaluates only the categories whose pool
reaches 3,000 (`category_threshold_check.csv`).

- CIR test evaluation: 19 of 152 test categories; 9311 of 15145 test FITB questions have their target in them (the official run evaluates 9,311 queries: same).
- Male-labelled categories in the data: 42 (42 of them by the first label of a duplicated ID in categories.csv, as the 2025 programs read it; category 21 is also listed as 'tshirt'). The largest pool among them is category 21 (male T-shirt / tshirt) with 2,725 items; none reaches 3,000, so no male-labelled item is a CIR target (421 test questions with a male-labelled target are skipped).
- Male-labelled items are nevertheless part of the training and test outfits: 1,930 items in 1,932 train outfits, 1,940 items in 1,719 test outfits. The 3,000 rule selects the CIR evaluation categories only; it does not filter the data used for training or for the compatibility and FITB evaluation, whereas Section 5.2 of the thesis says that only categories with more than 3,000 items were used in training.
- Validation during CIR training (train_cir.py, the same rule on the validation split): 7 of 127 categories, because the train loop stops at category 282 (male swim shorts), the first train category without validation items. Only the logged validation recall is affected; checkpoints are selected by validation FITB accuracy.
