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
