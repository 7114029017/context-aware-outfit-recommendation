# Post-training statistical evidence

Main: original vs context, full-data five paired seeds, two-sided paired t-test;
BH adjusted over the eight prespecified main metrics (2 CP + 6 OR).
95% CI covers the paired mean difference. These are new calculations, not archived thesis p-values.

| Main metric | Mean context−original | raw p | BH p (8 tests) | BH significance at .05 |
|---|---:|---:|---:|---|
| cp_auc | +0.01883374 | 0.000783481 | 0.00125357 | True |
| cp_fitb | +0.01247937 | 0.00610723 | 0.00697969 | True |
| or_r1 | +0.00128880 | 0.0947427 | 0.0947427 | False |
| or_r3 | +0.00382343 | 0.00332533 | 0.00443377 | True |
| or_r5 | +0.00788315 | 0.000157257 | 0.000419353 | True |
| or_r10 | +0.01452046 | 0.000432697 | 0.000865395 | True |
| or_r30 | +0.02012673 | 0.00015465 | 0.000419353 | True |
| or_r50 | +0.02291913 | 6.18231e-05 | 0.000419353 | True |

## Fair-subset ablation factors

Definition: full context − without factor. Negative means removing the factor had a higher mean for that metric.
Archived comparison uses T01 aggregates, NOT proven-identical historical memberwise IDs.
Signs treat absolute mean differences <= 1e-12 as zero (floating point roundoff only).

| Factor | Metric | Fresh delta | Fresh sign | Archived T01 delta | Archived sign |
|---|---|---:|---|---:|---|
| weather | AUC | +0.00152848 | positive | 0.0023999999999999577 | positive |
| weather | FITB Acc | +0.00306143 | positive | 0.0037000000000000366 | positive |
| weather | Recall@1 | -0.00000000 | zero | N/A | not_available |
| weather | Recall@3 | -0.00099068 | negative | N/A | not_available |
| weather | Recall@5 | -0.00139860 | negative | N/A | not_available |
| weather | Recall@10 | -0.00005828 | negative | 0.0021000000000000046 | positive |
| weather | Recall@30 | +0.00011655 | positive | 0.002999999999999975 | positive |
| weather | Recall@50 | -0.00518648 | negative | 0.002599999999999991 | positive |
| occasion | AUC | +0.00478017 | positive | 0.0029000000000000137 | positive |
| occasion | FITB Acc | +0.00235650 | positive | 0.0030999999999999917 | positive |
| occasion | Recall@1 | -0.00145688 | negative | N/A | not_available |
| occasion | Recall@3 | -0.00163170 | negative | N/A | not_available |
| occasion | Recall@5 | -0.00110723 | negative | N/A | not_available |
| occasion | Recall@10 | -0.00367133 | negative | 0.0007000000000000062 | positive |
| occasion | Recall@30 | -0.00168998 | negative | 0.0013999999999999846 | positive |
| occasion | Recall@50 | -0.00413753 | negative | 0.00040000000000001146 | positive |
| style | AUC | +0.01346521 | positive | 0.015499999999999958 | positive |
| style | FITB Acc | +0.01258812 | positive | 0.01639999999999997 | positive |
| style | Recall@1 | +0.00145688 | positive | N/A | not_available |
| style | Recall@3 | +0.00273893 | positive | N/A | not_available |
| style | Recall@5 | +0.00524476 | positive | N/A | not_available |
| style | Recall@10 | +0.01060606 | positive | 0.010999999999999996 | positive |
| style | Recall@30 | +0.02325175 | positive | 0.01959999999999998 | positive |
| style | Recall@50 | +0.02639860 | positive | 0.02410000000000001 | positive |
