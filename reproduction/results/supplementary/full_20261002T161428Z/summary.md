# Supplementary analyses of run `full_20261002T161428Z`

Source: the official run included in the repository (`full_20261002T161428Z`).

Computed by `reproduction/scripts/supplementary/official_run_analyses.py` from the run's saved
fair-subset outputs (query-level rows and per-seed metrics). No model was loaded and nothing was re-run.
These analyses come after the 35 training units and change none of the run's results.

## Hit@10 by target category (manuscript Figure 4)

| Category | Observations | This run: Full − Original | 2025: Full − Original |
|---|---:|---:|---:|
| dress (all body) | 2735 | +0.0161 | +0.0234 |
| heels (shoes) | 2375 | +0.0114 | +0.0126 |
| sunglasses (accessories) | 1235 | +0.0065 | -0.0040 |
| purse (bags) | 2485 | +0.0064 | +0.0129 |
| pump (shoes) | 2345 | +0.0047 | +0.0102 |
| clutch (bags) | 1985 | +0.0035 | +0.0030 |
| earrings (jewellery) | 2355 | +0.0025 | +0.0047 |
| necklace (jewellery) | 1645 | -0.0012 | +0.0012 |

Largest gain: dress (+0.0161). Categories with a decline in this run: necklace (-0.0012).

## Hit@10 by context subset

| Subset | Rule | Observations | This run: Full − Original | 2025 |
|---|---|---:|---:|---:|
| Cold | temperature <= 22.90 °C | 9920 | +0.0059 | +0.0078 |
| Warm | temperature > 22.90 °C | 7240 | +0.0080 | +0.0120 |
| Formal | formal keyword in occasion or title | 2330 | +0.0056 | +0.0090 |
| Casual | casual keyword in occasion or title | 2020 | +0.0114 | +0.0099 |
| High style | >= 2 style terms | 13260 | +0.0067 | +0.0095 |
| Low style | <= 1 style term | 3900 | +0.0072 | +0.0097 |
| Clothing-led | target in all body, bottoms, tops or outerwear | 2735 | +0.0161 | +0.0234 |
| Accessory-led | target in bags, shoes, accessories, hats, jewellery, scarves or sunglasses | 14425 | +0.0051 | +0.0069 |
| All | all fair-subset queries | 17160 | +0.0068 | +0.0096 |

## Case ranks (five-seed mean)

| Case | Original | Full | No-Weather | No-Occasion | No-Style |
|---|---:|---:|---:|---:|---:|
| purse (Figure A1, set 224499261) | 20.6 | 2.4 | 2.4 | 5.4 | 2.2 |
| dress (Figure A2, set 94771580) | 77.0 | 4.8 | 3.2 | 21.4 | 125.4 |
| sunglasses (Figure A3, set 200099867) | 27.0 | 593.4 | 509.4 | 465.4 | 100.6 |

## Benjamini-Hochberg family (retrospective sensitivity analysis)

Full vs Original on the fair subset. The manuscript corrects over the five reported metrics.

| Metric | Raw p | BH, 5 reported metrics | BH, all 8 metrics |
|---|---:|---:|---:|
| auc | < 0.0001 | < 0.0001 | < 0.0001 |
| fitb_acc | 0.0016 | 0.0027 | 0.0043 |
| recall_at_1 | 0.5162 |  | 0.5162 |
| recall_at_3 | 0.0378 |  | 0.0505 |
| recall_at_5 | 0.0174 |  | 0.0279 |
| recall_at_10 | 0.0467 | 0.0467 | 0.0534 |
| recall_at_30 | 0.0010 | 0.0025 | 0.0039 |
| recall_at_50 | 0.0089 | 0.0111 | 0.0177 |

## Cross-checks

- Overall Hit@10 from the query-level rows equals the mean per-seed Recall@10: Original 0.068531 vs 0.068531; Full 0.075350 vs 0.075350 (match).
- Observations per category equal the 2025 table T15: yes.
- Observations per subset equal the 2025 table T12: yes.
- BH over the five reported metrics equals the run's tables T03/T04: yes.
- Mean differences and p-values of the four comparisons equal the run's table T02 (20 rows): yes.
- Median temperature threshold: 22.90 °C.
