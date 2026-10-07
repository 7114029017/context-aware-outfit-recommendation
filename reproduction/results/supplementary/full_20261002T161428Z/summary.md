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

## Subsets: median rank and factor contributions (2025 table T12; thesis Figures 4-9 and 4-10)

Median rank improvement = median rank of Original minus median rank of Full (positive: the target
moves up). The last three columns are Hit@10 of Full minus Hit@10 of the simplified description.
Full table: `subset_robustness_summary.csv` (columns of T12).

| Subset | Observations | Median rank, Original → Full | Improvement | 2025 | No-Weather → Full | No-Occasion → Full | No-Style → Full |
|---|---:|---:|---:|---:|---:|---:|---:|
| Cold | 9920 | 256.0 → 231.0 | +25.0 | +38.0 | -0.0004 | -0.0032 | +0.0094 |
| Warm | 7240 | 250.0 → 213.0 | +37.0 | +39.0 | +0.0004 | -0.0043 | +0.0123 |
| All (union) (Weather) | 17160 | 255.0 → 223.0 | +32.0 | +37.5 | -0.0001 | -0.0037 | +0.0106 |
| Formal | 2330 | 247.0 → 207.5 | +39.5 | +37.0 | -0.0021 | -0.0009 | +0.0112 |
| Casual | 2020 | 268.5 → 211.0 | +57.5 | +67.5 | +0.0015 | -0.0025 | +0.0064 |
| All (union) (Occasion) | 4310 | 255.0 → 209.0 | +46.0 | +44.5 | -0.0005 | -0.0012 | +0.0093 |
| High style (≥2 terms) | 13260 | 261.5 → 231.5 | +30.0 | +38.0 | +0.0002 | -0.0041 | +0.0100 |
| Low style (≤1 term) | 3900 | 233.0 → 201.0 | +32.0 | +38.5 | -0.0008 | -0.0023 | +0.0126 |
| All (union) (Style Richness) | 17160 | 255.0 → 223.0 | +32.0 | +37.5 | -0.0001 | -0.0037 | +0.0106 |
| Clothing-led | 2735 | 176.0 → 140.0 | +36.0 | +42.0 | +0.0029 | -0.0059 | +0.0325 |
| Accessory-led | 14425 | 272.0 → 243.0 | +29.0 | +32.0 | -0.0006 | -0.0033 | +0.0064 |
| All (union) (Category) | 17160 | 255.0 → 223.0 | +32.0 | +37.5 | -0.0001 | -0.0037 | +0.0106 |
| Full dataset | 17160 | 255.0 → 223.0 | +32.0 | +37.5 | -0.0001 | -0.0037 | +0.0106 |

Table T13 (`subset_delta_hit10_pivot.csv`) and the 2025 figures F03 and F04 are redrawn in `figures/`.

## Temperature bands (weather rows of the 2025 table T14)

Bands of the leading temperature of the generated description: cold ≤ 15 °C < mild ≤ 22 °C < warm ≤ 28 °C
< hot. The program of T14 is not preserved; these cut-offs reproduce its four band sizes (yes, all 12 rows). Its occasion labels could not be recovered.

| Band | Observations | Full − Original | Full − No-Weather | Full − No-Style | 2025: Full − Original |
|---|---:|---:|---:|---:|---:|
| cold | 1380 | +0.0094 | -0.0029 | +0.0181 | +0.0087 |
| mild | 5145 | +0.0051 | +0.0006 | +0.0039 | +0.0082 |
| warm | 10540 | +0.0072 | -0.0002 | +0.0133 | +0.0103 |
| hot | 95 | +0.0211 | +0.0211 | -0.0316 | +0.0105 |

## Largest rank losses and gains (2025 tables T16 and T17)

`qualitative_failure_cases.csv` (T16) and `qualitative_user_cases.csv` (T17): for each comparison the 12
queries with the largest rank loss under Full, and the 12 with the largest rank gain among those Full ranks
in the top 10, as the archived rows show (the program is not preserved; the occasion label is left empty).

| Comparison | Losses (T16): largest, 12th | 2025 | Gains (T17): largest, 12th | 2025 |
|---|---|---|---|---|
| Original -> Proposed | 1970, 1377 | 1619, 1398 | 1271, 325 | 1332, 436 |
| No-weather -> Proposed | 1054, 666 | 879, 629 | 110, 28 | 96, 33 |
| No-occasion -> Proposed | 1330, 912 | 1308, 917 | 253, 29 | 87, 30 |
| No-style -> Proposed | 1750, 1377 | 1713, 1397 | 1328, 424 | 1185, 393 |

## Category effects of each factor (thesis Figures 4-8 and 4-11 (a) to 4-13 (a))

ΔHit@10 of Full minus the other condition; 2025 values (table T15) in parentheses.

| Category | Observations | Full − Original | Full − No-Weather | Full − No-Occasion | Full − No-Style |
|---|---:|---:|---:|---:|---:|
| dress (all-body) | 2735 | +0.016 (+0.023) | +0.003 (+0.005) | -0.006 (+0.005) | +0.033 (+0.038) |
| heels (shoes) | 2375 | +0.011 (+0.013) | +0.000 (+0.007) | -0.005 (+0.003) | +0.014 (+0.016) |
| sunglasses (accessories) | 1235 | +0.006 (-0.004) | +0.002 (-0.001) | +0.000 (-0.002) | -0.002 (+0.000) |
| purse (bags) | 2485 | +0.006 (+0.013) | -0.002 (+0.000) | -0.012 (+0.000) | +0.002 (+0.003) |
| pump (shoes) | 2345 | +0.005 (+0.010) | +0.000 (+0.001) | -0.001 (-0.001) | +0.011 (+0.013) |
| clutch (bags) | 1985 | +0.004 (+0.003) | -0.001 (+0.002) | +0.002 (-0.001) | +0.006 (+0.001) |
| earrings (jewellery) | 2355 | +0.003 (+0.005) | +0.000 (+0.002) | +0.000 (-0.000) | +0.003 (-0.001) |
| necklace (jewellery) | 1645 | -0.001 (+0.001) | -0.005 (-0.003) | -0.005 (-0.001) | +0.008 (+0.007) |

## Term effects of each factor (thesis Figures 4-11 (b, c) to 4-13 (b, c))

Terms are the weather, occasion and style fragments of each outfit; terms with fewer than 20
observations are left out, as in 2025. The 2025 columns are the eight bars of the archived figure
(three decimals as displayed) and this run's value for the same term. All terms:
`factor_term_effects.csv`.

**Weather terms, Full − Original (thesis Figure 4-11 (b))**: 209 of 480 distinct terms have at least 20 observations; terms of the 2025 figure again in this run's top 8: 2 of 8.

| Rank | This run: term | ΔHit@10 | Observations | 2025 figure: term | ΔHit@10 (2025) | This run |
|---:|---|---:|---:|---|---:|---:|
| 1 | summer evening | +0.160 | 25 | 16.9° C | +0.156 | +0.156 (n=45) |
| 2 | 12.7° C | +0.160 | 25 | 12.7° C | +0.120 | +0.160 (n=25) |
| 3 | 16.9° C | +0.156 | 45 | 14.7° C | +0.120 | +0.040 (n=25) |
| 4 | Warm | +0.109 | 55 | 22.1° C | +0.100 | +0.050 (n=20) |
| 5 | 26.9° C | +0.100 | 20 | 26.4° C | +0.091 | +0.064 (n=110) |
| 6 | breezy evening | +0.100 | 20 | Cool Evening | +0.089 | +0.044 (n=45) |
| 7 | 26.2° C | +0.084 | 95 | Warm Night | +0.080 | +0.000 (n=25) |
| 8 | 14.1° C | +0.080 | 25 | Cool | +0.074 | +0.053 (n=95) |

**Weather terms, Full − No-Weather (thesis Figure 4-11 (c))**: 209 of 480 distinct terms have at least 20 observations; terms of the 2025 figure again in this run's top 8: 1 of 8.

| Rank | This run: term | ΔHit@10 | Observations | 2025 figure: term | ΔHit@10 (2025) | This run |
|---:|---|---:|---:|---|---:|---:|
| 1 | 22.1° C | +0.100 | 20 | shimmering evening | +0.120 | -0.040 (n=25) |
| 2 | summer evening | +0.080 | 25 | 26.0° C | +0.100 | +0.000 (n=30) |
| 3 | 17.1° C | +0.080 | 25 | 18.5° C | +0.082 | +0.012 (n=85) |
| 4 | 16.9° C | +0.067 | 45 | Spring Day | +0.080 | +0.000 (n=50) |
| 5 | 16.4° C | +0.057 | 70 | 22.1° C | +0.050 | +0.100 (n=20) |
| 6 | 14.5° C | +0.050 | 40 | Breezy | +0.047 | +0.047 (n=170) |
| 7 | sun-drenched afternoon | +0.050 | 60 | Warm Evening | +0.047 | +0.035 (n=85) |
| 8 | breezy evening | +0.050 | 20 | 16.3° C | +0.044 | +0.022 (n=45) |

**Occasion terms, Full − Original (thesis Figure 4-12 (b))**: 129 of 835 distinct terms have at least 20 observations; terms of the 2025 figure again in this run's top 8: 5 of 8.

| Rank | This run: term | ΔHit@10 | Observations | 2025 figure: term | ΔHit@10 (2025) | This run |
|---:|---|---:|---:|---|---:|---:|
| 1 | Campus | +0.200 | 20 | Race Day | +0.167 | +0.033 (n=30) |
| 2 | Party | +0.200 | 20 | Party | +0.150 | +0.200 (n=20) |
| 3 | Art Museum | +0.150 | 20 | afternoon out | +0.133 | +0.100 (n=30) |
| 4 | Escape | +0.127 | 55 | Concert | +0.133 | +0.100 (n=30) |
| 5 | afternoon out | +0.100 | 30 | Escape | +0.127 | +0.127 (n=55) |
| 6 | Explorer | +0.100 | 20 | getaway | +0.120 | +0.060 (n=50) |
| 7 | Concert | +0.100 | 30 | Art Museum | +0.100 | +0.150 (n=20) |
| 8 | Night | +0.100 | 20 | Garden Wedding | +0.100 | +0.067 (n=30) |

**Occasion terms, Full − No-Occasion (thesis Figure 4-12 (c))**: 129 of 835 distinct terms have at least 20 observations; terms of the 2025 figure again in this run's top 8: 3 of 8.

| Rank | This run: term | ΔHit@10 | Observations | 2025 figure: term | ΔHit@10 (2025) | This run |
|---:|---|---:|---:|---|---:|---:|
| 1 | Race Day | +0.100 | 30 | evenings | +0.150 | +0.050 (n=20) |
| 2 | Adventures | +0.067 | 30 | Race Day | +0.100 | +0.100 (n=30) |
| 3 | Escape | +0.055 | 55 | Urban Explorer | +0.080 | +0.000 (n=25) |
| 4 | Explorer | +0.050 | 20 | Picnic | +0.080 | +0.040 (n=25) |
| 5 | Art Museum | +0.050 | 20 | back-to-school | +0.080 | +0.000 (n=25) |
| 6 | Campus | +0.050 | 20 | Escape | +0.055 | +0.055 (n=55) |
| 7 | Night | +0.050 | 20 | Look | +0.050 | +0.000 (n=20) |
| 8 | evenings | +0.050 | 20 | Salsa Night | +0.050 | +0.000 (n=20) |

**Style terms, Full − Original (thesis Figure 4-13 (b))**: 232 of 3360 distinct terms have at least 20 observations; terms of the 2025 figure again in this run's top 8: 5 of 8.

| Rank | This run: term | ΔHit@10 | Observations | 2025 figure: term | ΔHit@10 (2025) | This run |
|---:|---|---:|---:|---|---:|---:|
| 1 | Effortless red romance | +0.300 | 20 | Effortless red romance | +0.300 | +0.300 (n=20) |
| 2 | Pop of Red | +0.200 | 25 | breezy sandals | +0.250 | +0.200 (n=20) |
| 3 | breezy sandals | +0.200 | 20 | Effortless Summer Wedding Style | +0.250 | +0.150 (n=20) |
| 4 | delicate sparkle | +0.200 | 20 | Pop of Red | +0.160 | +0.200 (n=25) |
| 5 | Delicate Jewelry | +0.167 | 30 | Fringe Details | +0.160 | +0.120 (n=25) |
| 6 | Gladiator Sandals | +0.167 | 30 | Red-Hot | +0.150 | +0.125 (n=40) |
| 7 | Effortless Art Museum Chic | +0.150 | 20 | delicate sparkle | +0.150 | +0.200 (n=20) |
| 8 | Effortless Summer Wedding Style | +0.150 | 20 | Romance | +0.150 | +0.050 (n=20) |

**Style terms, Full − No-Style (thesis Figure 4-13 (c))**: 232 of 3360 distinct terms have at least 20 observations; terms of the 2025 figure again in this run's top 8: 4 of 8.

| Rank | This run: term | ΔHit@10 | Observations | 2025 figure: term | ΔHit@10 (2025) | This run |
|---:|---|---:|---:|---|---:|---:|
| 1 | Gladiator Sandals | +0.233 | 30 | Effortless Summer Wedding Style | +0.300 | +0.150 (n=20) |
| 2 | breezy sandals | +0.200 | 20 | breezy sandals | +0.250 | +0.200 (n=20) |
| 3 | Delicate Jewelry | +0.167 | 30 | Fringe Details | +0.240 | +0.120 (n=25) |
| 4 | Cozy Layered Look | +0.150 | 20 | Cozy Layered Look | +0.200 | +0.150 (n=20) |
| 5 | Effortless Art Museum Chic | +0.150 | 20 | Gladiator Sandals | +0.200 | +0.233 (n=30) |
| 6 | Effortless Summer Wedding Style | +0.150 | 20 | Floral Flair | +0.160 | +0.040 (n=25) |
| 7 | Velvet Evening Glam | +0.150 | 20 | Red-Hot | +0.150 | +0.100 (n=40) |
| 8 | glam | +0.150 | 20 | Delicate Lace | +0.150 | +0.100 (n=40) |

## Case ranks (five-seed mean)

| Case | Original | Full | No-Weather | No-Occasion | No-Style |
|---|---:|---:|---:|---:|---:|
| purse (Figure A1, set 224499261) | 20.6 | 2.4 | 2.4 | 5.4 | 2.2 |
| dress (Figure A2, set 94771580) | 77.0 | 4.8 | 3.2 | 21.4 | 125.4 |
| sunglasses (Figure A3, set 200099867) | 27.0 | 593.4 | 509.4 | 465.4 | 100.6 |

## Main experiment with Wilcoxon tests (2025 table A14)

Full vs Original over the five seeds of the main experiment (`main_statistical_rigor.csv`; notebook P12
statistical rigor, cell 4). BH is applied over these five tests, as in 2025.

| Metric | Original | Full | Difference (95% CI) | Paired t p | Wilcoxon p | dz | 2025: difference, t p, Wilcoxon p |
|---|---:|---:|---:|---:|---:|---:|---:|
| CP auc | 0.926803 | 0.945636 | 0.018834 [0.013135, 0.024533] | 0.000783 | 0.062500 | 4.103342 | 0.016205, 0.000000, 0.062500 |
| CP fitb_acc | 0.635669 | 0.648148 | 0.012479 [0.005936, 0.019023] | 0.006107 | 0.062500 | 2.368057 | 0.010340, 0.004492, 0.062500 |
| CIR recall_at_10 | 0.078187 | 0.092708 | 0.014520 [0.010752, 0.018289] | 0.000433 | 0.062500 | 4.783930 | 0.014112, 0.000639, 0.062500 |
| CIR recall_at_30 | 0.168360 | 0.188487 | 0.020127 [0.016111, 0.024142] | 0.000155 | 0.062500 | 6.223201 | 0.019697, 0.000246, 0.062500 |
| CIR recall_at_50 | 0.230330 | 0.253249 | 0.022919 [0.019294, 0.026544] | 0.000062 | 0.062500 | 7.851108 | 0.021458, 0.000384, 0.062500 |

With five seeds the exact two-sided Wilcoxon signed-rank test cannot go below 0.0625 (all five
differences of the same sign), so it never reaches 0.05; the 2025 table shows 0.0625 for every metric.

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
- Observations per subset equal the 2025 table T12: yes; all 13 rows of T12, including the unions: yes.
- Observations per category in all four comparisons equal the 2025 table T15: yes.
- BH over the five reported metrics equals the run's tables T03/T04: yes.
- `stage1_cp_seed_detail.csv` and `stage1_cir_seed_detail.csv` (the 2025 tables T08 and T09) agree with the run's T03/T04 at their printed precision: yes (5 of 5 metrics).
- Mean differences and p-values of the four comparisons equal the run's table T02 (20 rows): yes.
- Median temperature threshold: 22.90 °C.
- The means, differences, CIs, paired t p-values and dz of `main_statistical_rigor.csv` equal the run's statistics table main_paired_bh_8metrics.csv: yes.
