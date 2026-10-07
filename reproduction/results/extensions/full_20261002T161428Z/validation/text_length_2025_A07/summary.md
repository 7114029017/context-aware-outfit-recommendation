# Text-length analysis of run `2025 A07`

Source: row-level table `03_實驗與結果_experiments_results/01_文字長度影響分析/A07_length_performance_rowlevel_cir.csv`. Definitions copied from notebook P12 (01_文字長度影響分析).

| Manuscript value (Table 5, Text length) | This analysis |
|---|---:|
| N (query-target pairs) | 9311 |
| Pearson r, length difference vs ΔHit@10 (pair means over seeds) | 0.027647 (p = 0.007632) |
| Pearson r, length difference vs rank improvement | 0.043384 (p = 0.000028) |
| Pairs within five tokens (abs difference <= 5) | 3014 |
| ΔHit@10 within five tokens (Full − Original) | 0.0106 |

The manuscript reports N = 9,311, r = 0.027647 and 0.043384, n = 3,014 and ΔHit@10 = +0.0106 from the
2025 models.

Equal to the archived 2025 tables: A08 yes, A09 yes, A10 yes.
