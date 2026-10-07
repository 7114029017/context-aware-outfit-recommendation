# Text-length analysis of run `full_20261002T161428Z`

Source: the per-query files written by main_cir_per_query.py. Definitions copied from notebook P12 (01_文字長度影響分析).

| Manuscript value (Table 5, Text length) | This analysis |
|---|---:|
| N (query-target pairs) | 9311 |
| Pearson r, length difference vs ΔHit@10 (pair means over seeds) | 0.033985 (p = 0.001039) |
| Pearson r, length difference vs rank improvement | 0.045481 (p = 0.000011) |
| Pairs within five tokens (abs difference <= 5) | 3014 |
| ΔHit@10 within five tokens (Full − Original) | 0.0081 |

The manuscript reports N = 9,311, r = 0.027647 and 0.043384, n = 3,014 and ΔHit@10 = +0.0106 from the
2025 models.

Equal to the archived 2025 tables: A08 no, A09 no, A10 no.
