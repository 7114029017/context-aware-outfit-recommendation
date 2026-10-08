# Manuscript values recomputed from the input data and from 2025 outputs

Computed by `reproduction/scripts/supplementary/paper_value_checks.py`. Tables 6 and 7 come from the
official run and are produced by the reproduction pipeline; the values here do not come from the 35
training units. Table 8 and Table 9 are recomputed from the preserved 2025 outputs that the manuscript
reports (the 2025 Two-Tower models and the 2025 seed-1 counterfactual model); the same analyses with the
official run's models, the Two-Tower models retrained, are in `results/extensions/full_20261002T161428Z/`.

## Table 1

- train outfits: 16995
- validation outfits: 3000
- test outfits: 15145
- all outfits: 35140
- generated descriptions: 35140
- CP test pairs: 30290
- CP positive pairs: 15145
- CP negative pairs: 15145
- FITB questions: 15145
- CIR evaluable queries: 9311
- CIR excluded queries (no 3,000-item pool): 5834
- CIR target fine-grained categories: 19
- original text records: 68306

## Table 2

See `table2_text_fields.csv` (missing values, mean / median / max words, mean tokens).

## Table 5, target clues

`categories.csv` lists 34 category IDs more than once, 16 of them with different
labels. The 2025 programs keep the first occurrence; the frozen pipeline's secondary step
(`secondary/target_clue/`) keeps the last one. With the 2025 rule the six counts equal the archived table
A03 and the manuscript: yes. Fine-category labels of the 80 archived
examples (A04): 80 match the 2025 rule, 40 the last-occurrence rule.

| Metric | 2025 (A03) | Recomputed, 2025 rule | Last occurrence (pipeline) |
|---|---:|---:|---:|
| fine-category match in the generated description | 419 (4.50%) | 419 (4.50%) | 549 (5.90%) |
| fine- or main-category match in the generated description | 454 (4.88%) | 454 (4.88%) | 560 (6.01%) |
| target-item token overlap, generated description | 1030 (11.06%) | 1030 (11.06%) | 999 (10.73%) |
| target-item bigram overlap, generated description | 81 (0.87%) | 81 (0.87%) | 81 (0.87%) |
| target-item token overlap, original text | 1155 (12.40%) | 1155 (12.40%) | 1132 (12.16%) |

## Table 5, proxy values

See `table5_proxy_values.csv`. Counts: CLO > 4: 20; MET = 1.0: 25786; MET > 10: 8; temperature < 0 °C: 257; temperature < -20 °C: 51.

## Table 8 (2025 Two-Tower outputs)

| Metric | Original | Context-aware | Difference | Paired 95% CI |
|---|---|---|---:|---|
| cp_test_auc | 0.8796 ± 0.0029 | 0.9174 ± 0.0033 | +0.0378 | [0.0337, 0.0420] |
| cp_test_fitb_acc | 0.5460 ± 0.0051 | 0.5382 ± 0.0050 | -0.0077 | [-0.0180, 0.0025] |
| recall@10 | 0.0378 ± 0.0008 | 0.0479 ± 0.0017 | +0.0102 | [0.0087, 0.0117] |
| recall@30 | 0.0950 ± 0.0017 | 0.1103 ± 0.0030 | +0.0153 | [0.0127, 0.0179] |
| recall@50 | 0.1419 ± 0.0026 | 0.1589 ± 0.0046 | +0.0170 | [0.0139, 0.0200] |
| mean_rank | 601.3589 ± 7.0359 | 572.4678 ± 10.5809 | -28.8911 | [-44.8960, -12.8863] |
| median_rank | 361.2000 ± 6.2209 | 329.8000 ± 9.7826 | -31.4000 | [-44.0808, -18.7192] |

## Table 9 (2025 counterfactual outputs)

| Scope | n | Mean rank before → after | Mean rank change | Top-1 changed | Top-5 Jaccard |
|---|---:|---|---:|---:|---:|
| Overall | 24 | 586.0 → 603.0 | +17.0 | 0.500 | 0.351 |
| Weather | 8 | 850.6 → 704.2 | -146.4 | 0.125 | 0.508 |
| Occasion | 8 | 282.2 → 270.4 | -11.9 | 0.375 | 0.416 |
| Style | 8 | 625.2 → 834.4 | +209.1 | 1.000 | 0.130 |
