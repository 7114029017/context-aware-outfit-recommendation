# Checklist coverage (thesis Figure 4-2)

Computed by `reproduction/scripts/supplementary/checklist_coverage.py` (notebook P05 cell 1) with
nomic-ai/nomic-embed-text-v2-moe at revision 1066b6599d099fbb93dfcb64f9c37a7c9e503e85 (model code nomic-ai/nomic-bert-2048 at
7710840340a098cfb869c4f65e87cf2b1b70caca), on the CPU. A = Gemma checklist C* (15 items), B = Qwen checklist C* (10 items).

| Value | Recomputed | 2025 (stored notebook output) |
|---|---:|---:|
| AvgMaxSim A→B | 0.869612 | 0.869612 |
| AvgMaxSim B→A | 0.879224 | 0.879224 |
| Category JSD | 0.005029 | 0.005029 |
| Category L1 | 0.151515 | 0.151515 |
| tau focus (90% quantile of the best-match similarities) | 0.894595 | 0.894596 |
| Uncovered at tau focus: A / B | 13 / 9 | 13 / 9 |

Coverage table, 51 values of tau x 10 columns, at the six decimals printed in 2025: identical.

At tau = 0.80: Cov A→B 1.0000, Cov B→A 1.0000, F1 1.0000; the weighted F1 first drops below 1 at tau = 0.82 (the thesis: all three stay high up to 0.80).

Files: `coverage_vs_tau.csv` (the 2025 metrics table), `best_matches.csv`, `category_distribution.csv`,
`figures/figure_4_2_checklist_coverage.svg`.
