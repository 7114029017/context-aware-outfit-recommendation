# Thesis reproduction run summary

- Main reproduction: **numerically_close_or_exact**
- All 8 main Context−Original directions positive: **True**
- All 16 main condition means within retraining tolerance: **True**
- Fair-subset units verified: **25/25**
- Fair-subset provenance: **reproducibly reconstructed; historical memberwise identity not recovered**
- Secondary-analysis overall: **partial**
- Table 4-3 preserved proxy distribution paper-display match: **True**
- Chapter 4 table inventory: **20/20**, with **112** explicitly scoped numeric rows; NOT 20 fully verified thesis tables.

## Outputs

- Main: `main/summary/main_reproduction_summary.csv`
- Ablation: `ablation/summary/`
- Secondary: `secondary/results/remaining_reproduction_matrix.md`
- Table 4-3: `secondary/results/environment_proxy/table_4_3_manifest.json`
- Chapter 4 table-by-table overview: `chapter4/chapter4_report.md`
- Chapter 4 numeric source comparisons: `chapter4/chapter4_numeric_comparison.csv`

## Secondary modules

| Module | Status |
|---|---|
| length | **exact** |
| target_clue | **partial** |
| judge_preserved_outputs | **exact** |
| human_audit | **exact** |
| case_analysis | **partial** |
| two_tower | **exact** |
| judge_robustness_sampling_provenance | **partial** |
| judge_exact_generation | **partial** |
| case_visuals | **partial** |

## Interpretation

This run uses the archived handoff research programs through a thin wrapper layer. The wrapper prepares paths, seeds, the standardized decoder implementation, output directories, aggregation, and paper comparison; it does not replace the research model implementation.

Historical source/runtime provenance limitations remain separate from numerical reproduction.
