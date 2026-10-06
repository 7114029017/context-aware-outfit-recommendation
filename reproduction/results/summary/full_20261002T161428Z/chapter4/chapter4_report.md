# Chapter 4 reproduction comparison

- Source: **new single integrated full-run outputs**
- Numeric comparison rows: **112**, across the explicitly enumerated sources below.
- Table overview covers **4-1 through 4-20**, including unverified tables.
- The report does not assert all printed thesis table cells are exactly reproduced.

| Thesis table | Scope | Checked rows | Comparison result |
|---|---|---:|---|
| 4-1 | environment | 0 | provenance_only_or_not_independently_verified |
| 4-2 | method | 0 | provenance_only_or_not_independently_verified |
| 4-3 | preserved_numeric | 24 | all_checked_display_fields_match |
| 4-4 | preserved_numeric | 0 | archived_length_reanalysis_exact |
| 4-5 | preserved_numeric | 0 | archived_length_reanalysis_exact |
| 4-6 | partial | 6 | checked_counts_differ |
| 4-7 | partial | 3 | checked_counts_differ |
| 4-8 | preserved_numeric | 30 | all_checked_display_fields_match |
| 4-9 | preserved_numeric | 0 | archived_human_audit_reanalysis_exact |
| 4-10 | preserved_numeric | 0 | archived_human_audit_reanalysis_exact |
| 4-11 | main_numeric | 6 | main_condition_means_within_retrain_tolerance |
| 4-12 | main_numeric | 18 | main_condition_means_within_retrain_tolerance |
| 4-13 | candidate_numeric | 4 | candidate_ablation_vs_archived_T01_only_not_paper_table_exact |
| 4-14 | candidate_numeric | 6 | candidate_ablation_vs_archived_T01_only_not_paper_table_exact |
| 4-15 | candidate_numeric | 6 | candidate_ablation_vs_archived_T01_only_not_paper_table_exact |
| 4-16 | candidate_numeric | 9 | candidate_ablation_vs_archived_T01_only_not_paper_table_exact |
| 4-17 | archived_only | 0 | provenance_only_or_not_independently_verified |
| 4-18 | archived_only | 0 | provenance_only_or_not_independently_verified |
| 4-19 | archived_only | 0 | provenance_only_or_not_independently_verified |
| 4-20 | archived_only | 0 | provenance_only_or_not_independently_verified |

## Interpretive boundaries

- A full pipeline pass does not prove all 20 published tables match cell by cell.
- Table 4-13..4-16 numeric rows refer to the archived T01 aggregate, not each separately printed thesis table.
- Table 4-11..4-12 paper SD/CI/p/effect-size/significance cells are NOT individually compared.
- Tables without independently verified cell-level sources remain explicitly provenance-only.
- The original historical memberwise membership of the fair subset has not been recovered.
- Judge robustness uses preserved compare outputs; no Qwen/Gemma inference is rerun.

See `chapter4_numeric_comparison.csv` for source-specific references, recomputed values, differences, and provenance notes. See `chapter4_report.json` for input SHA-256 evidence.
