# Dataset tables (2025 T00, T01, A01, A02, A12, A13)

Computed by `reproduction/scripts/supplementary/dataset_tables_check.py` from the Polyvore metadata, the
generated descriptions and the fair subset, with the code of the 2025 programs; the CIR scope uses the
rule of the archived evaluator. Every value is compared with the archived table (`comparison.csv`; the
source paths and notes are not compared).

| Table | Rows | Values compared | Equal |
|---|---:|---:|---:|
| T00_dataset_overview_po_d_main.csv | 16 | 48 | 48 |
| T00_dataset_characterization_po_d_main.csv | 11 | 33 | 33 |
| T01_text_field_comparison_po_d_main.csv | 3 | 27 | 27 |
| A01_title_length_audit.csv | 3 | 45 | 45 |
| A02_generation_coverage_audit.csv | 11 | 22 | 22 |
| A12_po_d_split_item_outfit_stats.csv | 4 | 32 | 32 |
| A13_po_d_semantic_category_distribution.csv | 44 | 264 | 264 |

All 471 values equal the archived tables.
