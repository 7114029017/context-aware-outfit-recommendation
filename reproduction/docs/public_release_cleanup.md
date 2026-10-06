# Public release cleanup

This repository is the public reproducibility artifact for the TORS revision.
Polyvore images are not redistributed due to third-party rights. Third-party
publications and manuscripts under review are not redistributed either.

To publish the repository, the senior-student research folders (01–04) were
changed only as listed below; everything else in those folders is the frozen
handoff baseline that the official run `full_20261002T161428Z` used (commit
`7a5cc9c` of the original development repository).

## Removed files

| File | Reason |
|---|---|
| `01_資料建構_data_construction/clo_met_temperature/CLO_reference/2021 ASHRAE 69套套裝.pdf` | Third-party publication (ASHRAE) |
| `01_資料建構_data_construction/clo_met_temperature/CLO_reference/ASHRARE 2010 clo.pdf` | Third-party publication (ASHRAE) |
| `01_資料建構_data_construction/clo_met_temperature/CLO_reference/ssrn-5357611 實測.pdf` | Third-party publication (SSRN 5357611) |
| `01_資料建構_data_construction/clo_met_temperature/MET_reference/1_2024-adult-compendium_1_2024.pdf` | Third-party publication (2024 Adult Compendium of Physical Activities) |
| `01_資料建構_data_construction/clo_met_temperature/temperature_results/DonMcIntyreComfortRqtsECRCOct78.pdf` | Third-party publication (McIntyre, 1978) |
| `02_模型訓練和驗證_model_training_validation/main_hybrid_attention_code/Text-Conditioned_Outfit_Recommendation_With_Hybrid_Attention_Layer.pdf` | Third-party publication (Wang & Zhong, IEEE Access, 2024) |
| `03_實驗與結果_experiments_results/04_情境子集與三因子分析/圖表_figures_tables/figures/F21_qualitative_retrieval_case_original_vs_context_aware.svg` | Case figure built from Polyvore product photos |
| `03_實驗與結果_experiments_results/06_反事實情境敏感度/圖表_figures_tables/figures/F34a_counterfactual_context_larger_response_example.svg` | Case figure built from Polyvore product photos |
| `03_實驗與結果_experiments_results/06_反事實情境敏感度/圖表_figures_tables/figures/F34b_counterfactual_context_limited_response_example.svg` | Case figure built from Polyvore product photos |
| `04_文件資料_documents/journal/ACM_TORS_English_Condensed.docx` | Manuscript under review |
| `04_文件資料_documents/journal/ACM_TORS_English_v4.pdf` | Manuscript under review |
| `04_文件資料_documents/thesis/情境感知驅動的智慧穿搭推薦系統 論文終稿.pdf` | Thesis; its figures contain Polyvore product photos |

Folder `04_文件資料_documents/` contained only these three files and is
therefore no longer present. The publications are cited in the manuscript;
obtain them from their publishers.

## Notebooks with embedded images removed

The code cells and the text outputs are kept; only the embedded image outputs
were removed.

| Notebook | Images removed | Content of the removed images |
|---|---:|---|
| `01_資料建構_data_construction/clo_met_temperature/CLO_reference/Adding_CLO.ipynb` | 6 | Mostly Polyvore product photos (one statistics chart) |
| `03_實驗與結果_experiments_results/04_情境子集與三因子分析/source_programs/P02_subset_robustness_analysis.ipynb` | 9 | Product photos and virtual try-on renders |
| `03_實驗與結果_experiments_results/04_情境子集與三因子分析/source_programs/P03_qualitative_case_analysis.ipynb` | 65 | Case figures built from product photos |

Without its images, `P03_qualitative_case_analysis.ipynb` is 0.8 MB instead of
111.6 MB, so it is stored as a regular Git file instead of a Git LFS object.
Git LFS now holds only the eight precomputed feature files.

## Images that remain

The remaining 31 image files (statistics charts and one flow diagram), the
chart PDF `F01_main_factor_contribution_proportion.pdf`, and the 29 statistics
charts embedded in notebooks P01, P05 and P06 were reviewed one by one; none of
them shows a Polyvore image.

## Effects on the reproduction

- The reproduction pipeline reads none of the removed files or images. On the
  cleaned content, the preflight checks that do not need a GPU (feature files,
  data splits, original folders), the launch gates and the comparison with the
  official run (IDENTICAL, including the 70 checkpoints) all pass.
- `reproduction/environment/archived_sources_manifest.json` was regenerated
  for the published content (278 files in folders 01–03 instead of 287), so the
  launch gate checks the files as published here.
- `03_實驗與結果_experiments_results/source_map.csv` and `source_map.txt` are
  historical records and still list the three removed SVG figures.
- `docs/manuscript_reconciliation/reconciliation.md` and
  `scripts/build_manuscript_reconciliation.py` mention the removed thesis and
  manuscript PDFs as reference paths only; the tools do not open them.
- `scripts/preflight_remaining_thesis_reproduction.py` records notebook
  checksums. A new run records the checksums of the cleaned notebooks; this
  does not affect the comparison with the official run.
