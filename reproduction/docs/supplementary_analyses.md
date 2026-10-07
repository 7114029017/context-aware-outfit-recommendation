# Supplementary analyses

These analyses answer questions raised when the TORS manuscript and the thesis
were checked against this repository. They are not part of the 35 training
units: a full run (`reproduce_all.sh --fresh`) runs them after its
`RUN_STATUS.txt` has become `PASSED`, and a failure here does not change that
status. They read files that are already in the repository, in the run folder,
or downloaded by `scripts/bootstrap_data.sh`; they use no GPU and never modify
any run's results, and only the checklist coverage check (section 6) loads a
model, a text embedding model on the CPU. The manuscript's formal results remain
those in `results/summary/full_20261002T161428Z/`.

A full run writes them to its own folder:

| Folder | Content | Official run |
|---|---|---|
| `supplementary/run_analyses/` | Section 1, computed from this run's outputs | `results/supplementary/full_20261002T161428Z/` |
| `supplementary/input_data_audit/` | Section 2, independent of the run | `results/supplementary/input_data_audit/` |
| `supplementary/paper_value_checks/` | Section 3, independent of the run | `results/supplementary/paper_value_checks/` |
| `supplementary/judge_audit_checks/` | Section 4, independent of the run | `results/supplementary/judge_audit_checks/` |
| `supplementary/met_reference_check/` | Section 5, independent of the run; needs `bootstrap_data.sh --with-compendium` | `results/supplementary/met_reference_check/` |
| `supplementary/checklist_coverage/` | Section 6, independent of the run; needs `bootstrap_data.sh --with-nomic` | `results/supplementary/checklist_coverage/` |
| `logs/supplementary.log` | Console output | — |

Without its download, section 5 or 6 prints `SKIPPED` and writes nothing. The
official run was made before this step was added to the pipeline; its files
were computed afterwards with the same scripts. When the comparison with the
official run (README 0.8) reports `IDENTICAL`, a run's files equal the official
run's except for the run name in `run_analyses/summary.md` and in the figure
subtitles; when it reports `CONSISTENT`, the values in `run_analyses/` differ
slightly. The other folders do not depend on the run and should be identical.

To run the six scripts by hand (under a minute on a CPU), for example after the
step failed during a run:

    bash reproduction/scripts/supplementary/run_all.sh [--run-root PATH | --official] [--out-dir DIR]

| Option | Run analyzed | Output |
|---|---|---|
| `--run-root PATH` | The completed run folder at PATH, whose `RUN_STATUS.txt` must be `PASSED` | `PATH/supplementary/` |
| `--official` | The official run included in this repository | `results/supplementary/`; this regenerates the committed files |
| none | The newest `full_*` folder under `reproduction/runs/`, or the official run if there is none | As above |
| `--out-dir DIR` | — | `DIR/` instead |

They use the Polyvore metadata recorded by `bootstrap_data.sh`
(`reproduction/.local/polyvore_root.txt`); pass `--polyvore-root PATH` to use
another copy.

Figures. The 2025 programs drew their figures with matplotlib, which the
reproduction environment does not include. The scripts redraw the thesis
figures as SVG files (`figures/`, helper `_svg.py`) from the same numbers: the
same bars, bins, curves and selection (for example the top 8 terms), not the
same styling. Where a 2025 figure is archived as SVG, its values are compared
with the recomputed ones.

## 1. Analyses of a run's outputs

Script: `scripts/supplementary/official_run_analyses.py`.
Output: `results/supplementary/full_20261002T161428Z/` for the official run;
`supplementary/run_analyses/` in a run folder.

The official run stored, for every fair-subset unit, the rank and hits of each
of the 3,432 queries (`results/raw/full_20261002T161428Z/ablation/*/detail_cir_fresh_subset.csv`;
in a run folder, `ablation/runs/*/detail_cir_fresh_subset.csv`) and the per-seed
metrics. From these files the script computes:

| Output | Content | Manuscript / thesis |
|---|---|---|
| `category_hit10_delta.csv` | Hit@10 of Full minus Original by fine-grained target category, per seed, with the 2025 value | Manuscript Figure 4; thesis Figure 4-8 |
| `context_subset_hit10.csv` | The same by weather, occasion, style-richness and target-type subset; the rules follow the 2025 notebook P02 | Manuscript Section 5.5; thesis Figure 4-9 |
| `subset_robustness_summary.csv` | The 2025 table T12 recomputed: Hit@10 and median rank of Original and Full, and Hit@10 of each simplified description against Full, for every subset, the union of each dimension and all queries (P02 `METRIC_PAIRS`) | Thesis Figures 4-9 and 4-10 |
| `factor_category_effects.csv` | Hit@10 and median rank by target category for Full against Original, No-Weather, No-Occasion and No-Style, with the 2025 values of table T15 (P03 cell 4, `category_summary`) | Thesis Figures 4-8 and 4-11 (a) to 4-13 (a) |
| `factor_term_effects.csv` | The same by weather, occasion and style term, for terms with at least 20 observations, with the 2025 values of the eight terms in each archived figure (P03 cell 4, `term_summary`) | Thesis Figures 4-11 (b, c) to 4-13 (b, c) |
| `case_ranks.csv` | Five-seed mean ranks of the purse, dress and sunglasses cases under the five conditions | Manuscript Section 5.5, Figures A1–A3 |
| `fair_subset_factor_effects.csv` | Full minus Original, No-Weather, No-Occasion and No-Style for all eight metrics, with paired t-test p-values | Manuscript Section 5.3 |
| `fair_subset_bh_family_sensitivity.csv` | Retrospective sensitivity analysis: BH over the five reported metrics (as in the manuscript) and over all eight metrics | Manuscript Table 7 |
| `figures/` | Thesis Figures 4-8 to 4-13 (12 SVG files) | |

The manuscript's Figure 4, subset and case values come from 2025 outputs
(`known_limitations.md`). The official run gives the same subset conclusion
(every subset improves in Hit@10 and in median rank), different category values
(for example, sunglasses improves instead of declining) and different case
ranks with the same directions. Category and case values rest on few queries
and vary across seeds; the term values even more so: a term needs only 20
observations (four queries over five seeds). Of the 48 terms in the six 2025
top-8 lists, 20 are again in the official run's top 8 (weather 3 of 16,
occasion 8 of 16, style 9 of 16), and 9 of the 13 terms named in the thesis
text (`run_analyses/summary.md` lists both).

Cross-checks written to `summary.md`: the overall Hit@10 from the query rows
equals the per-seed Recall@10; the observation counts equal the 2025 tables T12
(all 13 rows) and T15 (all 32 rows); the BH values over the five reported
metrics equal the run's tables T03 and T04; the mean differences and p-values
of the four comparisons equal the run's table T02.

## 2. Audit of the input data

Script: `scripts/supplementary/input_data_audit.py`.
Output: `results/supplementary/input_data_audit/`; `supplementary/input_data_audit/` in a run folder.

The generated descriptions, the CLO / MET / temperature records, the W/O/S
annotations and the counterfactual pairs were built in 2025, and the official
run uses them unchanged. The script reports:

- the temperature at the start of each generated description against the
  computed reference (2,117 of 35,140 differ);
- whether the official run's context-aware text features encode the stored
  descriptions as written;
- equation (2) recomputed for all 35,140 outfits, and the outfits outside
  McIntyre's stated range;
- item IDs shared between the disjoint split files;
- counterfactual replacements that also removed words of a non-target factor;
- standalone "formal" fragments by factor, and activities outside the MET
  candidate list;
- the distribution of the CLO estimates (`clo_distribution_summary.csv`,
  `figures/figure_D_4_clo_distribution.svg`; notebook `Adding_CLO.ipynb`
  cell 17): all twelve statistics of thesis Table D-3 are reproduced (35,140
  values, mean 1.0066, SD 0.5586, IQR bounds −0.2437 and 2.0862, 1,726
  outliers), and the three panels of thesis Figure D-4 are redrawn;
- the category threshold (`category_threshold_check.csv`): the CIR candidate
  pool of every fine-grained category under the rule of the archived evaluator
  (`evaluate_cir.py`), which evaluates 19 of the 152 test categories, the
  9,311 queries of the official run; and the male-labelled categories, none of
  which reaches the 3,000-item pool (thesis Section 5.2).

The findings are listed in `known_limitations.md`.

## 3. Manuscript values from the input data and 2025 outputs

Script: `scripts/supplementary/paper_value_checks.py`.
Output: `results/supplementary/paper_value_checks/`; `supplementary/paper_value_checks/` in a run folder.

Recomputes the values that do not come from the 35 training units: Table 1
(dataset and evaluation scope), Table 2 (text fields), the proxy-value and
target-clue rows of Table 5, Table 8 (Two-Tower, from the preserved per-seed
outputs, including paired 95% confidence intervals) and Table 9 (counterfactual
analysis, from the preserved seed-1 retrieval outputs). Tables 8 and 9 and the
text-length row are regenerated from the run's own models by the extension
analyses (`extensions.md`).

The target-clue audit (notebook P12 in `02_目標單品線索檢查`) needs the
fine-grained category of each target. `categories.csv` lists 34 category IDs
more than once, 16 of them with different labels. The 2025 programs keep the
first occurrence; with that rule the six counts equal the archived table A03
and the manuscript (4.50%, 4.88%, 11.06%, 0.87%, 12.40%), and all 80 archived
examples (A04) carry the same labels. The pipeline's secondary step keeps the
last occurrence, which explains its different counts
(`table5_target_clues.csv` lists both).

## 4. Judge scores and human audit

Script: `scripts/supplementary/judge_audit_checks.py`.
Output: `results/supplementary/judge_audit_checks/`; `supplementary/judge_audit_checks/` in a run folder.

The judge scores and the 750 human answers are preserved 2025 data; the
pipeline recomputes their agreement statistics (thesis Tables 4-7 and 4-8, T19,
T21, T30). This script recomputes the remaining thesis analyses of the same
data, with the code of the 2025 notebooks P05 (cell 3) and P06 (cell 10):

| Output | Thesis | Check |
|---|---|---|
| `judge_score_diagnostics.csv`, `judge_score_histogram.csv`, `figures/figure_4_1_judge_score_histogram.svg` | Figure 4-1, the means ± SDs of Section 4.2.4 | Means and SDs equal T19 to all digits; the 100 bars equal the archived SVG of the thesis figure (F17), which bins both judges on [0, 1] (the notebook's own PNG binned each judge on its own range; the program that redrew it is not preserved) |
| `bottom_p_sensitivity_band.csv`, `figures/figure_4_3_bottom_p_overlap_band.svg`, `figures/bottom_p_lift_band.svg` | Figure 4-3 | p = 1% to 30%, bootstrap B = 500, seed 123. Many descriptions share a score, so the lowest p% depends on how `numpy.argsort` orders ties: the curves differ from the archived figure (F19) by at most 0.012 (Jaccard) and 0.016 (F1), the bands by at most 0.0034, and the intersections of Table 4-7 by 1, 15 and 44 descriptions, as in the pipeline |
| `item_disagreement.csv`, `figures/figure_4_4_item_disagreement.svg` | Figure 4-4, table T31 | All 25 rows equal the archived T31 in all compared columns and in order |
| `judge_reference_scan.csv`, `judge_input_roles.csv` | Section 3.5.3 (judges used only for assessment), tables A05 and A06 | The 2025 search terms in the 2025 file list (11 files found, none references a judge output; two t-test notebooks are not in the repository) and in the 29 files executed for the reproduction's 35 units (none); only the post-hoc analysis scripts read judge outputs |

## 5. MET reference list

Script: `scripts/supplementary/met_reference_check.py`.
Output: `results/supplementary/met_reference_check/`; `supplementary/met_reference_check/` in a run folder.
Needs: `bootstrap_data.sh --with-compendium` (the official 2024 Adult
Compendium PDF, SHA-256 pinned; the same file as in the 2025 handoff) and
`pdftotext` (poppler-utils).

The generation prompts used a 457-entry reduction of the Compendium
(`MET_reference/adult_activity_compendium_sorted_2024.json`); the program that
reduced it is not preserved, but thesis Section 3.3.2 states the rule (sort by
MET value; among equal MET values keep one activity per Major Heading). The
script rebuilds the Compendium table from the word positions in the PDF (1,111
activities) and applies the rule. In 14 rows the description is long enough to
wrap above and below the activity code, and a line-by-line text extraction
drops them; on the remaining 1,097 rows the rule gives the preserved list
exactly: the same 457 activity codes in the same order, with the same headings
and MET values (435 descriptions identical, 21 differing only in spacing, one
with the page title appended at a page break). On all 1,111 rows it would give
459 entries. The "after processing" column of thesis Table 3-1 equals the
rebuilt list; its "activities" column does not match the Compendium in 18 of 22
headings, and the thesis text gives 1,114 activities (the figure of the
Compendium's publication) while the table sums to 1,164.

## 6. Checklist coverage

Script: `scripts/supplementary/checklist_coverage.py`.
Output: `results/supplementary/checklist_coverage/`; `supplementary/checklist_coverage/` in a run folder.
Needs: `bootstrap_data.sh --with-nomic` (about 1.9 GB).

Thesis Figure 4-2 compares the two judges' weighted checklists through the text
embedding model nomic-ai/nomic-embed-text-v2-moe (notebook P05 cell 1). The
notebook used sentence-transformers, which the reproduction environment does
not include; the script runs the same steps with transformers on the CPU, with
the model weights at a pinned revision and the model code at the revision
recorded in the notebook's log, loaded the way sentence-transformers loads them.
The notebook kept its printed outputs, and the recomputation equals them: the
coverage table (51 thresholds × 10 columns) at the six printed decimals, both
AvgMaxSim values (0.869612 and 0.879224), the category JSD (0.005029) and L1
distance (0.151515), and the uncovered items at the focus threshold (13 of 15
and 9 of 10); the focus threshold itself differs in the sixth decimal.
