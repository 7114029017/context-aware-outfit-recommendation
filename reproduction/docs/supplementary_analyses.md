# Supplementary analyses

These analyses answer questions raised when the TORS manuscript was checked
against this repository. They are not part of the 35 training units: a full run
(`reproduce_all.sh --fresh`) runs them after its `RUN_STATUS.txt` has become
`PASSED`, and a failure here does not change that status. They read files that
are already in the repository, in the run folder, or downloaded by
`scripts/bootstrap_data.sh`; they load no model, use no GPU, and never modify
any run's results. The manuscript's formal results remain those in
`results/summary/full_20261002T161428Z/`.

A full run writes them to its own folder:

| Folder | Content | Official run |
|---|---|---|
| `supplementary/run_analyses/` | Section 1, computed from this run's outputs | `results/supplementary/full_20261002T161428Z/` |
| `supplementary/input_data_audit/` | Section 2, independent of the run | `results/supplementary/input_data_audit/` |
| `supplementary/paper_value_checks/` | Section 3, independent of the run | `results/supplementary/paper_value_checks/` |
| `logs/supplementary.log` | Console output | — |

The official run was made before this step was added to the pipeline; its
files were computed afterwards with the same scripts. When the comparison with
the official run (README 0.8) reports `IDENTICAL`, a run's files equal the
official run's except for the run name and source at the top of
`run_analyses/summary.md`; when it reports `CONSISTENT`, the values in
`run_analyses/` differ slightly. The other two folders do not depend on the run
and should be identical.

To run the three scripts by hand (a few seconds on a CPU), for example after the
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

## 1. Analyses of a run's outputs

Script: `scripts/supplementary/official_run_analyses.py`.
Output: `results/supplementary/full_20261002T161428Z/` for the official run;
`supplementary/run_analyses/` in a run folder.

The official run stored, for every fair-subset unit, the rank and hits of each
of the 3,432 queries (`results/raw/full_20261002T161428Z/ablation/*/detail_cir_fresh_subset.csv`;
in a run folder, `ablation/runs/*/detail_cir_fresh_subset.csv`) and the per-seed
metrics. From these files the script computes:

| Output | Content | Manuscript |
|---|---|---|
| `category_hit10_delta.csv` | Hit@10 of Full minus Original by fine-grained target category, per seed, with the 2025 value | Figure 4 and its discussion |
| `context_subset_hit10.csv` | The same by weather, occasion, style-richness and target-type subset; the rules follow the 2025 notebook P02 | Section 5.5 |
| `case_ranks.csv` | Five-seed mean ranks of the purse, dress and sunglasses cases under the five conditions | Section 5.5, Figures A1–A3 |
| `fair_subset_factor_effects.csv` | Full minus Original, No-Weather, No-Occasion and No-Style for all eight metrics, with paired t-test p-values | Section 5.3 |
| `fair_subset_bh_family_sensitivity.csv` | Retrospective sensitivity analysis: BH over the five reported metrics (as in the manuscript) and over all eight metrics | Table 7 |

The manuscript's Figure 4, subset and case values come from 2025 outputs
(`known_limitations.md`). The official run gives the same subset conclusion
(every subset improves), different category values (for example, sunglasses
improves instead of declining) and different case ranks with the same
directions. Category and case values rest on few queries and vary across seeds.

Cross-checks written to `summary.md`: the overall Hit@10 from the query rows
equals the per-seed Recall@10; the per-category and per-subset observation
counts equal the 2025 tables T15 and T12; the BH values over the five reported
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
  candidate list.

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
