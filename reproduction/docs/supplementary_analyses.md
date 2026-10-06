# Supplementary analyses

These analyses answer questions raised when the TORS manuscript was checked
against this repository. They are post-hoc analyses, not part of the 35-unit
pipeline (`reproduce_all.sh` does not run them): they read files that are
already in the repository, in a completed run folder, or downloaded by
`scripts/bootstrap_data.sh`; they load no model, use no GPU, and never modify
any run's results. The manuscript's formal results remain those in
`results/summary/full_20261002T161428Z/`.

Run all three scripts (a few seconds on a CPU):

    bash reproduction/scripts/supplementary/run_all.sh [--run-root PATH | --official]

They use the Polyvore metadata recorded by `bootstrap_data.sh`
(`reproduction/.local/polyvore_root.txt`); pass `--polyvore-root PATH` to use
another copy.

The first script analyzes one run:

| Option | Run analyzed | Output |
|---|---|---|
| none | The newest `full_*` folder under `reproduction/runs/`, whose `RUN_STATUS.txt` must be `PASSED`; the official run if there is no such folder | The run folder's `supplementary/`; `results/supplementary/full_20261002T161428Z/` for the official run |
| `--run-root PATH` | The completed run folder at PATH, for example a run written with `--output-base` or `--run-id` | `PATH/supplementary/` |
| `--official` | The official run included in this repository | `results/supplementary/full_20261002T161428Z/` |

After `reproduce_all.sh --fresh`, the default therefore analyzes the new run.
Compare its `supplementary/` folder with `results/supplementary/full_20261002T161428Z/`:
on the same NVIDIA GB10 and software stack, the files are identical except for
the run name and source in `summary.md`; on other GPUs the values differ
slightly. The other two scripts do not depend on a run.

## 1. Analyses of a run's outputs

Script: `scripts/supplementary/official_run_analyses.py`.
Output for the official run: `results/supplementary/full_20261002T161428Z/`.

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
Output: `results/supplementary/input_data_audit/`.

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
Output: `results/supplementary/paper_value_checks/`.

Recomputes the values that do not come from the 35 training units: Table 1
(dataset and evaluation scope), Table 2 (text fields), the proxy-value row of
Table 5, Table 8 (Two-Tower, from the preserved 2025 per-seed outputs, including
paired 95% confidence intervals) and Table 9 (counterfactual analysis, from the
preserved 2025 seed-1 retrieval outputs).
