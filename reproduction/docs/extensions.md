# Extension analyses

The 35 training units reproduce Tables 6 and 7. Several other analyses in the
TORS manuscript still rest on the senior student's preserved model outputs:
the text-length row of Table 5, the counterfactual analysis (Tables 4 and 9),
the Two-Tower check (Table 8), the color-shift numbers, the appendix case
figures and the thesis's color-shift case (Figure 4-14). The extension analyses
regenerate them from a run's own models, together with three 2025 analyses that
the manuscript does not use: the reliability analysis of notebook P04 (table
T18, figures F12-F16), the text-swap evaluation of the 2025 sweep, and the
step-by-step outfit generation of the try-on demo P02. They run after the 35
units, read the run's checkpoints and outputs, and never change the run's
results or its `RUN_STATUS.txt`; the code used by the 35 units is not changed.

Scripts: `reproduction/scripts/extensions/`. Each one is ported from the 2025
notebook that produced the corresponding values. Apart from paths and output
handling, the computation is copied unchanged, and each port is first checked
against the notebook's archived outputs (section "Validation").

## Running them

A full run (`reproduce_all.sh --fresh`) runs them after it has PASSED and writes
`<run folder>/extensions/`, with the status of each step in
`extensions/EXTENSIONS_STATUS.txt` and the console output in
`logs/extensions.log`. A failed step prints a warning; the run stays PASSED. To
run them by hand, for example after a failure:

    bash reproduction/scripts/extensions/run_all.sh --run-root RUN [--steps LIST] [--skip-gpu] [--validate]

| Step | Manuscript | Needs | Time |
|---|---|---|---|
| `text_length` | Table 5, "Text length" (thesis Tables 4-4, 4-5) | GPU | about 20 min |
| `counterfactual` | Tables 4 and 9 | FashionCLIP (`bootstrap_data.sh --with-fashionclip`) | about 1 min on a CPU |
| `two_tower` | Table 8 | GPU | 1.5-3 hours |
| `color` | Section 5.5, color-shift analysis (thesis Table 4-19) | Polyvore images (`bootstrap_data.sh --with-images`) | seconds |
| `figures` | Figures A1-A3; thesis Figure 4-14; 2025 figures F34a and F34b (notebook P16) | Polyvore images | seconds |
| `reliability` | 2025 table T18 and figures F12-F16 (notebook P04; not in the manuscript) | GPU | about 5 min |
| `text_swap` | 2025 sweep of `CP_evaluate.py` and `CIR_evaluate.py` (not in the manuscript) | GPU | about 45 min |
| `outfit_generation` | Outfit generation of the 2025 try-on demo (notebook P02; not in the manuscript) | Polyvore images, FashionCLIP | about 2 min on a CPU |

A step whose GPU or download is missing is reported as SKIPPED and the other
steps still run. `--validate` adds the checks of the next section (about 10 min
on a CPU, and with a GPU about 5 min for the reliability table T18 and 75 min
for the 20 preserved main checkpoints with both texts). The Polyvore images and
the figures that show Polyvore product photos stay on the local machine:
`case_figures.py` and `outfit_generation.py` refuse to write them into a tracked
folder, and only the figures' contents (item IDs, ranks and, for Figure 4-14,
the dominant colors) are recorded in `case_figures_manifest.csv`,
`color_case_figure_manifest.csv`, `counterfactual_figure_manifest.csv` and
`outfit_generation/generation.csv`.

## Validation of the ported code

Run with `--validate` on 2026-10-07; outputs in
`results/extensions/full_20261002T161428Z/validation/` and
`counterfactual/summary.md`.

| Check | Result |
|---|---|
| Text length: the archived 2025 row-level table A07 through `length_analysis.py` | A08, A09 and A10 equal the archived tables; r = 0.027647 and 0.043384, 3,014 pairs, ΔHit@10 = +0.0106 as in the manuscript |
| Text length: per-query files rebuilt from A07 through the `--detail-dir` path | the rebuilt A07 equals the archived one (46,555 rows) |
| Text length: one main unit (Original, seed 1) evaluated on the CPU | 9,311 queries; per-query hits average to the recalls; Recall@1, @10 and @50 equal the official run, Recall@3, @5 and @30 differ by one or two queries (float32 on the CPU, float16 autocast on the GPU in the official run), so this step runs on the GPU |
| Counterfactual: FashionCLIP at the revision in P16's log, on the 24 stored context-aware descriptions | cosine with the stored features 1.000000, largest absolute difference 6e-06 |
| Counterfactual: the preserved 2025 seed-1 model against the archived A45 | top-5 lists equal in 24/24 pairs for both conditions; ranks equal in 17/24 and 22/24, never more than 1 apart; Table 9 recomputed as +17.2 / 0.500 / 0.351 (manuscript +17.0 / 0.500 / 0.351) |
| Two-Tower: the 20 preserved checkpoints evaluated with `two_tower.py` against the archived A30 | FITB, Recall@1-50 and median rank equal; AUC within 3.3e-08 and mean rank within 0.0005 |
| Two-Tower: one-epoch training on a subset on the CPU (`--smoke`) | training, checkpoint selection and evaluation run end to end |
| Color: comparable rows on the official run | 745 rows over 149 cases, as printed by P03 in 2025 |
| Figures F34a and F34b: P16's selection rule (cell 9) on the archived review sheet A46 | selects the 2025 pairs CF04 and CF09 |
| Reliability labels: P04's query labels and P12's range checks (A16) on the 9,311 queries, without a model | all 16 label fields of the archived T18 equal for 9,311 of 9,311 queries; A16 equal; with the archived scores, the label, error-type, risk and log columns regenerate the 41 MB T18 file byte for byte; the metric code gives P04's printed performance table (18 values at six decimals) |
| Reliability: the preserved 2025 seed-1 models evaluated on the CPU (`validation/reliability_2025_seed1_cpu_check/`) | against the archived T18, per model: Hit@10 equal in 9,310 of 9,311 queries, top-1 item in 9,294, ranks in 7,342 and 7,387 (never more than 8 apart), confidences within 0.0054; Hit@1 and median rank as printed by P04, Hit@10 one query lower, ECE and Brier within 0.0002 (float32 on the CPU; P04 ran on CUDA with float16 autocast, so the step itself runs on the GPU) |
| Outfit generation: the preserved 2025 `cir_new_seed1` on the CPU | 6 of 6 picks equal to P02's printed output; scores within 0.0001 |
| Text swap: the official run's seed-1 models with the swapped texts, on the fair-subset outfits, on the CPU (`--subset-ids`) | both evaluators run end to end for both combinations (8 min); a plumbing test, not compared with stored results |

## Results for the official run

The official run's checkpoints are kept with the run folder and are not
public, so its extension outputs were produced once on the maintainer's
machine and committed to `results/extensions/full_20261002T161428Z/` (command:
`run_all.sh --run-root <official run folder> --out-dir
reproduction/results/extensions/full_20261002T161428Z --figures-dir <local
folder> --skip-gpu --validate`). The GPU steps (`text_length`, `two_tower`,
`reliability`, `text_swap`) and the GPU validations have not been run yet;
their outputs will be added to the same folder.

**Counterfactual (Tables 4 and 9).** Same 24 pairs, the official run's Context
models of seeds 1-5, CPU. Mean ± SD over the seeds (`table9_seeds_mean_sd.csv`;
per seed: `table9_seed<k>.csv`):

| Scope | Mean rank change | Top-1 changed | Top-5 Jaccard | Manuscript (2025 seed-1 model) |
|---|---:|---:|---:|---|
| Overall | +75.9 ± 41.3 | 0.658 ± 0.090 | 0.367 ± 0.015 | +17.0, 0.500, 0.351 |
| Weather | -77.1 ± 47.5 | 0.475 ± 0.240 | 0.522 ± 0.042 | -146.4, 0.125, 0.508 |
| Occasion | +20.0 ± 18.4 | 0.525 ± 0.105 | 0.490 ± 0.053 | -11.9, 0.375, 0.416 |
| Style | +285.0 ± 128.7 | 0.975 ± 0.056 | 0.088 ± 0.013 | +209.1, 1.000, 0.130 |

Style replacements again give by far the largest response. The overall rank
change is larger than in the manuscript and varies strongly between seeds, so
a single-seed table is fragile. The pair problems listed in
`known_limitations.md` (CF05, CF12, CF16, CF17, CF18, CF21) are unchanged; the
analysis uses the archived pairs as they are. `table9_live_baseline_*.csv`
compares with the context-aware description encoded by the same encoder, a
symmetric control that P16 did not have.

**Color-shift analysis.** 34 of 80 row-level problems (0.425) and 5 of 13
case-level problems (0.385) are solved under Full; the manuscript reports
39 / 89 (0.438) and 5 / 15 (0.333). The images come from the Hugging Face
mirror; the 2025 per-query files are not preserved, so the 2025 counts cannot
be recomputed.

**Figures A1-A3.** Same three cases, the seed whose Full rank is closest to the
five-seed mean: purse (seed 2) Original rank 21, Full 1; dress (seed 2)
Original 12, No-Style 55, Full 5; sunglasses (seed 1) Original 47, Full 500.
The five-seed mean ranks equal `results/supplementary/full_20261002T161428Z/case_ranks.csv`.
The captions must be updated if the manuscript uses these figures.

**Figure 4-14 (thesis, color-shift case).** Same case (set 174710752, a gray
dress), drawn as the notebook P03 (cell 7, `show_case`) drew it: the query
outfit, then the top-5 items under Original and under Full with their dominant
image colors. The seed shown is the notebook's representative row (largest
color problem under Original, then best Original rank): seed 1. All five
Original items are black and four of the five Full items gray, while the
target moves from rank 31 to 36; the thesis describes the same pattern (all
five black, gray items in front, rank 27 to 36). The case is a color problem
under Original in 4 of 5 seeds and under Full in 1 of 5
(`color_analysis/color_shift_cases.csv`).

**Figures F34a and F34b (2025, notebook P16).** The review sheet A46 rebuilt
from the seed-1 counterfactual results (`case_figures/counterfactual_review_seed1.csv`)
and P16's rule (largest and smallest absolute rank change) select CF18 (style,
high to low: rank 380 to 1,151, no top-5 item in common) and CF02 (weather,
warm to cold: rank 9 to 9, top-5 overlap 0.667); with the archived A46 the
rule selects the 2025 pairs CF04 and CF09. CF18 is one of the pairs whose
replacement also removed words of a non-target factor (`known_limitations.md`).
The figures are drawn locally; their contents are in
`case_figures/counterfactual_figure_manifest.csv`.

**Outfit generation (2025 try-on demo P02).** Case 224188768 ("26.2° C,
Effortless Summer Wedding Style with a Pop of Citrus"), the official run's
Context model of seed 1, CPU (`outfit_generation/`). From an empty outfit the
model picks the original dress (209144226) first, out of 5,170 candidates, and
then the shoe 82253359 and the bag 112382686, the items that the 2025 model
picked when the dress was kept; with the dress and the shoe kept it again picks
the bag 112382686. The item grid of P02 is drawn locally; the try-on images are
not reproduced (next section).

## What cannot be regenerated

| Item | Why |
|---|---|
| Generated descriptions, CLO / MET estimates, W/O/S split | The LLM sampling settings, and the split program and prompt, were not preserved; regenerating them would create a different dataset and require retraining 30 of the 35 units |
| LLM judge scores (prompt robustness, judge overlap) | Only the model names are recorded, without revisions, and the P0/P1/P2 prompts were not preserved; the statistics are recomputed from the preserved scores |
| The 750 judgments of the manual audit | Human judgments; the statistics are recomputed |
| Identity of the feature extraction | The extraction code was not preserved; the FashionCLIP text encoder reproduces the stored context-aware features (above), the image features would need all item images |
| Try-on images of the 2025 demo P02 | The person photo (`model_image.png`) was not preserved, and FastFit is not part of the handoff (its FastFit-MR-1024 weights, about 2 GB, are under the FastFit Non-Commercial License; the Human-Toolkit models add about 2.7 GB); the outfit generation before the try-on is reproduced |

The target-clue row of Table 5 is recomputed exactly from the input data by
the supplementary checks (`supplementary_analyses.md`).
