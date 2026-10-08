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

A full run (`reproduce_all.sh --fresh`) runs them after it has PASSED, through
`scripts/run_analyses.sh` and with `--validate`, and writes
`<run folder>/extensions/`, with the status of each step in
`extensions/EXTENSIONS_STATUS.txt`, the step logs in `extensions/logs/` and the
console output in `logs/extensions.log`. A failed step prints a warning; the run
stays PASSED. `run_analyses.sh` then lists every item with its state and evidence
in `ITEMS_STATUS.md` (README 0.8), where each item computed here also shows the
check of its port, and shows the final summary (`FINAL_SUMMARY.txt`). `reproduce_all.sh --analyses-only RUN` repeats all of this for
a completed run without retraining. To run the extension steps alone, for
example after a failure:

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
| Reliability: the same models on the GPU with float16 autocast, as P04 ran (`validation/reliability_2025_seed1/`) | Hit@10 equal in 9,311 and 9,310 of 9,311 queries, top-1 item in 9,298 and 9,301, ranks in 7,575 and 7,636 (never more than 5 apart); for the Original model Hit@1, Hit@10, median rank and error rate as printed by P04, ECE and Brier within 0.00001; the remaining differences come from the different GPU |
| Outfit generation: the preserved 2025 `cir_new_seed1` on the CPU | 6 of 6 picks equal to P02's printed output; scores within 0.0001 |
| Text swap: the 20 preserved 2025 main checkpoints with both texts on the GPU (`validation/text_swap_2025_checkpoints/`) | the matching-text means are within 0.0001 of the archived A14 (4 of the 10 identical: CP AUC of both models, Recall@10 of the Original models, Recall@50 of the Context models), so the preserved checkpoints are those behind the 2025 main results; the lost 2025 sweep is recovered: the context-aware text raises the Original model's AUC by 0.0064 (5/5 seeds), and the Context model given the original titles drops to AUC 0.9059, the pattern of the official run |
| Text swap: the official run's seed-1 models with the swapped texts, on the fair-subset outfits, on the CPU (`--subset-ids`) | both evaluators run end to end for both combinations (8 min); a plumbing test, not compared with stored results |

## Results for the official run

The official run's checkpoints are kept with the run folder and are not
public, so its extension outputs were produced once on the maintainer's
machine and committed to `results/extensions/full_20261002T161428Z/` (command:
`run_all.sh --run-root <official run folder> --out-dir
reproduction/results/extensions/full_20261002T161428Z --figures-dir <local
folder> --skip-gpu --validate`). The GPU steps were run afterwards on the GB10:
`text_length`, `reliability` and `text_swap` on 2026-10-07 and 08 (69 min),
then `two_tower` on 2026-10-08 (63 min), followed by the GPU validations
`validate_reliability` and `validate_text_swap` (80 min).

**Text length (Table 5; thesis Tables 4-4, 4-5).** The 10 main CIR checkpoints
evaluated again on the GPU with per-query output (`main_cir_per_query/`): all
10 reproduce the run's stored recalls exactly, and the per-query hits average
to them. The length analysis (`text_length/`, A07-A10 layout):

| Value | Official run | Manuscript (2025 models) |
|---|---:|---:|
| Query-target pairs | 9,311 | 9,311 |
| Pearson r, length difference vs ΔHit@10 (pair means) | 0.033985 (p = 0.0010) | 0.027647 (p = 0.0076) |
| Pearson r, length difference vs rank improvement | 0.045481 (p < 0.0001) | 0.043384 (p < 0.0001) |
| Pairs within five tokens | 3,014 | 3,014 |
| ΔHit@10 within five tokens (Full − Original) | +0.0081 | +0.0106 |

The correlations stay negligible in size (significant only because N is
large), and every matched-length group still improves (A10: +0.0127, +0.0161,
+0.0084, +0.0081 and +0.0081 for differences of at most 0, 1, 2, 3 and 5
tokens), so the manuscript's conclusion holds; its numbers change.

**Reliability (2025 notebook P04: table T18, figures F12-F16).** The run's
seed-1 Original and Context models, GPU with float16 autocast as in P04
(`reliability/`; the 40 MB file in the T18 layout is not committed):

| Model | Hit@1 | Hit@10 | Median rank | High-conf. error | ECE | Brier | 2025 (Hit@10, ECE, Brier) |
|---|---:|---:|---:|---:|---:|---:|---|
| Original | 0.0142 | 0.0795 | 228 | 0.2258 | 0.6444 | 0.5089 | 0.0780, 0.6423, 0.5050 |
| Context | 0.0159 | 0.0898 | 194 | 0.2067 | 0.6294 | 0.4964 | 0.0871, 0.6314, 0.4973 |

As in 2025, the Context model has the higher Hit@10, the lower ECE and Brier
score, fewer high-confidence errors, and a higher Hit@10 in all eight
subgroups (in 2025 the low-style gain was only +0.0005). Both models remain far from
calibrated (ECE above 0.6). P04 has its own evaluation loop: on the same models
and candidate pools its Hit@1 equals the run's Recall@1, its ranks equal the
evaluator's in 83% of the queries (never more than 4 apart), and its Hit@10
differs by one query per model, because P04 converts the candidate embeddings
to float32 before the cosine similarity while the evaluator keeps them in
float16.

**Two-Tower (Table 8).** The 10 Two-Tower models (original text and
context-aware description, seeds 1-5) retrained with the code of notebook P15
on the GPU (`two_tower/`; the model files and the row-level A31 stay local).
Mean ± SD and the paired difference (Context-aware minus Original) with its
95% CI:

| Metric | Original | Context-aware | Difference [95% CI] | Manuscript |
|---|---:|---:|---:|---:|
| CP AUC | 0.8803 ± 0.0031 | 0.9119 ± 0.0048 | +0.0316 [0.0224, 0.0409] | +0.0378 |
| CP FITB | 0.5440 ± 0.0051 | 0.5331 ± 0.0034 | -0.0109 [-0.0196, -0.0022] | -0.0077 |
| Recall@10 | 0.0387 ± 0.0024 | 0.0470 ± 0.0014 | +0.0083 [0.0040, 0.0126] | +0.0102 |
| Recall@30 | 0.0937 ± 0.0037 | 0.1117 ± 0.0032 | +0.0180 [0.0112, 0.0248] | +0.0153 |
| Recall@50 | 0.1379 ± 0.0039 | 0.1605 ± 0.0022 | +0.0226 [0.0169, 0.0283] | +0.0170 |
| Mean rank | 611.0 ± 10.6 | 573.3 ± 3.5 | -37.7 [-51.1, -24.4] | -28.89 |
| Median rank | 372.8 ± 10.6 | 328.8 ± 4.5 | -44.0 [-58.6, -29.4] | -31.4 |

Every difference has the manuscript's direction. The FITB decrease, whose
interval contained zero in the manuscript ([-0.0180, 0.0025]), now excludes
it.

**Text swap (2025 sweep).** Each main model of seeds 1-5 evaluated with the
other outfit text on the GPU (`text_swap/`); the seed-1 combinations with the
model's own text, evaluated again as a check, equal the stored results. Mean ±
SD over the five seeds:

| Model | Text | AUC | FITB | R@10 | R@50 |
|---|---|---:|---:|---:|---:|
| Original | original titles | 0.9268 ± 0.0040 | 0.6357 ± 0.0035 | 0.0782 ± 0.0012 | 0.2303 ± 0.0032 |
| Original | context-aware | 0.9338 ± 0.0041 | 0.6418 ± 0.0037 | 0.0818 ± 0.0024 | 0.2360 ± 0.0019 |
| Context | original titles | 0.9061 ± 0.0020 | 0.6179 ± 0.0013 | 0.0711 ± 0.0019 | 0.2136 ± 0.0023 |
| Context | context-aware | 0.9456 ± 0.0010 | 0.6481 ± 0.0029 | 0.0927 ± 0.0020 | 0.2532 ± 0.0023 |

The context-aware descriptions help even the model trained on the original
titles (AUC +0.0070, Recall@10 +0.0036, positive in all five seeds), and the
Context model depends on them: given the original titles it falls below the
Original model (AUC 0.9061, Recall@10 0.0711). The 2025 sweep, whose files were
lost, was recomputed from the preserved 2025 checkpoints and shows the same
pattern (validation table above).

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
