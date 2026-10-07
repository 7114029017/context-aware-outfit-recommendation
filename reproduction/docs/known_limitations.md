# Known limitations

Everything this artifact cannot reproduce source-exactly, or whose historical
source was not preserved (2026-09-30 guideline, section 十 item 9). For each
item: what is missing or not exact, what the artifact does instead, and where
the evidence is.

Overall: the 2026 results are fully reproducible from this repository (35
training units, statistics and tables, one command). The provenance of the
historical 2025 thesis pipeline is **partial**.

Differences between the latest results and the thesis (for example Recall@1
no longer significant after BH, and the Weather / Occasion OR signs) are not
limitations of the artifact. They are listed in
`docs/manuscript_reconciliation/reconciliation.md`.

## Model and training

| Item | Missing or not exact | What the artifact does | Evidence |
|---|---|---|---|
| CP decoder | The handoff's `DecoderLayerWithCrossAttn` has no definition (import commented out) | Uses the upstream authors' published decoder, `torch.nn.TransformerDecoderLayer`; historical forward behavior not claimed | `standardized_decoder.md` |
| 2025 runtime | Thesis trained on an RTX PRO 6000 Blackwell; exact 2025 package versions not recorded | 2026 runtime recorded (GB10, Python 3.12.3, PyTorch 2.9.1+cu130); bit-identical results are not expected on other GPU / CUDA / PyTorch builds | `environment/`, `reference_run_provenance.md` |
| Hyperparameter search | No tuning records; unknown whether alternatives were tried before adopting the upstream configuration; second-model rationale not recorded | Every setting sourced or marked missing; no supplementary sensitivity analysis | `hyperparameter_tuning.md`, `results/tuning.csv` |
| CP negatives | The manuscript and the upstream paper describe 10 negatives per positive, but the CP dataset pairs each positive outfit with one negative outfit (`num_negative=10` is used only by CIR) | Keeps the archived behavior | `hyperparameter_tuning.md`; `main_hybrid_attention_code/dataset.py` |
| CIR hard negatives | From epoch 40 the CIR dataset samples fine-grained hard negatives, but `dataset.py` line 581 loads `negatives` instead of `hard_negatives`, so the added loss term uses the hardest of the 10 random negatives | Keeps the archived behavior, because the training code is frozen after the official run; a corrected run would need a separate experiment tag (`configs/or.yaml`) | `hyperparameter_tuning.md`; `main_hybrid_attention_code/dataset.py` |

## Data and subsets

| Item | Missing or not exact | What the artifact does | Evidence |
|---|---|---|---|
| Fair-subset membership | Historical memberwise ID file, its SHA-256 and the generation program were not preserved | Reproducibly reconstructs the subset (21,903 / 9,930 test / 3,432 OR queries, same counts as the thesis) and uses it as the formal subset; member identity with 2025 not claimed | `fair_subset.md` |
| Dataset access | Original 2025 download date not recovered | 2026 bootstrap from the `Stylique/Polyvore` mirror, verified by byte-identical split manifests | `data/README.md` |
| Polyvore images | Not redistributed | `bootstrap_data.sh --with-images` downloads them from the same Hugging Face dataset for local use only; the color analysis and the case figures are regenerated with them. LLM regeneration is not rerun (see Description generation) | `data/README.md`, `extensions.md` |
| Item features | Exact FashionCLIP / SentenceBERT model revisions not established; the extraction code was not preserved | Uses the preserved feature files (Git LFS, SHA-256 verified); features are not re-extracted. The FashionCLIP text encoder at the revision in notebook P16's log reproduces the stored context-aware features (cosine 1.000000 on 24 descriptions) | `configs/cp.yaml`, `feature_manifest.md`, `extensions.md` |
| Description generation | The 35,140 context-aware descriptions and CLO / MET estimates came from Gemma-3-4B-IT; prompts and listed inference settings are preserved, sampling details beyond them are not certified | Uses the preserved generated descriptions; no regeneration | `prompts/semantic_rewrite.txt`, `prompts/clo_estimation.txt`, `prompts/activity_met_mapping.txt` |
| Temperature in the descriptions | 2,117 of the 35,140 descriptions (6.0%) begin with a temperature that differs from the computed reference `TSUB_target_C`: 2,077 in train (2,036 keep only the decimal digit, 41 drop the minus sign) and 40 in test, with no pattern. The train descriptions use an integer format written by a post-processing step that was not preserved | Uses the descriptions as preserved; the text features of the official run encode them. Correcting them would require regenerating the data and rerunning every experiment | `results/supplementary/input_data_audit/` |
| W/O/S annotation | Only the annotation outputs are preserved; the annotation program, prompt and model are not | Uses the preserved annotations | `01_資料建構_data_construction/generated_descriptions/02_三因子拆分_wos_factor_split/` |
| MET candidates | 245 of the 35,140 outfits (0.70%) use 14 activities that are not in the 457-entry candidate list | Uses the preserved values | `results/supplementary/input_data_audit/` |
| MET reference list (thesis 3.3.2, Table 3-1) | The program that reduced the 2024 Adult Compendium to the 457-entry list was not preserved. With the rule stated in the thesis, the official Compendium file gives the preserved list exactly (same 457 codes in the same order) only if the 14 rows whose descriptions wrap in the PDF are left out, as a line-by-line text extraction does; with all 1,111 rows the rule gives 459 entries. The "activities" column of Table 3-1 does not match the Compendium in 18 of 22 headings (it sums to 1,164; the text says 1,114, the figure of the Compendium's publication) | Uses the preserved list; the rebuild documents how it was made | `results/supplementary/met_reference_check/` |
| Thermal mapping range | McIntyre states the simplified relation for M < 150 W/m² and Icl < 1.5 clo; 6,348 of the 35,140 outfits (18.1%) fall outside this range | Uses the temperature reference only as a data-construction proxy | `results/supplementary/input_data_audit/` |
| Split item overlap | The released disjoint split files share item IDs: train and test 84, train and valid 3,781, valid and test 34 | Uses the released split unchanged | `results/supplementary/input_data_audit/` |
| Category threshold (thesis Section 5.2) | The thesis says that only categories with more than 3,000 items were used in training and that the male-labelled categories fall below this threshold. In the archived code the 3,000 threshold only selects the categories that get a CIR candidate pool (19 of 152 test categories, the 9,311 queries); it does not filter the training data. No male-labelled category reaches the pool size (largest: category 21, 2,725 items), but 1,930 male-labelled items are in the training outfits and 1,940 in the test outfits used for compatibility and FITB. During CIR training the logged validation recall covers 7 of 127 categories, because the loop that adds train items stops at the first category missing from the validation split; checkpoints are selected by validation FITB accuracy, which this does not affect | Keeps the archived code; the scope is documented | `results/supplementary/input_data_audit/category_threshold_check.csv` |

## LLM-as-a-Judge and audits

| Item | Missing or not exact | What the artifact does | Evidence |
|---|---|---|---|
| Judge models | Exact Qwen3-VL-32B-Instruct and Gemma-3-27B-it revisions not recovered | Re-analyzes the preserved formal Judge outputs; no Judge rerun | `judge_provenance.md`, `configs/llm_judge.yaml` |
| Judge prompts | Exact P0 / P1 / P2 prompt text not recovered | C / C* checklists preserved; robustness statistics recomputed from preserved outputs | `judge_provenance.md` |
| Robustness sample | Regenerating the 150-record sample yields only 3 / 150 of the archived IDs | Uses the archived 150-record files | `results/summary/reference_20260921T175217Z/secondary/remaining_reproduction_matrix.md` |
| Table 4-7 | Recomputed low-score overlaps 669 / 1,338 / 2,877 vs thesis 668 / 1,323 / 2,833 (cutoff ties; historical inputs) | Reports the recomputed values and the difference | same matrix; `chapter4_table_overview.csv` |
| Table 4-6 | Historical intermediate `reliability_meta_from_subset.csv` not archived. `categories.csv` lists 34 category IDs more than once; the 2025 programs keep the first occurrence, the pipeline's secondary step keeps the last, so its counts differ (549 / 560 / 999 / 81 / 1,132) | The supplementary checks apply the 2025 rule and reproduce all six counts exactly (419 / 454 / 1,030 / 81 / 1,155); the secondary output is frozen with the official run and keeps the last-occurrence counts | `results/supplementary/paper_value_checks/table5_target_clues.csv` |
| Human audit (Tables 4-9, 4-10) | Label-generation provenance not proven | Re-analyzes the 750 archived human judgments; no new judgments | `chapter4_table_overview.csv` |
| Thesis Figures 4-1 to 4-4 | Figure 4-1 in the thesis is a later redraw (archived SVG F17) that bins both judges on [0, 1]; notebook P05 binned each judge on its own range, and the redraw program is not preserved. Figure 4-3 depends on how tied scores are ordered (as Table 4-7). Figure 4-2 needs the Nomic embedding model, which the 2025 environment loaded through sentence-transformers | Recomputed from the preserved data: Figure 4-1 equals the thesis figure bar for bar; Figure 4-3 differs from it by at most 0.016; Figure 4-2 equals the notebook's stored output (pinned model, CPU); Figure 4-4 and table T31 are exact | `results/supplementary/judge_audit_checks/`, `results/supplementary/checklist_coverage/` |
| Judge independence (tables A05, A06) | The 2025 static scan listed two t-test notebooks that are not in the repository | The same scan on the code run for the reproduction's 35 units finds no reference to judge outputs | `results/supplementary/judge_audit_checks/judge_reference_scan.csv` |

## Analyses outside the 35 training units

The manuscript's values for these analyses come from historical checkpoints or
preserved outputs. The supplementary and extension analyses regenerate them
from the official run (`supplementary_analyses.md`, `extensions.md`).

| Analysis | Status | Evidence |
|---|---|---|
| Text length (TORS Table 5; thesis Tables 4-4, 4-5) | The manuscript's values come from the 2025 per-query file A07; the official run's main evaluation saved only the recalls. The extension re-evaluates the 10 main CIR checkpoints with the frozen evaluator and per-query output; not yet run (needs the GPU) | `extensions.md` |
| Two-Tower second model (TORS Table 8) | Trained after the oral defense (about June 2026) and never retrained. Means, SDs and 95% CIs recomputed exactly from the preserved per-seed outputs; the ported training code reproduces the 20 preserved checkpoints' results; retraining not yet run (needs the GPU) | `secondary/two_tower/`; `results/supplementary/paper_value_checks/`; `extensions.md` |
| Counterfactual retrieval (TORS Tables 4 and 9, 24 cases) | The manuscript's values come from the 2025 seed-1 model (notebook run on 2026-08-06). Rerun with the official run's Context models of seeds 1-5. The pairs are used as archived: in CF16, CF17, CF18 and CF21 the replacement also removed words annotated as another factor (in CF16 the word is "weekend", annotated as style), CF12 has no occasion fragment, and CF05 was grouped by the stated temperature (13.1 °C) although its temperature reference is 21.0 °C | `results/extensions/full_20261002T161428Z/counterfactual/`; `results/supplementary/input_data_audit/` |
| Category and context-subset analyses (thesis 4.5, Figures 4-8 to 4-13; TORS 5.5) | The manuscript's values come from historical checkpoints. The same analyses (table T12, the category and term summaries of notebook P03, Figures 4-8 to 4-13) computed from the official run's fair-subset rows keep the subset conclusion (every subset improves in Hit@10 and median rank) and differ by category (for example sunglasses). The term effects are less stable: a term needs only 20 observations (four queries over five seeds); the weather terms change almost completely (3 of the 16 terms in the two 2025 top-8 lists recur in the official run's top 8), the occasion and style terms about half (17 of 32), and 9 of the 13 terms named in the thesis text recur | `03_實驗與結果_experiments_results/04_情境子集與三因子分析/`; `results/supplementary/full_20261002T161428Z/` |
| Case studies and color analysis (Tables 4-17 to 4-20; TORS 5.5, Figures A1-A3; thesis Figure 4-14) | The manuscript's ranks are historical; the official-run ranks are in the supplementary outputs. The color analysis is automatic (color words and pixel colors; `chapter4_table_overview.csv` calls it a human judgment, which is not correct) and is regenerated from the official run with the downloaded images; Figures A1-A3 and the color case of Figure 4-14 can be redrawn locally (Figure 4-14 shows the same pattern: Original top-5 all black, Full mostly gray, target rank 31 to 36 against 27 to 36 in the thesis) | `results/supplementary/full_20261002T161428Z/`; `results/extensions/full_20261002T161428Z/` |
| 2025 tables T14, T16 and T17 (qualitative condition summary and case lists) | The program that wrote them is not preserved. The temperature bands of T14 are recovered: bands of the leading temperature of the description with the cut-offs 15, 22 and 28 °C reproduce all four band sizes. Its occasion labels are not: the closest keyword rule found still differs in 230 observations. T16 and T17 are rebuilt from the official run with the rule their rows show (per comparison, the 12 largest rank losses, and the 12 largest gains with Full in the top 10), without the occasion label | `results/supplementary/full_20261002T161428Z/` |
| Reliability analysis (2025 notebook P04: table T18, figures F12-F16; A16; not used in the manuscript) | Ported. The query labels and the range checks reproduce the archived T18 and A16 exactly, and the metric code reproduces P04's printed table; the per-query model results need the GPU, because P04 ran on CUDA with float16 autocast (a CPU check of the 2025 seed-1 models is in `extensions.md`) | `results/extensions/full_20261002T161428Z/validation/`; `extensions.md` |
| Text swap (2025 sweep of `CP_evaluate.py` and `CIR_evaluate.py`; not used in the manuscript) | The sweep's result files are not preserved, and the archived `CIR_evaluate.py` does not run as preserved (indentation error at line 272). The extension evaluates the main checkpoints with the frozen `evaluate_cp.py` / `evaluate_cir.py` and the other outfit text; not yet run (needs the GPU) | `extensions.md` |
| Try-on demo (2025 notebook P02, cells 1 and 2; not used in the manuscript) | The person photo (`model_image.png`) and FastFit (FastFit-MR-1024) are not part of the handoff, so the try-on images are not reproduced. The step-by-step outfit generation before the try-on is: the preserved 2025 model gives P02's six picks | `results/extensions/full_20261002T161428Z/outfit_generation/` |

## Records kept as they were

- The run records of `reference_20260921T175217Z` and `full_20260924T102513Z`
  keep their original wording ("temporary … candidate" decoder, "candidate"
  subset). They are historical records and are not edited.
- `results/README.md` and the reference manifest are part of the hashed
  reference result set (`results/SHA256SUMS.txt`) and keep their original
  wording for the same reason.

## Publication items still pending

The fixed GitHub release `v1.0.0-tors-reproduction` was published on
2026-10-07; the manuscript's Artifact Availability statement should link to
it. No DOI is used. Third-party terms are listed in `THIRD_PARTY_NOTICES.md` at
the repository root; the public-release changes are listed in
`public_release_cleanup.md`.
