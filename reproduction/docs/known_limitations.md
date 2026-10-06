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
| Polyvore images | Not redistributed | Image-dependent steps (case figures, color judgments, LLM regeneration) cannot be rerun from this artifact | `data/README.md` |
| Item features | Exact FashionCLIP / SentenceBERT model revisions not established | Uses the preserved feature files (Git LFS, SHA-256 verified); features are not re-extracted | `configs/cp.yaml`, `feature_manifest.md` |
| Description generation | The 35,140 context-aware descriptions and CLO / MET estimates came from Gemma-3-4B-IT; prompts and listed inference settings are preserved, sampling details beyond them are not certified | Uses the preserved generated descriptions; no regeneration | `prompts/semantic_rewrite.txt`, `prompts/clo_estimation.txt`, `prompts/activity_met_mapping.txt` |
| Temperature in the descriptions | 2,117 of the 35,140 descriptions (6.0%) begin with a temperature that differs from the computed reference `TSUB_target_C`: 2,077 in train (2,036 keep only the decimal digit, 41 drop the minus sign) and 40 in test, with no pattern. The train descriptions use an integer format written by a post-processing step that was not preserved | Uses the descriptions as preserved; the text features of the official run encode them. Correcting them would require regenerating the data and rerunning every experiment | `results/supplementary/input_data_audit/` |
| W/O/S annotation | Only the annotation outputs are preserved; the annotation program, prompt and model are not | Uses the preserved annotations | `01_資料建構_data_construction/generated_descriptions/02_三因子拆分_wos_factor_split/` |
| MET candidates | 245 of the 35,140 outfits (0.70%) use 14 activities that are not in the 457-entry candidate list | Uses the preserved values | `results/supplementary/input_data_audit/` |
| Thermal mapping range | McIntyre states the simplified relation for M < 150 W/m² and Icl < 1.5 clo; 6,348 of the 35,140 outfits (18.1%) fall outside this range | Uses the temperature reference only as a data-construction proxy | `results/supplementary/input_data_audit/` |
| Split item overlap | The released disjoint split files share item IDs: train and test 84, train and valid 3,781, valid and test 34 | Uses the released split unchanged | `results/supplementary/input_data_audit/` |

## LLM-as-a-Judge and audits

| Item | Missing or not exact | What the artifact does | Evidence |
|---|---|---|---|
| Judge models | Exact Qwen3-VL-32B-Instruct and Gemma-3-27B-it revisions not recovered | Re-analyzes the preserved formal Judge outputs; no Judge rerun | `judge_provenance.md`, `configs/llm_judge.yaml` |
| Judge prompts | Exact P0 / P1 / P2 prompt text not recovered | C / C* checklists preserved; robustness statistics recomputed from preserved outputs | `judge_provenance.md` |
| Robustness sample | Regenerating the 150-record sample yields only 3 / 150 of the archived IDs | Uses the archived 150-record files | `results/summary/reference_20260921T175217Z/secondary/remaining_reproduction_matrix.md` |
| Table 4-7 | Recomputed low-score overlaps 669 / 1,338 / 2,877 vs thesis 668 / 1,323 / 2,833 (cutoff ties; historical inputs) | Reports the recomputed values and the difference | same matrix; `chapter4_table_overview.csv` |
| Table 4-6 | Historical intermediate `reliability_meta_from_subset.csv` not archived | Scope (9,311) and bigram count reproduce; several lexical counts differ and are reported | same matrix |
| Human audit (Tables 4-9, 4-10) | Label-generation provenance not proven | Re-analyzes the 750 archived human judgments; no new judgments | `chapter4_table_overview.csv` |

## Analyses outside the 35 training units

These use historical checkpoints or preserved outputs, not the 2026 models.

| Analysis | Status | Evidence |
|---|---|---|
| Two-Tower second model (TORS Table 8) | Means and SDs recomputed exactly from preserved per-seed outputs, and 95% CIs recomputed from the same outputs; not retrained | `secondary/two_tower/`; `results/supplementary/paper_value_checks/` |
| Counterfactual retrieval (TORS Table 9, 24 cases) | Historical seed-1 checkpoint; not rerun. In pairs CF16, CF17, CF18 and CF21 the replacement also removed words annotated as another factor (in CF16 the word is "weekend", annotated as style), and CF12 has no occasion fragment | `03_實驗與結果_experiments_results/06_反事實情境敏感度/`; `results/supplementary/input_data_audit/` |
| Category and context-subset analyses (thesis 4.5, Figures 4-8 to 4-13; TORS 5.5) | The manuscript's values come from historical checkpoints. The same analyses computed from the official run's fair-subset rows differ by category (for example sunglasses) | `03_實驗與結果_experiments_results/04_情境子集與三因子分析/`; `results/supplementary/full_20261002T161428Z/` |
| Case studies and color analysis (Tables 4-17 to 4-20) | The manuscript's ranks are historical; the official-run ranks for the same set and target are in the supplementary outputs. Image-based judgments are not re-executed | `chapter4_table_overview.csv`; `results/supplementary/full_20261002T161428Z/` |

## Records kept as they were

- The run records of `reference_20260921T175217Z` and `full_20260924T102513Z`
  keep their original wording ("temporary … candidate" decoder, "candidate"
  subset). They are historical records and are not edited.
- `results/README.md` and the reference manifest are part of the hashed
  reference result set (`results/SHA256SUMS.txt`) and keep their original
  wording for the same reason.

## Publication items still pending

The fixed GitHub release `v1.0.0-tors-reproduction` and the manuscript's
Artifact Availability statement, which should link to that release. No DOI is
used. Third-party terms are listed in `THIRD_PARTY_NOTICES.md` at the
repository root; the public-release changes are listed in
`public_release_cleanup.md`.
