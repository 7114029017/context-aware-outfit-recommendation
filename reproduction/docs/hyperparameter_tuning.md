# Hyperparameter provenance

Every setting is classified into the four categories of the 2026-09-30 TORS
follow-up guideline (section 七): fixed by architecture or literature,
selected on validation, historical records missing, and supplementary
analysis. Machine-readable record: `reproduction/results/tuning.csv`.

## Summary

- **Main model (Hybrid Attention CP/CIR): all settings are the base model's
  published configuration, not tuned values.** The handoff's
  `config/base_config.py` is byte-identical to the upstream authors'
  repository, every hyperparameter line of `train_cp.py` / `train_cir.py`
  matches upstream, and the main values are stated in the upstream paper.
  This agrees with the thesis, which keeps the recommender's architecture and
  training procedure unchanged (abstract; 3.6.1).
- **The preserved historical runs used exactly this one configuration.** All
  ten historical CP training logs (2025-12) print an identical configuration
  block; no other configuration of this model appears in the handoff.
- **Validation selects only the checkpoint epoch** of each run (validation
  FITB accuracy). No hyperparameter value was selected on validation in the
  preserved evidence.
- **Second model (Two-Tower):** the values are fixed in its notebook; how they
  were chosen is not recorded.
- **No supplementary (retrospective) sensitivity analysis has been run.**

TORS A7 status: **PARTIAL**. The main model's settings are fully sourced. What
remains unrecoverable is whether any alternative values were tried before the
upstream configuration was adopted, and the rationale for the second model's
values.

## Sources

| Source | Identity |
|---|---|
| Upstream paper | X. Wang and Y. Zhong, "Text-Conditioned Outfit Recommendation With Hybrid Attention Layer," *IEEE Access*, vol. 12, 2024, doi:10.1109/ACCESS.2023.3346933 (not redistributed here; see `docs/public_release_cleanup.md`). Section IV-B "Implementation Details" |
| Upstream code | https://github.com/WangXin93/text-conditioned-outfit-recommendation, commit `91ed4fad653d44397fcf0bb4b8359b4a7d2c607c` (2025-05-13) |
| Handoff code | `02_模型訓練和驗證_model_training_validation/main_hybrid_attention_code/` |
| Historical logs | `02_…/main_hybrid_attention_checkpoints/{cp,cir}_{old,new}_seed{1..5}/log.txt` |

Comparison of the handoff code with the upstream commit:

| File | Result |
|---|---|
| `config/base_config.py` | identical |
| `dataset.py`, `utils.py` | identical |
| `train_cp.py`, `train_cir.py` | differ only in experiment plumbing (seed / subset / condition arguments, per-run seeding with deterministic cuDNN, fair-subset filter, Chinese log messages); optimizer, clipping, scheduler, hard-negative, GradScaler, margin and checkpoint-selection lines are identical |
| `outfit_transformer.py` | upstream builds the CP decoder with `torch.nn.TransformerDecoderLayer`; the handoff disabled that block and added the unrecoverable `DecoderLayerWithCrossAttn` (see `standardized_decoder.md`) |

## 1. Fixed by architecture or literature（依架構或文獻固定）

Values come from the upstream paper (P) or, where the paper is silent, from the
upstream implementation (C). They are fixed because the thesis adopts this
published model unchanged and studies only the text condition.

| Setting | Value | Source |
|---|---|---|
| Item image encoder | FashionCLIP features (512-d), linearly mapped to 64 | P §III-A, §IV-B; C `img_emb_size=64` |
| Item text encoder | SentenceBERT `distiluse-base-multilingual-cased-v2` (512-d), mapped to 64 | P §IV-B ("descriptions … use different languages"); C `txt_emb_size=64` |
| Outfit text embedding | mapped to 128 | P §IV-B ("same as the previous study [4]") |
| Maximum items per outfit | 16 (PO-D) | P §IV-B; C `max_item_len` |
| Decoder | 3 layers, 16 heads, `torch.nn.TransformerDecoderLayer` | P §IV-B ("found that the model capacity is enough"); C `num_layers=3`, `nhead=16`, upstream `outfit_transformer.py` |
| Dropout | 0.1 | C `dropout=0.1` |
| Epochs | 100 | P §IV-B; C `epochs=100` |
| Batch size | 50 (float16) | P §IV-B; C `batch_size = 50 if 'float16' in dtype else 30` |
| Optimizer | Adam, learning rate 5e-5 | P §IV-B; C `optim.Adam(parameters, lr=learning_rate)` |
| Weight decay | 0 (Adam default) | C, not stated in P |
| Learning-rate decay | ×0.5 every 10 epochs (`StepLR`) | P §IV-B; C train scripts |
| Gradient clipping | max norm 0.5 | C train scripts |
| Precision | float16 with GradScaler; TF32 allowed | C `dtype='float16'` |
| Negatives | CP: one negative outfit per positive, each item replaced by a random item of the same high-level category. CIR: 10 negative items per positive, from the same high-level category. `num_negative=10` is used only by CIR; P §IV-B states "10 per positive" | C `dataset.py` (`__getitem__` of the CP and CIR datasets) |
| Hard negatives | From epoch 40 (`hard_negatives_start_epoch=40`). CP adds one hard-negative outfit whose items are replaced within the same fine-grained category. CIR samples 10 fine-grained hard negatives, but `dataset.py` line 581 loads `negatives` instead of `hard_negatives`, so the added loss term uses the hardest of the 10 random negatives and the fine-grained hard negatives are never used. The archived behavior is kept (`configs/or.yaml`, `historical_behavior_note`) | P §IV-B; C `dataset.py`, `train_cp.py`, `train_cir.py` |
| CIR ranking loss margin | 0.3 | P §IV-B; C `margin=0.3` |
| CIR initialization | same-seed CP checkpoint | P §IV-B, item 1 "Pretraining by compatibility prediction task"; handoff `train_cir.py` |
| CIR scoring | cosine similarity | P §III |
| CIR candidate pool | fine-grained categories with ≥ 3,000 candidates | C `evaluate_cir.py`; thesis 4.1.1 ("沿用原始推薦模型的類別篩選設定") |

The ten historical CP logs (`cp_old_seed1–5`, `cp_new_seed1–5`) all print this
configuration (batch 50, dropout 0.1, float16, 100 epochs, hard negatives
from epoch 40, embeddings 64, learning rate 5e-5, margin 0.3, 16 items,
16 heads, 3 layers). The CIR logs show each run loading the same-seed CP
checkpoint and training at learning rate 5e-5 with batch 50.

## 2. Selected on validation（透過 validation 選擇）

| What | Candidates | Validation metric | Selected |
|---|---|---|---|
| Checkpoint epoch of every CP and CIR run | epochs 1–100, evaluated after each epoch | validation FITB accuracy (maximize) | the best epoch's checkpoint (`cp_best_ckpt.pt`, `cir_best_ckpt.pt`) |
| Second-model checkpoint epoch | epochs 1–100 per stage | validation FITB accuracy | best epoch (CP 10–28, OR 1–7 per `A34_second_model_training_report.txt`) |

Per-epoch validation results are in each run's training log. No
hyperparameter value was chosen this way in the preserved evidence.

## 3. Historical records missing（歷史紀錄遺失）

| Item | Status |
|---|---|
| Whether the thesis author tried alternative values before adopting the upstream configuration | No candidate ranges, trial logs or selection notes were found |
| Upstream authors' own search behind 3 layers / 16 heads and the other values | Only the paper's statements are available |
| Second model: why batch 256, learning rate 1e-3, AdamW, weight decay 1e-4, dropout 0.15 | Values are fixed in `P15_fashionclip_text_conditioned_retrieval_baseline.ipynb` (cell 2); no tuning or rationale recorded |

Search performed on 2026-10-03:

- every text file smaller than 20 MB in `01_`–`04_` (191 files: notebooks,
  scripts, logs, CSV/JSON/TXT/MD; 89 MB), for tuning terms (hyperparameter,
  超參數, 調參, grid search, sweep, learning rate, dropout, batch sizes,
  optuna, wandb);
- the ten CP and ten CIR historical training logs;
- the statistics notebooks P01 and P12 and table A14, where "sweep results"
  denotes the main experiment's result rows, not a hyperparameter search;
- the second-model folder (`A33`, `A34`, notebook P15).

Nothing in these records shows a hyperparameter search. Nothing is
reconstructed from assumptions.

## 4. Supplementary analysis（新增補充分析）

None has been performed. Any future sensitivity analysis will be labeled a
retrospective sensitivity analysis and kept separate from the original
settings.

## 2026 reproduction protocol (not tuning)

| Setting | Value | Note |
|---|---|---|
| Training seeds | 1, 2, 3, 4, 5 | the thesis's five seeds (historical runs `*_seed1`–`*_seed5`) |
| Determinism | cuDNN deterministic, benchmark off | per-run seeding in the handoff train scripts |
| CP decoder | standardized decoder (`torch.nn.TransformerDecoderLayer`) | the upstream implementation; see `standardized_decoder.md` |
| Human-audit sampling seed | 42 | analysis seed, not a training seed |
