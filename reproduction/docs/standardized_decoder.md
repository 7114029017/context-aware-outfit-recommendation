# Standardized decoder implementation

All 2026 experiments use `torch.nn.TransformerDecoderLayer`, stacked in
`torch.nn.TransformerDecoder`, as the CP decoder of the Hybrid Attention model.
This is the formal decoder implementation of the revised manuscript and of
this artifact. It covers main CP/CIR training, evaluation, the 25 fair-subset
ablation units and the smoke test.

## Why the archived decoder is not used

The archived model file
`02_模型訓練和驗證_model_training_validation/main_hybrid_attention_code/outfit_transformer.py`
builds the CP decoder with `DecoderLayerWithCrossAttn`, but:

- its import (`#from attn_utils import DecoderLayerWithCrossAttn`, line 11) is
  commented out, and
- no definition of the class exists anywhere in the handoff.

The archived file therefore cannot run as-is, and the early custom decoder
definition was not preserved, so its exact forward behavior cannot be
recovered.

The same file keeps a standard branch, disabled inside a triple-quoted string
(lines 174–180), with identical constructor arguments:

    decoder_layer = nn.TransformerDecoderLayer(
        d_model=img_emb_size+txt_emb_size, nhead=nhead, batch_first=True, dropout=dropout)

This branch is the base model's published implementation. In the upstream
authors' repository
(https://github.com/WangXin93/text-conditioned-outfit-recommendation, commit
`91ed4fa`), `outfit_transformer.py` builds the CP decoder with exactly this
`nn.TransformerDecoderLayer` call. Apart from whitespace, the handoff file
differs from upstream only by disabling that block and adding the
`DecoderLayerWithCrossAttn` block and its commented import.

The standardized implementation is exactly this branch, so the 2026 model is
the upstream authors' published model.

## Settings

| Setting | Value | Source |
|---|---|---|
| Layer | `torch.nn.TransformerDecoderLayer` | standard branch of the archived model file |
| Stack | `torch.nn.TransformerDecoder`, 3 layers | `num_layers=3` in `config/base_config.py` |
| `d_model` | 128 (64 image + 64 item text) | `img_emb_size=64`, `txt_emb_size=64` |
| Attention heads | 16 | `nhead=16` |
| Dropout | 0.1 | `dropout=0.1` |
| `batch_first` | `True` | constructor call |
| Other arguments | PyTorch defaults (feed-forward 2048, ReLU, post-norm) | not overridden |

## Input and output flow (CP)

1. Each item becomes one 128-d token: image embedding (64) concatenated with
   item-text embedding (64).
2. Target sequence: a learned outfit token followed by the item tokens.
3. Memory: the outfit-level text embedding followed by the item tokens.
4. Each decoder layer applies self-attention over the target sequence and
   cross-attention to the memory, with the same padding mask on both.
5. The output at the outfit-token position goes through an MLP
   (128 → 128 → 1) to a compatibility logit trained with
   `BCEWithLogitsLoss`. FITB scores candidates with the same CP model.

CIR uses `OutfitTransformerRetrieval`, which already builds its decoder from
`torch.nn.TransformerDecoderLayer` in the archived code. It is initialized from
the same-seed CP best checkpoint (parameters with matching names and shapes),
so CP and CIR share one decoder implementation.

## How it is applied

`reproduction/scripts/standard_decoder.py` switches the single
`DecoderLayerWithCrossAttn(` constructor to `nn.TransformerDecoderLayer(` in a
working copy of the archived source. Every entry point uses this module:

| Entry point | Role |
|---|---|
| `scripts/run_full_single_seed_training.py` | main CP → CIR training |
| `scripts/pipeline/evaluate_main_run.py` | main evaluation |
| `scripts/run_fair_subset_single_seed.py` | fair-subset ablation units |
| `scripts/run_single_seed_smoke_training.py` | smoke test |

The archived `02_…` tree is never modified (`REPRODUCIBILITY.md`). Run
manifests record the implementation under `cp_decoder` or `decoder_label`.

## Evidence that behavior is unchanged

- Moving the replacement from four per-script copies into the shared module
  left training bit-identical: the smoke CP and CIR checkpoints of
  `smoke_20260930T154824Z` match `smoke_20260924T102104Z` byte for byte.
- The completed runs `reference_20260921T175217Z` and
  `full_20260924T102513Z` used the same replacement. Their recorded manifests
  still carry the earlier wording "temporary … candidate" and are kept
  unchanged as historical records.

## What is not claimed

The historical `DecoderLayerWithCrossAttn` forward behavior is not recovered,
and the 2025 thesis numbers are not claimed to come from this implementation.
Hyperparameter provenance for the same model is in `hyperparameter_tuning.md`.
