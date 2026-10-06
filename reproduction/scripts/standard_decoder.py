"""Standardized decoder implementation for the Hybrid Attention CP model.

The archived handoff model file
02_模型訓練和驗證_model_training_validation/main_hybrid_attention_code/outfit_transformer.py
builds the CP decoder with ``DecoderLayerWithCrossAttn``, whose definition was
not preserved: its import in the archived file is commented out.  The same
file keeps the author's standard branch, commented out, with identical
constructor arguments:

    decoder_layer = nn.TransformerDecoderLayer(
        d_model=img_emb_size+txt_emb_size, nhead=nhead, batch_first=True, dropout=dropout)

Every 2026 experiment (training, evaluation, smoke test and fair-subset runs)
uses this standardized decoder implementation, applied by ``apply`` to a
working copy of the archived source.  The archived repository file itself is
never modified.  See reproduction/docs/standardized_decoder.md.
"""
from __future__ import annotations

from pathlib import Path

ARCHIVED_CONSTRUCTOR = "decoder_layer = DecoderLayerWithCrossAttn("
STANDARD_CONSTRUCTOR = "decoder_layer = nn.TransformerDecoderLayer("
LABEL = "standardized decoder implementation (torch.nn.TransformerDecoderLayer)"
MANIFEST = {
    "implementation": "torch.nn.TransformerDecoderLayer inside torch.nn.TransformerDecoder",
    "label": LABEL,
    "constructor_arguments": "d_model=img_emb_size+txt_emb_size, nhead=nhead, batch_first=True, dropout=dropout",
    "replaces_archived_symbol": "DecoderLayerWithCrossAttn (definition not preserved in the handoff)",
    "archived_source_modified": False,
    "documentation": "reproduction/docs/standardized_decoder.md",
}


def apply(model_file: Path) -> None:
    """Switch the CP decoder of a working copy of outfit_transformer.py to the standard layer."""
    text = model_file.read_text(encoding="utf-8")
    if text.count(ARCHIVED_CONSTRUCTOR) != 1:
        raise SystemExit(f"Expected exactly one {ARCHIVED_CONSTRUCTOR!r} in {model_file}")
    model_file.write_text(text.replace(ARCHIVED_CONSTRUCTOR, STANDARD_CONSTRUCTOR, 1), encoding="utf-8")
