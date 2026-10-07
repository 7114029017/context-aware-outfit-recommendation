"""Shared paths and helpers for the extension analyses (reproduction/docs/extensions.md).

The extensions regenerate manuscript analyses that the 35 training units do not
produce. They run after a full run has PASSED, read its saved checkpoints and
outputs, and never write into the run's training, evaluation or statistics
results; the code used by the 35 units is not changed.
"""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "supplementary"))
from _common import (D01, D02, D03, GENERATED, OFFICIAL_RUN, REPO, REPRO, WOS_JSONL, fmt,  # noqa: E402,F401
                     polyvore_root, read_csv, read_json, signed, write_csv, write_json, write_text)

SCRIPTS = REPRO / "scripts"
MODEL_CODE = D02 / "main_hybrid_attention_code"
FEATURES = D02 / "fashionclip_data"
CONTEXT_FEATURES = FEATURES / "encoded_NewoutfitUrlTitle_en_fashionClip.pkl"
ORIGINAL_FEATURES = FEATURES / "encoded_outfitUrlTitle_en_fashionClip.pkl"
CHECKPOINTS_2025 = D02 / "main_hybrid_attention_checkpoints"
LENGTH_DIR = D03 / "01_文字長度影響分析"
CASE_DIR = D03 / "04_情境子集與三因子分析"
TWO_TOWER_DIR = D03 / "05_第二模型驗證"
TWO_TOWER_MODELS = D02 / "second_model_two_tower" / "models"
COUNTERFACTUAL_DIR = D03 / "06_反事實情境敏感度"
EXTENSIONS = REPRO / "results" / "extensions"
SEEDS = (1, 2, 3, 4, 5)
MAIN_UNITS = {  # main/<unit>: (archived evaluator variant, outfit feature file)
    **{f"original_seed{s}": ("outfitUrlTitle", "encoded_outfitUrlTitle_en_fashionClip.pkl") for s in SEEDS},
    **{f"context_seed{s}": ("NewoutfitUrlTitle", "encoded_NewoutfitUrlTitle_en_fashionClip.pkl") for s in SEEDS},
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def passed_run(run_root: Path) -> Path:
    """A completed full run folder; stops unless its RUN_STATUS is PASSED."""
    run_root = run_root.expanduser().resolve()
    status = run_root / "RUN_STATUS.txt"
    state = status.read_text(encoding="utf-8").strip() if status.is_file() else "missing"
    if state != "PASSED":
        raise SystemExit(f"[EXTENSION BLOCKED] {run_root}: RUN_STATUS is {state}, not PASSED")
    return run_root


def output_dir(path: Path, run_root: Path | None = None) -> Path:
    """Resolved output folder; never inside the run's own results (only under <run>/extensions/)."""
    out = path.expanduser().resolve()
    if run_root is not None and out.is_relative_to(run_root) and not out.is_relative_to(run_root / "extensions"):
        raise SystemExit(f"[EXTENSION BLOCKED] output {out} would be inside the run's results; use <run>/extensions/")
    out.mkdir(parents=True, exist_ok=True)
    return out


def local_path(name: str, arg: str | None) -> Path | None:
    """--option value, else the path recorded by bootstrap_data.sh in reproduction/.local/<name>.txt."""
    if arg:
        return Path(arg).expanduser().resolve()
    recorded = REPRO / ".local" / f"{name}.txt"
    return Path(recorded.read_text(encoding="utf-8").strip()).resolve() if recorded.is_file() else None
