"""Shared paths and helpers for the supplementary analyses.

The supplementary scripts only read files that are already in the repository or
downloaded by reproduction/scripts/bootstrap_data.sh. They load no model, use no
GPU and do not modify the official run's results.
"""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
REPRO = REPO / "reproduction"
OFFICIAL_RUN = "full_20261002T161428Z"
SUPPLEMENTARY = REPRO / "results" / "supplementary"

D01 = REPO / "01_資料建構_data_construction"
D02 = REPO / "02_模型訓練和驗證_model_training_validation"
D03 = REPO / "03_實驗與結果_experiments_results"
GENERATED = D01 / "generated_descriptions" / "01_生成結果_generation_results" / "new_polyvore_outfit_titles.json"
WOS_JSONL = (D01 / "generated_descriptions" / "02_三因子拆分_wos_factor_split"
             / "wos_split_results_v5_merged_retry_round3.jsonl")
TEMPERATURE_DIR = D01 / "clo_met_temperature" / "temperature_results"
MET_CANDIDATES = D01 / "clo_met_temperature" / "MET_reference" / "adult_activity_compendium_sorted_2024.json"
CONTEXT_FEATURES = D02 / "fashionclip_data" / "encoded_NewoutfitUrlTitle_en_fashionClip.pkl"
CASE_TABLES = D03 / "04_情境子集與三因子分析" / "圖表_figures_tables" / "tables"
TARGET_CLUE = D03 / "02_目標單品線索檢查"
TWO_TOWER_SEEDS = D03 / "05_第二模型驗證" / "圖表_figures_tables" / "tables" / "A30_second_model_two_tower_seed_summary.csv"
COUNTERFACTUAL = D03 / "06_反事實情境敏感度"
SPLITS = ("train", "valid", "test")


def polyvore_root(arg: str | None) -> Path:
    """--polyvore-root, or the path recorded by bootstrap_data.sh."""
    if arg:
        root = Path(arg).expanduser().resolve()
    else:
        recorded = REPRO / ".local" / "polyvore_root.txt"
        if not recorded.is_file():
            raise SystemExit("[SUPPLEMENTARY BLOCKED] pass --polyvore-root or run "
                             "reproduction/scripts/bootstrap_data.sh first")
        root = Path(recorded.read_text(encoding="utf-8").strip()).expanduser().resolve()
    if not (root / "categories.csv").is_file():
        raise SystemExit(f"[SUPPLEMENTARY BLOCKED] categories.csv not found under the Polyvore root: {root}")
    return root


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, header: list[str], rows: list[list]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow(header)
        writer.writerows(rows)


def write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")


def write_text(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def fmt(value: float | None, digits: int = 4) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    return f"{value:.{digits}f}"


def signed(value: float | None, digits: int = 4) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    return f"{value:+.{digits}f}"


def load_temperatures() -> dict[str, tuple[str, dict]]:
    """set_id -> (split, record) from the temperature files; duplicated records are identical."""
    out: dict[str, tuple[str, dict]] = {}
    for split in SPLITS:
        for record in read_json(TEMPERATURE_DIR / f"{split}_temperature.json"):
            out.setdefault(str(record["set_id"]), (split, record))
    return out


def bh_adjust(pvalues: list[float]) -> list[float]:
    """Benjamini-Hochberg adjusted p-values, in the input order."""
    n = len(pvalues)
    order = sorted(range(n), key=lambda i: pvalues[i])
    adjusted = [0.0] * n
    running = 1.0
    for rank in range(n, 0, -1):
        i = order[rank - 1]
        running = min(running, pvalues[i] * n / rank)
        adjusted[i] = running
    return adjusted
