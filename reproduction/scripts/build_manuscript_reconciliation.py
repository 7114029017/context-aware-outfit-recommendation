#!/usr/bin/env python3
"""Reconcile thesis / TORS manuscript values against one completed run.

Inputs:
- reproduction/docs/manuscript_reconciliation/manuscript_values.csv, the
  hand-transcribed values and claims of the thesis final PDF and the TORS v4
  manuscript (both in 04_文件資料_documents/);
- the result files of the committed reference run (--reference) or of one
  completed integrated full run (--run-root).

Outputs reconciliation.csv and reconciliation.md.  Every latest value is read
from a result file; this script never trains, evaluates, or recomputes
statistics.  Derived values are limited to differences and ratios of
reported means and to boolean checks of manuscript claims.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
REPRO = ROOT / "reproduction"
REF_ID = "reference_20260921T175217Z"
RECON_DIR = REPRO / "docs" / "manuscript_reconciliation"
OFFICIAL_MANIFEST = REPRO / "results" / "final_reference_manifest.json"
OFFICIAL_EQUIVALENCE = ("`reference_20260921T175217Z`、`full_20260924T102513Z` 與此 run 逐位元相同（70 個 checkpoint、"
                        "逐 seed 結果），見 `reproduction/docs/clean_room_acceptance.md`")
REFERENCE_EQUIVALENCE = ("`full_20260924T102513Z`（重構後 `--fresh`，commit `2cd8d04`）與此 run bit-identical，"
                         "見 `reproduction/docs/post_restructure_fresh_run_verification.md`")
VALUES_CSV = RECON_DIR / "manuscript_values.csv"
SPLITS = REPRO / "splits"
FAIR_MANIFEST = SPLITS / "fair_subset" / "fair_subset_reconstruction_manifest.json"
FAIR_SCOPE = SPLITS / "fair_subset" / "fair_subset_cir_scope_audit.json"
CP_CONFIG = REPRO / "configs" / "cp.yaml"
OR_CONFIG = REPRO / "configs" / "or.yaml"
EXPECTED_RESULTS = REPRO / "docs" / "expected_results.md"
THESIS_PDF = "04_文件資料_documents/thesis/情境感知驅動的智慧穿搭推薦系統 論文終稿.pdf"
TORS_PDF = "04_文件資料_documents/journal/ACM_TORS_English_v4.pdf"

P_MAIN = "reproduction/scripts/compare_main_results.py"
P_STATS = "reproduction/scripts/pipeline/complete_statistics.py"
P_FAIR = "reproduction/scripts/summarize_fair_subset_5seed.py"
P_SPLITS = "reproduction/scripts/freeze_splits.py"
P_FAIR_IDS = "reproduction/scripts/reconstruct_fair_subset_from_wos.py"
P_FAIR_SCOPE = "reproduction/scripts/audit_fair_subset_cir_scope.py"
P_ENV = "reproduction/scripts/validate_reproduction_env.py"
P_TRAIN = "reproduction/scripts/run_full_single_seed_training.py"
P_BASE_CONFIG = "02_模型訓練和驗證_model_training_validation/main_hybrid_attention_code/config/base_config.py"
P_TWO_TOWER = "reproduction/scripts/reproduce_remaining_thesis.py"
P_CHAPTER4 = "reproduction/scripts/pipeline/build_chapter4_report.py"
P_SELF = "reproduction/scripts/build_manuscript_reconciliation.py"

MAIN_METRICS = ("cp_auc", "cp_fitb", "or_r1", "or_r3", "or_r5", "or_r10", "or_r30", "or_r50")
OR_METRICS = MAIN_METRICS[2:]
MAIN_LABEL = {
    "cp_auc": "AUC", "cp_fitb": "FITB", "or_r1": "R@1", "or_r3": "R@3",
    "or_r5": "R@5", "or_r10": "R@10", "or_r30": "R@30", "or_r50": "R@50",
}
FAIR_CP = ("auc", "fitb_acc")
FAIR_OR = ("recall_at_10", "recall_at_30", "recall_at_50")
FAIR_LABEL = {"auc": "AUC", "fitb_acc": "FITB", "recall_at_10": "R@10",
              "recall_at_30": "R@30", "recall_at_50": "R@50"}
SUBSET_TABLE_LABEL = {"auc": "AUC", "fitb_acc": "FITB Acc", "recall_at_10": "Recall@10",
                      "recall_at_30": "Recall@30", "recall_at_50": "Recall@50"}
FACTORS = ("no_weather", "no_occasion", "no_style")
FACTOR_LABEL = {"no_weather": "No-W", "no_occasion": "No-O", "no_style": "No-S"}

STATUS_LABEL = {
    "KEEP": "保留",
    "UPDATE_NUMBER": "更新數字",
    "REWRITE": "改寫敘述",
    "ADD": "需補寫",
    "OUTSIDE_FORMAL_RUN": "非正式 run",
}
STATUS_MEANING = {
    "KEEP": "最新值四捨五入到稿件位數後相同，或稿件敘述仍成立",
    "UPDATE_NUMBER": "數字改變，但方向與顯著性結論不變",
    "REWRITE": "方向、顯著性、筆數或文字結論與最新結果不一致，敘述必須改寫",
    "ADD": "稿件未記載，修訂稿需補寫",
    "OUTSIDE_FORMAL_RUN": "35 組正式訓練未重新產生；來源為保存輸出或其他分析",
}
GROUPS = (
    ("abstract", "一、Abstract 主要結果"),
    ("setup", "二、Experimental setup 與模型設定"),
    ("tables", "三、表 4-11 至表 4-16 全部欄位"),
    ("stats_text", "四、Results 內文的 p 值、BH 校正、效果量與信賴區間"),
    ("conclusions", "五、Discussion 與 Conclusion 核心結論"),
    ("required", "六、第九節要求保留的結論"),
    ("readme_supp", "七、README expected results 與補充表格"),
)
TABLE_TORS = {"4-11": "Table 6", "4-12": "Table 6", "4-13": "Table 7",
              "4-14": "Table 7", "4-15": "Table 7", "4-16": "Table 7"}


def fail(message: str) -> None:
    raise SystemExit("[RECONCILIATION BLOCKED] " + message)


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        fail(f"missing CSV: {path}")
    with path.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        fail(f"empty CSV: {path}")
    return rows


def read_json(path: Path) -> dict:
    if not path.is_file():
        fail(f"missing JSON: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def read_yaml(path: Path) -> dict:
    if not path.is_file():
        fail(f"missing YAML: {path}")
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path.resolve())


def stars(p: float) -> str:
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "n.s."


def parse_number(text: str) -> float:
    cleaned = text.replace("−", "-").replace(",", "").replace("%", "").strip()
    return float(cleaned)


def decimals(text: str) -> int:
    match = re.search(r"\.(\d+)", text)
    return len(match.group(1)) if match else 0


def sign(value: float) -> int:
    return (value > 0) - (value < 0)


def fmt(value: float, places: int, signed: bool = False) -> str:
    return f"{value:+.{places}f}" if signed else f"{value:.{places}f}"


def signed(value: float, places: int = 4) -> str:
    return fmt(value, places, signed=True)


@dataclass(frozen=True)
class Resolved:
    value: object = None
    text: str = "—"
    sources: tuple[Path, ...] = ()
    programs: tuple[str, ...] = ()
    holds: bool | None = None


class Latest:
    """Read-only view of one completed run's result files."""

    def __init__(self, run_id: str, paths: dict[str, Path]) -> None:
        self.run_id = run_id
        self.paths = paths
        self.f_main = paths["main"] / "main_reproduction_summary.csv"
        self.f_stats = paths["stats"] / "main_paired_bh_8metrics.csv"
        self.f_evidence = paths["stats"] / "statistical_evidence.json"
        self.f_fair = paths["ablation"] / "fresh_5variant_8metric_summary.csv"
        self.f_fair_seeds = paths["ablation"] / "fresh_25unit_metrics.csv"
        self.f_t03 = paths["ablation"] / "T03_stage1_cp_original_vs_full.csv"
        self.f_t04 = paths["ablation"] / "T04_stage1_cir_original_vs_full.csv"
        self.f_t07 = paths["ablation"] / "T07_stage2_factor_contribution_ratio.csv"
        self.f_fair_json = paths["ablation"] / "fresh_fair_subset_5seed_summary.json"
        self.f_two_tower = paths["secondary"] / "two_tower" / "recomputed_key_mean_std.csv"
        self.f_overview = paths["chapter4"] / "chapter4_table_overview.csv"

        self.main = {(r["metric"], r["condition"]): r for r in read_csv(self.f_main)}
        self.stats = {r["metric"]: r for r in read_csv(self.f_stats)}
        if set(self.stats) != set(MAIN_METRICS):
            fail(f"statistics file does not contain exactly the 8 main metrics: {self.f_stats}")
        self.evidence = read_json(self.f_evidence)
        self.fair = {(r["variant"], r["metric_key"]): r for r in read_csv(self.f_fair)}
        self.fair_seeds = read_csv(self.f_fair_seeds)
        # Thesis Tables 4-13/4-14 (T03/T04), keyed by fair-subset metric key.
        label_to_key = {label: key for key, label in SUBSET_TABLE_LABEL.items()}
        self.stage1 = {label_to_key[row["Metric"]]: (row, path)
                       for path in (self.f_t03, self.f_t04) for row in read_csv(path)}
        if set(self.stage1) != set(FAIR_CP + FAIR_OR):
            fail("T03/T04 do not contain exactly the five fair-subset table metrics")
        self.fair_json = read_json(self.f_fair_json)
        self.t07 = read_csv(self.f_t07)
        self.two_tower = {r["variant"]: r for r in read_csv(self.f_two_tower)}
        self.overview = {r["table"]: r for r in read_csv(self.f_overview)}
        self.env = read_json(paths["environment"])
        self.cp_config = read_yaml(CP_CONFIG)
        self.or_config = read_yaml(OR_CONFIG)
        self.fair_manifest = read_json(FAIR_MANIFEST)
        self.fair_scope = read_json(FAIR_SCOPE)

    # ---- accessors ------------------------------------------------------
    def stat(self, metric: str, field: str) -> float:
        return float(self.stats[metric][field])

    def bh(self, metric: str) -> float:
        return self.stat(metric, "bh_adjusted_p_8_main")

    def fair_mean(self, variant: str, key: str) -> float:
        return float(self.fair[(variant, key)]["mean"])

    def fair_delta(self, factor: str, key: str) -> float:
        """No-factor minus Context, i.e. thesis ΔX (negative = removal hurts)."""
        return self.fair_mean(factor, key) - self.fair_mean("context", key)

    def seeds(self) -> list[int]:
        return [int(s) for s in self.stats["cp_auc"]["seeds"].split(",")]

    def split_count(self, name: str) -> int:
        path = SPLITS / f"{name}_ids.csv"
        with path.open(encoding="utf-8", newline="") as stream:
            return sum(1 for _ in csv.DictReader(stream))

    # ---- key resolution -------------------------------------------------
    def resolve(self, key: str) -> Resolved:
        head = key.split(".", 1)[0]
        handler = {
            "main": self._main, "fair": self._fair, "split": self._split,
            "config": self._config, "env": self._env, "seeds": self._seeds,
            "decoder": self._decoder, "tt": self._two_tower,
            "overview": self._overview, "check": self._check,
            "none": lambda _k: Resolved(),
        }.get(head)
        if handler is None:
            fail(f"unknown latest_key: {key}")
        return handler(key)

    def _main(self, key: str) -> Resolved:
        _, metric, field = key.split(".")
        if metric == "or":
            if field == "dz_range":
                values = [self.stat(m, "cohen_dz") for m in OR_METRICS]
            else:
                values = [self.stat(m, "delta_mean") / self.stat(m, "original_mean") for m in OR_METRICS]
            programs = (P_STATS,) if field == "dz_range" else (P_STATS, P_SELF)
            return Resolved((min(values), max(values)), sources=(self.f_stats,), programs=programs)
        if field.endswith(("_mean", "_sd")):
            condition, stat = field.rsplit("_", 1)
            row = self.main[(metric, condition)]
            value = float(row["reproduced_mean" if stat == "mean" else "reproduced_sd"])
            return Resolved(value, sources=(self.f_main,), programs=(P_MAIN,))
        if field == "stars":
            p = self.bh(metric)
            return Resolved(stars(p), f"{stars(p)}（BH p = {p:.3g}）", (self.f_stats,), (P_STATS,))
        if field == "rel_gain":
            value = self.stat(metric, "delta_mean") / self.stat(metric, "original_mean")
            return Resolved(value, sources=(self.f_stats,), programs=(P_STATS, P_SELF))
        column = {"delta": "delta_mean", "ci_low": "ci95_low", "ci_high": "ci95_high", "dz": "cohen_dz"}[field]
        return Resolved(self.stat(metric, column), sources=(self.f_stats,), programs=(P_STATS,))

    def _fair(self, key: str) -> Resolved:
        parts = key.split(".")
        if parts[1] == "count":
            if parts[2] == "total":
                return Resolved(int(self.fair_manifest["candidate_unique_ids"]),
                                sources=(FAIR_MANIFEST,), programs=(P_FAIR_IDS,))
            split = self.fair_manifest["candidate_overlap_by_split"]
            text = f"{split['train']:,}／{split['valid']:,}／{split['test']:,}"
            return Resolved(text, text, (FAIR_MANIFEST,), (P_FAIR_IDS,))
        if parts[1] == "coverage":
            patterns = {k: int(v) for k, v in self.fair_manifest["wos_factor_presence_patterns"].items()}
            total = int(self.fair_manifest["wos_records_unique"])
            position = {"weather": 0, "occasion": 1, "style": 2}
            if parts[2] == "all":
                count = patterns.get("111", 0)
            else:
                count = sum(v for k, v in patterns.items() if k[position[parts[2]]] == "1")
            return Resolved(count / total, sources=(FAIR_MANIFEST,), programs=(P_FAIR_IDS,))
        if parts[1] == "cir_evaluable":
            value = int(self.fair_scope["observed_preserved_evaluator_semantics"]["evaluable_questions"])
            return Resolved(value, sources=(FAIR_SCOPE,), programs=(P_FAIR_SCOPE,))
        if parts[1] == "cir_seed_obs":
            queries = int(self.fair_scope["observed_preserved_evaluator_semantics"]["evaluable_questions"])
            n_seeds = len({row["seed"] for row in self.fair_seeds if row["variant"] == "context"})
            return Resolved(queries * n_seeds, sources=(FAIR_SCOPE, self.f_fair_seeds),
                            programs=(P_FAIR_SCOPE, P_FAIR))
        if parts[1] == "bh_family":
            text = self.fair_json["statistical_method"]["T03_T04_multiple_testing"]
            return Resolved(text, text, (self.f_fair_json,), (P_FAIR,))

        metric, field = parts[1], parts[2]
        if field in ("ci_low", "ci_high", "stars", "dz"):
            row, path = self.stage1[metric]
            if field == "stars":
                label = row["Sig. (BH)"]
                return Resolved(label, f"{label}（BH p = {row['BH-adjusted p']}）", (path,), (P_FAIR,))
            if field == "dz":
                return Resolved(float(row["Cohen’s dz"]), sources=(path,), programs=(P_FAIR,))
            low, high = (float(x) for x in row["95% CI of Δ"].strip("[]").split(","))
            return Resolved(low if field == "ci_low" else high, sources=(path,), programs=(P_FAIR,))
        if field == "delta_context":
            value = self.fair_mean("context", metric) - self.fair_mean("original", metric)
            return Resolved(value, sources=(self.f_fair,), programs=(P_FAIR, P_SELF))
        if field == "drop_no_style":
            value = self.fair_mean("context", metric) - self.fair_mean("no_style", metric)
            return Resolved(value, sources=(self.f_fair,), programs=(P_FAIR, P_SELF))
        if field.startswith("delta_"):
            value = self.fair_delta(field[len("delta_"):], metric)
            return Resolved(value, sources=(self.f_fair,), programs=(P_FAIR, P_SELF))
        variant, stat = field.rsplit("_", 1)
        row = self.fair[(variant, metric)]
        value = float(row["mean" if stat == "mean" else "std"])
        return Resolved(value, sources=(self.f_fair,), programs=(P_FAIR,))

    def _split(self, key: str) -> Resolved:
        name = key.split(".", 1)[1]
        if name == "total":
            names = ("train", "validation", "test")
            value = sum(self.split_count(n) for n in names)
            sources = tuple(SPLITS / f"{n}_ids.csv" for n in names)
        else:
            value = self.split_count(name)
            sources = (SPLITS / f"{name}_ids.csv",)
        return Resolved(value, sources=sources, programs=(P_SPLITS,))

    def _config(self, key: str) -> Resolved:
        parts = key.split(".")
        programs = (P_TRAIN, P_BASE_CONFIG)
        if parts[1] == "scheduler":
            sched = self.cp_config["training"]["scheduler"]
            text = (f"{sched['class']}（gamma {sched['gamma']}，"
                    f"{sched['scheduler_step_call'].replace('every_', '每 ').replace('_epochs', ' 個 epoch 呼叫一次')}）")
            return Resolved(text, text, (CP_CONFIG,), programs)
        config, path = (self.cp_config, CP_CONFIG) if parts[1] == "cp" else (self.or_config, OR_CONFIG)
        node: object = config
        for part in parts[2:]:
            if not isinstance(node, dict) or part not in node:
                fail(f"config key not found: {key}")
            node = node[part]
        return Resolved(node, str(node), (path,), programs)

    def _env(self, key: str) -> Resolved:
        path = self.paths["environment"]
        runtime = self.env["torch_runtime"]
        if key == "env.gpu":
            names = sorted({d["name"] for d in runtime["devices"]})
            text = "、".join(names)
            return Resolved(text, text, (path,), (P_ENV,))
        cudnn = int(runtime["cudnn_version"])
        text = (f"Python {self.env['python'].split()[0]}；"
                f"PyTorch {self.env['packages']['torch']['installed_version']}；"
                f"torchvision {self.env['packages']['torchvision']['installed_version']}；"
                f"CUDA {runtime['torch_cuda_version']}；"
                f"cuDNN {cudnn // 10000}.{cudnn % 10000 // 100}.{cudnn % 100}")
        return Resolved(text, text, (path,), (P_ENV,))

    def _seeds(self, key: str) -> Resolved:
        text = ", ".join(str(s) for s in self.seeds())
        return Resolved(text, text, (self.f_stats,), (P_STATS,))

    def _decoder(self, key: str) -> Resolved:
        model = self.cp_config["model"]
        if model.get("historical_decoder_source_available") is not False:
            fail("cp.yaml no longer records the historical decoder as unavailable; review decoder.impl")
        text = (f"torch.nn.TransformerDecoderLayer（{model['transformer_layers']} 層、"
                f"{model['attention_heads']} heads、dropout {model['dropout']}）；"
                f"歷史 {model['historical_decoder_symbol']} 定義未保存")
        return Resolved(text, text, (CP_CONFIG,), (P_TRAIN,))

    def _two_tower(self, key: str) -> Resolved:
        original = self.two_tower["original_text"]
        context = self.two_tower["context_aware_description"]
        metric = key.split(".", 1)[1]
        if metric == "summary":
            def diff(col: str) -> float:
                return float(context[f"{col}_mean"]) - float(original[f"{col}_mean"])
            text = (f"AUC {signed(diff('cp_test_auc'))}、FITB {signed(diff('cp_test_fitb_acc'))}、"
                    f"R@10 {signed(diff('recall@10'))}、R@30 {signed(diff('recall@30'))}、"
                    f"R@50 {signed(diff('recall@50'))}（保存輸出重算）")
            return Resolved(None, text, (self.f_two_tower,), (P_TWO_TOWER,))
        values = (float(original[f"{metric}_mean"]), float(original[f"{metric}_std"]),
                  float(context[f"{metric}_mean"]), float(context[f"{metric}_std"]),
                  float(context[f"{metric}_mean"]) - float(original[f"{metric}_mean"]))
        return Resolved(values, sources=(self.f_two_tower,), programs=(P_TWO_TOWER,))

    def _overview(self, key: str) -> Resolved:
        table = key.split(".", 1)[1]
        row = self.overview.get(table)
        if row is None:
            fail(f"chapter4 overview has no table {table}")
        text = f"{row['result']}：{row['limitation']}"
        return Resolved(row["result"], text, (self.f_overview,), (P_CHAPTER4,))

    # ---- claim checks ---------------------------------------------------
    def _check(self, key: str) -> Resolved:
        name = key.split(".", 1)[1]
        method = getattr(self, f"check_{name}", None)
        if method is None:
            fail(f"unknown check: {name}")
        holds, text, sources, programs = method()
        return Resolved(holds, text, sources, tuple(programs) + (P_SELF,), holds)

    def _main_sources(self) -> tuple[tuple[Path, ...], tuple[str, ...]]:
        return (self.f_stats,), (P_STATS,)

    def _fair_sources(self) -> tuple[tuple[Path, ...], tuple[str, ...]]:
        return (self.f_fair,), (P_FAIR,)

    def _deltas_text(self, factor: str, keys: tuple[str, ...]) -> str:
        return "、".join(f"{FAIR_LABEL[k]} {signed(self.fair_delta(factor, k))}" for k in keys)

    def check_main_all_positive(self):
        deltas = {m: self.stat(m, "delta_mean") for m in MAIN_METRICS}
        negative = [MAIN_LABEL[m] for m, d in deltas.items() if d <= 0]
        low = min(deltas, key=deltas.get)
        text = (f"{8 - len(negative)}／8 項平均差為正（最小：{MAIN_LABEL[low]} {signed(deltas[low])}）"
                + (f"；非正：{'、'.join(negative)}" if negative else ""))
        return not negative, text, *self._main_sources()

    def _significance(self) -> tuple[int, list[str]]:
        ns = [f"{MAIN_LABEL[m]}（BH p = {self.bh(m):.3g}）" for m in MAIN_METRICS if self.bh(m) >= 0.05]
        return 8 - len(ns), ns

    def check_main_all_sig(self):
        n_sig, ns = self._significance()
        text = f"BH 校正後 {n_sig}／8 顯著" + (f"；未顯著：{'、'.join(ns)}" if ns else "")
        return n_sig == 8, text, *self._main_sources()

    def check_main_sig_count7(self):
        n_sig, ns = self._significance()
        text = f"BH 校正後 {n_sig}／8 顯著" + (f"；未顯著：{'、'.join(ns)}" if ns else "")
        return n_sig == 7, text, *self._main_sources()

    def check_main_all_ci_sig(self):
        crosses = [f"{MAIN_LABEL[m]} [{signed(self.stat(m, 'ci95_low'), 5)}, {signed(self.stat(m, 'ci95_high'), 5)}]"
                   for m in MAIN_METRICS if self.stat(m, "ci95_low") <= 0]
        n_sig, ns = self._significance()
        parts = [f"BH 校正後 {n_sig}／8 顯著"]
        if crosses:
            parts.append(f"CI 含 0：{'、'.join(crosses)}")
        if ns:
            parts.append(f"未顯著：{'、'.join(ns)}")
        return not crosses and n_sig == 8, "；".join(parts), *self._main_sources()

    def check_cp_ci_sig_all(self):
        cp = ("cp_auc", "cp_fitb")
        holds = all(self.stat(m, "ci95_low") > 0 and self.bh(m) < 0.05 for m in cp)
        text = "；".join(f"{MAIN_LABEL[m]} CI [{signed(self.stat(m, 'ci95_low'), 5)}, "
                        f"{signed(self.stat(m, 'ci95_high'), 5)}]、BH p = {self.bh(m):.3g}" for m in cp)
        return holds, text, *self._main_sources()

    def check_auc_dz_largest(self):
        dz = {m: self.stat(m, "cohen_dz") for m in MAIN_METRICS}
        top = max(dz, key=dz.get)
        text = f"AUC dz = {dz['cp_auc']:.3f}；八項中最大為 {MAIN_LABEL[top]} dz = {dz[top]:.3f}"
        return top == "cp_auc", text, *self._main_sources()

    def check_r10_r50_level_drop(self):
        parts, holds = [], True
        for m in ("or_r10", "or_r50"):
            raw, adj = self.stat(m, "p_value"), self.bh(m)
            holds &= raw < 0.001 <= adj
            parts.append(f"{MAIN_LABEL[m]}：未校正 p = {raw:.3g}（{stars(raw)}）、BH p = {adj:.3g}（{stars(adj)}）")
        return holds, "；".join(parts), *self._main_sources()

    def check_r1_r3_ci_lower_positive(self):
        low = {m: self.stat(m, "ci95_low") for m in ("or_r1", "or_r3")}
        text = "；".join(f"{MAIN_LABEL[m]} CI 下界 {signed(v, 5)}" + ("（含 0）" if v <= 0 else "")
                        for m, v in low.items())
        return all(v > 0 for v in low.values()), text, *self._main_sources()

    def check_large_k_dz_larger(self):
        dz = {m: self.stat(m, "cohen_dz") for m in OR_METRICS}
        large, small = ("or_r10", "or_r30", "or_r50"), ("or_r1", "or_r3")
        holds = min(dz[m] for m in large) > max(dz[m] for m in small)
        text = ("R@10／30／50 dz = " + "／".join(f"{dz[m]:.3f}" for m in large)
                + "；R@1／3 dz = " + "／".join(f"{dz[m]:.3f}" for m in small))
        return holds, text, *self._main_sources()

    def check_r5_r10_largest_rel(self):
        gain = {m: self.stat(m, "delta_mean") / self.stat(m, "original_mean") for m in OR_METRICS}
        top2 = sorted(gain, key=gain.get, reverse=True)[:2]
        text = "相對提升最大：" + "、".join(f"{MAIN_LABEL[m]} {100 * gain[m]:.1f}%" for m in top2)
        return set(top2) == {"or_r5", "or_r10"}, text, *self._main_sources()

    def check_or_all_positive(self):
        deltas = {m: self.stat(m, "delta_mean") for m in OR_METRICS}
        text = "、".join(f"{MAIN_LABEL[m]} {signed(d)}" for m, d in deltas.items())
        return all(d > 0 for d in deltas.values()), text, *self._main_sources()

    def check_fair_all5_positive(self):
        keys = FAIR_CP + FAIR_OR
        deltas = {k: self.fair_mean("context", k) - self.fair_mean("original", k) for k in keys}
        n_pos = sum(d > 0 for d in deltas.values())
        text = f"{n_pos}／5 項為正（" + "、".join(f"{FAIR_LABEL[k]} {signed(d)}" for k, d in deltas.items()) + "）"
        return n_pos == 5, text, *self._fair_sources()

    def _stage1_text(self, keys: tuple[str, ...]) -> tuple[str, tuple[Path, ...]]:
        parts, sources = [], []
        for key in keys:
            row, path = self.stage1[key]
            parts.append(f"{FAIR_LABEL[key]} CI {row['95% CI of Δ']}、BH p = {row['BH-adjusted p']}（{row['Sig. (BH)']}）")
            sources.append(path)
        return "；".join(parts), tuple(dict.fromkeys(sources))

    def _stage1_ci_sig(self, keys: tuple[str, ...]) -> bool:
        return all(float(self.stage1[k][0]["95% CI of Δ"].strip("[]").split(",")[0]) > 0
                   and self.stage1[k][0]["Sig. (BH)"] != "n.s." for k in keys)

    def check_fair_cp_sig(self):
        text, sources = self._stage1_text(FAIR_CP)
        return self._stage1_ci_sig(FAIR_CP), text, sources, (P_FAIR,)

    def check_fair_or_ci_sig(self):
        text, sources = self._stage1_text(FAIR_OR)
        return self._stage1_ci_sig(FAIR_OR), text, sources, (P_FAIR,)

    def check_fair_r50_level_drop(self):
        row, path = self.stage1["recall_at_50"]
        raw = "***" if row["Raw p"].startswith("<") else stars(float(row["Raw p"]))
        text = f"R@50：未校正 p = {row['Raw p']}（{raw}）、BH p = {row['BH-adjusted p']}（{row['Sig. (BH)']}）"
        return raw == "***" and row["Sig. (BH)"] == "**", text, (path,), (P_FAIR,)

    def check_cp_wo_decrease_smaller(self):
        holds, parts = True, []
        for k in FAIR_CP:
            dw, do, ds = (self.fair_delta(f, k) for f in FACTORS)
            holds &= dw < 0 and do < 0 and abs(dw) < abs(ds) and abs(do) < abs(ds)
            parts.append(f"{FAIR_LABEL[k]}：ΔW {signed(dw)}、ΔO {signed(do)}、ΔS {signed(ds)}")
        return holds, "；".join(parts), *self._fair_sources()

    def _style_largest(self, keys: tuple[str, ...]) -> tuple[bool, str]:
        holds = True
        for k in keys:
            dw, do, ds = (self.fair_delta(f, k) for f in FACTORS)
            holds &= ds < 0 and abs(ds) > max(abs(dw), abs(do))
        text = "ΔS：" + "、".join(f"{FAIR_LABEL[k]} {signed(self.fair_delta('no_style', k))}" for k in keys)
        return holds, text

    def check_or_style_largest(self):
        holds, text = self._style_largest(FAIR_OR)
        return holds, text + "（皆為三因子中最大下降）" if holds else text, *self._fair_sources()

    def check_style_largest_all(self):
        holds, text = self._style_largest(FAIR_CP + FAIR_OR)
        return holds, text + "（CP 與 OR 皆為三因子中最大下降）" if holds else text, *self._fair_sources()

    def check_or_wo_decrease(self):
        holds = all(self.fair_delta(f, k) < 0 for f in ("no_weather", "no_occasion") for k in FAIR_OR)
        text = (f"No-W：{self._deltas_text('no_weather', FAIR_OR)}；"
                f"No-O：{self._deltas_text('no_occasion', FAIR_OR)}（正值表示移除後 Recall 反而上升）")
        return holds, text, *self._fair_sources()

    def check_or_wo_not_stable_positive(self):
        holds, text, sources, programs = self.check_or_wo_decrease()
        return not holds, text, sources, programs

    def check_or_style_drop_increases_with_k(self):
        ds = [self.fair_delta("no_style", k) for k in FAIR_OR]
        holds = ds[0] > ds[1] > ds[2]
        return holds, "ΔS：" + " → ".join(signed(d) for d in ds), *self._fair_sources()

    def check_all_removals_decrease(self):
        keys = FAIR_CP + FAIR_OR
        rising = [f"{FACTOR_LABEL[f]} {FAIR_LABEL[k]} {signed(self.fair_delta(f, k))}"
                  for f in FACTORS for k in keys if self.fair_delta(f, k) >= 0]
        text = (f"{len(FACTORS) * len(keys) - len(rising)}／{len(FACTORS) * len(keys)} 個 Δ 為負"
                + (f"；移除後上升：{'、'.join(rising)}" if rising else ""))
        return not rising, text, *self._fair_sources()

    def check_ratio_all_defined(self):
        parts, holds = [], True
        for row in self.t07:
            task, metric = row["Task"], row["Metric"]
            ratios = [row[f"{factor}_ratio"] for factor in ("style", "weather", "occasion")]
            if all(ratios):
                style, weather, occasion = (100 * float(r) for r in ratios)
                parts.append(f"{task} {metric}：風格 {style:.0f}%、天氣 {weather:.0f}%、場合 {occasion:.0f}%")
            else:
                holds = False
                parts.append(f"{task} {metric}：無法定義（天氣或場合移除後效能上升）")
        return holds, "；".join(parts), (self.f_t07,), (P_FAIR,)

    def check_r1_positive_ns(self):
        delta, p = self.stat("or_r1", "delta_mean"), self.bh("or_r1")
        return delta > 0 and p >= 0.05, f"R@1 Δ {signed(delta)}、BH p = {p:.3g}", *self._main_sources()

    def check_r3_sig(self):
        p = self.bh("or_r3")
        return p < 0.05, f"R@3 BH p = {p:.3g}（{stars(p)}）", *self._main_sources()

    def check_or_wo_small_seed_sensitive(self):
        by_seed = {(r["variant"], r["seed"]): r for r in self.fair_seeds}
        seeds = sorted({r["seed"] for r in self.fair_seeds if r["variant"] == "context"}, key=int)
        smaller = all(max(abs(self.fair_delta("no_weather", k)), abs(self.fair_delta("no_occasion", k)))
                      < abs(self.fair_delta("no_style", k)) for k in FAIR_OR)
        mixed = []
        for factor in ("no_weather", "no_occasion"):
            for k in FAIR_OR:
                diffs = [float(by_seed[(factor, s)][k]) - float(by_seed[("context", s)][k]) for s in seeds]
                up, down = sum(d > 0 for d in diffs), sum(d < 0 for d in diffs)
                if up and down:
                    mixed.append(f"{FACTOR_LABEL[factor]} {FAIR_LABEL[k]}（{up} 升／{down} 降）")
        text = (("|ΔW|、|ΔO| 皆小於 |ΔS|" if smaller else "|ΔW| 或 |ΔO| 不小於 |ΔS|")
                + f"；逐 seed 方向不一致：{'、'.join(mixed) if mixed else '無'}")
        return smaller and bool(mixed), text, (self.f_fair, self.f_fair_seeds), (P_FAIR,)

    def check_stats_method(self):
        test = self.evidence.get("test", "")
        ci = self.evidence.get("confidence_interval", "")
        family = self.evidence.get("bh_family", [])
        holds = "paired t-test" in test and "t confidence interval" in ci and set(family) == set(MAIN_METRICS)
        text = (f"{test}；{ci}；BH family = {len(family)} 項主要指標；α = {self.evidence.get('alpha')}；"
                f"Cohen's dz 由 main_paired_bh_8metrics.csv 的 cohen_dz 欄輸出")
        return holds, text, (self.f_evidence, self.f_stats), (P_STATS,)

    def check_arch_fashionclip(self):
        image = self.cp_config["features"]["image"]["source"]
        cls = self.cp_config["model"]["class"]
        return image == "FashionCLIP", f"{cls}（Hybrid Attention）；影像特徵 {image}", (CP_CONFIG,), (P_TRAIN, P_BASE_CONFIG)

    def check_cir_init_from_cp(self):
        source = self.or_config["initialization"]["source"]
        text = f"{source}：以同 seed 的 CP best checkpoint（validation FITB 選出）初始化"
        return source == "matching_CP_checkpoint", text, (OR_CONFIG,), (P_TRAIN,)

    def check_expected_results_cites_run(self):
        cited = self.run_id in EXPECTED_RESULTS.read_text(encoding="utf-8")
        text = (f"已引用 {self.run_id}" if cited
                else f"目前只列論文／歷史數值，未引用正式 run {self.run_id}")
        return cited, text, (EXPECTED_RESULTS,), ()

    def check_main_within_tol(self):
        diffs = [abs(float(r["absolute_difference"])) for (m, c), r in self.main.items()
                 if c in ("original", "context")]
        within = sum(d <= 0.005 for d in diffs)
        text = f"{within}／{len(diffs)} 個平均值在 0.005 內（最大差 {max(diffs):.4f}）"
        return within == len(diffs) == 16, text, (self.f_main,), (P_MAIN,)

    def check_readme_main_claims(self):
        positive, _, _, _ = self.check_main_all_positive()
        seven, sig_text, _, _ = self.check_main_sig_count7()
        r1_ns, r1_text, _, _ = self.check_r1_positive_ns()
        holds = positive and seven and r1_ns
        text = f"8 項平均差{'皆為正' if positive else '並非皆為正'}；{sig_text}；{r1_text}"
        return holds, text, *self._main_sources()


# ---- rule evaluation ----------------------------------------------------
def evaluate(rule: str, manuscript: str, res: Resolved) -> tuple[str, str, str]:
    """Return (status, latest display text, extra note)."""
    if rule in ("num", "delta", "ci"):
        places = decimals(manuscript)
        is_signed = manuscript.strip()[:1] in "+-−"
        latest = fmt(float(res.value), places, signed=is_signed)
        if rule != "num" and sign(float(res.value)) != sign(parse_number(manuscript)):
            return "REWRITE", latest, "方向改變" if rule == "delta" else "信賴區間是否包含 0 改變"
        same = latest == manuscript.replace("−", "-").strip()
        return ("KEEP" if same else "UPDATE_NUMBER"), latest, ""
    if rule == "stars":
        latest_label = res.value
        if (manuscript == "n.s.") != (latest_label == "n.s."):
            return "REWRITE", res.text, "顯著性結論改變"
        return ("KEEP" if latest_label == manuscript else "UPDATE_NUMBER"), res.text, ""
    if rule == "pct":
        places = decimals(manuscript)
        latest = f"{100 * float(res.value):.{places}f}%"
        return ("KEEP" if latest == manuscript.strip() else "UPDATE_NUMBER"), latest, ""
    if rule == "range":
        numbers = re.findall(r"-?\d+(?:\.\d+)?", manuscript)
        places = decimals(numbers[0])
        scale, suffix = (100, "%") if "%" in manuscript else (1, "")
        lo, hi = (scale * v for v in res.value)
        latest = f"{lo:.{places}f}{suffix}–{hi:.{places}f}{suffix}"
        stated = f"{float(numbers[0]):.{places}f}{suffix}–{float(numbers[1]):.{places}f}{suffix}"
        return ("KEEP" if latest == stated else "UPDATE_NUMBER"), latest, ""
    if rule == "count":
        latest = f"{int(res.value):,}"
        return ("KEEP" if int(res.value) == int(parse_number(manuscript)) else "REWRITE"), latest, ""
    if rule == "num_exact":
        same = math.isclose(float(res.value), parse_number(manuscript), rel_tol=1e-12)
        return ("KEEP" if same else "REWRITE"), f"{float(res.value):g}", ""
    if rule == "text":
        def norm(text: str) -> str:
            return re.sub(r"\s+", " ", text).strip().casefold()
        return ("KEEP" if norm(res.text) == norm(manuscript) else "REWRITE"), res.text, ""
    if rule == "check":
        return ("KEEP" if res.holds else "REWRITE"), res.text, ""
    if rule == "add":
        return "ADD", res.text, ""
    if rule == "external":
        if isinstance(res.value, tuple):
            numbers = re.findall(r"[-−+]?\d+(?:\.\d+)?", manuscript)
            places = [decimals(n) for n in numbers]
            o_mean, o_sd, c_mean, c_sd, diff = res.value
            latest = (f"{o_mean:.{places[0]}f} ± {o_sd:.{places[1]}f} | "
                      f"{c_mean:.{places[2]}f} ± {c_sd:.{places[3]}f} | {signed(diff, places[4])}")
            stated = (f"{parse_number(numbers[0]):.{places[0]}f} ± {parse_number(numbers[1]):.{places[1]}f} | "
                      f"{parse_number(numbers[2]):.{places[2]}f} ± {parse_number(numbers[3]):.{places[3]}f} | "
                      f"{signed(parse_number(numbers[4]), places[4])}")
            note = "保存輸出重算值與稿件一致" if latest == stated else "保存輸出重算值與稿件不一致"
            return "OUTSIDE_FORMAL_RUN", latest, note
        return "OUTSIDE_FORMAL_RUN", res.text, ""
    if rule == "overview":
        result = str(res.value)
        if "differ" in result:
            return "UPDATE_NUMBER", res.text, ""
        if "match" in result or "exact" in result:
            return "KEEP", res.text, ""
        return "OUTSIDE_FORMAL_RUN", res.text, ""
    fail(f"unknown rule: {rule}")


def build_rows(latest: Latest) -> list[dict[str, str]]:
    manuscript_rows = read_csv(VALUES_CSV)
    ids = [row["id"] for row in manuscript_rows]
    if len(ids) != len(set(ids)):
        fail("duplicate id in manuscript_values.csv")
    out = []
    for row in manuscript_rows:
        res = latest.resolve(row["latest_key"])
        status, shown, extra = evaluate(row["rule"], row["manuscript_value"], res)
        note = "；".join(part for part in (extra, row["note"]) if part)
        out.append({
            "id": row["id"],
            "group": row["group"],
            "thesis_ref": row["thesis_ref"],
            "tors_ref": row["tors_ref"],
            "item": row["item"],
            "manuscript_value": row["manuscript_value"],
            "latest_value": shown,
            "status": status,
            "source_files": "; ".join(dict.fromkeys(rel(p) for p in res.sources)),
            "programs": "; ".join(dict.fromkeys(res.programs)),
            "run_id": latest.run_id,
            "note": note,
            "transcription_note": row["note"],
        })
    return out


# ---- markdown -----------------------------------------------------------
def md_escape(text: str) -> str:
    return text.replace("|", "\\|").replace("*", "\\*").replace("\n", " ")


def short_path(path: str, run_prefix: str) -> str:
    if path.startswith(run_prefix):
        return "<RUN>/" + path[len(run_prefix):].lstrip("/")
    return path.removeprefix("reproduction/")


def source_cell(row: dict[str, str], run_prefix: str) -> str:
    files = [f"`{short_path(p, run_prefix)}`" for p in row["source_files"].split("; ") if p]
    programs = [f"`{Path(p).name}`" for p in row["programs"].split("; ")
                if p and not p.endswith(Path(P_SELF).name)]
    text = "、".join(files) if files else "—"
    return text + (f"（{'、'.join(programs)}）" if programs else "")


def cell_mark(row: dict[str, str]) -> str:
    """Markdown for one table cell; the text is escaped before bold markup is added."""
    stated = md_escape(row["manuscript_value"])
    latest = md_escape(row["latest_value"].split("（")[0])
    status = row["status"]
    if status == "KEEP":
        return f"{stated} ✓"
    if status == "REWRITE":
        return f"**{stated} → {latest} ⚠**"
    return f"{stated} → {latest}"


def render_table_group(rows: list[dict[str, str]], run_prefix: str) -> list[str]:
    lines = ["格式：`稿件 → 最新`；✓ 四捨五入後相同；⚠ 方向、CI 是否含 0 或顯著性改變。", ""]
    tables: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        tables.setdefault(row["id"].split(".")[0].removeprefix("T"), []).append(row)
    for table, table_rows in tables.items():
        lines.append(f"### 表 {table}（TORS {TABLE_TORS.get(table, '—')}）")
        lines.append("")
        fields: list[str] = []
        grid: dict[str, dict[str, dict[str, str]]] = {}
        for row in table_rows:
            metric, field = row["item"].split("｜", 1)
            if field not in fields:
                fields.append(field)
            grid.setdefault(metric, {})[field] = row
        lines.append("| 指標 | " + " | ".join(fields) + " |")
        lines.append("|---|" + "---|" * len(fields))
        for metric, cells in grid.items():
            lines.append(f"| {metric} | " + " | ".join(
                cell_mark(cells[f]) if f in cells else "" for f in fields) + " |")
        lines.append("")
        sources = dict.fromkeys(source_cell(r, run_prefix) for r in table_rows)
        lines.append("來源：" + "；".join(sources))
        notes: dict[str, list[str]] = {}
        for r in table_rows:
            if r["transcription_note"]:
                notes.setdefault(r["transcription_note"], []).append(r["item"])
        for text, items in notes.items():
            lines.append(f"- {md_escape(text)}（{md_escape('、'.join(items))}）")
        lines.append("")
    return lines


def render_row_table(rows: list[dict[str, str]], run_prefix: str) -> list[str]:
    lines = ["| ID | 論文 | TORS | 項目 | 稿件 | 最新 | 狀態 | 來源（程式） |",
             "|---|---|---|---|---|---|---|---|"]
    for row in rows:
        latest = row["latest_value"] + (f"（{row['note']}）" if row["note"] else "")
        lines.append("| " + " | ".join(md_escape(v) for v in (
            row["id"], row["thesis_ref"], row["tors_ref"], row["item"], row["manuscript_value"],
            latest, STATUS_LABEL[row["status"]], source_cell(row, run_prefix))) + " |")
    lines.append("")
    return lines


def render_markdown(rows: list[dict[str, str]], latest: Latest, commit: str, equivalence: str | None) -> str:
    run_prefix = rel(latest.paths["root"])
    lines = [
        "# 論文／TORS 稿結果對帳表",
        "",
        "> 本檔由 `reproduction/scripts/build_manuscript_reconciliation.py` 產生，請勿手動修改。"
        "稿件數值轉錄於同目錄的 `manuscript_values.csv`；逐列完整資料（含來源 CSV、程式與 run ID）見 `reconciliation.csv`。",
        "",
        "| 項目 | 內容 |",
        "|---|---|",
        f"| 對帳 run | `{latest.run_id}`（commit `{commit}`） |",
    ]
    if equivalence:
        lines.append(f"| 等價 run | {equivalence} |")
    lines += [
        f"| 稿件 | 碩論終稿 `{THESIS_PDF}`；TORS 投稿稿 `{TORS_PDF}`（兩者結果數值相同，頁碼為印刷頁碼） |",
        f"| `<RUN>` | `{run_prefix}` |",
        "",
        "## 狀態說明",
        "",
        "| 狀態 | 意義 | 筆數 |",
        "|---|---|---:|",
    ]
    counts = {status: sum(r["status"] == status for r in rows) for status in STATUS_LABEL}
    for status, label in STATUS_LABEL.items():
        lines.append(f"| {label}（`{status}`） | {STATUS_MEANING[status]} | {counts[status]} |")
    lines += ["", "## 修稿必須改寫的地方", ""]
    rewrite = [r for r in rows if r["status"] == "REWRITE"]
    lines += ["| ID | 位置（論文／TORS） | 稿件 | 最新結果 |", "|---|---|---|---|"]
    for row in rewrite:
        where = "／".join(v for v in (row["thesis_ref"], row["tors_ref"]) if v and v != "—")
        stated = f"{row['item']}：{row['manuscript_value']}" if row["group"] == "tables" else row["manuscript_value"]
        latest_text = row["latest_value"] + (f"（{row['note']}）" if row["note"] else "")
        lines.append("| " + " | ".join(md_escape(v) for v in (row["id"], where or "—", stated, latest_text)) + " |")
    lines.append("")
    lines += ["## 需補寫", ""]
    for row in (r for r in rows if r["status"] == "ADD"):
        lines.append(f"- `{row['id']}` {md_escape(row['item'])}：{md_escape(row['latest_value'])}")
    lines.append("")
    for group, title in GROUPS:
        subset = [r for r in rows if r["group"] == group]
        if not subset:
            continue
        lines += [f"## {title}", ""]
        if group == "tables":
            lines += render_table_group(subset, run_prefix)
        else:
            lines += render_row_table(subset, run_prefix)
    return "\n".join(lines).rstrip() + "\n"


def committed_layout(run_id: str) -> tuple[str, dict[str, Path], Path]:
    """Layout of a run packaged under reproduction/results/."""
    summary = REPRO / "results" / "summary" / run_id
    if not summary.is_dir():
        fail(f"no committed results for {run_id}")
    return run_id, {
        "root": summary,
        "main": summary / "main",
        "stats": summary / "statistics",
        "ablation": summary / "ablation",
        "secondary": summary / "secondary",
        "chapter4": summary / "chapter4",
        "environment": REPRO / "environment" / run_id / "environment_validation.json",
    }, REPRO / "results" / "raw" / run_id / "run_identity" / "git_commit.txt"


def load_layout(args: argparse.Namespace) -> tuple[str, dict[str, Path], Path]:
    if args.official:
        return committed_layout(read_json(OFFICIAL_MANIFEST)["official_run_id"])
    if args.reference:
        return committed_layout(REF_ID)
    run = args.run_root.expanduser().resolve()
    if not run.is_dir():
        fail(f"run directory missing: {run}")
    if (run / "RUN_STATUS.txt").read_text(encoding="utf-8").strip() != "PASSED":
        fail(f"run is not PASSED: {run}")
    return run.name, {
        "root": run,
        "main": run / "main" / "summary",
        "stats": run / "statistics",
        "ablation": run / "ablation" / "summary",
        "secondary": run / "secondary" / "results",
        "chapter4": run / "chapter4",
        "environment": run / "environment" / "environment_validation.json",
    }, run / "git_commit.txt"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--official", action="store_true",
                       help="Reconcile against the official run named in results/final_reference_manifest.json.")
    group.add_argument("--reference", action="store_true",
                       help=f"Reconcile against the committed {REF_ID} evidence.")
    group.add_argument("--run-root", type=Path,
                       help="Reconcile against one completed integrated full run.")
    parser.add_argument("--out-dir", type=Path, default=RECON_DIR,
                        help="Output directory (default: reproduction/docs/manuscript_reconciliation).")
    args = parser.parse_args()

    run_id, paths, commit_file = load_layout(args)
    latest = Latest(run_id, paths)
    commit = commit_file.read_text(encoding="utf-8").strip() if commit_file.is_file() else "not_available"
    rows = build_rows(latest)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = args.out_dir / "reconciliation.csv"
    with out_csv.open("w", encoding="utf-8", newline="") as stream:
        fields = [f for f in rows[0] if f != "transcription_note"]
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    out_md = args.out_dir / "reconciliation.md"
    equivalence = (OFFICIAL_EQUIVALENCE if args.official else REFERENCE_EQUIVALENCE if args.reference else None)
    out_md.write_text(render_markdown(rows, latest, commit, equivalence), encoding="utf-8")

    counts = {s: sum(r["status"] == s for r in rows) for s in STATUS_LABEL}
    print(f"[RECONCILIATION] run={run_id} rows={len(rows)} "
          + " ".join(f"{s}={n}" for s, n in counts.items()))
    print(f"[RECONCILIATION] wrote {rel(out_csv)}")
    print(f"[RECONCILIATION] wrote {rel(out_md)}")


if __name__ == "__main__":
    main()
