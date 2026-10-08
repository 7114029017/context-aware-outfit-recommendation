#!/usr/bin/env python3
"""Item-by-item status of the 46 reproduction items (data, checks, training, statistics, other experiments,
qualitative analyses) for one completed full run or for the official run included in the repository.

Every item is checked against the outputs that the reproduction writes: the 35 training units and their
statistics, the supplementary analyses (reproduction/scripts/supplementary/) and the extension analyses
(reproduction/scripts/extensions/), each with the check that shows it is correct: equal to the archived
2025 table or the thesis, equal to the run's own tables, or, where the retrained models give other values,
the check of the ported code on the preserved 2025 models and data (extensions/run_all.sh --validate). The
states:

- PASS: recomputed and equal to the 2025 values of the senior student's project (the archived outputs, the
  thesis or the original manuscript);
- DIFFERS: recomputed, with the computation checked, but the values differ from the 2025 values because the
  models were retrained or because tied scores are ordered differently; the evidence gives the reason and the
  check (the revised manuscript already reports the official run's Tables 6 and 7);
- REUSED: input data that cannot be regenerated (LLM outputs, the feature files) and is used as preserved,
  with its SHA-256 checked;
- NOT_REPRODUCIBLE: no record of the 2025 step exists;
- SKIPPED: an optional download or the GPU was missing, so the step did not run;
- MISSING / FAIL: an expected output is missing, or a check of the computation does not hold (a manuscript
  statement that the retrained models do not repeat is reported in the evidence, not as FAIL).

Usage:
  report_items.py --official                       the committed results of the official run
  report_items.py --run-root RUN [--supplementary-dir DIR] [--extensions-dir DIR]

Writes ITEMS_STATUS.md and ITEMS_STATUS.json to --out-dir (default: reproduction/results/ for --official,
the run folder otherwise) and prints a summary. Exit status 1 if an item is MISSING or FAIL.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import build_manuscript_reconciliation as recon  # noqa: E402

REPO = HERE.parents[1]
REPRO = REPO / "reproduction"
D01 = REPO / "01_資料建構_data_construction"
GENERATED = D01 / "generated_descriptions" / "01_生成結果_generation_results" / "new_polyvore_outfit_titles.json"
WOS = D01 / "generated_descriptions" / "02_三因子拆分_wos_factor_split" / "wos_split_results_v5_merged_retry_round3.jsonl"
SUPP_NAMES = ("input_data_audit", "dataset_tables_check", "counterfactual_pairs_check", "paper_value_checks",
              "judge_audit_checks", "met_reference_check", "checklist_coverage")
TABLE1 = {"train outfits": "16995", "validation outfits": "3000", "test outfits": "15145", "all outfits": "35140",
          "CP test pairs": "30290", "FITB questions": "15145", "CIR evaluable queries": "9311",
          "generated descriptions": "35140", "CIR excluded queries (no 3,000-item pool)": "5834",
          "original text records": "68306"}  # manuscript Table 1
TABLE2 = {"original": ("34", "0.10", "4.82", "5.00", "25", "4.90", "21.88"),
          "generated": ("0", "0.00", "10.08", "10.00", "27", "11.60", "0.00")}  # manuscript Table 2
TABLE2_COLUMNS = ("missing", "missing_percent", "mean_words", "median_words", "max_words", "mean_tokens",
                  "at_most_3_words_percent")
TABLE5_PROXY = {"CLO": ("0.93", "0.63", "1.21", "0.25", "2.07", "5.20"), "MET": ("1.00", "1.00", "1.30", "1.00", "1.50"),
                "temperature reference (°C)": ("22.9", "19.9", "24.5", "13.5", "27.1")}  # Table 5: median, Q1, Q3, P5, P95, max
TABLE5_COUNTS = "CLO > 4: 20; MET = 1.0: 25786; MET > 10: 8; temperature < 0 °C: 257; temperature < -20 °C: 51"
TABLE5_PROMPT = {("Qwen3-VL", "mean_abs_diff"): "0.0048", ("Gemma-3", "mean_abs_diff"): "0.0461",
                 ("Qwen3-VL", "within_0_05_rate"): "0.973", ("Gemma-3", "within_0_10_rate"): "0.893"}  # P0-R2
TABLE5_AUDIT = {"qwen": ("0.7242", "0.6667", "0.2273"), "gemma": ("0.6818", "0.6566", "0.2212")}  # human, model, MAE
TABLE8_MANUSCRIPT = {"cp_test_auc": "+0.0378", "cp_test_fitb_acc": "-0.0077", "recall@10": "+0.0102",
                     "recall@30": "+0.0153", "recall@50": "+0.0170", "mean_rank": "-28.89", "median_rank": "-31.4"}
T19_ARCHIVED = (REPO / "03_實驗與結果_experiments_results" / "00_控制檢查與附加稽核" / "圖表_figures_tables" / "tables" /
                "T19_judge_qwen_gemma_agreement_summary.csv")
CASES_MANUSCRIPT = {"purse": "23.4 -> 1.8", "dress": "122.6 -> 3.0, No-Style 60.2",
                    "sunglasses": "22.4 -> 505.0"}  # manuscript Section 5.5, mean target ranks
T12_ARCHIVED = (REPO / "03_實驗與結果_experiments_results" / "04_情境子集與三因子分析" / "圖表_figures_tables" / "tables" /
                "T12_subset_robustness_summary.csv")
P02_PICKS = {"keep0": ["56998165", "121507083", "161194648"], "keep1": ["82253359", "112382686"],
             "keep2": ["112382686"]}  # P02 cell 2 output (2025 cir_new_seed1), as in extensions/outfit_generation.py
STATES =("PASS", "DIFFERS", "REUSED", "NOT_REPRODUCIBLE", "SKIPPED", "MISSING", "FAIL")


def text(path: Path) -> str:
    return path.read_text(encoding="utf-8") if path.is_file() else ""


def rows(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def found(path: Path, pattern: str) -> bool:
    return re.search(pattern, text(path)) is not None


class Layout:
    """Where the outputs of a run folder or of the committed official run are."""

    def __init__(self, args: argparse.Namespace):
        self.official = args.official
        if args.official:
            self.run_id, self.paths, _ = recon.load_layout(argparse.Namespace(official=True, reference=False,
                                                                              run_root=None))
            raw = REPRO / "results" / "raw" / self.run_id
            self.root = None
            self.supp = REPRO / "results" / "supplementary"
            self.run_analyses = self.supp / self.run_id
            self.ext = REPRO / "results" / "extensions" / self.run_id
            self.fair_audit = raw / "fair_subset" / "fair_subset_cir_scope_audit.json"
            self.fair_manifest = raw / "fair_subset" / "fair_subset_reconstruction_manifest.json"
            self.split_sums = REPRO / "splits" / "SHA256SUMS.txt"
            self.features = REPRO / "environment" / self.run_id / "feature_verification.json"
            units = rows(REPRO / "results" / "summary" / self.run_id / "seed_index.csv")
            self.units = [(u["family"], u["status"]) for u in units]
            self.comparison = None
        else:
            run = args.run_root.expanduser().resolve()
            self.run_id, self.paths, _ = recon.load_layout(argparse.Namespace(official=False, reference=False,
                                                                              run_root=run))
            self.root = run
            self.supp = (args.supplementary_dir or run / "supplementary").expanduser().resolve()
            self.run_analyses = self.supp / "run_analyses"
            self.ext = (args.extensions_dir or run / "extensions").expanduser().resolve()
            self.fair_audit = run / "ablation" / "fair_subset_cir_scope_audit.json"
            self.fair_manifest = run / "ablation" / "fair_subset_reconstruction" / "fair_subset_reconstruction_manifest.json"
            self.split_sums = run / "frozen_splits" / "SHA256SUMS.txt"
            self.features = run / "environment" / "feature_verification.json"
            self.units = []
            for manifest in sorted((run / "main").glob("*/full_train_manifest.json")):
                self.units.append(("main", json.loads(text(manifest)).get("status", "")))
            for manifest in sorted((run / "ablation" / "runs").glob("*/fair_subset_run_manifest.json")):
                self.units.append(("ablation", json.loads(text(manifest)).get("status", "")))
            self.comparison = (args.out_dir or run).expanduser().resolve() / "official_comparison.json"

    def s(self, name: str) -> Path:
        return self.supp / name

    def where(self, path: Path) -> str:
        for base, label in ((self.root, "<run>"), (REPO, None)):
            if base is not None and path.is_relative_to(base):
                rel = path.relative_to(base)
                return f"{label}/{rel}" if label else str(rel)
        return str(path)


def extension_status(lay: Layout) -> dict[str, str]:
    out = {}
    for line in text(lay.ext / "EXTENSIONS_STATUS.txt").splitlines():
        if ":" in line:
            step, state = line.split(":", 1)
            out[step.strip()] = state.strip()
    return out


def result(state: str, evidence: str, *outputs: Path) -> tuple[str, str, list[Path]]:
    return state, evidence, list(outputs)


def archived_unchanged(path: Path) -> tuple[int, int]:
    """Files under path (a file or folder of the archived folders 01-03) whose SHA-256 equals
    reproduction/environment/archived_sources_manifest.json (verify_archived_sources.py), and how many it lists."""
    manifest = json.loads(text(REPRO / "environment" / "archived_sources_manifest.json") or "{}")
    listed = [REPO / f["path"] for f in manifest.get("files", [])]
    sha = {REPO / f["path"]: f["sha256"] for f in manifest.get("files", [])}
    listed = [q for q in listed if q == path or q.is_relative_to(path)]
    same = sum(1 for q in listed if q.is_file() and hashlib.sha256(q.read_bytes()).hexdigest() == sha[q])
    return same, len(listed)


def need(*paths: Path) -> Path | None:
    """The first missing path, or None."""
    return next((p for p in paths if not p.exists()), None)


def ext_step(lay: Layout, ext: dict, step: str, *paths: Path):
    """SKIPPED/FAILED from the extension status, MISSING for absent outputs, else None."""
    state = ext.get(step, "")
    if state.startswith("SKIPPED"):
        return result("SKIPPED", f"{step}: {state}")
    if state.startswith("FAILED"):
        return result("FAIL", f"{step}: {state}")
    missing = need(*paths)
    if missing:
        return result("MISSING", f"{lay.where(missing)} not found" + ("" if state else f" (step {step} not run)"))
    return None


def port_checks(ext: dict, steps: tuple[str, ...]) -> tuple[bool | None, str]:
    """The checks of the ported code against the 2025 outputs behind an item: False if one FAILED, True if all
    PASSED, None if one was not run (no --validate, or SKIPPED)."""
    states = {s: (ext.get(s) or "not run").split(" (")[0] for s in steps}
    note = "; port check " + ", ".join(f"{s} {v}" for s, v in states.items())
    if any(v.startswith("FAILED") for v in states.values()):
        return False, note
    return (True if all(v == "PASSED" for v in states.values()) else None), note


# ---------------------------------------------------------------- the 46 items
def items(lay: Layout) -> list[dict]:
    ext = extension_status(lay)
    ra = lay.run_analyses
    out = []

    def add(num, name, name_zh, ref, how, check, ports=()):
        """ports: the checks of the ported code that show the item's computation is right; a FAILED one makes
        the item FAIL."""
        state, evidence, outputs = check()
        if ports and state not in ("SKIPPED", "MISSING"):
            ok, note = port_checks(ext, ports)
            evidence += note
            state = "FAIL" if ok is False else state
        out.append({"id": num, "item": name, "item_zh": name_zh, "reference": ref, "how": how, "state": state,
                    "evidence": evidence, "outputs": [lay.where(p) for p in outputs]})

    # 一、資料準備
    def c1():
        if need(lay.split_sums):
            return result("MISSING", f"{lay.where(lay.split_sums)} not found")
        official = {l.split()[1].split("/")[-1]: l.split()[0] for l in text(REPRO / "splits" / "SHA256SUMS.txt").split("\n") if l.strip()}
        mine = {l.split()[1].split("/")[-1]: l.split()[0] for l in text(lay.split_sums).split("\n") if l.strip()}
        same = sum(1 for k, v in official.items() if mine.get(k) == v)
        local = REPRO / ".local" / "polyvore_root.txt"
        images = Path(text(local).strip()) / "images" / ".complete" if local.is_file() else None
        img = "images downloaded (local only)" if images and images.is_file() else "images not downloaded"
        state = "PASS" if same == len(official) and official else "FAIL"
        return result(state, f"split files: {same} of {len(official)} SHA-256 equal to the committed manifests; {img}",
                      lay.split_sums)
    add(1, "Polyvore dataset", "Polyvore 資料集", "full text", "public data, downloaded", c1)

    def c2():
        n = len(json.loads(text(GENERATED))) if GENERATED.is_file() else 0
        same, listed = archived_unchanged(GENERATED)
        return result("REUSED" if listed and same == listed else "FAIL",
                      f"{n:,} generated descriptions (LLM output, used as preserved); SHA-256 equal to the archived "
                      f"sources manifest: {same} of {listed} file", GENERATED)
    add(2, "Context-aware descriptions", "情境描述 35,140 筆（LLM 生成）", "Methods, Table 2", "input data", c2)

    def c3():
        folder = D01 / "clo_met_temperature" / "temperature_results"
        same, listed = archived_unchanged(folder)
        return result("REUSED" if listed and same == listed else "FAIL",
                      f"LLM estimates of CLO and MET, used as preserved; SHA-256 equal to the archived sources manifest: "
                      f"{same} of {listed} files", folder)
    add(3, "CLO / MET estimates", "CLO／MET 估計（LLM 估計）", "Table 5", "input data", c3)

    def c4():
        clo = {r["statistic"]: r["value"] for r in rows(lay.s("input_data_audit") / "clo_distribution_summary.csv")}
        outliers = next((v for k, v in clo.items() if "outlier" in k and "count" not in k), clo.get("outliers", "?"))
        return result("NOT_REPRODUCIBLE", f"no record of the re-inference; {outliers} values above the IQR bound remain",
                      lay.s("input_data_audit") / "clo_distribution_summary.csv")
    add(4, "CLO outlier re-inference", "CLO 離群值重新推論（稿件 3.2.1 節）", "manuscript 3.2.1", "not reproducible", c4)

    def c5():
        f = lay.s("met_reference_check") / "summary.md"
        if not f.is_file():
            return result("SKIPPED", "needs bootstrap_data.sh --with-compendium")
        ok = found(f, r"457 entries\. Same activity codes in the same order as the preserved list: yes")
        return result("PASS" if ok else "FAIL", "457 entries rebuilt from the official Compendium, same codes and order"
                      if ok else "rebuilt list differs", f)
    add(5, "MET candidate list (457)", "MET 候選清單 457 筆（稿件 3.2.2 節）", "manuscript 3.2.2; thesis Tables 3-1, 3-2",
        "public data, rebuilt by rule", c5)

    def c6():
        f = lay.s("input_data_audit") / "summary.md"
        ok = found(f, r"Equation \(2\) reproduces the stored temperature reference for 35140 of 35140 outfits")
        return result("PASS" if ok else ("MISSING" if not f.is_file() else "FAIL"),
                      "equation (2) gives the stored reference for 35,140 of 35,140 outfits", f)
    add(6, "Temperature reference (eq. 2)", "溫度參考值（式 2）", "eq. 2, Table 5", "recomputed from input data", c6)

    def c7():
        n = sum(1 for line in text(WOS).splitlines() if line.strip())
        same, listed = archived_unchanged(WOS)
        return result("REUSED" if listed and same == listed else "FAIL",
                      f"{n:,} annotated descriptions (the split program and prompt were not preserved); SHA-256 equal "
                      f"to the archived sources manifest: {same} of {listed} file", WOS)
    add(7, "Weather / occasion / style annotation", "天氣／場合／風格標註（表 3）", "Table 3", "input data", c7)

    def c8():
        feat = json.loads(text(lay.features)) if lay.features.is_file() else {}
        enc = rows(lay.ext / "counterfactual" / "encoder_check.csv")
        cos = min((float(r["cosine_stored_vs_live"]) for r in enc), default=None)
        ok = feat.get("all_features_ok") and feat.get("n_ok") == 8
        ev = (f"used as preserved (the extraction program and model versions were not kept); {feat.get('n_ok', 0)} "
              "of 8 feature files verified by SHA-256")
        ev += f"; FashionCLIP re-encodes 24 stored descriptions (min cosine {cos:.6f})" if cos is not None else ""
        return result("REUSED" if ok else "FAIL", ev, lay.features)
    add(8, "Eight feature files", "8 個特徵檔", "Methods", "input data, verified", c8)

    def c9():
        audit = json.loads(text(lay.fair_audit)) if lay.fair_audit.is_file() else {}
        manifest = json.loads(text(lay.fair_manifest)) if lay.fair_manifest.is_file() else {}
        ok = manifest.get("candidate_unique_ids") == 21903 and audit.get("expected_cir_evaluable_pairs_archived") == 3432 \
            and audit.get("count_matches_archived")
        return result("PASS" if ok else "FAIL", f"{manifest.get('candidate_unique_ids', 0):,} outfits, "
                      f"{audit.get('expected_cir_evaluable_pairs_archived', 0):,} OR queries (counts as in the thesis)",
                      lay.fair_manifest, lay.fair_audit)
    add(9, "Fair subset (21,903 outfits)", "公平子集 21,903 套", "Table 7", "rebuilt by rule", c9)

    def c10():
        f = lay.s("counterfactual_pairs_check") / "summary.md"
        ok = found(f, r"counterfactual description\): 24/24")
        return result("PASS" if ok else ("MISSING" if not f.is_file() else "FAIL"),
                      "24 of 24 pairs regenerated identically (P16 sampling)", f)
    add(10, "Counterfactual pairs (24)", "反事實配對 24 組（表 4）", "Table 4", "regenerated by rule", c10)

    # 二、資料檢查
    def c11():
        f = lay.s("paper_value_checks") / "table1_dataset_scope.csv"
        got = {r["quantity"]: r["value"] for r in rows(f)}
        same = sum(1 for k, v in TABLE1.items() if got.get(k) == v)
        return result("PASS" if same == len(TABLE1) else ("MISSING" if not got else "FAIL"),
                      f"{same} of {len(TABLE1)} values of manuscript Table 1 equal", f)
    add(11, "Dataset scale (Table 1)", "資料規模（表 1）", "Table 1", "recomputed from public data", c11)

    def c12():
        f = lay.s("input_data_audit") / "category_threshold_check.csv"
        r = rows(f)
        cats = [x for x in r if x["cir_evaluated"] == "yes"]
        questions = sum(int(x["cir_test_questions"] or 0) for x in cats)
        male = sum(1 for x in cats if x["male_any_label"])
        ok = len(cats) == 19 and questions == 9311 and male == 0
        return result("PASS" if ok else ("MISSING" if not r else "FAIL"),
                      f"{len(cats)} evaluated categories, {questions:,} queries, {male} male-labelled (thesis 5.2 "
                      "claim corrected: the threshold does not filter training data)", f)
    add(12, "3,000-item category threshold", "男裝類別的 3,000 件門檻（碩論 5.2 節）", "thesis 5.2", "recomputed", c12)

    def c13():
        f = lay.s("dataset_tables_check") / "summary.md"
        ok = found(f, r"All 471 values equal the archived tables")
        return result("PASS" if ok else ("MISSING" if not f.is_file() else "FAIL"),
                      "471 of 471 values equal T00, T01, A01, A02, A12, A13", f)
    add(13, "Dataset tables (not in the manuscript)", "資料集額外統計（論文沒用）", "2025 T00, T01, A01, A02, A12, A13",
        "recomputed", c13)

    def c14():
        f = lay.s("paper_value_checks") / "table2_text_fields.csv"
        r = {x["text"].split()[0]: x for x in rows(f)}
        same = sum(1 for t, values in TABLE2.items() for c, v in zip(TABLE2_COLUMNS, values)
                   if r.get(t, {}).get(c) == v)
        total = sum(len(v) for v in TABLE2.values())
        return result("PASS" if same == total else ("MISSING" if not r else "FAIL"),
                      f"{same} of {total} values of manuscript Table 2 equal (missing 34, mean words 4.82 and 10.08, "
                      "...)", f)
    add(14, "Text fields (Table 2)", "原始文字與情境描述的比較（表 2）", "Table 2", "recomputed from input data", c14)

    def c15():
        f = lay.paths["secondary"] / "environment_proxy" / "table_4_3_paper_field_comparison.csv"
        r = rows(f)
        same = sum(1 for x in r if x["paper_display_match"] == "True")
        proxy = {x["variable"]: x for x in rows(lay.s("paper_value_checks") / "table5_proxy_values.csv")}
        cols = ("median", "q1", "q3", "p5", "p95", "max")
        m_same = sum(1 for k, values in TABLE5_PROXY.items() for c, v in zip(cols, values) if proxy.get(k, {}).get(c) == v)
        m_total = sum(len(v) for v in TABLE5_PROXY.values()) + 5
        m_same += 5 if found(lay.s("paper_value_checks") / "summary.md", re.escape(TABLE5_COUNTS)) else 0
        ok = r and same == len(r) and m_same == m_total
        return result("PASS" if ok else ("MISSING" if not r else "FAIL"),
                      f"{same} of {len(r)} displayed values equal (thesis Table 4-3); {m_same} of {m_total} values of the "
                      "manuscript's Table 5 row equal (quantiles and counts)", f,
                      lay.s("paper_value_checks") / "table5_proxy_values.csv")
    add(15, "Proxy distributions (Table 5)", "代理值分布（表 5）", "Table 5; thesis Table 4-3", "recomputed from input data",
        c15)

    def c16():
        f = lay.s("input_data_audit") / "clo_distribution_summary.csv"
        r = [x for x in rows(f) if x.get("same_at_4_decimals")]
        same = sum(1 for x in r if x["same_at_4_decimals"] == "yes")
        return result("PASS" if r and same == len(r) else ("MISSING" if not r else "FAIL"),
                      f"{same} of {len(r)} statistics of thesis Table D-3 equal", f)
    add(16, "CLO distribution (thesis D-3, D-4)", "CLO 分布摘要（碩論表 D-3）", "thesis Table D-3, Figure D-4", "recomputed", c16)

    def c17():
        f = lay.s("paper_value_checks") / "table5_target_clues.csv"
        r = rows(f)
        same = sum(1 for x in r if x["count_first_occurrence"] == x["count_2025_A03"])
        return result("PASS" if r and same == len(r) else ("MISSING" if not r else "FAIL"),
                      f"{same} of {len(r)} counts equal A03 and the manuscript (4.50%, 4.88%, 11.06%, 0.87%, 12.40%)", f)
    add(17, "Target clues (Table 5)", "目標線索（表 5）", "Table 5; thesis Table 4-6", "recomputed from input data", c17)

    def c18():
        f = lay.s("judge_audit_checks") / "summary.md"
        ok = found(f, r"Mean and SD equal the 2025 table T19 to all digits: yes\. The 100 bars equal the thesis figure")
        return result("PASS" if ok else ("MISSING" if not f.is_file() else "FAIL"),
                      "means and SDs equal T19; the histogram equals thesis Figure 4-1", f)
    add(18, "Judge score distribution", "Judge 分數分布（碩論圖 4-1）", "thesis Figure 4-1, 4.2.4", "statistics recomputed", c18)

    def c19():
        t19 = lay.paths["secondary"] / "judge" / "T19_recomputed.csv"
        ties = lay.s("judge_audit_checks") / "bottom_p_tie_orders.csv"
        missing = need(t19, ties)
        if missing:
            return result("MISSING", f"{lay.where(missing)} not found")
        mine, archived = rows(t19)[0], rows(T19_ARCHIVED)[0]
        t19_same = all(abs(float(mine[k]) - float(archived[k])) <= 1e-12 for k in archived)
        r = rows(ties)
        inside = all(x["thesis_within_tie_range"] == "yes" for x in r)
        col = lambda key: ", ".join(x[key] for x in r)  # noqa: E731
        # numpy's argsort orders the tied scores by CPU type: as in 2025 on x86, differently on the GB10 (README 0.8)
        if t19_same and all(x["intersection_numpy_argsort"] == x["thesis_table_4_7"] for x in r):
            return result("PASS", "T19 (correlations, QWK) equal to the archived T19 within 1e-12: yes; bottom 5/10/20% "
                          f"overlaps {col('intersection_numpy_argsort')} equal to thesis Table 4-7", ties, t19)
        return result("DIFFERS" if t19_same and inside else "FAIL",
                      f"T19 (correlations, QWK) equal to the archived T19 within 1e-12: {'yes' if t19_same else 'NO'}; "
                      f"bottom 5/10/20% overlaps {col('intersection_numpy_argsort')} (thesis Table 4-7 "
                      f"{col('thesis_table_4_7')}), which differ only in the order of tied scores: each thesis value "
                      f"is a possible order ({', '.join(x['smallest_over_tie_orders'] + '-' + x['largest_over_tie_orders'] for x in r)}; "
                      f"medians of random orders {col('random_orders_median')}): {'yes' if inside else 'NO'}",
                      ties, t19)
    add(19, "Judge agreement (Table 5)", "Judge 一致性（表 5）", "Table 5; thesis Table 4-7", "statistics recomputed", c19)

    def c20():
        f = lay.s("judge_audit_checks") / "summary.md"
        curve = rows(lay.s("judge_audit_checks") / "bottom_p_curve_tie_range.csv")
        m = re.search(r"jaccard ([0-9.]+), jaccard_lo ([0-9.]+), jaccard_hi ([0-9.]+), f1 ([0-9.]+), f1_lo ([0-9.]+), "
                      r"f1_hi ([0-9.]+)\.", text(f))
        if not curve or m is None:
            return result("MISSING", f"{lay.where(f)} or bottom_p_curve_tie_range.csv not found")
        inside = sum(1 for x in curve if x["thesis_within_tie_range"] == "yes")
        if all(float(m.group(i)) == 0 for i in range(1, 7)):  # tied scores ordered as in 2025, see c19
            return result("PASS", "curves and bootstrap bands equal to thesis Figure 4-3 (largest difference 0.0000 over "
                          "the 30 values of p; the thesis values are read from the archived SVG)", f,
                          lay.s("judge_audit_checks") / "bottom_p_curve_tie_range.csv")
        return result("DIFFERS" if inside == len(curve) else "FAIL",
                      f"curves within {max(float(m.group(i)) for i in (1, 4)):.4f} of thesis Figure 4-3 and the "
                      f"bootstrap bands within {max(float(m.group(i)) for i in (2, 3, 5, 6)):.4f}; at {inside} of "
                      f"{len(curve)} values of p "
                      "the thesis curve is a possible order of the tied scores", f,
                      lay.s("judge_audit_checks") / "bottom_p_curve_tie_range.csv")
    add(20, "Bottom-p sensitivity curve", "低分重疊的敏感度曲線（碩論圖 4-3）", "thesis Figure 4-3", "statistics recomputed", c20)

    def c21():
        f = lay.s("checklist_coverage") / "summary.md"
        if not f.is_file():
            return result("SKIPPED", "needs bootstrap_data.sh --with-nomic")
        ok = found(f, r"Coverage table, 51 values of tau x 10 columns, at the six decimals printed in 2025: identical")
        return result("PASS" if ok else "FAIL", "coverage table equal to the 2025 notebook output (51 x 10 values)", f)
    add(21, "Checklist concept coverage", "兩份檢核表的概念涵蓋（碩論圖 4-2）", "thesis Figure 4-2", "recomputed", c21)

    def c22():
        f = lay.s("judge_audit_checks") / "judge_reference_scan.csv"
        r = rows(f)
        units = [x for x in r if "35 units" in x.get("scope", "")]
        unit_refs = [x for x in units if x.get("assessment") != "no_judge_artifact_reference_found"]
        scanned = len(units)
        return result("PASS" if r and not unit_refs else ("MISSING" if not r else "FAIL"),
                      f"{scanned} files run by the 35 units scanned; {len(unit_refs)} read judge outputs", f)
    add(22, "Judge not used in training", "Judge 未進入訓練的程式檢查", "A05, A06", "code check", c22)

    def c23():
        f = lay.paths["secondary"] / "judge" / "table_4_8_paper_field_comparison.csv"
        r = rows(f)
        same = sum(1 for x in r if x["paper_display_match"] == "True")
        pr = {x["judge"]: x for x in rows(lay.paths["secondary"] / "judge" / "prompt_robustness_summary.csv")
              if x["variant"] == "P0-R2"}
        m_same = sum(1 for (j, c), v in TABLE5_PROMPT.items() if j in pr and f"{float(pr[j][c]):.{len(v) - 2}f}" == v)
        ok = r and same == len(r) and m_same == len(TABLE5_PROMPT)
        return result("PASS" if ok else ("MISSING" if not r else "FAIL"),
                      f"{same} of {len(r)} displayed values equal (thesis Table 4-8); {m_same} of {len(TABLE5_PROMPT)} "
                      "values of the manuscript's Table 5 row equal", f)
    add(23, "Prompt robustness (Table 5)", "Prompt 穩健性（表 5）", "Table 5; thesis Table 4-8", "statistics recomputed", c23)

    def c24():
        f = lay.paths["chapter4"] / "chapter4_table_overview.csv"
        r = {x["table"]: x["result"] for x in rows(f)}
        ok = r.get("4-9") == r.get("4-10") == "archived_human_audit_reanalysis_exact"
        audit = {x["checklist"]: x for x in rows(lay.s("judge_audit_checks") / "audit_score_metrics.csv")}
        m_same = sum(1 for j, values in TABLE5_AUDIT.items() for c, v in zip(("human_mean", "model_mean", "mae"), values)
                     if audit.get(j, {}).get(c) == v)
        matrix = json.loads(text(lay.paths["secondary"] / "remaining_reproduction_matrix.json") or "{}")
        sel = next((m for m in matrix.get("modules", []) if m.get("module") == "human_audit"), {})
        sel_ok = sel.get("status") == "exact" and "selection match=True" in sel.get("evidence", "")
        ok = ok and m_same == 6 and sel_ok
        return result("PASS" if ok else ("MISSING" if not r else "FAIL"),
                      "thesis Tables 4-9 and 4-10 recomputed exactly from the 750 judgments; the stratified seed-42 "
                      f"selection of the 30 cases regenerated: {'yes' if sel_ok else 'NO'}; {m_same} of 6 values of the "
                      "manuscript's Table 5 row equal (human and model means, MAE)",
                      f, lay.paths["secondary"] / "human_audit" / "T30_recomputed.csv")
    add(24, "Human audit (Table 5)", "人工稽核（表 5）", "Table 5; thesis Tables 4-9, 4-10", "statistics recomputed", c24)

    def c25():
        f = lay.s("judge_audit_checks") / "summary.md"
        ok = found(f, r"Equal to the archived T31 in all 11 compared columns and the order of the 25 rows: yes")
        return result("PASS" if ok else ("MISSING" if not f.is_file() else "FAIL"),
                      "25 rows equal T31 in value and order", f)
    add(25, "Human-audit item disagreement", "人工稽核的逐項分歧（碩論圖 4-4）", "manuscript 6.2; thesis Figure 4-4",
        "statistics recomputed", c25)

    def c26():
        bad = ext_step(lay, ext, "text_length", lay.ext / "text_length" / "summary.md",
                       lay.ext / "main_cir_per_query" / "manifest.json")
        if bad:
            return bad
        manifest = json.loads(text(lay.ext / "main_cir_per_query" / "manifest.json"))
        units = manifest.get("units", [])
        ok = manifest.get("status") == "passed" and all(u.get("recalls_equal_stored") for u in units)
        m = re.search(r"ΔHit@10 within five tokens \(Full − Original\) \| ([-+0-9.]+)", text(lay.ext / "text_length" / "summary.md"))
        return result("DIFFERS" if ok else "FAIL",
                      f"{len(units)} main CIR units re-evaluated, recalls equal to the run's ({manifest.get('status')}); "
                      f"ΔHit@10 within five tokens {m.group(1) if m else '?'} (manuscript +0.0106, which the archived "
                      "2025 rows give again through the same code)", lay.ext / "text_length")
    add(26, "Text length (Table 5)", "文字長度（表 5）", "Table 5; thesis Tables 4-4, 4-5", "computed from the run's models",
        c26, ports=("validate_length",))

    # 三、模型訓練
    def manuscript_run() -> tuple[str, str]:
        """Retrained, so the values differ from 2025; the revised manuscript reports the official run's values."""
        note = "; retrained, so the values differ from the 2025 values; the revised manuscript already reports the "
        if lay.official:
            return "DIFFERS", note + "official run's values"
        if lay.comparison is not None and lay.comparison.is_file():
            verdict = json.loads(text(lay.comparison)).get("verdict", "?")
            return ("FAIL" if verdict == "NOT CONSISTENT" else "DIFFERS",
                    note + f"official run's values; comparison with the official run: {verdict}")
        return "MISSING", "; the comparison with the official run (official_comparison.json) was not found"

    def c27():
        main = [s for fam, s in lay.units if fam == "main"]
        ok = len(main) == 10 and all(s == "passed" for s in main)
        state, note = manuscript_run()
        return result(state if ok else "FAIL", f"{sum(s == 'passed' for s in main)} of 10 main units passed{note} "
                      "(docs/manuscript_reconciliation/)", lay.paths["main"])
    add(27, "Main experiment (Table 6)", "主實驗（表 6）", "Table 6", "retrained", c27)

    def c28():
        abl = [s for fam, s in lay.units if fam != "main"]  # status passed_candidate_source: the rebuilt subset
        ok = len(abl) == 25 and all(s.startswith("passed") for s in abl)
        state, note = manuscript_run()
        return result(state if ok else "FAIL", f"{sum(s.startswith('passed') for s in abl)} of 25 fair-subset units "
                      f"passed{note}", lay.paths["ablation"])
    add(28, "Fair-subset ablation (Table 7)", "公平子集消融（表 7）", "Table 7", "retrained", c28)

    def c29():
        f = lay.paths["stats"] / "main_paired_bh_8metrics.csv"
        ok = len(rows(f)) == 8 and (lay.paths["ablation"] / "T07_stage2_factor_contribution_ratio.csv").is_file()
        state, note = manuscript_run()
        return result(state if ok else ("MISSING" if not f.is_file() else "FAIL"),
                      f"8 metrics with mean, SD, CI, BH p and dz; ablation tables T02-T07, T10, T11{note}", f)
    add(29, "Statistics of Tables 6 and 7", "表 6、7 的統計檢定", "Tables 6, 7", "computed from the run", c29)

    def c30():
        latest = recon.Latest(lay.run_id, lay.paths)
        checks = [r for r in recon.read_csv(recon.VALUES_CSV) if r["group"] == "required" and r["rule"] == "check"]
        held = []
        for r in checks:
            ok, *_ = getattr(latest, "check_" + r["latest_key"].split(".", 1)[1])()
            held.append(bool(ok))
        return result("PASS" if held and all(held) else "FAIL", f"{sum(held)} of {len(held)} required conclusions "
                      "(R01-R07) hold", lay.paths["stats"])
    add(30, "The seven required conclusions", "必須成立的 7 條結論（R01～R07）",
        "abstract, 5.2, 5.3 (R01-R07 in docs/manuscript_reconciliation/)", "computed from the run",
        c30)

    # 四、統計與結論
    def c31():
        f = ra / "fair_subset_factor_effects.csv"
        ok = f.is_file() and (ra / "figures" / "figure_4_7_factor_contribution.svg").is_file() and \
            found(ra / "summary.md", r"Mean differences and p-values of the four comparisons equal the run's table T02 \(20 rows\): yes")
        return result("DIFFERS" if ok else ("MISSING" if not f.is_file() else "FAIL"),
                      "factor effects equal the run's T02; thesis Figure 4-7 from T07 (CIR shares not defined); "
                      "computed from the retrained units, so the values differ from 2025", f)
    add(31, "Ablation statistics, Figure 4-7", "消融的因素貢獻（碩論圖 4-7）", "thesis Figure 4-7", "computed from the run", c31)

    def c32():
        f = ra / "main_statistical_rigor.csv"
        s32 = text(ra / "summary.md")
        ok = "main_statistical_rigor.csv` equal the run's statistics table main_paired_bh_8metrics.csv: yes" in s32
        m = re.search(r"\| CP auc \| [0-9.]+ \| [0-9.]+ \| ([-0-9.]+) \[.*?\| ([-0-9.]+), ", s32)
        auc = f"; CP AUC difference {float(m.group(1)):+.4f} (2025 A14 {float(m.group(2)):+.4f})" if m else ""
        return result("DIFFERS" if ok else ("MISSING" if not f.is_file() else "FAIL"),
                      "A14 for the run; t-test values equal the run's statistics; Wilcoxon p = 0.0625 (five seeds)"
                      f"{auc}", f)
    add(32, "Wilcoxon tests (A14; not in the manuscript)", "Wilcoxon 檢定（論文沒用）", "2025 A14", "computed from the run", c32)

    # 五、其他實驗
    # Items computed with the retrained models differ from the manuscript in value. FAIL is kept for a computation
    # that does not hold (a check of the ported code, a count against 2025); a manuscript statement that the run
    # does not repeat is reported in the evidence.
    def c33():
        f = lay.ext / "two_tower" / "table8_two_tower.csv"
        bad = ext_step(lay, ext, "two_tower", f)
        if bad:
            return bad
        r = rows(f)
        same_sign = sum(1 for x in r if (x["mean_difference"][0] == "-") == (TABLE8_MANUSCRIPT[x["metric"]][0] == "-"))
        auc = next((x["mean_difference"] for x in r if x["metric"] == "cp_test_auc"), "?")
        return result("DIFFERS" if len(r) == 7 else "FAIL",
                      f"10 models retrained; {same_sign} of {len(r)} differences in the manuscript's direction "
                      f"(CP AUC {auc}, manuscript +0.0378)", f)
    add(33, "Two-Tower (Table 8)", "Two-Tower（表 8）", "Table 8", "retrained", c33, ports=("validate_two_tower",))

    def c34():
        f = lay.ext / "counterfactual" / "table9_seeds_mean_sd.csv"
        bad = ext_step(lay, ext, "counterfactual", f)
        if bad:
            return bad
        r = {x["scope"]: x for x in rows(f)}
        change = {k: float(v["mean_rank_change"].split(" ")[0]) for k, v in r.items()}
        style_largest = bool(change) and max(change, key=lambda k: abs(change[k])) == "Style"
        m = re.search(r"\| context_aware \| \d+/24 \| \d+ \| (\d+)/24 \|\n\| counterfactual \| \d+/24 \| \d+ \| (\d+)/24 \|",
                      text(lay.ext / "counterfactual" / "summary.md"))
        check = (f"the preserved 2025 seed-1 model gives the top-5 lists of A45 in {m.group(1)}/24 and {m.group(2)}/24 "
                 "pairs" if m else "the check with the 2025 model was not run (--validate)")
        return result("DIFFERS" if m is None or m.group(1) == m.group(2) == "24" else "FAIL",
                      f"Context models of 5 seeds; overall rank change {r.get('Overall', {}).get('mean_rank_change', '?')} "
                      f"(manuscript +17.0, one 2025 model); Style the largest change: {'yes' if style_largest else 'no'}; "
                      f"{check}", f)
    add(34, "Counterfactual analysis (Table 9)", "反事實分析（表 9）", "Tables 4, 9", "computed from the run's models", c34)

    def c35():
        f = lay.ext / "case_figures" / "counterfactual_figure_manifest.csv"
        bad = ext_step(lay, ext, "figures", f)
        if bad:
            return bad
        picked = ", ".join(x["pair_id"] for x in rows(f))
        if lay.official:  # the committed results leave out logs/ (results/extensions/.gitignore): apply the rule here
            sys.path.append(str(HERE / "extensions"))
            import case_figures
            a46 = case_figures.COUNTERFACTUAL_DIR / "A46_counterfactual_context_manual_review_sheet.csv"
            archived = [x["pair_id"] for x in case_figures.select_examples(case_figures.read_csv(a46))]
        else:
            log = text(lay.ext / "logs" / "figures.log")
            m = re.search(r"the same rule on the archived A46 selects (\S+) and (\S+) ", log)
            if m is None:
                return result("MISSING", "the check of P16's rule on the archived A46 is not in logs/figures.log", f)
            archived = list(m.groups())
        ok = set(archived) == {"CF04", "CF09"}
        return result("DIFFERS" if ok else "FAIL", f"P16's rule selects {picked} from the run's seed-1 results; on the "
                      f"archived A46 it selects {archived[0]} and {archived[1]} (2025 figures: CF04, CF09); figures "
                      "local only", f)
    add(35, "Counterfactual example figures (not in the manuscript)", "反事實範例圖（論文沒用）", "2025 F34a, F34b",
        "computed from the run's models", c35)

    def c36():
        f = lay.ext / "text_swap" / "summary.md"
        bad = ext_step(lay, ext, "text_swap", f)
        if bad:
            return bad
        ok = found(f, r"equal the run's stored results: yes")

        def swap(path: Path) -> str:
            r = rows(path)
            auc = lambda model, txt: [float(x["auc"]) for x in r if x["model"] == model and x["text"] == txt]  # noqa: E731
            if not r:
                return "?"
            gains = [a - b for a, b in zip(auc("original", "context"), auc("original", "original"))]
            swapped = auc("context", "original")
            return f"{sum(gains) / len(gains):+.4f}, {sum(swapped) / len(swapped):.4f}"
        old = lay.ext / "validation" / "text_swap_2025_checkpoints" / "text_swap_results.csv"
        ref = f" (2025 checkpoints {swap(old)})" if old.is_file() else ""
        return result("DIFFERS" if ok else "FAIL", "main models of 5 seeds with the other text; the re-evaluated "
                      "matching-text combinations equal the run's results; AUC gain of the Original models with the "
                      "context-aware text, AUC of the Context models with the original titles: "
                      f"{swap(lay.ext / 'text_swap' / 'text_swap_results.csv')}{ref}", f)
    add(36, "Text swap (not in the manuscript)", "文字互換評估（論文沒用）", "2025 sweep", "computed from the run's models", c36,
        ports=("validate_text_swap",))

    def c37():
        f = lay.ext / "reliability" / "manifest.json"
        bad = ext_step(lay, ext, "reliability", f)
        if bad:
            return bad
        m = json.loads(text(f))
        ok = m.get("status") == "complete" and m.get("labels_equal_archived_t18") and m.get("a16_equal_archived")

        def perf(path: Path) -> str:
            r = {x["run_tag"]: x for x in rows(path)}
            if set(r) < {"original", "proposed"}:
                return "?"
            return ", ".join(f"{k} {float(r['original'][c]):.4f}/{float(r['proposed'][c]):.4f}"
                             for k, c in (("Hit@10", "Hit@10"), ("ECE", "ECE"), ("Brier", "Brier Score")))
        old = lay.ext / "validation" / "reliability_2025_seed1" / "performance.csv"
        ref = f" (2025 models: {perf(old)})" if old.is_file() else ""
        return result("DIFFERS" if ok else "FAIL", "labels of 9,311 queries equal T18 and A16; seed-1 models evaluated "
                      f"({m.get('device')}, {m.get('precision')}), Original/Context "
                      f"{perf(f.parent / 'performance.csv')}{ref}", f.parent)
    add(37, "Reliability analysis (not in the manuscript)", "可靠度／校準分析（論文沒用）", "2025 P04: T18, F12-F16, A16",
        "computed from the run's models", c37, ports=("validate_reliability_labels", "validate_reliability"))

    # 六、質性分析 (fair subset; the 2025 ablation models and query rows are not preserved, so the counts are what
    # can be compared with 2025)
    def c38():
        f = ra / "category_hit10_delta.csv"
        r = rows(f)
        if not r:
            return result("MISSING", f"{lay.where(f)} not found")
        counts = found(ra / "summary.md", r"Observations per category equal the 2025 table T15: yes")
        d = {x["fine_category"]: float(x["delta_hit10"]) for x in r}
        top = max(d, key=d.get)
        sun = d.get("sunglasses", float("nan"))
        return result("DIFFERS" if counts else "FAIL", f"observations per category equal T15: {'yes' if counts else 'NO'}; largest gain {top} "
                      f"({d[top]:+.4f}); sunglasses {sun:+.4f} (manuscript: dress, bag and shoe gain most, sunglasses "
                      "declines slightly)", f)
    add(38, "Category improvement (Figure 4)", "類別改善（圖 4）", "Figure 4; thesis Figure 4-8", "computed from the run", c38)

    def c39():
        f = ra / "factor_category_effects.csv"
        missing = need(f, ra / "factor_term_effects.csv")
        if missing:
            return result("MISSING", f"{lay.where(missing)} not found")
        s = text(ra / "summary.md")
        counts = "Observations per category in all four comparisons equal the 2025 table T15: yes" in s
        again = re.findall(r"terms of the 2025 figure again in this run's top 8: (\d) of 8", s)
        return result("DIFFERS" if counts else "FAIL",
                      f"observations per category in the four comparisons equal T15: {'yes' if counts else 'NO'}; "
                      f"terms of the six 2025 term figures again in the run's top 8: {', '.join(again)} of 8",
                      f, ra / "factor_term_effects.csv")
    add(39, "Factor effects by category and term", "各因素對類別與詞彙的影響（碩論圖 4-11～4-13）", "thesis Figures 4-11 to 4-13",
        "computed from the run", c39)

    def c40():
        f = ra / "context_subset_hit10.csv"
        r = rows(f)
        if not r:
            return result("MISSING", f"{lay.where(f)} not found")
        counts = found(ra / "summary.md", r"Observations per subset equal the 2025 table T12: yes")
        deltas = [float(x["delta_hit10"]) for x in r if x.get("delta_hit10")]
        positive = sum(1 for d in deltas if d > 0)
        return result("DIFFERS" if counts else "FAIL", f"observations per subset equal T12: {'yes' if counts else 'NO'}; {positive} of "
                      f"{len(deltas)} subset differences positive (manuscript: all)", f)
    add(40, "Context subsets, Hit@10", "情境子集的 Hit@10（5.5 節）", "manuscript 5.5; thesis Figure 4-9", "computed from the run",
        c40)

    def c41():
        f = ra / "subset_robustness_summary.csv"
        r = rows(f)
        if not r:
            return result("MISSING", f"{lay.where(f)} not found")
        counts = found(ra / "summary.md", r"all 13 rows of T12, including the unions: yes")
        better = sum(1 for x in r if float(x["ΔMedian Rank"]) < 0)
        better_2025 = sum(1 for x in rows(T12_ARCHIVED) if float(x["ΔMedian Rank"]) < 0)
        return result("DIFFERS" if counts else "FAIL",
                      f"T12 recomputed; observations of all 13 rows equal: {'yes' if counts else 'NO'}; the median rank "
                      f"improves in {better} of {len(r)} rows (2025: {better_2025} of 13)", f)
    add(41, "Context subsets, median rank", "情境子集的中位名次（碩論圖 4-10）", "thesis Figure 4-10, T12", "computed from the run",
        c41)

    def c42():
        f = ra / "case_ranks.csv"
        r = rows(f)
        if len(r) != 15:
            return result("MISSING" if not r else "FAIL", f"{len(r)} case rows (3 cases x 5 conditions expected)", f)
        m = {(x["case"], x["condition"]): float(x["mean_rank"]) for x in r}
        mine = {"purse": f"{m['purse', 'Original']} -> {m['purse', 'Full']}",
                "dress": f"{m['dress', 'Original']} -> {m['dress', 'Full']}, No-Style {m['dress', 'No-Style']}",
                "sunglasses": f"{m['sunglasses', 'Original']} -> {m['sunglasses', 'Full']}"}
        holds = [m["purse", "Full"] < m["purse", "Original"], m["dress", "Full"] < m["dress", "Original"],
                 m["dress", "No-Style"] > m["dress", "Full"], m["sunglasses", "Full"] > m["sunglasses", "Original"]]
        return result("DIFFERS", "five-seed mean ranks " + "; ".join(f"{c} {mine[c]} (manuscript {CASES_MANUSCRIPT[c]})"
                                                                      for c in mine)
                      + f"; {sum(holds)} of 4 directions of Section 5.5 hold", f)
    add(42, "Case ranks", "個案名次（5.5 節）", "manuscript 5.5", "computed from the run", c42)

    def c43():
        f = lay.ext / "color_analysis" / "summary.md"
        bad = ext_step(lay, ext, "color", f, lay.ext / "color_analysis" / "color_shift_summary.csv")
        if bad:
            return bad
        m = re.search(r"text\): (\d+), over (\d+) cases", text(f))
        same_rows = m is not None and (m.group(1), m.group(2)) == ("745", "149")
        s = {x["level"]: x for x in rows(lay.ext / "color_analysis" / "color_shift_summary.csv")}
        return result("DIFFERS" if same_rows else "FAIL",
                      f"comparable rows {m.group(1) if m else '?'} over {m.group(2) if m else '?'} cases (P03 printed 745 "
                      f"over 149 in 2025); solved under Full: rows {s['row']['solved_n']}/{s['row']['denominator']}, "
                      f"cases {s['case']['solved_n']}/{s['case']['denominator']} (manuscript 39/89, 5/15)", f)
    add(43, "Color-shift analysis", "色彩分析（5.5 節）", "manuscript 5.5; thesis Table 4-19", "computed from the run + images", c43)

    def c44():
        f = lay.ext / "case_figures" / "color_case_figure_manifest.csv"
        bad = ext_step(lay, ext, "figures", f)
        if bad:
            return bad
        shown = {x["condition"]: x for x in rows(f) if x["shown_seed"] == "yes"}
        o, full = shown.get("Original", {}), shown.get("Full", {})
        black = o.get("top5_dominant_colors", "").split(" | ").count("black")
        gray = full.get("top5_dominant_colors", "").split(" | ").count("gray")
        return result("DIFFERS", f"seed {o.get('seed', '?')}: the Original top 5 has {black} of 5 black items, the Full "
                      f"top 5 {gray} of 5 gray; target rank {o.get('rank', '?')} -> {full.get('rank', '?')} (thesis: all "
                      "five black, gray items in front, 27 -> 36); figure local only", f)
    add(44, "Gray color-shift case figure", "灰色的顏色偏移個案圖（碩論圖 4-14）", "thesis Figure 4-14", "computed from the run + images",
        c44)

    def c45():
        f = lay.ext / "case_figures" / "case_figures_manifest.csv"
        bad = ext_step(lay, ext, "figures", f)
        if bad:
            return bad
        r = rows(f)
        rk = {(x["figure"], x["condition"].split(" (")[0]): int(x["rank"]) for x in r}
        seed = {x["figure"]: x["seed"] for x in r}
        holds = [rk["A1", "Full"] < rk["A1", "Original"], rk["A1", "No-Occasion"] > rk["A1", "Full"],
                 rk["A1", "No-Style"] > rk["A1", "Full"], rk["A2", "Full"] < rk["A2", "Original"],
                 rk["A2", "No-Style"] > max(rk["A2", "No-Weather"], rk["A2", "No-Occasion"]),
                 rk["A3", "Full"] > rk["A3", "Original"]]
        return result("DIFFERS", f"A1 purse (seed {seed['A1']}) {rk['A1', 'Original']} -> {rk['A1', 'Full']} (caption: "
                      f"25 -> 2); A2 dress (seed {seed['A2']}) {rk['A2', 'Original']} -> {rk['A2', 'Full']}, No-Style "
                      f"{rk['A2', 'No-Style']}; A3 sunglasses (seed {seed['A3']}) {rk['A3', 'Original']} -> "
                      f"{rk['A3', 'Full']}; {sum(holds)} of 6 statements of the captions hold, the ranks in the "
                      "captions must be updated; figures local only", f)
    add(45, "Appendix Figures A1-A3", "附錄圖 A1～A3", "manuscript appendix", "computed from the run + images", c45)

    def c46():
        f = lay.ext / "outfit_generation" / "generation.csv"
        bad = ext_step(lay, ext, "outfit_generation", f)
        if bad:
            return bad
        picks = {}
        for x in rows(f):
            picks.setdefault(x["setting"], []).append(x["item_id"])
        same = sum(a == b for k, v in P02_PICKS.items() for a, b in zip(v, picks.get(k, [])))
        total = sum(len(v) for v in P02_PICKS.values())
        return result("DIFFERS",
                      f"{sum(len(v) for v in picks.values())} generation steps with the run's Context seed-1 model; "
                      f"{same} of {total} picks equal to P02's (2025 model); the try-on images are not reproduced "
                      "(FastFit and the person photo are not part of the handoff)", f)
    add(46, "Outfit generation of the try-on demo (not in the manuscript)", "虛擬試穿示範的逐步選品（論文沒用）", "2025 P02",
        "computed from the run's models", c46, ports=("validate_outfit_generation",))
    return out


def validations(lay: Layout) -> list[list[str]]:
    ext = extension_status(lay)
    names = {"counterfactual": "2025 counterfactual model against A45 (with --validate)",
             "validate_length": "text length: archived A07 through the length analysis",
             "validate_two_tower": "Two-Tower: the 20 preserved checkpoints against A30",
             "validate_reliability_labels": "reliability labels against T18 and A16",
             "validate_outfit_generation": "outfit generation: the 2025 model against P02's picks",
             "validate_reliability": "reliability: the 2025 models on the GPU against T18",
             "validate_text_swap": "text swap: the 20 preserved main checkpoints against A14"}
    return [[step, label, ext.get(step, "not run")] for step, label in names.items()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--official", action="store_true", help="the official run's committed results")
    group.add_argument("--run-root", type=Path, help="a completed full run folder")
    parser.add_argument("--supplementary-dir", type=Path, default=None, help="default: <run>/supplementary")
    parser.add_argument("--extensions-dir", type=Path, default=None, help="default: <run>/extensions")
    parser.add_argument("--out-dir", type=Path, default=None)
    args = parser.parse_args()
    lay = Layout(args)
    out_dir = args.out_dir or (REPRO / "results" if args.official else lay.root)
    result_rows = items(lay)
    counts = Counter(r["state"] for r in result_rows)
    checks = validations(lay)
    bad = [r for r in result_rows if r["state"] in ("MISSING", "FAIL")]
    verdict = "INCOMPLETE" if bad else ("COMPLETE WITH SKIPPED ITEMS" if counts["SKIPPED"] else "COMPLETE")
    lines = [f"# Reproduction items: {lay.run_id}", "",
             f"Generated by `reproduction/scripts/report_items.py` from "
             f"{'the committed results of the official run' if args.official else 'the run folder'}. "
             "States: PASS (recomputed and equal to the 2025 values of the senior student's project: the archived "
             "outputs, the thesis or the original manuscript), DIFFERS (recomputed with the computation checked, but "
             "the values differ from the 2025 values because the models were retrained or because tied scores are "
             "ordered differently; the evidence gives the reason and the check; the revised manuscript already "
             "reports the official run's Tables 6 and 7), REUSED (preserved input data that cannot be regenerated, SHA-256 checked), NOT_REPRODUCIBLE, "
             "SKIPPED (download or GPU missing), MISSING / FAIL (an output is missing or a check of the computation "
             "does not hold).", "",
             f"**Verdict: {verdict}** — " + ", ".join(f"{s} {counts[s]}" for s in STATES if counts[s]) + ".", "",
             "| # | Item | 項目 | Manuscript / thesis | How | State | Evidence |", "|---:|---|---|---|---|---|---|"]
    for r in result_rows:
        lines.append(f"| {r['id']} | {r['item']} | {r['item_zh']} | {r['reference']} | {r['how']} | {r['state']} | "
                     f"{r['evidence']} |")
    lines += ["", "## Checks of the ported code against the 2025 outputs", "", "| Step | Check | Status |",
              "|---|---|---|"]
    lines += [f"| {a} | {b} | {c} |" for a, b, c in checks]
    lines += ["", "## Outputs", ""]
    lines += [f"- {r['id']}: " + ", ".join(f"`{o}`" for o in r["outputs"]) for r in result_rows if r["outputs"]]
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "ITEMS_STATUS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (out_dir / "ITEMS_STATUS.json").write_text(json.dumps({
        "run": lay.run_id, "official": args.official, "verdict": verdict, "counts": dict(counts),
        "items": result_rows, "port_checks": [dict(zip(("step", "check", "status"), c)) for c in checks],
    }, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"[ITEMS] {lay.run_id}: {verdict} — " + ", ".join(f"{s} {counts[s]}" for s in STATES if counts[s]))
    for r in result_rows:
        if r["state"] in ("MISSING", "FAIL", "SKIPPED"):
            print(f"[ITEMS] {r['id']:>2} {r['state']}: {r['item']} — {r['evidence']}")
    print(f"[ITEMS] report: {out_dir / 'ITEMS_STATUS.md'}")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
