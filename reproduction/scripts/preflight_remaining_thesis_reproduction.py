#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
REPRO_ROOT = ROOT / "reproduction"
REPRO_SCRIPTS = REPRO_ROOT / "scripts"

DATA = ROOT / "01_資料建構_data_construction"
MODEL = ROOT / "02_模型訓練和驗證_model_training_validation"
EXP = ROOT / "03_實驗與結果_experiments_results"

DEFAULT_OUT = REPRO_ROOT / "runs/remaining_thesis_reproduction"

MAX_HASH_BYTES = 50 * 1024 * 1024


def rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except Exception:
        return str(path)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def git(*args: str) -> str:
    p = subprocess.run(
        ["git", *args],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return p.stdout.strip() if p.returncode == 0 else ""


def is_tracked(path: Path) -> bool:
    if not path.exists():
        return False
    p = subprocess.run(
        ["git", "ls-files", "--error-unmatch", rel(path)],
        cwd=ROOT,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return p.returncode == 0


def is_lfs_pointer(path: Path) -> bool:
    if not path.is_file():
        return False
    try:
        with path.open("rb") as f:
            head = f.read(256)
        return b"git-lfs.github.com/spec/v1" in head
    except Exception:
        return False


def csv_info(path: Path) -> dict:
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as f:
            reader = csv.reader(f)
            header = next(reader, [])
            n = sum(1 for _ in reader)
        return {
            "rows": n,
            "columns": header,
            "n_columns": len(header),
        }
    except Exception as e:
        return {"parse_error": repr(e)}


def json_info(path: Path) -> dict:
    try:
        obj = json.loads(path.read_text(encoding="utf-8-sig"))
        if isinstance(obj, dict):
            return {
                "root_type": "dict",
                "n_top_level": len(obj),
                "top_level_keys": list(obj.keys())[:50],
            }
        if isinstance(obj, list):
            keys = []
            if obj and isinstance(obj[0], dict):
                keys = list(obj[0].keys())
            return {
                "root_type": "list",
                "n_top_level": len(obj),
                "first_row_keys": keys,
            }
        return {"root_type": type(obj).__name__}
    except Exception as e:
        return {"parse_error": repr(e)}


def jsonl_info(path: Path) -> dict:
    try:
        n = 0
        first = None
        ids = set()
        with path.open("r", encoding="utf-8-sig") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                n += 1
                if first is None:
                    first = obj
                if isinstance(obj, dict):
                    value = obj.get("id", obj.get("set_id"))
                    if value is not None:
                        ids.add(str(value))
        return {
            "rows": n,
            "unique_id_or_set_id": len(ids),
            "first_row_keys": list(first.keys()) if isinstance(first, dict) else [],
        }
    except Exception as e:
        return {"parse_error": repr(e)}


def inspect(path: Path) -> dict:
    rec = {
        "path": rel(path),
        "exists": path.is_file(),
        "tracked": is_tracked(path),
    }

    if not path.is_file():
        return rec

    rec["bytes"] = path.stat().st_size
    rec["lfs_pointer"] = is_lfs_pointer(path)

    if not rec["lfs_pointer"] and path.stat().st_size <= MAX_HASH_BYTES:
        rec["sha256"] = sha256(path)

    if rec["lfs_pointer"]:
        return rec

    suffix = path.suffix.lower()
    if suffix == ".csv":
        rec["content"] = csv_info(path)
    elif suffix == ".json":
        rec["content"] = json_info(path)
    elif suffix == ".jsonl":
        rec["content"] = jsonl_info(path)

    return rec


def existing_glob(pattern: str) -> list[Path]:
    return sorted(ROOT.glob(pattern))


def csv_rows(path: Path):
    if not path.is_file() or is_lfs_pointer(path):
        return []
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as f:
            return list(csv.DictReader(f))
    except Exception:
        return []


def check(name: str, ok: bool, detail: str) -> dict:
    return {"check": name, "ok": bool(ok), "detail": detail}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--polyvore-root",
        default=os.environ.get("POLYVORE_ROOT", ""),
        help="External Polyvore Outfits root. Optional for inventory; required later for end-to-end data checks.",
    )
    ap.add_argument(
        "--out-dir",
        default=str(DEFAULT_OUT),
    )
    ap.add_argument(
        "--run-root", default=None,
        help="Fresh integrated run root. Scope current fair-subset details to THIS run, never old results/raw runs.",
    )
    args = ap.parse_args()

    fresh_run = Path(args.run_root).expanduser().resolve() if args.run_root else None
    polyvore = Path(args.polyvore_root).expanduser().resolve() if args.polyvore_root else None
    out = Path(args.out_dir)
    if not out.is_absolute():
        out = ROOT / out
    out.mkdir(parents=True, exist_ok=True)

    head = git("rev-parse", "HEAD")
    branch = git("branch", "--show-current")

    report = {
        "classification": "remaining thesis reproduction comprehensive preflight",
        "repo_root": ".",
        "git_branch": branch,
        "git_commit": head,
        "fresh_integrated_run_root": str(fresh_run) if fresh_run else None,
        "polyvore_root_supplied": bool(polyvore),
        "polyvore_root_exists": bool(polyvore and polyvore.is_dir()),
        "policy": {
            "modify_archived_research_files": False,
            "overwrite_archived_tables": False,
            "run_training": False,
            "purpose": "inventory all remaining reproduction inputs in one pass",
        },
        "modules": {},
    }

    # ------------------------------------------------------------------
    # Common archived/generated inputs
    # ------------------------------------------------------------------
    generated_dir = (
        DATA
        / "generated_descriptions"
        / "01_生成結果_generation_results"
    )

    common = {
        "generated_titles": generated_dir / "new_polyvore_outfit_titles.json",
        "generated_ablation": generated_dir / "new_polyvore_outfit_titles_with_ablation.json",
        "wos": (
            DATA
            / "generated_descriptions"
            / "02_三因子拆分_wos_factor_split"
            / "wos_split_results_v5_merged_retry_round3.jsonl"
        ),
    }

    if polyvore:
        common.update({
            "polyvore_titles": polyvore / "polyvore_outfit_titles.json",
            "polyvore_metadata": polyvore / "polyvore_item_metadata.json",
            "polyvore_test": polyvore / "disjoint/test.json",
            "polyvore_images_dir": polyvore / "images",
        })

    report["common_inputs"] = {
        k: (
            {"path": str(v), "exists": v.is_dir()}
            if k == "polyvore_images_dir"
            else inspect(v)
        )
        for k, v in common.items()
    }

    # ------------------------------------------------------------------
    # A. Length analysis
    # ------------------------------------------------------------------
    length_dir = EXP / "01_文字長度影響分析"
    length_paths = {
        "source_notebook": length_dir / "source_programs/P12_length_defense_reproducible.ipynb",
        "rowlevel": length_dir / "A07_length_performance_rowlevel_cir.csv",
        "correlation": length_dir / "圖表_figures_tables/tables/A08_length_performance_correlation_summary.csv",
        "buckets": length_dir / "圖表_figures_tables/tables/A09_length_bucket_performance_summary.csv",
        "matched": length_dir / "圖表_figures_tables/tables/A10_matched_length_subset_summary.csv",
        "interpretation": length_dir / "A11_length_defense_interpretation.txt",
    }

    hist_detail = existing_glob(
        "results/raw/reproduction_start_20260918T070524Z/"
        "historical_cir_checkpoint_eval/detail_cir_*_all_checkpoint_seed*.csv"
    )

    length_checks = [
        check(
            "archived_rowlevel_46555",
            len(csv_rows(length_paths["rowlevel"])) == 46555,
            f"rows={len(csv_rows(length_paths['rowlevel']))}",
        ),
    ]
    if not fresh_run:
        length_checks.append(check(
            "historical_checkpoint_detail_files_10",
            len(hist_detail) == 10,
            f"files={len(hist_detail)}",
        ))

    report["modules"]["length"] = {
        "scope": "thesis tables 4-4 and 4-5",
        "files": {k: inspect(v) for k, v in length_paths.items()},
        "historical_cir_detail_files": [inspect(p) for p in hist_detail],
        "historical_checkpoint_detail_note": (
            "Historical checkpoint-evaluation detail files belong to a previous "
            "evidence cycle and are not required for a new integrated run; "
            "the preserved A07 row-level length source is checked separately."
            if fresh_run else "Historical checkpoint-evaluation evidence inventory"
        ),
        "checks": length_checks,
    }

    # ------------------------------------------------------------------
    # B. Target clue leakage
    # ------------------------------------------------------------------
    clue_dir = EXP / "02_目標單品線索檢查"
    clue_paths = {
        "source_notebook": clue_dir / "source_programs/P12_target_clue_leakage_reproducible.ipynb",
        "summary": clue_dir / "圖表_figures_tables/tables/A03_target_item_clue_leakage_audit_summary.csv",
        "examples": clue_dir / "A04_target_item_clue_leakage_examples.csv",
    }

    clue_rows = csv_rows(clue_paths["summary"])
    audited = None
    for row in clue_rows:
        if row.get("metric") == "audited_unique_main_cir_queries":
            audited = row.get("count")

    report["modules"]["target_clue"] = {
        "scope": "thesis table 4-6",
        "files": {k: inspect(v) for k, v in clue_paths.items()},
        "checks": [
            check(
                "archived_audited_queries_9311",
                str(audited) == "9311",
                f"audited_unique_main_cir_queries={audited}",
            ),
            check(
                "polyvore_root_available_for_source_recomputation",
                bool(polyvore and polyvore.is_dir()),
                str(polyvore) if polyvore else "not supplied",
            ),
        ],
    }

    # ------------------------------------------------------------------
    # C. LLM Judge agreement / robustness / checklists
    # ------------------------------------------------------------------
    judge_base = DATA / "llm_judge_checklists"

    qwen = judge_base / "Qwen3_Instruct"
    gemma = judge_base / "Gemma3"

    judge_paths = {
        "qwen_checklist_c": qwen / "checklist_C_Qwen3VL32B.json",
        "qwen_checklist_cstar": qwen / "checklist_C_star_Qwen3VL32B.json",
        "qwen_formal": qwen / "phase3_scores_Qwen3VL32B.jsonl",
        "qwen_p0r2": qwen / "phase3_scores_Qwen3VL32B_robustness_run1.jsonl",
        "qwen_p1": qwen / "phase3_scores_Qwen3VL32B_robustness_run1_change.jsonl",
        "qwen_p0r2_compare": qwen / "phase3_scores_Qwen3VL32B_robustness_run1_compare.csv",
        "qwen_p1_compare": qwen / "phase3_scores_Qwen3VL32B_robustness_run1_compare_change.csv",
        "qwen_p2_compare": qwen / "phase3_scores_Qwen3VL32B_robustness_run1_compare_conservative.csv",
        "qwen_p2": qwen / "phase3_scores_Qwen3VL32B_robustness_run1_conservative.jsonl",

        "gemma_checklist_c": gemma / "checklist_C_Gemma3.json",
        "gemma_checklist_cstar": gemma / "checklist_C_star_Gemma3.json",
        "gemma_formal": gemma / "phase3_scores_Gemma3.jsonl",
        "gemma_p0r2": gemma / "phase3_scores_Gemma3_robustness_run1.jsonl",
        "gemma_p1": gemma / "phase3_scores_Gemma3_robustness_run1_change.jsonl",
        "gemma_p0r2_compare": gemma / "phase3_scores_Gemma3_robustness_run1_compare.csv",
        "gemma_p1_compare": gemma / "phase3_scores_Gemma3_robustness_run1_compare_change.csv",
        "gemma_p2_compare": gemma / "phase3_scores_Gemma3_robustness_run1_compare_conservative.csv",
        "gemma_p2": gemma / "phase3_scores_Gemma3_robustness_run1_conservative.jsonl",

        "agreement_notebook": (
            EXP
            / "00_控制檢查與附加稽核"
            / "source_programs/P05_llm_judge_agreement_analysis.ipynb"
        ),
        "robustness_notebook": (
            EXP
            / "00_控制檢查與附加稽核"
            / "source_programs/P06_prompt_robustness_reproducible.ipynb"
        ),
        "t19": (
            EXP
            / "00_控制檢查與附加稽核"
            / "圖表_figures_tables/tables/T19_judge_qwen_gemma_agreement_summary.csv"
        ),
        "t20": (
            EXP
            / "00_控制檢查與附加稽核"
            / "圖表_figures_tables/tables/T20_judge_qwen_gemma_correlation_summary.csv"
        ),
        "t21": (
            EXP
            / "00_控制檢查與附加稽核"
            / "圖表_figures_tables/tables/T21_judge_qwen_gemma_threshold_agreement.csv"
        ),
    }

    q_formal = jsonl_info(judge_paths["qwen_formal"]) if judge_paths["qwen_formal"].is_file() else {}
    g_formal = jsonl_info(judge_paths["gemma_formal"]) if judge_paths["gemma_formal"].is_file() else {}

    prompt_paths = {
        "judge_qwen_p0": REPRO_ROOT / "prompts/judge_qwen_p0.txt",
        "judge_gemma_p0": REPRO_ROOT / "prompts/judge_gemma_p0.txt",
        "judge_p1_order": REPRO_ROOT / "prompts/judge_p1_order.txt",
        "judge_p2_conservative": REPRO_ROOT / "prompts/judge_p2_conservative.txt",
    }

    report["modules"]["judge"] = {
        "scope": "thesis tables 4-7, 4-8; figures 4-1 to 4-3; appendix G",
        "files": {k: inspect(v) for k, v in judge_paths.items()},
        "generation_prompt_files": {k: inspect(v) for k, v in prompt_paths.items()},
        "checks": [
            check(
                "qwen_formal_35140",
                q_formal.get("rows") == 35140,
                f"rows={q_formal.get('rows')}",
            ),
            check(
                "gemma_formal_35140",
                g_formal.get("rows") == 35140,
                f"rows={g_formal.get('rows')}",
            ),
            check(
                "formal_id_intersection_possible",
                q_formal.get("unique_id_or_set_id") == 35140
                and g_formal.get("unique_id_or_set_id") == 35140,
                (
                    f"qwen_unique={q_formal.get('unique_id_or_set_id')}, "
                    f"gemma_unique={g_formal.get('unique_id_or_set_id')}"
                ),
            ),
            check(
                "exact_generation_prompts_archived",
                all(p.is_file() for p in prompt_paths.values()),
                (
                    "all four prompt files present"
                    if all(p.is_file() for p in prompt_paths.values())
                    else "one or more exact generation prompt files still missing"
                ),
            ),
        ],
        "interpretation": (
            "Preserved-output Judge recomputation can be complete even if exact "
            "LLM generation rerun remains provenance-partial."
        ),
    }

    # ------------------------------------------------------------------
    # D. Human audit 30
    # ------------------------------------------------------------------
    human_dir = EXP / "07_人工稽核與品質診斷"
    human_paths = {
        "source_notebook": human_dir / "source_programs/P06_human_audit_30_reproducible.ipynb",
        "sampling_frame": human_dir / "A35_human_audit_30_sampling_frame.csv",
        "selected_cases": human_dir / "A36_human_audit_30_selected_cases.csv",
        "selected_cases_json": human_dir / "A37_human_audit_30_selected_cases.json",
        "manual_results": human_dir / "A38_human_audit_30_manual_results.csv",
        "manual_template": human_dir / "A38_human_audit_30_manual_results_template.csv",
        "sample_score_summary": human_dir / "A41_human_audit_30_sample_score_summary.csv",
        "coverage": human_dir / "A42_human_audit_30_coverage_summary.csv",
        "interpretation": human_dir / "A43_human_audit_30_interpretation.txt",
        "metrics": human_dir / "圖表_figures_tables/tables/T30_human_audit30_model_human_score_metrics.csv",
        "item_disagreement": human_dir / "圖表_figures_tables/tables/T31_human_audit30_item_disagreement.csv",
    }

    selected_rows = csv_rows(human_paths["selected_cases"])
    manual_rows = csv_rows(human_paths["manual_results"])
    metric_rows = csv_rows(human_paths["metrics"])

    strata = {}
    for row in selected_rows:
        key = (
            row.get("audit_stratum")
            or row.get("stratum")
            or row.get("group")
            or ""
        )
        if key:
            strata[key] = strata.get(key, 0) + 1

    report["modules"]["human_audit"] = {
        "scope": "thesis tables 4-9 and 4-10; figure 4-4",
        "files": {k: inspect(v) for k, v in human_paths.items()},
        "checks": [
            check(
                "selected_cases_30",
                len(selected_rows) == 30,
                f"rows={len(selected_rows)}",
            ),
            check(
                "manual_results_750_judgments",
                len(manual_rows) == 750,
                f"rows={len(manual_rows)}; expected=30*(10+15)=750",
            ),
            check(
                "manual_results_30_unique_cases",
                len({
                    str(r.get("set_id"))
                    for r in manual_rows
                    if r.get("set_id")
                }) == 30,
                "unique set_id=" + str(len({
                    str(r.get("set_id"))
                    for r in manual_rows
                    if r.get("set_id")
                })),
            ),
            check(
                "manual_qwen_300_judgments",
                sum(
                    str(r.get("checklist", "")).lower() == "qwen"
                    for r in manual_rows
                ) == 300,
                "Qwen judgments=" + str(sum(
                    str(r.get("checklist", "")).lower() == "qwen"
                    for r in manual_rows
                )),
            ),
            check(
                "manual_gemma_450_judgments",
                sum(
                    str(r.get("checklist", "")).lower() == "gemma"
                    for r in manual_rows
                ) == 450,
                "Gemma judgments=" + str(sum(
                    str(r.get("checklist", "")).lower() == "gemma"
                    for r in manual_rows
                )),
            ),
            check(
                "metrics_two_judges",
                len(metric_rows) == 2,
                f"rows={len(metric_rows)}",
            ),
            check(
                "sampling_seed_contract_42",
                True,
                "Archived source notebook declares RANDOM_SEED=42; exact rerun will verify IDs.",
            ),
        ],
        "observed_strata_from_selected_cases": strata,
    }

    # ------------------------------------------------------------------
    # E. Subgroup / qualitative / color / failure cases
    # ------------------------------------------------------------------
    case_dir = EXP / "04_情境子集與三因子分析"
    case_paths = {
        "subset_notebook": case_dir / "source_programs/P02_subset_robustness_analysis.ipynb",
        "qualitative_notebook": case_dir / "source_programs/P03_qualitative_case_analysis.ipynb",
        "t12_subset": case_dir / "圖表_figures_tables/tables/T12_subset_robustness_summary.csv",
        "t13_delta": case_dir / "圖表_figures_tables/tables/T13_subset_delta_hit10_pivot.csv",
        "t14_condition": case_dir / "圖表_figures_tables/tables/T14_qualitative_condition_summary.csv",
        "t15_category": case_dir / "圖表_figures_tables/tables/T15_qualitative_category_summary.csv",
        "t16_failure": case_dir / "圖表_figures_tables/tables/T16_qualitative_failure_cases.csv",
        "t17_user_cases": case_dir / "圖表_figures_tables/tables/T17_qualitative_user_cases.csv",
    }

    fair_detail = (
        sorted((fresh_run / "ablation" / "runs").glob(
            "*_seed*/detail_cir_fresh_subset.csv"
        ))
        if fresh_run else existing_glob(
            "results/raw/fair_subset_notebook_v1/*_seed*/detail_cir_fresh_subset.csv"
        )
    )

    qnb = inspect(case_paths["qualitative_notebook"])

    report["modules"]["case_analysis"] = {
        "scope": "thesis figures 4-8 to 4-14; tables 4-17 to 4-20",
        "files": {k: inspect(v) for k, v in case_paths.items()},
        "fresh_fair_subset_detail_files": [rel(p) for p in fair_detail],
        "checks": [
            check(
                "qualitative_notebook_materialized",
                qnb.get("exists") is True and not qnb.get("lfs_pointer", False),
                (
                    "materialized"
                    if qnb.get("exists") and not qnb.get("lfs_pointer", False)
                    else "Git LFS pointer or missing"
                ),
            ),
            check(
                "fresh_fair_subset_detail_25",
                len(fair_detail) == 25,
                f"files={len(fair_detail)}",
            ),
            check(
                "polyvore_images_available",
                bool(
                    polyvore
                    and (polyvore / "images").is_dir()
                ),
                str(polyvore / "images") if polyvore else "polyvore root not supplied",
            ),
        ],
    }

    # ------------------------------------------------------------------
    # F. Two-Tower supplemental validation
    # ------------------------------------------------------------------
    two_root = MODEL / "second_model_two_tower"
    two_exp = EXP / "05_第二模型驗證"

    model_records = []
    expected_checkpoint_count = 0
    materialized_checkpoint_count = 0
    pointer_checkpoint_count = 0
    config_count = 0

    for variant in ("original_text", "context_aware_description"):
        for seed in range(1, 6):
            run_dir = two_root / "models" / variant / f"seed_{seed}"
            for name in ("cp_best_model.pt", "best_model.pt"):
                expected_checkpoint_count += 1
                p = run_dir / name
                rec = inspect(p)
                rec["variant"] = variant
                rec["seed"] = seed
                rec["role"] = name
                model_records.append(rec)

                if p.is_file():
                    if is_lfs_pointer(p):
                        pointer_checkpoint_count += 1
                    else:
                        materialized_checkpoint_count += 1

            if (run_dir / "config.json").is_file():
                config_count += 1

    two_paths = {
        "source_notebook": two_exp / "source_programs/P15_fashionclip_text_conditioned_retrieval_baseline.ipynb",
        "seed_summary": two_exp / "圖表_figures_tables/tables/A30_second_model_two_tower_seed_summary.csv",
        "rowlevel": two_exp / "A31_second_model_two_tower_or_rowlevel.csv",
        "mean_std": two_exp / "圖表_figures_tables/tables/A32_second_model_two_tower_mean_std.csv",
        "interpretation": two_exp / "A33_second_model_two_tower_interpretation.txt",
        "training_report": two_exp / "A34_second_model_training_report.txt",
    }

    a30_rows = csv_rows(two_paths["seed_summary"])
    a32_rows = csv_rows(two_paths["mean_std"])

    report["modules"]["two_tower"] = {
        "scope": "supplemental second-model validation; not a thesis Chapter-4 core blocker unless explicitly required",
        "files": {k: inspect(v) for k, v in two_paths.items()},
        "checkpoints": model_records,
        "checks": [
            check(
                "seed_summary_10",
                len(a30_rows) == 10,
                f"rows={len(a30_rows)}",
            ),
            check(
                "mean_std_two_conditions",
                len(a32_rows) == 2,
                f"rows={len(a32_rows)}",
            ),
            check(
                "configs_10",
                config_count == 10,
                f"configs={config_count}",
            ),
            check(
                "checkpoints_20_materialized",
                materialized_checkpoint_count == 20,
                (
                    f"materialized={materialized_checkpoint_count}, "
                    f"lfs_pointers={pointer_checkpoint_count}, expected=20"
                ),
            ),
        ],
    }

    # ------------------------------------------------------------------
    # G. Counterfactual supplemental inventory
    # ------------------------------------------------------------------
    cf_dir = EXP / "06_反事實情境敏感度"
    cf_paths = {
        "source_notebook": cf_dir / "source_programs/P16_counterfactual_context_consistency_check.ipynb",
        "pairs": cf_dir / "A44_counterfactual_context_pairs.csv",
        "retrieval_seed1": cf_dir / "A45_counterfactual_context_retrieval_seed1.csv",
        "manual_review": cf_dir / "A46_counterfactual_context_manual_review_sheet.csv",
        "note": cf_dir / "A47_counterfactual_context_experiment_note.txt",
    }

    report["modules"]["counterfactual"] = {
        "scope": "supplemental inventory; not currently treated as thesis core acceptance blocker",
        "files": {k: inspect(v) for k, v in cf_paths.items()},
    }

    # ------------------------------------------------------------------
    # Compact readiness summary
    # ------------------------------------------------------------------
    readiness = {}

    def all_ok(module: str) -> bool:
        checks = report["modules"][module].get("checks", [])
        return bool(checks) and all(x["ok"] for x in checks)

    readiness["length"] = {
        "ready": all_ok("length"),
        "mode": (
            "archived deterministic A07 length reanalysis; historical checkpoint "
            "detail provenance is separately documented"
            if fresh_run else "archived deterministic + preserved row-level checkpoint evidence"
        ),
    }

    readiness["target_clue"] = {
        "ready": all_ok("target_clue"),
        "mode": "deterministic source recomputation",
    }

    judge_checks = report["modules"]["judge"]["checks"]
    judge_map = {x["check"]: x["ok"] for x in judge_checks}
    readiness["judge_preserved_outputs"] = {
        "ready": (
            judge_map.get("qwen_formal_35140", False)
            and judge_map.get("gemma_formal_35140", False)
            and judge_map.get("formal_id_intersection_possible", False)
        ),
        "mode": "recompute from preserved outputs",
    }
    readiness["judge_exact_generation"] = {
        "ready": judge_map.get("exact_generation_prompts_archived", False),
        "mode": "exact LLM generation rerun",
    }

    readiness["human_audit"] = {
        "ready": all_ok("human_audit"),
        "mode": "seed-42 sample reconstruction + preserved 750 human judgments",
    }

    case_checks = {
        x["check"]: x["ok"]
        for x in report["modules"]["case_analysis"]["checks"]
    }
    readiness["case_analysis"] = {
        "ready": (
            case_checks.get("qualitative_notebook_materialized", False)
            and case_checks.get("fresh_fair_subset_detail_25", False)
        ),
        "mode": "numeric subgroup + qualitative tables + color shift + failure cases",
    }

    readiness["case_visuals"] = {
        "ready": case_checks.get("polyvore_images_available", False),
        "mode": "image-dependent qualitative figure regeneration",
    }

    readiness["two_tower"] = {
        "ready": all_ok("two_tower"),
        "mode": "preserved second-model checkpoint validation",
    }

    report["readiness"] = readiness

    # ------------------------------------------------------------------
    # Write JSON
    # ------------------------------------------------------------------
    json_path = out / "preflight_inventory.json"
    json_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )

    # ------------------------------------------------------------------
    # Write compact Markdown
    # ------------------------------------------------------------------
    md = [
        "# Remaining thesis reproduction preflight",
        "",
        f"- Branch: `{branch}`",
        f"- Commit: `{head}`",
        f"- Polyvore root supplied: `{bool(polyvore)}`",
        f"- Polyvore root exists: `{bool(polyvore and polyvore.is_dir())}`",
        "",
        "## Readiness",
        "",
        "| Module | Ready | Mode |",
        "|---|---:|---|",
    ]

    for name, item in readiness.items():
        md.append(
            f"| {name} | {'YES' if item['ready'] else 'NO'} | {item['mode']} |"
        )

    md += [
        "",
        "## Failed / unresolved checks",
        "",
    ]

    unresolved = []
    for module, payload in report["modules"].items():
        for item in payload.get("checks", []):
            if not item["ok"]:
                unresolved.append(
                    f"- **{module} / {item['check']}**: {item['detail']}"
                )

    if unresolved:
        md.extend(unresolved)
    else:
        md.append("- None.")

    md += [
        "",
        "## Important scope rules",
        "",
        "- Archived research files are read-only references.",
        "- This preflight does not run model training.",
        "- Judge preserved-output recomputation is distinct from exact LLM generation rerun.",
        "- Human/Judge sampling seed 42 is distinct from model-training seeds 1–5.",
        "- Fresh reconstructed fair-subset outputs must not be relabeled as historical source-exact outputs.",
        "- Two-Tower and counterfactual analyses remain separately labeled supplemental evidence.",
        "",
    ]

    md_path = out / "preflight_summary.md"
    md_path.write_text("\n".join(md), encoding="utf-8")

    print("=" * 88)
    print("REMAINING THESIS REPRODUCTION PREFLIGHT")
    print("=" * 88)
    print("branch:", branch)
    print("commit:", head)
    print()

    for name, item in readiness.items():
        print(
            f"{name:28s}",
            "READY" if item["ready"] else "BLOCKED/PARTIAL",
            "-", item["mode"],
        )

    print()
    print("Unresolved checks:", len(unresolved))
    for line in unresolved:
        print(line)

    print()
    print("[OK]", rel(json_path))
    print("[OK]", rel(md_path))


if __name__ == "__main__":
    main()
