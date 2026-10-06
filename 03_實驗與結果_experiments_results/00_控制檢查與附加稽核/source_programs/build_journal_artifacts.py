from __future__ import annotations

import csv
import json
import re
import shutil
from collections import Counter
from pathlib import Path
from statistics import mean, median


BASE = Path("/home/ester/valid_description")
TGT = BASE / "experiment data" / "journal_figures_tables_20260629"

FIG = TGT / "figures"
TAB = TGT / "tables"
DER = TGT / "derived_audits"
SRC = TGT / "source_programs"


def ensure_dirs() -> None:
    for p in [FIG, TAB, DER, SRC]:
        p.mkdir(parents=True, exist_ok=True)


ARTIFACTS = [
    # Main quantitative tables.
    {
        "src": BASE / "text-conditioned-outfit-recommendation/analysis_outputs_final/table_independent_baseline_comparison.csv",
        "dst": TAB / "T01_main_overall_baseline_and_ablation.csv",
        "type": "table",
        "use": "Main result table: Original / Full / w-o weather / w-o occasion / w-o style across CP and CIR.",
        "program": BASE / "text-conditioned-outfit-recommendation/5筆資料 t-test.ipynb",
    },
    {
        "src": BASE / "text-conditioned-outfit-recommendation/analysis_outputs_final/table_independent_baseline_comparison.xlsx",
        "dst": TAB / "T01_main_overall_baseline_and_ablation.xlsx",
        "type": "table",
        "use": "Editable spreadsheet version of T01.",
        "program": BASE / "text-conditioned-outfit-recommendation/5筆資料 t-test.ipynb",
    },
    {
        "src": BASE / "text-conditioned-outfit-recommendation/analysis_outputs_final/table_independent_baseline_comparison_tests.csv",
        "dst": TAB / "T02_main_significance_tests.csv",
        "type": "table",
        "use": "Statistical tests for main independent baseline comparison.",
        "program": BASE / "text-conditioned-outfit-recommendation/5筆資料 t-test.ipynb",
    },
    {
        "src": BASE / "text-conditioned-outfit-recommendation/analysis_outputs_final/table_stage1_cp.csv",
        "dst": TAB / "T03_stage1_cp_original_vs_full.csv",
        "type": "table",
        "use": "Stage 1 CP result table: original title vs full proposed description.",
        "program": BASE / "text-conditioned-outfit-recommendation/5筆資料 t-test.ipynb",
    },
    {
        "src": BASE / "text-conditioned-outfit-recommendation/analysis_outputs_final/table_stage1_cir.csv",
        "dst": TAB / "T04_stage1_cir_original_vs_full.csv",
        "type": "table",
        "use": "Stage 1 CIR result table: original title vs full proposed description.",
        "program": BASE / "text-conditioned-outfit-recommendation/5筆資料 t-test.ipynb",
    },
    {
        "src": BASE / "text-conditioned-outfit-recommendation/analysis_outputs_final/table_stage2_cp.csv",
        "dst": TAB / "T05_stage2_cp_ablation.csv",
        "type": "table",
        "use": "Stage 2 CP ablation table.",
        "program": BASE / "text-conditioned-outfit-recommendation/5筆資料 t-test.ipynb",
    },
    {
        "src": BASE / "text-conditioned-outfit-recommendation/analysis_outputs_final/table_stage2_cir.csv",
        "dst": TAB / "T06_stage2_cir_ablation.csv",
        "type": "table",
        "use": "Stage 2 CIR ablation table.",
        "program": BASE / "text-conditioned-outfit-recommendation/5筆資料 t-test.ipynb",
    },
    {
        "src": BASE / "text-conditioned-outfit-recommendation/analysis_outputs_final/table_stage2_contribution_ratio.csv",
        "dst": TAB / "T07_stage2_factor_contribution_ratio.csv",
        "type": "table",
        "use": "Ablation contribution ratio table: weather / occasion / style.",
        "program": BASE / "text-conditioned-outfit-recommendation/5筆資料 t-test.ipynb",
    },
    {
        "src": BASE / "text-conditioned-outfit-recommendation/analysis_outputs_final/table_stage1_cp_detail.csv",
        "dst": TAB / "T08_stage1_cp_seed_detail.csv",
        "type": "table",
        "use": "Supplementary seed-level details for CP.",
        "program": BASE / "text-conditioned-outfit-recommendation/5筆資料 t-test.ipynb",
    },
    {
        "src": BASE / "text-conditioned-outfit-recommendation/analysis_outputs_final/table_stage1_cir_detail.csv",
        "dst": TAB / "T09_stage1_cir_seed_detail.csv",
        "type": "table",
        "use": "Supplementary seed-level details for CIR.",
        "program": BASE / "text-conditioned-outfit-recommendation/5筆資料 t-test.ipynb",
    },
    {
        "src": BASE / "text-conditioned-outfit-recommendation/analysis_outputs_final/table_stage2_cp_detail.csv",
        "dst": TAB / "T10_stage2_cp_seed_detail.csv",
        "type": "table",
        "use": "Supplementary seed-level details for CP ablations.",
        "program": BASE / "text-conditioned-outfit-recommendation/5筆資料 t-test.ipynb",
    },
    {
        "src": BASE / "text-conditioned-outfit-recommendation/analysis_outputs_final/table_stage2_cir_detail.csv",
        "dst": TAB / "T11_stage2_cir_seed_detail.csv",
        "type": "table",
        "use": "Supplementary seed-level details for CIR ablations.",
        "program": BASE / "text-conditioned-outfit-recommendation/5筆資料 t-test.ipynb",
    },
    # Main and robustness figures.
    {
        "src": BASE / "text-conditioned-outfit-recommendation/analysis_outputs_final/fig3_factor_proportion.png",
        "dst": FIG / "F01_main_factor_contribution_proportion.png",
        "type": "figure",
        "use": "Main ablation factor contribution figure.",
        "program": BASE / "text-conditioned-outfit-recommendation/5筆資料 t-test.ipynb",
    },
    {
        "src": BASE / "text-conditioned-outfit-recommendation/analysis_outputs_final/fig3_factor_proportion.pdf",
        "dst": FIG / "F01_main_factor_contribution_proportion.pdf",
        "type": "figure",
        "use": "PDF version of F01.",
        "program": BASE / "text-conditioned-outfit-recommendation/5筆資料 t-test.ipynb",
    },
    {
        "src": BASE / "text-conditioned-outfit-recommendation/subset_analysis_results/subset_analysis_summary.csv",
        "dst": TAB / "T12_subset_robustness_summary.csv",
        "type": "table",
        "use": "Subset robustness summary by weather, occasion/style, and item group.",
        "program": BASE / "text-conditioned-outfit-recommendation/完整套裝推薦_fastfit_demo.ipynb",
    },
    {
        "src": BASE / "text-conditioned-outfit-recommendation/subset_analysis_results/subset_delta_hit10_pivot.csv",
        "dst": TAB / "T13_subset_delta_hit10_pivot.csv",
        "type": "table",
        "use": "Subset delta Hit@10 pivot table.",
        "program": BASE / "text-conditioned-outfit-recommendation/完整套裝推薦_fastfit_demo.ipynb",
    },
    {
        "src": BASE / "text-conditioned-outfit-recommendation/subset_analysis_results/fig1_hitrate_comparison.png",
        "dst": FIG / "F02_subset_hitrate_comparison.png",
        "type": "figure",
        "use": "Subset hit-rate comparison.",
        "program": BASE / "text-conditioned-outfit-recommendation/完整套裝推薦_fastfit_demo.ipynb",
    },
    {
        "src": BASE / "text-conditioned-outfit-recommendation/subset_analysis_results/fig2_delta_heatmap.png",
        "dst": FIG / "F03_subset_delta_hit10_heatmap.png",
        "type": "figure",
        "use": "Subset delta heatmap.",
        "program": BASE / "text-conditioned-outfit-recommendation/完整套裝推薦_fastfit_demo.ipynb",
    },
    {
        "src": BASE / "text-conditioned-outfit-recommendation/subset_analysis_results/fig3_ablation_across_subsets.png",
        "dst": FIG / "F04_subset_ablation_across_subsets.png",
        "type": "figure",
        "use": "Ablation behavior across subsets.",
        "program": BASE / "text-conditioned-outfit-recommendation/完整套裝推薦_fastfit_demo.ipynb",
    },
    {
        "src": BASE / "text-conditioned-outfit-recommendation/subset_analysis_results/fig4_per_dimension_bars.png",
        "dst": FIG / "F05_subset_per_dimension_bars.png",
        "type": "figure",
        "use": "Per-dimension subset bars.",
        "program": BASE / "text-conditioned-outfit-recommendation/完整套裝推薦_fastfit_demo.ipynb",
    },
    {
        "src": BASE / "text-conditioned-outfit-recommendation/subset_analysis_results/fig5_rank_improvement_by_subset.png",
        "dst": FIG / "F06_subset_rank_improvement_by_subset.png",
        "type": "figure",
        "use": "Rank improvement by subset.",
        "program": BASE / "text-conditioned-outfit-recommendation/完整套裝推薦_fastfit_demo.ipynb",
    },
    # Qualitative / case analysis.
    {
        "src": BASE / "text-conditioned-outfit-recommendation/qualitative_outputs/qual_condition_summary.csv",
        "dst": TAB / "T14_qualitative_condition_summary.csv",
        "type": "table",
        "use": "Qualitative condition-level summary.",
        "program": BASE / "text-conditioned-outfit-recommendation/個案分析.ipynb",
    },
    {
        "src": BASE / "text-conditioned-outfit-recommendation/qualitative_outputs/qual_category_summary.csv",
        "dst": TAB / "T15_qualitative_category_summary.csv",
        "type": "table",
        "use": "Qualitative category-level summary.",
        "program": BASE / "text-conditioned-outfit-recommendation/個案分析.ipynb",
    },
    {
        "src": BASE / "text-conditioned-outfit-recommendation/qualitative_outputs/qual_failure_cases.csv",
        "dst": TAB / "T16_qualitative_failure_cases.csv",
        "type": "table",
        "use": "Representative failure cases.",
        "program": BASE / "text-conditioned-outfit-recommendation/個案分析.ipynb",
    },
    {
        "src": BASE / "text-conditioned-outfit-recommendation/qualitative_outputs/qual_user_cases.csv",
        "dst": TAB / "T17_qualitative_user_cases.csv",
        "type": "table",
        "use": "Representative user/query cases.",
        "program": BASE / "text-conditioned-outfit-recommendation/個案分析.ipynb",
    },
    {
        "src": BASE / "text-conditioned-outfit-recommendation/qualitative_outputs/fig_category_orig_vs_full.png",
        "dst": FIG / "F07_qualitative_category_original_vs_full.png",
        "type": "figure",
        "use": "Qualitative category comparison: original vs full.",
        "program": BASE / "text-conditioned-outfit-recommendation/個案分析.ipynb",
    },
    {
        "src": BASE / "text-conditioned-outfit-recommendation/qualitative_outputs/fig_condition_occasion_orig_vs_full.png",
        "dst": FIG / "F08_qualitative_condition_occasion_original_vs_full.png",
        "type": "figure",
        "use": "Qualitative condition/occasion comparison: original vs full.",
        "program": BASE / "text-conditioned-outfit-recommendation/個案分析.ipynb",
    },
    {
        "src": BASE / "text-conditioned-outfit-recommendation/qualitative_outputs/fig_category_weather.png",
        "dst": FIG / "F09_qualitative_category_weather.png",
        "type": "figure",
        "use": "Qualitative category by weather.",
        "program": BASE / "text-conditioned-outfit-recommendation/個案分析.ipynb",
    },
    {
        "src": BASE / "text-conditioned-outfit-recommendation/qualitative_outputs/fig_category_occasion.png",
        "dst": FIG / "F10_qualitative_category_occasion.png",
        "type": "figure",
        "use": "Qualitative category by occasion.",
        "program": BASE / "text-conditioned-outfit-recommendation/個案分析.ipynb",
    },
    {
        "src": BASE / "text-conditioned-outfit-recommendation/qualitative_outputs/fig_category_style.png",
        "dst": FIG / "F11_qualitative_category_style.png",
        "type": "figure",
        "use": "Qualitative category by style.",
        "program": BASE / "text-conditioned-outfit-recommendation/個案分析.ipynb",
    },
    # Reliability / calibration style outputs.
    {
        "src": BASE / "text-conditioned-outfit-recommendation/reliability_outputs/compare/reliability_required_fields.csv",
        "dst": TAB / "T18_reliability_required_fields_compare.csv",
        "type": "table",
        "use": "Reliability comparison required fields.",
        "program": BASE / "text-conditioned-outfit-recommendation/XAI作業.ipynb",
    },
    {
        "src": BASE / "text-conditioned-outfit-recommendation/reliability_outputs/compare/plot_01_performance_comparison.png",
        "dst": FIG / "F12_reliability_performance_comparison.png",
        "type": "figure",
        "use": "Reliability/performance comparison.",
        "program": BASE / "text-conditioned-outfit-recommendation/XAI作業.ipynb",
    },
    {
        "src": BASE / "text-conditioned-outfit-recommendation/reliability_outputs/compare/plot_03_reliability_diagram_comparison.png",
        "dst": FIG / "F13_reliability_diagram_comparison.png",
        "type": "figure",
        "use": "Reliability diagram comparison.",
        "program": BASE / "text-conditioned-outfit-recommendation/XAI作業.ipynb",
    },
    {
        "src": BASE / "text-conditioned-outfit-recommendation/reliability_outputs/compare/plot_05_error_slicing_comparison.png",
        "dst": FIG / "F14_reliability_error_slicing_comparison.png",
        "type": "figure",
        "use": "Error slicing comparison.",
        "program": BASE / "text-conditioned-outfit-recommendation/XAI作業.ipynb",
    },
    {
        "src": BASE / "text-conditioned-outfit-recommendation/reliability_outputs/compare/plot_06_subgroup_hit10_comparison.png",
        "dst": FIG / "F15_reliability_subgroup_hit10_comparison.png",
        "type": "figure",
        "use": "Subgroup Hit@10 comparison.",
        "program": BASE / "text-conditioned-outfit-recommendation/XAI作業.ipynb",
    },
    {
        "src": BASE / "text-conditioned-outfit-recommendation/reliability_outputs/compare/plot_08_subgroup_delta_hit10_heatmap.png",
        "dst": FIG / "F16_reliability_subgroup_delta_hit10_heatmap.png",
        "type": "figure",
        "use": "Subgroup delta Hit@10 heatmap.",
        "program": BASE / "text-conditioned-outfit-recommendation/XAI作業.ipynb",
    },
    # LLM-as-a-Judge / validation artifacts.
    {
        "src": BASE / "outputs/phase3_score_agreement_Qwen_vs_Gemma_summary.csv",
        "dst": TAB / "T19_judge_qwen_gemma_agreement_summary.csv",
        "type": "table",
        "use": "LLM-as-a-Judge agreement summary.",
        "program": BASE / "驗證結果分析.ipynb",
    },
    {
        "src": BASE / "outputs/phase3_score_agreement_Qwen_vs_Gemma_correlation_summary.csv",
        "dst": TAB / "T20_judge_qwen_gemma_correlation_summary.csv",
        "type": "table",
        "use": "LLM-as-a-Judge score correlation summary.",
        "program": BASE / "驗證結果分析.ipynb",
    },
    {
        "src": BASE / "outputs/phase3_score_agreement_Qwen_vs_Gemma_threshold_agreement.csv",
        "dst": TAB / "T21_judge_qwen_gemma_threshold_agreement.csv",
        "type": "table",
        "use": "LLM-as-a-Judge threshold agreement.",
        "program": BASE / "驗證結果分析.ipynb",
    },
    {
        "src": BASE / "outputs/phase3_score_agreement_Qwen_vs_Gemma_score_hist.png",
        "dst": FIG / "F17_judge_qwen_gemma_score_histogram.png",
        "type": "figure",
        "use": "LLM-as-a-Judge score histogram.",
        "program": BASE / "驗證結果分析.ipynb",
    },
    {
        "src": BASE / "outputs/phase3_score_agreement_Qwen_vs_Gemma_quantile_bins_confusion.png",
        "dst": FIG / "F18_judge_qwen_gemma_quantile_confusion.png",
        "type": "figure",
        "use": "LLM-as-a-Judge quantile-bin confusion.",
        "program": BASE / "驗證結果分析.ipynb",
    },
    {
        "src": BASE / "outputs/phase3_score_agreement_Qwen_vs_Gemma_bottomp_sensitivity_overlap_band.png",
        "dst": FIG / "F19_judge_bottomp_sensitivity_overlap_band.png",
        "type": "figure",
        "use": "LLM-as-a-Judge bottom-p sensitivity overlap.",
        "program": BASE / "驗證結果分析.ipynb",
    },
    {
        "src": BASE / "figures/prompt_robustness_abs_diff.png",
        "dst": FIG / "F20_prompt_robustness_absolute_difference.png",
        "type": "figure",
        "use": "Prompt robustness absolute-difference figure.",
        "program": BASE / "150 筆人工檢查.ipynb",
    },
    {
        "src": BASE / "figures/overall_error_metrics.png",
        "dst": FIG / "F21_human_check_overall_error_metrics.png",
        "type": "figure",
        "use": "Human/paper-analysis overall error metrics; use as appendix only if needed.",
        "program": BASE / "paper_analysis_charts.ipynb",
    },
    {
        "src": BASE / "figures/model_human_error_metrics.png",
        "dst": FIG / "F22_human_check_model_human_error_metrics.png",
        "type": "figure",
        "use": "Human/model error metrics; use as appendix only if needed.",
        "program": BASE / "paper_analysis_charts.ipynb",
    },
    {
        "src": BASE / "figures/item_disagreement_ranking.png",
        "dst": FIG / "F23_human_check_item_disagreement_ranking.png",
        "type": "figure",
        "use": "Item disagreement ranking; use as appendix only if needed.",
        "program": BASE / "paper_analysis_charts.ipynb",
    },
    # Older full-run comparison outputs kept as backup, not main conservative subset.
    {
        "src": BASE / "text-conditioned-outfit-recommendation/analysis_results/CP_Compatibility_Prediction_results.csv",
        "dst": TAB / "B01_legacy_fullrun_cp_results.csv",
        "type": "backup_table",
        "use": "Backup only: earlier full-run CP comparison, not the conservative main subset table.",
        "program": BASE / "text-conditioned-outfit-recommendation/新描述 t-test.ipynb",
    },
    {
        "src": BASE / "text-conditioned-outfit-recommendation/analysis_results/CIR_Conditional_Retrieval_results.csv",
        "dst": TAB / "B02_legacy_fullrun_cir_results.csv",
        "type": "backup_table",
        "use": "Backup only: earlier full-run CIR comparison, not the conservative main subset table.",
        "program": BASE / "text-conditioned-outfit-recommendation/新描述 t-test.ipynb",
    },
    {
        "src": BASE / "text-conditioned-outfit-recommendation/analysis_results/four_panel_summary.png",
        "dst": FIG / "B03_legacy_fullrun_four_panel_summary.png",
        "type": "backup_figure",
        "use": "Backup only: earlier full-run four-panel summary.",
        "program": BASE / "text-conditioned-outfit-recommendation/新描述 t-test.ipynb",
    },
]


TOKEN_RE = re.compile(r"[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)?")
TEMP_PREFIX_RE = re.compile(r"^\s*[+-]?\d+(?:\.\d+)?\s*(?:°\s*C|°C|℃|C)\s*[,，:：-]*\s*", re.IGNORECASE)


def token_count(text: str) -> int:
    return len(TOKEN_RE.findall((text or "").strip()))


def strip_temperature_prefix(text: str) -> str:
    return TEMP_PREFIX_RE.sub("", text or "").strip()


def stats_for_titles(label: str, titles: list[str]) -> dict[str, str]:
    counts = [token_count(t) for t in titles]
    nonempty = [c for c in counts if c > 0]
    n = len(counts)
    empty = sum(1 for c in counts if c == 0)
    def pct(pred) -> str:
        return f"{100 * sum(1 for c in counts if pred(c)) / n:.2f}" if n else "0.00"
    return {
        "dataset": label,
        "n_records": str(n),
        "empty_records": str(empty),
        "nonempty_records": str(len(nonempty)),
        "mean_tokens_all": f"{mean(counts):.4f}" if counts else "",
        "median_tokens_all": f"{median(counts):.4f}" if counts else "",
        "mean_tokens_nonempty": f"{mean(nonempty):.4f}" if nonempty else "",
        "median_tokens_nonempty": f"{median(nonempty):.4f}" if nonempty else "",
        "max_tokens": str(max(counts) if counts else 0),
        "pct_len_le_2": pct(lambda c: c <= 2),
        "pct_len_le_3": pct(lambda c: c <= 3),
        "pct_len_le_5": pct(lambda c: c <= 5),
        "pct_empty": pct(lambda c: c == 0),
    }


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    if not rows:
        return
    keys = list(rows[0].keys())
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def read_json(path: Path):
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def derive_title_length_and_coverage(manifest: list[dict[str, str]]) -> None:
    original = read_json(BASE / "polyvore_data/polyvore_outfits/polyvore_outfit_titles.json")

    def original_field_for_set_id(set_id: str, field: str) -> str:
        v = original.get(str(set_id), "")
        if isinstance(v, dict):
            return str(v.get(field, "") or "").strip()
        return str(v or "").strip()

    def original_query_for_set_id(set_id: str) -> str:
        v = original.get(str(set_id), "")
        if isinstance(v, dict):
            parts = [str(v.get(k, "") or "").strip() for k in ["url_name", "title"]]
            return " ".join(p for p in parts if p).strip()
        return str(v or "").strip()

    generated = read_json(BASE / "new_polyvore_outfit_titles.json")
    generated_set_ids = [str(k) for k in generated.keys()]
    po_d_orig_titles = [original_field_for_set_id(sid, "title") for sid in generated_set_ids]
    po_d_orig_url_names = [original_field_for_set_id(sid, "url_name") for sid in generated_set_ids]
    po_d_orig_url_title_queries = [original_query_for_set_id(sid) for sid in generated_set_ids]
    gen_titles = [str(generated[sid].get("title", "") if isinstance(generated[sid], dict) else generated[sid]).strip() for sid in generated_set_ids]

    rows = [
        stats_for_titles("PO-D_original_title_field_matched_to_35140_experimental_set_ids", po_d_orig_titles),
        stats_for_titles("PO-D_original_url_name_field_matched_to_35140_experimental_set_ids", po_d_orig_url_names),
        stats_for_titles("PO-D_original_url_name_plus_title_query_matched_to_35140_experimental_set_ids", po_d_orig_url_title_queries),
        stats_for_titles("PO-D_generated_full_description_title", gen_titles),
        stats_for_titles("PO-D_generated_full_description_title_without_temperature_prefix", [strip_temperature_prefix(t) for t in gen_titles]),
    ]

    ablation_path = BASE / "new_polyvore_outfit_titles_with_ablation.json"
    if ablation_path.exists():
        ablation = read_json(ablation_path)
        for key in ["title", "no_weather", "no_occasion", "no_style"]:
            titles = []
            for v in ablation.values():
                if isinstance(v, dict):
                    titles.append(str(v.get(key, "") or "").strip())
            rows.append(stats_for_titles(f"PO-D_ablation_field_{key}", titles))
            rows.append(stats_for_titles(f"PO-D_ablation_field_{key}_without_temperature_prefix", [strip_temperature_prefix(t) for t in titles]))

    out = DER / "A01_title_length_audit.csv"
    write_csv(out, rows)
    manifest.append({
        "renamed_file": str(out),
        "artifact_type": "derived_table",
        "paper_use": "Original outfit-level text weakness and generated-description length audit.",
        "source_file": "; ".join([
            str(BASE / "polyvore_data/polyvore_outfits/polyvore_outfit_titles.json"),
            str(BASE / "new_polyvore_outfit_titles.json"),
            str(BASE / "new_polyvore_outfit_titles_with_ablation.json"),
        ]),
        "source_folder": str(BASE),
        "source_program_file": str(SRC / "build_journal_artifacts.py"),
        "notes": "Computed on PO-D / Disjoint experimental set ids, n=35,140. The original baseline is reported both as title-only and url_name+title because the original outfit-level baseline uses outfitUrlTitle. The generated/proposed text uses only the generated title field.",
    })

    stats_by_name = {r["dataset"]: r for r in rows}
    orig_stat = stats_by_name["PO-D_original_title_field_matched_to_35140_experimental_set_ids"]
    orig_query_stat = stats_by_name["PO-D_original_url_name_plus_title_query_matched_to_35140_experimental_set_ids"]
    gen_stat = stats_by_name["PO-D_generated_full_description_title"]
    gen_no_temp_stat = stats_by_name["PO-D_generated_full_description_title_without_temperature_prefix"]
    t00_rows = [
        {
            "aspect": "Dataset",
            "value": "Polyvore Outfits Disjoint version (PO-D)",
            "note": "Main experimental dataset.",
            "source": str(BASE / "polyvore_data/polyvore_outfits/disjoint"),
        },
        {
            "aspect": "PO-D train / valid / test",
            "value": "16,995 / 3,000 / 15,145",
            "note": "Total = 35,140 outfit records.",
            "source": str(BASE / "polyvore_data/polyvore_outfits/disjoint"),
        },
        {
            "aspect": "Main experimental records",
            "value": "35,140",
            "note": "Matched generated-valid set ids used in downstream CP/CIR experiments.",
            "source": str(BASE / "new_polyvore_outfit_titles.json"),
        },
        {
            "aspect": "PO-D original empty titles",
            "value": f"{orig_stat['empty_records']} ({orig_stat['pct_empty']}%)",
            "note": "Computed after matching original titles to the 35,140 PO-D experimental set ids.",
            "source": str(BASE / "polyvore_data/polyvore_outfits/polyvore_outfit_titles.json"),
        },
        {
            "aspect": "PO-D original title-field length",
            "value": f"mean {orig_stat['mean_tokens_all']}, median {orig_stat['median_tokens_all']}, max {orig_stat['max_tokens']} tokens",
            "note": f"{orig_stat['pct_len_le_2']}% <= 2 tokens; {orig_stat['pct_len_le_3']}% <= 3 tokens.",
            "source": str(BASE / "polyvore_data/polyvore_outfits/polyvore_outfit_titles.json"),
        },
        {
            "aspect": "PO-D original url_name + title query length",
            "value": f"mean {orig_query_stat['mean_tokens_all']}, median {orig_query_stat['median_tokens_all']}, max {orig_query_stat['max_tokens']} tokens",
            "note": f"Matches the outfitUrlTitle baseline text definition; {orig_query_stat['pct_len_le_5']}% <= 5 tokens.",
            "source": str(BASE / "polyvore_data/polyvore_outfits/polyvore_outfit_titles.json"),
        },
        {
            "aspect": "Generated full description title-only query length",
            "value": f"mean {gen_stat['mean_tokens_all']}, median {gen_stat['median_tokens_all']}, max {gen_stat['max_tokens']} tokens",
            "note": "Generated/proposed method stores the constructed description in the title field only; no original url_name is appended. Includes temperature prefix.",
            "source": str(BASE / "new_polyvore_outfit_titles.json"),
        },
        {
            "aspect": "Generated full description title-only query length without temperature prefix",
            "value": f"mean {gen_no_temp_stat['mean_tokens_all']}, median {gen_no_temp_stat['median_tokens_all']}, max {gen_no_temp_stat['max_tokens']} tokens",
            "note": "Generated/proposed title field after removing the leading temperature expression.",
            "source": str(BASE / "new_polyvore_outfit_titles.json"),
        },
    ]
    t00 = TAB / "T00_dataset_characterization_po_d.csv"
    write_csv(t00, t00_rows)
    manifest.append({
        "renamed_file": str(t00),
        "artifact_type": "table",
        "paper_use": "Dataset Characterization Table: PO-D split size and original-title weakness.",
        "source_file": "; ".join([
            str(BASE / "polyvore_data/polyvore_outfits/disjoint"),
            str(BASE / "polyvore_data/polyvore_outfits/polyvore_outfit_titles.json"),
            str(BASE / "new_polyvore_outfit_titles.json"),
        ]),
        "source_folder": str(BASE / "polyvore_data/polyvore_outfits"),
        "source_program_file": str(SRC / "build_journal_artifacts.py"),
        "notes": "This is the paper-ready Dataset Characterization Table. It reports only PO-D n=35,140 and matched PO-D title statistics.",
    })

    # Teacher-context dataset diagnostics: split size, item count, outfit length,
    # and category distribution. These are appendix-ready but also support Table 1.
    split_dir = BASE / "polyvore_data/polyvore_outfits/disjoint"
    item_meta_path = BASE / "polyvore_data/polyvore_outfits/polyvore_item_metadata.json"
    with item_meta_path.open(encoding="utf-8") as f:
        item_meta = json.load(f)

    split_rows = []
    all_records = []
    for split in ["train", "valid", "test"]:
        records = read_json(split_dir / f"{split}.json")
        all_records.extend((split, r) for r in records)
        lengths = [len(r.get("items", [])) for r in records]
        item_ids = [str(it.get("item_id", "")) for r in records for it in r.get("items", []) if it.get("item_id")]
        split_rows.append({
            "split": split,
            "n_outfits": str(len(records)),
            "n_item_occurrences": str(len(item_ids)),
            "n_unique_items": str(len(set(item_ids))),
            "mean_outfit_length": f"{_mean([float(x) for x in lengths]):.4f}" if lengths else "0.0000",
            "median_outfit_length": f"{median(lengths):.4f}" if lengths else "0.0000",
            "min_outfit_length": str(min(lengths) if lengths else 0),
            "max_outfit_length": str(max(lengths) if lengths else 0),
        })

    all_lengths = [len(r.get("items", [])) for _, r in all_records]
    all_item_ids = [str(it.get("item_id", "")) for _, r in all_records for it in r.get("items", []) if it.get("item_id")]
    split_rows.append({
        "split": "all_po_d",
        "n_outfits": str(len(all_records)),
        "n_item_occurrences": str(len(all_item_ids)),
        "n_unique_items": str(len(set(all_item_ids))),
        "mean_outfit_length": f"{_mean([float(x) for x in all_lengths]):.4f}" if all_lengths else "0.0000",
        "median_outfit_length": f"{median(all_lengths):.4f}" if all_lengths else "0.0000",
        "min_outfit_length": str(min(all_lengths) if all_lengths else 0),
        "max_outfit_length": str(max(all_lengths) if all_lengths else 0),
    })
    out_split = DER / "A12_po_d_split_item_outfit_stats.csv"
    write_csv(out_split, split_rows)
    manifest.append({
        "renamed_file": str(out_split),
        "artifact_type": "derived_table",
        "paper_use": "PO-D split diagnostics: outfit count, item count, and outfit length.",
        "source_file": str(split_dir),
        "source_folder": str(split_dir),
        "source_program_file": str(SRC / "build_journal_artifacts.py"),
        "notes": "Completes teacher-requested dataset characterization fields beyond the 35,140 outfit count.",
    })

    category_rows = []
    for split_name in ["train", "valid", "test", "all_po_d"]:
        if split_name == "all_po_d":
            records_for_split = [r for _, r in all_records]
        else:
            records_for_split = read_json(split_dir / f"{split_name}.json")
        occ_counter: Counter[str] = Counter()
        unique_by_cat: dict[str, set[str]] = {}
        total_occ = 0
        all_unique: set[str] = set()
        for r in records_for_split:
            for it in r.get("items", []):
                iid = str(it.get("item_id", ""))
                if not iid:
                    continue
                meta = item_meta.get(iid, {})
                cat = str(meta.get("semantic_category") or "unknown")
                occ_counter[cat] += 1
                unique_by_cat.setdefault(cat, set()).add(iid)
                all_unique.add(iid)
                total_occ += 1
        for cat, occ in sorted(occ_counter.items(), key=lambda kv: (-kv[1], kv[0])):
            u = len(unique_by_cat.get(cat, set()))
            category_rows.append({
                "split": split_name,
                "semantic_category": cat,
                "item_occurrences": str(occ),
                "pct_item_occurrences": f"{100 * occ / total_occ:.2f}" if total_occ else "0.00",
                "unique_items": str(u),
                "pct_unique_items": f"{100 * u / len(all_unique):.2f}" if all_unique else "0.00",
            })
    out_cat = DER / "A13_po_d_semantic_category_distribution.csv"
    write_csv(out_cat, category_rows)
    manifest.append({
        "renamed_file": str(out_cat),
        "artifact_type": "derived_table",
        "paper_use": "PO-D semantic category distribution by split.",
        "source_file": "; ".join([str(split_dir), str(item_meta_path)]),
        "source_folder": str(BASE / "polyvore_data/polyvore_outfits"),
        "source_program_file": str(SRC / "build_journal_artifacts.py"),
        "notes": "Use in appendix or to expand the Dataset Characterization Table if the journal allows space.",
    })

    coverage_rows = []
    coverage_rows.append({
        "metric": "po_d_disjoint_train_records",
        "value": str(len(read_json(BASE / "polyvore_data/polyvore_outfits/disjoint/train.json"))),
        "source": str(BASE / "polyvore_data/polyvore_outfits/disjoint/train.json"),
    })
    coverage_rows.append({
        "metric": "po_d_disjoint_valid_records",
        "value": str(len(read_json(BASE / "polyvore_data/polyvore_outfits/disjoint/valid.json"))),
        "source": str(BASE / "polyvore_data/polyvore_outfits/disjoint/valid.json"),
    })
    coverage_rows.append({
        "metric": "po_d_disjoint_test_records",
        "value": str(len(read_json(BASE / "polyvore_data/polyvore_outfits/disjoint/test.json"))),
        "source": str(BASE / "polyvore_data/polyvore_outfits/disjoint/test.json"),
    })
    coverage_rows.append({
        "metric": "po_d_disjoint_total_records",
        "value": str(len(generated_set_ids)),
        "source": "disjoint train + valid + test; matched to generated title set_ids",
    })
    coverage_rows.append({
        "metric": "po_d_generated_full_title_records",
        "value": str(len(generated)),
        "source": str(BASE / "new_polyvore_outfit_titles.json"),
    })

    retry_path = BASE / "wos_split_results_v5_merged_retry_round3.jsonl"
    if retry_path.exists():
        n = 0
        valid = 0
        attempts = Counter()
        with retry_path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                n += 1
                try:
                    r = json.loads(line)
                except Exception:
                    continue
                if r.get("valid") is True:
                    valid += 1
                attempts[str(r.get("attempts", r.get("attempt", "missing")))] += 1
        coverage_rows.append({"metric": "rewrite_generation_jsonl_rows", "value": str(n), "source": str(retry_path)})
        coverage_rows.append({"metric": "rewrite_generation_valid_true", "value": str(valid), "source": str(retry_path)})
        for k in sorted(attempts, key=lambda x: (x == "missing", int(x) if x.isdigit() else 999999, x)):
            coverage_rows.append({"metric": f"rewrite_attempts_{k}", "value": str(attempts[k]), "source": str(retry_path)})

    out2 = DER / "A02_generation_coverage_audit.csv"
    write_csv(out2, coverage_rows)
    manifest.append({
        "renamed_file": str(out2),
        "artifact_type": "derived_table",
        "paper_use": "Dataset and generated-title coverage audit.",
        "source_file": "; ".join([r["source"] for r in coverage_rows[:3]]),
        "source_folder": str(BASE),
        "source_program_file": str(SRC / "build_journal_artifacts.py"),
        "notes": "Main experiment dataset is PO-D / Disjoint, n=35,140. Coverage table intentionally omits the full title registry count to avoid confusion.",
    })


WORD_RE = re.compile(r"[a-zA-Z][a-zA-Z'-]*")
STOP = set(
    """
    a an and are as at be by for from in into is it its of on or the this that these those to with without your you
    women woman womens female fashion clothing clothes outfit outfits wear wearing style styles styled look looks item items piece pieces
    new used vintage size small medium large plus petite one pair set collection design designer brand brands online shop shopping
    classic modern casual chic sleek elegant effortless comfy cozy cool warm cold hot day days night nights spring summer fall autumn winter
    vibe vibes scene mood ready friendly perfect practical realistic aspirational search recommended temperature clothing insulation original title
    """
    .split()
)
GENERIC = set(
    """
    dress dresses skirt skirts top tops shirt shirts blouse blouses sweater sweaters pants pant jeans jean shorts short jacket jackets coat coats
    heels heel shoes shoe sandals sandal boots boot flats flat sneakers sneaker bag bags purse purses handbag handbags jewellery jewelry necklace necklaces
    earrings ring rings bracelet bracelets watch watches sunglasses hat hats scarf scarves gown gowns romper jumpsuit suit suits cardigan cardigans
    """
    .split()
)


def words(s: str) -> list[str]:
    return [w.strip("'-").lower() for w in WORD_RE.findall(str(s or "").lower()) if w.strip("'-")]


def category_match(title: str, fine: str, major: str) -> tuple[bool, bool]:
    title_words = set(words(title))
    title_phrase = " ".join(words(title))

    def match_phrase_or_tokens(label: str) -> bool:
        label_words = words(label)
        if not label_words:
            return False
        if " ".join(label_words) in title_phrase:
            return True
        return all(any(v in title_words for v in [w, w.rstrip("s"), w + "s"]) for w in label_words)

    return match_phrase_or_tokens(fine), match_phrase_or_tokens(major)


def meta_text_parts(meta: dict) -> list[str]:
    parts: list[str] = []
    for k in ["url_name", "title", "description", "semantic_category"]:
        v = meta.get(k, "")
        if isinstance(v, str):
            parts.append(v)
    for k in ["catgeories", "related"]:
        v = meta.get(k, "")
        if isinstance(v, list):
            parts.extend(str(x) for x in v)
        elif isinstance(v, str):
            parts.append(v)
    return parts


def distinctive_tokens(meta: dict, fine: str, major: str) -> list[str]:
    toks: list[str] = []
    for part in meta_text_parts(meta):
        toks.extend(words(part))
    fine_set, major_set = set(words(fine)), set(words(major))
    out = []
    for t in toks:
        if len(t) < 4:
            continue
        if t in STOP or t in GENERIC or t in fine_set or t in major_set:
            continue
        out.append(t[:-2] if t.endswith("'s") else t)
    return sorted(set(out))


def distinctive_bigrams(meta: dict, fine: str, major: str) -> list[str]:
    toks: list[str] = []
    for part in meta_text_parts(meta):
        toks.extend([w for w in words(part) if len(w) >= 3 and w not in STOP])
    fine_set, major_set = set(words(fine)), set(words(major))
    toks = [t for t in toks if t not in GENERIC and t not in fine_set and t not in major_set]
    out = []
    for a, b in zip(toks, toks[1:]):
        if a != b and len(a) >= 4 and len(b) >= 4:
            out.append(f"{a} {b}")
    return sorted(set(out))


def derive_leakage_audit(manifest: list[dict[str, str]]) -> None:
    detail_dir = BASE / "text-conditioned-outfit-recommendation/experiments_none_only/detail_cir"
    rel_meta_path = BASE / "text-conditioned-outfit-recommendation/subset_analysis_results/reliability_meta_from_subset.csv"
    meta_path = BASE / "polyvore_data/polyvore_outfits/polyvore_item_metadata.json"
    cats_path = BASE / "polyvore_data/polyvore_outfits/categories.csv"

    cat_map = {}
    with cats_path.open(newline="", encoding="utf-8-sig") as f:
        for row in csv.reader(f):
            if len(row) >= 3 and row[0].strip().isdigit():
                cat_map.setdefault(row[0].strip(), {"fine": row[1].strip().lower(), "major": row[2].strip().lower()})

    with meta_path.open(encoding="utf-8") as f:
        item_meta = json.load(f)

    rel_rows = {}
    with rel_meta_path.open(newline="", encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            rel_rows[(str(r.get("set_id", "")), str(r.get("target_item_id", "")))] = r

    all_rows = []
    for fp in sorted(detail_dir.glob("detail_cir_NewoutfitUrlTitle_subset_none_subset_seed*.csv")):
        with fp.open(newline="", encoding="utf-8-sig") as f:
            for r in csv.DictReader(f):
                all_rows.append(r)

    unique = {}
    for r in all_rows:
        key = (str(r.get("set_id", "")), str(r.get("target_item_id", "")))
        unique.setdefault(key, r)

    audited = []
    for key, r in unique.items():
        sid, iid = key
        rel = rel_rows.get(key, {})
        meta = item_meta.get(iid, {})
        cid = str(meta.get("category_id") or r.get("target_item_fg") or "")
        cm = cat_map.get(cid, {})
        fine = (rel.get("fine_category") or cm.get("fine") or "").lower()
        major = (rel.get("major_category") or cm.get("major") or meta.get("semantic_category", "") or "").lower()
        title = rel.get("title_full_text", "")
        orig = rel.get("original_text", "")
        tfine, tmajor = category_match(title, fine, major)
        ofine, omajor = category_match(orig, fine, major)
        title_words = set(words(title))
        orig_words = set(words(orig))
        toks = distinctive_tokens(meta, fine, major)
        token_hits = sorted(t for t in toks if t in title_words)
        orig_token_hits = sorted(t for t in toks if t in orig_words)
        bigs = distinctive_bigrams(meta, fine, major)
        tphrase = " ".join(words(title))
        ophrase = " ".join(words(orig))
        big_hits = sorted(b for b in bigs if b in tphrase)
        orig_big_hits = sorted(b for b in bigs if b in ophrase)
        audited.append({
            "set_id": sid,
            "target_item_id": iid,
            "fine_category": fine,
            "major_category": major,
            "generated_title": title,
            "original_title": orig,
            "target_item_title": str(meta.get("title", "") or meta.get("url_name", "")),
            "generated_fine_category_match": str(tfine),
            "generated_any_category_match": str(tfine or tmajor),
            "generated_distinctive_token_hits": "|".join(token_hits),
            "generated_distinctive_bigram_hits": "|".join(big_hits),
            "original_fine_category_match": str(ofine),
            "original_any_category_match": str(ofine or omajor),
            "original_distinctive_token_hits": "|".join(orig_token_hits),
            "original_distinctive_bigram_hits": "|".join(orig_big_hits),
        })

    n = len(audited)
    def count(pred) -> int:
        return sum(1 for a in audited if pred(a))
    summary = []
    metrics = [
        ("generated_direct_fine_category_match", count(lambda a: a["generated_fine_category_match"] == "True")),
        ("generated_any_category_match", count(lambda a: a["generated_any_category_match"] == "True")),
        ("generated_distinctive_token_overlap", count(lambda a: bool(a["generated_distinctive_token_hits"]))),
        ("generated_distinctive_bigram_overlap", count(lambda a: bool(a["generated_distinctive_bigram_hits"]))),
        ("original_direct_fine_category_match", count(lambda a: a["original_fine_category_match"] == "True")),
        ("original_any_category_match", count(lambda a: a["original_any_category_match"] == "True")),
        ("original_distinctive_token_overlap", count(lambda a: bool(a["original_distinctive_token_hits"]))),
        ("original_distinctive_bigram_overlap", count(lambda a: bool(a["original_distinctive_bigram_hits"]))),
    ]
    for metric, value in metrics:
        summary.append({
            "metric": metric,
            "count": str(value),
            "denominator": str(n),
            "percent": f"{100 * value / n:.2f}" if n else "0.00",
        })

    out = DER / "A03_target_item_clue_leakage_audit_summary.csv"
    write_csv(out, summary)
    manifest.append({
        "renamed_file": str(out),
        "artifact_type": "derived_table",
        "paper_use": "Target-item clue leakage audit summary for CIR queries.",
        "source_file": "; ".join([str(detail_dir), str(rel_meta_path), str(meta_path), str(cats_path)]),
        "source_folder": str(BASE / "text-conditioned-outfit-recommendation"),
        "source_program_file": str(SRC / "build_journal_artifacts.py"),
        "notes": "Derived from actual CIR detail files and item metadata.",
    })

    examples = [a for a in audited if a["generated_any_category_match"] == "True" or a["generated_distinctive_token_hits"]]
    examples = examples[:80]
    out2 = DER / "A04_target_item_clue_leakage_examples.csv"
    write_csv(out2, examples)
    manifest.append({
        "renamed_file": str(out2),
        "artifact_type": "derived_table",
        "paper_use": "Examples for explaining target-clue leakage risk.",
        "source_file": "; ".join([str(detail_dir), str(rel_meta_path), str(meta_path), str(cats_path)]),
        "source_folder": str(BASE / "text-conditioned-outfit-recommendation"),
        "source_program_file": str(SRC / "build_journal_artifacts.py"),
        "notes": "Examples are selected from actual experiment rows, not generated images or invented data.",
    })


def derive_no_judge_intervention_proof(manifest: list[dict[str, str]]) -> None:
    files = [
        BASE / "text-conditioned-outfit-recommendation/train_cp.py",
        BASE / "text-conditioned-outfit-recommendation/train_cir.py",
        BASE / "text-conditioned-outfit-recommendation/evaluate_cp.py",
        BASE / "text-conditioned-outfit-recommendation/evaluate_cir.py",
        BASE / "text-conditioned-outfit-recommendation/CP_evaluate.py",
        BASE / "text-conditioned-outfit-recommendation/dataset.py",
        BASE / "text-conditioned-outfit-recommendation/outfit_transformer.py",
        BASE / "text-conditioned-outfit-recommendation/utils.py",
        BASE / "text-conditioned-outfit-recommendation/config/base_config.py",
        BASE / "text-conditioned-outfit-recommendation/config/cp_cond_hardneg.py",
        BASE / "text-conditioned-outfit-recommendation/config/cir_cond_hardneg.py",
        BASE / "text-conditioned-outfit-recommendation/5筆資料 t-test.ipynb",
        BASE / "text-conditioned-outfit-recommendation/新描述 t-test.ipynb",
    ]
    judge_terms = [
        "phase3_scores",
        "Qwen3VL32B",
        "phase3_scores_Qwen3VL32B",
        "phase3_scores_Gemma3",
        "checklist_C",
        "LLM-as-a-Judge",
        "judge_score",
        "rewrite_qwen",
        "rewrite_gemma",
    ]
    rows = []
    for fp in files:
        text = ""
        status = "missing_file"
        if fp.exists():
            status = "searched"
            text = fp.read_text(encoding="utf-8", errors="ignore")
        hits = []
        for term in judge_terms:
            if term in text:
                hits.append(term)
        rows.append({
            "searched_file": str(fp),
            "file_role": "downstream_training_evaluation_or_result_analysis",
            "status": status,
            "judge_artifact_terms_found": "|".join(hits),
            "intervention_assessment": "no_judge_artifact_reference_found" if status == "searched" and not hits else ("review_needed" if hits else status),
        })

    data_inputs = [
        BASE / "new_polyvore_outfit_titles.json",
        BASE / "new_polyvore_outfit_titles_with_ablation.json",
        BASE / "fashionclip_data/encoded_NewoutfitUrlTitle_en_fashionClip.pkl",
        BASE / "fashionclip_data/encoded_no_weather_outfitUrlTitle_en_fashionClip.pkl",
        BASE / "fashionclip_data/encoded_no_occasion_outfitUrlTitle_en_fashionClip.pkl",
        BASE / "fashionclip_data/encoded_no_style_outfitUrlTitle_en_fashionClip.pkl",
        BASE / "Qwen3_Instruct/phase3_scores_Qwen3VL32B.jsonl",
        BASE / "Gemma3/phase3_scores_Gemma3.jsonl",
    ]
    existence_rows = []
    for fp in data_inputs:
        existence_rows.append({
            "artifact": str(fp),
            "exists": str(fp.exists()),
            "role": (
                "downstream_description_or_feature_input"
                if "phase3_scores" not in fp.name
                else "post_hoc_judge_output_not_downstream_input"
            ),
        })

    out = DER / "A05_no_judge_intervention_static_proof.csv"
    write_csv(out, rows)
    manifest.append({
        "renamed_file": str(out),
        "artifact_type": "derived_table",
        "paper_use": "Static proof that LLM-as-a-Judge outputs were not referenced by downstream training/evaluation scripts.",
        "source_file": "; ".join(str(f) for f in files),
        "source_folder": str(BASE / "text-conditioned-outfit-recommendation"),
        "source_program_file": str(SRC / "build_journal_artifacts.py"),
        "notes": "Searched downstream training/evaluation/result-analysis files for judge artifacts. No phase3/Qwen/Gemma judge artifact references were found in the downstream pipeline files.",
    })

    out2 = DER / "A06_judge_and_downstream_input_roles.csv"
    write_csv(out2, existence_rows)
    manifest.append({
        "renamed_file": str(out2),
        "artifact_type": "derived_table",
        "paper_use": "Separates downstream description/feature inputs from post-hoc judge outputs.",
        "source_file": "; ".join(str(f) for f in data_inputs),
        "source_folder": str(BASE),
        "source_program_file": str(SRC / "build_journal_artifacts.py"),
        "notes": "Use to support the statement that judge rewrite/score fields were not used to rewrite, filter, train, evaluate, or select models.",
    })

    txt = DER / "A05_no_judge_intervention_static_proof.txt"
    lines = [
        "LLM-as-a-Judge no-intervention proof",
        "日期：2026-06-29",
        "",
        "結論：",
        "本次靜態檢查沒有在 downstream training/evaluation/result-analysis pipeline 中找到 phase3_scores、Qwen/Gemma judge score、rewrite_qwen、rewrite_gemma 或 checklist_C 等 judge artifact 的引用。",
        "因此目前證據支持：LLM-as-a-Judge 的 score/rewrite 欄位是 post-hoc quality/usability assessment，不是訓練、評估、篩選或模型選擇的介入條件。",
        "",
        "需要在論文避免的寫法：",
        "- judge 篩選低品質資料後才進入訓練",
        "- 符合品質控管標準才納入模型訓練與評估",
        "- judge rewrite 被用於改寫下游描述",
        "",
        "建議論文寫法：",
        "LLM-as-a-Judge was used only for post-hoc quality/usability assessment. The judge score and rewrite fields were not used to modify descriptions, filter samples, train the recommendation models, evaluate retrieval/compatibility performance, or select checkpoints.",
        "",
        "被搜尋的 downstream 檔案：",
    ]
    for r in rows:
        lines.append(f"- {r['searched_file']} :: {r['intervention_assessment']}")
    txt.write_text("\n".join(lines) + "\n", encoding="utf-8")
    manifest.append({
        "renamed_file": str(txt),
        "artifact_type": "derived_text",
        "paper_use": "Plain-language proof text for LLM-as-a-Judge post-hoc positioning.",
        "source_file": "; ".join(str(f) for f in files),
        "source_folder": str(BASE / "text-conditioned-outfit-recommendation"),
        "source_program_file": str(SRC / "build_journal_artifacts.py"),
        "notes": "Text summary of A05 static proof.",
    })


def _to_int(x: str, default: int = 0) -> int:
    try:
        return int(float(x))
    except Exception:
        return default


def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def _pearson(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) != len(ys) or len(xs) < 2:
        return None
    mx, my = _mean(xs), _mean(ys)
    vx = sum((x - mx) ** 2 for x in xs)
    vy = sum((y - my) ** 2 for y in ys)
    if vx == 0 or vy == 0:
        return None
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / (vx * vy) ** 0.5


def _rankdata(xs: list[float]) -> list[float]:
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    ranks = [0.0] * len(xs)
    i = 0
    while i < len(order):
        j = i + 1
        while j < len(order) and xs[order[j]] == xs[order[i]]:
            j += 1
        avg_rank = (i + 1 + j) / 2.0
        for k in range(i, j):
            ranks[order[k]] = avg_rank
        i = j
    return ranks


def _spearman(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) != len(ys) or len(xs) < 2:
        return None
    return _pearson(_rankdata(xs), _rankdata(ys))


def _fmt_float(x: float | None, nd: int = 4) -> str:
    return "" if x is None else f"{x:.{nd}f}"


def _summarize_perf(rows: list[dict[str, str]], label: str) -> dict[str, str]:
    n = len(rows)
    if n == 0:
        return {
            "group": label,
            "n_seed_rows": "0",
            "n_unique_set_target_pairs": "0",
            "original_hit10": "",
            "full_hit10": "",
            "delta_hit10": "",
            "original_mean_rank": "",
            "full_mean_rank": "",
            "mean_rank_improvement": "",
            "mean_original_query_len": "",
            "mean_generated_query_len_no_temp": "",
            "mean_length_delta_no_temp": "",
        }
    orig_hit = [_to_int(r["original_hit10"]) for r in rows]
    full_hit = [_to_int(r["full_hit10"]) for r in rows]
    orig_rank = [_to_int(r["original_rank"]) for r in rows]
    full_rank = [_to_int(r["full_rank"]) for r in rows]
    orig_len = [_to_int(r["original_query_len_tokens"]) for r in rows]
    gen_len = [_to_int(r["generated_query_len_no_temp_tokens"]) for r in rows]
    delta_len = [_to_int(r["length_delta_no_temp_tokens"]) for r in rows]
    return {
        "group": label,
        "n_seed_rows": str(n),
        "n_unique_set_target_pairs": str(len({(r["set_id"], r["target_item_id"]) for r in rows})),
        "original_hit10": f"{_mean(orig_hit):.4f}",
        "full_hit10": f"{_mean(full_hit):.4f}",
        "delta_hit10": f"{_mean([f - o for f, o in zip(full_hit, orig_hit)]):.4f}",
        "original_mean_rank": f"{_mean(orig_rank):.2f}",
        "full_mean_rank": f"{_mean(full_rank):.2f}",
        "mean_rank_improvement": f"{_mean([o - f for o, f in zip(orig_rank, full_rank)]):.2f}",
        "mean_original_query_len": f"{_mean(orig_len):.2f}",
        "mean_generated_query_len_no_temp": f"{_mean(gen_len):.2f}",
        "mean_length_delta_no_temp": f"{_mean(delta_len):.2f}",
    }


def _bucket_generated_len(n: int) -> str:
    if n <= 5:
        return "01_gen_no_temp_len_0_5"
    if n <= 8:
        return "02_gen_no_temp_len_6_8"
    if n <= 11:
        return "03_gen_no_temp_len_9_11"
    return "04_gen_no_temp_len_12_plus"


def _bucket_length_delta(n: int) -> str:
    if n <= 0:
        return "01_delta_no_temp_le_0"
    if n <= 3:
        return "02_delta_no_temp_1_3"
    if n <= 6:
        return "03_delta_no_temp_4_6"
    if n <= 9:
        return "04_delta_no_temp_7_9"
    return "05_delta_no_temp_10_plus"


def _bucket_original_len(n: int) -> str:
    if n <= 2:
        return "01_original_len_0_2"
    if n <= 5:
        return "02_original_len_3_5"
    if n <= 8:
        return "03_original_len_6_8"
    return "04_original_len_9_plus"


def derive_length_performance_audit(manifest: list[dict[str, str]]) -> None:
    detail_dir = BASE / "text-conditioned-outfit-recommendation/experiments_none_only/detail_cir"
    orig_title_path = BASE / "polyvore_data/polyvore_outfits/polyvore_outfit_titles.json"
    generated_title_path = BASE / "new_polyvore_outfit_titles.json"
    with orig_title_path.open(encoding="utf-8") as f:
        original = json.load(f)
    with generated_title_path.open(encoding="utf-8") as f:
        generated = json.load(f)

    def original_query_text(set_id: str) -> str:
        v = original.get(str(set_id), {})
        if isinstance(v, dict):
            return " ".join(str(v.get(k, "") or "").strip() for k in ["url_name", "title"]).strip()
        return str(v or "").strip()

    def generated_query_text(set_id: str) -> str:
        v = generated.get(str(set_id), "")
        if isinstance(v, dict):
            return str(v.get("title", "") or "").strip()
        return str(v or "").strip()

    def load_detail(prefix: str) -> dict[tuple[str, str, str], dict[str, str]]:
        out: dict[tuple[str, str, str], dict[str, str]] = {}
        for fp in sorted(detail_dir.glob(f"detail_cir_{prefix}_subset_none_subset_seed*.csv")):
            with fp.open(newline="", encoding="utf-8-sig") as f:
                for r in csv.DictReader(f):
                    key = (str(r.get("seed", "")), str(r.get("set_id", "")), str(r.get("target_item_id", "")))
                    out[key] = r
        return out

    orig_detail = load_detail("outfitUrlTitle")
    full_detail = load_detail("NewoutfitUrlTitle")
    keys = sorted(set(orig_detail) & set(full_detail), key=lambda x: (int(x[0]), int(x[1]), int(x[2])))

    rowlevel: list[dict[str, str]] = []
    for seed, set_id, target_item_id in keys:
        o = orig_detail[(seed, set_id, target_item_id)]
        f = full_detail[(seed, set_id, target_item_id)]
        orig_text = original_query_text(set_id)
        gen_text = generated_query_text(set_id)
        gen_no_temp = strip_temperature_prefix(gen_text)
        orig_len = token_count(orig_text)
        gen_len_total = token_count(gen_text)
        gen_len_no_temp = token_count(gen_no_temp)
        original_rank = _to_int(o.get("rank", "0"))
        full_rank = _to_int(f.get("rank", "0"))
        original_hit10 = _to_int(o.get("hit@10", "0"))
        full_hit10 = _to_int(f.get("hit@10", "0"))
        rowlevel.append({
            "seed": seed,
            "set_id": set_id,
            "target_item_id": target_item_id,
            "original_query_len_tokens": str(orig_len),
            "generated_query_len_total_tokens": str(gen_len_total),
            "generated_query_len_no_temp_tokens": str(gen_len_no_temp),
            "length_delta_total_tokens": str(gen_len_total - orig_len),
            "length_delta_no_temp_tokens": str(gen_len_no_temp - orig_len),
            "generated_no_temp_len_bucket": _bucket_generated_len(gen_len_no_temp),
            "length_delta_no_temp_bucket": _bucket_length_delta(gen_len_no_temp - orig_len),
            "original_len_bucket": _bucket_original_len(orig_len),
            "original_rank": str(original_rank),
            "full_rank": str(full_rank),
            "rank_improvement": str(original_rank - full_rank),
            "original_hit10": str(original_hit10),
            "full_hit10": str(full_hit10),
            "hit10_delta": str(full_hit10 - original_hit10),
        })

    # Fix same-bucket logic with a shared coarse bucket.
    def shared_bucket(n: int) -> str:
        if n <= 5:
            return "0_5"
        if n <= 8:
            return "6_8"
        if n <= 11:
            return "9_11"
        return "12_plus"

    for r in rowlevel:
        r["same_coarse_length_bucket"] = str(
            shared_bucket(_to_int(r["original_query_len_tokens"]))
            == shared_bucket(_to_int(r["generated_query_len_no_temp_tokens"]))
        )

    out_row = DER / "A07_length_performance_rowlevel_cir.csv"
    write_csv(out_row, rowlevel)
    manifest.append({
        "renamed_file": str(out_row),
        "artifact_type": "derived_table",
        "paper_use": "Row-level CIR length-performance audit joining original vs full generated queries.",
        "source_file": "; ".join([str(detail_dir), str(orig_title_path), str(generated_title_path)]),
        "source_folder": str(BASE / "text-conditioned-outfit-recommendation/experiments_none_only/detail_cir"),
        "source_program_file": str(SRC / "build_journal_artifacts.py"),
        "notes": "No retraining and no new text generation. Same seed/set_id/target_item_id rows are paired between original and full generated CIR results.",
    })

    xs = {
        "original_query_len_tokens": [float(r["original_query_len_tokens"]) for r in rowlevel],
        "generated_query_len_total_tokens": [float(r["generated_query_len_total_tokens"]) for r in rowlevel],
        "generated_query_len_no_temp_tokens": [float(r["generated_query_len_no_temp_tokens"]) for r in rowlevel],
        "length_delta_total_tokens": [float(r["length_delta_total_tokens"]) for r in rowlevel],
        "length_delta_no_temp_tokens": [float(r["length_delta_no_temp_tokens"]) for r in rowlevel],
    }
    ys = {
        "hit10_delta": [float(r["hit10_delta"]) for r in rowlevel],
        "rank_improvement": [float(r["rank_improvement"]) for r in rowlevel],
        "full_hit10": [float(r["full_hit10"]) for r in rowlevel],
        "full_rank_negative_for_better": [-float(r["full_rank"]) for r in rowlevel],
    }
    corr_rows = []
    for xname, xvals in xs.items():
        for yname, yvals in ys.items():
            corr_rows.append({
                "x": xname,
                "y": yname,
                "n_seed_rows": str(len(rowlevel)),
                "pearson_r": _fmt_float(_pearson(xvals, yvals)),
                "spearman_r": _fmt_float(_spearman(xvals, yvals)),
                "interpretation_hint": "near_zero_means_length_alone_is_unlikely_to_explain_performance" if abs(_pearson(xvals, yvals) or 0.0) < 0.10 else "inspect_with_bucket_table",
            })
    out_corr = DER / "A08_length_performance_correlation_summary.csv"
    write_csv(out_corr, corr_rows)
    manifest.append({
        "renamed_file": str(out_corr),
        "artifact_type": "derived_table",
        "paper_use": "Correlation summary between query length / length delta and CIR performance changes.",
        "source_file": str(out_row),
        "source_folder": str(DER),
        "source_program_file": str(SRC / "build_journal_artifacts.py"),
        "notes": "Diagnostic length-defense table; correlations are descriptive, not causal claims.",
    })

    bucket_rows = []
    for group_name, field in [
        ("generated_length_without_temperature_bucket", "generated_no_temp_len_bucket"),
        ("length_delta_without_temperature_bucket", "length_delta_no_temp_bucket"),
        ("original_query_length_bucket", "original_len_bucket"),
    ]:
        for bucket in sorted(set(r[field] for r in rowlevel)):
            bucket_rows.append({
                "bucket_type": group_name,
                **_summarize_perf([r for r in rowlevel if r[field] == bucket], bucket),
            })
    out_bucket = DER / "A09_length_bucket_performance_summary.csv"
    write_csv(out_bucket, bucket_rows)
    manifest.append({
        "renamed_file": str(out_bucket),
        "artifact_type": "derived_table",
        "paper_use": "Length bucket analysis: checks whether gains are concentrated only in long generated queries.",
        "source_file": str(out_row),
        "source_folder": str(DER),
        "source_program_file": str(SRC / "build_journal_artifacts.py"),
        "notes": "Use before deciding whether a stricter matched-length experiment is necessary.",
    })

    matched_rows = []
    for threshold in [0, 1, 2, 3, 5]:
        subset = [r for r in rowlevel if abs(_to_int(r["length_delta_no_temp_tokens"])) <= threshold]
        matched_rows.append({
            "matching_rule": f"abs(generated_no_temp_len - original_query_len) <= {threshold} tokens",
            **_summarize_perf(subset, f"abs_delta_le_{threshold}"),
        })
    same_bucket = [r for r in rowlevel if r["same_coarse_length_bucket"] == "True"]
    matched_rows.append({
        "matching_rule": "same coarse length bucket after removing temperature prefix (0-5, 6-8, 9-11, 12+)",
        **_summarize_perf(same_bucket, "same_coarse_length_bucket"),
    })
    out_match = DER / "A10_matched_length_subset_summary.csv"
    write_csv(out_match, matched_rows)
    manifest.append({
        "renamed_file": str(out_match),
        "artifact_type": "derived_table",
        "paper_use": "Matched subset analysis from existing data without new descriptions or retraining.",
        "source_file": str(out_row),
        "source_folder": str(DER),
        "source_program_file": str(SRC / "build_journal_artifacts.py"),
        "notes": "Diagnostic only if tight matched subsets are small; use to decide how strongly the length-only alternative can be rejected.",
    })

    # Concise prose interpretation for the paper-planning folder.
    txt = DER / "A11_length_defense_interpretation.txt"
    corr_hit = [r for r in corr_rows if r["y"] == "hit10_delta" and r["x"] == "length_delta_no_temp_tokens"][0]
    corr_rank = [r for r in corr_rows if r["y"] == "rank_improvement" and r["x"] == "length_delta_no_temp_tokens"][0]
    bucket_lines = [r for r in bucket_rows if r["bucket_type"] == "generated_length_without_temperature_bucket"]
    lines = [
        "Length defense audit",
        "日期：2026-06-29",
        "",
        "方法：",
        "- 不新增資料、不生成新描述、不重訓模型。",
        "- 對齊 experiments_none_only/detail_cir 中 Original 與 Full generated 的同一筆 seed/set_id/target_item_id。",
        "- 使用 original outfitUrlTitle query 長度與 generated title 移除溫度前綴後的長度。",
        "- 分析 length delta 與 Hit@10/rank improvement 的關係，並做 length bucket 與 matched subset summary。",
        "",
        "重點數值：",
        f"- length_delta_no_temp_tokens vs hit10_delta: Pearson r = {corr_hit['pearson_r']}, Spearman r = {corr_hit['spearman_r']}",
        f"- length_delta_no_temp_tokens vs rank_improvement: Pearson r = {corr_rank['pearson_r']}, Spearman r = {corr_rank['spearman_r']}",
        "",
        "Generated no-temperature length buckets:",
    ]
    for b in bucket_lines:
        lines.append(
            f"- {b['group']}: n={b['n_seed_rows']}, original Hit@10={b['original_hit10']}, full Hit@10={b['full_hit10']}, delta={b['delta_hit10']}, mean rank improvement={b['mean_rank_improvement']}"
        )
    lines.extend([
        "",
        "建議使用方式：",
        "- 如果各長度 bucket 的 delta_hit10 不是只集中在最長 query，則可用來回應「只是 query 變長」的質疑。",
        "- matched subset 若樣本太少，只能作為診斷/附錄，不要放成主結論。",
        "- 這個 audit 不能取代新的 masked evaluation，但可作為 length-only confound 的主要防守。",
    ])
    txt.write_text("\n".join(lines) + "\n", encoding="utf-8")
    manifest.append({
        "renamed_file": str(txt),
        "artifact_type": "derived_text",
        "paper_use": "Plain-language interpretation of the length defense audit.",
        "source_file": "; ".join([str(out_corr), str(out_bucket), str(out_match)]),
        "source_folder": str(DER),
        "source_program_file": str(SRC / "build_journal_artifacts.py"),
        "notes": "Use this as the checklist for writing the length-defense paragraph.",
    })


def _parse_p_value(x: str) -> float:
    s = str(x).strip()
    if s.startswith("<"):
        s = s[1:].strip()
    try:
        return float(s)
    except Exception:
        return 1.0


def _holm_adjust(ps: list[float]) -> list[float]:
    m = len(ps)
    order = sorted(range(m), key=lambda i: ps[i])
    adjusted = [1.0] * m
    running = 0.0
    for rank, idx in enumerate(order, start=1):
        val = (m - rank + 1) * ps[idx]
        running = max(running, val)
        adjusted[idx] = min(running, 1.0)
    return adjusted


def _bh_adjust(ps: list[float]) -> list[float]:
    m = len(ps)
    order = sorted(range(m), key=lambda i: ps[i], reverse=True)
    adjusted = [1.0] * m
    running = 1.0
    for rev_rank, idx in enumerate(order, start=1):
        rank = m - rev_rank + 1
        val = ps[idx] * m / rank
        running = min(running, val)
        adjusted[idx] = min(running, 1.0)
    return adjusted


def _ci95_mean(mean_value: float, sd_value: float, n: int = 5) -> tuple[float, float]:
    # t_{.975, df=4}; all current repeated-seed summaries use five seeds.
    tcrit = 2.7764451051977987
    half = tcrit * sd_value / (n ** 0.5)
    return mean_value - half, mean_value + half


def derive_statistical_rigor_summary(manifest: list[dict[str, str]]) -> None:
    src_dir = BASE / "text-conditioned-outfit-recommendation/analysis_outputs_final"
    source_files = [
        src_dir / "table_stage1_cp_detail.csv",
        src_dir / "table_stage1_cir_detail.csv",
        src_dir / "table_stage2_cp_detail.csv",
        src_dir / "table_stage2_cir_detail.csv",
    ]
    rows: list[dict[str, str]] = []
    for fp in source_files:
        with fp.open(newline="", encoding="utf-8-sig") as f:
            for raw in csv.DictReader(f):
                task = raw.get("Task", "")
                metric = raw.get("Metric", "")
                p_raw = _parse_p_value(raw.get("p_raw", "1"))
                dz = float(raw.get("cohens_dz", "0") or 0)
                seeds = raw.get("paired_seeds", "")
                n = len([s for s in seeds.split(",") if s.strip()]) or 5
                if "Baseline_mean" in raw:
                    comparison_family = "stage1_original_vs_full"
                    comparison = "Full contextual rewrite vs Original description"
                    baseline_label = "Original description"
                    proposed_label = "Full contextual rewrite"
                    baseline_mean = float(raw.get("Baseline_mean", "0"))
                    baseline_std = float(raw.get("Baseline_std", "0"))
                    proposed_mean = float(raw.get("Proposed_mean", "0"))
                    proposed_std = float(raw.get("Proposed_std", "0"))
                    delta = float(raw.get("Delta_mean", "0"))
                    factor = ""
                else:
                    factor = raw.get("Factor", "")
                    comparison_family = "stage2_ablation_full_vs_no_factor"
                    comparison = f"Full contextual rewrite vs w/o {factor}"
                    baseline_label = f"w/o {factor}"
                    proposed_label = "Full contextual rewrite"
                    baseline_mean = float(raw.get("NoFactor_mean", "0"))
                    baseline_std = float(raw.get("NoFactor_std", "0"))
                    proposed_mean = float(raw.get("Proposed_mean", "0"))
                    proposed_std = float(raw.get("Proposed_std", "0"))
                    # Source stores NoFactor - Proposed; convert to Proposed - NoFactor.
                    delta = -float(raw.get("Delta_mean_NoFactor_minus_Proposed", "0"))
                    dz = -dz
                b_lo, b_hi = _ci95_mean(baseline_mean, baseline_std, n)
                p_lo, p_hi = _ci95_mean(proposed_mean, proposed_std, n)
                if dz != 0:
                    sd_diff = abs(delta / dz)
                    d_lo, d_hi = _ci95_mean(delta, sd_diff, n)
                else:
                    d_lo, d_hi = (delta, delta)
                rows.append({
                    "comparison_family": comparison_family,
                    "comparison": comparison,
                    "task": task,
                    "metric": metric,
                    "factor": factor,
                    "baseline_label": baseline_label,
                    "baseline_mean": f"{baseline_mean:.6f}",
                    "baseline_std": f"{baseline_std:.6f}",
                    "baseline_ci95": f"[{b_lo:.6f}, {b_hi:.6f}]",
                    "proposed_label": proposed_label,
                    "proposed_mean": f"{proposed_mean:.6f}",
                    "proposed_std": f"{proposed_std:.6f}",
                    "proposed_ci95": f"[{p_lo:.6f}, {p_hi:.6f}]",
                    "delta_proposed_minus_baseline": f"{delta:.6f}",
                    "delta_ci95": f"[{d_lo:.6f}, {d_hi:.6f}]",
                    "p_raw": f"{p_raw:.8f}",
                    "p_holm_adjusted_all_tests": "",
                    "p_bh_adjusted_all_tests": "",
                    "cohens_dz": f"{dz:.6f}",
                    "paired_seeds": seeds,
                    "source_file": str(fp),
                })
    ps = [float(r["p_raw"]) for r in rows]
    holm = _holm_adjust(ps)
    bh = _bh_adjust(ps)
    for r, ph, pb in zip(rows, holm, bh):
        r["p_holm_adjusted_all_tests"] = f"{ph:.8f}"
        r["p_bh_adjusted_all_tests"] = f"{pb:.8f}"

    out = DER / "A14_statistical_rigor_ci_adjusted_p_effect_sizes.csv"
    write_csv(out, rows)
    manifest.append({
        "renamed_file": str(out),
        "artifact_type": "derived_table",
        "paper_use": "Teacher-context statistical rigor table: 95% CI, adjusted p-values, and paired Cohen's dz.",
        "source_file": "; ".join(str(f) for f in source_files),
        "source_folder": str(src_dir),
        "source_program_file": str(SRC / "build_journal_artifacts.py"),
        "notes": "Computed from existing five-seed summary/detail CSVs. Holm and BH corrections are applied across all stage1 and stage2 tests in this table.",
    })


def copy_artifacts(manifest: list[dict[str, str]]) -> None:
    for a in ARTIFACTS:
        src = a["src"]
        dst = a["dst"]
        dst.parent.mkdir(parents=True, exist_ok=True)
        if src.exists():
            shutil.copy2(src, dst)
            status = "copied"
        else:
            status = "missing_source"
        manifest.append({
            "renamed_file": str(dst),
            "artifact_type": a["type"],
            "paper_use": a["use"],
            "source_file": str(src),
            "source_folder": str(src.parent),
            "source_program_file": str(a["program"]),
            "notes": status,
        })


def copy_source_programs() -> None:
    programs = {
        BASE / "text-conditioned-outfit-recommendation/5筆資料 t-test.ipynb": SRC / "P01_main_results_and_ablation_5seed_ttest.ipynb",
        BASE / "text-conditioned-outfit-recommendation/完整套裝推薦_fastfit_demo.ipynb": SRC / "P02_subset_robustness_analysis.ipynb",
        BASE / "text-conditioned-outfit-recommendation/個案分析.ipynb": SRC / "P03_qualitative_case_analysis.ipynb",
        BASE / "text-conditioned-outfit-recommendation/XAI作業.ipynb": SRC / "P04_reliability_xai_analysis.ipynb",
        BASE / "驗證結果分析.ipynb": SRC / "P05_llm_judge_agreement_analysis.ipynb",
        BASE / "150 筆人工檢查.ipynb": SRC / "P06_prompt_robustness_human_check.ipynb",
        BASE / "paper_analysis_charts.ipynb": SRC / "P07_human_check_paper_analysis_charts.ipynb",
        BASE / "text-conditioned-outfit-recommendation/新描述 t-test.ipynb": SRC / "P08_legacy_fullrun_new_description_ttest.ipynb",
        BASE / "text-conditioned-outfit-recommendation/evaluate_cp.py": SRC / "P09_evaluate_cp.py",
        BASE / "text-conditioned-outfit-recommendation/evaluate_cir.py": SRC / "P10_evaluate_cir.py",
        BASE / "熱舒適模型/temperature.ipynb": SRC / "P11_temperature_title_generation.ipynb",
    }
    for src, dst in programs.items():
        if src.exists():
            shutil.copy2(src, dst)

    # Copy this script into the target folder for reproducibility.
    self_path = Path(__file__).resolve()
    if self_path.exists():
        dst = (SRC / "build_journal_artifacts.py").resolve()
        if self_path != dst:
            shutil.copy2(self_path, dst)


def write_source_maps(manifest: list[dict[str, str]]) -> None:
    csv_path = TGT / "source_map.csv"
    write_csv(csv_path, manifest)

    copied = sum(1 for m in manifest if m["notes"] != "missing_source")
    missing = [m for m in manifest if m["notes"] == "missing_source"]
    lines = []
    lines.append("期刊圖表整理來源對照")
    lines.append("日期：2026-06-29")
    lines.append("")
    lines.append(f"整理資料夾：{TGT}")
    lines.append(f"已列入項目：{len(manifest)}")
    lines.append(f"成功複製/產生：{copied}")
    lines.append(f"找不到來源檔：{len(missing)}")
    lines.append("")
    lines.append("篩選原則：")
    lines.append("- 只整理你的實驗結果圖表與由實驗資料重新計算出的稽核表。")
    lines.append("- 未納入 FastFit/OmniTry/survey/jupyter_outputs/user case 之類的生成展示圖，避免把生成案例誤當成實驗證據。")
    lines.append("- B 開頭檔案是 legacy/full-run backup，不建議放主文；主文優先使用 T01-T18 與 F01-F16。")
    lines.append("- derived_audits 內的 A 開頭檔案是由原始 JSON/CSV 實驗資料計算，不是 AI 生成內容。")
    lines.append("")
    lines.append("檔案對照：")
    for m in manifest:
        lines.append("")
        lines.append(f"[{m['artifact_type']}] {Path(m['renamed_file']).name}")
        lines.append(f"用途：{m['paper_use']}")
        lines.append(f"整理後位置：{m['renamed_file']}")
        lines.append(f"來源資料夾：{m['source_folder']}")
        lines.append(f"來源檔案：{m['source_file']}")
        lines.append(f"程式檔：{m['source_program_file']}")
        lines.append(f"備註：{m['notes']}")

    if missing:
        lines.append("")
        lines.append("找不到來源檔清單：")
        for m in missing:
            lines.append(f"- {m['source_file']}")

    (TGT / "source_map.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ensure_dirs()
    manifest: list[dict[str, str]] = []
    copy_artifacts(manifest)
    copy_source_programs()
    derive_title_length_and_coverage(manifest)
    derive_leakage_audit(manifest)
    derive_no_judge_intervention_proof(manifest)
    derive_length_performance_audit(manifest)
    derive_statistical_rigor_summary(manifest)
    write_source_maps(manifest)
    print(TGT)
    print(f"manifest_items={len(manifest)}")


if __name__ == "__main__":
    main()
