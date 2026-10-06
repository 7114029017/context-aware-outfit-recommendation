#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import argparse
import csv
import json

ROOT = Path(__file__).resolve().parents[3]
REPRO_ROOT = ROOT / "reproduction"
REPRO_SCRIPTS = REPRO_ROOT / "scripts"


def load_json(path: Path):
    if not path.is_file():
        raise SystemExit(f"Missing required summary: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def max_abs_mean_diff(path: Path):
    if not path.is_file():
        return None
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    vals = [float(r["abs_mean_difference"]) for r in rows if r.get("abs_mean_difference") not in (None, "")]
    return max(vals) if vals else None


def main():
    ap = argparse.ArgumentParser(description="Build one run-level reproduction summary.")
    ap.add_argument("--run-root", required=True)
    args = ap.parse_args()

    run = Path(args.run_root).expanduser()
    run = (run if run.is_absolute() else ROOT / run).resolve()

    main_acc = load_json(run / "main/summary/main_reproduction_acceptance.json")
    fair = load_json(run / "ablation/summary/fresh_fair_subset_5seed_summary.json")
    secondary = load_json(run / "secondary/results/remaining_reproduction_matrix.json")
    proxy = load_json(run / "secondary/results/environment_proxy/table_4_3_manifest.json")
    chapter4 = load_json(run / "chapter4/chapter4_report.json")
    if (
        chapter4.get("source_mode") != "full"
        or chapter4.get("total_chapter4_tables") != 20
        or chapter4.get("numeric_comparison_rows") != 112
    ):
        raise SystemExit("Missing/incomplete Chapter 4 full-run report; cannot declare full reproduction complete")

    fair_diff = max_abs_mean_diff(
        run / "ablation/summary/archived_vs_fresh_T01_numeric_comparison.csv"
    )

    modules = []
    for row in secondary.get("modules", []):
        modules.append({
            "module": row.get("module"),
            "status": row.get("status"),
            "evidence": row.get("evidence"),
        })

    payload = {
        "classification": "fresh thesis reproduction run summary",
        "run_root": str(run.relative_to(ROOT)) if ROOT in run.parents else run.name,
        "main": {
            "overall": main_acc.get("overall"),
            "all_primary_context_deltas_positive": main_acc.get("all_primary_context_deltas_positive"),
            "all_condition_means_within_numeric_threshold": main_acc.get("all_condition_means_within_numeric_threshold"),
            "summary_csv": "main/summary/main_reproduction_summary.csv",
        },
        "ablation": {
            "classification": fair.get("classification"),
            "units_verified": fair.get("units_verified"),
            "batch_status": fair.get("batch_status"),
            "max_abs_archived_vs_fresh_mean_difference": fair_diff,
            "historical_memberwise_identity": "not recovered; reproducibly reconstructed subset",
            "summary_dir": "ablation/summary",
        },
        "secondary": {
            "overall": secondary.get("overall"),
            "modules": modules,
            "summary_md": "secondary/results/remaining_reproduction_matrix.md",
        },
        "chapter4": {
            "classification": chapter4["classification"],
            "total_chapter4_tables": chapter4["total_chapter4_tables"],
            "numeric_comparison_rows": chapter4["numeric_comparison_rows"],
            "paper_display_match_4_3": chapter4.get("paper_display_match_4_3"),
            "paper_display_match_4_8": chapter4.get("paper_display_match_4_8"),
            "main_condition_means_within_retrain_tolerance": chapter4.get("main_mean_status"),
            "ablation_comparison_reference": "archived T01 aggregate; not each individual printed thesis table",
            "report_md": "chapter4/chapter4_report.md",
            "numeric_comparison_csv": "chapter4/chapter4_numeric_comparison.csv",
            "overview_csv": "chapter4/chapter4_table_overview.csv",
            "evidence_json": "chapter4/chapter4_report.json",
        },
        "table_4_3": {
            "source": "preserved aligned train/valid/test temperature.json outputs",
            "n_unique_primary_rows": proxy.get("n_unique_primary_rows"),
            "all_listed_paper_fields_match_display_precision":
                proxy.get("all_listed_paper_fields_match_display_precision"),
            "standalone_source_values_consistent_on_overlap":
                proxy.get("standalone_source_values_consistent_on_overlap"),
            "report": "secondary/results/environment_proxy/table_4_3_manifest.json",
            "historical_2025_runtime_or_selection_rule_exact":
                proxy.get("historical_2025_runtime_or_selection_rule_exact"),
        },
        "interpretation": [
            "The fresh main five-seed run is the primary independent numerical reproduction.",
            "The fair-subset ablation uses a reproducibly reconstructed subset because the original memberwise fair-subset ID file was not archived.",
            "Secondary preserved-output analyses keep their own exact/partial provenance labels.",
            "Historical source/runtime provenance is not upgraded merely because numerical results agree.",
        ],
    }

    out_json = run / "reproduction_summary.json"
    out_md = run / "reproduction_summary.md"
    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    lines = [
        "# Thesis reproduction run summary",
        "",
        f"- Main reproduction: **{payload['main']['overall']}**",
        f"- All 8 main Context−Original directions positive: **{payload['main']['all_primary_context_deltas_positive']}**",
        f"- All 16 main condition means within retraining tolerance: **{payload['main']['all_condition_means_within_numeric_threshold']}**",
        f"- Fair-subset units verified: **{payload['ablation']['units_verified']}/25**",
        f"- Fair-subset provenance: **reproducibly reconstructed; historical memberwise identity not recovered**",
        f"- Secondary-analysis overall: **{payload['secondary']['overall']}**",
        f"- Table 4-3 preserved proxy distribution paper-display match: **{payload['table_4_3']['all_listed_paper_fields_match_display_precision']}**",
        f"- Chapter 4 table inventory: **{payload['chapter4']['total_chapter4_tables']}/20**, with **{payload['chapter4']['numeric_comparison_rows']}** explicitly scoped numeric rows; NOT 20 fully verified thesis tables.",
        "",
        "## Outputs",
        "",
        "- Main: `main/summary/main_reproduction_summary.csv`",
        "- Ablation: `ablation/summary/`",
        "- Secondary: `secondary/results/remaining_reproduction_matrix.md`",
        "- Table 4-3: `secondary/results/environment_proxy/table_4_3_manifest.json`",
        "- Chapter 4 table-by-table overview: `chapter4/chapter4_report.md`",
        "- Chapter 4 numeric source comparisons: `chapter4/chapter4_numeric_comparison.csv`",
        "",
        "## Secondary modules",
        "",
        "| Module | Status |",
        "|---|---|",
    ]
    for row in modules:
        lines.append(f"| {row['module']} | **{row['status']}** |")
    lines += [
        "",
        "## Interpretation",
        "",
        "This run uses the archived handoff research programs through a thin wrapper layer. "
        "The wrapper prepares paths, seeds, the standardized decoder implementation, output directories, "
        "aggregation, and paper comparison; it does not replace the research model implementation.",
        "",
        "Historical source/runtime provenance limitations remain separate from numerical reproduction.",
        "",
    ]
    out_md.write_text("\n".join(lines), encoding="utf-8")
    print(f"[OK] wrote {out_json}")
    print(f"[OK] wrote {out_md}")


if __name__ == "__main__":
    main()
