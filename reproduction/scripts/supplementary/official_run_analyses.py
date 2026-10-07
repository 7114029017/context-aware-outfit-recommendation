#!/usr/bin/env python3
"""Supplementary analyses computed from the saved outputs of a completed full run.

Reads the run's fair-subset results (query-level rows and per-seed metrics that
the run wrote when it evaluated its retrained models). No model is loaded and
nothing is re-run. The run is chosen as follows:

- --run-root PATH: a completed full run folder, for example
  reproduction/runs/full_<UTC timestamp> from reproduce_all.sh --fresh;
- --official: the official run included in the repository
  (reproduction/results/raw/full_20261002T161428Z/);
- neither: the newest full_* folder under reproduction/runs/, or the official
  run if there is none.

A run folder must have RUN_STATUS PASSED. The outputs go to
reproduction/results/supplementary/<run>/ for the official run and to
<run folder>/supplementary/run_analyses/ for a run folder (or to --out-dir).
run_all.sh runs this script and the two input-data scripts; a full run calls
run_all.sh after it has PASSED. Outputs:

- category_hit10_delta.csv: Hit@10 of Full minus Original by fine-grained
  target category (manuscript Figure 4), with the 2025 values for comparison;
- context_subset_hit10.csv: Hit@10 by weather, occasion, style-richness and
  target-type subset (manuscript Section 5.5), with the 2025 values;
- case_ranks.csv: five-seed mean target ranks of the three manuscript cases
  (Figures A1-A3) under all five conditions;
- fair_subset_factor_effects.csv: Full minus each other condition for the
  eight metrics, with two-sided paired t-test p-values over the five seeds;
- fair_subset_bh_family_sensitivity.csv: retrospective sensitivity analysis of
  the Benjamini-Hochberg family for Full vs Original (the five reported
  metrics vs all eight metrics);
- summary.md: the main numbers and the cross-checks.

The subset rules follow the 2025 notebook
03_實驗與結果_experiments_results/04_情境子集與三因子分析/source_programs/P02_subset_robustness_analysis.ipynb
(FORMAL_KEYWORDS, CASUAL_KEYWORDS, TEMP_PATTERN and the median temperature split).
"""
from __future__ import annotations

import argparse
import json
import re
import statistics
from collections import defaultdict
from pathlib import Path

from scipy import stats

from _common import (CASE_TABLES, OFFICIAL_RUN, REPRO, SUPPLEMENTARY, WOS_JSONL, bh_adjust, fmt,
                     polyvore_root, read_csv, read_json, signed, write_csv, write_text)

VARIANTS = {  # run directory prefix -> condition name in the manuscript
    "original": "Original",
    "context": "Full",
    "no_weather": "No-Weather",
    "no_occasion": "No-Occasion",
    "no_style": "No-Style",
}
SEEDS = (1, 2, 3, 4, 5)
CP_METRICS = ("auc", "fitb_acc")
CIR_METRICS = ("recall_at_1", "recall_at_3", "recall_at_5", "recall_at_10", "recall_at_30", "recall_at_50")
METRICS = CP_METRICS + CIR_METRICS
REPORTED_FAMILY = ("auc", "fitb_acc", "recall_at_10", "recall_at_30", "recall_at_50")  # manuscript Table 7
CASES = (  # manuscript Figures A1-A3
    ("purse", "224499261", "Figure A1"),
    ("dress", "94771580", "Figure A2"),
    ("sunglasses", "200099867", "Figure A3"),
)
FORMAL_KEYWORDS = re.compile(r"\b(formal|office|business|work|professional|interview|meeting|wedding|party|"
                             r"cocktail|elegant|tailored|blazer|suit)\b")
CASUAL_KEYWORDS = re.compile(r"\b(casual|weekend|streetwear|street|relaxed|everyday|daily|chill|laid-back|"
                             r"lounge|vacation|travel)\b")
TEMP_PATTERN = re.compile(r"(-?\d+(?:\.\d+)?)\s*°?\s*([CFcf])")
CLOTHING = {"all body", "bottoms", "tops", "outerwear"}
ACCESSORY = {"bags", "shoes", "accessories", "hats", "jewellery", "scarves", "sunglasses"}


def norm_text(value) -> str:
    if value is None:
        return ""
    if isinstance(value, list):
        return " | ".join(str(v).strip() for v in value if str(v).strip())
    return str(value).strip()


def norm_category(value: str) -> str:
    value = re.sub(r"[_/|,-]+", " ", str(value).strip().lower())
    return re.sub(r"\s+", " ", value).strip()


def temperature(text: str) -> float | None:
    match = TEMP_PATTERN.search(text or "")
    if not match:
        return None
    value = float(match.group(1))
    return (value - 32) * 5 / 9 if match.group(2).upper() == "F" else value


def load_rows(ablation: Path) -> dict[str, list[dict]]:
    rows = {}
    for prefix in VARIANTS:
        rows[prefix] = []
        for seed in SEEDS:
            path = ablation / f"{prefix}_seed{seed}" / "detail_cir_fresh_subset.csv"
            for row in read_csv(path):
                row["seed"] = int(row["seed"])
                row["rank"] = int(row["rank"])
                row["hit@10"] = int(row["hit@10"])
                rows[prefix].append(row)
    return rows


def load_metrics(ablation: Path) -> dict[str, dict[str, list[float]]]:
    out = {}
    for prefix in VARIANTS:
        out[prefix] = {m: [] for m in METRICS}
        for seed in SEEDS:
            cp = read_csv(ablation / f"{prefix}_seed{seed}" / "results_cp_fresh_subset.csv")[0]
            cir = read_csv(ablation / f"{prefix}_seed{seed}" / "results_cir_fresh_subset.csv")[0]
            for m in CP_METRICS:
                out[prefix][m].append(float(cp[m]))
            for m in CIR_METRICS:
                out[prefix][m].append(float(cir[m]))
    return out


def official_source() -> dict:
    return {"run_id": OFFICIAL_RUN, "units": REPRO / "results" / "raw" / OFFICIAL_RUN / "ablation",
            "tables": REPRO / "results" / "summary" / OFFICIAL_RUN / "ablation", "out": SUPPLEMENTARY / OFFICIAL_RUN,
            "label": f"the official run included in the repository (`{OFFICIAL_RUN}`)"}


def run_folder_source(run_root: Path) -> dict:
    run_root = run_root.resolve()
    if not run_root.is_dir():
        raise SystemExit(f"[SUPPLEMENTARY BLOCKED] run folder not found: {run_root}")
    status = run_root / "RUN_STATUS.txt"
    state = status.read_text(encoding="utf-8").strip() if status.is_file() else "missing"
    if state != "PASSED":
        raise SystemExit(f"[SUPPLEMENTARY BLOCKED] {run_root}: RUN_STATUS is {state}, not PASSED. "
                         "Pass --run-root PATH for a completed run, or --official for the official run.")
    return {"run_id": run_root.name, "units": run_root / "ablation" / "runs", "tables": run_root / "ablation" / "summary",
            "out": run_root / "supplementary" / "run_analyses", "label": f"the run folder `{run_root}`"}


def choose_source(args) -> dict:
    if args.official and args.run_root:
        raise SystemExit("[SUPPLEMENTARY BLOCKED] pass either --run-root or --official, not both")
    if args.official:
        return official_source()
    if args.run_root:
        return run_folder_source(args.run_root)
    runs = sorted(p for p in (REPRO / "runs").glob("full_*") if p.is_dir())  # newest last, as in README 0.8
    return run_folder_source(runs[-1]) if runs else official_source()


def pfmt(p: float | None) -> str:
    if p is None:
        return ""
    return "< 0.0001" if p < 1e-4 else f"{p:.4f}"


def mean(values) -> float:
    values = list(values)
    return sum(values) / len(values)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-root", type=Path, default=None,
                        help="completed full run folder, e.g. reproduction/runs/full_<UTC timestamp>")
    parser.add_argument("--official", action="store_true", help="analyze the official run included in the repository")
    parser.add_argument("--polyvore-root", default=None)
    parser.add_argument("--out-dir", type=Path, default=None)
    args = parser.parse_args()

    source = choose_source(args)
    ablation = source["units"]
    if not ablation.is_dir():
        raise SystemExit(f"[SUPPLEMENTARY BLOCKED] run results not found: {ablation}")
    out = args.out_dir or source["out"]
    print(f"[SUPPLEMENTARY] analyzing {source['label'].replace('`', '')}")
    poly = polyvore_root(args.polyvore_root)

    rows = load_rows(ablation)
    metrics = load_metrics(ablation)
    original = {(r["seed"], r["set_id"], r["target_item_id"]): r for r in rows["original"]}
    full = {(r["seed"], r["set_id"], r["target_item_id"]): r for r in rows["context"]}
    if set(original) != set(full):
        raise SystemExit("[SUPPLEMENTARY BLOCKED] Original and Full rows cover different queries")
    keys = sorted(original)

    categories = {}
    for record in (poly / "categories.csv").read_text(encoding="utf-8-sig").splitlines():
        parts = record.split(",")
        if len(parts) >= 3 and parts[0].strip():
            categories.setdefault(parts[0].strip(), (norm_category(parts[1]), norm_category(parts[2])))

    fragments = {}
    for line in WOS_JSONL.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        if record.get("valid") is not True:
            continue
        frag = record.get("fragments") or {}
        style = frag.get("style") or []
        fragments[str(record.get("id", ""))] = {
            "title": norm_text(record.get("title")),
            "weather": norm_text(frag.get("weather")),
            "occasion": norm_text(frag.get("occasion")),
            "style_terms": len([t for t in style if str(t).strip()]) if isinstance(style, list) else 0,
        }
    titles = read_json(poly / "polyvore_outfit_titles.json")

    def original_text(set_id: str) -> str:
        value = titles.get(set_id, "")
        if isinstance(value, dict):
            parts = [norm_text(value.get("url_name", value.get("url", ""))), norm_text(value.get("title", ""))]
            return " ".join(p for p in parts if p).strip()
        return norm_text(value)

    # ---------------------------------------------------------------- category deltas
    by_category = defaultdict(lambda: {"o": [], "f": [], "seed": defaultdict(lambda: [[], []])})
    for key in keys:
        fg = original[key]["target_item_fg"]
        entry = by_category[fg]
        entry["o"].append(original[key]["hit@10"])
        entry["f"].append(full[key]["hit@10"])
        entry["seed"][key[0]][0].append(original[key]["hit@10"])
        entry["seed"][key[0]][1].append(full[key]["hit@10"])
    archived_category = {r["target_item_fg"]: r for r in read_csv(CASE_TABLES / "T15_qualitative_category_summary.csv")
                         if r["case_type"] == "orig_to_full"}
    category_rows = []
    for fg, entry in by_category.items():
        delta = mean(entry["f"]) - mean(entry["o"])
        per_seed = [mean(entry["seed"][s][1]) - mean(entry["seed"][s][0]) for s in SEEDS]
        fine, major = categories.get(fg, ("", ""))
        old = archived_category.get(fg, {})
        category_rows.append([fg, fine, major, len(entry["o"]), fmt(mean(entry["o"])), fmt(mean(entry["f"])),
                              signed(delta), *[signed(x) for x in per_seed],
                              old.get("n", ""), signed(float(old["delta_hit10"])) if old else ""])
    category_rows.sort(key=lambda r: -float(r[6]))
    write_csv(out / "category_hit10_delta.csv",
              ["target_item_fg", "fine_category", "main_category", "observations", "original_hit10", "full_hit10",
               "delta_hit10", *[f"delta_seed{s}" for s in SEEDS], "observations_2025", "delta_hit10_2025"],
              category_rows)

    # ---------------------------------------------------------------- context subsets
    info = {}
    for key in keys:
        set_id = key[1]
        frag = fragments.get(set_id, {})
        temp = None
        for text in (frag.get("weather", ""), frag.get("title", ""), original_text(set_id)):
            temp = temperature(text)
            if temp is not None:
                break
        occasion_text = (frag.get("occasion", "") + " " + frag.get("title", "")).lower()
        info[key] = {
            "temp": temp,
            "formal": bool(FORMAL_KEYWORDS.search(occasion_text)),
            "casual": bool(CASUAL_KEYWORDS.search(occasion_text)),
            "style_terms": frag.get("style_terms", 0),
            "major": categories.get(original[key]["target_item_fg"], ("", ""))[1],
        }
    temps = [v["temp"] for v in info.values() if v["temp"] is not None]
    threshold = statistics.median(temps)
    subsets = [
        ("Cold", f"temperature <= {threshold:.2f} °C", lambda v: v["temp"] is not None and v["temp"] <= threshold),
        ("Warm", f"temperature > {threshold:.2f} °C", lambda v: v["temp"] is not None and v["temp"] > threshold),
        ("Formal", "formal keyword in occasion or title", lambda v: v["formal"]),
        ("Casual", "casual keyword in occasion or title", lambda v: v["casual"]),
        ("High style", ">= 2 style terms", lambda v: v["style_terms"] >= 2),
        ("Low style", "<= 1 style term", lambda v: v["style_terms"] <= 1),
        ("Clothing-led", "target in all body, bottoms, tops or outerwear", lambda v: v["major"] in CLOTHING),
        ("Accessory-led", "target in bags, shoes, accessories, hats, jewellery, scarves or sunglasses",
         lambda v: v["major"] in ACCESSORY),
        ("All", "all fair-subset queries", lambda v: True),
    ]
    archived_subsets = {}
    for r in read_csv(CASE_TABLES / "T12_subset_robustness_summary.csv"):
        label = r["label"]
        name = ("High style" if label.startswith("High style") else "Low style" if label.startswith("Low style")
                else "All" if label == "Full dataset" else label)
        archived_subsets.setdefault(name, r)
    subset_rows = []
    for name, rule, test in subsets:
        members = [k for k in keys if test(info[k])]
        o = mean(original[k]["hit@10"] for k in members)
        f = mean(full[k]["hit@10"] for k in members)
        old = archived_subsets.get(name, {})
        subset_rows.append([name, rule, len(members), fmt(o), fmt(f), signed(f - o),
                            old.get("n", ""), signed(float(old["ΔHit@10"])) if old else ""])
    write_csv(out / "context_subset_hit10.csv",
              ["subset", "rule", "observations", "original_hit10", "full_hit10", "delta_hit10",
               "observations_2025", "delta_hit10_2025"], subset_rows)

    # ---------------------------------------------------------------- cases
    case_rows = []
    for name, set_id, figure in CASES:
        for prefix, label in VARIANTS.items():
            picked = sorted((r for r in rows[prefix] if r["set_id"] == set_id), key=lambda r: r["seed"])
            if len(picked) != len(SEEDS):
                raise SystemExit(f"[SUPPLEMENTARY BLOCKED] case {name} ({set_id}) has {len(picked)} rows for {label}")
            fg = picked[0]["target_item_fg"]
            case_rows.append([name, figure, set_id, picked[0]["target_item_id"], categories.get(fg, ("", ""))[0],
                              label, fmt(mean(r["rank"] for r in picked), 1), *[r["rank"] for r in picked]])
    write_csv(out / "case_ranks.csv",
              ["case", "manuscript_figure", "set_id", "target_item_id", "fine_category", "condition", "mean_rank",
               *[f"rank_seed{s}" for s in SEEDS]], case_rows)

    # ---------------------------------------------------------------- factor effects and BH family
    effect_rows = []
    effects = {}
    raw_p = {}
    for prefix in ("original", "no_weather", "no_occasion", "no_style"):
        for m in METRICS:
            diff = [a - b for a, b in zip(metrics["context"][m], metrics[prefix][m])]
            p = float(stats.ttest_rel(metrics["context"][m], metrics[prefix][m]).pvalue)
            effects[(prefix, m)] = (signed(mean(diff)), p)
            if prefix == "original":
                raw_p[m] = p
            effect_rows.append([f"Full - {VARIANTS[prefix]}", m, signed(mean(diff)),
                                sum(1 for d in diff if d > 0), f"{p:.6g}"])
    write_csv(out / "fair_subset_factor_effects.csv",
              ["comparison", "metric", "mean_difference", "seeds_positive", "paired_t_p_two_sided"], effect_rows)

    bh5 = dict(zip(REPORTED_FAMILY, bh_adjust([raw_p[m] for m in REPORTED_FAMILY])))
    bh8 = dict(zip(METRICS, bh_adjust([raw_p[m] for m in METRICS])))
    write_csv(out / "fair_subset_bh_family_sensitivity.csv",
              ["metric", "raw_p", "bh_p_reported_5_metrics", "bh_p_all_8_metrics"],
              [[m, f"{raw_p[m]:.6g}", f"{bh5[m]:.6g}" if m in bh5 else "", f"{bh8[m]:.6g}"] for m in METRICS])

    # ---------------------------------------------------------------- cross-checks and summary
    overall_o = mean(r["hit@10"] for r in rows["original"])
    overall_f = mean(r["hit@10"] for r in rows["context"])
    recall_o = mean(metrics["original"]["recall_at_10"])
    recall_f = mean(metrics["context"]["recall_at_10"])
    counts_match_t15 = all(str(len(e["o"])) == archived_category.get(fg, {}).get("n") for fg, e in by_category.items())
    counts_match_t12 = all(str(r[2]) == str(r[6]) for r in subset_rows)
    summary_dir = source["tables"]
    t03_t04 = {}
    for table in ("T03_stage1_cp_original_vs_full.csv", "T04_stage1_cir_original_vs_full.csv"):
        path = summary_dir / table
        if path.is_file():
            for r in read_csv(path):
                t03_t04[r["Metric"]] = r["BH-adjusted p"]
    names = {"AUC": "auc", "FITB Acc": "fitb_acc", "Recall@10": "recall_at_10", "Recall@30": "recall_at_30",
             "Recall@50": "recall_at_50"}

    def same_p(value: str, computed: float) -> bool:
        return (value == "< .001" and computed < 0.001) or (value != "< .001" and f"{computed:.4f}" == value)

    bh_matches = [(label, value, bh5[names[label]]) for label, value in t03_t04.items() if label in names]
    bh_ok = all(same_p(value, computed) for _, value, computed in bh_matches)
    comparisons = {"Original description": "original", "Simplified w/o weather": "no_weather",
                   "Simplified w/o occasion": "no_occasion", "Simplified w/o style": "no_style"}
    t02_rows = []
    t02 = summary_dir / "T02_main_significance_tests.csv"
    if t02.is_file():
        for r in read_csv(t02):
            prefix = comparisons.get(r["Comparison"].split(" vs ", 1)[-1])
            if prefix is not None and r["Metric"] in names:
                difference, p = effects[(prefix, names[r["Metric"]])]
                t02_rows.append(difference == r["Mean difference"] and same_p(r["p-value"], p))
    t02_ok = bool(t02_rows) and all(t02_rows)

    best = category_rows[0]
    lines = [
        f"# Supplementary analyses of run `{source['run_id']}`",
        "",
        f"Source: {source['label']}.",
        "",
        "Computed by `reproduction/scripts/supplementary/official_run_analyses.py` from the run's saved",
        "fair-subset outputs (query-level rows and per-seed metrics). No model was loaded and nothing was re-run.",
        "These analyses come after the 35 training units and change none of the run's results.",
        "",
        "## Hit@10 by target category (manuscript Figure 4)",
        "",
        "| Category | Observations | This run: Full − Original | 2025: Full − Original |",
        "|---|---:|---:|---:|",
        *[f"| {r[1]} ({r[2]}) | {r[3]} | {r[6]} | {r[-1]} |" for r in category_rows],
        "",
        f"Largest gain: {best[1]} ({best[6]}). Categories with a decline in this run: "
        + (", ".join(f"{r[1]} ({r[6]})" for r in category_rows if float(r[6]) < 0) or "none") + ".",
        "",
        "## Hit@10 by context subset",
        "",
        "| Subset | Rule | Observations | This run: Full − Original | 2025 |",
        "|---|---|---:|---:|---:|",
        *[f"| {r[0]} | {r[1]} | {r[2]} | {r[5]} | {r[7]} |" for r in subset_rows],
        "",
        "## Case ranks (five-seed mean)",
        "",
        "| Case | Original | Full | No-Weather | No-Occasion | No-Style |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, set_id, figure in CASES:
        by_label = {r[5]: r[6] for r in case_rows if r[0] == name}
        lines.append(f"| {name} ({figure}, set {set_id}) | {by_label['Original']} | {by_label['Full']} | "
                     f"{by_label['No-Weather']} | {by_label['No-Occasion']} | {by_label['No-Style']} |")
    lines += [
        "",
        "## Benjamini-Hochberg family (retrospective sensitivity analysis)",
        "",
        "Full vs Original on the fair subset. The manuscript corrects over the five reported metrics.",
        "",
        "| Metric | Raw p | BH, 5 reported metrics | BH, all 8 metrics |",
        "|---|---:|---:|---:|",
        *[f"| {m} | {pfmt(raw_p[m])} | {pfmt(bh5.get(m))} | {pfmt(bh8[m])} |" for m in METRICS],
        "",
        "## Cross-checks",
        "",
        f"- Overall Hit@10 from the query-level rows equals the mean per-seed Recall@10: "
        f"Original {overall_o:.6f} vs {recall_o:.6f}; Full {overall_f:.6f} vs {recall_f:.6f} "
        f"({'match' if abs(overall_o - recall_o) < 1e-9 and abs(overall_f - recall_f) < 1e-9 else 'MISMATCH'}).",
        f"- Observations per category equal the 2025 table T15: {'yes' if counts_match_t15 else 'NO'}.",
        f"- Observations per subset equal the 2025 table T12: {'yes' if counts_match_t12 else 'NO'}.",
        f"- BH over the five reported metrics equals the run's tables T03/T04: {'yes' if bh_ok and bh_matches else 'NO'}.",
        f"- Mean differences and p-values of the four comparisons equal the run's table T02 "
        f"({len(t02_rows)} rows): {'yes' if t02_ok else 'NO'}.",
        f"- Median temperature threshold: {threshold:.2f} °C.",
    ]
    write_text(out / "summary.md", lines)
    print(f"[SUPPLEMENTARY] run analyses written to {out}")


if __name__ == "__main__":
    main()
