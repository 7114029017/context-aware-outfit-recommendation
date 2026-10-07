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
- subset_robustness_summary.csv: the 2025 table T12 recomputed (Hit@10 and
  median rank of Full vs Original, and Hit@10 of Full vs each simplified
  description, by subset, with the union of each dimension and all queries);
- factor_category_effects.csv: Hit@10 and median rank by target category for
  Full vs Original and Full vs each simplified description, with the 2025
  values of T15;
- factor_term_effects.csv: the same by weather, occasion and style term
  (terms with at least 20 observations, as in 2025);
- figures/: thesis Figures 4-8 to 4-13 redrawn from this run (SVG);
- summary.md: the main numbers and the cross-checks.

The subset rules follow the 2025 notebook
03_實驗與結果_experiments_results/04_情境子集與三因子分析/source_programs/P02_subset_robustness_analysis.ipynb
(FORMAL_KEYWORDS, CASUAL_KEYWORDS, TEMP_PATTERN, the median temperature split and
METRIC_PAIRS), the category and term summaries the notebook P03 cell 4 in the
same folder (category_summary, term_summary with min_n=20, sorted by ΔHit@10 and
then by the number of observations).
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
from _svg import GREY, POSITIVE, grouped_vbar_chart, hbar_chart, vbar_chart

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
DIMENSIONS = (  # P02: subset dimensions in the order of table T12
    ("Weather: Cold vs Warm", ("Cold", "Warm")),
    ("Occasion: Formal vs Casual", ("Formal", "Casual")),
    ("Style Richness: High vs Low", ("High style", "Low style")),
    ("Category: Clothing vs Accessory", ("Clothing-led", "Accessory-led")),
)
T12_LABELS = {"High style": "High style (≥2 terms)", "Low style": "Low style (≤1 term)"}
METRIC_PAIRS = (  # P02 METRIC_PAIRS: left, right, T12 column; ranks by the median, Hit@10 by the mean
    ("original", "context", "Hit@10"),
    ("original", "context", "Median Rank"),
    ("no_weather", "context", "Hit@10 (no_weather→full)"),
    ("no_occasion", "context", "Hit@10 (no_occasion→full)"),
    ("no_style", "context", "Hit@10 (no_style→full)"),
)
FACTORS = ("weather", "occasion", "style")
CASE_TYPES = (("orig_to_full", "original"), ("weather", "no_weather"), ("occasion", "no_occasion"),
              ("style", "no_style"))  # T15 case types and the left condition; the right one is Full
TERM_MIN_N = 20  # P03 term_summary(min_n=20)
FIGURES_2025 = CASE_TABLES.parent / "figures"
TERM_FIGURES_2025 = {
    ("weather", "original"): "F09a_weather_terms_context_aware_vs_original.svg",
    ("weather", "no_weather"): "F09b_weather_term_contribution_no_weather_vs_full.svg",
    ("occasion", "original"): "F10a_occasion_terms_context_aware_vs_original.svg",
    ("occasion", "no_occasion"): "F10b_occasion_term_contribution_no_occasion_vs_full.svg",
    ("style", "original"): "F11a_style_terms_context_aware_vs_original.svg",
    ("style", "no_style"): "F11b_style_term_contribution_no_style_vs_full.svg",
}
THESIS_FIGURE = {"weather": "4-11", "occasion": "4-12", "style": "4-13"}


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


def evaluate_subset(members: list, label: str, dimension: str, by_key: dict) -> dict:
    """P02 evaluate_subset for the five METRIC_PAIRS."""
    row = {"n": len(members), "label": label}
    for left, right, name in METRIC_PAIRS:
        if name == "Median Rank":
            lv = float(statistics.median(by_key[k][left][0] for k in members))
            rv = float(statistics.median(by_key[k][right][0] for k in members))
        else:
            lv = mean(by_key[k][left][1] for k in members)
            rv = mean(by_key[k][right][1] for k in members)
        row[f"{name}_left"], row[f"{name}_right"], row[f"Δ{name}"] = lv, rv, rv - lv
    row["dimension"] = dimension
    return row


def summarize_groups(groups: dict, by_key: dict, left: str, min_n: int = 1) -> list[dict]:
    """P03 category_summary / term_summary: Full (right) vs `left`, sorted by ΔHit@10 and n, descending."""
    rows = []
    for name, members in groups.items():
        if len(members) < min_n:
            continue
        lh = mean(by_key[k][left][1] for k in members)
        rh = mean(by_key[k]["context"][1] for k in members)
        lr = float(statistics.median(by_key[k][left][0] for k in members))
        rr = float(statistics.median(by_key[k]["context"][0] for k in members))
        rows.append({"group": name, "n": len(members), "left_hit10": lh, "right_hit10": rh, "left_rank": lr,
                     "right_rank": rr, "delta_hit10": rh - lh, "delta_rank": rr - lr})
    # pandas sort_values is stable and groupby returns the groups sorted by key
    rows.sort(key=lambda r: str(r["group"]))
    rows.sort(key=lambda r: (-r["delta_hit10"], -r["n"]))
    return rows


def archived_bar_values(path: Path) -> list[tuple[str, float]]:
    """Bar labels and value labels of a 2025 matplotlib bar chart, top bar first (3 decimals as displayed)."""
    if not path.is_file():
        return []
    comments = re.findall(r"<!-- (.*?) -->", path.read_text(encoding="utf-8"))
    start = next((i for i, c in enumerate(comments) if "Hit@10" in c), None)
    if start is None:
        return []
    rest = comments[start + 1:]
    values = [c for c in rest if re.fullmatch(r"[+-]\d+\.\d+", c)]
    labels = [c for c in rest if not re.fullmatch(r"[+-]\d+\.\d+", c)]
    if not values or len(labels) != len(values):
        return []
    return list(zip(labels, (float(v) for v in values)))[::-1]


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

    by_key = defaultdict(dict)  # query -> condition -> (rank, hit@10)
    for prefix in VARIANTS:
        for r in rows[prefix]:
            by_key[(r["seed"], r["set_id"], r["target_item_id"])][prefix] = (r["rank"], r["hit@10"])
    if set(by_key) != set(keys) or any(len(v) != len(VARIANTS) for v in by_key.values()):
        raise SystemExit("[SUPPLEMENTARY BLOCKED] the five conditions cover different queries")

    categories = {}
    labels = {}  # P03 map_category: "name (group)", first occurrence of an ID
    for record in (poly / "categories.csv").read_text(encoding="utf-8-sig").splitlines():
        parts = record.split(",")
        if len(parts) >= 3 and parts[0].strip():
            categories.setdefault(parts[0].strip(), (norm_category(parts[1]), norm_category(parts[2])))
            name, group = parts[1].strip(), parts[2].strip()
            labels.setdefault(parts[0].strip(), f"{name} ({group})" if group else name)

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
            "terms": {f: frag.get(f) if isinstance(frag.get(f), list) else [] for f in FACTORS},
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

    # ---------------------------------------------------------------- table T12 and Figures 4-9, 4-10
    tests = {name: test for name, _, test in subsets}
    t12_rows = []
    for dimension, names in DIMENSIONS:
        union = set()
        for name in names:
            members = [k for k in keys if tests[name](info[k])]
            union.update(members)
            t12_rows.append(evaluate_subset(members, T12_LABELS.get(name, name), dimension, by_key))
        t12_rows.append(evaluate_subset([k for k in keys if k in union], "All (union)", dimension, by_key))
    t12_rows.append(evaluate_subset(keys, "Full dataset", "Overall", by_key))
    t12_archived = read_csv(CASE_TABLES / "T12_subset_robustness_summary.csv")
    t12_columns = list(t12_archived[0])
    write_csv(out / "subset_robustness_summary.csv", t12_columns,
              [[r[c] if isinstance(r[c], (int, str)) else repr(r[c]) for c in t12_columns] for r in t12_rows])
    plotted = [r for r in t12_rows if r["dimension"] != "Overall" and r["label"] != "All (union)"]
    tick_labels = [f"{r['label'].split(' (')[0]}\n(n={r['n']})" for r in plotted]
    grouped_vbar_chart(out / "figures" / "figure_4_9_subset_hit10.svg",
                       "Hit@10 by context subset: Original vs Full (thesis Figure 4-9)", tick_labels,
                       [("Original", [r["Hit@10_left"] for r in plotted], GREY),
                        ("Full (context-aware)", [r["Hit@10_right"] for r in plotted], POSITIVE)],
                       "Hit@10", breaks=(2, 4, 6), subtitle=f"Run {source['run_id']}, fair subset, five seeds pooled")
    vbar_chart(out / "figures" / "figure_4_10_median_rank_improvement.svg",
               "Median rank improvement, Original to Full, by subset (thesis Figure 4-10)", tick_labels,
               [0.0 - r["ΔMedian Rank"] for r in plotted], "Rank improvement (rank positions)", breaks=(2, 4, 6),
               subtitle=f"Run {source['run_id']}; improvement = median rank of Original minus median rank of Full")

    # ---------------------------------------------------------------- category and term effects (P03 cell 4)
    category_members = defaultdict(list)
    for key in keys:
        category_members[original[key]["target_item_fg"]].append(key)
    t15 = {(r["case_type"], r["target_item_fg"]): r for r in read_csv(CASE_TABLES / "T15_qualitative_category_summary.csv")}
    category_effects = {}
    effect_csv = []
    for case_type, left in CASE_TYPES:
        summary_rows = summarize_groups(category_members, by_key, left)
        category_effects[case_type] = summary_rows
        for r in summary_rows:
            old = t15.get((case_type, r["group"]), {})
            effect_csv.append([case_type, f"Full - {VARIANTS[left]}", r["group"], labels.get(r["group"], r["group"]),
                               r["n"], fmt(r["left_hit10"], 6), fmt(r["right_hit10"], 6), signed(r["delta_hit10"], 6),
                               fmt(r["left_rank"], 1), fmt(r["right_rank"], 1), signed(r["delta_rank"], 1),
                               old.get("n", ""), signed(float(old["delta_hit10"]), 6) if old else "",
                               signed(float(old["delta_rank"]), 1) if old else ""])
    write_csv(out / "factor_category_effects.csv",
              ["case_type", "comparison", "target_item_fg", "category", "observations", "left_hit10", "full_hit10",
               "delta_hit10", "left_median_rank", "full_median_rank", "delta_median_rank", "observations_2025",
               "delta_hit10_2025", "delta_median_rank_2025"], effect_csv)

    term_effects = {}
    term_csv = []
    for factor in FACTORS:
        groups = defaultdict(list)  # P03 explode_terms: one entry per listed term, empty terms dropped
        for key in keys:
            for term in fragments.get(key[1], {}).get("terms", {}).get(factor, []):
                if term is not None and str(term).strip():
                    groups[term].append(key)
        for left in ("original", f"no_{factor}"):
            summary_rows = summarize_groups(groups, by_key, left, TERM_MIN_N)
            shown_2025 = dict(archived_bar_values(FIGURES_2025 / TERM_FIGURES_2025[(factor, left)]))
            term_effects[(factor, left)] = (summary_rows, shown_2025, len(groups))
            for position, r in enumerate(summary_rows, 1):
                term_csv.append([factor, f"Full - {VARIANTS[left]}", position, r["group"], r["n"],
                                 fmt(r["left_hit10"], 6), fmt(r["right_hit10"], 6), signed(r["delta_hit10"], 6),
                                 fmt(r["left_rank"], 1), fmt(r["right_rank"], 1), signed(r["delta_rank"], 1),
                                 signed(shown_2025[r["group"]], 3) if r["group"] in shown_2025 else ""])
    write_csv(out / "factor_term_effects.csv",
              ["factor", "comparison", "position", "term", "observations", "left_hit10", "full_hit10", "delta_hit10",
               "left_median_rank", "full_median_rank", "delta_median_rank", "delta_hit10_2025_top8_figure"], term_csv)

    def category_chart(name, case_type, title, xlabel):
        shown = category_effects[case_type][:8]  # P03 plot_paper_category_bar(top_n=8)
        hbar_chart(out / "figures" / name, title, [labels.get(r["group"], r["group"]) for r in shown],
                   [r["delta_hit10"] for r in shown], xlabel,
                   subtitle=f"Run {source['run_id']}, fair subset, five seeds pooled; target categories")

    category_chart("figure_4_8_category_original_to_full.svg", "orig_to_full",
                   "Hit@10 change by target category (thesis Figure 4-8)", "ΔHit@10 (Full − Original)")
    for factor in FACTORS:
        no = f"No-{factor.capitalize()}"
        figure = THESIS_FIGURE[factor]
        category_chart(f"figure_{figure.replace('-', '_')}a_{factor}_categories.svg", factor,
                       f"Categories most sensitive to {factor} information (thesis Figure {figure} (a))",
                       f"ΔHit@10 (Full − {no})")
        for left, part, title in (("original", "b", f"{factor.capitalize()} terms improved by the context-aware description"),
                                  (f"no_{factor}", "c", f"{factor.capitalize()} term contribution")):
            summary_rows, _, total = term_effects[(factor, left)]
            shown = summary_rows[:8]
            hbar_chart(out / "figures" / f"figure_{figure.replace('-', '_')}{part}_{factor}_terms_"
                       f"{'original' if left == 'original' else 'no_' + factor}_to_full.svg",
                       f"{title} (thesis Figure {figure} ({part}))", [str(r["group"]) for r in shown],
                       [r["delta_hit10"] for r in shown], f"ΔHit@10 (Full − {VARIANTS[left]})",
                       subtitle=f"Run {source['run_id']}; top 8 of {len(summary_rows)} terms with at least "
                                f"{TERM_MIN_N} observations ({total} distinct terms)")

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
    t12_same_n = (len(t12_rows) == len(t12_archived)
                  and all(str(r["n"]) == a["n"] and r["label"] == a["label"] and r["dimension"] == a["dimension"]
                          for r, a in zip(t12_rows, t12_archived)))
    t15_same_n = all(str(r["n"]) == t15.get((case_type, r["group"]), {}).get("n")
                     for case_type, rs in category_effects.items() for r in rs) and \
        len(t15) == sum(len(rs) for rs in category_effects.values())
    old_t12 = {(a["dimension"], a["label"]): a for a in t12_archived}
    subset_lines = []
    for r in t12_rows:
        a = old_t12.get((r["dimension"], r["label"]), {})
        before = signed(0.0 - float(a["ΔMedian Rank"]), 1) if a else ""
        subset_lines.append(
            f"| {r['label']}{' (' + r['dimension'].split(':')[0] + ')' if r['label'] == 'All (union)' else ''} "
            f"| {r['n']} | {r['Median Rank_left']:.1f} → {r['Median Rank_right']:.1f} | {signed(0.0 - r['ΔMedian Rank'], 1)} "
            f"| {before} | {signed(r['ΔHit@10 (no_weather→full)'])} | {signed(r['ΔHit@10 (no_occasion→full)'])} "
            f"| {signed(r['ΔHit@10 (no_style→full)'])} |")
    effect_by = {(ct, r["group"]): r for ct, rs in category_effects.items() for r in rs}
    category_lines = []
    for r in category_effects["orig_to_full"]:
        cells = []
        for case_type, _ in CASE_TYPES:
            e, a = effect_by[(case_type, r["group"])], t15.get((case_type, r["group"]))
            cells.append(f"{signed(e['delta_hit10'], 3)} ({signed(float(a['delta_hit10']), 3) if a else '–'})")
        category_lines.append(f"| {labels.get(r['group'], r['group'])} | {r['n']} | " + " | ".join(cells) + " |")
    term_lines = []
    for factor in FACTORS:
        for left in ("original", f"no_{factor}"):
            summary_rows, shown_2025, total = term_effects[(factor, left)]
            mine = {r["group"]: r for r in summary_rows}
            top8 = {r["group"] for r in summary_rows[:8]}
            part = "b" if left == "original" else "c"
            term_lines += [
                "",
                f"**{factor.capitalize()} terms, Full − {VARIANTS[left]} (thesis Figure {THESIS_FIGURE[factor]} ({part}))**: "
                f"{len(summary_rows)} of {total} distinct terms have at least {TERM_MIN_N} observations; terms of the "
                f"2025 figure again in this run's top 8: {sum(1 for t in shown_2025 if t in top8)} of {len(shown_2025)}.",
                "",
                "| Rank | This run: term | ΔHit@10 | Observations | 2025 figure: term | ΔHit@10 (2025) | This run |",
                "|---:|---|---:|---:|---|---:|---:|",
            ]
            shown_list = list(shown_2025.items())
            for i in range(8):
                r = summary_rows[i] if i < len(summary_rows) else None
                old_term, old_value = shown_list[i] if i < len(shown_list) else ("", None)
                now = mine.get(old_term)
                term_lines.append(
                    f"| {i + 1} | {r['group'] if r else ''} | {signed(r['delta_hit10'], 3) if r else ''} "
                    f"| {r['n'] if r else ''} | {old_term} | {signed(old_value, 3) if old_value is not None else ''} "
                    f"| {signed(now['delta_hit10'], 3) + ' (n=' + str(now['n']) + ')' if now else ('below 20 observations' if old_term else '')} |")
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
        "## Subsets: median rank and factor contributions (2025 table T12; thesis Figures 4-9 and 4-10)",
        "",
        "Median rank improvement = median rank of Original minus median rank of Full (positive: the target",
        "moves up). The last three columns are Hit@10 of Full minus Hit@10 of the simplified description.",
        "Full table: `subset_robustness_summary.csv` (columns of T12).",
        "",
        "| Subset | Observations | Median rank, Original → Full | Improvement | 2025 | No-Weather → Full | "
        "No-Occasion → Full | No-Style → Full |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
        *subset_lines,
        "",
        "## Category effects of each factor (thesis Figures 4-8 and 4-11 (a) to 4-13 (a))",
        "",
        "ΔHit@10 of Full minus the other condition; 2025 values (table T15) in parentheses.",
        "",
        "| Category | Observations | Full − Original | Full − No-Weather | Full − No-Occasion | Full − No-Style |",
        "|---|---:|---:|---:|---:|---:|",
        *category_lines,
        "",
        "## Term effects of each factor (thesis Figures 4-11 (b, c) to 4-13 (b, c))",
        "",
        "Terms are the weather, occasion and style fragments of each outfit; terms with fewer than 20",
        "observations are left out, as in 2025. The 2025 columns are the eight bars of the archived figure",
        "(three decimals as displayed) and this run's value for the same term. All terms:",
        "`factor_term_effects.csv`.",
        *term_lines,
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
        f"- Observations per subset equal the 2025 table T12: {'yes' if counts_match_t12 else 'NO'}; "
        f"all 13 rows of T12, including the unions: {'yes' if t12_same_n else 'NO'}.",
        f"- Observations per category in all four comparisons equal the 2025 table T15: "
        f"{'yes' if t15_same_n else 'NO'}.",
        f"- BH over the five reported metrics equals the run's tables T03/T04: {'yes' if bh_ok and bh_matches else 'NO'}.",
        f"- Mean differences and p-values of the four comparisons equal the run's table T02 "
        f"({len(t02_rows)} rows): {'yes' if t02_ok else 'NO'}.",
        f"- Median temperature threshold: {threshold:.2f} °C.",
    ]
    write_text(out / "summary.md", lines)
    print(f"[SUPPLEMENTARY] run analyses written to {out}")


if __name__ == "__main__":
    main()
