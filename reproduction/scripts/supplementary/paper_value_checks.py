#!/usr/bin/env python3
"""Recompute manuscript values that come from the input data or from 2025 outputs.

Writes, under --out-dir (default reproduction/results/supplementary/paper_value_checks/):

- table1_dataset_scope.csv: split sizes, CP / FITB / CIR evaluation scope;
- table2_text_fields.csv: missing values and lengths of the original text
  (url_name + title) and of the generated descriptions;
- table5_proxy_values.csv: distribution of the CLO, MET and temperature
  reference values;
- table8_two_tower.csv: Two-Tower means, SDs and paired 95% confidence
  intervals from the preserved per-seed outputs (2025, not retrained);
- table9_counterfactual.csv: rank change, Top-1 change and Top-5 Jaccard of the
  24 counterfactual cases from the preserved retrieval outputs (2025 seed-1
  model);
- summary.md.

The official run's own tables are produced by the reproduction pipeline; this
script covers the values that do not come from the 35 training units.
"""
from __future__ import annotations

import argparse
import math
import re
from collections import defaultdict
from pathlib import Path
from statistics import mean, median

import numpy as np
from scipy import stats

from _common import (COUNTERFACTUAL, GENERATED, REPRO, SUPPLEMENTARY, TWO_TOWER_SEEDS, fmt, load_temperatures,
                     polyvore_root, read_csv, read_json, signed, write_csv, write_text)

WORD = re.compile(r"[A-Za-z]+(?:[-'][A-Za-z]+)?")
TOKEN = re.compile(r"[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)?")
TWO_TOWER_METRICS = ("cp_test_auc", "cp_test_fitb_acc", "recall@10", "recall@30", "recall@50", "mean_rank",
                     "median_rank")


def text_stats(texts: list[str]) -> list:
    words = [len(WORD.findall(t)) for t in texts]
    tokens = [len(TOKEN.findall(t)) for t in texts]
    n = len(texts)
    empty = sum(1 for w in words if w == 0)
    short = sum(1 for w in words if w <= 3)
    return [n, empty, f"{100 * empty / n:.2f}", fmt(mean(words), 2), fmt(float(median(words)), 2), max(words),
            fmt(mean(tokens), 2), f"{100 * short / n:.2f}"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--polyvore-root", default=None)
    parser.add_argument("--out-dir", type=Path, default=SUPPLEMENTARY / "paper_value_checks")
    args = parser.parse_args()
    out = args.out_dir
    poly = polyvore_root(args.polyvore_root)
    disjoint = poly / "disjoint"

    # ---------------------------------------------------------------- Table 1
    sizes = {s: len(read_json(disjoint / f"{s}.json")) for s in ("train", "valid", "test")}
    compat = [line.split() for line in (disjoint / "compatibility_test.txt").read_text(encoding="utf-8").splitlines()
              if line.strip()]
    fitb = read_json(disjoint / "fill_in_blank_test.json")
    or_rows = read_csv(REPRO / "splits" / "or_ids.csv")
    generated = read_json(GENERATED)
    titles = read_json(poly / "polyvore_outfit_titles.json")
    scope = [
        ["train outfits", sizes["train"]], ["validation outfits", sizes["valid"]], ["test outfits", sizes["test"]],
        ["all outfits", sum(sizes.values())],
        ["generated descriptions", len(generated)],
        ["CP test pairs", len(compat)],
        ["CP positive pairs", sum(1 for c in compat if c[0] == "1")],
        ["CP negative pairs", sum(1 for c in compat if c[0] == "0")],
        ["FITB questions", len(fitb)],
        ["CIR evaluable queries", len(or_rows)],
        ["CIR excluded queries (no 3,000-item pool)", len(fitb) - len(or_rows)],
        ["CIR target fine-grained categories", len({r["target_item_fg"] for r in or_rows})],
        ["original text records", len(titles)],
    ]
    write_csv(out / "table1_dataset_scope.csv", ["quantity", "value"], scope)

    # ---------------------------------------------------------------- Table 2
    ids = sorted(generated, key=lambda x: int(x) if x.isdigit() else x)

    def original(set_id: str) -> str:
        value = titles.get(set_id, "")
        if isinstance(value, dict):
            parts = [str(value.get("url_name", "") or "").strip(), str(value.get("title", "") or "").strip()]
            return " ".join(p for p in parts if p).strip()
        return str(value or "").strip()

    def generated_text(set_id: str) -> str:
        value = generated[set_id]
        return str(value.get("title", "") if isinstance(value, dict) else value).strip()

    header = ["text", "n", "missing", "missing_percent", "mean_words", "median_words", "max_words", "mean_tokens",
              "at_most_3_words_percent"]
    write_csv(out / "table2_text_fields.csv", header, [
        ["original (url_name + title)", *text_stats([original(s) for s in ids])],
        ["generated description", *text_stats([generated_text(s) for s in ids])],
    ])

    # ---------------------------------------------------------------- Table 5 proxy values
    records = [r for _, r in load_temperatures().values()]
    clo = np.array([float(r["total_clo"]) for r in records])
    met = np.array([float(r["met_value"]) for r in records])
    tsub = np.array([float(r["TSUB_target_C"]) for r in records])
    n = len(records)

    def dist(name: str, values: np.ndarray, digits: int) -> list:
        q = np.percentile(values, [5, 25, 50, 75, 95])
        return [name, n, fmt(q[2], digits), fmt(q[1], digits), fmt(q[3], digits), fmt(q[0], digits),
                fmt(q[4], digits), fmt(float(values.max()), digits)]

    write_csv(out / "table5_proxy_values.csv",
              ["variable", "n", "median", "q1", "q3", "p5", "p95", "max"],
              [dist("CLO", clo, 2), dist("MET", met, 2), dist("temperature reference (°C)", tsub, 1)])
    counts = {
        "CLO > 4": int((clo > 4).sum()),
        "MET = 1.0": int((met == 1.0).sum()),
        "MET > 10": int((met > 10).sum()),
        "temperature < 0 °C": int((tsub < 0).sum()),
        "temperature < -20 °C": int((tsub < -20).sum()),
    }

    # ---------------------------------------------------------------- Table 8 (2025 Two-Tower outputs)
    by_variant = defaultdict(dict)
    for row in read_csv(TWO_TOWER_SEEDS):
        by_variant[row["variant"]][int(row["seed"])] = row
    seeds = sorted(by_variant["original_text"])
    t_crit = stats.t.ppf(0.975, len(seeds) - 1)
    table8 = []
    for metric in TWO_TOWER_METRICS:
        o = np.array([float(by_variant["original_text"][s][metric]) for s in seeds])
        c = np.array([float(by_variant["context_aware_description"][s][metric]) for s in seeds])
        d = c - o
        half = t_crit * d.std(ddof=1) / math.sqrt(len(seeds))
        digits = 4
        table8.append([metric, f"{o.mean():.{digits}f} ± {o.std(ddof=1):.{digits}f}",
                       f"{c.mean():.{digits}f} ± {c.std(ddof=1):.{digits}f}", signed(float(d.mean()), digits),
                       f"[{d.mean() - half:.{digits}f}, {d.mean() + half:.{digits}f}]"])
    write_csv(out / "table8_two_tower.csv",
              ["metric", "original_mean_sd", "context_aware_mean_sd", "mean_difference", "paired_95_ci"], table8)

    # ---------------------------------------------------------------- Table 9 (2025 counterfactual outputs)
    rows = read_csv(COUNTERFACTUAL / "A45_counterfactual_context_retrieval_seed1.csv")
    base = {r["pair_id"]: r for r in rows if r["condition"] == "context_aware"}
    changed = {r["pair_id"]: r for r in rows if r["condition"] == "counterfactual"}
    cases = []
    for pair_id, b in base.items():
        c = changed[pair_id]
        top_b, top_c = set(b["top5_ids"].split(" | ")), set(c["top5_ids"].split(" | "))
        cases.append({"factor": b["factor"], "r0": float(b["rank"]), "r1": float(c["rank"]),
                      "top1": int(str(b["top1_id"]) != str(c["top1_id"])),
                      "jaccard": len(top_b & top_c) / len(top_b | top_c)})

    def summarise(label: str, group: list[dict]) -> list:
        return [label, len(group), fmt(mean(g["r0"] for g in group), 1), fmt(mean(g["r1"] for g in group), 1),
                signed(mean(g["r1"] - g["r0"] for g in group), 1), fmt(mean(g["top1"] for g in group), 3),
                fmt(mean(g["jaccard"] for g in group), 3)]

    table9 = [summarise("Overall", cases)]
    for factor in ("weather", "occasion", "style"):
        table9.append(summarise(factor.capitalize(), [g for g in cases if g["factor"] == factor]))
    write_csv(out / "table9_counterfactual.csv",
              ["scope", "n", "mean_rank_before", "mean_rank_after", "mean_rank_change", "top1_changed_share",
               "top5_jaccard"], table9)

    # ---------------------------------------------------------------- summary
    lines = [
        "# Manuscript values recomputed from the input data and from 2025 outputs",
        "",
        "Computed by `reproduction/scripts/supplementary/paper_value_checks.py`. Tables 6 and 7 come from the",
        "official run and are produced by the reproduction pipeline; the values here do not come from the 35",
        "training units. Table 8 and Table 9 use preserved 2025 outputs (the Two-Tower model was not retrained;",
        "the counterfactual analysis used the 2025 seed-1 model).",
        "",
        "## Table 1",
        "",
        *[f"- {q}: {v}" for q, v in scope],
        "",
        "## Table 2",
        "",
        "See `table2_text_fields.csv` (missing values, mean / median / max words, mean tokens).",
        "",
        "## Table 5, proxy values",
        "",
        "See `table5_proxy_values.csv`. Counts: " + "; ".join(f"{k}: {v}" for k, v in counts.items()) + ".",
        "",
        "## Table 8 (2025 Two-Tower outputs)",
        "",
        "| Metric | Original | Context-aware | Difference | Paired 95% CI |",
        "|---|---|---|---:|---|",
        *[f"| {r[0]} | {r[1]} | {r[2]} | {r[3]} | {r[4]} |" for r in table8],
        "",
        "## Table 9 (2025 counterfactual outputs)",
        "",
        "| Scope | n | Mean rank before → after | Mean rank change | Top-1 changed | Top-5 Jaccard |",
        "|---|---:|---|---:|---:|---:|",
        *[f"| {r[0]} | {r[1]} | {r[2]} → {r[3]} | {r[4]} | {r[5]} | {r[6]} |" for r in table9],
    ]
    write_text(out / "summary.md", lines)
    print(f"[SUPPLEMENTARY] manuscript value checks written to {out}")


if __name__ == "__main__":
    main()
