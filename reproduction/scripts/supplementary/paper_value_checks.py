#!/usr/bin/env python3
"""Recompute manuscript values that come from the input data or from 2025 outputs.

Writes, under --out-dir (default reproduction/results/supplementary/paper_value_checks/):

- table1_dataset_scope.csv: split sizes, CP / FITB / CIR evaluation scope;
- table2_text_fields.csv: missing values and lengths of the original text
  (url_name + title) and of the generated descriptions;
- table5_proxy_values.csv: distribution of the CLO, MET and temperature
  reference values;
- table5_target_clues.csv: the target-clue audit (notebook P12) with the
  category labels resolved as the 2025 programs did;
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
import csv
import math
import re
from collections import defaultdict
from pathlib import Path
from statistics import mean, median

import numpy as np
from scipy import stats

from _common import (COUNTERFACTUAL, GENERATED, REPRO, SUPPLEMENTARY, TARGET_CLUE, TWO_TOWER_SEEDS, fmt,
                     load_temperatures, polyvore_root, read_csv, read_json, signed, write_csv, write_text)

WORD = re.compile(r"[A-Za-z]+(?:[-'][A-Za-z]+)?")
TOKEN = re.compile(r"[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)?")
TWO_TOWER_METRICS = ("cp_test_auc", "cp_test_fitb_acc", "recall@10", "recall@30", "recall@50", "mean_rank",
                     "median_rank")
# Target-clue audit, copied from notebook P12 cell 4 (02_目標單品線索檢查 in the 2025 folders).
CLUE_STOP = set("""
a an and are as at be by for from in into is it its of on or the this that these those to with without your you
women woman womens female fashion clothing clothes outfit outfits wear wearing style styles styled look looks item items piece pieces
new used vintage size small medium large plus petite one pair set collection design designer brand brands online shop shopping
classic modern casual chic sleek elegant effortless comfy cozy cool warm cold hot day days night nights spring summer fall autumn winter
vibe vibes scene mood ready friendly perfect practical realistic recommended temperature original title url name
""".split())
CLUE_WORD = re.compile(r"[a-zA-Z][a-zA-Z'-]*")
CLUE_METRICS = (
    ("audited_unique_main_cir_queries", "CIR query-target pairs"),
    ("generated_fine_category_lexical_match", "fine-category match in the generated description"),
    ("generated_any_category_lexical_match", "fine- or main-category match in the generated description"),
    ("generated_target_item_distinctive_token_overlap", "target-item token overlap, generated description"),
    ("generated_target_item_distinctive_bigram_overlap", "target-item bigram overlap, generated description"),
    ("original_target_item_distinctive_token_overlap", "target-item token overlap, original text"),
)


def text_stats(texts: list[str]) -> list:
    words = [len(WORD.findall(t)) for t in texts]
    tokens = [len(TOKEN.findall(t)) for t in texts]
    n = len(texts)
    empty = sum(1 for w in words if w == 0)
    short = sum(1 for w in words if w <= 3)
    return [n, empty, f"{100 * empty / n:.2f}", fmt(mean(words), 2), fmt(float(median(words)), 2), max(words),
            fmt(mean(tokens), 2), f"{100 * short / n:.2f}"]


def clue_words(s) -> list[str]:
    return [w.strip("'-").lower() for w in CLUE_WORD.findall(str(s or "").lower()) if w.strip("'-")]


def clue_category_match(text: str, fine: str, major: str) -> tuple[bool, bool]:
    text_words = set(clue_words(text))
    text_phrase = " ".join(clue_words(text))

    def match_label(label: str) -> bool:
        ws = clue_words(label)
        return bool(ws) and (" ".join(ws) in text_phrase or any(w in text_words for w in ws))
    return match_label(fine), match_label(major)


def clue_meta_parts(meta: dict) -> list[str]:
    return [str(meta.get(k, "") or "").strip() for k in ("url_name", "title", "description", "semantic_category")
            if str(meta.get(k, "") or "").strip()]


def clue_tokens(meta: dict, fine: str, major: str) -> list[str]:
    blocked = set(clue_words(fine)) | set(clue_words(major)) | CLUE_STOP
    toks = []
    for part in clue_meta_parts(meta):
        toks.extend(w for w in clue_words(part) if len(w) >= 4 and w not in blocked)
    return sorted(set(toks))


def clue_bigrams(meta: dict, fine: str, major: str) -> list[str]:
    toks = []
    for part in clue_meta_parts(meta):
        ws = [w for w in clue_words(part) if w not in CLUE_STOP and len(w) >= 3]
        toks.extend(" ".join(ws[i:i + 2]) for i in range(len(ws) - 1))
    blocked = {" ".join(clue_words(fine)), " ".join(clue_words(major))}
    return sorted({b for b in toks if b and b not in blocked})


def target_clue_counts(pairs: list[tuple[str, str, str]], labels: dict, generated_text, original_text,
                       item_meta: dict) -> tuple[list[int], dict]:
    """The six P12 counts and the fine-category label used for each pair."""
    counts = [0] * len(CLUE_METRICS)
    used = {}
    for set_id, item_id, fg in pairs:
        fine, major = labels.get(fg, ("", ""))
        used[(set_id, item_id)] = fine
        gen, orig = generated_text(set_id), original_text(set_id)
        meta = item_meta.get(item_id, {})
        toks, bigrams = clue_tokens(meta, fine, major), clue_bigrams(meta, fine, major)
        gen_words, orig_words = set(clue_words(gen)), set(clue_words(orig))
        gen_phrase = " ".join(clue_words(gen))
        gf, gm = clue_category_match(gen, fine, major)
        hits = (True, gf, gf or gm, any(t in gen_words for t in toks), any(b in gen_phrase for b in bigrams),
                any(t in orig_words for t in toks))
        counts = [c + int(h) for c, h in zip(counts, hits)]
    return counts, used


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
    # ---------------------------------------------------------------- Table 5 target clues
    # categories.csv lists some category IDs more than once with different labels. The 2025 programs keep the
    # first occurrence (P02 and P04: drop_duplicates(keep="first")); the frozen pipeline's secondary step
    # (reproduce_remaining_thesis.py, run_target_clue) keeps the last one.
    first_labels, last_labels, seen_ids = {}, {}, []
    with (poly / "categories.csv").open("r", encoding="utf-8-sig", newline="") as f:
        for row in csv.reader(f):
            if row and row[0].strip():
                label = (row[1].strip().lower() if len(row) > 1 else "", row[2].strip().lower() if len(row) > 2 else "")
                first_labels.setdefault(row[0].strip(), label)
                last_labels[row[0].strip()] = label
                seen_ids.append(row[0].strip())
    duplicated = {i for i in seen_ids if seen_ids.count(i) > 1}
    relabelled = {i for i in duplicated if first_labels[i] != last_labels[i]}
    pairs = list({(r["set_id"], r["target_item_id"]): (r["set_id"], r["target_item_id"], r["target_item_fg"])
                  for r in or_rows}.values())
    item_meta = read_json(poly / "polyvore_item_metadata.json")
    first_counts, first_used = target_clue_counts(pairs, first_labels, generated_text, original, item_meta)
    last_counts, last_used = target_clue_counts(pairs, last_labels, generated_text, original, item_meta)
    archived = {r["metric"]: r for r in read_csv(TARGET_CLUE / "圖表_figures_tables" / "tables"
                                                 / "A03_target_item_clue_leakage_audit_summary.csv")}
    n_pairs = first_counts[0]
    clue_rows = []
    for (metric, label), first, last in zip(CLUE_METRICS, first_counts, last_counts):
        old = archived.get(metric, {})
        clue_rows.append([metric, label, old.get("count", ""), old.get("rate_pct", ""), first,
                          f"{100 * first / n_pairs:.2f}", last, f"{100 * last / n_pairs:.2f}"])
    write_csv(out / "table5_target_clues.csv",
              ["metric", "description", "count_2025_A03", "percent_2025_A03", "count_first_occurrence",
               "percent_first_occurrence", "count_last_occurrence", "percent_last_occurrence"], clue_rows)
    examples = read_csv(TARGET_CLUE / "A04_target_item_clue_leakage_examples.csv")
    examples_first = sum(1 for e in examples if first_used.get((e["set_id"], e["target_item_id"])) == e["fine_category"])
    examples_last = sum(1 for e in examples if last_used.get((e["set_id"], e["target_item_id"])) == e["fine_category"])
    clue_equal = all(str(r[4]) == r[2] for r in clue_rows)

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
        "## Table 5, target clues",
        "",
        f"`categories.csv` lists {len(duplicated)} category IDs more than once, {len(relabelled)} of them with different",
        "labels. The 2025 programs keep the first occurrence; the frozen pipeline's secondary step",
        "(`secondary/target_clue/`) keeps the last one. With the 2025 rule the six counts equal the archived table",
        f"A03 and the manuscript: {'yes' if clue_equal else 'NO'}. Fine-category labels of the {len(examples)} archived",
        f"examples (A04): {examples_first} match the 2025 rule, {examples_last} the last-occurrence rule.",
        "",
        "| Metric | 2025 (A03) | Recomputed, 2025 rule | Last occurrence (pipeline) |",
        "|---|---:|---:|---:|",
        *[f"| {r[1]} | {r[2]} ({r[3]}%) | {r[4]} ({r[5]}%) | {r[6]} ({r[7]}%) |" for r in clue_rows[1:]],
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
