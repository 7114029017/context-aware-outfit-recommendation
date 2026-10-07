#!/usr/bin/env python3
"""Text-length analysis of the main CIR experiment (manuscript Table 5, "Text length"; thesis Tables 4-4, 4-5).

The computation is copied from the 2025 notebook
03_實驗與結果_experiments_results/01_文字長度影響分析/source_programs/P12_length_defense_reproducible.ipynb
(cell 4): Original seed k is paired with Context seed k for each (set_id,
target_item_id); query length is the number of tokens of url_name + title
(Original) and of the generated description (Context); the outputs are the
row-level table (A07 layout), the correlations (A08), the length buckets (A09)
and the matched-length subsets (A10).

Input, one of:
- --detail-dir DIR: the per-query files written by main_cir_per_query.py
  (DIR/<original|context>_seed<k>/detail_cir_main.csv); this regenerates the
  analysis from a run's own models;
- --rowlevel FILE: an existing row-level table, for example the archived 2025
  A07 file; with it, the outputs must equal the archived A08-A10 tables.
CPU only, a few seconds.
"""
from __future__ import annotations

import argparse
import csv
import math
import re
from collections import defaultdict
from pathlib import Path
from statistics import mean

from scipy import stats as scipy_stats

from _ext import GENERATED, LENGTH_DIR, REPO, SEEDS, local_path, read_csv, read_json, write_text

TOKEN_RE = re.compile(r"[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)?")
ARCHIVED = {"A08": "A08_length_performance_correlation_summary.csv", "A09": "A09_length_bucket_performance_summary.csv",
            "A10": "A10_matched_length_subset_summary.csv"}


# ---------------------------------------------------------------- copied from P12 cells 2 and 4
def fmt_float(x, nd=4) -> str:
    try:
        if x is None or math.isnan(float(x)):
            return ""
        return f"{float(x):.{nd}f}"
    except Exception:
        return ""


def to_int(x, default=0) -> int:
    try:
        return int(float(x))
    except Exception:
        return default


def query_token_count(text: str) -> int:
    return len(TOKEN_RE.findall(str(text or "").strip()))


def original_url_name_plus_title(original_data: dict, set_id: str) -> str:
    value = original_data.get(str(set_id), "")
    if isinstance(value, dict):
        parts = [str(value.get("url_name", "") or "").strip(), str(value.get("title", "") or "").strip()]
        return " ".join(p for p in parts if p).strip()
    return str(value or "").strip()


def generated_title_only(generated_data: dict, set_id: str) -> str:
    value = generated_data.get(str(set_id), "")
    if isinstance(value, dict):
        return str(value.get("title", "") or "").strip()
    return str(value or "").strip()


def pearson_p(xs, ys):
    if len(xs) < 3 or len(set(xs)) <= 1 or len(set(ys)) <= 1:
        return None, None
    o = scipy_stats.pearsonr(xs, ys)
    return float(o.statistic), float(o.pvalue)


def spearman_p(xs, ys):
    if len(xs) < 3 or len(set(xs)) <= 1 or len(set(ys)) <= 1:
        return None, None
    o = scipy_stats.spearmanr(xs, ys)
    return float(o.statistic), float(o.pvalue)


def bucket_delta(n):
    if n <= 0: return "01_delta_le_0"  # noqa: E701
    if n <= 4: return "02_delta_1_4"  # noqa: E701
    if n <= 8: return "03_delta_5_8"  # noqa: E701
    if n <= 12: return "04_delta_9_12"  # noqa: E701
    return "05_delta_13_plus"


def bucket_gen(n):
    if n <= 8: return "01_gen_len_0_8"  # noqa: E701
    if n <= 12: return "02_gen_len_9_12"  # noqa: E701
    if n <= 16: return "03_gen_len_13_16"  # noqa: E701
    return "04_gen_len_17_plus"


def bucket_orig(n):
    if n <= 3: return "01_original_len_0_3"  # noqa: E701
    if n <= 6: return "02_original_len_4_6"  # noqa: E701
    if n <= 9: return "03_original_len_7_9"  # noqa: E701
    return "04_original_len_10_plus"


def summarize_perf(rows, label):
    oh = [to_int(r["original_hit10"]) for r in rows]
    fh = [to_int(r["full_hit10"]) for r in rows]
    orank = [to_int(r["original_rank"]) for r in rows]
    frank = [to_int(r["full_rank"]) for r in rows]
    olen = [to_int(r["original_query_len_tokens"]) for r in rows]
    glen = [to_int(r["generated_query_len_tokens"]) for r in rows]
    dlen = [to_int(r["length_delta_tokens"]) for r in rows]
    return {
        "group": label,
        "n_seed_rows": str(len(rows)),
        "n_unique_set_target_pairs": str(len({(r["set_id"], r["target_item_id"]) for r in rows})),
        "original_hit10": fmt_float(mean(oh), 4),
        "full_hit10": fmt_float(mean(fh), 4),
        "delta_hit10": fmt_float(mean([f - o for f, o in zip(fh, oh)]), 4),
        "original_mean_rank": fmt_float(mean(orank), 2),
        "full_mean_rank": fmt_float(mean(frank), 2),
        "mean_rank_improvement": fmt_float(mean([o - f for o, f in zip(orank, frank)]), 2),
        "mean_original_query_len": fmt_float(mean(olen), 2),
        "mean_generated_query_len": fmt_float(mean(glen), 2),
        "mean_length_delta": fmt_float(mean(dlen), 2),
    }
# ---------------------------------------------------------------- end of the copied code


def rowlevel_from_details(detail_dir: Path, polyvore: Path) -> list[dict]:
    original = read_json(polyvore / "polyvore_outfit_titles.json")
    generated = read_json(GENERATED)

    def load(prefix: str) -> dict:
        out = {}
        for seed in SEEDS:
            for r in read_csv(detail_dir / f"{prefix}_seed{seed}" / "detail_cir_main.csv"):
                out[(str(r["seed"]), str(r["set_id"]), str(r["target_item_id"]))] = r
        return out

    old, new = load("original"), load("context")
    if set(old) != set(new):
        raise SystemExit("[LENGTH BLOCKED] Original and Context per-query files cover different queries")
    rowlevel = []
    for seed, sid, iid in sorted(old, key=lambda x: (int(x[0]), int(x[1]), int(x[2]))):
        o, f = old[(seed, sid, iid)], new[(seed, sid, iid)]
        olen = query_token_count(original_url_name_plus_title(original, sid))
        glen = query_token_count(generated_title_only(generated, sid))
        orank, frank = to_int(o["rank"]), to_int(f["rank"])
        oh, fh = to_int(o["hit@10"]), to_int(f["hit@10"])
        rowlevel.append({
            "scope": "main_experiments_original_like",
            "seed": seed, "set_id": sid, "target_item_id": iid, "target_item_fg": str(o.get("target_item_fg", "")),
            "original_query_len_tokens": str(olen), "generated_query_len_tokens": str(glen),
            "length_delta_tokens": str(glen - olen),
            "generated_len_bucket": bucket_gen(glen), "length_delta_bucket": bucket_delta(glen - olen),
            "original_len_bucket": bucket_orig(olen),
            "original_rank": str(orank), "full_rank": str(frank), "rank_improvement": str(orank - frank),
            "original_hit10": str(oh), "full_hit10": str(fh), "hit10_delta": str(fh - oh),
        })
    return rowlevel


def analyse(rowlevel: list[dict]) -> tuple[list[dict], list[dict], list[dict]]:
    pair_acc = defaultdict(lambda: {"hit10_delta": [], "rank_improvement": []})
    for r in rowlevel:
        d = pair_acc[(r["set_id"], r["target_item_id"])]
        d["length_delta_tokens"] = float(r["length_delta_tokens"])
        d["original_query_len_tokens"] = float(r["original_query_len_tokens"])
        d["generated_query_len_tokens"] = float(r["generated_query_len_tokens"])
        d["hit10_delta"].append(float(r["hit10_delta"]))
        d["rank_improvement"].append(float(r["rank_improvement"]))
    pair_rows = [{"set_id": sid, "target_item_id": iid, "length_delta_tokens": d["length_delta_tokens"],
                  "original_query_len_tokens": d["original_query_len_tokens"],
                  "generated_query_len_tokens": d["generated_query_len_tokens"],
                  "hit10_delta": mean(d["hit10_delta"]), "rank_improvement": mean(d["rank_improvement"])}
                 for (sid, iid), d in pair_acc.items()]
    corr_rows = []
    for scope, rows in (("seed_row_level", rowlevel), ("pair_mean_across_seeds", pair_rows)):
        for xname in ("original_query_len_tokens", "generated_query_len_tokens", "length_delta_tokens"):
            for yname in ("hit10_delta", "rank_improvement"):
                xs = [float(r[xname]) for r in rows]
                ys = [float(r[yname]) for r in rows]
                pr, pp = pearson_p(xs, ys)
                sr, sp = spearman_p(xs, ys)
                abs_pr = abs(pr or 0)
                corr_rows.append({
                    "scope": scope, "x": xname, "y": yname, "n": str(len(rows)),
                    "pearson_r": fmt_float(pr, 6), "pearson_p": fmt_float(pp, 6),
                    "spearman_r": fmt_float(sr, 6), "spearman_p": fmt_float(sp, 6),
                    "practical_effect_size": "negligible" if abs_pr < 0.05 else ("small" if abs_pr < 0.10 else "inspect"),
                    "interpretation_hint": "Large N can make tiny correlations statistically significant; "
                                           "use effect size plus bucket/matched checks.",
                })
    bucket_rows = []
    for label, field in (("generated_query_length_bucket", "generated_len_bucket"),
                         ("length_delta_bucket", "length_delta_bucket"),
                         ("original_query_length_bucket", "original_len_bucket")):
        for b in sorted({r[field] for r in rowlevel}):
            bucket_rows.append({"bucket_type": label, **summarize_perf([r for r in rowlevel if r[field] == b], b)})
    matched = []
    for th in (0, 1, 2, 3, 5):
        subset = [r for r in rowlevel if abs(to_int(r["length_delta_tokens"])) <= th]
        matched.append({"matching_rule": f"abs(generated_query_len - original_query_len) <= {th} tokens",
                        **summarize_perf(subset, f"abs_delta_le_{th}")})
    return corr_rows, bucket_rows, matched


def write_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--detail-dir", type=Path, help="output folder of main_cir_per_query.py")
    source.add_argument("--rowlevel", type=Path, help="an A07-layout row-level table, e.g. the archived 2025 file")
    parser.add_argument("--polyvore-root", default=None)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--label", default=None, help="name of the analysed run, for summary.md")
    args = parser.parse_args()

    if args.detail_dir:
        polyvore = local_path("polyvore_root", args.polyvore_root)
        if polyvore is None:
            raise SystemExit("[LENGTH BLOCKED] Polyvore root not found; pass --polyvore-root or run bootstrap_data.sh")
        rowlevel = rowlevel_from_details(args.detail_dir.resolve(), polyvore)
        source_text = "the per-query files written by main_cir_per_query.py"
    else:
        rowlevel = read_csv(args.rowlevel)
        path = args.rowlevel.resolve()
        source_text = f"row-level table `{path.relative_to(REPO) if path.is_relative_to(REPO) else path.name}`"
    out = args.out_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    corr, buckets, matched = analyse(rowlevel)
    if args.detail_dir:
        write_rows(out / "A07_length_performance_rowlevel_cir.csv", rowlevel)
    write_rows(out / ARCHIVED["A08"], corr)
    write_rows(out / ARCHIVED["A09"], buckets)
    write_rows(out / ARCHIVED["A10"], matched)

    tables = LENGTH_DIR / "圖表_figures_tables" / "tables"
    same = {name: read_csv(out / file) == read_csv(tables / file) for name, file in ARCHIVED.items()}
    pair = {(r["x"], r["y"]): r for r in corr if r["scope"] == "pair_mean_across_seeds"}
    hit = pair[("length_delta_tokens", "hit10_delta")]
    rank = pair[("length_delta_tokens", "rank_improvement")]
    within5 = next(r for r in matched if r["group"] == "abs_delta_le_5")
    lines = [
        f"# Text-length analysis{f' of run `{args.label}`' if args.label else ''}",
        "",
        f"Source: {source_text}. Definitions copied from notebook P12 (01_文字長度影響分析).",
        "",
        "| Manuscript value (Table 5, Text length) | This analysis |",
        "|---|---:|",
        f"| N (query-target pairs) | {hit['n']} |",
        f"| Pearson r, length difference vs ΔHit@10 (pair means over seeds) | {hit['pearson_r']} (p = {hit['pearson_p']}) |",
        f"| Pearson r, length difference vs rank improvement | {rank['pearson_r']} (p = {rank['pearson_p']}) |",
        f"| Pairs within five tokens (abs difference <= 5) | {within5['n_unique_set_target_pairs']} |",
        f"| ΔHit@10 within five tokens (Full − Original) | {within5['delta_hit10']} |",
        "",
        "The manuscript reports N = 9,311, r = 0.027647 and 0.043384, n = 3,014 and ΔHit@10 = +0.0106 from the",
        "2025 models.",
        "",
        "Equal to the archived 2025 tables: " + ", ".join(f"{k} {'yes' if v else 'no'}" for k, v in same.items()) + ".",
    ]
    write_text(out / "summary.md", lines)
    print("\n".join(lines[4:11]))
    print(f"[LENGTH] written to {out}")


if __name__ == "__main__":
    main()
