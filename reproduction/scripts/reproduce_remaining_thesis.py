#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import statistics
import subprocess
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
REPRO_ROOT = ROOT / "reproduction"
REPRO_SCRIPTS = REPRO_ROOT / "scripts"
DATA = ROOT / "01_資料建構_data_construction"
MODEL = ROOT / "02_模型訓練和驗證_model_training_validation"
EXP = ROOT / "03_實驗與結果_experiments_results"

STOP = set("""
a an and are as at be by for from in into is it its of on or the this that these those to with without your you
women woman womens female fashion clothing clothes outfit outfits wear wearing style styles styled look looks item items piece pieces
new used vintage size small medium large plus petite one pair set collection design designer brand brands online shop shopping
classic modern casual chic sleek elegant effortless comfy cozy cool warm cold hot day days night nights spring summer fall autumn winter
vibe vibes scene mood ready friendly perfect practical realistic recommended temperature original title url name
""".split())


def git(*args):
    p = subprocess.run(["git", *args], cwd=ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return p.stdout.strip() if p.returncode == 0 else ""


def rel(path):
    try:
        return str(Path(path).resolve().relative_to(ROOT))
    except Exception:
        return Path(path).name


def read_csv(path):
    with Path(path).open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"No rows to write: {path}")
    fields = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                fields.append(key)
                seen.add(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
        w.writeheader()
        w.writerows(rows)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def read_jsonl(path):
    out = []
    with Path(path).open("r", encoding="utf-8-sig") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def num(x):
    return float(x)


def integer(x):
    return int(float(x))


def compare_tables(a, b, atol=1e-9):
    if len(a) != len(b):
        return {"same": False, "rows_a": len(a), "rows_b": len(b)}
    bad = 0
    maxdiff = 0.0
    for ra, rb in zip(a, b):
        for key in set(ra) | set(rb):
            va, vb = ra.get(key, ""), rb.get(key, "")
            if str(va) == str(vb):
                continue
            try:
                d = abs(float(va) - float(vb))
                maxdiff = max(maxdiff, d)
                if d > atol:
                    bad += 1
            except Exception:
                bad += 1
    return {"same": bad == 0, "mismatched_cells": bad, "max_numeric_abs_diff": maxdiff}


def result(module, status, evidence, outputs=None, details=None):
    return {
        "module": module,
        "status": status,
        "evidence": evidence,
        "outputs": outputs or [],
        "details": details or {},
    }


def load_scores(path):
    out = {}
    for obj in read_jsonl(path):
        if obj.get("score") is None:
            continue
        out[str(obj["id"])] = {
            "score": float(obj["score"]),
            "decisions": {
                str(item.get("id")): str(item.get("answer", "")).strip().lower()
                for item in obj.get("decisions", [])
            },
        }
    return out


def words(text):
    import re
    return [
        w.strip("'-").lower()
        for w in re.findall(r"[A-Za-z][A-Za-z'-]*", str(text or "").lower())
        if w.strip("'-")
    ]


def normalize_token_text(text):
    import re
    x = str(text or "").strip().lower()
    x = re.sub(r"[_/|,-]+", " ", x)
    return re.sub(r"\s+", " ", x).strip()


def title_only(data, sid):
    value = data.get(str(sid), "")
    if isinstance(value, dict):
        return str(value.get("title", "") or "").strip()
    return str(value or "").strip()


def original_text(data, sid):
    value = data.get(str(sid), "")
    if isinstance(value, dict):
        return " ".join(
            x for x in [
                str(value.get("url_name", "") or "").strip(),
                str(value.get("title", "") or "").strip(),
            ] if x
        )
    return str(value or "").strip()


def run_length(out_root):
    out = out_root / "length"
    out.mkdir(parents=True, exist_ok=True)
    base = EXP / "01_文字長度影響分析"
    rowlevel = read_csv(base / "A07_length_performance_rowlevel_cir.csv")
    if len(rowlevel) != 46555:
        return result("length", "failed", f"A07 row count={len(rowlevel)}, expected 46555.")

    pair_acc = defaultdict(lambda: {"hit10_delta": [], "rank_improvement": []})
    for r in rowlevel:
        key = (r["set_id"], r["target_item_id"])
        d = pair_acc[key]
        d["original_query_len_tokens"] = float(r["original_query_len_tokens"])
        d["generated_query_len_tokens"] = float(r["generated_query_len_tokens"])
        d["length_delta_tokens"] = float(r["length_delta_tokens"])
        d["hit10_delta"].append(float(r["hit10_delta"]))
        d["rank_improvement"].append(float(r["rank_improvement"]))

    pair_rows = []
    for (sid, iid), d in pair_acc.items():
        pair_rows.append({
            "set_id": sid,
            "target_item_id": iid,
            "original_query_len_tokens": d["original_query_len_tokens"],
            "generated_query_len_tokens": d["generated_query_len_tokens"],
            "length_delta_tokens": d["length_delta_tokens"],
            "hit10_delta": statistics.mean(d["hit10_delta"]),
            "rank_improvement": statistics.mean(d["rank_improvement"]),
        })

    corr = []
    for scope, rows in [("seed_row_level", rowlevel), ("pair_mean_across_seeds", pair_rows)]:
        for xname in ["original_query_len_tokens", "generated_query_len_tokens", "length_delta_tokens"]:
            for yname in ["hit10_delta", "rank_improvement"]:
                xs = [float(r[xname]) for r in rows]
                ys = [float(r[yname]) for r in rows]
                pr = stats.pearsonr(xs, ys)
                sr = stats.spearmanr(xs, ys)
                rr = float(pr.statistic)
                corr.append({
                    "scope": scope, "x": xname, "y": yname, "n": str(len(rows)),
                    "pearson_r": f"{rr:.6f}", "pearson_p": f"{float(pr.pvalue):.6f}",
                    "spearman_r": f"{float(sr.statistic):.6f}", "spearman_p": f"{float(sr.pvalue):.6f}",
                    "practical_effect_size": "negligible" if abs(rr) < .05 else ("small" if abs(rr) < .10 else "inspect"),
                    "interpretation_hint": "Large N can make tiny correlations statistically significant; use effect size plus bucket/matched checks.",
                })
    write_csv(out / "A08_recomputed.csv", corr)

    def summarize(rows, label):
        oh = [integer(r["original_hit10"]) for r in rows]
        fh = [integer(r["full_hit10"]) for r in rows]
        orank = [integer(r["original_rank"]) for r in rows]
        frank = [integer(r["full_rank"]) for r in rows]
        olen = [integer(r["original_query_len_tokens"]) for r in rows]
        glen = [integer(r["generated_query_len_tokens"]) for r in rows]
        dlen = [integer(r["length_delta_tokens"]) for r in rows]
        return {
            "group": label,
            "n_seed_rows": str(len(rows)),
            "n_unique_set_target_pairs": str(len({(r["set_id"], r["target_item_id"]) for r in rows})),
            "original_hit10": f"{statistics.mean(oh):.4f}",
            "full_hit10": f"{statistics.mean(fh):.4f}",
            "delta_hit10": f"{statistics.mean([f-o for f,o in zip(fh, oh)]):.4f}",
            "original_mean_rank": f"{statistics.mean(orank):.2f}",
            "full_mean_rank": f"{statistics.mean(frank):.2f}",
            "mean_rank_improvement": f"{statistics.mean([o-f for o,f in zip(orank, frank)]):.2f}",
            "mean_original_query_len": f"{statistics.mean(olen):.2f}",
            "mean_generated_query_len": f"{statistics.mean(glen):.2f}",
            "mean_length_delta": f"{statistics.mean(dlen):.2f}",
        }

    buckets = []
    for label, field in [
        ("generated_query_length_bucket", "generated_len_bucket"),
        ("length_delta_bucket", "length_delta_bucket"),
        ("original_query_length_bucket", "original_len_bucket"),
    ]:
        for value in sorted({r[field] for r in rowlevel}):
            buckets.append({"bucket_type": label, **summarize([r for r in rowlevel if r[field] == value], value)})
    write_csv(out / "A09_recomputed.csv", buckets)

    matched = []
    for th in [0, 1, 2, 3, 5]:
        sub = [r for r in rowlevel if abs(integer(r["length_delta_tokens"])) <= th]
        matched.append({
            "matching_rule": f"abs(generated_query_len - original_query_len) <= {th} tokens",
            **summarize(sub, f"abs_delta_le_{th}"),
        })
    write_csv(out / "A10_recomputed.csv", matched)

    tab = base / "圖表_figures_tables/tables"
    checks = {
        "A08": compare_tables(corr, read_csv(tab / "A08_length_performance_correlation_summary.csv"), 5e-7),
        "A09": compare_tables(buckets, read_csv(tab / "A09_length_bucket_performance_summary.csv"), 5e-7),
        "A10": compare_tables(matched, read_csv(tab / "A10_matched_length_subset_summary.csv"), 5e-7),
    }
    exact = len(pair_rows) == 9311 and all(v["same"] for v in checks.values())
    return result(
        "length",
        "exact" if exact else "partial",
        f"Recomputed length correlations/buckets/matched subsets from 46,555 archived seed-level rows ({len(pair_rows)} unique pairs).",
        [rel(out / "A08_recomputed.csv"), rel(out / "A09_recomputed.csv"), rel(out / "A10_recomputed.csv")],
        {"comparisons": checks},
    )


def run_target_clue(out_root, polyvore_root):
    out = out_root / "target_clue"
    out.mkdir(parents=True, exist_ok=True)
    rows = read_csv(REPRO_ROOT / "splits/or_ids.csv")
    metadata = read_json(polyvore_root / "polyvore_item_metadata.json")
    originals = read_json(polyvore_root / "polyvore_outfit_titles.json")
    generated = read_json(DATA / "generated_descriptions/01_生成結果_generation_results/new_polyvore_outfit_titles.json")
    wos_path = DATA / "generated_descriptions/02_三因子拆分_wos_factor_split/wos_split_results_v5_merged_retry_round3.jsonl"
    wos_titles = {}
    for obj in read_jsonl(wos_path):
        if obj.get("valid") is True:
            wos_titles[str(obj.get("id", ""))] = str(obj.get("title", "") or "").strip()

    categories = {}
    with (polyvore_root / "categories.csv").open("r", encoding="utf-8-sig", newline="") as f:
        for row in csv.reader(f):
            if row:
                categories[str(row[0]).strip()] = (
                    normalize_token_text(row[1]) if len(row) > 1 else "",
                    normalize_token_text(row[2]) if len(row) > 2 else "",
                )

    def cat_match(text, fine, major):
        twords = set(words(text))
        phrase = " ".join(words(text))
        def one(label):
            ws = words(label)
            return bool(ws) and (" ".join(ws) in phrase or any(w in twords for w in ws))
        return one(fine), one(major)

    def meta_parts(meta):
        return [
            str(meta.get(k, "") or "").strip()
            for k in ["url_name", "title", "description", "semantic_category"]
            if str(meta.get(k, "") or "").strip()
        ]

    def distinctive_tokens(meta, fine, major):
        blocked = set(words(fine)) | set(words(major)) | STOP
        toks = []
        for part in meta_parts(meta):
            toks.extend(w for w in words(part) if len(w) >= 4 and w not in blocked)
        return sorted(set(toks))

    def distinctive_bigrams(meta, fine, major):
        toks = []
        for part in meta_parts(meta):
            ws = [w for w in words(part) if w not in STOP and len(w) >= 3]
            toks.extend(" ".join(ws[i:i+2]) for i in range(len(ws)-1))
        blocked = {" ".join(words(fine)), " ".join(words(major))}
        return sorted({b for b in toks if b and b not in blocked})

    audited = []
    seen = set()
    for r in rows:
        sid, iid = str(r["set_id"]), str(r["target_item_id"])
        if (sid, iid) in seen:
            continue
        seen.add((sid, iid))
        fine, major = categories.get(str(r.get("target_item_fg", "")), ("", ""))
        gt = wos_titles.get(sid, "") or title_only(generated, sid)
        ot = original_text(originals, sid)
        meta = metadata.get(iid, {})
        toks, bigs = distinctive_tokens(meta, fine, major), distinctive_bigrams(meta, fine, major)
        gw, ow = set(words(gt)), set(words(ot))
        gp, op = " ".join(words(gt)), " ".join(words(ot))
        gf, gm = cat_match(gt, fine, major)
        of, om = cat_match(ot, fine, major)
        gtok = sorted(t for t in toks if t in gw)
        gbig = sorted(b for b in bigs if b in gp)
        otok = sorted(t for t in toks if t in ow)
        obig = sorted(b for b in bigs if b in op)
        audited.append({
            "set_id": sid, "target_item_id": iid, "fine_category": fine, "major_category": major,
            "generated_text": gt, "original_text": ot,
            "target_item_title_or_url": str(meta.get("title", "") or meta.get("url_name", "") or ""),
            "generated_fine_category_match": str(gf), "generated_any_category_match": str(gf or gm),
            "generated_target_token_overlap_count": str(len(gtok)), "generated_target_token_overlap": " | ".join(gtok[:12]),
            "generated_target_bigram_overlap_count": str(len(gbig)), "generated_target_bigram_overlap": " | ".join(gbig[:12]),
            "original_fine_category_match": str(of), "original_any_category_match": str(of or om),
            "original_target_token_overlap_count": str(len(otok)), "original_target_token_overlap": " | ".join(otok[:12]),
            "original_target_bigram_overlap_count": str(len(obig)), "original_target_bigram_overlap": " | ".join(obig[:12]),
        })

    audited.sort(key=lambda r: (int(r["set_id"]), int(r["target_item_id"])))
    # Preserve all 9,311 CURRENT reconstruction rows so that archived
    # historical examples can be inspected even when they are absent from
    # this run's top-80 example ranking. These are NOT historical P12 rows.
    write_csv(out / "A03_current_rowlevel_recomputed.csv", audited)

    # The original P12 source used an unarchived
    # subset_analysis_results/reliability_meta_from_subset.csv containing
    # fine_category, major_category, title_full_text and original_text.
    # A04 contains up to 80 *historical output examples*, not that complete
    # input table. Compare these concrete overlapping IDs field by field to
    # diagnose observed input drift, without claiming it explains all 9,311.
    archived_examples_path = (
        EXP / "02_目標單品線索檢查/A04_target_item_clue_leakage_examples.csv"
    )
    archived_examples = read_csv(archived_examples_path)
    current_by_pair = {(r["set_id"], r["target_item_id"]): r for r in audited}
    archived_compared_fields = [
        "fine_category", "major_category", "generated_text", "original_text",
        "target_item_title_or_url",
        "generated_fine_category_match", "generated_any_category_match",
        "generated_target_token_overlap_count", "generated_target_bigram_overlap_count",
        "original_target_token_overlap_count",
    ]
    archive_comparison = []
    archived_seen = set()
    for old_row in archived_examples:
        pair = (str(old_row["set_id"]), str(old_row["target_item_id"]))
        if pair in archived_seen:
            raise ValueError(f"Duplicated archived A04 example ID pair: {pair}")
        archived_seen.add(pair)
        new_row = current_by_pair.get(pair)
        same = {field: (str(old_row.get(field, "")) == str(new_row.get(field, "")))
                if new_row is not None else False
                for field in archived_compared_fields}
        archive_comparison.append({
            "set_id": pair[0], "target_item_id": pair[1],
            "historical_A04_row_found_in_current_scope": new_row is not None,
            **{field + "_matches_archived_A04": value for field, value in same.items()},
            "historical_fine_category": old_row.get("fine_category", ""),
            "current_fine_category": (new_row or {}).get("fine_category", ""),
            "historical_major_category": old_row.get("major_category", ""),
            "current_major_category": (new_row or {}).get("major_category", ""),
            "historical_generated_text": old_row.get("generated_text", ""),
            "current_generated_text": (new_row or {}).get("generated_text", ""),
            "historical_original_text": old_row.get("original_text", ""),
            "current_original_text": (new_row or {}).get("original_text", ""),
            "historical_target_item_title_or_url": old_row.get("target_item_title_or_url", ""),
            "current_target_item_title_or_url": (new_row or {}).get("target_item_title_or_url", ""),
            "historical_generated_fine_category_match": old_row.get("generated_fine_category_match", ""),
            "current_generated_fine_category_match": (new_row or {}).get("generated_fine_category_match", ""),
            "historical_generated_any_category_match": old_row.get("generated_any_category_match", ""),
            "current_generated_any_category_match": (new_row or {}).get("generated_any_category_match", ""),
            "historical_generated_target_token_overlap_count": old_row.get(
                "generated_target_token_overlap_count", ""),
            "current_generated_target_token_overlap_count": (new_row or {}).get(
                "generated_target_token_overlap_count", ""),
            "historical_generated_target_bigram_overlap_count": old_row.get(
                "generated_target_bigram_overlap_count", ""),
            "current_generated_target_bigram_overlap_count": (new_row or {}).get(
                "generated_target_bigram_overlap_count", ""),
            "historical_original_target_token_overlap_count": old_row.get(
                "original_target_token_overlap_count", ""),
            "current_original_target_token_overlap_count": (new_row or {}).get(
                "original_target_token_overlap_count", ""),
        })
    if len(archive_comparison) != len(archived_examples):
        raise AssertionError("Archived example comparison row count mismatch")
    write_csv(out / "A04_archived_example_provenance_comparison.csv", archive_comparison)
    source_check = {
        "historical_P12_input": "subset_analysis_results/reliability_meta_from_subset.csv",
        "historical_P12_input_archived": False,
        "historical_A04_example_count": len(archived_examples),
        "historical_A04_example_source_sha256": sha256(archived_examples_path),
        "archived_examples_found_in_current_9311_scope": sum(
            bool(r["historical_A04_row_found_in_current_scope"])
            for r in archive_comparison
        ),
        "field_exact_matches_among_archived_examples": {
            field: sum(bool(r[field + "_matches_archived_A04"])
                       for r in archive_comparison)
            for field in archived_compared_fields
        },
        "field_first_10_mismatched_pairs": {
            field: [
                [r["set_id"], r["target_item_id"]]
                for r in archive_comparison
                if not r[field + "_matches_archived_A04"]
            ][:10]
            for field in archived_compared_fields
        },
        "limitation": (
            "Archived A04 contains a selected historical top-example subset, "
            "not the missing 9,311-row historical P12 input. An observed "
            "difference demonstrates field-level drift on that example only; "
            "agreement cannot establish identity of unarchived rows or fully "
            "attribute the aggregate count mismatch."
        ),
    }
    (out / "A04_archived_example_provenance_summary.json").write_text(
        json.dumps(source_check, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8"
    )

    n = len(audited)
    def count(fn): return sum(1 for r in audited if fn(r))
    def pct(v): return f"{100*v/n:.2f}"
    vals = [
        count(lambda a: a["generated_fine_category_match"] == "True"),
        count(lambda a: a["generated_any_category_match"] == "True"),
        count(lambda a: integer(a["generated_target_token_overlap_count"]) > 0),
        count(lambda a: integer(a["generated_target_bigram_overlap_count"]) > 0),
        count(lambda a: integer(a["original_target_token_overlap_count"]) > 0),
    ]
    summary = [
        {"metric": "audited_unique_main_cir_queries", "count": str(n), "rate_pct": "100.00", "interpretation": "Main CIR query-target pairs, not no_missing subset."},
        {"metric": "generated_fine_category_lexical_match", "count": str(vals[0]), "rate_pct": pct(vals[0]), "interpretation": "Static category-token check; category mentions can be semantically natural and are not automatically answer leakage."},
        {"metric": "generated_any_category_lexical_match", "count": str(vals[1]), "rate_pct": pct(vals[1]), "interpretation": "Fine or major category mention in generated query."},
        {"metric": "generated_target_item_distinctive_token_overlap", "count": str(vals[2]), "rate_pct": pct(vals[2]), "interpretation": "Potential direct target-item lexical clue; inspect examples."},
        {"metric": "generated_target_item_distinctive_bigram_overlap", "count": str(vals[3]), "rate_pct": pct(vals[3]), "interpretation": "Stricter phrase-level clue check."},
        {"metric": "original_target_item_distinctive_token_overlap", "count": str(vals[4]), "rate_pct": pct(vals[4]), "interpretation": "Baseline reference: original url_name + title can also contain item-like tokens."},
    ]
    write_csv(out / "A03_recomputed.csv", summary)

    examples = [
        a for a in audited
        if integer(a["generated_target_token_overlap_count"]) > 0
        or integer(a["generated_target_bigram_overlap_count"]) > 0
    ]
    examples = sorted(
        examples,
        key=lambda a: (
            integer(a["generated_target_bigram_overlap_count"]),
            integer(a["generated_target_token_overlap_count"]),
        ),
        reverse=True,
    )[:80]
    write_csv(out / "A04_recomputed_examples.csv", examples)

    archived = read_csv(EXP / "02_目標單品線索檢查/圖表_figures_tables/tables/A03_target_item_clue_leakage_audit_summary.csv")
    cmp = compare_tables(summary, archived, 1e-12)
    archived_counts = {r["metric"]: int(float(r["count"])) for r in archived}
    recomputed_counts = {r["metric"]: int(float(r["count"])) for r in summary}
    return result(
        "target_clue",
        "exact" if cmp["same"] else "partial",
        (
            f"Recomputed target-clue leakage on the exact OR scope N={n}. "
            "The historical P12 intermediate source reliability_meta_from_subset.csv "
            "is not archived in the repository, so current reconstruction uses preserved "
            "OR IDs plus available WOS/Polyvore metadata. Scope and bigram count reproduce "
            "exactly, while several lexical-count rows differ."
        ),
        [rel(out / "A03_recomputed.csv"), rel(out / "A04_recomputed_examples.csv"),
         rel(out / "A03_current_rowlevel_recomputed.csv"),
         rel(out / "A04_archived_example_provenance_comparison.csv"),
         rel(out / "A04_archived_example_provenance_summary.json")],
        {
            "comparison": cmp,
            "archived_example_provenance_diagnostic": source_check,
            "historical_intermediate_source_archived": False,
            "historical_intermediate_source": "subset_analysis_results/reliability_meta_from_subset.csv",
            "archived_counts": archived_counts,
            "recomputed_counts": recomputed_counts,
        },
    )


def run_judge(out_root, bootstrap_n):
    out = out_root / "judge"
    out.mkdir(parents=True, exist_ok=True)
    qdir = DATA / "llm_judge_checklists/Qwen3_Instruct"
    gdir = DATA / "llm_judge_checklists/Gemma3"
    q = load_scores(qdir / "phase3_scores_Qwen3VL32B.jsonl")
    g = load_scores(gdir / "phase3_scores_Gemma3.jsonl")
    ids = sorted(set(q) & set(g))
    xa = np.array([q[i]["score"] for i in ids])
    xb = np.array([g[i]["score"] for i in ids])

    pr = stats.pearsonr(xa, xb)
    sr = stats.spearmanr(xa, xb)
    kr = stats.kendalltau(xa, xb, variant="b")
    index = np.arange(len(ids))
    pboot = np.empty(bootstrap_n)
    sboot = np.empty(bootstrap_n)

    # Historical P05 resets np.random.default_rng(seed=123) separately for
    # Pearson and Spearman bootstrap calls. Preserve that exact RNG contract.
    rng_p = np.random.default_rng(123)
    for b in range(bootstrap_n):
        samp = rng_p.choice(index, size=len(index), replace=True)
        pboot[b] = stats.pearsonr(xa[samp], xb[samp]).statistic

    rng_s = np.random.default_rng(123)
    for b in range(bootstrap_n):
        samp = rng_s.choice(index, size=len(index), replace=True)
        sboot[b] = stats.pearsonr(stats.rankdata(xa[samp]), stats.rankdata(xb[samp])).statistic

    pci = np.quantile(pboot, [.025, .975])
    sci = np.quantile(sboot, [.025, .975])

    def bins(x, n=10):
        qs = np.quantile(x, np.linspace(0, 1, n + 1))
        return np.digitize(x, qs[1:-1], right=True)

    def qwk(a, b, n=10):
        cm = np.zeros((n, n), dtype=float)
        for i, j in zip(a, b):
            cm[int(i), int(j)] += 1
        w = np.fromfunction(lambda i, j: ((i-j)**2)/((n-1)**2), (n, n))
        expected = np.outer(cm.sum(1), cm.sum(0)) / cm.sum()
        return float(1 - (w*cm).sum()/(w*expected).sum())

    agreement = [{
        "N_intersection": len(ids),
        "A_mean": float(xa.mean()), "A_std": float(xa.std(ddof=1)),
        "B_mean": float(xb.mean()), "B_std": float(xb.std(ddof=1)),
        "Pearson_r": float(pr.statistic), "Pearson_p": float(pr.pvalue),
        "Pearson_CI_lo": float(pci[0]), "Pearson_CI_hi": float(pci[1]),
        "Spearman_rho": float(sr.statistic), "Spearman_p": float(sr.pvalue),
        "Spearman_CI_lo": float(sci[0]), "Spearman_CI_hi": float(sci[1]),
        "Kendall_tau_b": float(kr.statistic), "Kendall_p": float(kr.pvalue),
        "QWK_bins": qwk(bins(xa), bins(xb)),
    }]
    write_csv(out / "T19_recomputed.csv", agreement)

    thresholds = []
    for t in [.8, .9, 1.0]:
        pa, pb = xa >= t, xb >= t
        thresholds.append({
            "threshold": t,
            "agreement": float((pa == pb).mean()),
            "pass_rate_A": float(pa.mean()),
            "pass_rate_B": float(pb.mean()),
        })
    write_csv(out / "T21_recomputed.csv", thresholds)

    def bottom(x, p):
        k = int(np.ceil(len(x) * p))
        idx = np.argsort(x)[:k]
        return {ids[i] for i in idx}

    bottom_rows = []
    for p in [.05, .10, .20]:
        A, B = bottom(xa, p), bottom(xb, p)
        inter = len(A & B)
        precision = inter / len(A)
        recall = inter / len(B)
        expected = p * p * len(ids)
        bottom_rows.append({
            "p(bottom)": p, "|LA|": len(A), "|LB|": len(B), "intersection": inter,
            "expected_random(p^2*N)": expected, "lift_over_random": inter/expected,
            "Jaccard": inter/len(A | B), "OverlapCoeff": inter/min(len(A), len(B)),
            "Precision(A->B)": precision, "Recall(A->B)": recall,
            "F1(A vs B)": 2*precision*recall/(precision+recall),
        })
    write_csv(out / "bottomp_recomputed.csv", bottom_rows)

    dfq = pd.DataFrame({"id": list(q), "score_qwen": [q[i]["score"] for i in q]})
    dfg = pd.DataFrame({"id": list(g), "score_gemma": [g[i]["score"] for i in g]})
    df = dfq.merge(dfg, on="id", how="inner")
    df["rank_qwen"] = df["score_qwen"].rank(pct=True, method="average")
    df["rank_gemma"] = df["score_gemma"].rank(pct=True, method="average")
    df["combined_rank"] = (df["rank_qwen"] + df["rank_gemma"]) / 2
    df["rank_gap"] = (df["rank_qwen"] - df["rank_gemma"]).abs()
    df = df.sort_values("combined_rank").reset_index(drop=True)
    n = len(df)
    low = df.iloc[:int(n*.20)]
    low = low[low["rank_gap"] <= .35]
    high = df.iloc[-int(n*.20):]
    high = high[high["rank_gap"] <= .35]
    sample = pd.concat([
        low.sample(n=75, random_state=42).assign(group="low"),
        high.sample(n=75, random_state=42).assign(group="high"),
    ], ignore_index=True).sample(frac=1, random_state=42).reset_index(drop=True)
    sampled_ids = set(sample["id"].astype(str))

    compare_files = {
        ("Qwen3-VL", "P0-R2"): qdir / "phase3_scores_Qwen3VL32B_robustness_run1_compare.csv",
        ("Qwen3-VL", "P1"): qdir / "phase3_scores_Qwen3VL32B_robustness_run1_compare_change.csv",
        ("Qwen3-VL", "P2"): qdir / "phase3_scores_Qwen3VL32B_robustness_run1_compare_conservative.csv",
        ("Gemma-3", "P0-R2"): gdir / "phase3_scores_Gemma3_robustness_run1_compare.csv",
        ("Gemma-3", "P1"): gdir / "phase3_scores_Gemma3_robustness_run1_compare_change.csv",
        ("Gemma-3", "P2"): gdir / "phase3_scores_Gemma3_robustness_run1_compare_conservative.csv",
    }
    robustness = []
    ids_match = True
    for (judge, variant), path in compare_files.items():
        rows = read_csv(path)
        here = {str(r["id"]) for r in rows}
        ids_match &= here == sampled_ids
        diffs = [float(r["abs_diff"]) for r in rows]
        robustness.append({
            "judge": judge, "variant": variant, "n": len(rows),
            "mean_abs_diff": statistics.mean(diffs),
            "median_abs_diff": statistics.median(diffs),
            # Table 4-8 also reports maximum absolute difference.  Preserve it
            # alongside the already verified mean/median/threshold statistics.
            "max_abs_diff": max(diffs),
            "within_0_05_rate": statistics.mean(str(r["within_0_05"]).lower() == "true" for r in rows),
            "within_0_10_rate": statistics.mean(str(r["within_0_10"]).lower() == "true" for r in rows),
            "sample_ids_match_seed42_reconstruction": here == sampled_ids,
        })
    write_csv(out / "prompt_robustness_summary.csv", robustness)

    # Thesis final PDF, Chapter 4, Table 4-8 (p. 47, printed pagination).
    # This is a display-precision numerical check of six PRESERVED compare
    # files, not a rerun of either LLM or a claim that the 2025 sampled IDs can
    # be reconstructed from the current sampling dataframe.
    table_4_8_paper = {
        ("Qwen3-VL", "P0-R2"): ("0.0048", "0.0000", "0.2727", "97.3%", "98.7%"),
        ("Gemma-3", "P0-R2"): ("0.0461", "0.0000", "0.3939", "62.0%", "89.3%"),
        ("Qwen3-VL", "P1"): ("0.0709", "0.0455", "0.6818", "64.0%", "72.0%"),
        ("Gemma-3", "P1"): ("0.0713", "0.0455", "0.4242", "50.0%", "75.3%"),
        ("Qwen3-VL", "P2"): ("0.0506", "0.0455", "0.4545", "80.0%", "85.3%"),
        ("Gemma-3", "P2"): ("0.0521", "0.0000", "0.6061", "71.3%", "84.0%"),
    }
    display_fields = (
        ("mean_abs_diff", "mean_abs_diff", lambda x: f"{x:.4f}"),
        ("median_abs_diff", "median_abs_diff", lambda x: f"{x:.4f}"),
        ("max_abs_diff", "max_abs_diff", lambda x: f"{x:.4f}"),
        ("within_0_05_rate", "within_0_05_rate", lambda x: f"{x * 100:.1f}%"),
        ("within_0_10_rate", "within_0_10_rate", lambda x: f"{x * 100:.1f}%"),
    )
    table_4_8_checks = []
    for row in robustness:
        key = (row["judge"], row["variant"])
        paper_fields = table_4_8_paper[key]
        for i, (field, output_field, render) in enumerate(display_fields):
            actual = render(float(row[output_field]))
            paper = paper_fields[i]
            table_4_8_checks.append({
                "judge": row["judge"], "variant": row["variant"],
                "n_archived": row["n"], "field": field,
                "paper_display": paper, "preserved_recomputed_display": actual,
                "paper_display_match": actual == paper,
            })
    write_csv(out / "table_4_8_paper_field_comparison.csv", table_4_8_checks)
    table_4_8_display_match = (
        len(robustness) == 6
        and len(table_4_8_checks) == 30
        and all(int(r["n"]) == 150 for r in robustness)
        and all(r["paper_display_match"] for r in table_4_8_checks)
    )

    tab = EXP / "00_控制檢查與附加稽核/圖表_figures_tables/tables"
    c19 = compare_tables(agreement, read_csv(tab / "T19_judge_qwen_gemma_agreement_summary.csv"), 1e-12 if bootstrap_n == 5000 else 1e-3)
    c21 = compare_tables(thresholds, read_csv(tab / "T21_judge_qwen_gemma_threshold_agreement.csv"), 1e-12)
    # Preserved-output numerical reproduction is distinct from whether the
    # historical 150-ID robustness sample can be regenerated from the current
    # dataframe/order/runtime. Verify the archived compare files themselves
    # against the preserved 35,140 formal outputs.
    archived_sets = []
    archived_score_mismatch = 0
    archived_missing_formal = 0
    for (judge, variant), path in compare_files.items():
        rows = read_csv(path)
        archived_sets.append({str(r["id"]) for r in rows})
        source = q if judge == "Qwen3-VL" else g
        for r in rows:
            sid = str(r["id"])
            if sid not in source:
                archived_missing_formal += 1
                continue
            if abs(float(r["original_score"]) - float(source[sid]["score"])) > 1e-12:
                archived_score_mismatch += 1

    all_six_same_ids = (
        len(archived_sets) == 6
        and all(x == archived_sets[0] for x in archived_sets[1:])
    )
    archived_ids = archived_sets[0] if archived_sets else set()
    sample_intersection = len(archived_ids & sampled_ids)

    numerical_exact = (
        len(ids) == 35140
        and c19["same"]
        and c21["same"]
        and all(r["n"] == 150 for r in robustness)
        and all_six_same_ids
        and archived_missing_formal == 0
        and archived_score_mismatch == 0
    )

    return result(
        "judge_preserved_outputs",
        "exact" if numerical_exact else "partial",
        (
            "Recomputed Judge T19 agreement, T21 threshold agreement, bottom-p "
            "overlaps, and preserved P0-R2/P1/P2 robustness outputs. The module's "
            "'exact' status applies to verified preserved-output T19/T21 and "
            "archived-robustness consistency; thesis Table 4-7 bottom-p overlaps "
            "are NOT exact (paper 668/1323/2833 vs current 669/1338/2877). "
            "Archived robustness original_score values match the preserved "
            "formal outputs; historical sample regeneration provenance is partial."
        ),
        [rel(out / "T19_recomputed.csv"), rel(out / "T21_recomputed.csv"), rel(out / "bottomp_recomputed.csv"),
         rel(out / "prompt_robustness_summary.csv"), rel(out / "table_4_8_paper_field_comparison.csv")],
        {
            "formal_intersection": len(ids),
            "table_4_7_paper_intersections": [668, 1323, 2833],
            "table_4_7_recomputed_intersections": [
                int(r["intersection"]) for r in bottom_rows
            ],
            "table_4_7_matches_paper": [
                int(r["intersection"]) for r in bottom_rows
            ] == [668, 1323, 2833],
            "T19_comparison": c19,
            "T21_comparison": c21,
            "table_4_8_paper_display_match_from_preserved_6x150": table_4_8_display_match,
            "table_4_8_paper_display_compared_fields": len(table_4_8_checks),
            "table_4_8_preserved_score_files_not_llm_rerun": True,
            "table_4_8_paper_comparison_csv": rel(out / "table_4_8_paper_field_comparison.csv"),
            "robustness_all_six_same_150_ids": all_six_same_ids,
            "robustness_missing_formal": archived_missing_formal,
            "robustness_original_score_mismatch": archived_score_mismatch,
            "historical_sample_regeneration_match": ids_match,
            "historical_sample_regeneration_intersection": sample_intersection,
            "historical_sample_regeneration_archived_only": len(archived_ids - sampled_ids),
            "historical_sample_regeneration_current_only": len(sampled_ids - archived_ids),
        },
    )


def run_human(out_root):
    out = out_root / "human_audit"
    out.mkdir(parents=True, exist_ok=True)
    hdir = EXP / "07_人工稽核與品質診斷"
    frame = read_csv(hdir / "A35_human_audit_30_sampling_frame.csv")
    archived = read_csv(hdir / "A36_human_audit_30_selected_cases.csv")
    manual = read_csv(hdir / "A38_human_audit_30_manual_results.csv")
    qdir = DATA / "llm_judge_checklists/Qwen3_Instruct"
    gdir = DATA / "llm_judge_checklists/Gemma3"
    q = load_scores(qdir / "phase3_scores_Qwen3VL32B.jsonl")
    g = load_scores(gdir / "phase3_scores_Gemma3.jsonl")
    qc = read_json(qdir / "checklist_C_star_Qwen3VL32B.json")["checklist"]
    gc = read_json(gdir / "checklist_C_star_Gemma3.json")["checklist"]

    rows = []
    for r in frame:
        z = dict(r)
        for key in ["combined_rank", "rank_gap", "score_gap"]:
            z[key] = float(z[key])
        rows.append(z)

    random.seed(42)
    low = [r for r in rows if r["combined_rank"] <= .25 and r["rank_gap"] <= .35]
    high = [r for r in rows if r["combined_rank"] >= .75 and r["rank_gap"] <= .35]
    conflict = [r for r in rows if r["rank_gap"] >= .50 or r["score_gap"] >= .25]

    def extremeness(r, stratum):
        if stratum == "low": return 1-r["combined_rank"]
        if stratum == "high": return r["combined_rank"]
        return r["rank_gap"] + r["score_gap"]

    def select(pool, stratum, n, used, covered):
        selected = []
        local = defaultdict(set)
        remaining = [r for r in pool if r["set_id"] not in used]
        dims = ["temp_bin", "occasion_bin", "met_bin", "style_bin"]
        while len(selected) < n and remaining:
            best, best_key = None, None
            for r in remaining:
                new_global = sum(r[d] not in covered[d] for d in dims)
                new_local = sum(r[d] not in local[d] for d in dims)
                key = (new_global*3 + new_local, extremeness(r, stratum), -random.random())
                if best_key is None or key > best_key:
                    best, best_key = r, key
            selected.append(best)
            used.add(best["set_id"])
            for d in dims:
                covered[d].add(best[d])
                local[d].add(best[d])
            remaining = [r for r in remaining if r["set_id"] not in used]
        return selected

    used, covered = set(), defaultdict(set)
    selected = []
    for stratum, part in [
        ("low", select(low, "low", 10, used, covered)),
        ("high", select(high, "high", 10, used, covered)),
        ("conflict", select(conflict, "conflict", 10, used, covered)),
    ]:
        for r in part:
            z = dict(r)
            z["audit_stratum"] = stratum
            selected.append(z)
    selected = sorted(selected, key=lambda r: (r["audit_stratum"], r["set_id"]))
    for i, r in enumerate(selected, 1):
        r["sample_order"] = str(i)

    sample_exact = [
        (r["set_id"], r["audit_stratum"], str(r["sample_order"])) for r in selected
    ] == [
        (r["set_id"], r["audit_stratum"], str(r["sample_order"])) for r in archived
    ]

    human = defaultdict(list)
    for r in manual:
        sid = str(r["set_id"])
        checklist = str(r["checklist"]).strip().lower()
        item = str(r["item_id"])
        ans = str(r.get("human_answer_yes", r.get("human_answer", ""))).strip().lower()
        if ans in {"yes", "y", "1", "true", "是"}:
            human[(sid, checklist, item)].append(1.0)
        elif ans in {"no", "n", "0", "false", "否"}:
            human[(sid, checklist, item)].append(0.0)
    human = {k: statistics.mean(v) for k, v in human.items()}
    checklists = {"qwen": qc, "gemma": gc}
    sources = {"qwen": q, "gemma": g}

    def human_score(sid, checklist):
        total = 0.0
        weight = 0.0
        for item in checklists[checklist]:
            key = (sid, checklist, str(item["id"]))
            if key in human:
                w = float(item.get("weight", 1))
                total += w * human[key]
                weight += w
        return total/weight if weight else None

    metrics = []
    selected_ids = [r["set_id"] for r in selected]
    for checklist in ["qwen", "gemma"]:
        hs, ms = [], []
        for sid in selected_ids:
            h = human_score(sid, checklist)
            if h is not None:
                hs.append(h)
                ms.append(sources[checklist][sid]["score"])
        diff = [m-h for h, m in zip(hs, ms)]
        metrics.append({
            "scope": "30_case_human_audit", "checklist": checklist, "n": len(hs),
            "bias_model_minus_human": f"{statistics.mean(diff):.4f}",
            "mae": f"{statistics.mean(abs(x) for x in diff):.4f}",
            "rmse": f"{math.sqrt(statistics.mean(x*x for x in diff)):.4f}",
            "pearson": f"{float(stats.pearsonr(hs, ms).statistic):.4f}",
            "spearman": f"{float(stats.spearmanr(hs, ms).statistic):.4f}",
            "human_mean": f"{statistics.mean(hs):.4f}",
            "model_mean": f"{statistics.mean(ms):.4f}",
            "human_sd": f"{statistics.stdev(hs):.4f}",
            "model_sd": f"{statistics.stdev(ms):.4f}",
        })
    write_csv(out / "T30_recomputed.csv", metrics)

    # Recompute Table 4-9 coverage *from the selected sample*, rather than
    # treating matching sample IDs as proof of every stratum/category count.
    coverage_dims = ("audit_stratum", "temp_bin", "met_bin", "occasion_bin", "style_bin")
    archived_coverage = read_csv(hdir / "A42_human_audit_30_coverage_summary.csv")
    coverage_rows = []
    all_categories_accounted_for = len(selected) == 30
    for dim in coverage_dims:
        observed = {r[dim] for r in selected}
        historical = {r["value"] for r in archived_coverage if r["dimension"] == dim}
        if observed != historical:
            all_categories_accounted_for = False
    for r in archived_coverage:
        dim, value = r["dimension"], r["value"]
        if dim not in coverage_dims:
            all_categories_accounted_for = False
            continue
        count = sum(row[dim] == value for row in selected)
        coverage_rows.append({
            "dimension": dim,
            "value": value,
            "n_samples": count,
            "ratio": f"{count / len(selected):.4f}" if selected else "",
        })
    write_csv(out / "A42_coverage_recomputed.csv", coverage_rows)
    coverage_comparison = compare_tables(coverage_rows, archived_coverage, 5e-5)
    coverage_exact = sample_exact and all_categories_accounted_for and coverage_comparison["same"]

    archived_t30 = read_csv(hdir / "圖表_figures_tables/tables/T30_human_audit30_model_human_score_metrics.csv")
    cmp = compare_tables(metrics, archived_t30, 5e-5)
    qn = sum(str(r["checklist"]).lower() == "qwen" for r in manual)
    gn = sum(str(r["checklist"]).lower() == "gemma" for r in manual)
    exact = sample_exact and len(manual) == 750 and qn == 300 and gn == 450 and cmp["same"]
    return result(
        "human_audit",
        "exact" if exact else "partial",
        f"Seed-42 30-case selection match={sample_exact}; manual judgments={len(manual)} ({qn} Qwen + {gn} Gemma).",
        [rel(out / "T30_recomputed.csv"), rel(out / "A42_coverage_recomputed.csv")],
        {
            "sample_selection_exact": sample_exact,
            "T30_comparison": cmp,
            "table_4_9_coverage_recomputed_from_selected_cases": True,
            "table_4_9_coverage_exact_against_archived_A42": coverage_exact,
            "table_4_9_coverage_comparison": coverage_comparison,
        },
    )


def run_case(out_root, polyvore_root, fair_subset_dir=None):
    out = out_root / "case_analysis"
    out.mkdir(parents=True, exist_ok=True)
    cdir = EXP / "04_情境子集與三因子分析"
    required = [
        cdir / "圖表_figures_tables/tables/T12_subset_robustness_summary.csv",
        cdir / "圖表_figures_tables/tables/T13_subset_delta_hit10_pivot.csv",
        cdir / "圖表_figures_tables/tables/T14_qualitative_condition_summary.csv",
        cdir / "圖表_figures_tables/tables/T15_qualitative_category_summary.csv",
        cdir / "圖表_figures_tables/tables/T16_qualitative_failure_cases.csv",
        cdir / "圖表_figures_tables/tables/T17_qualitative_user_cases.csv",
    ]
    fair_root = Path(fair_subset_dir).expanduser().resolve() if fair_subset_dir else (REPRO_ROOT / "runs/fair_subset_batch")
    fair = sorted(fair_root.glob("*_seed*/detail_cir_fresh_subset.csv"))
    summary = []
    for p in fair:
        variant, seed = p.parent.name.rsplit("_seed", 1)
        rows = read_csv(p)
        ranks = [integer(r["rank"]) for r in rows]
        hits = [integer(r["hit@10"]) for r in rows]
        summary.append({
            "variant": variant, "seed": int(seed), "n": len(rows),
            "hit10": statistics.mean(hits),
            "mean_rank": statistics.mean(ranks),
            "median_rank": statistics.median(ranks),
        })
    if summary:
        write_csv(out / "fresh_fair_subset_rowlevel_summary.csv", summary)

    notebook = cdir / "source_programs/P03_qualitative_case_analysis.ipynb"
    pointer = notebook.is_file() and b"git-lfs.github.com/spec/v1" in notebook.read_bytes()[:256]
    details = {
        "archived_tables_present": all(p.is_file() for p in required),
        "fresh_fair_subset_detail_files": len(fair),
        "qualitative_notebook_materialized": notebook.is_file() and not pointer,
        "images_available": (polyvore_root / "images").is_dir(),
        "historical_memberwise_fair_subset_identity_recovered": False,
    }
    return result(
        "case_analysis",
        "partial",
        f"Archived subgroup/case tables present={details['archived_tables_present']}; reconstructed fair-subset row-level details={len(fair)}/25. Historical memberwise fair-subset identity remains unresolved.",
        [rel(out / "fresh_fair_subset_rowlevel_summary.csv")] if summary else [],
        details,
    )


def run_two_tower(out_root):
    out = out_root / "two_tower"
    out.mkdir(parents=True, exist_ok=True)
    edir = EXP / "05_第二模型驗證"
    a30 = read_csv(edir / "圖表_figures_tables/tables/A30_second_model_two_tower_seed_summary.csv")
    a32 = read_csv(edir / "圖表_figures_tables/tables/A32_second_model_two_tower_mean_std.csv")
    metrics = [
        "cp_test_auc", "cp_test_fitb_acc", "or_test_fitb_acc",
        "recall@1", "recall@3", "recall@5", "recall@10", "recall@30", "recall@50",
        "mean_rank", "median_rank",
    ]
    recomputed = []
    for variant in ["original_text", "context_aware_description"]:
        sub = [r for r in a30 if r["variant"] == variant]
        row = {"variant": variant, "n_seeds": len(sub)}
        for metric in metrics:
            vals = [float(r[metric]) for r in sub]
            row[f"{metric}_mean"] = statistics.mean(vals)
            row[f"{metric}_std"] = statistics.stdev(vals)
        recomputed.append(row)
    write_csv(out / "recomputed_key_mean_std.csv", recomputed)

    archived = {r["variant"]: r for r in a32}
    matched = 0
    maxdiff = 0.0
    for row in recomputed:
        ref = archived[row["variant"]]
        for metric in metrics:
            for suffix in ["mean", "std"]:
                key = f"{metric}_{suffix}"
                if key in ref:
                    matched += 1
                    maxdiff = max(maxdiff, abs(float(row[key]) - float(ref[key])))

    import torch
    ckroot = MODEL / "second_model_two_tower/models"
    inventory = []
    errors = []
    for variant in ["original_text", "context_aware_description"]:
        for seed in range(1, 6):
            for role in ["cp_best_model.pt", "best_model.pt"]:
                path = ckroot / variant / f"seed_{seed}" / role
                rec = {"variant": variant, "seed": seed, "role": role, "path": rel(path), "exists": path.is_file()}
                if path.is_file():
                    rec["bytes"] = path.stat().st_size
                    rec["sha256"] = sha256(path)
                    try:
                        torch.load(path, map_location="cpu", weights_only=False)
                        rec["load_ok"] = True
                    except Exception as e:
                        rec["load_ok"] = False
                        rec["load_error"] = repr(e)
                        errors.append(rel(path))
                inventory.append(rec)
    (out / "checkpoint_inventory.json").write_text(json.dumps(inventory, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    all20 = len(inventory) == 20 and all(r["exists"] for r in inventory)
    load20 = all20 and not errors and all(r.get("load_ok") for r in inventory)
    exact = load20 and matched > 0 and maxdiff <= 1e-12
    return result(
        "two_tower",
        "exact" if exact else "partial",
        f"Recomputed A30 five-seed summary; A32 matched cells={matched}, max abs diff={maxdiff:.3g}; loadable checkpoints={sum(bool(r.get('load_ok')) for r in inventory)}/20.",
        [rel(out / "recomputed_key_mean_std.csv"), rel(out / "checkpoint_inventory.json")],
        {"matched_A32_cells": matched, "max_abs_difference": maxdiff, "all_20_checkpoints_load": load20},
    )


def main():
    ap = argparse.ArgumentParser(description="Integrated runner for the remaining thesis analyses.")
    ap.add_argument("--polyvore-root", required=True)
    ap.add_argument("--out-dir", default="reproduction/runs/remaining_thesis_reproduction")
    ap.add_argument("--judge-bootstrap", type=int, default=5000)
    ap.add_argument("--fair-subset-dir", default=None, help="Optional fresh 25-unit fair-subset run directory for case/subgroup summaries.")
    args = ap.parse_args()

    polyvore = Path(args.polyvore_root).expanduser().resolve()
    out = Path(args.out_dir)
    if not out.is_absolute():
        out = ROOT / out
    out.mkdir(parents=True, exist_ok=True)

    required = [
        polyvore / "polyvore_item_metadata.json",
        polyvore / "polyvore_outfit_titles.json",
        polyvore / "categories.csv",
        REPRO_ROOT / "splits/or_ids.csv",
    ]
    missing = [p for p in required if not p.is_file()]
    if missing:
        for p in missing:
            print("[MISSING]", p)
        raise SystemExit(2)

    print("=" * 88)
    print("REMAINING THESIS REPRODUCTION")
    print("=" * 88)

    modules = []
    runners = [
        ("length", lambda: run_length(out)),
        ("target_clue", lambda: run_target_clue(out, polyvore)),
        ("judge_preserved_outputs", lambda: run_judge(out, args.judge_bootstrap)),
        ("human_audit", lambda: run_human(out)),
        ("case_analysis", lambda: run_case(out, polyvore, args.fair_subset_dir)),
        ("two_tower", lambda: run_two_tower(out)),
    ]
    for name, fn in runners:
        print(f"\n[{name}]")
        try:
            r = fn()
        except Exception as e:
            r = result(name, "failed", repr(e))
        modules.append(r)
        print(r["status"], "-", r["evidence"])

    judge_row = next((r for r in modules if r["module"] == "judge_preserved_outputs"), None)
    if judge_row:
        jd = judge_row.get("details", {})
        sample_exact = bool(jd.get("historical_sample_regeneration_match"))
        inter = jd.get("historical_sample_regeneration_intersection")
        archived_only = jd.get("historical_sample_regeneration_archived_only")
        current_only = jd.get("historical_sample_regeneration_current_only")
        modules.append(result(
            "judge_robustness_sampling_provenance",
            "exact" if sample_exact else "partial",
            (
                "Historical robustness sample regeneration "
                + ("matches exactly." if sample_exact else
                   f"does not reproduce the archived 150 IDs: intersection={inter}/150, "
                   f"archived-only={archived_only}, current-only={current_only}. "
                   "All six archived robustness files use the same 150 IDs and their "
                   "original_score fields match the preserved formal outputs exactly; "
                   "the unresolved component is historical sample-generation provenance.")
            ),
        ))

    prompts = [
        REPRO_ROOT / "prompts/judge_qwen_p0.txt",
        REPRO_ROOT / "prompts/judge_gemma_p0.txt",
        REPRO_ROOT / "prompts/judge_p1_order.txt",
        REPRO_ROOT / "prompts/judge_p2_conservative.txt",
    ]
    modules.append(result(
        "judge_exact_generation",
        "exact" if all(p.is_file() for p in prompts) else "partial",
        "Exact Judge generation requires archived P0/P1/P2 prompt text; preserved-output recomputation is reported separately.",
    ))
    modules.append(result(
        "case_visuals",
        "exact" if (polyvore / "images").is_dir() else "partial",
        "Image-dependent qualitative figure regeneration requires Polyvore images/.",
    ))

    overall = "failed" if any(r["status"] == "failed" for r in modules) else (
        "partial" if any(r["status"] == "partial" for r in modules) else "exact"
    )

    payload = {
        "classification": "remaining thesis analyses integrated reproduction",
        "git_branch": git("branch", "--show-current"),
        "git_commit": git("rev-parse", "HEAD"),
        "dataset_root_name": polyvore.name,
        "overall": overall,
        "modules": modules,
        "important": [
            "Archived research files are read-only references and are not overwritten.",
            "Judge preserved-output recomputation is distinct from exact LLM generation reproduction.",
            "Human/Judge sampling seed 42 is distinct from model-training seeds 1-5.",
            "Fresh fair-subset case/subgroup evidence uses the reproducibly reconstructed fair subset because historical memberwise fair-subset identity is unresolved.",
        ],
    }
    jp = out / "remaining_reproduction_matrix.json"
    jp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    md = [
        "# Remaining thesis reproduction matrix",
        "",
        f"- Git commit: {payload['git_commit']}",
        f"- Dataset root identity: {payload['dataset_root_name']}",
        f"- Overall remaining-analysis status: **{overall}**",
        "",
        "| Module | Status | Evidence |",
        "|---|---|---|",
    ]
    for r in modules:
        ev = r["evidence"].replace("|", "\\|").replace("\n", " ")
        md.append(f"| {r['module']} | **{r['status']}** | {ev} |")
    md += [
        "",
        "The overall status is conservative: unresolved historical source/provenance components remain partial even when downstream numeric recomputation succeeds.",
        "",
    ]
    mp = out / "remaining_reproduction_matrix.md"
    mp.write_text("\n".join(md), encoding="utf-8")

    print("\n" + "=" * 88)
    print("SUMMARY")
    print("=" * 88)
    for r in modules:
        print(f"{r['module']:28s} {r['status']}")
    print("overall:", overall)
    print("[OK]", rel(jp))
    print("[OK]", rel(mp))


if __name__ == "__main__":
    main()