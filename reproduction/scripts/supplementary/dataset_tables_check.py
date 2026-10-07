#!/usr/bin/env python3
"""Rebuild the 2025 dataset tables (T00, T01, A01, A02, A12, A13) and compare them with the archived files.

The 2025 tables describe the Polyvore Outfits disjoint split, the original and generated text fields and
the evaluation scope. They were written by the notebook
01_資料建構_data_construction/01_使用資料與資料範圍_dataset_scope/source_programs/P12_dataset_scope_reproducible.ipynb
(T00_dataset_overview, T01_text_field_comparison, A01, A02) and by
03_實驗與結果_experiments_results/00_控制檢查與附加稽核/source_programs/build_journal_artifacts.py (A12, A13); the
program of T00_dataset_characterization is not preserved and its rows are rebuilt from the same statistics.
The code below is copied from these programs. Two inputs differ in kind from 2025: the CIR scope is
computed with the rule of the archived evaluator (the 2025 per-query files of the main runs are not
preserved), and the fair subset is the reconstruction used by the official run.

Writes under --out-dir (default reproduction/results/supplementary/dataset_tables_check/) one file per table with
the archived file name, comparison.csv (every value against the archived table) and summary.md.
"""
from __future__ import annotations

import argparse
import csv
import re
from collections import Counter
from pathlib import Path
from statistics import mean, median

from _common import D01, GENERATED, REPO, REPRO, SUPPLEMENTARY, WOS_JSONL, polyvore_root, read_csv, read_json, write_text
from input_data_audit import POOL_SIZE, evaluator_pools

SCOPE = D01 / "01_使用資料與資料範圍_dataset_scope"
WOS_DIR = WOS_JSONL.parent
ARCHIVED = {
    "T00_dataset_overview_po_d_main.csv": SCOPE / "圖表_figures_tables" / "tables" / "T00_dataset_overview_po_d_main.csv",
    "T00_dataset_characterization_po_d_main.csv": SCOPE / "圖表_figures_tables" / "tables" / "T00_dataset_characterization_po_d_main.csv",
    "T01_text_field_comparison_po_d_main.csv": SCOPE / "圖表_figures_tables" / "tables" / "T01_text_field_comparison_po_d_main.csv",
    "A01_title_length_audit.csv": SCOPE / "A01_title_length_audit.csv",
    "A02_generation_coverage_audit.csv": SCOPE / "A02_generation_coverage_audit.csv",
    "A12_po_d_split_item_outfit_stats.csv": WOS_DIR / "A12_po_d_split_item_outfit_stats.csv",
    "A13_po_d_semantic_category_distribution.csv": WOS_DIR / "A13_po_d_semantic_category_distribution.csv",
}
NOT_COMPARED = {"source", "notes"}  # 2025 file paths and free-text notes
FAIR_SUBSET = REPRO / "splits" / "fair_subset"

# ---------------------------------------------------------------- P12 dataset scope, cell 2 (copied)
TOKEN_RE = re.compile(r"[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)?")
WORD_RE = re.compile(r"[A-Za-z]+(?:[-'][A-Za-z]+)?")
# build_journal_artifacts.py
TEMP_PREFIX_RE = re.compile(r"^\s*[+-]?\d+(?:\.\d+)?\s*(?:°\s*C|°C|℃|C)\s*[,，:：-]*\s*", re.IGNORECASE)


def fmt_pct(num, den) -> str:
    return f"{100 * float(num) / float(den):.2f}" if den else "0.00"


def query_token_count(text: str) -> int:
    return len(TOKEN_RE.findall(str(text or "").strip()))


def word_tokens(text: str) -> list[str]:
    return [t.lower() for t in WORD_RE.findall(str(text or "").strip())]


def word_count(text: str) -> int:
    return len(word_tokens(text))


def original_field(original_data: dict, set_id: str, field: str) -> str:
    value = original_data.get(str(set_id), "")
    if isinstance(value, dict):
        return str(value.get(field, "") or "").strip()
    return str(value or "").strip()


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


def text_stats(texts: list[str]) -> dict:
    """P12 dataset scope, cell 4."""
    wc = [word_count(t) for t in texts]
    qc = [query_token_count(t) for t in texts]
    all_words = []
    for t in texts:
        all_words.extend(word_tokens(t))
    vocab = Counter(all_words)
    n = len(texts)
    empty = sum(1 for x in wc if x == 0)
    return {
        "n": n, "empty": empty, "pct_empty": fmt_pct(empty, n),
        "mean_words": mean(wc) if wc else 0, "median_words": median(wc) if wc else 0, "max_words": max(wc) if wc else 0,
        "mean_query_tokens": mean(qc) if qc else 0, "median_query_tokens": median(qc) if qc else 0,
        "max_query_tokens": max(qc) if qc else 0,
        "pct_le_3_words": fmt_pct(sum(1 for x in wc if x <= 3), n), "pct_le_5_words": fmt_pct(sum(1 for x in wc if x <= 5), n),
        "total_word_tokens": len(all_words), "unique_word_tokens": len(vocab),
        "type_token_ratio": (len(vocab) / len(all_words)) if all_words else 0,
    }


def write_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def cir_scope(poly: Path, meta: dict, split_records: dict) -> tuple[list[tuple[str, str]], set[str]]:
    """The query-target pairs that the archived evaluate_cir.py evaluates, and their categories."""
    fg2ims, id2im = {}, {}
    for split in ("train", "test"):
        fg2ims[split] = {}
        for outfit in split_records[split]:
            for item in outfit["items"]:
                fg = str(meta[item["item_id"]]["category_id"])
                fg2ims[split].setdefault(fg, {}).setdefault(outfit["set_id"], []).append(item["item_id"])
                if split == "test":
                    id2im[f"{outfit['set_id']}_{item['index']}"] = item["item_id"]
    pools, _ = evaluator_pools(fg2ims["test"], fg2ims["train"])
    evaluated = {k for k, n in pools.items() if n >= POOL_SIZE}
    pairs = []
    for q in read_json(poly / "disjoint" / "fill_in_blank_test.json"):
        gt = q["question"][0].split("_")[0]
        target = next(id2im[a] for a in q["answers"] if a.split("_")[0] == gt)
        if str(meta[target]["category_id"]) in evaluated:
            pairs.append((gt, target))
    return pairs, {str(meta[t]["category_id"]) for _, t in pairs}


def compare(name: str, rows: list[dict]) -> list[list]:
    archived = read_csv(ARCHIVED[name])
    out = []
    for i, (mine, old) in enumerate(zip(rows, archived), 1):
        for column, value in old.items():
            if column in NOT_COMPARED:
                continue
            now = str(mine.get(column, ""))
            out.append([name, i, column, now, value, "yes" if now == value else "NO"])
    if len(rows) != len(archived):
        out.append([name, "", "number of rows", len(rows), len(archived), "NO"])
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out-dir", type=Path, default=SUPPLEMENTARY / "dataset_tables_check")
    parser.add_argument("--polyvore-root", default=None)
    args = parser.parse_args()
    out = args.out_dir
    poly = polyvore_root(args.polyvore_root)
    split_dir = poly / "disjoint"

    records = {s: read_json(split_dir / f"{s}.json") for s in ("train", "valid", "test")}
    fitb_test_rows = read_json(split_dir / "fill_in_blank_test.json")
    compat_test_rows = [line.strip().split() for line in (split_dir / "compatibility_test.txt").open(encoding="utf-8")
                        if line.strip()]
    original_titles = read_json(poly / "polyvore_outfit_titles.json")
    generated_titles = read_json(GENERATED)
    meta = read_json(poly / "polyvore_item_metadata.json")

    train_n, valid_n, test_n = (len(records[s]) for s in ("train", "valid", "test"))
    all_records = records["train"] + records["valid"] + records["test"]
    all_split_ids = {str(r["set_id"]) for r in all_records}
    generated_ids = {str(k) for k in generated_titles}
    pairs, cir_fgs = cir_scope(poly, meta, records)
    cir_pairs = set(pairs)
    cir_rows_per_seed = [len(pairs)] * 5  # every seed evaluates the same queries
    subset_ids = {line.strip() for line in (FAIR_SUBSET / "fair_subset_ids.txt").read_text(encoding="utf-8").splitlines()
                  if line.strip()}
    test_ids = {str(r["set_id"]) for r in records["test"]}
    none_test_overlap = subset_ids & test_ids
    none_cir_rows = [len(read_csv(FAIR_SUBSET / "or_query_ids.csv"))] * 5
    outfit_lengths = [len(r.get("items", [])) for r in all_records]
    item_ids = [str(item["item_id"]) for r in all_records for item in r.get("items", []) if item.get("item_id")]

    ordered_ids = sorted(generated_ids, key=lambda x: int(x) if x.isdigit() else x)
    stat_original_title = text_stats([original_field(original_titles, sid, "title") for sid in ordered_ids])
    stat_original_url_title = text_stats([original_url_name_plus_title(original_titles, sid) for sid in ordered_ids])
    generated_texts = [generated_title_only(generated_titles, sid) for sid in ordered_ids]
    stat_generated = text_stats(generated_texts)
    stat_generated_no_temp = text_stats([TEMP_PREFIX_RE.sub("", t).strip() for t in generated_texts])

    # ---------------------------------------------------------------- P12 dataset scope, cell 4 (copied)
    cir_display = f"{cir_rows_per_seed[0]:,}" if len(set(cir_rows_per_seed)) == 1 else ", ".join(f"{x:,}" for x in cir_rows_per_seed)
    none_display = f"{none_cir_rows[0]:,}" if none_cir_rows and len(set(none_cir_rows)) == 1 else ", ".join(f"{x:,}" for x in none_cir_rows)
    main_total = train_n + valid_n + test_n
    cir_skipped = len(fitb_test_rows) - len(cir_pairs)
    t00 = [
        {"Category": "Dataset", "Statistic": "Source", "Value": "Polyvore Outfits Disjoint version (PO-D)"},
        {"Category": "Dataset split", "Statistic": "Train / validation / test / total outfits", "Value": f"{train_n:,} / {valid_n:,} / {test_n:,} / {main_total:,}"},
        {"Category": "Dataset split", "Statistic": "Generated-valid descriptions aligned to PO-D", "Value": f"{len(generated_ids):,} / {main_total:,} ({fmt_pct(len(generated_ids & all_split_ids), main_total)}%)"},
        {"Category": "Reference metadata", "Statistic": "Original Polyvore title metadata records", "Value": f"{len(original_titles):,} (metadata pool only; not main experimental N)"},
        {"Category": "CP evaluation", "Statistic": "Compatibility test pairs", "Value": f"{len(compat_test_rows):,}"},
        {"Category": "CP evaluation", "Statistic": "FITB test questions", "Value": f"{len(fitb_test_rows):,}"},
        {"Category": "CIR evaluation", "Statistic": "Initial test questions before CIR eligibility filtering", "Value": f"{len(fitb_test_rows):,}"},
        {"Category": "CIR evaluation", "Statistic": "Evaluable query-target pairs per seed", "Value": f"{cir_display} ({len(cir_pairs):,} unique pairs)"},
        {"Category": "CIR evaluation", "Statistic": "Excluded from CIR recall computation", "Value": f"{cir_skipped:,} test questions"},
        {"Category": "CIR evaluation", "Statistic": "Reason for 9,311 rather than 15,145", "Value": "CIR keeps only target fine-grained categories with an eligible same-category distractor pool in the original evaluation; skipped cases are not CP cases."},
        {"Category": "CIR evaluation", "Statistic": "Target fine-grained categories represented in CIR detail", "Value": f"{len(cir_fgs):,}"},
        {"Category": "CIR evaluation", "Statistic": "Seed-level paired CIR records across 5 seeds", "Value": f"{sum(cir_rows_per_seed):,}"},
        {"Category": "Supplementary no-missing ablation", "Statistic": "no_missing_none_ids total / test overlap", "Value": f"{len(subset_ids):,} / {len(none_test_overlap):,}"},
        {"Category": "Supplementary no-missing ablation", "Statistic": "CIR evaluable pairs per seed under experiments_none_only", "Value": f"{none_display or 'not found'} (not the main experiment scope)"},
        {"Category": "Outfit composition", "Statistic": "Item occurrences / unique items", "Value": f"{len(item_ids):,} / {len(set(item_ids)):,}"},
        {"Category": "Outfit composition", "Statistic": "Mean / median outfit length", "Value": f"{mean(outfit_lengths):.2f} / {median(outfit_lengths):.2f} items"},
    ]

    def t01_row(label, s, note):
        return {
            "Text field": label,
            "N records": f"{s['n']:,}",
            "Empty records": f"{s['empty']:,} ({s['pct_empty']}%)",
            "Mean / median / max word count": f"{s['mean_words']:.2f} / {s['median_words']:.2f} / {s['max_words']} words",
            "Mean query-token count": f"{s['mean_query_tokens']:.2f}",
            "Texts <= 3 words": f"{s['pct_le_3_words']}%",
            "Unique word tokens": f"{s['unique_word_tokens']:,}",
            "Type-token ratio": f"{s['type_token_ratio']:.4f}",
            "Note": note,
        }

    t01 = [
        t01_row("Original title field", stat_original_title, "Dataset characterization only; original retrieval baseline uses url_name + title."),
        t01_row("Original url_name + title query (outfitUrlTitle)", stat_original_url_title, "Original-paper text embedding context used for fair baseline comparison."),
        t01_row("Generated context-aware description (title only)", stat_generated, "Proposed/generated query; original url_name is not appended."),
    ]
    a01 = []
    for label, s in [("Original title field matched to PO-D ids", stat_original_title),
                     ("Original url_name + title query matched to PO-D ids", stat_original_url_title),
                     ("Generated context-aware title-only query", stat_generated)]:
        a01.append({
            "dataset": label, "n_records": str(s["n"]), "empty_records": str(s["empty"]), "pct_empty": s["pct_empty"],
            "mean_word_count": f"{s['mean_words']:.4f}", "median_word_count": f"{s['median_words']:.4f}",
            "max_word_count": str(s["max_words"]), "mean_query_token_count": f"{s['mean_query_tokens']:.4f}",
            "median_query_token_count": f"{s['median_query_tokens']:.4f}", "max_query_token_count": str(s["max_query_tokens"]),
            "pct_word_len_le_3": s["pct_le_3_words"], "pct_word_len_le_5": s["pct_le_5_words"],
            "total_word_tokens": str(s["total_word_tokens"]), "unique_word_tokens": str(s["unique_word_tokens"]),
            "type_token_ratio": f"{s['type_token_ratio']:.6f}",
        })
    rel = lambda p: str(Path(p).relative_to(REPO)) if str(p).startswith(str(REPO)) else str(p)
    a02 = [
        {"metric": "po_d_disjoint_train_records", "value": str(train_n), "source": "Polyvore disjoint/train.json", "notes": "Main PO-D split."},
        {"metric": "po_d_disjoint_valid_records", "value": str(valid_n), "source": "Polyvore disjoint/valid.json", "notes": "Main PO-D split."},
        {"metric": "po_d_disjoint_test_records", "value": str(test_n), "source": "Polyvore disjoint/test.json", "notes": "Main PO-D split."},
        {"metric": "po_d_disjoint_total_records", "value": str(main_total), "source": "disjoint train + valid + test", "notes": "Main dataset size."},
        {"metric": "generated_valid_title_records", "value": str(len(generated_titles)), "source": rel(GENERATED), "notes": "Aligned 1-to-1 with PO-D set_ids."},
        {"metric": "original_polyvore_title_metadata_records", "value": str(len(original_titles)), "source": "Polyvore polyvore_outfit_titles.json", "notes": "Reference metadata pool only."},
        {"metric": "cp_compatibility_test_pairs", "value": str(len(compat_test_rows)), "source": "Polyvore disjoint/compatibility_test.txt", "notes": "CP AUC evaluation scope."},
        {"metric": "cp_fitb_test_questions", "value": str(len(fitb_test_rows)), "source": "Polyvore disjoint/fill_in_blank_test.json", "notes": "CP FITB evaluation scope."},
        {"metric": "cir_evaluable_query_target_pairs_per_seed", "value": str(len(cir_pairs)), "source": "rule of the archived evaluate_cir.py", "notes": "CIR recall scope after candidate-pool eligibility filtering."},
        {"metric": "cir_test_questions_excluded_by_cir_eligibility", "value": str(cir_skipped), "source": "rule of the archived evaluate_cir.py", "notes": "Skipped by CIR candidate-pool eligibility; not CP reassignment."},
        {"metric": "no_missing_none_ids_total", "value": str(len(subset_ids)), "source": rel(FAIR_SUBSET / "fair_subset_ids.txt"), "notes": "Supplementary no-missing ablation/control only."},
    ]

    # ---------------------------------------------------------------- T00 characterization (program not preserved)
    def words_line(s):
        return f"{s['mean_words']:.2f} / {s['median_words']:.2f} / {s['max_words']} tokens"

    t00c = [
        {"Category": "Dataset", "Statistic": "Source", "Value": "Polyvore Outfits Disjoint version (PO-D)"},
        {"Category": "Dataset split", "Statistic": "Train / validation / test outfits", "Value": f"{train_n:,} / {valid_n:,} / {test_n:,}"},
        {"Category": "Main experiment", "Statistic": "Total outfits", "Value": f"{main_total:,}"},
        {"Category": "Outfit composition", "Statistic": "Item occurrences", "Value": f"{len(item_ids):,}"},
        {"Category": "Outfit composition", "Statistic": "Unique items", "Value": f"{len(set(item_ids)):,}"},
        {"Category": "Outfit composition", "Statistic": "Mean / median outfit length", "Value": f"{mean(outfit_lengths):.2f} / {median(outfit_lengths):.2f} items"},
        {"Category": "Original title field", "Statistic": "Empty titles", "Value": f"{stat_original_title['empty']} ({stat_original_title['pct_empty']}%)"},
        {"Category": "Original title field", "Statistic": "Mean / median / max length", "Value": words_line(stat_original_title)},
        {"Category": "Original title field", "Statistic": "Titles ≤ 3 tokens", "Value": f"{stat_original_title['pct_le_3_words']}%"},
        {"Category": "Generated description", "Statistic": "Mean / median / max length", "Value": words_line(stat_generated)},
        {"Category": "Generated description", "Statistic": "Mean / median / max length without temperature prefix", "Value": words_line(stat_generated_no_temp)},
    ]

    # ---------------------------------------------------------------- build_journal_artifacts.py: A12, A13 (copied)
    split_rows = []
    all_split_records = []
    for split in ["train", "valid", "test"]:
        recs = records[split]
        all_split_records.extend((split, r) for r in recs)
        lengths = [len(r.get("items", [])) for r in recs]
        ids = [str(it.get("item_id", "")) for r in recs for it in r.get("items", []) if it.get("item_id")]
        split_rows.append({
            "split": split, "n_outfits": str(len(recs)), "n_item_occurrences": str(len(ids)),
            "n_unique_items": str(len(set(ids))),
            "mean_outfit_length": f"{mean([float(x) for x in lengths]):.4f}" if lengths else "0.0000",
            "median_outfit_length": f"{median(lengths):.4f}" if lengths else "0.0000",
            "min_outfit_length": str(min(lengths) if lengths else 0), "max_outfit_length": str(max(lengths) if lengths else 0),
        })
    all_lengths = [len(r.get("items", [])) for _, r in all_split_records]
    all_item_ids = [str(it.get("item_id", "")) for _, r in all_split_records for it in r.get("items", []) if it.get("item_id")]
    split_rows.append({
        "split": "all_po_d", "n_outfits": str(len(all_split_records)), "n_item_occurrences": str(len(all_item_ids)),
        "n_unique_items": str(len(set(all_item_ids))),
        "mean_outfit_length": f"{mean([float(x) for x in all_lengths]):.4f}" if all_lengths else "0.0000",
        "median_outfit_length": f"{median(all_lengths):.4f}" if all_lengths else "0.0000",
        "min_outfit_length": str(min(all_lengths) if all_lengths else 0), "max_outfit_length": str(max(all_lengths) if all_lengths else 0),
    })
    category_rows = []
    for split_name in ["train", "valid", "test", "all_po_d"]:
        recs = [r for _, r in all_split_records] if split_name == "all_po_d" else records[split_name]
        occ_counter: Counter[str] = Counter()
        unique_by_cat: dict[str, set[str]] = {}
        total_occ = 0
        all_unique: set[str] = set()
        for r in recs:
            for it in r.get("items", []):
                iid = str(it.get("item_id", ""))
                if not iid:
                    continue
                cat = str(meta.get(iid, {}).get("semantic_category") or "unknown")
                occ_counter[cat] += 1
                unique_by_cat.setdefault(cat, set()).add(iid)
                all_unique.add(iid)
                total_occ += 1
        for cat, occ in sorted(occ_counter.items(), key=lambda kv: (-kv[1], kv[0])):
            u = len(unique_by_cat.get(cat, set()))
            category_rows.append({
                "split": split_name, "semantic_category": cat, "item_occurrences": str(occ),
                "pct_item_occurrences": f"{100 * occ / total_occ:.2f}" if total_occ else "0.00",
                "unique_items": str(u), "pct_unique_items": f"{100 * u / len(all_unique):.2f}" if all_unique else "0.00",
            })

    tables = {"T00_dataset_overview_po_d_main.csv": t00, "T00_dataset_characterization_po_d_main.csv": t00c,
              "T01_text_field_comparison_po_d_main.csv": t01, "A01_title_length_audit.csv": a01,
              "A02_generation_coverage_audit.csv": a02, "A12_po_d_split_item_outfit_stats.csv": split_rows,
              "A13_po_d_semantic_category_distribution.csv": category_rows}
    comparison = []
    for name, rows in tables.items():
        write_rows(out / name, rows)
        comparison += compare(name, rows)
    with (out / "comparison.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow(["table", "row", "column", "recomputed", "archived", "equal"])
        writer.writerows(comparison)
    differing = [c for c in comparison if c[5] != "yes"]
    lines = [
        "# Dataset tables (2025 T00, T01, A01, A02, A12, A13)",
        "",
        "Computed by `reproduction/scripts/supplementary/dataset_tables_check.py` from the Polyvore metadata, the",
        "generated descriptions and the fair subset, with the code of the 2025 programs; the CIR scope uses the",
        "rule of the archived evaluator. Every value is compared with the archived table (`comparison.csv`; the",
        "source paths and notes are not compared).",
        "",
        "| Table | Rows | Values compared | Equal |",
        "|---|---:|---:|---:|",
        *[f"| {name} | {len(rows)} | {sum(1 for c in comparison if c[0] == name)} | "
          f"{sum(1 for c in comparison if c[0] == name and c[5] == 'yes')} |" for name, rows in tables.items()],
        "",
        (f"All {len(comparison)} values equal the archived tables." if not differing else
         f"{len(differing)} of {len(comparison)} values differ: "
         + "; ".join(f"{c[0]} row {c[1]} {c[2]}: {c[3]} vs {c[4]}" for c in differing[:10])),
    ]
    write_text(out / "summary.md", lines)
    print(f"[SUPPLEMENTARY] dataset tables written to {out}")


if __name__ == "__main__":
    main()
