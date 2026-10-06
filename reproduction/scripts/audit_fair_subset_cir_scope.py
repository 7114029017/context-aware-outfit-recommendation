#!/usr/bin/env python3
"""Read-only reconstruction of the *denominator* of evaluate_cir.py's subset Recall@K.

No checkpoint, embeddings or CUDA required. Mirrors the preserved evaluator's
FG-pool construction (test first, train second, 3,000 unique items per FG,
including its existing train-loop `break`) and FITB query filtering. An
alternate `continue`-loop count is reported as sensitivity, never substituted.
A count match is an evaluation-scope gate, NOT historical memberwise ID proof.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
REPRO_ROOT = ROOT / "reproduction"
REPRO_SCRIPTS = REPRO_ROOT / "scripts"
EXPECTED_SPLITS = {"train": 16995, "valid": 3000, "test": 15145}
EXPECTED_SUBSET_SPLITS = {"train": 10225, "valid": 1748, "test": 9930}


def fail(msg: str) -> None:
    raise ValueError(msg)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_json(path: Path):
    if not path.is_file():
        fail(f"Input missing: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def build_fg2ims(outfits: list[dict], metadata: dict, keep: set[str]):
    # Same traversal order, insertion order, item duplication and FG key types
    # as UIUCPolyvoreRetrievalDataset.__init__ -> apply_subset_filter_inplace.
    fg2ims = {}
    im2fg = {}
    im_map = {}
    seen_outfits = set()
    for outfit in outfits:
        sid = str(outfit["set_id"])
        if sid in seen_outfits:
            fail(f"Duplicate split set_id: {sid}")
        seen_outfits.add(sid)
        for item in outfit["items"]:
            item_id = str(item["item_id"])
            if item_id not in metadata:
                fail(f"Metadata item missing: {item_id}")
            fg = metadata[item_id]["category_id"]
            if not isinstance(fg, (int, str)):
                fail(f"Invalid category_id for item {item_id}: {fg!r}")
            # Original dataset builds im2fg / id2im for the unfiltered split.
            im2fg[item_id] = fg
            image_key = f'{sid}_{int(item["index"])}'
            if image_key in im_map:
                fail(f"Duplicate indexed image key: {image_key}")
            im_map[image_key] = item_id
            if sid in keep:
                fg2ims.setdefault(fg, {}).setdefault(sid, []).append(item_id)
    return fg2ims, im2fg, im_map, seen_outfits


def pool_size_by_fg(test_fg: dict, train_fg: dict, *, historical_break: bool):
    """Recreate evaluate_cir.py lines 282–316 at item-ID level only."""
    pool = {}
    for fg, sid2items in test_fg.items():
        unique = set()
        for items in sid2items.values():
            for iid in items:
                if iid not in unique:
                    unique.add(iid)
                if len(unique) >= 3000:
                    break
            if len(unique) >= 3000:
                break
        pool[fg] = unique
    # Preserved source has 'break', not 'continue', if train FG not in test FG.
    # The distinction is recorded in the audit, not silently repaired.
    break_fg = None
    for fg, sid2items in train_fg.items():
        if fg not in pool:
            if historical_break:
                break_fg = str(fg)
                break
            continue
        unique = pool[fg]
        for items in sid2items.values():
            for iid in items:
                if iid not in unique:
                    unique.add(iid)
                if len(unique) >= 3000:
                    break
            if len(unique) >= 3000:
                break
    kept = {fg: unique for fg, unique in pool.items() if len(unique) >= 3000}
    return kept, break_fg, {str(fg): len(unique) for fg, unique in pool.items()}


def question_scope(questions: list, keep_ids: set[str], im_map: dict, im2fg: dict, eligible_fg: set,
                   pairs: list | None = None):
    """Count the evaluable CIR questions; append (set_id, target_item_id, target_item_fg) to pairs."""
    total = 0
    subset = 0
    evaluable = 0
    no_fg = Counter()
    included = Counter()
    subset_ids_seen = set()
    missing = []
    for i, question in enumerate(questions):
        total += 1
        if not isinstance(question, dict) or not question.get("question") or not question.get("answers"):
            fail(f"FITB question malformed at {i}")
        query_id = str(question["question"][0]).split("_")[0]
        # Mirrors load_fitb_questions / parse_iminfo: GT set_id = first question image's set_id.
        if query_id not in keep_ids:
            continue
        subset += 1
        subset_ids_seen.add(query_id)
        valid_answers = []
        for raw in question["answers"]:
            key = str(raw)
            if key not in im_map:
                missing.append({"query": query_id, "answer": key, "why": "not found in test id2im"})
                continue
            if key.split("_")[0] == query_id:
                valid_answers.append(im_map[key])
        if len(valid_answers) != 1:
            missing.append({"query": query_id, "matched_gt_answers": len(valid_answers)})
            continue
        target = valid_answers[0]
        if target not in im2fg:
            fail(f"Test im2fg missing GT target item {target}")
        fg = im2fg[target]
        if fg not in eligible_fg:
            no_fg[str(fg)] += 1
            continue
        evaluable += 1
        included[str(fg)] += 1
        if pairs is not None:
            pairs.append((query_id, str(target), str(fg)))
    return {
        "fitb_questions_total": total,
        "fitb_questions_in_subset": subset,
        "distinct_subset_fitb_ids": len(subset_ids_seen),
        "evaluable_questions": evaluable,
        "excluded_due_to_fg_pool_below_3000": subset - evaluable - len(missing),
        "included_questions_by_fg": dict(sorted(included.items())),
        "excluded_questions_by_fg": dict(sorted(no_fg.items())),
        "question_parse_anomalies": missing[:20],
        "question_parse_anomaly_count": len(missing),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--polyvore-root", required=True)
    ap.add_argument("--subset-ids", default="reproduction/splits/fair_subset/fair_subset_ids.txt")
    ap.add_argument("--subset-manifest", default="reproduction/splits/fair_subset/fair_subset_reconstruction_manifest.json")
    ap.add_argument("--out", default="reproduction/splits/fair_subset/fair_subset_cir_scope_audit.json")
    ap.add_argument("--out-query-ids", default="reproduction/splits/fair_subset/or_query_ids.csv",
                    help="Evaluable CIR (OR) queries, sorted as in the runs' evaluation-membership SHA-256.")
    ap.add_argument("--expect-count", type=int, default=3432)
    args = ap.parse_args()

    def resolve(s):
        p = Path(s).expanduser()
        return (p if p.is_absolute() else ROOT / p).resolve()

    polyvore = resolve(args.polyvore_root)
    ids_path = resolve(args.subset_ids)
    manifest_path = resolve(args.subset_manifest)
    out = resolve(args.out)
    query_ids_out = resolve(args.out_query_ids)
    if not ids_path.is_file() or not manifest_path.is_file():
        fail("Reconstructed subset candidate ID and manifest are required.")
    subset_manifest = load_json(manifest_path)
    if subset_manifest.get("gates_passed") is not True or subset_manifest.get("candidate_unique_ids") != 21903:
        fail("Candidate reconstruction provenance/count gate is not passed.")
    inputs = subset_manifest.get("source_files")
    if not isinstance(inputs, list) or not inputs or any(
        not Path(x["path"]).is_file() or sha256(Path(x["path"])) != x["sha256"]
        for x in inputs
    ):
        fail("Candidate reconstruction input SHA changed or manifest lacks input identities.")
    ids = ids_path.read_text(encoding="utf-8").splitlines()
    keep = set(ids)
    if len(ids) != len(keep) or len(ids) != 21903:
        fail("Expected 21,903 unique candidate IDs; subset list differs.")

    split_paths = {s: polyvore / "disjoint" / f"{s}.json" for s in EXPECTED_SPLITS}
    outfits = {s: load_json(p) for s, p in split_paths.items()}
    split_ids = {}
    coverage = {}
    for split, rows in outfits.items():
        if not isinstance(rows, list) or len(rows) != EXPECTED_SPLITS[split]:
            fail(f"Unexpected disjoint/{split}.json cardinality")
        sid = [str(row["set_id"]) for row in rows]
        if len(sid) != len(set(sid)):
            fail(f"Duplicate set_id in {split}")
        split_ids[split] = set(sid)
        coverage[split] = len(split_ids[split] & keep)
        if coverage[split] != EXPECTED_SUBSET_SPLITS[split]:
            fail(f"Unexpected {split} subset overlap: {coverage[split]}")
    if set.union(*split_ids.values()) != set.union(*split_ids.values(), keep):
        fail("Subset includes IDs outside PO-D splits.")
    if sum(map(len, split_ids.values())) != len(set.union(*split_ids.values())):
        fail("Disjoint splits overlap.")
    if sum(coverage.values()) != len(keep):
        fail("Subset membership does not partition into train/valid/test.")

    metadata_path = polyvore / "polyvore_item_metadata.json"
    metadata = load_json(metadata_path)
    if not isinstance(metadata, dict):
        fail("PO-D metadata must be mapping keyed by item_id")
    test_fg, test_im2fg, test_im_map, _ = build_fg2ims(outfits["test"], metadata, keep)
    train_fg, _, _, _ = build_fg2ims(outfits["train"], metadata, keep)
    questions_path = polyvore / "disjoint" / "fill_in_blank_test.json"
    questions = load_json(questions_path)
    if not isinstance(questions, list):
        fail("FITB test questions must be a list")

    historical_pools, stop_fg, historical_raw = pool_size_by_fg(test_fg, train_fg, historical_break=True)
    corrected_pools, _, corrected_raw = pool_size_by_fg(test_fg, train_fg, historical_break=False)
    pairs: list[tuple[str, str, str]] = []
    historical = question_scope(questions, keep, test_im_map, test_im2fg, set(historical_pools), pairs)
    if len(set(pairs)) != len(pairs):
        fail("Duplicate evaluable CIR question")
    pairs.sort()
    # Same canonical form as summarize_fair_subset_5seed.query_membership_sha.
    membership_sha = hashlib.sha256("\n".join("\t".join(p) for p in pairs).encode("utf-8")).hexdigest()
    sensitivity = question_scope(questions, keep, test_im_map, test_im2fg, set(corrected_pools))
    match = historical["evaluable_questions"] == args.expect_count
    source_consistent = (
        historical["question_parse_anomaly_count"] == 0
        and historical["fitb_questions_in_subset"] == coverage["test"]
        and historical["distinct_subset_fitb_ids"] == coverage["test"]
        and historical["fitb_questions_total"] == EXPECTED_SPLITS["test"]
    )
    passed = bool(match and source_consistent)

    report = {
        "classification": "fresh scope-only audit of reconstructed WOS fair-subset CIR Recall@K denominator; no checkpoint or inference",
        "source_semantics": "UIUCPolyvoreRetrievalDataset fg2ims creation -> evaluate_cir.py filter_fg2ims=True -> test unique pool then train unique pool with preserved break -> 3000-item FG threshold -> FITB filtered by set_id -> category eligible",
        "historical_membership_confirmed": False,
        "candidate_sha256": sha256(ids_path),
        "candidate_manifest_sha256": sha256(manifest_path),
        "subset_coverage": coverage,
        "expected_cir_evaluable_pairs_archived": args.expect_count,
        "observed_preserved_evaluator_semantics": historical,
        "evaluable_query_ids_file": query_ids_out.name,
        "evaluable_query_membership_sha256": membership_sha,
        "preserved_pool_categories_selected": len(historical_pools),
        "preserved_pool_category_sizes": {str(k): len(v) for k,v in historical_pools.items()},
        "preserved_train_loop_first_missing_test_fg_break": stop_fg,
        "preserved_unfiltered_pool_sizes": historical_raw,
        "sensitivity_train_loop_continue_instead_of_break": {
            "not_the_preserved_evaluator": True,
            "scope": sensitivity,
            "selected_categories": len(corrected_pools),
            "category_sizes": {str(k): len(v) for k,v in corrected_pools.items()},
            "unfiltered_pool_sizes": corrected_raw,
        },
        "count_matches_archived": match,
        "question_source_consistent": source_consistent,
        "gates_passed": passed,
        "inputs": [
            {"path": str(p), "bytes": p.stat().st_size, "sha256": sha256(p)}
            for p in [ids_path, manifest_path, *split_paths.values(), metadata_path, questions_path]
        ],
        "limits": [
            "Matches evaluation denominator only, not numerical Recall@K values or historical ID file.",
            "Does not verify actual runtime dataset loading/embeddings or the historical evaluator was identical.",
            "If the evaluable count differs from 3432, block training pending source/ID diagnosis."
        ],
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    query_ids_out.parent.mkdir(parents=True, exist_ok=True)
    for path in (out, query_ids_out):
        if path.exists():
            fail(f"Refusing to overwrite prior denominator audit output: {path}; supply a new path.")
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    query_ids_out.write_text(
        "set_id,target_item_id,target_item_fg\n" + "".join(",".join(p) + "\n" for p in pairs),
        encoding="utf-8",
    )
    print(json.dumps({
        "candidate_total": len(keep), "test_overlap": coverage["test"],
        "fitb_subset_questions": historical["fitb_questions_in_subset"],
        "evaluable_preserved_semantics": historical["evaluable_questions"],
        "expected": args.expect_count, "selected_categories": len(historical_pools),
        "corrected_loop_sensitivity_evaluable": sensitivity["evaluable_questions"],
        "evaluable_query_membership_sha256": membership_sha,
        "gates_passed": passed, "report": str(out), "query_ids": str(query_ids_out),
    }, ensure_ascii=False, indent=2), flush=True)
    if not passed:
        raise SystemExit(3)


if __name__ == "__main__":
    main()
