#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
from collections import OrderedDict
import argparse
import csv
import hashlib
import json
import sys

EXPECTED = {
    "train": 16995,
    "valid": 3000,
    "test": 15145,
    "compatibility_test": 30290,
    "fitb_test": 15145,
    "or_test": 9311,
}

def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def read_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)

def write_csv(path: Path, fieldnames, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n")
        w.writeheader()
        w.writerows(rows)

def split_rows(records):
    return [{"source_index": i, "set_id": str(rec["set_id"])} for i, rec in enumerate(records)]

def build_id2im(outfits):
    out = {}
    for outfit in outfits:
        sid = str(outfit["set_id"])
        for item in outfit["items"]:
            out[f"{sid}_{item['index']}"] = str(item["item_id"])
    return out

def build_fg2ims(outfits, metadata):
    fg2ims = OrderedDict()
    im2fg = {}
    for outfit in outfits:
        sid = str(outfit["set_id"])
        for item in outfit["items"]:
            iid = str(item["item_id"])
            fg = str(metadata[iid]["category_id"])
            im2fg[iid] = fg
            fg2ims.setdefault(fg, OrderedDict()).setdefault(sid, []).append(iid)
    return fg2ims, im2fg

def build_candidate_pools(test_fg2ims, train_fg2ims):
    # Replicates evaluate_cir.py insertion/order logic exactly.
    pools = OrderedDict()
    pool_sets = {}
    for fg, sid2items in test_fg2ims.items():
        pools[fg] = []
        pool_sets[fg] = set()
        stop = False
        for _, items in sid2items.items():
            for iid in items:
                if iid not in pool_sets[fg]:
                    pool_sets[fg].add(iid)
                    pools[fg].append(iid)
                if len(pools[fg]) >= 3000:
                    stop = True
                    break
            if stop:
                break

    for fg, sid2items in train_fg2ims.items():
        # Preserve the archived evaluator's early break behavior.
        if fg not in pools:
            break
        stop = False
        for _, items in sid2items.items():
            for iid in items:
                if iid not in pool_sets[fg]:
                    pool_sets[fg].add(iid)
                    pools[fg].append(iid)
                if len(pools[fg]) >= 3000:
                    stop = True
                    break
            if stop:
                break

    return OrderedDict((fg, ids) for fg, ids in pools.items() if len(ids) >= 3000)

def parse_compatibility(path: Path):
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            parts = line.strip().split()
            if not parts:
                continue
            label = int(parts[0])
            refs = parts[1:]
            gt = refs[0].split("_")[0] if refs else ""
            rows.append({"source_index": i, "label": label, "set_id": gt, "item_refs": " ".join(refs)})
    return rows

def parse_fitb(path: Path, id2im):
    data = read_json(path)
    rows = []
    for i, item in enumerate(data):
        q = [str(x) for x in item["question"]]
        a = [str(x) for x in item["answers"]]
        gt = q[0].split("_")[0] if q else ""
        correct_positions = [j for j, ref in enumerate(a) if ref.split("_")[0] == gt]
        correct_pos = correct_positions[0] if len(correct_positions) == 1 else ""
        target_ref = a[correct_pos] if correct_pos != "" else ""
        target_item_id = id2im.get(target_ref, "") if target_ref else ""
        rows.append({
            "source_index": i, "set_id": gt, "question_refs": " ".join(q),
            "answer_refs": " ".join(a), "correct_answer_position_zero_based": correct_pos,
            "target_ref": target_ref, "target_item_id": target_item_id,
        })
    return rows

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--polyvore-root", required=True, help="Path to polyvore_outfits containing disjoint/ and metadata")
    ap.add_argument("--out-dir", default="splits")
    args = ap.parse_args()

    root = Path(args.polyvore_root).expanduser().resolve()
    disjoint = root / "disjoint"
    out_dir = Path(args.out_dir).expanduser().resolve()
    metadata_path = root / "polyvore_item_metadata.json"
    required = [
        disjoint / "train.json", disjoint / "valid.json", disjoint / "test.json",
        disjoint / "compatibility_test.txt", disjoint / "fill_in_blank_test.json", metadata_path,
    ]
    missing = [p for p in required if not p.exists()]
    if missing:
        for p in missing:
            print("[MISSING]", p)
        sys.exit(2)

    train = read_json(disjoint / "train.json")
    valid = read_json(disjoint / "valid.json")
    test = read_json(disjoint / "test.json")
    metadata = read_json(metadata_path)

    write_csv(out_dir / "train_ids.csv", ["source_index", "set_id"], split_rows(train))
    write_csv(out_dir / "validation_ids.csv", ["source_index", "set_id"], split_rows(valid))
    write_csv(out_dir / "test_ids.csv", ["source_index", "set_id"], split_rows(test))

    compat = parse_compatibility(disjoint / "compatibility_test.txt")
    write_csv(out_dir / "cp_ids.csv", ["source_index", "label", "set_id", "item_refs"], compat)

    test_id2im = build_id2im(test)
    fitb = parse_fitb(disjoint / "fill_in_blank_test.json", test_id2im)
    write_csv(out_dir / "fitb_ids.csv", ["source_index", "set_id", "question_refs", "answer_refs", "correct_answer_position_zero_based", "target_ref", "target_item_id"], fitb)

    train_fg2ims, _ = build_fg2ims(train, metadata)
    test_fg2ims, test_im2fg = build_fg2ims(test, metadata)
    pools = build_candidate_pools(test_fg2ims, train_fg2ims)
    or_rows = []
    for row in fitb:
        iid = row["target_item_id"]
        fg = test_im2fg.get(iid, "")
        if fg not in pools:
            continue
        or_rows.append({"source_index": row["source_index"], "set_id": row["set_id"], "target_item_id": iid, "target_item_fg": fg, "candidate_pool_size": len(pools[fg])})
    write_csv(out_dir / "or_ids.csv", ["source_index", "set_id", "target_item_id", "target_item_fg", "candidate_pool_size"], or_rows)

    counts = {"train": len(train), "valid": len(valid), "test": len(test), "compatibility_test": len(compat), "fitb_test": len(fitb), "or_test": len(or_rows)}
    ok = True
    for key, expected in EXPECTED.items():
        got = counts[key]
        status = "OK" if got == expected else "MISMATCH"
        print(f"[{status}] {key}: got={got} expected={expected}")
        if got != expected:
            ok = False

    manifest_files = ["train_ids.csv", "validation_ids.csv", "test_ids.csv", "cp_ids.csv", "fitb_ids.csv", "or_ids.csv"]
    sums = [f"{sha256(out_dir / name)}  {name}" for name in manifest_files]
    (out_dir / "SHA256SUMS.txt").write_text("\n".join(sums) + "\n", encoding="utf-8")

    source_hashes = []
    for p in required:
        source_hashes.append({
            "path": str(p.relative_to(root)),
            "bytes": p.stat().st_size,
            "sha256": sha256(p),
        })
    (out_dir / "source_dataset_manifest.json").write_text(json.dumps({
        "polyvore_root": root.name,
        "polyvore_root_note": "Runtime absolute path intentionally omitted; source identity is established by relative paths, byte sizes, SHA-256 digests, and counts.",
        "counts": counts,
        "source_files": source_hashes,
        "notes": [
            "Manifest order preserves source file order.",
            "OR eligibility reproduces evaluate_cir.py candidate-pool construction, including its archived early-break behavior.",
            "Ablation/fair-subset IDs are not inferred because the canonical historical no-missing ID file is not archived in this handoff.",
        ],
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print("[OK] wrote manifests to", out_dir)
    sys.exit(0 if ok else 3)

if __name__ == "__main__":
    main()
