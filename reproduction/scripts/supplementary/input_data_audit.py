#!/usr/bin/env python3
"""Audits of the input data that the reproduction uses as fixed inputs.

The generated descriptions, the CLO / MET / temperature records, the W/O/S
annotations and the counterfactual pairs were built in 2025 and are used by the
official run unchanged. This script checks them and writes, under --out-dir
(default reproduction/results/supplementary/input_data_audit/):

- temperature_prefix_audit.csv and temperature_prefix_mismatches.csv: the
  temperature at the start of each generated description against the computed
  reference TSUB_target_C;
- feature_text_identity.json: whether the context-aware text features used by
  the official run encode the stored descriptions (descriptions with identical
  text but different temperature references must have identical vectors);
- thermal_mapping_checks.json: equation (2) recomputed for every outfit, and the
  outfits outside McIntyre's stated range (M < 150 W/m2, Icl < 1.5 clo);
- split_item_overlap.csv: item IDs shared between the disjoint split files;
- counterfactual_pair_check.csv: non-target words removed by each
  counterfactual replacement, and occasion pairs without an occasion fragment;
- wos_formal_fragments.csv and met_candidate_check.csv: two annotation counts;
- summary.md.
"""
from __future__ import annotations

import argparse
import json
import pickle
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from _common import (CONTEXT_FEATURES, COUNTERFACTUAL, GENERATED, MET_CANDIDATES, SUPPLEMENTARY, WOS_JSONL,
                     fmt, load_temperatures, polyvore_root, read_csv, read_json, write_csv, write_json, write_text)

PREFIX = re.compile(r"^\s*(-?\d+(?:\.\d+)?)\s*°\s*C", re.IGNORECASE)
WORD = re.compile(r"[a-z][a-z'-]+")


class NumpyOnlyUnpickler(pickle.Unpickler):
    """Loads the feature dictionary; refuses any class other than numpy arrays."""
    ALLOWED = {("numpy._core.multiarray", "_reconstruct"), ("numpy.core.multiarray", "_reconstruct"),
               ("numpy", "ndarray"), ("numpy", "dtype")}

    def find_class(self, module, name):
        if (module, name) in self.ALLOWED:
            return super().find_class(module, name)
        raise pickle.UnpicklingError(f"blocked {module}.{name}")


def matches_reference(stated: float, tsub: float) -> bool:
    """The prompt asks for the rounded temperature; one decimal place is also accepted."""
    half_up = int(tsub + 0.5) if tsub >= 0 else -int(-tsub + 0.5)
    return abs(stated - tsub) < 0.051 or stated in (round(tsub), int(tsub), half_up)


def mismatch_pattern(stated: float, tsub: float) -> str:
    if tsub < 0 and stated in (abs(round(tsub)), abs(int(tsub)), abs(int(-tsub + 0.5))):
        return "minus sign dropped"
    if stated == round(abs(tsub) * 10) % 10:
        return "only the decimal digit kept"
    return "no pattern"


def norm(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "").lower()).strip()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--polyvore-root", default=None)
    parser.add_argument("--out-dir", type=Path, default=SUPPLEMENTARY / "input_data_audit")
    args = parser.parse_args()
    out = args.out_dir
    poly = polyvore_root(args.polyvore_root)

    generated = read_json(GENERATED)
    temperatures = load_temperatures()

    # ---------------------------------------------------------------- temperature prefixes
    per_split = defaultdict(Counter)
    mismatches = []
    for set_id, value in generated.items():
        split, record = temperatures[set_id]
        tsub = float(record["TSUB_target_C"])
        text = value["title"] if isinstance(value, dict) else str(value)
        match = PREFIX.match(text)
        per_split[split]["descriptions"] += 1
        if not match:
            per_split[split]["no temperature prefix"] += 1
            mismatches.append([set_id, split, fmt(tsub, 1), "", "no temperature prefix", text[:60]])
            continue
        stated = float(match.group(1))
        per_split[split]["integer format" if "." not in match.group(1) else "one-decimal format"] += 1
        if matches_reference(stated, tsub):
            per_split[split]["matching"] += 1
        else:
            pattern = mismatch_pattern(stated, tsub)
            per_split[split]["mismatching"] += 1
            per_split[split][pattern] += 1
            mismatches.append([set_id, split, fmt(tsub, 1), match.group(1), pattern, text[:60]])
    columns = ["descriptions", "matching", "mismatching", "only the decimal digit kept", "minus sign dropped",
               "no pattern", "no temperature prefix", "integer format", "one-decimal format"]
    split_rows = [[split, *[per_split[split][c] for c in columns]] for split in ("train", "valid", "test")]
    split_rows.append(["all", *[sum(per_split[s][c] for s in per_split) for c in columns]])
    write_csv(out / "temperature_prefix_audit.csv", ["split", *columns], split_rows)
    mismatches.sort(key=lambda r: (r[1], r[0]))
    write_csv(out / "temperature_prefix_mismatches.csv",
              ["set_id", "split", "tsub_target_c", "stated_temperature", "pattern", "description_start"], mismatches)

    # ---------------------------------------------------------------- feature identity
    with CONTEXT_FEATURES.open("rb") as f:
        features = NumpyOnlyUnpickler(f).load()
    groups = defaultdict(list)
    for set_id, value in generated.items():
        groups[(value["title"] if isinstance(value, dict) else str(value)).strip()].append(set_id)
    pairs = identical = 0
    for members in groups.values():
        for i, a in enumerate(members):
            for b in members[i + 1:]:
                if abs(float(temperatures[a][1]["TSUB_target_C"]) - float(temperatures[b][1]["TSUB_target_C"])) > 0.05:
                    pairs += 1
                    identical += int(np.allclose(features[a], features[b], atol=1e-6))
    identity = {"feature_file": CONTEXT_FEATURES.name,
                "pairs_with_identical_text_and_different_reference": pairs,
                "pairs_with_identical_vectors": identical,
                "conclusion": ("the features encode the stored descriptions as written" if pairs and identical == pairs
                               else "inconclusive")}
    write_json(out / "feature_text_identity.json", identity)

    # ---------------------------------------------------------------- equation (2) and McIntyre range
    worst = 0.0
    matching = outside_m = outside_clo = outside_any = 0
    for _, record in temperatures.values():
        clo, met = float(record["total_clo"]), float(record["met_value"])
        m = met * 58.2
        tsub = 33.5 - 3 * clo - (0.08 + 0.05 * clo) * m
        diff = abs(round(tsub, 1) - float(record["TSUB_target_C"]))
        worst = max(worst, diff)
        matching += int(diff <= 0.051)
        outside_m += int(m >= 150)
        outside_clo += int(clo >= 1.5)
        outside_any += int(m >= 150 or clo >= 1.5)
    n = len(temperatures)
    thermal = {"outfits": n,
               "equation_2_matching_stored_reference": matching,
               "largest_difference_after_rounding": round(worst, 3),
               "metabolic_rate_w_m2": "MET x 58.2",
               "outside_mcintyre_range": {"M_at_least_150": outside_m, "Icl_at_least_1_5": outside_clo,
                                          "either": outside_any, "share": round(outside_any / n, 4)}}
    write_json(out / "thermal_mapping_checks.json", thermal)

    # ---------------------------------------------------------------- split overlap
    items = {}
    for split in ("train", "valid", "test"):
        outfits = read_json(poly / "disjoint" / f"{split}.json")
        items[split] = {str(item["item_id"]) for outfit in outfits for item in outfit["items"]}
    overlap_rows = [[a, b, len(items[a] & items[b])] for a, b in (("train", "test"), ("train", "valid"), ("valid", "test"))]
    write_csv(out / "split_item_overlap.csv", ["split_a", "split_b", "shared_item_ids"], overlap_rows)

    # ---------------------------------------------------------------- counterfactual pairs
    pair_rows = []
    for record in read_csv(COUNTERFACTUAL / "A44_counterfactual_context_pairs.csv"):
        factor, cf = record["factor"], norm(record["counterfactual_description"])
        removed = []
        for other, column in (("weather", "weather_text"), ("occasion", "occasion_text"), ("style", "style_text")):
            if other == factor:
                continue
            lost = [w for w in WORD.findall(norm(record[column])) if w not in cf]
            if lost:
                removed.append(f"{other}: {' / '.join(lost)}")
        no_fragment = factor == "occasion" and not record["occasion_text"].strip()
        pair_rows.append([record["pair_id"], factor, record["direction"], "; ".join(removed),
                          "yes" if no_fragment else ""])
    write_csv(out / "counterfactual_pair_check.csv",
              ["pair_id", "factor", "direction", "non_target_words_removed", "occasion_pair_without_occasion_fragment"],
              pair_rows)

    # ---------------------------------------------------------------- annotation counts
    formal = Counter()
    for line in WOS_JSONL.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        if record.get("valid") is not True:
            continue
        for factor, values in (record.get("fragments") or {}).items():
            for value in values if isinstance(values, list) else [values]:
                if str(value).strip().lower() == "formal":
                    formal[factor] += 1
    write_csv(out / "wos_formal_fragments.csv", ["factor", "standalone_formal_fragments"],
              [[k, formal[k]] for k in ("weather", "occasion", "style")])

    candidate_rows = read_json(MET_CANDIDATES)
    candidates = {str(c["ActivityDescription"]).strip() for c in candidate_rows}
    outside = Counter(str(r["activity"]).strip() for _, r in temperatures.values()
                      if str(r["activity"]).strip() not in candidates)
    write_csv(out / "met_candidate_check.csv", ["activity_not_in_candidate_list", "outfits"],
              sorted(([a, c] for a, c in outside.items()), key=lambda r: (-r[1], r[0])))

    # ---------------------------------------------------------------- summary
    total = split_rows[-1]
    lost_pairs = [r for r in pair_rows if r[3]]
    lines = [
        "# Input data audit",
        "",
        "Computed by `reproduction/scripts/supplementary/input_data_audit.py`. The audited files are the 2025",
        "inputs that the official run uses unchanged; nothing was regenerated.",
        "",
        "## Temperature at the start of the generated descriptions",
        "",
        "| Split | Descriptions | Matching | Mismatching | Decimal digit only | Minus sign dropped | No pattern |",
        "|---|---:|---:|---:|---:|---:|---:|",
        *[f"| {r[0]} | {r[1]} | {r[2]} | {r[3]} | {r[4]} | {r[5]} | {r[6]} |" for r in split_rows],
        "",
        f"{total[3]} of {total[1]} descriptions ({100 * total[3] / total[1]:.1f}%) state a temperature that differs",
        "from `TSUB_target_C`. The train descriptions use an integer format and the valid and test descriptions one",
        "decimal place; the preserved post-processing writes only the latter.",
        "",
        f"Feature check: {identical} of {pairs} description pairs with identical text but different temperature",
        f"references have identical context-aware text features ({identity['conclusion']}).",
        "",
        "## Thermal mapping",
        "",
        f"- Equation (2) reproduces the stored temperature reference for {matching} of {n} outfits.",
        f"- Outside McIntyre's stated range (M < 150 W/m², Icl < 1.5 clo): {outside_any} outfits "
        f"({100 * outside_any / n:.1f}%); Icl ≥ 1.5: {outside_clo}; M ≥ 150: {outside_m}.",
        "",
        "## Other checks",
        "",
        "- Item IDs shared between the disjoint split files: "
        + "; ".join(f"{a} and {b}: {c}" for a, b, c in overlap_rows) + ".",
        f"- Counterfactual pairs whose replacement also removed words of a non-target factor: "
        + (", ".join(f"{r[0]} ({r[3]})" for r in lost_pairs) or "none") + ".",
        f"- Occasion pairs without an occasion fragment: "
        + (", ".join(r[0] for r in pair_rows if r[4]) or "none") + ".",
        f"- Standalone \"formal\" fragments: occasion {formal['occasion']}, style {formal['style']}, "
        f"weather {formal['weather']}.",
        f"- Outfits whose activity is not in the {len(candidate_rows)}-entry MET candidate list "
        f"({len(candidates)} distinct descriptions): {sum(outside.values())} ({len(outside)} activities).",
    ]
    write_text(out / "summary.md", lines)
    print(f"[SUPPLEMENTARY] input data audit written to {out}")


if __name__ == "__main__":
    main()
