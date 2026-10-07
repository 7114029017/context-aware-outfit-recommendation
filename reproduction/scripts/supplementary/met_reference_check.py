#!/usr/bin/env python3
"""Rebuild the 457-entry MET reference list from the official 2024 Adult Compendium (thesis Section 3.3.2,
Tables 3-1 and 3-2).

The generation prompts used a reduced copy of the 2024 Adult Compendium of Physical Activities,
01_資料建構_data_construction/clo_met_temperature/MET_reference/adult_activity_compendium_sorted_2024.json
(457 entries). The program that reduced it is not preserved; the thesis states the rule: sort the
activities by MET value, and among activities with the same MET value keep one representative per
Major Heading. This script applies that rule to the official Compendium file and compares the result
with the preserved list, entry by entry.

Input: the Compendium PDF published at pacompendium.com (downloaded by `bootstrap_data.sh
--with-compendium`; its SHA-256 is pinned and equals the copy in the 2025 handoff), read with poppler's
`pdftotext -bbox` (word positions). The table rows are rebuilt from the positions of the activity
codes. A row whose long description wraps above and below its code has no description text on the
code's own line; such rows are flagged, because a line-by-line text extraction drops them. Without the
PDF or `pdftotext` the script reports SKIPPED and writes nothing.

Writes under --out-dir (default reproduction/results/supplementary/met_reference_check/):
met_reference_by_heading.csv (Table 3-1 counts against the Compendium and the rebuilt list),
met_reference_differences.csv (the rows that explain every difference) and summary.md. The Compendium
itself is not copied; only activity codes, headings, MET values and counts are written.
"""
from __future__ import annotations

import argparse
import hashlib
import html
import re
import shutil
import subprocess
from collections import Counter
from pathlib import Path

from _common import MET_CANDIDATES, REPRO, SUPPLEMENTARY, read_json, write_csv, write_text

PDF_SHA256 = "ac30234b8f8f813837e282773cfcb3e0fe062334777c2f7b3213429c4fbc251c"
PDF_URL = "https://pacompendium.com/wp-content/uploads/2025/02/1_2024-adult-compendium_1_2024.pdf"
WORD = re.compile(r'<word xMin="([\d.]+)" yMin="([\d.]+)" xMax="([\d.]+)" yMax="([\d.]+)">(.*?)</word>')
TABLE_3_1 = {  # thesis Table 3-1: Major Heading -> (activities, after processing)
    "Bicycling": (42, 25), "Conditioning Exercise": (89, 38), "Dancing": (35, 18), "Fishing & Hunting": (38, 18),
    "Home Activities": (71, 21), "Home Repair": (37, 10), "Inactivity": (14, 4), "Lawn & Garden": (53, 24),
    "Miscellaneous": (41, 9), "Music Playing": (27, 10), "Occupation": (143, 34), "Running": (82, 41),
    "Self Care": (11, 6), "Sexual Activity": (3, 3), "Sports": (157, 44), "Transportation": (14, 10),
    "Walking": (77, 34), "Water Activities": (102, 41), "Winter Activities": (78, 34),
    "Religious Activities": (20, 12), "Volunteer Activities": (22, 13), "Video Games": (8, 8),
}
TEXT_ORIGINAL_COUNT = 1114  # thesis text: "原始對照表共包含 1,114 筆活動類型"


def compendium_rows(pdf: Path) -> list[dict]:
    """Table rows from word positions: code column 120-195 pt, heading left of 125 pt, MET between the code
    and the description column. Each word belongs to the code whose line centre is nearest."""
    xhtml = subprocess.run(["pdftotext", "-bbox", str(pdf), "-"], capture_output=True, text=True, check=True).stdout
    rows = []
    for page, chunk in enumerate(xhtml.split("<page ")[1:], 1):
        words = [(float(a), float(b), float(c), float(d), html.unescape(w)) for a, b, c, d, w in WORD.findall(chunk)]
        header = [w[0] for w in words if w[4] == "Activity" and w[0] > 240]
        desc_x = min(header) if header else 240.0
        codes = sorted((w for w in words if re.fullmatch(r"\d{5}", w[4]) and 120 < w[0] < 195), key=lambda w: w[1])
        cells = [{"page": page, "y": (c[1] + c[3]) / 2, "code": c[4], "head": [], "met": [], "desc": []} for c in codes]
        for w in words:
            if w in codes or not cells:
                continue
            yc = (w[1] + w[3]) / 2
            if yc < cells[0]["y"] - 12 or yc > cells[-1]["y"] + 25:
                continue  # page title, column header, footer
            cell = min(cells, key=lambda c: abs(c["y"] - yc))
            column = "head" if w[0] < 125 else "met" if w[0] < desc_x - 5 else "desc"
            cell[column].append((round(yc, 1), w[0], w[4]))
        for c in cells:
            rows.append({"page": c["page"], "code": c["code"],
                         "head": " ".join(t for _, _, t in sorted(c["head"])),
                         "met": float(" ".join(t for _, _, t in sorted(c["met"]))),
                         "desc": " ".join(t for _, _, t in sorted(c["desc"])),
                         "desc_on_code_line": any(abs(y - c["y"]) < 2.0 for y, _, _ in c["desc"])})
    return rows


def reduce(rows: list[dict]) -> list[dict]:
    """The thesis rule: sort by MET value (ties keep the Compendium order); keep the first activity per
    (MET value, Major Heading)."""
    seen, kept = set(), []
    for r in sorted(rows, key=lambda r: r["met"]):
        if (r["met"], r["head"]) not in seen:
            seen.add((r["met"], r["head"]))
            kept.append(r)
    return kept


def norm(text: str) -> str:
    return re.sub(r"\s+", " ", str(text).replace("\xa0", " ")).strip()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out-dir", type=Path, default=SUPPLEMENTARY / "met_reference_check")
    parser.add_argument("--pdf", default=None, help="default: the path recorded by bootstrap_data.sh --with-compendium")
    parser.add_argument("--polyvore-root", default=None, help="accepted for run_all.sh; not used")
    args = parser.parse_args()
    recorded = REPRO / ".local" / "compendium_pdf.txt"
    pdf = Path(args.pdf).expanduser().resolve() if args.pdf else (
        Path(recorded.read_text(encoding="utf-8").strip()) if recorded.is_file() else None)
    if pdf is None or not pdf.is_file():
        print("[SUPPLEMENTARY] MET reference check SKIPPED: the Compendium PDF is not downloaded "
              "(bash reproduction/scripts/bootstrap_data.sh --with-compendium)")
        return
    if shutil.which("pdftotext") is None:
        print("[SUPPLEMENTARY] MET reference check SKIPPED: pdftotext (poppler-utils) is not installed")
        return
    if hashlib.sha256(pdf.read_bytes()).hexdigest() != PDF_SHA256:
        raise SystemExit(f"[SUPPLEMENTARY BLOCKED] {pdf} is not the pinned Compendium file ({PDF_URL})")
    out = args.out_dir

    rows = compendium_rows(pdf)
    preserved = read_json(MET_CANDIDATES)
    preserved_codes = [str(e["ActivityCode"]) for e in preserved]
    line_rows = [r for r in rows if r["desc_on_code_line"]]
    from_lines, from_all = reduce(line_rows), reduce(rows)
    same_codes = [r["code"] for r in from_lines] == preserved_codes
    same_values = all(r["head"] == e["MajorHeading"] and r["met"] == float(e["METValue"])
                      for r, e in zip(from_lines, preserved))
    by_code = {r["code"]: r for r in rows}
    pairs = [(e["ActivityDescription"], by_code[str(e["ActivityCode"])]["desc"]) for e in preserved]
    title = "2024 Adult Compendium of Physical Activities"
    desc_exact = sum(1 for a, b in pairs if a == b)
    desc_spacing = sum(1 for a, b in pairs if a != b and norm(a) == norm(b))
    desc_title = sum(1 for a, b in pairs if norm(a) != norm(b) and norm(a.replace(title, "")) == norm(b))

    count_all, count_lines = Counter(r["head"] for r in rows), Counter(r["head"] for r in line_rows)
    kept_lines, kept_all = Counter(r["head"] for r in from_lines), Counter(r["head"] for r in from_all)
    kept_preserved = Counter(e["MajorHeading"] for e in preserved)
    heading_rows = [[h, t[0], count_all[h], count_lines[h], t[1], kept_lines[h], kept_preserved[h], kept_all[h]]
                    for h, t in TABLE_3_1.items()]
    heading_rows.append(["Total", sum(t[0] for t in TABLE_3_1.values()), len(rows), len(line_rows),
                         sum(t[1] for t in TABLE_3_1.values()), len(from_lines), len(preserved), len(from_all)])
    write_csv(out / "met_reference_by_heading.csv",
              ["major_heading", "table_3_1_activities", "compendium_activities", "with_description_on_code_line",
               "table_3_1_after_processing", "rebuilt_from_line_rows", "preserved_list", "rebuilt_from_all_rows"],
              heading_rows)

    lost = [r for r in rows if not r["desc_on_code_line"]]
    lines_codes, all_codes = {r["code"] for r in from_lines}, {r["code"] for r in from_all}
    diff_rows = [["description wraps around the code (dropped by a line-based extraction)", r["code"], r["head"],
                  r["met"], "yes" if r["code"] in all_codes else ""] for r in lost]
    diff_rows += [["kept from all rows, not from line rows", c, by_code[c]["head"], by_code[c]["met"], ""]
                  for c in sorted(all_codes - lines_codes)]
    diff_rows += [["kept from line rows, not from all rows", c, by_code[c]["head"], by_code[c]["met"], ""]
                  for c in sorted(lines_codes - all_codes)]
    write_csv(out / "met_reference_differences.csv",
              ["row_type", "activity_code", "major_heading", "met_value", "kept_when_all_rows_are_used"], diff_rows)

    table_mismatch = [r[0] for r in heading_rows[:-1] if r[1] != r[2]]
    lines = [
        "# MET reference list (thesis Section 3.3.2, Tables 3-1 and 3-2)",
        "",
        "Computed by `reproduction/scripts/supplementary/met_reference_check.py` from the official 2024 Adult",
        f"Compendium PDF ({PDF_URL}; SHA-256 {PDF_SHA256[:16]}..., the same file as in the 2025 handoff).",
        "",
        f"- The Compendium table has {len(rows):,} activities in {len(count_all)} Major Headings (the thesis text gives "
        f"{TEXT_ORIGINAL_COUNT:,}, the figure of the Compendium's publication; Table 3-1 sums to "
        f"{sum(t[0] for t in TABLE_3_1.values()):,}).",
        f"- {len(lost)} rows have a description long enough to wrap above and below the activity code; a line-by-line",
        f"  text extraction of the PDF drops them and keeps {len(line_rows):,} rows.",
        f"- The thesis rule (sort by MET value; keep the first activity per MET value and Major Heading) applied to "
        f"those {len(line_rows):,} rows gives {len(from_lines)} entries. Same activity codes in the same order as the "
        f"preserved list: {'yes' if same_codes else 'NO'}; same Major Heading and MET value: "
        f"{'yes' if same_values else 'NO'}; descriptions: {desc_exact} identical, {desc_spacing} differ only in spacing "
        f"(double or non-breaking spaces kept by the 2025 extraction)"
        + (f", and in {desc_title} the preserved description ends with the page title, which a text extraction "
           "joins at a page break." if desc_title else "."),
        f"- Applied to all {len(rows):,} rows, the rule gives {len(from_all)} entries: "
        f"{len(from_all) - len(from_lines)} more "
        f"(MET value, Major Heading) groups, and {len(lines_codes - all_codes)} groups represented by another "
        "activity (`met_reference_differences.csv`).",
        f"- After processing, the per-heading counts of Table 3-1 equal the rebuilt list: "
        f"{'yes' if all(r[4] == r[5] for r in heading_rows) else 'NO'}. The 'activities' column of Table 3-1 does "
        f"not match the Compendium in {len(table_mismatch)} of {len(TABLE_3_1)} headings "
        f"(`met_reference_by_heading.csv`).",
        "",
        "Conclusion: the preserved 457-entry list is reproduced exactly from the official file with the rule in the",
        "thesis, given that the 2025 extraction lost the rows with wrapped descriptions. Done on all rows, the rule",
        "would give a slightly different list; the generated data used the preserved one.",
    ]
    write_text(out / "summary.md", lines)
    print(f"[SUPPLEMENTARY] MET reference check written to {out}")


if __name__ == "__main__":
    main()
