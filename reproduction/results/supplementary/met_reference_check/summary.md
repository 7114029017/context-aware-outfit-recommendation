# MET reference list (thesis Section 3.3.2, Tables 3-1 and 3-2)

Computed by `reproduction/scripts/supplementary/met_reference_check.py` from the official 2024 Adult
Compendium PDF (https://pacompendium.com/wp-content/uploads/2025/02/1_2024-adult-compendium_1_2024.pdf; SHA-256 ac30234b8f8f8138..., the same file as in the 2025 handoff).

- The Compendium table has 1,111 activities in 22 Major Headings (the thesis text gives 1,114, the figure of the Compendium's publication; Table 3-1 sums to 1,164).
- 14 rows have a description long enough to wrap above and below the activity code; a line-by-line
  text extraction of the PDF drops them and keeps 1,097 rows.
- The thesis rule (sort by MET value; keep the first activity per MET value and Major Heading) applied to those 1,097 rows gives 457 entries. Same activity codes in the same order as the preserved list: yes; same Major Heading and MET value: yes; descriptions: 435 identical, 21 differ only in spacing (double or non-breaking spaces kept by the 2025 extraction), and in 1 the preserved description ends with the page title, which a text extraction joins at a page break.
- Applied to all 1,111 rows, the rule gives 459 entries: 2 more (MET value, Major Heading) groups, and 3 groups represented by another activity (`met_reference_differences.csv`).
- After processing, the per-heading counts of Table 3-1 equal the rebuilt list: yes. The 'activities' column of Table 3-1 does not match the Compendium in 18 of 22 headings (`met_reference_by_heading.csv`).

Conclusion: the preserved 457-entry list is reproduced exactly from the official file with the rule in the
thesis, given that the 2025 extraction lost the rows with wrapped descriptions. Done on all rows, the rule
would give a slightly different list; the generated data used the preserved one.
