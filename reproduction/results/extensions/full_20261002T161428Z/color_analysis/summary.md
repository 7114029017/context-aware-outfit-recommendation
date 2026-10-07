# Color-shift analysis of run `full_20261002T161428Z`

Copied from notebook P03 (cell 7). Automatic: color words in the query texts and the dominant colors of
the item images (2827 images of targets and top-5 items). Comparable rows (the same single color
named in the Original and the Full text): 745, over 149 cases.

| Level | Problems under Original | Solved under Full | Share | Manuscript (2025 models) |
|---|---:|---:|---:|---:|
| Row (seed, set, target) | 80 | 34 | 0.4250 | 39 / 89 = 0.4382 |
| Case (set, target) | 13 | 5 | 0.3846 | 5 / 15 = 0.3333 |

The 2025 per-query files are not preserved, so the 2025 counts cannot be recomputed; the comparable rows
(745 over 149 cases in 2025) depend only on the texts and the fair subset.
