# Remaining thesis reproduction matrix

- Git commit: b9bf5aac7d07cb39cacc7f73eb28166848c19fe4
- Dataset root identity: uiuc-polyvore-hf
- Overall remaining-analysis status: **partial**

| Module | Status | Evidence |
|---|---|---|
| length | **exact** | Recomputed length correlations/buckets/matched subsets from 46,555 archived seed-level rows (9311 unique pairs). |
| target_clue | **partial** | Recomputed target-clue leakage on the exact OR scope N=9311. The historical P12 intermediate source reliability_meta_from_subset.csv is not archived in the repository, so current reconstruction uses preserved OR IDs plus available WOS/Polyvore metadata. Scope and bigram count reproduce exactly, while several lexical-count rows differ. |
| judge_preserved_outputs | **exact** | Recomputed Judge T19 agreement, T21 threshold agreement, bottom-p overlaps, and preserved P0-R2/P1/P2 robustness outputs. The module's 'exact' status applies to verified preserved-output T19/T21 and archived-robustness consistency; thesis Table 4-7 bottom-p overlaps are NOT exact (paper 668/1323/2833 vs current 669/1338/2877). Archived robustness original_score values match the preserved formal outputs; historical sample regeneration provenance is partial. |
| human_audit | **exact** | Seed-42 30-case selection match=True; manual judgments=750 (300 Qwen + 450 Gemma). |
| case_analysis | **partial** | Archived subgroup/case tables present=True; reconstructed fair-subset row-level details=25/25. Historical memberwise fair-subset identity remains unresolved. |
| two_tower | **exact** | Recomputed A30 five-seed summary; A32 matched cells=44, max abs diff=1.14e-13; loadable checkpoints=20/20. |
| judge_robustness_sampling_provenance | **partial** | Historical robustness sample regeneration does not reproduce the archived 150 IDs: intersection=3/150, archived-only=147, current-only=147. All six archived robustness files use the same 150 IDs and their original_score fields match the preserved formal outputs exactly; the unresolved component is historical sample-generation provenance. |
| judge_exact_generation | **partial** | Exact Judge generation requires archived P0/P1/P2 prompt text; preserved-output recomputation is reported separately. |
| case_visuals | **partial** | Image-dependent qualitative figure regeneration requires Polyvore images/. |

The overall status is conservative: unresolved historical source/provenance components remain partial even when downstream numeric recomputation succeeds.
