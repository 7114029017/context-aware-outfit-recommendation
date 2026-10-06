# Reproducibly reconstructed fair subset

Tables 4-13 to 4-16 (TORS Table 7) and the Weather / Occasion / Style
ablation use one fair subset: the outfits whose context-aware description
contains all three semantic factors (thesis Eq. 3-17,
`D_fair = {x | W(x) ≠ ∅, O(x) ≠ ∅, S(x) ≠ ∅}`). The historical memberwise ID
file was not preserved, so the 2026 artifact reconstructs the subset from the
preserved factor split. This reconstruction is the formal subset of the
revised manuscript.

The subset is selected by semantic-factor presence, not by text length. The
matched-length analysis of Table 4-5 uses a different subset.

## Construction rule

Select the WOS v5 merged-retry rows whose weather, occasion and style fragment
lists each contain at least one non-blank string.

| Source | SHA-256 (prefix) |
|---|---|
| `01_資料建構_data_construction/generated_descriptions/02_三因子拆分_wos_factor_split/wos_split_results_v5_merged_retry_round3.jsonl` | `d77d13c231b61a85` |
| `01_資料建構_data_construction/generated_descriptions/01_生成結果_generation_results/new_polyvore_outfit_titles_with_ablation.json` | `7365ce6b5f99d64c` |
| Polyvore Outfits Disjoint `train.json` / `valid.json` / `test.json` | `0a7c0f8dc60f420d` / `e3b3e2dca09333f8` / `df650429c987d73b` |

Full hashes are in `splits/fair_subset/fair_subset_reconstruction_manifest.json`.

## Exclusions

Of the 35,140 valid outfits, every description contains weather (W). The
remaining patterns are:

| Factors present | Outfits | In fair subset |
|---|---:|---|
| W only | 3 | no |
| W + S (no O) | 13,192 | no |
| W + O (no S) | 42 | no |
| W + O + S | 21,903 | yes |

Coverage: weather 100.0%, style 99.87%, occasion 62.45%; all three 62.33%.

## Splits and OR queries

| Scope | Count | File |
|---|---:|---|
| Fair subset | 21,903 | `fair_subset_ids.txt` |
| Train | 10,225 | `train_ids.txt` |
| Validation | 1,748 | `valid_ids.txt` |
| Test | 9,930 | `test_ids.txt` |
| Evaluable OR (CIR) queries | 3,432 | `or_query_ids.csv` |

The splits follow the PO-D disjoint split. Of the 9,930 FITB test questions in
the subset, 3,432 are evaluable under the preserved evaluator: the target's
fine-grained category must reach a 3,000-item candidate pool (8 categories).
The other 6,498 are excluded. `or_query_ids.csv` lists
`set_id, target_item_id, target_item_fg`, sorted.

## Checksums

`splits/fair_subset/SHA256SUMS.txt`:

| File | SHA-256 |
|---|---|
| `fair_subset_ids.txt` | `bd1d24067b8d5021d3dc1e90651855e9821b1df7b7bf7683ff55ee61c5a298e7` |
| `or_query_ids.csv` | `5d5c3a9a07ad22ecb32879d2fc5136c942d654aa088308aedceec9d67afee714` |
| `train_ids.txt` | `fa6b4565cadfdb3e859753dacfd7e9a2fb71ebd76a1bdfcedc754a731c0b64f9` |
| `valid_ids.txt` | `38bcfcff96c225b1a3273923ad17fa9d66366b83cb36800c9b7f0f50c2a5330b` |
| `test_ids.txt` | `838d617e88e13c266eb3ecbece144915e1fc92779d38f9238908f17d4a904dbb` |

## Programs

| Program | Role |
|---|---|
| `scripts/reconstruct_fair_subset_from_wos.py` | builds the IDs, splits and manifest |
| `scripts/audit_fair_subset_cir_scope.py` | verifies the 3,432-query scope and exports `or_query_ids.csv` |
| `scripts/run_fair_subset_single_seed.py` | trains and evaluates one variant × seed on the subset |
| `scripts/pipeline/build_fair_subset_report.py`, `scripts/summarize_fair_subset_5seed.py` | check that all 25 units evaluate the same 3,432 queries |

Every full run rebuilds the subset under `ablation/fair_subset_reconstruction/`
and re-exports the queries to `ablation/fair_subset_or_query_ids.csv`.

## One subset for Tables 4-13 to 4-16

- All 25 ablation units of `reference_20260921T175217Z` record the same subset
  SHA-256 (`bd1d2406…`) in `reference_seed_index.csv`.
- The membership SHA-256 of `or_query_ids.csv` (`a7cf547b…`) equals the
  `evaluation_query_membership_sha256` recorded by that run, and each of the
  25 units was evaluated on exactly these 3,432 queries.

## Naming

Before 2026-10-01 the same files were named `*_RECONSTRUCTED_CANDIDATE.txt`
under `splits/ablation_candidate/`; their content is unchanged. Internal field
names and status codes that say "candidate" (for example
`candidate_unique_ids` or `PASSED_CANDIDATE_SOURCE`) refer to this subset.

## Limitation

The historical generation program, memberwise ID file and its SHA-256 were not
recovered, so the 21,903 members are not claimed to be identical to the 2025
subset member by member.
