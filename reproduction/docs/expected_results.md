# Expected results

These are the values of the official run `full_20261002T161428Z` (clean-room run of commit
`b9bf5aa`, `PASSED` 2026-10-04; see `docs/clean_room_acceptance.md`). Every
table is produced automatically by the pipeline; the source files are named
under each heading, relative to `results/summary/full_20261002T161428Z/`.

On the same runtime (NVIDIA GB10, PyTorch 2.9.1+cu130) a fresh run is expected
to reproduce these files bit for bit: three independent 35-unit runs
(`reference_20260921T175217Z`, `full_20260924T102513Z`, `full_20261002T161428Z`) produced
identical checkpoints and seed-level results. On a different GPU, CUDA or
PyTorch build, bitwise equality is not expected. Compare the run with
`scripts/compare_with_official_run.py --run-root <run>`, which judges it by the
seven conclusions R01–R07 listed in README §0.8: the run passes when
`RUN_STATUS.txt` is `PASSED` and the verdict is `IDENTICAL` or `CONSISTENT`.
The tool also lists main-experiment means that differ from these values by more
than 0.005; that list records hardware and package differences and does not
change the verdict.

## Main experiment (thesis Tables 4-11 / 4-12)

Source: `statistics/table_4_11_cp_main.csv`, `statistics/table_4_12_or_main.csv`.
BH adjustment over the eight main metrics.

| Task | Metric | Original (mean±std) | Context-aware (mean±std) | Δ | 95% CI of Δ | BH-adjusted p | Sig. (BH) | Cohen’s dz |
|---|---|---|---|---|---|---|---|---|
| CP | AUC | 0.9268 ± 0.0040 | 0.9456 ± 0.0010 | +0.0188 | [+0.01313, +0.02453] | 0.0013 | \*\* | 4.103 |
| CP | FITB Acc | 0.6357 ± 0.0035 | 0.6481 ± 0.0029 | +0.0125 | [+0.00594, +0.01902] | 0.0070 | \*\* | 2.368 |

| Task | Metric | Original (mean±std) | Context-aware (mean±std) | Δ | 95% CI of Δ | BH-adjusted p | Sig. (BH) | Cohen’s dz |
|---|---|---|---|---|---|---|---|---|
| OR | Recall@1 | 0.0138 ± 0.0005 | 0.0151 ± 0.0016 | +0.0013 | [-0.00035, +0.00293] | 0.0947 | n.s. | 0.975 |
| OR | Recall@3 | 0.0333 ± 0.0011 | 0.0372 ± 0.0014 | +0.0038 | [+0.00213, +0.00552] | 0.0044 | \*\* | 2.799 |
| OR | Recall@5 | 0.0481 ± 0.0008 | 0.0560 ± 0.0012 | +0.0079 | [+0.00630, +0.00946] | < .001 | \*\*\* | 6.197 |
| OR | Recall@10 | 0.0782 ± 0.0012 | 0.0927 ± 0.0020 | +0.0145 | [+0.01075, +0.01829] | < .001 | \*\*\* | 4.784 |
| OR | Recall@30 | 0.1684 ± 0.0037 | 0.1885 ± 0.0028 | +0.0201 | [+0.01611, +0.02414] | < .001 | \*\*\* | 6.223 |
| OR | Recall@50 | 0.2303 ± 0.0032 | 0.2532 ± 0.0023 | +0.0229 | [+0.01929, +0.02654] | < .001 | \*\*\* | 7.851 |

## Fair subset (thesis Tables 4-13 / 4-14)

Source: `ablation/T03_stage1_cp_original_vs_full.csv`,
`ablation/T04_stage1_cir_original_vs_full.csv`. BH adjustment over the five
reported metrics (the archived table convention).

| Task | Metric | Original (mean±std) | Context-aware (mean±std) | Δ | 95% CI of Δ | BH-adjusted p | Sig. (BH) | Cohen’s dz |
|---|---|---|---|---|---|---|---|---|
| CP | AUC | 0.9121 ± 0.0018 | 0.9271 ± 0.0015 | +0.0150 | [+0.01426, +0.01576] | < .001 | \*\*\* | 24.874 |
| CP | FITB Acc | 0.6121 ± 0.0019 | 0.6226 ± 0.0031 | +0.0106 | [+0.00671, +0.01443] | 0.0027 | \*\* | 3.401 |

| Task | Metric | Original (mean±std) | Context-aware (mean±std) | Δ | 95% CI of Δ | BH-adjusted p | Sig. (BH) | Cohen’s dz |
|---|---|---|---|---|---|---|---|---|
| CIR | Recall@10 | 0.0685 ± 0.0052 | 0.0753 ± 0.0043 | +0.0068 | [+0.00016, +0.01348] | 0.0467 | \* | 1.271 |
| CIR | Recall@30 | 0.1512 ± 0.0042 | 0.1693 ± 0.0046 | +0.0182 | [+0.01234, +0.02402] | 0.0025 | \*\* | 3.864 |
| CIR | Recall@50 | 0.2143 ± 0.0043 | 0.2308 ± 0.0066 | +0.0165 | [+0.00689, +0.02610] | 0.0111 | \* | 2.132 |

## Ablation (thesis Tables 4-15 / 4-16)

Source: `ablation/T05_stage2_cp_ablation.csv`, `ablation/T06_stage2_cir_ablation.csv`.
Δ = without factor − context-aware (negative means removing the factor lowers
the metric).

| Metric | Proposed (mean±std) | No-weather (mean±std) | Δ_weather | No-occasion (mean±std) | Δ_occasion | No-style (mean±std) | Δ_style |
|---|---|---|---|---|---|---|---|
| AUC | 0.9271 ± 0.0015 | 0.9256 ± 0.0007 | -0.0015 | 0.9223 ± 0.0038 | -0.0048 | 0.9136 ± 0.0014 | -0.0135 |
| FITB Acc | 0.6226 ± 0.0031 | 0.6196 ± 0.0024 | -0.0031 | 0.6203 ± 0.0029 | -0.0024 | 0.6101 ± 0.0044 | -0.0126 |

| Metric | Proposed (mean±std) | No-weather (mean±std) | Δ_weather | No-occasion (mean±std) | Δ_occasion | No-style (mean±std) | Δ_style |
|---|---|---|---|---|---|---|---|
| Recall@10 | 0.0753 ± 0.0043 | 0.0754 ± 0.0034 | +0.0001 | 0.0790 ± 0.0012 | +0.0037 | 0.0647 ± 0.0057 | -0.0106 |
| Recall@30 | 0.1693 ± 0.0046 | 0.1692 ± 0.0063 | -0.0001 | 0.1710 ± 0.0041 | +0.0017 | 0.1461 ± 0.0081 | -0.0233 |
| Recall@50 | 0.2308 ± 0.0066 | 0.2360 ± 0.0042 | +0.0052 | 0.2350 ± 0.0030 | +0.0041 | 0.2044 ± 0.0106 | -0.0264 |

## Historical thesis values (comparison only)

The values below are the archived thesis results. They are not expected
values for this artifact; differences are listed in
`docs/manuscript_reconciliation/reconciliation.md`.

### Thesis Tables 4-11 / 4-12 (main experiment)

| Task | Metric | Original | Context-aware |
|---|---|---:|---:|
| CP | AUC | 0.9292 ± 0.0013 | 0.9454 ± 0.0013 |
| CP | FITB Accuracy | 0.6374 ± 0.0037 | 0.6478 ± 0.0021 |
| OR | Recall@1 | 0.0139 ± 0.0004 | 0.0151 ± 0.0011 |
| OR | Recall@3 | 0.0333 ± 0.0010 | 0.0369 ± 0.0020 |
| OR | Recall@5 | 0.0476 ± 0.0010 | 0.0557 ± 0.0023 |
| OR | Recall@10 | 0.0780 ± 0.0010 | 0.0921 ± 0.0035 |
| OR | Recall@30 | 0.1677 ± 0.0045 | 0.1874 ± 0.0038 |
| OR | Recall@50 | 0.2303 ± 0.0035 | 0.2518 ± 0.0038 |

Recovered paired seeds: `1,2,3,4,5`.

### Thesis Tables 4-13 / 4-14 (fair subset)

| Task | Metric | Original | Context-aware |
|---|---|---:|---:|
| CP | AUC | 0.9112 ± 0.0015 | 0.9265 ± 0.0015 |
| CP | FITB Accuracy | 0.6109 ± 0.0035 | 0.6228 ± 0.0034 |
| OR | Recall@10 | 0.0686 ± 0.0043 | 0.0781 ± 0.0023 |
| OR | Recall@30 | 0.1517 ± 0.0077 | 0.1720 ± 0.0034 |
| OR | Recall@50 | 0.2111 ± 0.0046 | 0.2351 ± 0.0045 |

### Thesis Tables 4-15 / 4-16 (ablation)

| Metric | Full | No-weather | No-occasion | No-style |
|---|---:|---:|---:|---:|
| CP AUC | 0.9265 | 0.9241 | 0.9236 | 0.9110 |
| CP FITB | 0.6228 | 0.6191 | 0.6197 | 0.6064 |
| OR R@10 | 0.0781 | 0.0760 | 0.0774 | 0.0671 |
| OR R@30 | 0.1720 | 0.1690 | 0.1706 | 0.1524 |
| OR R@50 | 0.2351 | 0.2325 | 0.2347 | 0.2110 |
