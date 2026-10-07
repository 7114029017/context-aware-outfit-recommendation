# Text-swap evaluation (2025 sweep of CP_evaluate.py and CIR_evaluate.py)

Models of run `full_20261002T161428Z`, seeds 1, 2, 3, 4, 5; device cuda. Each model is evaluated with the original titles and with the context-aware descriptions; the evaluators and the working copy are those of the full run's main evaluation.

Setup check: the matching-text combinations evaluated again (2) equal the run's stored results: yes. The other matching-text rows are the stored results.

| Model | Text | AUC | FITB | R@1 | R@10 | R@30 | R@50 |
|---|---|---:|---:|---:|---:|---:|---:|
| Original | original | 0.9268 ± 0.0040 | 0.6357 ± 0.0035 | 0.0138 ± 0.0005 | 0.0782 ± 0.0012 | 0.1684 ± 0.0037 | 0.2303 ± 0.0032 |
| Original | context | 0.9338 ± 0.0041 | 0.6418 ± 0.0037 | 0.0139 ± 0.0003 | 0.0818 ± 0.0024 | 0.1721 ± 0.0034 | 0.2360 ± 0.0019 |
| Context | original | 0.9061 ± 0.0020 | 0.6179 ± 0.0013 | 0.0119 ± 0.0004 | 0.0711 ± 0.0019 | 0.1551 ± 0.0022 | 0.2136 ± 0.0023 |
| Context | context | 0.9456 ± 0.0010 | 0.6481 ± 0.0029 | 0.0151 ± 0.0016 | 0.0927 ± 0.0020 | 0.1885 ± 0.0028 | 0.2532 ± 0.0023 |

Mean over the seeds ± SD. Effect of the text for each model (context-aware minus original text, mean over seeds; seeds with a positive difference):

| Model | AUC | FITB | R@10 | R@30 | R@50 |
|---|---:|---:|---:|---:|---:|
| Original | +0.0070 (5/5) | +0.0062 (5/5) | +0.0036 (5/5) | +0.0038 (5/5) | +0.0056 (5/5) |
| Context | +0.0395 (5/5) | +0.0303 (5/5) | +0.0216 (5/5) | +0.0334 (5/5) | +0.0397 (5/5) |

Files: `text_swap_results.csv` (one row per seed, model and text), `text_swap_summary.csv`, `logs/` (evaluator logs, not committed).

