# Text-swap evaluation (2025 sweep of CP_evaluate.py and CIR_evaluate.py)

Models of the preserved 2025 checkpoints, seeds 1, 2, 3, 4, 5; device cuda. Each model is evaluated with the original titles and with the context-aware descriptions; the evaluators and the working copy are those of the full run's main evaluation.

| Model | Text | AUC | FITB | R@1 | R@10 | R@30 | R@50 |
|---|---|---:|---:|---:|---:|---:|---:|
| Original | original | 0.9292 ± 0.0013 | 0.6374 ± 0.0035 | 0.0140 ± 0.0004 | 0.0780 ± 0.0009 | 0.1677 ± 0.0045 | 0.2304 ± 0.0036 |
| Original | context | 0.9356 ± 0.0017 | 0.6442 ± 0.0028 | 0.0140 ± 0.0006 | 0.0809 ± 0.0018 | 0.1721 ± 0.0040 | 0.2351 ± 0.0028 |
| Context | original | 0.9059 ± 0.0018 | 0.6168 ± 0.0027 | 0.0111 ± 0.0008 | 0.0701 ± 0.0020 | 0.1543 ± 0.0022 | 0.2110 ± 0.0034 |
| Context | context | 0.9454 ± 0.0013 | 0.6477 ± 0.0021 | 0.0151 ± 0.0011 | 0.0921 ± 0.0035 | 0.1874 ± 0.0037 | 0.2518 ± 0.0038 |

Mean over the seeds ± SD. Effect of the text for each model (context-aware minus original text, mean over seeds; seeds with a positive difference):

| Model | AUC | FITB | R@10 | R@30 | R@50 |
|---|---:|---:|---:|---:|---:|
| Original | +0.0064 (5/5) | +0.0067 (5/5) | +0.0029 (5/5) | +0.0045 (5/5) | +0.0047 (5/5) |
| Context | +0.0394 (5/5) | +0.0310 (5/5) | +0.0220 (5/5) | +0.0332 (5/5) | +0.0408 (5/5) |

Matching-text means and SDs against the archived A14 (`a14_check.csv`):

| Task | Metric | Model | Mean (A14) | SD (A14) | Equal |
|---|---|---|---|---|---|
| CP | auc | original | 0.929165 (0.929165) | 0.001302 (0.001302) | yes |
| CP | auc | context | 0.945370 (0.945370) | 0.001250 (0.001251) | NO |
| CP | fitb_acc | original | 0.637438 (0.637412) | 0.003522 (0.003674) | NO |
| CP | fitb_acc | context | 0.647725 (0.647752) | 0.002062 (0.002114) | NO |
| CIR | recall_at_10 | original | 0.077972 (0.077972) | 0.000945 (0.000967) | NO |
| CIR | recall_at_10 | context | 0.092128 (0.092085) | 0.003538 (0.003459) | NO |
| CIR | recall_at_30 | original | 0.167651 (0.167694) | 0.004490 (0.004539) | NO |
| CIR | recall_at_30 | context | 0.187434 (0.187391) | 0.003683 (0.003832) | NO |
| CIR | recall_at_50 | original | 0.230416 (0.230330) | 0.003589 (0.003465) | NO |
| CIR | recall_at_50 | context | 0.251788 (0.251788) | 0.003831 (0.003778) | NO |

Files: `text_swap_results.csv` (one row per seed, model and text), `text_swap_summary.csv`, `logs/` (evaluator logs, not committed).

