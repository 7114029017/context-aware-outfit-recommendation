# Per-query evaluation of one main unit on the CPU (plumbing check)

`main_cir_per_query.py --units original_seed1 --eval-device cpu` on the official run. The official run
evaluated on CUDA with float16 autocast; the CPU runs in float32.

| Metric | Official run (GPU) | CPU | Queries apart (of 9,311) |
|---|---:|---:|---:|
| recall_at_1 | 0.014177 | 0.014177 | 0 |
| recall_at_3 | 0.034368 | 0.034583 | 2 |
| recall_at_5 | 0.048545 | 0.048652 | 1 |
| recall_at_10 | 0.079583 | 0.079583 | 0 |
| recall_at_30 | 0.165181 | 0.165396 | 2 |
| recall_at_50 | 0.230910 | 0.230910 | 0 |

The per-query hits average to the CPU recalls and agree with the ranks. Because the CPU values differ
by one or two queries at some cut-offs, the text-length step runs on the GPU.
