# Step-by-step outfit generation (2025 try-on demo P02, without the try-on)

Case 224188768: "26.2° C, Effortless Summer Wedding Style with a Pop of Citrus". Model: run full_20261002T161428Z, main/context_seed1; device cpu.

Original outfit by slot: overall 209144226, shoe 208337886, bag 201548065; generation order overall → shoe → bag. Candidate bank: bag 11,688, lower 7,695, overall 5,170, shoe 13,225, upper 14,270 test items.

| Setting | Kept | Step | Slot | Item | Score | Runner-up (score) |
|---|---|---:|---|---|---:|---|
| keep0 | — | 1 | overall | 209144226 | 0.7933 | 196120491 (0.7817) |
| keep0 | — | 2 | shoe | 82253359 | 0.7568 | 133780489 (0.7355) |
| keep0 | — | 3 | bag | 112382686 | 0.7885 | 205926499 (0.7431) |
| keep1 | overall | 1 | shoe | 82253359 | 0.7568 | 133780489 (0.7355) |
| keep1 | overall | 2 | bag | 112382686 | 0.7885 | 205926499 (0.7431) |
| keep2 | overall shoe | 1 | bag | 112382686 | 0.6884 | 205926499 (0.6654) |

Item grid (Polyvore photos, not committed): `figure_P02_outfit_generation_224188768_seed1.png` in the figures folder.

The try-on images of P02 (FastFit on the person photo model_image.png) are not reproduced: the person photo was not preserved and FastFit is not part of the repository.

