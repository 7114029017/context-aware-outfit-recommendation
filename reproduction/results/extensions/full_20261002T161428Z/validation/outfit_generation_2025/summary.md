# Step-by-step outfit generation (2025 try-on demo P02, without the try-on)

Case 224188768: "26.2° C, Effortless Summer Wedding Style with a Pop of Citrus". Model: preserved 2025 cir_new_seed1; device cpu.

Original outfit by slot: overall 209144226, shoe 208337886, bag 201548065; generation order overall → shoe → bag. Candidate bank: bag 11,688, lower 7,695, overall 5,170, shoe 13,225, upper 14,270 test items.

| Setting | Kept | Step | Slot | Item | Score | Runner-up (score) |
|---|---|---:|---|---|---:|---|
| keep0 | — | 1 | overall | 56998165 | 0.7139 | 149930953 (0.7138) |
| keep0 | — | 2 | shoe | 121507083 | 0.8287 | 133306142 (0.8155) |
| keep0 | — | 3 | bag | 161194648 | 0.7800 | 203680817 (0.7792) |
| keep1 | overall | 1 | shoe | 82253359 | 0.7509 | 27368340 (0.7140) |
| keep1 | overall | 2 | bag | 112382686 | 0.7621 | 161194648 (0.7199) |
| keep2 | overall shoe | 1 | bag | 112382686 | 0.6873 | 158101683 (0.6433) |

Against P02's printed picks (2025 model on CUDA): 6 of 6 picks equal; largest difference of the four-decimal scores 0.0001.

Item grid (Polyvore photos, not committed): `figure_P02_outfit_generation_224188768_2025.png` in the figures folder.

The try-on images of P02 (FastFit on the person photo model_image.png) are not reproduced: the person photo was not preserved and FastFit is not part of the repository.

