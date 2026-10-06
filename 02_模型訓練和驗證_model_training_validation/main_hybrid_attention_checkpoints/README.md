# Main Hybrid Attention checkpoints

本資料夾只保留期刊主實驗最後採用的 best checkpoint，不保留每個 epoch 的 ckpt_*.pt。

整理原則：

- cp_old_seed1 到 cp_old_seed5：Original text 條件下的 CP 模型。
- cir_old_seed1 到 cir_old_seed5：Original text 條件下的 CIR 模型。
- cp_new_seed1 到 cp_new_seed5：Context-aware description 條件下的 CP 模型。
- cir_new_seed1 到 cir_new_seed5：Context-aware description 條件下的 CIR 模型。

每個資料夾只保留：

- ckpt.pt：訓練程式儲存的 best checkpoint。
- log.txt：訓練紀錄。

CIR 訓練關係：

CIR 由同 seed、同文字條件的 CP checkpoint 初始化後再訓練。