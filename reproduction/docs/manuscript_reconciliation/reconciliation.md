# 論文／TORS 稿結果對帳表

> 本檔由 `reproduction/scripts/build_manuscript_reconciliation.py` 產生，請勿手動修改。稿件數值轉錄於同目錄的 `manuscript_values.csv`；逐列完整資料（含來源 CSV、程式與 run ID）見 `reconciliation.csv`。

| 項目 | 內容 |
|---|---|
| 對帳 run | `full_20261002T161428Z`（commit `b9bf5aac7d07cb39cacc7f73eb28166848c19fe4`） |
| 等價 run | `reference_20260921T175217Z`、`full_20260924T102513Z` 與此 run 逐位元相同（70 個 checkpoint、逐 seed 結果），見 `reproduction/docs/clean_room_acceptance.md` |
| 稿件 | 碩論終稿 `04_文件資料_documents/thesis/情境感知驅動的智慧穿搭推薦系統 論文終稿.pdf`；TORS 投稿稿 `04_文件資料_documents/journal/ACM_TORS_English_v4.pdf`（兩者結果數值相同，頁碼為印刷頁碼） |
| `<RUN>` | `reproduction/results/summary/full_20261002T161428Z` |

## 狀態說明

| 狀態 | 意義 | 筆數 |
|---|---|---:|
| 保留（`KEEP`） | 最新值四捨五入到稿件位數後相同，或稿件敘述仍成立 | 62 |
| 更新數字（`UPDATE_NUMBER`） | 數字改變，但方向與顯著性結論不變 | 170 |
| 改寫敘述（`REWRITE`） | 方向、顯著性、筆數或文字結論與最新結果不一致，敘述必須改寫 | 21 |
| 需補寫（`ADD`） | 稿件未記載，修訂稿需補寫 | 15 |
| 非正式 run（`OUTSIDE_FORMAL_RUN`） | 35 組正式訓練未重新產生；來源為保存輸出或其他分析 | 19 |

## 修稿必須改寫的地方

| ID | 位置（論文／TORS） | 稿件 | 最新結果 |
|---|---|---|---|
| A12 | 摘要（p.i）；Abstract（p.ii）／Abstract（p.1） | 天氣與場合資訊能提供額外的情境限制／weather and occasion add complementary constraints | No-W：R@10 +0.0001、R@30 -0.0001、R@50 +0.0052；No-O：R@10 +0.0037、R@30 +0.0017、R@50 +0.0041（正值表示移除後 Recall 反而上升）（第九節：不得將 Weather 與 Occasion 寫成在所有 OR 指標都具有穩定正貢獻。） |
| S34 | 表4-1（p.41）／4.3（p.16） | NVIDIA RTX PRO 6000 Blackwell Workstation Edition（96 GB） | NVIDIA GB10（採用最新結果時，實驗環境描述必須改成正式 run 的硬體。） |
| T4-12.r1.ci_low | 表4-12（p.51）／Table 6（p.20） | Recall@1｜95% CI 下界：+0.00005 | -0.00035（信賴區間是否包含 0 改變） |
| T4-12.r1.sig | 表4-12（p.51）／Table 6（p.20） | Recall@1｜顯著性（BH）：\* | n.s.（BH p = 0.0947）（顯著性結論改變；TORS p 欄印為 < .05。） |
| T4-16.r10.dw | 表4-16（p.54） | Recall@10｜ΔW：-0.0022 | +0.0001（方向改變） |
| T4-16.r10.do | 表4-16（p.54） | Recall@10｜ΔO：-0.0008 | +0.0037（方向改變） |
| T4-16.r30.do | 表4-16（p.54） | Recall@30｜ΔO：-0.0014 | +0.0017（方向改變） |
| T4-16.r50.dw | 表4-16（p.54） | Recall@50｜ΔW：-0.0026 | +0.0052（方向改變） |
| T4-16.r50.do | 表4-16（p.54） | Recall@50｜ΔO：-0.0004 | +0.0041（方向改變） |
| X02 | 4.3.1（p.50） | AUC 的效果量特別大（dz 38.934），因五組配對差異極為穩定 | AUC dz = 4.103；八項中最大為 R@50 dz = 7.851 |
| X05 | 4.3.2（p.51）／5.2（p.20） | 各項平均差異的 95% CI 皆未包含 0，且經 BH 校正後仍達統計顯著／All 95% CIs exclude zero, and all improvements remain significant after Benjamini–Hochberg adjustment | BH 校正後 7／8 顯著；CI 含 0：R@1 [-0.00035, +0.00293]；未顯著：R@1（BH p = 0.0947） |
| X06 | 4.3.2（p.51） | Recall@10 與 Recall@50 由未校正的 p < .001 調整為 p < .01 | R@10：未校正 p = 0.000433（\*\*\*）、BH p = 0.000865（\*\*\*）；R@50：未校正 p = 6.18e-05（\*\*\*）、BH p = 0.000419（\*\*\*） |
| X08 | 4.3.2（p.51） | Recall@1 與 Recall@3 的信賴區間下限較接近 0 | R@1 CI 下界 -0.00035（含 0）；R@3 CI 下界 +0.00213 |
| X13b | 4.4.1（p.52） | Recall@50 的顯著性標記由校正前的 \*\*\* 調整為 \*\* | R@50：未校正 p = 0.0089（\*\*）、BH p = 0.0111（\*） |
| X17 | 4.4.2（p.53） | 移除天氣後各指標分別下降 0.0022、0.0030、0.0026；移除場合後分別下降 0.0008、0.0014、0.0004 | No-W：R@10 +0.0001、R@30 -0.0001、R@50 +0.0052；No-O：R@10 +0.0037、R@30 +0.0017、R@50 +0.0041（正值表示移除後 Recall 反而上升） |
| X19 | 4.4.2（p.54）；圖4-7 | 風格 71%–89%、天氣 10%–16%、場合 1%–14%；風格在 OR 的占比隨 K 增加而提高 | CP AUC：風格 68%、天氣 8%、場合 24%；CP FITB Acc：風格 70%、天氣 17%、場合 13%；CIR Recall@10：無法定義（天氣或場合移除後效能上升）；CIR Recall@30：無法定義（天氣或場合移除後效能上升）；CIR Recall@50：無法定義（天氣或場合移除後效能上升） |
| X20 | 4.4.2（p.54） | 移除三項因子後效能皆下降，其中以風格資訊的影響最大 | 10／15 個 Δ 為負；移除後上升：No-W R@10 +0.0001、No-W R@50 +0.0052、No-O R@10 +0.0037、No-O R@30 +0.0017、No-O R@50 +0.0041 |
| X21 | 5.3（p.20–21） | removing Weather or Occasion results in smaller changes ... Weather and Occasion provide smaller complementary benefits | No-W：R@10 +0.0001、R@30 -0.0001、R@50 +0.0052；No-O：R@10 +0.0037、R@30 +0.0017、R@50 +0.0041（正值表示移除後 Recall 反而上升） |
| C03 | 5.1（p.65） | 穿搭相容性預測與套裝推薦表現皆顯著優於使用原始文字的設定 | BH 校正後 7／8 顯著；未顯著：R@1（BH p = 0.0947） |
| C04 | 4.7（p.64）；5.1（p.65） | 移除天氣、場合與風格資訊後，模型效能皆出現不同程度的下降 | 10／15 個 Δ 為負；移除後上升：No-W R@10 +0.0001、No-W R@50 +0.0052、No-O R@10 +0.0037、No-O R@30 +0.0017、No-O R@50 +0.0041 |
| C06 | 4.7（p.64）；5.1（p.65）／6.1 RQ2（p.23） | 天氣與場合主要補充外在環境與使用情境限制／Weather and Occasion show smaller effects but provide complementary constraints | No-W：R@10 +0.0001、R@30 -0.0001、R@50 +0.0052；No-O：R@10 +0.0037、R@30 +0.0017、R@50 +0.0041（正值表示移除後 Recall 反而上升） |

## 需補寫

- `S13` 公平比較子集 train／validation／test：10,225／1,748／9,930
- `S16` 隨機種子：1, 2, 3, 4, 5
- `S23` Optimizer：Adam
- `S24` Weight decay：0.0
- `S25` Dropout：0.1
- `S26` Attention heads：16
- `S27` Transformer layers：3
- `S28` Gradient clipping norm：0.5
- `S29` Mixed precision：float16
- `S30` Learning-rate scheduler：StepLR（gamma 0.5，每 10 個 epoch 呼叫一次）
- `S31` Checkpoint 選擇指標：validation_fitb_accuracy
- `S32` OR／CIR ranking loss margin：0.3
- `S33` Decoder 實作：torch.nn.TransformerDecoderLayer（3 層、16 heads、dropout 0.1）；歷史 DecoderLayerWithCrossAttn 定義未保存
- `S35` 軟體環境：Python 3.12.3；PyTorch 2.9.1+cu130；torchvision 0.24.1；CUDA 13.0；cuDNN 9.13.0
- `S37` 公平比較子集的 BH 校正族群：Benjamini-Hochberg correction across the five CP+OR core metrics, matching archived published table convention

## 一、Abstract 主要結果

| ID | 論文 | TORS | 項目 | 稿件 | 最新 | 狀態 | 來源（程式） |
|---|---|---|---|---|---|---|---|
| A01 | 摘要（p.i）；Abstract（p.ii）；5.1（p.65） | Abstract（p.1） | CP AUC｜Original | 0.9292 | 0.9268 | 更新數字 | `<RUN>/main/main_reproduction_summary.csv`（`compare_main_results.py`） |
| A02 | 摘要（p.i）；Abstract（p.ii）；5.1（p.65） | Abstract（p.1） | CP AUC｜Context-aware | 0.9454 | 0.9456 | 更新數字 | `<RUN>/main/main_reproduction_summary.csv`（`compare_main_results.py`） |
| A03 | 摘要（p.i）；Abstract（p.ii）；5.1（p.65） | Abstract（p.1） | FITB Accuracy｜Original | 0.6374 | 0.6357 | 更新數字 | `<RUN>/main/main_reproduction_summary.csv`（`compare_main_results.py`） |
| A04 | 摘要（p.i）；Abstract（p.ii）；5.1（p.65） | Abstract（p.1） | FITB Accuracy｜Context-aware | 0.6478 | 0.6481 | 更新數字 | `<RUN>/main/main_reproduction_summary.csv`（`compare_main_results.py`） |
| A05 | 摘要（p.i）；Abstract（p.ii）；5.1（p.65） | Abstract（p.1） | Recall@10｜Original | 0.0780 | 0.0782 | 更新數字 | `<RUN>/main/main_reproduction_summary.csv`（`compare_main_results.py`） |
| A06 | 摘要（p.i）；Abstract（p.ii）；5.1（p.65） | Abstract（p.1） | Recall@10｜Context-aware | 0.0921 | 0.0927 | 更新數字 | `<RUN>/main/main_reproduction_summary.csv`（`compare_main_results.py`） |
| A07 | 摘要（p.i）；Abstract（p.ii）；5.1（p.65） | Abstract（p.1） | Recall@50｜Original | 0.2303 | 0.2303 | 保留 | `<RUN>/main/main_reproduction_summary.csv`（`compare_main_results.py`） |
| A08 | 摘要（p.i）；Abstract（p.ii）；5.1（p.65） | Abstract（p.1） | Recall@50｜Context-aware | 0.2518 | 0.2532 | 更新數字 | `<RUN>/main/main_reproduction_summary.csv`（`compare_main_results.py`） |
| A09 | 摘要（p.i）；Abstract（p.ii） | — | 移除風格後 CP AUC 下降幅度（縮減子集） | 0.0154 | 0.0135（TORS 摘要未寫此數值；對應 Table 7 的 Full–No-S。） | 更新數字 | `<RUN>/ablation/fresh_5variant_8metric_summary.csv`（`summarize_fair_subset_5seed.py`） |
| A10 | 摘要（p.i）；Abstract（p.ii） | — | 移除風格後 Recall@50 下降幅度（縮減子集） | 0.0241 | 0.0264（TORS 摘要未寫此數值；對應 Table 7 的 Full–No-S。） | 更新數字 | `<RUN>/ablation/fresh_5variant_8metric_summary.csv`（`summarize_fair_subset_5seed.py`） |
| A11 | 摘要（p.i）；Abstract（p.ii） | Abstract（p.1） | 風格是推薦效能的主要因素 | 風格條件是支撐語意對齊與排序品質的主要因素／style contributes most consistently | ΔS：AUC -0.0135、FITB -0.0126、R@10 -0.0106、R@30 -0.0233、R@50 -0.0264（CP 與 OR 皆為三因子中最大下降） | 保留 | `<RUN>/ablation/fresh_5variant_8metric_summary.csv`（`summarize_fair_subset_5seed.py`） |
| A12 | 摘要（p.i）；Abstract（p.ii） | Abstract（p.1） | 天氣與場合提供額外情境限制 | 天氣與場合資訊能提供額外的情境限制／weather and occasion add complementary constraints | No-W：R@10 +0.0001、R@30 -0.0001、R@50 +0.0052；No-O：R@10 +0.0037、R@30 +0.0017、R@50 +0.0041（正值表示移除後 Recall 反而上升）（第九節：不得將 Weather 與 Occasion 寫成在所有 OR 指標都具有穩定正貢獻。） | 改寫敘述 | `<RUN>/ablation/fresh_5variant_8metric_summary.csv`（`summarize_fair_subset_5seed.py`） |
| A13 | — | Abstract（p.1） | Two-Tower 驗證保留檢索增益 | Simpler Two-Tower validation preserves retrieval gains | AUC +0.0378、FITB -0.0077、R@10 +0.0102、R@30 +0.0153、R@50 +0.0170（保存輸出重算）（第二模型不在 35 組正式訓練內；數值由學姊保存的逐 seed 輸出重算。） | 非正式 run | `<RUN>/secondary/two_tower/recomputed_key_mean_std.csv`（`reproduce_remaining_thesis.py`） |
| A14 | — | Abstract（p.1） | 反事實替換改變排序 | controlled counterfactual replacements alter rankings observably | —（來源為 03_實驗與結果_experiments_results/06_反事實情境敏感度/（seed 1、歷史 checkpoint）；正式 run 未重做。） | 非正式 run | — |

## 二、Experimental setup 與模型設定

| ID | 論文 | TORS | 項目 | 稿件 | 最新 | 狀態 | 來源（程式） |
|---|---|---|---|---|---|---|---|
| S01 | 4.1.1（p.40）；5.1（p.65） | Abstract（p.1）；Table 1（p.7） | 有效套裝樣本總數 | 35,140 | 35,140 | 保留 | `splits/train_ids.csv`、`splits/validation_ids.csv`、`splits/test_ids.csv`（`freeze_splits.py`） |
| S02 | 4.1.1（p.40） | Table 1（p.7） | 訓練集套裝數 | 16,995 | 16,995 | 保留 | `splits/train_ids.csv`（`freeze_splits.py`） |
| S03 | 4.1.1（p.40） | Table 1（p.7） | 驗證集套裝數 | 3,000 | 3,000 | 保留 | `splits/validation_ids.csv`（`freeze_splits.py`） |
| S04 | 4.1.1（p.40） | Table 1（p.7） | 測試集套裝數 | 15,145 | 15,145 | 保留 | `splits/test_ids.csv`（`freeze_splits.py`） |
| S05 | 4.1.1（p.40） | Table 1（p.7） | CP 測試樣本數 | 30,290 | 30,290 | 保留 | `splits/cp_ids.csv`（`freeze_splits.py`） |
| S06 | 4.1.1（p.40） | Table 1（p.7） | FITB 題數 | 15,145 | 15,145 | 保留 | `splits/fitb_ids.csv`（`freeze_splits.py`） |
| S07 | 4.1.1（p.40） | Table 1（p.7） | OR／CIR 可評估查詢配對 | 9,311 | 9,311 | 保留 | `splits/or_ids.csv`（`freeze_splits.py`） |
| S08 | 4.1.1（p.40）；4.4.1（p.52） | — | 公平比較子集樣本數 | 21,903 | 21,903 | 保留 | `splits/fair_subset/fair_subset_reconstruction_manifest.json`（`reconstruct_fair_subset_from_wos.py`） |
| S09 | 4.4.1（p.52） | — | 公平比較子集占有效樣本比例 | 62.33% | 62.33% | 保留 | `splits/fair_subset/fair_subset_reconstruction_manifest.json`（`reconstruct_fair_subset_from_wos.py`） |
| S10 | 4.4.1（p.52） | — | 天氣因子覆蓋率 | 100.0% | 100.0% | 保留 | `splits/fair_subset/fair_subset_reconstruction_manifest.json`（`reconstruct_fair_subset_from_wos.py`） |
| S11 | 4.4.1（p.52） | — | 風格因子覆蓋率 | 99.87% | 99.87% | 保留 | `splits/fair_subset/fair_subset_reconstruction_manifest.json`（`reconstruct_fair_subset_from_wos.py`） |
| S12 | 4.4.1（p.52） | — | 場合因子覆蓋率 | 62.45% | 62.45% | 保留 | `splits/fair_subset/fair_subset_reconstruction_manifest.json`（`reconstruct_fair_subset_from_wos.py`） |
| S13 | — | — | 公平比較子集 train／validation／test | （未記載） | 10,225／1,748／9,930（第六項：修訂稿與 artifact 應公開此切分與 ID。） | 需補寫 | `splits/fair_subset/fair_subset_reconstruction_manifest.json`（`reconstruct_fair_subset_from_wos.py`） |
| S14 | 4.5（p.54） | 5.5（p.22） | 公平比較子集 OR 可評估查詢配對 | 3,432 | 3,432 | 保留 | `splits/fair_subset/fair_subset_cir_scope_audit.json`（`audit_fair_subset_cir_scope.py`） |
| S15 | 4.5（p.54） | 5.5（p.22） | 公平比較子集種子層級觀測值 | 17,160 | 17,160（= 可評估查詢配對 × seeds。） | 保留 | `splits/fair_subset/fair_subset_cir_scope_audit.json`、`<RUN>/ablation/fresh_25unit_metrics.csv`（`audit_fair_subset_cir_scope.py`、`summarize_fair_subset_5seed.py`） |
| S16 | 3.6（p.33） | 4.1（p.15） | 隨機種子 | 五組固定隨機種子（未列出數值） | 1, 2, 3, 4, 5（修訂稿應寫出 seed 值。） | 需補寫 | `<RUN>/statistics/main_paired_bh_8metrics.csv`（`complete_statistics.py`） |
| S17 | 表4-2（p.41） | 4.3（p.16） | Epoch | 100 | 100 | 保留 | `configs/cp.yaml`（`run_full_single_seed_training.py`、`base_config.py`） |
| S18 | 表4-2（p.41） | 4.3（p.16） | Batch size | 50 | 50 | 保留 | `configs/cp.yaml`（`run_full_single_seed_training.py`、`base_config.py`） |
| S19 | 表4-2（p.41） | 4.3（p.16） | Learning rate | 5e-5 | 5e-05（論文印為 5 × 10⁻⁵。） | 保留 | `configs/cp.yaml`（`run_full_single_seed_training.py`、`base_config.py`） |
| S20 | 表4-2（p.41） | 4.3（p.16） | 硬負樣本啟用 epoch | 40 | 40（論文寫「第 40 個 epoch 後啟用」、TORS 寫 starts after epoch 40；設定為 epoch 40（含）起啟用，修稿時統一措辭。） | 保留 | `configs/cp.yaml`（`run_full_single_seed_training.py`、`base_config.py`） |
| S21 | 表4-2（p.41）；3.6.1（p.33） | 4.3（p.16） | 推薦模型與特徵 | Hybrid Attention + FashionCLIP | OutfitTransformerPrediction（Hybrid Attention）；影像特徵 FashionCLIP | 保留 | `configs/cp.yaml`（`run_full_single_seed_training.py`、`base_config.py`） |
| S22 | 4.1.3（p.41） | 4.3（p.16） | OR／CIR 以 CP 權重初始化 | OR 以 CP 訓練完成後的權重初始化 | matching_CP_checkpoint：以同 seed 的 CP best checkpoint（validation FITB 選出）初始化 | 保留 | `configs/or.yaml`（`run_full_single_seed_training.py`） |
| S23 | — | — | Optimizer | （未記載） | Adam | 需補寫 | `configs/cp.yaml`（`run_full_single_seed_training.py`、`base_config.py`） |
| S24 | — | — | Weight decay | （未記載） | 0.0 | 需補寫 | `configs/cp.yaml`（`run_full_single_seed_training.py`、`base_config.py`） |
| S25 | — | — | Dropout | （未記載） | 0.1 | 需補寫 | `configs/cp.yaml`（`run_full_single_seed_training.py`、`base_config.py`） |
| S26 | — | — | Attention heads | （未記載） | 16 | 需補寫 | `configs/cp.yaml`（`run_full_single_seed_training.py`、`base_config.py`） |
| S27 | — | — | Transformer layers | （未記載） | 3 | 需補寫 | `configs/cp.yaml`（`run_full_single_seed_training.py`、`base_config.py`） |
| S28 | — | — | Gradient clipping norm | （未記載） | 0.5 | 需補寫 | `configs/cp.yaml`（`run_full_single_seed_training.py`、`base_config.py`） |
| S29 | — | — | Mixed precision | （未記載） | float16 | 需補寫 | `configs/cp.yaml`（`run_full_single_seed_training.py`、`base_config.py`） |
| S30 | — | — | Learning-rate scheduler | （未記載） | StepLR（gamma 0.5，每 10 個 epoch 呼叫一次） | 需補寫 | `configs/cp.yaml`（`run_full_single_seed_training.py`、`base_config.py`） |
| S31 | — | — | Checkpoint 選擇指標 | （未記載） | validation_fitb_accuracy | 需補寫 | `configs/cp.yaml`（`run_full_single_seed_training.py`、`base_config.py`） |
| S32 | — | — | OR／CIR ranking loss margin | （未記載） | 0.3（MarginRankingLoss。） | 需補寫 | `configs/or.yaml`（`run_full_single_seed_training.py`、`base_config.py`） |
| S33 | 3.6.1（p.33） | 4.3（p.16） | Decoder 實作 | （未記載；稿件稱沿用原模型架構） | torch.nn.TransformerDecoderLayer（3 層、16 heads、dropout 0.1）；歷史 DecoderLayerWithCrossAttn 定義未保存（第五項：修訂稿需說明全部實驗採標準化 decoder；摘要與 3.6.1「沿用原模型、不修改架構」的敘述需一併加註。） | 需補寫 | `configs/cp.yaml`（`run_full_single_seed_training.py`） |
| S34 | 表4-1（p.41） | 4.3（p.16） | 模型訓練 GPU | NVIDIA RTX PRO 6000 Blackwell Workstation Edition（96 GB） | NVIDIA GB10（採用最新結果時，實驗環境描述必須改成正式 run 的硬體。） | 改寫敘述 | `environment/full_20261002T161428Z/environment_validation.json`（`validate_reproduction_env.py`） |
| S35 | — | — | 軟體環境 | （未記載） | Python 3.12.3；PyTorch 2.9.1+cu130；torchvision 0.24.1；CUDA 13.0；cuDNN 9.13.0 | 需補寫 | `environment/full_20261002T161428Z/environment_validation.json`（`validate_reproduction_env.py`） |
| S36 | 3.6（p.33）；4.3（p.49–50） | 4.1（p.15） | 統計方法 | paired t-test、95% CI、Cohen's dz、8 項指標共同 BH 校正 | two-sided paired t-test on context-original per seed, df=4；95% two-sided t confidence interval on mean paired difference；BH family = 8 項主要指標；α = 0.05；Cohen's dz 由 main_paired_bh_8metrics.csv 的 cohen_dz 欄輸出 | 保留 | `<RUN>/statistics/statistical_evidence.json`、`<RUN>/statistics/main_paired_bh_8metrics.csv`（`complete_statistics.py`） |
| S37 | 4.4.1（p.52） | — | 公平比較子集的 BH 校正族群 | 未說明（表 4-13／4-14 標示為 BH 校正後顯著） | Benjamini-Hochberg correction across the five CP+OR core metrics, matching archived published table convention（修訂稿應寫明縮減子集以表列 5 項指標為同一 BH 族群（沿用學姊原表 T03／T04 的規則）。） | 需補寫 | `<RUN>/ablation/fresh_fair_subset_5seed_summary.json`（`summarize_fair_subset_5seed.py`） |
| S38 | — | 4.5.1（p.17） | Two-Tower 訓練設定 | 100 epochs, batch 256, lr 1e-3, AdamW | —（第二模型不在 35 組正式訓練內；來源 03_實驗與結果_experiments_results/05_第二模型驗證/。） | 非正式 run | — |

## 三、表 4-11 至表 4-16 全部欄位

格式：`稿件 → 最新`；✓ 四捨五入後相同；⚠ 方向、CI 是否含 0 或顯著性改變。

### 表 4-11（TORS Table 6）

| 指標 | Original 平均 | Original SD | Context 平均 | Context SD | Δ | 95% CI 下界 | 95% CI 上界 | 顯著性（BH） | Cohen's dz |
|---|---|---|---|---|---|---|---|---|---|
| AUC | 0.9292 → 0.9268 | 0.0013 → 0.0040 | 0.9454 → 0.9456 | 0.0013 → 0.0010 | +0.0162 → +0.0188 | +0.01569 → +0.01313 | +0.01672 → +0.02453 | \*\*\* → \*\* | 38.934 → 4.103 |
| FITB Acc | 0.6374 → 0.6357 | 0.0037 → 0.0035 | 0.6478 → 0.6481 | 0.0021 → 0.0029 | +0.0103 → +0.0125 | +0.00536 → +0.00594 | +0.01532 → +0.01902 | \*\* ✓ | 2.578 → 2.368 |

來源：`<RUN>/main/main_reproduction_summary.csv`（`compare_main_results.py`）；`<RUN>/statistics/main_paired_bh_8metrics.csv`（`complete_statistics.py`）
- TORS 印為 0.0157。（AUC｜95% CI 下界）
- TORS 印為 0.0167。（AUC｜95% CI 上界）
- TORS p 欄印為 < .001。（AUC｜顯著性（BH））
- TORS 印為 0.0054。（FITB Acc｜95% CI 下界）
- TORS 印為 0.0153。（FITB Acc｜95% CI 上界）
- TORS p 欄印為 < .01。（FITB Acc｜顯著性（BH））

### 表 4-12（TORS Table 6）

| 指標 | Original 平均 | Original SD | Context 平均 | Context SD | Δ | 95% CI 下界 | 95% CI 上界 | 顯著性（BH） | Cohen's dz |
|---|---|---|---|---|---|---|---|---|---|
| Recall@1 | 0.0139 → 0.0138 | 0.0004 → 0.0005 | 0.0151 ✓ | 0.0011 → 0.0016 | +0.0012 → +0.0013 | **+0.00005 → -0.00035 ⚠** | +0.00236 → +0.00293 | **\* → n.s. ⚠** | 1.292 → 0.975 |
| Recall@3 | 0.0333 ✓ | 0.0010 → 0.0011 | 0.0369 → 0.0372 | 0.0020 → 0.0014 | +0.0036 → +0.0038 | +0.00003 → +0.00213 | +0.00714 → +0.00552 | \* → \*\* | 1.253 → 2.799 |
| Recall@5 | 0.0476 → 0.0481 | 0.0010 → 0.0008 | 0.0557 → 0.0560 | 0.0023 → 0.0012 | +0.0082 → +0.0079 | +0.00443 → +0.00630 | +0.01189 → +0.00946 | \*\* → \*\*\* | 2.719 → 6.197 |
| Recall@10 | 0.0780 → 0.0782 | 0.0010 → 0.0012 | 0.0921 → 0.0927 | 0.0035 → 0.0020 | +0.0141 → +0.0145 | +0.01006 → +0.01075 | +0.01816 → +0.01829 | \*\* → \*\*\* | 4.326 → 4.784 |
| Recall@30 | 0.1677 → 0.1684 | 0.0045 → 0.0037 | 0.1874 → 0.1885 | 0.0038 → 0.0028 | +0.0197 → +0.0201 | +0.01527 → +0.01611 | +0.02412 → +0.02414 | \*\*\* ✓ | 5.526 → 6.223 |
| Recall@50 | 0.2303 ✓ | 0.0035 → 0.0032 | 0.2518 → 0.2532 | 0.0038 → 0.0023 | +0.0215 → +0.0229 | +0.01606 → +0.01929 | +0.02686 → +0.02654 | \*\* → \*\*\* | 4.934 → 7.851 |

來源：`<RUN>/main/main_reproduction_summary.csv`（`compare_main_results.py`）；`<RUN>/statistics/main_paired_bh_8metrics.csv`（`complete_statistics.py`）
- TORS p 欄印為 < .05。（Recall@1｜顯著性（BH）、Recall@3｜顯著性（BH））
- TORS p 欄印為 < .01。（Recall@5｜顯著性（BH）、Recall@10｜顯著性（BH）、Recall@50｜顯著性（BH））
- TORS p 欄印為 < .001。（Recall@30｜顯著性（BH））

### 表 4-13（TORS Table 7）

| 指標 | Original 平均 | Original SD | Context 平均 | Context SD | Δ | 95% CI 下界 | 95% CI 上界 | 顯著性（BH） | Cohen's dz |
|---|---|---|---|---|---|---|---|---|---|
| AUC | 0.9112 → 0.9121 | 0.0015 → 0.0018 | 0.9265 → 0.9271 | 0.0015 ✓ | +0.0153 → +0.0150 | +0.01308 → +0.01426 | +0.01750 → +0.01576 | \*\*\* ✓ | 8.601 → 24.874 |
| FITB Acc | 0.6109 → 0.6121 | 0.0035 → 0.0019 | 0.6228 → 0.6226 | 0.0034 → 0.0031 | +0.0119 → +0.0106 | +0.00474 → +0.00671 | +0.01915 → +0.01443 | \* → \*\* | 2.059 → 3.401 |

來源：`<RUN>/ablation/fresh_5variant_8metric_summary.csv`（`summarize_fair_subset_5seed.py`）；`<RUN>/ablation/T03_stage1_cp_original_vs_full.csv`（`summarize_fair_subset_5seed.py`）
- TORS 欄名 Full–Original。（AUC｜Δ、FITB Acc｜Δ）

### 表 4-14（TORS Table 7）

| 指標 | Original 平均 | Original SD | Context 平均 | Context SD | Δ | 95% CI 下界 | 95% CI 上界 | 顯著性（BH） | Cohen's dz |
|---|---|---|---|---|---|---|---|---|---|
| Recall@10 | 0.0686 → 0.0685 | 0.0043 → 0.0052 | 0.0781 → 0.0753 | 0.0023 → 0.0043 | +0.0096 → +0.0068 | +0.00153 → +0.00016 | +0.01758 → +0.01348 | \* ✓ | 1.479 → 1.271 |
| Recall@30 | 0.1517 → 0.1512 | 0.0077 → 0.0042 | 0.1720 → 0.1693 | 0.0034 → 0.0046 | +0.0203 → +0.0182 | +0.01069 → +0.01234 | +0.02999 → +0.02402 | \*\* ✓ | 2.617 → 3.864 |
| Recall@50 | 0.2111 → 0.2143 | 0.0046 → 0.0043 | 0.2351 → 0.2308 | 0.0045 → 0.0066 | +0.0241 → +0.0165 | +0.01781 → +0.00689 | +0.03033 → +0.02610 | \*\* → \* | 4.774 → 2.132 |

來源：`<RUN>/ablation/fresh_5variant_8metric_summary.csv`（`summarize_fair_subset_5seed.py`）；`<RUN>/ablation/T04_stage1_cir_original_vs_full.csv`（`summarize_fair_subset_5seed.py`）
- TORS 欄名 Full–Original。（Recall@10｜Δ、Recall@30｜Δ、Recall@50｜Δ）

### 表 4-15（TORS Table 7）

| 指標 | Context 平均 | Context SD | No-W 平均 | No-W SD | ΔW | No-O 平均 | No-O SD | ΔO | No-S 平均 | No-S SD | ΔS |
|---|---|---|---|---|---|---|---|---|---|---|---|
| AUC | 0.9265 → 0.9271 | 0.0015 ✓ | 0.9241 → 0.9256 | 0.0034 → 0.0007 | -0.0023 → -0.0015 | 0.9236 → 0.9223 | 0.0015 → 0.0038 | -0.0029 → -0.0048 | 0.9110 → 0.9136 | 0.0041 → 0.0014 | -0.0154 → -0.0135 |
| FITB Acc | 0.6228 → 0.6226 | 0.0034 → 0.0031 | 0.6191 → 0.6196 | 0.0060 → 0.0024 | -0.0037 → -0.0031 | 0.6197 → 0.6203 | 0.0024 → 0.0029 | -0.0031 → -0.0024 | 0.6064 → 0.6101 | 0.0051 → 0.0044 | -0.0164 → -0.0126 |

來源：`<RUN>/ablation/fresh_5variant_8metric_summary.csv`（`summarize_fair_subset_5seed.py`）
- TORS 以 Full–No-S 表示（正值，與 ΔS 反號）。（AUC｜ΔS、FITB Acc｜ΔS）

### 表 4-16（TORS Table 7）

| 指標 | Context 平均 | Context SD | No-W 平均 | No-W SD | ΔW | No-O 平均 | No-O SD | ΔO | No-S 平均 | No-S SD | ΔS |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Recall@10 | 0.0781 → 0.0753 | 0.0023 → 0.0043 | 0.0760 → 0.0754 | 0.0013 → 0.0034 | **-0.0022 → +0.0001 ⚠** | 0.0774 → 0.0790 | 0.0009 → 0.0012 | **-0.0008 → +0.0037 ⚠** | 0.0671 → 0.0647 | 0.0035 → 0.0057 | -0.0110 → -0.0106 |
| Recall@30 | 0.1720 → 0.1693 | 0.0034 → 0.0046 | 0.1690 → 0.1692 | 0.0047 → 0.0063 | -0.0030 → -0.0001 | 0.1706 → 0.1710 | 0.0030 → 0.0041 | **-0.0014 → +0.0017 ⚠** | 0.1524 → 0.1461 | 0.0042 → 0.0081 | -0.0196 → -0.0233 |
| Recall@50 | 0.2351 → 0.2308 | 0.0045 → 0.0066 | 0.2325 → 0.2360 | 0.0068 → 0.0042 | **-0.0026 → +0.0052 ⚠** | 0.2347 → 0.2350 | 0.0014 → 0.0030 | **-0.0004 → +0.0041 ⚠** | 0.2110 → 0.2044 | 0.0014 → 0.0106 | -0.0241 → -0.0264 |

來源：`<RUN>/ablation/fresh_5variant_8metric_summary.csv`（`summarize_fair_subset_5seed.py`）
- TORS 以 Full–No-S 表示（正值，與 ΔS 反號）。（Recall@10｜ΔS、Recall@30｜ΔS、Recall@50｜ΔS）

## 四、Results 內文的 p 值、BH 校正、效果量與信賴區間

| ID | 論文 | TORS | 項目 | 稿件 | 最新 | 狀態 | 來源（程式） |
|---|---|---|---|---|---|---|---|
| X01 | 4.3.1（p.50） | 5.2（p.20） | CP 兩項指標 CI 未含 0 且 BH 後顯著 | 兩項指標的信賴區間皆未包含 0，經 BH 校正後仍達統計顯著 | AUC CI [+0.01313, +0.02453]、BH p = 0.00125；FITB CI [+0.00594, +0.01902]、BH p = 0.00698 | 保留 | `<RUN>/statistics/main_paired_bh_8metrics.csv`（`complete_statistics.py`） |
| X02 | 4.3.1（p.50） | — | AUC 效果量特別大 | AUC 的效果量特別大（dz 38.934），因五組配對差異極為穩定 | AUC dz = 4.103；八項中最大為 R@50 dz = 7.851 | 改寫敘述 | `<RUN>/statistics/main_paired_bh_8metrics.csv`（`complete_statistics.py`） |
| X03 | 4.3.1（p.50）；圖4-5 | — | AUC 相對提升幅度 | 1.7% | 2.0% | 更新數字 | `<RUN>/statistics/main_paired_bh_8metrics.csv`（`complete_statistics.py`） |
| X04 | 4.3.1（p.50）；圖4-5 | — | FITB Accuracy 相對提升幅度 | 1.6% | 2.0% | 更新數字 | `<RUN>/statistics/main_paired_bh_8metrics.csv`（`complete_statistics.py`） |
| X05 | 4.3.2（p.51） | 5.2（p.20） | 全部指標 CI 未含 0 且 BH 後顯著 | 各項平均差異的 95% CI 皆未包含 0，且經 BH 校正後仍達統計顯著／All 95% CIs exclude zero, and all improvements remain significant after Benjamini–Hochberg adjustment | BH 校正後 7／8 顯著；CI 含 0：R@1 [-0.00035, +0.00293]；未顯著：R@1（BH p = 0.0947） | 改寫敘述 | `<RUN>/statistics/main_paired_bh_8metrics.csv`（`complete_statistics.py`） |
| X06 | 4.3.2（p.51） | — | R@10 與 R@50 顯著性層級因 BH 校正下降 | Recall@10 與 Recall@50 由未校正的 p < .001 調整為 p < .01 | R@10：未校正 p = 0.000433（\*\*\*）、BH p = 0.000865（\*\*\*）；R@50：未校正 p = 6.18e-05（\*\*\*）、BH p = 0.000419（\*\*\*） | 改寫敘述 | `<RUN>/statistics/main_paired_bh_8metrics.csv`（`complete_statistics.py`） |
| X07 | 4.3.2（p.51） | — | OR 各指標 Cohen's dz 範圍 | 1.253–5.526 | 0.975–7.851 | 更新數字 | `<RUN>/statistics/main_paired_bh_8metrics.csv`（`complete_statistics.py`） |
| X08 | 4.3.2（p.51） | — | R@1 與 R@3 的 CI 下限接近 0 但仍為正 | Recall@1 與 Recall@3 的信賴區間下限較接近 0 | R@1 CI 下界 -0.00035（含 0）；R@3 CI 下界 +0.00213 | 改寫敘述 | `<RUN>/statistics/main_paired_bh_8metrics.csv`（`complete_statistics.py`） |
| X09 | 4.3.2（p.51） | — | 較大 K 的改善較穩定 | Recall@10／30／50 的信賴區間距離 0 較遠、效果量較大 | R@10／30／50 dz = 4.784／6.223／7.851；R@1／3 dz = 0.975／2.799 | 保留 | `<RUN>/statistics/main_paired_bh_8metrics.csv`（`complete_statistics.py`） |
| X10 | 4.3.2（p.51）；圖4-6 | — | OR 相對提升幅度範圍 | 8.6%–18.1% | 9.3%–18.6% | 更新數字 | `<RUN>/statistics/main_paired_bh_8metrics.csv`（`complete_statistics.py`） |
| X11 | 4.3.2（p.51）；圖4-6 | — | R@5 與 R@10 相對提升最大 | Recall@5 與 Recall@10 的相對提升幅度最大 | 相對提升最大：R@10 18.6%、R@5 16.4% | 保留 | `<RUN>/statistics/main_paired_bh_8metrics.csv`（`complete_statistics.py`） |
| X12 | 4.4.1（p.52） | — | 縮減子集 CP 經 BH 校正後顯著 | 兩項指標經 BH 校正後仍達統計顯著 | AUC CI [+0.01426, +0.01576]、BH p = < .001（\*\*\*）；FITB CI [+0.00671, +0.01443]、BH p = 0.0027（\*\*） | 保留 | `<RUN>/ablation/T03_stage1_cp_original_vs_full.csv`（`summarize_fair_subset_5seed.py`） |
| X13 | 4.4.1（p.52） | — | 縮減子集 OR 的 CI 與 BH 顯著性 | 三項指標 95% CI 均未包含 0，經校正後仍達統計顯著 | R@10 CI [+0.00016, +0.01348]、BH p = 0.0467（\*）；R@30 CI [+0.01234, +0.02402]、BH p = 0.0025（\*\*）；R@50 CI [+0.00689, +0.02610]、BH p = 0.0111（\*） | 保留 | `<RUN>/ablation/T04_stage1_cir_original_vs_full.csv`（`summarize_fair_subset_5seed.py`） |
| X13b | 4.4.1（p.52） | — | 縮減子集 R@50 顯著性層級因 BH 校正下降 | Recall@50 的顯著性標記由校正前的 \*\*\* 調整為 \*\* | R@50：未校正 p = 0.0089（\*\*）、BH p = 0.0111（\*） | 改寫敘述 | `<RUN>/ablation/T04_stage1_cir_original_vs_full.csv`（`summarize_fair_subset_5seed.py`） |
| X14 | 4.4.1（p.52） | 5.3（p.21） | 縮減子集上 Context 全面優於 Original | 情境感知改寫描述在一致的樣本集合下仍維持正向效益／Full remains superior to Original on all five fair-subset metrics | 5／5 項為正（AUC +0.0150、FITB +0.0106、R@10 +0.0068、R@30 +0.0182、R@50 +0.0165） | 保留 | `<RUN>/ablation/fresh_5variant_8metric_summary.csv`（`summarize_fair_subset_5seed.py`） |
| X15 | 4.4.2（p.53） | — | CP：移除天氣與場合亦下降但幅度較小 | 天氣與場合資訊移除後亦造成效能下降，但幅度相對較小 | AUC：ΔW -0.0015、ΔO -0.0048、ΔS -0.0135；FITB：ΔW -0.0031、ΔO -0.0024、ΔS -0.0126 | 保留 | `<RUN>/ablation/fresh_5variant_8metric_summary.csv`（`summarize_fair_subset_5seed.py`） |
| X16 | 4.4.2（p.53） | 5.3（p.20） | OR：移除風格的下降最大 | 移除風格資訊後的下降皆明顯高於移除天氣或場合 | ΔS：R@10 -0.0106、R@30 -0.0233、R@50 -0.0264（皆為三因子中最大下降） | 保留 | `<RUN>/ablation/fresh_5variant_8metric_summary.csv`（`summarize_fair_subset_5seed.py`） |
| X17 | 4.4.2（p.53） | — | OR：移除天氣或場合後各指標皆下降 | 移除天氣後各指標分別下降 0.0022、0.0030、0.0026；移除場合後分別下降 0.0008、0.0014、0.0004 | No-W：R@10 +0.0001、R@30 -0.0001、R@50 +0.0052；No-O：R@10 +0.0037、R@30 +0.0017、R@50 +0.0041（正值表示移除後 Recall 反而上升） | 改寫敘述 | `<RUN>/ablation/fresh_5variant_8metric_summary.csv`（`summarize_fair_subset_5seed.py`） |
| X18 | 4.4.2（p.53） | — | OR：移除風格的下降隨 K 增加 | 隨著推薦範圍擴大，移除風格資訊所造成的下降幅度逐步增加 | ΔS：-0.0106 → -0.0233 → -0.0264 | 保留 | `<RUN>/ablation/fresh_5variant_8metric_summary.csv`（`summarize_fair_subset_5seed.py`） |
| X19 | 4.4.2（p.54）；圖4-7 | — | 各因子相對效能下降占比 | 風格 71%–89%、天氣 10%–16%、場合 1%–14%；風格在 OR 的占比隨 K 增加而提高 | CP AUC：風格 68%、天氣 8%、場合 24%；CP FITB Acc：風格 70%、天氣 17%、場合 13%；CIR Recall@10：無法定義（天氣或場合移除後效能上升）；CIR Recall@30：無法定義（天氣或場合移除後效能上升）；CIR Recall@50：無法定義（天氣或場合移除後效能上升） | 改寫敘述 | `<RUN>/ablation/T07_stage2_factor_contribution_ratio.csv`（`summarize_fair_subset_5seed.py`） |
| X20 | 4.4.2（p.54） | — | 移除三項因子後效能皆下降 | 移除三項因子後效能皆下降，其中以風格資訊的影響最大 | 10／15 個 Δ 為負；移除後上升：No-W R@10 +0.0001、No-W R@50 +0.0052、No-O R@10 +0.0037、No-O R@30 +0.0017、No-O R@50 +0.0041 | 改寫敘述 | `<RUN>/ablation/fresh_5variant_8metric_summary.csv`（`summarize_fair_subset_5seed.py`） |
| X21 | — | 5.3（p.20–21） | Weather 與 Occasion 的補充效益 | removing Weather or Occasion results in smaller changes ... Weather and Occasion provide smaller complementary benefits | No-W：R@10 +0.0001、R@30 -0.0001、R@50 +0.0052；No-O：R@10 +0.0037、R@30 +0.0017、R@50 +0.0041（正值表示移除後 Recall 反而上升） | 改寫敘述 | `<RUN>/ablation/fresh_5variant_8metric_summary.csv`（`summarize_fair_subset_5seed.py`） |
| X22 | — | 5.2（p.20） | CIR 各深度皆改善 | CIR also improves at every reported depth | R@1 +0.0013、R@3 +0.0038、R@5 +0.0079、R@10 +0.0145、R@30 +0.0201、R@50 +0.0229 | 保留 | `<RUN>/statistics/main_paired_bh_8metrics.csv`（`complete_statistics.py`） |

## 五、Discussion 與 Conclusion 核心結論

| ID | 論文 | TORS | 項目 | 稿件 | 最新 | 狀態 | 來源（程式） |
|---|---|---|---|---|---|---|---|
| C01 | 4.7（p.63）；5.1（p.65） | 6.1 RQ1（p.23）；7（p.24） | 情境感知描述提升 CP、FITB 與 OR／CIR | 僅替換文字條件即能提升穿搭相容性預測、填空式搭配與套裝推薦表現 | 8／8 項平均差為正（最小：R@1 +0.0013） | 保留 | `<RUN>/statistics/main_paired_bh_8metrics.csv`（`complete_statistics.py`） |
| C02 | 4.7（p.63）；5.1（p.65） | — | 公平比較子集維持相同改善方向 | 於公平比較子集上仍維持相同改善方向 | 5／5 項為正（AUC +0.0150、FITB +0.0106、R@10 +0.0068、R@30 +0.0182、R@50 +0.0165） | 保留 | `<RUN>/ablation/fresh_5variant_8metric_summary.csv`（`summarize_fair_subset_5seed.py`） |
| C03 | 5.1（p.65） | — | CP 與 OR 皆顯著優於原始文字 | 穿搭相容性預測與套裝推薦表現皆顯著優於使用原始文字的設定 | BH 校正後 7／8 顯著；未顯著：R@1（BH p = 0.0947） | 改寫敘述 | `<RUN>/statistics/main_paired_bh_8metrics.csv`（`complete_statistics.py`） |
| C04 | 4.7（p.64）；5.1（p.65） | — | 移除任一因子後效能皆下降 | 移除天氣、場合與風格資訊後，模型效能皆出現不同程度的下降 | 10／15 個 Δ 為負；移除後上升：No-W R@10 +0.0001、No-W R@50 +0.0052、No-O R@10 +0.0037、No-O R@30 +0.0017、No-O R@50 +0.0041 | 改寫敘述 | `<RUN>/ablation/fresh_5variant_8metric_summary.csv`（`summarize_fair_subset_5seed.py`） |
| C05 | 4.7（p.64）；5.1（p.65） | 6.1 RQ2（p.23）；7（p.24） | 風格是最強的情境因子 | 風格資訊的影響最為明顯／Removing Style causes the largest degradation in CP and CIR | ΔS：AUC -0.0135、FITB -0.0126、R@10 -0.0106、R@30 -0.0233、R@50 -0.0264（CP 與 OR 皆為三因子中最大下降） | 保留 | `<RUN>/ablation/fresh_5variant_8metric_summary.csv`（`summarize_fair_subset_5seed.py`） |
| C06 | 4.7（p.64）；5.1（p.65） | 6.1 RQ2（p.23） | 天氣與場合提供補充限制 | 天氣與場合主要補充外在環境與使用情境限制／Weather and Occasion show smaller effects but provide complementary constraints | No-W：R@10 +0.0001、R@30 -0.0001、R@50 +0.0052；No-O：R@10 +0.0037、R@30 +0.0017、R@50 +0.0041（正值表示移除後 Recall 反而上升） | 改寫敘述 | `<RUN>/ablation/fresh_5variant_8metric_summary.csv`（`summarize_fair_subset_5seed.py`） |
| C07 | — | 6.1 RQ1（p.23）；7（p.24） | Two-Tower 保留 AUC 與檢索增益 | Two-Tower validation retains AUC and retrieval gains but not a consistent FITB gain | AUC +0.0378、FITB -0.0077、R@10 +0.0102、R@30 +0.0153、R@50 +0.0170（保存輸出重算）（第二模型不在 35 組正式訓練內；數值由學姊保存的逐 seed 輸出重算。） | 非正式 run | `<RUN>/secondary/two_tower/recomputed_key_mean_std.csv`（`reproduce_remaining_thesis.py`） |
| C08 | — | 6.1 RQ2（p.23）；7（p.24） | 反事實分析支持 Style 最強 | Style replacement produces the highest Top-1 changed rate and the lowest Top-5 Jaccard overlap | —（來源為 03_實驗與結果_experiments_results/06_反事實情境敏感度/（seed 1、歷史 checkpoint）；正式 run 未重做。） | 非正式 run | — |
| C09 | 4.7（p.64）；5.1（p.65） | 5.5（p.22）；6.2（p.23） | 類別適用邊界 | 洋裝、鞋款與包款受益較明顯；太陽眼鏡與飾品效益有限 | —（來源為 03_實驗與結果_experiments_results/04_情境子集與三因子分析/ 的保存輸出（歷史 checkpoint）；正式 run 未重畫圖 4-8 至 4-13。） | 非正式 run | — |

## 六、第九節要求保留的結論

| ID | 論文 | TORS | 項目 | 稿件 | 最新 | 狀態 | 來源（程式） |
|---|---|---|---|---|---|---|---|
| R01 | — | — | 八項主要指標的平均值全部改善 | （第九節要求保留） | 8／8 項平均差為正（最小：R@1 +0.0013） | 保留 | `<RUN>/statistics/main_paired_bh_8metrics.csv`（`complete_statistics.py`） |
| R02 | — | — | 八項中七項經 BH 校正後顯著 | （第九節要求保留） | BH 校正後 7／8 顯著；未顯著：R@1（BH p = 0.0947） | 保留 | `<RUN>/statistics/main_paired_bh_8metrics.csv`（`complete_statistics.py`） |
| R03 | — | — | Recall@1 有平均改善但未達校正後顯著 | （第九節要求保留） | R@1 Δ +0.0013、BH p = 0.0947 | 保留 | `<RUN>/statistics/main_paired_bh_8metrics.csv`（`complete_statistics.py`） |
| R04 | — | — | Recall@3 仍達校正後顯著 | （第九節要求保留） | R@3 BH p = 0.00443（\*\*） | 保留 | `<RUN>/statistics/main_paired_bh_8metrics.csv`（`complete_statistics.py`） |
| R05 | — | — | Style 是 CP 與 OR 中最大且最穩定的貢獻因素 | （第九節要求保留） | ΔS：AUC -0.0135、FITB -0.0126、R@10 -0.0106、R@30 -0.0233、R@50 -0.0264（CP 與 OR 皆為三因子中最大下降） | 保留 | `<RUN>/ablation/fresh_5variant_8metric_summary.csv`（`summarize_fair_subset_5seed.py`） |
| R06 | — | — | Weather 與 Occasion 的 OR 效果較小，且對 seed 敏感 | （第九節要求保留） | \|ΔW\|、\|ΔO\| 皆小於 \|ΔS\|；逐 seed 方向不一致：No-W R@10（2 升／3 降）、No-W R@30（2 升／3 降）、No-W R@50（4 升／1 降）、No-O R@10（4 升／1 降）、No-O R@30（3 升／2 降）、No-O R@50（3 升／1 降） | 保留 | `<RUN>/ablation/fresh_5variant_8metric_summary.csv`、`<RUN>/ablation/fresh_25unit_metrics.csv`（`summarize_fair_subset_5seed.py`） |
| R07 | — | — | 不得將 Weather 與 Occasion 寫成在所有 OR 指標皆有穩定正貢獻 | （第九節要求保留） | No-W：R@10 +0.0001、R@30 -0.0001、R@50 +0.0052；No-O：R@10 +0.0037、R@30 +0.0017、R@50 +0.0041（正值表示移除後 Recall 反而上升） | 保留 | `<RUN>/ablation/fresh_5variant_8metric_summary.csv`（`summarize_fair_subset_5seed.py`） |

## 七、README expected results 與補充表格

| ID | 論文 | TORS | 項目 | 稿件 | 最新 | 狀態 | 來源（程式） |
|---|---|---|---|---|---|---|---|
| P01 | — | — | docs/expected_results.md 的比較基準 | 以論文／歷史數值作為 expected results | 已引用 full_20261002T161428Z（第三項：README expected results 只能引用正式 run；論文值改列為歷史參照。） | 保留 | `docs/expected_results.md` |
| P02 | — | — | README §11：16 個平均值落在 0.005 容許範圍 | 16 個 paper-comparison means 均落在 absolute tolerance 0.005 內 | 16／16 個平均值在 0.005 內（最大差 0.0024） | 保留 | `<RUN>/main/main_reproduction_summary.csv`（`compare_main_results.py`） |
| P03 | — | — | README §11：方向與顯著性摘要 | 八個方向均為正；BH 校正後 7／8 顯著；Recall@1 未達顯著 | 8 項平均差皆為正；BH 校正後 7／8 顯著；未顯著：R@1（BH p = 0.0947）；R@1 Δ +0.0013、BH p = 0.0947 | 保留 | `<RUN>/statistics/main_paired_bh_8metrics.csv`（`complete_statistics.py`） |
| P04 | 表4-3（p.42） | Table 5 Proxy values（p.19） | CLO／MET／舒適溫度分布 | 表 4-3 | all_checked_display_fields_match：24 checked PDF-display fields from aligned saved CLO/MET/Tsub proxy outputs; independent MET valid source incomplete. | 保留 | `<RUN>/chapter4/chapter4_table_overview.csv`（`build_chapter4_report.py`） |
| P05 | 表4-4（p.43） | Table 5 Text length（p.19） | 文字長度差相關分析 | 表 4-4 | archived_length_reanalysis_exact：Length correlations recalculated from archived seed-level rows; module-level archived-table comparison. | 保留 | `<RUN>/chapter4/chapter4_table_overview.csv`（`build_chapter4_report.py`） |
| P06 | 表4-5（p.43） | Table 5 Text length（p.19） | 相近長度子集 Hit@10 | 表 4-5 | archived_length_reanalysis_exact：Length matched-subset/bucket tables recalculated from archived outputs; module-level archived-table comparison. | 保留 | `<RUN>/chapter4/chapter4_table_overview.csv`（`build_chapter4_report.py`） |
| P07 | 表4-6（p.44） | Table 5 Target clues（p.20） | 目標單品線索洩漏檢查 | 表 4-6 | checked_counts_differ：Historical P12 reliability_meta_from_subset.csv missing; present-day category reconstruction differs. | 更新數字 | `<RUN>/chapter4/chapter4_table_overview.csv`（`build_chapter4_report.py`） |
| P08 | 表4-7（p.47） | Table 5 Judge agreement（p.20） | 評分者低分樣本重疊 | 表 4-7 | checked_counts_differ：Saved-output bottom-p intersections differ from thesis; cutoff ties and historical input/runtime identity unresolved. | 更新數字 | `<RUN>/chapter4/chapter4_table_overview.csv`（`build_chapter4_report.py`） |
| P09 | 表4-8（p.47） | Table 5 Prompt robustness（p.20） | 提示詞穩健性 | 表 4-8 | all_checked_display_fields_match：30 checked PDF-display fields from six saved Judge comparison files; no LLM rerun and historical 150-ID selection not recovered. | 保留 | `<RUN>/chapter4/chapter4_table_overview.csv`（`build_chapter4_report.py`） |
| P10 | 表4-9（p.48） | Table 5 Manual audit（p.20） | 人工稽核情境覆蓋 | 表 4-9 | archived_human_audit_reanalysis_exact：30-case coverage recomputed from selected archived records; source label-generation provenance not proven. | 保留 | `<RUN>/chapter4/chapter4_table_overview.csv`（`build_chapter4_report.py`） |
| P11 | 表4-10（p.48） | Table 5 Manual audit（p.20） | 人工稽核分數比較 | 表 4-10 | archived_human_audit_reanalysis_exact：750 archived human judgments recomputed and compared to archived T30; no new human judgments. | 保留 | `<RUN>/chapter4/chapter4_table_overview.csv`（`build_chapter4_report.py`） |
| P12 | 表4-17（p.60） | 5.5（p.22） | 案例一排名比較 | 表 4-17 | provenance_only_or_not_independently_verified：Fresh rankings for identical set_id/target/seed/pool not independently matched against published examples. | 非正式 run | `<RUN>/chapter4/chapter4_table_overview.csv`（`build_chapter4_report.py`） |
| P13 | 表4-18（p.61） | 5.5（p.22） | 案例二排名比較 | 表 4-18 | provenance_only_or_not_independently_verified：Fresh rankings for identical set_id/target/seed/pool not independently matched against published examples. | 非正式 run | `<RUN>/chapter4/chapter4_table_overview.csv`（`build_chapter4_report.py`） |
| P14 | 表4-19（p.62） | 5.5（p.22） | 顏色偏移改善分析 | 表 4-19 | provenance_only_or_not_independently_verified：Image-dependent human visual/color judgments not independently re-executed. | 非正式 run | `<RUN>/chapter4/chapter4_table_overview.csv`（`build_chapter4_report.py`） |
| P15 | 表4-20（p.63） | 5.5（p.22） | 失敗案例排名比較 | 表 4-20 | provenance_only_or_not_independently_verified：Identical-case fresh ranking comparison not independently verified. | 非正式 run | `<RUN>/chapter4/chapter4_table_overview.csv`（`build_chapter4_report.py`） |
| P16 | — | Table 8（p.21） | Two-Tower CP Test AUC | 0.8796 ± 0.0029 \| 0.9174 ± 0.0033 \| +0.0378 | 0.8796 ± 0.0029 \| 0.9174 ± 0.0033 \| +0.0378（保存輸出重算值與稿件一致；95% CI 未由重算輸出提供。） | 非正式 run | `<RUN>/secondary/two_tower/recomputed_key_mean_std.csv`（`reproduce_remaining_thesis.py`） |
| P17 | — | Table 8（p.21） | Two-Tower CP FITB Accuracy | 0.5460 ± 0.0051 \| 0.5382 ± 0.0050 \| −0.0077 | 0.5460 ± 0.0051 \| 0.5382 ± 0.0050 \| -0.0077（保存輸出重算值與稿件一致；95% CI 未由重算輸出提供。） | 非正式 run | `<RUN>/secondary/two_tower/recomputed_key_mean_std.csv`（`reproduce_remaining_thesis.py`） |
| P18 | — | Table 8（p.21） | Two-Tower Recall@10 | 0.0378 ± 0.0008 \| 0.0479 ± 0.0017 \| +0.0102 | 0.0378 ± 0.0008 \| 0.0479 ± 0.0017 \| +0.0102（保存輸出重算值與稿件一致；95% CI 未由重算輸出提供。） | 非正式 run | `<RUN>/secondary/two_tower/recomputed_key_mean_std.csv`（`reproduce_remaining_thesis.py`） |
| P19 | — | Table 8（p.21） | Two-Tower Recall@30 | 0.0950 ± 0.0017 \| 0.1103 ± 0.0030 \| +0.0153 | 0.0950 ± 0.0017 \| 0.1103 ± 0.0030 \| +0.0153（保存輸出重算值與稿件一致；95% CI 未由重算輸出提供。） | 非正式 run | `<RUN>/secondary/two_tower/recomputed_key_mean_std.csv`（`reproduce_remaining_thesis.py`） |
| P20 | — | Table 8（p.21） | Two-Tower Recall@50 | 0.1419 ± 0.0026 \| 0.1589 ± 0.0046 \| +0.0170 | 0.1419 ± 0.0026 \| 0.1589 ± 0.0046 \| +0.0170（保存輸出重算值與稿件一致；95% CI 未由重算輸出提供。） | 非正式 run | `<RUN>/secondary/two_tower/recomputed_key_mean_std.csv`（`reproduce_remaining_thesis.py`） |
| P21 | — | Table 8（p.21） | Two-Tower Mean rank | 601.3589 ± 7.0359 \| 572.4678 ± 10.5809 \| −28.8911 | 601.3589 ± 7.0359 \| 572.4678 ± 10.5809 \| -28.8911（保存輸出重算值與稿件一致；95% CI 未由重算輸出提供。） | 非正式 run | `<RUN>/secondary/two_tower/recomputed_key_mean_std.csv`（`reproduce_remaining_thesis.py`） |
| P22 | — | Table 8（p.21） | Two-Tower Median rank | 361.2000 ± 6.2209 \| 329.8000 ± 9.7826 \| −31.4000 | 361.2000 ± 6.2209 \| 329.8000 ± 9.7826 \| -31.4000（保存輸出重算值與稿件一致；95% CI 未由重算輸出提供。） | 非正式 run | `<RUN>/secondary/two_tower/recomputed_key_mean_std.csv`（`reproduce_remaining_thesis.py`） |
| P23 | — | Table 9（p.21） | 反事實檢索反應 | 24 cases；mean ΔRank +17.0；Top-5 Jaccard 0.351 | —（來源為 03_實驗與結果_experiments_results/06_反事實情境敏感度/（seed 1、歷史 checkpoint）；正式 run 未重做。） | 非正式 run | — |
| P24 | 4.5（p.54–56） | 5.5（p.22） | 情境子集 Hit@10 皆為正 | 3,432 pairs（17,160 seed-level observations）下各天氣、場合、風格與目標類型子集皆改善 | —（來源為 03_實驗與結果_experiments_results/04_情境子集與三因子分析/ 的保存輸出（歷史 checkpoint）；正式 run 只保存整體 Hit@10（secondary/case_analysis/fresh_fair_subset_rowlevel_summary.csv）。） | 非正式 run | — |
