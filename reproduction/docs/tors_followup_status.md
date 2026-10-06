# TORS 採最新結果後續工作：完成狀態

依老師 2026-09-30 的《TORS 採最新結果後續工作》逐項列出完成狀態與證據。
路徑相對於 `reproduction/`；`<RUN>` 代表正式 run `full_20261002T161428Z`。

狀態：✅ 完成；⚠️ 待老師確認；📝 修稿時處理。

## 二、乾淨環境端到端驗收

| 要求 | 狀態 | 證據 |
|---|---|---|
| 記錄 repository URL、branch、完整 commit | ✅ | `results/raw/<RUN>/run_identity/git_remote.txt`、`git_status_start.txt`、`git_commit.txt`（`b9bf5aa…`） |
| 完全依 README 建環境、準備資料 | ✅ | `results/raw/<RUN>/clean_room/cleanroom_setup.sh`、`cleanroom_setup.log`、`cleanroom_pip_freeze.txt` |
| 執行 `reproduce_all.sh --fresh` | ✅ | `results/raw/<RUN>/clean_room/cleanroom_launch.sh` |
| 不需人工搬檔、臨時改程式或隱藏步驟 | ✅ | `docs/clean_room_acceptance.md`（開跑時工作區乾淨；中途清快取是 README 0.6 已記載的系統操作，不影響 run） |
| 保存開始、結束時間、總耗時、GPU 與 runtime | ✅ | 2026-10-02T16:14:28Z → 2026-10-04T23:22:08Z，55:07:40；`environment/<RUN>/`、`results/final_reference_manifest.json` |
| 保存 terminal 最後摘要與 RUN_STATUS | ✅ | `results/raw/<RUN>/run_identity/final_terminal_summary.txt`、`RUN_STATUS.txt`（`PASSED`） |
| 中途失敗的紀錄 | ✅ | 無失敗 |
| 與 reference run 比較平均值、方向、統計結論 | ✅ | 70 / 70 checkpoint、70 / 70 逐 seed 結果、25 / 25 逐題明細逐位元相同，結論不變（`docs/clean_room_acceptance.md`） |

## 三、選定唯一正式 reference run

| 要求 | 狀態 | 證據 |
|---|---|---|
| 固定正式 run ID 與完整 commit | ✅ | `results/final_reference_manifest.json`：`full_20261002T161428Z`、`b9bf5aac7d07cb39cacc7f73eb28166848c19fe4` |
| seeds 固定為 1～5 | ✅ | 同上 `seeds`；`results/summary/<RUN>/seed_index.csv` |
| 保存每個 seed 的 raw results 與 summary | ✅ | `results/raw/<RUN>/`、`results/summary/<RUN>/` |
| 固定表格與統計程式版本 | ✅ | manifest 的 `programs`（commit 與統計程式 SHA-256） |
| README expected results 只引用正式 run | ✅ | `docs/expected_results.md` |
| final reference manifest（run ID、commit、seeds、環境、切分 checksum、checkpoint 雜湊、結果位置） | ✅ | `results/final_reference_manifest.json`；70 個 checkpoint 已重新計算雜湊驗證；雜湊清單 `results/SHA256SUMS_<RUN>.txt` |

## 四、以最新結果重新產生正式表格

| 要求 | 狀態 | 證據 |
|---|---|---|
| 表 4-11～4-16 由正式 run 自動產生 | ✅ | `results/summary/<RUN>/statistics/table_4_11_cp_main.csv`、`table_4_12_or_main.csv`；`results/summary/<RUN>/ablation/T03`～`T06` |
| 五 seed 平均、SD、paired test、BH、effect size、CI | ✅ | 主實驗 8 項同族群 BH；公平子集 5 項同族群 BH（學姊原表規則）；消融 T10 / T11 有 CI 與 dz |

## 五、正式化標準化 decoder

| 要求 | 狀態 | 證據 |
|---|---|---|
| 移除 temporary patch 措辭 | ✅ | 程式與文件已改；舊 run 紀錄保持原樣（歷史紀錄） |
| 記錄 3 層、16 heads、dropout 0.1 與輸入輸出流程 | ✅ | `docs/standardized_decoder.md` |
| 公開程式可直接執行，不依賴遺失的 class | ✅ | `scripts/standard_decoder.py`；clean-room run 通過 |
| 說明早期自訂 decoder 未保存、全部實驗採標準化實作 | ✅ | `docs/standardized_decoder.md`、README §14；原作者公開程式本身即使用此 decoder |

## 六、固定並公開最新版公平子集

| 要求 | 狀態 | 證據 |
|---|---|---|
| 完整 subset ID | ✅ | `splits/fair_subset/fair_subset_ids.txt`（21,903） |
| train／validation／test ID | ✅ | `train_ids.txt`、`valid_ids.txt`、`test_ids.txt`（10,225／1,748／9,930） |
| OR 可評估 3,432 筆 query ID | ✅ | `or_query_ids.csv`（與 25 組實際評估題目完全相同） |
| 建構規則、排除條件、產生程式、checksum | ✅ | `docs/fair_subset.md`、`splits/fair_subset/SHA256SUMS.txt` |
| 表 4-13～4-16 使用同一版本子集 | ✅ | `seed_index.csv` 25 組 `candidate_id_sha256` 相同 |
| 子集正式名稱 | ⚠️ | PDF 寫 length-controlled，但子集依三因子挑選，與文字長度無關；目前用「reproducibly reconstructed fair subset」 |

## 七、整理超參數文件

| 要求 | 狀態 | 證據 |
|---|---|---|
| 更新 `hyperparameter_tuning.md` | ✅ | 依四類分類，主模型設定全部來自原論文與原作者程式 |
| 更新 `tuning.csv` | ✅ | `results/tuning.csv`（每個設定一列，含類別與來源） |
| 再次搜尋舊紀錄，找不到者如實標記 | ✅ | `docs/hyperparameter_tuning.md` 第 3 節 |

## 八、論文結果對帳資料

| 要求 | 狀態 | 證據 |
|---|---|---|
| Abstract、setup、表 4-11～4-16、統計敘述、討論與結論、README expected results 與補充表格 | ✅ | `docs/manuscript_reconciliation/reconciliation.md`（287 項，每項含最新值、來源 CSV、run ID、程式） |
| 修訂稿逐項修正 | 📝 | 21 處改寫敘述、170 處更新數字、15 處補寫 |

## 九、必須保留的結論

七條結論都由正式 run 支持（對帳表 R01～R07）。📝 修稿時照寫。

## 十、封版前應交付的資料

| # | 交付內容 | 狀態 | 位置 |
|---|---|---|---|
| 1 | 乾淨環境驗收紀錄 | ✅ | `docs/clean_room_acceptance.md`、`results/raw/<RUN>/run_identity/`、`results/raw/<RUN>/clean_room/` |
| 2 | 正式 run manifest、五 seed raw 與 summary | ✅ | `results/final_reference_manifest.json`、`results/raw/<RUN>/`、`results/summary/<RUN>/` |
| 3 | 表 4-11～4-16 與完整統計 | ✅ | `results/summary/<RUN>/statistics/`、`results/summary/<RUN>/ablation/` |
| 4 | 標準化 decoder 程式與說明 | ✅ | `scripts/standard_decoder.py`、`docs/standardized_decoder.md` |
| 5 | 公平子集 ID、script、切分、checksum | ✅ | `splits/fair_subset/`、`docs/fair_subset.md` |
| 6 | `hyperparameter_tuning.md` 與 `tuning.csv` | ✅ | `docs/hyperparameter_tuning.md`、`results/tuning.csv` |
| 7 | 論文結果對帳表 | ✅ | `docs/manuscript_reconciliation/` |
| 8 | 最新版 README 與可執行指令 | ✅ | `README.md` §0（clean-room 指令；§0.8 跑完後的比對與結果位置）、§2、§11 |
| 9 | 已知限制清單 | ✅ | `docs/known_limitations.md` |

## 十一、完成判定

| 條件 | 狀態 | 說明 |
|---|---|---|
| 另一位接手者只依 README，在乾淨環境得到與正式 run 一致的結論 | ✅ | 正式 run 本身就是從全新 clone 照 README 跑出；三次獨立 35 組訓練逐位元相同。接手者跑完後以 `scripts/compare_with_official_run.py` 比對（README §0.8），判定與對帳表 R01～R07 相同 |
| 程式、切分、統計輸出與交付文件追溯到同一 commit 與 run | ✅ | run commit `b9bf5aa`；之後只新增收錄程式與文件產生程式，training、statistics、設定與切分都未再修改 |

## 封版之外仍待處理

- ⚠️ 老師確認：子集名稱、4.5 類別與情境子集分析是否用正式 run 重算、舊 checkpoint 分析（Two-Tower、反事實、個案）在稿件中的標示方式。
- 📝 修訂稿：依對帳表修正。
- 公開發佈：固定 Release `v1.0.0-tors-reproduction` 與論文的 Artifact Availability statement（不使用 DOI；`docs/tors_compliance.md` A1／A2／A10）。
