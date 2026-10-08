# 交接說明：專案現況與接續方式（2026-10-08）

這份文件給換電腦後接續的人（包括 AI 助手）。路徑除非另外註明，都相對於 repository 根目錄。

## 1. 現況摘要

- **已發布的版本**：GitHub Release `v1.0.0-tors-reproduction`（2026-10-07），正式 run 為
  `full_20261002T161428Z`。稿件目前引用這個版本。
- **開發分支 `next-release`**：v1.0.0 之後的全部工作（本文件所在的分支），尚未發布。內容：
  - 訓練以外的全部分析整合成一個指令（`scripts/run_analyses.sh`）；
  - 46 個分析項目的逐項報告（`scripts/report_items.py` → `ITEMS_STATUS.md`）；
  - 跑完時顯示的最終摘要（`scripts/show_final_summary.py` → `FINAL_SUMMARY.txt`）。
- **測試**：用 `reproduce_all.sh --analyses-only` 對正式 run 完整跑過兩次。
  - CPU：35 分鐘。
  - GPU：4 小時 13 分。
  - 所有輸出都和 repository 收錄的結果相同，只差時間欄位；重新訓練的 20 個 Two-Tower 模型檔也完全相同。
- **2026-10-08 指導老師的決定**：資料準備不要再沿用學姊產生的資料，要依學姊的專案重做一遍（見第 3 節）。
  - 因此原訂當天開跑的新正式 run 暫停。
  - 為那次 run 準備的候選版標籤沒有推送，已刪除。
- **下一步**：等老師決定重做的範圍後，先試跑，再完成資料準備，最後用單一指令跑新的正式 run（見第 6 節）。

## 2. 46 個分析項目的判定（正式 run）

逐項的依據與證據在 `results/ITEMS_STATUS.md`，總結在 `results/FINAL_SUMMARY.txt`。判定標準是
「與學姊專案 2025 年的數字是否相同」：

| 判定 | 項數 | 內容 |
|---|---:|---|
| 和學姊 2025 年相同（PASS） | 19 | 由公開資料或學姊保存的資料重算，數字相同 |
| 數值不同、計算已驗證（DIFFERS） | 22 | 20 項因模型重新訓練而不同。學姊的模型或資料用同一支程式重算時，會得到她的數字，所以程式本身是對的。另外 2 項只差在同分樣本的排序 |
| 沿用學姊的資料（REUSED） | 4 | 情境描述、CLO／MET 估計、三因子標註、8 個特徵檔 |
| 無法重現（NOT_REPRODUCIBLE） | 1 | CLO 離群值重新推論（學姊沒有留下紀錄） |

老師 10-08 的決定，就是要把「沿用」的這 4 項，以及依附在描述上的 LLM 評分與人工稽核，都改成重做。

## 3. 待重做的資料準備

學姊的程式與提示詞保留情況：

| 步驟 | 學姊的做法 | 留下的程式與提示詞 | 重做方式 |
|---|---|---|---|
| CLO 估計 | Gemma 3 4B IT 看商品拼貼圖，參考 CLO 對照表 | `01_資料建構_data_construction/clo_met_temperature/CLO_reference/Adding_CLO.ipynb`、`reproduction/prompts/clo_estimation.txt`、對照表 `ensemble_clo.json`、`single_items_clo.json` | 照筆記本重跑 |
| MET 活動選擇 | Gemma 3 4B IT 從 457 個活動選一個 | `MET_reference/metMapping.ipynb`、`reproduction/prompts/activity_met_mapping.txt`、`adult_activity_compendium_sorted_2024.json` | 照筆記本重跑 |
| 溫度參考值 | 稿件式 2 | `temperature_results/temperature.ipynb` | 用新的 CLO、MET 重算 |
| 情境描述 35,140 筆 | Gemma 3 4B IT 看拼貼圖，加上溫度、CLO 與原標題改寫 | `temperature_results/temperature.ipynb`、`reproduction/prompts/semantic_rewrite.txt` | 照筆記本重跑 |
| CLO 離群值重新推論 | 稿件 3.2.1 節：超過 IQR 上界者重新推論 | 沒有 | 依稿件描述自行實作 |
| 天氣／場合／風格拆分 | LLM（細節不明；最後採用的檔案是 `wos_split_results_v5_merged_retry_round3.jsonl`） | 沒有 | 依稿件表 3 的定義自行設計 |
| 特徵檔 | FashionCLIP（影像、outfit 文字）、SentenceBERT `distiluse-base-multilingual-cased-v2`（商品文字、類別） | 沒有。FashionCLIP 的版本已確認：學姊筆記本 P16 記錄的 revision 重算 24 筆描述，與原特徵相同 | 自行撰寫萃取程式 |
| LLM 評分 | Qwen3-VL-32B-Instruct 與 Gemma-3-27B-IT，三階段檢核表（稿件演算法 1） | 只有檢核表（`reproduction/prompts/checklists/`）與分數，評分程式沒有 | 依稿件演算法 1 自行實作 |
| Prompt 穩健性 | 150 筆以 3 種提示詞重評 | 只有結果 | 同上 |
| 人工稽核 | 研究者人工判斷 30 筆、750 個判斷 | `03_實驗與結果_experiments_results/07_人工稽核與品質診斷/`（表單、抽樣規則、結果） | 由人重新判斷新描述（抽樣依新評分） |

需要注意：

- **生成結果無法和學姊的一樣**：三個 Gemma 步驟用 Transformers 的 `pipeline`，生成設定是預設值，帶有隨機性。重做時要固定亂數種子並保存所有設定。
- **輸入圖片**：筆記本把每套服裝的商品圖拼成 4 欄拼貼圖（每格 224×224）。商品圖用
  `bash reproduction/scripts/bootstrap_data.sh --with-images` 下載。
- **需要下載的模型**：
  - Gemma 3 4B IT、Gemma-3-27B-IT、Qwen3-VL-32B-Instruct，共約 130 GB。
  - Gemma 需要先在 Hugging Face 同意授權，再用帳號的 token 下載。
- **重做後的影響**：描述換了，特徵、35 組訓練與所有分析都要重跑，稿件的數字幾乎全部會換。
- **時間粗估（單台 NVIDIA GB10，需試跑確認）**：
  - 三個 Gemma 步驟合計 1～5 天；
  - 三因子拆分 0.5～1 天；
  - LLM 評分可能 1～3 週（可考慮雲端 GPU）；
  - 訓練加分析約 2.5 天。

給老師的評估文件（含待決問題）是 `TORS_資料準備重做評估_20261008.pdf`。它不在 repository 中，見第 7 節。

**待老師決定的問題**：

1. 「不沿用」是否也包含 LLM 評分與人工稽核？
2. 學姊沒有留下程式的四步，是否同意由我們依稿件描述自行實作？
3. LLM 評分在哪裡跑：這台 GB10、雲端 GPU，還是縮小規模？
4. 論文是否接受數字全部換成新流程的結果？

## 4. 在新電腦上建立環境

完整步驟見 `reproduction/README.md` 第 0 節。要點：

    git clone --branch next-release https://github.com/7114029017/context-aware-outfit-recommendation.git
    cd context-aware-outfit-recommendation
    git lfs install && git lfs pull                     # 8 個特徵檔，約 2 GB
    python3.12 -m venv .venv-repro && source .venv-repro/bin/activate
    python -m pip install torch==2.9.1 torchvision==0.24.1 --index-url https://download.pytorch.org/whl/cu130
    python -m pip install -r reproduction/environment/requirements-reproduction-runtime.txt
    bash reproduction/scripts/bootstrap_data.sh --with-images --with-fashionclip --with-nomic --with-compendium
    bash reproduction/scripts/check.sh

- **Python 版本**：必須是 3.12。
- **GPU**：若不是 NVIDIA GB10，訓練結果不會逐位元相同。與正式 run 比對時判定為 `CONSISTENT` 是正常的（README 0.8）。
- **Windows 電腦**：在 WSL2 的 Ubuntu 24.04 裡照上面的步驟操作，repository 要 clone 在 WSL 的 Linux 檔案系統（README 0.0）。
  原生 Windows 不支援，`reproduction/scripts/` 的腳本會說明後停止。用 Git for Windows clone v1.0.0 時，換行會被改成
  CRLF，01～03 的 SHA-256 檢查會失敗；`next-release` 起由 `.gitattributes` 關閉換行轉換。
- **完整訓練的版本檢查**：只有 `main` 分支或 `v*-tors-reproduction` 標籤能開始完整訓練。從 `next-release` 只能做
  `reproduce_all.sh --analyses-only <run>`。
- **重做舊正式 run 的分析**：需要舊電腦上的 run 資料夾（含 70 個 checkpoint，見第 7 節）：

      bash reproduction/scripts/reproduce_all.sh --analyses-only <run 資料夾> --out-dir <輸出位置>

## 5. Git 與發布的狀態

| 位置 | 狀態 |
|---|---|
| GitHub `7114029017/context-aware-outfit-recommendation`：`main`、標籤 `v1.0.0-tors-reproduction` | `fb620bb`，已發布，未變動 |
| 同一 repository 的 `next-release` 分支 | v1.0.0 之後的全部工作，包含本文件；2026-10-08 推送 |
| 原始開發 repository（不公開，branch `thesis-full-reproduction`） | 舊電腦的工作目錄有 1 個 commit 沒推送：`6693528`（Zenodo 打包程式）。它的內容已包含在本 repository |

- **提交規則**：
  - 作者用 `7114029017 <338583826+7114029017@users.noreply.github.com>`。
  - 逐一指定要提交的檔案，不要用 `git add .`。
- **GitHub 權限**：舊電腦上推送用的 token 到 2026-10-13 到期，新電腦要重新設定推送權限。
- **Zenodo**：只準備了上傳草稿，沒有發布，也不使用 DOI。

## 6. 接下來的步驟

1. 老師決定第 3 節的四個問題。
2. 試跑約一天：
   - 照學姊的筆記本，對 100～200 套服裝跑 CLO、MET、情境描述與一小批評分；
   - 量出每一步的耗時，並比較新舊資料的分布。
3. 依決定的範圍完成全部資料準備，包括學姊沒有留下程式的四步：
   - 程式與設定收進 repository；
   - 新資料的位置與格式要能接上現有流程（`reproduction/configs/`、`02_模型訓練和驗證_model_training_validation/fashionclip_data/`）。
4. 用單一指令跑新的正式 run。
   - 要從 `main` 或發布用的標籤開始。
   - 做法照 2026-10-03 的 clean-room run：全新資料夾、照 README 第 0 節、保存 setup 與終端輸出紀錄。
5. 驗收後收錄新 run，把文件中的正式 run 換成新的（README §0.9、§2、§15，結果資料夾的說明，驗收報告等），並發布新版本。
6. 依新結果更新論文。
   - 稿件與 repository 的逐項對照可參考 `TORS_v8_與公開repo一致性檢查報告.pdf`（不在 repository 中）。
   - 該報告列出的 5 點稿件問題仍待處理：
     1. 稿件 3.2.1 節的 CLO 離群值重新推論；
     2. 稿件寫「1,114 筆活動」，官方是 1,111 筆；
     3. 太陽眼鏡「略降」的敘述；
     4. 6.5 節引用的舊 commit；
     5. 表 8、9 與 5.5 節要沿用舊數字還是換新。

## 7. 只在舊電腦上的檔案（需要手動帶走）

| 檔案 | 位置（舊電腦） | 大小 | 說明 |
|---|---|---|---|
| 正式 run 資料夾 | `~/Documents/course/tors-cleanroom-20261003/` | 13 GB | 含 70 個 checkpoint、setup 與終端輸出紀錄；checkpoint 不公開。重做舊 run 的分析時需要 |
| 給老師的評估 | `~/Documents/course/TORS_資料準備重做評估_20261008.pdf` | 0.3 MB | 第 3 節的評估與待決問題 |
| 重現狀態總表 | `~/Documents/course/TORS_重現狀態總表_20261007.pdf` | 0.8 MB | 46 項總表（項目編號以這份為準）與 v8 稿件對照 |
| 稿件一致性檢查報告 | `~/Documents/course/TORS_v8_與公開repo一致性檢查報告.pdf`、`TORS_v7_…pdf` | 3 MB | 稿件與 repository 的逐項檢查 |
| 稿件 | `~/Documents/course/ACM_TORS_English_v8_…docx`、`…v7_…docx` | 17 MB | 論文 docx 由老師維護，不要修改 |
| 指導老師的後續工作清單、早期報告、碩論終稿 | 原始開發 repository 工作目錄下的三份 PDF | — | 未納入任何 repository |
| 下載的資料與模型 | `_external/`（repository 內，不追蹤） | 5.5 GB | 不必帶走，用 `bootstrap_data.sh` 重新下載即可 |

## 8. 工作規則（從這次工作累積的約定）

- **原始資料夾 01～03**：不要修改學姊的原始研究資料夾 01～03（`verify_archived_sources.py` 會檢查）。
- **雜湊保護的檔案**：`results/README.md` 與 `results/SHA256SUMS*.txt` 列出的檔案都受雜湊清單保護，不要改。
- **商品照片**：Polyvore 商品圖，以及含商品照片的圖，只留在本機，不 commit、不散布。
- **要先確認的事**：
  - 推送前要先問；
  - 這台 GPU 也有其他人在用，長時間的 GPU 工作要先確認；
  - commit 時機由使用者決定。
- **不要邊跑邊改程式**：正在執行的 bash 腳本不要修改（bash 會邊讀邊執行）。
- **項目寫名稱**：給接手者看的輸出與文件，要直接寫項目名稱，例如「文字長度（表 5）」。不要只寫狀態總表的編號，接手者不一定有那份 PDF。
- **能重現就重現**：指導老師的原則是「能夠補的重現就要補」，2026-10-08 起資料準備也包含在內。
- **論文 docx 不要改**：它由老師維護；稿件需要修改的地方用報告列出。

## 9. 文件索引

- 完整流程與驗收：`reproduction/README.md`（第 0 節環境與執行、0.8 驗收與 46 項判定、§11 補充與延伸分析）
- 每項分析的說明：`docs/supplementary_analyses.md`、`docs/extensions.md`
- 限制與差異：`docs/known_limitations.md`
- 老師 2026-09-30 清單的完成狀態：`docs/tors_followup_status.md`
- TORS 投稿的 artifact 要求：`docs/tors_compliance.md`
- 稿件對帳表（對照碩論與 v4 稿）：`docs/manuscript_reconciliation/reconciliation.md`
- 正式 run 的驗收：`docs/clean_room_acceptance.md`
