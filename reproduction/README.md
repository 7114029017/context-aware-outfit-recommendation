# TORS thesis reproduction artifact

本資料夾是碩士論文《情境感知驅動的智慧穿搭推薦系統》與其 TORS 投稿稿的 2026 可重現性整理層。

目的不是改寫學姊原始研究資料，而是在保持原研究目錄不變的前提下（公開版只移除第三方文件、審查中的
稿件與碩論、3 張含 Polyvore 商品照片的圖，並去除 3 個筆記本內嵌的商品圖片，見
`docs/public_release_cleanup.md`），提供：

- 可執行的資料、環境與訓練檢查流程；
- 固定的 train / validation / test / CP / FITB / OR ID；
- 五個固定模型訓練 seeds；
- CP、FITB、OR 與 Judge 設定；
- 一鍵完整重現入口；
- seed-level raw results 與 summary；
- 正式 run 的結果與可核對證據；
- 對無法恢復的歷史來源限制做明確標示。

本 README 的用語：

| 用語 | 意思 |
|---|---|
| 正式 run | `full_20261002T161428Z`，修訂稿與本 README 的正式數值來源；稿件中稱為 frozen reference run |
| 你的 run | 依 §0 執行產生的新 run，位於 `reproduction/runs/full_<UTC 時間>/` |
| reference run | 較早的 `reference_20260921T175217Z`，與正式 run 逐位元相同，只作佐證 |
| 碩論／稿件 | 「碩論」指碩士論文，「稿件」指 TORS 投稿稿；「原論文」指 Hybrid Attention 原作者的論文（Wang & Zhong, 2024） |
| 表號 | 本 README 沿用碩論的表號（表 4-11 等）；與 TORS 稿件表號的對照見 §11 |

## 0. Clean-room full reproduction（接手者請先看）

這一節是**從全新 clone 開始的交接流程**，請依 0.0～0.8 的順序執行。若目標是驗證
「接手者只看本 README，能否從乾淨環境完成一次新的 35-unit reproduction」，請不要複製舊的
`.venv-repro`、`_external/` 或 `reproduction/runs/`。

稿件引用的固定版本是 Release 標籤：

`v1.0.0-tors-reproduction`

`main` 是之後可能繼續更新的分支。完整重現的啟動程式開始訓練前會檢查兩件事：
目前在 `main` 分支或 `v*-tors-reproduction` 標籤上，以及學姊原始研究資料夾 01～03 的
每個檔案都與 `reproduction/environment/archived_sources_manifest.json` 記錄的大小和
SHA-256 一致。任一項不符就拒絕開始訓練；第二項可在 0.5 先單獨檢查。

### 0.0 需求

| 項目 | 需求 |
|---|---|
| 作業系統 | Linux。正式 run 在 aarch64 的 NVIDIA GB10 上執行 |
| GPU | NVIDIA GPU，驅動需能執行 CUDA 13.0 版的 PyTorch（可用 `nvidia-smi` 確認）。訓練使用一張 GPU（`cuda:0`）；有多張 GPU 時用 `CUDA_VISIBLE_DEVICES` 指定。GB10 的 GPU 與 CPU 共用約 120 GB 記憶體；峰值 GPU 記憶體用量沒有記錄 |
| Python | **必須是 3.12**（正式 run 為 3.12.3），並含 venv 模組（Ubuntu／Debian：`sudo apt install python3.12-venv`）。其他版本無法安裝固定的套件版本：`scipy==1.18.1` 需要 3.12 以上，`numpy==2.0.2` 沒有 3.13 的套件 |
| Git | `git` 與 Git LFS（Ubuntu／Debian：`sudo apt install git-lfs`） |
| 網路 | github.com（含 Git LFS）、pypi.org、download.pytorch.org、huggingface.co |
| 磁碟 | 至少 20 GB：repo 與特徵檔約 5 GB、Python 環境約 5 GB、Polyvore metadata 約 140 MB、一次完整 run 約 2 GB，訓練期間另有暫存檔；延伸分析與補充分析用的圖片、FashionCLIP 與 Nomic 模型另需約 5.5 GB（0.4） |
| 時間 | clone 與 Git LFS 約 3 分鐘、安裝套件約 2 分鐘、下載資料與預檢各數秒（含圖片、FashionCLIP 與 Nomic 模型約 3 分鐘）、smoke test 約 40 秒；完整 35 組訓練在 GB10 上約 50～55 小時，之後的延伸分析約 3～4 小時 |

`sudo` 只有安裝系統套件與 0.6 的清除快取指令（選用）需要。

### 0.1 Clone 稿件引用的 Release

    cd /path/to/your/workspace

    git clone \
      --branch v1.0.0-tors-reproduction \
      https://github.com/7114029017/context-aware-outfit-recommendation.git

    cd context-aware-outfit-recommendation

    git describe --tags
    git rev-parse HEAD
    git status --short

預期 `git describe --tags` 顯示 `v1.0.0-tors-reproduction`（Git 會提示目前是 detached
HEAD，這是正常的），`git status --short` 沒有輸出。開始完整重現前，tracked working tree
不應有自行修改的學姊原始研究程式。
請用 `git clone`：Release 頁面的原始碼壓縮檔不是 git repository，也不含 Git LFS 的
特徵檔。本 repository 的版本來源見根目錄 `README.md` 的「版本來源」。

### 0.2 取得 Git LFS feature artifacts

模型訓練除了 Polyvore annotation / metadata 外，還需要 repository 中保存的
8 份 precomputed feature artifacts（共約 2 GB）。安裝 Git LFS（見 0.0）後執行：

    git lfs install
    git lfs pull
    git lfs ls-files

`git lfs ls-files` 應列出 8 個 `.pkl` 檔，每行中間是 `*`（已下載；`-` 表示仍是 pointer）。
0.5 的 `check.sh` 會再對這 8 份檔案做 byte-size 與 SHA-256 驗證；若仍是 LFS pointer、
檔案缺失或 hash 不符，不要開始完整重現。

### 0.3 建立新的 Python environment

正式 run 的執行環境為：

- Python 3.12.3
- PyTorch 2.9.1+cu130
- torchvision 0.24.1
- CUDA build 13.0
- cuDNN 9.13.0
- GPU: NVIDIA GB10

在 repository 根目錄建立新的 venv（必須是 Python 3.12，見 0.0）：

    python3.12 -m venv .venv-repro
    source .venv-repro/bin/activate
    python --version
    python -m pip install --upgrade pip

`python --version` 應顯示 `Python 3.12.x`。之後每開一個新的 terminal（包括 0.7 的 tmux），
都要先在 repository 根目錄執行 `source .venv-repro/bin/activate`。

若目標機器可使用 CUDA 13.0 wheel，可用下列版本組合盡量貼近正式 run：

    python -m pip install \
      torch==2.9.1 \
      torchvision==0.24.1 \
      --index-url https://download.pytorch.org/whl/cu130

若目標 GPU / driver 需要不同 CUDA-enabled PyTorch build，請安裝與該機器相容的
PyTorch / torchvision；不要用 CPU-only PyTorch 進行完整重現。
任何 runtime drift 都應保留在 environment evidence，不要為了對齊歷史版本而
偷偷替換或隱藏。

安裝其餘 reproduction dependencies：

    python -m pip install \
      -r reproduction/environment/requirements-reproduction-runtime.txt

快速確認 CUDA：

    python - <<'PY'
    import torch, torchvision
    print("torch:", torch.__version__)
    print("torchvision:", torchvision.__version__)
    print("cuda available:", torch.cuda.is_available())
    print("torch cuda:", torch.version.cuda)
    print("cudnn:", torch.backends.cudnn.version())
    print("gpu:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NONE")
    PY

開始完整重現前，`cuda available` 必須為 `True`（0.5 的 `check.sh` 只記錄 CUDA 狀態，
沒有 CUDA 也不會失敗）。在 NVIDIA GB10 上，PyTorch 會警告 GPU 的 compute capability 12.1
超出這個版本支援的範圍（8.0～12.0）；正式 run 也有同樣的警告，可以忽略。

### 0.4 重新下載 Polyvore annotations / metadata

不要從舊工作目錄複製 dataset。在 0.3 的環境中，由本 artifact 的 bootstrap 重新下載：

    bash reproduction/scripts/bootstrap_data.sh --with-images --with-fashionclip --with-nomic --with-compendium

來源是公開的 Hugging Face dataset `Stylique/Polyvore`
（https://huggingface.co/datasets/Stylique/Polyvore ，不需要帳號）。不加選項時只下載約
140 MB 的 annotation / metadata，35 組訓練只需要這些，預設存到：

`_external/uiuc-polyvore-hf/`

並記錄 local path 至：

`reproduction/.local/polyvore_root.txt`

四個選項是 35 組之後的延伸分析與補充分析（§11、`docs/extensions.md`、`docs/supplementary_analyses.md`）
要用的，下載固定版本並核對 SHA-256，全部存在 repo 之外：

- `--with-images`：同一個 dataset 的商品圖片（`images.zip`，2.5 GB；核對 SHA-256 後解壓到
  `_external/uiuc-polyvore-hf/images/`，約 2.6 GB）。色彩分析與附錄圖 A1～A3 使用。圖片只留在
  本機，**不 commit、不重新散布**。
- `--with-fashionclip`：FashionCLIP 模型（`patrickjohncyh/fashion-clip`，固定為學姊筆記本 P16
  使用的版本，約 610 MB），存到 `_external/fashion-clip/`，路徑記錄在
  `reproduction/.local/fashionclip_root.txt`。反事實分析用它編碼改寫後的描述。
- `--with-nomic`：文字嵌入模型 `nomic-ai/nomic-embed-text-v2-moe`（約 1.9 GB）與它的模型程式碼
  （`nomic-ai/nomic-bert-2048`，固定為學姊筆記本 P05 記錄的版本），存到 `_external/`。檢核清單
  涵蓋度（論文圖 4-2）使用。
- `--with-compendium`：官方的 2024 Adult Compendium of Physical Activities（PDF，0.6 MB，
  pacompendium.com；與學姊交接資料中的檔案相同），存到 `_external/compendium/`。MET 對照表的
  重建使用，讀取 PDF 需要 `pdftotext`（poppler-utils）。

不加這些選項時，對應的分析會略過，其他步驟不受影響。

成功時會顯示 `[READY] POLYVORE_ROOT=…`。要存到其他位置，可把路徑當成第一個參數；
要改用其他鏡像，可設定環境變數 `POLYVORE_HF_REPO`。下載固定在 dataset 的版本 `15d6c58`
（可用 `POLYVORE_HF_REVISION` 改）；這個版本的 12 個 annotation / metadata 檔與正式 run
使用的完全相同。

模型訓練使用的 precomputed image/text features 來自 0.2 的 Git LFS artifacts，不需要圖片。

### 0.5 Clean preflight

在 0.3 的環境中執行：

    bash reproduction/scripts/check.sh
    python3 reproduction/scripts/verify_archived_sources.py

`check.sh` 不做訓練。它記錄 Python 套件與 CUDA 狀態（套件缺失才會失敗；沒有 CUDA 不會
失敗，請以 0.3 的指令確認 GPU），並檢查 8 份 feature artifacts、dataset scope，以及
6 份 main ordered split manifests 的 byte identity。成功時會看到 8 份特徵檔各一行 `[OK]`、
6 行 `[SPLIT MATCH]`，最後是 `[DONE] preflight only -> …`。環境紀錄中的
`imports_ok_but_version_drift` 是正常的：PyTorch 回報 `2.9.1+cu130`，歷史紀錄寫的是
`2.9.1`（見 `environment/README.md`）。

`verify_archived_sources.py` 逐檔核對 01～03 資料夾，成功時顯示
`[ARCHIVED SOURCES] 278 files in folders 01-03 match archived_sources_manifest.json`。
0.7 開始訓練前會再核對一次。

check 的輸出預設位於：

`reproduction/runs/check_<UTC timestamp>/`

若檢查失敗，請先保留輸出與錯誤訊息；不要修改學姊 `01`～`03` 原始研究
目錄來讓檢查通過。

### 0.6 建議先跑 smoke test

在開始約兩天的完整訓練前，建議在 0.3 的環境中執行：

    bash reproduction/scripts/smoke_test.sh

Smoke test 只驗證小型 subset 的 CP -> CIR 執行鏈（約 40 秒），不可將其數值與正式 run
的結果比較。成功時最後顯示 `[DONE] smoke reproduction -> …`。

成功的 smoke 輸出預設位於：

`reproduction/runs/smoke_<UTC timestamp>/`

NVIDIA GB10 這類 CPU／GPU 共用記憶體的機器：CUDA 只把真正空閒的記憶體算作可用，
不含系統檔案快取。若沒有其他 GPU 程式在跑，smoke test 卻出現
`CUDA error: out of memory`，請先釋放檔案快取（只清快取，不影響資料），
再重跑 smoke test；完整重現開始前也建議先清一次：

    sudo sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'

### 0.7 從頭執行完整 35-unit reproduction

完整訓練約需兩天，請在斷線也不會結束的 session（例如 tmux）中執行，並在該 session 內
啟用 0.3 的環境：

    tmux new -s repro
    cd /path/to/your/workspace/context-aware-outfit-recommendation
    source .venv-repro/bin/activate
    bash reproduction/scripts/reproduce_all.sh --fresh

正式 run 就是這樣啟動的（`results/raw/full_20261002T161428Z/clean_room/cleanroom_launch.sh`）。
暫時離開 tmux 按 `Ctrl-b` 再按 `d`，之後用 `tmux attach -t repro` 回來。

`--fresh` 會真正重新執行：

- main Original × seeds 1,2,3,4,5；
- main Context × seeds 1,2,3,4,5；
- 公平子集消融的 Original / Context / No-Weather / No-Occasion / No-Style
  × seeds 1,2,3,4,5；
- 共 10 main + 25 公平子集消融 = **35 training units**；
- 後續 secondary analyses、Chapter 4 report、statistics、
  `THESIS_TABLES_SUMMARY.md` 與 final reproduction summary；
- 狀態成為 `PASSED` 之後，再執行補充分析（只用 CPU，一分鐘內；不改變 `RUN_STATUS.txt`），
  結果在 run 資料夾的 `supplementary/`（見 §11）；
- 接著執行延伸分析：用這次 run 的模型重做文字長度、反事實分析、Two-Tower、色彩分析與附錄圖
  （GPU 約 3～4 小時；同樣不改變 `RUN_STATUS.txt`），結果在 `extensions/`（見 §11）。

開始時會顯示 `[RUN ROOT]`（這次 run 的資料夾，預設為
`reproduction/runs/full_<UTC timestamp>/`）與 `[LIVE LOG]`。要看進度，另開一個 terminal：

    tail -f reproduction/runs/full_<UTC timestamp>/logs/full_console.log

可選參數（轉給 `pipeline/launch_full.sh`）：`--run-id NAME` 自訂 run 名稱、
`--output-base PATH` 把 run 放到其他位置、`--polyvore-root PATH` 指定 Polyvore 資料位置。

執行中會出現、屬於正常的訊息：

- GB10 的 compute capability 警告（見 0.3），每次載入 CUDA 都會出現；
- `judge_exact_generation` 與 `case_visuals` 標示為 `BLOCKED` / `PARTIAL`：流程不重跑 LLM、
  也沒有 Polyvore 圖片，這是預期的狀態（`docs/known_limitations.md`）。

流程不能從中斷處接續。若程式中斷（例如機器重新開機），該 run 的 `RUN_STATUS.txt` 會停在
`RUNNING`；請保留該資料夾，再執行一次 `--fresh`（會建立新的 run 資料夾並重新訓練）。

正式 run 的 wall time 為 55:07:40（同一台機器另有其他工作；先前兩次完整 run 為
49:42:27 與 49:09:31）；實際時間會依 GPU / runtime 而變化。

### 0.8 成功驗收條件

完整流程結束後，不要只看 terminal 最後一行。在 0.3 的環境中，先找到你的 run（0.7 開始時
顯示的 `[RUN ROOT]`；使用預設位置時，也可以用下列指令找最新的一次）：

    RUN="$(find reproduction/runs -maxdepth 1 -type d -name 'full_*' | sort | tail -n 1)"
    echo "$RUN"
    cat "$RUN/RUN_STATUS.txt"

| `RUN_STATUS.txt` | 意思 |
|---|---|
| `PASSED` | 全部完成，繼續下面的檢查 |
| `FAILED …` | 失敗；請保留整個 run 資料夾與 `logs/full_console.log` |
| `RUNNING`，但已沒有程式在跑 | 中斷；見 0.7 |

失敗時不要臨時改學姊原始程式或人工搬結果來把該 run 改成成功；先將失敗視為
reproduction artifact / environment 的可追查問題。

`PASSED` 時，確認關鍵 final outputs：

    test -f "$RUN/reproduction_summary.json" && echo "summary OK"
    test -f "$RUN/chapter4/chapter4_report.json" && echo "chapter4 OK"
    test -f "$RUN/chapter4/THESIS_TABLES_SUMMARY.md" && echo "table report OK"
    test -f "$RUN/logs/final_terminal_summary.txt" && echo "terminal summary OK"

可重新顯示 terminal 摘要（只讀檔案）：

    python3 reproduction/scripts/show_reproduction_summary.py \
      --run-root "$RUN"

摘要列出主實驗（表 4-11、4-12）的五 seed 平均與碩論值、R@1 與 R@3 的檢定、公平子集消融的
五 seed 摘要、Weather / Occasion 的方向比較、第四章 20 張表的涵蓋數，以及一份固定的 10 項
證據位置清單（指導老師 2026-09-22 的檢查清單）。逐表的來源與限制在
`chapter4/THESIS_TABLES_SUMMARY.md`。

**與正式 run 比對（驗收標準）。** terminal 摘要拿碩論的數字對照；驗收要確認的是你的 run
與正式 run `full_20261002T161428Z` 的結論是否一致：

    python3 reproduction/scripts/compare_with_official_run.py --run-root "$RUN"

工具會並排列出主實驗與公平子集消融的五 seed 平均，並逐條判定下列七條結論（判定方式與
`docs/manuscript_reconciliation/reconciliation.md` 的 R01～R07 相同）：

| 編號 | 結論 |
|---|---|
| R01 | 主實驗八項指標的平均值全部改善 |
| R02 | 八項中七項經 BH 校正後顯著 |
| R03 | Recall@1 有平均改善，但未達校正後顯著 |
| R04 | Recall@3 仍達校正後顯著 |
| R05 | Style 是 CP 與 OR 中最大且最穩定的貢獻因素 |
| R06 | Weather 與 Occasion 的 OR 效果較小，且對 seed 敏感 |
| R07 | 不把 Weather 與 Occasion 寫成在所有 OR 指標都有穩定的正貢獻 |

判定在以「判定：」開頭的那一行：

| 判定 | 意思 |
|---|---|
| `IDENTICAL` | 95 個逐 seed 結果檔與正式 run 逐位元相同。同一台 NVIDIA GB10、同一 runtime 應得到此結果。 |
| `CONSISTENT` | 數值不完全相同，但七條結論都一致。不同 GPU、CUDA 或 PyTorch build 時的預期結果。若有主實驗平均值與正式 run 相差超過 0.005，工具會另外註明，供記錄硬體與套件差異，不影響判定。 |
| `NOT CONSISTENT` | 至少一條結論不同，或公平子集、OR 題目與正式 run 不同。請保留整個 run 目錄，並在本 repository 開 GitHub issue 回報（附上 `official_comparison.txt`、`RUN_STATUS.txt`、硬體與套件版本）。 |

**驗收通過的條件：`RUN_STATUS.txt` 為 `PASSED`，且判定為 `IDENTICAL` 或 `CONSISTENT`。**
R02、R03 取決於 Recall@1 的 BH 校正後 p 值（正式 run 為 0.0947）；在其他硬體上若只有這兩條
不同，通常是這個 p 值跨過 0.05，回報時請附上它。

加 `--checkpoints` 會再核對 70 個 checkpoint 的 SHA-256。報告同時存成
`$RUN/official_comparison.txt` 與 `$RUN/official_comparison.json`。

**完整結果在哪裡。** 完整統計在 run 目錄的檔案裡（路徑相對於 `$RUN`；表號與稿件的對照見 §11）：

| 內容 | 檔案 |
|---|---|
| 表 4-11、4-12（mean ± SD、Δ、95% CI、BH 校正後 p、Cohen's dz） | `statistics/table_4_11_cp_main.csv`、`statistics/table_4_12_or_main.csv` |
| 8 項主要指標的 paired t-test 與 BH 明細 | `statistics/main_paired_bh_8metrics.csv`、`statistics/statistical_evidence.md` |
| 表 4-13、4-14（公平子集 Original vs Context） | `ablation/summary/T03_stage1_cp_original_vs_full.csv`、`ablation/summary/T04_stage1_cir_original_vs_full.csv` |
| 表 4-15、4-16（消融） | `ablation/summary/T05_stage2_cp_ablation.csv`、`ablation/summary/T06_stage2_cir_ablation.csv` |
| 消融的 95% CI、p 值、Cohen's dz | `ablation/summary/T10_stage2_cp_seed_detail.csv`、`ablation/summary/T11_stage2_cir_seed_detail.csv` |
| 每個 seed 的結果 | `main/<variant>_seed<N>/evaluation/results_cp.csv`、`results_cir.csv`；`ablation/summary/fresh_25unit_metrics.csv` |
| 第四章 20 張表的來源與限制 | `chapter4/THESIS_TABLES_SUMMARY.md` |
| 補充分析：類別、情境子集、因子與詞彙的效果（論文圖 4-8～4-13）、個案名次、因子效果、BH 族群、輸入資料稽核（含表 D-3、5.2 節的類別門檻）、稿件 Table 1、2、5、8、9 的重算、評分者與人工稽核（圖 4-1～4-4）、MET 對照表（表 3-1） | `supplementary/`（見 §11） |
| 延伸分析：文字長度、反事實分析（表 4、9）、Two-Tower（表 8）、色彩分析、附錄圖 A1～A3 與論文圖 4-14 | `extensions/`（各步驟狀態在 `extensions/EXTENSIONS_STATUS.txt`；見 §11） |
| 所有輸出的索引 | `INDEX.md` |

正式 run 的同一批檔案已收錄在 repository，數值整理在 `docs/expected_results.md`。
收錄後的位置：

| run 目錄中 | 收錄位置 |
|---|---|
| `statistics/`、`chapter4/`、`secondary/` | `results/summary/full_20261002T161428Z/` 下的同名資料夾 |
| `main/summary/` | `results/summary/full_20261002T161428Z/main/` |
| `ablation/summary/` | `results/summary/full_20261002T161428Z/ablation/` |
| `main/<variant>_seed<N>/evaluation/results_cp.csv`、`results_cir.csv` | `results/raw/full_20261002T161428Z/main/<variant>_seed<N>/` |
| `ablation/runs/<variant>_seed<N>/` | `results/raw/full_20261002T161428Z/ablation/<variant>_seed<N>/` |
| `supplementary/` | `results/supplementary/`（`run_analyses/` 收錄為 `full_20261002T161428Z/`；見 §11） |
| `extensions/` | `results/extensions/full_20261002T161428Z/`（見 §11） |

補充分析與延伸分析在 `PASSED` 之後自動執行，不屬於驗收條件：若失敗，terminal 會顯示
`[WARN]` 與重跑指令，`RUN_STATUS.txt` 仍是 `PASSED`（見 §11）。

### 0.9 目前 clean-room 驗證狀態

**正式 run（修訂稿唯一的數值來源）**：從原始開發 repository（不公開，branch
`thesis-full-reproduction`）全新 clone，依當時的本節流程建立新 environment 後執行的
clean-room run（版本關係見根目錄 `README.md` 的「版本來源」）：

- run ID：`full_20261002T161428Z`
- commit：`b9bf5aac7d07cb39cacc7f73eb28166848c19fe4`
- 2026-10-02T16:14:28Z → 2026-10-04T23:22:08Z（55:07:40）
- `RUN_STATUS.txt = PASSED`
- 與下列兩次 run 比對：70 / 70 checkpoint SHA-256、70 / 70 seed-level result
  CSV、25 / 25 fair-subset detail CSV 完全相同；所有 summary 數值一致

驗收紀錄見 `reproduction/docs/clean_room_acceptance.md`；結果收錄於
`results/raw/full_20261002T161428Z/`、`results/summary/full_20261002T161428Z/`
與 `results/final_reference_manifest.json`。

先前兩次完整 run 作為佐證：reference run `reference_20260921T175217Z` 已完成 35 units
並為 `PASSED`，該次執行完成於 repository restructuring 之前。

重構後已依當時的本節流程以 `bash reproduction/scripts/reproduce_all.sh --fresh`
完成第二次全新的 35-unit end-to-end run：

- run ID：`full_20260924T102513Z`
- commit：`2cd8d04694ef314955be48ba47689e98a4b13d35`
- 2026-09-24T10:25:13Z → 2026-09-26T11:34:44Z（49:09:31）
- `RUN_STATUS.txt = PASSED`
- 與 reference run 比對：70 / 70 checkpoint SHA-256、95 / 95 seed-level
  result CSV SHA-256 完全相同；main / ablation / statistics / Chapter 4
  數值輸出一致（同一台 NVIDIA GB10、同一 runtime）

逐項證據見 `reproduction/docs/post_restructure_fresh_run_verification.md`。
不同 GPU / CUDA / PyTorch build 不預期 bit-identical；此時依 §0.8 的驗收條件判讀
（`PASSED`，且比對結果為 `IDENTICAL` 或 `CONSISTENT`）。

## 1. Provenance boundary

學姊原始研究內容位於 repository 根目錄的：

- `01_資料建構_data_construction/`
- `02_模型訓練和驗證_model_training_validation/`
- `03_實驗與結果_experiments_results/`

2026 reproduction artifact 不修改這三個資料夾。公開版只做了下列改動：移除第三方文件與
3 張含 Polyvore 商品照片的圖（F21、F34a、F34b），並去除 3 個筆記本內嵌的 Polyvore 商品圖片；
原本的 `04_文件資料_documents/` 只有審查中的稿件與碩論，因此不在公開版中。完整清單見
`docs/public_release_cleanup.md`。

Frozen senior baseline：

`7a5cc9cd8f884865e1e27a234c18dd132f949598`

此 commit 屬於原始開發 repository，不在本 repository 的歷史中，也是正式 run 使用的內容。
01～03 公開版的內容記錄於 `reproduction/environment/archived_sources_manifest.json`，
完整重現開始前由 `reproduction/scripts/verify_archived_sources.py` 逐檔核對。

## 2. Official run and earlier reference runs

修訂稿的正式結果來自 clean-room run `full_20261002T161428Z`（commit `b9bf5aa`，
55:07:40，`PASSED`），細節見 §0.9 與 `docs/clean_room_acceptance.md`。若目的只是檢查已完成的
重現結果，不需要重新訓練，直接看這些檔案：

- raw results：`results/raw/full_20261002T161428Z/`
- summaries 與碩論表格（表 4-11～4-16）：`results/summary/full_20261002T161428Z/`
- seed index（每個 unit 的 checkpoint 雜湊、最佳 epoch、runtime）：
  `results/summary/full_20261002T161428Z/seed_index.csv`
- final manifest：`results/final_reference_manifest.json`
- 雜湊清單：`results/SHA256SUMS_full_20261002T161428Z.txt`
- 數值整理：`docs/expected_results.md`

較早的兩次完整 run 與正式 run 逐位元相同，作為佐證：

| run | training commit | 時間 | 說明 |
|---|---|---|---|
| `reference_20260921T175217Z` | `eb081e7a426cf5bd91acd8bb0f92a8398f08f45e` | 2026-09-21T17:52:17Z → 2026-09-23T19:34:44Z（49:42:27） | repository 重構前的 reference run（35 units，`PASSED`）；結果收錄於 `results/raw/reference_20260921T175217Z/` 與 `results/summary/reference_20260921T175217Z/`（seed index：`reference_seed_index.csv`） |
| `full_20260924T102513Z` | `2cd8d04694ef314955be48ba47689e98a4b13d35` | 2026-09-24T10:25:13Z → 2026-09-26T11:34:44Z（49:09:31） | 重構後的 `--fresh` run；證據見 `docs/post_restructure_fresh_run_verification.md` |

## 3. Repository contents

| Path | Purpose |
|---|---|
| `configs/` | CP、FITB、OR、Judge 與 shared model 設定的說明（說明用，流程不讀取，見 §7） |
| `data/` | 資料來源、取得方式與 provenance |
| `environment/` | runtime requirements、正式 run 與 reference run 的環境紀錄、01～03 的檔案清單 |
| `splits/` | 固定 main IDs 與重建的公平子集 IDs |
| `prompts/` | 可保存的 semantic prompt 與 Judge C/C* checklists |
| `scripts/` | check、smoke、full reproduction、統計與比對程式 |
| `scripts/supplementary/` | 補充分析程式（35 組訓練完成後執行，見 §11） |
| `scripts/extensions/` | 延伸分析程式（35 組訓練完成後執行，見 §11 與 `docs/extensions.md`） |
| `results/raw/` | 正式 run 與 reference run 的 seed-level results |
| `results/summary/` | 正式 run 與 reference run 的 summary、第四章比較與 statistics |
| `results/supplementary/` | 補充分析的輸出（見 §11） |
| `results/extensions/` | 正式 run 的延伸分析輸出（見 §11） |
| `results/tuning.csv` | 可恢復的 final-setting tuning provenance |
| `docs/` | provenance、expected results、稿件對帳表、TORS compliance 與限制 |
| `runs/` | 新執行產物；Git ignored。以 `check_` / `smoke_` / `full_` 區分執行模式 |

## 4. Data access

詳細資料 provenance：

`reproduction/data/README.md`

從乾淨環境下載所需 annotation / metadata（見 §0.4）：

    bash reproduction/scripts/bootstrap_data.sh --with-images --with-fashionclip --with-nomic --with-compendium

來源是 Hugging Face 的 `Stylique/Polyvore`；可用環境變數 `POLYVORE_HF_REPO` 改用其他鏡像。

`--with-images` 下載的 Polyvore 圖片只供本機的延伸分析使用，不 commit、不重新散布。

模型使用的 preserved feature artifacts 位於學姊原始模型資料夾，
並由 SHA-256 / byte-size verifier 在執行前檢查。

若只查看已收錄的正式 run 結果，不需要重新下載或重新訓練。

## 5. Environment setup

完整步驟見 §0.0 與 §0.3（必須使用 Python 3.12）。其他說明：

`reproduction/environment/README.md`

正式 run 的環境證據保存在：

`reproduction/environment/full_20261002T161428Z/`

reference run 的環境證據在 `reproduction/environment/reference_20260921T175217Z/`。

目前已知的歷史限制是：2026 runtime 並不能證明與 2025 原始環境完全相同。

## 6. Fixed IDs

Main manifests：

- `splits/train_ids.csv`
- `splits/validation_ids.csv`
- `splits/test_ids.csv`
- `splits/cp_ids.csv`
- `splits/fitb_ids.csv`
- `splits/or_ids.csv`

Main scope：

| Scope | Count |
|---|---:|
| train | 16,995 |
| validation | 3,000 |
| test | 15,145 |
| CP test pairs | 30,290 |
| FITB questions | 15,145 |
| OR evaluable pairs | 9,311 |

Main manifests 有固定 SHA-256，clean preflight 會重新建立並做 byte-for-byte
identity check。

公平比較子集（表 4-13～4-16 使用）IDs 位於：

`splits/fair_subset/`

| Scope | Count | 檔案 |
|---|---:|---|
| total | 21,903 | `fair_subset_ids.txt` |
| train | 10,225 | `train_ids.txt` |
| validation | 1,748 | `valid_ids.txt` |
| test | 9,930 | `test_ids.txt` |
| OR（CIR）evaluable queries | 3,432 | `or_query_ids.csv` |

這是修訂稿正式使用的 reproducibly reconstructed fair subset：由天氣、場合、
風格三個因子皆具備的樣本組成，每次完整重現都會重新建立並驗證。建構規則、
排除條件、產生程式與 checksum 見 `docs/fair_subset.md`。

歷史原始 memberwise fair-subset ID file 未被恢復，因此不宣稱與歷史 subset
逐筆完全相同。

## 7. Experiment configuration

設定說明（說明用的 YAML，記錄設定與證據來源；完整重現的流程不讀取它們，修改也不會影響訓練）：

- `configs/cp.yaml`
- `configs/fitb.yaml`
- `configs/or.yaml`
- `configs/main_hybrid_attention.yaml`
- `configs/llm_judge.yaml`

實際的訓練設定來自學姊原始程式
`02_模型訓練和驗證_model_training_validation/main_hybrid_attention_code/config/base_config.py`
與訓練程式。

Main fresh reproduction seeds：

`1, 2, 3, 4, 5`

seeds 由流程固定：主實驗在 `scripts/pipeline/reproduce_main.sh`（`SEEDS="1 2 3 4 5"`），
公平子集消融在 `scripts/pipeline/reproduce_ablation.sh`（`for seed in 1 2 3 4 5`）；每次訓練依
seed 固定亂數並使用 deterministic cuDNN（`docs/hyperparameter_tuning.md`）。
人工稽核 seed `42` 與模型訓練 seeds 不同。

Evidence-backed main settings 包括：

- Hybrid Attention；商品圖片特徵為 FashionCLIP，商品文字特徵為 SentenceBERT
  （`distiluse-base-multilingual-cased-v2`），整套的文字條件為 FashionCLIP
- 100 epochs
- batch size 50
- Adam
- learning rate 5e-5
- weight decay 0
- gradient clipping 0.5
- dropout 0.1
- 16 attention heads
- 3 transformer layers
- float16 mixed precision
- 負樣本：CP 每個正例搭配 1 個負樣本套裝，CIR 每個目標搭配 10 個負樣本；從第 40 個 epoch
  起加入 hard negative（CIR 的 hard negative 因原始程式的寫法實際上沒有使用，見
  `docs/hyperparameter_tuning.md`）
- validation FITB accuracy checkpoint selection

OR / CIR 使用 matching same-seed CP best checkpoint 初始化，
並使用 MarginRankingLoss，margin = 0.3。

## 8. Quick preflight

即 §0.5：

    bash reproduction/scripts/check.sh

不會執行訓練；檢查項目與成功訊息見 §0.5。未指定 `--run-root` 時，輸出預設位於
`reproduction/runs/check_<UTC timestamp>/`。

## 9. Optional smoke test

即 §0.6：

    bash reproduction/scripts/smoke_test.sh

Smoke test 只用小型 subset（context、seed 1、200 筆 train 與 100 筆 validation 套裝）與
1 epoch，驗證 CP -> CIR pipeline 是否能運作；它的數值不可拿來與正式 run 的結果比較。
未指定 `--run-root` 時，輸出預設位於 `reproduction/runs/smoke_<UTC timestamp>/`。

## 10. Full reproduction

接手者的完整重現請照 §0.7：

    bash reproduction/scripts/reproduce_all.sh --fresh

流程依序執行：

1. environment / feature / split identity gates；
2. Original 與 Context main experiments，5 seeds each；
3. 公平子集的五種條件 × 5 seeds；
4. preserved-output secondary analysis；
5. Chapter 4 comparison report；
6. five-seed summary 與 post-hoc statistics；
7. 狀態成為 `PASSED` 之後的補充分析（只用 CPU，見 §11）；
8. 延伸分析（文字長度與 Two-Tower 用 GPU，其餘用 CPU，見 §11）。

輸出位於 `reproduction/runs/full_<UTC timestamp>/`（可用 `--run-id`、`--output-base` 改變）。
若完成為 `PASSED`，最後會從該次 run 的 `main/summary/`、`ablation/summary/`、`statistics/`、
`chapter4/` 自動顯示 terminal 摘要，並存成 `logs/final_terminal_summary.txt`，接著執行補充
分析（結果在 `supplementary/`，輸出記錄在 `logs/supplementary.log`）與延伸分析（結果在
`extensions/`，輸出記錄在 `logs/extensions.log`）。摘要顯示、補充分析或延伸分析失敗都不會
改變已完成訓練的 `RUN_STATUS.txt`；原始 CSV / JSON 仍是權威結果。跑完後依 §0.8 與正式 run
比對。

完整流程也會產生一份**表 4-1～表 4-20 的逐表來源報告**：

    chapter4/THESIS_TABLES_SUMMARY.md

此報告會對每一張表明確列出：

- 是否由 2026 fresh 重新訓練得到；
- 是否由學姊保存的 row-level / score / audit / case evidence 重新統計或查證；
- 保存資訊的來源性質；
- 本流程實際做了什麼、沒有做什麼；
- 對應的 machine-readable evidence 路徑；
- source-exact / historical provenance 的限制。

不加 `--fresh` 時，`reproduce_all.sh` 會先尋找原始開發機器上保存的 reference run；全新 clone
沒有它，所以同樣會開始全新的完整重現。這個重用模式只供維護者使用，見 §19。

修訂稿的正式結果是 clean-room run `full_20261002T161428Z`（見 §0.9、§2）。它與
restructuring 前完成的 reference run 以及重構後的 `--fresh` run `full_20260924T102513Z`
逐位元相同。

## 11. Expected results

碩論表號與 TORS 稿件表號的對照（依 `docs/manuscript_reconciliation/manuscript_values.csv`）：

| 碩論 | TORS 稿件 | 內容 |
|---|---|---|
| 表 4-11、4-12 | Table 6 | 主實驗（CP、OR） |
| 表 4-13～4-16 | Table 7 與 5.3 節 | 公平子集與 Weather／Occasion／Style 消融 |
| 表 4-3～4-10 | Table 5 的各列 | 代理值、文字長度、target clues、Judge 一致性與穩健性、人工稽核 |
| 4.1.1 節 | Table 1 | 資料規模 |
| — | Table 8、Table 9 | Two-Tower 與反事實分析：稿件使用學姊保存的輸出；延伸分析用正式 run 的模型重做（見本節最後） |

正式 run 的預期結果（表 4-11～4-16 的數值，碩論值列為歷史參照）：

`docs/expected_results.md`

正式 run 的主實驗結果與統計：

`results/summary/full_20261002T161428Z/main/`、`results/summary/full_20261002T161428Z/statistics/`

完整重現會自動產生碩論表 4-11～4-16，欄位與碩論相同：表 4-11～4-14 有 mean ± SD、Δ、
95% CI、BH 校正後 p、顯著性與 Cohen's dz；表 4-15、4-16 有 mean ± SD 與 Δ，消融的 95% CI、
p 值與 Cohen's dz 在 `T10`、`T11`（見 §0.8）。路徑相對於 run 目錄：

| 碩論表 | 檔案 |
|---|---|
| 4-11 CP 主實驗 | `statistics/table_4_11_cp_main.csv` |
| 4-12 OR 主實驗 | `statistics/table_4_12_or_main.csv` |
| 4-13 公平子集 CP | `ablation/summary/T03_stage1_cp_original_vs_full.csv` |
| 4-14 公平子集 OR | `ablation/summary/T04_stage1_cir_original_vs_full.csv` |
| 4-15 CP 消融 | `ablation/summary/T05_stage2_cp_ablation.csv` |
| 4-16 OR 消融 | `ablation/summary/T06_stage2_cir_ablation.csv` |

主實驗的 BH 校正以 8 項主要指標為同一族群；公平子集以表列 5 項指標為同一族群，
沿用學姊原表（`03_實驗與結果_experiments_results/03_主推薦任務結果/圖表_figures_tables/tables/`
的 `T03`、`T04`）的規則。收錄到 `results/summary/<run>/` 後，
`ablation/summary/` 會變成 `ablation/`；正式 run 的六張表都在
`results/final_reference_manifest.json` 的 `thesis_tables`。

Main reproduction 的 16 個 paper-comparison means（正式 run 主實驗 Original 與 Context 各 8 項
平均值，與碩論值比較）均落在預先使用的 absolute tolerance 0.005 內。

八個 Context-minus-Original main aggregate metric directions 均為正。

五-seed paired analysis 經八個 main metrics 的 Benjamini-Hochberg correction
後，7 / 8 metrics 達 p < .05；Recall@1 未達顯著。

這些統計是描述已完成 reproduction evidence，不應解讀為所有歷史來源已被
精確恢復。

碩論終稿與 TORS 投稿稿的逐項對帳表（每項列出稿件值、最新值、來源 CSV、
run ID 與程式位置，並標示需保留、更新數字或改寫的敘述）：

`docs/manuscript_reconciliation/reconciliation.md`

稿件數值轉錄於 `docs/manuscript_reconciliation/manuscript_values.csv`；
對帳表以正式 run 為來源，由下列指令產生：

    python3 reproduction/scripts/build_manuscript_reconciliation.py --official

要以另一個已完成的 run 產生對帳表時，用 `--run-root <run 目錄>` 並加上
`--out-dir <其他目錄>`，以免覆寫正式對帳表。

補充分析不屬於 35 個 training units：`reproduce_all.sh --fresh` 在 run 的狀態成為 `PASSED`
之後才執行它，它失敗也不會改變 `RUN_STATUS.txt`。它只讀取 run 已存下的結果、2025 年的
輸入資料、保存的輸出與 0.4 下載的檔案，只用 CPU，一分鐘內完成；只有檢核清單涵蓋度一項載入
文字嵌入模型。結果在 run 資料夾的 `supplementary/`：

| 資料夾 | 內容 | 正式 run 的對應位置 |
|---|---|---|
| `run_analyses/` | 由該 run 的逐題結果計算：類別、情境子集、因子與詞彙的效果（表 T12～T17，論文圖 4-7～4-13，2025 年圖 F03、F04）、個案名次、因子效果（表 T08、T09）、BH 族群的敏感度分析、主實驗的 Wilcoxon 檢定（表 A14） | `results/supplementary/full_20261002T161428Z/` |
| `input_data_audit/` | 輸入資料稽核，含 CLO 分布（表 D-3、圖 D-4）與類別門檻（5.2 節），與 run 無關 | `results/supplementary/input_data_audit/` |
| `paper_value_checks/` | 稿件 Table 1、2、5、8、9 的重算，與 run 無關 | `results/supplementary/paper_value_checks/` |
| `judge_audit_checks/` | 評分者分數分布、低分樣本敏感度、人工稽核的題項分歧與分數（圖 4-1、4-3、4-4，表 T30、T31、A41，2025 年圖 F18、F20、F30～F32）與評分結果未進入訓練的程式掃描（表 A05、A06），與 run 無關 | `results/supplementary/judge_audit_checks/` |
| `met_reference_check/` | 由官方 Compendium 依論文規則重建 457 筆 MET 對照表（表 3-1），與 run 無關；需要 `--with-compendium` | `results/supplementary/met_reference_check/` |
| `checklist_coverage/` | 兩組檢核清單的概念涵蓋度（圖 4-2），與 run 無關；需要 `--with-nomic` | `results/supplementary/checklist_coverage/` |
| `dataset_tables_check/` | 2025 年資料表 T00、T01、A01、A02、A12、A13 的重算（471 個數值），與 run 無關 | `results/supplementary/dataset_tables_check/` |

執行時的輸出記錄在 `logs/supplementary.log`。正式 run 執行時流程還沒有這一步，右欄的檔案
是之後用同一套程式從正式 run 的輸出算出的。§0.8 的判定為 `IDENTICAL` 時，你的
`supplementary/` 除了 `run_analyses/` 的 `summary.md` 與圖中的 run 名稱與來源，應與右欄完全相同；
判定為 `CONSISTENT` 時，`run_analyses/` 的數字會略有不同。其他資料夾與 run 無關，
應完全相同。核對方式（`$RUN` 見 §0.8）：

    diff -r "$RUN/supplementary/run_analyses" reproduction/results/supplementary/full_20261002T161428Z
    diff -r "$RUN/supplementary/input_data_audit" reproduction/results/supplementary/input_data_audit
    diff -r "$RUN/supplementary/paper_value_checks" reproduction/results/supplementary/paper_value_checks
    for d in judge_audit_checks met_reference_check checklist_coverage dataset_tables_check; do
      diff -r "$RUN/supplementary/$d" "reproduction/results/supplementary/$d"
    done

也可以單獨執行，例如自動執行失敗時（terminal 會顯示 `[WARN]` 與重跑指令），修正原因後
重跑。需要 0.3 的環境與 0.4 下載的 Polyvore metadata（或用 `--polyvore-root PATH` 指定）：

    bash reproduction/scripts/supplementary/run_all.sh --run-root "$RUN"

- `--run-root <run 目錄>`：分析該 run（`RUN_STATUS` 必須是 `PASSED`），結果寫在該 run 資料夾的
  `supplementary/`。
- `--official`：分析 repo 收錄的正式 run，結果寫在 `results/supplementary/`，會重新產生與 repo
  相同的檔案。
- 不加參數：分析 `reproduction/runs/` 中最新的 `full_*`；沒有任何 run 時（例如剛 clone），
  分析正式 run。
- `--out-dir DIR`：結果改寫到 `DIR`。

說明見 `docs/supplementary_analyses.md`。

延伸分析用 run 自己的模型重做稿件中原本依賴學姊保存輸出的分析，同樣在 `PASSED` 之後才執行、
不改變 `RUN_STATUS.txt`，也不更動 35 組用的程式。每個程式都由產生稿件數值的筆記本移植，
並先用學姊保存的輸出驗證移植無誤：

| 延伸分析 | 稿件 | 需要 | 正式 run 的結果 |
|---|---|---|---|
| `text_length` | Table 5「Text length」 | GPU，約 20 分鐘 | `results/extensions/full_20261002T161428Z/text_length/`（10 個主實驗模型的逐題結果在 `main_cir_per_query/`） |
| `counterfactual` | Table 4、Table 9 | `--with-fashionclip`；CPU 約 1 分鐘 | `results/extensions/full_20261002T161428Z/counterfactual/`（5 個 seed） |
| `two_tower` | Table 8 | GPU，1.5～3 小時（GB10 上約 63 分鐘） | `results/extensions/full_20261002T161428Z/two_tower/`（模型檔只放本機） |
| `color` | 5.5 節色彩分析 | `--with-images`；CPU 幾秒 | `results/extensions/full_20261002T161428Z/color_analysis/` |
| `figures` | 附錄圖 A1～A3、論文圖 4-14（顏色偏移案例）、2025 年反事實範例圖 F34a、F34b | `--with-images`；CPU 幾秒 | 圖只在本機；內容（商品 ID、名次、顏色）在 `case_figures/` 的三個 manifest |
| `reliability` | 2025 年可靠度分析（P04：表 T18、圖 F12～F16；稿件沒用） | GPU，約 5 分鐘 | `results/extensions/full_20261002T161428Z/reliability/` |
| `text_swap` | 2025 年文字互換評估（`CP_evaluate.py`、`CIR_evaluate.py --sweep`；稿件沒用） | GPU，約 45 分鐘 | `results/extensions/full_20261002T161428Z/text_swap/` |
| `outfit_generation` | 2025 年虛擬試穿示範的逐步選品（P02，不含試穿；稿件沒用） | `--with-images`、`--with-fashionclip`；CPU 約 2 分鐘 | `results/extensions/full_20261002T161428Z/outfit_generation/` |

run 資料夾中的結果在 `extensions/`，各步驟的狀態在 `extensions/EXTENSIONS_STATUS.txt`；
缺少 GPU 或下載檔的步驟會標為 `SKIPPED`，其他步驟照常執行。單獨執行或重跑（例如修正失敗原因後）：

    bash reproduction/scripts/extensions/run_all.sh --run-root "$RUN" [--steps text_length,two_tower] [--skip-gpu] [--validate]

`--validate` 另外用學姊保存的輸出檢查移植的程式（CPU 約 10 分鐘；有 GPU 時另外約 5 分鐘重算
表 T18、約 75 分鐘用學姊的 20 個主實驗 checkpoint 重算文字互換）。驗證結果、正式 run 的數值與
仍無法重做的項目，見 `docs/extensions.md`。

## 12. Hyperparameter tuning provenance

文件：

`docs/hyperparameter_tuning.md`

Machine-readable record：

`results/tuning.csv`

主模型（Hybrid Attention CP／CIR）的全部設定都是原論文（Wang & Zhong, IEEE Access 2024）
與原作者公開程式的設定：交接版本的
`02_模型訓練和驗證_model_training_validation/main_hybrid_attention_code/config/base_config.py`
與原作者 repository 完全相同，
訓練程式中所有超參數相關程式行也相同；十份歷史 CP 訓練紀錄都是同一組設定。
validation 只用來選每次訓練的最佳 epoch，沒有證據顯示做過超參數搜尋。

每個設定依老師的四類（依文獻固定、validation 選擇、歷史紀錄遺失、補充分析）
逐項分類並附來源。第二模型（Two-Tower）的設定值有保存，但選擇理由未記錄；
也尚未做任何補充敏感度分析。因此 `docs/tors_compliance.md` 的 A7（超參數調整紀錄）
維持 `PARTIAL`。

本 artifact 不把 final settings 偽裝成歷史 tuning trials。

## 13. LLM-as-a-Judge

Judge provenance：

`docs/judge_provenance.md`

Archived machine-readable C / C* checklists：

`prompts/checklists/`

已保存：

- Qwen3-VL-32B-Instruct C / C*
- Gemma-3-27B-it C / C*
- weights / categories
- full preserved formal Judge outputs in the senior data tree
- robustness outputs used by preserved-output analysis

未恢復：

- exact historical Judge model revision；
- complete exact P0 / P1 / P2 executable prompt text；
- all historical generation parameters。

因此 preserved-output analysis 可重算，但 exact historical LLM generation
rerun provenance 為 partial。

## 14. Standardized decoder implementation

學姊 archived source 的 CP decoder 引用未保存的：

`DecoderLayerWithCrossAttn`

其 import 已被註解、定義不在交接資料中，因此 archived 程式無法直接執行，
早期自訂 decoder 的 exact forward behavior 也無法恢復。

修訂稿與本 artifact 的全部實驗（主實驗訓練、評估、公平子集消融與 smoke test）
統一採用 **standardized decoder implementation**：`torch.nn.TransformerDecoderLayer`
（3 層、16 heads、dropout 0.1，d_model 128）。這正是原論文作者公開程式
（WangXin93/text-conditioned-outfit-recommendation）的寫法；交接版本把它停用，
改用定義已遺失的自訂 class，因此標準化實作等同還原原作者的公開模型。

實作集中在 `scripts/standard_decoder.py`，只修改 working copy；學姊 repository
source 本身不修改。方法與輸入輸出說明見 `docs/standardized_decoder.md`。

本 artifact 不宣稱恢復歷史 `DecoderLayerWithCrossAttn` 的行為。

## 15. Result integrity

正式 run：

- 雜湊清單：`results/SHA256SUMS_full_20261002T161428Z.txt`
- run identity、環境、切分 checksum、checkpoint 雜湊與程式雜湊：`results/final_reference_manifest.json`
- seed index：`results/summary/full_20261002T161428Z/seed_index.csv`

reference run：`results/SHA256SUMS.txt`（201 個檔案）、
`results/reference_20260921T175217Z_manifest.json` 與
`results/summary/reference_20260921T175217Z/reference_seed_index.csv`。

大型 checkpoint 沒有重複 commit 到 result tree；正式 run 的 70 個 checkpoint 以 seed index 中的
SHA-256 記錄（`compare_with_official_run.py --checkpoints` 可核對你的 run）。

## 16. TORS compliance status

目前狀態：

`docs/tors_compliance.md`

老師 2026-09-30 後續工作清單的逐項完成狀態與證據：

`docs/tors_followup_status.md`

Repository artifact 已完成核心 reproduction evidence 整理。稿件引用的固定版本是 GitHub
Release `v1.0.0-tors-reproduction`（不使用 DOI）。公開版的整理見
`docs/public_release_cleanup.md`，第三方條款見根目錄 `THIRD_PARTY_NOTICES.md`。
A7（超參數調整紀錄）維持 `PARTIAL`，見 §12。

## 17. Known limitations

本 artifact 不宣稱以下項目已 source-exact recovery：

- historical fair-subset memberwise identity；
- historical DecoderLayerWithCrossAttn implementation；
- historical hyperparameter-search trial records；
- exact historical 2025 runtime；
- complete historical Judge generation prompt/parameter provenance；
- every qualitative/image-dependent thesis output。

因此整體研究來源層級 provenance 應描述為 `partial`。

逐項清單（缺什麼、本 artifact 怎麼處理、證據位置）見 `docs/known_limitations.md`。

另一方面，2026 main fresh numerical reproduction、seed-level evidence、
公平子集消融 reproduction artifact 與 preserved-output analysis 都有明確
保存與來源界線。

## 18. Handoff rule

不要修改學姊 `01` 至 `03` 原始研究目錄來讓數字符合碩論或稿件。

新的實驗輸出一律放在：

`reproduction/runs/`

新的可重現性 artifact 一律放在：

`reproduction/`

不要使用 `git add .`；應精確 stage 要提交的 reproduction artifact。

## 19. 維護者專用

以下功能只在原始開發機器上，或要更換稿件的正式 run 時使用；接手者驗收不需要，也不要執行。

**重用已完成的 reference run。** 不加 `--fresh` 時，`reproduce_all.sh` 若找到
`_reproduction_runs/full_20260921T175217Z/`（只存在於原始開發機器），會先做嚴格驗證：

1. local `RUN_STATUS.txt = PASSED`；
2. committed reference seed index 必須恰好 35 rows；
3. 10 個 main + 25 個公平子集消融 identity 必須完整；
4. 70 個 CP/CIR checkpoint SHA-256 必須逐一對上 seed index；
5. `reproduction/results/SHA256SUMS.txt` 的 201 個 committed reference results 必須全部 byte-identical。

全部通過後顯示 `[SKIP VERIFIED FULL TRAINING]`，直接使用已保存的 reference raw / summary /
Chapter 4 / statistics，產生 `reproduction/runs/reference_20260921T175217Z/THESIS_TABLES_SUMMARY.md`，
並執行 `python3 reproduction/scripts/show_reproduction_summary.py --reference`。這個模式
**不會啟動 CP/CIR 訓練，也不宣稱第二次完成 35-unit training**。
`--reuse-reference PATH` 可指定另一份已完成的 run，但它的 70 個 checkpoint 必須與 reference
seed index 完全相同（只有同一硬體與 runtime 才可能），其他 run 會被拒絕。

**收錄新的正式 run。** 只有要更換稿件的正式 run 時才執行：

    python3 reproduction/scripts/package_official_run.py --run-root <run 目錄>

它會產生 `results/raw/<run>/`、`results/summary/<run>/`（含 `seed_index.csv`）、
`environment/<run>/` 與 `results/SHA256SUMS_<run>.txt`，並**覆寫**
`results/final_reference_manifest.json`（加上 `--verify-checkpoints` 會再重新計算 70 個
checkpoint 的 SHA-256）。接手者不要對自己的 run 執行：覆寫後，`compare_with_official_run.py`
會把你的 run 當成正式 run，變成自己跟自己比對。
