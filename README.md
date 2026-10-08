# Context-Aware Outfit Recommendation

Reproducibility artifact for the manuscript *Context-Aware Semantic Construction
for Multimodal Outfit Recommendation* (submitted to ACM TORS). The version cited
by the manuscript is the release **`v1.1.0-tors-reproduction`**, published after
the acceptance of a new official run made with one command; this version is its
release candidate **`v1.1.0-rc1-tors-reproduction`**. The release published
before it is `v1.0.0-tors-reproduction`.

**Polyvore images are not redistributed due to third-party rights.** The
repository provides the instructions and scripts for obtaining the Polyvore
Outfits metadata and splits, together with fixed identifiers, derived
annotations, precomputed features, configurations and result outputs. See
`THIRD_PARTY_NOTICES.md`.

## English quick start

What the release contains:

- `reproduction/`: environment files, data preparation, fixed splits and
  fair-subset IDs, configurations, the training, evaluation and statistics
  scripts, the official run's seed-level results and summaries, the acceptance
  report and the provenance documents.
- `01_…`, `02_…`, `03_…`: the original 2025 research materials (data
  construction, model code, experiment outputs, derived annotations). The
  changes made for the public release are listed in
  `reproduction/docs/public_release_cleanup.md`.

Reproducing the 35 training units (10 main units and 25 fair-subset ablation
units, seeds 1–5) needs Linux, Git LFS, Python 3.12 (required by the pinned
packages), an NVIDIA GPU that runs CUDA 13.0 PyTorch, and about 20 GB of disk,
plus 5.5 GB for the extension and supplementary analyses (requirements:
`reproduction/README.md` §0.0):

    git clone --branch v1.1.0-rc1-tors-reproduction \
      https://github.com/7114029017/context-aware-outfit-recommendation.git
    cd context-aware-outfit-recommendation
    git lfs install && git lfs pull        # eight precomputed feature files
    python3.12 -m venv .venv-repro && source .venv-repro/bin/activate
    python -m pip install torch==2.9.1 torchvision==0.24.1 \
      --index-url https://download.pytorch.org/whl/cu130
    python -m pip install -r reproduction/environment/requirements-reproduction-runtime.txt
    bash reproduction/scripts/bootstrap_data.sh --with-images --with-fashionclip --with-nomic --with-compendium
                                    # Polyvore metadata and splits; images, two models and the
                                    # MET Compendium for the analyses after the 35 units
    bash reproduction/scripts/check.sh            # preflight
    bash reproduction/scripts/reproduce_all.sh --fresh
                                    # the 35 units, then every analysis, the comparison with the
                                    # official run and the 46-item report <run>/ITEMS_STATUS.md

Use `git clone`: the source-code archives on the release page are not git
repositories and do not contain the Git LFS feature files. The full run took
55:07:40 on an NVIDIA GB10; run it inside tmux with the venv activated. The
step-by-step guide, expected outputs and acceptance criteria are in
`reproduction/README.md` (in Chinese, with all commands). A run passes when
`RUN_STATUS.txt` is `PASSED` and the comparison with the official run
(`<run>/official_comparison.txt`, written at the end) reports `IDENTICAL` or
`CONSISTENT`. `<run>/ITEMS_STATUS.md` gives the state and evidence of each of the
46 analysis items; `reproduce_all.sh --analyses-only <run>` redoes all analyses of
a completed run without retraining.

Official run and results:

- The manuscript's "frozen reference run" is the official run
  `full_20261002T161428Z` (35 units, `RUN_STATUS = PASSED`).
- Seed-level results: `reproduction/results/raw/full_20261002T161428Z/`;
  summaries and statistics: `reproduction/results/summary/full_20261002T161428Z/`;
  acceptance report: `reproduction/docs/clean_room_acceptance.md`; expected
  results: `reproduction/docs/expected_results.md`; known limitations:
  `reproduction/docs/known_limitations.md`.
- Three complete runs on the same NVIDIA GB10 and software stack produced
  bit-identical checkpoints and results. On other GPUs or PyTorch builds the
  numbers can differ slightly; the comparison tool then checks that the
  conclusions hold.
- Supplementary analyses computed from the official run's outputs, audits of
  the input data, and the thesis analyses of the preserved judge scores, human
  audit and MET reference list: `reproduction/docs/supplementary_analyses.md`.
  A new full run writes the same analyses of its own outputs to its
  `supplementary/` folder after it has passed.
- Extension analyses that regenerate, from a run's own models, the analyses
  for which the manuscript used preserved outputs (text length, Tables 4, 8
  and 9, the color analysis, Figures A1-A3 and thesis Figure 4-14):
  `reproduction/docs/extensions.md`; results for the official run in
  `reproduction/results/extensions/`.
- The state and evidence of each of the 46 analysis items of the official run:
  `reproduction/results/ITEMS_STATUS.md`.

Version provenance: the official run was executed from commit `b9bf5aa` of the
original development repository, which is not public. This repository contains
the same training, evaluation and statistics code, configurations and data
splits.

License and citation: code and documentation written for this repository are
under the MIT License (`LICENSE`); third-party components keep their own terms
(`THIRD_PARTY_NOTICES.md`). How to cite: `CITATION.cff`.

## 中文說明

主要流程：

1. 資料建構：整理 PO-D 資料範圍、CLO / MET / temperature proxy、生成描述、LLM 評分清單與 W/O/S 三因子分割。
2. 模型訓練和驗證：保留主模型 Hybrid Attention 的 CP / CIR 程式；CIR 由對應 CP checkpoint 初始化後訓練。另保留 Two-Tower 第二模型的程式與輸出。
3. 實驗與結果：依期刊採納的實驗脈絡整理文字長度、target clue leakage、主推薦任務、W/O/S 消融、第二模型、反事實與人工稽核。各實驗資料夾內的 `圖表_figures_tables/` 保存該實驗對應的正式圖表資料。

**Polyvore 原始圖片因第三方權利不在 repo 中散布**，第三方文獻與審查中的稿件也不收錄。
模型訓練需要的 8 個預先萃取特徵檔以 Git LFS 收錄。正式 run 的 70 個 checkpoint 不隨 repo 公開，
SHA-256 記錄在 seed index 中；學姊 2025 年訓練的 20 個 checkpoint 保留在
`02_模型訓練和驗證_model_training_validation/main_hybrid_attention_checkpoints/`。

## 2026 reproduction artifact

學姊原始研究內容保留在：

- `01_資料建構_data_construction/`
- `02_模型訓練和驗證_model_training_validation/`
- `03_實驗與結果_experiments_results/`

這三個資料夾的內容就是 frozen baseline `7a5cc9cd8f884865e1e27a234c18dd132f949598`，
也就是正式 run 使用的內容；公開版只移除了第三方文件與 3 張含 Polyvore 商品照片的圖，
並去除 3 個筆記本內嵌的商品圖片。原本的 `04_文件資料_documents/` 只有審查中的稿件與碩論，
因此不在公開版中。
完整清單見 `reproduction/docs/public_release_cleanup.md`。完整重現開始前，
`reproduction/scripts/pipeline/launch_full.sh` 會依
`reproduction/environment/archived_sources_manifest.json` 逐檔核對 01～03 的內容。

新增的重現程式、設定、固定 ID、環境資訊、reference results 與 provenance
文件統一放在：

`reproduction/`

重現與交接請從：

`reproduction/README.md`

開始閱讀。

修訂稿的正式結果來自 clean-room run `full_20261002T161428Z`：從 GitHub 全新
clone、依 `reproduction/README.md` 建立環境後以 `reproduce_all.sh --fresh`
完成 35 組訓練（10 組主實驗、25 組公平子集消融），狀態 `PASSED`。它與先前的
reference run（`reference_20260921T175217Z`）及重構後的 `full_20260924T102513Z`
逐位元相同（同一台 NVIDIA GB10、同一套軟體）。驗收紀錄見
`reproduction/docs/clean_room_acceptance.md`。

論文用語對照：論文中的 frozen reference run，就是正式 run `full_20261002T161428Z`；
repo 文件中的 reference run 則指較早的 `reference_20260921T175217Z`。

## 版本來源

論文引用的固定版本是 Release `v1.1.0-tors-reproduction`：以單一指令重跑的新正式 run 驗收後發布，這個版本是它的候選版 `v1.1.0-rc1-tors-reproduction`；在那之前已發布的版本是 `v1.0.0-tors-reproduction`。v1.0.0 之後只加入訓練以外的分析、逐項報告與最終摘要，設定、資料切分、訓練、評估與統計程式都沒有改動。重現期間的開發在另一個原始
開發 repository 進行（不公開），正式 run 的紀錄和文件中出現的 commit 編號（例如正式 run 的
`b9bf5aa`、frozen baseline `7a5cc9c`）與 repository 網址，指的都是那個 repository。

本 repository 的內容以原始開發 repository 的 commit `6693528`（git tree
`00a91b400f1df790464d21fae41f72c1b27da4a1`）為基礎，再加上公開版的整理：啟動檢查改為
逐檔核對內容並接受 Release 標籤、移除第三方文件與商品圖片、補充文件與補充分析。正式 run
之後，設定、資料切分、訓練、評估與統計程式都沒有改動。

修訂稿與 README 引用的正式數值只來自 `reproduction/results/summary/full_20261002T161428Z/`；
補充分析的輸出另放在 `reproduction/results/supplementary/`；新的完整 run 通過後，會在自己的
`supplementary/` 資料夾產生同樣的分析。論文中以保存資料計算的分析（評分者分數分布、低分樣本、
檢核清單涵蓋度、人工稽核題項、MET 對照表、CLO 分布等）也由補充分析重新計算，見
`reproduction/docs/supplementary_analyses.md`。稿件中原本使用學姊保存輸出的分析（文字長度、表 4、8、9、
色彩分析、附錄圖與論文圖 4-14），由延伸分析以 run 自己的模型重做，見 `reproduction/docs/extensions.md`。
46 個分析項目的逐項狀態與證據在 `reproduction/results/ITEMS_STATUS.md`；新的完整 run 會產生自己的
`ITEMS_STATUS.md`。

授權：本 repository 撰寫的程式與文件採用 MIT License（`LICENSE`），第三方元件依其原本條款
（`THIRD_PARTY_NOTICES.md`）。引用方式見 `CITATION.cff`。
