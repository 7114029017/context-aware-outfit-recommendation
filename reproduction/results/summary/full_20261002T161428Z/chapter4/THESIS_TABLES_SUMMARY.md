# 論文第四章逐表重現來源與結果索引

- Report source: full_20261002T161428Z
- 範圍：論文表 4-1 ～ 表 4-20。
- 目的：清楚區分 2026 fresh 重新訓練、學姊保存輸出重新統計、部分重建、歷史／設定查證。

## 一眼看懂

- Fresh 重新訓練：表 4-11 ～ 4-16。
  - 4-11 / 4-12：main Original / Context × 5 seeds，共 10 units。
  - 4-13 ～ 4-16：重建公平子集的 5 variants × 5 seeds，共 25 units。
- 學姊保存輸出重新統計：表 4-3、4-4、4-5、4-7、4-8、4-9、4-10。
- 保存資料 + 目前 metadata 的部分重建：表 4-6。
- 歷史／設定／案例 provenance 查證：表 4-1、4-2、4-17、4-18、4-19、4-20。

## 20 張表總覽

| 表格 | 類型 | Fresh 重訓？ | 目前判定 |
|---|---|---|---|
| 4-1 實驗環境與主要軟硬體配置 | 環境／歷史資料查證 | 否 | 歷史／provenance 查證；未獨立 fresh 驗證 |
| 4-2 CP 任務訓練設定 | 程式／設定查證 | 否（設定本身）；4-11～4-16 使用此設定重訓 | 歷史／provenance 查證；未獨立 fresh 驗證 |
| 4-3 CLO、MET 與 Tsub 分布 | 學姊保存輸出重新統計 | 否 | 保存輸出重算：已檢查欄位一致 |
| 4-4 文字長度差與檢索表現變化 | 學姊保存輸出重新統計 | 否 | 保存 OR row-level 輸出重新統計一致 |
| 4-5 相近文字長度子集 | 學姊保存輸出重新統計 | 否 | 保存 OR row-level 輸出重新統計一致 |
| 4-6 目標單品線索檢查 | 保存資料 + 目前 metadata 的部分重建 | 否 | 部分重現；目前重算與論文有差異 |
| 4-7 兩位 Judge 低分樣本集合重疊 | 學姊歷史輸出查證 + 保存分數重算 | 否 | 部分重現；目前重算與論文有差異 |
| 4-8 Judge Prompt Robustness | 學姊保存輸出重新統計 | 否 | 保存輸出重算：已檢查欄位一致 |
| 4-9 30 筆人工稽核樣本情境覆蓋 | 學姊保存人工稽核資料重新統計 | 否 | 保存人工稽核資料重新統計一致 |
| 4-10 人工評分與 Judge 分數比較 | 學姊保存人工判斷 + Judge 分數重新統計 | 否 | 保存人工稽核資料重新統計一致 |
| 4-11 CP 主實驗 | 2026 fresh 重新訓練 | 是 | fresh main 重訓：主要 means 在容許範圍 |
| 4-12 OR / CIR 主實驗 | 2026 fresh 重新訓練 | 是 | fresh main 重訓：主要 means 在容許範圍 |
| 4-13 公平縮減子集 CP | 2026 fresh 重建公平子集重新訓練 | 是 | fresh 公平子集消融；比較基準為 archived T01 aggregate，非逐格 source-exact |
| 4-14 公平縮減子集 OR | 2026 fresh 重建公平子集重新訓練 | 是 | fresh 公平子集消融；比較基準為 archived T01 aggregate，非逐格 source-exact |
| 4-15 CP 消融實驗 | 2026 fresh 重建公平子集消融重訓 | 是 | fresh 公平子集消融；比較基準為 archived T01 aggregate，非逐格 source-exact |
| 4-16 OR 消融實驗 | 2026 fresh 重建公平子集消融重訓 | 是 | fresh 公平子集消融；比較基準為 archived T01 aggregate，非逐格 source-exact |
| 4-17 案例一：不同版本的排名 | 學姊保存歷史案例查證 | 否（未證明同一歷史 case source-exact rerun） | 歷史／provenance 查證；未獨立 fresh 驗證 |
| 4-18 案例二：不同版本的排名 | 學姊保存歷史案例查證 | 否（未證明同一歷史 case source-exact rerun） | 歷史／provenance 查證；未獨立 fresh 驗證 |
| 4-19 顏色偏移改善案例 | 學姊保存標註／歷史影像案例查證 | 否 | 歷史／provenance 查證；未獨立 fresh 驗證 |
| 4-20 失敗案例：不同版本的排名 | 學姊保存歷史案例查證 | 否（未證明同一歷史 case source-exact rerun） | 歷史／provenance 查證；未獨立 fresh 驗證 |

## 逐表來源說明

### 表 4-1｜實驗環境與主要軟硬體配置

- 重現類型：環境／歷史資料查證
- 是否 fresh 重新訓練得到：否
- 資料／資訊從哪裡來：論文記載的歷史環境資訊 + 本次 2026 reproduction runtime 紀錄。
- 本流程實際做什麼：記錄 Python / PyTorch / CUDA / GPU / 套件；不以重新訓練產生此表。
- 目前判定：歷史／provenance 查證；未獨立 fresh 驗證
- 重點解讀：2026 runtime 可記錄，但不宣稱等同 2025 歷史環境。
- 目前 evidence：environment/environment_validation.json
- Chapter 4 report limitation：Runtime of this reproduction is recorded separately; exact 2025 environment is not proven.

### 表 4-2｜CP 任務訓練設定

- 重現類型：程式／設定查證
- 是否 fresh 重新訓練得到：否（設定本身）；4-11～4-16 使用此設定重訓
- 資料／資訊從哪裡來：學姊保留的 CP training code + 2026 reproduction configs / wrapper。
- 本流程實際做什麼：固定 epoch、batch size、learning rate、hard negative 等設定；歷史 DecoderLayerWithCrossAttn 定義未恢復。
- 目前判定：歷史／provenance 查證；未獨立 fresh 驗證
- 重點解讀：設定可追溯；CP decoder 採標準化實作（torch.nn.TransformerDecoderLayer），歷史 DecoderLayerWithCrossAttn 定義未保存。
- 目前 evidence：git_commit.txt ; reproduction/configs/
- Chapter 4 report limitation：Archived model code runs with the standardized decoder implementation; historical CP decoder source not recovered.

### 表 4-3｜CLO、MET 與 Tsub 分布

- 重現類型：學姊保存輸出重新統計
- 是否 fresh 重新訓練得到：否
- 資料／資訊從哪裡來：學姊保存的 train / valid / test temperature proxy rows。
- 本流程實際做什麼：從保存 proxy output 重新計算分位數、極端值與計數；不重新呼叫當年的 LLM 產生 CLO/MET。
- 目前判定：保存輸出重算：已檢查欄位一致
- 重點解讀：24 個論文顯示欄位可由保存輸出重算到顯示精度一致。
- 目前 evidence：secondary/results/environment_proxy/
- Chapter 4 report limitation：24 checked PDF-display fields from aligned saved CLO/MET/Tsub proxy outputs; independent MET valid source incomplete.

### 表 4-4｜文字長度差與檢索表現變化

- 重現類型：學姊保存輸出重新統計
- 是否 fresh 重新訓練得到：否
- 資料／資訊從哪裡來：學姊保存的 OR row-level retrieval 結果；46,555 seed-level rows / 9,311 unique query-target pairs。
- 本流程實際做什麼：重新計算文字長度差與 Hit@10 / rank improvement 的相關係數與 p 值。
- 目前判定：保存 OR row-level 輸出重新統計一致
- 重點解讀：屬保存結果重算，不是重新訓練或重新生成 query。
- 目前 evidence：secondary/results/length/A08_recomputed.csv
- Chapter 4 report limitation：Length correlations recalculated from archived seed-level rows; module-level archived-table comparison.

### 表 4-5｜相近文字長度子集

- 重現類型：學姊保存輸出重新統計
- 是否 fresh 重新訓練得到：否
- 資料／資訊從哪裡來：與表 4-4 相同的學姊保存 Original / Context OR row-level retrieval 結果與文字長度。
- 本流程實際做什麼：按文字長度差重新分組，重算 N、Hit@10 與平均排名改善。
- 目前判定：保存 OR row-level 輸出重新統計一致
- 重點解讀：五個 matched-length thresholds 可由保存 row-level output 重新統計。
- 目前 evidence：secondary/results/length/A10_recomputed.csv
- Chapter 4 report limitation：Length matched-subset/bucket tables recalculated from archived outputs; module-level archived-table comparison.

### 表 4-6｜目標單品線索檢查

- 重現類型：保存資料 + 目前 metadata 的部分重建
- 是否 fresh 重新訓練得到：否
- 資料／資訊從哪裡來：學姊保存的 generated/original text + 現存 category metadata；歷史 reliability_meta_from_subset.csv 未保存。
- 本流程實際做什麼：重新計算 lexical/category clue counts；缺失的歷史中間欄位不以調規則方式補造。
- 目前判定：部分重現；目前重算與論文有差異
- 重點解讀：9311 pairs 與 distinctive bigram 可核對；其他 lexical counts 保留與論文的差異。
- 目前 evidence：secondary/results/target_clue/A03_recomputed.csv
- Chapter 4 report limitation：Historical P12 reliability_meta_from_subset.csv missing; present-day category reconstruction differs.

### 表 4-7｜兩位 Judge 低分樣本集合重疊

- 重現類型：學姊歷史輸出查證 + 保存分數重算
- 是否 fresh 重新訓練得到：否
- 資料／資訊從哪裡來：學姊保存的歷史 P05 notebook 輸出 + 兩位 Judge formal scores。
- 本流程實際做什麼：重新算 bottom-p intersection / lift / Jaccard / F1；不重新呼叫 Judge LLM。
- 目前判定：部分重現；目前重算與論文有差異
- 重點解讀：歷史 notebook 表值有保存證據；目前獨立重算有小差異，兩者分開報告。
- 目前 evidence：secondary/results/judge/bottomp_recomputed.csv
- Chapter 4 report limitation：Saved-output bottom-p intersections differ from thesis; cutoff ties and historical input/runtime identity unresolved.

### 表 4-8｜Judge Prompt Robustness

- 重現類型：學姊保存輸出重新統計
- 是否 fresh 重新訓練得到：否
- 資料／資訊從哪裡來：學姊保存的 Judge A/B × P0-R2/P1/P2 六份 150-ID compare outputs。
- 本流程實際做什麼：重算 MAD、median、max 與 threshold rates；不重新呼叫 Qwen/Gemma。
- 目前判定：保存輸出重算：已檢查欄位一致
- 重點解讀：30 個論文顯示欄位可由保存 compare outputs 重算；歷史 150-ID sampling provenance 仍有限制。
- 目前 evidence：secondary/results/judge/table_4_8_paper_field_comparison.csv
- Chapter 4 report limitation：30 checked PDF-display fields from six saved Judge comparison files; no LLM rerun and historical 150-ID selection not recovered.

### 表 4-9｜30 筆人工稽核樣本情境覆蓋

- 重現類型：學姊保存人工稽核資料重新統計
- 是否 fresh 重新訓練得到：否
- 資料／資訊從哪裡來：學姊保存的 30-case audit sample + temperature / MET / occasion / style labels。
- 本流程實際做什麼：對保存的 30 個 sample IDs 重新分組計數；不重新抽樣、不重新人工標註。
- 目前判定：保存人工稽核資料重新統計一致
- 重點解讀：覆蓋分類數量可由保存 audit records 重算。
- 目前 evidence：secondary/results/human_audit/A42_coverage_recomputed.csv
- Chapter 4 report limitation：30-case coverage recomputed from selected archived records; source label-generation provenance not proven.

### 表 4-10｜人工評分與 Judge 分數比較

- 重現類型：學姊保存人工判斷 + Judge 分數重新統計
- 是否 fresh 重新訓練得到：否
- 資料／資訊從哪裡來：學姊保存的 30 cases × 25 = 750 個人工 YES/NO judgments + 兩位 Judge preserved scores。
- 本流程實際做什麼：重算 Bias、MAE、RMSE、Pearson、Spearman；不新增人工判斷或 Judge inference。
- 目前判定：保存人工稽核資料重新統計一致
- 重點解讀：屬保存 evidence 的重新統計。
- 目前 evidence：secondary/results/human_audit/T30_recomputed.csv
- Chapter 4 report limitation：750 archived human judgments recomputed and compared to archived T30; no new human judgments.

### 表 4-11｜CP 主實驗

- 重現類型：2026 fresh 重新訓練
- 是否 fresh 重新訓練得到：是
- 資料／資訊從哪裡來：本次 full run：Original / Context × seeds 1～5，共 10 個 main units 的 CP 階段。
- 本流程實際做什麼：每組 100 epochs，產生 fresh checkpoints / evaluation outputs，再彙整 AUC、FITB。
- 目前判定：fresh main 重訓：主要 means 在容許範圍
- 重點解讀：這是主要 fresh-training 證據；與論文 means 做數值接近與方向比較。
- 目前 evidence：statistics/table_4_11_cp_main.csv ; main/summary/main_reproduction_summary.csv ; statistics/main_paired_bh_8metrics.csv
- Chapter 4 report limitation：Fresh main CP condition means versus paper; paper SD/CI/p/effect sizes are NOT fully verified by this output.

### 表 4-12｜OR / CIR 主實驗

- 重現類型：2026 fresh 重新訓練
- 是否 fresh 重新訓練得到：是
- 資料／資訊從哪裡來：與表 4-11 同一批 10 個 main runs；CIR 使用 same-seed CP checkpoint 初始化。
- 本流程實際做什麼：fresh CIR training / evaluation，彙整 Recall@1/3/5/10/30/50；另做 five-seed paired + BH statistics。
- 目前判定：fresh main 重訓：主要 means 在容許範圍
- 重點解讀：六個 mean directions 維持 Context > Original；R@1/R@3 顯著性以 fresh statistics 如實報告。
- 目前 evidence：statistics/table_4_12_or_main.csv ; main/summary/main_reproduction_summary.csv ; statistics/main_paired_bh_8metrics.csv
- Chapter 4 report limitation：Fresh main OR condition means versus paper; paper SD/CI/p/significance stars are NOT fully verified.

### 表 4-13｜公平縮減子集 CP

- 重現類型：2026 fresh 重建公平子集重新訓練
- 是否 fresh 重新訓練得到：是
- 資料／資訊從哪裡來：2026 可重現重建的公平子集；Original / Context × seeds 1～5。
- 本流程實際做什麼：fresh CP/FITB training/evaluation；歷史 memberwise fair-subset IDs 未恢復。
- 目前判定：fresh 公平子集消融；比較基準為 archived T01 aggregate，非逐格 source-exact
- 重點解讀：可稱重建公平子集上的 reproduction，不可稱歷史 subset identity exact。
- 目前 evidence：ablation/summary/T03_stage1_cp_original_vs_full.csv ; ablation/summary/fresh_fair_subset_5seed_summary.json
- Chapter 4 report limitation：CP Original/Context metrics compared ONLY against archived T01 aggregate; this is NOT the individual published Table 4-13 cell comparison.

### 表 4-14｜公平縮減子集 OR

- 重現類型：2026 fresh 重建公平子集重新訓練
- 是否 fresh 重新訓練得到：是
- 資料／資訊從哪裡來：與表 4-13 相同的重建公平子集；3,432-query CIR evaluable scope。
- 本流程實際做什麼：fresh CIR/OR evaluation，彙整 R@10/R@30/R@50。
- 目前判定：fresh 公平子集消融；比較基準為 archived T01 aggregate，非逐格 source-exact
- 重點解讀：核心 Context > Original 方向可重現；幅度不要求逐格相同。
- 目前 evidence：ablation/summary/T04_stage1_cir_original_vs_full.csv ; ablation/summary/fresh_fair_subset_5seed_summary.json
- Chapter 4 report limitation：OR Original/Context metrics compared ONLY against archived T01 aggregate; historical fair-subset membership not recovered.

### 表 4-15｜CP 消融實驗

- 重現類型：2026 fresh 重建公平子集消融重訓
- 是否 fresh 重新訓練得到：是
- 資料／資訊從哪裡來：重建公平子集：Original / Context / No-Weather / No-Occasion / No-Style × seeds 1～5，共 25 units。
- 本流程實際做什麼：fresh CP/FITB training/evaluation，重算 factor-removal effects。
- 目前判定：fresh 公平子集消融；比較基準為 archived T01 aggregate，非逐格 source-exact
- 重點解讀：No-Style 在 AUC / FITB 仍為最大且穩定下降。
- 目前 evidence：ablation/summary/T05_stage2_cp_ablation.csv ; ablation/summary/fresh_fair_subset_5seed_summary.json
- Chapter 4 report limitation：CP factor-removal metrics compared ONLY against archived T01 aggregate; not a published Table 4-15 cell-by-cell test.

### 表 4-16｜OR 消融實驗

- 重現類型：2026 fresh 重建公平子集消融重訓
- 是否 fresh 重新訓練得到：是
- 資料／資訊從哪裡來：與表 4-15 同一批 25 個公平子集消融 runs 的 CIR / OR outputs。
- 本流程實際做什麼：重算 R@10/R@30/R@50 factor-removal effects；報告採 Without Factor − Full Context convention。
- 目前判定：fresh 公平子集消融；比較基準為 archived T01 aggregate，非逐格 source-exact
- 重點解讀：Style 影響穩定；Weather / Occasion 部分符號改變，差異保留、不調規則硬配論文。
- 目前 evidence：ablation/summary/T06_stage2_cir_ablation.csv ; statistics/factor_direction_candidate_vs_archived_T01.csv
- Chapter 4 report limitation：OR factor-removal metrics compared ONLY against archived T01 aggregate; some factor directions differ; historical subset ID missing.

### 表 4-17｜案例一：不同版本的排名

- 重現類型：學姊保存歷史案例查證
- 是否 fresh 重新訓練得到：否（未證明同一歷史 case source-exact rerun）
- 資料／資訊從哪裡來：學姊保存的歷史案例排名表。
- 本流程實際做什麼：保留並呈現歷史案例數值；2026 fresh run 未獨立證明同 set_id / target / retrieval pool。
- 目前判定：歷史／provenance 查證；未獨立 fresh 驗證
- 重點解讀：只能稱歷史案例 evidence preserved，不列為 fresh source-exact reproduction。
- 目前 evidence：chapter4/chapter4_report.json
- Chapter 4 report limitation：Fresh rankings for identical set_id/target/seed/pool not independently matched against published examples.

### 表 4-18｜案例二：不同版本的排名

- 重現類型：學姊保存歷史案例查證
- 是否 fresh 重新訓練得到：否（未證明同一歷史 case source-exact rerun）
- 資料／資訊從哪裡來：學姊保存的歷史案例排名表。
- 本流程實際做什麼：保留歷史排名 evidence；不把 2026 fresh run 說成同一案例已重建。
- 目前判定：歷史／provenance 查證；未獨立 fresh 驗證
- 重點解讀：歷史資料可查證，fresh identical-case reproduction 未建立。
- 目前 evidence：chapter4/chapter4_report.json
- Chapter 4 report limitation：Fresh rankings for identical set_id/target/seed/pool not independently matched against published examples.

### 表 4-19｜顏色偏移改善案例

- 重現類型：學姊保存標註／歷史影像案例查證
- 是否 fresh 重新訓練得到：否
- 資料／資訊從哪裡來：學姊保存的 row-level / case-level 問題與改善標註；完整重判需要原始 Polyvore 圖片。
- 本流程實際做什麼：核對保存計數與比例；不宣稱已重新做人眼影像稽核。
- 目前判定：歷史／provenance 查證；未獨立 fresh 驗證
- 重點解讀：保存標註可以核對，image-dependent human review 未重新執行。
- 目前 evidence：chapter4/chapter4_report.json
- Chapter 4 report limitation：Image-dependent human visual/color judgments not independently re-executed.

### 表 4-20｜失敗案例：不同版本的排名

- 重現類型：學姊保存歷史案例查證
- 是否 fresh 重新訓練得到：否（未證明同一歷史 case source-exact rerun）
- 資料／資訊從哪裡來：學姊保存的歷史失敗案例排名表。
- 本流程實際做什麼：保留歷史案例排名；2026 fresh run 未獨立對上相同 case identity。
- 目前判定：歷史／provenance 查證；未獨立 fresh 驗證
- 重點解讀：可作歷史 evidence，不列為 fresh source-exact case reproduction。
- 目前 evidence：chapter4/chapter4_report.json
- Chapter 4 report limitation：Identical-case fresh ranking comparison not independently verified.

## 解讀邊界

- Fresh 重新訓練只用於表 4-11 ～ 4-16；其餘表格不可因 full pipeline PASSED 就改稱 fresh training。
- 表 4-13 ～ 4-16 雖為 fresh training，但使用 2026 可重現重建的公平子集；歷史逐筆 fair-subset ID 未恢復。
- 表 4-3 ～ 4-10 的可重現性主要來自學姊保存的 row-level / score / audit outputs 重新統計；其中 4-6、4-7 保留目前差異。
- 表 4-17 ～ 4-20 主要是歷史案例／標註 evidence；沒有證明新 run source-exact 重建相同 case / image / retrieval pool。
- Judge preserved-output analysis 不等於重新呼叫歷史 Qwen/Gemma；人工 audit 重算也不等於重新做人工作業。

## 其他 machine-readable evidence

- Chapter 4 overview: /home/nchu5503/Documents/course/tors-cleanroom-20261003/context-aware-outfit-reproduction/reproduction/runs/full_20261002T161428Z/chapter4/chapter4_table_overview.csv
- Chapter 4 numeric comparison: chapter4/chapter4_numeric_comparison.csv（fresh run）或 reference summary 對應路徑。
- 完整 main / ablation / secondary / statistics 路徑請見同一 run 的 reproduction_summary.md 或 reference summary。
