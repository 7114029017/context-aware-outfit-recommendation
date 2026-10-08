#!/usr/bin/env python3
"""Final summary of a completed full run, printed at the end of run_analyses.sh.

One screen in the order of the manuscript: the run (version, time, environment), the manuscript's tables
recomputed by the run (Tables 6 to 9 and the checks of Table 5), the seven required conclusions, the
comparison with the official run, the state of the 46 items and the acceptance. Every number is read from
the run's outputs; nothing is recomputed here. The text is also written to FINAL_SUMMARY.txt.

Usage:
  show_final_summary.py --official                       the committed results of the official run
  show_final_summary.py --run-root RUN [--supplementary-dir DIR] [--extensions-dir DIR] [--out-dir DIR]

--out-dir (default: the run folder; reproduction/results/ for --official) is where ITEMS_STATUS.json and
official_comparison.json are read and FINAL_SUMMARY.txt is written.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import unicodedata
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import report_items as ri  # noqa: E402
from report_items import REPO, recon, rows, text  # noqa: E402

WIDTH = 96
CONCLUSIONS = {"R01": "八項主要指標的平均值全部改善", "R02": "八項中七項經 BH 校正後顯著",
               "R03": "Recall@1 有平均改善，但未達校正後顯著", "R04": "Recall@3 仍達校正後顯著",
               "R05": "Style 是 CP 與 OR 中最大且最穩定的貢獻因素",
               "R06": "Weather 與 Occasion 的 OR 效果較小，且對 seed 敏感",
               "R07": "Weather 與 Occasion 並非在所有 OR 指標都有穩定的正貢獻"}
TWO_TOWER_ROWS = (("cp_test_auc", "CP Test AUC"), ("cp_test_fitb_acc", "CP FITB Accuracy"), ("recall@10", "Recall@10"),
                  ("recall@30", "Recall@30"), ("recall@50", "Recall@50"), ("mean_rank", "Mean rank (lower)"),
                  ("median_rank", "Median rank (lower)"))


# ---------------------------------------------------------------- text layout
def width(s: str) -> int:
    return sum(2 if unicodedata.east_asian_width(ch) in "WF" else 1 for ch in s)


def pad(s: str, n: int, right: bool = False) -> str:
    fill = " " * max(0, n - width(s))
    return fill + s if right else s + fill


def grid(headers: list[str], body: list[list[str]], right_from: int = 1) -> list[str]:
    """A plain table; columns from right_from on are right-aligned."""
    cols = [max(width(str(r[i])) for r in [headers, *body]) for i in range(len(headers))]
    line = lambda r: "  " + "  ".join(pad(str(c), cols[i], i >= right_from) for i, c in enumerate(r)).rstrip()  # noqa: E731
    return [line(headers), "  " + "  ".join("-" * c for c in cols), *[line(r) for r in body]]


def section(title: str) -> list[str]:
    return ["", "-" * WIDTH, " " + title, "-" * WIDTH]


def signed(v: float, digits: int = 4) -> str:
    return f"{v:+.{digits}f}"


# ---------------------------------------------------------------- run identity
def identity(lay: ri.Layout, out_dir: Path) -> dict:
    info = {"run": lay.run_id}
    if lay.official:
        m = json.loads(text(REPO / "reproduction" / "results" / "final_reference_manifest.json"))
        ex, repo, env = m["execution"], m["repository"], m["environment"]
        info.update(status=ex["run_status"], start=ex["started_utc"], finish=ex["finished_utc"],
                    train=ex["duration_hms"], commit=repo["commit"], source=f"{repo['url']} ({repo['branch']})",
                    env=f"{', '.join(env['gpu'])} | Python {env['python']} | PyTorch {env['packages']['torch']} | "
                        f"CUDA {env['torch_cuda']} | cuDNN {env['cudnn']}")
        return info
    run = lay.root
    info.update(status=text(run / "RUN_STATUS.txt").strip(), start=text(run / "started_utc.txt").strip(),
                finish=text(run / "finished_utc.txt").strip(), commit=text(run / "git_commit.txt").strip(),
                source=text(run / "git_remote.txt").strip())
    tag = subprocess.run(["git", "-C", str(REPO), "describe", "--tags", "--exact-match", info["commit"]],
                         capture_output=True, text=True).stdout.strip() if info["commit"] else ""
    info["tag"] = tag
    stamp = lambda s: datetime.strptime(s, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)  # noqa: E731
    hms = lambda sec: f"{int(sec // 3600)}:{int(sec % 3600 // 60):02d}:{int(sec % 60):02d}"  # noqa: E731
    try:
        start, finish = stamp(info["start"]), stamp(info["finish"])
        info["train"] = hms((finish - start).total_seconds())
        a0 = text(out_dir / "logs" / "analyses_started_utc.txt").strip()
        a1 = text(out_dir / "logs" / "analyses_finished_utc.txt").strip()
        if a0 and a1:
            a0, a1 = stamp(a0), stamp(a1)
            info["analyses"] = hms((a1 - a0).total_seconds())
            if (a0 - finish).total_seconds() < 3600:  # the analyses followed the training (reproduce_all.sh --fresh)
                info["total"] = hms((a1 - start).total_seconds())
    except ValueError:
        pass
    env = json.loads(text(run / "environment" / "environment_validation.json") or "{}")
    rt, pk = env.get("torch_runtime", {}), env.get("packages", {})
    torch_v = pk.get("torch", {}).get("installed_version", "?") if isinstance(pk.get("torch"), dict) else pk.get("torch", "?")
    gpu = ", ".join(d.get("name", "?") for d in rt.get("devices", [])) or "?"
    cudnn = rt.get("cudnn_version")
    cudnn = f"{cudnn // 10000}.{cudnn % 10000 // 100}.{cudnn % 100}" if isinstance(cudnn, int) else "?"
    info["env"] = (f"{gpu} | Python {str(env.get('python', '?')).split()[0]} | PyTorch {torch_v} | "
                   f"CUDA {rt.get('torch_cuda_version', '?')} | cuDNN {cudnn}")
    return info


# ---------------------------------------------------------------- the manuscript's tables
def table6(lay: ri.Layout) -> list[str]:
    stats = {r["metric"]: r for r in rows(lay.paths["stats"] / "main_paired_bh_8metrics.csv")}
    keys = {"AUC": "cp_auc", "FITB Acc": "cp_fitb", "Recall@1": "or_r1", "Recall@3": "or_r3", "Recall@5": "or_r5",
            "Recall@10": "or_r10", "Recall@30": "or_r30", "Recall@50": "or_r50"}
    body, sig = [], 0
    for r in rows(lay.paths["stats"] / "table_4_11_cp_main.csv") + rows(lay.paths["stats"] / "table_4_12_or_main.csv"):
        s = stats[keys[r["Metric"]]]
        sig += s["significant_bh_0_05"] == "True"
        body.append([("CP" if r["Task"] == "CP" else "CIR"), r["Metric"].replace("FITB Acc", "FITB Accuracy"),
                     r["Original (mean±std)"], r["Context-aware (mean±std)"], r["Δ"],
                     r["95% CI of Δ"].replace("+", ""), f"{float(s['bh_adjusted_p_8_main']):.6f}", r["Cohen’s dz"]])
    up = sum(1 for b in body if not b[4].startswith("-"))
    return [*section("表 6  主實驗：Hybrid Attention，原始文字 vs 情境描述（5 個 seed；BH 校正 8 項）"),
            *grid(["Task", "Metric", "Original text", "Context-aware", "Diff", "95% CI", "BH p", "dz"], body, 2),
            f"  → {up}/8 項平均改善；{sig}/8 項 BH 校正後顯著"
            + ("" if sig == 8 else "（未顯著：" + "、".join(b[1] for b, k in zip(body, keys.values())
                                                           if stats[k]["significant_bh_0_05"] != "True") + "）")]


def table7(lay: ri.Layout) -> list[str]:
    a = {r["Metric"]: r for r in rows(lay.paths["ablation"] / "T03_stage1_cp_original_vs_full.csv")
         + rows(lay.paths["ablation"] / "T04_stage1_cir_original_vs_full.csv")}
    b = {r["Metric"]: r for r in rows(lay.paths["ablation"] / "T05_stage2_cp_ablation.csv")
         + rows(lay.paths["ablation"] / "T06_stage2_cir_ablation.csv")}
    order = ["AUC", "FITB Acc", "Recall@10", "Recall@30", "Recall@50"]
    neg = lambda d: signed(-float(d))  # noqa: E731  (T05/T06 store No-X minus Full)
    main = [[m.replace("FITB Acc", "FITB"), a[m]["Original (mean±std)"], a[m]["Context-aware (mean±std)"],
             b[m]["No-style (mean±std)"], a[m]["Δ"], neg(b[m]["Δ_style"])] for m in order]
    drop = [[m.replace("FITB Acc", "FITB"), neg(b[m]["Δ_weather"]), neg(b[m]["Δ_occasion"]), neg(b[m]["Δ_style"])]
            for m in order]
    return [*section("表 7  公平子集與 Style 消融（21,903 套、3,432 題，5 個 seed）"),
            *grid(["Metric", "Original", "Full", "No-S", "Full-Original", "Full-No-S"], main),
            "", "  拿掉各因素後的下降（Full 減去拿掉該因素；正值代表這個因素有幫助）",
            *grid(["Metric", "Weather", "Occasion", "Style"], drop)]


def table8(lay: ri.Layout, ext: dict) -> list[str]:
    head = section("表 8  Two-Tower：重新訓練的 10 個模型（稿件表 8 是學姊保存的模型）")
    f = lay.ext / "two_tower" / "table8_two_tower.csv"
    if not f.is_file():
        return [*head, f"  沒有結果（two_tower：{ext.get('two_tower', 'not run')}）"]
    r = {x["metric"]: x for x in rows(f)}
    body = [[label, r[k]["original_mean_sd"], r[k]["context_aware_mean_sd"], r[k]["mean_difference"],
             r[k]["paired_95_ci"], r[k]["manuscript_difference"]] for k, label in TWO_TOWER_ROWS if k in r]
    return [*head, *grid(["Metric", "Original text", "Context-aware", "Diff", "95% CI", "稿件 Diff"], body)]


def table9(lay: ri.Layout, ext: dict) -> list[str]:
    head = section("表 9  反事實情境敏感度：24 組，正式 run 的 5 個 Context 模型（稿件表 9 是 1 個 2025 模型）")
    f = lay.ext / "counterfactual" / "table9_seeds_mean_sd.csv"
    if not f.is_file():
        return [*head, f"  沒有結果（counterfactual：{ext.get('counterfactual', 'not run')}）"]
    def fmt(v: str, sign: bool = False) -> str:
        m, sd = (float(x) for x in v.split(" ± "))
        return f"{m:+.1f} ± {sd:.1f}" if sign else f"{m:.1f} ± {sd:.1f}"
    body = [[r["scope"], r["n"], fmt(r["mean_rank_before"]), fmt(r["mean_rank_after"]), fmt(r["mean_rank_change"], True),
             r["top1_changed_share"], r["top5_jaccard"]] for r in rows(f)]
    return [*head, *grid(["Scope", "n", "Mean rank before", "after", "Mean ΔRank", "Top-1 changed", "Top-5 Jaccard"],
                         body)]


def table5(lay: ri.Layout) -> list[str]:
    out = section("表 5  資料品質檢查（重點數字）")
    proxy = {r["variable"]: r for r in rows(lay.s("paper_value_checks") / "table5_proxy_values.csv")}
    if proxy:
        c, m, t = proxy.get("CLO", {}), proxy.get("MET", {}), proxy.get("temperature reference (°C)", {})
        out.append(f"  {pad('代理值', 14)}CLO 中位數 {c.get('median')}（Q1-Q3 {c.get('q1')}-{c.get('q3')}）；MET 中位數 "
                   f"{m.get('median')}；溫度參考值中位數 {t.get('median')} °C")
    s = text(lay.ext / "text_length" / "summary.md")
    r1 = re.search(r"vs ΔHit@10 \(pair means over seeds\) \| ([0-9.]+)", s)
    r2 = re.search(r"vs rank improvement \| ([0-9.]+)", s)
    n = re.search(r"within five tokens \(abs difference <= 5\) \| (\d+)", s)
    d = re.search(r"ΔHit@10 within five tokens \(Full − Original\) \| ([-+0-9.]+)", s)
    if r1 and r2 and n and d:
        out.append(f"  {pad('文字長度', 14)}r = {float(r1.group(1)):.3f}（ΔHit@10）、{float(r2.group(1)):.3f}（名次）；長度相近 "
                   f"{int(n.group(1)):,} 對 ΔHit@10 {signed(float(d.group(1)))}（稿件 +0.0106，2025 年模型）")
    clue = {r["metric"]: r["percent_first_occurrence"] for r in rows(lay.s("paper_value_checks") / "table5_target_clues.csv")}
    if clue:
        out.append(f"  {pad('目標線索', 14)}細類別 {clue.get('generated_fine_category_lexical_match')}%、任一類別 "
                   f"{clue.get('generated_any_category_lexical_match')}%、詞組 "
                   f"{clue.get('generated_target_item_distinctive_bigram_overlap')}%；用詞重疊 "
                   f"{clue.get('generated_target_item_distinctive_token_overlap')}%（原始文字 "
                   f"{clue.get('original_target_item_distinctive_token_overlap')}%）")
    ties = rows(lay.s("judge_audit_checks") / "bottom_p_tie_orders.csv")
    if ties:
        t = ties[0]
        if t["intersection_numpy_argsort"] == t["thesis_table_4_7"]:
            out.append(f"  {pad('Judge 一致性', 14)}最低 5% 重疊 {t['intersection_numpy_argsort']} 筆（與稿件相同）")
        else:
            out.append(f"  {pad('Judge 一致性', 14)}最低 5% 重疊 {t['intersection_numpy_argsort']} 筆（稿件 {t['thesis_table_4_7']}，"
                       f"差在同分排序；同分可能的範圍 {t['smallest_over_tie_orders']}-{t['largest_over_tie_orders']}）")
    pr = {r["judge"]: r for r in rows(lay.paths["secondary"] / "judge" / "prompt_robustness_summary.csv")
          if r["variant"] == "P0-R2"}
    if pr:
        out.append(f"  {pad('Prompt 穩健性', 14)}P0-R2 平均絕對差 Judge A {float(pr['Qwen3-VL']['mean_abs_diff']):.4f}、"
                   f"Judge B {float(pr['Gemma-3']['mean_abs_diff']):.4f}")
    audit = {r["checklist"]: r for r in rows(lay.s("judge_audit_checks") / "audit_score_metrics.csv")}
    if audit:
        out.append(f"  {pad('人工稽核', 14)}30 筆；MAE Judge A {audit['qwen']['mae']}、Judge B {audit['gemma']['mae']}")
    return out


# ---------------------------------------------------------------- conclusions, comparison, items
def conclusions(lay: ri.Layout, out_dir: Path) -> tuple[list[str], bool, str]:
    latest = recon.Latest(lay.run_id, lay.paths)
    out, held = section("結論與比對"), []
    for r in recon.read_csv(recon.VALUES_CSV):
        if r["group"] == "required" and r["rule"] == "check":
            ok, *_ = getattr(latest, "check_" + r["latest_key"].split(".", 1)[1])()
            held.append(bool(ok))
            out.append(f"  {r['id']}  {'成立' if ok else '不成立'}  {CONCLUSIONS.get(r['id'], r['item'])}")
    if lay.official:
        verdict = "OFFICIAL"
        out.append("  與正式 run 比對：本 run 即正式 run")
    else:
        comp = json.loads(text(out_dir / "official_comparison.json") or "{}")
        verdict = comp.get("verdict", "NOT FOUND")
        line = re.search(r"^判定：.*$", text(out_dir / "official_comparison.txt"), re.M)
        official = comp.get("official_run", {})
        official = official.get("run_id", "") if isinstance(official, dict) else official
        out.append(f"  與正式 run {official} 比對：{line.group(0)[3:] if line else verdict}")
        diffs = [(f"{m.get('metric', '?')}（{m.get('condition', '?')}）", float(m["abs_diff"]))
                 for m in comp.get("main_means", []) if isinstance(m, dict) and "abs_diff" in m]
        if verdict != "IDENTICAL" and diffs:
            name, d = max(diffs, key=lambda x: x[1])
            out.append(f"  主實驗平均值與正式 run 的最大差異：{name} {d:.4f}（容許 {comp.get('tolerance', 0.005)}）")
    return out, all(held) and len(held) == 7, verdict


def wrap(names: list[str], indent: int) -> list[str]:
    """Item names joined with 、, wrapped to the summary width."""
    lines, cur = [], " " * indent
    for i, n in enumerate(names):
        piece = n + ("、" if i < len(names) - 1 else "")
        if cur.strip() and width(cur) + width(piece) > WIDTH:
            lines.append(cur.rstrip())
            cur = " " * indent
        cur += piece
    return [*lines, cur.rstrip()]


def items(out_dir: Path) -> tuple[list[str], dict]:
    data = json.loads(text(out_dir / "ITEMS_STATUS.json") or "{}")
    out = section(f"46 個分析項目（每項的依據與證據見 ITEMS_STATUS.md）：{data.get('verdict', 'NOT FOUND')}")
    by_state = {}
    for it in data.get("items", []):
        by_state.setdefault(it["state"], []).append(it)
    names = lambda group: [it["item_zh"] for it in group]  # noqa: E731
    for state, label in (("PASS", "和學姊 2025 年的數字相同"), ("DIFFERS", "數值不同，計算已驗證"),
                         ("REUSED", "沿用學姊的資料，核對 SHA-256"), ("NOT_REPRODUCIBLE", "無法重現，2025 年沒有紀錄"),
                         ("SKIPPED", "略過"), ("MISSING", "缺少輸出"), ("FAIL", "檢查不成立")):
        group = by_state.get(state, [])
        if not group:
            continue
        out.append(f"  {label}（{state}）{len(group)} 項")
        if state == "DIFFERS":
            ties = [it for it in group if "tied scores" in it["evidence"]]
            retrained = [it for it in group if it not in ties]
            if retrained:
                out += [f"    模型重新訓練（{len(retrained)} 項）", *wrap(names(retrained), 6)]
            if ties:
                out += [f"    同分樣本的排序不同（{len(ties)} 項）", *wrap(names(ties), 6)]
        elif state in ("SKIPPED", "MISSING", "FAIL"):
            out += [f"    ！{it['item_zh']}：{it['evidence']}" for it in group]
        else:
            out += wrap(names(group), 6)
    checks = data.get("port_checks", [])
    passed = sum(1 for c in checks if c["status"].startswith("PASSED"))
    skipped = sum(1 for c in checks if c["status"].startswith("SKIPPED"))
    out.append(f"  移植程式對 2025 年輸出的驗證：{passed}/{len(checks)} 通過" + (f"、{skipped} 項略過" if skipped else ""))
    return out, data


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--official", action="store_true", help="the official run's committed results")
    group.add_argument("--run-root", type=Path, help="a completed full run folder")
    parser.add_argument("--supplementary-dir", type=Path, default=None)
    parser.add_argument("--extensions-dir", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=None)
    args = parser.parse_args()
    lay = ri.Layout(args)
    out_dir = (args.out_dir or (REPO / "reproduction" / "results" if args.official else lay.root)).resolve()
    info = identity(lay, out_dir)
    ext = ri.extension_status(lay)

    version = info.get("tag") or ""
    version = (version + "，" if version else "") + f"commit {info.get('commit', '?')[:12]}"
    timing = f"訓練 {info.get('start')} → {info.get('finish')}（{info.get('train', '?')}）"
    if info.get("analyses"):
        timing += f"；之後的分析 {info['analyses']}" + (f"；合計 {info['total']}" if info.get("total") else "")
    lines = ["=" * WIDTH, f" TORS 論文重現 — 最終結果　{info['run']}", "=" * WIDTH,
             f"  狀態    {info.get('status')}（{sum(1 for _, s in lay.units if s.startswith('passed'))}/35 組訓練完成）",
             f"  版本    {version}", f"  來源    {info.get('source')}", f"  時間    {timing}", f"  環境    {info.get('env')}"]
    lines += table6(lay) + table7(lay) + table8(lay, ext) + table9(lay, ext) + table5(lay)
    concl, all_held, verdict = conclusions(lay, out_dir)
    item_lines, data = items(out_dir)
    lines += concl + item_lines
    where = "repo 的 reproduction/results/" if args.official else "run 資料夾"
    files = [("ITEMS_STATUS.md", "46 個項目的判定與證據")]
    files += ([("final_reference_manifest.json", "正式 run 的識別資訊與雜湊"), ("summary/<run>/、raw/<run>/", "統計表與逐 seed 結果")]
              if args.official else [("official_comparison.txt", "與正式 run 的逐項比對"), ("INDEX.md", "所有輸出的索引"),
                                     ("logs/", "每個步驟的完整紀錄")])
    files.append(("FINAL_SUMMARY.txt", "本摘要"))
    lines += [*section(f"詳細資料（{where}）"), *[f"  {pad(a, 31)}{b}" for a, b in files]]
    counts = data.get("counts", {})
    bad = [s for s in ("MISSING", "FAIL") if counts.get(s)]
    ok = info.get("status") == "PASSED" and all_held and verdict in ("IDENTICAL", "CONSISTENT", "OFFICIAL") and not bad
    compared = "本 run 即正式 run" if verdict == "OFFICIAL" else f"比對 {verdict}"
    detail = f"{info.get('status')}；七條結論{'全部成立' if all_held else '有不成立'}；{compared}；46 項 {data.get('verdict', '?')}"
    lines += ["", "=" * WIDTH, f" 驗收：{'通過' if ok else '未通過'}（{detail}）", "=" * WIDTH]
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "FINAL_SUMMARY.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    lines.append(f" 本摘要已存到 {out_dir / 'FINAL_SUMMARY.txt'}")
    print("\n".join(lines))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
