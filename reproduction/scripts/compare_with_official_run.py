#!/usr/bin/env python3
"""Compare one completed full run with the official run.

The official run is the one named in
reproduction/results/final_reference_manifest.json. For the run given by
--run-root, this script reports:

1. identity: whether the fair-subset IDs, the evaluable OR queries and the 95
   seed-level result files (with --checkpoints also the 70 checkpoints) are
   byte-identical to the official run;
2. the five-seed means of the main experiment (Tables 4-11, 4-12) and of the
   fair-subset ablation (Tables 4-13 to 4-16), side by side;
3. the seven conclusions of section 九 of the 2026-09-30 guideline, judged by
   the same checks as rows R01-R07 of the manuscript reconciliation
   (build_manuscript_reconciliation.py).

The verdict is IDENTICAL (seed-level results byte-identical and all seven
conclusions agree), CONSISTENT (numbers differ, all seven conclusions agree)
or NOT CONSISTENT (a conclusion differs, or the run used another fair subset
or OR query set; these do not depend on the hardware). The report is printed
and saved as official_comparison.txt and official_comparison.json in
--out-dir (default: the run directory). Nothing is trained, evaluated or
recomputed. Exit status: 0 for IDENTICAL or CONSISTENT, 1 for NOT
CONSISTENT, 2 when the run or the official results cannot be read.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import unicodedata
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
try:
    import build_manuscript_reconciliation as recon  # noqa: E402
except ModuleNotFoundError as exc:  # PyYAML comes with the reproduction environment
    print(f"[COMPARISON BLOCKED] {exc}; activate the reproduction environment first "
          "(source .venv-repro/bin/activate, README 0.3)", file=sys.stderr)
    raise SystemExit(2)

ROOT = recon.ROOT
REPRO = recon.REPRO
MEAN_TOLERANCE = 0.005  # retraining tolerance of compare_main_results.py
FAIR_KEYS = recon.FAIR_CP + recon.FAIR_OR
FAIR_VARIANTS = ("original", "context", "no_weather", "no_occasion", "no_style")
VARIANT_LABEL = {"original": "Original", "context": "Context", "no_weather": "No-weather",
                 "no_occasion": "No-occasion", "no_style": "No-style"}
MAIN_RESULTS = ("results_cp.csv", "results_cir.csv")
ABLATION_RESULTS = ("results_cp_fresh_subset.csv", "results_cir_fresh_subset.csv",
                    "detail_cir_fresh_subset.csv")
RULE = "=" * 96


def fail(message: str) -> None:
    print("[COMPARISON BLOCKED] " + message, file=sys.stderr)
    raise SystemExit(2)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8").strip() if path.is_file() else "not_available"


def width(text: str) -> int:
    return sum(2 if unicodedata.east_asian_width(ch) in "WF" else 1 for ch in text)


def pad(text: str, size: int) -> str:
    return text + " " * (size - width(text))


class Report:
    """Prints lines and keeps them for official_comparison.txt."""

    def __init__(self) -> None:
        self.lines: list[str] = []

    def line(self, text: str = "") -> None:
        print(text, flush=True)
        self.lines.append(text)

    def table(self, headers: list[str], rows: list[list[str]]) -> None:
        widths = [max(width(row[i]) for row in [headers, *rows]) for i in range(len(headers))]
        self.line("  ".join(pad(h, w) for h, w in zip(headers, widths)).rstrip())
        self.line("  ".join("-" * w for w in widths))
        for row in rows:
            self.line("  ".join(pad(c, w) for c, w in zip(row, widths)).rstrip())


def side_info(run_id: str, commit_file: Path, status_file: Path, environment_file: Path) -> dict[str, str]:
    env = recon.read_json(environment_file)
    runtime = env.get("torch_runtime") or {}
    devices = runtime.get("devices") or [{}]
    torch = (env.get("packages") or {}).get("torch") or {}
    return {
        "run_id": run_id,
        "commit": read_text(commit_file),
        "status": read_text(status_file),
        "gpu": str(devices[0].get("name", "not_available")),
        "torch": str(torch.get("installed_version", "not_available")),
        "cuda": str(runtime.get("torch_cuda_version", "not_available")),
    }


def unit_files(run: Path, row: dict[str, str]) -> tuple[Path, list[tuple[str, str]], list[tuple[str, str]]]:
    """Unit directory, its (result file, official SHA-256) and (checkpoint, official SHA-256) pairs."""
    unit = f"{row['variant']}_seed{row['seed']}"
    if row["family"] == "main":
        base = run / "main" / unit
        results = [("evaluation/" + MAIN_RESULTS[0], row["cp_results_sha256"]),
                   ("evaluation/" + MAIN_RESULTS[1], row["cir_results_sha256"])]
    else:
        base = run / "ablation" / "runs" / unit
        results = list(zip(ABLATION_RESULTS, (row["cp_results_sha256"], row["cir_results_sha256"],
                                              row["detail_results_sha256"])))
    checkpoints = [("cp_best_ckpt.pt", row["cp_checkpoint_sha256"]),
                   ("cir_best_ckpt.pt", row["cir_checkpoint_sha256"])]
    return base, results, checkpoints


def compare_files(run: Path, seed_index: list[dict[str, str]], kind: str) -> dict:
    pairs = []
    for row in seed_index:
        base, results, checkpoints = unit_files(run, row)
        for name, digest in (results if kind == "results" else checkpoints):
            pairs.append((base / name, digest))
    differing = []
    for index, (path, digest) in enumerate(pairs, 1):
        if not path.is_file() or sha256(path) != digest:
            differing.append(recon.rel(path) + ("" if path.is_file() else " (missing)"))
        if kind == "checkpoints" and (index % 10 == 0 or index == len(pairs)):
            print(f"[CHECKPOINT HASH] {index}/{len(pairs)}", file=sys.stderr, flush=True)
    return {"identical": len(pairs) - len(differing), "total": len(pairs), "differing": differing}


def compare_subset(run: Path, seed_index: list[dict[str, str]]) -> dict:
    """Subset ID file SHA-256 recorded by each of the 25 ablation units."""
    rows = [r for r in seed_index if r["family"] != "main"]
    differing = []
    for row in rows:
        path = run / "ablation" / "runs" / f"{row['variant']}_seed{row['seed']}" / "fair_subset_run_manifest.json"
        digest = recon.read_json(path)["identity"]["candidate_id_sha256"] if path.is_file() else None
        if digest != row["candidate_id_sha256"]:
            differing.append(recon.rel(path) + ("" if path.is_file() else " (missing)"))
    return {"identical": len(rows) - len(differing), "total": len(rows), "differing": differing}


def identity_checks(run: Path, manifest: dict, seed_index: list[dict[str, str]], checkpoints: bool) -> dict:
    audit = recon.read_json(run / "ablation" / "fair_subset_cir_scope_audit.json")
    query_sha = audit.get("evaluable_query_membership_sha256")
    queries_same = None if query_sha is None else query_sha == manifest["splits"]["or_query_membership_sha256"]
    return {
        "fair_subset_ids": compare_subset(run, seed_index),
        "or_queries": queries_same,
        "results": compare_files(run, seed_index, "results"),
        "checkpoints": compare_files(run, seed_index, "checkpoints") if checkpoints else None,
    }


def same_text(value: bool | None) -> str:
    return "未記錄（舊版 run）" if value is None else "相同" if value else "不同"


def count_text(check: dict | None) -> str:
    if check is None:
        return "未檢查（加 --checkpoints）"
    return f"{check['identical']}/{check['total']} 相同"


def conclusion_rows() -> list[dict[str, str]]:
    rows = [r for r in recon.read_csv(recon.VALUES_CSV) if r["group"] == "required" and r["rule"] == "check"]
    if [r["id"] for r in rows] != [f"R0{i}" for i in range(1, 8)]:
        fail(f"expected the seven required checks R01-R07 in {recon.rel(recon.VALUES_CSV)}")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-root", required=True, type=Path,
                        help="Completed full run, e.g. reproduction/runs/full_<UTC timestamp>.")
    parser.add_argument("--checkpoints", action="store_true",
                        help="Also hash the run's 70 CP/OR checkpoints and compare them with the official run.")
    parser.add_argument("--out-dir", type=Path,
                        help="Where to save official_comparison.txt/.json (default: the run directory).")
    args = parser.parse_args()

    run = args.run_root.expanduser().resolve()
    if not run.is_dir():
        fail(f"run directory missing: {run}")
    status = read_text(run / "RUN_STATUS.txt")
    if status != "PASSED":
        fail(f"run is not PASSED (RUN_STATUS: {status}): {run}")
    try:
        manifest = recon.read_json(recon.OFFICIAL_MANIFEST)
        official_id = manifest["official_run_id"]
        _, official_paths, official_commit = recon.committed_layout(official_id)
        run_id, run_paths, run_commit = recon.load_layout(
            argparse.Namespace(official=False, reference=False, run_root=run))
        official = recon.Latest(official_id, official_paths)
        latest = recon.Latest(run_id, run_paths)
        seed_index = recon.read_csv(ROOT / manifest["results"]["seed_index"])
        raw_identity = REPRO / "results" / "raw" / official_id / "run_identity"
        sides = {
            "official": side_info(official_id, official_commit, raw_identity / "RUN_STATUS.txt",
                                  official_paths["environment"]),
            "run": side_info(run_id, run_commit, run / "RUN_STATUS.txt", run_paths["environment"]),
        }
        required = conclusion_rows()
        if len(seed_index) != 35:
            fail(f"expected 35 units in the official seed index, got {len(seed_index)}")
        identity = identity_checks(run, manifest, seed_index, args.checkpoints)
    except SystemExit as exc:
        if isinstance(exc.code, str):  # a read failure reported by build_manuscript_reconciliation
            fail(exc.code.removeprefix("[RECONCILIATION BLOCKED] "))
        raise

    report = Report()
    report.line(RULE)
    report.line("與正式 run 比對 — COMPARISON WITH THE OFFICIAL RUN")
    report.line(RULE)
    report.table(["", "正式 run", "本次 run"], [
        ["Run ID", sides["official"]["run_id"], sides["run"]["run_id"]],
        ["Commit", sides["official"]["commit"][:12], sides["run"]["commit"][:12]],
        ["RUN_STATUS", sides["official"]["status"], sides["run"]["status"]],
        ["GPU", sides["official"]["gpu"], sides["run"]["gpu"]],
        ["PyTorch / CUDA", f"{sides['official']['torch']} / {sides['official']['cuda']}",
         f"{sides['run']['torch']} / {sides['run']['cuda']}"],
    ])
    report.line()

    report.line("一、逐位元比對")
    report.table(["項目", "結果"], [
        ["消融各組使用的公平子集 ID（21,903 筆）", count_text(identity["fair_subset_ids"])],
        ["OR 可評估題目（3,432 題）", same_text(identity["or_queries"])],
        ["逐 seed 結果檔（CP、OR、OR 逐題明細）", count_text(identity["results"])],
        ["Checkpoint（CP、OR）", count_text(identity["checkpoints"])],
    ])
    for kind in ("fair_subset_ids", "results", "checkpoints"):
        check = identity[kind]
        if check and check["differing"]:
            shown = check["differing"][:5]
            more = len(check["differing"]) - len(shown)
            report.line(f"不同的檔案（前 {len(shown)} 個）：" + "、".join(shown) + (f" 等 {more} 個" if more else ""))
    report.line("同一台 GPU、同一 runtime 時應逐位元相同；不同 GPU、CUDA 或 PyTorch build 時不預期相同，請看第四節。")
    report.line()

    main_means, main_rows = [], []
    for m in recon.MAIN_METRICS:
        cells = []
        for field, condition in (("original_mean", "original"), ("context_mean", "context")):
            a, b = official.stat(m, field), latest.stat(m, field)
            main_means.append({"metric": m, "condition": condition, "official": a, "run": b, "abs_diff": abs(a - b)})
            cells.append(f"{a:.4f} / {b:.4f}")
        cells.append(f"{recon.signed(official.stat(m, 'delta_mean'))} / {recon.signed(latest.stat(m, 'delta_mean'))}")
        cells.append(f"{official.bh(m):.4f} / {latest.bh(m):.4f}")
        main_rows.append([recon.MAIN_LABEL[m], *cells])
    within = sum(r["abs_diff"] <= MEAN_TOLERANCE for r in main_means)
    largest = max(r["abs_diff"] for r in main_means)
    report.line("二、主實驗五 seed 平均（表 4-11、4-12）：正式 run / 本次 run")
    report.table(["指標", "Original", "Context", "Δ（Context − Original）", "BH 校正後 p"], main_rows)
    report.line(f"16 個平均值與正式 run 的差距 ≤ {MEAN_TOLERANCE}：{within}/16（最大差距 {largest:.4f}）")
    report.line()

    fair_means, fair_rows = [], []
    for k in FAIR_KEYS:
        cells = []
        for v in FAIR_VARIANTS:
            a, b = official.fair_mean(v, k), latest.fair_mean(v, k)
            fair_means.append({"metric": k, "variant": v, "official": a, "run": b, "abs_diff": abs(a - b)})
            cells.append(f"{a:.4f} / {b:.4f}")
        fair_rows.append([recon.FAIR_LABEL[k], *cells])
    fair_largest = max(r["abs_diff"] for r in fair_means)
    report.line("三、公平子集消融五 seed 平均（表 4-13～4-16）：正式 run / 本次 run")
    report.table(["指標", *(VARIANT_LABEL[v] for v in FAIR_VARIANTS)], fair_rows)
    report.line(f"25 個平均值與正式 run 的最大差距：{fair_largest:.4f}")
    report.line()

    conclusions = []
    for row in required:
        check = "check_" + row["latest_key"].split(".", 1)[1]
        o_holds, o_text, *_ = getattr(official, check)()
        r_holds, r_text, *_ = getattr(latest, check)()
        conclusions.append({"id": row["id"], "statement": row["item"],
                            "official": {"supported": bool(o_holds), "detail": o_text},
                            "run": {"supported": bool(r_holds), "detail": r_text},
                            "consistent": bool(o_holds) == bool(r_holds)})
    report.line("四、必須保留的結論（指引第九節；判定方式同對帳表 R01～R07）")
    report.table(["", "結論", "正式 run", "本次 run", "比對"], [
        [c["id"], c["statement"], "支持" if c["official"]["supported"] else "不支持",
         "支持" if c["run"]["supported"] else "不支持", "一致" if c["consistent"] else "不一致"]
        for c in conclusions])
    report.line("本次 run 的依據：")
    for c in conclusions:
        report.line(f"  {c['id']} {c['run']['detail']}")
        if not c["consistent"]:
            report.line(f"      正式 run：{c['official']['detail']}")
    report.line()

    disagree = [c["id"] for c in conclusions if not c["consistent"]]
    data_differs = bool(identity["fair_subset_ids"]["differing"]) or identity["or_queries"] is False
    identical = (not identity["results"]["differing"] and not data_differs
                 and not (identity["checkpoints"] or {}).get("differing"))
    if disagree or data_differs:
        verdict = "NOT CONSISTENT"
        reasons = []
        if data_differs:
            reasons.append("公平子集或 OR 題目與正式 run 不同（與硬體無關，應完全相同）")
        if disagree:
            reasons.append(f"{len(disagree)} 條結論與正式 run 不同（{'、'.join(disagree)}）")
        message = "；".join(reasons) + "。請保留整個 run 目錄與 logs/full_console.log，並回報。"
    elif identical:
        verdict = "IDENTICAL"
        message = (f"{identity['results']['total']} 個逐 seed 結果檔與正式 run 逐位元相同，"
                   "七條結論一致。")
    else:
        verdict = "CONSISTENT"
        message = "數值與正式 run 不完全相同，但七條結論一致。"
        if within < 16:
            message += f"有 {16 - within} 個主實驗平均值差距超過 {MEAN_TOLERANCE}，請記錄硬體與套件差異。"
    report.line(RULE)
    report.line(f"判定：{verdict} — {message}")
    report.line(RULE)

    data = {
        "official_run": sides["official"],
        "run": {**sides["run"], "run_root": recon.rel(run)},
        "identity": identity,
        "main_means": main_means,
        "main_means_within_tolerance": within,
        "tolerance": MEAN_TOLERANCE,
        "fair_subset_means": fair_means,
        "fair_subset_max_abs_diff": fair_largest,
        "conclusions": conclusions,
        "conclusion_checks": "rows R01-R07 of reproduction/docs/manuscript_reconciliation/manuscript_values.csv, "
                             "judged by reproduction/scripts/build_manuscript_reconciliation.py",
        "verdict": verdict,
    }
    out_dir = (args.out_dir or run).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "official_comparison.txt").write_text("\n".join(report.lines) + "\n", encoding="utf-8")
    (out_dir / "official_comparison.json").write_text(
        json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"已存檔：{recon.rel(out_dir / 'official_comparison.txt')}、official_comparison.json")
    raise SystemExit(1 if verdict == "NOT CONSISTENT" else 0)


if __name__ == "__main__":
    main()
