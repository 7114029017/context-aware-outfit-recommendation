#!/usr/bin/env python3
"""Thesis analyses of the LLM-judge scores and the human audit that the pipeline does not produce.

The judge scores and the 750 human judgments are preserved 2025 data (they cannot be regenerated, see
reproduction/docs/known_limitations.md); the pipeline recomputes the agreement statistics (T19, T21,
Table 4-7, Table 4-8) and the audit metrics (T30). This script recomputes the remaining thesis analyses
of the same data, with the computation copied from the 2025 programs, and writes under --out-dir
(default reproduction/results/supplementary/judge_audit_checks/):

- judge_score_diagnostics.csv, judge_score_histogram.csv and figure_4_1_judge_score_histogram.svg: the
  score distribution of the two judges (thesis Figure 4-1; notebook P05 cell 3: discrimination_diag and
  the 50-bin histograms; the thesis figure, the archived F17 SVG, bins both judges on [0, 1], while P05's
  own PNG binned each judge on its own range, and the program that redrew it is not preserved);
- bottom_p_sensitivity_band.csv, figure_4_3_bottom_p_overlap_band.svg and bottom_p_lift_band.svg: the
  overlap of the two judges' lowest-scored p% for p = 1% to 30% with 95% bootstrap bands (B = 500,
  seed 123; thesis Figure 4-3; P05 cell 3: sensitivity_point_estimate, bootstrap_bands);
  the histogram and the curves are compared with the archived SVG figures F17 and F19;
- item_disagreement.csv and figure_4_4_item_disagreement.svg: for each checklist item, how often the
  judge's yes/no differs from the human answer over the 30 audited descriptions (2025 table T31, thesis
  Figure 4-4; notebook P06 cell 10);
- judge_reference_scan.csv and judge_input_roles.csv: a static search of the training and evaluation code
  for references to judge outputs (2025 tables A05 and A06; build_journal_artifacts.py,
  derive_no_judge_intervention_proof), repeated on the code that runs the reproduction's 35 units;
- summary.md.

The P05 / P06 notebooks are in 03_實驗與結果_experiments_results/00_控制檢查與附加稽核/source_programs/ and
03_實驗與結果_experiments_results/07_人工稽核與品質診斷/source_programs/.
"""
from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from statistics import mean

import numpy as np

from _common import D01, D02, D03, GENERATED, REPRO, SUPPLEMENTARY, read_csv, read_json, write_csv, write_text
from _svg import PALETTE, hbar_chart, histogram_chart, line_chart

JUDGES = D01 / "llm_judge_checklists"
QWEN_SCORES = JUDGES / "Qwen3_Instruct" / "phase3_scores_Qwen3VL32B.jsonl"
GEMMA_SCORES = JUDGES / "Gemma3" / "phase3_scores_Gemma3.jsonl"
QWEN_CSTAR = JUDGES / "Qwen3_Instruct" / "checklist_C_star_Qwen3VL32B.json"
GEMMA_CSTAR = JUDGES / "Gemma3" / "checklist_C_star_Gemma3.json"
CONTROL_TABLES = D03 / "00_控制檢查與附加稽核" / "圖表_figures_tables" / "tables"
CONTROL_FIGURES = CONTROL_TABLES.parent / "figures"
AUDIT = D03 / "07_人工稽核與品質診斷"
MODEL_CODE = D02 / "main_hybrid_attention_code"
SCRIPTS = REPRO / "scripts"

# P05 cell 3
P_GRID = np.arange(0.01, 0.301, 0.01)
N_BOOT, SEED_BAND, CI_LO, CI_HI = 500, 123, 0.025, 0.975
TABLE_4_7 = {0.05: (668, 0.2347, 0.3802), 0.10: (1323, 0.2319, 0.3765), 0.20: (2833, 0.2524, 0.4031)}
THESIS_MEANS = {"A(Qwen)": (0.828, 0.142), "B(Gemma)": (0.960, 0.089)}  # thesis Section 4.2.4
# P06 cell 9
QWEN_COLOR, GEMMA_COLOR = "#2f77b4", "#f28e2b"
T31_COLUMNS = ["checklist", "item_id", "category", "weight", "n", "item_disagreement_rate_0_1",
               "weighted_item_disagreement_0_1", "mean_signed_model_minus_human", "question_en", "short_label", "rank"]
# build_journal_artifacts.py, derive_no_judge_intervention_proof
JUDGE_TERMS = ["phase3_scores", "Qwen3VL32B", "phase3_scores_Qwen3VL32B", "phase3_scores_Gemma3", "checklist_C",
               "LLM-as-a-Judge", "judge_score", "rewrite_qwen", "rewrite_gemma"]
FILES_2025 = ["train_cp.py", "train_cir.py", "evaluate_cp.py", "evaluate_cir.py", "CP_evaluate.py", "dataset.py",
              "outfit_transformer.py", "utils.py", "config/base_config.py", "config/cp_cond_hardneg.py",
              "config/cir_cond_hardneg.py", "5筆資料 t-test.ipynb", "新描述 t-test.ipynb"]
TRAINING_ENTRY_POINTS = (  # the stages of reproduce_thesis.sh that produce the 35 units and their statistics
    SCRIPTS / "freeze_splits.py", SCRIPTS / "pipeline" / "reproduce_main.sh", SCRIPTS / "pipeline" / "reproduce_ablation.sh",
    SCRIPTS / "pipeline" / "complete_statistics.py")
TRAINING_SCOPE = "reproduction: 35 units (training, evaluation, statistics)"


def rel(path: Path) -> str:
    return str(path.relative_to(D01.parent))


def load_scores(path: Path) -> dict[str, dict]:
    """P05 load_scores_jsonl / P06 load_jsonl_scores: records without a score are skipped."""
    out = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        obj = json.loads(line)
        if obj.get("score") is None:
            continue
        out[str(obj["id"])] = {
            "score": float(obj["score"]),
            "decisions": {str(item.get("id")): str(item.get("answer", "")).strip().lower()
                          for item in obj.get("decisions", [])},
        }
    return out


# ---------------------------------------------------------------- P05 cell 3, copied
def discrimination_diag(x, name):
    q1, q3 = np.quantile(x, [0.25, 0.75])
    iqr = q3 - q1
    lt08 = (x < 0.8).mean()
    lt09 = (x < 0.9).mean()
    xr3 = np.round(x, 3)
    xr4 = np.round(x, 4)
    ur3 = len(np.unique(xr3)) / len(xr3)
    ur4 = len(np.unique(xr4)) / len(xr4)
    _, counts = np.unique(xr3, return_counts=True)
    ties_proxy = counts.max() / len(x)
    return {
        "judge": name,
        "mean": float(x.mean()),
        "std": float(x.std(ddof=1)),
        "iqr": float(iqr),
        "P(score<0.8)": float(lt08),
        "P(score<0.9)": float(lt09),
        "unique_ratio@3dp": float(ur3),
        "unique_ratio@4dp": float(ur4),
        "ties_proxy@3dp(maxfreq/N)": float(ties_proxy),
    }


def sensitivity_point_estimate(xa, xb, p_grid):
    N = len(xa)
    order_a = np.argsort(xa)
    order_b = np.argsort(xb)
    mask_a = np.zeros(N, dtype=bool)
    mask_b = np.zeros(N, dtype=bool)
    inter = 0
    prev_k = 0
    jacc = np.zeros(len(p_grid), dtype=np.float64)
    f1 = np.zeros(len(p_grid), dtype=np.float64)
    lift = np.zeros(len(p_grid), dtype=np.float64)
    for t, p in enumerate(p_grid):
        k = int(np.ceil(N * p))
        if k <= prev_k:
            k = prev_k + 1
        newA = order_a[prev_k:k]
        inter += np.count_nonzero(mask_b[newA])
        mask_a[newA] = True
        newB = order_b[prev_k:k]
        inter += np.count_nonzero(mask_a[newB] & (~mask_b[newB]))
        mask_b[newB] = True
        union = 2 * k - inter
        jacc[t] = inter / union if union > 0 else np.nan
        f1[t] = inter / k if k > 0 else np.nan
        exp_inter = (p * p) * N
        lift[t] = inter / exp_inter if exp_inter > 0 else np.nan
        prev_k = k
    return jacc, f1, lift


def bootstrap_bands(xa, xb, p_grid, n_boot=500, seed=123, qlo=0.025, qhi=0.975):
    rng = np.random.default_rng(seed)
    N = len(xa)
    P = len(p_grid)
    jacc_mat = np.zeros((n_boot, P), dtype=np.float64)
    f1_mat = np.zeros((n_boot, P), dtype=np.float64)
    lift_mat = np.zeros((n_boot, P), dtype=np.float64)
    idx_all = np.arange(N)
    for b in range(n_boot):
        samp = rng.choice(idx_all, size=N, replace=True)
        j, f, l = sensitivity_point_estimate(xa[samp], xb[samp], p_grid)
        jacc_mat[b] = j
        f1_mat[b] = f
        lift_mat[b] = l
    return ((np.quantile(jacc_mat, qlo, axis=0), np.quantile(jacc_mat, qhi, axis=0)),
            (np.quantile(f1_mat, qlo, axis=0), np.quantile(f1_mat, qhi, axis=0)),
            (np.quantile(lift_mat, qlo, axis=0), np.quantile(lift_mat, qhi, axis=0)))


# ---------------------------------------------------------------- archived matplotlib SVG figures
def svg_axis(svg: str, axis: str):
    """Pixel-to-value map of a matplotlib SVG axis, from its first and last tick."""
    ticks = [(float(x if axis == "x" else y), float(v.replace("−", "-"))) for x, y, v in re.findall(
        rf'<g id="{axis}tick_\d+">.*?<use [^>]*x="([\d.]+)" y="([\d.]+)".*?<text [^>]*>([^<]+)</text>', svg, re.S)]
    (p0, v0), (p1, v1) = ticks[0], ticks[-1]
    return lambda px: v0 + (px - p0) * (v1 - v0) / (p1 - p0)


def svg_points(body: str) -> list[tuple[float, float]]:
    path = re.search(r'd="([^"]+)"', body)
    return [(float(a), float(b)) for a, b in re.findall(r"[ML] ([\d.]+) ([\d.]+)", path.group(1))] if path else []


def archived_histogram(path: Path, colors: tuple[str, str]) -> list[list[int]]:
    """Bar heights of a matplotlib histogram SVG, per colour, in drawing order."""
    svg = path.read_text(encoding="utf-8")
    to_value = svg_axis(svg, "y")
    bars = {c: [] for c in colors}
    for d, style in re.findall(r'<path d="(M [^"]+)"[^>]*?style="([^"]*)"', svg):
        fill = re.search(r"fill: (#[0-9a-f]{6})", style)
        points = [(float(a), float(b)) for a, b in re.findall(r"[ML] ([\d.]+) ([\d.]+)", d)]
        if fill and fill.group(1) in bars and len(points) == 4:
            ys = [y for _, y in points]
            bars[fill.group(1)].append(round(to_value(min(ys)) - to_value(max(ys))))
    return [bars[c] for c in colors]


def archived_curves(path: Path) -> dict[str, list[float]]:
    """The two lines and two bands of the archived Figure 4-3 (F19), in data units, p = 1% to 30%."""
    svg = path.read_text(encoding="utf-8")
    to_value = svg_axis(svg, "y")
    groups = dict(re.findall(r'<g id="((?:line2d|FillBetweenPolyCollection)_\d+)">(.*?)</g>', svg, re.S))
    lines = [g for g in groups if g.startswith("line2d") and len(svg_points(groups[g])) == len(P_GRID)
             and re.search(r"stroke: #(1f77b4|ff7f0e)", groups[g])]
    bands = sorted(g for g in groups if g.startswith("FillBetween"))
    out = {}
    for name, line, band in zip(("jaccard", "f1"), lines, bands):
        out[name] = [to_value(y) for _, y in svg_points(groups[line])]
        points = svg_points(groups[band])  # fill_between: start, lower edge, upper edge backwards
        out[f"{name}_lo"] = [to_value(y) for _, y in points[1:31]]
        out[f"{name}_hi"] = [to_value(y) for _, y in points[32:62]][::-1]
    return out


# ---------------------------------------------------------------- P06 cell 10, copied
def answer_to_float(value):
    s = str(value).strip().lower()
    if s in {"yes", "y", "1", "true", "是"}:
        return 1.0
    if s in {"no", "n", "0", "false", "否"}:
        return 0.0
    return None


def fmt4(value):
    return "NA" if value is None else f"{value:.4f}"


def safe_float(value, default=None):
    try:
        return default if value is None or value == "" else float(value)
    except Exception:
        return default


def item_disagreement(scores: dict, checklists: dict, selected_ids: list[str], manual_rows: list[dict]) -> list[dict]:
    human_values = defaultdict(list)
    for row in manual_rows:
        set_id = str(row.get("set_id", ""))
        if set_id not in selected_ids:
            continue
        name = str(row.get("checklist", "")).strip().lower()
        item_id = str(row.get("item_id", "")).strip()
        ans = answer_to_float(row.get("human_answer_yes", row.get("human_answer", "")))
        if ans is None:
            ans = answer_to_float(row.get("human_answer", ""))
        if name in {"qwen", "gemma"} and item_id and ans is not None:
            human_values[(set_id, name, item_id)].append(ans)
    human_mean = {key: mean(values) for key, values in human_values.items() if values}

    def model_item_decisions(set_id, name):
        return {item_id: 1.0 if ans == "yes" else 0.0
                for item_id, ans in scores[name][set_id]["decisions"].items() if ans in {"yes", "no"}}

    rows = []
    for name, checklist in checklists.items():
        total_weight = sum(float(item.get("weight", 1)) for item in checklist)
        for item in checklist:
            item_id = str(item["id"])
            w = float(item.get("weight", 1))
            values, signed_values = [], []
            for set_id in selected_ids:
                h = human_mean.get((set_id, name, item_id))
                m = model_item_decisions(set_id, name).get(item_id)
                if h is None or m is None:
                    continue
                values.append(abs(m - h))
                signed_values.append(m - h)
            disagreement = mean(values) if values else None
            rows.append({
                "checklist": name, "item_id": item_id, "category": item.get("category", ""), "weight": w,
                "n": len(values), "item_disagreement_rate_0_1": fmt4(disagreement),
                "weighted_item_disagreement_0_1": fmt4(disagreement * (w / total_weight)
                                                       if disagreement is not None else None),
                "mean_signed_model_minus_human": fmt4(mean(signed_values) if signed_values else None),
                "question_en": item.get("text", ""), "short_label": f"{name.capitalize()} {item_id}",
            })
    rows = sorted(rows, key=lambda r: safe_float(r["weighted_item_disagreement_0_1"], -1), reverse=True)
    for i, row in enumerate(rows, start=1):
        row["rank"] = i
    return rows


# ---------------------------------------------------------------- A05 / A06
def search(path: Path) -> tuple[str, list[str]]:
    """The 2025 test: the file's text contains a judge term."""
    if not path.is_file():
        return "missing_file", []
    text = path.read_text(encoding="utf-8", errors="ignore")
    return "searched", [term for term in JUDGE_TERMS if term in text]


def referenced_files(path: Path) -> set[Path]:
    """Scripts named in a file (paths ending in .py or .sh) and the local modules it imports."""
    text = path.read_text(encoding="utf-8", errors="ignore")
    bases = (path.parent, SCRIPTS, SCRIPTS / "pipeline", MODEL_CODE, MODEL_CODE / "config")
    names = {token.rsplit("/", 1)[-1] for token in re.findall(r"[\w$./{}-]+\.(?:py|sh)\b", text)}
    if path.suffix == ".py":
        names |= {f"{m}.py" for m in re.findall(r"^\s*(?:from|import)\s+([A-Za-z_]\w*)", text, re.MULTILINE)}
    return {base / name for name in names for base in bases if (base / name).is_file()}


def training_closure() -> set[Path]:
    """The files executed for the 35 units: the archived model code and the split, main, ablation and
    statistics stages, with every script they name or import, followed recursively."""
    todo = list(TRAINING_ENTRY_POINTS) + sorted(p for p in MODEL_CODE.rglob("*.py"))
    seen: set[Path] = set()
    while todo:
        path = todo.pop().resolve()
        if path in seen or not path.is_file():
            continue
        seen.add(path)
        todo.extend(referenced_files(path) - seen)
    return seen


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out-dir", type=Path, default=SUPPLEMENTARY / "judge_audit_checks")
    parser.add_argument("--polyvore-root", default=None, help="accepted for run_all.sh; not used")
    args = parser.parse_args()
    out = args.out_dir
    figures = out / "figures"

    qwen, gemma = load_scores(QWEN_SCORES), load_scores(GEMMA_SCORES)
    ids = sorted(set(qwen) & set(gemma))
    xa = np.array([qwen[i]["score"] for i in ids], dtype=np.float64)
    xb = np.array([gemma[i]["score"] for i in ids], dtype=np.float64)

    # ---------------------------------------------------------------- Figure 4-1
    diag = [discrimination_diag(xa, "A(Qwen)"), discrimination_diag(xb, "B(Gemma)")]
    keys = list(diag[0])
    write_csv(out / "judge_score_diagnostics.csv", keys, [[d[k] if k == "judge" else repr(d[k]) for k in keys]
                                                          for d in diag])
    binned = histogram_chart(figures / "figure_4_1_judge_score_histogram.svg",
                             "Score distribution of the two judges (thesis Figure 4-1)",
                             [("Judge A (Qwen)", xa, PALETTE[0]), ("Judge B (Gemma)", xb, PALETTE[1])], 50,
                             "Judge score", lo=0.0, hi=1.0, ylabel="Number of descriptions",
                             subtitle=f"{len(ids)} descriptions scored by both judges; 50 bins on [0, 1]")
    f17 = archived_histogram(CONTROL_FIGURES / "F17_judge_qwen_gemma_score_histogram.svg", PALETTE[:2])
    f17_same = all(bars[:50] == counts for bars, (counts, _) in zip(f17, binned))
    hist_rows = []
    for judge, (counts, edges) in zip(("A(Qwen)", "B(Gemma)"), binned):
        hist_rows += [[judge, i + 1, repr(edges[i]), repr(edges[i + 1]), n] for i, n in enumerate(counts)]
    write_csv(out / "judge_score_histogram.csv", ["judge", "bin", "left_edge", "right_edge", "count"], hist_rows)
    t19 = read_csv(CONTROL_TABLES / "T19_judge_qwen_gemma_agreement_summary.csv")[0]
    t19_same = (len(ids) == int(t19["N_intersection"])
                and all(repr(d[k]) == t19[f"{p}_{s}"] for d, p in zip(diag, "AB")
                        for k, s in (("mean", "mean"), ("std", "std"))))

    # ---------------------------------------------------------------- Figure 4-3
    j_point, f_point, l_point = sensitivity_point_estimate(xa, xb, P_GRID)
    (j_lo, j_hi), (f_lo, f_hi), (l_lo, l_hi) = bootstrap_bands(xa, xb, P_GRID, N_BOOT, SEED_BAND, CI_LO, CI_HI)
    band_columns = ["p", "jaccard", "jacc_lo", "jacc_hi", "f1", "f1_lo", "f1_hi", "lift", "lift_lo", "lift_hi"]
    band = np.column_stack([P_GRID, j_point, j_lo, j_hi, f_point, f_lo, f_hi, l_point, l_lo, l_hi])
    write_csv(out / "bottom_p_sensitivity_band.csv", band_columns, [[repr(float(v)) for v in row] for row in band])
    line_chart(figures / "figure_4_3_bottom_p_overlap_band.svg",
               f"Sensitivity curve (bottom-p%) with 95% bootstrap band (B={N_BOOT}) (thesis Figure 4-3)", list(P_GRID),
               [{"name": "Jaccard (bottom-p%)", "y": list(j_point), "lo": list(j_lo), "hi": list(j_hi),
                 "color": PALETTE[0]},
                {"name": "F1 (bottom-p%)", "y": list(f_point), "lo": list(f_lo), "hi": list(f_hi),
                 "color": PALETTE[1]}],
               "p (bottom proportion)", "overlap score",
               subtitle="Lowest-scored p% of each judge; bootstrap over descriptions, seed 123, as in P05",
               legend_at="lower right")
    line_chart(figures / "bottom_p_lift_band.svg",
               f"Lift over random baseline with 95% bootstrap band (B={N_BOOT})", list(P_GRID),
               [{"name": "Lift over random", "y": list(l_point), "lo": list(l_lo), "hi": list(l_hi),
                 "color": PALETTE[0]}], "p (bottom proportion)", "lift over random", hline=1.0,
               subtitle="Intersection of the two bottom-p% sets divided by p² N")
    f19 = archived_curves(CONTROL_FIGURES / "F19_judge_bottomp_sensitivity_overlap_band.svg")
    mine = {"jaccard": j_point, "jaccard_lo": j_lo, "jaccard_hi": j_hi, "f1": f_point, "f1_lo": f_lo, "f1_hi": f_hi}
    f19_diff = {k: float(np.max(np.abs(np.array(f19[k]) - mine[k]))) for k in mine if k in f19}
    table_4_7 = []
    for p, (paper_inter, paper_j, paper_f1) in TABLE_4_7.items():
        t = int(round(p * 100)) - 1
        k = int(np.ceil(len(ids) * P_GRID[t]))  # as in sensitivity_point_estimate
        table_4_7.append((p, round(f_point[t] * k), paper_inter, j_point[t], paper_j, f_point[t], paper_f1))

    # ---------------------------------------------------------------- Figure 4-4 (T31)
    checklists = {"qwen": read_json(QWEN_CSTAR)["checklist"], "gemma": read_json(GEMMA_CSTAR)["checklist"]}
    selected_ids = [row["set_id"] for row in read_csv(AUDIT / "A36_human_audit_30_selected_cases.csv")]
    manual = read_csv(AUDIT / "A38_human_audit_30_manual_results.csv")
    items = item_disagreement({"qwen": qwen, "gemma": gemma}, checklists, selected_ids, manual)
    write_csv(out / "item_disagreement.csv", T31_COLUMNS, [[r[c] for c in T31_COLUMNS] for r in items])
    archived = read_csv(AUDIT / "圖表_figures_tables" / "tables" / "T31_human_audit30_item_disagreement.csv")
    t31_diff = [(i + 1, c, str(r[c]), a.get(c)) for i, (r, a) in enumerate(zip(items, archived)) for c in T31_COLUMNS
                if str(r[c]) != a.get(c)]
    t31_same = len(items) == len(archived) and not t31_diff
    top = items[:12]  # P06: the twelve items with the largest weighted disagreement
    hbar_chart(figures / "figure_4_4_item_disagreement.svg",
               "Weighted item disagreement, judge vs human audit (thesis Figure 4-4)",
               [r["short_label"] for r in top], [float(r["weighted_item_disagreement_0_1"]) for r in top],
               "Weighted item disagreement rate (0-1)", value_fmt="{:.3f}",
               subtitle=f"{len(selected_ids)} audited descriptions; the 12 of {len(items)} checklist items "
                        "with the largest disagreement",
               colors=[GEMMA_COLOR if r["checklist"] == "gemma" else QWEN_COLOR for r in top],
               legend=[("Qwen", QWEN_COLOR, "box"), ("Gemma", GEMMA_COLOR, "box")])

    # ---------------------------------------------------------------- A05 / A06
    scan_rows = []
    for name in FILES_2025:
        status, hits = search(MODEL_CODE / name)
        scan_rows.append(["2025 list (A05)", rel(MODEL_CODE / name), status, "|".join(hits),
                          "no_judge_artifact_reference_found" if status == "searched" and not hits
                          else ("review_needed" if hits else "not in the repository")])
    closure = training_closure()
    code = sorted({p.resolve() for p in list(SCRIPTS.rglob("*.py")) + list(SCRIPTS.rglob("*.sh"))
                   + list(MODEL_CODE.rglob("*.py")) if "__pycache__" not in p.parts} | closure)
    for path in code:
        status, hits = search(path)
        in_training = path in closure
        scan_rows.append([TRAINING_SCOPE if in_training else "reproduction: other scripts",
                          rel(path), status, "|".join(hits),
                          ("review_needed" if in_training else "reads judge outputs; not run by the 35 units")
                          if hits else "no_judge_artifact_reference_found"])
    write_csv(out / "judge_reference_scan.csv",
              ["scope", "searched_file", "status", "judge_artifact_terms_found", "assessment"], scan_rows)
    fashionclip = D02 / "fashionclip_data"
    inputs = [GENERATED, GENERATED.with_name("new_polyvore_outfit_titles_with_ablation.json"),
              fashionclip / "encoded_NewoutfitUrlTitle_en_fashionClip.pkl",
              fashionclip / "encoded_no_weather_outfitUrlTitle_en_fashionClip.pkl",
              fashionclip / "encoded_no_occasion_outfitUrlTitle_en_fashionClip.pkl",
              fashionclip / "encoded_no_style_outfitUrlTitle_en_fashionClip.pkl", QWEN_SCORES, GEMMA_SCORES]
    write_csv(out / "judge_input_roles.csv", ["artifact", "exists", "role"],
              [[rel(p), str(p.is_file()), "downstream_description_or_feature_input" if "phase3_scores" not in p.name
                else "post_hoc_judge_output_not_downstream_input"] for p in inputs])
    training_hits = [r for r in scan_rows if r[0] == TRAINING_SCOPE and r[3]]
    other_hits = [r for r in scan_rows if r[0] == "reproduction: other scripts" and r[3]]

    # ---------------------------------------------------------------- summary
    lines = [
        "# Judge and audit checks",
        "",
        "Computed by `reproduction/scripts/supplementary/judge_audit_checks.py` from the preserved judge scores",
        "and the preserved human audit; no judge or model was run.",
        "",
        "## Score distribution (thesis Figure 4-1; P05 cell 3)",
        "",
        f"{len(ids):,} descriptions scored by both judges. `figures/figure_4_1_judge_score_histogram.svg`.",
        "",
        "| Judge | Mean | SD | IQR | P(score < 0.8) | P(score < 0.9) | Thesis |",
        "|---|---:|---:|---:|---:|---:|---:|",
        *[f"| {d['judge']} | {d['mean']:.4f} | {d['std']:.4f} | {d['iqr']:.4f} | {d['P(score<0.8)']:.4f} | "
          f"{d['P(score<0.9)']:.4f} | {THESIS_MEANS[d['judge']][0]:.3f} ± {THESIS_MEANS[d['judge']][1]:.3f} |"
          for d in diag],
        "",
        f"Mean and SD equal the 2025 table T19 to all digits: {'yes' if t19_same else 'NO'}. The 100 bars equal "
        f"the thesis figure (the archived SVG F17): {'yes' if f17_same else 'NO'}.",
        "",
        "## Bottom-p% sensitivity (thesis Figure 4-3; P05 cell 3)",
        "",
        f"p = 1% to 30%, bootstrap B = {N_BOOT}, seed {SEED_BAND}. `bottom_p_sensitivity_band.csv`,",
        "`figures/figure_4_3_bottom_p_overlap_band.svg`, `figures/bottom_p_lift_band.svg`.",
        "",
        "| p | Intersection | Thesis Table 4-7 | Jaccard | Thesis | F1 | Thesis |",
        "|---:|---:|---:|---:|---:|---:|---:|",
        *[f"| {p:.2f} | {inter} | {paper_inter} | {j:.4f} | {pj:.4f} | {f:.4f} | {pf:.4f} |"
          for p, inter, paper_inter, j, pj, f, pf in table_4_7],
        "",
        "Many descriptions share a score, so which ones fall in the lowest p% depends on how ties are ordered",
        "(`numpy.argsort`); the intersections differ from the thesis by up to "
        f"{max(abs(inter - paper_inter) for _, inter, paper_inter, *_ in table_4_7)} descriptions, as in the",
        "pipeline's recomputation of Table 4-7. The curve and the band are computed with the same ordering.",
        "Largest difference from the thesis figure (the archived SVG F19), over the 30 values of p: "
        + ", ".join(f"{k} {v:.4f}" for k, v in f19_diff.items()) + ".",
        "",
        "## Item disagreement between judge and human audit (2025 table T31, thesis Figure 4-4; P06 cell 10)",
        "",
        f"{len(selected_ids)} audited descriptions (A36), {len(manual)} human answers (A38). `item_disagreement.csv`,",
        "`figures/figure_4_4_item_disagreement.svg`.",
        "",
        "| Rank | Item | Weight | Disagreement | Weighted | Model − human |",
        "|---:|---|---:|---:|---:|---:|",
        *[f"| {r['rank']} | {r['short_label']} ({r['category']}) | {r['weight']:g} | {r['item_disagreement_rate_0_1']} | "
          f"{r['weighted_item_disagreement_0_1']} | {r['mean_signed_model_minus_human']} |" for r in items[:12]],
        "",
        f"Equal to the archived T31 in all {len(T31_COLUMNS)} compared columns and the order of the "
        f"{len(archived)} rows: {'yes' if t31_same else 'NO ' + str(t31_diff[:5])} (the Chinese translation of "
        "each question, a display aid, is not recomputed).",
        "",
        "## Judge outputs in the training and evaluation code (2025 tables A05 and A06)",
        "",
        f"Search terms (2025): {', '.join(JUDGE_TERMS)}. `judge_reference_scan.csv`, `judge_input_roles.csv`.",
        "",
        f"- The 2025 list of {len(FILES_2025)} files: "
        f"{sum(1 for r in scan_rows if r[0] == '2025 list (A05)' and r[2] == 'searched' and not r[3])} found in "
        f"the repository and free of judge terms; "
        f"{sum(1 for r in scan_rows if r[0] == '2025 list (A05)' and r[2] == 'missing_file')} not in the "
        "repository (the two t-test notebooks).",
        f"- Files executed for the reproduction's 35 units ({len(closure)}: the archived model code, and the scripts "
        "named or imported by the split freezing, the main and ablation stages and the statistics, followed "
        "recursively): "
        + ("none references a judge output." if not training_hits else
           "REVIEW " + "; ".join(f"{r[1]} ({r[3]})" for r in training_hits)),
        "- Other reproduction scripts that read judge outputs (post-hoc analyses, not run by the 35 units): "
        + (", ".join(f"`{r[1].split('/')[-1]}`" for r in other_hits) or "none") + ".",
    ]
    write_text(out / "summary.md", lines)
    print(f"[SUPPLEMENTARY] judge and audit checks written to {out}")


if __name__ == "__main__":
    main()
