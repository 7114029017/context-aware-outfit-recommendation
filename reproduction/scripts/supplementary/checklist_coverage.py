#!/usr/bin/env python3
"""Concept coverage between the two judges' weighted checklists (thesis Figure 4-2).

The computation is copied from cell 1 of the 2025 notebook
03_實驗與結果_experiments_results/00_控制檢查與附加稽核/source_programs/P05_llm_judge_agreement_analysis.ipynb:
the items of the Gemma checklist C*_A and of the Qwen checklist C*_B are embedded with the text model
nomic-ai/nomic-embed-text-v2-moe (prompt "passage", normalized), every A item is matched to its most
similar B item and vice versa, and the weighted share of items whose best match reaches a similarity
threshold tau gives Cov A->B, Cov B->A and their F1 for tau = 0.50 to 1.00. The category-level JSD
and L1 distance of the two checklists' weights are also recomputed.

The notebook used sentence-transformers, which the reproduction environment does not include; this
script runs the same three steps of that model's sentence-transformers configuration (mean pooling over
all tokens including the prompt, L2 normalization) with transformers, on the CPU, and loads the model
code (nomic-ai/nomic-bert-2048, the revision recorded in the notebook's log) and the weights as
sentence-transformers did, with the same key remapping and strict loading. Both are downloaded by
`bootstrap_data.sh --with-nomic` (about 1.9 GB) at pinned revisions; without them the script reports
SKIPPED and writes nothing.

Writes under --out-dir (default reproduction/results/supplementary/checklist_coverage/):
coverage_vs_tau.csv, best_matches.csv, category_distribution.csv,
figures/figure_4_2_checklist_coverage.svg and summary.md, which compares every value with the
outputs stored in the 2025 notebook.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import re
import sys
import types
from pathlib import Path

import numpy as np
from scipy.spatial.distance import jensenshannon

from _common import D01, D03, REPRO, SUPPLEMENTARY, read_json, write_csv, write_text
from _svg import PALETTE, line_chart

JUDGES = D01 / "llm_judge_checklists"
PATH_A = JUDGES / "Gemma3" / "checklist_C_star_Gemma3.json"            # P05: path_a
PATH_B = JUDGES / "Qwen3_Instruct" / "checklist_C_star_Qwen3VL32B.json"  # P05: path_b
NOTEBOOK = D03 / "00_控制檢查與附加稽核" / "source_programs" / "P05_llm_judge_agreement_analysis.ipynb"
MODEL_REVISION = "1066b6599d099fbb93dfcb64f9c37a7c9e503e85"  # nomic-ai/nomic-embed-text-v2-moe
CODE_REVISION = "7710840340a098cfb869c4f65e87cf2b1b70caca"   # nomic-ai/nomic-bert-2048, as in the P05 log
PINNED = {  # SHA-256 of the downloaded files
    "model.safetensors": "097012b27af76d80af74fed4bc2ccc9091245286f776adf03ad1758a24ade9a0",
    "tokenizer.json": "3a56def25aa40facc030ea8b0b87f3688e4b3c39eb8b45d5702b3a1300fe2a20",
    "modeling_hf_nomic_bert.py": "3b24a366c4cc31b869466ccfb7bbb8879e138c97f8de06c83d4fa1e31a21f149",
    "configuration_hf_nomic_bert.py": "f7871694b8de3d3df4ac6640313d5799ce323261a0fb90c5cc567ecc34a0039e",
}
PROMPT = "search_document: "  # config_sentence_transformers.json, prompt "passage"
MAX_SEQ_LENGTH = 512          # sentence_bert_config.json
TAU_FOCUS_Q = 0.90
TAUS = [float(round(x, 4)) for x in np.arange(0.5, 1.0 + 1e-12, 0.01)]
COLUMNS = ["tau", "lenA", "lenB", "Cov_A_to_B_weighted (recall-like)", "Cov_B_to_A_weighted (precision-like)",
           "F1_weighted", "Cov_A_to_B_count", "Cov_B_to_A_count", "AvgMaxSim_A_to_B", "AvgMaxSim_B_to_A"]


def recorded(name: str, arg: str | None) -> Path | None:
    if arg:
        return Path(arg).expanduser().resolve()
    path = REPRO / ".local" / f"{name}.txt"
    return Path(path.read_text(encoding="utf-8").strip()).resolve() if path.is_file() else None


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_checklist(path: Path) -> list[dict]:
    """P05 load_checklist (id, text stripped, category, weight)."""
    return [{"id": str(i["id"]), "text": str(i["text"]).strip(), "category": str(i["category"]),
             "weight": float(i["weight"])} for i in read_json(path)["checklist"]]


class NomicEncoder:
    """SentenceTransformer("nomic-ai/nomic-embed-text-v2-moe", trust_remote_code=True).encode(..., prompt_name="passage",
    normalize_embeddings=True): Transformer -> mean Pooling (include_prompt) -> Normalize."""

    def __init__(self, model_dir: Path, code_dir: Path):
        import torch
        from safetensors.torch import load_file
        from transformers import AutoTokenizer

        sys.dont_write_bytecode = True
        package = types.ModuleType("nomic_bert_2048")
        package.__path__ = [str(code_dir)]
        sys.modules["nomic_bert_2048"] = package
        configuration = importlib.import_module("nomic_bert_2048.configuration_hf_nomic_bert")
        modeling = importlib.import_module("nomic_bert_2048.modeling_hf_nomic_bert")
        config = configuration.NomicBertConfig.from_pretrained(str(model_dir))
        # NomicBertModel.from_pretrained with a hub name, as sentence-transformers called it: remap, strict load
        model = modeling.NomicBertModel(config, add_pooling_layer=False)
        state = modeling.remap_bert_state_dict(load_file(str(model_dir / "model.safetensors")), config, remove_bert=True,
                                               remove_cls_weights=True,
                                               add_pooling_layer=getattr(config, "add_pooling_layer", False))
        model.load_state_dict(state, strict=True)
        self.model = model.eval()
        self.tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
        self.torch = torch

    def encode(self, texts: list[str]) -> np.ndarray:
        torch = self.torch
        texts = [(PROMPT + t).strip() for t in texts]
        features = self.tokenizer(texts, padding=True, truncation="longest_first", return_tensors="pt",
                                  max_length=MAX_SEQ_LENGTH)
        with torch.no_grad():
            tokens = self.model(**features, return_dict=False)[0]
        mask = features["attention_mask"].unsqueeze(-1).to(tokens.dtype)
        pooled = (tokens * mask).sum(1) / torch.clamp(mask.sum(1), min=1e-9)
        return torch.nn.functional.normalize(pooled, p=2, dim=1).numpy()


def stored_outputs() -> dict:
    """Values printed by P05 cell 1 in 2025 (the notebook keeps its outputs)."""
    cell = read_json(NOTEBOOK)["cells"][1]
    text = "".join("".join(o.get("text", "")) for o in cell["outputs"] if o.get("output_type") == "stream")
    out = {"table": []}
    for o in cell["outputs"]:
        html = "".join(o.get("data", {}).get("text/html", ""))
        if "Cov_A_to_B_weighted" in html:
            out["table"] = [re.findall(r"<td>([^<]*)</td>", row) for _, row in
                            re.findall(r"<tr>\s*<th>(\d+)</th>(.*?)</tr>", html, re.S)]
    for key, pattern in (("avg_ab", r"AvgMaxSim_A→B = ([\d.]+)"), ("avg_ba", r"AvgMaxSim_B→A = ([\d.]+)"),
                         ("jsd", r"JSD=([\d.]+)"), ("l1", r"L1=([\d.]+)"), ("tau_focus", r"TAU_FOCUS = ([\d.]+)"),
                         ("uncovered_a", r"A uncovered: (\d+)/"), ("uncovered_b", r"B uncovered: (\d+)/")):
        match = re.search(pattern, text)
        out[key] = match.group(1) if match else None
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out-dir", type=Path, default=SUPPLEMENTARY / "checklist_coverage")
    parser.add_argument("--model-dir", default=None, help="default: the path recorded by bootstrap_data.sh --with-nomic")
    parser.add_argument("--code-dir", default=None, help="default: the path recorded by bootstrap_data.sh --with-nomic")
    parser.add_argument("--polyvore-root", default=None, help="accepted for run_all.sh; not used")
    args = parser.parse_args()

    model_dir, code_dir = recorded("nomic_model_root", args.model_dir), recorded("nomic_code_root", args.code_dir)
    if model_dir is None or code_dir is None or not (model_dir / "model.safetensors").is_file() \
            or not (code_dir / "modeling_hf_nomic_bert.py").is_file():
        print("[SUPPLEMENTARY] checklist coverage SKIPPED: the Nomic embedding model is not downloaded "
              "(bash reproduction/scripts/bootstrap_data.sh --with-nomic)")
        return
    for name, digest in PINNED.items():
        path = (model_dir if name in ("model.safetensors", "tokenizer.json") else code_dir) / name
        if sha256(path) != digest:
            raise SystemExit(f"[SUPPLEMENTARY BLOCKED] {path} differs from the pinned revision")
    out = args.out_dir

    A, B = load_checklist(PATH_A), load_checklist(PATH_B)
    # category-level similarity (P05 section 2)
    def distribution(items):
        total = {}
        for i in items:
            total[i["category"]] = total.get(i["category"], 0.0) + i["weight"]
        s = sum(total.values())
        return {k: v / s for k, v in total.items()}
    pA, pB = distribution(A), distribution(B)
    cats = sorted(set(pA) | set(pB))
    vecA = np.array([pA.get(c, 0.0) for c in cats])
    vecB = np.array([pB.get(c, 0.0) for c in cats])
    jsd = float(jensenshannon(vecA, vecB, base=2.0) ** 2)
    l1 = float(np.abs(vecA - vecB).sum())
    write_csv(out / "category_distribution.csv", ["category", "A_weight_ratio", "B_weight_ratio"],
              [[c, repr(float(a)), repr(float(b))] for c, a, b in zip(cats, vecA, vecB)])

    # embeddings and many-to-many coverage (P05 sections 3 and 4)
    encoder = NomicEncoder(model_dir, code_dir)
    S = encoder.encode([i["text"] for i in A]) @ encoder.encode([i["text"] for i in B]).T
    wA, wB = np.array([i["weight"] for i in A]), np.array([i["weight"] for i in B])
    maxA, maxB = S.max(axis=1), S.max(axis=0)
    bestB, bestA = S.argmax(axis=1), S.argmax(axis=0)
    avg_ab, avg_ba = float((wA * maxA).sum() / wA.sum()), float((wB * maxB).sum() / wB.sum())
    tau_focus = float(np.quantile(np.concatenate([maxA, maxB]), TAU_FOCUS_Q))
    rows = []
    for tau in TAUS:
        cov_ab = float((wA * (maxA >= tau)).sum() / wA.sum())
        cov_ba = float((wB * (maxB >= tau)).sum() / wB.sum())
        f1 = 0.0 if (cov_ba + cov_ab) == 0 else float(2 * cov_ba * cov_ab / (cov_ba + cov_ab))
        rows.append([tau, len(A), len(B), cov_ab, cov_ba, f1, float((maxA >= tau).mean()), float((maxB >= tau).mean()),
                     avg_ab, avg_ba])
    write_csv(out / "coverage_vs_tau.csv", COLUMNS, [[v if isinstance(v, int) else repr(v) for v in r] for r in rows])
    write_csv(out / "best_matches.csv", ["direction", "item", "category", "weight", "best_match", "max_similarity",
                                         f"covered_at_tau_{tau_focus:.4f}"],
              [["A->B", f"Gemma {A[i]['id']}", A[i]["category"], A[i]["weight"], f"Qwen {B[bestB[i]]['id']}",
                f"{maxA[i]:.6f}", "yes" if maxA[i] >= tau_focus else ""] for i in np.argsort(-maxA, kind="stable")]
              + [["B->A", f"Qwen {B[j]['id']}", B[j]["category"], B[j]["weight"], f"Gemma {A[bestA[j]]['id']}",
                  f"{maxB[j]:.6f}", "yes" if maxB[j] >= tau_focus else ""] for j in np.argsort(-maxB, kind="stable")])
    line_chart(out / "figures" / "figure_4_2_checklist_coverage.svg",
               "Many-to-many rubric coverage vs tau (Nomic v2, passage/passage) (thesis Figure 4-2)", TAUS,
               [{"name": "Cov A→B (weighted)", "y": [r[3] for r in rows], "color": PALETTE[0]},
                {"name": "Cov B→A (weighted)", "y": [r[4] for r in rows], "color": PALETTE[1]},
                {"name": "F1 (weighted)", "y": [r[5] for r in rows], "color": PALETTE[2]}],
               "tau (similarity threshold)", "score", legend_at="lower left",
               subtitle=f"A = Gemma checklist ({len(A)} items), B = Qwen checklist ({len(B)} items)")

    # comparison with the stored 2025 outputs
    stored = stored_outputs()
    mine = [[f"{r[0]:.2f}", str(r[1]), str(r[2]), *[f"{v:.6f}" for v in r[3:8]], f"{r[8]:.6f}", f"{r[9]:.6f}"]
            for r in rows]
    diff = []
    for m, s in zip(mine, stored["table"]):
        for c, (a, b) in enumerate(zip(m, s)):
            if a != b and not (c == 7 and float(a) == float(b)):  # Cov_B_to_A_count is printed with one decimal
                diff.append((m[0], COLUMNS[c], a, b))
    uncovered_a, uncovered_b = int((maxA < tau_focus).sum()), int((maxB < tau_focus).sum())
    focus = next((r for r in rows if abs(r[0] - 0.80) < 1e-9), None)
    drop = next((r[0] for r in rows if r[5] < 1.0), None)
    lines = [
        "# Checklist coverage (thesis Figure 4-2)",
        "",
        "Computed by `reproduction/scripts/supplementary/checklist_coverage.py` (notebook P05 cell 1) with",
        f"nomic-ai/nomic-embed-text-v2-moe at revision {MODEL_REVISION} (model code nomic-ai/nomic-bert-2048 at",
        f"{CODE_REVISION}), on the CPU. A = Gemma checklist C* ({len(A)} items), B = Qwen checklist C* ({len(B)} items).",
        "",
        "| Value | Recomputed | 2025 (stored notebook output) |",
        "|---|---:|---:|",
        f"| AvgMaxSim A→B | {avg_ab:.6f} | {stored['avg_ab']} |",
        f"| AvgMaxSim B→A | {avg_ba:.6f} | {stored['avg_ba']} |",
        f"| Category JSD | {jsd:.6f} | {stored['jsd']} |",
        f"| Category L1 | {l1:.6f} | {stored['l1']} |",
        f"| tau focus (90% quantile of the best-match similarities) | {tau_focus:.6f} | "
        f"{float(stored['tau_focus']):.6f} |" if stored["tau_focus"] else "",
        f"| Uncovered at tau focus: A / B | {uncovered_a} / {uncovered_b} | {stored['uncovered_a']} / {stored['uncovered_b']} |",
        "",
        f"Coverage table, {len(rows)} values of tau x 10 columns, at the six decimals printed in 2025: "
        + ("identical." if not diff and len(stored["table"]) == len(rows) else f"{len(diff)} differences, e.g. {diff[:4]}."),
        "",
        f"At tau = 0.80: Cov A→B {focus[3]:.4f}, Cov B→A {focus[4]:.4f}, F1 {focus[5]:.4f}; the weighted F1 first drops "
        f"below 1 at tau = {drop:.2f} (the thesis: all three stay high up to 0.80)." if focus else "",
        "",
        "Files: `coverage_vs_tau.csv` (the 2025 metrics table), `best_matches.csv`, `category_distribution.csv`,",
        "`figures/figure_4_2_checklist_coverage.svg`.",
    ]
    write_text(out / "summary.md", [line for line in lines if line is not None])
    print(f"[SUPPLEMENTARY] checklist coverage written to {out}")


if __name__ == "__main__":
    main()
