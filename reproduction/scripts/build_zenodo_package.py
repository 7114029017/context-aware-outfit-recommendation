#!/usr/bin/env python3
"""Build the Zenodo upload set of this repository.

Writes to <out-dir>/upload/ three zip archives that share one top
directory, so they unpack into a single tree:

- <name>-<version>-code.zip: the tracked files of HEAD (Git LFS files
  materialized) except the large binaries in the data archive and the files
  in EXCLUDED (third-party publications, manuscripts and figures with
  Polyvore product photos). The notebooks in STRIP_IMAGES keep their code
  and text outputs but lose their embedded images;
- <name>-<version>-data.zip: the eight feature files, checked against
  verify_feature_files.py, and the 40 checkpoints of the handoff package;
- official-run-<run id>.zip: the complete directory of the official run,
  unpacking to reproduction/runs/<run id>/. Its 70 checkpoints are checked
  against results/final_reference_manifest.json and its 95 seed-level
  result files against the official seed index;

plus package_manifest.json, SHA256SUMS.txt and README.md. It also writes
<out-dir>/zenodo_form.md, the record metadata to enter on Zenodo (not
uploaded). Members are sorted and timestamps fixed, so rebuilding the same
commit gives byte-identical archives. The repository and the official run
directory are only read.
"""
from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import re
import shutil
import subprocess
import zipfile
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
REPRO = ROOT / "reproduction"
MANIFEST = REPRO / "results" / "final_reference_manifest.json"
FEATURE_SCRIPT = REPRO / "scripts" / "verify_feature_files.py"
NAME = "context-aware-outfit-recommendation"
TITLE = "Context-Aware Semantic Construction for Outfit Recommendation: reproducibility package"
D01 = "01_資料建構_data_construction"
D02 = "02_模型訓練和驗證_model_training_validation"
D03 = "03_實驗與結果_experiments_results"
D04 = "04_文件資料_documents"
FEATURE_DIR = f"{D02}/fashionclip_data/"
FIG00 = f"{D03}/00_控制檢查與附加稽核/圖表_figures_tables/figures"
FIG04 = f"{D03}/04_情境子集與三因子分析/圖表_figures_tables/figures"
FIG06 = f"{D03}/06_反事實情境敏感度/圖表_figures_tables/figures"

EXCLUDED = {
    f"{D01}/clo_met_temperature/CLO_reference/2021 ASHRAE 69套套裝.pdf": "third-party publication (ASHRAE)",
    f"{D01}/clo_met_temperature/CLO_reference/ASHRARE 2010 clo.pdf": "third-party publication (ASHRAE)",
    f"{D01}/clo_met_temperature/CLO_reference/ssrn-5357611 實測.pdf": "third-party publication (SSRN 5357611)",
    f"{D01}/clo_met_temperature/MET_reference/1_2024-adult-compendium_1_2024.pdf":
        "third-party publication (2024 Adult Compendium of Physical Activities)",
    f"{D01}/clo_met_temperature/temperature_results/DonMcIntyreComfortRqtsECRCOct78.pdf":
        "third-party publication (McIntyre, 1978)",
    f"{D02}/main_hybrid_attention_code/Text-Conditioned_Outfit_Recommendation_With_Hybrid_Attention_Layer.pdf":
        "third-party publication (Wang & Zhong, IEEE Access, 2024)",
    f"{D04}/journal/ACM_TORS_English_Condensed.docx": "manuscript under review",
    f"{D04}/journal/ACM_TORS_English_v4.pdf": "manuscript under review",
    f"{D04}/thesis/情境感知驅動的智慧穿搭推薦系統 論文終稿.pdf": "thesis; its figures contain Polyvore product images",
    f"{FIG04}/F21_qualitative_retrieval_case_original_vs_context_aware.svg": "case figure with Polyvore product photos",
    f"{FIG06}/F34a_counterfactual_context_larger_response_example.svg": "case figure with Polyvore product photos",
    f"{FIG06}/F34b_counterfactual_context_limited_response_example.svg": "case figure with Polyvore product photos",
}
STRIP_IMAGES = {
    f"{D01}/clo_met_temperature/CLO_reference/Adding_CLO.ipynb": "Polyvore product photos",
    f"{D03}/04_情境子集與三因子分析/source_programs/P02_subset_robustness_analysis.ipynb":
        "Polyvore product photos and virtual try-on renders",
    f"{D03}/04_情境子集與三因子分析/source_programs/P03_qualitative_case_analysis.ipynb":
        "case figures built from Polyvore product photos",
}
# Reviewed: their embedded images are statistics charts only.
CHART_NOTEBOOKS = {
    f"{D03}/00_控制檢查與附加稽核/source_programs/P05_llm_judge_agreement_analysis.ipynb",
    f"{D03}/00_控制檢查與附加稽核/source_programs/P06_prompt_robustness_reproducible.ipynb",
    f"{D03}/03_主推薦任務結果/source_programs/P01_main_results_and_ablation_5seed_ttest.ipynb",
}
# Reviewed: a Matplotlib chart.
ALLOWED_DOCUMENTS = {f"{D03}/03_主推薦任務結果/圖表_figures_tables/figures/F01_main_factor_contribution_proportion.pdf"}
# Reviewed: charts and a flow diagram. SVG files pass when they embed no raster image.
REVIEWED_IMAGES = {
    f"{D01}/generated_descriptions/圖表_figures_tables/figures/套裝描述改寫(含天氣、行程與風格) 的複本 (4).png",
    *(f"{FIG00}/{name}" for name in (
        "F12_reliability_performance_comparison.png", "F13_reliability_diagram_comparison.png",
        "F14_reliability_error_slicing_comparison.png", "F15_reliability_subgroup_hit10_comparison.png",
        "F16_reliability_subgroup_delta_hit10_heatmap.png", "F18_judge_qwen_gemma_quantile_confusion.png",
        "F20_prompt_robustness_absolute_difference.png")),
    *(f"{FIG04}/{name}" for name in (
        "F02_subset_hitrate_comparison.png", "F03_subset_delta_hit10_heatmap.png",
        "F04_subset_ablation_across_subsets.png", "F07_qualitative_category_original_vs_full.png",
        "F08_qualitative_condition_occasion_original_vs_full.png")),
}
DOCUMENT_SUFFIXES = (".pdf", ".doc", ".docx", ".ppt", ".pptx")
IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp", ".tif", ".tiff")
STORED_SUFFIXES = (".pkl", ".pt")  # already dense; stored without compression
SECRET = re.compile(rb"sk-[A-Za-z0-9_-]{20,}|hf_[A-Za-z0-9]{30,}|ghp_[A-Za-z0-9]{36}|AIza[0-9A-Za-z_-]{35}")
LFS_PREFIX = b"version https://git-lfs.github.com/spec/v1"
CHUNK = 8 * 1024 * 1024


def fail(message: str) -> None:
    raise SystemExit("[PACKAGE BLOCKED] " + message)


def git(*args: str) -> bytes:
    return subprocess.run(["git", "-C", str(ROOT), *args], check=True, capture_output=True).stdout


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def digests(path: Path) -> tuple[str, str]:
    """Git blob SHA-1 and SHA-256 of a file."""
    blob = hashlib.sha1(b"blob %d\0" % path.stat().st_size)
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(CHUNK), b""):
            blob.update(chunk)
            h.update(chunk)
    return blob.hexdigest(), h.hexdigest()


def sha256_file(path: Path) -> str:
    return digests(path)[1]


# ---- source tree --------------------------------------------------------
def head_entries() -> list[dict]:
    """Every blob of HEAD: path, mode, git blob ID and size."""
    entries = []
    for record in git("ls-tree", "-r", "-l", "-z", "HEAD").split(b"\0"):
        if not record:
            continue
        meta, path = record.split(b"\t", 1)
        mode, kind, oid, size = meta.split()
        if kind != b"blob":
            fail(f"unexpected {kind.decode()} entry: {path.decode()}")
        entries.append({"path": path.decode(), "mode": mode.decode(), "oid": oid.decode(), "size": int(size)})
    return entries


def lfs_pointers(entries: list[dict]) -> dict[str, dict]:
    """Path -> {oid, size} for the HEAD blobs that are Git LFS pointers."""
    small = [e for e in entries if e["size"] < 1024]
    out = subprocess.run(["git", "-C", str(ROOT), "cat-file", "--batch"], check=True, capture_output=True,
                         input=b"".join(e["oid"].encode() + b"\n" for e in small)).stdout
    pointers, pos = {}, 0
    for entry in small:
        header_end = out.index(b"\n", pos)
        size = int(out[pos:header_end].split()[2])
        content = out[header_end + 1:header_end + 1 + size]
        pos = header_end + 1 + size + 1
        if content.startswith(LFS_PREFIX):
            fields = dict(line.split(" ", 1) for line in content.decode().splitlines() if " " in line)
            pointers[entry["path"]] = {"oid": fields["oid"].removeprefix("sha256:"), "size": int(fields["size"])}
    return pointers


def verify_worktree(entries: list[dict], pointers: dict[str, dict]) -> dict[str, str]:
    """Check every working-tree file against HEAD; return path -> SHA-256."""
    if git("status", "--porcelain", "--untracked-files=no").strip():
        fail("tracked files differ from HEAD; commit or restore them first")
    sha = {}
    for entry in entries:
        path = ROOT / entry["path"]
        if not path.is_file():
            fail(f"missing in working tree: {entry['path']}")
        blob, digest = digests(path)
        pointer = pointers.get(entry["path"])
        if pointer:
            if digest != pointer["oid"] or path.stat().st_size != pointer["size"]:
                fail(f"Git LFS file not materialized or changed (run git lfs pull): {entry['path']}")
        elif blob != entry["oid"]:
            fail(f"working tree differs from HEAD: {entry['path']}")
        sha[entry["path"]] = digest
    return sha


def expected_features() -> dict[str, tuple[int, str]]:
    tree = ast.parse(FEATURE_SCRIPT.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(getattr(t, "id", None) == "EXPECTED" for t in node.targets):
            return ast.literal_eval(node.value)
    fail(f"no EXPECTED feature table in {FEATURE_SCRIPT}")


def notebook_images(data: dict) -> int:
    return sum(1 for cell in data.get("cells", []) for output in cell.get("outputs", [])
               for mime in (output.get("data") or {}) if mime.startswith("image/"))


def strip_images(raw: bytes) -> tuple[bytes, int]:
    """Notebook without embedded images; code and text outputs are kept."""
    data = json.loads(raw)
    removed = notebook_images(data)
    for cell in data.get("cells", []):
        kept = []
        for output in cell.get("outputs", []):
            mimes = output.get("data")
            if mimes and any(m.startswith("image/") for m in mimes):
                output["data"] = {m: v for m, v in mimes.items() if not m.startswith("image/")}
                output["metadata"] = {}
                if not output["data"]:
                    continue
            kept.append(output)
        if "outputs" in cell:
            cell["outputs"] = kept
    return (json.dumps(data, ensure_ascii=False, indent=1) + "\n").encode("utf-8"), removed


def text_for_scan(path: str, data: bytes) -> bytes | None:
    """Content to scan for credentials; embedded images are left out."""
    if path.endswith(".ipynb"):
        nb = json.loads(data)
        for cell in nb.get("cells", []):
            for output in cell.get("outputs", []):
                output["data"] = {m: v for m, v in (output.get("data") or {}).items() if not m.startswith("image/")}
        return json.dumps(nb, ensure_ascii=False).encode("utf-8")
    try:
        data.decode("utf-8")
    except UnicodeDecodeError:
        return None
    return data


# ---- archives -----------------------------------------------------------
def write_zip(target: Path, members: list[tuple[str, bytes | Path, bool]], when: tuple) -> None:
    """Deterministic zip; a member's payload is bytes or a file to stream."""
    with zipfile.ZipFile(target, "w") as archive:
        for name, payload, executable in sorted(members, key=lambda m: m[0]):
            info = zipfile.ZipInfo(name, when)
            info.external_attr = (0o100755 if executable else 0o100644) << 16
            info.compress_type = zipfile.ZIP_STORED if name.endswith(STORED_SUFFIXES) else zipfile.ZIP_DEFLATED
            if isinstance(payload, bytes):
                archive.writestr(info, payload)
            else:
                info.file_size = payload.stat().st_size
                with payload.open("rb") as source, archive.open(info, "w") as dest:
                    shutil.copyfileobj(source, dest, CHUNK)


def member_list(prefix: str, members: list[tuple[str, Path]]) -> list[dict]:
    return [{"path": prefix + name, "bytes": path.stat().st_size, "sha256": sha256_file(path)}
            for name, path in sorted(members)]


# ---- documents ----------------------------------------------------------
def snapshot_notes(commit: str, run_id: str, version: str, excluded: list[dict], stripped: list[dict]) -> str:
    lines = [
        f"# Snapshot notes ({version})",
        "",
        f"This directory holds the tracked files of commit `{commit}` of",
        f"`{NAME}` with these differences:",
        "",
        f"- The large binaries are in `{NAME}-{version}-data.zip` of this Zenodo record, which unpacks",
        "  into this directory:",
        f"  - the eight feature files in `{FEATURE_DIR}`;",
        "  - the 40 `.pt` checkpoints of the handoff package.",
        "- Not included:",
    ]
    lines += [f"  - `{e['path']}`: {e['reason']}." for e in excluded]
    lines += ["- Embedded images removed, code and text outputs kept:"]
    lines += [f"  - `{s['path']}`: {s['images_removed']} images ({s['reason']})." for s in stripped]
    lines += [
        "",
        "Every other file is byte-identical to the commit; `package_manifest.json` of the record lists",
        "each file's SHA-256 and git blob ID.",
        "",
        f"`official-run-{run_id}.zip` of the record unpacks the complete official run into",
        f"`reproduction/runs/{run_id}/`.",
        "",
        "To run the pipeline from this snapshot, unpack the data archive first, then",
        "",
        "    git init -q -b main && git add -A && git commit -qm snapshot",
        "",
        "and follow `reproduction/README.md` from section 0.3. The pipeline records the git commit of",
        "the copy it runs from, so the snapshot needs to be a git repository (`reproduction/runs/` is",
        "git-ignored). Compare a finished run with the official run as described in",
        "`reproduction/README.md` section 0.8.",
        "",
    ]
    return "\n".join(lines)


def record_readme(commit: str, commit_date: str, run_id: str, version: str, files: list[dict],
                  excluded: list[dict], stripped: list[dict]) -> str:
    rows = "\n".join(f"| `{f['file']}` | {f['contents']} | {len(f['members'])} | {f['bytes'] / 2**20:,.0f} MiB |"
                     for f in files)
    excluded_rows = "\n".join(f"- `{e['path']}`: {e['reason']}" for e in excluded)
    stripped_rows = "\n".join(f"- `{s['path']}`: {s['images_removed']} images removed ({s['reason']})" for s in stripped)
    top = f"{NAME}-{version}"
    return f"""# {TITLE}

Version {version}, built from commit `{commit}` ({commit_date}). The official
results come from the clean-room run `{run_id}`: 35 training units (10 main,
25 fair-subset ablation) over seeds 1-5, status `PASSED`.

## Files

| File | Contents | Files inside | Size |
|---|---|---|---|
{rows}

The three archives share the top directory `{top}/`. Unpacked
together, they give the repository as on GitHub (code and data) plus the
complete official run in `reproduction/runs/{run_id}/`.

## Restore

Download the archives into one directory, then:

    unzip {top}-code.zip
    unzip {top}-data.zip
    unzip official-run-{run_id}.zip    # optional: the complete official run
    cd {top}
    git init -q -b main && git add -A && git commit -qm snapshot

The pipeline records the git commit of the copy it runs from, hence the
`git init` step; `reproduction/runs/` is git-ignored. After creating the
Python environment (`reproduction/README.md` section 0.3), check the
restored official run against the published results:

    python3 reproduction/scripts/compare_with_official_run.py \\
      --run-root reproduction/runs/{run_id} --checkpoints

To reproduce, follow `reproduction/README.md` from section 0.3: create the
Python environment, download the Polyvore annotations with
`reproduction/scripts/bootstrap_data.sh`, run the preflight and start
`reproduction/scripts/reproduce_all.sh --fresh`. A full run took 49-55 hours
on one NVIDIA GB10. The results and statistics of the official run are in
`reproduction/results/summary/{run_id}/` and `reproduction/docs/expected_results.md`.

## Verify

    sha256sum -c SHA256SUMS.txt

`package_manifest.json` lists every archive member with its SHA-256; for
files unchanged from the source commit it also gives the git blob ID. The
70 official checkpoints match `reproduction/results/final_reference_manifest.json`,
the 95 seed-level result files match the official seed index, and the eight
feature files match `reproduction/scripts/verify_feature_files.py`.

## Not included

Polyvore images are not redistributed. The pipeline downloads the Polyvore
Outfits annotations from the `Stylique/Polyvore` mirror on Hugging Face;
steps that need the images (case figures, color judgments) cannot be rerun
from this package. The feature files are the original authors' precomputed
features, verified by SHA-256 and not re-extracted.

Left out of the code archive:

{excluded_rows}

Embedded images removed from notebooks, code and text outputs kept:

{stripped_rows}

## Licenses

Code: MIT. The model code in `{D02}/main_hybrid_attention_code/` keeps its
upstream MIT notice. Data, results and documentation: CC BY 4.0.
Polyvore-derived files remain subject to the terms of the Polyvore Outfits
dataset.
"""


def zenodo_form(version: str, run_id: str, commit: str) -> str:
    description = (
        'Reproducibility package for the TORS revision "Context-Aware Semantic Construction for Outfit '
        'Recommendation". The code archive holds the repository: code, configuration, environment records, '
        f"fixed data splits and fair-subset IDs, statistics scripts and the results of the official run {run_id} "
        "(35 training units, seeds 1-5). The data archive holds the precomputed item features and the 40 "
        "checkpoints of the original handoff package. The official-run archive holds the complete run "
        f"directory, including its 70 checkpoints and logs. Built from commit {commit}. Polyvore images, "
        "third-party publications and the manuscripts are not included; see README.md."
    )
    return f"""# Zenodo form ({version})

Enter these fields on the Zenodo upload page. Save as a draft; do not
publish until the authors are decided.

| Field | Value |
|---|---|
| Resource type | Software |
| Title | {TITLE} |
| Creators | (leave empty for now; required before publishing) |
| Version | {version} |
| Language | English |
| Licenses | MIT License; Creative Commons Attribution 4.0 International |
| Keywords | outfit recommendation; context-aware recommendation; fashion compatibility; reproducibility; Polyvore |
| DOI | click "Get a DOI now!" (reserve DOI) before uploading |
| Related works | add later: the GitHub repository (once public) and the paper DOI |

Description (paste as one paragraph):

{description}

Upload every file in `upload/`.
"""


# ---- main ---------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--official-run", required=True, type=Path,
                        help="Directory of the official run (with its checkpoints and logs).")
    parser.add_argument("--out-dir", required=True, type=Path, help="New directory for the upload set.")
    parser.add_argument("--version", default="v1.0.0")
    args = parser.parse_args()

    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    run_id = manifest["official_run_id"]
    run = args.official_run.expanduser().resolve()
    if run.name != run_id:
        fail(f"--official-run must be the directory of {run_id}, got {run}")
    if (run / "RUN_STATUS.txt").read_text(encoding="utf-8").strip() != "PASSED":
        fail(f"official run is not PASSED: {run}")
    out = args.out_dir.expanduser().resolve()
    if out.exists() and any(out.iterdir()):
        fail(f"output directory exists and is not empty: {out}")
    upload = out / "upload"
    top = f"{NAME}-{args.version}/"

    commit = git("rev-parse", "HEAD").decode().strip()
    stamp = int(git("show", "-s", "--format=%ct", "HEAD").decode().strip())
    when = datetime.fromtimestamp(stamp, tz=timezone.utc)
    commit_date = when.strftime("%Y-%m-%d")

    # Source tree checks.
    entries = head_entries()
    pointers = lfs_pointers(entries)
    sha = verify_worktree(entries, pointers)
    by_path = {e["path"]: e for e in entries}
    for listed in (*EXCLUDED, *STRIP_IMAGES, *CHART_NOTEBOOKS, *ALLOWED_DOCUMENTS, *REVIEWED_IMAGES):
        if listed not in by_path:
            fail(f"listed path is not tracked at HEAD: {listed}")
    features = sorted(p for p in by_path if p.startswith(FEATURE_DIR) and p.endswith(".pkl"))
    expected = expected_features()
    if [Path(p).name for p in features] != sorted(expected):
        fail("feature files differ from verify_feature_files.py")
    for p in features:
        size, digest = expected[Path(p).name]
        real_size = pointers[p]["size"] if p in pointers else by_path[p]["size"]
        if real_size != size or sha[p] != digest:
            fail(f"feature file does not match verify_feature_files.py: {p}")
    archived = sorted(p for p in by_path if p.endswith(".pt"))
    if len(archived) != 40:
        fail(f"expected 40 archived checkpoints, found {len(archived)}")
    snapshot = sorted(p for p in by_path if p not in EXCLUDED and p not in features and p not in archived)

    # Code archive content and content checks.
    code_members, records, stripped = [], [], []
    for p in snapshot:
        data = (ROOT / p).read_bytes()
        record = {"path": top + p, "git_blob": by_path[p]["oid"] if p not in pointers else None,
                  "lfs_sha256": pointers[p]["oid"] if p in pointers else None}
        if p in STRIP_IMAGES:
            data, removed = strip_images(data)
            stripped.append({"path": p, "reason": STRIP_IMAGES[p], "images_removed": removed,
                             "source_sha256": sha[p], "packaged_sha256": sha256_bytes(data)})
            record["git_blob"] = record["lfs_sha256"] = None
            record["modified"] = "embedded images removed"
        elif p.endswith(".ipynb") and notebook_images(json.loads(data)) and p not in CHART_NOTEBOOKS:
            fail(f"notebook with embedded images has not been reviewed: {p}")
        if p.lower().endswith(DOCUMENT_SUFFIXES) and p not in ALLOWED_DOCUMENTS:
            fail(f"document not reviewed for publication: {p}")
        if p.lower().endswith(IMAGE_SUFFIXES) and p not in REVIEWED_IMAGES:
            fail(f"image file not reviewed for publication: {p}")
        if p.lower().endswith(".svg") and b"<image" in data:
            fail(f"SVG with an embedded raster image: {p}")
        scan = text_for_scan(p, data)
        if scan is not None and SECRET.search(scan):
            fail(f"credential-like string in {p}")
        record.update({"bytes": len(data), "sha256": sha256_bytes(data)})
        records.append(record)
        code_members.append((top + p, data, by_path[p]["mode"] == "100755"))
    excluded = [{"path": p, "reason": r, "sha256": sha[p]} for p, r in sorted(EXCLUDED.items())]
    notes = snapshot_notes(commit, run_id, args.version, excluded, stripped).encode("utf-8")
    code_members.append((top + "ZENODO_SNAPSHOT.md", notes, False))
    records.append({"path": top + "ZENODO_SNAPSHOT.md", "bytes": len(notes), "sha256": sha256_bytes(notes),
                    "modified": "added by the packaging script"})
    data_items = [(p, ROOT / p) for p in features + archived]

    # Official run: the complete directory.
    run_files = sorted(p for p in run.rglob("*") if p.is_file() or p.is_symlink())
    if any(p.is_symlink() for p in run_files):
        fail(f"symbolic link in the official run directory: {run}")
    for unit in manifest["checkpoints"]["units"]:
        base = run / ("main" if unit["family"] == "main" else "ablation/runs") / f"{unit['variant']}_seed{unit['seed']}"
        for stage in ("cp", "cir"):
            path = base / f"{stage}_best_ckpt.pt"
            if not path.is_file() or sha256_file(path) != unit[f"{stage}_checkpoint_sha256"]:
                fail(f"official checkpoint missing or not matching the manifest: {path}")
    with (ROOT / manifest["results"]["seed_index"]).open(encoding="utf-8-sig") as stream:
        seed_index = list(csv.DictReader(stream))
    results_checked = 0
    for row in seed_index:
        unit = f"{row['variant']}_seed{row['seed']}"
        if row["family"] == "main":
            pairs = [(run / "main" / unit / "evaluation" / "results_cp.csv", row["cp_results_sha256"]),
                     (run / "main" / unit / "evaluation" / "results_cir.csv", row["cir_results_sha256"])]
        else:
            base = run / "ablation" / "runs" / unit
            pairs = [(base / "results_cp_fresh_subset.csv", row["cp_results_sha256"]),
                     (base / "results_cir_fresh_subset.csv", row["cir_results_sha256"]),
                     (base / "detail_cir_fresh_subset.csv", row["detail_results_sha256"])]
        for path, digest in pairs:
            if not path.is_file() or sha256_file(path) != digest:
                fail(f"official result file missing or not matching the seed index: {path}")
            results_checked += 1
    if results_checked != 95:
        fail(f"expected 95 seed-level result files, checked {results_checked}")
    for path in run_files:
        if not path.name.endswith(".pt") and SECRET.search(path.read_bytes()):
            fail(f"credential-like string in {path}")
    run_prefix = f"{top}reproduction/runs/{run_id}/"
    run_items = [(str(p.relative_to(run)), p) for p in run_files]

    # Archives.
    upload.mkdir(parents=True, exist_ok=True)
    when_tuple = when.timetuple()[:6]
    archives = [
        (f"{NAME}-{args.version}-code.zip", "the repository without its large binaries (code, configs, splits, "
         "results, docs)", code_members, records),
        (f"{NAME}-{args.version}-data.zip", "the eight item feature files and the 40 checkpoints of the original "
         "handoff package", [(top + n, p, False) for n, p in data_items], member_list(top, data_items)),
        (f"official-run-{run_id}.zip", "the complete official run directory, including its 70 checkpoints and logs",
         [(run_prefix + n, p, False) for n, p in run_items], member_list(run_prefix, run_items)),
    ]
    files = []
    for name, contents, members, listing in archives:
        write_zip(upload / name, members, when_tuple)
        files.append({"file": name, "contents": contents, "bytes": (upload / name).stat().st_size,
                      "sha256": sha256_file(upload / name), "members": listing})
        print(f"[PACKAGE] wrote {name} ({len(members)} files)", flush=True)

    package = {
        "title": TITLE,
        "version": args.version,
        "source_commit": commit,
        "source_commit_date": commit_date,
        "official_run_id": run_id,
        "builder": "reproduction/scripts/build_zenodo_package.py",
        "files": files,
        "excluded": excluded,
        "images_removed": stripped,
        "checks": {
            "worktree_vs_head": f"{len(entries)}/{len(entries)} tracked files match HEAD (LFS files materialized)",
            "features": f"{len(features)}/8 match verify_feature_files.py",
            "official_checkpoints": "70/70 match results/final_reference_manifest.json",
            "official_results": f"{results_checked}/95 seed-level result files match the official seed index",
            "credential_scan": "no credential-like strings in the code archive or the official run",
        },
    }
    (upload / "package_manifest.json").write_text(json.dumps(package, ensure_ascii=False, indent=2) + "\n",
                                                  encoding="utf-8")
    (upload / "README.md").write_text(
        record_readme(commit, commit_date, run_id, args.version, files, excluded, stripped), encoding="utf-8")
    sums = sorted(p.name for p in upload.iterdir() if p.name != "SHA256SUMS.txt")
    (upload / "SHA256SUMS.txt").write_text("".join(f"{sha256_file(upload / n)}  {n}\n" for n in sums),
                                           encoding="utf-8")
    (out / "zenodo_form.md").write_text(zenodo_form(args.version, run_id, commit), encoding="utf-8")

    print(f"[PACKAGE] commit {commit[:12]}; code archive {len(records)} files; "
          f"excluded {len(excluded)}; images removed {sum(s['images_removed'] for s in stripped)}")
    for key, value in package["checks"].items():
        print(f"[CHECK] {key}: {value}")
    print(f"[PACKAGE] upload set: {upload}")


if __name__ == "__main__":
    main()
