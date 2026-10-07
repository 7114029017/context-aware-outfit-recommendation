#!/usr/bin/env bash
# One official run -> one directory; original archived programs stay read-only.
set -Eeuo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/_common.sh"
OUTPUT_BASE="$REPRO_ROOT/runs"
RUN_ID=""
POLYVORE_ROOT=""

usage() {
  cat <<'HELP'
Usage:
  bash reproduction/scripts/reproduce_all.sh [--run-id NAME] [--output-base PATH] [--polyvore-root PATH]

This starts the entire 10-main + 25 fair-subset ablation training pipeline.
No --check-only mode: use bash reproduction/scripts/check.sh to preflight.
After the run has PASSED, the supplementary analyses (CPU only, outside the
35 units) are written to <run-id>/supplementary/, and the extension analyses
(text length, counterfactual, Two-Tower, color, case figures; about 2-3.5 GPU
hours) to <run-id>/extensions/. Neither ever changes RUN_STATUS.

All outputs for this new execution are stored within a single run directory:
  <output-base>/<run-id>/
The directory must NOT exist before starting; prior runs are never overwritten.
The default output base is reproduction/runs/ (Git-ignored).
For a large external disk, supply --output-base /absolute/disk/path.
HELP
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --run-id) RUN_ID="$2"; shift 2 ;;
    --output-base) OUTPUT_BASE="$2"; shift 2 ;;
    --polyvore-root) POLYVORE_ROOT="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "[ERROR] unknown argument: $1" >&2; usage; exit 2 ;;
  esac
done

if [[ -z "$RUN_ID" ]]; then
  RUN_ID="full_$(date -u +%Y%m%dT%H%M%SZ)"
fi
if [[ ! "$RUN_ID" =~ ^[a-zA-Z0-9][a-zA-Z0-9_-]*$ ]]; then
  echo "[ERROR] --run-id must contain only ASCII letters, numbers, underscores, hyphens" >&2
  exit 2
fi

if [[ "$OUTPUT_BASE" != /* ]]; then
  OUTPUT_BASE="$ROOT/$OUTPUT_BASE"
fi
OUTPUT_BASE="$(realpath -m "$OUTPUT_BASE")"
case "$OUTPUT_BASE" in
  "$ROOT" | "$ROOT/" | "$ROOT/01_資料建構_data_construction"* | \
  "$ROOT/02_模型訓練和驗證_model_training_validation"* | \
  "$ROOT/03_實驗與結果_experiments_results"* | "$ROOT/scripts"* | \
  "$ROOT/docs"* | "$ROOT/splits"* | "$ROOT/configs"* )
    echo "[ERROR] unsafe output base inside archived research/reproduction source: $OUTPUT_BASE" >&2
    exit 2 ;;
  "$ROOT/"*)
    if [[ "$OUTPUT_BASE" != "$REPRO_ROOT/runs" && "$OUTPUT_BASE" != "$REPRO_ROOT/runs/"* ]]; then
      echo "[ERROR] choose reproduction/runs or an external absolute output base" >&2
      exit 2
    fi ;;
  / | /tmp | /home | /mnt | /media)
    echo "[ERROR] choose a dedicated experiment output base, not a filesystem parent" >&2
    exit 2 ;;
esac

# Official source hygiene: start a costly run only from the main branch or from
# a release tag named v*-tors-reproduction, and only with unmodified original
# research programs. Folders 01-03 are checked by content against
# reproduction/environment/archived_sources_manifest.json.
BRANCH="$(git -C "$ROOT" branch --show-current 2>/dev/null || true)"
RELEASE_TAG="$(git -C "$ROOT" describe --tags --exact-match --match 'v*-tors-reproduction' HEAD 2>/dev/null || true)"
if [[ "$BRANCH" != "main" && -z "$RELEASE_TAG" ]]; then
  echo "[ERROR] run from a git clone of the main branch or of a v*-tors-reproduction release tag" >&2
  exit 3
fi
if ! "${PYTHON:-python3}" "$REPRO_SCRIPTS/verify_archived_sources.py"; then
  echo "[ERROR] original thesis research source differs from the frozen baseline" >&2
  echo "[ERROR] inspect your archived-source changes; do NOT begin training" >&2
  exit 3
fi

RUN_ROOT="$OUTPUT_BASE/$RUN_ID"
if [[ -e "$RUN_ROOT" || -L "$RUN_ROOT" ]]; then
  echo "[ERROR] refusing to overwrite existing run directory: $RUN_ROOT" >&2
  exit 3
fi
mkdir -p "$RUN_ROOT/logs"
LOG="$RUN_ROOT/logs/full_console.log"
date -u +%Y-%m-%dT%H:%M:%SZ > "$RUN_ROOT/started_utc.txt"
printf '%s\n' "RUNNING" > "$RUN_ROOT/RUN_STATUS.txt"

cat > "$RUN_ROOT/INDEX.md" <<'INDEX'
# Full thesis reproduction — this run only

All paths below are relative to THIS directory. Do not mix outputs from a
different run ID. The initial status is RUNNING; check RUN_STATUS.txt.

| Directory or file | Contents |
|---|---|
| RUN_STATUS.txt | RUNNING / PASSED / FAILED; check before using final results |
| started_utc.txt, finished_utc.txt | UTC execution timestamps |
| logs/full_console.log | Complete command-line output, including failures |
| logs/final_terminal_summary.txt | Final terminal summary; present only after PASSED |
| logs/supplementary.log | Output of the supplementary analyses; present only after PASSED |
| logs/extensions.log | Output of the extension analyses; present only after PASSED |
| environment/ | Python, CUDA, GPU and eight feature checks |
| frozen_splits/ | Six ordered training/evaluation scope identity manifests |
| git_commit.txt, git_status_start.txt, git_remote.txt | Source revision, branch and starting working tree, repository URL |
| main/original_seed1..5/, main/context_seed1..5/ | Ten main CP/CIR training checkpoints, manifests and evaluation CSVs |
| main/collected/, main/summary/ | Five-seed main collection and paper comparison |
| ablation/fair_subset_reconstruction/, ablation/fair_subset_cir_scope_audit.json | Reconstructed fair subset and 3,432-query audit |
| ablation/fair_subset_or_query_ids.csv | The 3,432 evaluable OR queries |
| ablation/runs/<variant>_seed<seed>/ | Twenty-five fair-subset CP/CIR ablation runs |
| ablation/summary/ | Five-seed ablation tables (thesis Tables 4-13 to 4-16) and preserved T01 comparison |
| statistics/ | Paired tests, BH, CI, Cohen's dz; thesis Tables 4-11 and 4-12 |
| secondary/preflight/, secondary/results/ | Preserved-output secondary analyses and historical provenance limits |
| chapter4/ | Twenty-table overview, 112 scoped numeric rows, JSON evidence |
| reproduction_summary.md, reproduction_summary.json | Final cross-module summary; present only after full completion |
| README.txt | Original entrypoint's final output pointers; present only after completion |
| supplementary/run_analyses/, input_data_audit/, paper_value_checks/, judge_audit_checks/, met_reference_check/, checklist_coverage/ | Supplementary analyses, run after PASSED (CPU only, outside the 35 units; they never change RUN_STATUS) |
| extensions/ | Extension analyses, run after PASSED (outside the 35 units; they never change RUN_STATUS); step status in extensions/EXTENSIONS_STATUS.txt |

The 35 training units run in series. Some historical sources are incomplete,
so a finished run must NOT claim exact reproduction of all published tables.
Full training is distinct from preserved-output Judge/human reanalysis.
INDEX

echo "[RUN ROOT] $RUN_ROOT"
echo "[LIVE LOG] $LOG"
echo "[CAUTION] launching 10 full main + 25 full ablation training units"

cmd=(bash "$REPRO_SCRIPTS/reproduce_thesis.sh" --mode full --run-root "$RUN_ROOT")
if [[ -n "$POLYVORE_ROOT" ]]; then
  cmd+=(--polyvore-root "$POLYVORE_ROOT")
fi

# PIPESTATUS[0] is the full entrypoint exit code; PIPESTATUS[1] is tee.
set +e
"${cmd[@]}" 2>&1 | tee "$LOG"
codes=("${PIPESTATUS[@]}")
set -e

date -u +%Y-%m-%dT%H:%M:%SZ > "$RUN_ROOT/finished_utc.txt"
if [[ "${codes[0]}" -eq 0 && "${codes[1]}" -eq 0 ]] &&
   [[ -f "$RUN_ROOT/reproduction_summary.json" ]] &&
   [[ -f "$RUN_ROOT/chapter4/chapter4_report.json" ]]; then
  printf '%s\n' "PASSED" > "$RUN_ROOT/RUN_STATUS.txt"
  echo "[PASSED] single-run reproduction complete: $RUN_ROOT"
  echo "[READ] $RUN_ROOT/INDEX.md"
  echo
  # Keep the final terminal summary with the run, not only on screen.
  if ! python3 "$REPRO_SCRIPTS/show_reproduction_summary.py" --run-root "$RUN_ROOT" |
       tee "$RUN_ROOT/logs/final_terminal_summary.txt"; then
    echo "[WARN] scientific run PASSED, but the final terminal summary could not be rendered." >&2
    echo "[WARN] inspect the saved CSV/JSON outputs under: $RUN_ROOT" >&2
  fi
  # Supplementary analyses: CPU only, under a minute. They read this run's saved
  # outputs and the fixed input data, and never change RUN_STATUS.txt.
  supplementary=(bash "$REPRO_SCRIPTS/supplementary/run_all.sh" --run-root "$RUN_ROOT"
                 --out-dir "$RUN_ROOT/supplementary")
  if [[ -n "$POLYVORE_ROOT" ]]; then
    supplementary+=(--polyvore-root "$POLYVORE_ROOT")
  fi
  echo
  if "${supplementary[@]}" > "$RUN_ROOT/logs/supplementary.log" 2>&1; then
    echo "[SUPPLEMENTARY] $RUN_ROOT/supplementary/ (log: logs/supplementary.log)"
  else
    echo "[WARN] scientific run PASSED, but the supplementary analyses failed; see $RUN_ROOT/logs/supplementary.log" >&2
    echo "[WARN] after fixing the cause, rerun: ${supplementary[*]}" >&2
  fi
  # Extension analyses (reproduction/docs/extensions.md): text length and Two-Tower on the GPU;
  # counterfactual, color analysis and case figures on the CPU. A step whose download is missing
  # is skipped. They read this run's checkpoints and outputs and never change RUN_STATUS.txt.
  extensions=(bash "$REPRO_SCRIPTS/extensions/run_all.sh" --run-root "$RUN_ROOT")
  if [[ -n "$POLYVORE_ROOT" ]]; then
    extensions+=(--polyvore-root "$POLYVORE_ROOT")
  fi
  echo "[EXTENSIONS] running the extension analyses (about 2-3.5 GPU hours); step logs in $RUN_ROOT/extensions/logs/"
  if "${extensions[@]}" 2>&1 | tee "$RUN_ROOT/logs/extensions.log"; then
    echo "[EXTENSIONS] $RUN_ROOT/extensions/ (status: extensions/EXTENSIONS_STATUS.txt)"
  else
    echo "[WARN] scientific run PASSED, but an extension analysis failed; see $RUN_ROOT/extensions/EXTENSIONS_STATUS.txt" >&2
    echo "[WARN] after fixing the cause, rerun: ${extensions[*]} --steps <failed steps>" >&2
  fi
else
  printf '%s\n' "FAILED entrypoint_exit=${codes[0]} tee_exit=${codes[1]}" > "$RUN_ROOT/RUN_STATUS.txt"
  echo "[FAILED] full run did not complete. Preserve this directory and inspect $LOG" >&2
  echo "[FAILED] no source-exact or complete scientific reproduction claim is allowed" >&2
  exit 1
fi
