#!/usr/bin/env bash
# Every step after the 35 training units, for one PASSED full run: the supplementary analyses, the
# extension analyses with the checks of the ported code against the 2025 outputs, the comparison with the
# official run, the item-by-item report of the 46 reproduction items (ITEMS_STATUS.md) and the final
# summary shown at the end (FINAL_SUMMARY.txt). reproduce_all.sh --fresh calls it after the run has
# PASSED; reproduce_all.sh --analyses-only RUN (or this script) redoes the analyses of a completed run
# without retraining.
# Usage: bash reproduction/scripts/run_analyses.sh --run-root RUN [--out-dir DIR] [--figures-dir DIR]
#            [--polyvore-root PATH] [--skip-gpu]
# DIR (default RUN) receives supplementary/, extensions/, official_comparison.{txt,json},
# ITEMS_STATUS.{md,json}, FINAL_SUMMARY.txt and logs/ (one log per step, analyses_started_utc.txt and
# analyses_finished_utc.txt). The figures with Polyvore photos go to --figures-dir (default
# DIR/extensions/case_figures; it must not be a tracked folder of the repository). About 4 hours 15 minutes
# on an NVIDIA GB10; every item also needs the optional downloads (bootstrap_data.sh --with-images
# --with-fashionclip --with-nomic --with-compendium), otherwise the affected items are SKIPPED. Nothing
# here changes RUN_STATUS.txt. Exit status 1 if an item is MISSING or FAIL or the acceptance does not pass.
set -Eeuo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/_common.sh"
PYTHON="${PYTHON:-python3}"
RUN_ROOT=""
OUT_DIR=""
FIGURES_ARGS=()
POLYVORE_ARGS=()
GPU_ARGS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --run-root) RUN_ROOT="${2:?--run-root needs a path}"; shift 2 ;;
    --out-dir) OUT_DIR="${2:?--out-dir needs a path}"; shift 2 ;;
    --figures-dir) FIGURES_ARGS=(--figures-dir "${2:?--figures-dir needs a path}"); shift 2 ;;
    --polyvore-root) POLYVORE_ARGS=(--polyvore-root "${2:?--polyvore-root needs a path}"); shift 2 ;;
    --skip-gpu) GPU_ARGS=(--skip-gpu); shift ;;
    -h|--help) sed -n '2,16p' "$0"; exit 0 ;;
    *) echo "[ERROR] unknown argument: $1" >&2; exit 2 ;;
  esac
done
[[ -n "$RUN_ROOT" ]] || { echo "[ERROR] --run-root is required" >&2; exit 2; }
[[ -d "$RUN_ROOT" ]] || { echo "[ERROR] run folder not found: $RUN_ROOT" >&2; exit 2; }
RUN_ROOT="$(cd "$RUN_ROOT" && pwd)"
[[ "$(cat "$RUN_ROOT/RUN_STATUS.txt" 2>/dev/null)" == "PASSED" ]] ||
  { echo "[ANALYSES BLOCKED] $RUN_ROOT: RUN_STATUS is not PASSED" >&2; exit 2; }
OUT_DIR="${OUT_DIR:-$RUN_ROOT}"
mkdir -p "$OUT_DIR/logs"
OUT_DIR="$(cd "$OUT_DIR" && pwd)"
date -u +%Y-%m-%dT%H:%M:%SZ > "$OUT_DIR/logs/analyses_started_utc.txt"
if [[ ${#FIGURES_ARGS[@]} -gt 0 && "${FIGURES_ARGS[1]}" != /* ]]; then
  FIGURES_ARGS=(--figures-dir "$PWD/${FIGURES_ARGS[1]}")
fi
if [[ ${#POLYVORE_ARGS[@]} -gt 0 && "${POLYVORE_ARGS[1]}" != /* ]]; then
  POLYVORE_ARGS=(--polyvore-root "$PWD/${POLYVORE_ARGS[1]}")
fi

step() {  # name, log file, command...
  local name="$1" log="$2"
  shift 2
  echo "[ANALYSES] $name (log: $log)"
  if ! "$@" > "$log" 2>&1; then
    echo "[WARN] $name failed or reported a problem; see $log" >&2
  fi
}
step "supplementary analyses" "$OUT_DIR/logs/supplementary.log" \
  bash "$HERE/supplementary/run_all.sh" --run-root "$RUN_ROOT" --out-dir "$OUT_DIR/supplementary" \
  ${POLYVORE_ARGS[@]+"${POLYVORE_ARGS[@]}"}
echo "[ANALYSES] extension analyses and checks of the ported code (step logs in $OUT_DIR/extensions/logs/)"
if ! bash "$HERE/extensions/run_all.sh" --run-root "$RUN_ROOT" --out-dir "$OUT_DIR/extensions" --validate \
     ${FIGURES_ARGS[@]+"${FIGURES_ARGS[@]}"} ${POLYVORE_ARGS[@]+"${POLYVORE_ARGS[@]}"} \
     ${GPU_ARGS[@]+"${GPU_ARGS[@]}"} > "$OUT_DIR/logs/extensions.log" 2>&1; then
  echo "[WARN] an extension step FAILED; see $OUT_DIR/extensions/EXTENSIONS_STATUS.txt" >&2
fi
cat "$OUT_DIR/extensions/EXTENSIONS_STATUS.txt" 2>/dev/null | sed 's/^/[ANALYSES]   /' || true
step "comparison with the official run" "$OUT_DIR/logs/comparison.log" \
  "$PYTHON" "$HERE/compare_with_official_run.py" --run-root "$RUN_ROOT" --out-dir "$OUT_DIR"
grep -m1 "^判定" "$OUT_DIR/official_comparison.txt" 2>/dev/null | sed 's/^/[ANALYSES]   /' || true
echo "[ANALYSES] item-by-item report (log: $OUT_DIR/logs/items.log) and final summary"
set +e
"$PYTHON" "$HERE/report_items.py" --run-root "$RUN_ROOT" --supplementary-dir "$OUT_DIR/supplementary" \
  --extensions-dir "$OUT_DIR/extensions" --out-dir "$OUT_DIR" > "$OUT_DIR/logs/items.log" 2>&1
items_code=$?
date -u +%Y-%m-%dT%H:%M:%SZ > "$OUT_DIR/logs/analyses_finished_utc.txt"
echo
"$PYTHON" "$HERE/show_final_summary.py" --run-root "$RUN_ROOT" --supplementary-dir "$OUT_DIR/supplementary" \
  --extensions-dir "$OUT_DIR/extensions" --out-dir "$OUT_DIR" 2>&1 | tee "$OUT_DIR/logs/final_summary.log"
summary_code="${PIPESTATUS[0]}"
set -e
[[ "$items_code" -eq 0 && "$summary_code" -eq 0 ]] || exit 1
