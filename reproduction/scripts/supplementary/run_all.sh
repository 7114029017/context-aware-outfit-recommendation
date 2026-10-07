#!/usr/bin/env bash
# Run the supplementary analyses: CPU only, a few seconds, no model is loaded.
# Usage: bash reproduction/scripts/supplementary/run_all.sh [--run-root PATH | --official]
#                                                           [--polyvore-root PATH] [--out-dir DIR]
# Without --run-root or --official, the newest full_* folder under reproduction/runs/
# is analyzed (its RUN_STATUS must be PASSED), or the official run included in the
# repository if there is none.
# Output: run_analyses/, input_data_audit/ and paper_value_checks/ under --out-dir,
# or under <run folder>/supplementary/ for a run folder. For the official run the
# default is reproduction/results/supplementary/, with the run analyses in
# full_20261002T161428Z/. A full run (reproduce_all.sh --fresh) calls this script
# after its RUN_STATUS has become PASSED.
set -Eeuo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUNS="$(cd "$HERE/../.." && pwd)/runs"
PYTHON="${PYTHON:-python3}"
RUN_ROOT=""
OFFICIAL=0
OUT_DIR=""
DATA_ARGS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --run-root) RUN_ROOT="${2:?--run-root needs a path}"; shift 2 ;;
    --official) OFFICIAL=1; shift ;;
    --polyvore-root) DATA_ARGS+=("$1" "${2:?--polyvore-root needs a path}"); shift 2 ;;
    --out-dir) OUT_DIR="${2:?--out-dir needs a path}"; shift 2 ;;
    -h|--help) sed -n '2,12p' "$0"; exit 0 ;;
    *) echo "[ERROR] unknown argument: $1" >&2; exit 2 ;;
  esac
done
if [[ -n "$RUN_ROOT" && "$OFFICIAL" -eq 1 ]]; then
  echo "[ERROR] pass either --run-root or --official, not both" >&2
  exit 2
fi
if [[ -z "$RUN_ROOT" && "$OFFICIAL" -eq 0 ]]; then
  # Newest run folder, found as in README 0.8.
  RUN_ROOT="$(find "$RUNS" -maxdepth 1 -type d -name 'full_*' 2>/dev/null | LC_ALL=C sort | tail -n 1 || true)"
  [[ -n "$RUN_ROOT" ]] || OFFICIAL=1
fi
if [[ "$OFFICIAL" -eq 1 ]]; then
  RUN_ARGS=(--official)
else
  RUN_ARGS=(--run-root "$RUN_ROOT")
  OUT_DIR="${OUT_DIR:-$RUN_ROOT/supplementary}"
fi
RUN_OUT=()
AUDIT_OUT=()
CHECK_OUT=()
if [[ -n "$OUT_DIR" ]]; then
  RUN_OUT=(--out-dir "$OUT_DIR/run_analyses")
  AUDIT_OUT=(--out-dir "$OUT_DIR/input_data_audit")
  CHECK_OUT=(--out-dir "$OUT_DIR/paper_value_checks")
fi
# The run analyses go first: they stop this script if the run has not PASSED.
"$PYTHON" "$HERE/official_run_analyses.py" "${RUN_ARGS[@]}" ${RUN_OUT[@]+"${RUN_OUT[@]}"} \
  ${DATA_ARGS[@]+"${DATA_ARGS[@]}"}
"$PYTHON" "$HERE/input_data_audit.py" ${AUDIT_OUT[@]+"${AUDIT_OUT[@]}"} ${DATA_ARGS[@]+"${DATA_ARGS[@]}"}
"$PYTHON" "$HERE/paper_value_checks.py" ${CHECK_OUT[@]+"${CHECK_OUT[@]}"} ${DATA_ARGS[@]+"${DATA_ARGS[@]}"}
