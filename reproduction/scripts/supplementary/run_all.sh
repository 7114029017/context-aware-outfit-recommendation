#!/usr/bin/env bash
# Run the supplementary analyses: CPU only, a few seconds, no model is loaded.
# Usage: bash reproduction/scripts/supplementary/run_all.sh [--run-root PATH | --official] [--polyvore-root PATH]
# Without --run-root or --official, the run analyses use the newest full_* folder
# under reproduction/runs/ (it must have RUN_STATUS PASSED), or the official run
# included in the repository if there is none.
set -Eeuo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${PYTHON:-python3}"
RUN_ARGS=()
DATA_ARGS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --run-root) RUN_ARGS+=("$1" "${2:?--run-root needs a path}"); shift 2 ;;
    --official) RUN_ARGS+=("$1"); shift ;;
    --polyvore-root) DATA_ARGS+=("$1" "${2:?--polyvore-root needs a path}"); shift 2 ;;
    -h|--help) sed -n '2,6p' "$0"; exit 0 ;;
    *) echo "[ERROR] unknown argument: $1" >&2; exit 2 ;;
  esac
done
"$PYTHON" "$HERE/official_run_analyses.py" ${RUN_ARGS[@]+"${RUN_ARGS[@]}"} ${DATA_ARGS[@]+"${DATA_ARGS[@]}"}
"$PYTHON" "$HERE/input_data_audit.py" ${DATA_ARGS[@]+"${DATA_ARGS[@]}"}
"$PYTHON" "$HERE/paper_value_checks.py" ${DATA_ARGS[@]+"${DATA_ARGS[@]}"}
