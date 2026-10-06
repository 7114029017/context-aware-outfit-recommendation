#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/_common.sh"
PYTHON="${PYTHON:-python3}"
POLYVORE_ROOT=""
RUN_ROOT=""

usage() {
  cat <<'EOF'
Usage:
  bash reproduction/scripts/pipeline/reproduce_analysis.sh --polyvore-root PATH --run-root PATH

Recomputes preserved secondary thesis analyses: length, target-clue audit,
Judge preserved-output analyses, 30-case human audit, subgroup/case numeric
evidence, Two-Tower validation, and the preserved Table 4-3 proxy distributions.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --polyvore-root) POLYVORE_ROOT="$2"; shift 2 ;;
    --run-root) RUN_ROOT="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "[ERROR] unknown argument: $1" >&2; usage; exit 2 ;;
  esac
done

[[ -n "$POLYVORE_ROOT" && -n "$RUN_ROOT" ]] || { usage; exit 2; }
OUT="$RUN_ROOT/secondary"
mkdir -p "$OUT"

"$PYTHON" "$REPRO_SCRIPTS/preflight_remaining_thesis_reproduction.py" \
  --polyvore-root "$POLYVORE_ROOT" \
  --run-root "$RUN_ROOT" \
  --out-dir "$OUT/preflight"

# Table 4-3: recompute distributions from the original archived proxy outputs.
# This is descriptive analysis only; it does not regenerate CLO/MET/Tsub or
# silently repair a missing standalone source file.
"$PYTHON" "$REPRO_SCRIPTS/pipeline/reproduce_table_4_3.py" \
  --out-dir "$OUT/results/environment_proxy"

"$PYTHON" "$REPRO_SCRIPTS/reproduce_remaining_thesis.py" \
  --polyvore-root "$POLYVORE_ROOT" \
  --out-dir "$OUT/results" \
  --fair-subset-dir "$RUN_ROOT/ablation/runs"

# The old analysis runner may catch individual module errors and still exit 0.
# Treat any failed or missing module as a failed full run; status=partial is
# allowed for documented missing historical source/provenance.
"$PYTHON" "$REPRO_SCRIPTS/pipeline/validate_secondary_matrix.py" \
  --matrix "$OUT/results/remaining_reproduction_matrix.json"

echo "[DONE] secondary analyses -> $OUT"