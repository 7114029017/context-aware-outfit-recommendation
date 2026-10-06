#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/_common.sh"
PYTHON="${PYTHON:-python3}"
POLYVORE_ROOT=""
RUN_ROOT=""

usage() {
  cat <<'EOF'
Usage:
  bash reproduction/scripts/pipeline/reproduce_ablation.sh --polyvore-root PATH --run-root PATH

Reconstructs the reproducibly reconstructed fair subset, verifies and exports the
3,432-query CIR scope, then runs 5 variants x 5 seeds using the archived handoff CP/CIR programs.
This reproduces the controlled ablation numerically; it does not claim recovery of the
missing historical memberwise fair-subset ID file.
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

BASE="$RUN_ROOT/ablation"
RECON="$BASE/fair_subset_reconstruction"
SCOPE="$BASE/fair_subset_cir_scope_audit.json"
RUNS="$BASE/runs"
REPORT="$BASE/fair_subset_batch_status.json"
SUMMARY="$BASE/summary"
mkdir -p "$BASE"

if [[ ! -f "$RECON/fair_subset_reconstruction_manifest.json" ]]; then
  "$PYTHON" "$REPRO_SCRIPTS/reconstruct_fair_subset_from_wos.py"     --polyvore-root "$POLYVORE_ROOT"     --out-dir "$RECON"
else
  echo "[SKIP] fair-subset reconstruction already exists: $RECON"
fi

"$PYTHON" "$REPRO_SCRIPTS/audit_fair_subset_cir_scope.py"   --polyvore-root "$POLYVORE_ROOT"   --subset-ids "$RECON/fair_subset_ids.txt"   --subset-manifest "$RECON/fair_subset_reconstruction_manifest.json"   --out "$SCOPE"   --out-query-ids "$BASE/fair_subset_or_query_ids.csv"   --expect-count 3432

for seed in 1 2 3 4 5; do
  for variant in original context no_weather no_occasion no_style; do
    out="$RUNS/${variant}_seed${seed}"
    "$PYTHON" "$REPRO_SCRIPTS/run_fair_subset_single_seed.py"       --polyvore-root "$POLYVORE_ROOT"       --variant "$variant"       --seed "$seed"       --out-dir "$out"       --scope-report "$SCOPE"       --subset-ids "$RECON/fair_subset_ids.txt"       --subset-manifest "$RECON/fair_subset_reconstruction_manifest.json"       --run       --acknowledge-candidate-source
  done
done

"$PYTHON" "$REPRO_SCRIPTS/pipeline/build_fair_subset_report.py"   --input-dir "$RUNS"   --out "$REPORT"

"$PYTHON" "$REPRO_SCRIPTS/summarize_fair_subset_5seed.py"   --input-dir "$RUNS"   --batch-report "$REPORT"   --out-dir "$SUMMARY"

echo "[DONE] fair-subset five-seed ablation -> $BASE"
