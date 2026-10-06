#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/_common.sh"
PYTHON="${PYTHON:-python3}"
POLYVORE_ROOT=""
RUN_ROOT=""
SEEDS="1 2 3 4 5"

usage() {
  cat <<'EOF'
Usage:
  bash reproduction/scripts/pipeline/reproduce_main.sh --polyvore-root PATH --run-root PATH [--seeds "1 2 3 4 5"]

Runs the thesis main Original vs Context experiment by invoking the archived
handoff train_cp.py / train_cir.py / evaluate_cp.py / evaluate_cir.py through
the validated wrapper layer.  The archived research source is never modified in place.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --polyvore-root) POLYVORE_ROOT="$2"; shift 2 ;;
    --run-root) RUN_ROOT="$2"; shift 2 ;;
    --seeds) SEEDS="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "[ERROR] unknown argument: $1" >&2; usage; exit 2 ;;
  esac
done

[[ -n "$POLYVORE_ROOT" && -n "$RUN_ROOT" ]] || { usage; exit 2; }
mkdir -p "$RUN_ROOT/main"

for seed in $SEEDS; do
  for variant in original context; do
    out="$RUN_ROOT/main/${variant}_seed${seed}"
    if [[ -f "$out/full_train_manifest.json" ]] &&
       "$PYTHON" "$REPRO_SCRIPTS/pipeline/verify_main_training_reuse.py" \
         --run-dir "$out" --variant "$variant" --seed "$seed"; then
      echo "[SKIP VERIFIED TRAIN] $variant seed$seed; full-data identity and both checkpoint SHA-256 verified"
    else
      if [[ -e "$out" ]]; then
        echo "[ERROR] incomplete existing run directory: $out" >&2
        echo "Use a new --run-root or inspect/remove only that incomplete run directory." >&2
        exit 3
      fi
      "$PYTHON" "$REPRO_SCRIPTS/run_full_single_seed_training.py"         --polyvore-root "$POLYVORE_ROOT"         --variant "$variant"         --seed "$seed"         --out-dir "$out"
    fi

    "$PYTHON" "$REPRO_SCRIPTS/pipeline/evaluate_main_run.py"       --polyvore-root "$POLYVORE_ROOT"       --run-dir "$out"       --out-dir "$out/evaluation"
  done
done

"$PYTHON" "$REPRO_SCRIPTS/pipeline/collect_main_results.py"   --input-dir "$RUN_ROOT/main"   --out-dir "$RUN_ROOT/main/collected"

"$PYTHON" "$REPRO_SCRIPTS/compare_main_results.py"   --cp "$RUN_ROOT/main/collected/results_cp.csv"   --cir "$RUN_ROOT/main/collected/results_cir.csv"   --out-dir "$RUN_ROOT/main/summary"   --mode retrain

echo "[DONE] main five-seed reproduction -> $RUN_ROOT/main"
