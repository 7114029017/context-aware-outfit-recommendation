#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"
PYTHON="${PYTHON:-python3}"
MODE="check"
POLYVORE_ROOT=""
RUN_ROOT=""

usage() {
  cat <<'EOF'
Usage:
  bash reproduction/scripts/reproduce_thesis.sh --mode check|smoke|main|full [--polyvore-root PATH] [--run-root PATH]

Modes:
  check  Validate environment, data availability, features, and frozen split identity.
  smoke  Run check + one short Context CP->CIR smoke training.
  main   Run check + full Original/Context five-seed main experiment and comparison.
  full   Run check + main + five-variant fair-subset ablation + preserved secondary analyses.

If --polyvore-root is omitted, the script first uses reproduction/.local/polyvore_root.txt, then falls back to the legacy .repro_local/polyvore_root.txt.
If --run-root is omitted, a UTC-tagged directory under reproduction/runs/ is created
with the mode as its prefix: check_<UTC>, smoke_<UTC>, main_<UTC>, or full_<UTC>.

The training wrappers call the archived handoff programs under:
  02_模型訓練和驗證_model_training_validation/main_hybrid_attention_code/

They do not modify that source tree in place.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --mode) MODE="$2"; shift 2 ;;
    --polyvore-root) POLYVORE_ROOT="$2"; shift 2 ;;
    --run-root) RUN_ROOT="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "[ERROR] unknown argument: $1" >&2; usage; exit 2 ;;
  esac
done

case "$MODE" in check|smoke|main|full) ;; *) usage; exit 2 ;; esac

# Fail during check/smoke (before expensive training) if a downstream bash
# launcher or stage runner cannot even be parsed. No models run here.
for shell_script in \
  "$REPRO_SCRIPTS/pipeline/launch_full.sh" \
  "$REPRO_SCRIPTS/pipeline/reproduce_main.sh" \
  "$REPRO_SCRIPTS/pipeline/reproduce_ablation.sh" \
  "$REPRO_SCRIPTS/pipeline/reproduce_analysis.sh"; do
  if ! bash -n "$shell_script"; then
    echo "[ERROR] shell syntax check failed: $shell_script" >&2
    exit 2
  fi
done
echo "[OK] full-run launcher and stage scripts parse correctly"

if [[ -z "$POLYVORE_ROOT" ]]; then
  if [[ -f "$REPRO_ROOT/.local/polyvore_root.txt" ]]; then
    POLYVORE_ROOT="$(cat "$REPRO_ROOT/.local/polyvore_root.txt")"
  elif [[ -f "$ROOT/.repro_local/polyvore_root.txt" ]]; then
    POLYVORE_ROOT="$(cat "$ROOT/.repro_local/polyvore_root.txt")"
  else
    echo "[ERROR] no --polyvore-root and .repro_local/polyvore_root.txt is absent" >&2
    echo "Run: bash reproduction/scripts/bootstrap_data.sh" >&2
    exit 3
  fi
fi
POLYVORE_ROOT="$(cd "$POLYVORE_ROOT" && pwd)"

if [[ -z "$RUN_ROOT" ]]; then
  tag="$(date -u +%Y%m%dT%H%M%SZ)"
  RUN_ROOT="$REPRO_ROOT/runs/${MODE}_$tag"
elif [[ "$RUN_ROOT" != /* ]]; then
  RUN_ROOT="$REPRO_ROOT/$RUN_ROOT"
fi
# Canonicalize before writing, to avoid symlinks escaping the protected
# archived-research tree. Inside this repository permit only dedicated
# Git-ignored run locations, never original model/data/code directories.
RUN_ROOT="$(realpath -m "$RUN_ROOT")"
case "$RUN_ROOT" in
  "$REPRO_ROOT/runs/"*) ;;
  "$ROOT" | "$ROOT/"*)
    echo "[ERROR] in-repository --run-root must be under reproduction/runs/<run-id>/" >&2
    exit 3 ;;
  / | /tmp | /home | /mnt | /media)
    echo "[ERROR] provide a dedicated external run subdirectory, not a filesystem parent" >&2
    exit 3 ;;
esac
mkdir -p "$RUN_ROOT/environment"

echo "[REPRO] mode=$MODE"
echo "[REPRO] repo=$ROOT"
echo "[REPRO] polyvore=$POLYVORE_ROOT"
echo "[REPRO] run_root=$RUN_ROOT"
git -C "$ROOT" rev-parse HEAD | tee "$RUN_ROOT/git_commit.txt"
git -C "$ROOT" status --short --branch | tee "$RUN_ROOT/git_status_start.txt"
# Repository URL with any embedded credentials removed.
{ git -C "$ROOT" remote get-url origin 2>/dev/null || echo "not_available"; } |
  sed -E 's#://[^/@]*@#://#' | tee "$RUN_ROOT/git_remote.txt"

"$PYTHON" "$REPRO_SCRIPTS/validate_reproduction_env.py"   --out "$RUN_ROOT/environment/environment_validation.json"

"$PYTHON" "$REPRO_SCRIPTS/verify_feature_files.py" \
  --out "$RUN_ROOT/environment/feature_verification.json" \
  | tee "$RUN_ROOT/environment/feature_verification_stdout.txt"

"$PYTHON" "$REPRO_SCRIPTS/freeze_splits.py"   --polyvore-root "$POLYVORE_ROOT"   --out-dir "$RUN_ROOT/frozen_splits"

# Counts alone do not establish dataset identity. Fail BEFORE smoke/main/full
# training if any regenerated ordered split/evaluation manifest differs from
# the preserved reference used for the completed reproduction.
for name in train_ids.csv validation_ids.csv test_ids.csv cp_ids.csv fitb_ids.csv or_ids.csv; do
  expected="$REPRO_ROOT/splits/$name"
  actual="$RUN_ROOT/frozen_splits/$name"
  if [[ ! -f "$expected" ]] || ! cmp -s "$expected" "$actual"; then
    echo "[ERROR] frozen split identity mismatch: $name" >&2
    echo "  preserved=$expected" >&2
    echo "  regenerated=$actual" >&2
    echo "Stop: do not start training with mismatched input/evaluation IDs." >&2
    exit 4
  fi
  echo "[SPLIT MATCH] $name"
done

if [[ "$MODE" == "check" ]]; then
  echo "[DONE] preflight only -> $RUN_ROOT"
  exit 0
fi

if [[ "$MODE" == "smoke" ]]; then
  "$PYTHON" "$REPRO_SCRIPTS/run_single_seed_smoke_training.py"     --polyvore-root "$POLYVORE_ROOT"     --variant context     --seed 1     --train-outfits 200     --valid-outfits 100     --epochs 1     --num-workers 0     --out-dir "$RUN_ROOT/smoke_context_seed1"
  echo "[DONE] smoke reproduction -> $RUN_ROOT"
  exit 0
fi

bash "$REPRO_SCRIPTS/pipeline/reproduce_main.sh"   --polyvore-root "$POLYVORE_ROOT"   --run-root "$RUN_ROOT"

if [[ "$MODE" == "main" ]]; then
  echo "[DONE] main reproduction -> $RUN_ROOT"
  exit 0
fi

bash "$REPRO_SCRIPTS/pipeline/reproduce_ablation.sh"   --polyvore-root "$POLYVORE_ROOT"   --run-root "$RUN_ROOT"

bash "$REPRO_SCRIPTS/pipeline/reproduce_analysis.sh" \
  --polyvore-root "$POLYVORE_ROOT" \
  --run-root "$RUN_ROOT"

# Generate the explicitly scoped paper-table evidence before marking the
# integrated full run complete. This includes twenty overview entries, with
# provenance-only gaps stated plainly, rather than claiming all twenty tables
# were independently regenerated or matched cell by cell.
"$PYTHON" "$REPRO_SCRIPTS/pipeline/build_chapter4_report.py" \
  --source full \
  --run-root "$RUN_ROOT"

# Do not publish a full-run completion summary merely because four report
# files were written. Validate their cross-file consistency, explicit
# source labels, numerical arithmetic and input evidence hashes first.
"$PYTHON" "$REPRO_SCRIPTS/pipeline/validate_chapter4_report.py" \
  --report-dir "$RUN_ROOT/chapter4" \
  --source full

"$PYTHON" "$REPRO_SCRIPTS/pipeline/build_run_summary.py" \
  --run-root "$RUN_ROOT"

# This is post-training-only analysis. It reads completed seed-level CSVs and
# does not load checkpoints or launch any additional model training.
"$PYTHON" "$REPRO_SCRIPTS/pipeline/complete_statistics.py" \
  --run-root "$RUN_ROOT" \
  --out-dir "$RUN_ROOT/statistics"

# Human-readable Table 4-1..4-20 provenance index. This reads completed
# outputs only; it does not launch training, LLM inference, or human annotation.
"$PYTHON" "$REPRO_SCRIPTS/build_thesis_tables_summary.py" \
  --run-root "$RUN_ROOT" \
  --out "$RUN_ROOT/chapter4/THESIS_TABLES_SUMMARY.md"

cat > "$RUN_ROOT/README.txt" <<EOF
Full reproduction completed.

Main summary:
  $RUN_ROOT/main/summary/main_reproduction_summary.csv
  $RUN_ROOT/main/summary/main_reproduction_acceptance.json

Ablation summary:
  $RUN_ROOT/ablation/summary/

Post-training paired BH and fair-subset factor direction report:
  $RUN_ROOT/statistics/statistical_evidence.md
  $RUN_ROOT/statistics/main_paired_bh_8metrics.csv

Secondary analyses:
  $RUN_ROOT/secondary/results/remaining_reproduction_matrix.md

Chapter 4 evidence-by-table comparison:
  $RUN_ROOT/chapter4/chapter4_report.md
  $RUN_ROOT/chapter4/chapter4_numeric_comparison.csv
  $RUN_ROOT/chapter4/chapter4_report.json
  $RUN_ROOT/chapter4/THESIS_TABLES_SUMMARY.md

Table-by-table provenance report:
  - labels which tables come from fresh retraining;
  - labels which tables are recomputed from senior-student preserved outputs;
  - states the source material, current evidence path, and unresolved limitation for Tables 4-1..4-20.

Interpretation:
  - fresh main retraining is compared to thesis means with the predeclared retraining tolerance;
  - fair-subset ablation uses the reproducibly reconstructed fair subset;
  - preserved secondary analyses retain documented historical provenance limitations.
EOF

echo "[DONE] full thesis reproduction -> $RUN_ROOT"