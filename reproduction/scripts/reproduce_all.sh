#!/usr/bin/env bash
set -Eeuo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
DEFAULT_COMPLETED_RUN="$ROOT/_reproduction_runs/full_20260921T175217Z"
REUSE_RUN=""
FORCE_FRESH=0
PASSTHRU=()

usage() {
  cat <<'HELP'
Usage:
  bash reproduction/scripts/reproduce_all.sh [--fresh] [--reuse-reference PATH]
                                              [launch_full options...]

Default behavior:
  - If the preserved local completed reference run exists at
      _reproduction_runs/full_20260921T175217Z/
    it is strictly verified against the committed 35-row seed index,
    all 70 checkpoint SHA-256 identities, and results/SHA256SUMS.txt.
    When verification passes, the expensive 35-unit training is skipped.
  - If that local completed run is absent (for example on a clean clone),
    a new full 10-main + 25 fair-subset ablation run is launched.

Options:
  --fresh
      Force a brand-new full training run even if the completed local
      reference run is available.

  --reuse-reference PATH
      Verify and reuse a specific completed local full-run directory.
      Training is skipped only if every verification gate passes.

Other options are forwarded to pipeline/launch_full.sh:
  --run-id NAME
  --output-base PATH
  --polyvore-root PATH

Reuse mode does not claim a second training execution. It verifies the
completed training/checkpoint evidence and reports the already committed
reference raw, summary, Chapter 4, and statistical result locations.
HELP
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --fresh)
      FORCE_FRESH=1
      shift
      ;;
    --reuse-reference)
      [[ $# -ge 2 ]] || { echo "[ERROR] --reuse-reference requires PATH" >&2; exit 2; }
      REUSE_RUN="$2"
      shift 2
      ;;
    -h|--help)
      usage
      echo
      bash "$HERE/pipeline/launch_full.sh" --help
      exit 0
      ;;
    *)
      PASSTHRU+=("$1")
      shift
      ;;
  esac
done

if [[ "$FORCE_FRESH" -eq 1 && -n "$REUSE_RUN" ]]; then
  echo "[ERROR] choose --fresh OR --reuse-reference, not both" >&2
  exit 2
fi

verify_and_reuse() {
  local run_dir="$1"
  echo "[REUSE CHECK] completed full run: $run_dir"
  if ! python3 "$HERE/verify_completed_reference.py" --full-run "$run_dir"; then
    echo "[REUSE BLOCKED] completed-run verification failed: $run_dir" >&2
    return 1
  fi

  echo
  echo "[SKIP VERIFIED FULL TRAINING] 35 completed training units are being reused."
  echo "[SKIP VERIFIED FULL TRAINING] No CP/CIR model training was launched."
  echo "[REFERENCE RAW] $ROOT/reproduction/results/raw/reference_20260921T175217Z/"
  echo "[REFERENCE SUMMARY] $ROOT/reproduction/results/summary/reference_20260921T175217Z/"
  echo "[REFERENCE STATISTICS] $ROOT/reproduction/results/summary/reference_20260921T175217Z/statistics/"
  echo "[REFERENCE MANIFEST] $ROOT/reproduction/results/reference_20260921T175217Z_manifest.json"
  echo

  local table_report="$ROOT/reproduction/runs/reference_20260921T175217Z/THESIS_TABLES_SUMMARY.md"
  if ! python3 "$HERE/build_thesis_tables_summary.py" --reference --out "$table_report"; then
    echo "[REUSE BLOCKED] verified evidence exists, but the table-by-table provenance report could not be generated." >&2
    return 1
  fi
  echo "[TABLE-BY-TABLE REPORT] $table_report"
  echo
  echo "[NOTE] Reuse mode verifies preserved training/checkpoints and committed result bytes."
  echo "[NOTE] It does not claim a second 35-unit execution. Use --fresh to retrain from scratch."
  echo
  if ! python3 "$HERE/show_reproduction_summary.py" --reference; then
    echo "[REUSE BLOCKED] verified evidence exists, but final result summary could not be rendered." >&2
    return 1
  fi
}

if [[ "$FORCE_FRESH" -eq 0 ]]; then
  if [[ -n "$REUSE_RUN" ]]; then
    REUSE_RUN="$(realpath -m "$REUSE_RUN")"
    [[ -d "$REUSE_RUN" ]] || { echo "[ERROR] reuse directory not found: $REUSE_RUN" >&2; exit 3; }
    if verify_and_reuse "$REUSE_RUN"; then
      exit 0
    fi
    echo "[ERROR] specified completed run failed verification; no training was started." >&2
    exit 4
  fi

  if [[ -d "$DEFAULT_COMPLETED_RUN" ]]; then
    if verify_and_reuse "$DEFAULT_COMPLETED_RUN"; then
      exit 0
    fi
    echo "[ERROR] completed local reference run exists but verification failed." >&2
    echo "[ERROR] Refusing to silently retrain. Inspect the preserved run, or use --fresh deliberately." >&2
    exit 4
  fi

  echo "[INFO] no preserved local completed reference run found."
  echo "[INFO] launching a brand-new full 35-unit reproduction."
fi

exec bash "$HERE/pipeline/launch_full.sh" "${PASSTHRU[@]}"
