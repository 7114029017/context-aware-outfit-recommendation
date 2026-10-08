#!/usr/bin/env bash
set -Eeuo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
DEFAULT_COMPLETED_RUN="$ROOT/_reproduction_runs/full_20260921T175217Z"
REUSE_RUN=""
FORCE_FRESH=0
ANALYSES_RUN=""
NO_DOWNLOADS=0
PASSTHRU=()
LAUNCH_ONLY=()
ANALYSES_ARGS=()
POLYVORE_ROOT=""

usage() {
  cat <<'HELP'
Usage:
  bash reproduction/scripts/reproduce_all.sh [--fresh | --analyses-only RUN | --reuse-reference PATH]
                                              [--no-optional-downloads] [launch_full options...]

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
      reference run is available. After the run has PASSED, every analysis
      runs as well (run_analyses.sh): the supplementary and extension
      analyses, the checks of the ported code, the comparison with the
      official run and the status of the 46 reproduction items in
      <run>/ITEMS_STATUS.md. One command reproduces every item.

  --analyses-only RUN
      Skip training and redo every analysis of the completed (PASSED) full
      run RUN with run_analyses.sh. Also accepts --out-dir DIR (default RUN),
      --figures-dir DIR and --skip-gpu.

  --no-optional-downloads
      Do not fetch the optional inputs of the analyses (Polyvore images,
      FashionCLIP, Nomic, Compendium). By default they are downloaded with
      bootstrap_data.sh before the run starts when they are missing (about
      5 GB); without them the items that need them are SKIPPED.

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
    --analyses-only)
      [[ $# -ge 2 ]] || { echo "[ERROR] --analyses-only requires RUN" >&2; exit 2; }
      ANALYSES_RUN="$2"
      shift 2
      ;;
    --no-optional-downloads)
      NO_DOWNLOADS=1
      shift
      ;;
    --out-dir|--figures-dir)
      [[ $# -ge 2 ]] || { echo "[ERROR] $1 requires a path" >&2; exit 2; }
      ANALYSES_ARGS+=("$1" "$2")
      shift 2
      ;;
    --skip-gpu)
      ANALYSES_ARGS+=("$1")
      shift
      ;;
    --polyvore-root)
      [[ $# -ge 2 ]] || { echo "[ERROR] --polyvore-root requires a path" >&2; exit 2; }
      POLYVORE_ROOT="$2"
      PASSTHRU+=("$1" "$2")
      shift 2
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
      LAUNCH_ONLY+=("$1")
      shift
      ;;
  esac
done

if [[ "$FORCE_FRESH" -eq 1 && -n "$REUSE_RUN" ]]; then
  echo "[ERROR] choose --fresh OR --reuse-reference, not both" >&2
  exit 2
fi
if [[ -n "$ANALYSES_RUN" && ( "$FORCE_FRESH" -eq 1 || -n "$REUSE_RUN" ) ]]; then
  echo "[ERROR] --analyses-only cannot be combined with --fresh or --reuse-reference" >&2
  exit 2
fi
if [[ -n "$ANALYSES_RUN" && ${#LAUNCH_ONLY[@]} -gt 0 ]]; then
  echo "[ERROR] --analyses-only does not train; these options do not apply: ${LAUNCH_ONLY[*]}" >&2
  exit 2
fi
if [[ ${#ANALYSES_ARGS[@]} -gt 0 && -z "$ANALYSES_RUN" ]]; then
  echo "[ERROR] --out-dir, --figures-dir and --skip-gpu belong to --analyses-only" >&2
  exit 2
fi

# The optional inputs of the analyses after the 35 units (all items of ITEMS_STATUS.md).
ensure_optional_downloads() {
  local local_dir="$ROOT/reproduction/.local" poly fclip nomic_model nomic_code compendium
  poly="${POLYVORE_ROOT:-$(cat "$local_dir/polyvore_root.txt" 2>/dev/null || true)}"
  fclip="$(cat "$local_dir/fashionclip_root.txt" 2>/dev/null || true)"
  nomic_model="$(cat "$local_dir/nomic_model_root.txt" 2>/dev/null || true)"
  nomic_code="$(cat "$local_dir/nomic_code_root.txt" 2>/dev/null || true)"
  compendium="$(cat "$local_dir/compendium_pdf.txt" 2>/dev/null || true)"
  local flags=()
  [[ -n "$poly" && -f "$poly/images/.complete" ]] || flags+=(--with-images)
  [[ -n "$fclip" && -f "$fclip/model.safetensors" ]] || flags+=(--with-fashionclip)
  [[ -n "$nomic_model" && -d "$nomic_model" && -n "$nomic_code" && -d "$nomic_code" ]] || flags+=(--with-nomic)
  [[ -n "$compendium" && -f "$compendium" ]] || flags+=(--with-compendium)
  command -v pdftotext >/dev/null 2>&1 ||
    echo "[DOWNLOADS] pdftotext (poppler-utils) is not installed: the MET reference check (the 457-entry MET candidate list) will be SKIPPED" >&2
  if [[ ${#flags[@]} -eq 0 ]]; then
    echo "[DOWNLOADS] the optional inputs of the analyses are present (images, FashionCLIP, Nomic, Compendium)"
    return 0
  fi
  if [[ "$NO_DOWNLOADS" -eq 1 ]]; then
    echo "[DOWNLOADS] missing: ${flags[*]}; --no-optional-downloads was given, so the items that need them will be SKIPPED" >&2
    return 0
  fi
  echo "[DOWNLOADS] fetching the optional inputs of the analyses: ${flags[*]}"
  if ! bash "$HERE/bootstrap_data.sh" ${poly:+"$poly"} "${flags[@]}"; then
    echo "[DOWNLOADS] the download failed; nothing was started. Fix the cause and run the same command again," >&2
    echo "[DOWNLOADS] or add --no-optional-downloads (the items that need these inputs are then SKIPPED)." >&2
    exit 5
  fi
}

if [[ -n "$ANALYSES_RUN" ]]; then
  [[ "$(cat "$ANALYSES_RUN/RUN_STATUS.txt" 2>/dev/null)" == "PASSED" ]] ||
    { echo "[ERROR] $ANALYSES_RUN is not a completed full run (its RUN_STATUS.txt is not PASSED)" >&2; exit 2; }
  ensure_optional_downloads
  analyses=(bash "$HERE/run_analyses.sh" --run-root "$ANALYSES_RUN")
  [[ -n "$POLYVORE_ROOT" ]] && analyses+=(--polyvore-root "$POLYVORE_ROOT")
  exec "${analyses[@]}" ${ANALYSES_ARGS[@]+"${ANALYSES_ARGS[@]}"}
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

ensure_optional_downloads
exec bash "$HERE/pipeline/launch_full.sh" ${PASSTHRU[@]+"${PASSTHRU[@]}"}
