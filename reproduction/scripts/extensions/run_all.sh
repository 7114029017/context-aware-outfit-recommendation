#!/usr/bin/env bash
# Extension analyses after a PASSED full run (reproduction/docs/extensions.md). They regenerate
# manuscript analyses that the 35 training units do not produce, from the run's own models, and
# never change the run's results or RUN_STATUS.txt.
# Usage: bash reproduction/scripts/extensions/run_all.sh --run-root RUN [--out-dir DIR]
#            [--figures-dir DIR] [--polyvore-root PATH] [--steps LIST] [--skip-gpu] [--validate]
# Steps, in this order (default: all of them):
#   text_length     GPU, about 20 min: per-query CIR results of the 10 main units, then the length analysis
#   counterfactual  CPU, about 1 min: Tables 4 and 9 with the run's Context models (needs FashionCLIP)
#   two_tower       GPU, 1.5-3 hours: retrain the Two-Tower model, Table 8
#   color           CPU, seconds: color-shift analysis (needs the Polyvore images)
#   figures         CPU, seconds: Figures A1-A3, thesis Figure 4-14 and the 2025 figures F34a/F34b (needs the
#                   images; the PNG files stay local)
#   reliability     GPU, about 5 min: reliability analysis of the seed-1 CIR models (2025 T18, F12-F16)
#   text_swap       GPU, about 45 min: the main models evaluated with the other outfit text (2025 sweep)
#   outfit_generation CPU, about 2 min: step-by-step outfit generation of the 2025 try-on demo (needs the
#                   images and FashionCLIP; the item grid stays local)
# --validate also checks the ported code against the 2025 outputs: archived A07 table, the 20 preserved
# Two-Tower checkpoints, the 2025 counterfactual model and labels A16/T18 (CPU, about 5 min), the
# 2025 outfit-generation picks (CPU, about 2 min), and with a GPU the 2025 reliability table T18
# (about 5 min) and the 20 preserved main checkpoints with both texts against A14 (about 75 min).
# Single checks: list them in --steps with --validate, e.g. --validate --steps validate_text_swap.
# A step whose GPU or download is missing is SKIPPED; a failing step does not stop the others.
# DIR (default RUN/extensions) receives the outputs, logs/<step>.log and EXTENSIONS_STATUS.txt (a step
# run again replaces its line; the lines of the other steps are kept).
# Exit status 1 if a step FAILED.
set -Eeuo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPRO="$(cd "$HERE/../.." && pwd)"
PYTHON="${PYTHON:-python3}"
RUN_ROOT=""
OUT_DIR=""
FIGURES_DIR=""
POLYVORE_ARGS=()
STEPS="text_length,counterfactual,two_tower,color,figures,reliability,text_swap,outfit_generation"
SKIP_GPU=0
VALIDATE=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --run-root) RUN_ROOT="${2:?--run-root needs a path}"; shift 2 ;;
    --out-dir) OUT_DIR="${2:?--out-dir needs a path}"; shift 2 ;;
    --figures-dir) FIGURES_DIR="${2:?--figures-dir needs a path}"; shift 2 ;;
    --polyvore-root) POLYVORE_ARGS=(--polyvore-root "${2:?--polyvore-root needs a path}"); shift 2 ;;
    --steps) STEPS="${2:?--steps needs a list}"; shift 2 ;;
    --skip-gpu) SKIP_GPU=1; shift ;;
    --validate) VALIDATE=1; shift ;;
    -h|--help) sed -n '2,26p' "$0"; exit 0 ;;
    *) echo "[ERROR] unknown argument: $1" >&2; exit 2 ;;
  esac
done
[[ -n "$RUN_ROOT" ]] || { echo "[ERROR] --run-root is required" >&2; exit 2; }
RUN_ROOT="$(cd "$RUN_ROOT" && pwd)"
[[ "$(cat "$RUN_ROOT/RUN_STATUS.txt" 2>/dev/null)" == "PASSED" ]] ||
  { echo "[EXTENSION BLOCKED] $RUN_ROOT: RUN_STATUS is not PASSED" >&2; exit 2; }
OUT_DIR="${OUT_DIR:-$RUN_ROOT/extensions}"
mkdir -p "$OUT_DIR/logs"
OUT_DIR="$(cd "$OUT_DIR" && pwd)"
FIGURES_DIR="${FIGURES_DIR:-$OUT_DIR/case_figures}"
# The steps run from this script's folder, so relative paths are made absolute here.
[[ "$FIGURES_DIR" == /* ]] || FIGURES_DIR="$PWD/$FIGURES_DIR"
if [[ ${#POLYVORE_ARGS[@]} -gt 0 && "${POLYVORE_ARGS[1]}" != /* ]]; then
  POLYVORE_ARGS=(--polyvore-root "$PWD/${POLYVORE_ARGS[1]}")
fi
STATUS="$OUT_DIR/EXTENSIONS_STATUS.txt"
touch "$STATUS"

if [[ ${#POLYVORE_ARGS[@]} -gt 0 ]]; then
  POLYVORE="${POLYVORE_ARGS[1]}"
else
  POLYVORE="$(cat "$REPRO/.local/polyvore_root.txt" 2>/dev/null || true)"
fi
FASHIONCLIP="$(cat "$REPRO/.local/fashionclip_root.txt" 2>/dev/null || true)"
HAS_GPU=0
if [[ "$SKIP_GPU" -eq 0 ]] && "$PYTHON" -c 'import sys, torch; sys.exit(0 if torch.cuda.is_available() else 1)' 2>/dev/null; then
  HAS_GPU=1
fi

FAILED=0
note() {  # replaces the step's line left by an earlier invocation
  local line
  line="$(printf '%-15s %s' "$1:" "$2")"
  { grep -v "^$1:" "$STATUS" || true; echo "$line"; } > "$STATUS.tmp"
  mv "$STATUS.tmp" "$STATUS"
  echo "$line"
}
run_step() {  # name, command...
  local name="$1"; shift
  if (cd "$HERE" && "$@") > "$OUT_DIR/logs/$name.log" 2>&1; then
    note "$name" "PASSED"
  else
    note "$name" "FAILED (see $OUT_DIR/logs/$name.log)"
    FAILED=1
  fi
}
wanted() { [[ ",$STEPS," == *",$1,"* ]]; }
gpu_or_skip() {
  if [[ "$HAS_GPU" -eq 1 ]]; then return 0; fi
  note "$1" "SKIPPED ($([[ "$SKIP_GPU" -eq 1 ]] && echo "--skip-gpu" || echo "no CUDA GPU"); needs the GPU)"
  return 1
}
images_or_skip() {
  if [[ -n "$POLYVORE" && -f "$POLYVORE/images/.complete" ]]; then return 0; fi
  note "$1" "SKIPPED (no Polyvore images; run bootstrap_data.sh --with-images)"
  return 1
}

text_length_step() {
  "$PYTHON" main_cir_per_query.py --run-root "$RUN_ROOT" --out-dir "$OUT_DIR/main_cir_per_query" \
    ${POLYVORE_ARGS[@]+"${POLYVORE_ARGS[@]}"} &&
  "$PYTHON" length_analysis.py --detail-dir "$OUT_DIR/main_cir_per_query" --out-dir "$OUT_DIR/text_length" \
    --label "$(basename "$RUN_ROOT")" ${POLYVORE_ARGS[@]+"${POLYVORE_ARGS[@]}"}
}
if wanted text_length && gpu_or_skip text_length; then
  run_step text_length text_length_step
fi
if wanted counterfactual; then
  if [[ -n "$FASHIONCLIP" && -f "$FASHIONCLIP/model.safetensors" ]]; then
    CF_ARGS=(--run-root "$RUN_ROOT" --out-dir "$OUT_DIR/counterfactual")
    [[ "$VALIDATE" -eq 1 ]] && CF_ARGS+=(--validate-2025)
    run_step counterfactual "$PYTHON" counterfactual.py "${CF_ARGS[@]}" ${POLYVORE_ARGS[@]+"${POLYVORE_ARGS[@]}"}
  else
    note counterfactual "SKIPPED (no FashionCLIP; run bootstrap_data.sh --with-fashionclip)"
  fi
fi
if wanted two_tower && gpu_or_skip two_tower; then
  run_step two_tower "$PYTHON" two_tower.py --train --out-dir "$OUT_DIR/two_tower" ${POLYVORE_ARGS[@]+"${POLYVORE_ARGS[@]}"}
fi
if wanted color && images_or_skip color; then
  run_step color "$PYTHON" color_analysis.py --run-root "$RUN_ROOT" --out-dir "$OUT_DIR/color_analysis" \
    ${POLYVORE_ARGS[@]+"${POLYVORE_ARGS[@]}"}
fi
if wanted figures && images_or_skip figures; then
  run_step figures "$PYTHON" case_figures.py --run-root "$RUN_ROOT" --out-dir "$FIGURES_DIR" \
    --manifest-dir "$OUT_DIR/case_figures" --counterfactual-dir "$OUT_DIR/counterfactual" \
    ${POLYVORE_ARGS[@]+"${POLYVORE_ARGS[@]}"}
fi
if wanted reliability && gpu_or_skip reliability; then
  run_step reliability "$PYTHON" reliability_analysis.py --run-root "$RUN_ROOT" --out-dir "$OUT_DIR/reliability" \
    ${POLYVORE_ARGS[@]+"${POLYVORE_ARGS[@]}"}
fi
if wanted text_swap && gpu_or_skip text_swap; then
  run_step text_swap "$PYTHON" text_swap.py --run-root "$RUN_ROOT" --out-dir "$OUT_DIR/text_swap" \
    ${POLYVORE_ARGS[@]+"${POLYVORE_ARGS[@]}"}
fi
fashionclip_or_skip() {
  if [[ -n "$FASHIONCLIP" && -f "$FASHIONCLIP/model.safetensors" ]]; then return 0; fi
  note "$1" "SKIPPED (no FashionCLIP; run bootstrap_data.sh --with-fashionclip)"
  return 1
}
if wanted outfit_generation && images_or_skip outfit_generation && fashionclip_or_skip outfit_generation; then
  run_step outfit_generation "$PYTHON" outfit_generation.py --run-root "$RUN_ROOT" \
    --out-dir "$OUT_DIR/outfit_generation" --figures-dir "$FIGURES_DIR" ${POLYVORE_ARGS[@]+"${POLYVORE_ARGS[@]}"}
fi
checked() {  # with --validate: every check, or only the checks named in --steps if it names any
  [[ ",$STEPS," == *",validate_"* ]] || return 0
  wanted "$1"
}
if [[ "$VALIDATE" -eq 1 ]]; then
  if checked validate_length; then
    run_step validate_length "$PYTHON" length_analysis.py \
      --rowlevel "$(cd "$REPRO/.." && pwd)/03_實驗與結果_experiments_results/01_文字長度影響分析/A07_length_performance_rowlevel_cir.csv" \
      --out-dir "$OUT_DIR/validation/text_length_2025_A07" --label "2025 A07"
  fi
  if checked validate_two_tower; then
    run_step validate_two_tower "$PYTHON" two_tower.py --evaluate-preserved \
      --out-dir "$OUT_DIR/validation/two_tower_preserved_checkpoints" ${POLYVORE_ARGS[@]+"${POLYVORE_ARGS[@]}"}
  fi
  if checked validate_reliability_labels; then
    run_step validate_reliability_labels "$PYTHON" reliability_analysis.py --labels-only \
      --out-dir "$OUT_DIR/validation/reliability_labels" ${POLYVORE_ARGS[@]+"${POLYVORE_ARGS[@]}"}
  fi
  if checked validate_outfit_generation && images_or_skip validate_outfit_generation &&
     fashionclip_or_skip validate_outfit_generation; then
    run_step validate_outfit_generation "$PYTHON" outfit_generation.py --validate-2025 \
      --out-dir "$OUT_DIR/validation/outfit_generation_2025" --figures-dir "$FIGURES_DIR" \
      ${POLYVORE_ARGS[@]+"${POLYVORE_ARGS[@]}"}
  fi
  if checked validate_reliability && gpu_or_skip validate_reliability; then
    run_step validate_reliability "$PYTHON" reliability_analysis.py --validate-2025 \
      --out-dir "$OUT_DIR/validation/reliability_2025_seed1" ${POLYVORE_ARGS[@]+"${POLYVORE_ARGS[@]}"}
  fi
  if checked validate_text_swap && gpu_or_skip validate_text_swap; then
    run_step validate_text_swap "$PYTHON" text_swap.py --checkpoints-2025 \
      --out-dir "$OUT_DIR/validation/text_swap_2025_checkpoints" ${POLYVORE_ARGS[@]+"${POLYVORE_ARGS[@]}"}
  fi
fi
echo "[EXTENSIONS] status: $STATUS"
exit "$FAILED"
