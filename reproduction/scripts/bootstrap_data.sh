#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/_common.sh"

DEST="${1:-$REPO_ROOT/_external/uiuc-polyvore-hf}"
REPO_ID="${POLYVORE_HF_REPO:-Stylique/Polyvore}"

echo "[1/4] Checking Python environment"
python - <<'PY'
import sys
print(sys.version)
PY

echo "[2/4] Checking huggingface_hub from the prepared environment"
python - <<'PY2'
import importlib.metadata as md
print("huggingface_hub:", md.version("huggingface_hub"))
PY2

echo "[3/4] Downloading only files required for checkpoint reproduction"
echo "[HF DATASET] $REPO_ID"
echo "[DEST]       $DEST"

python - "$REPO_ID" "$DEST" <<'PY'
from huggingface_hub import snapshot_download
import sys

repo_id, dest = sys.argv[1], sys.argv[2]
patterns = [
    "disjoint/train.json",
    "disjoint/valid.json",
    "disjoint/test.json",
    "disjoint/compatibility_train.txt",
    "disjoint/compatibility_valid.txt",
    "disjoint/compatibility_test.txt",
    "disjoint/fill_in_blank_train.json",
    "disjoint/fill_in_blank_valid.json",
    "disjoint/fill_in_blank_test.json",
    "categories.csv",
    "polyvore_item_metadata.json",
    "polyvore_outfit_titles.json",
]
snapshot_download(
    repo_id=repo_id,
    repo_type="dataset",
    local_dir=dest,
    allow_patterns=patterns,
)
PY

echo "[4/4] Verifying expected layout"
REQUIRED=(
  "polyvore_item_metadata.json"
  "disjoint/train.json"
  "disjoint/valid.json"
  "disjoint/test.json"
  "disjoint/compatibility_train.txt"
  "disjoint/compatibility_test.txt"
  "disjoint/fill_in_blank_train.json"
  "disjoint/fill_in_blank_test.json"
)

FAILED=0
for REL in "${REQUIRED[@]}"; do
  if [[ -f "$DEST/$REL" ]]; then
    echo "[OK] $REL"
  else
    echo "[MISSING] $REL"
    FAILED=1
  fi
done

if [[ "$FAILED" -ne 0 ]]; then
  echo "[ERROR] Hugging Face mirror does not satisfy the expected raw UIUC layout."
  exit 4
fi

mkdir -p "$REPRO_ROOT/.local"
ABS_DEST="$(cd "$DEST" && pwd)"
printf '%s\n' "$ABS_DEST" > "$REPRO_ROOT/.local/polyvore_root.txt"
printf '%s\n' "$REPO_ID" > "$REPRO_ROOT/.local/polyvore_source.txt"

echo
echo "[READY] POLYVORE_ROOT=$ABS_DEST"
echo "[SOURCE] $REPO_ID"
echo
echo "IMPORTANT: this bootstrap downloads annotations/metadata, NOT the Polyvore images."
echo "Next: pull archived LFS features and run the reproduction preflight:"
echo "  git lfs pull"
echo "  bash reproduction/scripts/check.sh"
echo "Do NOT overwrite reproduction/splits/: these are versioned reference IDs."
