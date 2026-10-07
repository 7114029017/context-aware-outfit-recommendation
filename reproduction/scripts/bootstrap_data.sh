#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/_common.sh"

usage() {
  cat <<'HELP'
Usage: bash reproduction/scripts/bootstrap_data.sh [DEST] [--with-images] [--with-fashionclip]

Downloads the Polyvore Outfits annotations and metadata (no images) from the
Hugging Face dataset Stylique/Polyvore into DEST (default
_external/uiuc-polyvore-hf) and records DEST in reproduction/.local/. The 35
training units need nothing else.

Optional downloads for the extension analyses that run after the 35 units
(reproduction/docs/extensions.md); both stay outside the repository:
  --with-images      the Polyvore item images (images.zip, 2.5 GB, from the
                     same dataset), extracted to DEST/images/. Used by the
                     color analysis and the case figures; never committed or
                     redistributed (THIRD_PARTY_NOTICES.md).
  --with-fashionclip the FashionCLIP model (patrickjohncyh/fashion-clip, about
                     610 MB) into _external/fashion-clip/. Used by the
                     counterfactual analysis to encode the edited descriptions.
HELP
}

DEST=""
WITH_IMAGES=0
WITH_FASHIONCLIP=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --with-images) WITH_IMAGES=1; shift ;;
    --with-fashionclip) WITH_FASHIONCLIP=1; shift ;;
    -h|--help) usage; exit 0 ;;
    -*) echo "[ERROR] unknown option: $1" >&2; usage; exit 2 ;;
    *) [[ -z "$DEST" ]] || { echo "[ERROR] only one DEST may be given" >&2; exit 2; }; DEST="$1"; shift ;;
  esac
done
DEST="${DEST:-$REPO_ROOT/_external/uiuc-polyvore-hf}"
REPO_ID="${POLYVORE_HF_REPO:-Stylique/Polyvore}"
# Fixed revisions, so that every download gets the same files.
REVISION="${POLYVORE_HF_REVISION:-15d6c58de0c2a2040297558ede70b2d2f7a070ed}"
IMAGES_ZIP_SHA256="a2d6868087c72083fcddca64fa493ab9d88a467e5b0576ea2f3aa60987c5b904"
FASHIONCLIP_REPO="patrickjohncyh/fashion-clip"
FASHIONCLIP_REVISION="7e3ba62ce16b379a1ab479346b66f192e76f51b7"  # the revision used by notebook P16
FASHIONCLIP_DEST="$REPO_ROOT/_external/fashion-clip"

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
echo "[HF DATASET] $REPO_ID @ $REVISION"
echo "[DEST]       $DEST"

python - "$REPO_ID" "$DEST" "$REVISION" <<'PY'
from huggingface_hub import snapshot_download
import sys

repo_id, dest, revision = sys.argv[1], sys.argv[2], sys.argv[3]
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
    revision=revision,
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

if [[ "$WITH_IMAGES" -eq 1 ]]; then
  echo
  echo "[IMAGES] $REPO_ID images.zip -> $ABS_DEST/images/"
  python - "$REPO_ID" "$ABS_DEST" "$REVISION" "$IMAGES_ZIP_SHA256" <<'PY'
import hashlib
import sys
import zipfile
from pathlib import Path

from huggingface_hub import hf_hub_download

repo_id, dest, revision, expected = sys.argv[1], Path(sys.argv[2]), sys.argv[3], sys.argv[4]
images = dest / "images"
marker = images / ".complete"
if marker.is_file() and marker.read_text(encoding="utf-8").split()[0] == expected:
    print(f"[IMAGES] already extracted: {marker.read_text(encoding='utf-8').strip()}")
    raise SystemExit(0)
archive = Path(hf_hub_download(repo_id=repo_id, repo_type="dataset", filename="images.zip",
                               revision=revision, local_dir=dest))
h = hashlib.sha256()
with archive.open("rb") as stream:
    for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
        h.update(chunk)
if h.hexdigest() != expected:
    raise SystemExit(f"[ERROR] images.zip SHA-256 {h.hexdigest()} differs from the expected {expected}")
count = 0
images.mkdir(parents=True, exist_ok=True)
with zipfile.ZipFile(archive) as z:
    for info in z.infolist():
        # Only images/<item id>.jpg; the archive also holds macOS __MACOSX/ entries.
        parts = info.filename.split("/")
        if info.is_dir() or len(parts) != 2 or parts[0] != "images" or not parts[1].endswith(".jpg"):
            continue
        if not parts[1][:-4].isdigit():
            continue
        (images / parts[1]).write_bytes(z.read(info))
        count += 1
marker.write_text(f"{expected} {count} images\n", encoding="utf-8")
archive.unlink()
print(f"[IMAGES] extracted {count} images; removed the archive")
PY
fi

if [[ "$WITH_FASHIONCLIP" -eq 1 ]]; then
  echo
  echo "[FASHIONCLIP] $FASHIONCLIP_REPO @ $FASHIONCLIP_REVISION -> $FASHIONCLIP_DEST"
  python - "$FASHIONCLIP_REPO" "$FASHIONCLIP_REVISION" "$FASHIONCLIP_DEST" <<'PY'
import sys

from huggingface_hub import snapshot_download

repo_id, revision, dest = sys.argv[1:4]
snapshot_download(repo_id=repo_id, revision=revision, local_dir=dest,
                  allow_patterns=["config.json", "preprocessor_config.json", "special_tokens_map.json",
                                  "tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt",
                                  "model.safetensors"])
PY
  printf '%s\n' "$FASHIONCLIP_DEST" > "$REPRO_ROOT/.local/fashionclip_root.txt"
fi

echo
echo "[READY] POLYVORE_ROOT=$ABS_DEST"
echo "[SOURCE] $REPO_ID"
echo
if [[ "$WITH_IMAGES" -eq 1 ]]; then
  echo "The Polyvore images in $ABS_DEST/images/ are for local use only; do not commit or redistribute them."
else
  echo "IMPORTANT: this bootstrap downloads annotations/metadata, NOT the Polyvore images."
fi
echo "Next: pull archived LFS features and run the reproduction preflight:"
echo "  git lfs pull"
echo "  bash reproduction/scripts/check.sh"
echo "Do NOT overwrite reproduction/splits/: these are versioned reference IDs."
