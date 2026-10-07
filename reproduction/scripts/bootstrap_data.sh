#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/_common.sh"

usage() {
  cat <<'HELP'
Usage: bash reproduction/scripts/bootstrap_data.sh [DEST] [--with-images] [--with-fashionclip]
                                                   [--with-nomic] [--with-compendium]

Downloads the Polyvore Outfits annotations and metadata (no images) from the
Hugging Face dataset Stylique/Polyvore into DEST (default
_external/uiuc-polyvore-hf) and records DEST in reproduction/.local/. The 35
training units need nothing else.

Optional downloads for the analyses that run after the 35 units
(reproduction/docs/extensions.md, reproduction/docs/supplementary_analyses.md);
all stay outside the repository, at pinned revisions:
  --with-images      the Polyvore item images (images.zip, 2.5 GB, from the
                     same dataset), extracted to DEST/images/. Used by the
                     color analysis and the case figures; never committed or
                     redistributed (THIRD_PARTY_NOTICES.md).
  --with-fashionclip the FashionCLIP model (patrickjohncyh/fashion-clip, about
                     610 MB) into _external/fashion-clip/. Used by the
                     counterfactual analysis to encode the edited descriptions.
  --with-nomic       the text embedding model nomic-ai/nomic-embed-text-v2-moe
                     (about 1.9 GB) and its model code (nomic-ai/nomic-bert-2048)
                     into _external/. Used by the checklist coverage check
                     (thesis Figure 4-2).
  --with-compendium  the 2024 Adult Compendium of Physical Activities (PDF,
                     0.6 MB, pacompendium.com) into _external/compendium/. Used
                     by the MET reference check; reading it needs pdftotext
                     (poppler-utils).
HELP
}

DEST=""
WITH_IMAGES=0
WITH_FASHIONCLIP=0
WITH_NOMIC=0
WITH_COMPENDIUM=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --with-images) WITH_IMAGES=1; shift ;;
    --with-fashionclip) WITH_FASHIONCLIP=1; shift ;;
    --with-nomic) WITH_NOMIC=1; shift ;;
    --with-compendium) WITH_COMPENDIUM=1; shift ;;
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
NOMIC_REPO="nomic-ai/nomic-embed-text-v2-moe"
NOMIC_REVISION="1066b6599d099fbb93dfcb64f9c37a7c9e503e85"
NOMIC_CODE_REPO="nomic-ai/nomic-bert-2048"
NOMIC_CODE_REVISION="7710840340a098cfb869c4f65e87cf2b1b70caca"  # the revision in notebook P05's log
NOMIC_DEST="$REPO_ROOT/_external/nomic-embed-text-v2-moe"
NOMIC_CODE_DEST="$REPO_ROOT/_external/nomic-bert-2048"
COMPENDIUM_URL="https://pacompendium.com/wp-content/uploads/2025/02/1_2024-adult-compendium_1_2024.pdf"
COMPENDIUM_SHA256="ac30234b8f8f813837e282773cfcb3e0fe062334777c2f7b3213429c4fbc251c"  # = the 2025 handoff copy
COMPENDIUM_DEST="$REPO_ROOT/_external/compendium/1_2024-adult-compendium_1_2024.pdf"

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

if [[ "$WITH_NOMIC" -eq 1 ]]; then
  echo
  echo "[NOMIC] $NOMIC_REPO @ $NOMIC_REVISION -> $NOMIC_DEST"
  echo "[NOMIC] $NOMIC_CODE_REPO @ $NOMIC_CODE_REVISION -> $NOMIC_CODE_DEST"
  python - "$NOMIC_REPO" "$NOMIC_REVISION" "$NOMIC_DEST" "$NOMIC_CODE_REPO" "$NOMIC_CODE_REVISION" "$NOMIC_CODE_DEST" <<'PY'
import sys

from huggingface_hub import snapshot_download

repo, revision, dest, code_repo, code_revision, code_dest = sys.argv[1:7]
snapshot_download(repo_id=repo, revision=revision, local_dir=dest,
                  allow_patterns=["config.json", "config_sentence_transformers.json", "modules.json",
                                  "sentence_bert_config.json", "1_Pooling/config.json", "model.safetensors",
                                  "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
                                  "sentencepiece.bpe.model"])
snapshot_download(repo_id=code_repo, revision=code_revision, local_dir=code_dest,
                  allow_patterns=["configuration_hf_nomic_bert.py", "modeling_hf_nomic_bert.py", "config.json"])
PY
  printf '%s\n' "$NOMIC_DEST" > "$REPRO_ROOT/.local/nomic_model_root.txt"
  printf '%s\n' "$NOMIC_CODE_DEST" > "$REPRO_ROOT/.local/nomic_code_root.txt"
fi

if [[ "$WITH_COMPENDIUM" -eq 1 ]]; then
  echo
  echo "[COMPENDIUM] $COMPENDIUM_URL -> $COMPENDIUM_DEST"
  python - "$COMPENDIUM_URL" "$COMPENDIUM_DEST" "$COMPENDIUM_SHA256" <<'PY'
import hashlib
import sys
import urllib.request
from pathlib import Path

url, dest, expected = sys.argv[1], Path(sys.argv[2]), sys.argv[3]
if dest.is_file() and hashlib.sha256(dest.read_bytes()).hexdigest() == expected:
    print("[COMPENDIUM] already downloaded")
    raise SystemExit(0)
request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (reproduction bootstrap)"})
data = urllib.request.urlopen(request, timeout=120).read()
digest = hashlib.sha256(data).hexdigest()
if digest != expected:
    raise SystemExit(f"[ERROR] the Compendium file has SHA-256 {digest}, not the pinned {expected}; the site may have "
                     "published a new version")
dest.parent.mkdir(parents=True, exist_ok=True)
dest.write_bytes(data)
print(f"[COMPENDIUM] {len(data)} bytes")
PY
  printf '%s\n' "$COMPENDIUM_DEST" > "$REPRO_ROOT/.local/compendium_pdf.txt"
  command -v pdftotext >/dev/null || echo "[WARN] pdftotext not found; install poppler-utils for the MET reference check"
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
