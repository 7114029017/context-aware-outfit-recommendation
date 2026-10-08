#!/usr/bin/env bash
# Clean-room setup for an official TORS run, following reproduction/README.md section 0 (0.1-0.6)
# verbatim at the release tag RELEASE_TAG. Every command is echoed with a UTC
# timestamp; output goes to cleanroom_setup.log. The clone's HEAD must be EXPECTED_COMMIT.
set -euo pipefail
export PS4='+ [$(date -u +%Y-%m-%dT%H:%M:%SZ)] '
set -x

WORKSPACE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RELEASE_TAG="${RELEASE_TAG:?set RELEASE_TAG, for example v1.0.0-tors-reproduction}"
EXPECTED_COMMIT="${EXPECTED_COMMIT:?set EXPECTED_COMMIT to the full commit hash of RELEASE_TAG}"
cd "$WORKSPACE"

# 0.1 Clone the release tag.
git clone \
  --branch "$RELEASE_TAG" \
  https://github.com/7114029017/context-aware-outfit-recommendation.git
cd context-aware-outfit-recommendation
git describe --tags
git rev-parse HEAD
test "$(git rev-parse HEAD)" = "$EXPECTED_COMMIT"
git status --short

# 0.2 Git LFS feature artifacts.
git lfs install
git lfs pull
git lfs ls-files

# 0.3 New Python environment.
python3.12 -m venv .venv-repro
set +x
source .venv-repro/bin/activate
set -x
python --version
python -m pip install --upgrade pip
python -m pip install \
  torch==2.9.1 \
  torchvision==0.24.1 \
  --index-url https://download.pytorch.org/whl/cu130
python -m pip install \
  -r reproduction/environment/requirements-reproduction-runtime.txt
python - <<'PY'
import torch, torchvision
print("torch:", torch.__version__)
print("torchvision:", torchvision.__version__)
print("cuda available:", torch.cuda.is_available())
print("torch cuda:", torch.version.cuda)
print("cudnn:", torch.backends.cudnn.version())
print("gpu:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NONE")
PY
python -m pip freeze > "$WORKSPACE/cleanroom_pip_freeze.txt"

# 0.4 Polyvore annotations / metadata and the optional inputs of the analyses.
bash reproduction/scripts/bootstrap_data.sh --with-images --with-fashionclip --with-nomic --with-compendium

# 0.5 Clean preflight.
bash reproduction/scripts/check.sh
python3 reproduction/scripts/verify_archived_sources.py

# 0.6 Smoke test.
bash reproduction/scripts/smoke_test.sh

git status --short
echo "[CLEANROOM SETUP DONE] $(date -u +%Y-%m-%dT%H:%M:%SZ)"
