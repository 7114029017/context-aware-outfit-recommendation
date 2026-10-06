#!/usr/bin/env bash
# Clean-room setup for the official TORS rerun, following
# reproduction/README.md section 0 (0.1-0.6) verbatim. Every command is
# echoed with a UTC timestamp; output goes to cleanroom_setup.log.
set -euo pipefail
export PS4='+ [$(date -u +%Y-%m-%dT%H:%M:%SZ)] '
set -x

WORKSPACE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$WORKSPACE"

# 0.1 Clone the official reproduction branch.
git clone \
  --branch thesis-full-reproduction \
  --single-branch \
  https://github.com/guoyouuu/context-aware-outfit-recommendation.git \
  context-aware-outfit-reproduction
cd context-aware-outfit-reproduction
git branch --show-current
git rev-parse HEAD
git status --short

# 0.2 Git LFS feature artifacts.
git lfs install
git lfs pull

# 0.3 New Python environment.
python3 -m venv .venv-repro
set +x
source .venv-repro/bin/activate
set -x
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

# 0.4 Re-download Polyvore annotations / metadata.
bash reproduction/scripts/bootstrap_data.sh

# 0.5 Clean preflight.
bash reproduction/scripts/check.sh

# 0.6 Smoke test.
bash reproduction/scripts/smoke_test.sh

git status --short
echo "[CLEANROOM SETUP DONE] $(date -u +%Y-%m-%dT%H:%M:%SZ)"
