#!/usr/bin/env bash
# Official clean-room TORS run (README section 0.7), started inside tmux.
# The complete terminal output is also kept in cleanroom_run_terminal.log.
set -o pipefail
WORKSPACE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$WORKSPACE/context-aware-outfit-reproduction"
source .venv-repro/bin/activate
echo "[LAUNCH] $(date -u +%Y-%m-%dT%H:%M:%SZ) bash reproduction/scripts/reproduce_all.sh --fresh"
bash reproduction/scripts/reproduce_all.sh --fresh 2>&1 | tee -a "$WORKSPACE/cleanroom_run_terminal.log"
echo "[reproduce_all.sh exit=$?] $(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a "$WORKSPACE/cleanroom_run_terminal.log"
exec bash
