#!/usr/bin/env bash

REPRO_SCRIPTS="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPRO_ROOT="$(cd "$REPRO_SCRIPTS/.." && pwd)"
REPO_ROOT="$(cd "$REPRO_ROOT/.." && pwd)"

# Backward-compatible name used by the existing reproduction scripts.
ROOT="$REPO_ROOT"

export REPRO_SCRIPTS REPRO_ROOT REPO_ROOT ROOT
