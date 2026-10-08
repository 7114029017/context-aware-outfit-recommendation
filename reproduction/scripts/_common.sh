#!/usr/bin/env bash

# The reproduction scripts source this file first. Native Windows (Git Bash, MSYS2, Cygwin) cannot
# run the pipeline, so stop here before any step does work; WSL2 reports Linux and is supported.
case "$(uname -s)" in
  MINGW*|MSYS*|CYGWIN*)
    cat >&2 <<'MSG'
[PLATFORM BLOCKED] native Windows (Git Bash, MSYS2 or Cygwin) is not supported.
Run the scripts inside WSL2 (Ubuntu 24.04), from a clone in the Linux file system (for example ~/),
as described in reproduction/README.md 0.0. On native Windows the archived training code cannot
start its DataLoader workers, the scripts call python3, and Python writes CRLF line endings, so
the SHA-256 checks of the outputs fail.
MSG
    exit 2 ;;
esac

REPRO_SCRIPTS="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPRO_ROOT="$(cd "$REPRO_SCRIPTS/.." && pwd)"
REPO_ROOT="$(cd "$REPRO_ROOT/.." && pwd)"

# Backward-compatible name used by the existing reproduction scripts.
ROOT="$REPO_ROOT"

export REPRO_SCRIPTS REPRO_ROOT REPO_ROOT ROOT
