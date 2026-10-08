# Clean-room run templates

Scripts for running an official reproduction in a new, empty folder outside any
repository, as the clean-room run of 2026-10-03 was made
(`reproduction/docs/clean_room_acceptance.md`). Copy the three scripts into the
folder, then:

    RELEASE_TAG=<release tag> EXPECTED_COMMIT=<full commit of the tag> \
      bash cleanroom_setup.sh > cleanroom_setup.log 2>&1
    # On an NVIDIA GB10, flush the page cache first (README 0.6).
    tmux new-session -d -s tors-official "bash cleanroom_launch.sh"
    tmux new-session -d -s tors-monitor "bash cleanroom_monitor.sh"

| Script | What it does |
|---|---|
| `cleanroom_setup.sh` | README sections 0.1-0.6 verbatim: clones the tag, Git LFS, a new venv, the packages and their `pip freeze`, the downloads including the optional inputs of the analyses, `check.sh`, `verify_archived_sources.py` and the smoke test; every command with a UTC time stamp. Stops if the clone is not `EXPECTED_COMMIT` |
| `cleanroom_launch.sh` | `reproduce_all.sh --fresh` (README 0.7); the complete terminal output is kept in `cleanroom_run_terminal.log` |
| `cleanroom_monitor.sh` | Every 10 minutes one line in `cleanroom_resources.log`: free and available memory, page cache, GPU use, free disk, the training units finished and the command running; `LOW MEMORY` marks free memory under 8 GiB. Stops when the run has its `FINAL_SUMMARY.txt` or has failed |

Write every manual action during the run (for example a page-cache flush) in a
notes file next to the scripts.
