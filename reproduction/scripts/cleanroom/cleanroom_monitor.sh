#!/usr/bin/env bash
# Every 10 minutes during the run, one line in cleanroom_resources.log: free and available memory and page
# cache (GiB), GPU utilization (%), free disk (GiB), the run folder, its RUN_STATUS, the training units
# finished (of 35) and the command running. "LOW MEMORY" marks free memory under 8 GiB (README 0.6: on a GB10
# CUDA counts only free memory). Stops when the run has its FINAL_SUMMARY.txt or has FAILED.
WORKSPACE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG="$WORKSPACE/cleanroom_resources.log"
RUNS="$WORKSPACE/context-aware-outfit-recommendation/reproduction/runs"
[[ -f "$LOG" ]] || echo "utc,mem_free_gib,mem_available_gib,page_cache_gib,gpu_util_pct,disk_free_gib,run,run_status,units_done,running,note" > "$LOG"
while true; do
  run="$(ls -d "$RUNS"/full_* 2>/dev/null | tail -n 1)"
  status="$(cat "$run/RUN_STATUS.txt" 2>/dev/null || echo none)"
  read -r free avail cache < <(free -g | awk '/^Mem:/ {print $4, $7, $6}')
  gpu="$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null | head -n 1)"
  disk="$(df -BG --output=avail "$WORKSPACE" | tail -n 1 | tr -dc 0-9)"
  units="$(grep -l '"status": "passed' "$run"/main/*/full_train_manifest.json "$run"/ablation/runs/*/fair_subset_run_manifest.json 2>/dev/null | wc -l)"
  running="$(grep -a '^\[EXEC\]' "$run/logs/full_console.log" 2>/dev/null | tail -n 1 | sed 's#.*/python3* ##' | cut -c1-90 | tr ',"' '; ')"
  note=""; [[ "$free" -lt 8 ]] && note="LOW MEMORY"
  echo "$(date -u +%Y-%m-%dT%H:%M:%SZ),$free,$avail,$cache,$gpu,$disk,$(basename "${run:-none}"),$status,$units,\"$running\",$note" >> "$LOG"
  if [[ -f "$run/FINAL_SUMMARY.txt" || "$status" == FAILED* ]]; then
    echo "$(date -u +%Y-%m-%dT%H:%M:%SZ),,,,,,$(basename "$run"),$status,$units,\"monitor stopped\"," >> "$LOG"
    break
  fi
  sleep 600
done
