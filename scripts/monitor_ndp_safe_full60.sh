#!/usr/bin/env bash
set -u

BASE="${1:?usage: monitor_ndp_safe_full60.sh OUTPUT_ROOT}"
PYTHON="/usr/local/bin/python3.10"
SUMMARY="/home/DIFFER/scripts/summarize_ndp_safe_full60.py"
LOG="$BASE/monitor_summary.log"

mkdir -p "$BASE"
echo "MONITOR_STARTED $(date -Is)" >> "$LOG"

while :; do
  complete=0
  for f in \
    "$BASE/LTCC/RUN_COMPLETE.json" \
    "$BASE/NDP_SAFE/LTCC/RUN_COMPLETE.json" \
    "$BASE/PRCC/RUN_COMPLETE.json" \
    "$BASE/NDP_SAFE/PRCC/RUN_COMPLETE.json"; do
    if [ -f "$f" ]; then
      complete=$((complete + 1))
    fi
  done

  forbidden=$(find "$BASE" -type f \( \
    -iname 'rank*state.pt' -o -iname 'rank*state.pth' \
    -o -iname 'checkpoint_e*_rank*.pt' -o -iname 'checkpoint_e*_rank*.pth' \
  \) -print)
  if [ -n "$forbidden" ]; then
    echo "FORBIDDEN_RANK_FILES_DETECTED $(date -Is)" >> "$LOG"
    echo "$forbidden" >> "$LOG"
    exit 2
  fi

  if [ "$complete" -eq 4 ]; then
    echo "ALL_TRAIN_COMPLETE $(date -Is)" >> "$LOG"
    "$PYTHON" "$SUMMARY" --root "$BASE" >> "$LOG" 2>&1
    rc=$?
    echo "SUMMARY_EXIT=$rc $(date -Is)" >> "$LOG"
    exit "$rc"
  fi
  sleep 60
done
