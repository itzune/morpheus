#!/bin/bash
# Monitor training; stop cleanly right after a checkpoint save.
#
# Uses FILE-BASED detection (not log-line) because stdout buffering can
# delay the "Saved:" line. The checkpoint file appearing on disk is atomic
# and immediate — it's the reliable trigger.
#
# At a checkpoint step (multiple of save_interval=2000), the order in
# train.py is:
#   [VALID] → [AUTOCOMPLETE] → [BEST] (best.pt) → save checkpoint file
# So when the checkpoint FILE appears, eval is ALREADY complete and W&B
# log() calls for that eval have already been issued.
#
# We send SIGINT (not SIGKILL) so W&B catches it, flushes buffers, and
# marks the run as finished cleanly → continuous wandb graph on resume.
#
# Usage:
#   scripts/stop_after_checkpoint.sh              # stop at the NEXT checkpoint
#   scripts/stop_after_checkpoint.sh --target 50000  # stop at/after step >= 50000
#
# In --target mode the monitor loops, waiting for each checkpoint to appear
# and stabilize WITHOUT stopping training, until the next checkpoint step
# reaches the target. This survives SSH drops (runs via nohup on the server).

cd ~/morpheus-mamba || exit 1
LOG=data/train.log
MONITOR_LOG=data/stop_monitor.log
SAVE_INTERVAL=2000
MIN_SIZE=540000000
STABLE_REQUIRED=3

# --- Parse args ---
TARGET_STEP=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --target) TARGET_STEP="$2"; shift 2;;
    *) echo "Unknown arg: $1" >&2; exit 1;;
  esac
done

echo "[$(date)] Stop monitor started (file-based detection)" | tee "$MONITOR_LOG"
if [ "$TARGET_STEP" -gt 0 ]; then
  echo "[$(date)] Target: stop after checkpoint at step >= $TARGET_STEP" | tee -a "$MONITOR_LOG"
else
  echo "[$(date)] No target — will stop at the NEXT checkpoint" | tee -a "$MONITOR_LOG"
fi

# --- Wait for a specific checkpoint file to appear + write to complete ---
# Returns 0 on success, 1 if training died before the file appeared.
wait_for_checkpoint() {
  local ckpt="$1"
  local step="$2"
  echo "[$(date)] Waiting for checkpoint file: $ckpt (step $step)" | tee -a "$MONITOR_LOG"

  # Wait for the file to APPEAR
  while true; do
    if [ -f "$ckpt" ]; then
      echo "[$(date)] ✓ Checkpoint file appeared: $ckpt" | tee -a "$MONITOR_LOG"
      break
    fi
    if ! pgrep -f "train.py --config" >/dev/null 2>&1; then
      echo "[$(date)] WARNING: training process not found — it may have stopped already" | tee -a "$MONITOR_LOG"
      return 1
    fi
    sleep 10
  done

  # Wait for the write to COMPLETE (size stability)
  # Poll file size; only proceed when unchanged for 3 consecutive polls (15s)
  # AND large enough to be a complete checkpoint (~547MB). This guarantees
  # torch.save() has finished writing before we send any signal (or loop on).
  local prev_size=-1 stable_count=0
  echo "[$(date)] Waiting for checkpoint write to complete (size stability)…" | tee -a "$MONITOR_LOG"
  while true; do
    local curr_size=$(stat -c%s "$ckpt" 2>/dev/null || echo 0)
    if [ "$curr_size" -eq "$prev_size" ] && [ "$curr_size" -gt 0 ]; then
      stable_count=$((stable_count + 1))
    else
      stable_count=0
    fi
    if [ "$stable_count" -ge "$STABLE_REQUIRED" ] && [ "$curr_size" -ge "$MIN_SIZE" ]; then
      echo "[$(date)] ✓ Size stable at $curr_size bytes for 15s (≥ ${MIN_SIZE}) — write COMPLETE" | tee -a "$MONITOR_LOG"
      break
    fi
    prev_size=$curr_size
    sleep 5
  done
  return 0
}

# --- Main loop ---
while true; do
  # Auto-detect the NEXT expected checkpoint step
  LAST_STEP_FILE=$(ls -1 checkpoints/step_*.pt 2>/dev/null | sort | tail -1)
  if [ -z "$LAST_STEP_FILE" ]; then
    echo "[$(date)] ERROR: No existing checkpoints found" | tee -a "$MONITOR_LOG"
    exit 1
  fi
  LAST_STEP=$(basename "$LAST_STEP_FILE" | sed 's/step_0*//; s/\.pt//')
  LAST_STEP=$((10#$LAST_STEP))  # strip leading zeros
  NEXT_STEP=$((LAST_STEP + SAVE_INTERVAL))
  NEXT_CKPT="checkpoints/step_$(printf '%07d' $NEXT_STEP).pt"
  echo "[$(date)] Last checkpoint: step_$LAST_STEP → next expected: step_$NEXT_STEP" | tee -a "$MONITOR_LOG"

  # In target mode: if next checkpoint is still below target, wait for it
  # (without stopping) then loop to re-detect the following checkpoint.
  if [ "$TARGET_STEP" -gt 0 ] && [ "$NEXT_STEP" -lt "$TARGET_STEP" ]; then
    if ! wait_for_checkpoint "$NEXT_CKPT" "$NEXT_STEP"; then
      echo "[$(date)] Training stopped unexpectedly; monitor exiting" | tee -a "$MONITOR_LOG"
      exit 1
    fi
    echo "[$(date)] Reached step $NEXT_STEP (below target $TARGET_STEP) — continuing…" | tee -a "$MONITOR_LOG"
    sleep 5  # brief pause before re-detecting the next checkpoint
    continue
  fi

  # At/past target (or no target): wait for this checkpoint, then STOP.
  if ! wait_for_checkpoint "$NEXT_CKPT" "$NEXT_STEP"; then
    echo "[$(date)] Training stopped unexpectedly; monitor exiting" | tee -a "$MONITOR_LOG"
    exit 1
  fi
  break
done

# Small grace period so the W&B log() calls for this eval flush
sleep 3

# --- Find MAIN training PID ---
# Must be the PYTHON interpreter running train.py, NOT:
#   - the bash -c wrapper that launched it (matches 'train.py --config' too)
#   - forkserver workers (have 'forkserver' + 'import sys' in args)
# We filter on comm (^python) so the bash wrapper is never selected.
MAIN_PID=$(ps -eo pid,comm,args | awk '$2 ~ /^python/ && /train\.py --config/ && !/forkserver/ && !/import sys/ {print $1; exit}')
echo "[$(date)] Main training PID: ${MAIN_PID:-none}" | tee -a "$MONITOR_LOG"

if [ -z "$MAIN_PID" ]; then
  echo "[$(date)] No main PID to signal — already stopped?" | tee -a "$MONITOR_LOG"
else
  echo "[$(date)] Sending SIGINT to PID $MAIN_PID (clean W&B shutdown)..." | tee -a "$MONITOR_LOG"
  kill -INT "$MAIN_PID"

  # Wait up to 90s for clean exit
  for i in $(seq 1 18); do
    sleep 5
    if ! kill -0 "$MAIN_PID" 2>/dev/null; then
      echo "[$(date)] Process $MAIN_PID exited cleanly after $((i*5))s" | tee -a "$MONITOR_LOG"
      break
    fi
    [ $((i % 2)) -eq 0 ] && echo "[$(date)] Still waiting for exit... (${i}x5s)" | tee -a "$MONITOR_LOG"
  done

  # If still alive, escalate to SIGTERM then SIGKILL
  if kill -0 "$MAIN_PID" 2>/dev/null; then
    echo "[$(date)] SIGINT did not stop it — sending SIGTERM" | tee -a "$MONITOR_LOG"
    kill -TERM "$MAIN_PID"; sleep 10
    if kill -0 "$MAIN_PID" 2>/dev/null; then
      echo "[$(date)] Still alive — sending SIGKILL" | tee -a "$MONITOR_LOG"
      kill -9 "$MAIN_PID"; sleep 3
    fi
  fi
fi

# --- Verify the TRAINER (python) is really gone, not just the wrapper ---
# This catches the false-success case where we signalled the wrong PID.
REMAINING_TRAINER=$(ps -eo pid,comm,args | awk '$2 ~ /^python/ && /train\.py --config/ && !/forkserver/ && !/import sys/ {print $1}')
if [ -n "$REMAINING_TRAINER" ]; then
  echo "[$(date)] !! Trainer still alive after stop attempt: $REMAINING_TRAINER — force killing" | tee -a "$MONITOR_LOG"
  echo "$REMAINING_TRAINER" | xargs kill -9 2>/dev/null
  sleep 3
fi

# --- Clean up any orphaned DataLoader forkserver workers ---
ORPHANS=$(ps -eo pid,args | grep "multiprocessing.forkserver" | grep "train.py" | grep -v grep | awk '{print $1}')
if [ -n "$ORPHANS" ]; then
  echo "[$(date)] Cleaning up orphaned workers: $ORPHANS" | tee -a "$MONITOR_LOG"
  echo "$ORPHANS" | xargs kill -9 2>/dev/null
  sleep 2
fi

# --- Final verification ---
echo "" | tee -a "$MONITOR_LOG"
echo "[$(date)] ===== FINAL STATE =====" | tee -a "$MONITOR_LOG"
echo "[$(date)] Checkpoint files (last 5):" | tee -a "$MONITOR_LOG"
ls -la checkpoints/step_*.pt 2>&1 | tail -5 | tee -a "$MONITOR_LOG"
echo "[$(date)] Last eval points in log:" | tee -a "$MONITOR_LOG"
grep -E "\[VALID\]|\[AUTOCOMPLETE\]|Saved:" "$LOG" | tail -8 | tee -a "$MONITOR_LOG"
echo "[$(date)] GPU state:" | tee -a "$MONITOR_LOG"
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader 2>&1 | tee -a "$MONITOR_LOG"
echo "[$(date)] Remaining train.py processes (should be none):" | tee -a "$MONITOR_LOG"
ps -eo pid,args | grep "[t]rain.py" | tee -a "$MONITOR_LOG"
echo "[$(date)] W&B finish lines in log:" | tee -a "$MONITOR_LOG"
grep -iE "wandb.*finish|wandb.*sync|wandb.*exit" "$LOG" | tail -5 | tee -a "$MONITOR_LOG"
echo "[$(date)] Stop monitor complete." | tee -a "$MONITOR_LOG"
