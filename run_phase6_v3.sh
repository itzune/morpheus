#!/bin/bash
# Phase 6 v3 pipeline: wait for 70/30 data rebuild → stop servers → train
# Resumes from v2 best.pt with 70/30 FIM ratio + 5x EOT loss weighting
set -e
cd /root/morpheus-mamba

echo "========================================"
echo "Phase 6 v3 pipeline"
echo "Started: $(date)"
echo "========================================"

# Step 1: Wait for rebuild to complete
echo "Waiting for data/train_fim_7030.npy rebuild..."
while true; do
  # Check if rebuild screen is still running
  if ! screen -ls 2>/dev/null | grep -q "rebuild"; then
    # Screen exited — check if file was created
    if [ -f data/train_fim_7030.npy ]; then
      echo "[$(date)] Rebuild complete!"
      break
    else
      echo "[$(date)] ERROR: Rebuild screen exited but train_fim_7030.npy not found!"
      exit 1
    fi
  fi
  sleep 30
done

# Step 2: Verify the new dataset
echo ""
echo "========================================"
echo "Verifying data/train_fim_7030.npy"
echo "========================================"
python3 -c "
import numpy as np
a = np.load('data/train_fim_7030.npy', mmap_mode='r')
print(f'Tokens: {len(a):,}')
print(f'Size: {a.nbytes/1e9:.2f} GB')
for name, tid in [('PRE',4000),('SUF',4001),('MID',4002),('EOT',4003),('</s>',2)]:
    print(f'  <{name}>: {int((a==tid).sum()):,}')
"

# Step 3: Stop inference servers (free GPU)
echo ""
echo "========================================"
echo "Stopping inference servers"
echo "========================================"
screen -S llama -X quit 2>/dev/null || true
screen -S demo -X quit 2>/dev/null || true
sleep 3
nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv,noheader

# Step 4: Launch Phase 6 v3 training
echo ""
echo "========================================"
echo "Step 4: Phase 6 v3 training (70/30 FIM + 5x EOT weight)"
echo "Started: $(date)"
echo "========================================"
PYTHONUNBUFFERED=1 WANDB_MODE=disabled python3 -u train.py \
  --config config/phase6_fim_v3.yaml \
  --pretrained checkpoints/phase6/best.pt

echo ""
echo "========================================"
echo "Phase 6 v3 training complete!"
echo "Finished: $(date)"
echo "========================================"
