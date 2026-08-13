#!/bin/bash
# Overnight keepalive: ensure a read-only training run always holds gpu1/2/7 (never idle).
# Read-only is stable (no write-path leak) and trains the gate + read-path (Phase-1).
PY=/shared/ssd_14T/home/adenelmi/miniconda3/envs/longlive/bin/python
REPO=/shared/ssd_14T/home/adenelmi/Anchor-Forcing
LOG=/tmp/keepalive.log
PORT=30095
echo "[$(date +%H:%M:%S)] keepalive up (holds gpu1/2/7 with read-only)" >> "$LOG"
while true; do
  if ! pgrep -u adenelmi -f "_read1L.yaml" >/dev/null 2>&1; then
    PORT=$((PORT+2)); [ $PORT -gt 30180 ] && PORT=30095
    : > /tmp/read3gpu.log
    cd "$REPO"
    CUDA_VISIBLE_DEVICES=1,2,6 NCCL_P2P_DISABLE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
      setsid nohup "$PY" -m torch.distributed.run --nproc_per_node=3 --max-restarts=0 \
      --master_port=$PORT train.py --config_path /tmp/_read1L.yaml --no-one-logger --no-auto-resume \
      >> /tmp/read3gpu.log 2>&1 < /dev/null &
    echo "[$(date +%H:%M:%S)] read-only was DOWN -> relaunched on port $PORT" >> "$LOG"
    sleep 180   # let it load + run before re-checking
  fi
  sleep 30
done
