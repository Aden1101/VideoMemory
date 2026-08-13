#!/bin/bash
# Keepalive for overfit14: keep it running (auto_resume) until step >= DONE, then exit.
# Handles the max_iters 400->1000 extension (clean stop at 400 -> relaunch continues) AND crashes.
# Surgical: only touches processes whose cmdline contains "train_overfit14" (no other run affected).
PY=/shared/ssd_14T/home/adenelmi/miniconda3/envs/longlive/bin/python
BASE=/shared/ssd_14T/home/adenelmi/Anchor-Forcing
CFG=$BASE/reference/runs/e3_overfit14/configs/train_overfit14.yaml
LOG=$BASE/reference/runs/e3_overfit14/train.log
KLOG=$BASE/reference/runs/e3_overfit14/keepalive.log
DONE=1000
cd "$BASE"
echo "[$(date '+%m-%d %H:%M:%S')] keepalive up (target step $DONE)" >> "$KLOG"
port=30486
while true; do
  if ! pgrep -u adenelmi -f "train_overfit14" >/dev/null 2>&1; then
    s=$(grep -oE "mem stats step=[0-9]+" "$LOG" 2>/dev/null | tail -1 | tr -dc '0-9')
    if [ -n "$s" ] && [ "$s" -ge $((DONE-10)) ] 2>/dev/null; then
      echo "[$(date '+%m-%d %H:%M:%S')] reached step $s (>=$DONE) -> DONE, exiting" >> "$KLOG"; exit 0
    fi
    echo "[$(date '+%m-%d %H:%M:%S')] overfit14 down at step ${s:-?} -> relaunch (auto_resume)" >> "$KLOG"
    pkill -9 -f "train_overfit14" 2>/dev/null; sleep 5   # clean any orphan workers
    port=$((port+2)); [ $port -gt 30510 ] && port=30486
    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=3,4,5 NCCL_P2P_DISABLE=1 PG_TIMEOUT_MIN=20 \
      PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
      setsid nohup "$PY" -m torch.distributed.run --nproc_per_node=3 --max-restarts=0 --master_port=$port \
      train.py --config_path "$CFG" --no-one-logger >> "$LOG" 2>&1 </dev/null & disown
    echo "[$(date '+%m-%d %H:%M:%S')] relaunched (port $port)" >> "$KLOG"
    sleep 150
  fi
  sleep 20
done
