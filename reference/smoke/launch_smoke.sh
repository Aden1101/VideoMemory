#!/bin/bash
# Smoke test of the audit fixes: a few training steps (exercises id_loss/CRIT-1, snapshot/restore/CRIT-3,
# bptt_window=2, train_write, multi-GPU sync/MAJ-1). Usage: ./launch_smoke.sh "1"  or  ./launch_smoke.sh "1,3"
PY=/shared/ssd_14T/home/adenelmi/miniconda3/envs/longlive/bin/python
BASE=/shared/ssd_14T/home/adenelmi/Anchor-Forcing
CVD="${1:-1}"
NPROC=$(echo "$CVD" | awk -F, '{print NF}')
TAG=$(echo "$CVD" | tr ',' '-')
LOG=$BASE/reference/smoke/train_${TAG}.log
cd "$BASE"; : > "$LOG"
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$CVD NCCL_P2P_DISABLE=1 PG_TIMEOUT_MIN=15 \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  setsid nohup "$PY" -m torch.distributed.run --nproc_per_node=$NPROC --max-restarts=0 --master_port=30733 \
  train.py --config_path "$BASE/reference/smoke/smoke.yaml" --no-one-logger >> "$LOG" 2>&1 </dev/null & disown
echo "[$(date '+%H:%M:%S')] smoke launched CVD $CVD (nproc=$NPROC) pid $! -> $LOG"
