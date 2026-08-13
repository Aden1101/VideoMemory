#!/bin/bash
PY=/shared/ssd_14T/home/adenelmi/miniconda3/envs/longlive/bin/python
BASE=/shared/ssd_14T/home/adenelmi/Anchor-Forcing
CFG=$BASE/reference/runs/e3_overfit14/configs/train_overfit14.yaml
LOG=$BASE/reference/runs/e3_overfit14/train.log
cd "$BASE"
: > "$LOG"
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=3,4,5 NCCL_P2P_DISABLE=1 PG_TIMEOUT_MIN=20 \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  setsid nohup "$PY" -m torch.distributed.run --nproc_per_node=3 --max-restarts=0 --master_port=30482 \
  train.py --config_path "$CFG" --no-one-logger >> "$LOG" 2>&1 </dev/null & disown
echo "[$(date '+%H:%M:%S')] launched overfit14 (7 no-switch + 7 switch) on CVD 3,4,5 (pid $!)"
