#!/bin/bash
# Overfit-E1 smoke + calibration pass: E1 objective (id + tv + hygiene) on the 8 E0 dynamic prompts.
# First goal = plumbing survives (id_loss computes, no crash) + read e1_latent_motion to calibrate tv floor.
PY=/shared/ssd_14T/home/adenelmi/miniconda3/envs/longlive/bin/python
BASE=/shared/ssd_14T/home/adenelmi/Anchor-Forcing
CFG=$BASE/reference/runs/e1_identity/configs/e1_overfit_v1.yaml
LOG=$BASE/reference/runs/e1_identity/train_overfit.log
CVD="${1:-0,5}"          # default 2 free GPUs; pass "0,3,4,5" style to use more
NPROC=$(echo "$CVD" | awk -F, '{print NF}')
cd "$BASE"
: > "$LOG"
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$CVD NCCL_P2P_DISABLE=1 PG_TIMEOUT_MIN=20 \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  setsid nohup "$PY" -m torch.distributed.run --nproc_per_node=$NPROC --max-restarts=0 --master_port=30542 \
  train.py --config_path "$CFG" --no-one-logger >> "$LOG" 2>&1 </dev/null & disown
echo "[$(date '+%H:%M:%S')] launched e1_overfit_v1 on CVD $CVD (nproc=$NPROC) pid $! -> $LOG"
