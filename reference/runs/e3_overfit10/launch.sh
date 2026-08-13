#!/bin/bash
# Pause e3_real3 and launch the 10-prompt overfit on the freed GPUs 3/4/5.
# (patterns live inside this file, so pkill won't self-match the launcher.)
set -u
PY=/shared/ssd_14T/home/adenelmi/miniconda3/envs/longlive/bin/python
BASE=/shared/ssd_14T/home/adenelmi/Anchor-Forcing
CFG=$BASE/reference/runs/e3_overfit10/configs/train_overfit10.yaml
LOG=$BASE/reference/runs/e3_overfit10/train.log
cd "$BASE"
echo "[$(date '+%H:%M:%S')] stopping guardian watcher"
pkill -9 -f "gpu_accumulate_watcher.sh" 2>/dev/null
sleep 2
echo "[$(date '+%H:%M:%S')] pausing e3_real3 (checkpoints kept, resumable via auto_resume)"
pkill -9 -f "_real3.yaml" 2>/dev/null
pkill -9 -f "gpu_holder.py" 2>/dev/null
sleep 4
cp -f "$BASE/reference/runs/e3_real3_gate01_scal3r/eval10_prompts.txt" "$BASE/reference/runs/e3_overfit10/prompts10.txt"
# wait for smi 3/4/5 to free
m3=9999; m4=9999; m5=9999
for k in $(seq 1 40); do
  m3=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 3 2>/dev/null | tr -dc 0-9)
  m4=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 4 2>/dev/null | tr -dc 0-9)
  m5=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 5 2>/dev/null | tr -dc 0-9)
  { [ "${m3:-9999}" -lt 3000 ] && [ "${m4:-9999}" -lt 3000 ] && [ "${m5:-9999}" -lt 3000 ]; } && break
  sleep 3
done
echo "[$(date '+%H:%M:%S')] GPUs after pause: smi3=$m3 smi4=$m4 smi5=$m5"
: > "$LOG"
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=3,4,5 NCCL_P2P_DISABLE=1 PG_TIMEOUT_MIN=20 \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  setsid nohup "$PY" -m torch.distributed.run --nproc_per_node=3 --max-restarts=0 --master_port=30480 \
  train.py --config_path "$CFG" --no-one-logger >> "$LOG" 2>&1 </dev/null & disown
echo "[$(date '+%H:%M:%S')] launched overfit10 on CVD 3,4,5 (pid $!), log $LOG"
