#!/bin/bash
# Kill orphan train.py workers (from previously killed runs) then relaunch overfit14 clean.
PY=/shared/ssd_14T/home/adenelmi/miniconda3/envs/longlive/bin/python
BASE=/shared/ssd_14T/home/adenelmi/Anchor-Forcing
CFG=$BASE/reference/runs/e3_overfit14/configs/train_overfit14.yaml
LOG=$BASE/reference/runs/e3_overfit14/train.log
cd "$BASE"
echo "[$(date '+%H:%M:%S')] killing lingering train.py / torchrun workers (orphans + crashed overfit14)"
pkill -9 -f "torch.distributed.run" 2>/dev/null
pkill -9 -f "train.py --config_path" 2>/dev/null
sleep 5
m3=9999; m4=9999; m5=9999
for k in $(seq 1 40); do
  m3=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 3 2>/dev/null | tr -dc 0-9)
  m4=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 4 2>/dev/null | tr -dc 0-9)
  m5=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 5 2>/dev/null | tr -dc 0-9)
  { [ "${m3:-9999}" -lt 2000 ] && [ "${m4:-9999}" -lt 2000 ] && [ "${m5:-9999}" -lt 2000 ]; } && break
  sleep 3
done
echo "[$(date '+%H:%M:%S')] smi after cleanup: 3=$m3 4=$m4 5=$m5"
if [ "${m3:-9999}" -ge 2000 ] || [ "${m4:-9999}" -ge 2000 ] || [ "${m5:-9999}" -ge 2000 ]; then
  echo "!! GPUs still busy, NOT relaunching (investigate)"; exit 1
fi
: > "$LOG"
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=3,4,5 NCCL_P2P_DISABLE=1 PG_TIMEOUT_MIN=20 \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  setsid nohup "$PY" -m torch.distributed.run --nproc_per_node=3 --max-restarts=0 --master_port=30484 \
  train.py --config_path "$CFG" --no-one-logger >> "$LOG" 2>&1 </dev/null & disown
echo "[$(date '+%H:%M:%S')] relaunched overfit14 clean on CVD 3,4,5 (pid $!)"
