#!/bin/bash
# e4_antiflicker launch on 4 GPUs. Usage: ./launch_e4.sh "0,1,3,5" 30097
set +e
cd /shared/ssd_14T/home/adenelmi/Anchor-Forcing
GPUS="${1:-0,1,3,5}"
PORT="${2:-30097}"
PY=/shared/ssd_14T/home/adenelmi/miniconda3/envs/longlive/bin/python
CFG=reference/runs/e4_antiflicker/e4_antiflicker.yaml
RUNLOG=reference/e4_antiflicker.log

pkill -9 -f "e4_antiflicker/e4_antiflicker.yaml" 2>/dev/null
sleep 4
: > "$RUNLOG"
rm -rf /shared/ssd_14T_project/adenelmi/e4_antiflicker
mkdir -p /shared/ssd_14T_project/adenelmi/e4_antiflicker

export CUDA_VISIBLE_DEVICES="$GPUS"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export NCCL_P2P_DISABLE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,garbage_collection_threshold:0.8

setsid nohup "$PY" -m torch.distributed.run --nproc_per_node=4 --max-restarts=0 \
  --master_port="$PORT" train.py --config_path "$CFG" --no-one-logger \
  >> "$RUNLOG" 2>&1 < /dev/null &
disown
echo "LAUNCHED gpus=$GPUS port=$PORT pid=$!"
