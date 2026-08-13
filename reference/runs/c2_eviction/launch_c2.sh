#!/bin/bash
# c2_eviction launch on 4 GPUs. Usage: bash launch_c2.sh "0,1,2,3" 30112
set +e
cd /shared/ssd_14T/home/adenelmi/Anchor-Forcing
GPUS="${1:-0,1,2,3}"
PORT="${2:-30112}"
PY=/shared/ssd_14T/home/adenelmi/miniconda3/envs/longlive/bin/python
CFG=reference/runs/c2_eviction/c2_eviction.yaml
RUNLOG=reference/c2_eviction.log

pkill -9 -f "c2_eviction/c2_eviction.yaml" 2>/dev/null
sleep 4
: > "$RUNLOG"
rm -rf /shared/ssd_14T_project/adenelmi/c2_eviction
mkdir -p /shared/ssd_14T_project/adenelmi/c2_eviction

export CUDA_VISIBLE_DEVICES="$GPUS"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export NCCL_P2P_DISABLE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,garbage_collection_threshold:0.8

setsid nohup "$PY" -m torch.distributed.run --nproc_per_node=4 --max-restarts=0 \
  --master_port="$PORT" train.py --config_path "$CFG" --no-one-logger \
  >> "$RUNLOG" 2>&1 < /dev/null &
disown
echo "LAUNCHED gpus=$GPUS port=$PORT pid=$!"
