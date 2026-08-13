#!/bin/bash
# c3_stable launch on 4 GPUs. Usage: bash launch_c3.sh "0,1,2,3" 30113
set +e
cd /shared/ssd_14T/home/adenelmi/Anchor-Forcing
GPUS="${1:-0,1,2,3}"
PORT="${2:-30113}"
PY=/shared/ssd_14T/home/adenelmi/miniconda3/envs/longlive/bin/python
CFG=reference/runs/c3_stable/c3_stable.yaml
RUNLOG=reference/c3_stable.log

pkill -9 -f "c3_stable/c3_stable.yaml" 2>/dev/null
sleep 4
: > "$RUNLOG"
rm -rf /shared/ssd_14T_project/adenelmi/c3_stable
mkdir -p /shared/ssd_14T_project/adenelmi/c3_stable

export CUDA_VISIBLE_DEVICES="$GPUS"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export NCCL_P2P_DISABLE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,garbage_collection_threshold:0.8

setsid nohup "$PY" -m torch.distributed.run --nproc_per_node=4 --max-restarts=0 \
  --master_port="$PORT" train.py --config_path "$CFG" --no-one-logger \
  >> "$RUNLOG" 2>&1 < /dev/null &
disown
echo "LAUNCHED gpus=$GPUS port=$PORT pid=$!"
