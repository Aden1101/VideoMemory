#!/bin/bash
# Launch ONE shard of the arbiter-battery campaign (memory2_probes_registry.md).
#   ./launch_probes.sh <gpu_index> <pairs> [tag]
# The GPU pre-check is INSIDE this script (house rule): the shard refuses to start unless that
# exact GPU has > 25 GB free AT THIS INSTANT, and index 6 is never used (broken in smi).
set -u
REPO=/shared/ssd_14T/home/adenelmi/Anchor-Forcing
PY=/shared/ssd_14T/home/adenelmi/miniconda3/envs/longlive/bin/python
GPU=${1:?usage: launch_probes.sh <gpu> <pairs> [tag]}
PAIRS=${2:?usage: launch_probes.sh <gpu> <pairs> [tag]}
TAG=${3:-shard$GPU}
ARMS=${ARMS:-oracle1thru,oracle1blur,memon20,oracle1twin}
LOGS=$REPO/reference/runs/mem2/logs
mkdir -p "$LOGS"

export CUDA_DEVICE_ORDER=PCI_BUS_ID
if [ "$GPU" = "6" ]; then echo "GPU 6 is BROKEN in smi -- refusing"; exit 1; fi
FREE=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$GPU")
if [ "$FREE" -lt 25000 ]; then
  echo "GPU $GPU has only ${FREE} MiB free (< 25000) -- refusing to launch $TAG"; exit 1
fi
echo "GPU $GPU: ${FREE} MiB free -> launching $TAG (pairs $PAIRS, arms $ARMS)"

cd "$REPO" || exit 1
CUDA_VISIBLE_DEVICES=$GPU nohup "$PY" reference/runs/reid_probe_l1/run_reid_probe.py \
  --mode run --pairs "$PAIRS" --seeds 0,1,2 --arms "$ARMS" \
  > "$LOGS/probes_$TAG.log" 2>&1 &
echo $! > "$LOGS/probes_$TAG.pid"
echo "pid $(cat "$LOGS/probes_$TAG.pid") -> $LOGS/probes_$TAG.log"
