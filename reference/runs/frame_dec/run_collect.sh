#!/bin/bash
# frame_dec pair collection, sharded one-process-per-GPU (pattern of reid_probe_l1/run_all.sh).
#   bash run_collect.sh              # launch shards on GPUS (default 0 1 2 3), logs in logs/
#   GPUS="0 1" N=100 POS=8 bash run_collect.sh
#   bash run_collect.sh manifest     # merge shard manifests once done
# NOTE: default N=400 x POS=8 projects ~544 GB > the 150 GB ceiling -> every shard will print
# the projection after its first video and ABORT. Pick N/POS so that N*POS*0.17GB <= 150GB
# (e.g. N=400 POS=2, or N=110 POS=8). GPU 6 is BROKEN; never use it.
set -u
REPO=/shared/ssd_14T/home/adenelmi/Anchor-Forcing
PY=/shared/ssd_14T/home/adenelmi/miniconda3/envs/longlive/bin/python
HERE="$REPO/reference/runs/frame_dec"

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "$REPO"

if [ "${1:-}" = "manifest" ]; then
  exec "$PY" "$HERE/collect_pairs.py" --mode manifest
fi

N="${N:-400}"
POS="${POS:-8}"
RESUME="${RESUME:-}"           # RESUME=--resume to continue existing shards
read -r -a GPU_ARR <<< "${GPUS:-0 1 2 3}"
NSHARD=${#GPU_ARR[@]}

mkdir -p "$HERE/logs"
for i in "${!GPU_ARR[@]}"; do
  g="${GPU_ARR[$i]}"
  if [ "$g" = "6" ]; then echo "REFUSE: GPU 6 is BROKEN."; exit 1; fi
  CUDA_VISIBLE_DEVICES=$g nohup "$PY" "$HERE/collect_pairs.py" \
      --mode run --shard "$i/$NSHARD" --n "$N" --positions "$POS" $RESUME \
      > "$HERE/logs/shard_gpu${g}.log" 2>&1 &
  echo "shard $i/$NSHARD -> GPU $g (pid $!)"
done
wait
echo "ALL SHARDS DONE"
