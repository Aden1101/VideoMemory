#!/bin/bash
# L1 re-ID revisit probe, sharded across GPUs 0-3 (2 pairs x 3 seeds x 3 arms = 18 videos/shard).
#   bash run_all.sh            # launch the 4 shards in the background, logs in logs/
#   bash run_all.sh summary    # aggregate stats once shards are done
set -u
REPO=/shared/ssd_14T/home/adenelmi/Anchor-Forcing
PY=/shared/ssd_14T/home/adenelmi/miniconda3/envs/longlive/bin/python
HERE="$REPO/reference/runs/reid_probe_l1"

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "$REPO"

if [ "${1:-}" = "summary" ]; then
  exec "$PY" "$HERE/run_reid_probe.py" --mode summary --pairs 0,1,2,3,4,5,6,7 --seeds 0,1,2
fi

mkdir -p "$HERE/logs"
GPUS=(0 1 2 3)
SHARDS=("0,1" "2,3" "4,5" "6,7")
for i in "${!GPUS[@]}"; do
  g="${GPUS[$i]}"
  if [ "$g" = "6" ]; then echo "REFUSE: GPU 6 is BROKEN."; exit 1; fi
  CUDA_VISIBLE_DEVICES=$g nohup "$PY" "$HERE/run_reid_probe.py" \
      --mode run --pairs "${SHARDS[$i]}" --seeds 0,1,2 --arms "${ARMS:-memon,memzero,stock}" \
      > "$HERE/logs/shard_gpu${g}.log" 2>&1 &
  echo "shard pairs=${SHARDS[$i]} -> GPU $g (pid $!)"
done
wait
echo "ALL SHARDS DONE"
