#!/bin/bash
# ============================================================================================
# SCENARIO MATRIX launcher. Shards by TRIPLE across GPUs, with the occupancy pre-check INSIDE
# the launch command (a GPU that looked free 30 s ago is not a free GPU) and a hard refusal on
# the broken index 6. NEVER more than 4 GPUs.
#
#   bash launch_scen.sh S2,S3,S4 "0,1,2" "0,1"        # scenarios, GPU list, seeds
#   bash launch_scen.sh summary                        # aggregate
#
# Env overrides: ARMS (default = scenarios.DEFAULT_ARMS per scenario), TRIPLES (default 0-11).
# ============================================================================================
set -u
REPO=/shared/ssd_14T/home/adenelmi/Anchor-Forcing
PY=/shared/ssd_14T/home/adenelmi/miniconda3/envs/longlive/bin/python
HERE="$REPO/reference/runs/reid_scen"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "$REPO" || exit 1
mkdir -p "$HERE/logs"

SCENS="${1:-S2,S3,S4}"
GPULIST="${2:-0,1,3,4}"
SEEDS="${3:-0,1}"
TRIPLES="${TRIPLES:-0,1,2,3,4,5,6,7,8,9,10,11}"

if [ "$SCENS" = "summary" ]; then
  exec "$PY" "$HERE/run_reid_scen.py" --mode summary --scen "${2:-S2,S3,S4}" \
       --triples "$TRIPLES" --seeds "${3:-0,1}"
fi

IFS=',' read -r -a GPUS <<< "$GPULIST"
IFS=',' read -r -a TRI <<< "$TRIPLES"
if [ "${#GPUS[@]}" -gt 4 ]; then echo "REFUSE: ${#GPUS[@]} GPUs > 4 (ask Aden first)."; exit 1; fi

# ---- pre-check every GPU AT LAUNCH TIME ----
for g in "${GPUS[@]}"; do
  if [ "$g" = "6" ]; then echo "REFUSE: GPU 6 is BROKEN."; exit 1; fi
  FREE=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$g")
  echo "GPU $g free=${FREE}MiB"
  if [ "$FREE" -lt 25000 ]; then echo "REFUSE: GPU $g has ${FREE}MiB free (< 25000)."; exit 1; fi
done

# ---- round-robin the triples over the shards ----
NG=${#GPUS[@]}
for i in "${!GPUS[@]}"; do
  SHARD=""
  for j in "${!TRI[@]}"; do
    if [ $(( j % NG )) -eq "$i" ]; then SHARD="${SHARD:+$SHARD,}${TRI[$j]}"; fi
  done
  [ -z "$SHARD" ] && continue
  g="${GPUS[$i]}"
  CUDA_VISIBLE_DEVICES=$g nohup "$PY" "$HERE/run_reid_scen.py" \
      --mode run --scen "$SCENS" --triples "$SHARD" --seeds "$SEEDS" \
      ${ARMS:+--arms "$ARMS"} \
      > "$HERE/logs/scen_gpu${g}.log" 2>&1 &
  echo "shard triples=$SHARD scen=$SCENS -> GPU $g (pid $!)"
done
wait
echo "ALL SHARDS DONE"
"$PY" "$HERE/run_reid_scen.py" --mode summary --scen "$SCENS" --triples "$TRIPLES" --seeds "$SEEDS"
