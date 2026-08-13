#!/bin/bash
# Re-ID revisit probe launcher.
#   bash run_probe.sh [GPU] [mode] [pairs] [extra args...]
#   e.g. bash run_probe.sh 2 selftest
#        bash run_probe.sh 2 run 0
#        bash run_probe.sh 2 score 0,1,2
set -u
GPU="${1:-2}"
if [ "$GPU" = "6" ]; then echo "REFUSE: GPU 6 is BROKEN."; exit 1; fi
MODE="${2:-run}"
PAIRS="${3:-0}"

REPO=/shared/ssd_14T/home/adenelmi/Anchor-Forcing
PY=/shared/ssd_14T/home/adenelmi/miniconda3/envs/longlive/bin/python

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=$GPU
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd "$REPO"
exec "$PY" reference/runs/reid_probe/run_reid_probe.py --mode "$MODE" --pairs "$PAIRS" "${@:4}"
