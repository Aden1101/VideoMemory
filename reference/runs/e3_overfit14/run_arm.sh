#!/bin/bash
# Run one eval arm. args: $1=config.yaml  $2=logfile  $3=CVD  $4=master_port  $5(optional)=MEM_ZERO_COLD_FRAMES
CFG="$1"; LOG="$2"; CVD="$3"; PORT="$4"; MZCF="${5:-}"
PYBIN=/shared/ssd_14T/home/adenelmi/miniconda3/envs/longlive/bin
cd /shared/ssd_14T/home/adenelmi/Anchor-Forcing
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="$CVD"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
[ -n "$MZCF" ] && export MEM_ZERO_COLD_FRAMES="$MZCF"
echo "===== arm start $(date '+%H:%M:%S') cfg=$CFG cvd=$CVD port=$PORT MZCF=${MZCF:-none} =====" >> "$LOG"
"$PYBIN/torchrun" --nproc_per_node=1 --master_port="$PORT" inference/inference.py --config_path "$CFG" >> "$LOG" 2>&1
echo "===== arm done $(date '+%H:%M:%S') rc=$? =====" >> "$LOG"
