#!/bin/bash
# Wait for >=3 clean GPUs (<500MiB, exclude broken idx6), then launch the clean E1 re-run (e1_rerun2).
cd /shared/ssd_14T/home/adenelmi/Anchor-Forcing
PY=/shared/ssd_14T/home/adenelmi/miniconda3/envs/longlive/bin/python
LOG=reference/runs/e1_rerun/train2.log
MARK=reference/runs/e1_rerun/LAUNCHED2.txt
export CUDA_DEVICE_ORDER=PCI_BUS_ID
while true; do
  CLEAN=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits 2>/dev/null \
          | awk -F', ' '$1!=6 && ($2+0)<500 {print $1}')
  N=$(printf '%s\n' "$CLEAN" | grep -c .)
  if [ "$N" -ge 3 ]; then
    CVD=$(printf '%s\n' "$CLEAN" | head -3 | paste -sd,)
    echo "$(date '+%F %T') launching e1_rerun2 on CVD $CVD" | tee "$MARK"
    : > "$LOG"
    CUDA_VISIBLE_DEVICES=$CVD NCCL_P2P_DISABLE=1 PG_TIMEOUT_MIN=20 \
      PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
      "$PY" -m torch.distributed.run --nproc_per_node=3 --max-restarts=0 --master_port=30811 \
      train.py --config_path reference/runs/e1_rerun/e1_rerun2.yaml --no-one-logger >> "$LOG" 2>&1
    echo "$(date '+%F %T') training exited code $?" >> "$MARK"
    break
  fi
  sleep 45
done
