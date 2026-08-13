#!/bin/bash
# Keepalive for e3_realk2 (real K=2 cross-chunk BPTT, DMD-pure, -> 3000). Holders only when DOWN.
PY=/shared/ssd_14T/home/adenelmi/miniconda3/envs/longlive/bin/python
HOLDER=/shared/ssd_14T/home/adenelmi/.claude/jobs/3118b292/tmp/gpu_holder.py
CFG=/shared/ssd_14T/home/adenelmi/.claude/jobs/3118b292/tmp/_realk2.yaml
RUNLOG=/shared/ssd_14T/home/adenelmi/Anchor-Forcing/reference/e3_realk2.log
LOG=/shared/ssd_14T/home/adenelmi/Anchor-Forcing/reference/realk2_keepalive.log
cd /shared/ssd_14T/home/adenelmi/Anchor-Forcing
declare -A U=( [1]=GPU-7972ddf1-78b9-ff67-c13b-aaaa6f9f5e57 \
               [2]=GPU-16b40df4-c8ef-04d3-e8a5-054f40cf5ec7 \
               [7]=GPU-df09dc6f-c851-343b-7a73-73f3f34b0107 )
port=30400
echo "[$(date '+%m-%d %H:%M:%S')] realk2 keepalive up" >> "$LOG"
while true; do
  if ! pgrep -u adenelmi -f "_realk2.yaml" >/dev/null 2>&1; then
    ds=$(grep -oE 'step=[0-9]+' "$RUNLOG" 2>/dev/null | tail -1 | tr -dc '0-9')
    if [ -n "$ds" ] && [ "$ds" -ge 2990 ] 2>/dev/null; then
      echo "[$(date '+%m-%d %H:%M:%S')] realk2 reached $ds (>=2990) -> DONE, exiting" >> "$LOG"; exit 0
    fi
    echo "[$(date '+%m-%d %H:%M:%S')] realk2 DOWN at step ${ds:-?} -> grab + relaunch" >> "$LOG"
    pkill -9 -f gpu_holder.py 2>/dev/null; sleep 2
    for g in 1 2 7; do CUDA_VISIBLE_DEVICES=${U[$g]} setsid nohup "$PY" "$HOLDER" >> /tmp/holder_$g.log 2>&1 </dev/null & disown; done
    for k in $(seq 1 120); do
      m1=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 1 2>/dev/null || echo 9999)
      m2=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 2 2>/dev/null || echo 9999)
      m7=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 7 2>/dev/null || echo 9999)
      if [ "$m1" -lt 2500 ] && [ "$m2" -lt 2500 ] && [ "$m7" -lt 2500 ]; then break; fi
      sleep 5
    done
    pkill -9 -f gpu_holder.py 2>/dev/null; sleep 3
    port=$((port+2)); [ $port -gt 30460 ] && port=30400
    CUDA_VISIBLE_DEVICES=1,2,6 NCCL_P2P_DISABLE=1 PG_TIMEOUT_MIN=20 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
      setsid nohup "$PY" -m torch.distributed.run --nproc_per_node=3 --max-restarts=0 --master_port=$port \
      train.py --config_path "$CFG" --no-one-logger >> "$RUNLOG" 2>&1 </dev/null & disown
    echo "[$(date '+%m-%d %H:%M:%S')] relaunched realk2 (auto-resume) port $port" >> "$LOG"
    sleep 300
  fi
  sleep 20
done
