#!/bin/bash
# Keepalive for run1 (e3_run1_decay): maintain 3 holders + relaunch run1 (auto-resume) if it dies.
PY=/shared/ssd_14T/home/adenelmi/miniconda3/envs/longlive/bin/python
cd /shared/ssd_14T/home/adenelmi/Anchor-Forcing
LOG=/shared/ssd_14T/home/adenelmi/Anchor-Forcing/reference/run1_keepalive.log
declare -A U=( [1]=GPU-7972ddf1-78b9-ff67-c13b-aaaa6f9f5e57 [2]=GPU-16b40df4-c8ef-04d3-e8a5-054f40cf5ec7 [7]=GPU-df09dc6f-c851-343b-7a73-73f3f34b0107 )
port=30270
echo "[$(date '+%m-%d %H:%M:%S')] run1 keepalive up" >> "$LOG"
while true; do
  n=$(pgrep -u adenelmi -fc gpu_holder.py)
  if [ "${n:-0}" -lt 3 ]; then
    pkill -9 -f gpu_holder.py 2>/dev/null; sleep 2
    for g in 1 2 7; do CUDA_VISIBLE_DEVICES=${U[$g]} setsid nohup "$PY" /tmp/gpu_holder.py >> /tmp/holder_$g.log 2>&1 </dev/null & disown; done
    echo "[$(date '+%m-%d %H:%M:%S')] (re)launched 3 holders" >> "$LOG"
  fi
  if ! pgrep -u adenelmi -f "_run1_decay.yaml" >/dev/null 2>&1; then
    echo "[$(date '+%m-%d %H:%M:%S')] run1 DOWN -> waiting GPU release" >> "$LOG"
    for i in $(seq 1 80); do m=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 1 2>/dev/null||echo 9999); [ "$m" -lt 2500 ] && break; sleep 5; done
    port=$((port+2)); [ $port -gt 30330 ] && port=30270
    CUDA_VISIBLE_DEVICES=1,2,6 NCCL_P2P_DISABLE=1 PG_TIMEOUT_MIN=20 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
      setsid nohup "$PY" -m torch.distributed.run --nproc_per_node=3 --max-restarts=0 --master_port=$port \
      train.py --config_path /tmp/_run1_decay.yaml --no-one-logger >> reference/e3_run1.log 2>&1 </dev/null & disown
    echo "[$(date '+%m-%d %H:%M:%S')] relaunched run1 (auto-resume) port $port" >> "$LOG"
    sleep 200
  fi
  sleep 30
done
