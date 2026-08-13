#!/bin/bash
# run1b keepalive: 3 holders + relaunch run1b (auto-resume) if dead. FIXED: checks ALL 3 GPUs free + cooldown.
PY=/shared/ssd_14T/home/adenelmi/miniconda3/envs/longlive/bin/python
cd /shared/ssd_14T/home/adenelmi/Anchor-Forcing
LOG=/shared/ssd_14T/home/adenelmi/Anchor-Forcing/reference/run1b_keepalive.log
declare -A U=( [1]=GPU-7972ddf1-78b9-ff67-c13b-aaaa6f9f5e57 [2]=GPU-16b40df4-c8ef-04d3-e8a5-054f40cf5ec7 [7]=GPU-df09dc6f-c851-343b-7a73-73f3f34b0107 )
port=30350
echo "[$(date '+%m-%d %H:%M:%S')] run1b keepalive up" >> "$LOG"
while true; do
  n=$(pgrep -u adenelmi -fc gpu_holder.py)
  if [ "${n:-0}" -lt 3 ]; then
    pkill -9 -f gpu_holder.py 2>/dev/null; sleep 2
    for g in 1 2 7; do CUDA_VISIBLE_DEVICES=${U[$g]} setsid nohup "$PY" /tmp/gpu_holder.py >> /tmp/holder_$g.log 2>&1 </dev/null & disown; done
    echo "[$(date '+%m-%d %H:%M:%S')] (re)launched 3 holders" >> "$LOG"
  fi
  if ! pgrep -u adenelmi -f "_run1b_decay.yaml" >/dev/null 2>&1; then
    echo "[$(date '+%m-%d %H:%M:%S')] run1b DOWN -> waiting ALL 3 GPUs to free" >> "$LOG"
    for k in $(seq 1 100); do
      m1=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 1 2>/dev/null||echo 9999)
      m2=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 2 2>/dev/null||echo 9999)
      m7=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 7 2>/dev/null||echo 9999)
      if [ "$m1" -lt 2500 ] && [ "$m2" -lt 2500 ] && [ "$m7" -lt 2500 ]; then break; fi
      sleep 5
    done
    sleep 10
    port=$((port+2)); [ $port -gt 30420 ] && port=30350
    CUDA_VISIBLE_DEVICES=1,2,6 NCCL_P2P_DISABLE=1 PG_TIMEOUT_MIN=20 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
      setsid nohup "$PY" -m torch.distributed.run --nproc_per_node=3 --max-restarts=0 --master_port=$port \
      train.py --config_path /tmp/_run1b_decay.yaml --no-one-logger >> reference/e3_run1b.log 2>&1 </dev/null & disown
    echo "[$(date '+%m-%d %H:%M:%S')] relaunched run1b (auto-resume) port $port" >> "$LOG"
    sleep 300
  fi
  sleep 30
done
