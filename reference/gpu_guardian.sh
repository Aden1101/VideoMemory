#!/bin/bash
# GPU guardian for gpu1/2/7: NEVER lose a GPU.
#   (1) keep 3 holders alive (GPU never drops to 0 / grabbable);
#   (2) as soon as one of our GPUs is idle (eval/training done) AND no eval-script is still using it,
#       launch a read-only memory training run on it (productive + wandb), relaunch if it dies.
# Eval-aware: will NOT fill a GPU while its eval script is still running (no collision).
PY=/shared/ssd_14T/home/adenelmi/miniconda3/envs/longlive/bin/python
cd /shared/ssd_14T/home/adenelmi/Anchor-Forcing
LOG=/shared/ssd_14T/home/adenelmi/Anchor-Forcing/reference/gpu_guardian.log
declare -A U=( [1]=GPU-7972ddf1-78b9-ff67-c13b-aaaa6f9f5e57 [2]=GPU-16b40df4-c8ef-04d3-e8a5-054f40cf5ec7 [7]=GPU-df09dc6f-c851-343b-7a73-73f3f34b0107 )
# which eval script currently OWNS each GPU (don't fill while it runs)
# pgrep uses ERE: alternation is | (not \|). Any of these scripts "owns" a GPU (incl. ad-hoc diagnostics).
declare -A EV=( [1]="e2_eval_horizon|e2_causal_test|e2_perframe|e2_gate" [2]="e2_eval_parallel|e2_eval_horizon|e2_causal_test|e2_perframe|e2_gate" [7]="e2_eval_mem960|e2_causal_test|e2_perframe|e2_gate" )
declare -A PORT=( [1]=31010 [2]=31020 [7]=31070 )
echo "[$(date '+%m-%d %H:%M:%S')] guardian up" >> "$LOG"
while true; do
  # (1) keep exactly 3 holders alive
  n=$(pgrep -u adenelmi -fc gpu_holder.py)
  if [ "${n:-0}" -lt 3 ]; then
    pkill -9 -f gpu_holder.py 2>/dev/null; sleep 2
    for g in 1 2 7; do CUDA_VISIBLE_DEVICES=${U[$g]} setsid nohup "$PY" /tmp/gpu_holder.py >> /tmp/holder_$g.log 2>&1 </dev/null & disown; done
    echo "[$(date '+%m-%d %H:%M:%S')] (re)launched 3 holders" >> "$LOG"
  fi
  # (2) fill any idle OUR-gpu with a read-only run (unless its eval script still runs, or a filler is up)
  for g in 1 2 7; do
    pgrep -u adenelmi -f "${EV[$g]}" >/dev/null 2>&1 && continue            # eval still owns this GPU
    pgrep -u adenelmi -f "_ro_fill_g${g}.yaml" >/dev/null 2>&1 && continue   # filler already running
    mem=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $g 2>/dev/null || echo 99999)
    # cooldown: don't relaunch within ~90s of a prior launch (covers the model-load window where mem<2000)
    cd_recent=$(find /tmp/.ro_cd_g${g} -mmin -1.5 2>/dev/null)
    if [ "${mem:-99999}" -lt 2000 ] && [ -z "$cd_recent" ]; then
      touch /tmp/.ro_cd_g${g}
      : > /tmp/ro_fill_g${g}.log
      CUDA_VISIBLE_DEVICES=${U[$g]} NCCL_P2P_DISABLE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
        setsid nohup "$PY" -m torch.distributed.run --nproc_per_node=1 --max-restarts=0 --master_port=${PORT[$g]} \
        train.py --config_path /tmp/_ro_fill_g${g}.yaml --no-one-logger --no-auto-resume \
        >> /tmp/ro_fill_g${g}.log 2>&1 </dev/null & disown
      echo "[$(date '+%m-%d %H:%M:%S')] launched read-only filler on gpu$g (idle ${mem}MB)" >> "$LOG"
    fi
  done
  sleep 30
done
