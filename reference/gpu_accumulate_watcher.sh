#!/bin/bash
# GPU accumulate watcher + real-run guardian.
# Phase 1 (accumulate): grab working GPUs as they free (1-GPU read-only holds), reach NEED=3.
# Phase 2 (launch): kill holds, launch the REAL 3-GPU run.
# Phase 3 (guardian): watch the real run. If it dies:
#   - progressing (transient hiccup) -> auto-resume relaunch,
#   - stuck / startup crash -> GIVE UP: kill it, hold the 3 GPUs read-only forever (for Aden), preserve the crash log.
# Broken GPU ab1015a0 (smi idx6) EXCLUDED. Detection in UUID space; launch with numeric CVD + PCI_BUS_ID.
set -u
PY=/shared/ssd_14T/home/adenelmi/miniconda3/envs/longlive/bin/python
JT=/shared/ssd_14T/home/adenelmi/.claude/jobs/3118b292/tmp
RO_CFG=$JT/_ro1.yaml
HOLDER=$JT/gpu_holder.py
REAL_CFG=$JT/_real3.yaml
BASE=/shared/ssd_14T/home/adenelmi/Anchor-Forcing
LOG=$BASE/reference/gpu_watcher.log
REALLOG=$BASE/reference/e3_real3.log
HELD=$JT/held
NEED=3
FREE_THRESH=2500
DONE_STEP=2990
cd "$BASE"
mkdir -p "$HELD"; rm -f "$HELD"/*

declare -A CVD=( [c916c446]=0 [7972ddf1]=1 [16b40df4]=2 [07f5cef9]=3 [7d78717a]=4 [c4ff2462]=5 [df09dc6f]=6 )
log(){ echo "[$(date '+%m-%d %H:%M:%S')] $*" >> "$LOG"; }
laststep(){ grep -oE 'step=[0-9]+' "$REALLOG" 2>/dev/null | tail -1 | tr -dc '0-9'; }
real_alive(){ pgrep -u adenelmi -f "_real3.yaml" >/dev/null 2>&1; }
ro_alive_on(){ local c=$1; local p; p=$(cat "$HELD/$c" 2>/dev/null); [ -n "$p" ] && kill -0 "$p" 2>/dev/null; }

launch_ro(){   # $1=cvd  -- robust holder (grabs up to ~36GB, catches OOM, never crash-loops)
  local c=$1
  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$c PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    setsid nohup "$PY" "$HOLDER" >> "$BASE/reference/ro_cvd${c}.log" 2>&1 </dev/null & disown
  echo $! > "$HELD/$c"
  log "hold up on CVD=$c (robust holder ~36GB, pid $!)"
}

RPORT=30470
launch_real(){ # $1,$2,$3 = cvds ; appends to REALLOG (never truncate -> keep crash tracebacks)
  local cs="$1,$2,$3"
  pkill -9 -f "gpu_holder.py" 2>/dev/null; sleep 3
  RPORT=$((RPORT+2)); [ $RPORT -gt 30498 ] && RPORT=30470
  echo "===== launch/resume real run $(date '+%m-%d %H:%M:%S') on CVD=$cs port $RPORT =====" >> "$REALLOG"
  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$cs NCCL_P2P_DISABLE=1 PG_TIMEOUT_MIN=20 \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    setsid nohup "$PY" -m torch.distributed.run --nproc_per_node=3 --max-restarts=0 --master_port=$RPORT \
    train.py --config_path "$REAL_CFG" --no-one-logger >> "$REALLOG" 2>&1 </dev/null & disown
  log "real 3-GPU run launched on CVD=$cs (pid $!, port $RPORT)"
}

# ---------- Phase 1: accumulate ----------
log "watcher up. accumulate to $NEED (working CVD 0-6, broken idx6 excluded)."
REAL_CVDS=""
while true; do
  for f in "$HELD"/*; do [ -e "$f" ] || continue; p=$(cat "$f" 2>/dev/null)
    if [ -z "$p" ] || ! kill -0 "$p" 2>/dev/null; then c=$(basename "$f"); rm -f "$f"; log "hold CVD=$c died -> released"; fi
  done
  while IFS=',' read -r uu used; do
    uu=$(echo "$uu" | tr -d ' '); used=$(echo "$used" | tr -dc '0-9'); key=${uu:4:8}; c=${CVD[$key]:-}
    [ -z "$c" ] && continue; [ -e "$HELD/$c" ] && continue
    if [ -n "$used" ] && [ "$used" -lt "$FREE_THRESH" ] 2>/dev/null; then launch_ro "$c"; fi
  done < <(nvidia-smi --query-gpu=gpu_uuid,memory.used --format=csv,noheader,nounits 2>/dev/null)
  mapfile -t H < <(ls -1 "$HELD" 2>/dev/null | sort)
  if [ "${#H[@]}" -ge "$NEED" ]; then
    REAL_CVDS="${H[0]},${H[1]},${H[2]}"
    log "have ${#H[@]} holds (${H[*]}) -> launching REAL run on ${H[0]},${H[1]},${H[2]}"
    launch_real "${H[0]}" "${H[1]}" "${H[2]}"; break
  fi
  [ "${#H[@]}" -gt 0 ] && log "holding ${#H[@]}/$NEED: ${H[*]}"
  sleep 20
done

# ---------- Phase 3: guardian ----------
IFS=',' read -r RC0 RC1 RC2 <<< "$REAL_CVDS"
sleep 120   # startup grace
prev=-1; stuck=0; relaunches=0; giveup=0
while [ $giveup -eq 0 ]; do
  if real_alive; then sleep 30; continue; fi
  s=$(laststep); s=${s:-0}
  if [ "$s" -ge "$DONE_STEP" ] 2>/dev/null; then log "real run reached $s (>=${DONE_STEP}) -> DONE. exiting."; exit 0; fi
  if [ "$s" -gt "$prev" ] 2>/dev/null; then stuck=0; else stuck=$((stuck+1)); fi
  prev=$s
  if [ $stuck -ge 2 ] || [ $relaunches -ge 8 ]; then giveup=1; break; fi
  relaunches=$((relaunches+1))
  log "real run DOWN at step $s (stuck=$stuck) -> auto-resume #$relaunches"
  launch_real "$RC0" "$RC1" "$RC2"; sleep 120
done

# ---------- give up -> conserve GPUs read-only until Aden ----------
log "=== GIVING UP on real run at step ${prev} after ${relaunches} tries. Crash log kept: $REALLOG ==="
log "=== switching CVD ${RC0},${RC1},${RC2} to READ-ONLY HOLD mode (conserve for Aden) ==="
pkill -9 -f "_real3.yaml" 2>/dev/null; sleep 3; rm -f "$HELD"/*
while true; do
  for c in "$RC0" "$RC1" "$RC2"; do
    if ! ro_alive_on "$c"; then launch_ro "$c"; fi
  done
  sleep 30
done
