#!/usr/bin/env bash
# ============================================================================================
# SUPPLEMENT (4 nouvelles scenes, PRIORITAIRE sur l'echelle de flou fine).
# Regle projet : carte eligible si >= 25 GB libres au pre-check (co-run ok), jamais l'idx 6,
# au plus 2 cartes (verrous), la plus libre d'abord ; OOM = arret sans relance ;
# la moitie B relance l'echelle de flou fine (declassee) une fois SA part terminee.
#   bash launch_supplement.sh A|B
# ============================================================================================
set -euo pipefail
HALF="${1:?usage: launch_supplement.sh A|B}"
REPO=/shared/ssd_14T/home/adenelmi/Anchor-Forcing
PY=/shared/ssd_14T/home/adenelmi/miniconda3/envs/longlive/bin/python
HERE="$REPO/reference/runs/fidelity_ladder"
HEAVY=/shared/ssd_14T_project/adenelmi/fidelity_ladder
LOG="$HEAVY/logs/run_sup_half${HALF}.log"
mkdir -p "$HEAVY/logs"
export CUDA_DEVICE_ORDER=PCI_BUS_ID

free_disk_gb() { df -BG /shared/ssd_14T_project | awk 'NR==2 {gsub("G","",$4); print $4}'; }

pick_gpu() {
  local idx free
  while read -r idx free; do
    idx="${idx%,}"
    [ "$idx" = "6" ] && continue
    [ "$free" -lt 25000 ] && continue
    [ -e "$HEAVY/logs/gpu_${idx}.lock" ] && continue
    echo "$idx"; return 0
  done < <(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits \
           | sort -t, -k2 -rn)
  return 1
}

echo "[sup $HALF] $(date '+%F %T') recherche d'une carte (>=25 GB, co-run ok)..." | tee -a "$LOG"
WAITED=0
TRIES=0
while true; do
  DGB=$(free_disk_gb)
  if [ "$DGB" -lt 15 ]; then
    echo "[sup $HALF] $(date '+%F %T') disque ${DGB}G < 15G : ATTENTE" | tee -a "$LOG"; sleep 600; continue
  fi
  if GPU=$(pick_gpu); then
    echo "$$ sup$HALF $(date '+%F %T')" > "$HEAVY/logs/gpu_${GPU}.lock"
    sleep 2
    FREE=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$GPU")
    if [ "$FREE" -lt 25000 ]; then
      rm -f "$HEAVY/logs/gpu_${GPU}.lock"; sleep 60; continue
    fi
    echo "[sup $HALF] $(date '+%F %T') GPU $GPU (libre ${FREE} MiB, ${DGB}G) apres ${WAITED} min" | tee -a "$LOG"
    set +e
    CUDA_VISIBLE_DEVICES="$GPU" "$PY" "$HERE/run_fidelity2.py" --campaign supplement \
      --half "$HALF" >> "$LOG" 2>&1
    RC=$?
    set -e
    rm -f "$HEAVY/logs/gpu_${GPU}.lock"
    if [ "$RC" -eq 0 ]; then
      echo "[sup $HALF] $(date '+%F %T') TERMINE rc=0" | tee -a "$LOG"
      if [ "$HALF" = "B" ]; then
        echo "[sup B] relance de l'echelle de flou fine (declassee)" | tee -a "$LOG"
        bash "$HERE/start_blur.sh" >> "$LOG" 2>&1 || true
      fi
      exit 0
    fi
    if tail -n 80 "$LOG" | grep -qiE "out of memory|CUDA error|cublas.*alloc"; then
      echo "[sup $HALF] $(date '+%F %T') OOM DETECTE (rc=$RC) : ARRET sans relance" | tee -a "$LOG"; exit 3
    fi
    TRIES=$((TRIES + 1))
    if [ "$TRIES" -ge 3 ]; then
      echo "[sup $HALF] $(date '+%F %T') echec rc=$RC (tentative $TRIES/3) : ABANDON" | tee -a "$LOG"; exit 2
    fi
    echo "[sup $HALF] $(date '+%F %T') echec rc=$RC (tentative $TRIES/3), retry dans 10 min" | tee -a "$LOG"
    sleep 600
  else
    sleep 180; WAITED=$((WAITED + 3))
    if [ $((WAITED % 60)) -eq 0 ]; then
      echo "[sup $HALF] $(date '+%F %T') toujours en attente (${WAITED} min)" | tee -a "$LOG"
      nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | tee -a "$LOG"
    fi
  fi
done
