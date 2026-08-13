#!/usr/bin/env bash
# ============================================================================================
# ECHELLE DE FLOU FINE -- 12 videos (scenarios 08-10 x sigma 1/2/3/4) sur la CARTE 5
# (celle qui a genere ces scenarios en campagne 2 : le re-check bit a bit de l'ancre par
# ORACLE_ANCHOR_DUMP suppose la meme carte). >= 25 GB au pre-check, co-run ok, jamais l'idx 6.
# Si la carte 5 est inelegible on attend (pas de bascule silencieuse de carte).
# ============================================================================================
set -euo pipefail
REPO=/shared/ssd_14T/home/adenelmi/Anchor-Forcing
PY=/shared/ssd_14T/home/adenelmi/miniconda3/envs/longlive/bin/python
HERE="$REPO/reference/runs/fidelity_ladder"
LOG=/shared/ssd_14T_project/adenelmi/fidelity_ladder/logs/run_blur.log
export CUDA_DEVICE_ORDER=PCI_BUS_ID
WAITED=0
TRIED_OTHER=0     # 1 apres un essai rate sur une carte != 5 (assert d'ancre) -> on n'attend plus que la 5

pick() {          # prefere la 5 ; sinon la plus libre eligible (jamais la 6)
  local f5 idx free
  f5=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i 5)
  if [ "$f5" -ge 25000 ]; then echo 5; return 0; fi
  [ "$TRIED_OTHER" = "1" ] && return 1
  while read -r idx free; do
    idx="${idx%,}"
    [ "$idx" = "6" ] && continue
    [ "$free" -lt 25000 ] && continue
    echo "$idx"; return 0
  done < <(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | sort -t, -k2 -rn)
  return 1
}

while true; do
  DGB=$(df -BG /shared/ssd_14T_project | awk 'NR==2 {gsub("G","",$4); print $4}')
  if [ "$DGB" -ge 15 ] && GPU=$(pick); then
    FREE=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$GPU")
    if [ "$FREE" -lt 25000 ]; then sleep 60; continue; fi
    echo "[blur] $(date '+%F %T') GPU $GPU (libre ${FREE} MiB, ${DGB}G) apres ${WAITED} min" | tee -a "$LOG"
    set +e
    CUDA_VISIBLE_DEVICES="$GPU" "$PY" "$HERE/run_fidelity2.py" \
      --scenarios scenario08,scenario09,scenario10 \
      --arms flou_s1,flou_s2,flou_s3,flou_s4 >> "$LOG" 2>&1
    RC=$?
    set -e
    if [ "$RC" -eq 0 ]; then echo "[blur] $(date '+%F %T') TERMINE rc=0" | tee -a "$LOG"; exit 0; fi
    if tail -n 60 "$LOG" | grep -q "segment A differs from the stored anchor"; then
      echo "[blur] $(date '+%F %T') carte $GPU : segment A != ancre stockee (non-determinisme inter-carte) ; on n'attend plus que la carte 5" | tee -a "$LOG"
      TRIED_OTHER=1; sleep 60; continue
    fi
    if tail -n 60 "$LOG" | grep -qiE "out of memory|CUDA error"; then
      echo "[blur] $(date '+%F %T') OOM DETECTE : ARRET sans relance" | tee -a "$LOG"; exit 3
    fi
    echo "[blur] $(date '+%F %T') echec rc=$RC : ABANDON, voir $LOG" | tee -a "$LOG"; exit 2
  fi
  sleep 180; WAITED=$((WAITED + 3))
  [ $((WAITED % 30)) -eq 0 ] && echo "[blur] $(date '+%F %T') attente d'une carte (${WAITED} min)" | tee -a "$LOG"
done
