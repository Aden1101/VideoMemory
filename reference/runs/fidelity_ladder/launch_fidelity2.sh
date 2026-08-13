#!/usr/bin/env bash
# ============================================================================================
# ECHELLE DE FIDELITE -- lanceur avec ATTENTE DE FENETRE GPU.
#
# Regles (assouplies par le coordinateur 2026-08-11, regle du projet) :
# CUDA_DEVICE_ORDER=PCI_BUS_ID toujours ; index 6 INTERDIT ; au plus 2 cartes ;
# une carte est ELIGIBLE si >= 25 GB libres AU PRE-CHECK, quel que soit le proprietaire des
# process qui y tournent (le co-run est PREFERE a l'attente) ; on prend la carte la PLUS
# LIBRE ; pre-check final DANS la meme iteration que le lancement ; verrou fichier pour que
# les deux moities ne prennent pas la meme carte ; garde-fou disque.
# OOM : on NE relance PAS en boucle -- la moitie s'arrete et le rapporte (reduction a 1 carte
# geree par l'operateur). Echec non-OOM : 3 tentatives max.
#
#   bash launch_fidelity.sh A     # scenarios 1-5
#   bash launch_fidelity.sh B     # scenarios 6-10
# ============================================================================================
set -euo pipefail
HALF="${1:?usage: launch_fidelity.sh A|B}"
REPO=/shared/ssd_14T/home/adenelmi/Anchor-Forcing
PY=/shared/ssd_14T/home/adenelmi/miniconda3/envs/longlive/bin/python
HERE="$REPO/reference/runs/fidelity_ladder"
HEAVY=/shared/ssd_14T_project/adenelmi/fidelity_ladder
LOG="$HEAVY/logs/run2_half${HALF}.log"
mkdir -p "$HEAVY/logs"
export CUDA_DEVICE_ORDER=PCI_BUS_ID

free_disk_gb() { df -BG /shared/ssd_14T_project | awk 'NR==2 {gsub("G","",$4); print $4}'; }

# carte candidate : PCI idx != 6, >= 25 GB libres au pre-check, pas de verrou ;
# co-run autorise (regle du projet) ; on renvoie la carte la PLUS LIBRE
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

echo "[launch $HALF] $(date '+%F %T') recherche d'une carte (>=25 GB libres, co-run ok)..." | tee -a "$LOG"
WAITED=0
TRIES=0
while true; do
  DGB=$(free_disk_gb)
  if [ "$DGB" -lt 15 ]; then
    echo "[launch $HALF] $(date '+%F %T') disque projet ${DGB}G < 15G : ATTENTE" | tee -a "$LOG"
    sleep 600; continue
  fi
  if GPU=$(pick_gpu); then
    # verrou puis RE-CHECK immediat (pre-check dans la commande, regle de la maison)
    echo "$$ half$HALF $(date '+%F %T')" > "$HEAVY/logs/gpu_${GPU}.lock"
    sleep 2
    FREE=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$GPU")
    if [ "$FREE" -lt 25000 ]; then
      echo "[launch $HALF] carte $GPU retombee a ${FREE} MiB entre-temps, on rend le verrou" | tee -a "$LOG"
      rm -f "$HEAVY/logs/gpu_${GPU}.lock"; sleep 60; continue
    fi
    echo "[launch $HALF] $(date '+%F %T') GPU $GPU (libre ${FREE} MiB, ${DGB}G disque, co-run) apres ${WAITED} min d'attente" | tee -a "$LOG"
    set +e
    CUDA_VISIBLE_DEVICES="$GPU" "$PY" "$HERE/run_fidelity2.py" --half "$HALF" >> "$LOG" 2>&1
    RC=$?
    set -e
    rm -f "$HEAVY/logs/gpu_${GPU}.lock"
    if [ "$RC" -eq 0 ]; then
      echo "[launch $HALF] $(date '+%F %T') TERMINE rc=0" | tee -a "$LOG"; exit 0
    fi
    if tail -n 80 "$LOG" | grep -qiE "out of memory|CUDA error|cublas.*alloc"; then
      echo "[launch $HALF] $(date '+%F %T') OOM DETECTE (rc=$RC) : ARRET de cette moitie sans relance (regle coordinateur : reduire a une carte, rapporter)" | tee -a "$LOG"
      exit 3
    fi
    TRIES=$((TRIES + 1))
    if [ "$TRIES" -ge 3 ]; then
      echo "[launch $HALF] $(date '+%F %T') echec rc=$RC (tentative $TRIES/3) : ABANDON, voir $LOG" | tee -a "$LOG"
      exit 2
    fi
    echo "[launch $HALF] $(date '+%F %T') echec rc=$RC (tentative $TRIES/3) -- nouvelle tentative dans 10 min" | tee -a "$LOG"
    sleep 600
  else
    sleep 180; WAITED=$((WAITED + 3))
    if [ $((WAITED % 60)) -eq 0 ]; then
      echo "[launch $HALF] $(date '+%F %T') toujours en attente (${WAITED} min) -- occupation:" | tee -a "$LOG"
      nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | tee -a "$LOG"
    fi
  fi
done
