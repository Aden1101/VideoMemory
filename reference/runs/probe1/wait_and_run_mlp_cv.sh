#!/usr/bin/env bash
# ============================================================================================
# Lanceur d'attente pour le sondage CV des bras MLP (GO du coordinateur, 07/08).
# Au moment du GO, les 8 cartes portaient un process de calcul d'un AUTRE utilisateur (zhou sur
# 0/1/6/7, selimlali sur 2-5) et le maximum libre etait 23.3 GB, sous la marge de 25 GB. La
# regle est explicite : on ne prend pas une carte occupee par quelqu'un d'autre. Ce script
# attend donc une carte qui satisfait LES DEUX conditions, puis lance le run tel quel.
#   - AUCUN process de calcul d'un autre utilisateur sur la carte
#   - >= 25 GB libres
# L'index 6 est ecarte (smi casse). Une seule carte est prise (jamais 5/6 cartes).
# Usage : setsid nohup bash reference/runs/probe1/wait_and_run_mlp_cv.sh > <log> 2>&1 &
# ============================================================================================
set -uo pipefail
export CUDA_DEVICE_ORDER=PCI_BUS_ID
REPO=/shared/ssd_14T/home/adenelmi/Anchor-Forcing
cd "$REPO"
ME=$(id -un)

others_on() {                      # -> liste des autres utilisateurs sur la carte $1
  nvidia-smi -i "$1" --query-compute-apps=pid --format=csv,noheader | while read -r p; do
    [ -n "$p" ] && ps -o user= -p "$p" 2>/dev/null
  done | sort -u | grep -v "^${ME}$" | tr -d ' \n'
}

while true; do
  for i in 0 1 2 3 4 5 7; do
    FREE=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$i" 2>/dev/null)
    [ -z "${FREE:-}" ] && continue
    OTH=$(others_on "$i")
    if [ "$FREE" -ge 25000 ] && [ -z "$OTH" ]; then
      echo "[attente] GPU $i libre (${FREE} MiB, aucun autre utilisateur) a $(date) -> lancement"
      exec bash reference/runs/probe1/run_mlp_delta.sh "$i" --with-cv
    fi
  done
  echo "[attente] $(date +%H:%M) aucune carte libre (marge 25 GB + zero process tiers), on repasse dans 60 s"
  sleep 60
done
