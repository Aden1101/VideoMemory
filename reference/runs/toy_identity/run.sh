#!/usr/bin/env bash
# ============================================================================================
# TOY D'ECRITURE-IDENTITE : les DEUX runs scelles, sur UNE carte, l'un apres l'autre.
#   bras 'trained' = ecrivain theta_w + tete co-entraines (la question)
#   bras 'frozen'  = LE MEME entrainement, ecrivain GELE (le critere 3)
# Puis le verdict mecanique des 4 criteres.
#
# Pre-check GPU DANS la commande de lancement (regle de la maison) : EXCLUSIVITE demandee
# (aucun voisin), jamais l'index 6.
# Usage : bash reference/runs/toy_identity/run.sh [GPU_INDEX]
# ============================================================================================
set -euo pipefail
export CUDA_DEVICE_ORDER=PCI_BUS_ID
REPO=/shared/ssd_14T/home/adenelmi/Anchor-Forcing
PY=/shared/ssd_14T/home/adenelmi/miniconda3/envs/longlive/bin/python
LOGS=/shared/ssd_14T_project/adenelmi/toy_identity
mkdir -p "$LOGS"

GPU=${1:-}
if [ -z "$GPU" ]; then
  GPU=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits \
        | awk -F', ' '$1!=6 && $2<1000 {print $1; exit}')
fi
[ -n "$GPU" ] || { echo "AUCUN GPU LIBRE (exclusivite exigee)"; exit 1; }
[ "$GPU" != "6" ] || { echo "index 6 INTERDIT (carte cassee)"; exit 1; }
USED=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits \
       | awk -F', ' -v g="$GPU" '$1==g {print $2}')
[ "$USED" -lt 1000 ] || { echo "GPU $GPU occupe ($USED MiB)"; exit 1; }
echo "[run] GPU $GPU (occupation $USED MiB) -- pre-check OK"

cd "$REPO"
for TAG in trained frozen; do
  EXTRA=""
  [ "$TAG" = "frozen" ] && EXTRA="--freeze_writer"
  echo "[run] === bras $TAG ==="
  CUDA_VISIBLE_DEVICES=$GPU $PY reference/runs/toy_identity/toy_identity.py \
      --tag "$TAG" $EXTRA "${@:2}" > "$LOGS/$TAG.log" 2>&1
  tail -3 "$LOGS/$TAG.log"
done

$PY reference/runs/toy_identity/toy_identity.py --stage verdict | tee "$LOGS/verdict.txt"
