#!/usr/bin/env bash
# ============================================================================================
# MATERIAUX v4 -- MASQUES + PLANCHE DE LABELLISATION (numerotee 61-84).
# A lancer APRES run_collect_v4.sh. Rien n'est score ici : la planche part a l'oeil d'Aden,
# et c'est SA labellisation qui ouvre (ou non) la suite (instrument seul d'abord, registre).
#
# Particularite v4 : le repli "a person" de l'extraction de sujet est COUPE pour ce corpus
# (C.NO_PERSON_CORPORA) -- sujets animaux/vehicules/objets, aucun humain ne doit pouvoir
# capter le masque.
#
# Usage : bash reference/runs/probe1/run_masks_v4.sh [GPU]
# ============================================================================================
set -euo pipefail
REPO=/shared/ssd_14T/home/adenelmi/Anchor-Forcing
PY=/shared/ssd_14T/home/adenelmi/miniconda3/envs/longlive/bin/python
HEAVY=/shared/ssd_14T_project/adenelmi/probe1
export CUDA_DEVICE_ORDER=PCI_BUS_ID HF_HOME="$HEAVY/hf"
cd "$REPO"
ME="$(id -un)"

gpu_ok() {
  local G="$1"
  case "$G" in 5|6) return 1;; esac
  local UUID FREE OWNER P
  UUID=$(nvidia-smi --query-gpu=uuid --format=csv,noheader -i "$G" 2>/dev/null) || return 1
  # EXCLUSIVITE : on ecarte toute carte portant un process de calcul, QUEL QUE SOIT son
  # proprietaire -- y compris le notre. Filtrer sur "un autre utilisateur" ne suffit pas :
  # un AUTRE AGENT du meme projet (bras MLP sur le GPU 2, 07/08) tourne sous le meme compte
  # et se ferait voler sa carte.
  for P in $(nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader \
             | awk -F', ' -v u="$UUID" '$1==u {print $2}'); do
    OWNER=$(ps -o user= -p "$P" 2>/dev/null | tr -d ' ')
    [ -n "$OWNER" ] && return 1
  done
  FREE=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$G")
  [ "$FREE" -ge 25000 ]
}

echo "=== pre-check GPU (au meme instant que le lancement) ==="
nvidia-smi --query-gpu=index,memory.free,memory.used,utilization.gpu --format=csv
GPU="${1:-}"
if [ -n "$GPU" ]; then
  gpu_ok "$GPU" || { echo "GPU $GPU non eligible : REFUS"; exit 1; }
else
  for G in 0 1 2 3 4 7; do gpu_ok "$G" && { GPU="$G"; break; }; done
  [ -z "$GPU" ] && { echo "aucune carte eligible : REFUS"; exit 1; }
fi
echo "=== carte retenue : GPU $GPU ==="
export CUDA_VISIBLE_DEVICES="$GPU"

echo "=== 0) index du corpus v4 (reconstruit : la generation vient de finir) ==="
$PY -c "
import sys; sys.path.insert(0, '$REPO/reference/runs/probe1')
import probe1_common as C
C.load_index(corpus='v4', rebuild=True, verbose=True)
_, p = C.pair_dataset(corpus='v4', verbose=False)
assert len(p) == 24, f'{len(p)} paires v4 (24 attendues)'
nos = sorted(q['board_no'] for q in p)
assert nos == list(range(61, 85)), nos
print(f'paires v4 : {len(p)}, planches {nos[0]}..{nos[-1]}')"

echo "=== 1) masques (OWL-ViT + SAM) sur les 48 videos ==="
$PY reference/runs/probe1/build_masks.py --corpus v4 --only_pairs 2>&1 | tail -12

echo "=== 2) coherence des jumeaux (une requete par PAIRE) + planches ==="
$PY reference/runs/probe1/build_masks.py --corpus v4 --fix_pairs 2>&1 | tail -12

echo "=== planches produites ==="
ls -la "$HEAVY/board/" | grep v4
echo "MASKS_V4_DONE"
