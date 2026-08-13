#!/usr/bin/env bash
# ============================================================================================
# ETAPE 1 v2 -- construction des artefacts MANQUANTS (AUCUN run de verdict).
#   A. corpus v1 : la nouvelle borne `hdirect` (60 videos de paires) + le jeu de
#      PRE-ENTRAINEMENT (105 videos switch : hdirect ET state, ce dernier en --stream, donc
#      sans ecrire un seul octet d'etat -- 71 GB economises) ;
#   B. corpus v2 : masques (OWL-ViT + SAM) + planches (dont la NUMEROTEE) + signatures des 3
#      sources, toutes en --stream.
# Pre-check GPU DANS la commande, index 6 INTERDIT.
# Usage :  bash reference/runs/probe1/run_build_v2.sh 7 [A|B|AB]
# ============================================================================================
set -euo pipefail
GPU="${1:-7}"
WHAT="${2:-AB}"
[ "$GPU" = "6" ] && { echo "index 6 est CASSE, refus"; exit 1; }
REPO=/shared/ssd_14T/home/adenelmi/Anchor-Forcing
PY=/shared/ssd_14T/home/adenelmi/miniconda3/envs/longlive/bin/python
HEAVY=/shared/ssd_14T_project/adenelmi/probe1
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="$GPU" HF_HOME="$HEAVY/hf"
cd "$REPO"

echo "=== pre-check GPU $GPU ==="
nvidia-smi --query-gpu=index,memory.free,memory.used,utilization.gpu --format=csv
FREE=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$GPU")
[ "$FREE" -lt 15000 ] && { echo "GPU $GPU : ${FREE} MiB libres, REFUS"; exit 1; }
df -h /shared/ssd_14T_project

if [[ "$WHAT" == *A* ]]; then
  echo "=== A1 borne hdirect, corpus v1 (60 videos de paires) ==="
  $PY reference/runs/probe1/build_signatures.py --source hdirect 2>&1 | tail -4
  echo "=== A2 pre-entrainement : hdirect sur les 105 videos switch ==="
  $PY reference/runs/probe1/build_signatures.py --source hdirect --switch_only 2>&1 | tail -4
  echo "=== A3 pre-entrainement : state sur les 105 videos switch (--stream, 0 octet d'etat) ==="
  $PY reference/runs/probe1/build_signatures.py --source state --switch_only --stream 2>&1 | tail -4
fi

if [[ "$WHAT" == *B* ]]; then
  echo "=== B0 index du corpus v2 (reconstruit : la collecte vient de finir) ==="
  $PY -c "import sys; sys.path.insert(0,'$REPO/reference/runs/probe1'); \
import probe1_common as C; v=C.load_index(corpus='v2', rebuild=True, verbose=True); \
_, p = C.pair_dataset(corpus='v2', verbose=False); print('paires v2 :', len(p)); \
assert len(p) == 30, f'{len(p)} paires v2 (30 attendues) : la collecte est-elle finie ?'"
  echo "=== B1 masques du corpus v2 (OWL-ViT + SAM) ==="
  $PY reference/runs/probe1/build_masks.py --corpus v2 --only_pairs 2>&1 | tail -8
  echo "=== B2 coherence des jumeaux (une requete par PAIRE) + planches ==="
  $PY reference/runs/probe1/build_masks.py --corpus v2 --fix_pairs 2>&1 | tail -8
  echo "=== B3 etats v2 : REJOUES A LA VOLEE (--stream), pas stockes ==="
  # 4 ages x 60 videos x 170 MB = 41 GB d'intermediaires bit a bit reproductibles (T19) sur un
  # disque projet a 100 % : on rejoue et on ne garde que les signatures. Pour les materialiser
  # quand meme : `build_states.py --corpus v2 --only_pairs` (2,5 s/video).
  AVAIL=$(df -BG --output=avail /shared/ssd_14T_project | tail -1 | tr -dc '0-9')
  echo "    disque projet : ${AVAIL} GB libres (les etats couteraient 41 GB)"
  echo "=== B4 signatures v2 : state (--stream), hdirect, h ==="
  $PY reference/runs/probe1/build_signatures.py --corpus v2 --source state --stream 2>&1 | tail -3
  $PY reference/runs/probe1/build_signatures.py --corpus v2 --source hdirect 2>&1 | tail -3
  $PY reference/runs/probe1/build_signatures.py --corpus v2 --source h 2>&1 | tail -3
fi

echo "=== selftests ==="
$PY reference/runs/probe1/selftest_probe1.py --gpu 2>&1 | grep -viE "futurewarning|warnings.warn" | tail -25
df -h /shared/ssd_14T_project
echo "BUILD_V2_DONE"
