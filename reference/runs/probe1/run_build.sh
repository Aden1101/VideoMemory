#!/usr/bin/env bash
# ============================================================================================
# ETAPE 1 -- construction complete des artefacts (AUCUN run de verdict).
# Ordre impose par les dependances :
#   masques -> (etats | sondes) -> signatures {state, h} -> selftests
# Les deux premieres etapes sont longues (~35 min) et se lancent a part, une seule fois :
#   python reference/runs/probe1/build_masks.py --all        # tache 0, 165 videos
#   python reference/runs/probe1/build_masks.py --fix_pairs  # une requete par PAIRE
#   python reference/runs/probe1/build_probes.py             # la batterie fixe
# Le pre-check GPU est DANS la commande (regle de la maison) et l'index 6 est exclu.
# Usage :  bash reference/runs/probe1/run_build.sh 7
# ============================================================================================
set -euo pipefail
GPU="${1:-7}"
[ "$GPU" = "6" ] && { echo "index 6 est CASSE, refus"; exit 1; }
REPO=/shared/ssd_14T/home/adenelmi/Anchor-Forcing
PY=/shared/ssd_14T/home/adenelmi/miniconda3/envs/longlive/bin/python
HEAVY=/shared/ssd_14T_project/adenelmi/probe1
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="$GPU" HF_HOME="$HEAVY/hf"
cd "$REPO"

echo "=== pre-check GPU (exclusivite) ==="
nvidia-smi --query-gpu=index,memory.free,memory.used,utilization.gpu --format=csv

echo "=== 1/4 etats (rejeu des ecritures, 60 videos x 4 ages, ~41 GB) ==="
$PY reference/runs/probe1/build_states.py --only_pairs 2>&1 | tail -20

echo "=== 2/4 signatures : les ETATS ==="
$PY reference/runs/probe1/build_signatures.py --source state 2>&1 | tail -8

echo "=== 3/4 signatures : la BORNE SUPERIEURE (h) ==="
$PY reference/runs/probe1/build_signatures.py --source h 2>&1 | tail -8

echo "=== 4/4 selftests complets ==="
$PY reference/runs/probe1/selftest_probe1.py --gpu 2>&1 | grep -viE "futurewarning|warnings.warn"
