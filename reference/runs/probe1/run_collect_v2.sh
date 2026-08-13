#!/usr/bin/env bash
# ============================================================================================
# MATERIAUX v2 : 30 paires a prompts SANS ATTRIBUT (60 videos single 90f).
# Vague 1 = les 30 "main" (graine de base), vague 2 = les 30 "twin" (graine +100000, le MEME
# offset que la campagne v3 : mem2_data.TWIN_SEED_OFFSET).
# Les deux vagues sont CHAINEES sur le meme GPU (une seule carte libre au lancement).
# Pre-check GPU DANS la commande (regle de la maison), index 6 INTERDIT.
# Usage : bash reference/runs/probe1/run_collect_v2.sh 7
# ============================================================================================
set -euo pipefail
GPU="${1:-7}"
[ "$GPU" = "6" ] && { echo "index 6 est CASSE, refus"; exit 1; }
REPO=/shared/ssd_14T/home/adenelmi/Anchor-Forcing
PY=/shared/ssd_14T/home/adenelmi/miniconda3/envs/longlive/bin/python
PROMPTS="$REPO/reference/runs/probe1/prompts_noattr_v2.txt"
OUT=/shared/ssd_14T_project/adenelmi/prompts_noattr_v2
LOG=/shared/ssd_14T_project/adenelmi/probe1/collect_v2.log
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="$GPU"
cd "$REPO"

echo "=== pre-check GPU $GPU (exclusivite) ===" | tee -a "$LOG"
nvidia-smi --query-gpu=index,memory.free,memory.used,utilization.gpu --format=csv | tee -a "$LOG"
FREE=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$GPU")
if [ "$FREE" -lt 25000 ]; then
  echo "GPU $GPU n'a que ${FREE} MiB libres (< 25 GB) : REFUS" | tee -a "$LOG"; exit 1
fi
echo "=== disque ===" | tee -a "$LOG"
df -h /shared/ssd_14T_project | tee -a "$LOG"

echo "=== vague 1/2 : les 30 MAIN (90f, single) ===" | tee -a "$LOG"
$PY reference/runs/frame_dec/collect_pairs.py --v3 --mode run \
    --prompts-file "$PROMPTS" --frames-single 90 --videos-per-part 10 \
    --check-videos 2 --check-every 12 --max-gb 90 --shard 0/1 2>&1 | tee -a "$LOG"

echo "=== vague 2/2 : les 30 TWIN (graine +100000) ===" | tee -a "$LOG"
$PY reference/runs/frame_dec/collect_pairs.py --v3 --mode run \
    --prompts-file "$PROMPTS" --frames-single 90 --videos-per-part 10 \
    --twin-offset 100000 --max-gb 90 --shard 0/1 --out "${OUT}_twins" 2>&1 | tee -a "$LOG"

echo "=== manifestes ===" | tee -a "$LOG"
$PY reference/runs/frame_dec/collect_pairs.py --v3 --mode manifest --out "$OUT" 2>&1 | tee -a "$LOG"
$PY reference/runs/frame_dec/collect_pairs.py --v3 --mode manifest --out "${OUT}_twins" 2>&1 | tee -a "$LOG"
du -sh "$OUT" "${OUT}_twins" | tee -a "$LOG"
echo "COLLECT_V2_DONE" | tee -a "$LOG"
