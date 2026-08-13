#!/usr/bin/env bash
# ============================================================================================
# MATERIAUX v3 : 13 paires a prompts SANS ATTRIBUT (26 videos single 90f), nuit du 06/08.
# C2-generation (verdict C1 VERT scelle au registre). SANS collecte h (--no-h : latents/conds/
# graines seulement, la collecte h sera transitoire apres labellisation). Base corpus_idx
# 9 100 000 (--prompts-base) : aucune collision de video_id avec v2 dans masks/ sig/ states/.
#
# Les DEUX vagues tournent EN PARALLELE : main sur $1, twins (graine +100000) sur $2.
# Pre-check GPU DANS la commande pour CHAQUE carte ; 0/1/2 (occupes par d'autres) et 5/6
# INTERDITS ce soir ; garde-fou disque >= 50 GB.
# Usage : bash reference/runs/probe1/run_collect_v3.sh 7 3          (vague 1)
#         bash reference/runs/probe1/run_collect_v3.sh 7 3 resume   (vague 2+ : REPREND les
#         memes sorties ; le collecteur recharge le manifeste, continue la numerotation des
#         parts et SAUTE les corpus_idx deja faits -- seuls les prompts ajoutes au fichier
#         sont generes ; logs *_wave2.log pour ne pas ecraser ceux de la vague 1)
# ============================================================================================
set -euo pipefail
GPU_MAIN="${1:-7}"
GPU_TWIN="${2:-3}"
MODE3="${3:-}"
RESUME=()
SUF=""
if [ "$MODE3" = "resume" ]; then
  RESUME=(--resume)
  SUF="_wave2"
fi
for G in "$GPU_MAIN" "$GPU_TWIN"; do
  case "$G" in 0|1|2|5|6) echo "GPU $G INTERDIT ce soir (occupe ou casse) : refus"; exit 1;; esac
done
[ "$GPU_MAIN" = "$GPU_TWIN" ] && { echo "meme GPU pour les 2 vagues : passe-le en seul arg"; exit 1; }
REPO=/shared/ssd_14T/home/adenelmi/Anchor-Forcing
PY=/shared/ssd_14T/home/adenelmi/miniconda3/envs/longlive/bin/python
PROMPTS="$REPO/reference/runs/frame_dec/prompts_noattr_v3.txt"
OUT=/shared/ssd_14T_project/adenelmi/prompts_noattr_v3
export CUDA_DEVICE_ORDER=PCI_BUS_ID
cd "$REPO"
mkdir -p "$OUT" "${OUT}_twins"

echo "=== garde-fou disque (>= 50 GB) ==="
df -h /shared/ssd_14T_project
AVAIL=$(df -BG --output=avail /shared/ssd_14T_project | tail -1 | tr -dc '0-9')
[ "$AVAIL" -lt 50 ] && { echo "disque projet : ${AVAIL} GB libres (< 50) : REFUS"; exit 1; }

echo "=== pre-check GPUs $GPU_MAIN et $GPU_TWIN (au meme instant que le lancement) ==="
nvidia-smi --query-gpu=index,memory.free,memory.used,utilization.gpu --format=csv
for G in "$GPU_MAIN" "$GPU_TWIN"; do
  FREE=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$G")
  [ "$FREE" -lt 25000 ] && { echo "GPU $G : ${FREE} MiB libres (< 25 GB) : REFUS"; exit 1; }
done

COMMON=(reference/runs/frame_dec/collect_pairs.py --v3 --mode run
        --prompts-file "$PROMPTS" --prompts-base 9100000 --no-h
        --frames-single 90 --videos-per-part 10 --check-videos 0
        --max-gb 90 --shard 0/1)
if [ ${#RESUME[@]} -gt 0 ]; then COMMON+=("${RESUME[@]}"); fi

echo "=== vague MAIN sur GPU $GPU_MAIN + vague TWIN (graine +100000) sur GPU $GPU_TWIN, EN PARALLELE${SUF:+ (reprise)} ==="
( CUDA_VISIBLE_DEVICES="$GPU_MAIN" $PY "${COMMON[@]}" \
    && CUDA_VISIBLE_DEVICES="$GPU_MAIN" $PY reference/runs/frame_dec/collect_pairs.py \
         --v3 --mode manifest --out "$OUT" ) \
  > "$OUT/collect_v3_main${SUF}.log" 2>&1 &
P_MAIN=$!
( CUDA_VISIBLE_DEVICES="$GPU_TWIN" $PY "${COMMON[@]}" --twin-offset 100000 \
    --out "${OUT}_twins" \
    && CUDA_VISIBLE_DEVICES="$GPU_TWIN" $PY reference/runs/frame_dec/collect_pairs.py \
         --v3 --mode manifest --out "${OUT}_twins" ) \
  > "${OUT}_twins/collect_v3_twins${SUF}.log" 2>&1 &
P_TWIN=$!

RC=0
wait "$P_MAIN" || { echo "vague MAIN en ECHEC (voir $OUT/collect_v3_main${SUF}.log)"; RC=1; }
wait "$P_TWIN" || { echo "vague TWIN en ECHEC (voir ${OUT}_twins/collect_v3_twins${SUF}.log)"; RC=1; }
# tail -n N (jamais l'obsolete `tail -3` : avec DEUX fichiers, GNU le rejette et, sous
# set -e, ce rapport cosmetique faisait passer une collecte REUSSIE pour un echec, 06/08 nuit)
tail -n 3 "$OUT/collect_v3_main${SUF}.log" "${OUT}_twins/collect_v3_twins${SUF}.log" || true
du -sh "$OUT" "${OUT}_twins"
df -h /shared/ssd_14T_project | tail -1
[ "$RC" -ne 0 ] && { echo "COLLECT_V3_FAILED"; exit "$RC"; }
echo "COLLECT_V3_DONE"
