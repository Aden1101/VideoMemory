#!/usr/bin/env bash
# ============================================================================================
# MATERIAUX v4 : 24 paires jumelles (48 videos single 90f), sujets ANIMAUX / VEHICULES /
# OBJETS, ZERO humain. Registre : "C2 v3 : MATERIAU v4" (scelle 07/08). DERNIER materiau.
#
# SANS collecte h (--no-h : latents/conds/graines seulement, ~20 MB/video) : les h seront
# transitoires APRES la labellisation d'Aden, comme pour v3. Base corpus_idx 9 200 000.
#
# GPU : choix par le FAIT (jamais une carte portant le process d'un AUTRE utilisateur, jamais
# 5/6, marge >= 25 GB au moment MEME du lancement). Les deux vagues (main / twins) tournent
# en parallele sur deux cartes eligibles.
#
# Usage : bash reference/runs/probe1/run_collect_v4.sh [GPU_MAIN GPU_TWIN]
#         (sans argument : les deux premieres cartes eligibles sont choisies automatiquement)
# ============================================================================================
set -euo pipefail
REPO=/shared/ssd_14T/home/adenelmi/Anchor-Forcing
PY=/shared/ssd_14T/home/adenelmi/miniconda3/envs/longlive/bin/python
PROMPTS="$REPO/reference/runs/frame_dec/prompts_noattr_v4.txt"
OUT=/shared/ssd_14T_project/adenelmi/prompts_noattr_v4
export CUDA_DEVICE_ORDER=PCI_BUS_ID
cd "$REPO"
ME="$(id -un)"

gpu_ok() {   # 1 = eligible
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
if [ "$#" -ge 2 ]; then
  GPU_MAIN="$1"; GPU_TWIN="$2"
  gpu_ok "$GPU_MAIN" || { echo "GPU $GPU_MAIN non eligible : REFUS"; exit 1; }
  gpu_ok "$GPU_TWIN" || { echo "GPU $GPU_TWIN non eligible : REFUS"; exit 1; }
else
  GPU_MAIN=""; GPU_TWIN=""
  for G in 0 1 2 3 4 7; do
    if gpu_ok "$G"; then
      [ -z "$GPU_MAIN" ] && { GPU_MAIN="$G"; continue; }
      [ -z "$GPU_TWIN" ] && { GPU_TWIN="$G"; break; }
    fi
  done
  [ -z "$GPU_MAIN" ] || [ -z "$GPU_TWIN" ] && {
    echo "moins de 2 cartes eligibles (libres, >=25 GB, sans process tiers) : REFUS"; exit 1; }
fi
echo "=== cartes retenues : MAIN=$GPU_MAIN  TWIN=$GPU_TWIN ==="

echo "=== garde-fou disque (>= 20 GB ; --no-h pese ~1 GB au total) ==="
df -h /shared/ssd_14T_project
AVAIL=$(df -BG --output=avail /shared/ssd_14T_project | tail -1 | tr -dc '0-9')
[ "$AVAIL" -lt 20 ] && { echo "disque projet : ${AVAIL} GB libres (< 20) : REFUS"; exit 1; }

mkdir -p "$OUT" "${OUT}_twins"
COMMON=(reference/runs/frame_dec/collect_pairs.py --v3 --mode run
        --prompts-file "$PROMPTS" --prompts-base 9200000 --no-h
        --frames-single 90 --videos-per-part 12 --check-videos 0
        --max-gb 90 --shard 0/1)

echo "=== vague MAIN (24 videos) sur GPU $GPU_MAIN + vague TWIN sur GPU $GPU_TWIN ==="
( CUDA_VISIBLE_DEVICES="$GPU_MAIN" $PY "${COMMON[@]}" --out "$OUT" \
    && CUDA_VISIBLE_DEVICES="$GPU_MAIN" $PY reference/runs/frame_dec/collect_pairs.py \
         --v3 --mode manifest --out "$OUT" ) \
  > "$OUT/collect_v4_main.log" 2>&1 &
P_MAIN=$!
( CUDA_VISIBLE_DEVICES="$GPU_TWIN" $PY "${COMMON[@]}" --twin-offset 100000 \
    --out "${OUT}_twins" \
    && CUDA_VISIBLE_DEVICES="$GPU_TWIN" $PY reference/runs/frame_dec/collect_pairs.py \
         --v3 --mode manifest --out "${OUT}_twins" ) \
  > "${OUT}_twins/collect_v4_twins.log" 2>&1 &
P_TWIN=$!

RC=0
wait "$P_MAIN" || { echo "vague MAIN en ECHEC (voir $OUT/collect_v4_main.log)"; RC=1; }
wait "$P_TWIN" || { echo "vague TWIN en ECHEC (voir ${OUT}_twins/collect_v4_twins.log)"; RC=1; }
tail -n 3 "$OUT/collect_v4_main.log" "${OUT}_twins/collect_v4_twins.log" || true
du -sh "$OUT" "${OUT}_twins" || true
df -h /shared/ssd_14T_project | tail -1
[ "$RC" -ne 0 ] && { echo "COLLECT_V4_FAILED"; exit "$RC"; }
echo "COLLECT_V4_DONE"
