#!/usr/bin/env bash
# ============================================================================================
# ETAPE 1 DE LA CHAINE C2 : COLLECTE h DU CORPUS v3 (transitoire, ~46 GB, purgee apres les
# signatures). Les 42 videos v3 ont ete generees avec --no-h ; on REJOUE exactement les memes
# rollouts (memes prompts, meme base 9 100 000, memes graines, memes GPUs qu'a la generation :
# main sur 7, twins sur 3) en STOCKANT les h cette fois.
#
# Sorties DEDIEES : prompts_noattr_v3_h{,_twins} -- les dossiers d'origine sont CONSERVES
# INTACTS, car ce sont eux qui portent les latents de reference du CONTROLE D'IDENTITE
# (selftest_v3h.py : latents re-generes == latents d'origine, BIT A BIT, sinon les masques
# d'Aden ne correspondent plus aux h et tout le test serait invalide).
#
# Usage : bash reference/runs/probe1/run_collect_v3h.sh 7 3
# ============================================================================================
set -euo pipefail
GPU_MAIN="${1:-7}"
GPU_TWIN="${2:-3}"
# Garde-fou GPU par le FAIT, pas par des index figes : la liste noire d'index se perime en
# quelques minutes (nuit du 07/08 : 0/1/2 declares occupes le sont devenus libres, tandis
# qu'un job 5-shards d'un AUTRE utilisateur prenait 3-7). Regle appliquee ici, dans cet ordre :
#   (1) 5 et 6 : INTERDITS en dur (regle de la maison, l'index smi 6 est casse) ;
#   (2) toute carte portant le process d'un AUTRE utilisateur : refus (ne jamais perturber
#       un run sain qui n'est pas le notre) ;
#   (3) marge : >= 25 GB libres au moment MEME du lancement.
[ "$GPU_MAIN" = "$GPU_TWIN" ] && { echo "meme GPU pour les 2 vagues : refus"; exit 1; }
ME="$(id -un)"
for G in "$GPU_MAIN" "$GPU_TWIN"; do
  case "$G" in 5|6) echo "GPU $G INTERDIT (regle de la maison / smi casse) : refus"; exit 1;; esac
  UUID=$(nvidia-smi --query-gpu=uuid --format=csv,noheader -i "$G")
  for P in $(nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader \
             | awk -F', ' -v u="$UUID" '$1==u {print $2}'); do
    OWNER=$(ps -o user= -p "$P" 2>/dev/null | tr -d ' ')
    [ -n "$OWNER" ] && [ "$OWNER" != "$ME" ] && {
      echo "GPU $G porte le process $P de '$OWNER' (pas nous) : REFUS de co-run"; exit 1; }
  done
done
REPO=/shared/ssd_14T/home/adenelmi/Anchor-Forcing
PY=/shared/ssd_14T/home/adenelmi/miniconda3/envs/longlive/bin/python
PROMPTS="$REPO/reference/runs/frame_dec/prompts_noattr_v3.txt"
OUT=/shared/ssd_14T_project/adenelmi/prompts_noattr_v3_h
export CUDA_DEVICE_ORDER=PCI_BUS_ID
cd "$REPO"
mkdir -p "$OUT" "${OUT}_twins"

echo "=== garde-fou disque (>= 60 GB : la collecte h pese ~46 GB) ==="
df -h /shared/ssd_14T_project
AVAIL=$(df -BG --output=avail /shared/ssd_14T_project | tail -1 | tr -dc '0-9')
[ "$AVAIL" -lt 60 ] && { echo "disque projet : ${AVAIL} GB libres (< 60) : REFUS"; exit 1; }

echo "=== pre-check GPUs $GPU_MAIN et $GPU_TWIN (au meme instant que le lancement) ==="
nvidia-smi --query-gpu=index,memory.free,memory.used,utilization.gpu --format=csv
for G in "$GPU_MAIN" "$GPU_TWIN"; do
  FREE=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$G")
  [ "$FREE" -lt 25000 ] && { echo "GPU $G : ${FREE} MiB libres (< 25 GB) : REFUS"; exit 1; }
done

# ATTENTION : PAS de --no-h ici (c'est tout l'objet de l'etape). Tout le reste est identique
# a la generation d'origine, au flag pres.
COMMON=(reference/runs/frame_dec/collect_pairs.py --v3 --mode run
        --prompts-file "$PROMPTS" --prompts-base 9100000
        --frames-single 90 --videos-per-part 10 --check-videos 0
        --max-gb 90 --shard 0/1)

echo "=== vague MAIN (21 videos, h) sur GPU $GPU_MAIN + vague TWIN sur GPU $GPU_TWIN ==="
( CUDA_VISIBLE_DEVICES="$GPU_MAIN" $PY "${COMMON[@]}" --out "$OUT" \
    && CUDA_VISIBLE_DEVICES="$GPU_MAIN" $PY reference/runs/frame_dec/collect_pairs.py \
         --v3 --mode manifest --out "$OUT" ) \
  > "$OUT/collect_v3h_main.log" 2>&1 &
P_MAIN=$!
( CUDA_VISIBLE_DEVICES="$GPU_TWIN" $PY "${COMMON[@]}" --twin-offset 100000 \
    --out "${OUT}_twins" \
    && CUDA_VISIBLE_DEVICES="$GPU_TWIN" $PY reference/runs/frame_dec/collect_pairs.py \
         --v3 --mode manifest --out "${OUT}_twins" ) \
  > "${OUT}_twins/collect_v3h_twins.log" 2>&1 &
P_TWIN=$!

RC=0
wait "$P_MAIN" || { echo "vague MAIN en ECHEC (voir $OUT/collect_v3h_main.log)"; RC=1; }
wait "$P_TWIN" || { echo "vague TWIN en ECHEC (voir ${OUT}_twins/collect_v3h_twins.log)"; RC=1; }
tail -n 3 "$OUT/collect_v3h_main.log" "${OUT}_twins/collect_v3h_twins.log" || true
du -sh "$OUT" "${OUT}_twins" || true
df -h /shared/ssd_14T_project | tail -1
[ "$RC" -ne 0 ] && { echo "COLLECT_V3H_FAILED"; exit "$RC"; }
echo "COLLECT_V3H_DONE"
