#!/usr/bin/env bash
# ============================================================================================
# ANNEXE DE ROBUSTESSE 10 GRAINES (demandee par le coordinateur le 07/08, DECLAREE AVANT tout
# score MLP ; motif : le controle de device a montre une variabilite de chaine d'environ 2
# paires sur strong19, du meme ordre que les ecarts qu'on interprete).
#
# CE QUE C'EST : les memes bras, la MEME configuration, seule la liste de graines change
# (0..9 au lieu de 0,1,2). Les runs 3 graines SCELLES restent la reference : rien n'est retire
# ni remplace, cette annexe les ACCOMPAGNE. Les sorties portent donc des noms distincts
# (annex_seeds10_*.json) et n'ecrasent aucun json scelle.
# BRAS : les deux ancres (`state` = A0 et `lin_delta_all`) puis, si le budget le permet, le
# PRIMARY MLP (A3 mlp_full_muon). L'ordre est celui-la : les ancres d'abord.
#
# Co-run autorise pour les SONDAGES (regle scellee 07/08) : marge >= 8 GB, jamais l'index 6.
# Usage : setsid nohup bash reference/runs/probe1/annex_seeds10.sh <GPU> > <log> 2>&1 &
# ============================================================================================
set -uo pipefail
GPU="${1:-7}"
[ "$GPU" = "6" ] && { echo "index smi 6 est CASSE, refus"; exit 1; }
REPO=/shared/ssd_14T/home/adenelmi/Anchor-Forcing
PY=/shared/ssd_14T/home/adenelmi/miniconda3/envs/longlive/bin/python
HEAVY=/shared/ssd_14T_project/adenelmi/probe1
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="$GPU" HF_HOME="$HEAVY/hf"
cd "$REPO"

# l'annexe passe APRES les 4 sondages MLP (c'est ce qu'attend Aden en premier)
while true; do
  n=0
  for s in mlp_full_muon mlp_delta_all mlp_full mlp_hebb_seq; do
    [ -f "$HEAVY/runs/cv_${s}_subject_pool_v2.json" ] && n=$((n+1))
  done
  [ "$n" -eq 4 ] && break
  sleep 60
done
echo "[annexe] les 4 sondages MLP sont deposes, demarrage a $(date)"

FREE=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$GPU")
[ "$FREE" -lt 8000 ] && { echo "GPU $GPU : ${FREE} MiB libres (< 8 GB), REFUS"; exit 1; }
echo "[annexe] GPU $GPU, ${FREE} MiB libres (co-run sondage autorise)"

for SRC in state lin_delta_all mlp_full_muon; do
  echo "=== ANNEXE 10 GRAINES : $SRC (config identique, seules les graines changent) ==="
  $PY reference/runs/probe1/train_probe.py --stage cv --audited --source "$SRC" \
      --target subject --features pool --corpus v2 --seeds 0,1,2,3,4,5,6,7,8,9 \
      --out "$HEAVY/runs/annex_seeds10_${SRC}_subject_pool_v2.json"
done
echo "ANNEX_SEEDS10_DONE"
