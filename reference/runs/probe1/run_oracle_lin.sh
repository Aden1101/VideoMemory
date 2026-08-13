#!/usr/bin/env bash
# ============================================================================================
# ORACLE DE LA REGLE D'ECRITURE LINEAIRE -- LE RUN COMPLET (registre : section "ORACLE DE LA
# REGLE D'ECRITURE LINEAIRE", GO Aden 2026-08-06 soir, seuils scelles AVANT implementation).
#
# NE LANCER QU'APRES : (1) selftest_lin.py TOUT PASSE ; (2) la relecture de code (le
# --audited des CV ATTESTE cette relecture, il ne la remplace pas).
#
# 4 bras, ZERO parametre appris, mode STREAM (aucun W sur disque) :
#   L1 lin_mean_subject  (PRIMARY, seuil scelle : strong19 >= 14/19)
#   L2 lin_mean_all      L3 lin_delta_subject      L4 lin_delta_all
# Signatures : memes sondes, memes ages, meme compression ; sondage CV 6 plis, pool, v2,
# SANS pre-entrainement, memes plis que le verdict legal.
#
# Usage :  bash reference/runs/probe1/run_oracle_lin.sh 7
# ============================================================================================
set -euo pipefail
GPU="${1:-7}"
[ "$GPU" = "6" ] && { echo "index smi 6 est CASSE, refus"; exit 1; }
REPO=/shared/ssd_14T/home/adenelmi/Anchor-Forcing
PY=/shared/ssd_14T/home/adenelmi/miniconda3/envs/longlive/bin/python
HEAVY=/shared/ssd_14T_project/adenelmi/probe1
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="$GPU" HF_HOME="$HEAVY/hf"
cd "$REPO"

echo "=== pre-check GPU $GPU (au meme instant que le lancement) ==="
nvidia-smi --query-gpu=index,memory.free,memory.used,utilization.gpu --format=csv
FREE=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$GPU")
[ "$FREE" -lt 15000 ] && { echo "GPU $GPU : ${FREE} MiB libres, REFUS"; exit 1; }

echo "=== 1) SIGNATURES (rejeu lineaire a la volee, L1..L4) ==="
for SRC in lin_mean_subject lin_mean_all lin_delta_subject lin_delta_all; do
  $PY reference/runs/probe1/build_signatures.py --corpus v2 --source "$SRC" --stream
done

echo "=== 2) SONDAGE CV (6 plis, pool, v2, SANS pre-entrainement, 4 bras) ==="
for SRC in lin_mean_subject lin_mean_all lin_delta_subject lin_delta_all; do
  $PY reference/runs/probe1/train_probe.py --stage cv --audited --source "$SRC" \
      --target subject --features pool --corpus v2
done

echo "=== resume (strong19 a lire contre les seuils scelles ; L1 = PRIMARY >= 14/19) ==="
$PY - <<'EOF'
import json
R = "/shared/ssd_14T_project/adenelmi/probe1/runs"
for arm, src in (("L1 mean/sujet (PRIMARY)", "lin_mean_subject"),
                 ("L2 mean/tous          ", "lin_mean_all"),
                 ("L3 delta/sujet        ", "lin_delta_subject"),
                 ("L4 delta/tous         ", "lin_delta_all")):
    d = json.load(open(f"{R}/cv_{src}_subject_pool_v2.json"))
    s19, s25, pm = (d["summary"].get(k) for k in ("strong19", "valid25", "permuted"))
    print(f"{arm}: strong19 {s19['n_pairs_correct']:.1f}/19 (pair {s19['acc_pair']:.3f}, "
          f"p={s19['wilcoxon_pair_p']:.3g}) | valid25 {s25['n_pairs_correct']:.1f}/25 | "
          f"permute {pm['acc_pair_25perm']:.3f}")
EOF
echo "ORACLE_LIN_DONE"
