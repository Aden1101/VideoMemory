#!/usr/bin/env bash
# ============================================================================================
# ORACLE D'ECRITURE MASQUEE -- LE RUN COMPLET (registre : "ORACLE D'ECRITURE MASQUEE",
# GO Aden 2026-08-06, seuils scelles AVANT implementation).
#
# NE LANCER QU'APRES : (1) selftest_maskwrite.py TOUT PASSE ; (2) la relecture de code
# (le --audited des CV ci-dessous ATTESTE cette relecture, il ne la remplace pas).
#
# 3 bras, 60 videos v2, protocole de signatures INCHANGE (memes ages, memes sondes),
# CV 6 plis memes plis que le verdict legal (graine SPLIT_SEED par corpus_idx), pool,
# SANS pre-entrainement pour les TROIS bras (le bras A original est RE-COURU ici meme).
#
# Usage :  bash reference/runs/probe1/run_oracle_maskwrite.sh 7 [disk|stream]
#   disk   (defaut) = materialise les etats sous oracle_maskwrite/states_{subject,bg_matched}
#                     (~38 GB PAR bras, exige >= 95 GB libres sur le disque projet)
#   stream          = rejoue les ecritures masquees a la volee, ZERO octet d'etat
#                     (meme trajectoire bit a bit : replay_video est le meme code)
# ============================================================================================
set -euo pipefail
GPU="${1:-7}"
MODE="${2:-disk}"
[ "$GPU" = "6" ] && { echo "index smi 6 est CASSE, refus"; exit 1; }
[[ "$MODE" == "disk" || "$MODE" == "stream" ]] || { echo "mode $MODE inconnu"; exit 1; }
REPO=/shared/ssd_14T/home/adenelmi/Anchor-Forcing
PY=/shared/ssd_14T/home/adenelmi/miniconda3/envs/longlive/bin/python
HEAVY=/shared/ssd_14T_project/adenelmi/probe1
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="$GPU" HF_HOME="$HEAVY/hf"
cd "$REPO"

echo "=== pre-check GPU $GPU (au meme instant que le lancement) ==="
nvidia-smi --query-gpu=index,memory.free,memory.used,utilization.gpu --format=csv
FREE=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$GPU")
[ "$FREE" -lt 15000 ] && { echo "GPU $GPU : ${FREE} MiB libres, REFUS"; exit 1; }

AVAIL=$(df -BG --output=avail /shared/ssd_14T_project | tail -1 | tr -dc '0-9')
echo "disque projet : ${AVAIL} GB libres"
if [ "$MODE" = "disk" ] && [ "$AVAIL" -lt 95 ]; then
  echo "REFUS : le mode disk ecrit ~76 GB (2 bras x 38 GB) ; relance en mode stream"
  exit 1
fi

echo "=== 1) ETATS REJOUES (bras subject puis bg_matched) ==="
if [ "$MODE" = "disk" ]; then
  $PY reference/runs/probe1/build_states.py --corpus v2 --only_pairs --write-mask subject
  $PY reference/runs/probe1/build_states.py --corpus v2 --only_pairs --write-mask bg_matched
  echo "=== 2) SIGNATURES (protocole inchange, lues depuis les nouveaux repertoires) ==="
  $PY reference/runs/probe1/build_signatures.py --corpus v2 --source state \
      --states-dir "$HEAVY/oracle_maskwrite/states_subject"    --sig-tag state_mw_subject
  $PY reference/runs/probe1/build_signatures.py --corpus v2 --source state \
      --states-dir "$HEAVY/oracle_maskwrite/states_bg_matched" --sig-tag state_mw_bg_matched
else
  echo "(mode stream : etats rejoues a la volee, rien n'est materialise)"
  echo "=== 2) SIGNATURES (rejeu masque a la volee) ==="
  $PY reference/runs/probe1/build_signatures.py --corpus v2 --source state --stream \
      --write-mask subject    --sig-tag state_mw_subject
  $PY reference/runs/probe1/build_signatures.py --corpus v2 --source state --stream \
      --write-mask bg_matched --sig-tag state_mw_bg_matched
fi

echo "=== 3) SONDAGE CV (6 plis, pool, v2, SANS pre-entrainement, 3 bras) ==="
# bras A (original) RE-COURU dans la config exacte de la comparaison (armes egales) ;
# sortie cv_state_subject_pool_v2.json : n'ecrase PAS le verdict legal (qui est _pre_).
$PY reference/runs/probe1/train_probe.py --stage cv --audited --source state \
    --target subject --features pool --corpus v2
# bras B (sujet-seul) -- PRIMARY scelle : strong19 >= 14/19
$PY reference/runs/probe1/train_probe.py --stage cv --audited --source state_mw_subject \
    --target subject --features pool --corpus v2
# bras C (fond-apparie) -- DISSOCIATION scellee : B - C >= 3 paires
$PY reference/runs/probe1/train_probe.py --stage cv --audited --source state_mw_bg_matched \
    --target subject --features pool --corpus v2

echo "=== resume (les 3 lignes strong19 a lire contre les seuils scelles) ==="
$PY - <<'EOF'
import json
R = "/shared/ssd_14T_project/adenelmi/probe1/runs"
for arm, f in (("A original  ", "cv_state_subject_pool_v2.json"),
               ("B sujet-seul", "cv_state_mw_subject_subject_pool_v2.json"),
               ("C fond-app. ", "cv_state_mw_bg_matched_subject_pool_v2.json")):
    d = json.load(open(f"{R}/{f}"))
    s19, s25 = d["summary"].get("strong19"), d["summary"].get("valid25")
    print(f"{arm}: strong19 {s19['n_pairs_correct']:.1f}/19 (pair {s19['acc_pair']:.3f}, "
          f"p={s19['wilcoxon_pair_p']:.3g}) | valid25 {s25['n_pairs_correct']:.1f}/25 | "
          f"permute {d['summary']['permuted']['acc_pair_25perm']:.3f}")
EOF
echo "ORACLE_MASKWRITE_DONE"
