#!/usr/bin/env bash
# ============================================================================================
# ARM MLP-CORRECTEUR -- LE RUN (registre : sections "ARM MLP-CORRECTEUR", "AMENDEMENT (Aden
# 07/08)" et "REPARATION DU CRITERE AVEUGLE" ; seuil scelle : strong19 >= 14/19 sur le PRIMARY,
# bras EXPLORATOIRE, multiplicite a corriger au rapport).
#
# NE LANCER QU'APRES : (1) selftest_mlp_delta.py TOUT PASSE (D0-D6) ; (2) la relecture de code
# d'Aden (le --audited des CV ATTESTE cette relecture, il ne la remplace pas) ; (3) le GO.
# PRIORITE GPU : la replication v4 passe AVANT ce bras.
# LE SONDAGE (CV) EST VERROUILLE : ce script ne fait que les SIGNATURES par defaut. Il faut
# --with-cv, explicitement, pour lancer les CV -- et ce drapeau ne doit etre pose qu'apres le GO
# du coordinateur (qui vient apres le verdict de la replication v4).
#
# PLAN, apres l'arbitrage du 07/08 ("REPARATION DU CRITERE AVEUGLE"). ZERO entrainement, mode
# STREAM (aucun etat sur disque), TOUS les tokens, corpus v2 :
#   A3 mlp_full_muon   residu + sequentiel 512 + Muon  <- PRIMARY (seuil >= 14/19)
#   A1 mlp_delta_all   residu + moyenne (1 pas) + Muon    isole le CALENDRIER (S)
#   A4 mlp_hebb_seq    Hebb   + sequentiel 512 + Muon     isole l'OBJECTIF (R), calendrier et
#                                                         pas APPARIES au PRIMARY
#   A2' mlp_full       residu + sequentiel 32 + BRUT x0.5 isole MUON (M) ; ecrit (part 1.03)
#                                                         mais ne descend pas (R = -0.577)
# (A0 = notre ecriture actuelle = le bras `state`, deja mesure 8.00/19 : JAMAIS relance.)
# DISSYMETRIE DECLAREE : tous les bras sont en micro-lot 512 SAUF A2', qui ne peut pas y etre
# (a 512 le brut a part ecrite 1.04 et R = -124). Contrainte physique du bras, pas opportunisme.
# LES DEUX CHIFFRES AVEUGLES DE CHAQUE BRAS (sweep_mlp_step.py --arms, runs/sweep_mlp_arms.json ;
# a rapporter A COTE des scores) : R = reduction relative moyenne de ||f_W(k)-v|| par ecriture,
# part = poids de l'ecriture dans l'etat (plancher scelle 0.5) --
#   A3 R = +0.0251 part 0.952 | A1 R = -0.1737 part 0.728 | A2' R = -0.5772 part 1.034 |
#   A4 R = -4.6205 part 0.947 | A0 (reference) R = -0.3596 part 0.883
# PLAFOND DECLARE : le pur sequentiel (micro-lot 1) est HORS BUDGET (~3.9 h/bras) : non couru.
# COUT MESURE (selftest, par video, 3 blocs x 4 ages) : mlp_delta_all 2.1 s, mlp_full 8.8 s,
# mlp_full_muon 9.4 s, mlp_hebb_seq 9.5 s -> ~30 min de signatures pour les 60 videos des 4 bras.
# Signatures : memes sondes, memes ages, meme compression, MEME lecture (swiglu_apply) que le
# bras `state` ; sondage CV 6 plis, pool, v2, SANS pre-entrainement, memes plis, 3 graines.
#
# Usage :  bash reference/runs/probe1/run_mlp_delta.sh 0            # signatures seules
#          bash reference/runs/probe1/run_mlp_delta.sh 0 --with-cv  # + CV, APRES LE GO
set -euo pipefail
GPU="${1:-2}"
WITH_CV="${2:-}"
[ "$GPU" = "6" ] && { echo "index smi 6 est CASSE, refus"; exit 1; }
ARMS="mlp_full_muon mlp_delta_all mlp_hebb_seq mlp_full"   # PRIMARY d'abord
REPO=/shared/ssd_14T/home/adenelmi/Anchor-Forcing
PY=/shared/ssd_14T/home/adenelmi/miniconda3/envs/longlive/bin/python
HEAVY=/shared/ssd_14T_project/adenelmi/probe1
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="$GPU" HF_HOME="$HEAVY/hf"
cd "$REPO"

echo "=== pre-check GPU $GPU (au meme instant que le lancement) ==="
nvidia-smi --query-gpu=index,memory.free,memory.used,utilization.gpu --format=csv
FREE=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$GPU")
# Marge exigee : 25 GB tant qu'il reste des SIGNATURES a construire (rejeu du modele + h), mais
# 8 GB si elles sont deja toutes sur disque et qu'il ne reste que le SONDAGE. C'est la regle de
# co-run scellee au registre le 07/08 : un sondage tient dans moins de 2 GB (240 signatures de
# 1536 nombres), la marge de 25 GB visait la generation et l'entrainement, pas les CV.
NEED=8000
for SRC in mlp_full_muon mlp_delta_all mlp_hebb_seq mlp_full; do
  [ -f "$HEAVY/sig/_index_${SRC}_v2.json" ] || NEED=25000
done
[ "$FREE" -lt "$NEED" ] && { echo "GPU $GPU : ${FREE} MiB libres (< ${NEED} MiB requis), REFUS"; exit 1; }
echo "    marge requise ${NEED} MiB (8 GB = sondage seul, co-run autorise ; 25 GB = signatures a batir)"

echo "=== 1) SIGNATURES (rejeu a la volee, les 4 bras du plan, PRIMARY d'abord) ==="
for SRC in $ARMS; do
  $PY reference/runs/probe1/build_signatures.py --corpus v2 --source "$SRC" --stream
done

if [ "$WITH_CV" != "--with-cv" ]; then
  echo "=== 2) SONDAGE CV : NON LANCE (verrou). Aucun score d'identite sans le GO du"
  echo "    coordinateur, qui vient APRES le verdict de la replication v4."
  echo "    Pour le lancer ensuite :  bash $0 $GPU --with-cv"
  echo "MLP_SIGNATURES_DONE"
  exit 0
fi

echo "=== 2) SONDAGE CV (6 plis, pool, v2, SANS pre-entrainement, 3 graines) ==="
for SRC in $ARMS; do
  $PY reference/runs/probe1/train_probe.py --stage cv --audited --source "$SRC" \
      --target subject --features pool --corpus v2
done

echo "=== resume (strong19 contre le seuil scelle >= 14/19 ; EXPLORATOIRE, p_corr = 8 p) ==="
$PY - <<'EOF'
import json
R = "/shared/ssd_14T_project/adenelmi/probe1/runs"
rows = [("A3 mlp_full_muon PRIM", "mlp_full_muon"),
        ("A1 mlp_delta_all     ", "mlp_delta_all"),
        ("A4 mlp_hebb_seq      ", "mlp_hebb_seq"),
        ("A2' mlp_full (brut)  ", "mlp_full"),
        ("(ref) lin_delta_all  ", "lin_delta_all"),
        ("(ref A0) state       ", "state")]
for arm, src in rows:
    try:
        d = json.load(open(f"{R}/cv_{src}_subject_pool_v2.json"))
    except FileNotFoundError:
        print(f"{arm}: (pas de run)")
        continue
    s19, s25, pm = (d["summary"].get(k) for k in ("strong19", "valid25", "permuted"))
    print(f"{arm}: strong19 {s19['n_pairs_correct']:.2f}/19 (pair {s19['acc_pair']:.3f}, "
          f"p={s19['wilcoxon_pair_p']:.3g}) | valid25 {s25['n_pairs_correct']:.2f}/25 | "
          f"permute {pm['acc_pair_25perm']:.3f}")
EOF
echo "MLP_DELTA_DONE"
