#!/usr/bin/env bash
# ============================================================================================
# ETAPE 2 DE LA CHAINE C2 : SIGNATURES DU CORPUS v3, puis PURGE DES h.
#
# Deux sources SEULEMENT, protocole identique a v2 (memes ages, memes sondes, meme
# compression) :
#   - lin_delta_all    : LE bras teste (etat W delta-rule tous-tokens ; --stream obligatoire,
#                        W n'est jamais materialise) ;
#   - hdirect_masked   : la BORNE DE VALIDITE d'instrument, exigee sur le meme materiau.
#
# Les signatures sont indexees par video_id (main:91000xx / twin:91000xx), IDENTIQUES entre
# les corpus `v3` (sans h, conserve) et `v3h` (avec h, transitoire) : une fois les h purges,
# l'etape 3 lit ces memes fichiers via `--test-corpus v3`.
#
# Usage : bash reference/runs/probe1/run_sig_v3h.sh 0 [nopurge]
# ============================================================================================
set -euo pipefail
GPU="${1:-0}"
NOPURGE="${2:-}"
REPO=/shared/ssd_14T/home/adenelmi/Anchor-Forcing
PY=/shared/ssd_14T/home/adenelmi/miniconda3/envs/longlive/bin/python
HEAVY=/shared/ssd_14T_project/adenelmi/probe1
HDIR=/shared/ssd_14T_project/adenelmi/prompts_noattr_v3_h
export CUDA_DEVICE_ORDER=PCI_BUS_ID HF_HOME="$HEAVY/hf"
cd "$REPO"

# meme garde-fou que la collecte : 5/6 interdits, jamais la carte d'un AUTRE utilisateur,
# marge >= 15 GB (les signatures sont bien plus legeres qu'un rollout).
case "$GPU" in 5|6) echo "GPU $GPU INTERDIT : refus"; exit 1;; esac
ME="$(id -un)"
UUID=$(nvidia-smi --query-gpu=uuid --format=csv,noheader -i "$GPU")
for P in $(nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader \
           | awk -F', ' -v u="$UUID" '$1==u {print $2}'); do
  OWNER=$(ps -o user= -p "$P" 2>/dev/null | tr -d ' ')
  [ -n "$OWNER" ] && [ "$OWNER" != "$ME" ] && {
    echo "GPU $GPU porte le process $P de '$OWNER' : REFUS"; exit 1; }
done
echo "=== pre-check GPU $GPU (au meme instant que le lancement) ==="
nvidia-smi --query-gpu=index,memory.free,memory.used,utilization.gpu --format=csv
FREE=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$GPU")
[ "$FREE" -lt 15000 ] && { echo "GPU $GPU : ${FREE} MiB libres : REFUS"; exit 1; }
export CUDA_VISIBLE_DEVICES="$GPU"

echo "=== CONTROLE D'IDENTITE (obligatoire AVANT toute signature) ==="
$PY reference/runs/probe1/selftest_v3h.py 2>&1 | grep -viE "futurewarning|warnings.warn"

echo "=== 1) signatures lin_delta_all (--stream) ==="
$PY reference/runs/probe1/build_signatures.py --corpus v3h --source lin_delta_all --stream
echo "=== 2) signatures hdirect_masked (borne de validite) ==="
$PY reference/runs/probe1/build_signatures.py --corpus v3h --source hdirect_masked

echo "=== verification : 42 videos x 4 ages = 168 signatures par source ==="
$PY - <<'EOF'
import json, sys, os
sys.path.insert(0, "reference/runs/probe1")
import probe1_common as C
from build_signatures import sig_path
_, pairs = C.pair_dataset(corpus="v3", verbose=False)
vids = C.load_index(corpus="v3", verbose=False)
byid = C.vids_by_id(vids)
ok = True
for src in ("lin_delta_all", "hdirect_masked"):
    n = miss = 0
    for p in pairs:
        for role in ("main", "twin"):
            for a in C.state_ages(byid[p[role]]):
                n += 1
                if not os.path.exists(sig_path(src, p[role], a["name"])):
                    miss += 1
    print(f"  {src:16s} {n - miss}/{n} signatures" + (f"  MANQUANTES: {miss}" if miss else ""))
    ok &= miss == 0
sys.exit(0 if ok else 1)
EOF

if [ "$NOPURGE" = "nopurge" ]; then
  echo "=== purge des h SAUTEE (nopurge) ==="
else
  echo "=== 3) PURGE des h transitoires (les signatures sont en boite) ==="
  du -sh "$HDIR" "${HDIR}_twins" || true
  rm -rf "$HDIR" "${HDIR}_twins"
  rm -f "$HEAVY/index_v3h.json"
  echo "    purges. Le corpus v3 (sans h, avec les latents de reference) est CONSERVE."
fi
df -h /shared/ssd_14T_project | tail -1
echo "SIG_V3H_DONE"
