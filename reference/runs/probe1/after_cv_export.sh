#!/usr/bin/env bash
# Chaine automatiquement l'export des marges (export_verdict_arms.py) des que les 4 sondages
# MLP ont depose leurs json : le livrable "planche" se termine tout seul apres le CV.
set -uo pipefail
R=/shared/ssd_14T_project/adenelmi/probe1/runs
PY=/shared/ssd_14T/home/adenelmi/miniconda3/envs/longlive/bin/python
cd /shared/ssd_14T/home/adenelmi/Anchor-Forcing
while true; do
  n=0
  for s in mlp_full_muon mlp_delta_all mlp_full mlp_hebb_seq; do
    [ -f "$R/cv_${s}_subject_pool_v2.json" ] && n=$((n+1))
  done
  if [ "$n" -eq 4 ]; then
    echo "[chainage] les 4 sondages sont la ($(date)) -> export des marges"
    exec $PY reference/runs/probe1/export_verdict_arms.py
  fi
  sleep 60
done
