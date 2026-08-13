#!/usr/bin/env bash
# Demarre les deux moities de la campagne echelle de fidelite, detachees de la session
# (setsid + nohup) pour survivre au cycle de vie des taches de l'agent. Idempotent :
# ne relance pas une moitie deja en cours.
set -u
LOGS=/shared/ssd_14T_project/adenelmi/fidelity_ladder/logs
HERE="$(cd "$(dirname "$0")" && pwd)"
mkdir -p "$LOGS"
for H in A B; do
  if pgrep -f "launch_fidelity.sh $H" > /dev/null 2>&1; then
    echo "moitie $H deja en cours"
    continue
  fi
  setsid nohup bash "$HERE/launch_fidelity.sh" "$H" \
    > "$LOGS/launcher$H.out" 2>&1 < /dev/null &
  disown || true
  echo "moitie $H lancee (pid $!)"
  sleep 5
done
pgrep -af "launch_fidelity.sh" || true
