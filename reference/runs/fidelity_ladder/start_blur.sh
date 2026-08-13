#!/usr/bin/env bash
# Demarre l'echelle de flou fine, detachee de la session (survit au cycle de vie de l'agent).
set -u
LOGS=/shared/ssd_14T_project/adenelmi/fidelity_ladder/logs
HERE="$(cd "$(dirname "$0")" && pwd)"
mkdir -p "$LOGS"
if pgrep -f '^bash [^ ]*launch_blur\.sh$' > /dev/null 2>&1; then
  echo "deja en cours"; pgrep -af '^bash [^ ]*launch_blur\.sh$'; exit 0
fi
setsid nohup bash "$HERE/launch_blur.sh" > "$LOGS/launcher_blur.out" 2>&1 < /dev/null &
disown || true
echo "lance (pid $!)"
