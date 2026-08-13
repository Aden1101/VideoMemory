#!/bin/bash
# Silent waiter: exits when all E0 generation processes are gone (then the session is re-invoked).
cd /shared/ssd_14T/home/adenelmi/Anchor-Forcing
for i in $(seq 1 300); do   # cap ~7.5h (300 * 90s)
  running=$(pgrep -af "inference/inference.py" 2>/dev/null | grep -c "e0_reid/configs")
  n=$(ls reference/runs/e0_reid/videos/*/*.mp4 2>/dev/null | wc -l)
  if [ "$running" -eq 0 ]; then
    echo "E0_GEN_COMPLETE running=0 mp4=$n"
    for a in stock_sink3 stock_sink9 stock_sink21 memon_sink3; do
      echo "  $a: $(ls reference/runs/e0_reid/videos/$a/*.mp4 2>/dev/null | wc -l)/8"
    done
    exit 0
  fi
  sleep 90
done
echo "E0_WAITER_TIMEOUT mp4=$(ls reference/runs/e0_reid/videos/*/*.mp4 2>/dev/null | wc -l)"
