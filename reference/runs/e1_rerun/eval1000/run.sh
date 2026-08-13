#!/bin/bash
# step-1000 mem-ON vs mem-OFF ablation on one GPU. usage: run.sh <cvd_index>
set -u
GPU="${1:?need gpu index}"
cd /shared/ssd_14T/home/adenelmi/Anchor-Forcing
ENV=/shared/ssd_14T/home/adenelmi/miniconda3/envs/longlive
export CUDA_DEVICE_ORDER=PCI_BUS_ID
ED=reference/runs/e1_rerun/eval1000
JOBTMP=/shared/ssd_14T/home/adenelmi/.claude/jobs/3118b292/tmp
rm -f "$ED"/videos/memon/*.mp4 "$ED"/videos/memoff/*.mp4 2>/dev/null

echo "=== gen mem-ON on CVD $GPU ==="
MEM_ZERO_ALL=0 CUDA_VISIBLE_DEVICES=$GPU "$ENV/bin/torchrun" --nproc_per_node=1 --master_port=29660 \
  inference/inference.py --config_path "$ED/memon.yaml" > "$ED/gen_memon.log" 2>&1
echo "memon exit $? ($(ls "$ED/videos/memon"/*.mp4 2>/dev/null | wc -l) vids)"
echo "=== gen mem-OFF on CVD $GPU ==="
MEM_ZERO_ALL=1 CUDA_VISIBLE_DEVICES=$GPU "$ENV/bin/torchrun" --nproc_per_node=1 --master_port=29661 \
  inference/inference.py --config_path "$ED/memoff.yaml" > "$ED/gen_memoff.log" 2>&1
echo "memoff exit $? ($(ls "$ED/videos/memoff"/*.mp4 2>/dev/null | wc -l) vids)"

echo "=== build manifest + score ==="
"$ENV/bin/python" - "$ED" <<'PY'
import json, os, sys, glob
ED=sys.argv[1]
labels=["jog","cycle","ballet","boy","skate","elder","surf","martial"]
items=[]
for arm in ["memon","memoff"]:
    for f in sorted(glob.glob(os.path.join(ED,"videos",arm,"*.mp4"))):
        idx=int(os.path.basename(f).split("-")[0])
        items.append({"path":os.path.abspath(f),"prompt":labels[idx],"arm":"mem_on" if arm=="memon" else "mem_off"})
json.dump(items, open(os.path.join(ED,"manifest.json"),"w"), indent=1)
print(f"manifest {len(items)} items")
PY
CUDA_VISIBLE_DEVICES=$GPU "$ENV/bin/python" reference/runs/e0_reid/drift_metric.py \
  --manifest "$ED/manifest.json" --out "$ED/drift_out" --fps 4 --anchor_sec 2 --motion_floor 0.5 --device cuda:0 \
  > "$ED/score.log" 2>&1
echo "scorer exit $?"
echo "EVAL1000_DONE"
