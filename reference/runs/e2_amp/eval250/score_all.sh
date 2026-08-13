#!/bin/bash
# e2_amp step-250 scoring: DINO drift + VBench (subject_consistency + background_consistency), mem-ON vs mem-OFF.
set -u
AF=/shared/ssd_14T/home/adenelmi/Anchor-Forcing
ED=$AF/reference/runs/e2_amp/eval250
ENV=/shared/ssd_14T/home/adenelmi/miniconda3/envs/longlive
VB=/shared/ssd_14T/home/adenelmi/LongLive/vbench_repo
GPU="${1:-0}"
export CUDA_DEVICE_ORDER=PCI_BUS_ID

# wait for both arms fully generated (16 videos)
for i in $(seq 1 120); do
  n=$(( $(ls $ED/videos/memon/*.mp4 2>/dev/null | wc -l) + $(ls $ED/videos/memoff/*.mp4 2>/dev/null | wc -l) ))
  [ "$n" -ge 16 ] && break
  sleep 30
done
echo "gen: memon=$(ls $ED/videos/memon/*.mp4 2>/dev/null|wc -l) memoff=$(ls $ED/videos/memoff/*.mp4 2>/dev/null|wc -l)"

# ---------- DINO drift (the sensitive metric) ----------
cd $AF
$ENV/bin/python - <<'PY'
import json,os,glob
ED="/shared/ssd_14T/home/adenelmi/Anchor-Forcing/reference/runs/e2_amp/eval250"
labels=["jog","cycle","ballet","boy","skate","elder","surf","martial"]
items=[]
for d,arm in [("memon","mem_on"),("memoff","mem_off")]:
  for f in sorted(glob.glob(os.path.join(ED,"videos",d,"*.mp4"))):
    idx=int(os.path.basename(f).split("-")[0]); items.append({"path":os.path.abspath(f),"prompt":labels[idx],"arm":arm})
json.dump(items,open(os.path.join(ED,"manifest.json"),"w"))
print("manifest",len(items))
PY
CUDA_VISIBLE_DEVICES=$GPU $ENV/bin/python reference/runs/e0_reid/drift_metric.py \
  --manifest $ED/manifest.json --out $ED/drift_out --fps 4 --anchor_sec 2 --motion_floor 0.5 --device cuda:0 \
  > $ED/score_dino.log 2>&1
echo "=== DINO drift done (exit $?) ==="

# ---------- VBench SC + BC (the standard-but-blind metric) ----------
cd $VB
for arm in memon memoff; do
  for DIM in subject_consistency background_consistency; do
    CUDA_VISIBLE_DEVICES=$GPU PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
      $ENV/bin/python vbench2_beta_long/eval_long.py \
      --videos_path $ED/videos/$arm --dimension $DIM --mode long_custom_input \
      --output_path $ED/vbench_$arm > $ED/vbench_${arm}_${DIM}.log 2>&1
    echo "vbench $arm $DIM exit $?"
  done
done
echo "SCORE_ALL_DONE"
