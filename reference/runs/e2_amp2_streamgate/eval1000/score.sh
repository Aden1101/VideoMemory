#!/bin/bash
# streamgate step-1000 3-criteria eval: mem-ON(β1.0) / mem-OFF / mem-β0.5, DINO drift.
set -u
AF=/shared/ssd_14T/home/adenelmi/Anchor-Forcing
SED=$AF/reference/runs/e2_amp2_streamgate/eval1000
ENV=/shared/ssd_14T/home/adenelmi/miniconda3/envs/longlive
GPU="${1:-0}"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
cd $AF
gen() {  # $1=arm-dir $2=envstring $3=port
  rm -f $SED/videos/$1/*.mp4 2>/dev/null
  env $2 CUDA_VISIBLE_DEVICES=$GPU $ENV/bin/torchrun --nproc_per_node=1 --master_port=$3 \
    inference/inference.py --config_path $SED/$1.yaml > $SED/gen_$1.log 2>&1
  echo "$1 exit $? ($(ls $SED/videos/$1/*.mp4 2>/dev/null|wc -l) vids)"
}
gen memon  "MEM_STREAM_SCALE=1.0 MEM_ZERO_ALL=0" 29691
gen memoff "MEM_STREAM_SCALE=1.0 MEM_ZERO_ALL=1" 29692
gen memb05 "MEM_STREAM_SCALE=0.5 MEM_ZERO_ALL=0" 29693
# --- DINO drift, 3 arms ---
$ENV/bin/python - <<'PY'
import json,os,glob
SED="/shared/ssd_14T/home/adenelmi/Anchor-Forcing/reference/runs/e2_amp2_streamgate/eval1000"
labels=["jog","cycle","ballet","boy","skate","elder","surf","martial"]
items=[]
for d,arm in [("memon","mem_on"),("memoff","mem_off"),("memb05","mem_b05")]:
  for f in sorted(glob.glob(os.path.join(SED,"videos",d,"*.mp4"))):
    idx=int(os.path.basename(f).split("-")[0]); items.append({"path":os.path.abspath(f),"prompt":labels[idx],"arm":arm})
json.dump(items,open(os.path.join(SED,"manifest.json"),"w")); print("manifest",len(items))
PY
CUDA_VISIBLE_DEVICES=$GPU $ENV/bin/python reference/runs/e0_reid/drift_metric.py \
  --manifest $SED/manifest.json --out $SED/drift_out --fps 4 --anchor_sec 2 --motion_floor 0.5 --device cuda:0 \
  > $SED/score_dino.log 2>&1
echo "dino exit $?"
echo "STREAMGATE_EVAL_DONE"
