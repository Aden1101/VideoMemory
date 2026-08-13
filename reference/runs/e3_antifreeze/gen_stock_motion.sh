#!/bin/bash
# e3 motion-target precompute, step 1/2: generate stock-AF reference videos (3 seeds x 8 prompts,
# 300 latent frames) on ONE gpu, then step 2/2: build motion_targets.json.
# usage: gen_stock_motion.sh <cvd_index>
set -u
GPU="${1:?need gpu index}"
cd /shared/ssd_14T/home/adenelmi/Anchor-Forcing
ENV=/shared/ssd_14T/home/adenelmi/miniconda3/envs/longlive
export CUDA_DEVICE_ORDER=PCI_BUS_ID
ED=reference/runs/e3_antifreeze

for SEED in 42 43 44; do
  CFG="$ED/.stockmotion_seed$SEED.yaml"
  sed -e "s/^seed: 42/seed: $SEED/" -e "s#/seed42#/seed$SEED#" "$ED/stockmotion.yaml" > "$CFG"
  mkdir -p "$ED/stock_videos/seed$SEED"
  echo "=== stock-AF gen seed $SEED on CVD $GPU ==="
  CUDA_VISIBLE_DEVICES=$GPU "$ENV/bin/torchrun" --nproc_per_node=1 --master_port=2971$((SEED-42)) \
    inference/inference.py --config_path "$CFG" > "$ED/gen_seed$SEED.log" 2>&1
  echo "seed $SEED exit $? ($(ls "$ED/stock_videos/seed$SEED"/*.mp4 2>/dev/null | wc -l) vids)"
done

echo "=== build motion_targets.json ==="
CUDA_VISIBLE_DEVICES=$GPU "$ENV/bin/python" "$ED/precompute_motion_targets.py" \
  --video_dirs "$ED/stock_videos/seed42" "$ED/stock_videos/seed43" "$ED/stock_videos/seed44" \
  --prompts reference/runs/e0_reid/prompts/e0_dynamic8.txt \
  --out "$ED/motion_targets.json" > "$ED/precompute.log" 2>&1
echo "precompute exit $?"
tail -3 "$ED/precompute.log"
echo "E3_MOTION_TARGETS_DONE"
