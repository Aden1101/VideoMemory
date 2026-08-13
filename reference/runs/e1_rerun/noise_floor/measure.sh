#!/bin/bash
# Noise-floor measurement: mem-OFF + mem-DC at seeds 43,44,45 (seed 42 reused from eval1000), then score.
set -u
AF=/shared/ssd_14T/home/adenelmi/Anchor-Forcing
ND=$AF/reference/runs/e1_rerun/noise_floor
ENV=/shared/ssd_14T/home/adenelmi/miniconda3/envs/longlive
export CUDA_DEVICE_ORDER=PCI_BUS_ID
mkdir -p $ND/videos
cd $AF

# per-seed configs (off = MEM_ZERO_ALL, dc = MEM_DC_REMOVE at gen time)
for s in 43 44 45; do
  for arm in off dc; do
    base=$([ $arm = off ] && echo memoff || echo memon)
    sed "s/^seed: 42/seed: $s/" reference/runs/e1_rerun/eval1000/$base.yaml > $ND/mem${arm}_s$s.yaml
    sed -i "s#output_folder:.*#output_folder: $ND/videos/mem${arm}_s$s#" $ND/mem${arm}_s$s.yaml
  done
done

# launch 6 gens across GPUs 0,1,3,4,5
gpus=(0 1 3 4 5); i=0
for s in 43 44 45; do for arm in off dc; do
  g=${gpus[$((i%5))]}
  E=$([ $arm = off ] && echo "MEM_ZERO_ALL=1" || echo "MEM_DC_REMOVE=1 MEM_ZERO_ALL=0")
  rm -f $ND/videos/mem${arm}_s$s/*.mp4 2>/dev/null
  env $E CUDA_VISIBLE_DEVICES=$g $ENV/bin/torchrun --nproc_per_node=1 --master_port=$((29700+i)) \
    inference/inference.py --config_path $ND/mem${arm}_s$s.yaml > $ND/gen_mem${arm}_s$s.log 2>&1 &
  i=$((i+1))
  [ $((i%5)) -eq 0 ] && wait
done; done
wait
echo "GENS_DONE ($(ls $ND/videos/*/*.mp4 2>/dev/null | wc -l) videos)"

echo "=== NOISE-FLOOR SCORE ==="
CUDA_VISIBLE_DEVICES=0 $ENV/bin/python $ND/noise_score.py
echo "NOISE_FLOOR_DONE"
