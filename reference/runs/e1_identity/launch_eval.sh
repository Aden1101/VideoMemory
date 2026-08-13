#!/bin/bash
# E1 eval: generate mem-on + mem-zero for checkpoints 150 & 300 across 4 GPUs, then the drift ruler.
set -u
export CUDA_DEVICE_ORDER=PCI_BUS_ID
cd /shared/ssd_14T/home/adenelmi/Anchor-Forcing
ENV=/shared/ssd_14T/home/adenelmi/miniconda3/envs/longlive
CFG=reference/runs/e1_identity/eval/configs
LOG=reference/runs/e1_identity/eval/out
mkdir -p "$LOG"

# arm -> (gpu, memzero?)  ; memzero arm zeroes mem_out for all frames via MEM_ZERO_ALL
launch() {
  local name=$1 gpu=$2 port=$3 mz=$4
  local env_mz=""
  [ "$mz" = "1" ] && env_mz="MEM_ZERO_ALL=100000"
  CUDA_VISIBLE_DEVICES=$gpu env $env_mz "$ENV/bin/torchrun" --nproc_per_node=1 --master_port=$port \
    inference/inference.py --config_path "$CFG/$name.yaml" > "$LOG/gen_$name.log" 2>&1 &
  echo "launched $name on CVD $gpu (mz=$mz) pid $!"
}

launch ck150_memon   0 29710 0
launch ck150_memzero 3 29711 1
launch ck300_memon   4 29712 0
launch ck300_memzero 5 29713 1
wait
echo "ALL_E1_EVAL_GEN_DONE"
