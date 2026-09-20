#!/bin/bash
# Training launcher. The memory runs used 4 GPUs; two 48 GB cards also work.
#   bash train.sh                                    # run 1
#   bash train.sh configs/runs/run3_write_evicted_stable.yaml 4

CONFIG="${1:-configs/runs/run1_write_current_chunk.yaml}"
NGPU="${2:-4}"
echo "CONFIG=$CONFIG on $NGPU GPU(s)"

# expandable segments keeps the long rollouts from fragmenting the allocator
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,garbage_collection_threshold:0.8

torchrun \
  --nnodes 1 \
  --nproc_per_node="$NGPU" \
  train.py \
  --config_path "$CONFIG" \
  --no-one-logger
