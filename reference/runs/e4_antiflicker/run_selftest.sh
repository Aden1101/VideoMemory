#!/bin/bash
# e4 twin-selftest + grad-probe gate (shrunk rollout so it fits ONE 48GB GPU; all e4 code paths
# active: EMA injection low-pass, mem-ON twin, advantage loss with flick, seed-step per-chunk backward).
set -u
cd /shared/ssd_14T/home/adenelmi/Anchor-Forcing
ENV=/shared/ssd_14T/home/adenelmi/miniconda3/envs/longlive
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export PYTHONPATH=/shared/ssd_14T/home/adenelmi/Anchor-Forcing
export E3_TWIN_SELFTEST=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=0
rm -rf /shared/ssd_14T_project/adenelmi/e4_selftest
"$ENV/bin/torchrun" --nproc_per_node=1 --master_port=29760 \
  train.py --config_path reference/runs/e4_antiflicker/.e4_selftest_min.yaml \
  --no_save --no_visualize --no-one-logger
echo "SELFTEST_EXIT=$?"
