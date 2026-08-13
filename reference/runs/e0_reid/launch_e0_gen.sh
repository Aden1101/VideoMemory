#!/bin/bash
# E0 generation: stock-AF baseline drift + anchoring probe (sink sweep) + freeze adversary.
# One arm per free GPU (CVD 0,3,4,5), ~120s videos, 8 dynamic subject prompts.
set -u
export CUDA_DEVICE_ORDER=PCI_BUS_ID
cd /shared/ssd_14T/home/adenelmi/Anchor-Forcing
ENV=/shared/ssd_14T/home/adenelmi/miniconda3/envs/longlive
CFG=reference/runs/e0_reid/configs
LOG=reference/runs/e0_reid/out
mkdir -p "$LOG"

# arm -> gpu (CVD, PCI_BUS_ID order)
arms=(stock_sink3 stock_sink9 stock_sink21 memon_sink3)
gpus=(0 3 4 5)
port=29610
pids=()
for i in "${!arms[@]}"; do
  arm=${arms[$i]}; gpu=${gpus[$i]}
  CUDA_VISIBLE_DEVICES=$gpu "$ENV/bin/torchrun" --nproc_per_node=1 --master_port=$port \
    inference/inference.py --config_path "$CFG/$arm.yaml" \
    > "$LOG/gen_$arm.log" 2>&1 &
  pid=$!; pids+=($pid)
  echo "launched $arm on CVD $gpu (port $port) pid $pid"
  port=$((port+1))
done

fail=0
for i in "${!pids[@]}"; do
  if wait "${pids[$i]}"; then
    echo "OK: ${arms[$i]}"
  else
    echo "FAIL: ${arms[$i]} (exit $?)"; fail=1
  fi
done
echo "ALL_E0_GEN_DONE fail=$fail"
