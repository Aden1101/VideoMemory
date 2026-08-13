#!/bin/bash
# One-GPU fit-test: gate_init=0.01, Scal3R-faithful (renorm, decay OFF, muon 5), 1 memory layer, K=1.
PY=/shared/ssd_14T/home/adenelmi/miniconda3/envs/longlive/bin/python
CFG=/shared/ssd_14T/home/adenelmi/.claude/jobs/3118b292/tmp/_gate01_scal3r.yaml
RUNLOG=/shared/ssd_14T/home/adenelmi/Anchor-Forcing/reference/e3_gate01_scal3r.log
cd /shared/ssd_14T/home/adenelmi/Anchor-Forcing
: > "$RUNLOG"
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=6 NCCL_P2P_DISABLE=1 PG_TIMEOUT_MIN=20 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  setsid nohup "$PY" -m torch.distributed.run --nproc_per_node=1 --max-restarts=0 --master_port=30500 \
  train.py --config_path "$CFG" --no-one-logger >> "$RUNLOG" 2>&1 </dev/null & disown
echo "launched gate01 on idx6 (pid group $!), log: $RUNLOG"
