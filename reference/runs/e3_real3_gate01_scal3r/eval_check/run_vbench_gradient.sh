#!/bin/bash
# VBench on the same 4 prompts across the gradient: baseline / gate01_450 / gate01_750 / overfit.
CK=/shared/ssd_14T/home/adenelmi/Anchor-Forcing/reference/runs/e3_real3_gate01_scal3r/eval_check
OF=/shared/ssd_14T/home/adenelmi/Anchor-Forcing/reference/runs/e3_overfit14/eval
VB=/shared/ssd_14T/home/adenelmi/LongLive/vbench_repo
PY=/shared/ssd_14T/home/adenelmi/miniconda3/envs/longlive/bin/python
LOG=$CK/vbench.log
# build 4-prompt subsets for baseline + overfit (man=0 sonia=3 youngman=1 woman=13 -> renamed 0..3)
mkdir -p $CK/baseline4 $CK/overfit4
cp -f $OF/baseline_af/0-0.mp4 $CK/baseline4/0-0.mp4; cp -f $OF/baseline_af/3-0.mp4 $CK/baseline4/1-0.mp4
cp -f $OF/baseline_af/1-0.mp4 $CK/baseline4/2-0.mp4; cp -f $OF/baseline_af/13-0.mp4 $CK/baseline4/3-0.mp4
cp -f $OF/memory/0-0.mp4 $CK/overfit4/0-0.mp4; cp -f $OF/memory/3-0.mp4 $CK/overfit4/1-0.mp4
cp -f $OF/memory/1-0.mp4 $CK/overfit4/2-0.mp4; cp -f $OF/memory/13-0.mp4 $CK/overfit4/3-0.mp4
: > $LOG
cd $VB
for D in baseline4 gate01_450 gate01_750 overfit4; do
  for DIM in subject_consistency background_consistency; do
    echo "[$(date '+%H:%M:%S')] $D / $DIM" >> $LOG
    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=3 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
      $PY vbench2_beta_long/eval_long.py --videos_path $CK/$D --dimension $DIM \
      --mode long_custom_input --output_path $CK/score_$D >> $LOG 2>&1
  done
done
echo "[$(date '+%H:%M:%S')] === VBENCH DONE ===" >> $LOG
