#!/bin/bash
# Eval overfit14 ckpt 1000: 3 arms in parallel on CVD 3/4/5, then VBench (subject+background consistency).
#   memory  = MAG on (ckpt 1000)          -> GPU3
#   ablate  = same ckpt, MAG output zeroed -> GPU4 (MEM_ZERO_COLD_FRAMES)
#   baseline_af = stock Anchor-Forcing     -> GPU5 (ckpt/final.pt, memory_layers=[])
BASE=/shared/ssd_14T/home/adenelmi/Anchor-Forcing
RD=$BASE/reference/runs/e3_overfit14
CFG=$RD/configs
PY=/shared/ssd_14T/home/adenelmi/miniconda3/envs/longlive/bin/python
ARM=$RD/run_arm.sh
VB=/shared/ssd_14T/home/adenelmi/LongLive/vbench_repo
EVAL=$RD/eval
LOG=$EVAL/orchestrator.log
cd "$BASE"
rm -rf "$EVAL/memory" "$EVAL/ablate" "$EVAL/baseline_af" "$EVAL"/score_* 2>/dev/null
mkdir -p "$EVAL"
echo "[$(date '+%H:%M:%S')] === GENERATION (3 arms parallel, CVD 3/4/5) ===" > "$LOG"
bash "$ARM" "$CFG/eval_memory.yaml"      "$EVAL/gen_memory.log"      3 29970        >/dev/null 2>&1 &
bash "$ARM" "$CFG/eval_ablate.yaml"      "$EVAL/gen_ablate.log"      4 29972 100000 >/dev/null 2>&1 &
bash "$ARM" "$CFG/eval_baseline_af.yaml" "$EVAL/gen_baseline_af.log" 5 29974        >/dev/null 2>&1 &
wait
echo "[$(date '+%H:%M:%S')] generation done. videos:" >> "$LOG"
for D in memory ablate baseline_af; do echo "  $D: $(ls "$EVAL/$D"/*.mp4 2>/dev/null | wc -l) mp4" >> "$LOG"; done
echo "[$(date '+%H:%M:%S')] === VBENCH (subject + background consistency) ===" >> "$LOG"
cd "$VB"
for D in memory ablate baseline_af; do
  for DIM in subject_consistency background_consistency; do
    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=3 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
      "$PY" vbench2_beta_long/eval_long.py --videos_path "$EVAL/$D" --dimension "$DIM" \
      --mode long_custom_input --output_path "$EVAL/score_$D" >> "$LOG" 2>&1
  done
done
echo "[$(date '+%H:%M:%S')] === DONE ===" >> "$LOG"
grep -rhoE "\"[a-z_]+\": \[?[0-9.]+" "$EVAL"/score_* 2>/dev/null | tail -30 >> "$LOG"
