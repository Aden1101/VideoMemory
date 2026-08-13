#!/bin/bash
# ============================================================================================
# c1_cotrain step-250 :: single-entrypoint 3-way memory eval.
#   ARM1 memon    = c1 ckpt, memory ON (trained MAG injected)
#   ARM2 memzero  = c1 ckpt, MEM_ZERO_ALL=1 (injection zeroed) = LoRA-drift-only / inert proxy
#   ARM3 AF-stock = REUSED held-out stock column (e3_antifreeze/heldout, memory_layers [], final.pt)
#
# Does: gen memon -> gen memzero (same GPU) -> VBench SC/BC/DD on both -> (fill missing BC on
#       reused stock) -> metrics.py -> pre-registered verdict.
#
# GUARDED: refuses to run until the step-250 checkpoint exists. Idempotent (skips finished stages).
# Usage:   bash run_eval.sh [GPU]      (default GPU 5; NEVER 6=broken; training holds 0-3)
# ============================================================================================
set -u
GPU="${1:-5}"
if [ "$GPU" = "6" ]; then echo "REFUSE: GPU 6 is BROKEN. Pick 5 or 7."; exit 1; fi

REPO=/shared/ssd_14T/home/adenelmi/Anchor-Forcing
ENV=/shared/ssd_14T/home/adenelmi/miniconda3/envs/longlive
VB=/shared/ssd_14T/home/adenelmi/LongLive/vbench_repo
PY=$ENV/bin/python
TORCHRUN=$ENV/bin/torchrun

ED=$REPO/reference/runs/c1_cotrain/eval250
CKPT=/shared/ssd_14T_project/adenelmi/c1_cotrain/checkpoint_model_000250/model.pt
PROMPTS=$REPO/reference/runs/e3_antifreeze/heldout/prompts.txt
STOCK_VID=$REPO/reference/runs/e3_antifreeze/heldout/videos/stockaf
STOCK_VB=$REPO/reference/runs/e3_antifreeze/heldout/vbench_stockaf

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ---- GUARD: checkpoint must exist ----
if [ ! -f "$CKPT" ]; then
  echo "GUARD: checkpoint not found -> $CKPT"
  echo "       step-250 not saved yet. Aborting (nothing generated)."
  exit 2
fi
echo "checkpoint present: $CKPT"

NP=$(grep -cve '^[[:space:]]*$' "$PROMPTS")   # non-empty prompt lines
echo "held-out prompts: $NP  (file: $PROMPTS)"

port() { echo $(( (RANDOM % 10000) + 20000 )); }

# ---- helper: have all {idx}-0.mp4 for NP prompts been generated? ----
have_all_videos() { local d="$1"; local i; for ((i=0;i<NP;i++)); do [ -f "$d/$i-0.mp4" ] || return 1; done; return 0; }

# ---- helper: vbench dim already scored in dir? ----
have_vbench() { ls "$1"/results_*_"$2"_eval_results.json >/dev/null 2>&1; }

gen_arm() {  # $1=cfg $2=outdir $3=extra-env
  local cfg="$1" out="$2" env="$3"
  if have_all_videos "$out"; then echo "  [skip gen] $out already has $NP videos"; return 0; fi
  mkdir -p "$out"
  echo "  [gen] $cfg -> $out   (env: ${env:-none})"
  cd "$REPO"
  env $env CUDA_VISIBLE_DEVICES=$GPU "$TORCHRUN" --nproc_per_node=1 --master_port=$(port) \
    inference/inference.py --config_path "$cfg" > "$ED/gen_$(basename "$out").log" 2>&1
  local rc=$?
  echo "  [gen] $(basename "$out") exit=$rc  ($(ls "$out"/*.mp4 2>/dev/null | wc -l) mp4)"
  return $rc
}

vbench_arm() {  # $1=videos_dir $2=output_dir  (SC/BC/DD)
  local vids="$1" outp="$2"
  mkdir -p "$outp"; cd "$VB"
  for DIM in subject_consistency background_consistency dynamic_degree; do
    if have_vbench "$outp" "$DIM"; then echo "  [skip vbench] $(basename "$outp")/$DIM"; continue; fi
    echo "  [vbench] $(basename "$outp") / $DIM"
    CUDA_VISIBLE_DEVICES=$GPU "$PY" vbench2_beta_long/eval_long.py \
      --videos_path "$vids" --dimension $DIM --mode long_custom_input --output_path "$outp" \
      >> "$ED/vb_$(basename "$outp")_$DIM.log" 2>&1
    echo "  [vbench] $DIM exit=$?"
  done
}

echo "=== STAGE 1: generate memon (GPU $GPU) ==="
gen_arm "$ED/memon.yaml"   "$ED/videos/memon"   ""

echo "=== STAGE 2: generate memzero (GPU $GPU, MEM_ZERO_ALL=1) ==="
gen_arm "$ED/memzero.yaml" "$ED/videos/memzero" "MEM_ZERO_ALL=1"

echo "=== STAGE 3: VBench SC/BC/DD on memon + memzero ==="
vbench_arm "$ED/videos/memon"   "$ED/vbench_memon"
vbench_arm "$ED/videos/memzero" "$ED/vbench_memzero"

echo "=== STAGE 3b: fill missing VBench dims on REUSED AF-stock (SC+DD already exist; adds BC) ==="
if [ -d "$STOCK_VID" ] && have_all_videos "$STOCK_VID"; then
  vbench_arm "$STOCK_VID" "$STOCK_VB"
else
  echo "  WARN: reused stock videos missing at $STOCK_VID -- regenerate via stockaf.yaml first."
fi

echo "=== STAGE 4: metrics + pre-registered verdict ==="
cd "$ED"
CUDA_VISIBLE_DEVICES=$GPU "$PY" metrics.py \
  --memon_dir   "$ED/videos/memon" \
  --memzero_dir "$ED/videos/memzero" \
  --stock_dir   "$STOCK_VID" \
  --vbench_memon "$ED/vbench_memon" \
  --vbench_memzero "$ED/vbench_memzero" \
  --vbench_stock "$STOCK_VB" \
  --device cuda:0 \
  --out "$ED/metrics_c1_step250.json" | tee "$ED/metrics_c1_step250.stdout.txt"

echo "=== DONE. Report: $ED/metrics_c1_step250.txt ==="
