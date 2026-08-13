#!/usr/bin/env python
"""STEP 0 falsification probe (a): leak check. Run the MEM model but with the memory READ zeroed on ALL frames
(MEM_ZERO_COLD_FRAMES huge). Output must equal the AF baseline (same base+LoRA+seed). If subject_consistency !=
baseline (0.9597 @60s), the memory's mere PRESENCE corrupts something (init clobber / shared-state leak)."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
os.environ["MEM_ZERO_COLD_FRAMES"] = "99999"   # zero the read for every frame -> x = x + 0 = baseline (if no leak)
from e2_eval_parallel import infer, vbench, MEM_CKPT, MEM_LAYERS, OUT, GPUS, log

G1 = GPUS[0]  # gpu1
log("STEP0(a) leak check: mem with read zeroed EVERYWHERE, 60s -> must match baseline 0.9597")
infer("mem_zeroall_L240", MEM_CKPT, list(MEM_LAYERS), f"{OUT}/mem_zeroall_L240", 240, G1, 30096)
vbench("mem_zeroall_L240", f"{OUT}/mem_zeroall_L240", f"{OUT}/eval_mem_zeroall_L240", G1)
log("STEP0(a) done")
