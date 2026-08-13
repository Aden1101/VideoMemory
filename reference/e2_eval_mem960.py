#!/usr/bin/env python
"""Run ONLY mem_L960 (240s) on gpu7, reusing the fixed infer()/vbench() from e2_eval_parallel.
Launched separately so it fills the idle 3rd GPU while mem_240/mem_480 run on gpu1/gpu2."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from e2_eval_parallel import infer, vbench, MEM_CKPT, MEM_LAYERS, OUT, GPUS, log

G7 = GPUS[2]  # gpu7 UUID
log("mem_L960 (240s) standalone on gpu7")
infer("mem_L960", MEM_CKPT, list(MEM_LAYERS), f"{OUT}/mem_L960", 960, G7, 30055)
vbench("mem_L960", f"{OUT}/mem_L960", f"{OUT}/eval_mem_L960", G7)
log("mem_L960 done")
