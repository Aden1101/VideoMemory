#!/usr/bin/env python
"""Causal confirmation of root cause #1 (unsupervised cold-start read). On gpu1, regenerate the mem 6s
videos but with the COLD memory READ (first 12 latent frames) ZEROED (the write/state update is kept,
env MEM_ZERO_COLD_FRAMES=12). Then VBench. Compare subject_consistency:
  baseline 6s = 0.9637 ; mem 6s (normal) = 0.8009 ; mem 6s (cold read zeroed) = ?
If it recovers from ~0.80 toward ~0.88+ (the long-range mem level), the cold-start read is PROVEN the cause."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
os.environ["MEM_ZERO_COLD_FRAMES"] = "12"   # zero the cold READ for global latent frame < 12
from e2_eval_parallel import infer, vbench, MEM_CKPT, MEM_LAYERS, OUT, GPUS, log

G1 = GPUS[0]  # gpu1
log("CAUSAL TEST: mem 24 latent, cold read (frames<12) zeroed, writes kept")
infer("mem_H24_coldzero", MEM_CKPT, list(MEM_LAYERS), f"{OUT}/mem_H24_coldzero", 24, G1, 30090)
vbench("mem_H24_coldzero", f"{OUT}/mem_H24_coldzero", f"{OUT}/eval_mem_H24_coldzero", G1)
log("CAUSAL TEST done")
