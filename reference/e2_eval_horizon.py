#!/usr/bin/env python
"""Training-horizon A/B on gpu1: eval mem vs baseline at 24 latent (= streaming_max_length, the
EXACT length the memory state was trained on, ~6s) and 48 latent (2 chunks, first cross-chunk regime).
Isolates the train/test state-accumulation gap: if mem >= base at 24 but < base at 60/120/240, the
degradation is the carried-state horizon gap (cross-chunk), not a within-chunk failure."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from e2_eval_parallel import infer, vbench, MEM_CKPT, MEM_LAYERS, OUT, GPUS, log

G1 = GPUS[0]  # gpu1
port = 30070
for H in [24, 48]:
    for tag, ckpt, layers in [(f"mem_H{H}", MEM_CKPT, list(MEM_LAYERS)),
                              (f"base_H{H}", "ckpt/final.pt", [])]:
        infer(tag, ckpt, layers, f"{OUT}/{tag}", H, G1, port); port += 1
        vbench(tag, f"{OUT}/{tag}", f"{OUT}/eval_{tag}", G1)
log("horizon eval (24, 48 latent) done")
