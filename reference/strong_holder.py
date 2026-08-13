#!/usr/bin/env python
"""Truly RESERVE the single visible GPU by occupying most of it -- a small holder does NOT
stop grabbers (proven: khoutaibi allocated on top of our 2.5GB holder). Our write run needs
a near-whole GPU, so any other allocation breaks it; hence we reserve ~all of it.
HOLD_GB via env (default 44). CUDA_VISIBLE_DEVICES must select exactly one GPU.
The watcher kills this right before launching the real run."""
import os, time, torch

gb = float(os.environ.get("HOLD_GB", "44"))
target = int(gb * 1e9)
chunk = int(2e9)            # grow in 2GB float16 blocks so we hold whatever fits
buf, got = [], 0
while got < target:
    sz = min(chunk, target - got)
    try:
        buf.append(torch.empty(sz // 2, dtype=torch.float16, device="cuda:0"))
        got += sz
    except RuntimeError:
        break               # GPU filled by someone else -> hold what we secured
print(f"[strong_holder] reserved ~{got/1e9:.1f}GB on visible GPU; holding until killed", flush=True)
while True:
    time.sleep(15)
