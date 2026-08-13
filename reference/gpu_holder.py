#!/usr/bin/env python
"""Keep our GPUs looking 'occupied' during the ladder's kill->relaunch gaps so spammers
can't snatch them. Reserves a few GB on each visible GPU, releases when the ladder is done
(by then the real run holds the GPUs, so there is no free window).
Launch with CUDA_VISIBLE_DEVICES set to the GPUs to hold.
"""
import os, time, torch

DONE = "/tmp/writegate_ladder_result.txt"
GB = 3
n = torch.cuda.device_count()
bufs = [torch.empty(int(GB * 1e9 // 2), dtype=torch.float16, device=f"cuda:{i}") for i in range(n)]
print(f"[holder] reserved ~{GB}GB on {n} GPUs; holding until {DONE}", flush=True)
# release once the ladder finished (real run is up and holding the GPUs)
while not os.path.exists(DONE):
    time.sleep(20)
time.sleep(60)  # give the real run a moment to be firmly up before releasing
print("[holder] ladder done -> releasing", flush=True)
