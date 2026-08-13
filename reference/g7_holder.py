#!/usr/bin/env python
"""Reserve a few GB on G7 so we don't lose our one clean GPU while waiting for a 2nd.
The watcher kills this right before it launches the real run. CUDA_VISIBLE_DEVICES must = G7."""
import time, torch

buf = torch.empty(int(2.5e9 // 2), dtype=torch.float16, device="cuda:0")  # ~2.5 GB
print("[g7_holder] reserved ~2.5GB on G7; holding until killed", flush=True)
while True:
    time.sleep(20)
