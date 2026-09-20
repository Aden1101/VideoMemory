#!/usr/bin/env python
"""A real miniature CausalWanModel with a memory branch, rolled out over three chunks. No
image is judged: this checks that the chain executes and the caches advance. Catches what an
import test cannot, which is everything that only fails at run time.

    python tests/test_inference_smoke.py            # detected device
    AF_DEVICE=cpu python tests/test_inference_smoke.py
"""
import os
import sys
import time

import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils.device import describe, device, synchronize  # noqa: E402
from wan.modules.causal_model import CausalWanModel  # noqa: E402

HEADS, HEAD_DIM = 12, 128          # required by the pipeline's _initialize_kv_cache
DIM = HEADS * HEAD_DIM
LAYERS, F, H, W = 2, 3, 16, 16
FRAME_SEQ = (H // 2) * (W // 2)
LOCAL, CHUNKS = 12, 3
DT = torch.bfloat16                # the real pipeline runs in bf16; the TTT memory downcasts anyway

OK = []


def check(name, cond, detail=""):
    assert cond, f"FAIL {name} {detail}"
    OK.append(name)
    print(f"  ok  {name}")


def main():
    print(describe())
    torch.manual_seed(0)
    m = CausalWanModel(dim=DIM, ffn_dim=2048, num_heads=HEADS, num_layers=LAYERS, text_dim=64,
                       text_len=16, local_attn_size=LOCAL, sink_size=3, memory_layers=(1,),
                       in_dim=16, out_dim=16).to(device=device, dtype=DT).eval()
    print(f"  miniature model: {sum(p.numel() for p in m.parameters())/1e6:.1f} M params, "
          f"{LAYERS} layers, {FRAME_SEQ} tokens/frame")

    kv = [{"k": torch.zeros(1, LOCAL * FRAME_SEQ, HEADS, HEAD_DIM, dtype=DT, device=device),
           "v": torch.zeros(1, LOCAL * FRAME_SEQ, HEADS, HEAD_DIM, dtype=DT, device=device),
           "global_end_index": torch.tensor([0], dtype=torch.long, device=device),
           "local_end_index": torch.tensor([0], dtype=torch.long, device=device)}
          for _ in range(LAYERS)]
    xa = [{"k": torch.zeros(1, 512, HEADS, HEAD_DIM, dtype=DT, device=device),
           "v": torch.zeros(1, 512, HEADS, HEAD_DIM, dtype=DT, device=device),
           "is_init": False} for _ in range(LAYERS)]

    # The output head is zero-init (causal_model.py:1780, the "never noise into the cache"
    # rule), so a fresh model returns exactly zero. It is perturbed here, otherwise S4 and S5
    # would be comparing 0 with 0 and would never catch a regression of the path.
    with torch.no_grad():
        m.head.head.weight.normal_(0, 0.02)

    ctx = [torch.randn(16, 64, device=device, dtype=DT)]
    t = torch.ones(1, F, device=device, dtype=DT) * 1000.0     # one timestep per frame
    ends, outs = [], []

    with torch.no_grad():
        for c in range(CHUNKS):
            x = [torch.randn(16, F, H, W, device=device, dtype=DT)]
            t0 = time.time()
            out = m(x, t=t, context=ctx, seq_len=F * FRAME_SEQ, kv_cache=kv, crossattn_cache=xa,
                    current_start=c * F * FRAME_SEQ)
            synchronize()
            ends.append(int(kv[0]["global_end_index"]))
            outs.append(out)
            print(f"  .. chunk {c}: {tuple(out[0].shape)} in {(time.time()-t0)*1000:5.0f} ms, "
                  f"cache at {ends[-1]} tokens")

    check("S1 output shape matches the input",
          tuple(outs[0][0].shape) == (16, F, H, W), str(tuple(outs[0][0].shape)))
    check("S2 no non-finite value", all(bool(torch.isfinite(o).all()) for o in outs))
    check("S3 the KV cache advances from chunk to chunk",
          ends == [(c + 1) * F * FRAME_SEQ for c in range(CHUNKS)], str(ends))
    check("S4 output is not identically zero (head perturbed)",
          float(outs[0].abs().max()) > 0)
    check("S5 two different chunks give different outputs",
          not torch.equal(outs[0], outs[1]))
    check("S6 the cross-attention cache was initialised", xa[0]["is_init"])

    print(f"\n{len(OK)}/{len(OK)} inference smoke PASS on {device}")


if __name__ == "__main__":
    main()
