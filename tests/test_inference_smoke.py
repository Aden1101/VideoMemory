#!/usr/bin/env python
"""A real miniature CausalWanModel with a memory branch, rolled out over three chunks, each
with its read pass and its write pass, as the inference pipeline does. No image is judged:
this checks that the chain executes, the caches advance and the memory writes. Catches what an
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
from wan.modules.video_memory import TTTBlockCache  # noqa: E402

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
    # one memory cache per block, as the pipelines allocate them. Without it the memory branch
    # returns at its first line and the test would cover the model but not the memory.
    ttt = [TTTBlockCache() for _ in range(LAYERS)]

    # The output head is zero-init (CausalWanModel.init_weights, the "never noise into the cache"
    # rule), so a fresh model returns exactly zero. It is perturbed here, otherwise the two checks on
    # the output values would be comparing 0 with 0 and would never catch a regression of the path.
    with torch.no_grad():
        m.head.head.weight.normal_(0, 0.02)

    ctx = [torch.randn(16, 64, device=device, dtype=DT)]
    t = torch.ones(1, F, device=device, dtype=DT) * 1000.0     # one timestep per frame
    ends, outs = [], []

    with torch.no_grad():
        for c in range(CHUNKS):
            x = [torch.randn(16, F, H, W, device=device, dtype=DT)]
            t0 = time.time()
            common = dict(context=ctx, seq_len=F * FRAME_SEQ, kv_cache=kv, crossattn_cache=xa,
                          current_start=c * F * FRAME_SEQ, ttt_cache=ttt)
            out = m(x, t=t, ttt_update=False, **common)                        # read pass
            m(out, t=torch.zeros_like(t), ttt_update=True, **common)           # write pass
            synchronize()
            ends.append(int(kv[0]["global_end_index"]))
            outs.append(out)
            print(f"  .. chunk {c}: {tuple(out[0].shape)} in {(time.time()-t0)*1000:5.0f} ms, "
                  f"cache at {ends[-1]} tokens")

        # the same chunk read twice, memory live then zeroed: the denoising loop also calls the
        # model several times on one chunk, so the caches accept it
        x = [torch.randn(16, F, H, W, device=device, dtype=DT)]
        common["current_start"] = CHUNKS * F * FRAME_SEQ
        live = m(x, t=t, ttt_update=False, **common)
        os.environ["MEM_ZERO_ALL"] = "1"
        try:
            zeroed = m(x, t=t, ttt_update=False, **common)
        finally:
            del os.environ["MEM_ZERO_ALL"]

    check("output shape matches the input",
          tuple(outs[0][0].shape) == (16, F, H, W), str(tuple(outs[0][0].shape)))
    check("no non-finite value", all(bool(torch.isfinite(o).all()) for o in outs))
    check("the KV cache advances from chunk to chunk",
          ends == [(c + 1) * F * FRAME_SEQ for c in range(CHUNKS)], str(ends))
    check("output is not identically zero (head perturbed)",
          float(outs[0].abs().max()) > 0)
    check("two different chunks give different outputs",
          not torch.equal(outs[0], outs[1]))
    check("the cross-attention cache was initialised", xa[0]["is_init"])
    check("the memory block wrote a state, the plain block did not",
          ttt[1].state is not None and ttt[0].state is None)
    init = m.blocks[1].memory.init_state(1, device, DT)
    check("that state moved away from its init",
          any(not torch.equal(w, w0) for w, w0 in zip(ttt[1].state, init)))
    check("reading the memory changes the output, MEM_ZERO_ALL takes that away",
          not torch.equal(live, zeroed))

    print(f"\n{len(OK)}/{len(OK)} inference smoke PASS on {device}")


if __name__ == "__main__":
    main()
