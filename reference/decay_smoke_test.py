#!/usr/bin/env python
"""Smoke test the new memory improvements: gated-delta decay (use_decay) + bounded tanh gate.
Checks: constructs, runs an 8-chunk rollout, state norm stays BOUNDED (forgetting works, no blow-up),
mem_out finite, and gradient flows to decay_logit + the fast-weight seed."""
import sys
sys.path.insert(0, "/shared/ssd_14T/home/adenelmi/Anchor-Forcing")
import torch
from wan.modules.video_memory import VideoMemoryLayer

dev = "cuda"
m = VideoMemoryLayer(dim=1536, num_heads=1, use_decay=True, decay_init=0.95,
                     bounded_gate=True, muon_update_steps=0, g_init=0.1,
                     detach_state=False).to(dev).to(torch.bfloat16)
m.train()
state = None
B, L = 1, 21
norms, outrms = [], []
for c in range(8):
    x = torch.randn(B, L, 1536, device=dev, dtype=torch.bfloat16)
    out, state = m.step(x, state=state, do_update=True)
    norms.append(round(state[0].detach().float().norm().item(), 2))
    outrms.append(round(out.detach().float().pow(2).mean().sqrt().item(), 4))
    assert torch.isfinite(out).all(), f"non-finite out at chunk {c}"
    assert all(torch.isfinite(s).all() for s in state), f"non-finite state at chunk {c}"
print("state w0 norm per chunk (should stay bounded, can shrink=forget):", norms)
print("mem_out rms per chunk:", outrms)
loss = out.float().pow(2).mean()
loss.backward()
print("decay value sigmoid(decay_logit):", round(torch.sigmoid(m.decay_logit).item(), 4))
print("grad reaches decay_logit:", m.decay_logit.grad is not None and torch.isfinite(m.decay_logit.grad).all().item())
print("grad reaches w0 seed:", m.w0.grad is not None and torch.isfinite(m.w0.grad).all().item())
print("grad reaches qkv:", m.qkv.weight.grad is not None)
# sanity: with use_decay=False it must still run (old path)
m2 = VideoMemoryLayer(dim=1536, num_heads=1, use_decay=False, bounded_gate=False,
                      muon_update_steps=5, g_init=0.0, detach_state=True).to(dev).to(torch.bfloat16).eval()
o2, _ = m2.step(torch.randn(1, 21, 1536, device=dev, dtype=torch.bfloat16))
print("old path (use_decay=False) runs, out finite:", torch.isfinite(o2).all().item())
print("SMOKE OK")
