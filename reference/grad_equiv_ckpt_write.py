#!/usr/bin/env python
"""Gradient-equivalence test for the write-step checkpointing (MAG_CKPT_WRITE).

Compares the gradients of every memory parameter with checkpoint_write ON vs OFF,
same seed / same params / same input. Two regimes:
  1. PLUMBING  (muon=0, no Newton-Schulz): must be ~identical. Large => wiring bug
     (wrong tensors recomputed under checkpoint).
  2. REAL      (Muon NS=5, bf16): the deployment error. Expect a few % (the FIXME).
  3. REAL fp32 (Muon NS=5, fp32): isolates how much of the error is bf16 vs logic.

Run (1 GPU):
  CUDA_VISIBLE_DEVICES=<free> <longlive>/python reference/grad_equiv_ckpt_write.py
"""
import os
os.environ.pop("MAG_CKPT_WRITE", None)  # we toggle layer.checkpoint_write directly
import torch
import wan.modules.video_memory as vm
from wan.modules.video_memory import VideoMemoryLayer

# Instrument: count how many times the REAL torch checkpoint() is actually invoked,
# so a "0 error" cannot be a false positive from the checkpoint path being skipped.
_orig_ckpt = vm.checkpoint
def _counting_ckpt(*a, **k):
    _counting_ckpt.n += 1
    return _orig_ckpt(*a, **k)
_counting_ckpt.n = 0
vm.checkpoint = _counting_ckpt


def grads_for(layer, x, use_ckpt):
    layer.zero_grad(set_to_none=True)
    layer.checkpoint_write = use_ckpt
    mem_out, new_state = layer.step(x, state=None, do_update=True)
    # loss touches BOTH the read (mem_out) and the write (new_state = updated fast weights)
    loss = mem_out.float().sum() + sum(s.float().sum() for s in new_state)
    loss.backward()
    return {n: p.grad.detach().float().clone()
            for n, p in layer.named_parameters() if p.grad is not None}


def run_case(label, muon_steps, dtype, L=4680, dim=1536):
    torch.manual_seed(0)
    layer = VideoMemoryLayer(dim=dim, num_heads=1, muon_update_steps=muon_steps,
                             g_init=0.1, detach_state=False).cuda().to(dtype)
    layer.train()
    torch.manual_seed(1)
    x = torch.randn(1, L, dim, device="cuda", dtype=dtype)

    _counting_ckpt.n = 0
    g_no = grads_for(layer, x, use_ckpt=False)
    n_off = _counting_ckpt.n
    g_ck = grads_for(layer, x, use_ckpt=True)
    n_on = _counting_ckpt.n - n_off

    print(f"\n=== {label}  (muon={muon_steps}, dtype={str(dtype).split('.')[-1]}, L={L}) ===")
    print(f"  [checkpoint() invoked: OFF-run={n_off}, ON-run={n_on}]  (ON must be >=1)")
    worst = 0.0
    for n in g_no:
        a, b = g_no[n], g_ck[n]
        rel = (b - a).norm().item() / (a.norm().item() + 1e-12)
        maxd = (b - a).abs().max().item()
        nan = torch.isnan(b).any().item()
        print(f"  {n:22s} relL2={rel:.3e}  maxdiff={maxd:.3e}  |g|={a.norm().item():.3e}"
              + ("  <-- NaN!" if nan else ""))
        worst = max(worst, rel)
    print(f"  --> worst relative-L2 gradient error: {worst:.3e}")
    return worst


def grads_for_chain(layer, xs, use_ckpt):
    """Chain K writes, threading the state across blocks (the real rollout / BPTT-through-time)."""
    layer.zero_grad(set_to_none=True)
    layer.checkpoint_write = use_ckpt
    state, outs = None, []
    for x in xs:
        mem_out, state = layer.step(x, state=state, do_update=True)
        outs.append(mem_out)
    loss = sum(o.float().sum() for o in outs) + sum(s.float().sum() for s in state)
    loss.backward()
    return {n: p.grad.detach().float().clone()
            for n, p in layer.named_parameters() if p.grad is not None}


def run_chain(label, muon_steps, dtype, K=8, L=4680, dim=1536):
    torch.manual_seed(0)
    layer = VideoMemoryLayer(dim=dim, num_heads=1, muon_update_steps=muon_steps,
                             g_init=0.1, detach_state=False).cuda().to(dtype)
    layer.train()
    torch.manual_seed(2)
    xs = [torch.randn(1, L, dim, device="cuda", dtype=dtype) for _ in range(K)]

    _counting_ckpt.n = 0
    g_no = grads_for_chain(layer, xs, use_ckpt=False)
    n_off = _counting_ckpt.n
    g_ck = grads_for_chain(layer, xs, use_ckpt=True)
    n_on = _counting_ckpt.n - n_off

    print(f"\n=== {label}  (muon={muon_steps}, dtype={str(dtype).split('.')[-1]}, K={K} writes, L={L}) ===")
    print(f"  [checkpoint() invoked: OFF-run={n_off}, ON-run={n_on}]  (ON must be {K})")
    worst = 0.0
    for n in g_no:
        a, b = g_no[n], g_ck[n]
        rel = (b - a).norm().item() / (a.norm().item() + 1e-12)
        maxd = (b - a).abs().max().item()
        nan = torch.isnan(b).any().item()
        print(f"  {n:22s} relL2={rel:.3e}  maxdiff={maxd:.3e}  |g|={a.norm().item():.3e}"
              + ("  <-- NaN!" if nan else ""))
        worst = max(worst, rel)
    print(f"  --> worst relative-L2 gradient error: {worst:.3e}")
    return worst


if __name__ == "__main__":
    print("Gradient equivalence: checkpoint_write ON vs OFF (identical seed/params/input)")
    w_plumb = run_case("PLUMBING single write (no Newton-Schulz)", muon_steps=0, dtype=torch.bfloat16)
    w_real  = run_case("REAL single write (Muon NS=5, bf16)", muon_steps=5, dtype=torch.bfloat16)
    w_fp32  = run_case("REAL single write (Muon NS=5, fp32)", muon_steps=5, dtype=torch.float32)
    w_chain = run_chain("THROUGH-TIME 8 chained writes (Muon NS=5, bf16)", muon_steps=5, dtype=torch.bfloat16)

    print("\n================ VERDICT ================")
    print(f"  plumbing  single (muon=0, bf16): {w_plumb:.3e}   (~0 expected; large => WIRING BUG)")
    print(f"  real      single (muon=5, bf16): {w_real:.3e}")
    print(f"  real      single (muon=5, fp32): {w_fp32:.3e}")
    print(f"  THROUGH-TIME 8 writes (bf16):    {w_chain:.3e}   (the FIXME's 'through time' case)")
    ok = (w_plumb < 1e-2) and (w_chain < 0.10)
    print("  RESULT:", "OK -- checkpoint write is correct (error within tolerance)" if ok
          else "SUSPECT -- investigate (error larger than expected)")
