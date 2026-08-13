"""Minimal repro of the seed-step per-chunk-backward hazard.

Mirrors the exact mechanics of the windowed seed path:
  chunk 0 (grad): block runs under non-reentrant checkpoint; its k is written
  IN-PLACE into a preallocated KV buffer (like _apply_cache_updates);
  loss0.backward() runs immediately (per-chunk backward, no retain_graph).
  chunk 1 (grad): attention clones the KV buffer (temp_k = cache.clone())
  under grad -> graph links back into chunk 0's consumed checkpoint.
  loss1.backward() -> expected RuntimeError (backward through freed graph).

Also checks the control: detaching the cache entry between the two backwards
(equivalent to _clear_cache_gradients) makes both backwards succeed.
"""
import torch
from torch.utils.checkpoint import checkpoint

torch.manual_seed(0)
D = 8
lin = torch.nn.Linear(D, D)          # stands in for the backbone (LoRA/frozen mix)
w0 = torch.nn.Parameter(torch.randn(D, D) * 0.1)  # stands in for the memory seed

def block(x, state):
    h = lin(x)
    k = h * 2.0                       # "roped key" computed inside the block
    out = h.tanh() + x @ state        # read path touches the seed-state
    return out, k

def run(detach_cache_between):
    cache = torch.zeros(2, D)         # preallocated KV buffer (requires_grad=False leaf)
    cache_ref = [cache]

    # ---- chunk 0 (grad, seed step) ----
    x0 = torch.randn(1, D)
    state0 = w0 + 0.0                 # differentiable init_state (w0.repeat analog)
    out0, k0 = checkpoint(block, x0, state0, use_reentrant=False)
    cache_ref[0][0:1] = k0            # in-place grad-mode write (_apply_cache_updates analog)
    loss0 = out0.pow(2).mean()
    loss0.backward()                  # per-chunk backward, frees/consumes chunk-0 graph
    g_after_0 = None if w0.grad is None else w0.grad.norm().item()

    if detach_cache_between:          # _clear_cache_gradients analog
        cache_ref[0] = cache_ref[0].detach()

    # ---- chunk 1 (grad) ----
    x1 = torch.randn(1, D)
    temp_k = cache_ref[0].clone()     # attention clones the cache under grad
    out1, k1 = checkpoint(block, x1, w0.detach() + 0.0, use_reentrant=False)
    y = out1 + temp_k[0:1]            # chunk 1 output depends on cached chunk-0 k
    loss1 = y.pow(2).mean()
    loss1.backward()
    return g_after_0

print("cache requires_grad after grad-mode in-place write:", end=" ")
c = torch.zeros(2, D); k = (torch.randn(1, D) * torch.nn.Parameter(torch.ones(1))).clone()
c[0:1] = k
print(c.requires_grad)

for detach in (False, True):
    if w0.grad is not None: w0.grad = None
    try:
        g = run(detach)
        print(f"detach_between={detach}: OK, seed grad after chunk0 backward = {g:.4f}")
    except RuntimeError as e:
        print(f"detach_between={detach}: RuntimeError: {str(e)[:140]}")
