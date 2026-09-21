"""The memory's write gradient, which is derived by hand rather than by autograd, checked
against autograd. How the cell behaves chunk after chunk is tested in test_memory_cell.py,
through VideoMemoryLayer.step itself.

    python -m pytest tests/test_grad.py
"""
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from wan.modules.video_memory import swiglu_apply, swiglu_fast_weight_gradient


def _kaiming(shape, fan_in, generator):
    return torch.randn(*shape, generator=generator, dtype=torch.float64) * (math.sqrt(2) / math.sqrt(fan_in))


def _make_case(B=1, L=5, head_dim=4, inter_multi=2, seed=0):
    """Tiny float64 case in the module's batched (B*nh, L, d) layout with nh=1."""
    g = torch.Generator().manual_seed(seed)
    di = do = head_dim
    dh = head_dim * inter_multi
    k = torch.randn(B, L, di, generator=g, dtype=torch.float64)
    v = torch.randn(B, L, do, generator=g, dtype=torch.float64)
    w0 = _kaiming((B, di, dh), di, g)
    w1 = _kaiming((B, dh, do), dh, g)
    w2 = _kaiming((B, di, dh), di, g)
    return k, v, w0, w1, w2


def test_closed_form_gradient_matches_autograd():
    k, v, w0, w1, w2 = _make_case(B=1, L=5, head_dim=4, inter_multi=2)
    B, L, _ = k.shape
    ones = torch.ones(B, L, 1, dtype=torch.float64)  # lr0 = lr1 = lr2 = 1 -> raw gradient

    # closed-form raw gradient of dot = <fW(k), v>
    g0, g1, g2 = swiglu_fast_weight_gradient(k, v, w0, w1, w2, ones, ones, ones)

    # autograd gradient of the same dot
    w0a = w0.clone().requires_grad_(True)
    w1a = w1.clone().requires_grad_(True)
    w2a = w2.clone().requires_grad_(True)
    dot = (swiglu_apply(k, w0a, w1a, w2a) * v).sum()
    ag0, ag1, ag2 = torch.autograd.grad(dot, [w0a, w1a, w2a])

    for name, analytic, auto in (("w0", g0, ag0), ("w1", g1, ag1), ("w2", g2, ag2)):
        err = (analytic - auto).abs().max().item()
        print(f"  {name}: max|closed-form - autograd| = {err:.3e}")
        assert torch.allclose(analytic, auto, rtol=1e-5, atol=1e-8), f"{name} mismatch: {err:.3e}"
