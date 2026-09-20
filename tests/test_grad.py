"""The memory's write gradient, which is derived by hand rather than by autograd.

Checks it against autograd, and that a future chunk never leaks into a past one.

    python -m pytest tests/test_grad.py
"""
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from wan.modules.video_memory import (
    swiglu_apply,
    swiglu_fast_weight_gradient,
    weight_norm_update,
)


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


def _run_chunks(q, k, v, state, chunks, clean_mask, muon_steps=0):
    """Apply-first then (on clean chunks) update, threading the fast-weight state."""
    w0, w1, w2 = state
    outs = []
    for c, (s, e) in enumerate(chunks):
        outs.append(swiglu_apply(q[:, s:e], w0, w1, w2))  # apply-first: state from chunks < c
        if clean_mask[c]:
            ones = torch.ones(q.shape[0], e - s, 1, dtype=q.dtype)
            g0, g1, g2 = swiglu_fast_weight_gradient(
                k[:, s:e], v[:, s:e], w0, w1, w2, ones, ones, ones)
            w0, w1, w2 = weight_norm_update(w0, w1, w2, g0, g1, g2, muon_steps)
    return torch.cat(outs, dim=1)


def test_apply_first_inter_chunk_causality():
    B, head_dim, inter_multi = 1, 4, 2
    di = do = head_dim
    dh = head_dim * inter_multi
    g = torch.Generator().manual_seed(1)
    q = torch.randn(B, 9, di, generator=g, dtype=torch.float64)
    k = torch.randn(B, 9, di, generator=g, dtype=torch.float64)
    v = torch.randn(B, 9, do, generator=g, dtype=torch.float64)
    state = (_kaiming((B, di, dh), di, g), _kaiming((B, dh, do), dh, g), _kaiming((B, di, dh), di, g))
    chunks = [(0, 3), (3, 6), (6, 9)]
    clean = [True, True, True]

    o_ref = _run_chunks(q, k, v, state, chunks, clean)

    # perturb only the 3rd (future) chunk, tokens 6:9
    qp, kp, vp = q.clone(), k.clone(), v.clone()
    p = torch.Generator().manual_seed(2)
    qp[:, 6:9] += torch.randn(B, 3, di, generator=p, dtype=torch.float64)
    kp[:, 6:9] += torch.randn(B, 3, di, generator=p, dtype=torch.float64)
    vp[:, 6:9] += torch.randn(B, 3, do, generator=p, dtype=torch.float64)
    o_pert = _run_chunks(qp, kp, vp, state, chunks, clean)

    leak_past = (o_ref[:, :6] - o_pert[:, :6]).abs().max().item()
    change_future = (o_ref[:, 6:9] - o_pert[:, 6:9]).abs().max().item()
    print(f"  leak into past chunks   = {leak_past:.3e}  (must be 0)")
    print(f"  change in future chunk  = {change_future:.3e}  (must respond)")

    assert leak_past == 0.0, f"future perturbation leaked into past chunks: {leak_past:.3e}"
    assert change_future > 1e-6, f"perturbed future chunk did not respond: {change_future:.3e}"
