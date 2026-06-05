import math
from typing import List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

RMSNorm = getattr(nn, "RMSNorm", None)
if RMSNorm is None:  # pragma: no cover  (torch < 2.4 fallback)
    class RMSNorm(nn.Module):
        def __init__(self, d, eps=1e-5, elementwise_affine=True):
            super().__init__()
            self.eps = eps
            self.weight = nn.Parameter(torch.ones(d)) if elementwise_affine else None
        def forward(self, x):
            x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
            return x * self.weight if self.weight is not None else x


class TTTBlockCache:
    """Per-block fast-weight state holder, threaded through generator calls like a KV cache.

    Deliberately a plain object and NOT a dict: FSDP mixed-precision input casting and
    activation/gradient checkpointing traverse kwargs as pytrees and REBUILD dict/list
    containers (a fresh dict per forward call), silently swallowing in-place dict writes.
    A custom object is a pytree LEAF: passed by reference, so `cache.state = new_state`
    mutates the shared object and survives the call stack."""
    __slots__ = ("state",)

    def __init__(self):
        self.state = None


def inv_softplus(x: float) -> float:
    return x + math.log(-math.expm1(-x))


def silu_backprop(dy: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    s = torch.sigmoid(x)
    return dy * s * (1.0 + x * (1.0 - s))


def zeropower_via_newtonschulz5(G: torch.Tensor, steps: int) -> torch.Tensor:
    if steps <= 0:
        return G
    assert G.ndim == 3
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.to(torch.bfloat16)
    transposed = X.size(1) > X.size(2)
    if transposed:
        X = X.transpose(1, 2)
    X = X / (X.norm(dim=(1, 2), keepdim=True) + 1e-7)
    for _ in range(steps):
        A = X @ X.transpose(1, 2)
        B = b * A + c * A @ A
        X = a * X + B @ X
    if transposed:
        X = X.transpose(1, 2)
    return X.to(G.dtype)


def swiglu_apply(q, w0, w1, w2):
    return (F.silu(q @ w0) * (q @ w2)) @ w1


def swiglu_fast_weight_gradient(k, v, w0, w1, w2, lr0, lr1, lr2):
    """RAW gradient of dot=<fW(k),v> w.r.t (w0,w1,w2), token-wise lr folded in. (verified)"""
    g = k @ w0
    u = k @ w2
    hidden = F.silu(g) * u
    dhidden = v @ w1.transpose(-1, -2)
    dup = dhidden * F.silu(g)
    dgate = silu_backprop(dhidden * u, g)
    w0_grad = (k * lr0).transpose(-1, -2) @ dgate
    w1_grad = (hidden * lr1).transpose(-1, -2) @ v
    w2_grad = (k * lr2).transpose(-1, -2) @ dup
    return w0_grad, w1_grad, w2_grad


def weight_norm_update(w0, w1, w2, g0, g1, g2, muon_steps, allreduce_fn=None):
    n0 = w0.detach().norm(dim=1, keepdim=True)
    n1 = w1.detach().norm(dim=1, keepdim=True)
    n2 = w2.detach().norm(dim=1, keepdim=True)
    if allreduce_fn is not None:                 # GCS on RAW grads, before Muon
        g0, g1, g2 = allreduce_fn(g0, g1, g2)
    g0 = zeropower_via_newtonschulz5(g0, muon_steps)
    g1 = zeropower_via_newtonschulz5(g1, muon_steps)
    g2 = zeropower_via_newtonschulz5(g2, muon_steps)
    w0 = w0 + g0                                 # PLUS
    w1 = w1 + g1
    w2 = w2 + g2
    w0 = w0 / (w0.norm(dim=1, keepdim=True) + 1e-5) * n0
    w1 = w1 / (w1.norm(dim=1, keepdim=True) + 1e-5) * n1
    w2 = w2 / (w2.norm(dim=1, keepdim=True) + 1e-5) * n2
    return w0, w1, w2


class VideoMemoryLayer(nn.Module):
    """MAG long-term memory branch. Use `step(x, state, do_update)` per generator call
    (AR/inference/rollout), or `forward_sequence(x, chunks, clean_mask)` for the
    fully-parallel training path. Output is the gated contribution to ADD to the residual.
    """
    def __init__(
        self,
        dim: int = 1536,
        num_heads: int = 1,        # nh=1 -> head_dim=dim, max state size (paper choice)
        inter_multi: int = 4,      # k=4
        base_lr: float = 1e-2,
        muon_steps: int = 5,       # 0 = plain SGD on the fast weight
        qk_norm: bool = True,      # RMSNorm on head_dim (on for AR stability)
        use_modulation: bool = True,
        g_init: float = 0.0,       # 0.0 -> inert warm start (identical to baseline at step 0)
        detach_state: Optional[bool] = None,  # None -> detach in eval, keep graph in train
    ):
        super().__init__()
        assert dim % num_heads == 0
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.muon_steps = muon_steps
        self.use_modulation = use_modulation
        self.detach_state = detach_state

        di = do = self.head_dim
        dh = self.head_dim * inter_multi

        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.q_norm = RMSNorm(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = RMSNorm(self.head_dim) if qk_norm else nn.Identity()

        self.w0 = nn.Parameter(torch.randn(num_heads, di, dh) * (math.sqrt(2) / math.sqrt(di)))
        self.w1 = nn.Parameter(torch.randn(num_heads, dh, do) * (math.sqrt(2) / math.sqrt(dh)))
        self.w2 = nn.Parameter(torch.randn(num_heads, di, dh) * (math.sqrt(2) / math.sqrt(di)))

        self.lrs = nn.Linear(dim, num_heads * 3)
        self.inv_base_lr = inv_softplus(base_lr)

        self.o_norm = RMSNorm(self.head_dim, eps=1e-5)
        self.proj = nn.Linear(dim, dim * (3 if use_modulation else 1), bias=True)
        if use_modulation:
            self.a_norm = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
            nn.init.zeros_(self.proj.weight)
            nn.init.zeros_(self.proj.bias)
            self.proj.bias.data[2 * dim:] = g_init   # gate bias -> warm start

    # ---- internals ----
    def _qkv_lr(self, x):
        B, L, _ = x.shape
        qkv = self.qkv(x)
        q, k, v = rearrange(qkv, "b l (qkv h d) -> qkv b h l d", qkv=3, h=self.num_heads)
        q = self.q_norm(q)
        k = self.k_norm(k)
        q = rearrange(q, "b h l d -> (b h) l d")
        k = rearrange(k, "b h l d -> (b h) l d")
        v = rearrange(v, "b h l d -> (b h) l d")
        with torch.amp.autocast("cuda", enabled=False):
            # lr path in float32 regardless of the module dtype. The original
            # `self.lrs(x.float())` crashed under bf16 deployment (fp32 input vs bf16 weight:
            # "mat1 and mat2 must have the same dtype") -- cast the tiny 1536->3 linear's
            # weights per-call instead, then fold the lrs back in the compute dtype.
            lrs = F.softplus(
                F.linear(x.float(), self.lrs.weight.float(),
                         self.lrs.bias.float() if self.lrs.bias is not None else None)
                + self.inv_base_lr)
            lr0, lr1, lr2 = rearrange(lrs, "b l (n h c) -> n (b h) l c", n=3, h=self.num_heads)
        lr0, lr1, lr2 = lr0.type_as(k), lr1.type_as(k), lr2.type_as(k)
        return q, k, v, lr0, lr1, lr2, B

    def _project_output(self, o, B):
        o = self.o_norm(o)
        o = rearrange(o, "(b h) l d -> b l (h d)", h=self.num_heads, b=B)
        if not self.use_modulation:
            return self.proj(o)
        shift, scale, gate = self.proj(o).chunk(3, dim=-1)
        return gate * (self.a_norm(o) * (1 + scale) + shift)

    def init_state(self, B, device, dtype):
        return (self.w0.repeat(B, 1, 1).to(device=device, dtype=dtype),
                self.w1.repeat(B, 1, 1).to(device=device, dtype=dtype),
                self.w2.repeat(B, 1, 1).to(device=device, dtype=dtype))

    def _detach(self):
        return (not self.training) if self.detach_state is None else self.detach_state

    # ---- AR / inference / rollout: one temporal chunk per call ----
    def step(self, x, state=None, do_update=True, allreduce_fn=None):
        """x:(B,L,dim) for the current chunk. Returns (mem_out:(B,L,dim), new_state)."""
        q, k, v, lr0, lr1, lr2, B = self._qkv_lr(x)
        if state is None:
            state = self.init_state(B, x.device, q.dtype)
        w0, w1, w2 = state
        o = swiglu_apply(q, w0, w1, w2)                       # APPLY-FIRST (state from < this chunk)
        new_state = state
        if do_update:                                          # write the clean chunk
            g0, g1, g2 = swiglu_fast_weight_gradient(k, v, w0, w1, w2, lr0, lr1, lr2)
            w0n, w1n, w2n = weight_norm_update(w0, w1, w2, g0, g1, g2, self.muon_steps, allreduce_fn)
            if self._detach():
                w0n, w1n, w2n = w0n.detach(), w1n.detach(), w2n.detach()
            new_state = (w0n, w1n, w2n)
        return self._project_output(o, B), new_state

    # ---- training (fully parallel over all chunks of a clip) ----
    def forward_sequence(self, x, chunks: Sequence[Tuple[int, int]], clean_mask: Sequence[bool],
                         allreduce_fn=None):
        q, k, v, lr0, lr1, lr2, B = self._qkv_lr(x)
        w0, w1, w2 = self.init_state(B, x.device, q.dtype)
        o = torch.zeros_like(q)
        for c, (s, e) in enumerate(chunks):
            o[:, s:e] = swiglu_apply(q[:, s:e], w0, w1, w2)    # apply-first
            if clean_mask[c]:
                g0, g1, g2 = swiglu_fast_weight_gradient(
                    k[:, s:e], v[:, s:e], w0, w1, w2, lr0[:, s:e], lr1[:, s:e], lr2[:, s:e])
                w0, w1, w2 = weight_norm_update(w0, w1, w2, g0, g1, g2, self.muon_steps, allreduce_fn)
        return self._project_output(o, B)
