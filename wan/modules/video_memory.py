import math
import os
from typing import Optional, Sequence

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint


class CompatRMSNorm(nn.Module):
    def __init__(
        self,
        normalized_shape,
        eps=None,
        elementwise_affine: bool = True,
        device=None,
        dtype=None,
    ):
        super().__init__()
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = tuple(normalized_shape)
        self.eps = eps
        self.elementwise_affine = elementwise_affine

        if elementwise_affine:
            self.weight = nn.Parameter(torch.ones(self.normalized_shape, device=device, dtype=dtype))
        else:
            self.register_parameter('weight', None)

    def forward(self, x: Tensor):
        dims = tuple(range(-len(self.normalized_shape), 0))
        eps = self.eps if self.eps is not None else torch.finfo(x.dtype).eps
        output = x * torch.rsqrt(x.pow(2).mean(dims, keepdim=True) + eps)
        if self.weight is not None:
            output = output * self.weight
        return output


RMSNorm = getattr(nn, 'RMSNorm', CompatRMSNorm)


class TTTBlockCache:
    """Per-block fast-weight state holder, threaded through generator calls like a KV cache.

    Deliberately a plain object and not a dict: FSDP mixed-precision input casting and
    activation/gradient checkpointing traverse kwargs as pytrees and rebuild dict/list
    containers (a fresh dict per forward call), silently swallowing in-place dict writes.
    A custom object is a pytree leaf: passed by reference, so `cache.state = new_state`
    mutates the shared object and survives the call stack. (Scal3R threaded a tuple instead,
    which works in its single-process inference path but not under our FSDP training stack.)"""
    __slots__ = ("state", "ema", "fifo")

    def __init__(self):
        self.state = None
        # anti-flicker: causal temporal-EMA buffers for the injection low-pass, keyed by
        # denoising-step (timestep value). Entries are replaced (never mutated in place) so a
        # shallow dict() copy suffices for the twin snapshot, exactly like the state tuples.
        self.ema = {}
        # eviction feed: FIFO of (chunk_idx, detached h) for the last window_chunks write
        # chunks; the popped (= just-evicted) entry becomes the memory's x_write. Entries are
        # appended/popped (tuples never mutated in place), so a shallow list() copy suffices
        # for the twin/BPTT snapshots. Empty and untouched unless eviction_feed is on.
        self.fifo = []


def inv_softplus(x):
    """ Inverse of the softplus function """
    if isinstance(x, torch.Tensor):
        y = x + torch.log(-torch.expm1(-x))
    else:
        y = x + math.log(-math.expm1(-x))
    return y


def silu_backprop(dy: torch.Tensor, x: torch.Tensor):
    """ Backpropagation for the Swish (Silu) activation function

    Args:
        dy (torch.Tensor): (B, C, L), gradient of the outer loss wrt the y
        x (torch.Tensor): (B, C, L), input of the silu activation
    Returns:
        dx (torch.Tensor): (B, C, L), gradient of the outer loss wrt the x
    """
    sigma = torch.sigmoid(x)
    dx = dy * sigma * (1 + x * (1 - sigma))
    return dx


def zeropower_via_newtonschulz5(G, steps):
    """ Modified from https://github.com/MoonshotAI/Moonlight/blob/master/examples/toy_train.py#L49
    Major change: G is (B, C, C) rather than (C, C).

    Newton-Schulz iteration to compute the zeroth power / orthogonalization of G. We opt to use a
    quintic iteration whose coefficients are selected to maximize the slope at zero. For the purpose
    of minimizing steps, it turns out to be empirically effective to keep increasing the slope at
    zero even beyond the point where the iteration no longer converges all the way to one everywhere
    on the interval. This iteration therefore does not produce UV^T but rather something like US'V^T
    where S' is diagonal with S_{ii}' ~ Uniform(0.5, 1.5), which turns out not to hurt model
    performance at all relative to UV^T, where USV^T = G is the SVD.

    Args:
        G (torch.Tensor): (B, C, C) matrix to orthogonalize
        steps, (int): Number of iterations to perform
    Returns:
        X (torch.Tensor): (B, C, C) orthogonalized matrix
    """
    assert len(G.shape) == 3
    if steps == 0:                       # 0 = plain sgd, keep the raw grad (no orthogonalization)
        return G
    a, b, c = (3.4445, -4.7750, 2.0315)

    X = G.bfloat16()  # FIXME: may cause 2% - 4% relative gradient error when using checkpoint through time
    if G.size(1) > G.size(2):
        X = X.transpose(1, 2)
    # Ensure spectral norm is at most 1
    X = X / (X.norm(dim=(1, 2), keepdim=True) + 1e-7)

    # Perform the NS iterations
    for _ in range(steps):
        A = X @ X.transpose(1, 2)
        B = (
            b * A + c * A @ A
        )  # adapted from suggestion by @jxbz, @leloykun, and @YouJiacheng
        X = a * X + B @ X

    if G.size(1) > G.size(2):
        X = X.transpose(1, 2)
    return X


def swiglu_apply(q, w0, w1, w2):
    """ f_W(q) = (silu(q @ w0) * (q @ w2)) @ w1, Scal3R's apply formula (the `oi = ...` line of
    fast_weight_swish_glu_weight_norm_mini_batch_apply), pulled out so the causal step() can
    apply before it writes (apply-first). """
    return (F.silu(q @ w0, inplace=False) * (q @ w2)) @ w1


def swiglu_fast_weight_gradient(k, v, w0, w1, w2, lr0, lr1, lr2):
    """ Raw inner-loop gradient of dot = <f_W(k), v> wrt (w0, w1, w2), token-wise lr folded in.

    Body is identical to Scal3R's fast_weight_swish_glu_weight_norm_mini_batch_gradient; the only
    change is the signature is pre-sliced to a single causal chunk, so there is no ttt_ua_order to
    index (k, v, lr* are already the chunk's tokens). No minus sign: the fast weight ascends this
    dot (it is the TTT objective); Muon is applied later, in weight_norm_update.
    """
    # Calculate hidden -> SiLU(W1x) ◦ (W3x)
    gate_before_act = k @ w0  # (B, L, C) @ (B, C, Ch) -> (B, L, Ch)
    hide_before_mul = k @ w2  # (B, L, C) @ (B, C, Ch) -> (B, L, Ch)
    hidden = F.silu(gate_before_act, inplace=False) * hide_before_mul  # (B, L, Ch)

    # Compute the middle local gradients
    dhidden = v @ w1.transpose(-1, -2)  # (B, L, C) @ (B, C, Ch) -> (B, L, Ch)
    dhide_before_mul = dhidden * F.silu(gate_before_act, inplace=False)  # (B, L, Ch)
    dgate = dhidden * hide_before_mul  # (B, L, Ch)
    dgate_before_act = silu_backprop(dgate, gate_before_act)  # (B, L, Ch)

    # Calculate the gradients for w0, w1, w2
    w0_grad = (k * lr0).transpose(-1, -2) @ dgate_before_act  # (B, C, Ch)
    w1_grad = (hidden * lr1).transpose(-1, -2) @ v  # (B, Ch, C)
    w2_grad = (k * lr2).transpose(-1, -2) @ dhide_before_mul  # (B, C, Ch)

    # note: we return the raw local gradients here, not the orthogonalized ones
    # note: the muon update is applied in the update method (here: weight_norm_update)
    return w0_grad, w1_grad, w2_grad


def weight_norm_update(w0, w1, w2, w0_grad, w1_grad, w2_grad, muon_update_steps, allreduce_fn=None,
                       decay=None, init_norms=None):
    """ Muon (Newton-Schulz) + fast-weight update.
    Default (decay=None): Scal3R renorm, ascend the Muon'd grad, then renorm each row to its
    pre-update norm (norm conserved, no forgetting; the state only rotates in a fixed-norm shell).
    Gated-delta (decay/init_norms set): w <- decay*w + muon(grad), then cap each row to its init norm
    (shrink allowed = explicit forgetting). `allreduce_fn` = Scal3R's context-parallel allreduce (raw grads).
    """
    # Remember that the weight norm before the update is applied
    w0_norm = w0.detach().norm(dim=1, keepdim=True)
    w1_norm = w1.detach().norm(dim=1, keepdim=True)
    w2_norm = w2.detach().norm(dim=1, keepdim=True)

    if allreduce_fn is not None:
        w0_grad, w1_grad, w2_grad = allreduce_fn(w0_grad, w1_grad, w2_grad)

    # Apply the Muon update here
    w0_grad = zeropower_via_newtonschulz5(w0_grad, muon_update_steps)
    w1_grad = zeropower_via_newtonschulz5(w1_grad, muon_update_steps)
    w2_grad = zeropower_via_newtonschulz5(w2_grad, muon_update_steps)

    if decay is None:
        # original: ascend then renorm each row back to its pre-update norm (no forgetting)
        w0 = w0 + w0_grad
        w1 = w1 + w1_grad
        w2 = w2 + w2_grad
        w0 = w0 / (w0.norm(dim=1, keepdim=True) + 1e-5) * w0_norm
        w1 = w1 / (w1.norm(dim=1, keepdim=True) + 1e-5) * w1_norm
        w2 = w2 / (w2.norm(dim=1, keepdim=True) + 1e-5) * w2_norm
        return w0, w1, w2

    # gated-delta: decay (forget) the old state, add the update, cap each row norm to its init norm
    d0, d1, d2 = decay
    n0, n1, n2 = init_norms
    w0 = d0 * w0 + w0_grad
    w1 = d1 * w1 + w1_grad
    w2 = d2 * w2 + w2_grad
    w0 = w0 / (w0.norm(dim=1, keepdim=True) / (n0 + 1e-5)).clamp(min=1.0)
    w1 = w1 / (w1.norm(dim=1, keepdim=True) / (n1 + 1e-5)).clamp(min=1.0)
    w2 = w2 / (w2.norm(dim=1, keepdim=True) / (n2 + 1e-5)).clamp(min=1.0)
    return w0, w1, w2


def _sinusoid_embed(vals, dim):
    """Vals: (...,) scalars -> (..., dim) parameter-free sinusoidal embedding."""
    half = max(dim // 2, 1)
    freqs = torch.exp(-math.log(10000.0) * torch.arange(half, device=vals.device).float() / half)
    ang = vals.float().unsqueeze(-1) * freqs
    return torch.cat([ang.sin(), ang.cos()], dim=-1)


def linear_delta_chunk_update(x, W, beta):
    """One linear delta-rule write on raw tokens x (T, d) fp32: W <- W + beta (v - W k) k^T,
    sequential token by token in row order, k = h/||h|| (L2 per token), v = h. W (d, d) fp32
    is updated in place and returned.

    P0 (plan_entrainement_video_2026-08-07, bras M2): this is the deployment twin of the
    offline oracle `lin_chunk_update` (reference/runs/probe1/build_states.py, rule='delta',
    arm lin_delta_all, 15.33/19). The body is arithmetic-identical to that function's delta
    branch, same ops, same order, same dtypes, same kernel shapes (2-D (T, d) fp32 input) --
    so the state produced in generation is bitwise the state of the offline replay
    (asserted by reference/runs/probe1/selftest_lin_native.py). Any edit here must keep that
    parity or re-run the equivalence selftest. Zero learned parameter, training-free."""
    assert x.dim() == 2 and x.dtype == torch.float32, f"tokens {tuple(x.shape)} {x.dtype}"
    assert W.dim() == 2 and W.dtype == torch.float32, f"state {tuple(W.shape)} {W.dtype}"
    nrm = x.norm(dim=-1, keepdim=True)
    assert torch.isfinite(nrm).all() and bool((nrm > 0).all()), "token de norme nulle"
    K = x / nrm                                   # k = h / ||h||, v = h
    for t in range(x.shape[0]):                   # sequentiel, scelle au registre (oracle L4)
        k = K[t]
        W.add_(torch.outer(x[t] - W @ k, k), alpha=beta)
    return W


class VideoMemoryLayer(nn.Module):
    """ Causal adaptation of Scal3R's FastWeightGluMLPMultihead.

    Same fast-weight SwiGLU memory (qkv/lr projection, inner-loop gradient, Muon + weight-norm
    update, gated AdaLN output), wired for autoregressive video instead of bidirectional VGGT
    patches. Use `step(x, state, do_update)` per generator call (ar/inference/rollout). Output is
    the gated contribution to add to the residual.
    """
    def __init__(
        self,
        dim: int = 1536,
        num_heads: int = 1,        # nh=1 -> head_dim=dim, max state size (large head_dim is better for TTT)
        inter_multi: int = 4,      # k=4
        base_lr: float = 1e-2,
        muon_update_steps: int = 5,  # 0 = plain SGD on the fast weight
        qk_norm: bool = True,      # RMSNorm on head_dim (on for ar stability; Scal3R default False)
        use_modulation: bool = True,
        g_init: float = 0.0,       # 0.0 -> inert warm start (identical to baseline at step 0; Scal3R: 0.1)
        use_decay: bool = False,   # gated-delta forgetting in weight_norm_update (else renorm-to-init)
        decay_init: float = 0.95,  # initial forget gate value = sigmoid(decay_logit); 1.0 = no forgetting
        bounded_gate: bool = False,  # tanh-bound the output gate (kills the unbounded fixed-RMS injection)
        single_norm: bool = False,   # read-path hygiene: drop the redundant o_norm so proj sees the raw
                                     # read magnitude (relevance-coupled injection); keep only AdaLN a_norm
        bounded_mod: bool = False,   # tanh-bound scale/shift too, so injection amplitude is state-independent
        rel_norm: bool = False,      # amplitude fix: replace per-token o_norm with one chunk-scalar RMS, so
                                     # per-token relative read magnitude (relevance) survives to proj + carrier
                                     # while the global write-count growth cancels every chunk; also drops a_norm
        dc_remove: bool = False,     # amplitude fix: subtract the token-mean of the injection (train + inference)
                                     # so proj cannot learn the token-constant DC-bias crutch
        detach_state: Optional[bool] = None,  # None -> detach in eval, keep graph in train
        checkpoint_write: bool = False,  # gradient-checkpoint the write update
        gate_extra: Optional[Sequence[str]] = None,  # extra gate inputs (off=[]): "agreement","denoising_step","frames_since_switch"
        state_type: str = "swiglu",   # P0/M2: "swiglu" (default, strictly the historical cell)
                                      # | "linear_delta" (matrix state W, delta rule on raw h)
        linear_beta: float = 0.5,     # beta of the delta rule (LIN_BETA of the sealed oracle)
    ):
        super().__init__()
        assert dim % num_heads == 0, "dim should be divisible by num_heads"
        # ---- P0 linear-delta state (plan 2026-08-07, bras M2), additive ------------------
        # state_type="linear_delta" swaps only the state/read/write core of the cell:
        #   state : one matrix W (head_dim x head_dim) per block, init 0, fp32 (the sealed
        #           accumulation dtype of the offline oracle, fp16/bf16 would change the
        #           rounding of the sequential rule);
        #   write : linear_delta_chunk_update on the raw evicted-chunk tokens (same tokens
        #           the historical write receives, x_write of the eviction feed), k = h/||h||,
        #           v = h, no qkv projection, no learned parameter, training-free;
        #   read  : o = W @ q for the existing read-path queries q (qkv projection + q_norm),
        #           plugged exactly where swiglu_apply reads today, so the whole downstream
        #           (o_norm/rel_norm, gate, modulation, injection, env probes, hooks) is
        #           unchanged.
        # GPU footprint (nh=1, d=1536, B=1): W fp32 = 1536*1536*4 = 9.44 MB per block
        # (28.3 MB for the 3 deployed blocks; fp16 snapshot 4.72 MB/block, ~14.2 MB/age),
        # vs the SwiGLU state (w0+w1+w2 bf16, inter_multi=4) = 56.6 MB per block (169.9 MB
        # for 3 blocks): ~6x smaller. The module still owns every historical parameter
        # (qkv/lrs/proj/w0/w1/w2/...), so checkpoints load unchanged in both modes.
        assert state_type in ("swiglu", "linear_delta"), f"unknown state_type {state_type!r}"
        if state_type == "linear_delta":
            # the sealed rule is defined on full raw tokens (d = 1536); per-head splitting of
            # h would be a different rule than the validated oracle -> refuse.
            assert num_heads == 1, "linear_delta state requires num_heads == 1"
        self.state_type = state_type
        self.linear_beta = float(linear_beta)

        self.dim = dim
        self.g_init = g_init
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.muon_update_steps = muon_update_steps
        self.use_modulation = use_modulation
        self.use_decay = use_decay
        self.bounded_gate = bounded_gate
        self.single_norm = single_norm
        self.bounded_mod = bounded_mod
        self.rel_norm = rel_norm
        self.dc_remove = dc_remove
        self.detach_state = detach_state
        self.checkpoint_write = checkpoint_write
        self.gate_extra = list(gate_extra) if gate_extra else []
        if use_decay:
            # one learned forget gate (scalar), sigmoid(decay_logit); init to decay_init
            _di = min(max(float(decay_init), 1e-6), 1.0 - 1e-6)   # keep in (0,1) for the logit
            self.decay_logit = nn.Parameter(torch.tensor(math.log(_di / (1.0 - _di))))

        # Query, Key, Value linear layers
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.q_norm = RMSNorm(self.head_dim, eps=1e-5) if qk_norm else nn.Identity()
        self.k_norm = RMSNorm(self.head_dim, eps=1e-5) if qk_norm else nn.Identity()

        # Projection layers
        self.proj = nn.Linear(dim, dim if not use_modulation else dim * 3, bias=True)
        if use_modulation:
            # Adaptive layer normalization without affine parameters
            self.a_norm = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
            self._init_modulation_params(dim, g_init=g_init)
            if self.gate_extra:
                self._build_gate_extra(dim)

        # TTT fast weight dimensions
        di = self.head_dim
        do = self.head_dim
        dh = self.head_dim * inter_multi

        # TTT fast weight parameters initialization, LaCT/Scal3R: 1/sqrt(fan_in) (no sqrt(2) gain)
        self.w0 = nn.Parameter(torch.randn(num_heads, di, dh) / math.sqrt(di))
        self.w1 = nn.Parameter(torch.randn(num_heads, dh, do) / math.sqrt(dh))
        self.w2 = nn.Parameter(torch.randn(num_heads, di, dh) / math.sqrt(di))

        # Learning rates for the fast weight
        self.lrs = nn.Linear(dim, num_heads * 3)
        self.inv_base_lr = inv_softplus(base_lr)
        self.base_lr = base_lr

        # Output normalization
        self.o_norm = RMSNorm(self.head_dim, eps=1e-5, elementwise_affine=True)

    def _init_modulation_params(self, dim: int, g_init=0.1):
        with torch.no_grad():
            nn.init.zeros_(self.proj.weight)
            nn.init.zeros_(self.proj.bias)
            self.proj.bias[2 * dim:] = g_init  # gate bias only

    # ---- optional extra gate inputs; zero-init proj so they contribute nothing until trained ----
    GX_RANK = 64   # shared-space rank for the memory<->prompt agreement
    GX_EMB = 16    # sinusoidal dim for scalar signals (denoising step / frames-since-switch)

    def _build_gate_extra(self, dim: int):
        _known = {"agreement", "denoising_step", "frames_since_switch"}
        _bad = [g for g in self.gate_extra if g not in _known]
        assert not _bad, f"unknown gate_extra entries {_bad}, expected subset of {sorted(_known)}"
        extra_dim = 0
        if "agreement" in self.gate_extra:
            self.gx_po = nn.Linear(dim, self.GX_RANK, bias=False)  # project memory read
            self.gx_pc = nn.Linear(dim, self.GX_RANK, bias=False)  # project prompt c (same dim)
            extra_dim += 1                                          # the scalar a in [0,1]
        if "denoising_step" in self.gate_extra:
            extra_dim += self.GX_EMB
        if "frames_since_switch" in self.gate_extra:
            extra_dim += self.GX_EMB
        # separate proj, added to the base proj(o_hat); zero-init so it contributes nothing at init
        self.proj_extra = nn.Linear(extra_dim, 3 * dim, bias=False)
        nn.init.zeros_(self.proj_extra.weight)

    def _gate_extra_vec(self, o_hat, prompt_emb, frames_since_switch, denoising_step):
        """Build the (B, L, extra_dim) feature vector fed to proj_extra."""
        B, L, _ = o_hat.shape
        feats = []
        if "agreement" in self.gate_extra:
            if prompt_emb is None:
                raise ValueError("gate_extra 'agreement' needs prompt_emb (the text context c)")
            c = prompt_emb.mean(dim=1)                                       # (B, dim) pooled prompt
            a = torch.sigmoid((self.gx_po(o_hat) * self.gx_pc(c)[:, None, :]).sum(-1, keepdim=True))
            self.dbg_agreement = a.detach().float().mean()                  # for logging (0-dim, no host sync)
            feats.append(a.to(o_hat.dtype))                                  # (B, L, 1)
        if "denoising_step" in self.gate_extra:
            if denoising_step is None:
                raise ValueError("gate_extra 'denoising_step' needs the denoising step signal (not wired yet)")
            feats.append(_sinusoid_embed(denoising_step, self.GX_EMB).to(o_hat.dtype))
        if "frames_since_switch" in self.gate_extra:
            if frames_since_switch is None:
                raise ValueError("gate_extra 'frames_since_switch' needs the frames-since-switch signal (not wired yet)")
            feats.append(_sinusoid_embed(frames_since_switch, self.GX_EMB).to(o_hat.dtype))
        return torch.cat(feats, dim=-1)

    def reset_parameters(self):
        """Re-apply the memory branch's own initialization. Must be called after any global
        model init (e.g. causalWanModel.init_weights(), which xavier-inits every nn.Linear in
        the model)."""
        assert self.use_modulation, "reset_parameters assumes use_modulation=True (the only supported mode)"
        self._init_modulation_params(self.dim, g_init=self.g_init)
        if self.gate_extra:
            nn.init.zeros_(self.proj_extra.weight)  # re-zero extras after global init

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
            # Scal3R does `lrs = self.lrs(x.float())`; that crashes when the module is deployed in
            # bf16 (fp32 input vs bf16 weight). Cast the tiny lrs Linear's weights per call instead.
            lrs = F.linear(x.float(), self.lrs.weight.float(),
                           self.lrs.bias.float() if self.lrs.bias is not None else None)
        lrs = F.softplus(lrs.float() + self.inv_base_lr)
        lr0, lr1, lr2 = rearrange(lrs, "b l (n h c) -> n (b h) l c", n=3, h=self.num_heads)
        # Scal3R keeps lr in fp32 and lets the ambient autocast harmonize the matmuls; our pipelines
        # run in plain bf16 (no autocast), so fold the lr into the compute dtype to avoid bf16@fp32.
        lr0, lr1, lr2 = lr0.type_as(k), lr1.type_as(k), lr2.type_as(k)
        return q, k, v, lr0, lr1, lr2, B

    def _project_output(self, x, batch_size, prompt_emb=None,
                        frames_since_switch=None, denoising_step=None, num_frames=None):
        if self.rel_norm:                      # amplitude fix: one chunk-scalar RMS per (b*h). relative
            s = x.float().pow(2).mean(dim=(1, 2), keepdim=True).add(1e-10).sqrt()  # eps inside sqrt: safe backward
            x = (x.float() / s).type_as(x)   # per-token read magnitude survives; write-count growth cancels/chunk
        elif not self.single_norm:             # legacy: per-token o_norm (erases the read magnitude)
            x = self.o_norm(x)
        x = rearrange(x, '(b h) l d -> b l (h d)', h=self.num_heads, b=batch_size)
        if not self.use_modulation:
            return self.proj(x)
        shift, scale, gate = self.proj(x).chunk(3, dim=-1)
        if self.gate_extra:  # extra inputs -> added via the zero-init proj_extra (no-op at init)
            es, esc, eg = self.proj_extra(
                self._gate_extra_vec(x, prompt_emb, frames_since_switch, denoising_step)).chunk(3, dim=-1)
            shift, scale, gate = shift + es, scale + esc, gate + eg
        if self.bounded_gate:                 # tanh-bound -> no unbounded fixed-RMS injection
            gate = torch.tanh(gate)
        if self.bounded_mod:                  # bound scale/shift too -> injection amplitude state-independent
            scale, shift = torch.tanh(scale), torch.tanh(shift)
        # rel_norm: the raw (chunk-normed) read is the carrier so amplitude tracks read relevance;
        # legacy: a_norm re-pins the carrier to unit variance (relevance-decoupled).
        carrier = x if self.rel_norm else self.a_norm(x)
        out = gate * (carrier * (1 + scale) + shift)
        # diagnostics on the pre-dc-removal injection (0-dim scalars, no host sync)
        _o = out.detach().float()
        self.dbg_dc_frac = _o.mean(dim=1, keepdim=True).pow(2).mean() / _o.pow(2).mean().clamp_min(1e-12)
        _tok = _o.pow(2).mean(dim=-1).sqrt()                          # (b, l) per-token injection RMS
        self.dbg_amp_disp = _tok.std() / _tok.mean().clamp_min(1e-12) # 0 = flat = amplitude still decoupled
        # split the joint dispersion into the wanted spatial axis (relevance coupling across
        # sites) vs the unwanted temporal axis (per-site frame-to-frame std = the flicker). Needs the
        # frame count to reshape l=(F*S). Health target: temporal small while spatial stays > 0.
        if num_frames is not None and int(num_frames) > 0 and _tok.shape[1] % int(num_frames) == 0:
            _F = int(num_frames)
            _amp = _tok.reshape(_tok.shape[0], _F, _tok.shape[1] // _F)   # (b, F, S)
            _site = _amp.mean(dim=1)                                       # (b, S) time-mean amp / site
            self.dbg_ampdisp_spatial = _site.std() / _site.mean().clamp_min(1e-12)
            _fstd = _amp.std(dim=1)                                        # (b, S) across-frame std / site
            self.dbg_ampdisp_temporal = _fstd.mean() / _amp.mean().clamp_min(1e-12)
        self.dbg_gate_mean = gate.detach().float().abs().mean()
        if self.dc_remove or int(os.environ.get("MEM_DC_REMOVE", "0")):
            out = out - out.mean(dim=1, keepdim=True)  # kill the token-constant DC channel, train + inference
        return out

    def init_state(self, B, device, dtype):
        if getattr(self, "state_type", "swiglu") == "linear_delta":
            # W = 0, one matrix per block (nh = 1). fp32 sealed (the oracle's accumulation
            # dtype); the `dtype` arg (deployment bf16) is deliberately not applied to the
            # state, it still selects nothing here because the rule fixes its own dtype.
            return (torch.zeros(B * self.num_heads, self.head_dim, self.head_dim,
                                device=device, dtype=torch.float32),)
        return (self.w0.repeat(B, 1, 1).to(device=device, dtype=dtype),
                self.w1.repeat(B, 1, 1).to(device=device, dtype=dtype),
                self.w2.repeat(B, 1, 1).to(device=device, dtype=dtype))

    def _linear_write(self, x_write, W):
        """Linear-delta write: new W from the raw evicted-chunk tokens x_write (B, Lw, dim).

        Parity contract with the offline oracle (build_states.lin_chunk_update, delta):
          - x_write[b] is handed to linear_delta_chunk_update as the same 2-D (T, d) fp32
            tensor the oracle sees (bf16 -> fp32 cast is exact, like the oracle's stored
            fp16 -> fp32);
          - TF32 is forced OFF around the update (saved/restored): the offline replay runs
            under probe1 set_determinism (TF32 off), and the state must be the same function
            of the data whatever the ambient backend flags of the calling pipeline are;
          - the state is functional: W is cloned, the clone is updated in place (bit-copy +
            identical in-place ops = the oracle's persistent-W trajectory), so snapshots of
            older state tuples (twin/BPTT/episodic) are never mutated;
          - no_grad: the rule is training-free by design (no learned parameter in the write;
            the read path above stays differentiable wrt q)."""
        with torch.no_grad():
            saved_tf32 = torch.backends.cuda.matmul.allow_tf32
            torch.backends.cuda.matmul.allow_tf32 = False
            try:
                Wn = W.detach().clone()
                for b in range(x_write.shape[0]):
                    linear_delta_chunk_update(x_write[b].float(), Wn[b], self.linear_beta)
            finally:
                torch.backends.cuda.matmul.allow_tf32 = saved_tf32
        return Wn

    def _detach(self):
        return (not self.training) if self.detach_state is None else self.detach_state

    def _decay_and_norms(self, B):
        """Gated-delta forget gate (scalar sigmoid) + per-row INIT norms repeated to the (B*nh) state.
        Returns (None, None) when use_decay is off (-> weight_norm_update keeps renorm-to-init)."""
        if not self.use_decay:
            return None, None
        dec = torch.sigmoid(self.decay_logit)
        decay = (dec, dec, dec)
        init_norms = (self.w0.detach().norm(dim=1, keepdim=True).repeat(B, 1, 1),
                      self.w1.detach().norm(dim=1, keepdim=True).repeat(B, 1, 1),
                      self.w2.detach().norm(dim=1, keepdim=True).repeat(B, 1, 1))
        return decay, init_norms

    # ---- AR / inference / rollout: one temporal chunk per call ----
    def step(self, x, state=None, do_update=True, allreduce_fn=None,
             prompt_emb=None, frames_since_switch=None, denoising_step=None, num_frames=None,
             x_write=None):
        """x:(B,L,dim) for the current chunk. Returns (mem_out:(B,L,dim), new_state).
        prompt_emb/frames_since_switch/denoising_step feed the optional gate_extra inputs.
        num_frames (latent frames in this call) only feeds the spatial/temporal dispersion split.
        x_write (B,Lw,dim): the eviction feed. When given, the read query is still taken from the
        current chunk `x` (what we are generating), but the WRITE stores the k->v association of
        `x_write` = the EVICTED tokens (what fell out of the KV window), so the memory holds the
        cache-complement (VideoSSM/AHN-style) instead of the redundant current chunk. x_write=None
        (default) => write the current chunk, which is the behaviour before the eviction feed."""
        q, k, v, lr0, lr1, lr2, B = self._qkv_lr(x)
        if state is None:
            state = self.init_state(B, x.device, q.dtype)
        if getattr(self, "state_type", "swiglu") == "linear_delta":
            # ---- P0 linear-delta core (additive; the else-branch below is the historical
            # path, untouched). read, apply-first, plugged exactly where swiglu_apply reads:
            # o_i = W @ q_i for the existing read-path queries q (fp32 matmul with the fp32
            # state, cast back to the compute dtype so all downstream, gate, modulation,
            # injection, env probes, is unchanged). write: delta rule on the raw tokens of
            # the same chunk the historical write receives (x_write of the eviction feed,
            # else the current chunk). k/v/lr* of the qkv projection are unused by the rule.
            assert allreduce_fn is None, "linear_delta: context-parallel write not supported"
            (W,) = state
            o = torch.matmul(q.float(), W.transpose(-1, -2)).type_as(q)
            new_state = state
            if do_update:
                new_state = (self._linear_write(
                    x_write if x_write is not None else x, W),)
        else:
            w0, w1, w2 = state
            o = swiglu_apply(q, w0, w1, w2)                       # apply-first (state from < this chunk)
            new_state = state
            if do_update:                                          # write the (evicted, or current) chunk
                if x_write is not None:                            # eviction-feed: write source = evicted tokens
                    _, k, v, lr0, lr1, lr2, _ = self._qkv_lr(x_write)
                decay, init_norms = self._decay_and_norms(B)
                def _write(k, v, w0, w1, w2, lr0, lr1, lr2):
                    g0, g1, g2 = swiglu_fast_weight_gradient(k, v, w0, w1, w2, lr0, lr1, lr2)
                    return weight_norm_update(
                        w0, w1, w2, g0, g1, g2, self.muon_update_steps, allreduce_fn,
                        decay=decay, init_norms=init_norms)
                if self.checkpoint_write and torch.is_grad_enabled():
                    w0n, w1n, w2n = checkpoint(_write, k, v, w0, w1, w2, lr0, lr1, lr2,
                                               use_reentrant=False)
                else:
                    w0n, w1n, w2n = _write(k, v, w0, w1, w2, lr0, lr1, lr2)
                if self._detach():
                    w0n, w1n, w2n = w0n.detach(), w1n.detach(), w2n.detach()
                new_state = (w0n, w1n, w2n)
        out = self._project_output(o, B, prompt_emb=prompt_emb,
                                   frames_since_switch=frames_since_switch,
                                   denoising_step=denoising_step, num_frames=num_frames)
        # lightweight diagnostics (0-dim tensors -> no host sync here; .item() at log time)
        self.dbg_read_rms = o.detach().float().pow(2).mean().sqrt()
        self.dbg_out_rms = out.detach().float().pow(2).mean().sqrt()
        return out, new_state

    # ---- memory 2.0 write-replay (offline writer training), additive, step() untouched ----
    def write_only(self, x_write, state, allreduce_fn=None, use_checkpoint=None, grad_fn=None,
                   micro_batch=None):
        """One fast-weight write: state -> new state from the evicted-chunk hidden x_write.

        Exactly the do_update branch of step() with x_write given, factored out so the
        write-replay trainer (memory 2.0) can re-run the write chain offline on collected h
        sequences and get bitwise the same states as the in-situ rollout
        (selftest: reference/runs/frame_dec/selftest_replay.py). Parity contract:
          - `_qkv_lr(x_write)` is reused as is: the qkv projection stays one fused matmul
            (1536 -> 4608) with q computed and thrown away, exactly like step(), re-doing
            it as separate k/v matmuls would change the rounding;
          - same `_decay_and_norms` per call, same `_write` op order, same checkpoint
            mechanism (`use_reentrant=False`), no dtype cast added anywhere;
          - no `_detach()` here: the caller owns the graph (replay_writes keeps it so the
            gradient reaches qkv / k_norm / lrs / decay_logit through the whole chain);
          - caller-side requirements for bitwise parity vs inference: module and x_write in
            the deployment dtype (bf16), no ambient autocast, same TF32/backends settings
            and same GPU model as the in-situ rollout.
        use_checkpoint: None -> self.checkpoint_write (the config flag, same as step());
        the checkpoint branch is only taken when grads are enabled, so a no_grad replay
        follows the exact same code path as no_grad inference.
        grad_fn (additive, diagnostic only, registry section "arm MLP-CORRECTEUR", 07/08):
        the inner-loop gradient function, with the exact signature of
        `swiglu_fast_weight_gradient` (k, v, w0, w1, w2, lr0, lr1, lr2) -> (g0, g1, g2).
        None (the default, and the only value any training/inference path ever passes) calls
        `swiglu_fast_weight_gradient` itself, so the default write is bitwise the historical
        one, calling the same function object through a local name changes nothing.
        This exists so an offline replay can swap the write objective (error-correcting
        variant) without duplicating a single line of the parity contract above.
        micro_batch (additive, diagnostic only, registry "plan FACTORIEL", 07/08): None (the
        default, and the only value any training/inference path ever passes) = one update on
        the whole chunk, bitwise the historical write. An int m > 0 makes the write sequential:
        the chunk's tokens are consumed in slices of m in their natural index order, each slice
        seeing the state left by the previous ones. Invariants deliberately preserved so that
        only the update schedule changes:
          - `_qkv_lr` still runs once on the whole chunk (one fused matmul) and the slices are
            taken on its output, so k/v/lr are token-for-token identical to the one-step write;
          - `_decay_and_norms` still runs once per write and the forget gate is applied on the
            first slice only (total = decay*w + sum of the slice updates, exactly the one-step
            algebra); the later slices pass decay 1.0 and keep the same init_norms cap;
          - the per-row norm cap is applied at every slice: in a sequential rule each
            intermediate state is a real state (the next slice reads it), so it must be valid.
        No gradient checkpointing on this path: it is an offline no_grad diagnostic replay.
        Returns the new (w0, w1, w2) tuple. Nothing on the module is mutated."""
        # P0 guard: this is the SwiGLU write-replay path; the linear_delta state replays
        # offline via build_states.lin_states / lin_chunk_update, never through here.
        assert getattr(self, "state_type", "swiglu") == "swiglu", \
            f"write_only is undefined for state_type={self.state_type!r}"
        _, k, v, lr0, lr1, lr2, B = self._qkv_lr(x_write)
        w0, w1, w2 = state
        decay, init_norms = self._decay_and_norms(B)
        gfn = swiglu_fast_weight_gradient if grad_fn is None else grad_fn

        def _write(k, v, w0, w1, w2, lr0, lr1, lr2):
            g0, g1, g2 = gfn(k, v, w0, w1, w2, lr0, lr1, lr2)
            return weight_norm_update(
                w0, w1, w2, g0, g1, g2, self.muon_update_steps, allreduce_fn,
                decay=decay, init_norms=init_norms)

        if micro_batch is None:
            use_ckpt = self.checkpoint_write if use_checkpoint is None else bool(use_checkpoint)
            if use_ckpt and torch.is_grad_enabled():
                # non-reentrant checkpoint: activations of the write are recomputed in backward;
                # gradients also flow to closure-captured tensors (decay <- decay_logit), which
                # use_reentrant=False supports (reentrant would silently drop them).
                return checkpoint(_write, k, v, w0, w1, w2, lr0, lr1, lr2, use_reentrant=False)
            return _write(k, v, w0, w1, w2, lr0, lr1, lr2)

        # ---- sequential micro-batch schedule (same two calls as `_write`, in a loop) ----
        m, L = int(micro_batch), int(k.shape[-2])
        assert m > 0, f"micro_batch must be > 0, got {micro_batch}"
        keep = (1.0, 1.0, 1.0)               # "no forgetting": the decay already fired, slice 0
        for s in range(0, L, m):
            e = min(s + m, L)
            g0, g1, g2 = gfn(k[..., s:e, :], v[..., s:e, :], w0, w1, w2,
                             lr0[..., s:e, :], lr1[..., s:e, :], lr2[..., s:e, :])
            w0, w1, w2 = weight_norm_update(
                w0, w1, w2, g0, g1, g2, self.muon_update_steps, allreduce_fn,
                decay=None if decay is None else (decay if s == 0 else keep),
                init_norms=init_norms)
        return w0, w1, w2


def replay_writes(layer, h_seq, detach_state=False, state=None, detach_init=True,
                  n_bptt_writes=None, use_checkpoint=None, allreduce_fn=None,
                  return_all=False, grad_fn=None, micro_batch=None):
    """Differentiable write-replay (memory 2.0): re-run a cell's whole write sequence on an
    ordered list of evicted-chunk hiddens, offline, keeping the graph toward the write
    parameters (qkv, k_norm, lrs, decay_logit).

    Premise (memory2_design.md 1.1): under MEM_ZERO_ALL=1 the h fed to the writes are pure
    functions of the frozen backbone (data, independent of the writer's parameters), so
    replaying `state <- write_theta(state, h_i)` from the init state reproduces the in-situ
    state trajectory exactly, bitwise, not approximately (selftest_replay.py asserts it).

    Args:
        layer: a VideoMemoryLayer (typically blocks[5/16/27].memory of the loaded pipeline),
            in the deployment dtype (bf16). Its own detach_state flag is irrelevant here:
            this function never calls step()/_detach(); graph control is explicit below.
        h_seq: ordered list of write inputs, h_seq[i] = the mem_x_write of the i-th write
            (the fifo-popped evicted chunk, shape (B, Lw, dim)), in write_chunks order.
            Must be on layer's device and in layer's compute dtype (cast fp16-stored shards
            back to bf16 before calling, exact for in-range values).
        detach_state: True cuts the graph after every write (values unchanged; useful for
            probing). False (default, the training mode) keeps the full graph.
        state: initial (w0, w1, w2); None -> layer.init_state(B, device, dtype) exactly like
            the first in-situ write pass (state is None until the first write pass, which
            initializes it from the w0/w1/w2 init parameters).
        detach_init: detach the initial state from the w0/w1/w2 init parameters (True =
            phase A/B, w-inits frozen; False = phase C, gradient flows into the inits).
        n_bptt_writes: truncated BPTT, only the last n writes build a graph, older writes
            run under no_grad (stop-gradient at the boundary). The forward is unchanged, so
            bitwise parity with inference is preserved (design 1.4 mitigation 3). None =
            full BPTT. 0 = pure forward.
        use_checkpoint: per-write gradient checkpointing; None -> layer.checkpoint_write.
        return_all: also return the list of per-write states (for the bitwise selftest).
        grad_fn, micro_batch: forwarded verbatim to write_only; None (default) = the historical
            write objective and the historical one-step schedule, bitwise (see write_only's
            docstring). Diagnostic replays only.

    Returns: final (w0, w1, w2), or (final, [per-write states]) if return_all."""
    assert len(h_seq) > 0, "replay_writes: empty h_seq"
    wdt = layer.qkv.weight.dtype
    for i, h in enumerate(h_seq):
        assert h.dim() == 3 and h.shape[-1] == layer.dim, \
            f"h_seq[{i}]: shape {tuple(h.shape)}, expected (B, Lw, {layer.dim})"
        assert h.dtype == wdt, \
            (f"h_seq[{i}] dtype {h.dtype} != module dtype {wdt}: cast stored h back to the "
             f"deployment dtype BEFORE replay (fp16 shard -> bf16), never inside the chain")
        assert h.device == layer.qkv.weight.device, \
            f"h_seq[{i}] on {h.device}, module on {layer.qkv.weight.device}"
    if state is None:
        state = layer.init_state(h_seq[0].shape[0], h_seq[0].device, wdt)
    if detach_init:
        state = tuple(t.detach() for t in state)

    n = len(h_seq)
    n_grad = n if n_bptt_writes is None else max(0, min(int(n_bptt_writes), n))
    cut = n - n_grad                     # writes [0, cut) run under no_grad (truncated BPTT)
    all_states = [] if return_all else None
    for i, h in enumerate(h_seq):
        if i < cut:
            with torch.no_grad():        # forward identical; only the graph is cut
                state = layer.write_only(h, state, allreduce_fn=allreduce_fn,
                                         use_checkpoint=use_checkpoint, grad_fn=grad_fn,
                                         micro_batch=micro_batch)
        else:
            state = layer.write_only(h, state, allreduce_fn=allreduce_fn,
                                     use_checkpoint=use_checkpoint, grad_fn=grad_fn,
                                     micro_batch=micro_batch)
        if detach_state:
            state = tuple(t.detach() for t in state)
        if return_all:
            all_states.append(state)
    return (state, all_states) if return_all else state


class FrameMixerBlock(nn.Module):
    """v2 anti-checkerboard spatial mixer block (memory2 phase B).

    The v1 decoder is strictly per token: each of the 30x52 queries reads the state on its own
    and the head expands it into a 2x2 latent patch, so nothing ever forces two neighbouring
    patches to agree, and the rendered frame carries a visible 2x2 checkerboard. This block
    adds the one thing v1 lacks, a local spatial mixing on the token grid, as a residual
    correction:

        x <- x + pw2(GELU(pw1(depthwise3x3(RMSNorm(x)))))

    With pw2 ZERO-INIT (weight and bias) so the block is exactly the identity at init: a decoder
    built with mixer=True and warm-started from a v1/A2 checkpoint computes the v1 function
    bitwise (selftest_phaseB test 1), and the extra keys are the only ones missing from the
    old state_dict."""

    def __init__(self, dim, hidden, kernel=3, eps=1e-5):
        super().__init__()
        self.norm = RMSNorm(dim, eps=eps)
        self.dw = nn.Conv2d(dim, dim, kernel, padding=kernel // 2, groups=dim)
        self.pw1 = nn.Conv2d(dim, hidden, 1)
        self.pw2 = nn.Conv2d(hidden, dim, 1)
        nn.init.zeros_(self.pw2.weight)     # identity at init: the mixer starts inert
        nn.init.zeros_(self.pw2.bias)

    def forward(self, x, grid_h, grid_w):
        """x: (B, m, dim) tokens in (h, w) row-major order -> (B, m, dim)."""
        B, m, dim = x.shape
        assert m == grid_h * grid_w, f"mixer: {m} tokens != {grid_h}x{grid_w}"
        g = self.norm(x).view(B, grid_h, grid_w, dim).permute(0, 3, 1, 2)   # (B, dim, gh, gw)
        g = self.pw2(F.gelu(self.pw1(self.dw(g))))
        g = g.permute(0, 2, 3, 1).reshape(B, m, dim)
        return x + g


class AnchorFrameDecoder(nn.Module):
    """State -> latent-frame decoder ("frame de rappel decodee", plan_final_sink_virtuel DESIGN v1).

    STANDALONE readout module: reads the fast-weight states of the (3) VideoMemoryLayer cells
    and synthesizes ONE latent video frame (B, 1, 16, 60, 104), to be spliced into the recache
    via the existing frames_override hook at the revisit switch (the exact oracle1 mechanism,
    +0.224 / 24+/0-). Trained OFFLINE on (states, evicted-frame) pairs -- this class is NOT
    wired into any forward or pipeline here (additive code only; nothing existing is touched).

    Inputs: `states` = sequence of num_blocks (w0, w1, w2) tuples, in memory_layers order
    (e.g. blocks [5, 16, 27]). Shapes exactly as produced by VideoMemoryLayer(num_heads=1)
    init_state()/step(): the leading axis is B*num_heads == B (nh=1 in causal_model.py),
    w0 (B, dim, dim*inter_multi), w1 (B, dim*inter_multi, dim), w2 like w0.
    States are INPUTS, not parameters: detached by default so no gradient ever reaches them
    (detach_states=False is the v2 end-to-end escape hatch, unused in v1 offline training).

    Architecture (localplan_B section 3.1 adapted to the frame readout of design 2):
      - m = 30*52 = 1560 learned queries, ONE PER 2x2 LATENT PATCH of the frame. A factorized
        learned 2D grid embedding (row 30 + col 52) is SHARED across blocks so the three reads
        of a given patch address the same spatial slot; the query TABLE is PER-BLOCK (the
        block-5/16/27 states live in depth-specific feature spaces, one shared table cannot
        speak all three key spaces).
      - per block b: q_norm (RMSNorm, mirroring the cell's own read path) on (queries_b + pos2d),
        then swiglu_apply(q, w0, w1, w2) -> read (B, 1560, 1536); per-block RMSNorm on the read
        (a plain normalization of a fresh read: nothing is subtracted anywhere in this module,
        so there is no renorm-after-subtraction).
      - fusion: concat of the 3 reads (4608) -> Linear 4608->3072 -> GELU -> Linear 3072->1536
        -> LayerNorm (per token; no cross-token mixing: the state is a global associative
        memory, each patch retrieves its own content by address).
      - head: Linear 1536 -> 16*2*2 = 64 per token, ZERO-INIT (weight AND bias) => the decoder
        emits an EXACTLY-NULL frame at init (never xavier noise into the cache), then
        unpatchify to (B, 1, 16, 60, 104) mirroring CausalWanModel.unpatchify's layout
        (token dim = (ph, pw, c), tokens (h, w) row-major).

    Param count (defaults dim=1536, num_blocks=3, grid 30x52, fuse_hidden=3072) = 26,304,064:
      queries 3*1560*1536 = 7,188,480 ; 2D grid emb (30+52)*1536 = 125,952 ;
      q_norms + read_norms 6*1536 = 9,216 ; fuse 4608->3072->1536 (+biases) = 18,878,976 ;
      out LayerNorm 3,072 ; head 1536->64 (+bias) = 98,368  => ~26.3M, under the 40M budget.

    v1.1 OPTIONAL CONDITIONING (recency-bias attack, plan_final CORRECTIF v1): cond_dim > 0
    adds a light MLP LayerNorm(cond_dim) -> Linear(cond_dim, cond_hidden) -> GELU ->
    Linear(cond_hidden, num_blocks*2*dim) producing per-block FiLM (scale, shift) applied to
    the NORMALIZED read queries: q <- q*(1+scale) + shift = "look in the state for what
    matches THIS description" (cond = the POOLED T5 embedding of the prompt whose content
    must be resurrected, i.e. prompt A at a revisit). The last linear is ZERO-INIT (weight
    AND bias) so at init scale = shift = 0 and the conditioned decoder computes EXACTLY the
    v1.0 function -- clean warm-start from a v1.0 checkpoint (its state_dict loads with
    strict=False; the only missing keys are cond_norm/cond_in/cond_out at their inert init).
    cond_dim = 0 (default) creates NO cond modules: state_dict and forward are byte-identical
    to v1.0. The input LayerNorm makes the FiLM robust to the pooled-embedding scale (the
    T5 mean pools over zero-padded positions, so the raw norm varies with prompt length).
    Extra params at cond_dim=4096, cond_hidden=512: 8,192 + 2,097,664 + 4,727,808 =
    6,833,664 => total 33,137,728, still under the 40M budget.

    v2 OPTIONAL SPATIAL MIXER (memory2 phase B, anti-checkerboard): mixer=True inserts
    `mixer_blocks` FrameMixerBlock AFTER the per-token reads/fusion and BEFORE the head, the
    only place in the module where two neighbouring 2x2 patches can talk to each other. Every
    block is zero-init on its last pointwise => the mixer is EXACTLY the identity at init, so
    a v1/A2 checkpoint warm-starts into it with strict=False and the ONLY missing keys are
    `mixer.*` (asserted by the trainer and by selftest_phaseB). mixer=False (default) creates
    NO module: state_dict and forward are byte-identical to v1.
    Extra params at dim=1536, mixer_hidden=768, 2 blocks: 2 * (1536 (norm) + 15,360 (dw) +
    1,180,416 (pw1) + 1,181,184 (pw2)) = 4,756,992 => total 37,894,720, still under 40M.
    """

    def __init__(
        self,
        dim: int = 1536,           # cell head_dim == dim (num_heads=1 in causal_model.py)
        inter_multi: int = 4,      # cell hidden multiplier -> dh = 6144
        num_blocks: int = 3,       # memory_layers cells, e.g. [5, 16, 27]
        grid_h: int = 30,          # latent 60 / patch 2
        grid_w: int = 52,          # latent 104 / patch 2
        patch_h: int = 2,
        patch_w: int = 2,
        z_dim: int = 16,           # latent channels
        fuse_hidden: int = 3072,
        detach_states: bool = True,  # v1 offline: states are data, never receive gradient
        cond_dim: int = 0,           # v1.1: pooled-prompt FiLM conditioning; 0 = off = exact v1.0
        cond_hidden: int = 512,      # hidden width of the cond MLP (only used when cond_dim > 0)
        mixer: bool = False,         # v2: spatial mixing on the token grid (default off = v1)
        mixer_blocks: int = 2,       # number of residual mixer blocks
        mixer_hidden: int = 768,     # pointwise bottleneck width of a mixer block
        mixer_kernel: int = 3,       # depthwise kernel
    ):
        super().__init__()
        self.dim = dim
        self.dh = dim * inter_multi
        self.num_blocks = num_blocks
        self.grid_h, self.grid_w = grid_h, grid_w
        self.patch_h, self.patch_w = patch_h, patch_w
        self.z_dim = z_dim
        self.m = grid_h * grid_w
        self.detach_states = detach_states

        # per-block query tables + shared factorized 2D grid embedding (init scale as in the
        # localplan_B queries: randn / sqrt(dim); the q_norm renormalizes anyway)
        self.queries = nn.Parameter(torch.randn(num_blocks, self.m, dim) / math.sqrt(dim))
        self.pos_row = nn.Parameter(torch.randn(grid_h, dim) / math.sqrt(dim))
        self.pos_col = nn.Parameter(torch.randn(grid_w, dim) / math.sqrt(dim))
        self.q_norms = nn.ModuleList([RMSNorm(dim, eps=1e-5) for _ in range(num_blocks)])
        self.read_norms = nn.ModuleList([RMSNorm(dim, eps=1e-5) for _ in range(num_blocks)])

        # fusion MLP (per token) + zero-init output head
        self.fuse1 = nn.Linear(num_blocks * dim, fuse_hidden)
        self.fuse2 = nn.Linear(fuse_hidden, dim)
        self.out_norm = nn.LayerNorm(dim, eps=1e-6)
        self.head = nn.Linear(dim, z_dim * patch_h * patch_w)
        nn.init.zeros_(self.head.weight)   # null frame at init, never xavier noise
        nn.init.zeros_(self.head.bias)

        # v1.1 optional FiLM conditioning on the read queries (cond_dim == 0 -> no modules
        # created: the unconditioned decoder is strictly the v1.0 module, same state_dict)
        # v2 spatial mixer (identity at init; no module created when mixer=False -> exact v1)
        self.use_mixer = bool(mixer)
        if self.use_mixer:
            assert mixer_blocks > 0, "mixer=True with mixer_blocks=0"
            self.mixer = nn.ModuleList([
                FrameMixerBlock(dim, mixer_hidden, kernel=mixer_kernel)
                for _ in range(mixer_blocks)])

        self.cond_dim = cond_dim
        self.cond_hidden = cond_hidden
        if cond_dim > 0:
            self.cond_norm = nn.LayerNorm(cond_dim, eps=1e-6)
            self.cond_in = nn.Linear(cond_dim, cond_hidden)
            self.cond_out = nn.Linear(cond_hidden, num_blocks * 2 * dim)
            nn.init.zeros_(self.cond_out.weight)  # FiLM identity at init: scale = shift = 0
            nn.init.zeros_(self.cond_out.bias)    # -> conditioned model == v1.0 model exactly

    def num_params(self):
        return sum(p.numel() for p in self.parameters())

    def _unpatchify(self, tok):
        """(B, m, z*ph*pw) -> (B, 1, z, H, W). Token dim laid out (ph, pw, c) and tokens in
        (h, w) row-major order, exactly mirroring CausalWanModel.unpatchify for one frame
        (its (f, h, w, pf, ph, pw, c) einsum with pf = f = 1)."""
        B = tok.shape[0]
        u = tok.view(B, self.grid_h, self.grid_w, self.patch_h, self.patch_w, self.z_dim)
        u = torch.einsum("bhwpqc->bchpwq", u)
        u = u.reshape(B, self.z_dim, self.grid_h * self.patch_h, self.grid_w * self.patch_w)
        return u.unsqueeze(1)  # (B, 1, 16, 60, 104)

    def forward(self, states, cond=None):
        """States: sequence of num_blocks (w0, w1, w2) tuples (memory_layers order).
        cond: optional pooled prompt embedding, (B, cond_dim) or (cond_dim,), the FiLM
        condition ("retrieve what matches this description"). Requires cond_dim > 0.
        cond=None with cond_dim > 0 runs the unconditioned read path (no FiLM applied),
        which at init is also the exact conditioned output (zero-init FiLM).
        Returns the decoded latent frame (B, 1, z_dim, grid_h*patch_h, grid_w*patch_w)."""
        assert len(states) == self.num_blocks, \
            f"expected {self.num_blocks} block states, got {len(states)}"
        pos = (self.pos_row[:, None, :] + self.pos_col[None, :, :]).reshape(self.m, self.dim)
        dtype = self.queries.dtype
        film = None
        if cond is not None:
            assert self.cond_dim > 0, \
                "cond passed to an UNconditioned decoder (built with cond_dim=0)"
            if cond.dim() == 1:  # unbatched (cond_dim,) convenience -> (1, cond_dim)
                cond = cond[None]
            assert cond.dim() == 2 and cond.shape[-1] == self.cond_dim, \
                f"cond {tuple(cond.shape)} != (B, {self.cond_dim})"
            if self.detach_states:  # cond is data in v1 offline training, like the states
                cond = cond.detach()
            f = self.cond_out(F.gelu(self.cond_in(self.cond_norm(cond.to(dtype)))))
            film = f.view(f.shape[0], self.num_blocks, 2, self.dim)  # (Bc, nb, {scale,shift}, dim)
        B = None
        reads = []
        for b, st in enumerate(states):
            assert st is not None and len(st) == 3, f"block {b}: state must be a (w0, w1, w2) tuple"
            w0, w1, w2 = st
            if w0.dim() == 2:  # unbatched (di, dh) convenience -> (1, di, dh)
                w0, w1, w2 = w0[None], w1[None], w2[None]
            if self.detach_states:
                w0, w1, w2 = w0.detach(), w1.detach(), w2.detach()
            assert w0.shape[-2:] == (self.dim, self.dh), \
                f"block {b}: w0 {tuple(w0.shape)} != (*, {self.dim}, {self.dh})"
            assert w1.shape[-2:] == (self.dh, self.dim), \
                f"block {b}: w1 {tuple(w1.shape)} != (*, {self.dh}, {self.dim})"
            assert w2.shape[-2:] == (self.dim, self.dh), \
                f"block {b}: w2 {tuple(w2.shape)} != (*, {self.dim}, {self.dh})"
            if B is None:
                B = w0.shape[0]
            assert w0.shape[0] == B and w1.shape[0] == B and w2.shape[0] == B, \
                f"block {b}: inconsistent state batch (expected leading axis B*nh with nh=1)"
            w0, w1, w2 = w0.to(dtype), w1.to(dtype), w2.to(dtype)
            q = self.q_norms[b](self.queries[b] + pos)              # (m, dim)
            q = q.unsqueeze(0).expand(B, -1, -1)                    # (B, m, dim)
            if film is not None:
                fb = film.expand(B, -1, -1, -1) if film.shape[0] == 1 else film
                assert fb.shape[0] == B, \
                    f"cond batch {film.shape[0]} != state batch {B} (and not broadcastable)"
                # FiLM after the q_norm so the modulation is not renormalized away; at the
                # zero-init (scale = shift = 0) this is q*1 + 0 == q exactly (v1.0 identity)
                q = q * (1.0 + fb[:, b, 0].unsqueeze(1)) + fb[:, b, 1].unsqueeze(1)
            r = swiglu_apply(q, w0, w1, w2)                         # read: (B, m, dim)
            reads.append(self.read_norms[b](r))
        h = torch.cat(reads, dim=-1)                                # (B, m, 3*dim)
        h = self.fuse2(F.gelu(self.fuse1(h)))
        if getattr(self, "use_mixer", False):                       # v2: inter-token spatial
            for blk in self.mixer:                                  # mixing, identity at init
                h = blk(h, self.grid_h, self.grid_w)
        tok = self.head(self.out_norm(h))                           # (B, m, 64), == 0 at init
        return self._unpatchify(tok)
