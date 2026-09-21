"""The fast-weight memory cell: a small MLP whose weights are rewritten during generation.

The read, the write gradient and the Muon update follow LaCT (Large Chunk Test-Time Training,
Zhang et al.), whose reference implementation this adapts; the names of its functions are kept
so the two can be compared side by side. What is ours is the causal chunk-by-chunk stepping,
the renormalization back to the initial norms, the gated injection and the eviction feed.
"""
import math
from typing import Optional, Sequence

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint

from utils.env import env_int


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
    mutates the shared object and survives the call stack. (LaCT threads a tuple instead, which
    works in its single-process inference path but not under our FSDP training stack.)"""
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

    def snapshot(self):
        """The cache as it stands, to be handed back to restore(). Shallow on purpose: state
        tuples, EMA entries and FIFO entries are always replaced, never edited in place, so
        holding the references is enough and costs no memory."""
        return (self.state, dict(self.ema), list(self.fifo))

    def restore(self, snap):
        state, ema, fifo = snap
        # fresh containers, so later writes to this cache leave the snapshot reusable
        self.state, self.ema, self.fifo = state, dict(ema), list(fifo)


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
    """ f_W(q) = (silu(q @ w0) * (q @ w2)) @ w1, LaCT's apply formula (the `oi = ...` line of
    fast_weight_swish_glu_weight_norm_mini_batch_apply), pulled out so the causal step() can
    apply before it writes (apply-first). """
    return (F.silu(q @ w0, inplace=False) * (q @ w2)) @ w1


def swiglu_fast_weight_gradient(k, v, w0, w1, w2, lr0, lr1, lr2):
    """ Raw inner-loop gradient of dot = <f_W(k), v> wrt (w0, w1, w2), token-wise lr folded in.

    Body is identical to LaCT's fast_weight_swish_glu_weight_norm_mini_batch_gradient; the only
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
    Default (decay=None): LaCT renorm, ascend the Muon'd grad, then renorm each column to its
    pre-update norm (norm conserved, no forgetting; the state only rotates in a fixed-norm shell).
    Gated-delta (decay/init_norms set): w <- decay*w + muon(grad), then cap each column to its init norm
    (shrink allowed = explicit forgetting). `allreduce_fn` = LaCT's context-parallel allreduce,
    on the raw gradients.
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
        # ascend, then renorm each column back to its pre-update norm (no forgetting)
        w0 = w0 + w0_grad
        w1 = w1 + w1_grad
        w2 = w2 + w2_grad
        w0 = w0 / (w0.norm(dim=1, keepdim=True) + 1e-5) * w0_norm
        w1 = w1 / (w1.norm(dim=1, keepdim=True) + 1e-5) * w1_norm
        w2 = w2 / (w2.norm(dim=1, keepdim=True) + 1e-5) * w2_norm
        return w0, w1, w2

    # gated-delta: decay (forget) the old state, add the update, cap each column norm to its init norm
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

    This is the deployment side of the rule that was validated offline: same operations, same
    order, same dtypes, same shapes, so the state built during generation matches the one the
    offline replay builds. Any edit here has to preserve that. No learned parameter."""
    assert x.dim() == 2 and x.dtype == torch.float32, f"tokens {tuple(x.shape)} {x.dtype}"
    assert W.dim() == 2 and W.dtype == torch.float32, f"state {tuple(W.shape)} {W.dtype}"
    nrm = x.norm(dim=-1, keepdim=True)
    assert torch.isfinite(nrm).all() and bool((nrm > 0).all()), "token of zero norm"
    K = x / nrm                                   # k = h / ||h||, v = h
    for t in range(x.shape[0]):                   # sequential, one token at a time
        k = K[t]
        W.add_(torch.outer(x[t] - W @ k, k), alpha=beta)
    return W


class VideoMemoryLayer(nn.Module):
    """The memory cell: a fast-weight state read with the current chunk and rewritten with the
    chunk that left the attention window. Call `step` once per generator call; what it returns
    is the gated contribution to add to the residual stream.

    A causal adaptation of LaCT's FastWeightGluMLPMultihead: same qkv/lr projection, inner-loop
    gradient, Muon update and gated AdaLN output, stepped chunk by chunk instead of applied to a
    bidirectional sequence.

    Arguments
      num_heads, inter_multi  one head of width dim, hidden width 4 * dim: the largest state,
                              which is what test-time training benefits from
      base_lr                 the inner learning rate the per-token rates start from
      muon_update_steps       Newton-Schulz iterations on the write gradient; 0 = plain SGD
      qk_norm                 RMSNorm on q and k, for stability over a long rollout
      g_init                  initial gate bias; 0 makes the branch inert at step 0
      use_decay, decay_init   let the state forget: w <- decay * w + update, norms capped at
                              their init value, instead of renormalising to a fixed norm
      bounded_gate            tanh on the gate, so the injection cannot grow without bound
      bounded_mod             tanh on scale and shift too
      rel_norm                normalise the read by one RMS per chunk instead of per token, so
                              a token's read magnitude, its relevance, reaches the gate
      single_norm             without rel_norm: skip the per-token norm of the read
      dc_remove               subtract the token mean of the injection, so the branch cannot
                              learn a constant bias
      detach_state            None = detach the new state in eval, keep its graph in training
      checkpoint_write        gradient-checkpoint the write
      gate_extra              extra gate inputs among "agreement", "denoising_step",
                              "frames_since_switch"; zero-initialised, so inert until trained
      state_type              "swiglu", the fast-weight MLP (w0, w1, w2), or "linear_delta":
                              one fp32 matrix W per block, written by a training-free delta
                              rule on the raw evicted tokens and read as W q. Six times
                              smaller, and the module keeps every SwiGLU parameter, so a
                              checkpoint loads in both modes
      linear_beta             step size of that delta rule
    """
    def __init__(
        self,
        dim: int = 1536,
        num_heads: int = 1,
        inter_multi: int = 4,
        base_lr: float = 1e-2,
        muon_update_steps: int = 5,
        qk_norm: bool = True,
        g_init: float = 0.0,
        use_decay: bool = False,
        decay_init: float = 0.95,
        bounded_gate: bool = False,
        single_norm: bool = False,
        bounded_mod: bool = False,
        rel_norm: bool = False,
        dc_remove: bool = False,
        detach_state: Optional[bool] = None,
        checkpoint_write: bool = False,
        gate_extra: Optional[Sequence[str]] = None,
        state_type: str = "swiglu",
        linear_beta: float = 0.5,
    ):
        super().__init__()
        assert dim % num_heads == 0, "dim should be divisible by num_heads"
        assert state_type in ("swiglu", "linear_delta"), f"unknown state_type {state_type!r}"
        # the delta rule is defined on whole tokens; splitting them per head is another rule
        assert state_type == "swiglu" or num_heads == 1, "linear_delta state requires num_heads == 1"
        self.state_type = state_type
        self.linear_beta = float(linear_beta)

        self.dim = dim
        self.g_init = g_init
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.muon_update_steps = muon_update_steps
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
        # one projection of the read gives the shift, the scale and the gate of the injection
        self.proj = nn.Linear(dim, dim * 3, bias=True)
        # Adaptive layer normalization without affine parameters
        self.a_norm = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self._init_modulation_params(dim, g_init=g_init)
        if self.gate_extra:
            self._build_gate_extra(dim)

        # TTT fast weight dimensions
        di = self.head_dim
        do = self.head_dim
        dh = self.head_dim * inter_multi

        # fast weight initialization, as in LaCT: 1/sqrt(fan_in), no sqrt(2) gain
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
        """Zero projection, so shift and scale start at 0 and the gate at its bias g_init."""
        with torch.no_grad():
            nn.init.zeros_(self.proj.weight)
            nn.init.zeros_(self.proj.bias)
            self.proj.bias[2 * dim:] = g_init       # proj outputs [shift | scale | gate]

    # ---- optional extra inputs of the gate ----------------------------------------------------
    GX_RANK = 64   # rank of the space where the read and the prompt are compared
    GX_EMB = 16    # width of the sinusoidal embedding of a scalar input
    # width of each input, in the order they are concatenated whatever order the config lists
    # them in: the columns of proj_extra are logged per input by the trainer
    GATE_INPUTS = {"agreement": 1, "denoising_step": GX_EMB, "frames_since_switch": GX_EMB}

    def _build_gate_extra(self, dim: int):
        unknown = [g for g in self.gate_extra if g not in self.GATE_INPUTS]
        assert not unknown, \
            f"unknown gate_extra entries {unknown}, expected subset of {sorted(self.GATE_INPUTS)}"
        if "agreement" in self.gate_extra:
            self.gx_po = nn.Linear(dim, self.GX_RANK, bias=False)     # of the memory read
            self.gx_pc = nn.Linear(dim, self.GX_RANK, bias=False)     # of the pooled prompt
        width = sum(w for name, w in self.GATE_INPUTS.items() if name in self.gate_extra)
        # added to proj's output, and zero-initialised: the inputs do nothing until trained
        self.proj_extra = nn.Linear(width, 3 * dim, bias=False)
        nn.init.zeros_(self.proj_extra.weight)

    def _gate_extra_vec(self, o_hat, prompt_emb, frames_since_switch, denoising_step):
        """The (B, L, width) input of proj_extra."""
        given = {"agreement": prompt_emb, "denoising_step": denoising_step,
                 "frames_since_switch": frames_since_switch}
        feats = []
        for name in self.GATE_INPUTS:
            if name not in self.gate_extra:
                continue
            if given[name] is None:
                raise ValueError(f"gate_extra {name!r} is on but its input was not passed to step()")
            if name == "agreement":
                # how much the read agrees with the prompt, one scalar in [0, 1] per token
                prompt = prompt_emb.mean(dim=1)                                 # (B, dim)
                a = torch.sigmoid((self.gx_po(o_hat) * self.gx_pc(prompt)[:, None, :])
                                  .sum(-1, keepdim=True))
                self.dbg_agreement = a.detach().float().mean()                  # 0-dim: no host sync
                feats.append(a.to(o_hat.dtype))
            else:
                feats.append(_sinusoid_embed(given[name], self.GX_EMB).to(o_hat.dtype))
        return torch.cat(feats, dim=-1)

    def reset_parameters(self):
        """Re-apply the branch's own initialisation. To be called after any global init of the
        model: CausalWanModel.init_weights xavier-initialises every nn.Linear, this one included."""
        self._init_modulation_params(self.dim, g_init=self.g_init)
        if self.gate_extra:
            nn.init.zeros_(self.proj_extra.weight)

    def _qkv_lr(self, x):
        """x (B, L, dim) -> q, k, v and the three per-token learning rates, each
        (B * heads, L, .), plus B. q and k are RMS-normalised per head."""
        B = x.shape[0]
        q, k, v = rearrange(self.qkv(x), "b l (qkv h d) -> qkv b h l d", qkv=3, h=self.num_heads)
        q, k = self.q_norm(q), self.k_norm(k)
        q, k, v = (rearrange(t, "b h l d -> (b h) l d") for t in (q, k, v))

        with torch.amp.autocast("cuda", enabled=False):
            # in fp32 whatever the module's dtype. LaCT writes self.lrs(x.float()), which fails
            # on a bf16 module (fp32 input, bf16 weight), so the small weights are cast instead.
            lrs = F.linear(x.float(), self.lrs.weight.float(),
                           self.lrs.bias.float() if self.lrs.bias is not None else None)
        lrs = F.softplus(lrs.float() + self.inv_base_lr)
        lr0, lr1, lr2 = rearrange(lrs, "b l (n h c) -> n (b h) l c", n=3, h=self.num_heads)
        # LaCT leaves them in fp32 for autocast to reconcile; the pipelines here run in plain
        # bf16 with no autocast, so they are cast to the compute dtype
        return q, k, v, lr0.type_as(k), lr1.type_as(k), lr2.type_as(k), B

    def _project_output(self, x, batch_size, prompt_emb=None,
                        frames_since_switch=None, denoising_step=None, num_frames=None):
        """The read -> the injection: gate * (carrier * (1 + scale) + shift), where shift, scale
        and gate are one projection of the normalised read and the carrier is that read."""
        if self.rel_norm:
            # one RMS per chunk: the growth of the state with the write count cancels, while a
            # token's read magnitude relative to the others survives. eps inside the sqrt keeps
            # the backward finite.
            rms = x.float().pow(2).mean(dim=(1, 2), keepdim=True).add(1e-10).sqrt()
            x = (x.float() / rms).type_as(x)
        elif not self.single_norm:
            x = self.o_norm(x)                  # per token, which erases the read magnitude
        x = rearrange(x, '(b h) l d -> b l (h d)', h=self.num_heads, b=batch_size)
        shift, scale, gate = self.proj(x).chunk(3, dim=-1)
        if self.gate_extra:                     # through a zero-init projection: inert until trained
            extra = self._gate_extra_vec(x, prompt_emb, frames_since_switch, denoising_step)
            es, esc, eg = self.proj_extra(extra).chunk(3, dim=-1)
            shift, scale, gate = shift + es, scale + esc, gate + eg
        if self.bounded_gate:
            gate = torch.tanh(gate)
        if self.bounded_mod:
            scale, shift = torch.tanh(scale), torch.tanh(shift)
        # under rel_norm the amplitude of the carrier tracks the relevance of the read;
        # otherwise a_norm pins it to unit variance
        carrier = x if self.rel_norm else self.a_norm(x)
        out = gate * (carrier * (1 + scale) + shift)
        self._record_diagnostics(out, gate, num_frames)
        if self.dc_remove or env_int("MEM_DC_REMOVE"):
            out = out - out.mean(dim=1, keepdim=True)
        return out

    def _record_diagnostics(self, out, gate, num_frames):
        """Health numbers of the injection before the DC removal, kept as 0-dim tensors so that
        nothing here forces a host sync; the trainer reads them at log time."""
        o = out.detach().float()
        self.dbg_dc_frac = o.mean(dim=1, keepdim=True).pow(2).mean() / o.pow(2).mean().clamp_min(1e-12)
        tok = o.pow(2).mean(dim=-1).sqrt()                          # (b, l) per-token injection RMS
        self.dbg_amp_disp = tok.std() / tok.mean().clamp_min(1e-12) # 0 = flat = amplitude still decoupled
        # split the joint dispersion into the wanted spatial axis (relevance coupling across
        # sites) vs the unwanted temporal axis (per-site frame-to-frame std = the flicker). Needs the
        # frame count to reshape l=(F*S). Health target: temporal small while spatial stays > 0.
        if num_frames is not None and int(num_frames) > 0 and tok.shape[1] % int(num_frames) == 0:
            _F = int(num_frames)
            amp = tok.reshape(tok.shape[0], _F, tok.shape[1] // _F)   # (b, F, S)
            site = amp.mean(dim=1)                                       # (b, S) time-mean amp / site
            self.dbg_ampdisp_spatial = site.std() / site.mean().clamp_min(1e-12)
            fstd = amp.std(dim=1)                                        # (b, S) across-frame std / site
            self.dbg_ampdisp_temporal = fstd.mean() / amp.mean().clamp_min(1e-12)
        self.dbg_gate_mean = gate.detach().float().abs().mean()

    def init_state(self, B, device, dtype):
        """The state before any write: the learned (w0, w1, w2), or W = 0 for the linear rule,
        which stays in fp32 whatever `dtype` says because its rounding is part of the rule."""
        if self.state_type == "linear_delta":
            return (torch.zeros(B * self.num_heads, self.head_dim, self.head_dim,
                                device=device, dtype=torch.float32),)
        return tuple(w.repeat(B, 1, 1).to(device=device, dtype=dtype)
                     for w in (self.w0, self.w1, self.w2))

    def _linear_write(self, x_write, W):
        """The delta-rule write: a new W from the raw tokens x_write (B, Lw, dim).

        It has to be the same function of the data as the offline replay that validated the
        rule. So each x_write[b] goes to linear_delta_chunk_update as a 2-D fp32 tensor, as the
        replay does (bf16 to fp32 is exact); TF32 is switched off around the update, so the
        result does not depend on the caller's backend flags; and W is cloned before being
        updated, so the state tuples held by a snapshot are never mutated. No grad: the rule
        has no learned parameter, and the read stays differentiable with respect to q."""
        with torch.no_grad():
            tf32 = torch.backends.cuda.matmul.allow_tf32
            torch.backends.cuda.matmul.allow_tf32 = False
            try:
                new_W = W.detach().clone()
                for b in range(x_write.shape[0]):
                    linear_delta_chunk_update(x_write[b].float(), new_W[b], self.linear_beta)
            finally:
                torch.backends.cuda.matmul.allow_tf32 = tf32
        return new_W

    def _detach(self):
        """Cut the graph of the new state? Yes in eval, no in training, unless detach_state says."""
        return (not self.training) if self.detach_state is None else self.detach_state

    def _decay_and_norms(self, B):
        """Under use_decay: the forget gate, one learned scalar used for the three weights, and
        the column norms of the init weights, the ceiling weight_norm_update caps the state at.
        (None, None) otherwise, and weight_norm_update then renormalises to a fixed norm."""
        if not self.use_decay:
            return None, None
        decay = torch.sigmoid(self.decay_logit)
        init_norms = tuple(w.detach().norm(dim=1, keepdim=True).repeat(B, 1, 1)
                           for w in (self.w0, self.w1, self.w2))
        return (decay, decay, decay), init_norms

    # ---- AR / inference / rollout: one temporal chunk per call ----
    def step(self, x, state=None, do_update=True, allreduce_fn=None,
             prompt_emb=None, frames_since_switch=None, denoising_step=None, num_frames=None,
             x_write=None):
        """One chunk: read the state, then write to it. Returns (mem_out (B, L, dim), new_state).

        x (B, L, dim) is the current chunk; the read query always comes from it. The write
        stores the k -> v association of `x_write` (B, Lw, dim), the chunk that just left the
        attention window, so the memory holds what attention can no longer see; with
        x_write=None it stores the current chunk instead. The read uses the state as it was
        before this write. prompt_emb, frames_since_switch and denoising_step feed the optional
        gate_extra inputs; num_frames only feeds the diagnostics."""
        q, k, v, lr0, lr1, lr2, B = self._qkv_lr(x)
        if state is None:
            state = self.init_state(B, x.device, q.dtype)
        o = self._read(q, state)
        new_state = state
        if do_update:
            if self.state_type == "linear_delta":
                assert allreduce_fn is None, "linear_delta: context-parallel write not supported"
                new_state = (self._linear_write(x if x_write is None else x_write, state[0]),)
            else:
                if x_write is not None:
                    _, k, v, lr0, lr1, lr2, _ = self._qkv_lr(x_write)
                new_state = self._swiglu_write(k, v, lr0, lr1, lr2, state, B, allreduce_fn)
            if self._detach():
                new_state = tuple(t.detach() for t in new_state)
        out = self._project_output(o, B, prompt_emb=prompt_emb,
                                   frames_since_switch=frames_since_switch,
                                   denoising_step=denoising_step, num_frames=num_frames)
        # 0-dim tensors, so no host sync here; the trainer calls .item() at log time
        self.dbg_read_rms = o.detach().float().pow(2).mean().sqrt()
        self.dbg_out_rms = out.detach().float().pow(2).mean().sqrt()
        return out, new_state

    def _read(self, q, state):
        """f_W(q) for the SwiGLU state, W q for the linear one. The linear state is fp32, so the
        product is taken in fp32 and cast back: everything downstream sees the compute dtype."""
        if self.state_type == "linear_delta":
            return torch.matmul(q.float(), state[0].transpose(-1, -2)).type_as(q)
        return swiglu_apply(q, *state)

    def _swiglu_write(self, k, v, lr0, lr1, lr2, state, B, allreduce_fn=None,
                      use_checkpoint=None, grad_fn=None, micro_batch=None):
        """The SwiGLU write, from projected keys, values and per-token learning rates. The one
        place it is written: step() and write_only() both end here, so an offline replay builds
        bitwise the states of the rollout it replays.

        grad_fn replaces the inner-loop gradient (same signature as swiglu_fast_weight_gradient)
        and micro_batch=m makes the write sequential, m tokens at a time in index order, each
        slice seeing the state the previous one left. Both are for diagnostic replays; no
        training or inference path sets them. Under micro_batch the forget gate fires on the
        first slice only, so the total is still decay * w + the sum of the updates, and the norm
        cap applies at every slice because each intermediate state is read by the next."""
        w0, w1, w2 = state
        decay, init_norms = self._decay_and_norms(B)
        gfn = swiglu_fast_weight_gradient if grad_fn is None else grad_fn

        def update(k, v, w0, w1, w2, lr0, lr1, lr2, decay=decay):
            g0, g1, g2 = gfn(k, v, w0, w1, w2, lr0, lr1, lr2)
            return weight_norm_update(w0, w1, w2, g0, g1, g2, self.muon_update_steps,
                                      allreduce_fn, decay=decay, init_norms=init_norms)

        if micro_batch is None:
            use_ckpt = self.checkpoint_write if use_checkpoint is None else bool(use_checkpoint)
            if use_ckpt and torch.is_grad_enabled():
                # non-reentrant: gradients also reach the tensors the closure captures
                # (decay <- decay_logit), which the reentrant variant silently drops
                return checkpoint(update, k, v, w0, w1, w2, lr0, lr1, lr2, use_reentrant=False)
            return update(k, v, w0, w1, w2, lr0, lr1, lr2)

        m, L = int(micro_batch), int(k.shape[-2])
        assert m > 0, f"micro_batch must be > 0, got {micro_batch}"
        for s in range(0, L, m):
            e = min(s + m, L)
            slice_decay = decay if (decay is None or s == 0) else (1.0, 1.0, 1.0)
            w0, w1, w2 = update(k[..., s:e, :], v[..., s:e, :], w0, w1, w2,
                                lr0[..., s:e, :], lr1[..., s:e, :], lr2[..., s:e, :],
                                decay=slice_decay)
        return w0, w1, w2

    def write_only(self, x_write, state, allreduce_fn=None, use_checkpoint=None, grad_fn=None,
                   micro_batch=None):
        """The write of step() without the read, for replaying a rollout's writes offline:
        state -> new (w0, w1, w2) from x_write, the hidden states of an evicted chunk.

        The graph is kept (no detach), so a replay can train the write parameters through the
        whole chain. For the states to match the rollout bitwise, the caller must give the module
        and x_write in the deployment dtype (bf16), run without ambient autocast, and keep the
        TF32 flags and GPU model of the rollout. grad_fn and micro_batch: see _swiglu_write."""
        assert self.state_type == "swiglu", \
            f"write_only is undefined for state_type={self.state_type!r}"
        _, k, v, lr0, lr1, lr2, B = self._qkv_lr(x_write)
        return self._swiglu_write(k, v, lr0, lr1, lr2, state, B, allreduce_fn,
                                  use_checkpoint, grad_fn, micro_batch)


def replay_writes(layer, h_seq, detach_state=False, state=None, detach_init=True,
                  n_bptt_writes=None, use_checkpoint=None, allreduce_fn=None,
                  return_all=False, grad_fn=None, micro_batch=None):
    """Re-run a cell's whole sequence of writes offline, on the ordered hidden states of the
    chunks it evicted, keeping the graph toward the write parameters (qkv, k_norm, lrs,
    decay_logit). Returns the final (w0, w1, w2), or (final, [state after each write]) with
    return_all.

    The replay is exact and not an approximation because, with the injection zeroed
    (MEM_ZERO_ALL=1), the hidden states fed to the writes are functions of the frozen backbone
    alone: they do not depend on the write parameters, so replaying
    state <- write(state, h_i) from the init state retraces the rollout's states.

    layer           a VideoMemoryLayer in the deployment dtype (bf16). Its detach_state flag
                    plays no part: the graph is controlled by the arguments below.
    h_seq           h_seq[i] (B, Lw, dim) is the input of the i-th write, on the layer's device
                    and in its dtype: cast fp16 shards back to bf16 before calling, never
                    inside the chain
    state           where to start; None = the init state, as the first write of a rollout
    detach_init     keep the init weights out of the graph (True) or train them too
    n_bptt_writes   only the last n writes build a graph; None = all, 0 = a pure forward.
                    The values do not change, so parity with the rollout holds either way.
    detach_state    cut the graph after every write, for probing
    use_checkpoint, grad_fn, micro_batch, allreduce_fn   passed on to write_only"""
    assert h_seq, "replay_writes: empty h_seq"
    weight = layer.qkv.weight
    for i, h in enumerate(h_seq):
        assert h.dim() == 3 and h.shape[-1] == layer.dim, \
            f"h_seq[{i}]: shape {tuple(h.shape)}, expected (B, Lw, {layer.dim})"
        assert h.dtype == weight.dtype, \
            (f"h_seq[{i}] dtype {h.dtype} != module dtype {weight.dtype}: cast stored h back to "
             f"the deployment dtype BEFORE replay (fp16 shard -> bf16), never inside the chain")
        assert h.device == weight.device, f"h_seq[{i}] on {h.device}, module on {weight.device}"
    if state is None:
        state = layer.init_state(h_seq[0].shape[0], h_seq[0].device, weight.dtype)
    if detach_init:
        state = tuple(t.detach() for t in state)

    n_grad = len(h_seq) if n_bptt_writes is None else max(0, min(int(n_bptt_writes), len(h_seq)))
    first_with_grad = len(h_seq) - n_grad
    all_states = []
    for i, h in enumerate(h_seq):
        with torch.set_grad_enabled(i >= first_with_grad and torch.is_grad_enabled()):
            state = layer.write_only(h, state, allreduce_fn=allreduce_fn,
                                     use_checkpoint=use_checkpoint, grad_fn=grad_fn,
                                     micro_batch=micro_batch)
        if detach_state:
            state = tuple(t.detach() for t in state)
        if return_all:                  # otherwise each state is freed as the next replaces it
            all_states.append(state)
    return (state, all_states) if return_all else state
