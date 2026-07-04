import math
from typing import Optional, Sequence, Tuple

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

    Deliberately a plain object and NOT a dict: FSDP mixed-precision input casting and
    activation/gradient checkpointing traverse kwargs as pytrees and REBUILD dict/list
    containers (a fresh dict per forward call), silently swallowing in-place dict writes.
    A custom object is a pytree LEAF: passed by reference, so `cache.state = new_state`
    mutates the shared object and survives the call stack. (Scal3R threaded a tuple instead,
    which works in its single-process inference path but not under our FSDP training stack.)"""
    __slots__ = ("state",)

    def __init__(self):
        self.state = None


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
    """ f_W(q) = (silu(q @ w0) * (q @ w2)) @ w1 -- Scal3R's apply formula (the `oi = ...` line of
    fast_weight_swish_glu_weight_norm_mini_batch_apply), pulled out so the causal step() can
    APPLY before it writes (apply-first). """
    return (F.silu(q @ w0, inplace=False) * (q @ w2)) @ w1


def swiglu_fast_weight_gradient(k, v, w0, w1, w2, lr0, lr1, lr2):
    """ Raw inner-loop gradient of dot = <f_W(k), v> wrt (w0, w1, w2), token-wise lr folded in.

    Body is identical to Scal3R's fast_weight_swish_glu_weight_norm_mini_batch_gradient; the only
    change is the signature is pre-sliced to a single causal chunk, so there is no ttt_ua_order to
    index (k, v, lr* are already the chunk's tokens). No minus sign: the fast weight ASCENDS this
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

    # NOTE: we return the raw local gradients here, not the orthogonalized ones
    # NOTE: the muon update is applied in the update method (here: weight_norm_update)
    return w0_grad, w1_grad, w2_grad


def weight_norm_update(w0, w1, w2, w0_grad, w1_grad, w2_grad, muon_update_steps, allreduce_fn=None,
                       decay=None, init_norms=None):
    """ Muon (Newton-Schulz) + fast-weight update.
    Default (decay=None): Scal3R renorm -- ascend the Muon'd grad, then renorm each row to its
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
    """vals: (...,) scalars -> (..., dim) parameter-free sinusoidal embedding."""
    half = max(dim // 2, 1)
    freqs = torch.exp(-math.log(10000.0) * torch.arange(half, device=vals.device).float() / half)
    ang = vals.float().unsqueeze(-1) * freqs
    return torch.cat([ang.sin(), ang.cos()], dim=-1)


class VideoMemoryLayer(nn.Module):
    """ Causal adaptation of Scal3R's FastWeightGluMLPMultihead.

    Same fast-weight SwiGLU memory (qkv/lr projection, inner-loop gradient, Muon + weight-norm
    update, gated AdaLN output), wired for autoregressive video instead of bidirectional VGGT
    patches. Use `step(x, state, do_update)` per generator call (AR/inference/rollout) or
    `forward_sequence(x, chunks, clean_mask)` for the fully-parallel training path. Output is the
    gated contribution to ADD to the residual.
    """
    def __init__(
        self,
        dim: int = 1536,
        num_heads: int = 1,        # nh=1 -> head_dim=dim, max state size (large head_dim is better for TTT)
        inter_multi: int = 4,      # k=4
        base_lr: float = 1e-2,
        muon_update_steps: int = 5,  # 0 = plain SGD on the fast weight
        qk_norm: bool = True,      # RMSNorm on head_dim (on for AR stability; Scal3R default False)
        use_modulation: bool = True,
        g_init: float = 0.0,       # 0.0 -> inert warm start (identical to baseline at step 0; Scal3R: 0.1)
        use_decay: bool = False,   # gated-delta forgetting in weight_norm_update (else renorm-to-init)
        decay_init: float = 0.95,  # initial forget gate value = sigmoid(decay_logit); 1.0 = no forgetting
        bounded_gate: bool = False,  # tanh-bound the output gate (kills the unbounded fixed-RMS injection)
        single_norm: bool = False,   # read-path hygiene: drop the redundant o_norm so proj sees the raw
                                     # read magnitude (relevance-coupled injection); keep only AdaLN a_norm
        detach_state: Optional[bool] = None,  # None -> detach in eval, keep graph in train
        checkpoint_write: bool = False,  # gradient-checkpoint the write update
        gate_extra: Optional[Sequence[str]] = None,  # extra gate inputs (off=[]): "agreement","denoising_step","frames_since_switch"
    ):
        super().__init__()
        assert dim % num_heads == 0, "dim should be divisible by num_heads"

        self.dim = dim
        self.g_init = g_init
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.muon_update_steps = muon_update_steps
        self.use_modulation = use_modulation
        self.use_decay = use_decay
        self.bounded_gate = bounded_gate
        self.single_norm = single_norm
        self.detach_state = detach_state
        self.checkpoint_write = checkpoint_write
        self.gate_extra = list(gate_extra) if gate_extra else []
        if use_decay:
            # one learned forget gate (scalar), sigmoid(decay_logit); init to decay_init
            self.decay_logit = nn.Parameter(torch.tensor(math.log(decay_init / (1.0 - decay_init))))

        # Query, Key, Value linear layers
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.q_norm = RMSNorm(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = RMSNorm(self.head_dim) if qk_norm else nn.Identity()

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

        # TTT fast weight parameters initialization -- LaCT/Scal3R: 1/sqrt(fan_in) (no sqrt(2) gain)
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
        """Re-apply the memory branch's own initialization. must be called after any global
        model init (e.g. CausalWanModel.init_weights(), which xavier-inits every nn.Linear in
        the model)."""
        if self.use_modulation:
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
                        frames_since_switch=None, denoising_step=None):
        if not self.single_norm:               # default: legacy double-norm (o_norm then a_norm)
            x = self.o_norm(x)                 # single_norm=True -> skip, let raw read magnitude reach proj
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
        return gate * (self.a_norm(x) * (1 + scale) + shift)

    def init_state(self, B, device, dtype):
        return (self.w0.repeat(B, 1, 1).to(device=device, dtype=dtype),
                self.w1.repeat(B, 1, 1).to(device=device, dtype=dtype),
                self.w2.repeat(B, 1, 1).to(device=device, dtype=dtype))

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
             prompt_emb=None, frames_since_switch=None, denoising_step=None):
        """x:(B,L,dim) for the current chunk. Returns (mem_out:(B,L,dim), new_state).
        prompt_emb/frames_since_switch/denoising_step feed the optional gate_extra inputs."""
        q, k, v, lr0, lr1, lr2, B = self._qkv_lr(x)
        if state is None:
            state = self.init_state(B, x.device, q.dtype)
        w0, w1, w2 = state
        o = swiglu_apply(q, w0, w1, w2)                       # APPLY-FIRST (state from < this chunk)
        new_state = state
        if do_update:                                          # write the clean chunk
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
                                   denoising_step=denoising_step)
        # lightweight diagnostics (0-dim tensors -> no host sync here; .item() at log time)
        self.dbg_read_rms = o.detach().float().pow(2).mean().sqrt()
        self.dbg_out_rms = out.detach().float().pow(2).mean().sqrt()
        return out, new_state

    # ---- training (fully parallel over all chunks of a clip) ----
    def forward_sequence(self, x, chunks: Sequence[Tuple[int, int]], clean_mask: Sequence[bool],
                         allreduce_fn=None):
        q, k, v, lr0, lr1, lr2, B = self._qkv_lr(x)
        w0, w1, w2 = self.init_state(B, x.device, q.dtype)
        decay, init_norms = self._decay_and_norms(B)
        o = torch.zeros_like(q)
        for c, (s, e) in enumerate(chunks):
            o[:, s:e] = swiglu_apply(q[:, s:e], w0, w1, w2)    # apply-first
            if clean_mask[c]:
                w0_grad, w1_grad, w2_grad = swiglu_fast_weight_gradient(
                    k[:, s:e], v[:, s:e], w0, w1, w2, lr0[:, s:e], lr1[:, s:e], lr2[:, s:e])
                w0, w1, w2 = weight_norm_update(
                    w0, w1, w2, w0_grad, w1_grad, w2_grad, self.muon_update_steps, allreduce_fn,
                    decay=decay, init_norms=init_norms)
        return self._project_output(o, B)
