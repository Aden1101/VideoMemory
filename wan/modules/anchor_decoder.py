"""The decoder that turns a memory state back into a picture.

Trained offline, on pairs collected from real rollouts, and never run during generation: it is
an analysis tool. Loading it is what `MEM_DECODE` does when an experiment asks for an anchor
rebuilt from the memory rather than taken from the real past.
"""
import math

import torch
import torch.nn.functional as F
from torch import nn

from wan.modules.video_memory import RMSNorm, swiglu_apply


class FrameMixerBlock(nn.Module):
    """Spatial mixer block: what keeps the decoded frame free of a 2x2 checkerboard.

    Without it the decoder is strictly per token: each of the 30x52 queries reads the state on
    its own and the head expands it into a 2x2 latent patch, so nothing forces two neighbouring
    patches to agree and the rendered frame carries a visible 2x2 checkerboard. This block adds
    the missing piece, a local spatial mixing on the token grid, as a residual correction:

        x <- x + pw2(GELU(pw1(depthwise3x3(RMSNorm(x)))))

    `pw2` is zero-initialised, weight and bias, so the block is exactly the identity at
    initialisation: a decoder built with mixer=True and warm-started from a checkpoint trained
    without the mixer computes the same function, and `mixer.*` are the only keys the old
    state_dict is missing."""

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
    """Turns the memory states back into one latent video frame.

    A standalone readout module: it reads the fast weights of the three VideoMemoryLayer cells
    and synthesizes one latent frame (B, 1, 16, 60, 104), which the pipeline splices into the
    cache at a revisit through the existing `frames_override` hook. Trained offline on
    (states, evicted-frame) pairs. Nothing here runs during generation.

    Inputs: `states` = one (w0, w1, w2) tuple per memory block, in `memory_layers` order
    (blocks 5, 16, 27). Shapes exactly as `VideoMemoryLayer(num_heads=1).init_state()` and
    `.step()` produce them, so the leading axis is B: w0 (B, dim, dim*inter_multi),
    w1 (B, dim*inter_multi, dim), w2 like w0. States are inputs, not parameters, and are
    detached by default so no gradient ever reaches them.

    Architecture
      - m = 30*52 = 1560 learned queries, one per 2x2 latent patch of the frame. A factorized
        learned 2D grid embedding (30 rows + 52 columns) is shared across blocks, so the three
        reads of a given patch address the same spatial slot. The query table itself is per
        block: the states of blocks 5, 16 and 27 live in depth-specific feature spaces and one
        shared table cannot speak all three.
      - per block: RMSNorm on (query + grid embedding), mirroring the cell's own read path,
        then swiglu_apply(q, w0, w1, w2) giving a read of (B, 1560, 1536), then a per-block
        RMSNorm on that read.
      - fusion: the 3 reads concatenated (4608) -> Linear 4608 to 3072 -> GELU -> Linear 3072
        to 1536 -> LayerNorm, per token. No cross-token mixing here: the state is a global
        associative memory and each patch retrieves its own content by address.
      - head: Linear 1536 to 16*2*2 = 64 per token, zero-initialised in weight and bias, so the
        decoder emits an exactly null frame at initialisation rather than noise. Unpatchify to
        (B, 1, 16, 60, 104) follows CausalWanModel.unpatchify: token dim = (ph, pw, c), tokens
        in (h, w) row-major order.

    About 26.3M parameters with the defaults, most of them in the fusion MLP (18.9M) and the
    query tables (7.2M); 33.1M with the conditioning, 37.9M with the mixer on top.

    Optional conditioning: cond_dim > 0 adds LayerNorm(cond_dim) -> Linear(cond_dim,
    cond_hidden) -> GELU -> Linear(cond_hidden, num_blocks*2*dim), producing a per-block
    (scale, shift) pair applied to the normalized queries as q <- q*(1+scale) + shift. That is
    "look in the state for what matches this description", where the description is the pooled
    T5 embedding of the prompt whose subject must come back. The last linear is zero-initialised
    in weight and bias, so scale = shift = 0 at initialisation and the conditioned decoder
    computes the unconditioned function: an unconditioned checkpoint warm-starts with
    strict=False, the only missing keys being the inert cond modules. cond_dim = 0, the default,
    creates no conditioning module at all. The input LayerNorm makes the conditioning robust to
    the scale of the pooled embedding, which varies with prompt length because T5 averages over
    zero-padded positions.

    Optional spatial mixer: mixer=True inserts `mixer_blocks` FrameMixerBlock between the
    per-token fusion and the head, the only place in the module where two neighbouring patches
    can talk to each other. Each is the identity at initialisation, so a checkpoint trained
    without the mixer warm-starts with strict=False and `mixer.*` are the only missing keys.
    mixer=False, the default, creates no module.
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
        detach_states: bool = True,  # states are data, they never receive gradient
        cond_dim: int = 0,           # pooled-prompt conditioning of the queries; 0 = off
        cond_hidden: int = 512,      # hidden width of the cond MLP (only used when cond_dim > 0)
        mixer: bool = False,         # spatial mixing on the token grid (default off)
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

        # per-block query tables + shared factorized 2D grid embedding, initialised at
        # randn / sqrt(dim); the norm on the queries renormalizes anyway
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

        # spatial mixer, the identity at init; mixer=False creates no module
        self.use_mixer = bool(mixer)
        if self.use_mixer:
            assert mixer_blocks > 0, "mixer=True with mixer_blocks=0"
            self.mixer = nn.ModuleList([
                FrameMixerBlock(dim, mixer_hidden, kernel=mixer_kernel)
                for _ in range(mixer_blocks)])

        # conditioning of the read queries; cond_dim == 0 creates no module, so the
        # unconditioned decoder keeps exactly the same state_dict
        self.cond_dim = cond_dim
        self.cond_hidden = cond_hidden
        if cond_dim > 0:
            self.cond_norm = nn.LayerNorm(cond_dim, eps=1e-6)
            self.cond_in = nn.Linear(cond_dim, cond_hidden)
            self.cond_out = nn.Linear(cond_hidden, num_blocks * 2 * dim)
            nn.init.zeros_(self.cond_out.weight)  # FiLM identity at init: scale = shift = 0
            nn.init.zeros_(self.cond_out.bias)    # conditioned model == unconditioned one at init

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
        """states: one (w0, w1, w2) tuple per memory block, in memory_layers order.
        cond: the pooled prompt embedding, (B, cond_dim) or (cond_dim,), "retrieve what matches
        this description"; it needs a decoder built with cond_dim > 0. With cond=None such a
        decoder reads unconditioned, which at initialisation is also what the conditioned read
        gives. Returns the latent frame (B, 1, z_dim, grid_h * patch_h, grid_w * patch_w)."""
        assert len(states) == self.num_blocks, \
            f"expected {self.num_blocks} block states, got {len(states)}"
        dtype = self.queries.dtype
        pos = (self.pos_row[:, None, :] + self.pos_col[None, :, :]).reshape(self.m, self.dim)
        film = self._film(cond, dtype)
        B, reads = None, []
        for b, state in enumerate(states):
            w0, w1, w2 = self._checked_state(b, state, B)
            B = w0.shape[0]
            q = self.q_norms[b](self.queries[b] + pos).unsqueeze(0).expand(B, -1, -1)   # (B, m, dim)
            if film is not None:
                fb = film.expand(B, -1, -1, -1) if film.shape[0] == 1 else film
                assert fb.shape[0] == B, \
                    f"cond batch {film.shape[0]} != state batch {B} (and not broadcastable)"
                # after the norm, so the modulation is not normalised away; scale = shift = 0
                # at initialisation, which leaves q exactly as it is
                q = q * (1.0 + fb[:, b, 0].unsqueeze(1)) + fb[:, b, 1].unsqueeze(1)
            reads.append(self.read_norms[b](swiglu_apply(q, w0.to(dtype), w1.to(dtype), w2.to(dtype))))
        h = self.fuse2(F.gelu(self.fuse1(torch.cat(reads, dim=-1))))        # (B, m, dim), per token
        if self.use_mixer:                                                  # the only cross-token step
            for block in self.mixer:
                h = block(h, self.grid_h, self.grid_w)
        return self._unpatchify(self.head(self.out_norm(h)))                # exactly 0 at initialisation

    def _film(self, cond, dtype):
        """cond -> (Bc, num_blocks, 2, dim), a (scale, shift) pair per block, or None."""
        if cond is None:
            return None
        assert self.cond_dim > 0, "cond passed to an UNconditioned decoder (built with cond_dim=0)"
        if cond.dim() == 1:
            cond = cond[None]
        assert cond.dim() == 2 and cond.shape[-1] == self.cond_dim, \
            f"cond {tuple(cond.shape)} != (B, {self.cond_dim})"
        if self.detach_states:          # the conditioning is data, like the states
            cond = cond.detach()
        f = self.cond_out(F.gelu(self.cond_in(self.cond_norm(cond.to(dtype)))))
        return f.view(f.shape[0], self.num_blocks, 2, self.dim)

    def _checked_state(self, b, state, B):
        """One block's (w0, w1, w2), batched, detached if asked, with its shapes checked against
        the cell's and its batch against the blocks already read (B, None for the first)."""
        assert state is not None and len(state) == 3, f"block {b}: state must be a (w0, w1, w2) tuple"
        weights = [w[None] if w.dim() == 2 else w for w in state]           # unbatched -> (1, ., .)
        if self.detach_states:
            weights = [w.detach() for w in weights]
        expected = ((self.dim, self.dh), (self.dh, self.dim), (self.dim, self.dh))
        for name, w, shape in zip(("w0", "w1", "w2"), weights, expected):
            assert w.shape[-2:] == shape, f"block {b}: {name} {tuple(w.shape)} != (*, {shape[0]}, {shape[1]})"
        batch = weights[0].shape[0] if B is None else B
        assert all(w.shape[0] == batch for w in weights), \
            f"block {b}: inconsistent state batch (expected leading axis B*nh with nh=1)"
        return weights
