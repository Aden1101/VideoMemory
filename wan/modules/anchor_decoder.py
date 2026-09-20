"""The decoder that turns a memory state back into a picture.

Trained offline, on pairs collected from real rollouts, and never run during generation: it is
an analysis tool. Loading it is what `MEM_DECODE` does when an experiment asks for an anchor
rebuilt from the memory rather than taken from the real past.
"""
import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn

from wan.modules.video_memory import swiglu_apply


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
