"""Adding the anchor to the attention context instead of evicting a true frame for it."""

from pipeline.switch.memory_state import find_causal_model
from utils.env import env_float, env_str
from utils.log import log


class AdditiveKVMixin:

    # ---- adding the anchor instead of evicting a frame for it ---------------------------------
    # Every other way of bringing a frame back takes a slot of the 12-frame window, so a true
    # frame of the current scene is evicted and the model's declared past becomes partly a
    # foreign world. Here the anchor becomes 1560 extra key/value tokens per block, concatenated
    # at attention time in the free band between the sink and the window. Nothing is evicted.
    #   MEM_ADDITIVE_KV=1   turn it on (off by default)
    #   MEM_ADD_GAIN=g      attention-mass gain of the anchor tokens (1 = plain concatenation)
    #   MEM_ADD_POS=k|auto  RoPE band start of the anchor tokens; auto = one frame before the
    #                       local window (the distances the model already reads every step)
    def _add_blocks(self):
        cm = find_causal_model(self.generator.model)
        assert cm is not None, "MEM_ADDITIVE_KV: no CausalWanModel found under the wrappers"
        return cm.blocks

    def _reset_add_kv(self):
        """Per rollout: nothing installed, an empty trace of the attention mass, and no extra
        tokens left on the attention modules by a previous arm of the same process."""
        self._add_installed = False
        self._add_mass_log = []
        self._clear_add_kv()

    def _clear_add_kv(self):
        """Called once per rollout, unconditionally: an additive channel installed by a previous
        arm of the same process must never leak into the next one."""
        cm = find_causal_model(self.generator.model)
        if cm is None:
            return
        for blk in cm.blocks:
            sa = blk.self_attn
            sa._add_kv = None
            sa._add_capture = False
            sa._add_captured = None

    def _install_add_kv(self, anchor_frames, conditional_dict):
        """Phantom forward of the anchor frame through a fresh 1-frame cache: every block sees a
        clean position-0 write and we capture the raw (pre-RoPE) k/v it produces. Same
        construction as _write_sink_anchor, except nothing is written into the real cache, the
        tokens are read at attention time only. Draws no RNG (the memzero pairing is preserved).
        """
        anchor_frames = anchor_frames.to(self._generator_device)
        nf = anchor_frames.shape[1]
        blocks = self._add_blocks()
        for blk in blocks:
            blk.self_attn._add_kv = None            # the phantom pass itself must be clean
            blk.self_attn._add_capture = True
            blk.self_attn._add_captured = None
        self._prefill_scratch_cache(anchor_frames, conditional_dict)    # the blocks capture k/v as it runs
        gain = env_float("MEM_ADD_GAIN", 1)
        p = env_str("MEM_ADD_POS").lower()
        pos = -1 if p in ("", "auto") else int(p)
        n_tok = nf * self.frame_seq_length
        for bi, blk in enumerate(blocks):
            sa = blk.self_attn
            assert sa._add_captured is not None, \
                f"MEM_ADDITIVE_KV: the phantom pass captured nothing on block {bi}"
            k, v = sa._add_captured
            assert k.shape[1] == n_tok and v.shape[1] == n_tok, \
                f"MEM_ADDITIVE_KV: block {bi} captured {k.shape[1]} tokens, expected {n_tok}"
            sa._add_kv = (k, v)
            sa._add_gain = gain
            sa._add_pos = pos
            sa._add_capture = False
            sa._add_captured = None
        self._reset_crossattn_cache()
        self._add_installed = True
        log(f"[MEM_ADDITIVE_KV] anchor frame installed as {n_tok} EXTRA k/v tokens on "
            f"{len(blocks)} blocks (gain={gain}, rope band "
            f"{'auto (one frame before the window)' if pos < 0 else pos}); the local window "
            f"keeps its {self.local_attn_size} true frames, nothing evicted")

    def _add_mass_report(self):
        """Mean attention mass the anchor tokens took, per block, the anti-inertia tripwire of
        the design plan (a channel that reads 0 % is dead by construction)."""
        cm = find_causal_model(self.generator.model)
        if cm is None or not self._add_installed:
            return None
        return {bi: float(getattr(blk.self_attn, "_add_mass", float("nan")))
                for bi, blk in enumerate(cm.blocks)}

    def _add_mass_src_report(self):
        """Attention mass by source [sink | window | anchor] per
        block, at the last forward. `window` = everything in the context that is not the sink
        (the prompt cache, when active, is counted there). None unless MEM_ADD_MASS_SRC=1."""
        cm = find_causal_model(self.generator.model)
        if cm is None or not self._add_installed:
            return None
        out = {}
        for bi, blk in enumerate(cm.blocks):
            m = getattr(blk.self_attn, "_add_mass_src", None)
            if m is not None:
                out[bi] = {"sink": m[0], "window": m[1], "anchor": m[2],
                           "n_tok": getattr(blk.self_attn, "_add_ntok_src", None)}
        return out or None
