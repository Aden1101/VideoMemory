"""Adding the anchor to the attention context instead of evicting a true frame for it."""
import os

import torch

from pipeline.switch.memory_state import _find_causal_model



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
        cm = _find_causal_model(self.generator.model)
        assert cm is not None, "MEM_ADDITIVE_KV: no CausalWanModel found under the wrappers"
        return cm.blocks

    def _clear_add_kv(self):
        """Called once per rollout, unconditionally: an additive channel installed by a previous
        arm of the same process must never leak into the next one."""
        cm = _find_causal_model(self.generator.model)
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
        device = next(self.generator.parameters()).device
        anchor_frames = anchor_frames.to(device)
        B, nf = anchor_frames.shape[0], anchor_frames.shape[1]
        blocks = self._add_blocks()
        for blk in blocks:
            blk.self_attn._add_kv = None            # the phantom pass itself must be clean
            blk.self_attn._add_capture = True
            blk.self_attn._add_captured = None
        _real = self.kv_cache1
        self._initialize_kv_cache(batch_size=B, dtype=anchor_frames.dtype, device=device,
                                  kv_cache_size_override=nf * self.frame_seq_length)
        _tmp = self.kv_cache1
        self.kv_cache1 = _real
        block_mask = self.generator.model._prepare_blockwise_causal_attn_mask(
            device=device, num_frames=nf, frame_seqlen=self.frame_seq_length,
            num_frame_per_block=self.num_frame_per_block, local_attn_size=self.local_attn_size)
        context_timestep = torch.ones([B, nf], device=device,
                                      dtype=torch.int64) * self.args.context_noise
        self.generator.model.block_mask = block_mask
        with torch.no_grad():
            self.generator(
                noisy_image_or_video=anchor_frames,
                conditional_dict=conditional_dict,
                timestep=context_timestep,
                kv_cache=_tmp,
                crossattn_cache=self.crossattn_cache,
                current_start=0,
                rope_enhance=self.rope_enhance,
                ttt_cache=getattr(self, "ttt_cache", None),
                ttt_update=False,
            )
        del _tmp
        gain = float(os.environ.get("MEM_ADD_GAIN", "1") or 1)
        _p = (os.environ.get("MEM_ADD_POS", "") or "").strip().lower()
        pos = -1 if _p in ("", "auto") else int(_p)
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
        for blk in self.crossattn_cache:
            blk["k"].zero_()
            blk["v"].zero_()
            blk["is_init"] = False
        self._add_installed = True
        print(f"[MEM_ADDITIVE_KV] anchor frame installed as {n_tok} EXTRA k/v tokens on "
              f"{len(blocks)} blocks (gain={gain}, rope band "
              f"{'auto (one frame before the window)' if pos < 0 else pos}); the local window "
              f"keeps its 12 TRUE frames, nothing evicted", flush=True)

    def _add_mass_report(self):
        """Mean attention mass the anchor tokens took, per block, the anti-inertia tripwire of
        the design plan (a channel that reads 0 % is dead by construction)."""
        cm = _find_causal_model(self.generator.model)
        if cm is None or not getattr(self, "_add_installed", False):
            return None
        return {bi: float(getattr(blk.self_attn, "_add_mass", float("nan")))
                for bi, blk in enumerate(cm.blocks)}

    def _add_mass_src_report(self):
        """Attention mass by source [sink | window | anchor] per
        block, at the last forward. `window` = everything in the context that is not the sink
        (the prompt cache, when active, is counted there). None unless MEM_ADD_MASS_SRC=1."""
        cm = _find_causal_model(self.generator.model)
        if cm is None or not getattr(self, "_add_installed", False):
            return None
        out = {}
        for bi, blk in enumerate(cm.blocks):
            m = getattr(blk.self_attn, "_add_mass_src", None)
            if m is not None:
                out[bi] = {"sink": m[0], "window": m[1], "anchor": m[2],
                           "n_tok": getattr(blk.self_attn, "_add_ntok_src", None)}
        return out or None
