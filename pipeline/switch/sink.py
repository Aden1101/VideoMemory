# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES
#
# Licensed under the Apache License, Version 2.0 (the "License").
# You may not use this file except in compliance with the License.
# To view a copy of this license, visit http://www.apache.org/licenses/LICENSE-2.0
#
# No warranties are given. The work is provided "AS IS", without warranty of any kind, express or implied.
#
# SPDX-License-Identifier: Apache-2.0
"""Rewriting the pinned sink frames instead of a slot of the rolling window."""
import os

import torch

from pipeline.switch.memory_state import _find_causal_model



class SinkMixin:

    # ---- refreshing the pinned sink, the cheapest possible explanation ------------------------
    # The sink holds the opening frames forever and is never rewritten, so when the subject does
    # not open the video, the opener keeps a permanent grip on every later segment. The obvious
    # objection to our gain is then: you are only lifting that grip, and refreshing the sink at
    # the switch would give the same thing with no memory at all. This arm is that objection,
    # implemented: no recall, no memory, no decoder, just the sink slots overwritten with the
    # last true frame of the current scene. The rolling window is left alone.
    #   MEM_SINKPURGE=0/unset  off (default)
    #   MEM_SINKPURGE=1        purge at revisit switches only (paired with the splice arms: same
    #                          instant, same cell, so the comparison is apples to apples)
    #   MEM_SINKPURGE=2        purge at every switch (the "always refresh" variant)
    def _sink_frames(self):
        """Runtime sink size in latent FRAMES, read from the model (set_runtime_sink_size aware)."""
        cm = _find_causal_model(self.generator.model)
        assert cm is not None, "MEM_SINKPURGE: no CausalWanModel found under the wrappers"
        return int(getattr(cm.blocks[0].self_attn, "_sink_runtime",
                           getattr(cm.blocks[0].self_attn, "sink_size", 0)))

    def _maybe_sinkpurge(self, output, current_start_frame, cond_new, is_revisit):
        mode = int(os.environ.get("MEM_SINKPURGE", "0") or 0)
        if not mode or not (is_revisit or mode >= 2):
            return
        assert os.environ.get("REPLAY_PAST_POS", "") != "sink0", \
            "MEM_SINKPURGE and REPLAY_PAST_POS=sink0 both rewrite the sink: not a stack"
        assert not os.environ.get("MEM_DECODE", "").strip() \
            and not int(os.environ.get("REPLAY_PAST", "0") or 0) and not self._fb_mode(), \
            "MEM_SINKPURGE is the control with neither a splice nor a memory: do not combine " \
            "it with a recall arm (MEM_DECODE / REPLAY_PAST / FRAMEBANK)"
        assert getattr(self, "global_sink", False), \
            "MEM_SINKPURGE targets the pinned (global) sink; global_sink is off in this config"
        k = self._sink_frames()
        assert k >= 1, "MEM_SINKPURGE: sink_size is 0, there is nothing pinned to purge"
        assert current_start_frame >= k, \
            f"MEM_SINKPURGE needs {k} true frame(s) before the switch (have {current_start_frame})"
        n_tok = k * self.frame_seq_length
        before = self.kv_cache1[0]["k"][:, :n_tok].detach().float().clone()
        tail = self.kv_cache1[0]["k"][:, n_tok:].detach().float().clone()
        cur = output[:, current_start_frame - k:current_start_frame]
        self._write_sink_anchor(cur, cond_new)
        after = self.kv_cache1[0]["k"][:, :n_tok].detach().float()
        rel = float((after - before).norm() / (before.norm() + 1e-12))
        untouched = float((self.kv_cache1[0]["k"][:, n_tok:].detach().float() - tail).abs().max())
        print(f"[MEM_SINKPURGE] the {k} pinned sink frame(s) of the OPENER were overwritten with "
              f"the current true frame(s) [{current_start_frame - k}, {current_start_frame}) at "
              f"frame {current_start_frame} (mode {mode}): sink k rel-L2 change {rel:.4f}, rest of "
              f"the window max|diff| {untouched:.3e} (must be 0)", flush=True)
        assert rel > 1e-3, "MEM_SINKPURGE fired but the sink k/v did not change"
        assert untouched == 0.0, "MEM_SINKPURGE touched the local window, it must not"

    def _write_sink_anchor(self, anchor_frames, conditional_dict):
        """Overwrite a sink slot with a fresh anchor frame
        (the persistent-position variant: the anchor lives where the stale first-frame anchor
        lived, read by every future query, never evicted). Runs a 1-frame forward at
        current_start=0 with sink_recache_after_switch=True so the protected sink slots are
        legitimately rewritten (the flag exists for exactly this)."""
        device = anchor_frames.device if anchor_frames.device.type == "cuda" \
            else next(self.generator.parameters()).device
        anchor_frames = anchor_frames.to(device)
        B, nf = anchor_frames.shape[0], anchor_frames.shape[1]
        # The main cache is in rolled/recompute state: a forward at current_start=0 cannot
        # address the sink slots through the window arithmetic (empty write span). Instead:
        # prefill the anchor into a fresh temporary 1-frame cache (clean position-0 write),
        # then copy its k/v into the sink slots of the real cache (buffer layout [sink|window],
        # sink = first sink_tokens entries, stored UNroped under rope_enhance like any key).
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
            n_tok = nf * self.frame_seq_length
            for bi in range(self.num_transformer_blocks):
                self.kv_cache1[bi]["k"][:, :n_tok].copy_(_tmp[bi]["k"][:, :n_tok])
                self.kv_cache1[bi]["v"][:, :n_tok].copy_(_tmp[bi]["v"][:, :n_tok])
        del _tmp
        for blk in self.crossattn_cache:
            blk["k"].zero_()
            blk["v"].zero_()
            blk["is_init"] = False
