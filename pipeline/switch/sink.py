"""Rewriting the pinned sink frames instead of a slot of the rolling window."""
import torch

from pipeline.switch.memory_state import _find_causal_model
from pipeline.switch.replay import replay_position
from utils.env import env_int, env_str


class SinkMixin:

    # ---- refreshing the pinned sink, the cheapest possible explanation ------------------------
    # The sink holds the opening frames forever, so when the subject does not open the video,
    # the opener weighs on every later segment. Any recall arm also lifts that weight, and the
    # question is whether that alone explains what it gains. This arm answers it: no recall, no
    # memory, no decoder, only the sink slots overwritten with the last true frame of the current
    # scene. The rolling window is left alone.
    #   MEM_SINKPURGE=0/unset  off (default)
    #   MEM_SINKPURGE=1        purge at revisit switches only: the same instant as the splice
    #                          arms, so the two compare
    #   MEM_SINKPURGE=2        purge at every switch (the "always refresh" variant)
    def _sink_frames(self):
        """Runtime sink size in latent FRAMES, read from the model (set_runtime_sink_size aware)."""
        cm = _find_causal_model(self.generator.model)
        assert cm is not None, "MEM_SINKPURGE: no CausalWanModel found under the wrappers"
        return int(getattr(cm.blocks[0].self_attn, "_sink_runtime",
                           getattr(cm.blocks[0].self_attn, "sink_size", 0)))

    def _maybe_sinkpurge(self, output, current_start_frame, cond_new, is_revisit):
        """MEM_SINKPURGE: overwrite the sink slots with the last true frames of the current
        scene, at revisits (1) or at every switch (2), and check that the sink changed while the
        rolling window did not. Does nothing when the arm is off."""
        mode = env_int("MEM_SINKPURGE")
        if not mode or not (is_revisit or mode >= 2):
            return
        assert replay_position() != "sink0", \
            "MEM_SINKPURGE and REPLAY_PAST_POS=sink0 both rewrite the sink: not a stack"
        assert not env_str("MEM_DECODE") \
            and not env_int("REPLAY_PAST") and not self._fb_mode(), \
            "MEM_SINKPURGE is the control with neither a splice nor a memory: do not combine " \
            "it with a recall arm (MEM_DECODE / REPLAY_PAST / FRAMEBANK)"
        assert self.global_sink, \
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
        if anchor_frames.device.type != "cuda":
            anchor_frames = anchor_frames.to(self._generator_device)
        nf = anchor_frames.shape[1]
        # The main cache is in rolled/recompute state: a forward at current_start=0 cannot
        # address the sink slots through the window arithmetic (empty write span). Instead:
        # prefill the anchor into a fresh temporary 1-frame cache (clean position-0 write),
        # then copy its k/v into the sink slots of the real cache (buffer layout [sink|window],
        # sink = first sink_tokens entries, stored without RoPE under rope_enhance like any key).
        tmp = self._prefill_scratch_cache(anchor_frames, conditional_dict)
        with torch.no_grad():
            n_tok = nf * self.frame_seq_length
            for bi in range(self.num_transformer_blocks):
                self.kv_cache1[bi]["k"][:, :n_tok].copy_(tmp[bi]["k"][:, :n_tok])
                self.kv_cache1[bi]["v"][:, :n_tok].copy_(tmp[bi]["v"][:, :n_tok])
        del tmp
        self._reset_crossattn_cache()
