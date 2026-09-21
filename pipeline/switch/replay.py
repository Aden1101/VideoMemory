"""Rebuilding the attention window from the real frames of the subject's first appearance.

This is a measurement arm, not a feature: it hands the model the frames that a perfect
memory would have recovered, so the score it reaches is the ceiling the memory is aiming at.
Off by default.

    REPLAY_PAST=1          turn the arm on
    REPLAY_PAST_FRAMES=k   only the last k window slots come from the past, the rest stays
                           the true recent tail. 0 or unset = the whole window.
    REPLAY_PAST_POS=head   put those k slots at the oldest end of the window instead of the
                           most recent one: a distant reference rather than a recent frame.
    REPLAY_PAST_POS=sink0  leave the window alone and write the frame into the sink slot,
                           which is read forever and never evicted.
    REPLAY_PAST_SLOT=j     a single past frame, placed j slots before the most recent one.
    REPLAY_PAST_NOISE=s    degrade the replayed frames with noise of standard deviation s,
                           to imitate an imperfect recall.
    REPLAY_PAST_BLOCKS=a,b keep the replayed window only on these transformer blocks.
"""
import torch

from pipeline.switch.noise import REPLAY_NOISE_SEED, mix_with_noise, private_noise
from utils.env import env_choice, env_float, env_int, env_str
from utils.log import log


def replay_position():
    """REPLAY_PAST_POS: "" for the most recent slots, "head" for the oldest end of the window,
    "sink0" for the sink slot."""
    return env_choice("REPLAY_PAST_POS", ("head", "sink0"))


class ReplayPastMixin:
    def _build_replay_window(self, output, current_start_frame, a_end, *, composite=False):
        """The window of latent frames to recache, or None when the arm is off.

        `a_end` is the end latent frame (exclusive) of the segment where the subject first
        appeared, so the anchor frame is `a_end - 1`. `composite=True` keeps only the subject
        region of the replayed tail and takes the border from the current scene.
        """
        if not env_int("REPLAY_PAST"):
            return None
        n = self._window_frames(current_start_frame)
        nf = env_int("REPLAY_PAST_FRAMES")
        need = nf if 0 < nf < n else n
        assert a_end is not None and a_end - need >= 0, \
            f"replaying the past needs {need} frames of the first appearance before frame {a_end}"
        slot_k = env_int("REPLAY_PAST_SLOT", -1)
        if slot_k >= 0:
            window = output[:, current_start_frame - n:current_start_frame].clone()
            pos = n - 1 - min(slot_k, n - 1)
            window[:, pos:pos + 1] = self._anchor_content(output, a_end - 1, a_end)
            log(f"[REPLAY_PAST] anchor at window slot {pos}/{n} "
                f"({slot_k} frames before most-recent)")
        elif 0 < nf < n:
            if replay_position() == "head":
                window = torch.cat([
                    self._anchor_content(output, a_end - nf, a_end),
                    output[:, current_start_frame - (n - nf):current_start_frame]], dim=1)
                log(f"[REPLAY_PAST] first {nf}/{n} window slots from the first appearance "
                    f"[{a_end - nf}, {a_end}), rest = recent tail")
            else:
                tail = self._anchor_content(output, a_end - nf, a_end)
                if composite:
                    tail = self._maybe_composite(tail, output, current_start_frame)
                window = torch.cat([
                    output[:, current_start_frame - n:current_start_frame - nf], tail], dim=1)
                log(f"[REPLAY_PAST] last {nf}/{n} window slots from the first appearance "
                    f"[{a_end - nf}, {a_end}), rest = true recent tail")
        else:
            window = self._anchor_content(output, a_end - n, a_end)
            log(f"[REPLAY_PAST] recaching frames [{a_end - n}, {a_end}) of the first "
                f"appearance at positions [{current_start_frame - n}, {current_start_frame})")
        return window

    def _degrade_replay_window(self, window, current_start_frame):
        """Add noise of standard deviation `REPLAY_PAST_NOISE` to the replayed frames, keeping
        the root-mean-square level of the latents. Its own generator, so the main random stream
        is untouched and the paired arms keep identical noise."""
        dose = env_float("REPLAY_PAST_NOISE")
        if dose <= 0.0 or window is None:
            return window
        nf_env = env_int("REPLAY_PAST_FRAMES")
        na = nf_env if 0 < nf_env < window.shape[1] else window.shape[1]
        part = window[:, -na:]
        eps = private_noise(part, REPLAY_NOISE_SEED + current_start_frame).to(part.dtype)
        degraded = mix_with_noise(part, eps, dose)
        log(f"[REPLAY_PAST_NOISE] degraded the last {na} replayed frame(s), sigma={dose}")
        return torch.cat([window[:, :-na], degraded], dim=1)

    def _recache_with_replay(self, output, current_start_frame, cond, window,
                             *, timestep_override=None):
        """Refill the kv cache at a switch, with `window` in place of the true recent frames.
        `window=None` gives the plain recache."""
        blocks_env = env_str("REPLAY_PAST_BLOCKS")
        sink0 = replay_position() == "sink0"
        assert timestep_override is None or not (blocks_env or sink0), \
            "MEM_ANCHOR_T declares the noise level of the slot the recache writes, so it is " \
            "defined for the plain window, not for REPLAY_PAST_BLOCKS / REPLAY_PAST_POS=sink0"
        if window is not None and blocks_env:
            # Keep the replayed frames only on the listed blocks and a normal window elsewhere:
            # does the anchor at the memory blocks alone bring the subject back? Both passes see
            # the same frames at the same positions, so restoring k and v is exact.
            keep = {int(b) for b in blocks_env.split(",") if b != ""}
            self._recache_after_switch(output, current_start_frame, cond)
            snap_kv = {bi: {kk: self.kv_cache1[bi][kk].detach().to("cpu", copy=True)
                            for kk in ("k", "v")}
                       for bi in range(self.num_transformer_blocks) if bi not in keep}
            self._recache_after_switch(output, current_start_frame, cond, frames_override=window)
            with torch.no_grad():
                for bi, kv in snap_kv.items():
                    self.kv_cache1[bi]["k"].copy_(kv["k"].to(self.kv_cache1[bi]["k"].device))
                    self.kv_cache1[bi]["v"].copy_(kv["v"].to(self.kv_cache1[bi]["v"].device))
            del snap_kv
            log(f"[REPLAY_PAST_BLOCKS] replayed k/v kept only on blocks {sorted(keep)}, "
                f"other blocks restored to the normal recache")
        elif sink0 and window is not None:
            # normal recache, so the window stays the true recent tail, then the anchor frame
            # overwrites the sink slot, which is read forever and never evicted
            self._recache_after_switch(output, current_start_frame, cond)
            self._write_sink_anchor(window[:, -1:], cond)
            log("[REPLAY_PAST] sink0: anchor frame written into the sink slot (persistent), "
                "window = normal recent tail")
        else:
            self._recache_after_switch(output, current_start_frame, cond,
                                       frames_override=window,
                                       timestep_override=timestep_override)
