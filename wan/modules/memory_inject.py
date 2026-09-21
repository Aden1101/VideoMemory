"""Where the memory meets the transformer block: read the cell, bound the injection, add it to
the residual stream, and store the state the write produced.

Mixed into CausalWanAttentionBlock. The block itself, its attention and its caches are
upstream's and stay in wan/modules/causal_model.py.
"""
import math

import torch

from utils.env import env_int, env_str

_ABLATION_ANNOUNCED = False

# The options of the memory cell, under the names the configs give them, mapped to the
# VideoMemoryLayer argument each one sets. This is the only list of them outside the cell:
# the wrapper, the model and the block pass `memory_kwargs` through untouched.
MEMORY_OPTIONS = {
    "gate_init": "g_init", "ckpt_write": "checkpoint_write", "gate_extra": "gate_extra",
    "use_decay": "use_decay", "decay_init": "decay_init", "bounded_gate": "bounded_gate",
    "single_norm": "single_norm", "bounded_mod": "bounded_mod", "rel_norm": "rel_norm",
    "dc_remove": "dc_remove", "muon_update_steps": "muon_update_steps",
    "mem_state_type": "state_type", "mem_linear_beta": "linear_beta",
}


def memory_cell_args(memory_kwargs):
    """Config-named memory options -> VideoMemoryLayer arguments. An unknown name is an error:
    a misspelt option would otherwise leave the cell on its default with nothing said."""
    memory_kwargs = dict(memory_kwargs or {})
    unknown = sorted(set(memory_kwargs) - set(MEMORY_OPTIONS))
    if unknown:
        raise TypeError(f"unknown memory option(s) {unknown}, expected among {sorted(MEMORY_OPTIONS)}")
    # a block's gate opens at 0.1 unless told otherwise; the bare cell defaults to 0.0
    args = {"g_init": 0.1}
    args.update({MEMORY_OPTIONS[k]: v for k, v in memory_kwargs.items()})
    return args



def _reject_parallel(mem_out, x):
    """MEM_REJECT_STREAM, a control arm (results/run1_write_current_chunk/eval_R1): remove, per
    token, the component of the injection parallel to the residual stream x, "more of what is
    already there" and the part suspected of damping motion. No renormalisation afterwards, so
    a read that is entirely self-similar injects nearly nothing."""
    direction = x.detach().float()
    direction = direction / direction.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    m = mem_out.float()
    return (m - (m * direction).sum(-1, keepdim=True) * direction).to(mem_out.dtype)

class MemoryInjectMixin:
    def _memory_inject(self, x, h, *, ttt_cache, ttt_state, ttt_update, mem_x_write,
                       mem_ema_key, mem_ema_prev, context, denoising_step,
                       switch_global_frame, current_start, num_frames, frame_seqlen):
        """Read the memory with h = norm1(x_in), guard what comes out, add it to the residual
        stream x, and on a write pass store the new state. Returns x unchanged when this block
        has no memory branch.

        ttt_state, mem_ema_prev and mem_x_write are read by the model outside the gradient-
        checkpointed region and passed in, so the backward recompute sees the same tensors."""
        if self.memory is None or ttt_cache is None:
            return x
        do_update, x_write = ttt_update, None
        if self.eviction_feed and ttt_update:
            # queue this chunk; the write takes the one that just left the attention window,
            # and nothing evicted yet means no write. The backward recompute appends again,
            # which the trainer's snapshot and restore undoes.
            chunk = int(current_start) // (frame_seqlen * num_frames)
            ttt_cache.fifo.append((chunk, h.detach()))
            do_update, x_write = mem_x_write is not None, mem_x_write
        step_of_denoising, frames_since_switch = self._gate_signals(
            x, denoising_step, switch_global_frame, current_start, num_frames, frame_seqlen)
        mem_out, new_state = self.memory.step(
            h, state=ttt_cache.state if ttt_state is None else ttt_state,
            do_update=do_update, x_write=x_write, prompt_emb=context,
            denoising_step=step_of_denoising, frames_since_switch=frames_since_switch,
            num_frames=num_frames)

        ablated = env_int("MEM_ZERO_ALL") != 0
        if ablated:
            mem_out = self._zeroed(mem_out)
        mem_out = self._scale_to_stream(mem_out, x)
        if not ttt_update and not ablated:
            mem_out = self._smooth(mem_out, ttt_cache, mem_ema_key, mem_ema_prev)
        if env_int("MEM_REJECT_STREAM") != 0:
            mem_out = _reject_parallel(mem_out, x)

        if not ttt_update:
            return x + mem_out
        # On the write pass the output only feeds the detached clean cache, so detach the
        # injection: otherwise every frozen block downstream builds a ~15 GB backward graph for
        # nothing. The write gradient reaches the memory through new_state, not through x.
        ttt_cache.state = new_state     # an attribute of an object, so it survives the call stack
        return x + mem_out.detach()

    def _gate_signals(self, x, denoising_step, switch_global_frame, current_start, num_frames,
                      frame_seqlen):
        """Per-token (B, L) inputs of the optional gate_extra: the denoising step of this call and
        the frame index counted from the last prompt switch. None for each one not asked for."""
        wanted = self.memory.gate_extra
        step_of_denoising = frames_since_switch = None
        B = x.shape[0]
        if "denoising_step" in wanted and denoising_step is not None:
            ds = denoising_step.float()
            if ds.dim() == 1:                                    # (B,) -> (B, F)
                ds = ds[:, None].expand(B, num_frames)
            elif ds.shape[1] != num_frames:
                ds = ds[:, :1].expand(B, num_frames)
            step_of_denoising = ds.repeat_interleave(frame_seqlen, dim=1)
        if "frames_since_switch" in wanted:
            frame = current_start // frame_seqlen + torch.arange(num_frames, device=x.device)
            if switch_global_frame is not None:                  # the count restarts at the switch
                frame = torch.where(frame >= switch_global_frame, frame - switch_global_frame, frame)
            frames_since_switch = frame.float()[None, :].expand(B, num_frames) \
                .repeat_interleave(frame_seqlen, dim=1)
        return step_of_denoising, frames_since_switch

    @staticmethod
    def _zeroed(mem_out):
        """MEM_ZERO_ALL: the memory-off arm. Announced once per process so a log shows which
        arm it was."""
        global _ABLATION_ANNOUNCED
        if not _ABLATION_ANNOUNCED:
            print("[MEM_ZERO_ALL] memory injection zeroed, mem-OFF ablation arm", flush=True)
            _ABLATION_ANNOUNCED = True
        return mem_out * 0.0

    def _scale_to_stream(self, mem_out, x):
        """Scale the injection by a fraction of the local RMS of the residual stream, so it
        cannot pile up into an artifact over a long rollout. Measured on x and not on the
        normalised h, and detached, so the memory cannot inflate the stream to earn a bigger
        injection. MEM_STREAM_SCALE overrides the config value at inference."""
        override = env_str("MEM_STREAM_SCALE")
        scale = float(override) if override else float(self.mem_stream_scale or 0.0)
        if scale <= 0.0:
            return mem_out
        stream_rms = x.detach().float().pow(2).mean(dim=-1, keepdim=True).add(1e-8).sqrt()
        self.memory.dbg_inj_stream = (
            (mem_out.detach().float().pow(2).mean(-1, keepdim=True).sqrt() * scale).mean()
            / stream_rms.mean().clamp_min(1e-8))
        return mem_out * (scale * stream_rms).to(mem_out.dtype)

    def _smooth(self, mem_out, ttt_cache, key, previous):
        """Causal exponential average of the injection over chunks, to damp flicker; off at
        ema_alpha = 0. One buffer per denoising step (`key`), so consecutive updates of a buffer
        are consecutive chunks. The first chunk at a key passes through and seeds the buffer."""
        alpha = float(self.ema_alpha or 0.0)
        if alpha <= 0.0 or key is None:
            return mem_out
        if previous is not None and previous.shape == mem_out.shape:
            mem_out = alpha * mem_out + (1.0 - alpha) * previous.to(device=mem_out.device,
                                                                    dtype=mem_out.dtype)
        # kept on the CPU (~165 MB per GPU otherwise); a bf16 round trip is exact
        ttt_cache.ema[key] = mem_out.detach().to("cpu", non_blocking=True)
        return mem_out


class EvictionFeedMixin:
    def _pop_evicted_chunks(self, ttt_cache, ttt_update, grid_sizes, current_start):
        """{block index: hidden states of the chunk that just left the attention window}: what
        each memory writes on this pass. Empty unless this is a write pass with the eviction
        feed on, and for a block until its queue holds a full window.

        The chunk is popped here, outside the gradient-checkpointed block, and handed to the
        block as an argument, so the backward recompute sees the same tensor. Eviction is
        modelled as a delay and not token by token: a chunk counts as evicted window_chunks
        write passes after it was queued. The chunks that hold the sink stay in the cache for
        good, so they are dropped from the queue without being written."""
        evicted = {}
        if not (self.eviction_feed and ttt_update and ttt_cache is not None):
            return evicted
        frames = int(grid_sizes[0][0].item())
        if frames <= 0:
            return evicted
        window = self._local_window_frames()
        assert window > 0, "eviction_feed requires local attention (local_attn_size > 0)"
        window_chunks = max(1, math.ceil(window / frames))
        sink_chunks = math.ceil(int(getattr(self.blocks[0].self_attn, "_sink_runtime", 0)) / frames)
        tokens_per_frame = int(math.prod(grid_sizes[0][1:]).item())
        chunk = int(current_start) // (tokens_per_frame * frames)
        debug = env_int("MEM_EVICT_DEBUG")
        for index, block in enumerate(self.blocks):
            if getattr(block, "memory", None) is None or ttt_cache[index] is None:
                continue
            queue, popped = ttt_cache[index].fifo, None
            if len(queue) >= window_chunks:        # popped before this pass appends its chunk
                popped, hidden = queue.pop(0)
                if popped >= sink_chunks:
                    evicted[index] = hidden
            if debug:
                if not hasattr(self, "_evict_dbg"):
                    self._evict_dbg = []
                self._evict_dbg.append({"block": index, "chunk": chunk, "popped": popped,
                                        "wrote": index in evicted,
                                        "window_chunks": window_chunks, "sink_chunks": sink_chunks})
                print(f"[evict-feed] chunk={chunk} block={index} popped={popped} "
                      f"wrote={index in evicted} window_chunks={window_chunks} "
                      f"sink_chunks={sink_chunks}", flush=True)
        return evicted

    def _local_window_frames(self):
        """local_attn_size as one number of frames: it may be an int or one value per block,
        -1 meaning global attention. Returns -1 when no block is local."""
        size = self.local_attn_size
        if isinstance(size, int):
            return size
        sizes = [int(v) for v in size if int(v) != -1] if hasattr(size, "__iter__") else [int(size)]
        return max(sizes) if sizes else -1
