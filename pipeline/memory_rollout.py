"""What training the memory adds to the streaming rollout: the write pass run under grad
(train_write), and cache dropout, which now and then starves the attention sink for a whole
rollout so the memory is the only carrier of the early video.

Mixed into StreamingTrainingPipeline; the rollout itself is upstream's.
"""
import torch
import torch.distributed as dist


class MemoryRolloutMixin:
    def _init_memory_rollout(self, kwargs):
        """Read the two training options of the memory from the pipeline's keyword arguments:
        train_write, and the cache dropout with its anneal."""
        # train_write=True: write under enable_grad + keep state graph so the write-path trains.
        self.train_write = kwargs.get("train_write", False)
        if self.train_write:
            from wan.modules.video_memory import VideoMemoryLayer
            model = getattr(self.generator, "model", None)
            if model is not None:
                for _, m in model.named_modules():
                    if isinstance(m, VideoMemoryLayer):
                        m.detach_state = False
            print("[train_write] enabled")

        # cache dropout: with prob cache_dropout_prob, a whole rollout runs
        # with a starved runtime sink (cache_dropout_sink frames, default 0) and the memory
        # ON, so the memory is the only carrier of the early-video anchor. At 0.0, the
        # default, the sink is never touched.
        self.cache_dropout_prob = float(kwargs.get("cache_dropout_prob", 0.0) or 0.0)
        self.cache_dropout_sink = int(kwargs.get("cache_dropout_sink", 0) or 0)
        self._cache_dropout_active = False
        # linear anneal of the dropout prob from cache_dropout_prob (start) to
        # cache_dropout_prob_end over cache_dropout_anneal_steps. When end is None or the anneal
        # window is <= 0, the prob stays at the start value, which is the un-annealed behaviour.
        cd_end = kwargs.get("cache_dropout_prob_end", None)
        self.cache_dropout_prob_end = (float(cd_end) if cd_end is not None
                                       else self.cache_dropout_prob)
        self.cache_dropout_anneal_steps = int(kwargs.get("cache_dropout_anneal_steps", 0) or 0)
        self._train_step = 0  # pushed by the trainer each step; drives the anneal
        if self.cache_dropout_prob > 0.0:
            msg = (f"[cache-dropout policy] prob={self.cache_dropout_prob} "
                   f"dropped_sink={self.cache_dropout_sink}")
            if self.cache_dropout_anneal_steps > 0 and self.cache_dropout_prob_end != self.cache_dropout_prob:
                msg += (f" ANNEAL-> {self.cache_dropout_prob_end} over "
                         f"{self.cache_dropout_anneal_steps} steps")
            print(msg)

    def set_train_step(self, step):
        """The trainer pushes its global step so _maybe_cache_dropout can anneal the prob."""
        self._train_step = int(step)

    def _effective_cache_dropout_prob(self):
        """Linearly interpolated dropout prob at the current train step. Constant (== start)
        when the anneal window is <= 0 or end == start."""
        p0, p1, Ta = self.cache_dropout_prob, self.cache_dropout_prob_end, self.cache_dropout_anneal_steps
        if Ta <= 0 or p1 == p0:
            return p0
        frac = min(1.0, max(0.0, self._train_step / Ta))
        return p0 + (p1 - p0) * frac

    def _set_runtime_sink(self, sink_size=None):
        """Set the runtime attention sink on every self-attn module (None = config value)."""
        model = self.generator.model
        setter = getattr(model, "set_runtime_sink_size", None)
        if callable(setter):
            setter(sink_size)
            return
        for _, m in model.named_modules():  # fallback: attribute walk (FSDP-safe)
            if hasattr(m, "_sink_runtime") and hasattr(m, "sink_size"):
                m._sink_runtime = int(m.sink_size if sink_size is None else sink_size)

    def _maybe_cache_dropout(self):
        """Roll the per-rollout cache-dropout coin (rank-synced) and set the runtime sink.
        The override stays in place for the whole rollout (all chunks and their backwards:
        gradient-checkpoint recompute must see the same sink the forward saw); the next
        rollout re-decides, restoring the full sink on a no-drop draw."""
        if self.cache_dropout_prob <= 0.0:
            return
        p = self._effective_cache_dropout_prob()   # annealed (== cache_dropout_prob when no anneal)
        if dist.is_initialized():
            flag = torch.zeros(1, device="cuda")
            if dist.get_rank() == 0:
                flag.fill_(float(torch.rand(()).item() < p))
            dist.broadcast(flag, src=0)
            drop = bool(flag.item())
        else:
            drop = bool(torch.rand(()).item() < p)
        self._cache_dropout_active = drop
        self._set_runtime_sink(self.cache_dropout_sink if drop else None)
        if drop and (not dist.is_initialized() or dist.get_rank() == 0):
            print(f"[cache-dropout] rollout with runtime sink={self.cache_dropout_sink} "
                  f"(memory must carry the anchor)", flush=True)

    def _write_context(self, requires_grad: bool = True):
        # Only build the write graph when this chunk will actually be backpropped.
        # On critic steps generate_chunk runs with requires_grad=False (chunk is detached,
        # critic backward goes through fake_score only) -> building the clean-rerun graph
        # under enable_grad there is pure waste and the dominant VRAM leak. Gate on requires_grad.
        return torch.enable_grad() if (self.train_write and requires_grad) else torch.no_grad()
