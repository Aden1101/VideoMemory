"""What training the memory adds to the distillation trainer: truncated backpropagation across
chunks, the snapshot that protects the fast-weight state from the backward recompute, the
memory's own gradient clip and learning-rate schedule, its checkpointing and its logs.

Mixed into Trainer. The DMD loop, the FSDP setup and the critic are upstream's and stay in
trainer/distillation.py.
"""
import math
import re
from utils.misc import merge_dict_list
import torch.distributed as dist
import torch
import os
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import StateDictType, FullStateDictConfig


_BLOCK = re.compile(r"blocks\.(\d+)\.")
# (attribute a cell records, tag it is logged under)
_CELL_DIAGNOSTICS = (
    ("dbg_out_rms", "memout"), ("dbg_read_rms", "read"), ("dbg_dc_frac", "dcfrac"),
    ("dbg_amp_disp", "ampdisp"), ("dbg_ampdisp_spatial", "ampdisp_spatial"),
    ("dbg_ampdisp_temporal", "ampdisp_temporal"), ("dbg_gate_mean", "gatemean"),
    ("dbg_inj_stream", "injstream"), ("dbg_agreement", "agree"))


def _block_of(name):
    """Index of the transformer block a parameter or module name belongs to, as text."""
    found = _BLOCK.search(name)
    return found.group(1) if found else "x"


def load_without_memory_keys(generator, state_dict):
    """Load a generator state dict that may predate the memory branch. The memory parameters
    are then absent and keep their fresh init; any other missing or unexpected key is an error."""
    result = generator.load_state_dict(state_dict, strict=False)
    non_memory_missing = [k for k in result.missing_keys if "memory" not in k]
    assert not non_memory_missing, f"Missing non-memory keys: {non_memory_missing[:5]}"
    assert not result.unexpected_keys, f"Unexpected keys: {list(result.unexpected_keys)[:5]}"
    return result


class MemoryTrainingMixin:
    def _memory_cells(self):
        """(name, module) of every memory cell of the generator. Found by class name, which
        holds whatever wrapper FSDP, peft or checkpointing puts around the blocks."""
        return [(name, module) for name, module in self.model.generator.named_modules()
                if type(module).__name__ == "VideoMemoryLayer"]

    @property
    def _rollout_ttt_cache(self):
        """The memory caches of the training rollout, or None: a trainer that is not streaming
        has no rollout pipeline, and a pipeline has no cache before its first sequence."""
        pipeline = getattr(getattr(self, "streaming_model", None), "inference_pipeline", None)
        return getattr(pipeline, "ttt_cache", None)

    def _vram_profile_tick(self):
        """VRAM profiler hook"""
        n = getattr(self.config, "profile_vram", 0)
        if not n:
            return
        warmup = getattr(self.config, "profile_vram_warmup", 3)
        if self.step == warmup:
            torch.cuda.reset_peak_memory_stats()
        elif self.step >= warmup + n:
            torch.cuda.synchronize()
            alloc = torch.cuda.max_memory_allocated() / 2**30
            reserved = torch.cuda.max_memory_reserved() / 2**30
            if not dist.is_initialized() or dist.get_rank() == 0:
                layers = getattr(self.config, "model_kwargs", {}).get("memory_layers", [])
                print(f"[VRAM_PROFILE] layers={layers} "
                      f"train_write={getattr(self.config, 'train_write', False)} "
                      f"alloc_GiB={alloc:.2f} reserved_GiB={reserved:.2f}", flush=True)
            if dist.is_initialized():
                dist.barrier()
            os._exit(0)

    def _clip_mem_grads(self):
        """Clip the gradients of the memory, after they were averaged across ranks and before
        the optimizer step."""
        if self.mem_params and self.max_grad_norm_mem is not None:
            torch.nn.utils.clip_grad_norm_(self.mem_params, self.max_grad_norm_mem)

    def _lr_sched_step(self):
        """Cosine decay of both generator groups from lr_decay_start to max_iters, each down to
        its own fraction of its base learning rate."""
        start = self._lr_decay_start
        if start is None or self.step < start:
            return
        progress = min(1.0, (self.step - start) / max(1, self.config.max_iters - start))
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))              # 1 -> 0 over the decay
        groups = self.generator_optimizer.param_groups
        for group, base, floor in zip(groups, self._base_lrs, self._lr_floors):
            group["lr"] = base * (floor + (1.0 - floor) * cosine)

    def _log_memory_stats(self):
        """Opt-in instrumentation of the memory, on the main process only.
        config.log_memory_stats: log the gate, the injection diagnostics and the drift of the
        live fast-weight state. config.assert_memory_drift: fail loudly if that state stops
        accumulating, which is how a container rebuilt by FSDP or by checkpointing swallows
        the writes; worth running after any change to how the state is threaded, or a torch
        upgrade."""
        do_log = getattr(self.config, "log_memory_stats", False)
        do_assert = getattr(self.config, "assert_memory_drift", False)
        if not (do_log or do_assert) or not self.is_main_process:
            return
        try:
            if do_log and hasattr(self, "writer"):
                self._log_memory_parameters()
                self._log_cell_diagnostics()
            ttt = self._rollout_ttt_cache
            if ttt is None:
                if do_log:
                    print(f"mem stats step={self.step} no ttt_cache")
                return
            found, max_drift = self._log_state_drift(ttt, do_log)
            if do_assert and self.step >= getattr(self.config, "dfake_gen_update_ratio", 1):
                assert found, "drift guard: no live fast-weight state (writes swallowed again?)"
                assert max_drift > 1.0, f"drift guard: drift={max_drift:.3e} ~ 0, memory stuck at init"
        except AssertionError:
            raise
        except Exception as e:  # never let logging kill a training run
            print(f"mem stats logging failed: {e}")

    def _scalar(self, tag, value):
        self.writer.add_scalar(f"memory/{tag}", value, self.step)

    def _log_memory_parameters(self):
        """What the trained parameters of each cell are doing: the gate and modulation biases,
        the weight of the gate, of the optional gate inputs, and how far the read (qkv) and
        output (proj) weights have moved from their value at the first call."""
        if not hasattr(self, "_mem_init"):
            self._mem_init = {}
        first_gate = None
        for name, p in self.model.generator.named_parameters():
            if "memory" not in name:
                continue
            b, w = _block_of(name), p.detach().float()
            if name.endswith("proj.bias"):            # laid out as [shift | scale | gate]
                third = w.numel() // 3
                gate = w[2 * third:].abs().mean().item()
                self._scalar(f"gate_b{b}", gate)
                self._scalar(f"shiftscale_b{b}", w[:2 * third].abs().mean().item())
                first_gate = gate if first_gate is None else first_gate
            elif name.endswith("proj.weight"):
                self._scalar(f"gatew_b{b}", w[2 * (w.shape[0] // 3):].norm().item())
            elif name.endswith("gx_po.weight"):       # agreement: projection of the read
                self._scalar(f"Po_b{b}", w.norm().item())
            elif name.endswith("gx_pc.weight"):       # agreement: projection of the prompt
                self._scalar(f"Pc_b{b}", w.norm().item())
            elif name.endswith("proj_extra.weight"):
                # zero-initialised, so the norm of the columns fed by one input says how much
                # the gate learned to use it. Columns: [agreement (1) | denoising step (E) |
                # frames since switch (E)]; the E-wide ones are divided by sqrt(E) to compare.
                E = 16
                if w.dim() == 2 and w.shape[1] >= 1 + 2 * E:
                    self._scalar(f"gx_agree_b{b}", w[:, 0:1].norm().item())
                    self._scalar(f"gx_denoise_b{b}", w[:, 1:1 + E].norm().item() / (E ** 0.5))
                    self._scalar(f"gx_frames_b{b}", w[:, 1 + E:1 + 2 * E].norm().item() / (E ** 0.5))
            if name.endswith(("qkv.weight", "proj.weight")):
                if name not in self._mem_init:
                    self._mem_init[name] = w.cpu().clone()
                else:
                    tag = "dqkv" if "qkv" in name else "dproj"
                    self._scalar(f"{tag}_b{b}", (w.cpu() - self._mem_init[name]).norm().item())
        if first_gate is not None:
            print(f"mem stats step={self.step} gate(first)={first_gate:.3e}")

    def _log_cell_diagnostics(self):
        """The numbers each cell recorded during its last forward (see VideoMemoryLayer)."""
        for name, mod in self._memory_cells():
            for attr, tag in _CELL_DIAGNOSTICS:
                if getattr(mod, attr, None) is not None:
                    self._scalar(f"{tag}_b{_block_of(name)}", float(getattr(mod, attr)))

    def _log_state_drift(self, ttt, do_log):
        """How far each live fast-weight state is from its init. The norm of w is pinned by
        weight_norm_update, so only ||w - w_init|| tells a frozen state from a living one.
        Also the state-norm and mean column-norm ratios to the init: under use_decay with
        muon_update_steps = 0 the state collapses and the ratio falls well under 1, which the
        drift alone cannot show. Returns (a live state was found, the largest drift)."""
        inits = {}
        for name, p in self.model.generator.named_parameters():
            key = name.rsplit(".", 1)[-1]
            if "memory" in name and key in ("w0", "w1", "w2"):
                block = _BLOCK.search(name)
                inits.setdefault(int(block.group(1)) if block else -1, {})[key] = p.detach()
        found, max_drift = False, 0.0
        for i, cache in enumerate(ttt):
            state = getattr(cache, "state", None)
            if state is None:
                continue
            found = True
            pairs = [(w.detach().float(), inits.get(i, {}).get(key))
                     for w, key in zip(state, ("w0", "w1", "w2"))]
            drifts = [(w - init.float()).norm().item() if init is not None else None
                      for w, init in pairs]
            block_drift = max([d for d in drifts if d is not None], default=0.0)
            max_drift = max(max_drift, block_drift)
            if not do_log:
                continue
            norm_ratio, column_ratio = [], []
            for w, init in pairs:
                if init is None:
                    norm_ratio.append(float("nan"))
                    column_ratio.append(float("nan"))
                    continue
                init = init.float()
                norm_ratio.append(w.norm().item() / (init.norm().item() + 1e-9))
                column_ratio.append((w.norm(dim=-2) / (init.norm(dim=-2) + 1e-9)).mean().item())
            shown = ", ".join("n/a" if d is None else f"{d:.3e}" for d in drifts)
            print(f"mem stats step={self.step} block={i} drift=({shown})")
            print(f"mem stats step={self.step} block={i} "
                  f"statenorm/init=({', '.join(f'{r:.3f}' for r in norm_ratio)}) "
                  f"colnorm/init=({', '.join(f'{r:.3f}' for r in column_ratio)})")
            if hasattr(self, "writer"):
                self._scalar(f"drift_b{i}", block_drift)
                self._scalar(f"statenorm_ratio_b{i}", norm_ratio[0])
                self._scalar(f"colnorm_ratio_b{i}", column_ratio[0])
        if do_log and not found:
            print(f"mem stats step={self.step} ttt allocated but no state")
        if do_log and found and hasattr(self, "writer"):
            self._scalar("drift_max", max_drift)
        return found, max_drift

    def _unfreeze_memory_and_select_lora(self, config):
        """peft froze everything that is not a LoRA adapter, the memory included: make the memory
        trainable again, and freeze the adapters of the blocks below lora_unfreeze_from_block."""
        # memory params are new, full-rank and trainable (excluded from LoRA targets).
        # peft freezes every non-LoRA param, so we un-freeze them here.
        for n, p in self.model.generator.named_parameters():
            if "memory" in n:
                p.requires_grad_(True)

        # Selective LoRA unfreeze: only blocks >= lora_unfreeze_from_block (which consume the memory
        # injection) stay trainable; earlier blocks' LoRA is frozen, dropping their grad + AdamW state
        # (frees their VRAM). None = all LoRA trainable. Critic LoRA untouched.
        unfreeze_from = getattr(config, "lora_unfreeze_from_block", None)
        if unfreeze_from is not None:
            import re as _re
            frozen_blocks, kept_blocks, kept = set(), set(), 0
            for n, p in self.model.generator.named_parameters():
                if "memory" in n or not p.requires_grad:
                    continue
                m = _re.search(r"blocks\.(\d+)\.", n)
                if m is None or int(m.group(1)) < int(unfreeze_from):
                    p.requires_grad_(False)
                    if m is not None:
                        frozen_blocks.add(int(m.group(1)))
                else:
                    kept += 1
                    kept_blocks.add(int(m.group(1)))
            if self.is_main_process:
                print(f"[selective LoRA] unfreeze from block {unfreeze_from}: "
                      f"froze LoRA of blocks {sorted(frozen_blocks)}, "
                      f"kept {kept} trainable LoRA tensors across blocks {sorted(kept_blocks)}")

    def _memory_modules_outside_fsdp(self):
        """The memory cells, to be left out of FSDP sharding, moved to the device and made
        identical on every rank. Empty on a single process."""
        # fast weights (w0/w1/w2) are excluded from FSDP
        mem_ignored = []
        if self.world_size > 1:
            mem_ignored = [cell for _, cell in self._memory_cells()]
            for m in mem_ignored:
                m.to(device=self.device, dtype=self.dtype)
            # memory is excluded from FSDP: sync the per-rank divergent init from rank 0 so all ranks
            # share one memory (its grads are averaged before each optimizer step)
            for m in mem_ignored:
                for p in m.parameters():
                    dist.broadcast(p.data, src=0)
                for b in m.buffers():
                    dist.broadcast(b.data, src=0)
        return mem_ignored

    def _generator_param_groups(self, config):
        """Optimizer groups of the generator: the LoRA adapters at config.lr and, when there is a
        memory, its parameters at config.lr_mem. Keeps self.mem_params for the gradient clip."""
        gen = self.model.generator
        mem_params = [p for n, p in gen.named_parameters() if "memory" in n and p.requires_grad]
        other_params = [p for n, p in gen.named_parameters() if "memory" not in n and p.requires_grad]
        # safety net
        assert mem_params or not any("memory" in n for n, _ in gen.named_parameters()), \
            "memory params exist but none landed in the optimizer group (name-match broke post-FSDP)"
        generator_param_groups = [{"params": other_params, "lr": config.lr}]  # LoRA adapters
        if mem_params:  # no memory layers -> single group, strictly identical to baseline
            generator_param_groups.append({"params": mem_params, "lr": getattr(config, "lr_mem", 1e-4)})
        self.mem_params = mem_params
        return generator_param_groups

    def _init_memory_clip_and_schedule(self, config):
        """What _clip_mem_grads and _lr_sched_step read. Left out of the config, the first
        clips nothing and the second never starts."""
        # the memory is kept out of FSDP, so generator.clip_grad_norm_, an FSDP method, skips
        # it: this is the only clip its gradients get
        clip = getattr(config, "max_grad_norm_mem", None)
        # a norm of 1e8 or more is read as "no clip": a config writes 1e9 to turn the clip off
        self.max_grad_norm_mem = None if clip is None or float(clip) >= 1e8 else float(clip)
        # base learning rate of each optimizer group, [LoRA, memory]; the schedule scales them
        self._base_lrs = [g["lr"] for g in self.generator_optimizer.param_groups]
        start = getattr(config, "lr_decay_start", None)
        self._lr_decay_start = None if start is None else int(start)
        self._lr_floors = [float(getattr(config, "lr_floor_frac", 1.0)),
                           float(getattr(config, "lr_mem_floor_frac", 1.0))]

    def _sample_bptt_horizon(self):
        """How many chunks this generator step rolls out: `bptt_window`, or a draw between
        `bptt_window_min` and it. The state is detached only where the window starts and one
        backward runs over the window, so the write is trained across chunk boundaries. A random
        horizon lets a short gradient tail supervise states of many ages at the memory cost of
        that tail alone. Rank 0 draws and broadcasts: FSDP needs every rank to run the same
        number of forwards."""
        horizon = int(getattr(self.config, "bptt_window", 1))
        shortest = int(getattr(self.config, "bptt_window_min", horizon))
        if horizon <= shortest:
            return horizon
        def draw():
            return int(torch.randint(shortest, horizon + 1, (1,)).item())

        if not dist.is_initialized():
            return draw()
        # only rank 0 draws, so the other ranks' random streams are left alone
        shared = torch.tensor(draw() if dist.get_rank() == 0 else 0, device=self.device, dtype=torch.int32)
        dist.broadcast(shared, src=0)
        return int(shared.item())

    def _bptt_generator_step(self, horizon):
        """One generator step over a window of `horizon` chunks. Returns the log dict."""
        # The first horizon - tail chunks run under no_grad: they only age the state past the
        # KV eviction, cheaply. The last `tail` chunks carry the gradient. A tail of one chunk
        # means no backward ever recomputes a previous chunk, whose cache entries may be gone by
        # then. `grad_tail_W` left out of the config, the tail is the whole window.
        tail = int(getattr(self.config, "grad_tail_W", horizon))
        tail = max(1, min(tail, horizon))
        losses = []
        logs = []
        used = 0
        # force a fresh sequence so the window always starts at frame 0 (chunk 0 sets the
        # identity anchor, the grad tail lands in the same sequence).
        # seed step: every train_seed_every steps also put chunk 0 in-graph so the fast-weight
        # seed w0/w1/w2 (only reachable via init_state at chunk 0) receives gradient.
        seed_every = int(getattr(self.config, "train_seed_every", 0))
        gen_idx = self.step // max(1, int(getattr(self.config, "dfake_gen_update_ratio", 1)))
        seed_step = seed_every > 0 and (gen_idx % seed_every == 0)   # every N-th generator step
        twin_on = self.streaming_model.twin.on   # one definition, the model's
        if getattr(self.config, "id_loss_weight", 0.0) > 0.0 or twin_on or seed_step:
            self.streaming_active = False
            self.start_new_sequence()
        # seed steps also grad chunk 0 -> more grad chunks; normalize each per-chunk backward so
        # the total gradient equals the normal path's mean(losses)/gas
        n_grad = tail + (1 if (seed_step and horizon - tail > 0) else 0)
        for i in range(horizon):
            if i > 0 and not self.streaming_model.can_generate_more():
                break
            used += 1
            with_grad = (i >= horizon - tail) or (seed_step and i == 0)   # +chunk 0 on seed steps
            # build the mem-OFF counterfactual twin only for the last (grad-tail) chunk
            make_twin = (with_grad and i == horizon - 1 and twin_on)
            chunk, chunk_info = self.streaming_model.generate_next_chunk(
                requires_grad=with_grad, detach_state=(i == horizon - tail),
                keep_last_write_grad=(with_grad and i < horizon - 1 and i + 1 > horizon - tail),
                make_twin=make_twin)
            if with_grad:
                loss, chunk_log = self.streaming_model.compute_generator_loss(
                    chunk=chunk, chunk_info=chunk_info)
                logs.append(chunk_log)
                if seed_step:   # per-chunk backward; sever the KV-cache grad link between chunks
                    snap = self._snapshot_ttt_state()
                    torch.cuda.empty_cache()   # reclaim the ~800MB reserved-but-unallocated cache
                                               # at the post-forward low-water mark, before chunk-0's
                                               # backward peak (seed step only ~1/20). Fits the 4-GPU budget.
                    try:
                        (loss / (self.gradient_accumulation_steps * n_grad)).backward()
                    finally:
                        self._restore_ttt_state(snap)
                    self.streaming_model._clear_cache_gradients()
                    del snap
                    torch.cuda.empty_cache()
                    losses.append(loss.detach())
                else:
                    losses.append(loss)
        if losses:
            generator_loss = sum(losses) / len(losses)
            if not seed_step:                             # normal step: one backward over the window
                ttt_snap = self._snapshot_ttt_state()
                try:
                    (generator_loss / self.gradient_accumulation_steps).backward()
                finally:
                    self._restore_ttt_state(ttt_snap)
            self._log_memory_grads()
        else:
            generator_loss = torch.tensor(0.0, device=self.device)
        generator_log_dict = merge_dict_list(logs) if logs else {}
        generator_log_dict.update({
            "generator_loss": generator_loss,
            "generator_grad_norm": torch.tensor(0.0, device=self.device),
            "bptt_chunks_used": torch.tensor(float(used), device=self.device),
            "bptt_grad_chunks": torch.tensor(float(len(losses)), device=self.device),
        })
        return generator_log_dict

    def _log_memory_grads(self):
        """On the first few steps, print the gradient norms of the memory's write (lrs, decay),
        read (qkv, proj) and seed (w0) parameters, to confirm each of them is reached."""
        if self.step not in (0, 5, 10, 15, 20, 30) or (dist.is_initialized() and dist.get_rank() != 0):
            return
        try:
            def gn(p):      # gradient norm, -1 when the parameter is absent or got no gradient
                grad = getattr(p, "grad", None)
                return float(grad.detach().float().norm()) if grad is not None else -1.0
            for i, (_, m) in enumerate(self._memory_cells()):
                print(f"mem-grad step={self.step} mem{i} "
                      f"WRITE lrs={gn(m.lrs.weight):.3e} decay={gn(getattr(m,'decay_logit',None)):.3e} "
                      f"READ qkv={gn(m.qkv.weight):.3e} proj={gn(m.proj.weight):.3e} "
                      f"SEED w0={gn(m.w0):.3e}")
        except Exception as _e:
            print(f"mem-grad check failed: {_e}")

    def _snapshot_ttt_state(self):
        """The memory caches as they stand after generation, to be put back after the backward.

        The state is written inside the gradient-checkpointed block, so the backward recompute
        writes it again and leaves the cache on an earlier chunk's state; the same goes for the
        injection EMA buffers and for the eviction queue, which the recompute appends to. The
        state and the buffers are detached and parked on the CPU; the queue holds tensors that
        are never edited in place, so a copy of the list is enough."""
        cache = self._rollout_ttt_cache
        if not cache:
            return None
        return [{"state": None if c.state is None else tuple(t.detach().to("cpu") for t in c.state),
                 "ema": {k: v.detach().to("cpu") for k, v in c.ema.items()},
                 "fifo": list(c.fifo)}
                for c in cache]

    def _restore_ttt_state(self, snap):
        cache = self._rollout_ttt_cache
        if snap is None or not cache:
            return
        for c, s in zip(cache, snap):
            c.state = None if s["state"] is None else tuple(t.to(self.device) for t in s["state"])
            c.ema = {k: v.to(self.device) for k, v in s["ema"].items()}
            c.fifo = list(s["fifo"])

    def _gather_memory_state_dict(self, fsdp_model):
        """The memory parameters, gathered in full on rank 0. They are full-rank and not LoRA
        adapters, so the LoRA state dict does not carry them and the checkpoint has to."""
        with FSDP.state_dict_type(
            fsdp_model,
            StateDictType.FULL_STATE_DICT,
            FullStateDictConfig(rank0_only=True, offload_to_cpu=True)
        ):
            full = fsdp_model.state_dict()
        return {k: v for k, v in full.items() if "memory" in k}
