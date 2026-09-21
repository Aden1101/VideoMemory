"""The rollout with any number of prompt switches: generate chunk by chunk under a schedule,
and at each switch rebuild the attention context, feeding a past subject back into it when the
segment is a revisit. Which arm feeds it back is chosen by the environment; each arm lives in
its own module of this package.

Mixed into SwitchCausalInferencePipeline, whose `inference()` is the two-switch case of it.
"""
import json
import os
from typing import List, Optional
import torch
from utils.memory import gpu, get_cuda_free_memory_gb, move_model_to_device_with_memory_preservation
from utils.env import check_arm_variables, env_int, env_str
from utils.log import log
from wan.modules.video_memory import TTTBlockCache
from pipeline.switch.memory_state import sync_memory_writer
from pipeline.switch.schedule import parse_schedule


class ScheduleDriverMixin:
    def _reset_crossattn_cache(self):
        """Forget the cached text keys and values, so the next forward cross-attends afresh."""
        for blk in self.crossattn_cache:
            blk["k"].zero_()
            blk["v"].zero_()
            blk["is_init"] = False

    @property
    def _generator_device(self):
        return next(self.generator.parameters()).device

    def _prefill_scratch_cache(self, frames, conditional_dict):
        """Run `frames` on their own through the model, at the context noise level and at
        position 0, into a scratch KV cache, and return that cache: every block then holds the
        keys and values of a clean first-position write of those frames. No grad, no memory
        write, no draw from the RNG, and the rollout's own cache is not touched."""
        batch_size, num_frames = frames.shape[:2]
        scratch = self._scratch_kv_cache(batch_size, num_frames, frames.dtype, frames.device)
        timestep = torch.ones([batch_size, num_frames], device=frames.device,
                              dtype=torch.int64) * self.args.context_noise
        with torch.no_grad():
            self.generator(
                noisy_image_or_video=frames, conditional_dict=conditional_dict,
                timestep=timestep, kv_cache=scratch, crossattn_cache=self.crossattn_cache,
                current_start=0, rope_enhance=self.rope_enhance,
                ttt_cache=self.ttt_cache, ttt_update=False)
        return scratch

    def _scratch_kv_cache(self, batch_size, num_frames, dtype, device):
        """A KV cache of `num_frames` frames for a forward that must not touch the rollout: the
        phantom pass of an arm that encodes or sharpens an anchor on its own. The rollout's cache
        is put back as it was, and the model's block mask is set for the scratch size."""
        rollout_cache = self.kv_cache1
        self._initialize_kv_cache(batch_size=batch_size, dtype=dtype, device=device,
                                  kv_cache_size_override=num_frames * self.frame_seq_length)
        scratch, self.kv_cache1 = self.kv_cache1, rollout_cache
        self.generator.model.block_mask = self.generator.model._prepare_blockwise_causal_attn_mask(
            device=device, num_frames=num_frames, frame_seqlen=self.frame_seq_length,
            num_frame_per_block=self.num_frame_per_block, local_attn_size=self.local_attn_size)
        return scratch

    def _window_frames(self, current_start_frame):
        """How many past frames the attention context holds when generation reaches
        `current_start_frame`: all of them under global attention, else the local window."""
        if self.local_attn_size == -1:
            return current_start_frame
        return min(self.local_attn_size, current_start_frame)

    def _begin_rollout(self, noise, low_memory, first_switch_frame=None):
        """Everything both drivers do before the first chunk: check the arm variables, free
        the text encoder, allocate the output and the caches, and reset what each arm keeps
        from one rollout to the next. Returns the output tensor."""
        check_arm_variables()
        batch_size, num_output_frames, num_channels, height, width = noise.shape
        if low_memory:
            move_model_to_device_with_memory_preservation(
                self.text_encoder, target_device=gpu,
                preserved_memory_gb=get_cuda_free_memory_gb(gpu) + 5)
        # in low-memory mode the output lives on the CPU, to leave the GPU to the model
        output = torch.zeros([batch_size, num_output_frames, num_channels, height, width],
                             device=torch.device("cpu") if low_memory else noise.device,
                             dtype=noise.dtype)
        self._allocate_caches(noise)

        self._switch_global_frame = None
        self._first_switch_frame = first_switch_frame
        # MEM_WRITER: overlay write parameters trained offline, before the first write. It
        # draws nothing from the RNG, so paired arms keep identical noise.
        sync_memory_writer(self.generator.model)
        for reset in (self._reset_memory_slots, self._reset_early_decode, self._reset_composite,
                      self._reset_add_kv, self._fb_reset):
            reset()
        self.generator.model.local_attn_size = self.local_attn_size
        log(f"[inference] local_attn_size set on model: {self.generator.model.local_attn_size}")
        self._set_all_modules_max_attention_size(self.local_attn_size)
        return output

    def _allocate_caches(self, noise):
        """A KV cache of the local window, or of the whole video under global attention; the
        cross-attention cache; and a fresh memory state per block."""
        batch_size, num_output_frames = noise.shape[:2]
        local = getattr(self.args.model_kwargs, "local_attn_size", -1)
        frames = num_output_frames if local == -1 else local
        log(f"kv_cache_size: {frames * self.frame_seq_length} "
            f"(policy: {'global (-1)' if local == -1 else f'int->local, size={local}'}, "
            f"frame_seq_length: {self.frame_seq_length}, num_output_frames: {num_output_frames})")
        self._initialize_kv_cache(batch_size, dtype=noise.dtype, device=noise.device,
                                  kv_cache_size_override=frames * self.frame_seq_length)
        self._initialize_crossattn_cache(batch_size=batch_size, dtype=noise.dtype,
                                         device=noise.device)
        self.ttt_cache = [TTTBlockCache() for _ in range(self.num_transformer_blocks)]

    def _denoise_chunk(self, noise, output, cond, current_start_frame, num_input_frames,
                       current_num_frames):
        """Generate one chunk under `cond` and write it into `output`.

        Two kinds of pass over the same positions. The read passes are the sampler's denoising
        steps on the chunk's slice of `noise`: they read the KV cache and the memory. The write
        pass runs the result once more at the context noise level: it is the one that refreshes
        the KV cache with clean keys and lets the memory write."""
        batch_size = noise.shape[0]
        shape = [batch_size, current_num_frames]
        steps = self.denoising_step_list

        def generator(frames, timestep, write):
            return self.generator(
                noisy_image_or_video=frames, conditional_dict=cond, timestep=timestep,
                kv_cache=self.kv_cache1, crossattn_cache=self.crossattn_cache,
                current_start=current_start_frame * self.frame_seq_length,
                rope_enhance=self.rope_enhance, ttt_cache=self.ttt_cache, ttt_update=write,
                switch_global_frame=self._switch_global_frame)

        noisy_input = noise[:, current_start_frame - num_input_frames:
                            current_start_frame + current_num_frames - num_input_frames]
        for index, current_timestep in enumerate(steps):
            timestep = torch.ones(shape, device=noise.device, dtype=torch.int64) * current_timestep
            _, denoised_pred = generator(noisy_input, timestep, write=False)
            if index < len(steps) - 1:              # renoise to the next step of the sampler
                noisy_input = self.scheduler.add_noise(
                    denoised_pred.flatten(0, 1),
                    torch.randn_like(denoised_pred.flatten(0, 1)),
                    steps[index + 1] * torch.ones([batch_size * current_num_frames],
                                                  device=noise.device, dtype=torch.long),
                ).unflatten(0, denoised_pred.shape[:2])

        output[:, current_start_frame:
               current_start_frame + current_num_frames] = denoised_pred.to(output.device)
        generator(denoised_pred, torch.ones_like(timestep) * self.args.context_noise, write=True)

    def _switch_recache(self, output, current_start_frame, cond_new, *, a_end=None,
                        seg_idx=None, prompt_idx=None, is_revisit=False, subject=None,
                        cond_prev=None):
        """Cache surgery at one schedule boundary.

        is_revisit=False -> plain switch: no recall hook fires and, with FRAMEBANK off, this is
        upstream's recache under the new prompt.
        is_revisit=True  -> the entering subject has been seen before: the recall hooks fire.
        `a_end` = end latent frame (exclusive) of the segment where the entering subject first
        appeared, i.e. the anchor frame is `a_end - 1`.

        Note on FRAMEBANK: its router runs at every transition, revisit or not, Echo-Forcing
        is never told which switch is a revisit, so a false recall on a fresh scene
        is a real failure mode of the baseline and must be measurable, not hidden.
        """
        window = None
        timesteps = None                  # MEM_ANCHOR_T: per-slot declared timestep of the recache
        composites_before = self._comp_applied
        if is_revisit:
            assert a_end is not None and a_end >= 1, "revisit without a first-appearance segment"
            # MEM_EPISODIC: restore the state snapshotted when this subject was last left.
            snapshots = self._ttt_snap_by_prompt
            subject_key = prompt_idx if subject is None else subject
            if env_int("MEM_EPISODIC") and snapshots.get(subject_key):
                self._restore_ttt(snapshots[subject_key])
                log(f"[MEM_EPISODIC] restored the memory state of subject {subject_key} at frame "
                    f"{current_start_frame}")
            self._dump_anchor(output, a_end)
            if env_int("MEM_COMP_PROMPT_GATE"):
                self._comp_gate_from_prompts(cond_new, cond_prev)
        # ---- MEM_ADDITIVE_KV: the anchor enters as extra k/v tokens, no splice ----------------
        additive = bool(is_revisit and env_int("MEM_ADDITIVE_KV"))
        if additive:
            assert not env_int("REPLAY_PAST") and not self._fb_mode(), \
                "MEM_ADDITIVE_KV is the channel that evicts nothing: do not combine it with a " \
                "splice arm (REPLAY_PAST / FRAMEBANK)"
            anchor_ckpt = env_str("MEM_DECODE")
            if anchor_ckpt:                       # the decoded anchor, additive
                content, _ = self._memdec_frame(anchor_ckpt, cond_new, output.dtype, output.device)
            else:                         # the true anchor frame, additive
                content = self._anchor_content(output, a_end - 1, a_end)
            self._install_add_kv(content, cond_new)
        if is_revisit:
            window = self._build_replay_window(output, current_start_frame, a_end,
                                                composite=True)
        window = self._degrade_replay_window(window, current_start_frame)
        if self._fb_mode():                 # the frame bank routes at every switch
            assert window is None and not env_str("MEM_DECODE"), \
                "FRAMEBANK, REPLAY_PAST and MEM_DECODE are mutually exclusive"
            window = self._framebank_window(output, current_start_frame, cond_new, seg_idx,
                                             prompt_idx, subject, is_revisit)
        # ---- when the decoded anchor fires --------------------------------------------------
        # By default the schedule says when a subject is revisited, which is correct but is a
        # signal the frame-bank baseline does not get. MEM_DECODE_ROUTED=1 uses that baseline's
        # own router instead, so the two arms differ only by what they bring back.
        recalls = is_revisit
        if env_int("MEM_DECODE_ROUTED"):
            assert not self._fb_mode(), \
                "MEM_DECODE_ROUTED and FRAMEBANK are mutually exclusive (one router, one arm)"
            _, routing = self._fb_select(seg_idx, prompt_idx, cond_new, subject=subject, mode="key")
            recalls = bool(routing["recalled"])
            log(f"[MEM_DECODE_ROUTED] segment {seg_idx}: router says "
                f"{'REVISIT -> splice' if recalls else 'fresh scene -> no splice'} "
                f"(ground truth was {'revisit' if is_revisit else 'fresh'})")
        decoder_ckpt = env_str("MEM_DECODE") if (recalls and not additive) else ""
        if decoder_ckpt:
            assert window is None, "MEM_DECODE and REPLAY_PAST are mutually exclusive"
            window, timesteps = self._decoded_anchor_window(output, current_start_frame, cond_new,
                                                            decoder_ckpt)
        if env_int("MEM_DECODE_COMPOSITE") and window is not None:
            assert self._comp_applied > composites_before, \
                "MEM_DECODE_COMPOSITE=1 but the splice built at this switch carries no " \
                "composited anchor, unsupported placement (SLOT / POS=head / full window / " \
                "FRAMEBANK content)"
        self._recache_with_replay(output, current_start_frame, cond_new, window,
                                  timestep_override=timesteps)
        # MEM_SINKPURGE: after the recache, so it rewrites the sink slots of the
        # cache the return will actually read. No-op unless the hook is on (default off).
        self._maybe_sinkpurge(output, current_start_frame, cond_new, is_revisit)

    def _framebank_window(self, output, current_start_frame, cond_new, seg_idx, prompt_idx,
                          subject, is_revisit):
        """The recache window of the frame-bank arm, or None when the router recalls nothing.
        The recalled frame takes the most recent slot, as the real-frame arm places it, and the
        rest of the window stays the true recent tail.

        FRAMEBANK_GT takes the moment of the recall from the schedule instead of the router, so
        every arm brings its frame back at the same instant and only the content differs; the
        router still runs and its decision is logged."""
        ground_truth = bool(env_int("FRAMEBANK_GT"))
        frame, _ = self._fb_select(seg_idx, prompt_idx, cond_new, subject=subject,
                                   force=(bool(is_revisit) if ground_truth else None))
        if frame is None or (ground_truth and not is_revisit):
            return None
        n = self._window_frames(current_start_frame)
        log(f"[FRAMEBANK] recalled frame spliced at recency (1/{n} slots)")
        return torch.cat([output[:, current_start_frame - n:current_start_frame - 1],
                          frame.to(dtype=output.dtype, device=output.device)], dim=1)

    def _decoded_anchor_window(self, output, current_start_frame, cond_new, ckpt):
        """(recache window, per-slot timesteps or None) for the arm that tests the memory: the
        anchor is decoded from the live memory state, put through the content controls, then
        sharpened, composited and placed. The order is part of the design: the dump comes first,
        so a dumped anchor stays the raw decoder output another run reads back; the controls
        come before the sharpening, so sharpening a wrong anchor is the exact twin of
        sharpening the right one; the composite comes last, so it applies to whatever they left."""
        n = self._window_frames(current_start_frame)
        if env_int("MEM_DECODE_THRU"):
            frame, cond_on = self._early_decoded_frame(), "thru"
        else:
            frame, cond_on = self._memdec_frame(ckpt, cond_new, output.dtype, output.device)
        kslots = max(1, env_int("MEM_DECODE_SLOTS", 1))
        dump = env_str("MEM_DECODE_DUMP")
        if dump:
            # another run may be reading this file right now: write aside, then rename
            path = f"{dump}.f{current_start_frame}.pt"
            torch.save(frame.detach().to("cpu", torch.float16), path + ".tmp")
            os.replace(path + ".tmp", path)
            log(f"[MEM_DECODE_DUMP] anchor latent saved -> {path}")
        frame = self._maybe_wrong_anchor(frame)
        frame = self._maybe_noise_anchor(frame, output, current_start_frame, n)
        frame = self._maybe_sdedit(frame, cond_new, current_start_frame)
        frame = self._maybe_composite(frame, output, current_start_frame)
        offset = env_int("MEM_DECODE_SLOT_OFFSET", -1)
        window = self._memdec_window(output, current_start_frame, n, frame, kslots, offset)
        # MEM_ANCHOR_T: noise the anchor and declare that noise level for its slots
        window, timesteps = self._maybe_declare_anchor_noise(window, n, kslots, offset,
                                                             current_start_frame)
        log(f"[MEM_DECODE] decoded anchor spliced ({kslots}/{n} slots, window convention "
            f"{'recency duplicated' if offset < 0 else f'slot_offset={offset}'}"
            f"), cond={cond_on if cond_on == 'thru' else ('ON' if cond_on else 'off')}")
        return window, timesteps

    def _report_attention_mass(self):
        """MEM_ADDITIVE_KV: how much attention the extra anchor tokens took, on the last forward
        and, with MEM_ADD_MASS_SRC=1, per source [sink | window | anchor] over the chunks after the
        return. A channel read at about 0 % is dead by construction, whatever it carries."""
        mass = self._add_mass_report()
        if mass:
            values = [m for m in mass.values() if m == m]      # drop the NaN of blocks without the hook
            log(f"[MEM_ADDITIVE_KV] attention mass of the anchor tokens on the last forward: "
                f"mean {sum(values) / max(1, len(values)):.4f}, min {min(values):.4f}, "
                f"max {max(values):.4f} "
                f"| per block " + " ".join(f"{b}:{m:.3f}" for b, m in sorted(mass.items())))
        trace = self._add_mass_log
        if trace:           # one entry per chunk; MEM_ADD_MASS_DUMP names a file for the whole of it
            after_return = [e for e in trace if e["frame"] > (self._switch_global_frame or 0)]
            steps = after_return or trace
            share = {k: sum(b[k] for e in steps for b in e["per_block"].values())
                    / max(1, sum(len(e["per_block"]) for e in steps))
                    for k in ("sink", "window", "anchor")}
            n_tokens = next(iter(steps[-1]["per_block"].values()))["n_tok"]
            log(f"[MEM_ADD_MASS_SRC] attention mass per source after the return "
                f"({len(steps)} steps x {len(steps[-1]['per_block'])} blocks): "
                f"sink {share['sink']:.4f} | window {share['window']:.4f} | "
                f"anchor {share['anchor']:.4f}   (tokens {n_tokens} = sink/window/anchor, "
                f"uniform share of the anchor "
                f"{n_tokens[2] / max(1, sum(n_tokens)):.4f})")
            dump = env_str("MEM_ADD_MASS_DUMP")
            if dump:
                os.makedirs(os.path.dirname(dump) or ".", exist_ok=True)
                with open(dump, "w") as f:
                    json.dump({"summary_after_return": share, "n_tok": n_tokens, "steps": trace}, f)
                log(f"[MEM_ADD_MASS_SRC] full trace -> {dump}")

    def _enter_segment(self, plan, conds, seg, output, current_start_frame):
        """The switch into segment `seg`: bank and snapshot the segment that just ended, then
        rebuild the attention context under the new prompt, with a recall if it is a revisit."""
        left = seg - 1
        # before the recall, so the scene just left is in the bank and excluded as the
        # adjacent one, which is Echo-Forcing's routing rule
        self._fb_store(output, current_start_frame, left, plan.prompt[left], conds[left],
                       subject=plan.subject[left])
        if env_int("MEM_EPISODIC") and plan.returns_later(left):
            self._ttt_snap_by_prompt[plan.subject[left]] = self._snapshot_ttt()
            log(f"[MEM_EPISODIC] snapshotted the state of subject {plan.subject[left]} "
                f"at frame {current_start_frame}")
        self._switch_recache(
            output, current_start_frame, conds[seg], a_end=plan.anchor_end(seg), seg_idx=seg,
            prompt_idx=plan.prompt[seg], subject=plan.subject[seg],
            is_revisit=plan.is_revisit[seg], cond_prev=conds[left])
        self._switch_global_frame = current_start_frame
        log(f"[schedule] switch to segment {seg} (prompt {plan.prompt[seg]}) at frame "
            f"{current_start_frame}")

    def _trace_attention_mass(self, current_start_frame, seg):
        """One point per generated chunk of the attention mass per source, for the additive
        arm. Nothing is kept unless MEM_ADD_MASS_SRC=1, so it costs nothing otherwise."""
        if not self._add_installed:
            return
        per_source = self._add_mass_src_report()
        if per_source is not None:
            self._add_mass_log.append({"frame": int(current_start_frame), "segment": int(seg),
                                       "per_block": per_source})

    def inference_schedule(
        self,
        noise: torch.Tensor,
        *,
        prompts: List[str],
        schedule: List,
        subjects: Optional[List[int]] = None,
        return_latents: bool = False,
        low_memory: bool = False,
    ):
        """N-switch generation.

        Prompts  : the distinct prompt strings, indexed by the schedule.
        schedule : [(start_frame, prompt_idx), ...], strictly increasing start frames, the first
                   one 0, every start frame a multiple of num_frame_per_block. A segment whose
                   subject already appeared earlier is a revisit: the recall hooks fire and the
                   anchor is taken from the segment of that subject's first appearance.
        subjects : optional ground-truth identity per segment (same length as schedule).
                   Defaults to the prompt indices, i.e. "same prompt string = same subject".
                   Give it explicitly when a revisit is described by a paraphrase: then the
                   conditioning differs but the subject (hence the re-ID reference and the
                   frame-bank ceiling) is the same, and the router has a real matching job.

        A prompt is a string, or a list of strings aligned with the batch. The text encoder is
        called once per schedule entry, in schedule order.
        """
        num_output_frames = noise.shape[1]
        assert num_output_frames % self.num_frame_per_block == 0
        plan = parse_schedule(schedule, len(prompts), subjects, num_output_frames,
                              self.num_frame_per_block)
        for line in plan.describe(prompts):
            log(f"[schedule] {line}")
        conds = [self.text_encoder(text_prompts=[prompts[p]] if isinstance(prompts[p], str)
                                   else list(prompts[p])) for p in plan.prompt]
        starts = [start for start, _ in plan.schedule]
        # MEM_DECODE_THRU decodes the anchor when its chunk leaves the window, so it has to know
        # the anchor from the start: the one of the only revisit
        revisits = [i for i, revisit in enumerate(plan.is_revisit) if revisit]
        assert not env_int("MEM_DECODE_THRU") or len(revisits) == 1, \
            f"MEM_DECODE_THRU follows a single anchor: the schedule needs one revisit, got {len(revisits)}"
        early = revisits[0] if len(revisits) == 1 else None

        output = self._begin_rollout(
            noise, low_memory, first_switch_frame=None if early is None else plan.anchor_end(early))
        seg = 0
        for current_start_frame in range(0, num_output_frames, self.num_frame_per_block):
            if seg + 1 < len(starts) and current_start_frame >= starts[seg + 1]:
                seg += 1
                self._enter_segment(plan, conds, seg, output, current_start_frame)
            self._denoise_chunk(noise, output, conds[seg], current_start_frame,
                                num_input_frames=0, current_num_frames=self.num_frame_per_block)
            if early is not None:
                self._thru_maybe_decode(current_start_frame, conds[early], output.dtype, output.device)
            self._trace_attention_mass(current_start_frame, seg)

        self._report_attention_mass()

        # the generation loop is over and nothing reads these caches again, so release them
        # (~3.5 GB) right before the VAE decode, the biggest allocation of the rollout.
        if low_memory:
            import gc
            self.kv_cache1 = None
            self.crossattn_cache = None
            self._clear_add_kv()
            gc.collect()
            torch.cuda.empty_cache()
        video = self.vae.decode_to_pixel(output.to(noise.device), use_cache=False, to_cpu=low_memory)
        video = (video * 0.5 + 0.5).clamp(0, 1)

        if return_latents:
            return video, output
        return video
