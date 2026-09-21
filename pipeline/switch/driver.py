"""The rollout with any number of prompt switches: generate chunk by chunk under a schedule,
and at each switch rebuild the attention context, feeding a past subject back into it when the
segment is a revisit. Which arm feeds it back is chosen by the environment: real past frames
(replay.py) or an anchor decoded from the memory (decoded_anchor.py), with the controls on
their content in anchor_controls.py.

Mixed into SwitchCausalInferencePipeline, whose `inference()` is the two-switch case of it.
"""
import os
from typing import List, Optional
import torch
from utils.memory import gpu, get_cuda_free_memory_gb, move_model_to_device_with_memory_preservation
from utils.env import check_arm_variables, env_int, env_str
from utils.log import log
from wan.modules.video_memory import TTTBlockCache
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

    def _window_frames(self, current_start_frame):
        """How many past frames the attention context holds when generation reaches
        `current_start_frame`: all of them under global attention, else the local window."""
        if self.local_attn_size == -1:
            return current_start_frame
        return min(self.local_attn_size, current_start_frame)

    def _begin_rollout(self, noise, low_memory):
        """Before the first chunk: check the arm variables, free the text encoder, allocate the
        output and the caches. Returns the output tensor."""
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
                        is_revisit=False):
        """Rebuild the attention context at a schedule boundary. A plain switch is upstream's
        recache under the new prompt. At a revisit the window can carry the entering subject
        instead: its real past frames (REPLAY_PAST) or an anchor decoded from the memory
        (MEM_DECODE). `a_end` is the end frame, exclusive, of the segment where that subject
        first appeared, so the anchor frame is `a_end - 1`."""
        window = None
        if is_revisit:
            assert a_end is not None and a_end >= 1, "revisit without a first-appearance segment"
            self._dump_anchor(output, a_end)
            window = self._build_replay_window(output, current_start_frame, a_end)
            window = self._degrade_replay_window(window, current_start_frame)
            decoder_ckpt = env_str("MEM_DECODE")
            if decoder_ckpt:
                assert window is None, "MEM_DECODE and REPLAY_PAST are mutually exclusive"
                window = self._decoded_anchor_window(output, current_start_frame, cond_new,
                                                     decoder_ckpt)
        self._recache_with_replay(output, current_start_frame, cond_new, window)

    def _decoded_anchor_window(self, output, current_start_frame, cond_new, ckpt):
        """The recache window of the arm that tests the memory: the anchor is decoded from the
        live memory state, put through the content controls, then placed. The dump comes first,
        so a dumped anchor stays the raw decoder output another run reads back."""
        n = self._window_frames(current_start_frame)
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
        offset = env_int("MEM_DECODE_SLOT_OFFSET", -1)
        log(f"[MEM_DECODE] decoded anchor spliced ({kslots}/{n} slots, window convention "
            f"{'recency duplicated' if offset < 0 else f'slot_offset={offset}'}"
            f"), cond={'ON' if cond_on else 'off'}")
        return self._memdec_window(output, current_start_frame, n, frame, kslots, offset)

    def _enter_segment(self, plan, conds, seg, output, current_start_frame):
        """The switch into segment `seg`: rebuild the attention context under the new prompt,
        with a recall if the segment is a revisit."""
        self._switch_recache(output, current_start_frame, conds[seg],
                             a_end=plan.anchor_end(seg), is_revisit=plan.is_revisit[seg])
        self._switch_global_frame = current_start_frame
        log(f"[schedule] switch to segment {seg} (prompt {plan.prompt[seg]}) at frame "
            f"{current_start_frame}")

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
                   conditioning differs but the subject, hence the anchor, is the same.

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

        output = self._begin_rollout(noise, low_memory)
        seg = 0
        for current_start_frame in range(0, num_output_frames, self.num_frame_per_block):
            if seg + 1 < len(starts) and current_start_frame >= starts[seg + 1]:
                seg += 1
                self._enter_segment(plan, conds, seg, output, current_start_frame)
            self._denoise_chunk(noise, output, conds[seg], current_start_frame,
                                num_input_frames=0, current_num_frames=self.num_frame_per_block)

        # the generation loop is over and nothing reads these caches again, so release them
        # (~3.5 GB) right before the VAE decode, the biggest allocation of the rollout.
        if low_memory:
            import gc
            self.kv_cache1 = None
            self.crossattn_cache = None
            gc.collect()
            torch.cuda.empty_cache()
        video = self.vae.decode_to_pixel(output.to(noise.device), use_cache=False, to_cpu=low_memory)
        video = (video * 0.5 + 0.5).clamp(0, 1)

        if return_latents:
            return video, output
        return video
