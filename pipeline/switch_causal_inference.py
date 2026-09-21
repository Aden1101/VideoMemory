# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES
#
# Licensed under the Apache License, Version 2.0 (the "License").
# You may not use this file except in compliance with the License.
# To view a copy of this license, visit http://www.apache.org/licenses/LICENSE-2.0
#
# No warranties are given. The work is provided "AS IS", without warranty of any kind, express or implied.
#
# SPDX-License-Identifier: Apache-2.0
import json
import os
from typing import List, Optional
import torch

from utils.wan_wrapper import WanDiffusionWrapper, WanTextEncoder, WanVAEWrapper
from utils.memory import gpu, get_cuda_free_memory_gb, move_model_to_device_with_memory_preservation
from utils.env import env_int, env_on, env_str
from pipeline.causal_inference import CausalInferencePipeline  # reuse init utilities
from wan.modules.video_memory import TTTBlockCache
from pipeline.switch.memory_state import MemoryStateMixin, sync_memory_writer
from pipeline.switch.decoded_anchor import AnchorDecodeMixin
from pipeline.switch.anchor_controls import AnchorControlsMixin
from pipeline.switch.sdedit import SdeditMixin
from pipeline.switch.sink import SinkMixin
from pipeline.switch.composite import CompositeMixin
from pipeline.switch.additive_kv import AdditiveKVMixin
from pipeline.switch.frame_bank import FrameBankMixin
from pipeline.switch.replay import ReplayPastMixin
from pipeline.switch.schedule import parse_schedule
from pipeline.switch.driver import ScheduleDriverMixin


# arms that only inference_schedule() implements; inference() refuses to run with one of them on
_SCHEDULE_ONLY_ARMS = ("MEM_DECODE_COMPOSITE", "MEM_ADDITIVE_KV", "MEM_ANCHOR_T", "MEM_SDEDIT",
                       "MEM_SINKPURGE", "MEM_DECODE_NOISE", "MEM_DECODE_SRC")


class SwitchCausalInferencePipeline(ScheduleDriverMixin, MemoryStateMixin, AnchorDecodeMixin, AnchorControlsMixin,
                                    SdeditMixin, SinkMixin, CompositeMixin, AdditiveKVMixin,
                                    FrameBankMixin, ReplayPastMixin, CausalInferencePipeline):
    """Generate a video, switch prompt one or more times, and optionally feed a past
    subject back into the attention context at the return. Each way of feeding it back
    lives in its own module under pipeline/switch/."""

    def __init__(
        self,
        args,
        device,
        *,
        generator: WanDiffusionWrapper | None = None,
        text_encoder: WanTextEncoder | None = None,
        vae: WanVAEWrapper | None = None,
    ):
        super().__init__(args, device, generator=generator, text_encoder=text_encoder, vae=vae)
        self.global_sink = getattr(args, "global_sink", False)

    def _recache_after_switch(self, output, current_start_frame, new_conditional_dict,
                              frames_override=None, timestep_override=None):
        if not self.global_sink:
            # reset kv cache
            for block_idx in range(self.num_transformer_blocks):
                cache = self.kv_cache1[block_idx]
                cache["k"].zero_()
                cache["v"].zero_()
                # cache["global_end_index"].zero_()
                # cache["local_end_index"].zero_()
            
        # reset cross-attention cache
        for blk in self.crossattn_cache:
            blk["k"].zero_()
            blk["v"].zero_()
            blk["is_init"] = False
        
        # recache
        if current_start_frame == 0:
            return

        num_recache_frames = current_start_frame if self.local_attn_size == -1 else min(self.local_attn_size, current_start_frame)
        recache_start_frame = current_start_frame - num_recache_frames

        if frames_override is not None:
            # recache frames taken from elsewhere in the video, at the current positions:
            # this is how a past frame is fed back into the attention context.
            assert frames_override.shape[1] == num_recache_frames, \
                f"frames_override has {frames_override.shape[1]} frames, recache needs {num_recache_frames}"
            frames_to_recache = frames_override
        else:
            frames_to_recache = output[:, recache_start_frame:current_start_frame]
        # move to gpu
        if frames_to_recache.device.type == 'cpu':
            target_device = next(self.generator.parameters()).device
            frames_to_recache = frames_to_recache.to(target_device)
        batch_size = frames_to_recache.shape[0]
        print(f"num_recache_frames: {num_recache_frames}, recache_start_frame: {recache_start_frame}, current_start_frame: {current_start_frame}")
        
        # prepare blockwise causal mask
        device = frames_to_recache.device
        block_mask = self.generator.model._prepare_blockwise_causal_attn_mask(
            device=device,
            num_frames=num_recache_frames,
            frame_seqlen=self.frame_seq_length,
            num_frame_per_block=self.num_frame_per_block,
            local_attn_size=self.local_attn_size
        )

        context_timestep = torch.ones([batch_size, num_recache_frames],
                                    device=device, dtype=torch.int64) * self.args.context_noise
        # MEM_ANCHOR_T: the declared noise level is already per frame, so one slot can be
        # declared noisy while the 11 true frames keep context_noise. None = off.
        if timestep_override is not None:
            # With uniform_timestep, the wrapper reduces the per-frame timestep to slot 0's,
            # which would silently drop the declaration and turn this into plain noising.
            assert not getattr(self.generator, "uniform_timestep", False), \
                "MEM_ANCHOR_T needs a causal generator (uniform_timestep=False): otherwise the " \
                "per-frame timestep is reduced to slot 0's and the declaration is lost"
            assert timestep_override.shape[-1] == num_recache_frames, \
                f"timestep_override has {timestep_override.shape[-1]} entries, recache needs " \
                f"{num_recache_frames}"
            context_timestep = timestep_override.to(device=device).round().to(torch.int64) \
                .reshape(1, num_recache_frames).expand(batch_size, -1).contiguous()
            _d = [(i, int(x)) for i, x in enumerate(context_timestep[0].tolist())
                  if int(x) != int(self.args.context_noise)]
            print(f"[MEM_ANCHOR_T] recache timesteps: {len(_d)} slot(s) declared off-baseline "
                  f"{_d} (baseline {int(self.args.context_noise)}, {num_recache_frames} slots)",
                  flush=True)

        self.generator.model.block_mask = block_mask

        with torch.no_grad():
            self.generator(
                noisy_image_or_video=frames_to_recache,
                conditional_dict=new_conditional_dict,
                timestep=context_timestep,
                kv_cache=self.kv_cache1,
                crossattn_cache=self.crossattn_cache,
                current_start=recache_start_frame * self.frame_seq_length,
                rope_enhance=self.rope_enhance,
                sink_recache_after_switch=not self.global_sink,
                ttt_cache=getattr(self, "ttt_cache", None),
                ttt_update=False,
            )

        # reset cross-attention cache
        for blk in self.crossattn_cache:
            blk["k"].zero_()
            blk["v"].zero_()
            blk["is_init"] = False

    def inference(
        self,
        noise: torch.Tensor,
        *,
        text_prompts_first: List[str],
        text_prompts_second: List[str],
        switch_frame_index: int,
        text_prompts_third: Optional[List[str]] = None,
        second_switch_frame_index: Optional[int] = None,
        initial_latent: Optional[torch.Tensor] = None,
        return_latents: bool = False,
        low_memory: bool = False,
    ):
        """Generate video with prompt switch.

        Notes (differences from base inference):
            text_prompts_first: prompt list used before switching (aligned with batch)
            text_prompts_second: prompt list used after switching
            switch_frame_index: 0-based frame index; frames >= this value use the second prompt
            text_prompts_third: optional prompt list for a second switch (A->B->A revisit);
                defaults to text_prompts_first when second_switch_frame_index is set
            second_switch_frame_index: optional 0-based frame index; frames >= this value use
                the third prompt (must be > switch_frame_index). None = single-switch A->B
                behaviour, byte-identical to before.
        """
        batch_size, num_output_frames, num_channels, height, width = noise.shape
        assert num_output_frames % self.num_frame_per_block == 0
        num_blocks = num_output_frames // self.num_frame_per_block

        schedule_only = [name for name in _SCHEDULE_ONLY_ARMS if env_on(name)]
        if env_int("MEM_DECODE_SLOT_OFFSET", -1) >= 0:
            schedule_only.append("MEM_DECODE_SLOT_OFFSET")
        assert not schedule_only, \
            f"{schedule_only} exist in the N-switch driver only, run them through inference_schedule()"

        # Encode both prompts upfront
        cond_first = self.text_encoder(text_prompts=text_prompts_first)
        cond_second = self.text_encoder(text_prompts=text_prompts_second)
        # Optional A->B->A revisit: encode the third prompt (defaults to the first prompt)
        cond_third = None
        if second_switch_frame_index is not None:
            assert second_switch_frame_index > switch_frame_index, \
                "second_switch_frame_index must come after switch_frame_index"
            if text_prompts_third is None:
                text_prompts_third = text_prompts_first
            cond_third = self.text_encoder(text_prompts=text_prompts_third)

        output = self._begin_rollout(noise, low_memory)
        current_start_frame = 0
        num_input_frames = 0

        # Temporal denoising by blocks
        all_num_frames = [self.num_frame_per_block] * num_blocks

        using_second = False
        using_third = False
        for current_num_frames in all_num_frames:
            if (second_switch_frame_index is not None) and using_second and (not using_third) \
                    and (current_start_frame >= second_switch_frame_index):
                # MEM_EPISODIC (off by default): restore the memory state
                # snapshotted at the A->B switch, before the recache, so the return-of-A rollout
                # (recache reads included) runs on the A-era state instead of B's recency.
                if env_int("MEM_EPISODIC") and getattr(self, "_ttt_snap_A", None):
                    self._restore_ttt(self._ttt_snap_A)
                    print(f"[MEM_EPISODIC] restored A-era memory state at frame {current_start_frame}")
                # Replaying the past: rebuild the attention window from the real frames of
                # the subject's first appearance instead of the current ones. Deliberate
                # cheating, it gives the ceiling that a perfect memory could reach.
                # The anchor is dumped first, before any replay touches `output`.
                self._dump_anchor(output, self._first_switch_frame)
                _splice = self._build_replay_window(output, current_start_frame,
                                                    self._first_switch_frame)
                _splice = self._degrade_replay_window(_splice, current_start_frame)
                # the frame fed back is decoded from the live memory instead of being the real
                # past frame. Same mechanism as above, but nothing is read from the true past:
                # this is the arm that actually tests the memory.
                _dec_ckpt = env_str("MEM_DECODE")
                if _dec_ckpt:
                    assert _splice is None, "MEM_DECODE and REPLAY_PAST are mutually exclusive"
                    _splice = self._single_anchor_window(output, current_start_frame, cond_third,
                                                         _dec_ckpt)
                self._recache_with_replay(output, current_start_frame, cond_third, _splice)
                cond_in_use = cond_third
                using_third = True
                self._switch_global_frame = current_start_frame
                print("second_switch_frame_index", second_switch_frame_index)
                print("current_start_frame", current_start_frame)
            elif (not using_second) and (current_start_frame >= switch_frame_index):
                # MEM_EPISODIC: freeze the memory of the opening scene before the next one overwrites it.
                if env_int("MEM_EPISODIC"):
                    self._ttt_snap_A = self._snapshot_ttt()
                    print(f"[MEM_EPISODIC] snapshotted A-era memory state at frame {current_start_frame}")
                self._first_switch_frame = current_start_frame
                self._recache_after_switch(output, current_start_frame, cond_second)
                cond_in_use = cond_second
                using_second = True
                self._switch_global_frame = current_start_frame
                print("switch_frame_index", switch_frame_index)
                print("current_start_frame", current_start_frame)
            else:
                cond_in_use = (cond_third if using_third
                               else (cond_second if using_second else cond_first))
            
            self._denoise_chunk(noise, output, cond_in_use, current_start_frame,
                                num_input_frames, current_num_frames)

            # MEM_DECODE_THRU (default off = no-op): the write that stores the
            # anchor chunk has just happened -> decode and cache the frame now (age 1 write).
            self._thru_maybe_decode(current_start_frame, cond_third, output.dtype, output.device)

            current_start_frame += current_num_frames

        # Standard decoding (to_cpu mirrors causal_inference.py: stack decoded chunks in RAM
        # instead of VRAM for long videos)
        video = self.vae.decode_to_pixel(output.to(noise.device), use_cache=False, to_cpu=low_memory)
        video = (video * 0.5 + 0.5).clamp(0, 1)

        if return_latents:
            return video, output
        return video
