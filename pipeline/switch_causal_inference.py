# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES
#
# Licensed under the Apache License, Version 2.0 (the "License").
# You may not use this file except in compliance with the License.
# To view a copy of this license, visit http://www.apache.org/licenses/LICENSE-2.0
#
# No warranties are given. The work is provided "AS IS", without warranty of any kind, express or implied.
#
# SPDX-License-Identifier: Apache-2.0
from typing import List, Optional
import torch

from utils.wan_wrapper import WanDiffusionWrapper, WanTextEncoder, WanVAEWrapper
from pipeline.causal_inference import CausalInferencePipeline  # reuse init utilities
from pipeline.switch.decoded_anchor import AnchorDecodeMixin
from pipeline.switch.anchor_controls import AnchorControlsMixin
from pipeline.switch.replay import ReplayPastMixin
from pipeline.switch.schedule import two_switch_schedule
from pipeline.switch.driver import ScheduleDriverMixin


class SwitchCausalInferencePipeline(ScheduleDriverMixin, AnchorDecodeMixin, AnchorControlsMixin,
                                    ReplayPastMixin, CausalInferencePipeline):
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
                              frames_override=None):
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

    def inference(self, noise: torch.Tensor, *, text_prompts_first: List[str],
                  text_prompts_second: List[str], switch_frame_index: int,
                  text_prompts_third: Optional[List[str]] = None,
                  second_switch_frame_index: Optional[int] = None,
                  return_latents: bool = False, low_memory: bool = False):
        """One prompt switch, A -> B, or two when `second_switch_frame_index` is given: A -> B
        and back. Each prompt is a list aligned with the batch. This is `inference_schedule` on a
        schedule of two or three segments: a switch index inside a block takes effect at the next
        block, and the third prompt, the first one unless given, counts as the return of the
        first subject whatever its wording (see two_switch_schedule)."""
        prompts = [text_prompts_first, text_prompts_second] + (
            [text_prompts_third] if text_prompts_third is not None else [])
        schedule, subjects = two_switch_schedule(
            switch_frame_index, second_switch_frame_index, text_prompts_third is not None,
            noise.shape[1], self.num_frame_per_block)
        return self.inference_schedule(noise, prompts=prompts, schedule=schedule, subjects=subjects,
                                       return_latents=return_latents, low_memory=low_memory)
