# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES
#
# Licensed under the Apache License, Version 2.0 (the "License").
# You may not use this file except in compliance with the License.
# To view a copy of this license, visit http://www.apache.org/licenses/LICENSE-2.0
#
# No warranties are given. The work is provided "AS IS", without warranty of any kind, express or implied.
#
# SPDX-License-Identifier: Apache-2.0
import os
import time
import random
import torch
import torch.distributed as dist
from typing import Tuple, Dict, Any, Optional, List
from einops import rearrange

from utils.debug_option import DEBUG, LOG_GPU_MEMORY, MY_VISUAL_DEBUG
from utils.memory import log_gpu_memory
from pipeline.streaming_switch_training import StreamingSwitchTrainingPipeline
import torch.nn.functional as F



class StreamingTrainingModel:
    """
    A model wrapper specifically for streaming/serialized training.

    This class wraps existing models (DMD, DMDSwitch, etc.) and provides a unified
    interface for streaming training. Main features:
    1. Manage streaming generation state
    2. Reuse KV cache and cross-attention cache
    3. Support prompt switching for DMD Switch
    4. Provide chunk-wise loss computation
    5. Support overlapping frames to ensure continuity
    """
    
    def __init__(self, base_model, config):
        """
        Initialize the streaming training model.

        Args:
            base_model: underlying model (DMD, DMDSwitch, etc.)
            config: configuration object
        """
        self.base_model = base_model
        self.config = config
        self.device = base_model.device
        self.dtype = base_model.dtype
        self.image_or_video_shape = getattr(config, 'image_or_video_shape', None)
        
        # Streaming training configuration
        self.chunk_size = getattr(config, "streaming_chunk_size", 21)  # Fixed chunk size used for loss computation
        self.max_length = getattr(config, "streaming_max_length", 57)
        self.possible_max_length = getattr(config, "streaming_possible_max_length", None)
        self.min_new_frame = getattr(config, "streaming_min_new_frame", 18)

        # Get required components from the underlying model
        self.generator = base_model.generator
        self.fake_score = base_model.fake_score
        self.scheduler = base_model.scheduler
        self.denoising_loss_func = base_model.denoising_loss_func
        
        # Fetch model configuration
        self.num_frame_per_block = base_model.num_frame_per_block
        self.frame_seq_length = getattr(base_model.inference_pipeline, 'frame_seq_length', 1560)
        
        # Initialize inference pipeline
        self.inference_pipeline = base_model.inference_pipeline
        if self.inference_pipeline is None:
            base_model._initialize_inference_pipeline()
            self.inference_pipeline = base_model.inference_pipeline
        
        # Streaming state
        self.reset_state()

        # sink update config
        self.sink_update_ratio = getattr(config, "sink_update_ratio", 0)
        self.use_prompt_cache = getattr(config, "use_prompt_cache", False)
        self.use_gan = getattr(config, "use_gan", False)
        self.use_decouple_dmd = getattr(config, "use_decouple_dmd", False)
        # Text-dropout (identity signal): prob of generating a chunk with null text so the
        # model must rely on memory/history for the subject. 0.0 = off (default = unchanged).
        self.text_dropout_prob = getattr(config, "text_dropout_prob", 0.0)
        # identity + anti-freeze objective terms; all weights 0.0 => unchanged.
        # id term: decode a few frames, DINO-embed, cosine to the chunk-0 anchor (grad flows through the decoder).
        self.id_loss_weight = getattr(config, "id_loss_weight", 0.0)
        self.id_decode_frames = getattr(config, "id_decode_frames", 2)   # latent frames decoded (grad) / chunk
        # Differentiable temporal-variance floor (anti-freeze): hinge on adjacent-latent motion.
        self.tv_loss_weight = getattr(config, "tv_loss_weight", 0.0)
        self.tv_motion_floor = getattr(config, "tv_motion_floor", 0.0)   # latent-diff floor; 0 => inactive
        # counterfactual-advantage objective (all defaults 0/off leave the baseline unchanged).
        # Rewards the memory only for beating its own motion-valid mem-OFF twin on identity:
        #   L = -w_adv * g * clamp(cos_ON - cos_CF, -neg, +pos) + w_rel*(log m_ON - log m_CF)^2
        #       + w_ext * relu(log(frac*m_stock) - log m_ON)^2
        self.adv_loss_weight = getattr(config, "adv_loss_weight", 0.0)   # bonus mode only (default OFF)
        self.adv_clamp_pos = getattr(config, "adv_clamp_pos", 0.05)      # reward saturates here (cos units)
        self.adv_clamp_neg = getattr(config, "adv_clamp_neg", 0.20)      # do-no-harm side stays linear 4x deeper
        self.adv_gate_sigma = getattr(config, "adv_gate_sigma", 0.405)   # ln(1.5): log-motion validity band
        self.adv_floor_frac = getattr(config, "adv_floor_frac", 0.5)     # ext floor fires below frac*stock motion
        self.mm_rel_weight = getattr(config, "mm_rel_weight", 0.0)       # two-sided log-motion match to the twin
        self.mm_ext_weight = getattr(config, "mm_ext_weight", 0.0)       # one-sided log floor vs stock-AF curve
        # do-no-harm hinge (identity may not be worse than the twin; nothing pays for beating it):
        self.dnh_weight = getattr(config, "dnh_weight", 0.0)
        self.dnh_deadband = getattr(config, "dnh_deadband", 0.01)        # cos units of tolerated noise
        # activity floor: the injection's causal AC-dose (real-vs-twin latent difference, global
        # per-channel mean removed so a DC/style bias cannot satisfy it) must stay >= act_floor.
        # relu => zero gradient once satisfied: a floor, never a maximizer.
        self.act_weight = getattr(config, "act_weight", 0.0)
        self.act_floor = getattr(config, "act_floor", 0.05)              # relative AC-dose target (calibrate)
        # anti-flicker (every default leaves the objective as it is without them):
        #  - motion_lag2: use the flicker-blind lag-2 motion statistic in mm_rel/mm_ext + the dose,
        #    so frame-rate shimmer cannot counterfeit motion credit (removes the flicker payer).
        #  - dose_tmean_remove: also subtract the per-site temporal mean in the AC-dose, so a static
        #    overlay (wallpaper) earns zero dose (escalation named in the docstring).
        #  - flick_weight/flick_deadband: latent second-temporal-difference hinge vs the twin (tripwire).
        self.motion_lag2 = bool(getattr(config, "motion_lag2", False))
        self.dose_tmean_remove = bool(getattr(config, "dose_tmean_remove", False))
        self.flick_weight = getattr(config, "flick_weight", 0.0)
        self.flick_deadband = getattr(config, "flick_deadband", 0.25)    # relative to the twin's roughness
        self._twin_anchor_on = (self.adv_loss_weight > 0.0 or self.dnh_weight > 0.0)
        self._twin_on = (self._twin_anchor_on or self.act_weight > 0.0
                       or self.mm_rel_weight > 0.0 or self.mm_ext_weight > 0.0
                       or self.flick_weight > 0.0)
        self._motion_targets = None                                      # {sha1(prompt)[:16]: [m_c0..], "_global": [..]}
        _mt_path = getattr(config, "motion_target_path", None)
        if _mt_path and self._twin_on:
            import json as _json
            with open(_mt_path) as _f:
                self._motion_targets = _json.load(_f)
            if (not dist.is_initialized() or dist.get_rank() == 0):
                print(f"[StreamingTrain-Model] twin objective on: w_adv={self.adv_loss_weight} "
                      f"w_dnh={self.dnh_weight} (deadband {self.dnh_deadband}) "
                      f"w_act={self.act_weight} (floor {self.act_floor}) "
                      f"w_rel={self.mm_rel_weight} w_ext={self.mm_ext_weight} floor_frac={self.adv_floor_frac} "
                      f"gate_sigma={self.adv_gate_sigma} targets={_mt_path} ({len(self._motion_targets)} keys)")
        self._dino = None                                                # lazy frozen DINOv2
        self._id_anchor = None                                           # per-sequence chunk-0 subject anchor
        if dist.get_rank() == 0:
            print(f"[StreamingTrain-Model] use_prompt_cache={self.use_prompt_cache}. use_gan={self.use_gan}. use_decouple_dmd={self.use_decouple_dmd}. text_dropout_prob={self.text_dropout_prob}")
            print(f"[StreamingTrain-Model] id/tv terms: id_loss_weight={self.id_loss_weight} id_decode_frames={self.id_decode_frames} tv_loss_weight={self.tv_loss_weight} tv_motion_floor={self.tv_motion_floor}")
        
        if DEBUG and (not dist.is_initialized() or dist.get_rank() == 0):
            print(f"[StreamingTrain-Model] streamingTrainingModel initialized:")
            print(f"[StreamingTrain-Model] chunk_size={self.chunk_size}, max_length={self.max_length}")
            print(f"[StreamingTrain-Model] min_new_frame={self.min_new_frame}")
            print(f"[StreamingTrain-Model] base_model type: {type(self.base_model).__name__}")

    def _process_first_frame_encoding(self, frames: torch.Tensor) -> torch.Tensor:
        """
        Apply special encoding to the first frame, following the logic in _run_generator.

        Args:
            frames: frame sequence [batch_size, num_frames, C, H, W]

        Returns:
            processed_frames: processed frame sequence where the first frame is re-encoded as an image latent
        """
        total_frames = frames.shape[1]
        
        if total_frames <= 1:
            # Only one or zero frames, return as is
            return frames
        
        # Determine the range to process: last 21 frames
        process_frames = min(21, total_frames)
        
        if DEBUG and (not dist.is_initialized() or dist.get_rank() == 0):
            print(f"[StreamingTrain-Model] Processing first frame encoding for loss: total_frames={total_frames}, processing last {process_frames} frames")
            
        with torch.no_grad():
            # Decode the frames to be processed into pixels
            frames_to_decode = frames[:, :-(process_frames - 1), ...]
            pixels = self.base_model.vae.decode_to_pixel(frames_to_decode)
            
            # Take the last frame's pixel representation
            last_frame_pixel = pixels[:, -1:, ...].to(self.dtype)  # 拿到最后一节的第一帧
            last_frame_pixel = rearrange(last_frame_pixel, "b t c h w -> b c t h w")
            
            # Re-encode as image latent
            image_latent = self.base_model.vae.encode_to_latent(last_frame_pixel).to(self.dtype)
            
        remaining_frames = frames[:, -(process_frames - 1):, ...]  # 最后的20帧
        processed_frames = torch.cat([image_latent, remaining_frames], dim=1)  # 拼上，符合wan输入形式
                
        if DEBUG and (not dist.is_initialized() or dist.get_rank() == 0):
            print(f"[StreamingTrain-Model] Processed first frame encoding: {frames.shape} -> {processed_frames.shape}")
            
        return processed_frames

    def reset_state(self):
        """Reset streaming training state"""
        if DEBUG and (not dist.is_initialized() or dist.get_rank() == 0):
            print(f"[StreamingTrain-Model] Resetting streaming training state")
            
        self.state = {
            "current_length": 0,
            "conditional_info": None,
            "has_switched": False,  # Track whether prompt has been switched
            "previous_frames": None,  # Store last generated frames for overlap (up to 21)
            "temp_max_length": None,  # Temporary max length for the current sequence
            # gan part
            "frames4gan": None,  # 用于存储generator后续送入gan的两帧
            "frames4gan_gt": None,  # 用于存储GT后续送入gan的两帧
            "frames4gan_mismatch": None, # 用于存储GT后续送入gan的mismatch两帧
            "frames4gan_gt_prompt": None,  # 用于存储GT后续送入gan的两帧的prompt
            "frames4gan_mismatch_gt_prompt": None,  # 用于存储GT后续送入gan的两帧的prompt
            "chunk_cnt": 0  # 用于记录chunk的个数，用于取出gt中对应的内容
        }

        self.inference_pipeline.clear_kv_cache()
        self.inference_pipeline.clear_flow_cache()
        self.inference_pipeline.clear_prompt_cache()
        
    
    def _should_switch_prompt(self, chunk_start_frame: int, chunk_size: int) -> bool:
        """Determine whether to switch prompt (DMDSwitch only)"""
        # Check if the model supports switching (DMDSwitch)
        if not isinstance(self.inference_pipeline, StreamingSwitchTrainingPipeline):
            if DEBUG and (not dist.is_initialized() or dist.get_rank() == 0):
                print(f"[StreamingTrain-Model] Not a switch pipeline, no switching")
            return False
        
        # If already switched, do not switch again
        if self.state.get("has_switched", False):
            if DEBUG and (not dist.is_initialized() or dist.get_rank() == 0):
                print(f"[StreamingTrain-Model] Already switched, not switching again")
            return False
        
        switch_info = self.state["conditional_info"].get("switch_info", {})
        switch_frame_index = switch_info.get("switch_frame_index")
        
        if switch_frame_index is None:
            if DEBUG and (not dist.is_initialized() or dist.get_rank() == 0):
                print(f"[StreamingTrain-Model] No switch_frame_index, not switching")
            return False
        
        # Check if the switch point falls within the current chunk range [chunk_start_frame, chunk_start_frame + chunk_size)
        chunk_end_frame = chunk_start_frame + chunk_size
        should_switch = chunk_start_frame <= switch_frame_index < chunk_end_frame
        
        if DEBUG and (not dist.is_initialized() or dist.get_rank() == 0):
            print(f"[StreamingTrain-Model] Switch check: switch_frame={switch_frame_index}, chunk=[{chunk_start_frame}, {chunk_end_frame}), should_switch={should_switch}")
            
        return should_switch

    def _get_current_conditional_dict(self, chunk_start_frame: int) -> dict:
        """Get the conditional_dict to use for the current chunk"""
        cond_info = self.state["conditional_info"]
        
        # Check whether it has switched already or should switch now
        switch_info = cond_info.get("switch_info", {})
        if switch_info:
            switch_frame_index = switch_info.get("switch_frame_index")
            if switch_frame_index is not None:
                if self.state.get("has_switched", False) or chunk_start_frame >= switch_frame_index:
                    # If already switched, or current frame has reached the switch point, use the switched prompt
                    if DEBUG and (not dist.is_initialized() or dist.get_rank() == 0):
                        print(f"[StreamingTrain-Model] Using switch conditional_dict for chunk starting at frame {chunk_start_frame}")
                    return switch_info.get("switch_conditional_dict", cond_info["conditional_dict"])
        
        # Otherwise use the original prompt
        if DEBUG and (not dist.is_initialized() or dist.get_rank() == 0):
            print(f"[StreamingTrain-Model] Using original conditional_dict for chunk starting at frame {chunk_start_frame}")
        return cond_info["conditional_dict"]
    
    def _generate_chunk(
        self,
        noise_chunk: torch.Tensor,
        chunk_start_frame: int,
        requires_grad: bool = True,
        detach_state: bool = True,
        keep_last_write_grad: bool = False,
    ) -> Tuple[torch.Tensor, Optional[int], Optional[int]]:
        """
        Generate a single chunk.

        Args:
            noise_chunk: noise input [batch_size, chunk_frames, C, H, W]
            chunk_start_frame: start frame index of the chunk in the full sequence
            requires_grad: whether gradients are required

        Returns:
            generated_chunk: generated chunk [batch_size, chunk_frames, C, H, W]
            denoised_timestep_from: starting timestep for denoising
            denoised_timestep_to: ending timestep for denoising
        """
        if (not dist.is_initialized() or dist.get_rank() == 0) and LOG_GPU_MEMORY:
            log_gpu_memory(f"StreamingTrain-Model: Before generate chunk {chunk_start_frame}", device=self.device, rank=dist.get_rank() if dist.is_initialized() else 0)
        
        if DEBUG and (not dist.is_initialized() or dist.get_rank() == 0):
            print(f"[StreamingTrain-Model] _generate_chunk: chunk_start_frame={chunk_start_frame}, chunk_size={noise_chunk.shape[1]}")
            print(f"[StreamingTrain-Model] requires_grad={requires_grad}")
            
        # Get the conditional_dict to use now
        current_conditional_dict = self._get_current_conditional_dict(chunk_start_frame)

        # Text-dropout: with prob text_dropout_prob generate this chunk on null text (the DMD target
        # and critic keep the true prompt) so the student must carry the subject from memory. Chunk 0 is
        # never dropped. The cross-attn cache holds the chunk-0 true prompt, so a dropped chunk also
        # bypasses that cache, else the denoising passes would still attend to the true text.
        disable_crossattn_cache = False
        if (self.text_dropout_prob > 0.0 and chunk_start_frame > 0
                and self.state.get("previous_frames") is not None
                and random.random() < self.text_dropout_prob):
            # copy: the cached null dict must not be mutated by generation (it may add initial_latent)
            current_conditional_dict = dict(self.state["conditional_info"]["unconditional_dict"])
            disable_crossattn_cache = True
            if DEBUG and (not dist.is_initialized() or dist.get_rank() == 0):
                print(f"[StreamingTrain-Model] text-dropout: null text at chunk {chunk_start_frame}")

        # Prepare generation parameters
        kwargs = {
            "noise": noise_chunk,
            "conditional_dict": current_conditional_dict,
            "current_start_frame": chunk_start_frame,
            "requires_grad": requires_grad,
            "return_sim_step": False,
            "disable_crossattn_cache": disable_crossattn_cache,
            "keep_last_write_grad": keep_last_write_grad,
        }
        
        # Add switching logic for DMDSwitch models
        if isinstance(self.inference_pipeline, StreamingSwitchTrainingPipeline):
            switch_info = self.state["conditional_info"].get("switch_info", {})
            if switch_info and self._should_switch_prompt(chunk_start_frame, noise_chunk.shape[1]):
                if (not dist.is_initialized() or dist.get_rank() == 0):
                    print(f"[StreamingTrain-Model] Switching prompt at frame {switch_info['switch_frame_index']}")
                
                # Compute the relative switch frame index (relative to current chunk start)
                relative_switch_index = max(0, switch_info["switch_frame_index"] - chunk_start_frame)
                kwargs["switch_frame_index"] = relative_switch_index
                kwargs["switch_conditional_dict"] = switch_info["switch_conditional_dict"]
                
                # Pass previous_frames for recache when switching
                if self.state["previous_frames"] is not None:
                    kwargs["switch_recache_frames"] = self.state["previous_frames"]
                    
                    if DEBUG and (not dist.is_initialized() or dist.get_rank() == 0):
                        print(f"[StreamingTrain-Model] Passed previous_frames for switch recache: {self.state['previous_frames'].shape}")
                
                if DEBUG and (not dist.is_initialized() or dist.get_rank() == 0):
                    print(f"[StreamingTrain-Model] Adding switch parameters: relative_switch_index={relative_switch_index}")
                
                # Mark switched to avoid switching again in later chunks
                self.state["has_switched"] = True

            # prompt cache初始化
            if self.inference_pipeline.prompt_cache is not None:
                self.state["conditional_info"].get("switch_info", {})
                switch_frame_index = switch_info.get("switch_frame_index")

                if MY_VISUAL_DEBUG and (not dist.is_initialized() or dist.get_rank() == 0):
                    print("[StreamingTrain-Model] switch_frame_index is {}".format(switch_frame_index))

                # no switch this sequence keeps the far-future sentinel so the switch gate never fires
                # (clear_prompt_cache zeroes switch_frame, which would spuriously trigger the gate)
                fill_val = switch_frame_index if switch_frame_index is not None else int(1e8)
                for block_index in range(self.inference_pipeline.num_transformer_blocks):
                    self.inference_pipeline.prompt_cache[block_index]["switch_frame"].fill_(fill_val)

        # the TTT state is carried chunk->chunk by reference, but training backwards once per chunk
        # (its graph is then freed). detach_state=True detaches the entering state so each chunk's
        # write-graph starts from a leaf (else "backward a second time" crash). detach_state=False
        # (chunks 2..K of a BPTT window) keeps the grad_fn so the loss backprops into the previous
        # chunk; the trainer sums the window's losses and backwards once (no freed-graph, no leak).
        ttt_cache = getattr(self.inference_pipeline, "ttt_cache", None)
        if detach_state and ttt_cache is not None:
            for _c in ttt_cache:
                if _c is not None and getattr(_c, "state", None) is not None:
                    _c.state = tuple(s.detach() for s in _c.state)

        # Call the pipeline-specific method to generate a chunk
        if DEBUG and (not dist.is_initialized() or dist.get_rank() == 0):
            print(f"[StreamingTrain-Model] Calling pipeline.generate_chunk_with_cache")
        
        if (not dist.is_initialized() or dist.get_rank() == 0) and LOG_GPU_MEMORY:
            log_gpu_memory(f"StreamingTrain-Model: Before pipeline.generate_chunk_with_cache", device=self.device, rank=dist.get_rank() if dist.is_initialized() else 0)
            
        output, denoised_timestep_from, denoised_timestep_to = self.inference_pipeline.generate_chunk_with_cache(**kwargs)
        # if DEBUG:
        #     # Inspect timestep info
        #     print(f"[DEBUG-SeqModel-GenChunk] denoised_timestep_from: {denoised_timestep_from}")
        #     print(f"[DEBUG-SeqModel-GenChunk] denoised_timestep_to: {denoised_timestep_to}")

        #     print(f"output shape: {output.shape}")
        #     with torch.no_grad():
        #         if self.state["previous_frames"] is not None:
        #             output_vis = torch.cat([self.state["previous_frames"], output], dim=1)
        #         else:
        #             output_vis = output
        #         video = self.base_model.vae.decode_to_pixel(output_vis, use_cache=False)
        #         video = (video * 0.5 + 0.5).clamp(0, 1)
        #         video = video.permute(0, 1, 3, 4, 2).cpu().numpy() * 255.0
        #         video_tensor = torch.from_numpy(video[0].astype("uint8"))
        #         from torchvision.io import write_video
        #         write_video(f"debug_save/output_{chunk_start_frame}_to_{chunk_start_frame+output.shape[1]}_denoise_{denoised_timestep_from}_{denoised_timestep_to}_rank{dist.get_rank()}.mp4", video_tensor, fps=16)
        if (not dist.is_initialized() or dist.get_rank() == 0) and LOG_GPU_MEMORY:
            log_gpu_memory(f"StreamingTrain-Model: After pipeline.generate_chunk_with_cache", device=self.device, rank=dist.get_rank() if dist.is_initialized() else 0)

        return output, denoised_timestep_from, denoised_timestep_to
    
    def setup_sequence(
        self,
        conditional_dict: Dict,
        unconditional_dict: Dict,
        initial_latent: Optional[torch.Tensor] = None,
        switch_conditional_dict: Optional[Dict] = None,
        switch_frame_index: Optional[int] = None,
        temp_max_length: Optional[int] = None,
        gt_prompts=None,
        gt_mismatch_prompts=None,
        gt_pairs=None,
        mismatch_pairs=None
    ):
        """Set up a new sequence"""
        self._id_anchor = None  # reset the chunk-0 identity anchor for this new sequence
        if (not dist.is_initialized() or dist.get_rank() == 0) and LOG_GPU_MEMORY:
            log_gpu_memory(f"StreamingTrain-Model: Before setup_sequence", device=self.device, rank=dist.get_rank() if dist.is_initialized() else 0)
            
        if DEBUG and (not dist.is_initialized() or dist.get_rank() == 0):
            print(f"[StreamingTrain-Model] Setting up new sequence:")
            print(f"[StreamingTrain-Model] image_or_video_shape={self.image_or_video_shape}")
            print(f"[StreamingTrain-Model] initial_latent shape: {initial_latent.shape if initial_latent is not None else None}")
            print(f"[StreamingTrain-Model] switch_frame_index={switch_frame_index}")

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        batch_size = self.image_or_video_shape[0]
        if self.inference_pipeline.kv_cache1 is None:
            self.inference_pipeline._initialize_kv_cache(
                batch_size=batch_size,
                dtype=self.dtype,
                device=self.device,
            )
            if DEBUG and (not dist.is_initialized() or dist.get_rank() == 0):
                print(f"[StreamingTrain-Model] init kv_cache1: {self.inference_pipeline.kv_cache1[0]['k'].shape}")

        if self.inference_pipeline.crossattn_cache is None:
            self.inference_pipeline._initialize_crossattn_cache(
                batch_size=batch_size,
                dtype=self.dtype,
                device=self.device
            )
            if DEBUG and (not dist.is_initialized() or dist.get_rank() == 0):
                print(f"[StreamingTrain-Model] init crossattn_cache: {self.inference_pipeline.crossattn_cache[0]['k'].shape}")
        
        if self.inference_pipeline.flow_cache is None and self.sink_update_ratio != 0:
            self.inference_pipeline._initialize_flow_cache(
                batch_size=batch_size,
                dtype=self.dtype,
                device=self.device,
                video_shape=self.image_or_video_shape[-2:],
                ratio=self.sink_update_ratio
            )
        
        if self.inference_pipeline.prompt_cache is None and self.use_prompt_cache:
            self.inference_pipeline._initialize_prompt_cache(
                batch_size=batch_size,
                dtype=self.dtype,
                device=self.device,
            )
        
        # Reset state
        self.reset_state()
        self.state["temp_max_length"] = temp_max_length
        
        # if self.possible_max_length is not None:
        #     # Ensure all processes select the same length
        #     if dist.is_initialized():
        #         if dist.get_rank() == 0:
        #             import random
        #             selected_idx = random.randint(0, len(self.possible_max_length) - 1)
        #         else:
        #             selected_idx = 0
        #         selected_idx_tensor = torch.tensor(selected_idx, device=self.device, dtype=torch.int32)
        #         dist.broadcast(selected_idx_tensor, src=0)
        #         selected_idx = selected_idx_tensor.item()
        #     else:
        #         import random
        #         selected_idx = random.randint(0, len(self.possible_max_length) - 1)
            
        #     self.state["temp_max_length"] = self.possible_max_length[selected_idx]
        # else:
        #     self.state["temp_max_length"] = self.max_length
            
        #     if DEBUG and (not dist.is_initialized() or dist.get_rank() == 0):
        #         print(f"[StreamingTrain-Model] Selected temporary max length: {self.state['temp_max_length']} (from {self.possible_max_length})")
        
        # Prepare initial sequence
        if initial_latent is not None:
            self.state["current_length"] = initial_latent.shape[1]
            if DEBUG and (not dist.is_initialized() or dist.get_rank() == 0):
                print(f"[StreamingTrain-Model] Starting with initial_latent, length={self.state['current_length']}")
        else:
            self.state["current_length"] = 0
            if DEBUG and (not dist.is_initialized() or dist.get_rank() == 0):
                print(f"[StreamingTrain-Model] Starting with empty sequence")
        
        # Save conditional information
        self.state["conditional_info"] = {
            "conditional_dict": conditional_dict,
            "unconditional_dict": unconditional_dict,
        }
        
        # DMDSwitch related information
        if switch_conditional_dict is not None and switch_frame_index is not None:
            self.state["conditional_info"]["switch_info"] = {
                "switch_conditional_dict": switch_conditional_dict,
                "switch_frame_index": switch_frame_index,
            }
            if DEBUG and (not dist.is_initialized() or dist.get_rank() == 0):
                print(f"[StreamingTrain-Model] DMDSwitch info saved: switch_frame_index={switch_frame_index}")
        
        # Handle cache updates for initial_latent
        if initial_latent is not None:
            if DEBUG and (not dist.is_initialized() or dist.get_rank() == 0):
                print(f"[StreamingTrain-Model] Initializing cache with initial_latent")
            
            # Use initial latent to update cache
            timestep = torch.zeros([batch_size, initial_latent.shape[1]], device=self.device, dtype=torch.int64)
            with torch.no_grad():
                self.inference_pipeline.generator(
                    noisy_image_or_video=initial_latent,
                    conditional_dict=conditional_dict,
                    timestep=timestep,
                    kv_cache=self.inference_pipeline.kv_cache1,
                    crossattn_cache=self.inference_pipeline.crossattn_cache,
                    current_start=0
                )
            
            if (not dist.is_initialized() or dist.get_rank() == 0) and LOG_GPU_MEMORY:
                log_gpu_memory(f"StreamingTrain-Model: After initial latent processing", device=self.device, rank=dist.get_rank() if dist.is_initialized() else 0)
        else:
            if DEBUG and (not dist.is_initialized() or dist.get_rank() == 0):
                print(f"[StreamingTrain-Model] No initial latent")
        
        # gan part
        if gt_prompts is not None:
            self.state["frames4gan_gt"] = gt_pairs.to(self.device)
            self.state["frames4gan_mismatch"] = mismatch_pairs.to(self.device)
            self.state["frames4gan_gt_prompt"] = gt_prompts
            self.state["frames4gan_mismatch_gt_prompt"] = gt_mismatch_prompts


    def can_generate_more(self) -> bool:
        """Check whether more chunks can be generated"""
        current_length = self.state["current_length"]
        temp_max_length = self.state.get("temp_max_length")
        can_generate = current_length < temp_max_length and (current_length + self.min_new_frame) <= temp_max_length
        
        if DEBUG and (not dist.is_initialized() or dist.get_rank() == 0):
            print(f"[StreamingTrain-Model] can_generate_more: current_length={current_length}, temp_max_length={temp_max_length}, global_max_length={self.max_length}, can_generate={can_generate}")

        return can_generate

    # ---- counterfactual-twin machinery: snapshot/restore everything one chunk mutates ----
    def _twin_snapshot(self):
        """Clone every piece of state a chunk generation mutates (KV / cross-attn / flow / prompt
        caches, fast-weight state, RNG streams, switch flag) so a mem-OFF twin of the same chunk can
        be generated and then fully undone. Tensor clones go to CPU to spare VRAM. Completeness is
        validated empirically by TWIN_SELFTEST=1 (twin with memory ON must equal the real chunk
        bitwise)."""
        import random as _pyrand
        p = self.inference_pipeline
        tensors = []

        def _walk(obj):
            if torch.is_tensor(obj):
                tensors.append((obj, obj.detach().to("cpu", copy=True)))
            elif isinstance(obj, dict):
                for v in obj.values():
                    _walk(v)
            elif isinstance(obj, (list, tuple)):
                for v in obj:
                    _walk(v)

        for c in (getattr(p, "kv_cache1", None), getattr(p, "crossattn_cache", None),
                  getattr(p, "flow_cache", None), getattr(p, "prompt_cache", None)):
            if c is not None:
                _walk(c)
        ttt = None
        ttt_ema = None
        ttt_fifo = None
        if getattr(p, "ttt_cache", None):
            # state tuples are replaced per write (never mutated in place) -> keeping refs suffices
            ttt = [None if getattr(c, "state", None) is None else tuple(c.state) for c in p.ttt_cache]
            # the EMA buffers are mutable state a chunk mutates -> snapshot them too, else the
            # mem-OFF twin's buffer writes leak into the real rollout (breaks the counterfactual). Dict
            # entries are replaced per update (never mutated in place), so a shallow dict() copy suffices.
            ttt_ema = [dict(getattr(c, "ema", {}) or {}) for c in p.ttt_cache]
            # eviction feed: the fifo is appended/popped per write chunk ((idx, tensor) tuples,
            # tensors never mutated in place) -> a shallow list() copy suffices, like the EMA dicts.
            ttt_fifo = [list(getattr(c, "fifo", []) or []) for c in p.ttt_cache]
        return {
            "tensors": tensors, "ttt": ttt, "ttt_ema": ttt_ema, "ttt_fifo": ttt_fifo,
            "has_switched": self.state.get("has_switched", False),
            "rng_cpu": torch.get_rng_state(),
            "rng_cuda": torch.cuda.get_rng_state(self.device),
            "rng_py": _pyrand.getstate(),
        }

    def _twin_restore(self, snap):
        import random as _pyrand
        p = self.inference_pipeline
        with torch.no_grad():
            for ref, saved in snap["tensors"]:
                ref.copy_(saved.to(ref.device))
        if snap["ttt"] is not None and getattr(p, "ttt_cache", None):
            for c, s in zip(p.ttt_cache, snap["ttt"]):
                c.state = s
        if snap.get("ttt_ema") is not None and getattr(p, "ttt_cache", None):
            for c, e in zip(p.ttt_cache, snap["ttt_ema"]):
                c.ema = dict(e)   # fresh dict so later mutations do not touch the snapshot
        if snap.get("ttt_fifo") is not None and getattr(p, "ttt_cache", None):
            for c, f in zip(p.ttt_cache, snap["ttt_fifo"]):
                c.fifo = list(f)  # fresh list so later appends/pops do not touch the snapshot
        self.state["has_switched"] = snap["has_switched"]
        torch.set_rng_state(snap["rng_cpu"])
        torch.cuda.set_rng_state(snap["rng_cuda"], self.device)
        _pyrand.setstate(snap["rng_py"])

    def generate_next_chunk(self, requires_grad: bool = True, detach_state: bool = True, keep_last_write_grad: bool = False, make_twin: bool = False) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Generate the next chunk, supporting overlap to ensure temporal continuity.

        Args:
            requires_grad: whether gradients are required

        Returns:
            generated_chunk: the full generated chunk (including overlap frames)
            info: generation info (including timestep, gradient_mask, etc.)
        """
        if DEBUG and (not dist.is_initialized() or dist.get_rank() == 0):
            print(f"[StreamingTrain-Model] generate_next_chunk called: requires_grad={requires_grad}")
            
        # DEBUG: inspect the generator model gradient state
        if DEBUG and (not dist.is_initialized() or dist.get_rank() == 0):
            gen_training_mode = self.generator.training
            gen_params_requiring_grad = sum(1 for p in self.generator.parameters() if p.requires_grad)
            gen_params_total = sum(1 for p in self.generator.parameters())
            print(f"[DEBUG-SeqModel] Generator training mode: {gen_training_mode}")
            print(f"[DEBUG-SeqModel] Generator params requiring grad: {gen_params_requiring_grad}/{gen_params_total}")
            
        if not self.can_generate_more():
            raise ValueError("Cannot generate more chunks")
        
        current_length = self.state["current_length"]
        batch_size = self.image_or_video_shape[0]
        
        if DEBUG and (not dist.is_initialized() or dist.get_rank() == 0):
            print(f"[StreamingTrain-Model] Generating chunk: current_length={current_length}")
        
        # Check if previous_frames can be used for overlap and auto-compute overlap frame count
        previous_frames = self.state.get("previous_frames")
        if previous_frames is not None:
            # Randomly select number of new frames (min=min_new_frame, max=chunk_size, step=3)
            max_new_frames = min(self.state["temp_max_length"] - current_length + 1, self.chunk_size)
            possible_new_frames = list(range(self.min_new_frame, max_new_frames, 3))  # TODO这里为什么要设定随机生成的帧数，是为了训练任意长度视频的生成吗？不过这里应该只有self.min_new_frame=18这一个选项
            
            # Ensure all processes choose the same random value
            if dist.is_initialized():
                if dist.get_rank() == 0:
                    import random
                    selected_idx = random.randint(0, len(possible_new_frames) - 1)
                else:
                    selected_idx = 0
                selected_idx_tensor = torch.tensor(selected_idx, device=self.device, dtype=torch.int32)
                dist.broadcast(selected_idx_tensor, src=0)
                selected_idx = selected_idx_tensor.item()
            else:
                import random
                selected_idx = random.randint(0, len(possible_new_frames) - 1)
            
            new_frames_to_generate = possible_new_frames[selected_idx]  # 随机挑选生成的帧数

            # Auto-compute required overlap frames to ensure the final chunk has 21 frames
            overlap_frames = self.chunk_size - new_frames_to_generate  # 计算overlap帧数，不计算sink的梯度，后续生成gradient_mask时使用
            if overlap_frames > 0 and overlap_frames <= previous_frames.shape[1]:
                overlap_frames_to_use = overlap_frames
            else:
                # If overlap can't be used, generate a full chunk_size without overlap
                overlap_frames_to_use = 0
                new_frames_to_generate = self.chunk_size
            
            if DEBUG and (not dist.is_initialized() or dist.get_rank() == 0):
                print(f"[StreamingTrain-Model] With auto overlap: generating {new_frames_to_generate} new frames, reusing {overlap_frames_to_use} overlap frames")
        else:  # 第一次前向推理时
            overlap_frames_to_use = 0
            new_frames_to_generate = self.chunk_size
            if DEBUG and (not dist.is_initialized() or dist.get_rank() == 0):
                print(f"[StreamingTrain-Model] First chunk: generating {new_frames_to_generate} frames (no overlap)")
        
        if DEBUG and (not dist.is_initialized() or dist.get_rank() == 0):
            print(f"[StreamingTrain-Model] Random frame selection: selected={new_frames_to_generate}")
            print(f"[StreamingTrain-Model] Auto overlap calculation: overlap_frames={overlap_frames_to_use}")
        
        # Sample noise for new frames
        noise_chunk = torch.randn(
            [batch_size, new_frames_to_generate, *self.image_or_video_shape[2:]],
            device=self.device,
            dtype=self.dtype
        )

        # ---- counterfactual mem-OFF twin (advantage objective) ----
        # Same pre-chunk state, same noise, same RNG stream, injection zeroed -> the only difference
        # vs the real chunk is the memory's injection during this chunk. Generated first, then every
        # mutated state is restored, so the real rollout below is bit-identical to a no-twin run.
        # Self-check: TWIN_SELFTEST=1 keeps the memory on in the twin and
        # the code below asserts twin == real, which validates snapshot completeness + RNG pairing.
        twin_new_frames = None
        if make_twin and self.state.get("previous_frames") is not None:
            _selftest = os.environ.get("TWIN_SELFTEST", "0") == "1"
            _snap = self._twin_snapshot()
            _prev_mz = os.environ.get("MEM_ZERO_ALL")
            if not _selftest:
                os.environ["MEM_ZERO_ALL"] = "1"
            try:
                with torch.no_grad():
                    twin_new_frames, _, _ = self._generate_chunk(
                        noise_chunk=noise_chunk,
                        chunk_start_frame=current_length,
                        requires_grad=False,
                        detach_state=True,
                        keep_last_write_grad=False,
                    )
                twin_new_frames = twin_new_frames.detach()
            finally:
                if _prev_mz is None:
                    os.environ.pop("MEM_ZERO_ALL", None)
                else:
                    os.environ["MEM_ZERO_ALL"] = _prev_mz
                self._twin_restore(_snap)
                del _snap

        # Generate new frames - note chunk_start_frame should consider overlap
        generated_new_frames, denoised_timestep_from, denoised_timestep_to = self._generate_chunk(
            noise_chunk=noise_chunk,
            chunk_start_frame=current_length,
            requires_grad=requires_grad,
            detach_state=detach_state,
            keep_last_write_grad=keep_last_write_grad,
        )

        if twin_new_frames is not None and os.environ.get("TWIN_SELFTEST", "0") == "1":
            _diff = (generated_new_frames.detach() - twin_new_frames).abs().max().item()
            print(f"[E3-TWIN-SELFTEST] rank{dist.get_rank() if dist.is_initialized() else 0} "
                  f"max|real - twin(mem ON)| = {_diff:.3e} (MUST be 0.0)", flush=True)

        # Build the full chunk for loss computation
        if previous_frames is not None:
            # Concatenate specified overlap frames and newly generated frames
            full_chunk = torch.cat([previous_frames, generated_new_frames], dim=1)

            ###### 取出两帧用于作为gan的输入 #####
            idx1 = torch.randint(low=0, high=previous_frames.shape[1], size=(1,))
            idx2 = torch.randint(low=0, high=generated_new_frames.shape[1], size=(1,))
            frame_prev = previous_frames[:, idx1: idx1+1]
            frame_gen = generated_new_frames[:, idx2: idx2+1]

            if dist.get_rank() == 0:
                print("[StreamingTrain-Model] select frame_prev id is: {}. frame_gen id is {}".format(idx1, idx2))

        else:
            full_chunk = generated_new_frames

            ###### 取出两帧用于作为gan的输入 #####
            high = generated_new_frames.shape[1]
            idx = torch.randperm(high)[:2]          # 随机选取两个不重复的索引
            sorted_idx, _ = torch.sort(idx)         # 排序
            idx1, idx2 = sorted_idx.tolist()        # 变成 Python 整数
            frame_prev = generated_new_frames[:, idx1: idx1+1]
            frame_gen = generated_new_frames[:, idx2: idx2+1]
        
            if dist.get_rank() == 0:
                print("[StreamingTrain-Model] select frame id is: {} and {}. frame_prev.shape is {}".format(idx1, idx2, frame_prev.shape))

        self.state["frames4gan"] = torch.cat([frame_prev, frame_gen], dim=0)
        
        # Update state - save the last 21 frames as previous_frames for the next chunk
        # The frames saved here should be those before _process_first_frame_encoding
        frames_to_save = full_chunk.detach().clone()[:, -self.chunk_size:, ...]
        if DEBUG and (not dist.is_initialized() or dist.get_rank() == 0):
            print(f"[StreamingTrain-Model] Saved last {frames_to_save.shape[1]} frames as previous_frames")
        

        # Process first-frame encoding (if there is overlap)
        if previous_frames is not None:
            full_chunk = self._process_first_frame_encoding(full_chunk)  # 重新处理最后一个chunk的第一帧。重新组织最后一个chunk，用于计算梯度
        
        if previous_frames is not None:
            # Create gradient_mask: only newly generated frames require gradients
            gradient_mask = torch.zeros_like(full_chunk, dtype=torch.bool)
            # Overlap frames do not compute gradients; new frames do
            gradient_mask[:, overlap_frames_to_use:overlap_frames_to_use + new_frames_to_generate, ...] = True
            if DEBUG and (not dist.is_initialized() or dist.get_rank() == 0):
                print(f"[StreamingTrain-Model] Built chunk with auto overlap: shape={full_chunk.shape}")
                print(f"[StreamingTrain-Model] Gradient mask: {new_frames_to_generate} frames will have gradients out of {full_chunk.shape[1]}")
        else:
            # For the first chunk, all frames are newly generated
            gradient_mask = torch.ones_like(full_chunk, dtype=torch.bool)

        self.state["current_length"] += new_frames_to_generate  # Increase only by newly generated frames
        
        if DEBUG and (not dist.is_initialized() or dist.get_rank() == 0):
            print(f"[StreamingTrain-Model] Updated state: current_length={self.state['current_length']}")
            if self.state["previous_frames"] is not None:
                print(f"[StreamingTrain-Model] Saved {self.state['previous_frames'].shape[1]} frames as previous_frames for next chunk")

        self.state["previous_frames"] = frames_to_save

        # cache the identity anchor on the first chunk of the sequence, here in the rollout, so it is
        # set whether a generator or a critic step rolls frame 0 (the generator-loss path alone never
        # sees frame 0 under the windowed schedule).
        if previous_frames is None:
            self._set_id_anchor(full_chunk)

        # Return info
        info = {
            "denoised_timestep_from": denoised_timestep_from,
            "denoised_timestep_to": denoised_timestep_to,
            "chunk_start_frame": current_length,  # Start frame position in the full sequence
            "chunk_frames": full_chunk.shape[1],  # Chunk size used for loss (fixed 21 frames)
            "new_frames_generated": new_frames_to_generate,
            "current_length": self.state["current_length"],
            "gradient_mask": gradient_mask,  # Mask frames that do not require gradients for loss computation
            "overlap_frames_used": overlap_frames_to_use,
            "twin_new_frames": twin_new_frames,  # mem-OFF counterfactual of the new frames, or None
        }
        if (not dist.is_initialized() or dist.get_rank() == 0):
            print(f"[StreamingTrain-Model] current_training_chunk: ({self.state['current_length'] - new_frames_to_generate} -> {self.state['current_length']})/{self.state['temp_max_length']}")
        return full_chunk, info
    
    # ---- frame-level identity anchor + temporal-variance floor ----
    _DINO_MEAN = (0.485, 0.456, 0.406)
    _DINO_STD = (0.229, 0.224, 0.225)

    def _get_dino(self):
        """Lazily load a frozen DINOv2-ViT-B/14 (offline from the torch.hub cache)."""
        if self._dino is None:
            hub = os.path.expanduser("~/.cache/torch/hub")
            torch.hub.set_dir(hub)
            local = os.path.join(hub, "facebookresearch_dinov2_main")
            m = torch.hub.load(local, "dinov2_vitb14", source="local", pretrained=True)
            m.eval().to(self.device)
            for p in m.parameters():
                p.requires_grad_(False)
            self._dino = m
            if (not dist.is_initialized() or dist.get_rank() == 0):
                print("[StreamingTrain-Model] loaded frozen DINOv2-vitb14 for identity loss")
        return self._dino

    def _dino_embed_pixels(self, pixels):
        """Pixels [B,T,C,H,W] in [-1,1] -> L2-normalized DINO CLS embeddings [B*T, D] (fp32)."""
        dino = self._get_dino()
        b, t, c, h, w = pixels.shape
        x = pixels.reshape(b * t, c, h, w).float()
        x = (x * 0.5 + 0.5).clamp(0, 1)                       # [-1,1] -> [0,1]
        x = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)
        mean = torch.tensor(self._DINO_MEAN, device=x.device).view(1, 3, 1, 1)
        std = torch.tensor(self._DINO_STD, device=x.device).view(1, 3, 1, 1)
        x = (x - mean) / std
        f = dino(x)                                           # [B*T, D] CLS embedding
        return F.normalize(f, dim=-1)

    # -----------------------------------------------------------------------------------------
    # The extra loss terms, and why the runs this repo presents have all of them off.
    #
    # Eight were built, in three generations, each closing the channel the previous one left
    # open. Every one is a hinge with a deadband rather than a maximiser: an unbounded "be more
    # active" term is always satisfied by the cheapest route, which is noise or a constant bias.
    #
    #   id, tv            identity to the chunk-0 anchor, and a floor on latent motion. Used by
    #                     the first generation of runs, then dropped: the identity term is a
    #                     freeze ally, since the cheapest way to keep a face is to stop moving,
    #                     and the motion floor is subsumed by the mm terms below.
    #   dnh, act,         do-no-harm on identity; a floor on how much the injection changes the
    #   mm_rel, mm_ext    frame; a two-sided motion match to the twin; and an absolute floor
    #                     against the unmodified model's own motion curve, which is what catches
    #                     a twin that froze too. Used by the second generation of runs.
    #   flick,            a second-temporal-difference hinge against the twin, plus two switches,
    #   motion_lag2,      added after the model learned to counterfeit motion with frame-rate
    #   dose_tmean_remove shimmer and to satisfy the activity floor with a static overlay. Each
    #                     switch closes one of those two routes.
    #   adv               a clamped reward for beating the twin on identity. Written, bounded,
    #                     gated on motion validity, and never enabled in any run of this project:
    #                     it was judged a freeze risk. adv_floor_frac is still used, by mm_ext.
    #
    # The four runs behind the README figure turn all of this off and keep DMD alone, because the
    # stack fought the distillation and swept the model between freezing and flickering. With
    # adv, dnh, act and flick at 0, _twin_on is false and the trainer never even builds the twin
    # (make_twin in trainer/distillation.py), so none of the code below runs.
    # -----------------------------------------------------------------------------------------
    def _set_id_anchor(self, chunk):
        """Cache the chunk-0 subject identity anchor (no grad), called from the rollout on the first
        chunk. Uses the head frames (highest prompt fidelity, last to degrade under collapse) so the
        anchor is a real subject; supervised chunks compare their tail frames, so id_cos=1.0 becomes a
        genuine static-video alarm instead of a vacuous garbage-vs-garbage fixed point."""
        if self.id_loss_weight <= 0.0 and not self._twin_anchor_on:
            return
        k = max(1, int(self.id_decode_frames))
        with torch.no_grad():
            px = self.base_model.vae.decode_to_pixel(chunk[:, :k, ...], use_cache=False)  # head, pre-collapse
            emb = self._dino_embed_pixels(px)
            self._id_anchor = F.normalize(emb.mean(0, keepdim=True), dim=-1).detach()

    def _compute_identity_tv_loss(self, chunk, chunk_start_frame, gradient_mask):
        """Frame-level identity anchor loss + temporal-variance floor. Returns (id_loss, tv_loss, log).
        All log values are 0-dim tensors present on every chunk (chunk-0 defaults) so merge_dict_list
        (which keys off the first chunk's dict) keeps them and stacks them across chunks."""
        device = chunk.device
        id_loss = torch.zeros((), device=device)
        tv_loss = torch.zeros((), device=device)
        # consistent keys every chunk (chunk-0 identity is perfect by construction -> cos=1, loss=0)
        log = {
            "e1_latent_motion": torch.zeros((), device=device),
            "e1_tv_loss": torch.zeros((), device=device),
            "e1_id_cos": torch.ones((), device=device),
            "e1_id_loss": torch.zeros((), device=device),
        }

        # temporal-variance floor (anti-freeze): hinge on adjacent-latent motion; penalize only if frozen
        if self.tv_loss_weight > 0.0 and chunk.shape[1] >= 2:
            motion = (chunk[:, 1:] - chunk[:, :-1]).abs().mean()
            tv_loss = F.relu(self.tv_motion_floor - motion)
            log["e1_latent_motion"] = motion.detach()
            log["e1_tv_loss"] = tv_loss.detach()

        # frame-level identity: decode a few frames -> DINO -> cosine to the chunk-0 anchor, on later
        # chunks only, with grad through decode -> gate -> read -> write (anchor cached in the rollout).
        # Skip after a real prompt switch: the anchor is the pre-switch subject, penalizing the new one.
        if (self.id_loss_weight > 0.0 and chunk_start_frame > 0 and self._id_anchor is not None
                and not self.state.get("has_switched", False)):
            k = max(1, int(self.id_decode_frames))
            px = self.base_model.vae.decode_to_pixel(chunk[:, -k:, ...], use_cache=False)
            emb = self._dino_embed_pixels(px)
            cos = (emb * self._id_anchor).sum(-1)            # [k]
            id_loss = (1.0 - cos).mean()
            log["e1_id_cos"] = cos.mean().detach()
            log["e1_id_loss"] = id_loss.detach()
        return id_loss, tv_loss, log

    # ---- counterfactual, motion-gated, clamped identity advantage over the mem-OFF twin ----
    def _motion_target_index(self, chunk_start_frame):
        """chunk_start (0, 21, 39, 57, ...) -> index (0, 1, 2, 3, ...) into the motion-target curve."""
        if chunk_start_frame <= 0:
            return 0
        return (chunk_start_frame - self.chunk_size) // self.min_new_frame + 1

    def _get_motion_target(self, chunk_start_frame):
        """Per-prompt (fallback: _global) stock-AF latent-motion target for this chunk position.
        Keys = sha1(prompt.strip())[:16]; the trainer stores the prompt in state["current_prompt"]."""
        if self._motion_targets is None:
            return None
        import hashlib
        p = self.state.get("current_prompt")
        key = hashlib.sha1(p.strip().encode()).hexdigest()[:16] if isinstance(p, str) else None
        curve = self._motion_targets.get(key) if key else None
        if curve is None:
            curve = self._motion_targets.get("_global")
        if not curve:
            return None
        i = min(self._motion_target_index(chunk_start_frame), len(curve) - 1)
        return torch.tensor(float(curve[i]), device=self.device)

    def _compute_advantage_loss(self, chunk, chunk_info):
        """Twin objective: clean + active + do-no-harm + dynamic, via the mem-OFF counterfactual twin.
            L = w_dnh * relu(cos_CF - cos_ON - deadband)            (do-no-harm identity hinge)
              + w_act * relu(act_floor - D_ac)                      (activity floor: not inert)
              + w_rel * (log m_ON - log m_CF)^2                     (motion two-sided vs twin)
              + w_ext * relu(log(frac * m_stock) - log m_ON)^2      (anti-co-freeze vs stock curve)
              [ - w_adv * g * clamp(cos_ON - cos_CF, -neg, +pos)    bonus mode only, default OFF ]
        with D_ac = RMS((c_ON - c_CF) - per-channel-mean) / RMS_ac(c_CF): the injection's causal
        dose, DC/global-style component removed.
        Anti-degeneracy structure (each blocked in-construction, not by tuning):
        - The twin is no-grad and shares history + KV + noise: the only differential lever is the
          memory injection during this chunk; nothing can be earned by degrading the baseline, and
          the paired design cancels prompt/noise variance.
        - nothing pays for identity self-similarity (no id_loss, reward off): freeze has
          no payer, and mm_rel/mm_ext punish it; a frozen chunk also zeroes the dnh hinge trivially,
          so freeze buys nothing that neutrality does not already buy, at strictly higher mm cost.
        - The activity floor is a RELU: zero gradient once D_ac >= act_floor -> a floor, never a
          maximizer -> no runaway (contrast: rel_norm alone pins a fixed dose at every chunk;
          here the floor only forbids ~zero dose, the envelope bounds it above, and D_ac's mean
          removal forbids satisfying it with the resurrected DC bias).
        - Under the box {identity not-worse, motion matched, DMD on-manifold, dose >= floor}, the
          cheapest admissible behavior is an active-but-neutral content/detail difference: active by
          construction, harmless by constraint. Wallpaper (static texture watermark) is the residual
          gaming channel -> judged at eval (visual + dcfrac), escalation = temporal-mean removal in D_ac.
        - Decoded frames are position-randomized in the tail so the read cannot position-lock.
        Returns (loss, log). All log keys present on every chunk (merge_dict_list keys off chunk 0)."""
        device = chunk.device
        _z = lambda: torch.zeros((), device=device)
        log = {
            "twin_adv_loss": _z(), "twin_adv_raw": _z(), "twin_gate": _z(),
            "twin_dnh": _z(), "twin_act": _z(), "twin_dose": _z(),
            "twin_mm_rel": _z(), "twin_mm_ext": _z(), "twin_flick": _z(),
            "twin_cos_on": torch.ones((), device=device), "twin_cos_cf": torch.ones((), device=device),
            "twin_motion_on": _z(), "twin_motion_cf": _z(), "twin_motion_tgt": _z(),
            "twin_active": _z(), "twin_oom_skip": _z(),
        }
        twin = chunk_info.get("twin_new_frames")
        chunk_start_frame = chunk_info["chunk_start_frame"]
        if twin is None or chunk_start_frame <= 0 or self._id_anchor is None:
            return _z(), log
        eps = 1e-6
        ov = int(chunk_info.get("overlap_frames_used", 0))
        new_real = chunk[:, ov:]                        # newly generated frames only (grad); twin spans
        # the same global frames and is decode-comparable (no first-frame re-encode contamination)
        # lag-2 (frame-rate-blind) motion statistic: frame-rate alternation contributes ~0 to a
        # lag-2 delta while smooth motion contributes fully, so the memory cannot counterfeit
        # motion credit with shimmer. Falls back to the first difference when the flag is off or the
        # chunk is too short for a lag-2 window. (motion_targets.json must be regenerated to match.)
        if getattr(self, "motion_lag2", False) and new_real.shape[1] >= 3:
            m_on = (new_real[:, 2:] - new_real[:, :-2]).abs().mean() * 0.5
            with torch.no_grad():
                m_cf = (twin[:, 2:] - twin[:, :-2]).abs().mean() * 0.5
        else:
            m_on = (new_real[:, 1:] - new_real[:, :-1]).abs().mean()
            with torch.no_grad():
                m_cf = (twin[:, 1:] - twin[:, :-1]).abs().mean()
        m_tgt = self._get_motion_target(chunk_start_frame)
        # (1) two-sided motion match to the twin: the injection may not change motion either way
        mm_rel = (torch.log(m_on + eps) - torch.log(m_cf + eps)) ** 2
        # (2) absolute anti-co-freeze floor vs the external stock-AF curve (catches a co-frozen twin)
        if m_tgt is not None:
            mm_ext = F.relu(torch.log(self.adv_floor_frac * m_tgt + eps) - torch.log(m_on + eps)) ** 2
        else:
            mm_ext = _z()
        # (3) activity floor on the injection's causal AC-dose (be measurably non-inert)
        d = new_real - twin                                             # [B, T, C, H, W]
        d_ac = d - d.mean(dim=(1, 3, 4), keepdim=True)                  # remove global per-channel bias
        with torch.no_grad():
            cf_ac = twin - twin.mean(dim=(1, 3, 4), keepdim=True)
            if getattr(self, "dose_tmean_remove", False):
                cf_ac = cf_ac - cf_ac.mean(dim=1, keepdim=True)         # match the per-site temporal removal
            cf_rms = cf_ac.float().pow(2).mean().sqrt().clamp_min(eps)
        # also remove the per-site temporal mean so a static overlay (wallpaper) earns zero dose;
        # only a slow, state-evolving, spatially-structured change can satisfy the floor.
        if getattr(self, "dose_tmean_remove", False):
            d_ac = d_ac - d_ac.mean(dim=1, keepdim=True)                # per (B,C,H,W) site temporal mean
        dose = d_ac.float().pow(2).mean().sqrt() / cf_rms
        act = F.relu(self.act_floor - dose)
        # (3b) anti-flicker tripwire: penalize the per-site second-temporal-difference roughness the
        # injection adds beyond the mem-OFF twin. One-sided hinge, paired (real acceleration in the twin
        # is priced in), relative to the twin's own roughness. D2 (not D1) maximizes on frame-rate
        # alternation while smooth constant-velocity motion contributes ~0. Latent-space (covers all new
        # frames, free); the EMA low-pass should make this structurally ~0 -> a rise flags conversion of
        # slow injection into fast output downstream (a reportable finding), hence "tripwire" weight.
        if self.flick_weight > 0.0 and new_real.shape[1] >= 3:
            d2_on = new_real[:, 2:] - 2 * new_real[:, 1:-1] + new_real[:, :-2]
            r_on = d2_on.float().pow(2).mean(dim=2).clamp_min(1e-12).sqrt()               # [B, T-2, H, W] per-site
            with torch.no_grad():
                d2_cf = twin[:, 2:] - 2 * twin[:, 1:-1] + twin[:, :-2]
                r_cf = d2_cf.float().pow(2).mean(dim=2).clamp_min(1e-12).sqrt()
                r_ref = r_cf.mean().clamp_min(eps)
            flick = F.relu(r_on - r_cf - self.flick_deadband * r_ref).mean() / r_ref
        else:
            flick = _z()
        # (4) paired identity terms on decoded frames, position-randomized in the tail
        k = max(1, int(self.id_decode_frames))
        span = min(6, new_real.shape[1])
        off = int(torch.randint(0, max(1, span - k + 1), (1,)).item())
        s0, s1 = new_real.shape[1] - off - k, new_real.shape[1] - off
        px_on = self.base_model.vae.decode_to_pixel(new_real[:, s0:s1], use_cache=False)
        cos_on = (self._dino_embed_pixels(px_on) * self._id_anchor).sum(-1).mean()
        with torch.no_grad():
            px_cf = self.base_model.vae.decode_to_pixel(twin[:, s0:s1], use_cache=False)
            cos_cf = (self._dino_embed_pixels(px_cf) * self._id_anchor).sum(-1).mean()
        A = cos_on - cos_cf
        # do-no-harm hinge: fires only when the injection makes identity worse than mem-OFF (beyond
        # the noise deadband); no gate needed, under freeze it is zero, and the mm terms own freeze.
        dnh = F.relu(-A - self.dnh_deadband)
        # bonus-mode clamped reward (off by default): motion-validity gate detached, never a lever
        with torch.no_grad():
            ref = m_tgt if m_tgt is not None else m_cf
            gate = torch.exp(-((torch.log(m_on.detach() + eps) - torch.log(ref + eps)) ** 2)
                             / (2.0 * self.adv_gate_sigma ** 2))
        adv = gate * torch.clamp(A, min=-self.adv_clamp_neg, max=self.adv_clamp_pos)
        loss = (self.dnh_weight * dnh
                + self.act_weight * act
                + self.mm_rel_weight * mm_rel
                + self.mm_ext_weight * mm_ext
                + self.flick_weight * flick
                + self.adv_loss_weight * (-adv))
        log.update({
            "twin_adv_loss": (self.adv_loss_weight * (-adv)).detach(),
            "twin_adv_raw": A.detach(), "twin_gate": gate.detach(),
            "twin_dnh": dnh.detach(), "twin_act": act.detach(), "twin_dose": dose.detach(),
            "twin_mm_rel": mm_rel.detach(), "twin_mm_ext": mm_ext.detach(),
            "twin_flick": flick.detach(),
            "twin_cos_on": cos_on.detach(), "twin_cos_cf": cos_cf.detach(),
            "twin_motion_on": m_on.detach(), "twin_motion_cf": m_cf.detach(),
            "twin_motion_tgt": (m_tgt.detach() if m_tgt is not None else m_cf.detach()),
            "twin_active": torch.ones((), device=device),
        })
        return loss, log

    def compute_generator_loss(self,
        chunk: torch.Tensor,
        chunk_info: Dict[str, Any]
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Compute the generator loss.

        Args:
            chunk: generated chunk
            chunk_info: chunk metadata

        Returns:
            loss: loss value
            log_dict: log dictionary
        """
        _t_loss_start = time.time()
        if (not dist.is_initialized() or dist.get_rank() == 0) and LOG_GPU_MEMORY:
            log_gpu_memory(f"StreamingTrain-Model: Before compute generator loss", device=self.device, rank=dist.get_rank() if dist.is_initialized() else 0)

        # Fetch conditional_dict for loss computation
        chunk_start_frame = chunk_info["chunk_start_frame"]
        conditional_dict = self._get_current_conditional_dict(chunk_start_frame)
        unconditional_dict = self.state["conditional_info"]["unconditional_dict"]
        
        # Fetch gradient_mask to compute loss only on newly generated frames
        gradient_mask = chunk_info.get("gradient_mask", None)
        
        if DEBUG and (not dist.is_initialized() or dist.get_rank() == 0):
            print(f"[StreamingTrain-Model] Using conditional_dict and unconditional_dict for loss calculation at frame {chunk_start_frame}")
        
        if not self.use_decouple_dmd:
            # Compute DMD loss
            dmd_loss, dmd_log_dict = self.base_model.compute_distribution_matching_loss(
                image_or_video=chunk,
                conditional_dict=conditional_dict,
                unconditional_dict=unconditional_dict,
                gradient_mask=gradient_mask,  # Pass gradient_mask
                denoised_timestep_from=chunk_info["denoised_timestep_from"],
                denoised_timestep_to=chunk_info["denoised_timestep_to"]
            )
        else:
            dmd_loss, dmd_log_dict = self.base_model.compute_distribution_matching_loss_decouple(
                image_or_video=chunk,
                conditional_dict=conditional_dict,
                unconditional_dict=unconditional_dict,
                gradient_mask=gradient_mask,  # Pass gradient_mask
                denoised_timestep_from=chunk_info["denoised_timestep_from"],
                denoised_timestep_to=chunk_info["denoised_timestep_to"]
            )
        
        if (not dist.is_initialized() or dist.get_rank() == 0) and LOG_GPU_MEMORY:
            log_gpu_memory(f"StreamingTrain-Model: After DMD loss computation", device=self.device, rank=dist.get_rank() if dist.is_initialized() else 0)
        
        # Update log dict
        dmd_log_dict.update({
            "loss_time": time.time() - _t_loss_start,
            "new_frames_supervised": chunk_info.get("new_frames_generated", chunk.shape[1]),
        })

        ###### identity + temporal-variance objective ######
        if self.id_loss_weight > 0.0 or self.tv_loss_weight > 0.0:
            try:
                id_loss, tv_loss, e1_log = self._compute_identity_tv_loss(
                    chunk, chunk_start_frame, gradient_mask)
                dmd_loss = dmd_loss + self.id_loss_weight * id_loss + self.tv_loss_weight * tv_loss
                dmd_log_dict.update(e1_log)
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                # keep the log keys present so merge_dict_list (keyed off chunk 0) does not raise
                dmd_log_dict.update({
                    "e1_latent_motion": torch.zeros((), device=self.device),
                    "e1_tv_loss": torch.zeros((), device=self.device),
                    "e1_id_cos": torch.ones((), device=self.device),
                    "e1_id_loss": torch.zeros((), device=self.device),
                })
                if (not dist.is_initialized() or dist.get_rank() == 0):
                    print("[StreamingTrain-Model] identity/tv loss OOM, skipped this step")

        ###### counterfactual objective (do-no-harm + activity floor + motion match) ######
        if self._twin_on:
            try:
                adv_loss, twin_log = self._compute_advantage_loss(chunk, chunk_info)
                dmd_loss = dmd_loss + adv_loss
                dmd_log_dict.update(twin_log)
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                # keep the log keys present so merge_dict_list (keyed off chunk 0) does not raise;
                # twin_oom_skip=1 makes silent skipping visible in wandb
                dmd_log_dict.update({
                    "twin_adv_loss": torch.zeros((), device=self.device),
                    "twin_adv_raw": torch.zeros((), device=self.device),
                    "twin_gate": torch.zeros((), device=self.device),
                    "twin_dnh": torch.zeros((), device=self.device),
                    "twin_act": torch.zeros((), device=self.device),
                    "twin_dose": torch.zeros((), device=self.device),
                    "twin_mm_rel": torch.zeros((), device=self.device),
                    "twin_mm_ext": torch.zeros((), device=self.device),
                    "twin_flick": torch.zeros((), device=self.device),
                    "twin_cos_on": torch.ones((), device=self.device),
                    "twin_cos_cf": torch.ones((), device=self.device),
                    "twin_motion_on": torch.zeros((), device=self.device),
                    "twin_motion_cf": torch.zeros((), device=self.device),
                    "twin_motion_tgt": torch.zeros((), device=self.device),
                    "twin_active": torch.zeros((), device=self.device),
                    "twin_oom_skip": torch.ones((), device=self.device),
                })
                if (not dist.is_initialized() or dist.get_rank() == 0):
                    print("[StreamingTrain-Model] advantage loss OOM, skipped this chunk")

        ###### gan process #####
        if self.use_gan:
            generator_cls_loss = self.compute_generator_cls_loss(
                fake_image=self.state["frames4gan"], 
                text_embedding=conditional_dict)
            dmd_loss += generator_cls_loss * 1e-3
            dmd_log_dict.update({"generator_cls_loss": generator_cls_loss})
        
        return dmd_loss, dmd_log_dict
    
    def _clear_cache_gradients(self):
        """
        Clear possible gradient references in KV cache and cross-attention cache.
        This is important for preventing memory leaks, especially before critic training.
        """
        if DEBUG and (not dist.is_initialized() or dist.get_rank() == 0):
            print(f"[StreamingTrain-Model] Clearing cache gradients")
            
        # Clear gradient refs in KV cache
        if hasattr(self.inference_pipeline, 'kv_cache1') and self.inference_pipeline.kv_cache1 is not None:
            for cache_block in self.inference_pipeline.kv_cache1:
                if 'k' in cache_block and cache_block['k'].requires_grad:
                    cache_block['k'] = cache_block['k'].detach()
                if 'v' in cache_block and cache_block['v'].requires_grad:
                    cache_block['v'] = cache_block['v'].detach()
        
        # Clear gradient refs in cross-attention cache        
        if hasattr(self.inference_pipeline, 'crossattn_cache') and self.inference_pipeline.crossattn_cache is not None:
            for cache_block in self.inference_pipeline.crossattn_cache:
                if 'k' in cache_block and cache_block['k'].requires_grad:
                    cache_block['k'] = cache_block['k'].detach()
                if 'v' in cache_block and cache_block['v'].requires_grad:
                    cache_block['v'] = cache_block['v'].detach()
        
        if DEBUG and (not dist.is_initialized() or dist.get_rank() == 0):
            print(f"[StreamingTrain-Model] Cache gradients cleared")

    def compute_critic_loss(self, chunk: torch.Tensor, chunk_info: Dict[str, Any]) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Compute critic loss.

        Args:
            chunk: generated chunk
            chunk_info: chunk metadata

        Returns:
            loss: loss value
            log_dict: log dictionary
        """
        _t_loss_start = time.time()
        if (not dist.is_initialized() or dist.get_rank() == 0) and LOG_GPU_MEMORY:
            log_gpu_memory(f"StreamingTrain-Model: Before compute critic loss", device=self.device, rank=dist.get_rank() if dist.is_initialized() else 0)
            
        if DEBUG and (not dist.is_initialized() or dist.get_rank() == 0):
            print(f"[StreamingTrain-Model] compute_critic_loss: chunk_shape={chunk.shape}")
            for k, v in chunk_info.items():
                if k == "gradient_mask":
                    print(f"[StreamingTrain-Model] chunk_info {k}: {v[0, :, 0, 0, 0]}")
                else:
                    print(f"[StreamingTrain-Model] chunk_info {k}: {v}")
            print(f"[StreamingTrain-Model] chunk requires_grad: {chunk.requires_grad}")
            
        # Critical fix: ensure chunk has no gradient connections
        if chunk.requires_grad:
            chunk = chunk.detach()

        # Critical fix: clear gradient references inside caches
        self._clear_cache_gradients()
        
        # Force clear CUDA cache to ensure previous graphs are released
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        if (not dist.is_initialized() or dist.get_rank() == 0) and LOG_GPU_MEMORY:
            log_gpu_memory(f"StreamingTrain-Model: After chunk detachment and cache cleanup", device=self.device, rank=dist.get_rank() if dist.is_initialized() else 0)
            
        # Fetch conditional_dict for loss computation
        chunk_start_frame = chunk_info["chunk_start_frame"]
        conditional_dict = self._get_current_conditional_dict(chunk_start_frame)
        
        # Fetch gradient_mask to compute loss only on newly generated frames
        gradient_mask = chunk_info.get("gradient_mask", None)
        
        batch_size, num_frame = chunk.shape[:2]
        
        if DEBUG and (not dist.is_initialized() or dist.get_rank() == 0):
            print(f"[StreamingTrain-Model] Preparing critic loss: batch_size={batch_size}, num_frame={num_frame}")
        
        # Use the same timestep range logic as non-streaming training
        denoised_timestep_from = chunk_info.get("denoised_timestep_from", None)
        denoised_timestep_to = chunk_info.get("denoised_timestep_to", None)
        
        min_timestep = denoised_timestep_to if (getattr(self.base_model, 'ts_schedule', False) and denoised_timestep_to is not None) else getattr(self.base_model, 'min_score_timestep')
        max_timestep = denoised_timestep_from if (getattr(self.base_model, 'ts_schedule_max', False) and denoised_timestep_from is not None) else getattr(self.base_model, 'num_train_timestep')
        
        # Randomly select time steps
        critic_timestep = self.base_model._get_timestep(
            min_timestep=min_timestep,
            max_timestep=max_timestep,
            batch_size=batch_size,
            num_frame=num_frame,
            num_frame_per_block=getattr(self.base_model, 'num_frame_per_block', 3),
            uniform_timestep=True  # Set to True to match non-streaming training
        ).to(self.device)
        
        # Apply the same timestep shift logic as non-streaming training
        if getattr(self.base_model, 'timestep_shift') > 1:
            timestep_shift = self.base_model.timestep_shift
            critic_timestep = timestep_shift * \
                (critic_timestep / 1000) / (1 + (timestep_shift - 1) * (critic_timestep / 1000)) * 1000
        
        critic_timestep = critic_timestep.clamp(self.base_model.min_step, self.base_model.max_step)
        
        # Sample noise
        critic_noise = torch.randn_like(chunk)
        
        # Add noise to chunk
        noisy_chunk = self.scheduler.add_noise(
            chunk.flatten(0, 1),
            critic_noise.flatten(0, 1),
            critic_timestep.flatten(0, 1)
        ).unflatten(0, (batch_size, num_frame))

        if DEBUG and (not dist.is_initialized() or dist.get_rank() == 0):
            print(f"[StreamingTrain-Model] Added noise, timestep range: [{critic_timestep.min().item()}, {critic_timestep.max().item()}]")
        
        if (not dist.is_initialized() or dist.get_rank() == 0) and LOG_GPU_MEMORY:
            log_gpu_memory(f"StreamingTrain-Model: Before fake score computation", device=self.device, rank=dist.get_rank() if dist.is_initialized() else 0)
        
        # Compute fake prediction
        _, pred_fake_image = self.fake_score(
            noisy_image_or_video=noisy_chunk,
            conditional_dict=conditional_dict,
            timestep=critic_timestep
        )
        
        if (not dist.is_initialized() or dist.get_rank() == 0) and LOG_GPU_MEMORY:
            log_gpu_memory(f"StreamingTrain-Model: After fake score computation", device=self.device, rank=dist.get_rank() if dist.is_initialized() else 0)

        # Compute denoising loss
        denoising_loss_type = getattr(self.base_model.args, 'denoising_loss_type', 'mse')
        if denoising_loss_type == "flow":
            from utils.wan_wrapper import WanDiffusionWrapper
            flow_pred = WanDiffusionWrapper._convert_x0_to_flow_pred(
                scheduler=self.scheduler,
                x0_pred=pred_fake_image.flatten(0, 1),
                xt=noisy_chunk.flatten(0, 1),
                timestep=critic_timestep.flatten(0, 1)
            )
            pred_fake_noise = None
            if DEBUG and (not dist.is_initialized() or dist.get_rank() == 0):
                print(f"[StreamingTrain-Model] Using flow-based denoising loss")
        else:
            flow_pred = None
            pred_fake_noise = self.scheduler.convert_x0_to_noise(
                x0=pred_fake_image.flatten(0, 1),
                xt=noisy_chunk.flatten(0, 1),
                timestep=critic_timestep.flatten(0, 1)
            ).unflatten(0, (batch_size, num_frame))
            if DEBUG and (not dist.is_initialized() or dist.get_rank() == 0):
                print(f"[StreamingTrain-Model] Using MSE-based denoising loss")

        gradient_mask_flat = gradient_mask.flatten(0, 1) if gradient_mask is not None else None
        denoising_loss = self.denoising_loss_func(
            x=chunk.flatten(0, 1),
            x_pred=pred_fake_image.flatten(0, 1),
            noise=critic_noise.flatten(0, 1),
            noise_pred=pred_fake_noise,
            alphas_cumprod=self.scheduler.alphas_cumprod,
            timestep=critic_timestep.flatten(0, 1),
            flow_pred=flow_pred,
            gradient_mask=gradient_mask_flat  # Pass gradient_mask
        )
        
        if (not dist.is_initialized() or dist.get_rank() == 0) and LOG_GPU_MEMORY:
            log_gpu_memory(f"StreamingTrain-Model: After denoising loss computation", device=self.device, rank=dist.get_rank() if dist.is_initialized() else 0)
        
        if DEBUG and (not dist.is_initialized() or dist.get_rank() == 0):
            print(f"[StreamingTrain-Model] Critic loss computed: {denoising_loss.item()}")

        if 'flow_pred' in locals():
            del flow_pred
        if 'pred_fake_noise' in locals():
            del pred_fake_noise

        # Build log dict
        critic_log_dict = {
            "loss_time": time.time() - _t_loss_start,
            "new_frames_supervised": chunk_info.get("new_frames_generated", num_frame),
        }

        ###### gan process #####
        if self.use_gan:
            # gt_prompts_emb = self.model.text_encoder(text_prompts=batch["gt_prompts"])
            # gt_mismatch_prompts_emb = self.model.text_encoder(text_prompts=batch["mismatch_prompts"])
            # gt_pairs_emb = self.model.vae.encode_to_latent(batch["gt_pairs"])
            # mismatch_pairs_emb = self.model.vae.encode_to_latent(batch["mismatch_pairs"])
            real_img = self.state["frames4gan_gt"][self.state["chunk_cnt"]]
            mismatch_img = self.state["frames4gan_mismatch"][self.state["chunk_cnt"]]
            if dist.get_rank()==0:
                print("[test dataloader shape] real_img: {}. mismatch_img: {}. chunk_cnt: {}".format(real_img.shape, mismatch_img.shape, self.state["chunk_cnt"]))
            real_img_emb = self.base_model.vae.encode_to_latent(real_img.unsqueeze(2))
            mismatch_img_emb = self.base_model.vae.encode_to_latent(mismatch_img.unsqueeze(2))

            gt_prompt = self.state["frames4gan_gt_prompt"][self.state["chunk_cnt"]]
            gt_mismatch_prompt = self.state["frames4gan_mismatch_gt_prompt"][self.state["chunk_cnt"]]
            gt_prompt_emb = self.base_model.text_encoder(text_prompts=gt_prompt + gt_prompt)
            gt_mismatch_prompt_emb = self.base_model.text_encoder(text_prompts=gt_prompt + gt_mismatch_prompt)
            if dist.get_rank()==0:
                print("[test txt_emb shape] gt_prompt_emb: {}. gt_mismatch_prompt_emb: {}.".format(gt_prompt_emb["prompt_embeds"].shape, gt_mismatch_prompt_emb["prompt_embeds"].shape))
                print("[test txt prompt] gt_prompt: {}. gt_mismatch_prompt: {}".format(gt_prompt[0][0:10], gt_mismatch_prompt[0][0:10]))

            critic_cls_loss, critic_cls_loss_log_dict = self.compute_critic_cls_loss(
                real_image=real_img_emb,
                fake_image=self.state["frames4gan"], 
                mismatch_img=mismatch_img_emb, 
                real_text_embedding=gt_prompt_emb,
                fake_text_embedding=conditional_dict,
                mismatch_text_embedding=gt_mismatch_prompt_emb)
            denoising_loss += critic_cls_loss * 1e-3
            critic_log_dict.update(critic_cls_loss_log_dict)
            self.state["chunk_cnt"] += 1

        # Critical: clean up intermediate variables after critic loss
        del conditional_dict, critic_noise, noisy_chunk, pred_fake_image

        return denoising_loss, critic_log_dict
    
    def get_sequence_length(self) -> int:
        """Get current sequence length"""
        return self.state.get("current_length", 0) 

    def compute_generator_cls_loss(self, 
        fake_image, text_embedding, 
    ):
        pred_realism_on_fake_with_grad = self.compute_cls_logits(
            fake_image, 
            text_embedding=text_embedding, 
        )
        loss = F.softplus(-pred_realism_on_fake_with_grad).mean()
        return loss 
    
    def compute_critic_cls_loss(
            self, real_image, fake_image, mismatch_img,
            real_text_embedding, fake_text_embedding, mismatch_text_embedding
        ):
        logit_real = self.compute_cls_logits(
            real_image.detach(), 
            text_embedding=real_text_embedding,
        )
        logit_fake = self.compute_cls_logits(
            fake_image.detach(), 
            text_embedding=fake_text_embedding,
        )
        logit_real_mismatch = self.compute_cls_logits(
            mismatch_img.detach(), 
            text_embedding=mismatch_text_embedding,
        )

        loss = F.softplus(logit_real_mismatch).mean() + 0.5*F.softplus(logit_fake).mean() + F.softplus(-logit_real).mean()

        log_dict = {
            "critic_logit_real": logit_real.detach(),
            "critic_logit_fake": logit_fake.detach(),
            "critic_logit_real_mismatch": logit_real_mismatch.detach(),
            "critic_cls_loss": loss.detach()
        }
        return loss, log_dict
    
    def compute_cls_logits(self, images, text_embedding):
        # if dist.get_rank()==0:
        #     print("[compute_cls_logits] images shape is {}".format(images.shape))
        # TODO这里需要修改
        batch_size = 2
        num_frame = 1
        # timestep = self.base_model._get_timestep(
        #     min_timestep=0,
        #     max_timestep=1000,
        #     batch_size=batch_size,
        #     num_frame=num_frame,
        #     num_frame_per_block=getattr(self.base_model, 'num_frame_per_block', 3),
        #     uniform_timestep=True  # Set to True to match non-streaming training
        # ).to(self.device)
        
        # # Apply the same timestep shift logic as non-streaming training
        # if getattr(self.base_model, 'timestep_shift') > 1:
        #     timestep_shift = self.base_model.timestep_shift
        #     timestep = timestep_shift * \
        #         (timestep / 1000) / (1 + (timestep_shift - 1) * (timestep / 1000)) * 1000
        
        # timestep = timestep.clamp(self.base_model.min_step, self.base_model.max_step)

        timestep = torch.zeros([batch_size, num_frame], device=images.device, dtype=torch.int64)  # 用最后一步的噪声
        
        # Sample noise
        noise = torch.randn_like(images)
        
        # Add noise to chunk
        noisy_imgs = self.scheduler.add_noise(
            images.flatten(0, 1),
            noise.flatten(0, 1),
            timestep.flatten(0, 1)
        ).unflatten(0, (batch_size, num_frame))

        # Compute fake prediction
        flow_pred, pred_x0, logits = self.fake_score(
            noisy_image_or_video=noisy_imgs,
            conditional_dict=text_embedding,
            timestep=timestep,
            classify_mode=True
        )
        return logits