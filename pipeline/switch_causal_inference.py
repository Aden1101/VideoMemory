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
import math
import os
from typing import List, Optional
import torch

from utils.wan_wrapper import WanDiffusionWrapper, WanTextEncoder, WanVAEWrapper
from utils.memory import gpu, get_cuda_free_memory_gb, move_model_to_device_with_memory_preservation
from pipeline.causal_inference import CausalInferencePipeline  # reuse init utilities
from wan.modules.video_memory import TTTBlockCache
import torch.distributed as dist
from utils.debug_option import DEBUG
from pipeline.switch.anchor_decode import _memdec_get_decoder  # noqa: F401  (imported elsewhere)
from pipeline.switch.memory_state import (MemoryStateMixin, _find_causal_model,  # noqa: F401
                                          _memwriter_sync)
from pipeline.switch.anchor_decode import AnchorDecodeMixin
from pipeline.switch.anchor_controls import AnchorControlsMixin
from pipeline.switch.sdedit import SdeditMixin
from pipeline.switch.sink import SinkMixin
from pipeline.switch.composite import CompositeMixin
from pipeline.switch.additive_kv import AdditiveKVMixin
from pipeline.switch.frame_bank import FrameBankMixin


class SwitchCausalInferencePipeline(MemoryStateMixin, AnchorDecodeMixin, AnchorControlsMixin, SdeditMixin, SinkMixin, CompositeMixin, AdditiveKVMixin, FrameBankMixin, CausalInferencePipeline):
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
                "MEM_ANCHOR_T exige un generateur CAUSAL (uniform_timestep=False) : sinon le " \
                "timestep par frame est reduit a celui du slot 0 et la declaration est perdue"
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

        assert not int(os.environ.get("MEM_DECODE_COMPOSITE", "0") or 0), \
            "MEM_DECODE_COMPOSITE (arms oracle1comp / memdec21c) is implemented in the N-switch " \
            "driver only, run those arms through inference_schedule()"
        assert not int(os.environ.get("MEM_ADDITIVE_KV", "0") or 0), \
            "MEM_ADDITIVE_KV (arm oracle1add) is implemented in the N-switch driver only, " \
            "run it through inference_schedule()"
        assert int(os.environ.get("MEM_DECODE_SLOT_OFFSET", "-1") or -1) < 0, \
            "MEM_DECODE_SLOT_OFFSET (gate 2) is implemented in the N-switch driver only"
        assert float(os.environ.get("MEM_ANCHOR_T", "0") or 0) <= 0.0, \
            "MEM_ANCHOR_T (lot offensif 2026-08-04, arms i1_*) is implemented in the N-switch " \
            "driver only, run it through inference_schedule()"
        assert float(os.environ.get("MEM_SDEDIT", "0") or 0) <= 0.0 \
            and not int(os.environ.get("MEM_SINKPURGE", "0") or 0), \
            "MEM_SDEDIT / MEM_SINKPURGE (mini-batterie training-free 2026-08-04, arms " \
            "sdedit_ours / sdedit_wrong / sinkpurge) are implemented in the N-switch driver only"
        assert float(os.environ.get("MEM_DECODE_NOISE", "0") or 0) <= 0.0 \
            and not os.environ.get("MEM_DECODE_SRC", "").strip(), \
            "MEM_DECODE_NOISE / MEM_DECODE_SRC (reviewer controls noiseanchor / wronganchor, " \
            "2026-08-03) are implemented in the N-switch driver only"

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

        if low_memory:
            gpu_memory_preservation = get_cuda_free_memory_gb(gpu) + 5
            move_model_to_device_with_memory_preservation(
                self.text_encoder, target_device=gpu, preserved_memory_gb=gpu_memory_preservation
            )

        # Place output on CPU to save GPU memory
        output_device = torch.device('cpu') if low_memory else noise.device
        output = torch.zeros(
            [batch_size, num_output_frames, num_channels, height, width],
            device=output_device,
            dtype=noise.dtype
        )

        local_attn_cfg = getattr(self.args.model_kwargs, "local_attn_size", -1)
        kv_policy = ""
        if local_attn_cfg != -1:
            # local attention
            kv_cache_size = local_attn_cfg * self.frame_seq_length
            kv_policy = f"int->local, size={local_attn_cfg}"
        else:
            # global attention
            kv_cache_size = num_output_frames * self.frame_seq_length
            kv_policy = "global (-1)"
        print(f"kv_cache_size: {kv_cache_size} (policy: {kv_policy}, frame_seq_length: {self.frame_seq_length}, num_output_frames: {num_output_frames})")

        self._initialize_kv_cache(
            batch_size,
            dtype=noise.dtype,
            device=noise.device,
            kv_cache_size_override=kv_cache_size
        )
        self._initialize_crossattn_cache(
            batch_size=batch_size,
            dtype=noise.dtype,
            device=noise.device
        )
        # fast-weight memory state, fresh per sequence
        # stock model with the memory never read or written)
        self.ttt_cache = [TTTBlockCache() for _ in range(self.num_transformer_blocks)]
        # MEM_WRITER (arm memdec20, memory 2.0): overlay the offline-trained write parameters on
        # the memory cells before the first write of this rollout. Default off = no-op; consumes
        # no RNG draw, so the paired arms keep identical noise streams.
        _memwriter_sync(self.generator.model)

        current_start_frame = 0
        num_input_frames = 0
        self._switch_global_frame = None
        self._first_switch_frame = None
        self._ttt_snap_A = None
        self._thru_frame = None          # MEM_DECODE_THRU cache (arm oracle1thru), per rollout
        self._thru_info = None
        self._comp_gate_now = 1.0
        self._add_installed = False
        self._add_mass_log = []          # T1: masse par source, par chunk et par bloc
        self._clear_add_kv()             # no additive channel may survive from a previous arm
        self.generator.model.local_attn_size = self.local_attn_size
        print(f"[inference] local_attn_size set on model: {self.generator.model.local_attn_size}")
        self._set_all_modules_max_attention_size(self.local_attn_size)

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
                if int(os.environ.get("MEM_EPISODIC", "0")) and getattr(self, "_ttt_snap_A", None):
                    self._restore_ttt(self._ttt_snap_A)
                    print(f"[MEM_EPISODIC] restored A-era memory state at frame {current_start_frame}")
                # ORACLE_SPLICE rebuilds the attention window from the REAL frames of the
                # subject's first appearance instead of the current ones. It is deliberate
                # cheating: it gives the ceiling that a perfect memory could reach.
                _splice = None
                # dump the original-A anchor latent for the twin arm (free,
                # no RNG draw, off by default) before any splice touches `output`.
                self._dump_anchor(output, self._first_switch_frame)
                if int(os.environ.get("ORACLE_SPLICE", "0")):
                    _n = current_start_frame if self.local_attn_size == -1 \
                        else min(self.local_attn_size, current_start_frame)
                    _a_end = self._first_switch_frame
                    assert _a_end is not None and _a_end - _n >= 0, \
                        f"oracle splice needs {_n} frames of original A before frame {_a_end}"
                    # how few real frames still suffice: only the last nf slots carry the past,
                    # the rest keeps the true recent tail. 0 = the whole window.
                    _nf = int(os.environ.get("ORACLE_SPLICE_FRAMES", "0") or 0)
                    # does the position matter: the single past frame replaces the window slot
                    # k frames before the most recent one. k=0 is the most recent slot, k of 3 to
                    # 6 reads as "a few moments ago". -1 or unset = off.
                    _slot_k = int(os.environ.get("ORACLE_SPLICE_SLOT", "-1") or -1)
                    if _slot_k >= 0:
                        _splice = output[:, current_start_frame - _n:current_start_frame].clone()
                        _pos = _n - 1 - min(_slot_k, _n - 1)
                        _splice[:, _pos:_pos + 1] = self._anchor_content(output, _a_end - 1, _a_end)
                        print(f"[ORACLE_SPLICE] anchor at window slot {_pos}/{_n} "
                              f"({_slot_k} frames before most-recent)")
                    elif 0 < _nf < _n:
                        # ORACLE_SPLICE_POS=head puts the anchor in the oldest window slots
                        # instead of the most recent: a distant reference, not a recent frame.
                        if os.environ.get("ORACLE_SPLICE_POS", "") == "head":
                            _splice = torch.cat([
                                self._anchor_content(output, _a_end - _nf, _a_end),
                                output[:, current_start_frame - (_n - _nf):current_start_frame]],
                                dim=1)
                            print(f"[ORACLE_SPLICE] partial HEAD: first {_nf}/{_n} window slots "
                                  f"from original A [{_a_end - _nf}, {_a_end}), rest = recent tail")
                        else:
                            _splice = torch.cat([
                                output[:, current_start_frame - _n:current_start_frame - _nf],
                                self._anchor_content(output, _a_end - _nf, _a_end)], dim=1)
                            print(f"[ORACLE_SPLICE] partial: last {_nf}/{_n} window slots from original A "
                                  f"[{_a_end - _nf}, {_a_end}), rest = true recent tail")
                    else:
                        _splice = self._anchor_content(output, _a_end - _n, _a_end)
                        print(f"[ORACLE_SPLICE] recaching frames [{_a_end - _n}, {_a_end}) of original A "
                              f"at positions [{current_start_frame - _n}, {current_start_frame})")
                # ORACLE_SPLICE_NOISE degrades the anchor with RMS-preserving noise, to imitate
                # an imperfect recall. Its own generator, so the main random stream is untouched.
                _dose = float(os.environ.get("ORACLE_SPLICE_NOISE", "0") or 0)
                if _dose > 0.0 and _splice is not None:
                    _nf_env = int(os.environ.get("ORACLE_SPLICE_FRAMES", "0") or 0)
                    _na = _nf_env if 0 < _nf_env < _splice.shape[1] else _splice.shape[1]
                    _g = torch.Generator(device=_splice.device)
                    _g.manual_seed(987654321 + current_start_frame)
                    _a_part = _splice[:, -_na:]
                    _eps = torch.randn(_a_part.shape, generator=_g, device=_a_part.device,
                                       dtype=torch.float32).to(_a_part.dtype)
                    _a_deg = ((1.0 - _dose ** 2) ** 0.5) * _a_part + _dose * _eps
                    _splice = torch.cat([_splice[:, :-_na], _a_deg], dim=1)
                    print(f"[ORACLE_SPLICE_NOISE] degraded last {_na} splice frame(s), sigma={_dose}")
                # the frame fed back is decoded from the live memory instead of being the real
                # past frame. Same mechanism as above, but nothing is read from the true past:
                # this is the arm that actually tests the memory.
                _dec_ckpt = os.environ.get("MEM_DECODE", "").strip()
                if _dec_ckpt:
                    assert _splice is None, "MEM_DECODE and ORACLE_SPLICE are mutually exclusive"
                    _n = current_start_frame if self.local_attn_size == -1 \
                        else min(self.local_attn_size, current_start_frame)
                    if int(os.environ.get("MEM_DECODE_THRU", "0") or 0):
                        # arm oracle1thru: the frame was already decoded at the eviction of the
                        # anchor chunk (write age 1); nothing is decoded here.
                        assert self._thru_frame is not None, \
                            "MEM_DECODE_THRU: the anchor chunk was never evicted before the " \
                            "return switch (check switch indices / local_attn_size)"
                        _frame, _cond_on = self._thru_frame, None
                        print(f"[MEM_DECODE_THRU] splicing the frame decoded at eviction "
                              f"{self._thru_info}", flush=True)
                    else:
                        _frame, _cond_on = self._memdec_frame(_dec_ckpt, cond_third,
                                                              output.dtype, output.device)
                    _cond = _cond_on
                    # MEM_DECODE_SLOTS (default 1): repeat the decoded anchor over the last k
                    # window slots, evicts B from the readable context k times faster (window-
                    # mixing fix; oracle3 proved multi-slot anchors keep the same re-ID score).
                    _kslots = max(1, int(os.environ.get("MEM_DECODE_SLOTS", "1") or 1))
                    # MEM_DECODE_DUMP (debug/eye-check): save the decoded anchor latent so it
                    # can be VAE-decoded and inspected (is the anchor clean-A or an A/B fusion?)
                    _dump = os.environ.get("MEM_DECODE_DUMP", "").strip()
                    if _dump:
                        torch.save(_frame.detach().to("cpu", torch.float16),
                                   f"{_dump}.f{current_start_frame}.pt")
                        print(f"[MEM_DECODE_DUMP] anchor latent saved -> {_dump}.f{current_start_frame}.pt")
                    _splice = torch.cat(
                        [output[:, current_start_frame - (_n - _kslots):current_start_frame]]
                        + [_frame] * _kslots, dim=1)
                    print(f"[MEM_DECODE] decoded anchor spliced at recency ({_kslots}/{_n} slots), "
                          f"cond={'thru' if _cond is None else ('ON' if _cond else 'off')}",
                          flush=True)
                _blocks_env = os.environ.get("ORACLE_SPLICE_BLOCKS", "").strip()
                if _splice is not None and _blocks_env:
                    # keep the anchor recache only on the listed blocks, a normal one elsewhere:
                    # does the anchor at the memory blocks alone bring the subject back? Both
                    # passes see the same frames at the same positions, so restoring k/v is exact.
                    _keep = {int(b) for b in _blocks_env.split(",") if b != ""}
                    self._recache_after_switch(output, current_start_frame, cond_third)
                    _snap_kv = {bi: {kk: self.kv_cache1[bi][kk].detach().to("cpu", copy=True)
                                     for kk in ("k", "v")}
                                for bi in range(self.num_transformer_blocks) if bi not in _keep}
                    self._recache_after_switch(output, current_start_frame, cond_third,
                                               frames_override=_splice)
                    with torch.no_grad():
                        for bi, kv in _snap_kv.items():
                            self.kv_cache1[bi]["k"].copy_(kv["k"].to(self.kv_cache1[bi]["k"].device))
                            self.kv_cache1[bi]["v"].copy_(kv["v"].to(self.kv_cache1[bi]["v"].device))
                    del _snap_kv
                    print(f"[ORACLE_SPLICE_BLOCKS] oracle KV kept ONLY on blocks {sorted(_keep)}, "
                          f"other blocks restored to the normal recache")
                elif os.environ.get("ORACLE_SPLICE_POS", "") == "sink0" and _splice is not None:
                    # normal recache (window = true recent tail),
                    # then the anchor frame overwrites the sink slot (read forever, never evicted)
                    self._recache_after_switch(output, current_start_frame, cond_third)
                    self._write_sink_anchor(_splice[:, -1:], cond_third)
                    print(f"[ORACLE_SPLICE] sink0: anchor frame written into the SINK slot "
                          f"(persistent), window = normal recent tail")
                else:
                    self._recache_after_switch(output, current_start_frame, cond_third,
                                               frames_override=_splice)
                cond_in_use = cond_third
                using_third = True
                self._switch_global_frame = current_start_frame
                print("second_switch_frame_index", second_switch_frame_index)
                print("current_start_frame", current_start_frame)
            elif (not using_second) and (current_start_frame >= switch_frame_index):
                # MEM_EPISODIC: freeze the memory of the opening scene before the next one overwrites it.
                if int(os.environ.get("MEM_EPISODIC", "0")):
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
            
            noisy_input = noise[:, current_start_frame - num_input_frames : current_start_frame + current_num_frames - num_input_frames]

            # Spatial denoising loop (same as parent but uses cond_in_use)
            for index, current_timestep in enumerate(self.denoising_step_list):
                timestep = torch.ones([batch_size, current_num_frames], device=noise.device, dtype=torch.int64) * current_timestep
                if index < len(self.denoising_step_list) - 1:
                    _, denoised_pred = self.generator(
                        noisy_image_or_video=noisy_input,
                        conditional_dict=cond_in_use,
                        timestep=timestep,
                        kv_cache=self.kv_cache1,
                        crossattn_cache=self.crossattn_cache,
                        current_start=current_start_frame * self.frame_seq_length,
                        rope_enhance=self.rope_enhance,
                        ttt_cache=self.ttt_cache,
                        ttt_update=False,
                        switch_global_frame=self._switch_global_frame,
                    )
                    next_timestep = self.denoising_step_list[index + 1]
                    noisy_input = self.scheduler.add_noise(
                        denoised_pred.flatten(0, 1),
                        torch.randn_like(denoised_pred.flatten(0, 1)),
                        next_timestep * torch.ones([batch_size * current_num_frames], device=noise.device, dtype=torch.long),
                    ).unflatten(0, denoised_pred.shape[:2])
                else:
                    _, denoised_pred = self.generator(
                        noisy_image_or_video=noisy_input,
                        conditional_dict=cond_in_use,
                        timestep=timestep,
                        kv_cache=self.kv_cache1,
                        crossattn_cache=self.crossattn_cache,
                        current_start=current_start_frame * self.frame_seq_length,
                        rope_enhance=self.rope_enhance,
                        ttt_cache=self.ttt_cache,
                        ttt_update=False,
                        switch_global_frame=self._switch_global_frame,
                    )

            output[:, current_start_frame : current_start_frame + current_num_frames] = denoised_pred.to(output.device)

            # rerun with clean context noise for cache update
            context_timestep = torch.ones_like(timestep) * self.args.context_noise
            self.generator(
                noisy_image_or_video=denoised_pred,
                conditional_dict=cond_in_use,
                timestep=context_timestep,
                kv_cache=self.kv_cache1,
                crossattn_cache=self.crossattn_cache,
                current_start=current_start_frame * self.frame_seq_length,
                rope_enhance=self.rope_enhance,
                ttt_cache=self.ttt_cache,
                ttt_update=True,
                switch_global_frame=self._switch_global_frame,
            )

            # MEM_DECODE_THRU (arm oracle1thru, default off = no-op): the write that stores the
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

    def _switch_recache(self, output, current_start_frame, cond_new, *, a_end=None,
                        seg_idx=None, prompt_idx=None, is_revisit=False, subject=None,
                        cond_prev=None):
        """Cache surgery at one schedule boundary.

        is_revisit=False -> plain switch: no recall hook fires and, with FRAMEBANK
        off, the call is byte-identical to the legacy first switch.
        is_revisit=True  -> the entering prompt has been seen before: the recall hooks fire.
        `a_end` = end latent frame (exclusive) of the segment where the entering subject first
        appeared, i.e. the anchor frame is `a_end - 1`. In the legacy A->B->A rollout that is
        `self._first_switch_frame`, so this is a strict generalization of the block in
        `inference()` (same statements, same order, same tensors).

        Note on FRAMEBANK: its router runs at every transition, revisit or not, Echo-Forcing
        has no oracle telling it which switch is a revisit, so a false recall on a fresh scene
        is a real failure mode of the baseline and must be measurable, not hidden.
        """
        assert not int(os.environ.get("MEM_DECODE_THRU", "0") or 0), \
            "MEM_DECODE_THRU (arm oracle1thru) is not supported by the N-switch driver: its " \
            "eviction arithmetic assumes the single legacy anchor, run it through inference()"
        _splice = None
        _ts_over = None                  # MEM_ANCHOR_T: per-slot declared timestep of the recache
        _comp_before = getattr(self, "_comp_applied", 0)
        if is_revisit:
            assert a_end is not None and a_end >= 1, "revisit without a first-appearance segment"
            # MEM_EPISODIC: restore the state snapshotted when this subject was last left.
            _snaps = getattr(self, "_ttt_snap_by_prompt", None) or {}
            _sk = prompt_idx if subject is None else subject
            if int(os.environ.get("MEM_EPISODIC", "0")) and _snaps.get(_sk):
                self._restore_ttt(_snaps[_sk])
                print(f"[MEM_EPISODIC] restored the memory state of subject {_sk} at frame "
                      f"{current_start_frame}")
            self._dump_anchor(output, a_end)
            if int(os.environ.get("MEM_COMP_PROMPT_GATE", "0") or 0):
                self._comp_gate_from_prompts(cond_new, cond_prev)
        # ---- MEM_ADDITIVE_KV (gate 1): the anchor enters as extra k/v tokens, no splice ------
        _addk = bool(is_revisit and int(os.environ.get("MEM_ADDITIVE_KV", "0") or 0))
        if _addk:
            assert not int(os.environ.get("ORACLE_SPLICE", "0") or 0) and not self._fb_mode(), \
                "MEM_ADDITIVE_KV is the NO-EVICTION channel: it must not be combined with a " \
                "splice arm (ORACLE_SPLICE / FRAMEBANK)"
            _ck = os.environ.get("MEM_DECODE", "").strip()
            if _ck:                       # future arm memdec20add: the decoded anchor, additive
                _content, _ = self._memdec_frame(_ck, cond_new, output.dtype, output.device)
            else:                         # arm oracle1add: the true anchor frame, additive
                _content = self._anchor_content(output, a_end - 1, a_end)
            self._install_add_kv(_content, cond_new)
        if is_revisit and int(os.environ.get("ORACLE_SPLICE", "0")):
            _n = current_start_frame if self.local_attn_size == -1 \
                else min(self.local_attn_size, current_start_frame)
            _a_end = a_end
            _nf = int(os.environ.get("ORACLE_SPLICE_FRAMES", "0") or 0)
            _need = _nf if 0 < _nf < _n else _n
            assert _a_end - _need >= 0, \
                f"oracle splice needs {_need} frames of the first appearance before frame {_a_end}"
            _slot_k = int(os.environ.get("ORACLE_SPLICE_SLOT", "-1") or -1)
            if _slot_k >= 0:
                _splice = output[:, current_start_frame - _n:current_start_frame].clone()
                _pos = _n - 1 - min(_slot_k, _n - 1)
                _splice[:, _pos:_pos + 1] = self._anchor_content(output, _a_end - 1, _a_end)
                print(f"[ORACLE_SPLICE] anchor at window slot {_pos}/{_n} "
                      f"({_slot_k} frames before most-recent)")
            elif 0 < _nf < _n:
                if os.environ.get("ORACLE_SPLICE_POS", "") == "head":
                    _splice = torch.cat([
                        self._anchor_content(output, _a_end - _nf, _a_end),
                        output[:, current_start_frame - (_n - _nf):current_start_frame]], dim=1)
                    print(f"[ORACLE_SPLICE] partial HEAD: first {_nf}/{_n} window slots "
                          f"from first appearance [{_a_end - _nf}, {_a_end}), rest = recent tail")
                else:
                    _splice = torch.cat([
                        output[:, current_start_frame - _n:current_start_frame - _nf],
                        self._maybe_composite(
                            self._anchor_content(output, _a_end - _nf, _a_end),
                            output, current_start_frame)], dim=1)
                    print(f"[ORACLE_SPLICE] partial: last {_nf}/{_n} window slots from the first "
                          f"appearance [{_a_end - _nf}, {_a_end}), rest = true recent tail")
            else:
                _splice = self._anchor_content(output, _a_end - _n, _a_end)
                print(f"[ORACLE_SPLICE] recaching frames [{_a_end - _n}, {_a_end}) of the first "
                      f"appearance at positions [{current_start_frame - _n}, {current_start_frame})")
        _dose = float(os.environ.get("ORACLE_SPLICE_NOISE", "0") or 0)
        if _dose > 0.0 and _splice is not None:
            _nf_env = int(os.environ.get("ORACLE_SPLICE_FRAMES", "0") or 0)
            _na = _nf_env if 0 < _nf_env < _splice.shape[1] else _splice.shape[1]
            _g = torch.Generator(device=_splice.device)
            _g.manual_seed(987654321 + current_start_frame)
            _a_part = _splice[:, -_na:]
            _eps = torch.randn(_a_part.shape, generator=_g, device=_a_part.device,
                               dtype=torch.float32).to(_a_part.dtype)
            _a_deg = ((1.0 - _dose ** 2) ** 0.5) * _a_part + _dose * _eps
            _splice = torch.cat([_splice[:, :-_na], _a_deg], dim=1)
            print(f"[ORACLE_SPLICE_NOISE] degraded last {_na} splice frame(s), sigma={_dose}")
        # ---- FRAMEBANK (Echo-Forcing-style real-content recall), routes at every switch ------
        if self._fb_mode():
            assert _splice is None and not os.environ.get("MEM_DECODE", "").strip(), \
                "FRAMEBANK, ORACLE_SPLICE and MEM_DECODE are mutually exclusive"
            _n = current_start_frame if self.local_attn_size == -1 \
                else min(self.local_attn_size, current_start_frame)
            # FRAMEBANK_GT (pilote texte-insuffisant): le QUAND = la verite du planning, pour que
            # framebank, memdec20 et wronganchor splicent EXACTEMENT au meme instant (egalite
            # stricte). Le routeur tourne quand meme et sa decision est loguee.
            _gt = bool(int(os.environ.get("FRAMEBANK_GT", "0") or 0))
            _fb_frame, _ = self._fb_select(seg_idx, prompt_idx, cond_new, subject=subject,
                                           force=(bool(is_revisit) if _gt else None))
            if _gt and not is_revisit:
                _fb_frame = None
            if _fb_frame is not None:
                # same placement as the real-frame arm: the recalled frame takes
                # the most-recent window slot, the rest of the window stays the true recent tail.
                _splice = torch.cat(
                    [output[:, current_start_frame - _n:current_start_frame - 1],
                     _fb_frame.to(dtype=output.dtype, device=output.device)], dim=1)
                print(f"[FRAMEBANK] recalled frame spliced at recency (1/{_n} slots)", flush=True)
        # ---- when the decoded anchor fires --------------------------------------------------
        # By default the schedule says when a subject is revisited, which is correct but is a
        # signal the frame-bank baseline does not get. MEM_DECODE_ROUTED=1 uses that baseline's
        # own router instead, so the two arms differ only by what they bring back.
        _fires = is_revisit
        if int(os.environ.get("MEM_DECODE_ROUTED", "0") or 0):
            assert not self._fb_mode(), \
                "MEM_DECODE_ROUTED and FRAMEBANK are mutually exclusive (one router, one arm)"
            _, _rev = self._fb_select(seg_idx, prompt_idx, cond_new, subject=subject, mode="key")
            _fires = bool(_rev["recalled"])
            print(f"[MEM_DECODE_ROUTED] segment {seg_idx}: router says "
                  f"{'REVISIT -> splice' if _fires else 'fresh scene -> no splice'} "
                  f"(ground truth was {'revisit' if is_revisit else 'fresh'})", flush=True)
        _dec_ckpt = os.environ.get("MEM_DECODE", "").strip() if (_fires and not _addk) else ""
        if _dec_ckpt:
            assert _splice is None, "MEM_DECODE and ORACLE_SPLICE are mutually exclusive"
            _n = current_start_frame if self.local_attn_size == -1 \
                else min(self.local_attn_size, current_start_frame)
            _frame, _cond_on = self._memdec_frame(_dec_ckpt, cond_new, output.dtype, output.device)
            _cond = _cond_on
            _kslots = max(1, int(os.environ.get("MEM_DECODE_SLOTS", "1") or 1))
            _dump = os.environ.get("MEM_DECODE_DUMP", "").strip()
            if _dump:
                # another run may be reading this file at the same moment, so write to .tmp and
                # rename: torch.save alone leaves a window where the file is truncated.
                _dp = f"{_dump}.f{current_start_frame}.pt"
                torch.save(_frame.detach().to("cpu", torch.float16), _dp + ".tmp")
                os.replace(_dp + ".tmp", _dp)
                print(f"[MEM_DECODE_DUMP] anchor latent saved -> {_dp}")
            # the controls run after the dump, so the dumped anchor stays the raw decoder
            # output that another run reads back, and before the composite, so they compose.
            _frame = self._maybe_wrong_anchor(_frame)
            _frame = self._maybe_noise_anchor(_frame, output, current_start_frame, _n)
            # sharpen the anchor with the model's own prior, after the content controls on
            # purpose: sharpening a wrong anchor is then the exact twin of sharpening the right one.
            _frame = self._maybe_sdedit(_frame, cond_new, current_start_frame)
            # keep only the subject region of the anchor, take the border from the current scene
            _frame = self._maybe_composite(_frame, output, current_start_frame)
            _soff = int(os.environ.get("MEM_DECODE_SLOT_OFFSET", "-1") or -1)
            _splice = self._memdec_window(output, current_start_frame, _n, _frame, _kslots, _soff)
            # MEM_ANCHOR_T (arms i1_*): stop lying about the anchor's noise level. OFF by default,
            # and it returns the splice object itself then, so every arm on disk is unchanged.
            _splice, _ts_over = self._maybe_declare_anchor_noise(
                _splice, _n, _kslots, _soff, current_start_frame)
            print(f"[MEM_DECODE] decoded anchor spliced ({_kslots}/{_n} slots, window convention "
                  f"{'LEGACY (recency duplicated)' if _soff < 0 else f'oracle, slot_offset={_soff}'}"
                  f"), cond={'thru' if _cond is None else ('ON' if _cond else 'off')}", flush=True)
        if int(os.environ.get("MEM_DECODE_COMPOSITE", "0") or 0) and _splice is not None:
            assert getattr(self, "_comp_applied", 0) > _comp_before, \
                "MEM_DECODE_COMPOSITE=1 but the splice built at this switch carries NO " \
                "composited anchor, unsupported placement (SLOT / POS=head / full window / " \
                "FRAMEBANK content)"
        _blocks_env = os.environ.get("ORACLE_SPLICE_BLOCKS", "").strip()
        assert _ts_over is None or not (_blocks_env
                                        or os.environ.get("ORACLE_SPLICE_POS", "") == "sink0"), \
            "MEM_ANCHOR_T is defined for the plain recency splice (it declares the noise level " \
            "of the slot the recache writes), not for ORACLE_SPLICE_BLOCKS / POS=sink0"
        if _splice is not None and _blocks_env:
            _keep = {int(b) for b in _blocks_env.split(",") if b != ""}
            self._recache_after_switch(output, current_start_frame, cond_new)
            _snap_kv = {bi: {kk: self.kv_cache1[bi][kk].detach().to("cpu", copy=True)
                             for kk in ("k", "v")}
                        for bi in range(self.num_transformer_blocks) if bi not in _keep}
            self._recache_after_switch(output, current_start_frame, cond_new,
                                       frames_override=_splice)
            with torch.no_grad():
                for bi, kv in _snap_kv.items():
                    self.kv_cache1[bi]["k"].copy_(kv["k"].to(self.kv_cache1[bi]["k"].device))
                    self.kv_cache1[bi]["v"].copy_(kv["v"].to(self.kv_cache1[bi]["v"].device))
            del _snap_kv
            print(f"[ORACLE_SPLICE_BLOCKS] oracle KV kept ONLY on blocks {sorted(_keep)}, "
                  f"other blocks restored to the normal recache")
        elif os.environ.get("ORACLE_SPLICE_POS", "") == "sink0" and _splice is not None:
            self._recache_after_switch(output, current_start_frame, cond_new)
            self._write_sink_anchor(_splice[:, -1:], cond_new)
            print(f"[ORACLE_SPLICE] sink0: anchor frame written into the SINK slot "
                  f"(persistent), window = normal recent tail")
        else:
            self._recache_after_switch(output, current_start_frame, cond_new,
                                       frames_override=_splice, timestep_override=_ts_over)
        # MEM_SINKPURGE (arm sinkpurge): after the recache, so it rewrites the sink slots of the
        # cache the return will actually read. No-op unless the hook is on (default off).
        self._maybe_sinkpurge(output, current_start_frame, cond_new, is_revisit)

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

        Legacy equivalence: schedule [(0, 0), (sw1, 1), (sw2, 0)] with prompts [A, B] reproduces
        `inference(noise, text_prompts_first=[A], text_prompts_second=[B], switch_frame_index=sw1,
        second_switch_frame_index=sw2)` bitwise (selftest a). The text encoder is called once per
        schedule entry, in schedule order, exactly the legacy call sequence (A, B, A).
        """
        batch_size, num_output_frames, num_channels, height, width = noise.shape
        assert num_output_frames % self.num_frame_per_block == 0
        num_blocks = num_output_frames // self.num_frame_per_block

        assert not int(os.environ.get("MEM_DECODE_THRU", "0") or 0), \
            "MEM_DECODE_THRU (arm oracle1thru) is not supported by the N-switch driver: its " \
            "eviction arithmetic assumes the single legacy anchor, run it through inference()"
        schedule = [(int(f), int(p)) for f, p in schedule]
        assert schedule and schedule[0][0] == 0, "schedule must start at frame 0"
        for i, (f, p) in enumerate(schedule):
            assert 0 <= p < len(prompts), f"schedule[{i}] prompt index {p} out of range"
            assert f % self.num_frame_per_block == 0, \
                f"schedule[{i}] start frame {f} is not a multiple of {self.num_frame_per_block}"
            if i:
                assert f > schedule[i - 1][0], "schedule start frames must be strictly increasing"
        assert schedule[-1][0] < num_output_frames, "last segment is empty"
        seg_prompt = [p for _, p in schedule]
        seg_subj = [int(x) for x in (subjects if subjects is not None else seg_prompt)]
        assert len(seg_subj) == len(schedule), \
            f"subjects has {len(seg_subj)} entries for {len(schedule)} segments"
        # first appearance of each subject + the end frame of that segment (= anchor frame + 1)
        seg_end = [schedule[i + 1][0] for i in range(len(schedule) - 1)] + [num_output_frames]
        first_app = {}
        for i, s in enumerate(seg_subj):
            first_app.setdefault(s, i)
        is_revisit = [i > 0 and first_app[s] < i for i, s in enumerate(seg_subj)]
        print(f"[schedule] {len(schedule)} segments, {len(prompts)} distinct prompts, subjects "
              f"{seg_subj}, revisits at segments "
              f"{[i for i, r in enumerate(is_revisit) if r]}", flush=True)
        for i, (f, p) in enumerate(schedule):
            print(f"[schedule]   seg{i}: frames [{f}, {seg_end[i]})  prompt{p} subject{seg_subj[i]}"
                  f"{'  REVISIT of seg' + str(first_app[seg_subj[i]]) if is_revisit[i] else ''}"
                  f"  '{prompts[p][:60]}'", flush=True)

        # one text-encoder call per schedule entry, in schedule order (legacy call sequence).
        conds = [self.text_encoder(text_prompts=[prompts[p]]) for _, p in schedule]

        if low_memory:
            gpu_memory_preservation = get_cuda_free_memory_gb(gpu) + 5
            move_model_to_device_with_memory_preservation(
                self.text_encoder, target_device=gpu, preserved_memory_gb=gpu_memory_preservation
            )

        output_device = torch.device('cpu') if low_memory else noise.device
        output = torch.zeros(
            [batch_size, num_output_frames, num_channels, height, width],
            device=output_device,
            dtype=noise.dtype
        )

        local_attn_cfg = getattr(self.args.model_kwargs, "local_attn_size", -1)
        kv_policy = ""
        if local_attn_cfg != -1:
            kv_cache_size = local_attn_cfg * self.frame_seq_length
            kv_policy = f"int->local, size={local_attn_cfg}"
        else:
            kv_cache_size = num_output_frames * self.frame_seq_length
            kv_policy = "global (-1)"
        print(f"kv_cache_size: {kv_cache_size} (policy: {kv_policy}, frame_seq_length: {self.frame_seq_length}, num_output_frames: {num_output_frames})")

        self._initialize_kv_cache(
            batch_size, dtype=noise.dtype, device=noise.device,
            kv_cache_size_override=kv_cache_size)
        self._initialize_crossattn_cache(
            batch_size=batch_size, dtype=noise.dtype, device=noise.device)
        self.ttt_cache = [TTTBlockCache() for _ in range(self.num_transformer_blocks)]
        _memwriter_sync(self.generator.model)

        current_start_frame = 0
        num_input_frames = 0
        self._switch_global_frame = None
        self._first_switch_frame = schedule[1][0] if len(schedule) > 1 else None
        self._ttt_snap_A = None
        self._ttt_snap_by_prompt = {}
        self._thru_frame = None
        self._thru_info = None
        self._comp_applied = 0           # MEM_DECODE_COMPOSITE bookkeeping, per rollout
        self._comp_gate_now = 1.0
        self._add_installed = False
        self._add_mass_log = []          # T1: masse par source, par chunk et par bloc
        self._clear_add_kv()             # no additive channel may survive from a previous arm
        self._fb_reset()
        self.generator.model.local_attn_size = self.local_attn_size
        print(f"[inference] local_attn_size set on model: {self.generator.model.local_attn_size}")
        self._set_all_modules_max_attention_size(self.local_attn_size)

        all_num_frames = [self.num_frame_per_block] * num_blocks

        seg = 0
        for current_num_frames in all_num_frames:
            if seg + 1 < len(schedule) and current_start_frame >= schedule[seg + 1][0]:
                prev_seg, seg = seg, seg + 1
                # FRAMEBANK 'preserve': the segment that just ended enters the scene pool
                # (before the recall, so the just-left scene is in the bank and excluded as the
                # adjacent one, exactly Echo-Forcing's routing rule).
                self._fb_store(output, current_start_frame, prev_seg, seg_prompt[prev_seg],
                               conds[prev_seg], subject=seg_subj[prev_seg])
                # MEM_EPISODIC: freeze the state of a first appearance that will be revisited.
                if int(os.environ.get("MEM_EPISODIC", "0")) \
                        and first_app[seg_subj[prev_seg]] == prev_seg \
                        and seg_subj[prev_seg] in seg_subj[prev_seg + 1:]:
                    self._ttt_snap_by_prompt[seg_subj[prev_seg]] = self._snapshot_ttt()
                    print(f"[MEM_EPISODIC] snapshotted the state of subject "
                          f"{seg_subj[prev_seg]} at frame {current_start_frame}")
                self._switch_recache(
                    output, current_start_frame, conds[seg],
                    a_end=seg_end[first_app[seg_subj[seg]]],
                    seg_idx=seg, prompt_idx=seg_prompt[seg], subject=seg_subj[seg],
                    is_revisit=is_revisit[seg], cond_prev=conds[prev_seg])
                self._switch_global_frame = current_start_frame
                print(f"[schedule] switch to segment {seg} (prompt {seg_prompt[seg]}) at frame "
                      f"{current_start_frame}", flush=True)
            cond_in_use = conds[seg]

            noisy_input = noise[:, current_start_frame - num_input_frames : current_start_frame + current_num_frames - num_input_frames]

            for index, current_timestep in enumerate(self.denoising_step_list):
                timestep = torch.ones([batch_size, current_num_frames], device=noise.device, dtype=torch.int64) * current_timestep
                if index < len(self.denoising_step_list) - 1:
                    _, denoised_pred = self.generator(
                        noisy_image_or_video=noisy_input,
                        conditional_dict=cond_in_use,
                        timestep=timestep,
                        kv_cache=self.kv_cache1,
                        crossattn_cache=self.crossattn_cache,
                        current_start=current_start_frame * self.frame_seq_length,
                        rope_enhance=self.rope_enhance,
                        ttt_cache=self.ttt_cache,
                        ttt_update=False,
                        switch_global_frame=self._switch_global_frame,
                    )
                    next_timestep = self.denoising_step_list[index + 1]
                    noisy_input = self.scheduler.add_noise(
                        denoised_pred.flatten(0, 1),
                        torch.randn_like(denoised_pred.flatten(0, 1)),
                        next_timestep * torch.ones([batch_size * current_num_frames], device=noise.device, dtype=torch.long),
                    ).unflatten(0, denoised_pred.shape[:2])
                else:
                    _, denoised_pred = self.generator(
                        noisy_image_or_video=noisy_input,
                        conditional_dict=cond_in_use,
                        timestep=timestep,
                        kv_cache=self.kv_cache1,
                        crossattn_cache=self.crossattn_cache,
                        current_start=current_start_frame * self.frame_seq_length,
                        rope_enhance=self.rope_enhance,
                        ttt_cache=self.ttt_cache,
                        ttt_update=False,
                        switch_global_frame=self._switch_global_frame,
                    )

            output[:, current_start_frame : current_start_frame + current_num_frames] = denoised_pred.to(output.device)

            context_timestep = torch.ones_like(timestep) * self.args.context_noise
            self.generator(
                noisy_image_or_video=denoised_pred,
                conditional_dict=cond_in_use,
                timestep=context_timestep,
                kv_cache=self.kv_cache1,
                crossattn_cache=self.crossattn_cache,
                current_start=current_start_frame * self.frame_seq_length,
                rope_enhance=self.rope_enhance,
                ttt_cache=self.ttt_cache,
                ttt_update=True,
                switch_global_frame=self._switch_global_frame,
            )

            # T1 (lot offensif 2026-08-04): trace de la masse d'attention par source, un point par
            # chunk genere, par bloc. Sans MEM_ADD_MASS_SRC=1 le report vaut None et rien n'est
            # accumule (aucun cout, aucun changement de sortie).
            if getattr(self, "_add_installed", False):
                _src = self._add_mass_src_report()
                if _src is not None:
                    self._add_mass_log.append({"frame": int(current_start_frame),
                                               "segment": int(seg), "per_block": _src})

            current_start_frame += current_num_frames

        # MEM_ADDITIVE_KV tripwire: how much attention mass did the extra tokens actually take on
        # the last forward? A channel read at ~0 % is dead by construction, not by content.
        _mass = self._add_mass_report()
        if _mass:
            _v = [m for m in _mass.values() if m == m]
            print(f"[MEM_ADDITIVE_KV] attention mass of the anchor tokens on the last forward: "
                  f"mean {sum(_v) / max(1, len(_v)):.4f}, min {min(_v):.4f}, max {max(_v):.4f} "
                  f"| per block " + " ".join(f"{b}:{m:.3f}" for b, m in sorted(_mass.items())),
                  flush=True)
        # T1: masse par source [sink | fenetre | ancre]. Resume + dump complet (bloc x pas).
        _log = getattr(self, "_add_mass_log", None)
        if _log:
            _post = [e for e in _log if e["frame"] > (self._switch_global_frame or 0)]
            _use = _post or _log
            _agg = {k: sum(b[k] for e in _use for b in e["per_block"].values())
                    / max(1, sum(len(e["per_block"]) for e in _use))
                    for k in ("sink", "window", "anchor")}
            _ntok = next(iter(_use[-1]["per_block"].values()))["n_tok"]
            print(f"[MEM_ADD_MASS_SRC] masse d'attention par SOURCE apres le retour "
                  f"({len(_use)} pas x {len(_use[-1]['per_block'])} blocs) : "
                  f"sink {_agg['sink']:.4f} | fenetre {_agg['window']:.4f} | "
                  f"ANCRE {_agg['anchor']:.4f}   (tokens {_ntok} = sink/fenetre/ancre, "
                  f"part uniforme de l'ancre "
                  f"{_ntok[2] / max(1, sum(_ntok)):.4f})", flush=True)
            _dump = os.environ.get("MEM_ADD_MASS_DUMP", "").strip()
            if _dump:
                os.makedirs(os.path.dirname(_dump) or ".", exist_ok=True)
                with open(_dump, "w") as _f:
                    json.dump({"summary_after_return": _agg, "n_tok": _ntok, "steps": _log}, _f)
                print(f"[MEM_ADD_MASS_SRC] trace complete -> {_dump}", flush=True)

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
