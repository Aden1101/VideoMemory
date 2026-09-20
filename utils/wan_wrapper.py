# Adopted from https://github.com/guandeh17/Self-Forcing
# SPDX-License-Identifier: Apache-2.0
import types
from typing import List, Optional
import torch
from torch import nn

from utils.scheduler import SchedulerInterface, FlowMatchScheduler
from wan.modules.tokenizers import HuggingfaceTokenizer
from wan.modules.model import WanModel, RegisterTokens, GanAttentionBlock
from wan.modules.vae import _video_vae
from wan.modules.t5 import umt5_xxl
from wan.modules.causal_model import CausalWanModel
import torch.distributed as dist
from utils.debug_option import DEBUG, MY_VISUAL_DEBUG
from utils.device import device as af_device, hp
from utils import prompt_cache
import os


class WanTextEncoder(torch.nn.Module):
    """Text encoder, with an on-disk cache and optionally lazy construction.

    Two additions, both inert by default on the original install:

    1. `forward` consults the embedding cache first (utils/prompt_cache). A cached prompt
       returns the tensor exactly as it was produced, same dtype, so the output is bitwise
       identical to an uncached run. Always on: it is a pure time saving (encoding one prompt
       costs ~90 s here) with no effect on results.

    2. AF_LAZY_TEXT_ENCODER=1 defers construction of the 11.4 GB until the first prompt that
       misses the cache. With a full cache the encoder is never built and the memory stays with
       the generator. By default the flag is absent and loading stays eager, exactly as before
      , the CUDA path is untouched, including its interaction with DynamicSwapInstaller, which
       walks the submodules at install time.
    """

    def __init__(self) -> None:
        super().__init__()
        self.cache_dir = os.environ.get("AF_PROMPT_CACHE") or prompt_cache.DEFAULT_DIR
        self._lazy = os.environ.get("AF_LAZY_TEXT_ENCODER", "") == "1"
        self.text_encoder = None
        self._resident_device = af_device
        # dtype marker. Inference.py does `pipeline = pipeline.to(dtype=torch.bfloat16)`, which
        # casts this encoder too. In lazy mode it does not exist yet at that moment and would
        # escape the cast; this buffer does undergo it, and tells us which precision the pipeline
        # wants to work in. Without it the cache would hand float32 to a bfloat16 model (Metal
        # assertion, or worse, a silent promotion).
        self.register_buffer("_dtype_probe", torch.zeros((), dtype=torch.float32),
                             persistent=False)
        self.tokenizer = HuggingfaceTokenizer(
            name="wan_models/Wan2.1-T2V-1.3B/google/umt5-xxl/", seq_len=512, clean='whitespace')
        if not self._lazy:
            self._build()

    def _build(self) -> None:
        """Build and load the weights. Called at init (default) or on the first cache miss
        (lazy mode)."""
        if self.text_encoder is not None:
            return
        if self._lazy and (not dist.is_initialized() or dist.get_rank() == 0):
            print("[text_encoder] cache miss -> loading 11.4 GB", flush=True)
        self.text_encoder = umt5_xxl(
            encoder_only=True,
            return_tokenizer=False,
            dtype=torch.float32,
            device=torch.device('cpu')
        ).eval().requires_grad_(False)
        self.text_encoder.load_state_dict(
            torch.load("wan_models/Wan2.1-T2V-1.3B/models_t5_umt5-xxl-enc-bf16.pth",
                       map_location='cpu', weights_only=False)
        )
        
        # The text encoder stays on CPU unless CUDA is present. The test is deliberately
        # `torch.cuda.is_available()` and not "any accelerator": widening it puts umT5-XXL's
        # 11 GB into the accelerator pool, which is painless on a 48 GB card and starves the
        # generator on a Mac's 17.8 GB pool, where the VAE decode then OOMs. Keeping it on CPU
        # is what DynamicSwapInstaller needs: it streams the weights from RAM on each access,
        # and can do nothing for a model already sitting on the device.
        if torch.cuda.is_available():
            self.text_encoder = self.text_encoder.cuda()
        # where the weights actually reside, so inputs can be sent there
        self._resident_device = next(self.text_encoder.parameters()).device


    @property
    def device(self):
        """The device to send inputs to. It is not necessarily the accelerator.

        Three cases, and confusing them gives "Passed CPU tensor to mps op":
          - model not built yet (lazy mode) -> the accelerator, by default;
          - model wrapped by DynamicSwapInstaller -> its weights are streamed to the accelerator
            on every access, so the inputs must be there too;
          - otherwise -> wherever the model actually resides (CPU off CUDA, since the original
            semantics were restored).
        """
        if self.text_encoder is None:
            return af_device
        if "forge_backup_original_class" in self.text_encoder.__dict__:
            return af_device
        return self._resident_device

    def encode(self, text_prompts: List[str]) -> torch.Tensor:
        """The actual encoding, cache-free; `forward` wraps it with the prompt cache."""
        self._build()
        ids, mask = self.tokenizer(
            text_prompts, return_mask=True, add_special_tokens=True)
        ids = ids.to(self.device)
        mask = mask.to(self.device)
        seq_lens = mask.gt(0).sum(dim=1).long()
        context = self.text_encoder(ids, mask)
        # ids = ids.to(torch.device('cpu'))
        # mask = mask.to(torch.device('cpu'))
        for u, v in zip(context, seq_lens):
            u[v:] = 0.0  # set padding to 0.0
        return context

    def forward(self, text_prompts: List[str]) -> dict:
        """Cache first, encoder second, and only for the missing prompts.

        Missing prompts are encoded in a single batch (the encoder works batch-wise) then written
        to the cache. The result is reassembled in input order: shuffling the rows would pair a
        prompt with another one's conditioning, an error that raises no exception and would
        falsify a whole evaluation run."""
        want = self._dtype_probe.dtype
        cached, missing = prompt_cache.get_many(text_prompts, self.cache_dir, want)
        if missing:
            fresh = self.encode([text_prompts[i] for i in missing]).to(want)
            for slot, i in enumerate(missing):
                e = fresh[slot]
                prompt_cache.put(text_prompts[i], e, self.cache_dir)
                cached[i] = e
        context = torch.stack([e.to(device=self.device, dtype=want) for e in cached], dim=0)
        return {
            "prompt_embeds": context
        }


class WanVAEWrapper(torch.nn.Module):
    def __init__(self):
        super().__init__()
        mean = [
            -0.7571, -0.7089, -0.9113, 0.1075, -0.1745, 0.9653, -0.1517, 1.5508,
            0.4134, -0.0715, 0.5517, -0.3632, -0.1922, -0.9497, 0.2503, -0.2921
        ]
        std = [
            2.8184, 1.4541, 2.3275, 2.6558, 1.2196, 1.7708, 2.6052, 2.0743,
            3.2687, 2.1526, 2.8652, 1.5579, 1.6382, 1.1253, 2.8251, 1.9160
        ]
        self.mean = torch.tensor(mean, dtype=torch.float32)
        self.std = torch.tensor(std, dtype=torch.float32)

        # init model
        self.model = _video_vae(
            pretrained_path="wan_models/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth",
            z_dim=16,
        ).eval().requires_grad_(False)

    def encode_to_latent(self, pixel: torch.Tensor) -> torch.Tensor:
        # pixel: [batch_size, num_channels, num_frames, height, width]
        device, dtype = pixel.device, pixel.dtype
        scale = [self.mean.to(device=device, dtype=dtype),
                 1.0 / self.std.to(device=device, dtype=dtype)]

        output = [
            self.model.encode(u.unsqueeze(0), scale).float().squeeze(0)
            for u in pixel
        ]
        output = torch.stack(output, dim=0)
        # from [batch_size, num_channels, num_frames, height, width]
        # to [batch_size, num_frames, num_channels, height, width]
        output = output.permute(0, 2, 1, 3, 4)
        return output

    def decode_to_pixel(self, latent: torch.Tensor, use_cache: bool = False, to_cpu: bool = False) -> torch.Tensor:
        # to_cpu: move each decoded chunk to CPU immediately so the final stack happens in RAM
        # instead of VRAM (long videos, e.g. 240s, OOM otherwise). Default False: re-encode the pixels on GPU
        zs = latent.permute(0, 2, 1, 3, 4)
        if use_cache:
            assert latent.shape[0] == 1, "Batch size must be 1 when using cache"

        device, dtype = latent.device, latent.dtype
        scale = [self.mean.to(device=device, dtype=dtype),
                 1.0 / self.std.to(device=device, dtype=dtype)]

        if use_cache:
            decode_function = self.model.cached_decode
        else:
            decode_function = self.model.decode

        output = []
        for u in zs:
            out = decode_function(u.unsqueeze(0), scale).float().clamp_(-1, 1).squeeze(0)
            output.append(out.cpu() if to_cpu else out)
        output = torch.stack(output, dim=0)
        # from [batch_size, num_channels, num_frames, height, width]
        # to [batch_size, num_frames, num_channels, height, width]
        output = output.permute(0, 2, 1, 3, 4)
        return output


class WanDiffusionWrapper(torch.nn.Module):
    def __init__(
            self,
            model_name="wan_models/Wan2.1-T2V-1.3B",
            timestep_shift=8.0,
            is_causal=False,
            local_attn_size=-1,
            sink_size=0,
            use_gan=False,
            memory_layers=(),
            gate_init=0.1,
            ckpt_write=False,
            gate_extra=None,
            use_decay=False,
            decay_init=0.95,
            bounded_gate=False,
            single_norm=False,
            bounded_mod=False,
            rel_norm=False,
            dc_remove=False,
            muon_update_steps=5,
            mem_stream_scale=0.0,
            ema_alpha=0.0,
            eviction_feed=False,
            mem_state_type="swiglu",   # or "linear_delta": matrix state, training-free delta rule
            mem_linear_beta=0.5
    ):
        super().__init__()
        # 修改为模型路径，方便修改
        if is_causal:
            # low_cpu_mem_usage=False: the pretrained Wan checkpoint predates the memory
            # branch (blocks.5.memory.*), diffusers must build the full model first so the
            # missing memory keys keep their __init__ initialization (Kaiming + gate 0)
            self.model = CausalWanModel.from_pretrained(
                model_name, local_attn_size=local_attn_size, sink_size=sink_size,
                memory_layers=memory_layers, gate_init=gate_init, ckpt_write=ckpt_write,
                gate_extra=gate_extra, use_decay=use_decay, decay_init=decay_init,
                bounded_gate=bounded_gate, single_norm=single_norm, bounded_mod=bounded_mod, rel_norm=rel_norm, dc_remove=dc_remove, muon_update_steps=muon_update_steps,
                mem_stream_scale=mem_stream_scale, ema_alpha=ema_alpha,
                eviction_feed=eviction_feed,
                mem_state_type=mem_state_type, mem_linear_beta=mem_linear_beta,
                low_cpu_mem_usage=False, device_map=None)
        else:
            self.model = WanModel.from_pretrained(model_name)
            if use_gan:
                print("[use_gan] activate")
                self.adding_cls_branch()

        self.model.eval()

        # For non-causal diffusion, all frames share the same timestep
        self.uniform_timestep = not is_causal

        self.scheduler = FlowMatchScheduler(
            shift=timestep_shift, sigma_min=0.0, extra_one_step=True
        )
        self.scheduler.set_timesteps(1000, training=True)

        # self.seq_len = 1560 * local_attn_size if local_attn_size != -1 else 32760 # [1, 21, 16, 60, 104]
        self.seq_len = 1560 * local_attn_size if local_attn_size > 21 else 32760 # [1, 21, 16, 60, 104]
        self.post_init()

    def enable_gradient_checkpointing(self) -> None:
        self.model.enable_gradient_checkpointing()

    def adding_cls_branch(self, atten_dim=1536, num_class=1, time_embed_dim=0) -> None:
        # NOTE: This is hard coded for WAN2.1-T2V-1.3B for now!!!!!!!!!!!!!!!!!!!!
        self._cls_pred_branch = nn.Sequential(
            # Input: [B, 384, 21, 60, 104]
            nn.LayerNorm(atten_dim * 3 + time_embed_dim),
            nn.Linear(atten_dim * 3 + time_embed_dim, 1536),
            nn.SiLU(),
            nn.Linear(atten_dim, num_class)
        )
        self._cls_pred_branch.requires_grad_(True)

        num_registers = 3
        self._register_tokens = RegisterTokens(num_registers=num_registers, dim=atten_dim)
        self._register_tokens.requires_grad_(True)

        gan_ca_blocks = []
        for _ in range(num_registers):
            block = GanAttentionBlock()
            gan_ca_blocks.append(block)
        self._gan_ca_blocks = nn.ModuleList(gan_ca_blocks)
        self._gan_ca_blocks.requires_grad_(True)
        # self.has_cls_branch = True

    def _convert_flow_pred_to_x0(self, flow_pred: torch.Tensor, xt: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        """
        Convert flow matching's prediction to x0 prediction.
        flow_pred: the prediction with shape [B, C, H, W]
        xt: the input noisy data with shape [B, C, H, W]
        timestep: the timestep with shape [B]

        pred = noise - x0
        x_t = (1-sigma_t) * x0 + sigma_t * noise
        we have x0 = x_t - sigma_t * pred
        see derivations https://chatgpt.com/share/67bf8589-3d04-8008-bc6e-4cf1a24e2d0e
        """
        # use higher precision for calculations
        original_dtype = flow_pred.dtype
        flow_pred, xt, sigmas, timesteps = map(
            lambda x: hp(x, flow_pred.device), [flow_pred, xt,
                                                        self.scheduler.sigmas,
                                                        self.scheduler.timesteps]
        )

        timestep_id = torch.argmin(
            (timesteps.unsqueeze(0) - timestep.unsqueeze(1)).abs(), dim=1)
        sigma_t = sigmas[timestep_id].reshape(-1, 1, 1, 1)
        x0_pred = xt - sigma_t * flow_pred
        return x0_pred.to(original_dtype)

    @staticmethod
    def _convert_x0_to_flow_pred(scheduler, x0_pred: torch.Tensor, xt: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        """
        Convert x0 prediction to flow matching's prediction.
        x0_pred: the x0 prediction with shape [B, C, H, W]
        xt: the input noisy data with shape [B, C, H, W]
        timestep: the timestep with shape [B]

        pred = (x_t - x_0) / sigma_t
        """
        # use higher precision for calculations
        original_dtype = x0_pred.dtype
        x0_pred, xt, sigmas, timesteps = map(
            lambda x: hp(x, x0_pred.device), [x0_pred, xt,
                                                      scheduler.sigmas,
                                                      scheduler.timesteps]
        )
        timestep_id = torch.argmin(
            (timesteps.unsqueeze(0) - timestep.unsqueeze(1)).abs(), dim=1)
        sigma_t = sigmas[timestep_id].reshape(-1, 1, 1, 1)
        flow_pred = (xt - x0_pred) / sigma_t
        return flow_pred.to(original_dtype)

    def forward(
        self,
        noisy_image_or_video: torch.Tensor, conditional_dict: dict,
        timestep: torch.Tensor, kv_cache: Optional[List[dict]] = None,
        crossattn_cache: Optional[List[dict]] = None,
        current_start: Optional[int] = None,
        classify_mode: Optional[bool] = False,
        concat_time_embeddings: Optional[bool] = False,
        clean_x: Optional[torch.Tensor] = None,
        aug_t: Optional[torch.Tensor] = None,
        cache_start: Optional[int] = None,
        sink_recache_after_switch=False,
        flow_cache=None,
        rope_enhance=False,
        prompt_cache=None,
        new_prompt_chunk_flag=False,
        ttt_cache=None,
        ttt_update=False,
        switch_global_frame=None,
    ) -> torch.Tensor:
        prompt_embeds = conditional_dict["prompt_embeds"]

        # [B, F] -> [B]
        if self.uniform_timestep:
            input_timestep = timestep[:, 0]
        else:
            input_timestep = timestep

        logits = None
        # X0 prediction
        if kv_cache is not None:  # generator
            flow_pred = self.model(
                noisy_image_or_video.permute(0, 2, 1, 3, 4),
                t=input_timestep, context=prompt_embeds,
                seq_len=self.seq_len,
                kv_cache=kv_cache,
                crossattn_cache=crossattn_cache,
                current_start=current_start,
                cache_start=cache_start,
                sink_recache_after_switch=sink_recache_after_switch,
                flow_cache=flow_cache,
                rope_enhance=rope_enhance,
                prompt_cache=prompt_cache,
                new_prompt_chunk_flag=new_prompt_chunk_flag,
                ttt_cache=ttt_cache,
                ttt_update=ttt_update,
                switch_global_frame=switch_global_frame,
            ).permute(0, 2, 1, 3, 4)
        else:
            if clean_x is not None:
                # teacher forcing
                flow_pred = self.model(
                    noisy_image_or_video.permute(0, 2, 1, 3, 4),
                    t=input_timestep, context=prompt_embeds,
                    seq_len=self.seq_len,
                    clean_x=clean_x.permute(0, 2, 1, 3, 4),
                    aug_t=aug_t,
                    # sink_recache_after_switch=sink_recache_after_switch
                ).permute(0, 2, 1, 3, 4)

            else:
                if classify_mode: # gan
                    flow_pred, logits = self.model(
                        noisy_image_or_video.permute(0, 2, 1, 3, 4),
                        t=input_timestep, context=prompt_embeds,
                        seq_len=self.seq_len,
                        classify_mode=True,
                        register_tokens=self._register_tokens,
                        cls_pred_branch=self._cls_pred_branch,
                        gan_ca_blocks=self._gan_ca_blocks,
                        concat_time_embeddings=concat_time_embeddings,
                        # sink_recache_after_switch=sink_recache_after_switch
                    )
                    flow_pred = flow_pred.permute(0, 2, 1, 3, 4)
                else:  # teacher & student
                    flow_pred = self.model(
                        noisy_image_or_video.permute(0, 2, 1, 3, 4),
                        t=input_timestep, context=prompt_embeds,
                        seq_len=self.seq_len,
                        # sink_recache_after_switch=sink_recache_after_switch
                    ).permute(0, 2, 1, 3, 4)

        pred_x0 = self._convert_flow_pred_to_x0(
            flow_pred=flow_pred.flatten(0, 1),
            xt=noisy_image_or_video.flatten(0, 1),
            timestep=timestep.flatten(0, 1)
        ).unflatten(0, flow_pred.shape[:2])

        if logits is not None:
            return flow_pred, pred_x0, logits

        return flow_pred, pred_x0

    def get_scheduler(self) -> SchedulerInterface:
        """
        Update the current scheduler with the interface's static method
        """
        scheduler = self.scheduler
        scheduler.convert_x0_to_noise = types.MethodType(
            SchedulerInterface.convert_x0_to_noise, scheduler)
        scheduler.convert_noise_to_x0 = types.MethodType(
            SchedulerInterface.convert_noise_to_x0, scheduler)
        scheduler.convert_velocity_to_x0 = types.MethodType(
            SchedulerInterface.convert_velocity_to_x0, scheduler)
        self.scheduler = scheduler
        return scheduler

    def post_init(self):
        """
        A few custom initialization steps that should be called after the object is created.
        Currently, the only one we have is to bind a few methods to scheduler.
        We can gradually add more methods here if needed.
        """
        self.get_scheduler()
