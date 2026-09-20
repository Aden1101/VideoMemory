# Adopted from https://github.com/guandeh17/Self-Forcing
# SPDX-License-Identifier: Apache-2.0
import gc
import logging
import math
import random
import re
from pathlib import Path

from utils.dataset import TextDataset, TwoTextDataset, TwoTextVideoPairDataset, cycle
from utils.distributed import EMA_FSDP, fsdp_wrap, fsdp_state_dict, launch_distributed_job
from utils.misc import (
    set_seed,
    merge_dict_list
)
import torch.distributed as dist
from omegaconf import OmegaConf
from model import DMD, DMDSwitch
from model.streaming_training import StreamingTrainingModel
import torch
import time
import os
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import (
    StateDictType, FullStateDictConfig, FullOptimStateDictConfig
)
from utils.video_io import write_video  # torchvision when available, else imageio

# LoRA related imports
import peft
from peft import get_peft_model_state_dict
import safetensors.torch

from utils.memory import gpu, get_cuda_free_memory_gb, log_gpu_memory
from pipeline import (
    CausalInferencePipeline,
    SwitchCausalInferencePipeline
)
from utils.debug_option import DEBUG, LOG_GPU_MEMORY, MY_VISUAL_DEBUG
# from one_logger_utils import OneLoggerUtils
import time
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm


class Trainer:
    
    def __init__(self, config):
        self.config = config
        self.step = 0

        # Step 1: Initialize the distributed training environment (rank, seed, dtype, logging etc.)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

        launch_distributed_job()
        global_rank = dist.get_rank()
        self.world_size = dist.get_world_size()

        self.dtype = torch.bfloat16 if config.mixed_precision else torch.float32
        self.device = torch.cuda.current_device()
        self.is_main_process = global_rank == 0
        self.causal = config.causal
        if self.is_main_process:
            # wandb.init must come before SummaryWriter so sync_tensorboard can patch the TB writer
            # (otherwise only wandb's auto system metrics show, not our add_scalar curves).
            if getattr(config, "use_wandb", False):
                import wandb
                run_name = os.path.basename(config.logdir.rstrip("/"))
                wandb.init(
                    project=getattr(config, "wandb_project", "anchor-forcing-memory"),
                    name=run_name,
                    id=run_name,
                    resume="allow",
                    dir=config.logdir,
                    config=OmegaConf.to_container(config, resolve=True),
                    sync_tensorboard=True,
                )
            self.writer = SummaryWriter(log_dir=config.logdir)

        # use a random seed for the training
        if config.seed == 0:
            random_seed = torch.randint(0, 10000000, (1,), device=self.device)
            dist.broadcast(random_seed, src=0)
            config.seed = random_seed.item()

        set_seed(config.seed + global_rank)

        self.output_path = config.logdir

        # Step 2: Initialize the model

        if config.distribution_loss == "causvid":
            self.model = CausVid(config, device=self.device)
        elif config.distribution_loss == "dmd":
            self.model = DMD(config, device=self.device)
        elif config.distribution_loss == "dmd_switch":
            self.model = DMDSwitch(config, device=self.device)
        elif config.distribution_loss == "dmd_window":
            self.model = DMDWindow(config, device=self.device)
        elif config.distribution_loss == "sid":
            self.model = SiD(config, device=self.device)
        else:
            raise ValueError("Invalid distribution matching loss")

        # Save pretrained model state_dicts to CPU
        self.fake_score_state_dict_cpu = self.model.fake_score.state_dict()

        # Auto resume configuration (needed for LoRA checkpoint loading)
        auto_resume = getattr(config, "auto_resume", True)  # Default to True

        # ================================= LoRA Configuration =================================
        self.is_lora_enabled = False
        self.lora_config = None
        if hasattr(config, 'adapter') and config.adapter is not None:
            self.is_lora_enabled = True
            self.lora_config = config.adapter
            
            if self.is_main_process:
                print(f"LoRA enabled with config: {self.lora_config}")
                print("Loading base model and applying LoRA before FSDP wrapping...")
            
            # 1. Load base model first (config.generator_ckpt) - before applying LoRA and FSDP
            base_checkpoint_path = getattr(config, "generator_ckpt", None)
            if base_checkpoint_path:
                if self.is_main_process:
                    print(f"Loading base model from {base_checkpoint_path} (before applying LoRA)")
                base_checkpoint = torch.load(base_checkpoint_path, map_location="cpu")
                
                # Load generator (directly; no key alignment needed since LoRA not applied yet)
                if "generator" in base_checkpoint:
                    if self.is_main_process:
                        print(f"Loading pretrained generator from {base_checkpoint_path}")
                    # strict=False: the base checkpoint predates the memory branch; only acceptable missing keys are memory params
                    result = self.model.generator.load_state_dict(base_checkpoint["generator"], strict=False)
                    non_memory_missing = [k for k in result.missing_keys if "memory" not in k]
                    assert not non_memory_missing, f"Missing non-memory keys: {non_memory_missing[:5]}"
                    assert not result.unexpected_keys, f"Unexpected keys: {list(result.unexpected_keys)[:5]}"
                    if self.is_main_process:
                        print(f"Generator weights loaded successfully ({len(result.missing_keys)} memory keys kept at init)")
                elif "model" in base_checkpoint:
                    if self.is_main_process:
                        print(f"Loading pretrained generator from {base_checkpoint_path}")
                    result = self.model.generator.load_state_dict(base_checkpoint["model"], strict=False)
                    non_memory_missing = [k for k in result.missing_keys if "memory" not in k]
                    assert not non_memory_missing, f"Missing non-memory keys: {non_memory_missing[:5]}"
                    assert not result.unexpected_keys, f"Unexpected keys: {list(result.unexpected_keys)[:5]}"
                    if self.is_main_process:
                        print(f"Generator weights loaded successfully ({len(result.missing_keys)} memory keys kept at init)")
                else:
                    if self.is_main_process:
                        print("Warning: Generator checkpoint not found in base model.")
                
                # Load critic
                if "critic" in base_checkpoint:
                    if self.is_main_process:
                        print(f"Loading pretrained critic from {base_checkpoint_path}")
                    result = self.model.fake_score.load_state_dict(base_checkpoint["critic"], strict=True)
                    if self.is_main_process:
                        print("Critic weights loaded successfully")
                else:
                    if self.is_main_process:
                        print("Warning: Critic checkpoint not found in base model.")
            else:
                if self.is_main_process:
                    raise ValueError("No base model checkpoint specified for LoRA training.")
            
            # Load training step
            if "step" in base_checkpoint:
                self.step = base_checkpoint["step"]
                if self.is_main_process:
                    print(f"base_checkpoint step: {self.step}")
            else:
                if self.is_main_process:
                    print("Warning: Step not found in checkpoint, starting from step 0.")
            
            # 2. Apply LoRA wrapping now (after loading base model, before FSDP wrapping)
            if self.is_main_process:
                print("Applying LoRA to models...")
            self.model.generator.model = self._configure_lora_for_model(self.model.generator.model, "generator")
            
            # Configure LoRA for fake_score if needed
            if getattr(self.lora_config, 'apply_to_critic', True):
                self.model.fake_score.model = self._configure_lora_for_model(self.model.fake_score.model, "fake_score")
                if self.is_main_process:
                    print("LoRA applied to both generator and critic")
            else:
                if self.is_main_process:
                    print("LoRA applied to generator only")
            
            # 3. Load LoRA weights before FSDP wrapping (if a checkpoint is available)
            # lora_checkpoint_path = None
            lora_checkpoint_path = getattr(config, "lora_checkpoint_path", None)
            if auto_resume and self.output_path:
                # Find the latest checkpoint and verify it is a LoRA checkpoint
                latest_checkpoint = self.find_latest_checkpoint(self.output_path)
                if latest_checkpoint:
                    try:
                        checkpoint = torch.load(latest_checkpoint, map_location="cpu")
                        if "generator_lora" in checkpoint and "critic_lora" in checkpoint:
                            lora_checkpoint_path = latest_checkpoint
                            if self.is_main_process:
                                print(f"Auto resume: Found LoRA checkpoint at {lora_checkpoint_path}")
                        else:
                            raise ValueError(f"Checkpoint {latest_checkpoint} is not a LoRA checkpoint. "
                                           f"Found keys: {list(checkpoint.keys())}")
                    except Exception as e:
                        if self.is_main_process:
                            print(f"Error validating checkpoint: {e}")
                        raise e
                else:
                    if self.is_main_process:
                        print("Auto resume: No LoRA checkpoint found in logdir")
            elif auto_resume:
                if self.is_main_process:
                    print("Auto resume enabled but no logdir specified for LoRA")
            else:
                if self.is_main_process:
                    print("Auto resume disabled for LoRA")
            
            # If no auto-resumed LoRA checkpoint found, try config.lora_ckpt
            if lora_checkpoint_path is None:
                lora_ckpt_path = getattr(config, "lora_ckpt", None)
                if lora_ckpt_path:
                    try:
                        checkpoint = torch.load(lora_ckpt_path, map_location="cpu")
                        if "generator_lora" in checkpoint and "critic_lora" in checkpoint:
                            lora_checkpoint_path = lora_ckpt_path
                            if self.is_main_process:
                                print(f"Using explicit LoRA checkpoint: {lora_checkpoint_path}")
                        else:
                            raise ValueError(f"Explicit LoRA checkpoint {lora_ckpt_path} is not a valid LoRA checkpoint. "
                                           f"Found keys: {list(checkpoint.keys())}")
                    except Exception as e:
                        if self.is_main_process:
                            print(f"Error loading explicit LoRA checkpoint: {e}")
                        raise e
                else:
                    if self.is_main_process:
                        print("No LoRA checkpoint specified, starting LoRA training from scratch")
            
            # Load LoRA checkpoint (before FSDP wrapping)
            if lora_checkpoint_path:
                if self.is_main_process:
                    print(f"Loading LoRA checkpoint from {lora_checkpoint_path} (before FSDP wrapping)")
                lora_checkpoint = torch.load(lora_checkpoint_path, map_location="cpu")
                
                # Load LoRA weights using PEFT's standard method
                if "generator_lora" in lora_checkpoint:
                    if self.is_main_process:
                        print(f"Loading LoRA generator weights: {len(lora_checkpoint['generator_lora'])} keys in checkpoint")
                    
                    # Use PEFT's set_peft_model_state_dict; it automatically aligns key names
                    peft.set_peft_model_state_dict(self.model.generator.model, lora_checkpoint["generator_lora"])
                
                if "critic_lora" in lora_checkpoint:
                    if self.is_main_process:
                        print(f"Loading LoRA critic weights: {len(lora_checkpoint['critic_lora'])} keys in checkpoint")

                    # Use PEFT's set_peft_model_state_dict; it automatically aligns key names
                    peft.set_peft_model_state_dict(self.model.fake_score.model, lora_checkpoint["critic_lora"])

                # memory params, restore them too,
                if "generator_memory" in lora_checkpoint:
                    if self.is_main_process:
                        print(f"Loading memory weights: {len(lora_checkpoint['generator_memory'])} keys in checkpoint")
                    result = self.model.generator.load_state_dict(lora_checkpoint["generator_memory"], strict=False)
                    assert not result.unexpected_keys, f"Unexpected memory keys: {list(result.unexpected_keys)[:5]}"
                    still_missing_mem = [k for k in result.missing_keys if "memory" in k]
                    assert not still_missing_mem, f"Memory NOT loaded (would stay at init!): {still_missing_mem[:5]}"

                # stash the EMA shadow (crash-resume must keep the accumulated smoothing, not
                # re-snapshot the raw resumed weights). Applied after the EMA is (re)created below.
                if "generator_ema" in lora_checkpoint:
                    self._resumed_ema_shadow = lora_checkpoint["generator_ema"]
                    if self.is_main_process:
                        print(f"Stashed EMA shadow from checkpoint: {len(self._resumed_ema_shadow)} keys "
                              f"(applied after EMA setup)")

                # Load training step
                resume_step = getattr(config, "resume_step", True)
                if "step" in lora_checkpoint and resume_step:
                    self.step = lora_checkpoint["step"]
                    if self.is_main_process:
                        print(f"Resuming LoRA training from step {self.step}")
            else:
                if self.is_main_process:
                    print("No LoRA checkpoint to load, starting from scratch")

        # memory params are new, full-rank and trainable (excluded from LoRA targets).
        # peft freezes every non-LoRA param, so we un-freeze them here.
        for n, p in self.model.generator.named_parameters():
            if "memory" in n:
                p.requires_grad_(True)

        # Selective LoRA unfreeze: only blocks >= lora_unfreeze_from_block (which consume the memory
        # injection) stay trainable; earlier blocks' LoRA is frozen, dropping their grad + AdamW state
        # (frees their VRAM). None = all LoRA trainable. Critic LoRA untouched.
        _unfreeze_from = getattr(config, "lora_unfreeze_from_block", None)
        if _unfreeze_from is not None:
            import re as _re
            _frozen_blocks, _kept_blocks, _kept = set(), set(), 0
            for n, p in self.model.generator.named_parameters():
                if "memory" in n or not p.requires_grad:
                    continue
                _m = _re.search(r"blocks\.(\d+)\.", n)
                if _m is None or int(_m.group(1)) < int(_unfreeze_from):
                    p.requires_grad_(False)
                    if _m is not None:
                        _frozen_blocks.add(int(_m.group(1)))
                else:
                    _kept += 1
                    _kept_blocks.add(int(_m.group(1)))
            if self.is_main_process:
                print(f"[selective LoRA] unfreeze from block {_unfreeze_from}: "
                      f"froze LoRA of blocks {sorted(_frozen_blocks)}, "
                      f"kept {_kept} trainable LoRA tensors across blocks {sorted(_kept_blocks)}")

        # fast weights (w0/w1/w2) are excluded from FSDP
        mem_ignored = []
        if self.world_size > 1:
            mem_ignored = [m for m in self.model.generator.modules()
                           if type(m).__name__ == "VideoMemoryLayer"]
            for m in mem_ignored:
                m.to(device=self.device, dtype=self.dtype)
            # memory is excluded from FSDP: sync the per-rank divergent init from rank 0 so all ranks
            # share one memory (its grads are averaged before each optimizer step)
            for m in mem_ignored:
                for _p in m.parameters():
                    dist.broadcast(_p.data, src=0)
                for _b in m.buffers():
                    dist.broadcast(_b.data, src=0)

        self.model.generator = fsdp_wrap(
            self.model.generator,
            sharding_strategy=config.sharding_strategy,
            mixed_precision=config.mixed_precision,
            wrap_strategy=config.generator_fsdp_wrap_strategy,
            ignored_modules=mem_ignored or None
        )

        self.model.real_score = fsdp_wrap(
            self.model.real_score,
            sharding_strategy=config.sharding_strategy,
            mixed_precision=config.mixed_precision,
            wrap_strategy=config.real_score_fsdp_wrap_strategy,
            # Mirrors text_encoder_cpu_offload below; default False = unchanged behavior.
            cpu_offload=getattr(config, "real_score_cpu_offload", False)
        )

        self.model.fake_score = fsdp_wrap(
            self.model.fake_score,
            sharding_strategy=config.sharding_strategy,
            mixed_precision=config.mixed_precision,
            wrap_strategy=config.fake_score_fsdp_wrap_strategy,
            # Mirrors real_score_cpu_offload; default False = unchanged behavior.
            # The critic is trainable, so this offloads its params/grads/optim to CPU
            # (frees ~2.6GB/rank of peak VRAM) at the cost of slower critic steps.
            cpu_offload=getattr(config, "fake_score_cpu_offload", False)
        )

        self.model.text_encoder = fsdp_wrap(
            self.model.text_encoder,
            sharding_strategy=config.sharding_strategy,
            mixed_precision=config.mixed_precision,
            wrap_strategy=config.text_encoder_fsdp_wrap_strategy,
            cpu_offload=getattr(config, "text_encoder_cpu_offload", False)
        )
        self.model.vae = self.model.vae.to(
            device=self.device, dtype=torch.bfloat16 if config.mixed_precision else torch.float32)

        # if not config.no_visualize or config.load_raw_video:
        #     print("Moving vae to device 2, self.device: ", self.device)
        #     self.model.vae = self.model.vae.to(
        #         device=self.device, dtype=torch.bfloat16 if config.mixed_precision else torch.float32)

        # Step 3: Set up EMA parameter containers
        rename_param = (
            lambda name: name.replace("_fsdp_wrapped_module.", "")
            .replace("_checkpoint_wrapped_module.", "")
            .replace("_orig_mod.", "")
        )
        self.name_to_trainable_params = {}
        for n, p in self.model.generator.named_parameters():
            if not p.requires_grad:
                continue

            renamed_n = rename_param(n)
            self.name_to_trainable_params[renamed_n] = p
        ema_weight = config.ema_weight
        self.generator_ema = None
        if (ema_weight is not None) and (ema_weight > 0.0):
            # EMA runs in LoRA mode too, as a trainable-only shadow (EMA_FSDP skips
            # non-requires_grad params). It damps the late-training slow-mem-param ramp that
            # drives the flicker. It does not alter the training trajectory (a read-only CPU
            # shadow updated after optimizer.step); eval must set use_ema:true to load it.
            if self.is_main_process:
                _mode = "trainable-only (LoRA)" if self.is_lora_enabled else "full"
                print(f"Setting up {_mode} EMA with weight {ema_weight}")
            self.generator_ema = EMA_FSDP(self.model.generator, decay=ema_weight)
            # restore a stashed shadow from a resume checkpoint (keeps accumulated smoothing
            # instead of the raw resumed weights EMA_FSDP just snapshotted).
            _stashed = getattr(self, "_resumed_ema_shadow", None)
            if _stashed is not None:
                self.generator_ema.load_state_dict(_stashed)
                self._resumed_ema_shadow = None
                if self.is_main_process:
                    print(f"Restored EMA shadow from resume checkpoint ({len(_stashed)} keys)")

        
        
        # Step 4: Initialize the optimizer
        # memory params (full-rank, excluded from LoRA, un-frozen before the FSDP wrap), get their own LR group.  
        gen = self.model.generator
        mem_params = [p for n, p in gen.named_parameters() if "memory" in n and p.requires_grad]
        other_params = [p for n, p in gen.named_parameters() if "memory" not in n and p.requires_grad]
        # safety net
        assert mem_params or not any("memory" in n for n, _ in gen.named_parameters()), \
            "memory params exist but none landed in the optimizer group (name-match broke post-FSDP)"
        generator_param_groups = [{"params": other_params, "lr": config.lr}]  # LoRA adapters
        if mem_params:  # no memory layers -> single group, strictly identical to baseline
            generator_param_groups.append({"params": mem_params, "lr": getattr(config, "lr_mem", 1e-4)})
        self.generator_optimizer = torch.optim.AdamW(
            generator_param_groups,
            betas=(config.beta1, config.beta2),
            weight_decay=config.weight_decay
        )
        # handles for (C) explicit mem-grad clip and (B) cosine lr schedule.
        # mem params are FSDP-ignored, so generator.clip_grad_norm_() (an FSDP method) skips
        # them and this is the only clip they get. Default is effectively off (1e9).
        self.mem_params = mem_params
        self.max_grad_norm_mem = float(getattr(config, "max_grad_norm_mem", 1e9))
        # per-group base lrs [LoRA, (mem)]; the cosine multiplier is applied to these each step.
        self._base_lrs = [g["lr"] for g in self.generator_optimizer.param_groups]
        # lr_decay_start defaults beyond max_iters => schedule never fires => lr constant (byte-identical).
        self._lr_decay_start = int(getattr(config, "lr_decay_start", 10**9))
        self._lr_floor_frac = float(getattr(config, "lr_floor_frac", 1.0))       # LoRA group floor
        self._lr_mem_floor_frac = float(getattr(config, "lr_mem_floor_frac", 1.0))  # mem group floor

        self.critic_optimizer = torch.optim.AdamW(
            [param for param in self.model.fake_score.parameters()
             if param.requires_grad],
            lr=config.lr_critic if hasattr(config, "lr_critic") else config.lr,
            betas=(config.beta1_critic, config.beta2_critic),
            weight_decay=config.weight_decay
        )

        # Step 5: Initialize the dataloader
        if self.config.i2v:
            dataset = ShardingLMDBDataset(config.data_path, max_pair=int(1e8))
        elif self.config.distribution_loss == "dmd_switch":
            if config.use_gan:
                dataset = TwoTextVideoPairDataset(config.data_path, config.switch_prompt_path, config.video_json_path)
            else:
                dataset = TwoTextDataset(config.data_path, config.switch_prompt_path)
        else:
            dataset = TextDataset(config.data_path)
        sampler = torch.utils.data.distributed.DistributedSampler(
            dataset, shuffle=True, drop_last=True)
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=config.batch_size,
            sampler=sampler,
            num_workers=8)

        if dist.get_rank() == 0:
            print("DATASET SIZE %d" % len(dataset))
        self.dataloader = cycle(dataloader)

        # Step 6: Initialize the validation dataloader for visualization (fixed prompts)
        self.fixed_vis_batch = None
        self.vis_interval = getattr(config, "vis_interval", -1)
        if self.vis_interval > 0 and len(getattr(config, "vis_video_lengths", [])) > 0:
            # Determine validation data path
            val_data_path = getattr(config, "val_data_path", None) or config.data_path

            if self.config.i2v:
                val_dataset = ShardingLMDBDataset(val_data_path, max_pair=int(1e8))
            elif self.config.distribution_loss == "dmd_switch":
                val_dataset = TwoTextDataset(val_data_path, config.val_switch_prompt_path)
            else:
                val_dataset = TextDataset(val_data_path)

            if dist.get_rank() == 0:
                print("VAL DATASET SIZE %d" % len(val_dataset))

            sampler = torch.utils.data.distributed.DistributedSampler(
                val_dataset, shuffle=False, drop_last=False)
            # streaming sampling to keep prompts fixed
            val_dataloader = torch.utils.data.DataLoader(
                val_dataset,
                batch_size=getattr(config, "val_batch_size", 1),
                sampler=sampler,
                num_workers=8,
            )

            # Take the first batch as fixed visualization batch
            try:
                self.fixed_vis_batch = next(iter(val_dataloader))
            except StopIteration:
                self.fixed_vis_batch = None
            
            # ----------------------------------------------------------------------------------------------------------
            # Visualization settings
            # ----------------------------------------------------------------------------------------------------------
            # List of video lengths to visualize, e.g. [8, 16, 32]
            self.vis_video_lengths = getattr(config, "vis_video_lengths", [])

            if self.vis_interval > 0 and len(self.vis_video_lengths) > 0:
                self._setup_visualizer()
            
        if not self.is_lora_enabled:
            # ================================= Standard (non-LoRA) model logic =================================
            checkpoint_path = None
            
            if auto_resume and self.output_path:
                # Auto resume: find latest checkpoint in logdir
                latest_checkpoint = self.find_latest_checkpoint(self.output_path)
                if latest_checkpoint:
                    checkpoint_path = latest_checkpoint
                    if self.is_main_process:
                        print(f"Auto resume: Found latest checkpoint at {checkpoint_path}")
                else:
                    if self.is_main_process:
                        print("Auto resume: No checkpoint found in logdir, starting from scratch")
            elif auto_resume:
                if self.is_main_process:
                    print("Auto resume enabled but no logdir specified, starting from scratch")
            else:
                if self.is_main_process:
                    print("Auto resume disabled, starting from scratch")
            
            if checkpoint_path is None:
                if getattr(config, "generator_ckpt", False):
                    # Explicit checkpoint path provided
                    checkpoint_path = config.generator_ckpt
                    if self.is_main_process:
                        print(f"Using explicit checkpoint: {checkpoint_path}")

            if checkpoint_path:
                if self.is_main_process:
                    print(f"Loading checkpoint from {checkpoint_path}")
                checkpoint = torch.load(checkpoint_path, map_location="cpu")
                
                # Load generator
                if "generator" in checkpoint:
                    if self.is_main_process:
                        print(f"Loading pretrained generator from {checkpoint_path}")
                    # tolerate checkpoints that predate the memory branch
                    result = self.model.generator.load_state_dict(checkpoint["generator"], strict=False)
                    non_memory_missing = [k for k in result.missing_keys if "memory" not in k]
                    assert not non_memory_missing, f"Missing non-memory keys: {non_memory_missing[:5]}"
                    assert not result.unexpected_keys, f"Unexpected keys: {list(result.unexpected_keys)[:5]}"
                elif "model" in checkpoint:
                    if self.is_main_process:
                        print(f"Loading pretrained generator from {checkpoint_path}")
                    result = self.model.generator.load_state_dict(checkpoint["model"], strict=False)
                    non_memory_missing = [k for k in result.missing_keys if "memory" not in k]
                    assert not non_memory_missing, f"Missing non-memory keys: {non_memory_missing[:5]}"
                    assert not result.unexpected_keys, f"Unexpected keys: {list(result.unexpected_keys)[:5]}"
                else:
                    if self.is_main_process:
                        print("Warning: Generator checkpoint not found.")
                
                # Load critic
                if "critic" in checkpoint:
                    if self.is_main_process:
                        print(f"Loading pretrained critic from {checkpoint_path}")
                    self.model.fake_score.load_state_dict(checkpoint["critic"], strict=True)
                else:
                    if self.is_main_process:
                        print("Warning: Critic checkpoint not found.")
                
                # Load EMA
                if "generator_ema" in checkpoint and self.generator_ema is not None:
                    if self.is_main_process:
                        print(f"Loading pretrained EMA from {checkpoint_path}")
                    self.generator_ema.load_state_dict(checkpoint["generator_ema"])
                else:
                    if self.is_main_process:
                        print("Warning: EMA checkpoint not found or EMA not initialized.")
                
                # For auto resume, always resume full training state
                # Load optimizers
                if "generator_optimizer" in checkpoint:
                    if self.is_main_process:
                        print("Resuming generator optimizer...")
                    gen_osd = FSDP.optim_state_dict_to_load(
                        self.model.generator,              # FSDP root module
                        self.generator_optimizer,          # newly created optimizer
                        checkpoint["generator_optimizer"]  # optimizer state dict at save time
                    )
                    self.generator_optimizer.load_state_dict(gen_osd)
                else:
                    if self.is_main_process:
                        print("Warning: Generator optimizer checkpoint not found.")
                
                if "critic_optimizer" in checkpoint:
                    if self.is_main_process:
                        print("Resuming critic optimizer...")
                    crit_osd = FSDP.optim_state_dict_to_load(
                        self.model.fake_score,
                        self.critic_optimizer,
                        checkpoint["critic_optimizer"]
                    )
                    self.critic_optimizer.load_state_dict(crit_osd)
                else:
                    if self.is_main_process:
                        print("Warning: Critic optimizer checkpoint not found.")
                
                # Load training step
                if "step" in checkpoint:
                    self.step = checkpoint["step"]
                    if self.is_main_process:
                        print(f"Resuming from step {self.step}")
                else:
                    if self.is_main_process:
                        print("Warning: Step not found in checkpoint, starting from step 0.")

        ##############################################################################################################

        # Let's delete EMA params for early steps to save some computes at training and inference
        # Note: This should be done after potential resume to avoid accidentally deleting resumed EMA
        if self.step < config.ema_start_step:
            self.generator_ema = None

        self.max_grad_norm_generator = getattr(config, "max_grad_norm_generator", 10.0)
        self.max_grad_norm_critic = getattr(config, "max_grad_norm_critic", 10.0)
        self.gradient_accumulation_steps = getattr(config, "gradient_accumulation_steps", 1)
        self.previous_time = None
        
        # streaming training configuration
        self.streaming_training = getattr(config, "streaming_training", False)
        self.streaming_chunk_size = getattr(config, "streaming_chunk_size", 21)
        self.streaming_max_length = getattr(config, "streaming_max_length", 63)
        
        # Create streaming training model if enabled
        if self.streaming_training:
            self.streaming_model = StreamingTrainingModel(self.model, config)
            if self.is_main_process:
                print(f"streaming training enabled: chunk_size={self.streaming_chunk_size}, max_length={self.streaming_max_length}")
        else:
            self.streaming_model = None
        
        # streaming training state (simplified)
        self.streaming_active = False  # Whether we're currently in a sequence
        
        if self.is_main_process:
            print(f"Gradient accumulation steps: {self.gradient_accumulation_steps}")
            if self.gradient_accumulation_steps > 1:
                print(f"Effective batch size: {config.batch_size * self.gradient_accumulation_steps * self.world_size}")
            if self.streaming_training:
                print(f"streaming training enabled: chunk_size={self.streaming_chunk_size}, max_length={self.streaming_max_length}")
            if LOG_GPU_MEMORY:
                log_gpu_memory("After initialization", device=self.device, rank=dist.get_rank())

        
    def _move_optimizer_to_device(self, optimizer, device):
        """Move optimizer state to the specified device."""
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)

    def find_latest_checkpoint(self, logdir):
        """Find the latest checkpoint in the logdir."""
        if not os.path.exists(logdir):
            return None
        
        checkpoint_dirs = []
        for item in os.listdir(logdir):
            if item.startswith("checkpoint_model_") and os.path.isdir(os.path.join(logdir, item)):
                try:
                    # Extract step number from directory name
                    step_str = item.replace("checkpoint_model_", "")
                    step = int(step_str)
                    checkpoint_path = os.path.join(logdir, item, "model.pt")
                    if os.path.exists(checkpoint_path):
                        checkpoint_dirs.append((step, checkpoint_path))
                except ValueError:
                    continue
        
        if not checkpoint_dirs:
            return None
        
        # Sort by step number and return the latest one
        checkpoint_dirs.sort(key=lambda x: x[0])
        latest_step, latest_path = checkpoint_dirs[-1]
        return latest_path

    def get_all_checkpoints(self, logdir):
        """Get all checkpoints in the logdir sorted by step number."""
        if not os.path.exists(logdir):
            return []
        
        checkpoint_dirs = []
        for item in os.listdir(logdir):
            if item.startswith("checkpoint_model_") and os.path.isdir(os.path.join(logdir, item)):
                try:
                    # Extract step number from directory name
                    step_str = item.replace("checkpoint_model_", "")
                    step = int(step_str)
                    checkpoint_dir_path = os.path.join(logdir, item)
                    checkpoint_file_path = os.path.join(checkpoint_dir_path, "model.pt")
                    if os.path.exists(checkpoint_file_path):
                        checkpoint_dirs.append((step, checkpoint_dir_path, item))
                except ValueError:
                    continue
        
        # Sort by step number (ascending order)
        checkpoint_dirs.sort(key=lambda x: x[0])
        return checkpoint_dirs

    def cleanup_old_checkpoints(self, logdir, max_checkpoints):
        """Remove old checkpoints if the number exceeds max_checkpoints.
        
        Only the main process performs the actual deletion to avoid race conditions
        in distributed training.
        """
        if max_checkpoints <= 0:
            return
        
        # Only main process should perform cleanup to avoid race conditions
        if not self.is_main_process:
            return
            
        checkpoints = self.get_all_checkpoints(logdir)
        if len(checkpoints) > max_checkpoints:
            # Calculate how many to remove
            num_to_remove = len(checkpoints) - max_checkpoints
            checkpoints_to_remove = checkpoints[:num_to_remove]  # Remove oldest ones
            
            print(f"Checkpoint cleanup: Found {len(checkpoints)} checkpoints, removing {num_to_remove} oldest ones (keeping {max_checkpoints})")
            
            import shutil
            removed_count = 0
            for step, checkpoint_dir_path, dir_name in checkpoints_to_remove:
                try:
                    print(f"  Removing: {dir_name} (step {step})")
                    shutil.rmtree(checkpoint_dir_path)
                    removed_count += 1
                except Exception as e:
                    print(f"  Warning: Failed to remove checkpoint {dir_name}: {e}")
            
            print(f"Checkpoint cleanup completed: removed {removed_count}/{num_to_remove} old checkpoints")
        else:
            if len(checkpoints) > 0:
                print(f"Checkpoint cleanup: Found {len(checkpoints)} checkpoints (max: {max_checkpoints}, no cleanup needed)")

    def _get_switch_frame_index(self, max_length=None):
        if getattr(self.config, "switch_mode", "fixed") == "random":  # 随机在某一次三帧的过程中切换prompt
            block = self.config.num_frame_per_block
            min_idx = self.config.min_switch_frame_index
            max_idx = self.config.max_switch_frame_index
            if min_idx == max_idx:
                switch_idx = min_idx
            else:
                choices = list(range(min_idx, max_idx, block))
                if max_length is not None:
                    choices = [choice for choice in choices if choice < max_length]
                
                if len(choices) == 0:
                    if max_length is not None:
                        raise ValueError(f"No valid switch choices available (all choices >= max_length {max_length})")
                    else:
                        switch_idx = block
                else:
                    if dist.get_rank() == 0:
                        switch_idx = random.choice(choices)
                    else:
                        switch_idx = 0  # placeholder; will be overwritten by broadcast
                switch_idx_tensor = torch.tensor(switch_idx, device=self.device)
                dist.broadcast(switch_idx_tensor, src=0)
                switch_idx = switch_idx_tensor.item()
        elif getattr(self.config, "switch_mode", "fixed") == "fixed":  # 21帧为一段，换一次prompt
            switch_idx = getattr(self.config, "fixed_switch_index", 21)
            if max_length is not None:
                assert max_length > switch_idx, f"max_length {max_length} is not greater than switch_idx {switch_idx}"
        elif getattr(self.config, "switch_mode", "fixed") == "random_choice":  # 根据自行定义的帧数，换prompt
            switch_choices = getattr(self.config, "switch_choices", [])
            if len(switch_choices) == 0:
                raise ValueError("switch_choices is empty")
            else:
                if max_length is not None:
                    valid = [choice for choice in switch_choices if choice < max_length]
                    # all choices beyond the video = deliberate no-switch (plain-long): keep the
                    # out-of-range sentinel so _should_switch_prompt never fires, instead of crashing
                    switch_choices = valid if valid else [max(switch_choices)]
                
                if dist.get_rank() == 0:
                    switch_idx = random.choice(switch_choices)
                else:
                    switch_idx = 0
            switch_idx_tensor = torch.tensor(switch_idx, device=self.device)
            dist.broadcast(switch_idx_tensor, src=0)
            switch_idx = switch_idx_tensor.item()
        else:
            raise ValueError(f"Invalid switch_mode: {getattr(self.config, 'switch_mode', 'fixed')}")
        return switch_idx


    def save(self):
        print("Start gathering distributed model states...")

        if self.is_lora_enabled:
            gen_lora_sd = self._gather_lora_state_dict(
                self.model.generator.model)
            crit_lora_sd = self._gather_lora_state_dict(
                self.model.fake_score.model)
            # memory params are not part of the LoRA state dict, persist them explicitly
            gen_memory_sd = self._gather_memory_state_dict(self.model.generator)

            state_dict = {
                "generator_lora": gen_lora_sd,
                "critic_lora": crit_lora_sd,
                "step": self.step,
            }
            if gen_memory_sd:
                state_dict["generator_memory"] = gen_memory_sd
            # persist the trainable-only EMA shadow so eval (use_ema) and crash-resume can
            # recover it. Without this the LoRA-EMA mechanism leaves nothing on disk.
            if self.generator_ema is not None:
                state_dict["generator_ema"] = self.generator_ema.state_dict()
        else:
            with FSDP.state_dict_type(
                self.model.generator,
                StateDictType.FULL_STATE_DICT,
                FullStateDictConfig(rank0_only=True, offload_to_cpu=True),
                FullOptimStateDictConfig(rank0_only=True),          # newly added
            ):
                generator_state_dict  = self.model.generator.state_dict()
                generator_opim_state_dict = FSDP.optim_state_dict(self.model.generator,
                                                self.generator_optimizer)

            with FSDP.state_dict_type(
                self.model.fake_score,
                StateDictType.FULL_STATE_DICT,
                FullStateDictConfig(rank0_only=True, offload_to_cpu=True),
                FullOptimStateDictConfig(rank0_only=True),          # newly added
            ):
                critic_state_dict  = self.model.fake_score.state_dict()
                critic_opim_state_dict = FSDP.optim_state_dict(self.model.fake_score,
                                                self.critic_optimizer)

            if self.config.ema_start_step < self.step and self.generator_ema is not None:
                state_dict = {
                    "generator": generator_state_dict,
                    "critic": critic_state_dict,
                    "generator_ema": self.generator_ema.state_dict(),
                    "generator_optimizer": generator_opim_state_dict,
                    "critic_optimizer": critic_opim_state_dict,
                    "step": self.step,
                }
            else:
                state_dict = {
                    "generator": generator_state_dict,
                    "critic": critic_state_dict,
                    "generator_optimizer": generator_opim_state_dict,
                    "critic_optimizer": critic_opim_state_dict,
                    "step": self.step,
                }

        if self.is_main_process:
            checkpoint_dir = os.path.join(self.output_path, f"checkpoint_model_{self.step:06d}")
            os.makedirs(checkpoint_dir, exist_ok=True)
            checkpoint_file = os.path.join(checkpoint_dir, "model.pt")
            torch.save(state_dict, checkpoint_file)
            print("Model saved to", checkpoint_file)
            
            # Cleanup old checkpoints if max_checkpoints is set
            max_checkpoints = getattr(self.config, "max_checkpoints", 0)
            if max_checkpoints > 0:
                self.cleanup_old_checkpoints(self.output_path, max_checkpoints)

        torch.cuda.empty_cache()
        import gc
        gc.collect()
    
    def fwdbwd_one_step(self, batch, train_generator):
        self.model.eval()  # prevent any randomness (e.g. dropout)

        if self.step % 5 == 0:
            torch.cuda.empty_cache()

        # Step 1: Get the next batch of text prompts
        text_prompts = batch["prompts"]

        batch_size = len(text_prompts)
        image_or_video_shape = list(self.config.image_or_video_shape)
        image_or_video_shape[0] = batch_size

        # Step 2: Extract the conditional infos
        with torch.no_grad():
            conditional_dict = self.model.text_encoder(
                text_prompts=text_prompts)

            if not getattr(self, "unconditional_dict", None):
                unconditional_dict = self.model.text_encoder(
                    text_prompts=[self.config.negative_prompt] * batch_size)
                unconditional_dict = {k: v.detach()
                                      for k, v in unconditional_dict.items()}
                self.unconditional_dict = unconditional_dict  # cache the unconditional_dict
            else:
                unconditional_dict = self.unconditional_dict

        # Step 3: Store gradients for the generator (if training the generator)
        if train_generator:
            generator_loss, generator_log_dict = self.model.generator_loss(
                image_or_video_shape=image_or_video_shape,
                conditional_dict=conditional_dict,
                unconditional_dict=unconditional_dict,
                clean_latent=None,
                initial_latent=None
            )

            # Scale loss for gradient accumulation and backward
            scaled_generator_loss = generator_loss / self.gradient_accumulation_steps
            scaled_generator_loss.backward()
            if LOG_GPU_MEMORY:
                log_gpu_memory("After train_generator backward pass", device=self.device, rank=dist.get_rank())
            # Return original loss for logging
            generator_log_dict.update({"generator_loss": generator_loss,
                                       "generator_grad_norm": torch.tensor(0.0, device=self.device)})  # Will be computed after accumulation

            return generator_log_dict
        else:
            generator_log_dict = {}

        critic_loss, critic_log_dict = self.model.critic_loss(
            image_or_video_shape=image_or_video_shape,
            conditional_dict=conditional_dict,
            unconditional_dict=unconditional_dict,
            clean_latent=None,
            initial_latent=None
        )

        # Scale loss for gradient accumulation and backward
        scaled_critic_loss = critic_loss / self.gradient_accumulation_steps
        scaled_critic_loss.backward()
        if LOG_GPU_MEMORY:
            log_gpu_memory("After train_critic backward pass", device=self.device, rank=dist.get_rank())
        # Return original loss for logging
        critic_log_dict.update({"critic_loss": critic_loss,
                                "critic_grad_norm": torch.tensor(0.0, device=self.device)})  # Will be computed after accumulation

        return critic_log_dict

    def generate_video(self, pipeline, num_frames, prompts, image=None):
        batch_size = len(prompts)
        if image is not None:
            image = image.squeeze(0).unsqueeze(0).unsqueeze(2).to(device="cuda", dtype=torch.bfloat16)

            # Encode the input image as the first latent
            initial_latent = pipeline.vae.encode_to_latent(image).to(device="cuda", dtype=torch.bfloat16)
            initial_latent = initial_latent.repeat(batch_size, 1, 1, 1, 1)
            sampled_noise = torch.randn(
                [batch_size, num_frames - 1, 16, 60, 104],
                device="cuda",
                dtype=self.dtype
            )
        else:
            initial_latent = None
            sampled_noise = torch.randn(
                [batch_size, num_frames, 16, 60, 104],
                device=self.device,
                dtype=self.dtype
            )
        with torch.no_grad():
            video, _ = pipeline.inference(
                noise=sampled_noise,
                text_prompts=prompts,
                return_latents=True,
            )
        current_video = video.permute(0, 1, 3, 4, 2).cpu().numpy() * 255.0
        pipeline.vae.model.clear_cache()
        return current_video
    

    def generate_video_with_switch(self, pipeline, num_frames, prompts, switch_prompts, switch_frame_index, image=None):
        batch_size = len(prompts)
        if image is not None:
            image = image.squeeze(0).unsqueeze(0).unsqueeze(2).to(device="cuda", dtype=torch.bfloat16)

            # Encode the input image as the first latent
            initial_latent = pipeline.vae.encode_to_latent(image).to(device="cuda", dtype=torch.bfloat16)
            initial_latent = initial_latent.repeat(batch_size, 1, 1, 1, 1)
            sampled_noise = torch.randn(
                [batch_size, num_frames - 1, 16, 60, 104],
                device="cuda",
                dtype=self.dtype
            )
        else:
            initial_latent = None
            sampled_noise = torch.randn(
                [batch_size, num_frames, 16, 60, 104],
                device=self.device,
                dtype=self.dtype
            )
        with torch.no_grad():
            video, _ = pipeline.inference(
                noise=sampled_noise,
                text_prompts_first=prompts,
                text_prompts_second=switch_prompts,
                switch_frame_index=switch_frame_index,
                return_latents=True
            )
        current_video = video.permute(0, 1, 3, 4, 2).cpu().numpy() * 255.0
        pipeline.vae.model.clear_cache()
        return current_video

    def start_new_sequence(self):
        if DEBUG and (not dist.is_initialized() or dist.get_rank() == 0):
            print(f"[SeqTrain-Trainer] start_new_sequence called")
        
        if LOG_GPU_MEMORY:
            log_gpu_memory(f"streaming Training: Before start_new_sequence", device=self.device, rank=dist.get_rank())
        
        # Fetch a new batch
        if DEBUG and (not dist.is_initialized() or dist.get_rank() == 0):
            print(f"[SeqTrain-Trainer] start_new_sequence: fetch new batch")
        batch = next(self.dataloader)

        # Prepare conditional information
        text_prompts = batch["prompts"]
        if self.config.i2v:
            image_latent = batch["ode_latent"][:, -1][:, 0:1, ].to(
                device=self.device, dtype=self.dtype)
        else:
            image_latent = None

        batch_size = len(text_prompts)
        image_or_video_shape = list(self.config.image_or_video_shape)
        image_or_video_shape[0] = batch_size
        
        if DEBUG and (not dist.is_initialized() or dist.get_rank() == 0):
            print(f"[SeqTrain-Trainer] Setting up sequence: batch_size={batch_size}, i2v={self.config.i2v}")
            print(f"[SeqTrain-Trainer] image_or_video_shape={image_or_video_shape}")
        
        with torch.no_grad():
            conditional_dict = self.model.text_encoder(text_prompts=text_prompts)
            if DEBUG and (not dist.is_initialized() or dist.get_rank() == 0):
                print(f"[SeqTrain-Trainer] Created and cached conditional_dict")
            if not getattr(self, "unconditional_dict", None):
                unconditional_dict = self.model.text_encoder(
                    text_prompts=[self.config.negative_prompt] * batch_size)
                unconditional_dict = {k: v.detach() for k, v in unconditional_dict.items()}
                self.unconditional_dict = unconditional_dict
                if DEBUG and (not dist.is_initialized() or dist.get_rank() == 0):
                    print(f"[SeqTrain-Trainer] Created and cached unconditional_dict")
            else:
                unconditional_dict = self.unconditional_dict
        
        if (not dist.is_initialized() or dist.get_rank() == 0) and LOG_GPU_MEMORY:
            log_gpu_memory(f"streaming Training: After text encoding", device=self.device, rank=dist.get_rank())
        
        if self.streaming_model.possible_max_length is not None:
            # Ensure all processes choose the same length
            if dist.is_initialized():
                if dist.get_rank() == 0:
                    import random
                    selected_idx = random.randint(0, len(self.streaming_model.possible_max_length) - 1)
                else:
                    selected_idx = 0
                selected_idx_tensor = torch.tensor(selected_idx, device=self.device, dtype=torch.int32)
                dist.broadcast(selected_idx_tensor, src=0)
                selected_idx = selected_idx_tensor.item()
            else:
                import random
                selected_idx = random.randint(0, len(self.streaming_model.possible_max_length) - 1)
            
            temp_max_length = self.streaming_model.possible_max_length[selected_idx]
        else:
            temp_max_length = self.streaming_model.max_length
            
            if DEBUG and (not dist.is_initialized() or dist.get_rank() == 0):
                print(f"[SeqTrain-Model] Selected temporary max length: {temp_max_length} (from {self.streaming_model.possible_max_length})")
        

        # Handle DMD Switch related information
        switch_conditional_dict = None
        switch_frame_index = None
        if isinstance(self.model, DMDSwitch) and "switch_prompts" in batch:
            if DEBUG and (not dist.is_initialized() or dist.get_rank() == 0):
                print(f"[SeqTrain-Trainer] Processing DMDSwitch info")
                
            with torch.no_grad():
                switch_conditional_dict = self.model.text_encoder(
                    text_prompts=batch["switch_prompts"]
                )
            switch_frame_index = self._get_switch_frame_index(temp_max_length)
            
            if MY_VISUAL_DEBUG and (not dist.is_initialized() or dist.get_rank() == 0):
                print(f"[SeqTrain-Trainer] switch_frame_index={switch_frame_index}")
            
            if (not dist.is_initialized() or dist.get_rank() == 0) and LOG_GPU_MEMORY:
                log_gpu_memory(f"streaming Training: After switch text encoding", device=self.device, rank=dist.get_rank())
        
        # Set up the sequence
        if DEBUG and (not dist.is_initialized() or dist.get_rank() == 0):
            print(f"[SeqTrain-Trainer] Calling streaming_model.setup_sequence")
        
        # gan dataset process
        gt_pairs = batch.get("gt_pairs", None)
        if gt_pairs is not None:
            # gt_prompts_emb = self.model.text_encoder(text_prompts=batch["gt_prompts"])
            # gt_mismatch_prompts_emb = self.model.text_encoder(text_prompts=batch["mismatch_prompts"])
            # gt_pairs_emb = self.model.vae.encode_to_latent(batch["gt_pairs"])
            # mismatch_pairs_emb = self.model.vae.encode_to_latent(batch["mismatch_pairs"])
            gt_prompts = batch["gt_prompts"]
            gt_mismatch_prompts = batch["mismatch_prompts"]
            gt_pairs = batch["gt_pairs"][0]  # TODO 这里只支持batch为1
            mismatch_pairs = batch["mismatch_pairs"][0]
            # if dist.get_rank()==0:
            #     print("[test dataloader shape] gt_pairs: {}. mismatch_pairs: {}".format(gt_pairs.shape, mismatch_pairs.shape))
        else:
            gt_prompts = None
            gt_mismatch_prompts= None
            gt_pairs = None
            mismatch_pairs = None
            
        self.streaming_model.setup_sequence(  # 初始化kv cache prompt cache，第二段开始清空state、kv cache
            conditional_dict=conditional_dict,  # 第一段的prompt
            unconditional_dict=unconditional_dict,
            initial_latent=image_latent,
            switch_conditional_dict=switch_conditional_dict,  # 后一段的prompt
            switch_frame_index=switch_frame_index,  # 切换的帧的idx
            temp_max_length=temp_max_length,
            # gan_part
            gt_prompts=gt_prompts,
            gt_mismatch_prompts=gt_mismatch_prompts,
            gt_pairs=gt_pairs,
            mismatch_pairs=mismatch_pairs
        )
        # the model needs the prompt text to key the per-prompt stock motion-target curve
        self.streaming_model.state["current_prompt"] = text_prompts[0] if text_prompts else None

        self.streaming_active = True
        
        if DEBUG and (not dist.is_initialized() or dist.get_rank() == 0):
            print(f"[SeqTrain-Trainer] streaming training sequence setup completed")
            
        if (not dist.is_initialized() or dist.get_rank() == 0) and LOG_GPU_MEMORY:
            log_gpu_memory(f"streaming Training: After sequence setup", device=self.device, rank=dist.get_rank())

    def fwdbwd_one_step_streaming(self, train_generator):
        """Forward/backward pass using the new StreamingTrainingModel for serialized training"""
        self.model.eval()  # prevent any randomness (e.g. dropout)

        if self.step % self.config.dfake_gen_update_ratio == 0:
            torch.cuda.empty_cache()

        # If no active sequence, start a new one
        if not self.streaming_active:
            if DEBUG and (not dist.is_initialized() or dist.get_rank() == 0):
                print(f"[SeqTrain-Trainer] No active sequence, starting new one")
            self.start_new_sequence()  # 计算切换点、计算prompt、计算第一次kv cache和cross-attention cache、取出一个batch
        
        # Check whether we can generate more chunks
        if not self.streaming_model.can_generate_more():
            # Current sequence is finished; start a new one
            if DEBUG and (not dist.is_initialized() or dist.get_rank() == 0):
                print(f"[SeqTrain-Trainer] Current sequence completed, starting new one")
            self.streaming_active = False
            self.start_new_sequence()
        
        self.kv_cache_before_generator_rollout = None
        self.kv_cache_after_generator_rollout = None
        self.kv_cache_after_generator_backward = None
        self.kv_cache_before_critic_rollout = None
        self.kv_cache_after_critic_rollout = None
        self.kv_cache_after_critic_backward = None
        
        if train_generator:
            # if MY_VISUAL_DEBUG and (not dist.is_initialized() or dist.get_rank() == 0):
            if dist.get_rank() == 0:
                print(f"[SeqTrain-Trainer] Training generator: generating next chunk")

            # Cross-chunk truncated BPTT over K = bptt_window chunks: roll K chunks, detach the state
            # only at the window start, sum the per-chunk losses, one backward over the window (trains
            # the write across chunk boundaries; graph alive for one backward, freed after = no leak).
            K_bptt = int(getattr(self.config, "bptt_window", 1))
            # per-step random horizon K ~ U{bptt_window_min .. bptt_window} so the single grad
            # chunk (W=1) supervises many horizons up to the freeze horizon, at proven W=1 VRAM.
            # Rank-synced: FSDP collectives require every rank to run the same number of forwards.
            K_min = int(getattr(self.config, "bptt_window_min", K_bptt))
            if K_bptt > K_min:
                if not dist.is_initialized():
                    K_bptt = int(torch.randint(K_min, K_bptt + 1, (1,)).item())
                else:
                    _K = torch.tensor(
                        int(torch.randint(K_min, K_bptt + 1, (1,)).item()) if dist.get_rank() == 0 else 0,
                        device=self.device, dtype=torch.int32)
                    dist.broadcast(_K, src=0)
                    K_bptt = int(_K.item())
            if K_bptt > 1:
                # roll K_bptt chunks: the first (K_bptt - W) under no_grad (deep state past KV-eviction,
                # cheap), the last W with grad = the supervised tail. W=1 avoids the cross-chunk
                # checkpoint-vs-eviction error (no backward recomputes a previous chunk). Absent -> W=K.
                W = int(getattr(self.config, "grad_tail_W", K_bptt))
                W = max(1, min(W, K_bptt))
                _losses = []
                _logs = []
                _used = 0
                # force a fresh sequence so the K-chunk window always starts at frame 0 (chunk 0 sets the
                # identity anchor, the grad tail lands in the same sequence).
                # seed step: every train_seed_every steps also put chunk 0 in-graph so the fast-weight
                # seed w0/w1/w2 (only reachable via init_state at chunk 0) receives gradient.
                _seed_every = int(getattr(self.config, "train_seed_every", 0))
                _gen_idx = self.step // max(1, int(getattr(self.config, "dfake_gen_update_ratio", 1)))
                seed_step = _seed_every > 0 and (_gen_idx % _seed_every == 0)   # every N-th generator step
                _twin_on = any(float(getattr(self.config, _k, 0.0)) > 0.0
                             for _k in ("adv_loss_weight", "dnh_weight", "act_weight", "flick_weight"))
                if getattr(self.config, "id_loss_weight", 0.0) > 0.0 or _twin_on or seed_step:
                    self.streaming_active = False
                    self.start_new_sequence()
                # seed steps also grad chunk 0 -> more grad chunks; normalize each per-chunk backward so
                # the total gradient equals the normal path's mean(losses)/gas
                _n_grad = W + (1 if (seed_step and K_bptt - W > 0) else 0)
                for _k in range(K_bptt):
                    if _k > 0 and not self.streaming_model.can_generate_more():
                        break
                    _used += 1
                    rg = (_k >= K_bptt - W) or (seed_step and _k == 0)   # +chunk 0 on seed steps
                    # build the mem-OFF counterfactual twin only for the last (grad-tail) chunk
                    _mk_twin = (rg and _k == K_bptt - 1 and _twin_on)
                    _gc, _ci = self.streaming_model.generate_next_chunk(
                        requires_grad=rg, detach_state=(_k == K_bptt - W),
                        keep_last_write_grad=(rg and _k < K_bptt - 1 and _k + 1 > K_bptt - W),
                        make_twin=_mk_twin)
                    if rg:
                        _l, _ld = self.streaming_model.compute_generator_loss(chunk=_gc, chunk_info=_ci)
                        _logs.append(_ld)
                        if seed_step:   # per-chunk backward; sever the KV-cache grad link between chunks
                            _snap = self._snapshot_ttt_state()
                            torch.cuda.empty_cache()   # reclaim the ~800MB reserved-but-unallocated cache
                                                       # at the post-forward low-water mark, before chunk-0's
                                                       # backward peak (seed step only ~1/20). Fits the 4-GPU budget.
                            try:
                                (_l / (self.gradient_accumulation_steps * _n_grad)).backward()
                            finally:
                                self._restore_ttt_state(_snap)
                            self.streaming_model._clear_cache_gradients()
                            del _snap
                            torch.cuda.empty_cache()
                            _losses.append(_l.detach())
                        else:
                            _losses.append(_l)
                if _losses:
                    generator_loss = sum(_losses) / len(_losses)
                    if not seed_step:                             # normal step: one backward over the window
                        _ttt_snap = self._snapshot_ttt_state()
                        try:
                            (generator_loss / self.gradient_accumulation_steps).backward()
                        finally:
                            self._restore_ttt_state(_ttt_snap)
                    # first few steps: confirm the read (qkv/proj) and write (lrs/decay) memory params get gradients
                    if self.step in (0, 5, 10, 15, 20, 30) and (not dist.is_initialized() or dist.get_rank() == 0):
                        try:
                            gn = lambda p: (float(p.grad.detach().float().norm()) if (p is not None and getattr(p, "grad", None) is not None) else -1.0)
                            mems = [mm for mm in self.model.generator.modules() if type(mm).__name__ == "VideoMemoryLayer"]
                            for i, m in enumerate(mems):
                                print(f"mem-grad step={self.step} mem{i} "
                                      f"WRITE lrs={gn(m.lrs.weight):.3e} decay={gn(getattr(m,'decay_logit',None)):.3e} "
                                      f"READ qkv={gn(m.qkv.weight):.3e} proj={gn(m.proj.weight):.3e} "
                                      f"SEED w0={gn(m.w0):.3e}")
                        except Exception as _e:
                            print(f"mem-grad check failed: {_e}")
                else:
                    generator_loss = torch.tensor(0.0, device=self.device)
                generator_log_dict = merge_dict_list(_logs) if _logs else {}
                generator_log_dict.update({
                    "generator_loss": generator_loss,
                    "generator_grad_norm": torch.tensor(0.0, device=self.device),
                    "bptt_chunks_used": torch.tensor(float(_used), device=self.device),
                    "bptt_grad_chunks": torch.tensor(float(len(_losses)), device=self.device),
                })
                return generator_log_dict

            train_first_chunk = getattr(self.config, "train_first_chunk", False)
            if train_first_chunk:
                # 这里的梯度mask会记录新生成的chunk
                generated_chunk, chunk_info = self.streaming_model.generate_next_chunk(requires_grad=True)
            else:
                current_seq_length = self.streaming_model.state.get("current_length")
                if current_seq_length == 0:
                    if DEBUG and (not dist.is_initialized() or dist.get_rank() == 0):
                        print(f"[SeqTrain-Trainer] train_first_chunk={train_first_chunk}, current_seq_length={current_seq_length}, generate first chunk")
                    generated_chunk, chunk_info = self.streaming_model.generate_next_chunk(requires_grad=False)

                generated_chunk, chunk_info = self.streaming_model.generate_next_chunk(requires_grad=True)  # 这里生成新的chunk时，不计算sink的梯度。只会生成长度为18的chunk，一段段的生成，并不是在里面一次性生成240
            
                if DEBUG and (not dist.is_initialized() or dist.get_rank() == 0):
                    print(f"[SeqTrain-Trainer] train_first_chunk={train_first_chunk}, current_seq_length={current_seq_length}")

            self._log_memory_stats()

            # Compute generator loss
            generator_loss, generator_log_dict = self.streaming_model.compute_generator_loss(
                chunk=generated_chunk,
                chunk_info=chunk_info
            )

            # Scale loss for gradient accumulation and backward
            scaled_generator_loss = generator_loss / self.gradient_accumulation_steps
            
            if DEBUG and (not dist.is_initialized() or dist.get_rank() == 0):
                print(f"[DEBUG] Scaled generator loss: {scaled_generator_loss.item()}")

            _ttt_snap = self._snapshot_ttt_state()
            try:
                scaled_generator_loss.backward()
            finally:
                self._restore_ttt_state(_ttt_snap)

            generator_log_dict.update({
                "generator_loss": generator_loss,
                "generator_grad_norm": torch.tensor(0.0, device=self.device),
            })
            
            return generator_log_dict
        else:
            # if MY_VISUAL_DEBUG and (not dist.is_initialized() or dist.get_rank() == 0):
            if dist.get_rank() == 0:
                print(f"[SeqTrain-Trainer] Training critic: generating next chunk")

            train_first_chunk = getattr(self.config, "train_first_chunk", False)
            if train_first_chunk:
                generated_chunk, chunk_info = self.streaming_model.generate_next_chunk(requires_grad=False)
            else:
                current_seq_length = self.streaming_model.state.get("current_length")
                if current_seq_length == 0:
                    if DEBUG and (not dist.is_initialized() or dist.get_rank() == 0):
                        print(f"[SeqTrain-Trainer] train_first_chunk={train_first_chunk}, current_seq_length={current_seq_length}, generate first chunk")
                    generated_chunk, chunk_info = self.streaming_model.generate_next_chunk(requires_grad=False)

                generated_chunk, chunk_info = self.streaming_model.generate_next_chunk(requires_grad=False)
            
                if DEBUG and (not dist.is_initialized() or dist.get_rank() == 0):
                    print(f"[SeqTrain-Trainer] train_first_chunk={train_first_chunk}, current_seq_length={current_seq_length}")

            if DEBUG and (not dist.is_initialized() or dist.get_rank() == 0):
                print(f"[SeqTrain-Trainer] Generated chunk shape: {generated_chunk.shape}")
                print(f"[SeqTrain-Trainer] Generated chunk requires_grad: {generated_chunk.requires_grad}")
            
            if generated_chunk.requires_grad:
                generated_chunk = generated_chunk.detach()

            # Compute critic loss
            critic_loss, critic_log_dict = self.streaming_model.compute_critic_loss(
                chunk=generated_chunk,
                chunk_info=chunk_info
            )
            
            if DEBUG and (not dist.is_initialized() or dist.get_rank() == 0):
                print(f"[SeqTrain-Trainer] Critic loss: {critic_loss.item()}")
            
            # Scale loss for gradient accumulation and backward
            scaled_critic_loss = critic_loss / self.gradient_accumulation_steps
            scaled_critic_loss.backward()
            
            critic_log_dict.update({
                "critic_loss": critic_loss,
                "critic_grad_norm": torch.tensor(0.0, device=self.device),
            })
            
            return critic_log_dict

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
                print(f"[VRAM_PROFILE] layers={getattr(self.config, 'model_kwargs', {}).get('memory_layers', [])} "
                      f"train_write={getattr(self.config, 'train_write', False)} "
                      f"alloc_GiB={alloc:.2f} reserved_GiB={reserved:.2f}", flush=True)
            if dist.is_initialized():
                dist.barrier()
            os._exit(0)

    def _clip_mem_grads(self):
        """Explicit grad clip on the FSDP-ignored memory params. The generator FSDP
        clip_grad_norm_ does NOT cover them. Called after the mem-grad hand-allreduce and before
        the optimizer step. Default max_grad_norm_mem ~1e9 => no-op (byte-identical)."""
        if getattr(self, "mem_params", None) and self.max_grad_norm_mem < 1e8:
            torch.nn.utils.clip_grad_norm_(self.mem_params, self.max_grad_norm_mem)

    def _lr_sched_step(self):
        """Cosine lr decay for both generator groups (LoRA + mem) from lr_decay_start to
        max_iters, each to its own floor fraction. lr_decay_start defaults beyond max_iters =>
        never fires => lr stays at its base value (byte-identical)."""
        s0 = self._lr_decay_start
        if self.step < s0:
            return
        T = self.config.max_iters
        prog = min(1.0, (self.step - s0) / max(1, T - s0))
        cos = 0.5 * (1.0 + math.cos(math.pi * prog))           # 1 -> 0 over [s0, T]
        fracs = [self._lr_floor_frac, self._lr_mem_floor_frac]  # [LoRA, mem]; mem group is index 1
        for i, (g, base) in enumerate(zip(self.generator_optimizer.param_groups, self._base_lrs)):
            fl = fracs[i] if i < len(fracs) else fracs[-1]
            g["lr"] = base * (fl + (1.0 - fl) * cos)

    def train(self):
        start_step = self.step
        pbar_total = max(0, self.config.max_iters - self.step)
        pbar = tqdm(total=pbar_total, initial=0, desc=f"Training steps (start={self.step})", disable=not (self.is_main_process))
        try:
            while True:
                pbar.update(1)
                pbar.set_postfix(step=self.step)
                # Reclaim cached-but-unallocated VRAM before the step's peak (frees the
                # ~1.6GB the caching allocator strands). Lets write-training fit on the
                # 48GB cards by a hair. Perf-only; gated off by default.
                if getattr(self.config, "empty_cache_each_step", False):
                    torch.cuda.empty_cache()
                # Check if we should train generator on this optimization step
                TRAIN_GENERATOR = self.step % self.config.dfake_gen_update_ratio == 0
                if LOG_GPU_MEMORY:
                    log_gpu_memory(f"Before training", device=self.device, rank=dist.get_rank())
                
                if dist.get_rank() == 0 and DEBUG:
                    print(f"[Debug] Step {self.step}: switch_mode={getattr(self.config,'switch_mode','fixed')}")

                if self.streaming_training:
                    # (D) push the global step to the rollout pipeline so it can anneal
                    # cache_dropout_prob (no-op when the anneal window is off). The pipeline is the
                    # StreamingSwitchTrainingPipeline reached via streaming_model.inference_pipeline;
                    # its clear_kv_cache() -> _maybe_cache_dropout() reads this step.
                    _ip = getattr(self.streaming_model, "inference_pipeline", None)
                    if _ip is not None and hasattr(_ip, "set_train_step"):
                        _ip.set_train_step(self.step)
                    # Zero-out all optimizer gradients
                    if TRAIN_GENERATOR:
                        self.generator_optimizer.zero_grad(set_to_none=True)
                    self.critic_optimizer.zero_grad(set_to_none=True)

                    # Whole-cycle gradient accumulation loop
                    accumulated_generator_logs = []
                    accumulated_critic_logs = []
                    
                    for accumulation_step in range(self.gradient_accumulation_steps):
                        if DEBUG and (not dist.is_initialized() or dist.get_rank() == 0):
                            print(f"[SeqTrain-Trainer] Whole-cycle accumulation step {accumulation_step + 1}/{self.gradient_accumulation_steps}")
                        
                        if (not dist.is_initialized() or dist.get_rank() == 0) and LOG_GPU_MEMORY and accumulation_step == 0:
                            log_gpu_memory(f"streaming Training Step {self.step}: Before whole-cycle forward/backward", device=self.device, rank=dist.get_rank())
                        
                        # Train generator (if needed)
                        if TRAIN_GENERATOR:
                            if DEBUG and (not dist.is_initialized() or dist.get_rank() == 0):
                                print(f"[SeqTrain-Trainer] Accumulation step {accumulation_step + 1}: Training generator")
                            extra_gen = self.fwdbwd_one_step_streaming(True)
                            accumulated_generator_logs.append(extra_gen)
                        
                        # Train critic
                        if DEBUG and (not dist.is_initialized() or dist.get_rank() == 0):
                            print(f"[SeqTrain-Trainer] Accumulation step {accumulation_step + 1}: Training critic")
                        extra_crit = self.fwdbwd_one_step_streaming(False)
                        accumulated_critic_logs.append(extra_crit)
                        
                        if (not dist.is_initialized() or dist.get_rank() == 0) and LOG_GPU_MEMORY and accumulation_step == 0:
                            log_gpu_memory(f"streaming Training Step {self.step}: After whole-cycle forward/backward", device=self.device, rank=dist.get_rank())
                    
                    # Compute grad norm and update parameters
                    if TRAIN_GENERATOR:
                        if self.world_size > 1:
                            # memory is FSDP-ignored (replicated): average its grads across ranks by hand
                            for _n, _p in self.model.generator.named_parameters():
                                if "memory" in _n and _p.grad is not None:
                                    dist.all_reduce(_p.grad, op=dist.ReduceOp.AVG)
                        generator_grad_norm = self.model.generator.clip_grad_norm_(self.max_grad_norm_generator)
                        generator_log_dict = merge_dict_list(accumulated_generator_logs)
                        generator_log_dict["generator_grad_norm"] = generator_grad_norm
                        
                        if DEBUG and (not dist.is_initialized() or dist.get_rank() == 0):
                            print(f"[SeqTrain-Trainer] Generator training completed, grad_norm={generator_grad_norm.item()}")

                        self._clip_mem_grads()                 # (C) clip FSDP-ignored mem grads
                        self.generator_optimizer.step()
                        self._lr_sched_step()                  # (B) cosine lr decay (no-op pre-decay_start)
                        if self.generator_ema is not None:
                            self.generator_ema.update(self.model.generator)
                        self._log_memory_stats()
                    else:
                        generator_log_dict = {}
                    
                    critic_grad_norm = self.model.fake_score.clip_grad_norm_(self.max_grad_norm_critic)
                    critic_log_dict = merge_dict_list(accumulated_critic_logs)
                    critic_log_dict["critic_grad_norm"] = critic_grad_norm
                    
                    if DEBUG and (not dist.is_initialized() or dist.get_rank() == 0):
                        print(f"[SeqTrain-Trainer] Critic training completed, grad_norm={critic_grad_norm.item()}")
                    
                    self.critic_optimizer.step()

                    if (not dist.is_initialized() or dist.get_rank() == 0) and LOG_GPU_MEMORY:
                        log_gpu_memory(f"streaming Training Step {self.step}: After optimizer steps", device=self.device, rank=dist.get_rank())

                    # Increase step count
                    self.step += 1
                    self._vram_profile_tick()

                    if DEBUG and (not dist.is_initialized() or dist.get_rank() == 0):
                        print(f"[SeqTrain-Trainer] streaming training step completed: step={self.step}")
                        if hasattr(self, 'streaming_model') and self.streaming_model is not None:
                            current_seq_length = self.streaming_model.state.get("current_length", 0)
                            print(f"[SeqTrain-Trainer] Current sequence length: {current_seq_length}/{self.streaming_model.max_length}")
                            
                    if (not dist.is_initialized() or dist.get_rank() == 0) and LOG_GPU_MEMORY:
                        log_gpu_memory(f"streaming Training Step {self.step}: Training step completed", device=self.device, rank=dist.get_rank())
                else:
                    if TRAIN_GENERATOR:
                        self.generator_optimizer.zero_grad(set_to_none=True)
                    self.critic_optimizer.zero_grad(set_to_none=True)
                    
                    # Whole-cycle gradient accumulation loop
                    accumulated_generator_logs = []
                    accumulated_critic_logs = []
                    
                    for accumulation_step in range(self.gradient_accumulation_steps):
                        batch = next(self.dataloader)
                        
                        # Train generator (if needed)
                        if TRAIN_GENERATOR:
                            extra_gen = self.fwdbwd_one_step(batch, True)
                            accumulated_generator_logs.append(extra_gen)
                        
                        # Train critic
                        extra_crit = self.fwdbwd_one_step(batch, False)
                        accumulated_critic_logs.append(extra_crit)
                    
                    # Compute grad norm and update parameters
                    if TRAIN_GENERATOR:
                        if self.world_size > 1:
                            # memory is FSDP-ignored (replicated): average its grads across ranks by hand
                            for _n, _p in self.model.generator.named_parameters():
                                if "memory" in _n and _p.grad is not None:
                                    dist.all_reduce(_p.grad, op=dist.ReduceOp.AVG)
                        generator_grad_norm = self.model.generator.clip_grad_norm_(self.max_grad_norm_generator)
                        generator_log_dict = merge_dict_list(accumulated_generator_logs)
                        generator_log_dict["generator_grad_norm"] = generator_grad_norm

                        self._clip_mem_grads()                 # (C) clip FSDP-ignored mem grads
                        self.generator_optimizer.step()
                        self._lr_sched_step()                  # (B) cosine lr decay (no-op pre-decay_start)
                        if self.generator_ema is not None:
                            self.generator_ema.update(self.model.generator)
                    else:
                        generator_log_dict = {}
                    
                    critic_grad_norm = self.model.fake_score.clip_grad_norm_(self.max_grad_norm_critic)
                    critic_log_dict = merge_dict_list(accumulated_critic_logs)
                    critic_log_dict["critic_grad_norm"] = critic_grad_norm
                    
                    self.critic_optimizer.step()

                    # Increment the step since we finished gradient update
                    self.step += 1

                # Create EMA params (if not already created). Also in LoRA mode (trainable-only).
                if (self.step >= self.config.ema_start_step) and \
                        (self.generator_ema is None) and (self.config.ema_weight > 0):
                    self.generator_ema = EMA_FSDP(self.model.generator, decay=self.config.ema_weight)
                    if self.is_main_process:
                        _mode = "trainable-only (LoRA)" if self.is_lora_enabled else "full"
                        print(f"{_mode} EMA created at step {self.step} with weight {self.config.ema_weight}")

                # Save the model
                if (not self.config.no_save) and (self.step - start_step) > 0 and self.step % self.config.log_iters == 0:
                    torch.cuda.empty_cache()
                    self.save()
                    torch.cuda.empty_cache()

                # Logging
                if self.is_main_process:
                    if TRAIN_GENERATOR and generator_log_dict:
                        self.writer.add_scalar("generator/generator_loss", generator_log_dict["generator_loss"].mean().item(), self.step)
                        # dmd loss norm
                        self.writer.add_scalar("generator/generator_grad_norm", generator_log_dict["generator_grad_norm"].mean().item(), self.step)
                        # 训练时梯度回传的norm (absent in the windowed path)
                        if "dmdtrain_gradient_norm" in generator_log_dict:
                            self.writer.add_scalar("generator/dmdtrain_gradient_norm", generator_log_dict["dmdtrain_gradient_norm"].mean().item(), self.step)
                        # gan loss
                        try:
                            self.writer.add_scalar("generator/generator_cls_loss", generator_log_dict["generator_cls_loss"].mean().item(), self.step)
                        except:
                            pass
                        # identity + temporal-variance terms
                        for _k in ["e1_id_cos", "e1_id_loss", "e1_latent_motion", "e1_tv_loss"]:
                            if _k in generator_log_dict:
                                try:
                                    self.writer.add_scalar(f"generator/{_k}", generator_log_dict[_k].mean().item(), self.step)
                                except Exception:
                                    pass
                        # counterfactual-objective terms (dose/dnh/motion-match), logged every
                        # generator step so act_floor is calibratable and mm_ext/motion is observable
                        for _k, _v in generator_log_dict.items():
                            if _k.startswith("twin_"):
                                try:
                                    _val = _v.mean().item() if hasattr(_v, "mean") else float(_v)
                                    self.writer.add_scalar(f"twin/{_k}", _val, self.step)
                                except Exception:
                                    pass


                    self.writer.add_scalar("critic/critic_loss", critic_log_dict["critic_loss"].mean().item(), self.step)
                    self.writer.add_scalar("critic/critic_grad_norm", critic_log_dict["critic_grad_norm"].mean().item(), self.step)
                    # gan loss
                    try:
                        self.writer.add_scalar("critic/critic_logit_real", critic_log_dict["critic_logit_real"].mean().item(), self.step)
                        self.writer.add_scalar("critic/critic_logit_fake", critic_log_dict["critic_logit_fake"].mean().item(), self.step)
                        self.writer.add_scalar("critic/critic_logit_real_mismatch", critic_log_dict["critic_logit_real_mismatch"].mean().item(), self.step)
                        self.writer.add_scalar("critic/critic_cls_loss", critic_log_dict["critic_cls_loss"].mean().item(), self.step)
                    except:
                        pass

                if self.step % self.config.gc_interval == 0:
                    if dist.get_rank() == 0:
                        logging.info("DistGarbageCollector: Running GC.")
                    gc.collect()
                    torch.cuda.empty_cache()

                if self.is_main_process:
                    current_time = time.time()
                    iteration_time = 0 if self.previous_time is None else current_time - self.previous_time
                    self.writer.add_scalar("time/per_iteration_time", iteration_time, self.step)
                    self.previous_time = current_time

                # ---------------------------------------- Visualization ---------------------------------------------------

                if self.vis_interval > 0 and (self.step % self.vis_interval == 0):
                    try:
                        self._visualize()
                    except Exception as e:
                        print(f"[Warning] Visualization failed at step {self.step}: {e}")
                
                if self.step > self.config.max_iters:
                    break
            pbar.close()
        
        except Exception as e:
            _rk = dist.get_rank() if dist.is_initialized() else 0
            print(f"[ERROR] rank {_rk} training crashed at step {self.step}: {e}", flush=True)
            import traceback
            traceback.print_exc()
            raise


    def _configure_lora_for_model(self, transformer, model_name):
        """Configure LoRA for a WanDiffusionWrapper model"""
        # Find all Linear modules in WanAttentionBlock modules
        target_linear_modules = set()
        
        # Define the specific modules we want to apply LoRA to
        if model_name == 'generator':
            adapter_target_modules = ['CausalWanAttentionBlock']
        elif model_name == 'fake_score':
            adapter_target_modules = ['WanAttentionBlock']
        else:
            raise ValueError(f"Invalid model name: {model_name}")
        
        for name, module in transformer.named_modules():
            if module.__class__.__name__ in adapter_target_modules:
                for full_submodule_name, submodule in module.named_modules(prefix=name):
                    # exclude the memory branch: its Linears (qkv/lrs/proj) are
                    # full-rank trainable params, not LoRA targets (see the lr_mem group).
                    if isinstance(submodule, torch.nn.Linear) and ".memory." not in full_submodule_name:
                        target_linear_modules.add(full_submodule_name)
        
        target_linear_modules = list(target_linear_modules)
        
        if self.is_main_process:
            print(f"LoRA target modules for {model_name}: {len(target_linear_modules)} Linear layers")
            if getattr(self.lora_config, 'verbose', False):
                for module_name in sorted(target_linear_modules):
                    print(f"  - {module_name}")
        
        # Create LoRA config
        adapter_type = self.lora_config.get('type', 'lora')
        if adapter_type == 'lora':
            peft_config = peft.LoraConfig(
                r=self.lora_config.get('rank', 16),
                lora_alpha=self.lora_config.get('alpha', None) or self.lora_config.get('rank', 16),
                lora_dropout=self.lora_config.get('dropout', 0.0),
                target_modules=target_linear_modules,
                # task_type="FEATURE_EXTRACTION"        # Remove this; not needed for diffusion models
            )
        else:
            raise NotImplementedError(f'Adapter type {adapter_type} is not implemented')
        
        # Apply LoRA to the transformer
        lora_model = peft.get_peft_model(transformer, peft_config)

        if self.is_main_process:
            print('peft_config', peft_config)
            lora_model.print_trainable_parameters()

        return lora_model


    def _gather_lora_state_dict(self, lora_model):
        "On rank-0, gather FULL_STATE_DICT, then filter only LoRA weights"
        with FSDP.state_dict_type(
            lora_model,                       # lora_model contains nested FSDP submodules
            StateDictType.FULL_STATE_DICT,
            FullStateDictConfig(rank0_only=True, offload_to_cpu=True)
        ):
            full = lora_model.state_dict()
        return get_peft_model_state_dict(lora_model, state_dict=full)

    def _log_memory_stats(self):
        """opt-in memory instrumentation.
        config.log_memory_stats: print gate stats and live fast-weight drift.
        config.assert_memory_drift: fail loudly if the state stops accumulating
        (guards against fsdp/pytree container rebuilds swallowing the state writes;
        run after any change to the ttt threading or a torch upgrade).
        note: |w| is pinned at its init norm by weight_norm_update, so only
        ||w - w_init|| discriminates a frozen state from a living one."""
        do_log = getattr(self.config, "log_memory_stats", False)
        do_assert = getattr(self.config, "assert_memory_drift", False)
        if not (do_log or do_assert) or not self.is_main_process:
            return
        try:
            if do_log and hasattr(self, "writer"):
                first_gate = None
                if not hasattr(self, "_mem_init"):
                    self._mem_init = {}   # lazy init-weight snapshots (CPU) for weight-delta proof
                # per-layer gate bias, shift+scale bias, gate-weight norm, P_o/P_c norms, weight-deltas
                for n, p in self.model.generator.named_parameters():
                    if "memory" not in n:
                        continue
                    mm = re.search(r"blocks\.(\d+)\.", n)
                    li = mm.group(1) if mm else "x"
                    if n.endswith("proj.bias"):
                        b = p.detach().float(); third = b.numel() // 3
                        g = b[2 * third:].abs().mean().item()
                        ss = b[:2 * third].abs().mean().item()
                        self.writer.add_scalar(f"memory/gate_b{li}", g, self.step)
                        self.writer.add_scalar(f"memory/shiftscale_b{li}", ss, self.step)
                        first_gate = g if first_gate is None else first_gate
                    elif n.endswith("proj.weight"):
                        wt = p.detach().float(); third = wt.shape[0] // 3
                        self.writer.add_scalar(f"memory/gatew_b{li}", wt[2 * third:].norm().item(), self.step)
                    elif n.endswith("gx_po.weight"):   # agreement projection P_o
                        self.writer.add_scalar(f"memory/Po_b{li}", p.detach().float().norm().item(), self.step)
                    elif n.endswith("gx_pc.weight"):   # agreement projection P_c
                        self.writer.add_scalar(f"memory/Pc_b{li}", p.detach().float().norm().item(), self.step)
                    elif n.endswith("proj_extra.weight"):
                        # gate_extra usage: proj_extra is zero-init, so per-input column-norm = how much
                        # the gate learned to use each combo input. Feature order is code-fixed:
                        # [agreement(1), denoising_step(GX_EMB=16), frames_since_switch(16)]. Normalize the
                        # 16-dim ones by sqrt(16) so all three are per-weight-RMS comparable.
                        w = p.detach().float(); E = 16
                        if w.dim() == 2 and w.shape[1] >= 1 + 2 * E:
                            self.writer.add_scalar(f"memory/gx_agree_b{li}", w[:, 0:1].norm().item(), self.step)
                            self.writer.add_scalar(f"memory/gx_denoise_b{li}", w[:, 1:1 + E].norm().item() / (E ** 0.5), self.step)
                            self.writer.add_scalar(f"memory/gx_frames_b{li}", w[:, 1 + E:1 + 2 * E].norm().item() / (E ** 0.5), self.step)
                    # weight-delta from init -> proof the read (qkv) / gate (proj) weights actually move
                    if n.endswith("qkv.weight") or n.endswith("proj.weight"):
                        cur = p.detach().float().cpu()
                        if n not in self._mem_init:
                            self._mem_init[n] = cur.clone()
                        else:
                            d = (cur - self._mem_init[n]).norm().item()
                            tag = "dqkv" if "qkv" in n else "dproj"
                            self.writer.add_scalar(f"memory/{tag}_b{li}", d, self.step)
                # effective contribution + read magnitude + agreement score (captured in step())
                for n, mod in self.model.generator.named_modules():
                    if type(mod).__name__ != "VideoMemoryLayer":
                        continue
                    mm = re.search(r"blocks\.(\d+)\.", n)
                    li = mm.group(1) if mm else "x"
                    if getattr(mod, "dbg_out_rms", None) is not None:
                        self.writer.add_scalar(f"memory/memout_b{li}", float(mod.dbg_out_rms), self.step)
                    if getattr(mod, "dbg_read_rms", None) is not None:
                        self.writer.add_scalar(f"memory/read_b{li}", float(mod.dbg_read_rms), self.step)
                    if getattr(mod, "dbg_dc_frac", None) is not None:
                        self.writer.add_scalar(f"memory/dcfrac_b{li}", float(mod.dbg_dc_frac), self.step)
                    if getattr(mod, "dbg_amp_disp", None) is not None:
                        self.writer.add_scalar(f"memory/ampdisp_b{li}", float(mod.dbg_amp_disp), self.step)
                    if getattr(mod, "dbg_ampdisp_spatial", None) is not None:
                        self.writer.add_scalar(f"memory/ampdisp_spatial_b{li}", float(mod.dbg_ampdisp_spatial), self.step)
                    if getattr(mod, "dbg_ampdisp_temporal", None) is not None:
                        self.writer.add_scalar(f"memory/ampdisp_temporal_b{li}", float(mod.dbg_ampdisp_temporal), self.step)
                    if getattr(mod, "dbg_gate_mean", None) is not None:
                        self.writer.add_scalar(f"memory/gatemean_b{li}", float(mod.dbg_gate_mean), self.step)
                    if getattr(mod, "dbg_inj_stream", None) is not None:
                        self.writer.add_scalar(f"memory/injstream_b{li}", float(mod.dbg_inj_stream), self.step)
                    if getattr(mod, "dbg_agreement", None) is not None:
                        self.writer.add_scalar(f"memory/agree_b{li}", float(mod.dbg_agreement), self.step)
                if first_gate is not None:
                    print(f"mem stats step={self.step} gate(first)={first_gate:.3e}")
            pipe = getattr(getattr(self, "streaming_model", None), "inference_pipeline", None)
            ttt = getattr(pipe, "ttt_cache", None)
            if ttt is None:
                if do_log:
                    print(f"mem stats step={self.step} no ttt_cache")
                return
            import re as _re
            inits = {}
            for _n, _p in self.model.generator.named_parameters():
                _key = _n.rsplit(".", 1)[-1]
                if "memory" not in _n or _key not in ("w0", "w1", "w2"):
                    continue
                _mb = _re.search(r"blocks\.(\d+)\.", _n)
                inits.setdefault(int(_mb.group(1)) if _mb else -1, {})[_key] = _p.detach()
            found = False
            max_drift = 0.0
            for i, blk in enumerate(ttt):
                st = getattr(blk, "state", None)
                if st is None:
                    continue
                found = True
                blk_inits = inits.get(i, {})
                drifts = []; d_block = 0.0
                for w, key in zip(st, ("w0", "w1", "w2")):
                    if key in blk_inits:
                        d = (w.detach().float() - blk_inits[key].float()).norm().item()
                        max_drift = max(max_drift, d); d_block = max(d_block, d)
                        drifts.append(f"{d:.3e}")
                    else:
                        drifts.append("n/a")
                if do_log and hasattr(self, "writer"):
                    self.writer.add_scalar(f"memory/drift_b{i}", d_block, self.step)
                if do_log:
                    print(f"mem stats step={self.step} block={i} drift=({', '.join(drifts)})")
                # state-norm / init-norm ratio: the muon=0 collapse shows as ratio << 1 (the cap
                # never binds); muon>0 keeps it ~1 (cap binds at init). drift (dist-from-init)
                # cannot see this. Also report mean per-column ratio (the cap acts per column).
                sn_ratios = []
                col_ratios = []
                for w, key in zip(st, ("w0", "w1", "w2")):
                    if key in blk_inits:
                        wf = w.detach().float(); inf = blk_inits[key].float()
                        sn_ratios.append(wf.norm().item() / (inf.norm().item() + 1e-9))
                        try:
                            cr = (wf.norm(dim=-2) / (inf.norm(dim=-2) + 1e-9)).mean().item()
                        except Exception:
                            cr = float("nan")
                        col_ratios.append(cr)
                    else:
                        sn_ratios.append(float("nan")); col_ratios.append(float("nan"))
                if do_log:
                    print(f"mem stats step={self.step} block={i} "
                          f"statenorm/init=({', '.join(f'{r:.3f}' for r in sn_ratios)}) "
                          f"colnorm/init=({', '.join(f'{r:.3f}' for r in col_ratios)})")
                    if hasattr(self, "writer"):
                        self.writer.add_scalar(f"memory/statenorm_ratio_b{i}", sn_ratios[0], self.step)
                        self.writer.add_scalar(f"memory/colnorm_ratio_b{i}", col_ratios[0], self.step)
            if not found and do_log:
                print(f"mem stats step={self.step} ttt allocated but no state")
            if do_log and found and hasattr(self, "writer"):
                self.writer.add_scalar("memory/drift_max", max_drift, self.step)
            if do_assert and self.step >= getattr(self.config, "dfake_gen_update_ratio", 1):
                assert found, "drift guard: no live fast-weight state (writes swallowed again?)"
                assert max_drift > 1.0, f"drift guard: drift={max_drift:.3e} ~ 0, memory stuck at init"
        except AssertionError:
            raise
        except Exception as e:  # never let logging kill a training run
            print(f"mem stats logging failed: {e}")

    def _snapshot_ttt_state(self):
        # the fast-weight state is mutated inside the gradient-checkpointed block, so backward-time
        # recompute re-runs the write and leaves the cache holding an earlier chunk's state. Snapshot
        # the post-generation state before backward, restore it after. The injection EMA buffer is
        # mutated the same way (recompute rewrites ttt_cache.ema[key]), so snapshot/restore it too --
        # else the per-chunk (seed-step) backward corrupts the EMA going into the next chunk's forward.
        pipe = getattr(getattr(self, "streaming_model", None), "inference_pipeline", None)
        cache = getattr(pipe, "ttt_cache", None) if pipe is not None else None
        if not cache:
            return None
        return [
            {
                "state": (None if getattr(c, "state", None) is None
                          else tuple(t.detach().to("cpu") for t in c.state)),
                "ema": {k: v.detach().to("cpu") for k, v in (getattr(c, "ema", {}) or {}).items()},
                # eviction-feed fifo: backward recompute re-appends inside the checkpointed
                # block, so restore the pre-backward list. Entries are (idx, tensor) tuples
                # replaced per write (tensors never mutated in place) -> a shallow list copy
                # of refs suffices, no device round-trip needed.
                "fifo": list(getattr(c, "fifo", []) or []),
            }
            for c in cache
        ]

    def _restore_ttt_state(self, snap):
        if snap is None:
            return
        pipe = getattr(getattr(self, "streaming_model", None), "inference_pipeline", None)
        cache = getattr(pipe, "ttt_cache", None) if pipe is not None else None
        if not cache:
            return
        for c, s in zip(cache, snap):
            c.state = None if s["state"] is None else tuple(t.to(self.device) for t in s["state"])
            c.ema = {k: v.to(self.device) for k, v in s["ema"].items()}
            c.fifo = list(s.get("fifo", []))  # undo recompute-time appends

    def _gather_memory_state_dict(self, fsdp_model):
        "gather the full state dict on rank 0 and keep only the memory params (full-rank, not lora)"
        with FSDP.state_dict_type(
            fsdp_model,
            StateDictType.FULL_STATE_DICT,
            FullStateDictConfig(rank0_only=True, offload_to_cpu=True)
        ):
            full = fsdp_model.state_dict()
        return {k: v for k, v in full.items() if "memory" in k}
    
    # --------------------------------------------------------------------------------------------------------------
    # Visualization helpers
    # --------------------------------------------------------------------------------------------------------------

    def _setup_visualizer(self):
        """Initialize the inference pipeline for visualization on CPU, to be moved to GPU only when needed."""

        # Choose pipeline class depending on causal flag
        if 'switch' in self.config.distribution_loss:
            self.vis_pipeline = SwitchCausalInferencePipeline(
                args=self.config,
                device=self.device,
                generator=self.model.generator,
                text_encoder=self.model.text_encoder,
                vae=self.model.vae)
        else:
            self.vis_pipeline = CausalInferencePipeline(
                args=self.config,
                device=self.device,
                generator=self.model.generator,
                text_encoder=self.model.text_encoder,
                vae=self.model.vae)

        # Visualization output directory (default: <logdir>/vis)
        self.vis_output_dir = os.path.join(os.path.dirname(self.output_path), "vis")
        os.makedirs(self.vis_output_dir, exist_ok=True)
        if self.config.vis_ema:
            raise NotImplementedError("Visualization with EMA is not implemented")

    def _visualize(self):
        """Generate and save sample videos to monitor training progress."""
        if self.vis_interval <= 0 or not hasattr(self, "vis_pipeline"):
            return

        # Use the fixed batch of prompts/images prepared from val_loader
        if not getattr(self, "fixed_vis_batch", None):
            print("[Warning] No fixed validation batch available for visualization.")
            return

        # cache dropout: the vis pipeline shares the generator modules, so force the full
        # config sink for visualization and re-assert the training rollout's runtime sink
        # afterwards (a dropout rollout may be mid-sequence). No-op when the flag is off.
        _cd_pipe = getattr(getattr(self, "streaming_model", None), "inference_pipeline", None)
        _cd_on = _cd_pipe is not None and getattr(_cd_pipe, "cache_dropout_prob", 0.0) > 0.0
        if _cd_on:
            _cd_pipe._set_runtime_sink(None)

        step_vis_dir = os.path.join(self.vis_output_dir, f"step_{self.step:07d}")
        os.makedirs(step_vis_dir, exist_ok=True)
        batch = self.fixed_vis_batch
        if isinstance(self.vis_pipeline, SwitchCausalInferencePipeline):
            prompts = batch["prompts"]
            switch_prompts = batch["switch_prompts"]
            switch_frame_index = self._get_switch_frame_index()
        else:
            prompts = batch["prompts"]

        image = None
        if self.config.i2v and ("image" in batch):
            image = batch["image"]

        # Prepare model mode info for filename
        mode_info = ""
        if self.is_lora_enabled:
            mode_info = "_lora"
            if self.is_main_process:
                print(f"Generating videos in LoRA mode (step {self.step})")
        
        for vid_len in self.vis_video_lengths:
            print(f"Generating video of length {vid_len}")
            if isinstance(self.vis_pipeline, SwitchCausalInferencePipeline):
                videos = self.generate_video_with_switch(self.vis_pipeline, vid_len, prompts, switch_prompts, switch_frame_index, image=image)
            else:
                videos = self.generate_video(self.vis_pipeline, vid_len, prompts, image=image)

            # Save each sample
            for idx, video_np in enumerate(videos):
                if isinstance(self.vis_pipeline, SwitchCausalInferencePipeline):
                    video_name = f"step_{self.step:07d}_rank_{dist.get_rank()}_sample_{idx}_len_{vid_len}{mode_info}_switch_frame_{switch_frame_index}.mp4"
                else:
                    video_name = f"step_{self.step:07d}_rank_{dist.get_rank()}_sample_{idx}_len_{vid_len}{mode_info}.mp4"
                out_path = os.path.join(
                    step_vis_dir,
                    video_name,
                )
                video_tensor = torch.from_numpy(video_np.astype("uint8"))
                write_video(out_path, video_tensor, fps=16)

            # After saving current length videos, release related tensors to reduce peak memory
            del videos, video_np, video_tensor  # type: ignore
            torch.cuda.empty_cache()

        # drop the persistent inference caches (kv/crossattn/ttt, ~3.4GB) so the next training
        # step gets its VRAM back; they are re-allocated on the next _visualize call.
        for _attr in ("kv_cache1", "crossattn_cache", "ttt_cache"):
            if hasattr(self.vis_pipeline, _attr):
                setattr(self.vis_pipeline, _attr, None)
        torch.cuda.empty_cache()
        import gc
        gc.collect()

        # cache dropout: put the training rollout's runtime sink back (see above)
        if _cd_on and getattr(_cd_pipe, "_cache_dropout_active", False):
            _cd_pipe._set_runtime_sink(_cd_pipe.cache_dropout_sink)
