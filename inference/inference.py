# Adopted from https://github.com/guandeh17/Self-Forcing
# SPDX-License-Identifier: Apache-2.0
import argparse
import torch
import os
from omegaconf import OmegaConf
from tqdm import tqdm
from torchvision import transforms
from torchvision.io import write_video
from einops import rearrange
import torch.distributed as dist
from torch.utils.data import DataLoader, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from pipeline import (
    CausalInferencePipeline,
)
from utils.dataset import TextDataset
from utils.misc import set_seed

from utils.memory import gpu, get_cuda_free_memory_gb, DynamicSwapInstaller, log_gpu_memory

parser = argparse.ArgumentParser()
parser.add_argument("--config_path", type=str, help="Path to the config file")
args = parser.parse_args()

config = OmegaConf.load(args.config_path)

# Initialize distributed inference
if "LOCAL_RANK" in os.environ:
    os.environ["NCCL_CROSS_NIC"] = "1"
    os.environ["NCCL_DEBUG"] = os.environ.get("NCCL_DEBUG", "INFO")
    os.environ["NCCL_TIMEOUT"] = os.environ.get("NCCL_TIMEOUT", "1800")

    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", str(local_rank)))

    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    if not dist.is_initialized():
        dist.init_process_group(
            backend="nccl",
            rank=rank,
            world_size=world_size,
            timeout=torch.distributed.constants.default_pg_timeout,
        )
    set_seed(config.seed + local_rank)
    config.distributed = True  # Mark as distributed for pipeline
    if rank == 0:
        print(f"[Rank {rank}] Initialized distributed processing on device {device}")
else:
    local_rank = 0
    rank = 0
    device = torch.device("cuda")
    set_seed(config.seed)
    config.distributed = False  # Mark as non-distributed
    print(f"Single GPU mode on device {device}")

print(f'Free VRAM {get_cuda_free_memory_gb(device)} GB')
low_memory = get_cuda_free_memory_gb(device) < 40
low_memory = True

torch.set_grad_enabled(False)


# Initialize pipeline
# Note: checkpoint loading is now handled inside the pipeline __init__ method
pipeline = CausalInferencePipeline(config, device=device)

# Load generator checkpoint
if config.generator_ckpt:
    state_dict = torch.load(config.generator_ckpt, map_location="cpu")
    # c3: use_ema loads the EMA shadow. In the MAG eval recipe the trained weights (incl. the
    # shadow) live in lora_ckpt, while generator_ckpt is the plain base.pt (no shadow) -> fall
    # back to "generator" here and apply the EMA overlay in the lora_ckpt block below.
    _use_ema_here = bool(getattr(config, "use_ema", False)) and ("generator_ema" in state_dict)
    if "generator" in state_dict or "generator_ema" in state_dict:
        raw_gen_state_dict = state_dict["generator_ema" if _use_ema_here else "generator"]
    elif "model" in state_dict:
        raw_gen_state_dict = state_dict["model"]
    else:
        raise ValueError(f"Generator state dict not found in {config.generator_ckpt}")
    if _use_ema_here:
        def _clean_key(name: str) -> str:
            """Remove FSDP / checkpoint wrapper prefixes from parameter names."""
            name = name.replace("_fsdp_wrapped_module.", "")
            return name

        cleaned_state_dict = { _clean_key(k): v for k, v in raw_gen_state_dict.items() }
        missing, unexpected = pipeline.generator.load_state_dict(cleaned_state_dict, strict=False)
        if local_rank == 0:
            if len(missing) > 0:
                print(f"[Warning] {len(missing)} parameters are missing when loading checkpoint: {missing[:8]} ...")
            if len(unexpected) > 0:
                print(f"[Warning] {len(unexpected)} unexpected parameters encountered when loading checkpoint: {unexpected[:8]} ...")
        # same memory-aware safety net as the non-EMA branch: only memory keys may be missing
        non_memory_missing = [k for k in missing if "memory" not in k]
        assert not non_memory_missing, f"Missing non-memory keys: {non_memory_missing[:5]}"
        assert not unexpected, f"Unexpected keys: {list(unexpected)[:5]}"
    else:
        # acceptable missing keys are the memory params (they keep their fresh init)
        missing, unexpected = pipeline.generator.load_state_dict(raw_gen_state_dict, strict=False)
        non_memory_missing = [k for k in missing if "memory" not in k]
        assert not non_memory_missing, f"Missing non-memory keys: {non_memory_missing[:5]}"
        assert not unexpected, f"Unexpected keys: {list(unexpected)[:5]}"
        if local_rank == 0 and len(missing) > 0:
            print(f"[MAG] {len(missing)} memory keys kept at init (not in checkpoint)")

# --------------------------- LoRA support (optional) ---------------------------
from utils.lora_utils import configure_lora_for_model
import peft

pipeline.is_lora_enabled = False
if getattr(config, "adapter", None) and configure_lora_for_model is not None:
    if local_rank == 0:
        print(f"LoRA enabled with config: {config.adapter}")
        print("Applying LoRA to generator (inference)...")
    # 在加载基础权重后，对 generator 的 transformer 模型应用 LoRA 包装
    pipeline.generator.model = configure_lora_for_model(
        pipeline.generator.model,
        model_name="generator",
        lora_config=config.adapter,
        is_main_process=(local_rank == 0),
    )

    # 加载 LoRA 权重（如果提供了 lora_ckpt）
    lora_ckpt_path = getattr(config, "lora_ckpt", None)
    if lora_ckpt_path:
        if local_rank == 0:
            print(f"Loading LoRA checkpoint from {lora_ckpt_path}")
        lora_checkpoint = torch.load(lora_ckpt_path, map_location="cpu")
        # 兼容包含 `generator_lora` 键或直接是 LoRA state dict 两种格式
        if isinstance(lora_checkpoint, dict) and "generator_lora" in lora_checkpoint:
            peft.set_peft_model_state_dict(pipeline.generator.model, lora_checkpoint["generator_lora"])  # type: ignore
        else:
            peft.set_peft_model_state_dict(pipeline.generator.model, lora_checkpoint)  # type: ignore
        if local_rank == 0:
            print("LoRA weights loaded for generator")

        # MAG memory (full-rank), restore the trained weights.
        if isinstance(lora_checkpoint, dict) and "generator_memory" in lora_checkpoint:
            mem_sd = lora_checkpoint["generator_memory"]
            model_mem_keys = {k for k in pipeline.generator.state_dict() if "memory" in k}
            result = pipeline.generator.load_state_dict(mem_sd, strict=False)
            assert not result.unexpected_keys, f"Unexpected memory keys: {list(result.unexpected_keys)[:5]}"
            still_missing_mem = [k for k in result.missing_keys if "memory" in k]
            assert not still_missing_mem, f"Memory NOT loaded (would stay at init!): {still_missing_mem[:5]}"
            if local_rank == 0:
                print(f"[MAG] {len(mem_sd)} trained memory keys loaded "
                      f"(model has {len(model_mem_keys)} memory keys)")

        # c3 EMA OVERLAY: with use_ema (config or env USE_EMA_WEIGHTS), replace the trainable subset
        # (LoRA adapters + memory) with the smoothed EMA shadow saved in the trained checkpoint. The
        # shadow is TRAINABLE-ONLY, so it is overlaid on top of the already-loaded raw weights AFTER
        # LoRA is applied: frozen base keeps its base.pt values, trainable params get the EMA values.
        _use_ema_cfg = bool(getattr(config, "use_ema", False))
        _use_ema_env = os.environ.get("USE_EMA_WEIGHTS", "0") not in ("0", "", "false", "False")
        _use_ema = _use_ema_cfg or _use_ema_env
        if _use_ema and isinstance(lora_checkpoint, dict) and "generator_ema" in lora_checkpoint:
            def _clean_ema_key(name: str) -> str:
                for pfx in ("_fsdp_wrapped_module.", "_checkpoint_wrapped_module.", "_orig_mod."):
                    name = name.replace(pfx, "")
                return name
            shadow = {_clean_ema_key(k): v for k, v in lora_checkpoint["generator_ema"].items()}
            ema_res = pipeline.generator.load_state_dict(shadow, strict=False)
            # trainable-only shadow -> the frozen base is legitimately "missing"; NOTHING may be
            # unexpected, and a nonzero number of (LoRA/memory) keys MUST have matched, else the
            # EMA silently did nothing (naming mismatch) and we'd wrongly report an EMA eval.
            n_matched = len(shadow) - len(ema_res.unexpected_keys)
            assert not ema_res.unexpected_keys, \
                f"[EMA] unexpected shadow keys (naming mismatch): {list(ema_res.unexpected_keys)[:5]}"
            assert n_matched > 0, \
                "[EMA] use_ema requested but ZERO shadow keys matched the model (naming mismatch); refusing silent raw-weight eval"
            if local_rank == 0:
                print(f"[EMA] overlaid {n_matched}/{len(shadow)} trainable EMA-shadow keys "
                      f"(smoothed LoRA+memory); frozen base kept from base.pt")
        elif _use_ema:
            # metric-integrity: a CONFIG use_ema:true must NEVER silently eval raw weights. Only the
            # env-var convenience path (USE_EMA_WEIGHTS) is allowed to soft-fall-back to raw.
            _m = "[EMA] use_ema requested but no 'generator_ema' shadow in the checkpoint"
            if _use_ema_cfg:
                raise ValueError(
                    _m + " -- refusing to silently evaluate RAW weights (metric-integrity trap). "
                    "Set use_ema:false to eval raw, or point lora_ckpt at a checkpoint that has an "
                    "EMA shadow (saved from step >= ema_start_step).")
            elif local_rank == 0:
                print(_m + " -> (env USE_EMA_WEIGHTS) falling back to RAW trained weights")
    else:
        if local_rank == 0:
            print("No LoRA checkpoint specified; using base weights with LoRA adapters initialized")

    pipeline.is_lora_enabled = True


# Move pipeline to appropriate dtype and device
pipeline = pipeline.to(dtype=torch.bfloat16)
if low_memory:
    DynamicSwapInstaller.install_model(pipeline.text_encoder, device=device)
pipeline.generator.to(device=device)
pipeline.vae.to(device=device)

extended_prompt_path = config.data_path
dataset = TextDataset(prompt_path=config.data_path, extended_prompt_path=extended_prompt_path)
num_prompts = len(dataset)
print(f"Number of prompts: {num_prompts}")

if dist.is_initialized():
    sampler = DistributedSampler(dataset, shuffle=False, drop_last=False)
else:
    sampler = SequentialSampler(dataset)
dataloader = DataLoader(dataset, batch_size=1, sampler=sampler, num_workers=0, drop_last=False)

# Create output directory (only on main process to avoid race conditions)
if local_rank == 0:
    os.makedirs(config.output_folder, exist_ok=True)

if dist.is_initialized():
    dist.barrier()


def encode(self, videos: torch.Tensor) -> torch.Tensor:
    device, dtype = videos[0].device, videos[0].dtype
    scale = [self.mean.to(device=device, dtype=dtype),
             1.0 / self.std.to(device=device, dtype=dtype)]
    output = [
        self.model.encode(u.unsqueeze(0), scale).float().squeeze(0)
        for u in videos
    ]

    output = torch.stack(output, dim=0)
    return output

for i, batch_data in tqdm(enumerate(dataloader), total=len(dataloader), disable=(local_rank != 0)):
    idx = batch_data['idx'].item()

    # For DataLoader batch_size=1, the batch_data is already a single item, but in a batch container
    # Unpack the batch data for convenience
    if isinstance(batch_data, dict):
        batch = batch_data
    elif isinstance(batch_data, list):
        batch = batch_data[0]  # First (and only) item in the batch

    all_video = []

    # For text-to-video, batch is just the text prompt
    prompt = batch['prompts'][0]
    extended_prompt = batch['extended_prompts'][0] if 'extended_prompts' in batch else None
    if extended_prompt is not None:
        prompts = [extended_prompt] * config.num_samples
    else:
        prompts = [prompt] * config.num_samples
    sampled_noise = torch.randn(
        [config.num_samples, config.num_output_frames, 16, 60, 104], device=device, dtype=torch.bfloat16
    )

    print("sampled_noise.device", sampled_noise.device)
    # print("initial_latent.device", initial_latent.device)
    print("prompts", prompts)
    # Generate 81 frames
    # print('sampled_noise.shape', sampled_noise.shape, 'prompts', prompts)
    # print('pipeline.generator', pipeline.generator)
    # print('pipeline.text_encoder', pipeline.text_encoder)
    # print('pipeline.vae', pipeline.vae)

    # 跳过已经生成过的文件
    gen_list = []
    for seed_idx in range(config.num_samples):
        if not os.path.exists(os.path.join(config.output_folder, f'{prompt}-{seed_idx}.mp4')):
            gen_list.append(seed_idx)
    if len(gen_list) == 0:
        continue

    for num in gen_list:
        video, latents = pipeline.inference(
            noise=sampled_noise[num:num + 1],
            text_prompts=prompts,
            return_latents=True,
            low_memory=low_memory,
            profile=False,
        )
        current_video = rearrange(video, 'b t c h w -> b t h w c').cpu()
        all_video.append(current_video)

    # Final output video
    video = 255.0 * torch.cat(all_video, dim=0)

    # Clear VAE cache
    pipeline.vae.model.clear_cache()

    if dist.is_initialized():
        rank = dist.get_rank()
    else:
        rank = 0

    # Save the video if the current prompt is not a dummy prompt
    if idx < num_prompts:
        # Determine model type for filename
        if hasattr(pipeline, 'is_lora_enabled') and pipeline.is_lora_enabled:
            model_type = "lora"
        elif getattr(config, 'use_ema', False):
            model_type = "ema"
        else:
            model_type = "regular"
            
        for seed_idx in gen_list:
            # All processes save their videos
            if config.save_with_index:
                # output_path = os.path.join(config.output_folder, f'rank{rank}-{idx}-{seed_idx}_{model_type}.mp4')
                output_path = os.path.join(config.output_folder, f'{idx}-{seed_idx}.mp4')
            else:
                # output_path = os.path.join(config.output_folder, f'rank{rank}-{prompt[:100]}-{seed_idx}.mp4')
                output_path = os.path.join(config.output_folder, f'{prompt}-{seed_idx}.mp4')
            write_video(output_path, video[seed_idx], fps=16)

    if config.inference_iter != -1 and i >= config.inference_iter:
        break
if dist.is_initialized():
    dist.destroy_process_group()