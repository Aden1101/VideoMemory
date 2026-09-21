# Adopted from https://github.com/guandeh17/Self-Forcing
# SPDX-License-Identifier: Apache-2.0
import argparse
import torch
import os
from omegaconf import OmegaConf
from tqdm import tqdm
from utils.video_io import write_video  # torchvision when available, else imageio
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

from utils.memory import get_cuda_free_memory_gb, DynamicSwapInstaller
from utils.device import device as af_device, describe as af_describe
from utils.env import check_arm_variables

parser = argparse.ArgumentParser()
parser.add_argument("--config_path", type=str, help="Path to the config file")
args = parser.parse_args()
check_arm_variables()    # a misspelt MEM_* variable stops the run here, not mid-way

config = OmegaConf.load(args.config_path)

# Initialize distributed inference
# The distributed branch is CUDA + NCCL by construction, and inference.sh calls torchrun even
# for a single process, which sets LOCAL_RANK. Requiring CUDA here keeps a lone process on the
# single-device path instead of sending it into torch.cuda.set_device.
if "LOCAL_RANK" in os.environ and torch.cuda.is_available():
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
    if "LOCAL_RANK" in os.environ:
        print("[inference] LOCAL_RANK is set but CUDA is absent: taking the single-device path.")
    local_rank = 0
    rank = 0
    device = af_device
    set_seed(config.seed)
    config.distributed = False  # Mark as non-distributed
    print(f"Single device mode | {af_describe()}")

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
    # The EMA shadow lives in lora_ckpt together with the trained weights, while generator_ckpt
    # is the plain base.pt and has none. Fall back to "generator" here and apply the overlay
    # in the lora_ckpt block below.
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
    else:
        missing, unexpected = pipeline.generator.load_state_dict(raw_gen_state_dict, strict=False)
        if local_rank == 0 and len(missing) > 0:
            print(f"[memory] {len(missing)} keys kept at init (not in checkpoint)")
    # only the memory parameters may be absent from a checkpoint: they keep their fresh init
    # here and are loaded from lora_ckpt below
    non_memory_missing = [k for k in missing if "memory" not in k]
    assert not non_memory_missing, f"Missing non-memory keys: {non_memory_missing[:5]}"
    assert not unexpected, f"Unexpected keys: {list(unexpected)[:5]}"

# --------------------------- LoRA support (optional) ---------------------------
from utils.lora_utils import configure_lora_for_model
import peft

def load_trained_memory(generator, checkpoint, verbose=False):
    """Restore the trained memory weights, which are full-rank and so travel beside the LoRA state
    dict. Fails loudly if the key is there but nothing matched: a memory left at its random init
    still produces plausible video, so the run would not be the arm it claims to be."""
    if not (isinstance(checkpoint, dict) and "generator_memory" in checkpoint):
        return
    state = checkpoint["generator_memory"]
    result = generator.load_state_dict(state, strict=False)
    assert not result.unexpected_keys, f"unexpected memory keys: {list(result.unexpected_keys)[:5]}"
    missing = [k for k in result.missing_keys if "memory" in k]
    assert not missing, f"memory would stay at init: {missing[:5]}"
    if verbose:
        print(f"[memory] {len(state)} trained keys loaded")


def overlay_ema_shadow(generator, checkpoint, from_config, verbose=False):
    """Replace the trainable weights with their smoothed EMA copy.

    The shadow holds only what was trained, so it is overlaid on the already-loaded weights and the
    frozen base keeps its base.pt values. A config asking for EMA may never fall back to the raw
    weights, since the result would be reported as an EMA measurement; USE_EMA_WEIGHTS may.
    """
    if not (isinstance(checkpoint, dict) and "generator_ema" in checkpoint):
        if from_config:
            raise ValueError("use_ema is set but this checkpoint has no 'generator_ema' shadow; "
                             "set use_ema:false to evaluate the raw weights instead")
        if verbose:
            print("[EMA] no shadow in the checkpoint, using the raw trained weights")
        return

    def strip(name):
        for prefix in ("_fsdp_wrapped_module.", "_checkpoint_wrapped_module.", "_orig_mod."):
            name = name.replace(prefix, "")
        return name

    shadow = {strip(k): v for k, v in checkpoint["generator_ema"].items()}
    result = generator.load_state_dict(shadow, strict=False)
    # the frozen base is legitimately missing, but a shadow matching nothing would leave the raw
    # weights in place and still be reported as an EMA evaluation
    assert not result.unexpected_keys, f"[EMA] unexpected shadow keys: {list(result.unexpected_keys)[:5]}"
    assert shadow, "[EMA] the shadow is empty"
    if verbose:
        print(f"[EMA] overlaid {len(shadow)} trainable keys")


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

        load_trained_memory(pipeline.generator, lora_checkpoint, verbose=(local_rank == 0))

        use_ema_cfg = bool(getattr(config, "use_ema", False))
        use_ema_env = os.environ.get("USE_EMA_WEIGHTS", "0") not in ("0", "", "false", "False")
        if use_ema_cfg or use_ema_env:
            overlay_ema_shadow(pipeline.generator, lora_checkpoint,
                               from_config=use_ema_cfg, verbose=(local_rank == 0))
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