#!/usr/bin/env python
"""Capture the memory contribution (mem_out) as feature maps during inference.

Loads the read-only memory model, runs a short single-prompt generation, and via a monkeypatch
on each VideoMemoryLayer.step it collects mem_out at the 5 memory layers per chunk. Then saves:
  - reference/featuremaps/rms_vs_frame.png : RMS of mem_out per frame, per layer (tests the
    "time-varying injection" mechanism -> if it wiggles frame to frame, that lowers consistency).
  - reference/featuremaps/heatmaps_b{li}.png : spatial |mem_out| maps for sampled frames.

Run single-GPU (no torchrun): CUDA_VISIBLE_DEVICES=<gpu> python reference/feature_maps.py
"""
import os, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from omegaconf import OmegaConf
from einops import rearrange

REPO = "/shared/ssd_14T/home/adenelmi/Anchor-Forcing"
sys.path.insert(0, REPO)
os.chdir(REPO)

RO_CKPT = "/shared/ssd_14T_project/adenelmi/e1_ro_5L_metrics/checkpoint_model_003000/model.pt"
MEM_LAYERS = [5, 11, 17, 23, 29]
OUTDIR = os.path.join(REPO, "reference/featuremaps")
NFRAMES = 120  # latent (~30s), short + fast
PROMPT = ("A serene countryside scene featuring a playful golden retriever and a gentle brown horse "
          "grazing in a green pasture under a clear blue sky. Medium shot.")

os.makedirs(OUTDIR, exist_ok=True)
from pipeline import CausalInferencePipeline
from utils.memory import gpu, get_cuda_free_memory_gb, DynamicSwapInstaller

device = torch.device("cuda")
cfg = OmegaConf.load(os.path.join(REPO, "configs/baseline.yaml"))
cfg.model_kwargs.memory_layers = MEM_LAYERS
cfg.lora_ckpt = RO_CKPT
cfg.num_output_frames = NFRAMES
cfg.distributed = False
from utils.misc import set_seed
set_seed(cfg.seed)
torch.set_grad_enabled(False)

pipeline = CausalInferencePipeline(cfg, device=device)
# generator base
sd = torch.load(cfg.generator_ckpt, map_location="cpu")
if "generator" in sd or "generator_ema" in sd:
    raw = sd["generator"]
elif "model" in sd:
    raw = sd["model"]
else:
    raw = sd
missing, unexpected = pipeline.generator.load_state_dict(raw, strict=False)
assert not [k for k in missing if "memory" not in k], f"missing non-mem: {missing[:5]}"
# LoRA + memory
from utils.lora_utils import configure_lora_for_model
import peft
pipeline.generator.model = configure_lora_for_model(pipeline.generator.model, model_name="generator",
                                                    lora_config=cfg.adapter, is_main_process=True)
lck = torch.load(cfg.lora_ckpt, map_location="cpu")
peft.set_peft_model_state_dict(pipeline.generator.model, lck["generator_lora"])
if "generator_memory" in lck:
    r = pipeline.generator.load_state_dict(lck["generator_memory"], strict=False)
    assert not r.unexpected_keys, r.unexpected_keys[:5]
    print(f"[fmap] loaded {len(lck['generator_memory'])} memory keys")
pipeline.is_lora_enabled = True
pipeline = pipeline.to(dtype=torch.bfloat16)
DynamicSwapInstaller.install_model(pipeline.text_encoder, device=device)
pipeline.generator.to(device=device); pipeline.vae.to(device=device)

# ---- monkeypatch step() on the 5 memory layers to capture mem_out ----
caps = {li: [] for li in MEM_LAYERS}
for li in MEM_LAYERS:
    layer = pipeline.generator.model.blocks[li].memory
    orig = layer.step
    def wrapped(x, *a, _orig=orig, _li=li, **k):
        out, st = _orig(x, *a, **k)
        caps[_li].append(out.detach().float().cpu())
        return out, st
    layer.step = wrapped

noise = torch.randn([1, NFRAMES, 16, 60, 104], device=device, dtype=torch.bfloat16)
print("[fmap] generating...", flush=True)
pipeline.inference(noise=noise, text_prompts=[PROMPT], return_latents=True, low_memory=True, profile=False)

# ---- process captures: each entry (1, Lchunk, dim); concat -> (1, F*1560, dim) ----
plt.figure(figsize=(10, 5))
for li in MEM_LAYERS:
    if not caps[li]:
        print(f"[fmap] layer {li}: no capture"); continue
    full = torch.cat(caps[li], dim=1)[0]          # (L, dim)
    L, dim = full.shape
    fseq = 1560
    F = L // fseq
    full = full[: F * fseq].reshape(F, fseq, dim)
    norm = full.norm(dim=-1)                        # (F, 1560)  per-token magnitude
    rms = norm.pow(2).mean(dim=-1).sqrt().numpy()   # (F,) per-frame RMS
    plt.plot(range(F), rms, label=f"layer {li}")
    # spatial heatmaps for sampled frames
    H, W = 30, 52
    sample = [int(x) for x in torch.linspace(0, F - 1, 6).tolist()]
    fig, axs = plt.subplots(1, len(sample), figsize=(3 * len(sample), 3))
    for ax, fr in zip(axs, sample):
        m = norm[fr].reshape(H, W).numpy()
        ax.imshow(m, cmap="magma"); ax.set_title(f"f{fr}"); ax.axis("off")
    fig.suptitle(f"|mem_out| spatial, layer {li}")
    fig.tight_layout(); fig.savefig(os.path.join(OUTDIR, f"heatmaps_b{li}.png"), dpi=90); plt.close(fig)
    print(f"[fmap] layer {li}: F={F} rms[min/max]={rms.min():.3f}/{rms.max():.3f}")

plt.xlabel("frame (latent)"); plt.ylabel("mem_out RMS"); plt.title("Memory contribution RMS vs frame")
plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
plt.savefig(os.path.join(OUTDIR, "rms_vs_frame.png"), dpi=110); plt.close()
print(f"[fmap] DONE -> {OUTDIR}")
open("/tmp/featuremaps_done.txt", "w").write(f"feature maps saved to {OUTDIR}\n")
