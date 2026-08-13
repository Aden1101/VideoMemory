#!/usr/bin/env python
# ============================================================================================
# Re-ID REVISIT probe (inference-only): does the fast-weight memory (MAG) help SUBJECT
# RESURRECTION on an A->B->A prompt schedule that the plain-drift eval structurally cannot see?
#
#   For each (A,B) pair: generate ONE A->B->A video (first third A, middle third B, last third
#   A again) with the c3-trained memory-grafted model, in TWO arms:
#       memon   = memory injection active (default)
#       memzero = MEM_ZERO_ALL=1 (injection zeroed on every forward; state still written)
#   SAME seed / SAME noise across arms -> the ONLY difference is the memory read injection.
#
#   RE-ID SCORE (DINOv2-vitb14, loading copied verbatim from c3 eval metrics.py):
#       reid_resA_origA  = mean pairwise cos(DINO(resurrected-A frames), DINO(original-A frames))
#                          KEY number: higher = A came back looking more like the original A.
#       ctrl_B_origA     = cos(B segment, original-A)      (sanity: should be LOW)
#       ctrl_resA_B      = cos(resurrected-A, B segment)   (sanity: did B actually leave?)
#       self_origA       = cos(origA 1st half, origA 2nd half)  (self-consistency upper bound)
#   Both FULL-FRAME and CENTER-CROP (central 60%) DINO variants are reported.
#
# Frame units: switch indices are in LATENT frames (current_start_frame units, 240 total in the
# c3 recipe; num_frame_per_block=3 so use multiples of 3). Latent frame f occupies pixel frames
# [4f-3, 4f] (f>=1; latent 0 -> pixel 0), 957 pixel frames total for 240 latents @ 16fps.
#
# Usage (see run_probe.sh):
#   python run_reid_probe.py --mode selftest              # tiny 24-frame end-to-end check
#   python run_reid_probe.py --mode run --pairs 0         # generate+score pair 0, both arms
#   python run_reid_probe.py --mode score --pairs 0,1,2   # re-score existing videos only
# ============================================================================================
import argparse
import gc
import json
import os
import sys

REPO = "/shared/ssd_14T/home/adenelmi/Anchor-Forcing"
sys.path.insert(0, REPO)
HERE = os.path.dirname(os.path.abspath(__file__))

import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from einops import rearrange
from torchvision.io import write_video, read_video

# (A, B) pairs: A = distinctive re-identifiable subject, B = unrelated interlude
PAIRS = [
    ("a brown dog with a white chest running in a green park",
     "a red vintage car driving down a city street"),
    ("a woman in a yellow raincoat walking on a beach",
     "a group of penguins on ice"),
    ("an orange tabby cat sitting on a wooden table",
     "a hot air balloon rising over mountains"),
]

# ----------------------------- DINO (copied verbatim from c3 eval1000_sink1/metrics.py) -----
DINO_MEAN = (0.485, 0.456, 0.406)
DINO_STD = (0.229, 0.224, 0.225)
_DINO = None


def get_dino(device):
    global _DINO
    if _DINO is None:
        hub = os.path.expanduser("~/.cache/torch/hub")
        torch.hub.set_dir(hub)
        local = os.path.join(hub, "facebookresearch_dinov2_main")
        m = torch.hub.load(local, "dinov2_vitb14", source="local", pretrained=True)
        m.eval().to(device)
        for p in m.parameters():
            p.requires_grad_(False)
        _DINO = m
        print(f"[probe] loaded frozen DINOv2-vitb14 on {device}")
    return _DINO


@torch.no_grad()
def dino_embed(frames01, device, batch=64):
    """frames01 [T,3,H,W] in [0,1] -> L2-normalized DINO CLS embeddings [T,D] (cpu fp32)."""
    dino = get_dino(device)
    mean = torch.tensor(DINO_MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(DINO_STD, device=device).view(1, 3, 1, 1)
    out = []
    for i in range(0, frames01.shape[0], batch):
        x = frames01[i:i + batch].to(device).float()
        x = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)
        x = (x - mean) / std
        f = dino(x)
        out.append(F.normalize(f, dim=-1).cpu())
    return torch.cat(out, 0)


# ----------------------------- pipeline build (mirrors inference/inference.py) --------------
def build_pipeline(config, device):
    """SwitchCausalInferencePipeline with the SAME trained weights as the c3 eval:
    base.pt -> LoRA adapters -> generator_lora -> generator_memory (all asserted loaded)."""
    from pipeline.switch_causal_inference import SwitchCausalInferencePipeline
    from utils.lora_utils import configure_lora_for_model
    from utils.memory import DynamicSwapInstaller
    import peft

    pipeline = SwitchCausalInferencePipeline(config, device=device)

    # base weights (inference.py non-EMA branch)
    state_dict = torch.load(config.generator_ckpt, map_location="cpu")
    if "generator" in state_dict:
        raw = state_dict["generator"]
    elif "model" in state_dict:
        raw = state_dict["model"]
    else:
        raise ValueError(f"Generator state dict not found in {config.generator_ckpt}")
    missing, unexpected = pipeline.generator.load_state_dict(raw, strict=False)
    non_memory_missing = [k for k in missing if "memory" not in k]
    assert not non_memory_missing, f"Missing non-memory keys: {non_memory_missing[:5]}"
    assert not unexpected, f"Unexpected keys: {list(unexpected)[:5]}"

    # LoRA wrap + trained adapters
    pipeline.generator.model = configure_lora_for_model(
        pipeline.generator.model, model_name="generator",
        lora_config=config.adapter, is_main_process=True)
    lora_checkpoint = torch.load(config.lora_ckpt, map_location="cpu")
    assert isinstance(lora_checkpoint, dict) and "generator_lora" in lora_checkpoint, \
        f"no generator_lora in {config.lora_ckpt}"
    peft.set_peft_model_state_dict(pipeline.generator.model, lora_checkpoint["generator_lora"])
    print(f"[probe] LoRA weights loaded from {config.lora_ckpt}")

    # trained MAG memory (full-rank) -- must load, else the probe runs a random memory
    assert "generator_memory" in lora_checkpoint, \
        f"no generator_memory in {config.lora_ckpt} -- refusing to probe a random-init memory"
    mem_sd = lora_checkpoint["generator_memory"]
    result = pipeline.generator.load_state_dict(mem_sd, strict=False)
    assert not result.unexpected_keys, f"Unexpected memory keys: {list(result.unexpected_keys)[:5]}"
    still_missing_mem = [k for k in result.missing_keys if "memory" in k]
    assert not still_missing_mem, f"Memory NOT loaded: {still_missing_mem[:5]}"
    print(f"[probe] {len(mem_sd)} trained memory keys loaded")

    pipeline = pipeline.to(dtype=torch.bfloat16)
    DynamicSwapInstaller.install_model(pipeline.text_encoder, device=device)
    pipeline.generator.to(device=device)
    pipeline.vae.to(device=device)
    return pipeline


# ----------------------------- generation ---------------------------------------------------
def run_arm(pipeline, config, pair_idx, arm, frames, sw1, sw2, device, out_mp4):
    """One A->B->A generation. Returns (latents_cpu, blocks_with_written_ttt_state)."""
    from utils.misc import set_seed
    A, B = PAIRS[pair_idx]
    os.environ["MEM_ZERO_ALL"] = "1" if arm == "memzero" else "0"
    seed = int(config.seed) + pair_idx  # same seed across ARMS (what matters), distinct per pair
    set_seed(seed)
    noise = torch.randn([1, frames, 16, 60, 104], device=device, dtype=torch.bfloat16)
    print(f"[probe] pair{pair_idx} arm={arm} seed={seed} frames={frames} "
          f"switch@{sw1} switchback@{sw2}\n        A='{A}'\n        B='{B}'")
    video, latents = pipeline.inference(
        noise=noise,
        text_prompts_first=[A],
        text_prompts_second=[B],
        switch_frame_index=sw1,
        text_prompts_third=None,          # None -> defaults back to prompt A
        second_switch_frame_index=sw2,
        return_latents=True,
        low_memory=True,
    )
    # ttt evidence: which blocks actually hold a written fast-weight state after the rollout
    written = [bi for bi in range(pipeline.num_transformer_blocks)
               if pipeline.ttt_cache[bi].state is not None]
    print(f"[probe][{arm}] ttt_cache.state written on blocks: {written}")

    vid = (rearrange(video[0].cpu().float(), "t c h w -> t h w c") * 255.0)
    write_video(out_mp4, vid.clamp(0, 255).to(torch.uint8), fps=16)
    print(f"[probe][{arm}] saved {out_mp4} ({vid.shape[0]} pixel frames)")

    lat = latents.float().cpu()
    # free per-sequence caches before the next arm
    pipeline.vae.model.clear_cache()
    pipeline.kv_cache1 = None
    pipeline.crossattn_cache = None
    pipeline.ttt_cache = None
    gc.collect()
    torch.cuda.empty_cache()
    return lat, written


# ----------------------------- scoring ------------------------------------------------------
def latent_to_pix(f_lo, f_hi):
    """Inclusive latent-frame range -> inclusive pixel-frame range (Wan VAE 4x temporal)."""
    lo = 0 if f_lo == 0 else 4 * f_lo - 3
    hi = 4 * f_hi
    return lo, hi


def sample_seg(vid01, f_lo, f_hi, stride):
    lo, hi = latent_to_pix(f_lo, f_hi)
    hi = min(hi, vid01.shape[0] - 1)
    return vid01[lo:hi + 1:stride]


def center_crop(frames, frac=0.6):
    _, _, H, W = frames.shape
    ch, cw = int(H * frac), int(W * frac)
    top, left = (H - ch) // 2, (W - cw) // 2
    return frames[:, :, top:top + ch, left:left + cw]


def seg_cos(Ea, Eb):
    """Mean pairwise cosine between two sets of L2-normalized embeddings."""
    return float((Ea @ Eb.T).mean())


def score_video(path, frames, sw1, sw2, device, settle=20, resettle=12, stride=8):
    """DINO re-ID scores for one A->B->A mp4. Segments (latent units, both switches at block
    boundaries): origA = [settle, sw1-3] (skip warm-up + pre-switch chunk), B = [sw1+resettle,
    sw2-3] (skip transition), resA = [sw2+resettle, frames-1]."""
    v, _, _ = read_video(path, pts_unit="sec", output_format="THWC")  # [T,H,W,3] uint8
    vid = v.permute(0, 3, 1, 2).float() / 255.0  # CPU [T,3,H,W]; dino_embed batches to GPU
    segs = {
        "origA": (settle, sw1 - 3),
        "B": (sw1 + resettle, sw2 - 3),
        "resA": (sw2 + resettle, frames - 1),
    }
    out = {"segments_latent": {k: list(rng) for k, rng in segs.items()}}
    for variant, xform in (("full", lambda f: f), ("crop60", center_crop)):
        E = {k: dino_embed(xform(sample_seg(vid, *rng, stride)), device) for k, rng in segs.items()}
        half = max(1, E["origA"].shape[0] // 2)
        out[variant] = {
            "reid_resA_origA": seg_cos(E["resA"], E["origA"]),
            "ctrl_B_origA": seg_cos(E["B"], E["origA"]),
            "ctrl_resA_B": seg_cos(E["resA"], E["B"]),
            "self_origA": seg_cos(E["origA"][:half], E["origA"][half:]),
            "n_frames": {k: int(E[k].shape[0]) for k in E},
        }
    return out


# ----------------------------- selftest -----------------------------------------------------
def selftest(pipeline, config, device):
    """Tiny end-to-end check: (1) A->B->A runs; (2) ttt state written EXACTLY on memory_layers;
    (3) eviction-feed write path fires (MEM_EVICT_DEBUG prints); (4) MEM_ZERO_ALL changes the
    output latents (memory injection is live); (5) legacy single-switch path still works."""
    from utils.misc import set_seed
    os.environ["MEM_EVICT_DEBUG"] = "1"
    frames, sw1, sw2 = 24, 9, 18
    mem_layers = sorted(int(x) for x in config.model_kwargs.memory_layers)
    lat = {}
    for arm in ("memon", "memzero"):
        vid_path = os.path.join(HERE, "videos", f"selftest_{arm}.mp4")
        lat[arm], written = run_arm(pipeline, config, 0, arm, frames, sw1, sw2, device, vid_path)
        if arm == "memon":
            assert written == mem_layers, \
                f"ttt state written on {written}, expected exactly {mem_layers}"
    d = (lat["memon"] - lat["memzero"]).abs().max().item()
    m = (lat["memon"] - lat["memzero"]).abs().mean().item()
    print(f"[selftest] latent diff mem-ON vs mem-OFF: max={d:.6f} mean={m:.6f}")
    assert d > 0, "MEM_ZERO_ALL had NO effect on the output -> memory injection NOT live"
    os.environ["MEM_EVICT_DEBUG"] = "0"

    # legacy single-switch A->B path (second_switch_frame_index=None) must still run
    os.environ["MEM_ZERO_ALL"] = "0"
    set_seed(int(config.seed))
    noise = torch.randn([1, 12, 16, 60, 104], device=device, dtype=torch.bfloat16)
    A, B = PAIRS[0]
    pipeline.inference(noise=noise, text_prompts_first=[A], text_prompts_second=[B],
                       switch_frame_index=6, return_latents=True, low_memory=True)
    print("[selftest] legacy single-switch A->B path OK")
    print("[selftest] PASS")


# ----------------------------- main ---------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["selftest", "run", "score"], default="run")
    ap.add_argument("--pairs", type=str, default="0", help="comma list of pair indices")
    ap.add_argument("--arms", type=str, default="memon,memzero")
    ap.add_argument("--frames", type=int, default=240, help="latent frames (c3 recipe = 240)")
    ap.add_argument("--switch1", type=int, default=81, help="latent frame of A->B switch")
    ap.add_argument("--switch2", type=int, default=162, help="latent frame of B->A switch-back")
    ap.add_argument("--config", type=str, default=os.path.join(HERE, "probe.yaml"))
    args = ap.parse_args()

    assert args.switch1 % 3 == 0 and args.switch2 % 3 == 0, "switch indices must be multiples of num_frame_per_block=3"
    device = torch.device("cuda")
    config = OmegaConf.load(args.config)
    config.distributed = False
    torch.set_grad_enabled(False)

    pair_ids = [int(x) for x in args.pairs.split(",") if x != ""]
    arms = [a for a in args.arms.split(",") if a != ""]
    vid_dir = os.path.join(HERE, "videos")
    res_dir = os.path.join(HERE, "results")
    os.makedirs(vid_dir, exist_ok=True)
    os.makedirs(res_dir, exist_ok=True)

    pipeline = None
    if args.mode in ("selftest", "run"):
        pipeline = build_pipeline(config, device)

    if args.mode == "selftest":
        selftest(pipeline, config, device)
        return

    all_scores = {}
    for pi in pair_ids:
        pair_scores = {"pair": PAIRS[pi],
                       "frames": args.frames, "switch1": args.switch1, "switch2": args.switch2,
                       "checkpoint": str(config.lora_ckpt)}
        for arm in arms:
            mp4 = os.path.join(vid_dir, f"pair{pi}_{arm}.mp4")
            if args.mode == "run":
                if os.path.exists(mp4):
                    print(f"[probe] SKIP existing {mp4}")
                else:
                    run_arm(pipeline, config, pi, arm, args.frames, args.switch1, args.switch2,
                            device, mp4)
            assert os.path.exists(mp4), f"missing video for scoring: {mp4}"
            pair_scores[arm] = score_video(mp4, args.frames, args.switch1, args.switch2, device)
        all_scores[f"pair{pi}"] = pair_scores

        out_json = os.path.join(res_dir, f"pair{pi}_scores.json")
        with open(out_json, "w") as f:
            json.dump(pair_scores, f, indent=2)
        print(f"[probe] wrote {out_json}")

    # summary table
    print("\n================= RE-ID REVISIT PROBE SUMMARY =================")
    print(f"{'pair':<6}{'variant':<8}{'arm':<9}{'reid_resA_origA':>16}{'ctrl_B_origA':>14}"
          f"{'ctrl_resA_B':>13}{'self_origA':>12}")
    for pi in pair_ids:
        ps = all_scores[f"pair{pi}"]
        for variant in ("full", "crop60"):
            for arm in arms:
                s = ps[arm][variant]
                print(f"{pi:<6}{variant:<8}{arm:<9}{s['reid_resA_origA']:>16.4f}"
                      f"{s['ctrl_B_origA']:>14.4f}{s['ctrl_resA_B']:>13.4f}{s['self_origA']:>12.4f}")
        if "memon" in arms and "memzero" in arms:
            for variant in ("full", "crop60"):
                dlt = ps["memon"][variant]["reid_resA_origA"] - ps["memzero"][variant]["reid_resA_origA"]
                print(f"{pi:<6}{variant:<8}{'DELTA':<9}{dlt:>+16.4f}   (memon - memzero, KEY signal)")
    print("===============================================================")


if __name__ == "__main__":
    main()
