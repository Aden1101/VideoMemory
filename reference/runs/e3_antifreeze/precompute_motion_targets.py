#!/usr/bin/env python
"""Precompute the per-prompt stock-AF latent-motion target curves for e3_antifreeze.

For every prompt p and chunk position i, computes m[p][i] = mean |x_t+1 - x_t| over the latent
frames of that chunk window, using EXACTLY the training statistic and index mapping
(model/streaming_training.py::_motion_target_index):
    chunk 0 : latent frames [0, 21)
    chunk i : latent frames [21 + 18*(i-1), 21 + 18*i)          (the 18 NEW frames)
Averaged over seeds. Output JSON: { sha1(prompt.strip())[:16]: [m_0, m_1, ...], "_global": [...] }.

Input videos: stock-AF mp4s from gen_stock_motion.sh (mem-OFF also fine via MEM_ZERO_ALL=1).
Latents are recovered by VAE-encoding each chunk's pixel window with one latent of left context
(dropped) — a small systematic offset vs true streaming latents is possible; the training-side
calibration step (compare wandb e3_motion_on at healthy EARLY chunks vs this table; rescale by the
constant if off >1.3x) absorbs it.

Usage (repo root, longlive env, 1 GPU):
  python reference/runs/e3_antifreeze/precompute_motion_targets.py \
      --video_dirs reference/runs/e3_antifreeze/stock_videos/seed42 \
                   reference/runs/e3_antifreeze/stock_videos/seed43 \
                   reference/runs/e3_antifreeze/stock_videos/seed44 \
      --prompts reference/runs/e0_reid/prompts/e0_dynamic8.txt \
      --out reference/runs/e3_antifreeze/motion_targets.json

Calibration mode (--pair_dose A_DIR B_DIR): computes the training-side AC-dose statistic
(RMS of the per-channel-mean-removed latent difference, / AC-RMS of B) per chunk between two
SAME-SEED video dirs. Run it on eval1000 memdc vs memoff = the INERT reference; set the config
act_floor ~3-5x above it:
  python reference/runs/e3_antifreeze/precompute_motion_targets.py \
      --pair_dose reference/runs/e1_rerun/eval1000/videos/memdc \
                  reference/runs/e1_rerun/eval1000/videos/memoff \
      --prompts reference/runs/e0_reid/prompts/e0_dynamic8.txt --out /dev/null
"""
import argparse, glob, hashlib, json, os, sys

import torch

CHUNK, STEP = 21, 18                      # latent chunk size / new frames per chunk (training values)


def pix_range_for_latents(a, b):
    """Pixel-frame range that decodes to latent frames [a, b) (Wan: T_pix = 4*(T_lat-1)+1).
    latent 0 <- pixel 0; latent j>=1 <- pixels [4j-3 .. 4j]."""
    p0 = 0 if a == 0 else 4 * a - 3
    p1 = 4 * (b - 1) + 1
    return p0, p1


def encode_window(vae, frames_u8, a, b, device):
    """VAE-encode pixel frames covering latents [a-ctx, b) and return latents [a, b).
    frames_u8: [T, H, W, C] uint8. Returns [n, C, h, w] float latents."""
    ctx = 0 if a == 0 else 1                            # one latent of left context, dropped
    p0, p1 = pix_range_for_latents(a - ctx, b)
    seg = frames_u8[p0:p1].to(device)
    seg = seg.permute(3, 0, 1, 2).unsqueeze(0).float() / 127.5 - 1.0   # [1, C, T, H, W] in [-1,1]
    with torch.no_grad():
        lat = vae.encode_to_latent(seg.to(torch.bfloat16))             # [1, T_lat, C, h, w]
    return lat[0, ctx:].float()


def ac_dose(lat_a, lat_b):
    """Training-side AC-dose between two latent windows [n, C, h, w] (t=dim0 here)."""
    d = lat_a - lat_b
    d_ac = d - d.mean(dim=(0, 2, 3), keepdim=True)
    b_ac = lat_b - lat_b.mean(dim=(0, 2, 3), keepdim=True)
    return (d_ac.pow(2).mean().sqrt() / b_ac.pow(2).mean().sqrt().clamp_min(1e-6)).item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video_dirs", nargs="+", default=None)
    ap.add_argument("--pair_dose", nargs=2, default=None, metavar=("A_DIR", "B_DIR"))
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--lag2", action="store_true",
                    help="e4: use the flicker-blind lag-2 motion statistic 0.5*mean|x_t+2 - x_t| "
                         "(must match streaming_training.py::_compute_advantage_loss motion_lag2)")
    args = ap.parse_args()

    sys.path.insert(0, os.getcwd())                     # repo root (wan_models/ relative path)
    from torchvision.io import read_video
    from utils.wan_wrapper import WanVAEWrapper

    prompts = [l.strip() for l in open(args.prompts) if l.strip()]
    vae = WanVAEWrapper().to(args.device, torch.bfloat16).eval()

    if args.pair_dose:
        a_dir, b_dir = args.pair_dose
        doses = []
        for fa in sorted(glob.glob(os.path.join(a_dir, "*.mp4"))):
            fb = os.path.join(b_dir, os.path.basename(fa))
            if not os.path.exists(fb):
                continue
            fr_a, _, _ = read_video(fa, pts_unit="sec", output_format="THWC")
            fr_b, _, _ = read_video(fb, pts_unit="sec", output_format="THWC")
            T_lat = (min(fr_a.shape[0], fr_b.shape[0]) - 1) // 4 + 1
            n_chunks = 1 + max(0, (T_lat - CHUNK) // STEP)
            per_chunk = []
            for i in range(n_chunks):
                a = 0 if i == 0 else CHUNK + STEP * (i - 1)
                b = CHUNK if i == 0 else CHUNK + STEP * i
                la = encode_window(vae, fr_a, a, b, args.device)
                lb = encode_window(vae, fr_b, a, b, args.device)
                per_chunk.append(ac_dose(la, lb))
            doses.append(sum(per_chunk) / len(per_chunk))
            print(f"{os.path.basename(fa)}: mean AC-dose {doses[-1]:.4f} "
                  f"(per-chunk {['%.3f' % v for v in per_chunk]})", flush=True)
        print(f"\nPAIR AC-DOSE mean over {len(doses)} pairs = {sum(doses)/len(doses):.4f} "
              f"-> set act_floor ~3-5x this if this is the inert (memdc vs memoff) reference")
        return

    if not args.video_dirs:
        ap.error("either --video_dirs or --pair_dose is required")

    per_prompt = {}                                     # idx -> list over seeds of [m_0, m_1, ...]
    for d in args.video_dirs:
        vids = sorted(glob.glob(os.path.join(d, "*.mp4")))
        for f in vids:
            idx = int(os.path.basename(f).split("-")[0])    # save_with_index naming
            frames, _, _ = read_video(f, pts_unit="sec", output_format="THWC")
            T_lat = (frames.shape[0] - 1) // 4 + 1
            n_chunks = 1 + max(0, (T_lat - CHUNK) // STEP)
            curve = []
            for i in range(n_chunks):
                a = 0 if i == 0 else CHUNK + STEP * (i - 1)
                b = CHUNK if i == 0 else CHUNK + STEP * i
                lat = encode_window(vae, frames, a, b, args.device)     # [n, C, h, w]
                if args.lag2 and lat.shape[0] >= 3:
                    m = ((lat[2:] - lat[:-2]).abs().mean() * 0.5).item()
                else:
                    m = (lat[1:] - lat[:-1]).abs().mean().item()
                curve.append(m)
            per_prompt.setdefault(idx, []).append(curve)
            print(f"[{os.path.basename(d)}] prompt {idx}: {len(curve)} chunks, "
                  f"m0={curve[0]:.4f} mLast={curve[-1]:.4f}", flush=True)

    out = {}
    all_curves = []
    for idx, curves in sorted(per_prompt.items()):
        L = min(len(c) for c in curves)
        avg = [sum(c[i] for c in curves) / len(curves) for i in range(L)]
        key = hashlib.sha1(prompts[idx].strip().encode()).hexdigest()[:16]
        out[key] = avg
        all_curves.append(avg)
        print(f"prompt {idx} ({key}): {['%.4f' % v for v in avg]}")
    L = min(len(c) for c in all_curves)
    out["_global"] = [sum(c[i] for c in all_curves) / len(all_curves) for i in range(L)]

    with open(args.out, "w") as f:
        json.dump(out, f, indent=1)
    print(f"wrote {args.out}: {len(out) - 1} prompts + _global, {L}+ chunk positions")


if __name__ == "__main__":
    main()
