#!/usr/bin/env python
# ============================================================================================
# EYE-PACK of the ARBITER BATTERY (obligatory, not optional -- Aden checks the anchors himself).
# For each requested pair: ONE row, four columns, all through the REAL VAE at 480x832:
#   [ vraie frame d'ancre | thruwriter relue | twin (autre seed) | blur sigma ]
#     = original A anchor    = the frame the      = what the twin    = what the blur
#       (what oracle1          writer+decoder       arm splices        arm splices
#        splices)              re-read at the       (same prompt,      (true frame,
#                              EVICTION (age 1)     other instance)    our fidelity)
# The blur column is recomputed here with the EXACT inference round trip (VAE decode ->
# gaussian blur -> VAE encode -> VAE decode), so what you see is what gets spliced.
# The thruwriter mp4s of those pairs are copied next to the PNG.
#
# Usage (GPU pre-check in the SAME command):
#   CUDA_DEVICE_ORDER=PCI_BUS_ID nvidia-smi --query-gpu=index,memory.free --format=csv && \
#   CUDA_VISIBLE_DEVICES=1 python reference/runs/mem2/make_eyepack_probes.py
# ============================================================================================
import argparse
import glob
import os
import shutil
import sys

REPO = "/shared/ssd_14T/home/adenelmi/Anchor-Forcing"
PROBE_DIR = os.path.join(REPO, "reference/runs/reid_probe_l1")
FRAME_DEC = os.path.join(REPO, "reference/runs/frame_dec")
HERE = os.path.dirname(os.path.abspath(__file__))
for p in (REPO, PROBE_DIR, FRAME_DEC, HERE):
    sys.path.insert(0, p)
os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")

import torch  # noqa: E402
import torchvision.transforms.functional as TVF  # noqa: E402

from make_eyepack import grid, vae_frames  # noqa: E402
from run_reid_probe import ANCHOR_DIR, PAIRS  # noqa: E402
from train_decoder import build_vae  # noqa: E402

OUT = os.path.join(HERE, "eyepack_probes")


def load_lat(path):
    t = torch.load(path, map_location="cpu", weights_only=False)
    t = t["anchor"] if isinstance(t, dict) and "anchor" in t else t
    assert t.dim() == 5 and t.shape[1] == 1, f"{path}: shape {tuple(t.shape)}"
    return t


@torch.no_grad()
def blur_roundtrip(vae, lat, sigma, device):
    """The EXACT content the oracle1blur arm splices: VAE decode -> gaussian blur (sigma px)
    -> VAE encode (same ops as SwitchCausalInferencePipeline._blur_latent)."""
    px = vae.decode_to_pixel(lat.to(device=device, dtype=torch.bfloat16), use_cache=False)
    k = int(6 * sigma) | 1
    px = TVF.gaussian_blur(px[:, 0], kernel_size=[k, k], sigma=[sigma, sigma])[:, None]
    out = vae.encode_to_latent(px.permute(0, 2, 1, 3, 4).to(torch.bfloat16))
    vae.model.clear_cache()
    return out.float().cpu()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", default="1,4,7")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n_seeds", type=int, default=3)
    ap.add_argument("--twin_offset", type=int, default=1)
    ap.add_argument("--sigma", type=float, default=24.0)
    ap.add_argument("--anchor_dir", default=ANCHOR_DIR)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    os.chdir(REPO)
    os.makedirs(OUT, exist_ok=True)
    device = torch.device(args.device)
    torch.set_grad_enabled(False)
    vae = build_vae(device)

    rows, labels = [], []
    for pi in [int(x) for x in args.pairs.split(",") if x != ""]:
        si = args.seed
        tw = (si + args.twin_offset) % args.n_seeds
        a_true = os.path.join(args.anchor_dir, f"pair{pi}_s{si}.pt")
        a_twin = os.path.join(args.anchor_dir, f"pair{pi}_s{tw}.pt")
        thru = sorted(glob.glob(os.path.join(args.anchor_dir, f"thru_pair{pi}_s{si}.f*.pt")))
        missing = [p for p in (a_true, a_twin) if not os.path.exists(p)]
        if missing or not thru:
            print(f"[eyepack] pair{pi}: missing {missing or 'thru dump'} -- skipped", flush=True)
            continue
        lt, lw = load_lat(a_true), load_lat(a_twin)
        lh = load_lat(thru[-1])
        lb = blur_roundtrip(vae, lt, args.sigma, device)
        imgs = vae_frames(vae, torch.cat([lt[:, 0], lh[:, 0].float(), lw[:, 0], lb[:, 0]], 0),
                          device)
        rows.append(imgs)
        labels.append(f"pair{pi} s{si}\n{PAIRS[pi][0][:40]}\n(twin = s{tw})")
        print(f"[eyepack] pair{pi}: true {os.path.basename(a_true)}, thru "
              f"{os.path.basename(thru[-1])}, twin s{tw}", flush=True)
        # the mp4 the eye must watch: the thruwriter cell itself
        src = os.path.join(PROBE_DIR, "videos", f"pair{pi}_s{si}_oracle1thru.mp4")
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(OUT, f"pair{pi}_s{si}_oracle1thru.mp4"))
            print(f"[eyepack] copied {os.path.basename(src)}", flush=True)

    assert rows, "no pair had all its artefacts -- run the campaign first"
    p = os.path.join(OUT, "anchors_true_thru_twin_blur.png")
    grid(rows, ["frame VRAIE (oracle1)", "thruwriter relue (age 1)",
                f"twin (autre seed)", f"blur sigma={args.sigma:g}"], p, row_labels=labels)
    print(f"[eyepack] -> {p}", flush=True)


if __name__ == "__main__":
    main()
