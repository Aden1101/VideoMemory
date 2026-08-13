#!/usr/bin/env python
# ============================================================================================
# ECHELLE DE FIDELITE -- mesure de la degradation des barreaux de flou (GPU, VAE seul).
#
# Replique EXACTEMENT les operations du hook _blur_latent (pipeline/switch_causal_inference.py) :
# VAE decode -> flou gaussien sigma px (noyau int(6*sigma)|1) -> VAE encode, en bf16, et mesure
# pour chaque scenario et chaque sigma dans {0 (plancher aller-retour VAE), 6, 24} :
#   psnr_blur_px : PSNR du pixel flou vs pixel original (la degradation du flou seul)
#   psnr_rt_px   : PSNR du pixel decode(encode(flou)) vs original (ce que le contexte recoit,
#                  flou + aller-retour VAE)
#   rel_l2_lat   : ||lat_rt - lat_orig|| / ||lat_orig|| en espace latent
# + une planche JPEG par scenario [original | rt sigma=0 | rt sigma=6 | rt sigma=24].
# L'ancre mesuree = refs/anchor_<sid>.pt (le dump natif du rollout `rien`).
# Sorties : refs/blur_metrics.json, check/blur_board_<sid>.jpg. AUCUN verdict : documentation.
# ============================================================================================
import argparse
import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

_ap = argparse.ArgumentParser()
_ap.add_argument("--campaign", type=str, default="v1", choices=["v1", "nonopening"])
_ARGS = _ap.parse_args()
if _ARGS.campaign == "nonopening":
    import fidelity2_common as C                                         # noqa: E402
else:
    import fidelity_common as C                                          # noqa: E402

sys.path.insert(0, C.REPO)
os.chdir(C.REPO)

import torch                                                             # noqa: E402
import torchvision.transforms.functional as TVF                          # noqa: E402
import torchvision.utils as vutils                                       # noqa: E402

SIGMAS = (0.0, C.BLUR_SIGMA_LIGHT, C.BLUR_SIGMA_STRONG)


def psnr(a, b):
    """a, b dans [0,1]."""
    mse = float(((a - b) ** 2).mean())
    return 99.0 if mse == 0 else 10.0 * math.log10(1.0 / mse)


def main():
    from utils.wan_wrapper import WanVAEWrapper
    device = torch.device("cuda")
    torch.set_grad_enabled(False)
    vae = WanVAEWrapper().to(device=device, dtype=torch.bfloat16)

    out = {}
    for sc in C.SCENARIOS:
        sid = sc["sid"]
        p = C.anchor_pt(sid)
        assert os.path.exists(p), f"{p} manquant (lancer d'abord le bras `rien` de {sid})"
        d = torch.load(p, map_location="cpu", weights_only=False)
        lat = d["anchor"].to(device=device, dtype=torch.bfloat16)     # [1,1,16,60,104]
        px0 = vae.decode_to_pixel(lat, use_cache=False)               # [1,1,3,H,W] in [-1,1]
        vae.model.clear_cache()
        px0_01 = (px0[:, 0].float() * 0.5 + 0.5).clamp(0, 1)
        tiles = [px0_01[0].cpu()]
        out[sid] = {"anchor_frame": int(d.get("frame", C.ANCHOR_FRAME)), "sigmas": {}}
        for sig in SIGMAS:
            if sig > 0:
                k = int(6 * sig) | 1
                pxb = TVF.gaussian_blur(px0[:, 0], kernel_size=[k, k],
                                        sigma=[sig, sig])[:, None]
            else:
                pxb = px0
            lat_rt = vae.encode_to_latent(
                pxb.permute(0, 2, 1, 3, 4).to(torch.bfloat16))
            vae.model.clear_cache()
            px_rt = vae.decode_to_pixel(lat_rt.to(torch.bfloat16), use_cache=False)
            vae.model.clear_cache()
            pxb_01 = (pxb[:, 0].float() * 0.5 + 0.5).clamp(0, 1)
            px_rt_01 = (px_rt[:, 0].float() * 0.5 + 0.5).clamp(0, 1)
            rel = float((lat_rt.float() - lat.float()).norm() / (lat.float().norm() + 1e-12))
            m = {"psnr_blur_px": round(psnr(pxb_01, px0_01), 2),
                 "psnr_rt_px": round(psnr(px_rt_01, px0_01), 2),
                 "rel_l2_lat": round(rel, 4),
                 "kernel_px": (int(6 * sig) | 1) if sig > 0 else 0}
            out[sid]["sigmas"][str(sig)] = m
            tiles.append(px_rt_01[0].cpu())
            print(f"[blur] {sid} sigma={sig:>4}: blur-only PSNR {m['psnr_blur_px']} dB, "
                  f"apres aller-retour VAE {m['psnr_rt_px']} dB, rel-L2 latent "
                  f"{m['rel_l2_lat']}", flush=True)
        grid = vutils.make_grid(torch.stack(tiles), nrow=len(tiles), padding=6)
        board = os.path.join(C.CHECK_DIR, f"blur_board_{sid}.jpg")
        vutils.save_image(grid, board)
        print(f"[blur] planche [orig | rt0 | rt{C.BLUR_SIGMA_LIGHT:g} | "
              f"rt{C.BLUR_SIGMA_STRONG:g}] -> {board}", flush=True)

    dst = os.path.join(C.REFS_DIR, "blur_metrics.json")
    with open(dst, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[blur] metriques -> {dst}")


if __name__ == "__main__":
    main()
