#!/usr/bin/env python
"""ECHELLE de FLOU fine, similarite DINO entre l'image INSEREE et l'image nette d'origine.

Pour chaque scenario du paquet (08, 09, 10) et chaque sigma de l'echelle {0, 1, 2, 3, 4,
6, 24} : l'image inseree est reconstruite par les operations exactes du hook _blur_latent
(decode -> flou sigma -> encode), puis re-decodee ; on mesure
  dino_cos  : cos(DINO(image inseree re-decodee), DINO(image nette d'origine))
  psnr_px   : PSNR pixel de la meme paire
sigma=0 (barreau `oracle`) insere la VRAIE frame latente sans aller-retour VAE ->
dino_cos = 1.0 par construction ; la ligne `rt0` (aller-retour VAE sans flou) est ajoutee
a titre informatif (= ce que rendrait un decodeur parfait qui repasse par le VAE).
Sortie : refs/blur_fine_dino.json + table imprimee -> conversion du verdict de l'oeil en
specification chiffree ("un decodeur doit atteindre au moins X de similarite DINO").
"""
import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import fidelity2_common as C                                             # noqa: E402

sys.path.insert(0, C.REPO)
sys.path.insert(0, C.PROBE_DIR)
os.chdir(C.REPO)

import torch                                                             # noqa: E402
import torchvision.transforms.functional as TVF                          # noqa: E402

SIGMAS = (0.0, 1.0, 2.0, 3.0, 4.0, 6.0, 24.0)


def psnr(a, b):
    mse = float(((a - b) ** 2).mean())
    return 99.0 if mse == 0 else 10.0 * math.log10(1.0 / mse)


def main():
    from utils.wan_wrapper import WanVAEWrapper
    from run_reid_probe import dino_embed
    device = torch.device("cuda")
    torch.set_grad_enabled(False)
    vae = WanVAEWrapper().to(device=device, dtype=torch.bfloat16)

    out = {}
    for sid in C.BLUR_SIDS:
        p = C.anchor_pt(sid)
        assert os.path.exists(p), f"{p} manquant"
        lat = torch.load(p, map_location="cpu",
                         weights_only=False)["anchor"].to(device, torch.bfloat16)
        px0 = vae.decode_to_pixel(lat, use_cache=False)
        vae.model.clear_cache()
        px0_01 = (px0[:, 0].float() * 0.5 + 0.5).clamp(0, 1)
        e0 = dino_embed(px0_01, device)[0]
        out[sid] = {}
        for tag, sig in [("oracle_sigma0", None)] + [(f"sigma{int(s)}" if s else "rt0", s)
                                                     for s in SIGMAS]:
            if sig is None:                      # barreau oracle : la vraie frame, telle quelle
                m = {"dino_cos": 1.0, "psnr_px": 99.0, "note": "vraie frame latente inseree"}
            else:
                if sig > 0:
                    k = int(6 * sig) | 1
                    pxb = TVF.gaussian_blur(px0[:, 0], kernel_size=[k, k],
                                            sigma=[sig, sig])[:, None]
                else:
                    pxb = px0
                lat_rt = vae.encode_to_latent(pxb.permute(0, 2, 1, 3, 4).to(torch.bfloat16))
                vae.model.clear_cache()
                px_ins = vae.decode_to_pixel(lat_rt.to(torch.bfloat16), use_cache=False)
                vae.model.clear_cache()
                px_ins_01 = (px_ins[:, 0].float() * 0.5 + 0.5).clamp(0, 1)
                ei = dino_embed(px_ins_01, device)[0]
                m = {"dino_cos": round(float(torch.dot(e0, ei)), 4),
                     "psnr_px": round(psnr(px_ins_01, px0_01), 2)}
            out[sid][tag] = m
            print(f"[dino] {sid} {tag:>13}: dino_cos {m['dino_cos']:.4f}  "
                  f"psnr {m['psnr_px']} dB", flush=True)

    # moyenne par sigma (la ligne de la spec)
    agg = {}
    for tag in next(iter(out.values())):
        vals = [out[sid][tag]["dino_cos"] for sid in out]
        agg[tag] = round(sum(vals) / len(vals), 4)
    print("[dino] moyenne dino_cos par niveau : "
          + "  ".join(f"{t}={v}" for t, v in agg.items()), flush=True)

    dst = os.path.join(C.REFS_DIR, "blur_fine_dino.json")
    with open(dst, "w") as f:
        json.dump({"per_scenario": out, "mean_dino_cos": agg,
                   "definition": "cos(DINO(image inseree re-decodee), DINO(frame nette "
                                 "d'origine)) ; sigma0=oracle (vraie frame, cos=1) ; rt0 = "
                                 "aller-retour VAE sans flou"}, f, indent=2)
    print(f"[dino] -> {dst}")


if __name__ == "__main__":
    main()
