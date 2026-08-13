#!/usr/bin/env python
# ============================================================================================
# 16.1 quater, LE TEST QUI TRANCHE : sur le tier CROSS-SCENE, dire "l'ancre ressemble a B" peut
# vouloir dire seulement "l'ancre ressemble a la SCENE de B". Sur le tier MEME-DECOR (SS2), A et
# B partagent le decor VERBATIM et ne different QUE PAR LE SUJET. Si l'ancre y ressemble encore
# davantage a B qu'a A, alors elle ne porte pas "la mauvaise scene" : elle porte LE MAUVAIS SUJET.
#
# References de la cellule (memes conventions que le reste du dossier) :
#   A* = SS1 segment 0 (A ouvre, rien avant lui)   B = SS2 segment 2   X = SS2 segment 0
#
#   python anchor_what_ss2.py
# ============================================================================================
import json
import os
import sys

REPO = "/shared/ssd_14T/home/adenelmi/Anchor-Forcing"
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [REPO, os.path.join(REPO, "reference/runs/reid_probe_l1"), HERE]
os.chdir(REPO)

import torch                                                            # noqa: E402
from torchvision.io import read_video                                   # noqa: E402
import scenarios as S                                                   # noqa: E402
from prescreen_offensif import CACHE, VID, OUT, sign_p, lowband         # noqa: E402
import prescreen_offensif as PR                                         # noqa: E402

CELLS = [(12, 0), (13, 0), (14, 0), (15, 0)]


def anchor_px_ss2(ti, si, device):
    cpath = os.path.join(CACHE, f"SS2_t{ti}_s{si}_anchor.pt")
    if os.path.exists(cpath):
        return torch.load(cpath, map_location="cpu")
    lat = torch.load(os.path.join(S.DEC_ANCHOR_DIR, f"SS2_t{ti}_s{si}_decanchor.f180.pt"),
                     map_location="cpu")
    from utils.wan_wrapper import WanVAEWrapper
    if getattr(PR, "_VAE", None) is None:
        PR._VAE = WanVAEWrapper().to(device=device, dtype=torch.bfloat16).eval()
    px = PR._VAE.decode_to_pixel(lat.to(device=device, dtype=torch.bfloat16), use_cache=False)
    PR._VAE.model.clear_cache()
    im = (px[0].float() * 0.5 + 0.5).clamp(0, 1).mean(0).cpu()
    os.makedirs(CACHE, exist_ok=True)
    torch.save(im, cpath)
    return im


def main():
    from run_reid_probe import sample_seg, dino_embed, center_crop, seg_cos
    torch.set_grad_enabled(False)
    dev = torch.device("cuda")
    wA = S.score_segments("SS1", 240)[0]
    segs = S.score_segments("SS2", 240)
    wB, wX = segs[2], segs[0]
    rows = []
    for ti, si in CELLS:
        pA, pS = f"{VID}/SS1_t{ti}_s{si}_memzero.mp4", f"{VID}/SS2_t{ti}_s{si}_memzero.mp4"
        src = os.path.join(S.DEC_ANCHOR_DIR, f"SS2_t{ti}_s{si}_decanchor.f180.pt")
        if not all(map(os.path.exists, (pA, pS, src))):
            print(f"  t{ti}: reference ou dump manquant, saute")
            continue
        vA, _, _ = read_video(pA, pts_unit="sec", output_format="THWC")
        vS, _, _ = read_video(pS, pts_unit="sec", output_format="THWC")
        f = {"A": sample_seg(vA.permute(0, 3, 1, 2).float() / 255., *wA, 8),
             "B": sample_seg(vS.permute(0, 3, 1, 2).float() / 255., *wB, 8),
             "X": sample_seg(vS.permute(0, 3, 1, 2).float() / 255., *wX, 8)}
        anc = anchor_px_ss2(ti, si, dev)[None]
        la, ea = lowband(anc)[0], dino_embed(center_crop(anc), dev)
        r = {"triple": ti, "seed": si}
        for nm, fr in f.items():
            r["lb_" + nm] = float((lowband(fr) @ la).mean())
            r["dino_" + nm] = seg_cos(ea, dino_embed(center_crop(fr), dev))
        rows.append(r)
        print(f"  t{ti}: bande basse  A {r['lb_A']:+.3f}  B {r['lb_B']:+.3f}  X {r['lb_X']:+.3f}"
              f"   |  DINO crop60  A {r['dino_A']:.3f}  B {r['dino_B']:.3f}  X {r['dino_X']:.3f}",
              flush=True)
    if not rows:
        print("aucune cellule exploitable")
        return
    print("\n" + "=" * 96)
    print("TIER MEME-DECOR : A et B partagent le decor VERBATIM, seul le SUJET change.")
    print("Si l'ancre reste plus proche de B, elle porte LE MAUVAIS SUJET, pas la mauvaise scene.")
    print("=" * 96)
    summ = {"n": len(rows), "cells": rows, "verdict": {}}
    for k, lab in (("lb", "bande basse 8x8"), ("dino", "DINO crop60")):
        mm = {nm: sum(r[f"{k}_{nm}"] for r in rows) / len(rows) for nm in ("A", "B", "X")}
        nb = sum(1 for r in rows if r[f"{k}_B"] > r[f"{k}_A"])
        summ["verdict"][k] = dict(vs_A=mm["A"], vs_B=mm["B"], vs_X=mm["X"], n_B_gt_A=nb,
                                  n=len(rows), sign_p=sign_p(nb, len(rows) - nb))
        print(f"  [{lab:16s}] ancre vs A* {mm['A']:+.4f} | vs B {mm['B']:+.4f} | "
              f"vs X {mm['X']:+.4f}   -- B plus proche que A sur {nb}/{len(rows)}")
    with open(os.path.join(OUT, "anchor_what_ss2.json"), "w") as f:
        json.dump(summ, f, indent=2)
    print(f"\n[anc] -> {OUT}/anchor_what_ss2.json")


if __name__ == "__main__":
    main()
