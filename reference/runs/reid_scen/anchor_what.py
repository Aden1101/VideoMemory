#!/usr/bin/env python
# ============================================================================================
# DE QUOI L'ANCRE DECODEE EST-ELLE L'IMAGE ?  (0 rollout)
#
# Les mesures de la nuit disent que l'ancre ne porte ni l'identite de A (DINO +0.0007, p = 0.84),
# ni sa couleur (+0.128, 13/24, p = 0.84), ni meme sa structure grossiere (+0.014, 13/24). Elle
# n'est pourtant PAS generique : T0 montre qu'elle est copiee dans le rendu de facon SPECIFIQUE
# (0.47-0.52 contre un plancher de 0.10). Donc elle est l'image de QUELQUE CHOSE. De quoi ?
#
# L'hypothese mecanique, qui vient du DECODEUR lui-meme : il a ete entraine a reconstruire la
# frame EVINCEE de la fenetre. Au switch de RETOUR (frame 180), les frames qui viennent d'etre
# evincees appartiennent a l'INTERLUDE B, pas a A. Un decodeur PARFAIT rendrait donc une frame
# de B. Test : comparer l'ancre a A*, a B (l'interlude) et a X (l'ouvreur epingle dans le sink).
#
#   python anchor_what.py
# ============================================================================================
import json
import os
import sys

REPO = "/shared/ssd_14T/home/adenelmi/Anchor-Forcing"
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [REPO, os.path.join(REPO, "reference/runs/reid_probe_l1"), HERE]

import torch                                                            # noqa: E402
from torchvision.io import read_video                                   # noqa: E402
import scenarios as S                                                   # noqa: E402
from prescreen_offensif import CACHE, VID, OUT, CELLS, sign_p, lowband  # noqa: E402


def main():
    from run_reid_probe import sample_seg, dino_embed, center_crop, seg_cos
    torch.set_grad_enabled(False)
    dev = torch.device("cuda")
    wA = S.score_segments("S3", 300)[0]          # A* propre (A ouvre, rien avant lui)
    segs = S.score_segments("S2", 240)
    wB, wX = segs[2], segs[0]                    # l'interlude B ; l'ouvreur X (epingle au sink)
    rows = []
    for ti, si in CELLS:
        cp = os.path.join(CACHE, f"S2_t{ti}_s{si}_anchor.pt")
        pA, pS = f"{VID}/S3_t{ti}_s{si}_memzero.mp4", f"{VID}/S2_t{ti}_s{si}_memzero.mp4"
        if not all(map(os.path.exists, (cp, pA, pS))):
            continue
        anc = torch.load(cp, map_location="cpu")[None]
        vA, _, _ = read_video(pA, pts_unit="sec", output_format="THWC")
        vS, _, _ = read_video(pS, pts_unit="sec", output_format="THWC")
        f = {"A": sample_seg(vA.permute(0, 3, 1, 2).float() / 255., *wA, 16),
             "B": sample_seg(vS.permute(0, 3, 1, 2).float() / 255., *wB, 16),
             "X": sample_seg(vS.permute(0, 3, 1, 2).float() / 255., *wX, 16)}
        la, ea = lowband(anc)[0], dino_embed(center_crop(anc), dev)
        r = {"triple": ti, "seed": si}
        for nm, fr in f.items():
            r["lb_" + nm] = float((lowband(fr) @ la).mean())
            r["dino_" + nm] = seg_cos(ea, dino_embed(center_crop(fr), dev))
        rows.append(r)
        print(f"  t{ti} s{si}: bande basse  A {r['lb_A']:+.3f}  B {r['lb_B']:+.3f}  "
              f"X {r['lb_X']:+.3f}   |  DINO  A {r['dino_A']:.3f}  B {r['dino_B']:.3f}  "
              f"X {r['dino_X']:.3f}", flush=True)

    print("\n" + "=" * 96)
    print("DE QUOI L'ANCRE EST-ELLE L'IMAGE ?  A* = le sujet a retrouver | B = l'interlude dont "
          "les frames\nviennent d'etre EVINCEES (ce que le decodeur a appris a reconstruire) | "
          "X = l'ouvreur epingle au sink")
    print("=" * 96)
    summ = {"n": len(rows), "cells": rows, "verdict": {}}
    for k, lab in (("lb", "bande basse 8x8"), ("dino", "DINO crop60")):
        mm = {nm: sum(r[f"{k}_{nm}"] for r in rows) / len(rows) for nm in ("A", "B", "X")}
        nb = sum(1 for r in rows if r[f"{k}_B"] > r[f"{k}_A"])
        nx = sum(1 for r in rows if r[f"{k}_X"] > r[f"{k}_A"])
        summ["verdict"][k] = {"vs_A": mm["A"], "vs_B": mm["B"], "vs_X": mm["X"],
                              "n_B_gt_A": nb, "n_X_gt_A": nx, "n": len(rows),
                              "sign_p_B_gt_A": sign_p(nb, len(rows) - nb)}
        print(f"  [{lab:16s}] ancre vs A* {mm['A']:+.4f} | vs B {mm['B']:+.4f} | "
              f"vs X {mm['X']:+.4f}")
        print(f"                     B plus proche que A sur {nb}/{len(rows)} cellules "
              f"(p {sign_p(nb, len(rows) - nb):.4f}) ; X plus proche que A sur {nx}/{len(rows)}")
    with open(os.path.join(OUT, "anchor_what_is_it.json"), "w") as f:
        json.dump(summ, f, indent=2)
    print(f"\n[anc] -> {OUT}/anchor_what_is_it.json")


if __name__ == "__main__":
    main()
