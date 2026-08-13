#!/usr/bin/env python
# ============================================================================================
# SONDE D'IDENTITE DE L'ANCRE (2026-08-04) -- la mesure que la planche d'oeil a rendue evidente
# et qu'aucune campagne n'avait faite : **est-ce qu'il y a seulement de l'identite DANS l'ancre ?**
#
# Tout le dossier (§11, §12, §13) mesure ce que le RENDU fait du contenu splice, donc il ne peut
# jamais separer "le canal splice -> rendu est aveugle au contenu" de "il n'y a pas de contenu".
# Ici on court-circuite le canal : on VAE-decode l'ancre en PIXELS et on la compare directement,
# en DINO, a deux references PROPRES :
#     - le A* de SA cellule       (le sujet qu'elle est censee porter)
#     - le A* de la cellule SOURCE (le sujet que `wronganchor` lui substitue)
# Si les deux cosinus sont EGAUX, l'ancre ne porte aucune identite d'instance : le negatif du
# §11 n'a plus besoin d'une explication sur le canal, il est explique en amont.
#
# La meme mesure est refaite sur les ancres AFFUTEES : l'affutage peut-il faire apparaitre une
# identite que le decodeur n'a pas produite ? (Non, par construction -- mais il faut le montrer.)
#
#   python probe_anchor_identity.py --dir eyepack_tfree/grid
# ============================================================================================
import argparse
import glob
import json
import os
import re
import sys

REPO = "/shared/ssd_14T/home/adenelmi/Anchor-Forcing"
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [REPO, os.path.join(REPO, "reference/runs/reid_probe_l1"), HERE]

VID = "/shared/ssd_14T_project/adenelmi/reid_scen/videos"
CLEAN_REF = {"SS2": ("SS1", 240), "S2": ("S3", 300)}
CELLS = [("SS2", 12), ("SS2", 13), ("SS2", 14), ("S2", 2), ("S2", 4), ("S2", 11)]


def main():
    import torch
    import numpy as np
    from PIL import Image
    from torchvision.io import read_video
    from run_reid_probe import dino_embed, sample_seg, center_crop, seg_cos
    import scenarios as S

    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=os.path.join(HERE, "eyepack_tfree", "grid"))
    ap.add_argument("--out", default=os.path.join(HERE, "results", "tfree"))
    args = ap.parse_args()
    torch.set_grad_enabled(False)
    dev = torch.device("cuda")

    def emb_png(p):
        x = torch.from_numpy(np.array(Image.open(p).convert("RGB"))).permute(2, 0, 1)[None] / 255.
        return dino_embed(x, dev), dino_embed(center_crop(x), dev)

    def emb_astar(scen, ti):
        rs, rf = CLEAN_REF[scen]
        lo, hi = S.score_segments(rs, rf)[0]
        v, _, _ = read_video(f"{VID}/{rs}_t{ti}_s0_memzero.mp4", pts_unit="sec",
                             output_format="THWC")
        seg = sample_seg(v.permute(0, 3, 1, 2).float() / 255., lo, hi, 8)
        return dino_embed(seg, dev), dino_embed(center_crop(seg), dev)

    astar = {}
    res, rows = {}, []
    for scen, ti in CELLS:
        src = S.wrong_anchor_triple(ti)
        for t in (ti, src):
            if (scen, t) not in astar:
                astar[(scen, t)] = emb_astar(scen, t)
        variants = {}
        for p in sorted(glob.glob(os.path.join(args.dir, f"anchor_*_{scen}_t{ti}.png"))):
            m = re.match(r"anchor_(raw|s[\d.]+)_", os.path.basename(p))
            variants[m.group(1)] = p
        cell = {"source_triple": src, "variants": {}}
        for name, p in variants.items():
            e = emb_png(p)
            own = {"full": seg_cos(e[0], astar[(scen, ti)][0]),
                   "crop60": seg_cos(e[1], astar[(scen, ti)][1])}
            oth = {"full": seg_cos(e[0], astar[(scen, src)][0]),
                   "crop60": seg_cos(e[1], astar[(scen, src)][1])}
            cell["variants"][name] = {"vs_own_Astar": own, "vs_source_Astar": oth,
                                      "margin_full": own["full"] - oth["full"],
                                      "margin_crop60": own["crop60"] - oth["crop60"]}
            rows.append((scen, ti, name, own["crop60"], oth["crop60"],
                         own["crop60"] - oth["crop60"]))
        res[f"{scen}_t{ti}"] = cell

    names = sorted({r[2] for r in rows}, key=lambda x: (x != "raw", x))
    print("\n===== L'ANCRE PORTE-T-ELLE UNE IDENTITE ?  DINO crop60 de l'ancre DECODEE (pixels) "
          "contre 2 A* PROPRES =====")
    print(f"{'cellule':>10} {'variante':>8} {'vs SON A*':>11} {'vs A* SOURCE':>13} "
          f"{'MARGE':>9}   (marge ~ 0 => aucune identite d'instance)")
    for scen, ti, name, o, x, mg in rows:
        print(f"{f'{scen} t{ti}':>10} {name:>8} {o:11.4f} {x:13.4f} {mg:+9.4f}")
    print()
    summ = {}
    for n in names:
        sel = [r for r in rows if r[2] == n]
        mo = sum(r[3] for r in sel) / len(sel)
        mx = sum(r[4] for r in sel) / len(sel)
        mg = sum(r[5] for r in sel) / len(sel)
        npos = sum(1 for r in sel if r[5] > 0)
        summ[n] = {"mean_vs_own": mo, "mean_vs_source": mx, "mean_margin": mg,
                   "n_pos": npos, "n": len(sel)}
        print(f"MOYENNE {n:>8}: vs SON A* {mo:.4f}   vs A* SOURCE {mx:.4f}   "
              f"MARGE {mg:+.4f}  ({npos}/{len(sel)} cellules positives)")
    os.makedirs(args.out, exist_ok=True)
    p = os.path.join(args.out, "anchor_identity.json")
    with open(p, "w") as f:
        json.dump({"cells": res, "summary": summ}, f, indent=2)
    print(f"\n[probe] ecrit {p}")


if __name__ == "__main__":
    main()
