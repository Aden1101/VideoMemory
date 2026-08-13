#!/usr/bin/env python
# ============================================================================================
# SCORING DU LOT OFFENSIF (2026-08-04) -- un seul juge pour les trois leviers, celui du dossier :
# re-ID du RETOUR contre le A* PROPRE (le segment 0 du rollout ou A OUVRE la video : SS1 pour le
# tier meme-decor, S3 pour le tier cross-scene), crop60 en primaire et full en secondaire.
#
# LA REGLE TRANSVERSE DU LOT : le comparateur de CHAQUE bras est son PROPRE jumeau, jamais
# memzero. On imprime donc pour chaque levier le CONTRASTE APPARIE par cellule et le compte de
# cellules au-dessus du seuil pre-enregistre (+0.08), pas seulement des moyennes.
#
#   python score_offensif.py --lever 1     # legende  : ours - wrong (et - padded)
#   python score_offensif.py --lever 2     # I1       : i1_ours - i1_wrong, par t*
#   python score_offensif.py --lever 3     # additif  : add(p17) - memzero, chemin vers oracle1
# ============================================================================================
import argparse
import json
import os
import sys

REPO = "/shared/ssd_14T/home/adenelmi/Anchor-Forcing"
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [REPO, os.path.join(REPO, "reference/runs/reid_probe_l1"), HERE]

VID = "/shared/ssd_14T_project/adenelmi/reid_scen/videos"
OUT = os.path.join(HERE, "results", "offensif")
# scenario -> (scenario du A* propre, ses frames). Les scenarios de legende partagent la
# reference de leur forme de base : meme triple, meme seed, meme A qui ouvre.
CLEAN_REF = {"SS2": ("SS1", 240), "S2": ("S3", 300), "SS2c": ("SS1", 240), "S2c": ("S3", 300)}

LEVERS = {
    1: {"name": "LEVIER 1 -- LEGENDE (canal texte)",
        "cells": [("S2c", t, 0) for t in (0, 2, 3, 4, 11)]
                 + [("SS2c", t, 0) for t in (12, 13, 14)],
        "arms": ["caption_none", "padded", "caption_wrong", "caption_ours", "framebank_vlm"],
        "contrasts": [("caption_ours", "caption_wrong"), ("caption_ours", "padded"),
                      ("caption_wrong", "padded"), ("framebank_vlm", "caption_wrong")]},
    2: {"name": "LEVIER 2 -- I1, l'ancre DECLAREE bruitee",
        "cells": [("SS2", t, 0) for t in (12, 13, 14)] + [("S2", t, 0) for t in (2, 4, 11)],
        "arms": ["memzero", "wronganchor", "memdec20",
                 "i1_wrong250", "i1_ours250", "i1_wrong500", "i1_ours500"],
        "contrasts": [("i1_ours500", "i1_wrong500"), ("i1_ours250", "i1_wrong250"),
                      ("memdec20", "wronganchor"),
                      ("i1_ours500", "memdec20"), ("i1_ours250", "memdec20")]},
    4: {"name": "LEVIER 4 -- DECODER LE SNAPSHOT DE FIN DE SEGMENT A",
        # comparateurs deja tous sur disque : le jumeau contenu-faux (wronganchor) est LE juge,
        # memdec20 dit si changer d'INSTANT change quelque chose, memzero est le plancher.
        "cells": [("S2", t, 0) for t in (2, 3, 0, 4, 6, 11)],
        "arms": ["memzero", "wronganchor", "memdec20", "memdec_snap"],
        "contrasts": [("memdec_snap", "wronganchor"), ("memdec_snap", "memdec20"),
                      ("memdec20", "wronganchor"), ("memdec_snap", "memzero")]},
    3: {"name": "LEVIER 3 -- ADDITIF a Delta = 1",
        "cells": [("SS2", t, 0) for t in (12, 13, 14, 15)],
        "arms": ["memzero", "oracle1", "oracle1add8", "oracle1add8p9", "oracle1add1p17",
                 "oracle1add8p17", "memdec20add8p17", "memdec20"],
        "contrasts": [("oracle1add8p17", "memzero"), ("oracle1add1p17", "memzero"),
                      ("oracle1add8p9", "memzero"), ("oracle1", "memzero"),
                      ("oracle1add8p17", "oracle1add8p9")]},
}


def sign_p(npos, nneg):
    from run_reid_probe import sign_test_p
    return sign_test_p(npos, nneg)


def main():
    import torch
    from torchvision.io import read_video
    from run_reid_probe import dino_embed, sample_seg, center_crop, seg_cos
    import scenarios as S

    ap = argparse.ArgumentParser()
    ap.add_argument("--lever", type=int, required=True, choices=[1, 2, 3, 4])
    ap.add_argument("--thr", type=float, default=0.08)
    args = ap.parse_args()
    torch.set_grad_enabled(False)
    device = torch.device("cuda")
    spec = LEVERS[args.lever]

    cache = {}

    def emb(path, win, crop):
        k = (path, win, crop)
        if k not in cache:
            v, _, _ = read_video(path, pts_unit="sec", output_format="THWC")
            seg = sample_seg(v.permute(0, 3, 1, 2).float() / 255., win[0], win[1], 8)
            cache[k] = dino_embed(center_crop(seg) if crop else seg, device)
        return cache[k]

    res = {"lever": spec["name"], "cells": {}}
    print("=" * 100)
    print(spec["name"] + "   (re-ID contre le A* PROPRE, crop60 primaire)")
    print("=" * 100)
    for scen, ti, si in spec["cells"]:
        ref_scen, ref_frames = CLEAN_REF[scen]
        w_rev = S.score_segments(scen, S.SCENARIOS[scen]["frames"])[S.revisit_segments(scen)[0]]
        w_clean = S.score_segments(ref_scen, ref_frames)[0]
        p_ref = f"{VID}/{ref_scen}_t{ti}_s{si}_memzero.mp4"
        if not os.path.exists(p_ref):
            print(f"  {scen} t{ti} s{si}: A* propre manquant ({p_ref}), cellule sautee")
            continue
        e_ref = {c: emb(p_ref, w_clean, c) for c in (0, 1)}
        row = {}
        for a in spec["arms"]:
            p = f"{VID}/{scen}_t{ti}_s{si}_{a}.mp4"
            if not os.path.exists(p):
                continue
            row[a] = {"crop60": seg_cos(emb(p, w_rev, 1), e_ref[1]),
                      "full": seg_cos(emb(p, w_rev, 0), e_ref[0])}
        res["cells"][f"{scen}_t{ti}_s{si}"] = row
        print(f"\n  {scen} t{ti} s{si}   " + "  ".join(
            f"{a}={row[a]['crop60']:.4f}" for a in spec["arms"] if a in row))

    print("\n" + "=" * 100)
    print(f"CONTRASTES APPARIES (seuil pre-enregistre {args.thr:+.2f})")
    print("=" * 100)
    res["contrasts"] = {}
    for a, b in spec["contrasts"]:
        for view in ("crop60", "full"):
            d = {c: r[a][view] - r[b][view] for c, r in res["cells"].items() if a in r and b in r}
            if not d:
                continue
            v = list(d.values())
            m = sum(v) / len(v)
            npos = sum(1 for x in v if x > 0)
            nabove = sum(1 for x in v if x >= args.thr)
            res["contrasts"][f"{a}-{b}|{view}"] = {
                "mean": m, "n": len(v), "n_pos": npos, "n_above_thr": nabove,
                "sign_p": sign_p(npos, len(v) - npos), "per_cell": d}
            print(f"  {a:16s} - {b:16s} [{view:6s}] : moyenne {m:+.4f}  "
                  f"signes {npos}+/{len(v) - npos}-  p {sign_p(npos, len(v) - npos):.4f}  "
                  f"|  AU-DESSUS DU SEUIL {nabove}/{len(v)}")
            if view == "crop60":
                print("        par cellule : " + "  ".join(f"{k.split('_t')[1]}:{x:+.3f}"
                                                           for k, x in sorted(d.items())))
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, f"scores_lever{args.lever}.json"), "w") as f:
        json.dump(res, f, indent=2)
    print(f"\n[off] -> {OUT}/scores_lever{args.lever}.json")


if __name__ == "__main__":
    main()
