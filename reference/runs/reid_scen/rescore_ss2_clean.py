#!/usr/bin/env python
# ============================================================================================
# SS2 RE-SCORE CONTRE UNE REFERENCE PROPRE  (pilote same-scene, 2026-08-02)
#
# POURQUOI. Dans SS2 (X -> A -> B -> A) le sujet A n'ouvre pas la video, et avec local_attn_size
# 12 + sink_size 1 le SUJET DE L'OUVERTURE reste epingle dans le sink tout le rollout : le
# segment que la campagne prend comme reference ("1re apparition de A", segment 1) montre en
# fait X (le barista / le jardinier / le livreur), pas A. Meme pathologie que S2 (cf.
# rescore_clean.py et section 8 de reference/scenarios_design_report.md), mais ici elle est
# VISUELLEMENT indiscutable puisque le decor est commun : seul le sujet distingue X de A.
#
# LA REFERENCE PROPRE. A* = segment 0 du rollout SS1 de la MEME cellule (meme triple, meme
# seed : cell_seed ne depend pas du scenario), ou A OUVRE la video et rien ne le precede. Le
# segment 0 est produit avant tout switch, donc identique pour tous les bras MAG ; stockoff est
# score contre SON PROPRE A* (autre checkpoint => une reference MAG confondrait pipeline et bras).
#
#   python rescore_ss2_clean.py
# ============================================================================================
import json
import os
import sys

REPO = "/shared/ssd_14T/home/adenelmi/Anchor-Forcing"
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [REPO, os.path.join(REPO, "reference/runs/reid_probe_l1"), HERE]

VID = "/shared/ssd_14T_project/adenelmi/reid_scen/videos"
OUT = os.path.join(HERE, "results", "ss2_clean")
ARMS = [("stockoff", "LongLive"), ("memzero", "sans memoire"),
        ("framebank", "banque Echo"), ("memdec20", "notre memoire"), ("oracle1", "oracle1")]
CELLS = [(12, "cafe"), (13, "pelouse"), (14, "boulangerie")]
SEED = 0


def main():
    import torch
    from torchvision.io import read_video
    from run_reid_probe import dino_embed, sample_seg, center_crop, seg_cos
    import scenarios as S

    torch.set_grad_enabled(False)
    device = torch.device("cuda")
    w_rev = S.score_segments("SS2", 240)[S.revisit_segments("SS2")[0]]      # (192, 239)
    w_fa = S.score_segments("SS2", 240)[1]                                  # reference contaminee
    w_clean = S.score_segments("SS1", 240)[0]                               # (20, 78), A ouvre

    cache = {}

    def emb(path, lo, hi):
        key = (path, lo, hi)
        if key not in cache:
            assert os.path.exists(path), f"video manquante {path}"
            v, _, _ = read_video(path, pts_unit="sec", output_format="THWC")
            seg = sample_seg(v.permute(0, 3, 1, 2).float() / 255., lo, hi, 8)
            cache[key] = (dino_embed(seg, device), dino_embed(center_crop(seg), device))
        return cache[key]

    res = {"reference": "SS1 segment 0 (A ouvre, meme triple / meme seed)", "cells": {}}
    for ti, name in CELLS:
        clean = {"mag": emb(f"{VID}/SS1_t{ti}_s{SEED}_memzero.mp4", *w_clean),
                 "stock": emb(f"{VID}/SS1_t{ti}_s{SEED}_stockoff.mp4", *w_clean)}
        rec = {"name": name}
        fa = emb(f"{VID}/SS2_t{ti}_s{SEED}_memzero.mp4", *w_fa)
        rec["_reference_contaminee_vs_Aclean"] = {
            "full": seg_cos(fa[0], clean["mag"][0]), "crop60": seg_cos(fa[1], clean["mag"][1])}
        for a, _ in ARMS:
            e = emb(f"{VID}/SS2_t{ti}_s{SEED}_{a}.mp4", *w_rev)
            pipe = "stock" if a == "stockoff" else "mag"
            rec[a] = {"full": seg_cos(e[0], clean[pipe][0]),
                      "crop60": seg_cos(e[1], clean[pipe][1]), "ref_pipeline": pipe}
        res["cells"][f"t{ti}"] = rec
        print(f"[ss2clean] t{ti} ({name}) fait", flush=True)

    # ---- tableaux ------------------------------------------------------------------------
    for v in ("full", "crop60"):
        print(f"\n===== SS2 : reID(retour de A vs A* PROPRE = SS1 seg0)   [{v}] =====")
        print(f"{'cellule':>14} " + " ".join(f"{lab:>13}" for _, lab in ARMS))
        for ti, name in CELLS:
            r = res["cells"][f"t{ti}"]
            print(f"{'t%d %s' % (ti, name):>14} " + " ".join(f"{r[a][v]:13.4f}" for a, _ in ARMS))
        print(f"{'MOYENNE':>14} " + " ".join(
            f"{sum(res['cells'][f't{ti}'][a][v] for ti, _ in CELLS) / len(CELLS):13.4f}"
            for a, _ in ARMS))
        m = sum(res["cells"][f"t{ti}"]["_reference_contaminee_vs_Aclean"][v]
                for ti, _ in CELLS) / len(CELLS)
        print(f"  [sanity] la reference de la campagne ('1re apparition de A', SS2 seg1) vs A* "
              f"PROPRE : {m:.4f}   <-- c'est contre CA que la campagne scorait")

    os.makedirs(OUT, exist_ok=True)
    p = os.path.join(OUT, "ss2_clean_rescore.json")
    with open(p, "w") as f:
        json.dump(res, f, indent=2)
    print(f"\n[ss2clean] ecrit {p}")


if __name__ == "__main__":
    main()
