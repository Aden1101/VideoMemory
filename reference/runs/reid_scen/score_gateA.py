#!/usr/bin/env python
# ============================================================================================
# GATE A (lot defensif, 2026-08-04) -- les 2 controles reviewer sur le tier MEME DECOR.
#
# Le §11 a tue le claim "contenu" sur le tier CROSS-SCENE : wronganchor y rendait 106 % du gain.
# Mais ce tier-la est celui ou le fond change a chaque switch, donc ou une frame quelconque
# suffit a "casser" l'interlude. Le tier MEME DECOR (SS2 : X -> A -> B -> A, meme lieu, seul le
# sujet change) est le seul ou le score ne peut venir QUE du sujet -- c'est le terrain de la
# these. GATE A y rejoue exactement les memes controles.
#
#   reference : A* PROPRE = segment 0 du rollout SS1 de la MEME cellule (A ouvre, rien avant).
#   contraste : reid(memdec20) - reid(wronganchor), par cellule.
#   REGLE SCELLEE : >= +0.08 sur >= 4/5 cellules  =>  le finetune est AUTORISE.
#                   sinon, la premisse ("notre memoire apporte son CONTENU") est falsifiee.
#
#   python score_gateA.py
# ============================================================================================
import json
import os
import statistics as st
import sys

REPO = "/shared/ssd_14T/home/adenelmi/Anchor-Forcing"
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [REPO, os.path.join(REPO, "reference/runs/reid_probe_l1"), HERE]

VID = "/shared/ssd_14T_project/adenelmi/reid_scen/videos"
OUT = os.path.join(HERE, "results", "gateA")
CELLS = [12, 13, 14, 15, 16]
SEED = 0
ARMS = ["memzero", "memdec20", "wronganchor", "noiseanchor", "framebank", "oracle1"]
THRESH = 0.08


def main():
    import torch
    from torchvision.io import read_video
    from run_reid_probe import dino_embed, sample_seg, center_crop, seg_cos, sign_test_p
    import scenarios as S

    torch.set_grad_enabled(False)
    dev = torch.device("cuda")
    w_rev = S.score_segments("SS2", 240)[S.revisit_segments("SS2")[0]]
    w_clean = S.score_segments("SS1", 240)[0]
    cache = {}

    def emb(p, w):
        if (p, w) not in cache:
            assert os.path.exists(p), f"video manquante {p}"
            v, _, _ = read_video(p, pts_unit="sec", output_format="THWC")
            seg = sample_seg(v.permute(0, 3, 1, 2).float() / 255., *w, 8)
            cache[(p, w)] = (dino_embed(seg, dev), dino_embed(center_crop(seg), dev))
        return cache[(p, w)]

    res = {"reference": "SS1 seg0 (A ouvre, meme cellule)", "cells": {}}
    for ti in CELLS:
        c = f"t{ti}_s{SEED}"
        clean = emb(f"{VID}/SS1_{c}_memzero.mp4", w_clean)
        rec = {}
        for a in ARMS:
            p = f"{VID}/SS2_{c}_{a}.mp4"
            if not os.path.exists(p):
                continue
            e = emb(p, w_rev)
            rec[a] = {"full": seg_cos(e[0], clean[0]), "crop60": seg_cos(e[1], clean[1])}
        res["cells"][c] = rec
        print(f"[gateA] {c}: {len(rec)} bras", flush=True)

    verdict = {}
    for v in ("crop60", "full"):
        print(f"\n===== GATE A -- SS2 (meme decor), re-ID du retour vs A* PROPRE  [{v}] =====")
        arms = [a for a in ARMS if any(a in r for r in res["cells"].values())]
        print(f"{'cellule':>10} " + " ".join(f"{a:>13}" for a in arms))
        for c, r in res["cells"].items():
            print(f"{c:>10} " + " ".join(
                f"{r[a][v]:13.4f}" if a in r else f"{'--':>13}" for a in arms))
        print(f"{'MOYENNE':>10} " + " ".join(
            f"{st.mean([r[a][v] for r in res['cells'].values() if a in r]):13.4f}"
            if any(a in r for r in res["cells"].values()) else f"{'--':>13}" for a in arms))
        # contrastes apparies
        print(f"\n  contrastes apparies (seuil scelle {THRESH:+.2f}) :")
        vv = {}
        for a, b in (("memdec20", "wronganchor"), ("memdec20", "noiseanchor"),
                     ("memdec20", "memzero"), ("wronganchor", "memzero")):
            d = {c: r[a][v] - r[b][v] for c, r in res["cells"].items() if a in r and b in r}
            if not d:
                continue
            npos = sum(1 for x in d.values() if x > 0)
            pas = sum(1 for x in d.values() if x >= THRESH)
            vv[f"{a}-{b}"] = {"per_cell": d, "mean": st.mean(d.values()), "n": len(d),
                              "n_pos": npos, "n_ge_thresh": pas,
                              "sign_p": sign_test_p(npos, len(d) - npos)}
            print(f"    {a:12s} - {b:12s} : " + "  ".join(f"{c.split('_')[0]} {x:+.3f}"
                                                          for c, x in sorted(d.items()))
                  + f"  | moyenne {st.mean(d.values()):+.4f}  {npos}+/{len(d) - npos}-  "
                    f"**{pas}/{len(d)} >= {THRESH:+.2f}**")
        verdict[v] = vv

    key = verdict["crop60"].get("memdec20-wronganchor", {})
    n_ok, n = key.get("n_ge_thresh", 0), key.get("n", 0)
    ok = n and n_ok >= 4 and n >= 5
    print(f"\n{'=' * 92}\nVERDICT GATE A (crop60, la vue primaire du tier meme-decor) : "
          f"{n_ok}/{n} cellules avec memdec20 - wronganchor >= {THRESH:+.2f}")
    print(f"  => {'FINETUNE AUTORISE (la premisse tient)' if ok else 'PREMISSE FALSIFIEE : sur le tier meme-decor non plus, le contenu de notre memoire ne fait pas la difference'}")
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "gateA.json"), "w") as f:
        json.dump({"scores": res, "contrasts": verdict,
                   "rule": {"threshold": THRESH, "need": "4/5 crop60"},
                   "n_ge_thresh": n_ok, "n_cells": n, "finetune_allowed": bool(ok)}, f, indent=2)
    print(f"[gateA] ecrit {OUT}/gateA.json")


if __name__ == "__main__":
    main()
