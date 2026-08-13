#!/usr/bin/env python
# ============================================================================================
# SCORING du PILOTE "TEXTE INSUFFISANT" (2026-08-03) -- 2 cellules, 4+2 bras.
#
# Metriques, telles que PRE-ENREGISTREES dans memory2_probes_registry.md :
#   reid        cos(retour, A*) full et crop60, A* = segment 0 du rollout SS1 de la MEME cellule
#               (A ouvre, prompt complet, rien avant lui)
#   contraste   reid(memdec20) - reid(wronganchor) : LA signature d'une lecture du contenu
#   margin      t18 seulement : cos(retour, A_ref) - cos(retour, B_ref) avec A_ref = segment 1
#               (chien brun) et B_ref = segment 2 (chien blanc) du rollout memzero -- reference
#               commune a tous les bras, dans le meme rollout, donc symetrique
#   attributs   t17 seulement : CLIP ViT-B/32, cos(retour, "jet-black coat / white mane")
#               - cos(retour, "brown horse / dark mane"), + la luminance du crop central
#               (une robe noire est SOMBRE : mesure objective, aucun modele)
#
#   python score_textpoor.py
# ============================================================================================
import json
import os
import sys

REPO = "/shared/ssd_14T/home/adenelmi/Anchor-Forcing"
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [REPO, os.path.join(REPO, "reference/runs/reid_probe_l1"), HERE]

import torch                                                            # noqa: E402
from torchvision.io import read_video                                   # noqa: E402
from run_reid_probe import dino_embed, sample_seg, center_crop, seg_cos  # noqa: E402
import scenarios as S                                                   # noqa: E402

VID = "/shared/ssd_14T_project/adenelmi/reid_scen/videos"
OUT = os.path.join(HERE, "results", "textpoor")
ARMS = ["memzero", "memdec20", "wronganchor", "framebank_gt", "framebank_oracle_gt", "framebank"]
CELLS = [(17, 0), (18, 0)]
ATTR = {
    17: [("cible", "a jet-black horse with a snow-white mane"),
         ("stereotype", "a brown horse with a dark mane"),
         ("clair", "a white horse with a light mane")],
}


def main():
    torch.set_grad_enabled(False)
    dev = torch.device("cuda")
    from score_comp import clip_text_score
    w_rev = S.score_segments("S2p", 240)[3]                             # (192, 239) le retour
    w_a = S.score_segments("S2p", 240)[1]                               # (72, 117)  A vu
    w_b = S.score_segments("S2p", 240)[2]                               # (132, 177) B vu
    w_clean = S.score_segments("SS1", 240)[0]                           # (20, 78)   A* propre
    cache = {}

    def frames(path, win):
        if (path, win) not in cache:
            assert os.path.exists(path), f"missing {path}"
            v, _, _ = read_video(path, pts_unit="sec", output_format="THWC")
            cache[(path, win)] = sample_seg(v.permute(0, 3, 1, 2).float() / 255., *win, 8)
        return cache[(path, win)]

    def emb(path, win):
        f = frames(path, win)
        return dino_embed(f, dev), dino_embed(center_crop(f), dev)

    res = {}
    for ti, si in CELLS:
        cell = f"t{ti}_s{si}"
        clean = emb(f"{VID}/SS1_{cell}_memzero.mp4", w_clean)
        aref = emb(f"{VID}/S2p_{cell}_memzero.mp4", w_a)
        bref = emb(f"{VID}/S2p_{cell}_memzero.mp4", w_b)
        rec = {"prompt_A": S.TRIPLES[ti][0], "prompt_B": S.TRIPLES[ti][1],
               "prompt_retour": S.PARAPHRASES[ti], "arms": {}}
        for a in ARMS:
            p = f"{VID}/S2p_{cell}_{a}.mp4"
            if not os.path.exists(p):
                continue
            e = emb(p, w_rev)
            row = {"reid_full": seg_cos(e[0], clean[0]), "reid_crop60": seg_cos(e[1], clean[1]),
                   "vs_Aref_crop60": seg_cos(e[1], aref[1]),
                   "vs_Bref_crop60": seg_cos(e[1], bref[1])}
            row["margin_crop60"] = row["vs_Aref_crop60"] - row["vs_Bref_crop60"]
            f = frames(p, w_rev)
            row["luminance_crop60"] = float(center_crop(f).mean())
            if ti in ATTR:
                for name, txt in ATTR[ti]:
                    row[f"clip_{name}"] = clip_text_score(f, txt, dev)
                row["attr_margin"] = row["clip_cible"] - row["clip_stereotype"]
            rec["arms"][a] = row
        # les references elles-memes, pour situer les echelles
        rec["ref"] = {"Aclean_vs_Aref_crop60": seg_cos(clean[1], aref[1]),
                      "Aclean_vs_Bref_crop60": seg_cos(clean[1], bref[1]),
                      "luminance_Aclean": float(center_crop(frames(
                          f"{VID}/SS1_{cell}_memzero.mp4", w_clean)).mean())}
        if ti in ATTR:
            fc = frames(f"{VID}/SS1_{cell}_memzero.mp4", w_clean)
            for name, txt in ATTR[ti]:
                rec["ref"][f"clip_{name}"] = clip_text_score(fc, txt, dev)
            rec["ref"]["attr_margin"] = rec["ref"]["clip_cible"] - rec["ref"]["clip_stereotype"]
        res[cell] = rec
        print(f"[textpoor] {cell} done", flush=True)

    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "scores_textpoor.json"), "w") as f:
        json.dump(res, f, indent=2)

    # ---------------- tableaux -------------------------------------------------------------
    for cell, rec in res.items():
        print(f"\n================ {cell} : A = '{rec['prompt_A'][:60]}...'")
        print(f"                 retour = '{rec['prompt_retour']}'")
        print(f"  {'bras':22s} {'reid full':>10s} {'reid crop60':>12s} {'vs A':>8s} {'vs B':>8s} "
              f"{'margin':>8s} {'lum':>6s}" + ("  attr_margin" if "17" in cell else ""))
        for a, r in rec["arms"].items():
            line = (f"  {a:22s} {r['reid_full']:10.4f} {r['reid_crop60']:12.4f} "
                    f"{r['vs_Aref_crop60']:8.4f} {r['vs_Bref_crop60']:8.4f} "
                    f"{r['margin_crop60']:+8.4f} {r['luminance_crop60']:6.3f}")
            if "attr_margin" in r:
                line += f"  {r['attr_margin']:+.4f}"
            print(line)
        rf = rec["ref"]
        print(f"  {'[A* propre lui-meme]':22s} {'':10s} {'':12s} {rf['Aclean_vs_Aref_crop60']:8.4f} "
              f"{rf['Aclean_vs_Bref_crop60']:8.4f} "
              f"{rf['Aclean_vs_Aref_crop60'] - rf['Aclean_vs_Bref_crop60']:+8.4f} "
              f"{rf['luminance_Aclean']:6.3f}"
              + (f"  {rf['attr_margin']:+.4f}" if "attr_margin" in rf else ""))
        if "memdec20" in rec["arms"] and "wronganchor" in rec["arms"]:
            for v in ("reid_crop60", "reid_full", "margin_crop60"):
                d = rec["arms"]["memdec20"][v] - rec["arms"]["wronganchor"][v]
                print(f"  >>> CONTRASTE memdec20 - wronganchor sur {v:14s} = {d:+.4f}")
    print(f"\n[textpoor] wrote {os.path.join(OUT, 'scores_textpoor.json')}")


if __name__ == "__main__":
    main()
