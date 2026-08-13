#!/usr/bin/env python
# ============================================================================================
# LA QUESTION QUE 16.1 bis IMPOSE : l'ancre decodee contient-elle L'IDENTITE, tout court ?
#
# T0 a montre que le canal TRANSMET (l'ancre recue est copiee a 0.47-0.52 de correlation, contre
# 0.10 pour une ancre recue par personne) et que la quantite transmise est ORTHOGONALE au gain de
# re-ID. Il reste donc exactement deux mondes possibles :
#   M1 "transport" : l'ancre PORTE l'identite, elle n'est simplement pas transmise -> il faut
#      travailler le canal (les 3 leviers de la nuit).
#   M2 "contenu"   : l'ancre ne porte PAS l'identite -> aucun canal ne la fera apparaitre, et le
#      chemin critique est l'ECRITURE + le DECODEUR.
# Le test est direct et ne coute aucun rollout : le juge du dossier (DINO) applique a l'IMAGE de
# l'ancre elle-meme, contre le A* PROPRE de sa cellule, avec pour comparateur l'ancre ETRANGERE
# (celle que splice `wronganchor`) -- exactement la paire du §11.
#
#   python anchor_identity.py
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
from prescreen_offensif import CACHE, VID, OUT, CELLS, sign_p           # noqa: E402


def main():
    from run_reid_probe import dino_embed, sample_seg, center_crop, seg_cos
    torch.set_grad_enabled(False)
    dev = torch.device("cuda")
    w_clean = S.score_segments("S3", 300)[0]
    rows = []
    for ti, si in CELLS:
        other = S.wrong_anchor_triple(ti)
        p_ref = f"{VID}/S3_t{ti}_s{si}_memzero.mp4"
        cp = {k: os.path.join(CACHE, f"S2_t{t}_s{si}_anchor.pt")
              for k, t in (("own", ti), ("for", other))}
        if not (os.path.exists(p_ref) and all(os.path.exists(v) for v in cp.values())):
            continue
        v, _, _ = read_video(p_ref, pts_unit="sec", output_format="THWC")
        ref = sample_seg(v.permute(0, 3, 1, 2).float() / 255., *w_clean, 8)
        anc = {k: torch.load(p, map_location="cpu")[None] for k, p in cp.items()}
        r = {"triple": ti, "seed": si, "source_triple": other}
        for view, fn in (("full", lambda x: x), ("crop60", center_crop)):
            e_ref = dino_embed(fn(ref), dev)
            for k in ("own", "for"):
                r[f"{view}_{k}"] = seg_cos(dino_embed(fn(anc[k]), dev), e_ref)
            r[f"{view}_adv"] = r[f"{view}_own"] - r[f"{view}_for"]
        rows.append(r)
        print(f"  t{ti} s{si}: DINO(ancre PROPRE, A*) crop60 {r['crop60_own']:.4f} | "
              f"(ancre de t{other}, A*) {r['crop60_for']:.4f} | avantage "
              f"{r['crop60_adv']:+.4f}", flush=True)

    print("\n" + "=" * 96)
    print("L'ANCRE DECODEE PORTE-T-ELLE L'IDENTITE ? (juge DINO du dossier, comparateur = "
          "l'ancre ETRANGERE)")
    print("=" * 96)
    summ = {"n": len(rows), "cells": rows, "verdict": {}}
    for view in ("crop60", "full"):
        v = [r[f"{view}_adv"] for r in rows]
        m = sum(v) / len(v)
        sd = (sum((x - m) ** 2 for x in v) / max(1, len(v) - 1)) ** 0.5
        npos = sum(1 for x in v if x > 0)
        o = sum(r[f"{view}_own"] for r in rows) / len(rows)
        f_ = sum(r[f"{view}_for"] for r in rows) / len(rows)
        summ["verdict"][view] = {"own": o, "foreign": f_, "adv_mean": m,
                                 "sem": sd / len(v) ** 0.5, "n_pos": npos, "n": len(v),
                                 "sign_p": sign_p(npos, len(v) - npos)}
        print(f"  [{view:6s}] ancre PROPRE {o:.4f} | ancre ETRANGERE {f_:.4f} | avantage "
              f"{m:+.4f} sem {sd / len(v) ** 0.5:.4f}  signes {npos}+/{len(v) - npos}-  "
              f"p {sign_p(npos, len(v) - npos):.4f}")
    with open(os.path.join(OUT, "anchor_identity.json"), "w") as f:
        json.dump(summ, f, indent=2)
    print(f"\n[anc] -> {OUT}/anchor_identity.json")


if __name__ == "__main__":
    main()
