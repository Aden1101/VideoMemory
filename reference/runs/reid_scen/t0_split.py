#!/usr/bin/env python
# ============================================================================================
# T0, TROISIEME PASSE (0 GPU, les ancres sont deja decodees et en cache) : la copie mesuree par
# T0 est-elle celle du DECOR ou celle du SUJET ?
#
# L'objection reviewer evidente : une correlation 8x8 sur l'image entiere est dominee par le
# fond, donc "l'ancre est copiee" pourrait vouloir dire seulement "l'ancre fixe le decor". On
# refait donc EXACTEMENT le meme design apparie 2x2, mais separement sur
#   * le CENTRE (crop 60 %, majoritairement le sujet)
#   * le BORD  (complement du crop 60 %, le decor)
# Si seul le bord copie, le canal transporte une scene ; si le centre copie aussi, il transporte
# la place et la silhouette du sujet -- et l'absence d'identite est alors un defaut de CONTENU
# de l'ancre, pas un defaut de transport.
#
#   python t0_split.py
# ============================================================================================
import json
import os
import sys

REPO = "/shared/ssd_14T/home/adenelmi/Anchor-Forcing"
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [REPO, os.path.join(REPO, "reference/runs/reid_probe_l1"), HERE]

import torch                                                            # noqa: E402
import torch.nn.functional as F                                         # noqa: E402
from torchvision.io import read_video                                   # noqa: E402
import scenarios as S                                                   # noqa: E402
from prescreen_offensif import CACHE, VID, OUT, CELLS, SWITCH_F, sign_p  # noqa: E402


def band(frames01, n=8, part="all"):
    """(T,3,H,W) -> (T, n*n) gris, centre-reduit. part = all | center | border."""
    x = frames01
    if part == "center":
        _, _, H, W = x.shape
        h, w = int(H * 0.6), int(W * 0.6)
        t, l = (H - h) // 2, (W - w) // 2
        x = x[:, :, t:t + h, l:l + w]
    g = (0.299 * x[:, 0] + 0.587 * x[:, 1] + 0.114 * x[:, 2])[:, None]
    if part == "border":                       # le centre mis a zero: "tout sauf le sujet"
        _, _, H, W = g.shape
        h, w = int(H * 0.6), int(W * 0.6)
        t, l = (H - h) // 2, (W - w) // 2
        g = g.clone()
        g[:, :, t:t + h, l:l + w] = 0.0
    z = F.adaptive_avg_pool2d(g, n).flatten(1)
    z = z - z.mean(1, keepdim=True)
    return z / z.norm(dim=1, keepdim=True).clamp_min(1e-6)


def main():
    from run_reid_probe import latent_to_pix
    torch.set_grad_enabled(False)
    rows = []
    wins = {"early": (SWITCH_F, SWITCH_F + 12), "scored": S.score_segments("S2", 240)[3]}
    for ti, si in CELLS:
        other = S.wrong_anchor_triple(ti)
        paths = {"wrong": f"{VID}/S2_t{ti}_s{si}_wronganchor.mp4",
                 "dec": f"{VID}/S2_t{ti}_s{si}_memdec20.mp4"}
        cp = {w: os.path.join(CACHE, f"S2_t{t}_s{si}_anchor.pt") for w, t in
              (("own", ti), ("for", other))}
        if not all(os.path.exists(p) for p in list(paths.values()) + list(cp.values())):
            continue
        anc = {k: torch.load(v, map_location="cpu")[None] for k, v in cp.items()}
        r = {"triple": ti, "seed": si}
        for name, p in paths.items():
            v, _, _ = read_video(p, pts_unit="sec", output_format="THWC")
            vid = v.permute(0, 3, 1, 2).float() / 255.
            for tag, (lo, hi) in wins.items():
                plo, phi = latent_to_pix(lo, hi)
                f01 = vid[min(plo, vid.shape[0] - 1):min(phi, vid.shape[0] - 1) + 1:4]
                for part in ("center", "border"):
                    A = band(f01, 8, part)
                    for who in ("own", "for"):
                        r[f"{tag}_{name}_{who}_{part}"] = float((A @ band(anc[who], 8, part)[0])
                                                                .mean())
        for tag in wins:
            for part in ("center", "border"):
                r[f"excess_{tag}_{part}"] = (
                    (r[f"{tag}_wrong_for_{part}"] - r[f"{tag}_wrong_own_{part}"])
                    + (r[f"{tag}_dec_own_{part}"] - r[f"{tag}_dec_for_{part}"]))
        rows.append(r)
        print(f"  t{ti} s{si}: exces scored centre {r['excess_scored_center']:+.3f} | "
              f"bord {r['excess_scored_border']:+.3f}", flush=True)

    print("\n" + "=" * 96)
    print("T0 SPLIT -- la copie est-elle celle du SUJET (centre) ou du DECOR (bord) ?")
    print("=" * 96)
    summ = {"n": len(rows), "cells": rows, "verdict": {}}
    for tag in ("early", "scored"):
        for part in ("center", "border"):
            v = [r[f"excess_{tag}_{part}"] for r in rows]
            m = sum(v) / len(v)
            sd = (sum((x - m) ** 2 for x in v) / max(1, len(v) - 1)) ** 0.5
            npos = sum(1 for x in v if x > 0)
            summ["verdict"][f"{tag}_{part}"] = {"mean": m, "sem": sd / len(v) ** 0.5,
                                                "n_pos": npos, "n": len(v),
                                                "sign_p": sign_p(npos, len(v) - npos)}
            print(f"  {tag:7s} {part:7s}: exces {m:+.4f}  sem {sd / len(v) ** 0.5:.4f}  "
                  f"signes {npos}+/{len(v) - npos}-  p {sign_p(npos, len(v) - npos):.4f}")
    for tag in ("early", "scored"):
        for name in ("wrong", "dec"):
            recue = "for" if name == "wrong" else "own"
            for part in ("center", "border"):
                a = sum(r[f"{tag}_{name}_{recue}_{part}"] for r in rows) / len(rows)
                b = sum(r[f"{tag}_{name}_{'own' if name == 'wrong' else 'for'}_{part}"]
                        for r in rows) / len(rows)
                print(f"  [niveaux] {tag:7s} {name:5s} {part:7s}: ancre RECUE {a:+.4f} | "
                      f"NON RECUE {b:+.4f}")
    with open(os.path.join(OUT, "t0_split.json"), "w") as f:
        json.dump(summ, f, indent=2)
    print(f"\n[t0] -> {OUT}/t0_split.json")


if __name__ == "__main__":
    main()
