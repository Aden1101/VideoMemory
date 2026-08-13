#!/usr/bin/env python
# ============================================================================================
# MEMOIRE 2.1 -- GATE G2b : PRE-ECRAN EN SITU DE L'ANCRE DU BRAS `memdec_t0`.
#
# Mesure identique a celle de 16.1 quater (`anchor_what.py`, 16 frames par segment ; ATTENTION,
# `snap_prescreen.py` du levier 4 en echantillonne 8, ses chiffres ne sont donc PAS bit a bit
# comparables) :
# correlation bande-basse 8x8 de l'ancre DECODEE avec
#     A* = le sujet a retrouver (segment 0 du rollout S3/SS1 PROPRE de la cellule)
#     B  = l'interlude (le segment que le decodeur a appris a reconstruire)
#     X  = l'ouvreur epingle au sink
# Reference a battre, mesuree sur 24 cellules : ancre COURANTE vs A* +0.042 | vs B +0.298 |
# vs X +0.344 (B > A sur 21/24, p = 3e-4). Levier 4 (snapshot) : +0.027 / +0.431 / +0.565, 0/2.
#
# REGLE SCELLEE (registre, G2b) : corr(ancre, A*) >= +0.15 ET > corr(ancre, B) sur >= 4/6
# cellules => on score les videos (G3). Sinon STOP, on ne score pas.
#
# Le dump d'ancre est GRATUIT : il sort du meme rollout que la video du bras (MEM_DECODE_DUMP).
#
#   python t0_prescreen.py --arm memdec_t0
# ============================================================================================
import argparse
import glob
import json
import os
import sys

REPO = "/shared/ssd_14T/home/adenelmi/Anchor-Forcing"
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [REPO, os.path.join(REPO, "reference/runs/reid_probe_l1"), HERE]
os.chdir(REPO)                                   # le VAE resout wan_models/... en relatif

import torch                                                             # noqa: E402
from torchvision.io import read_video                                    # noqa: E402
import scenarios as S                                                    # noqa: E402
import prescreen_offensif as PR                                          # noqa: E402
from prescreen_offensif import CACHE, OUT, VID, lowband, sign_p          # noqa: E402

# La reference PROPRE de A*, par scenario : le rollout ou A OUVRE (rien avant lui), donc jamais
# contamine par le sink (regle transverse du dossier, payee deux fois).
CLEAN_SCEN = {"S2": "S3", "SS2": "SS1"}


def anchor_px_arm(scen, ti, si, arm, device, switch_f=180):
    """Image RGB (3,H,W) de l'ancre decodee par `arm` sur la cellule, avec cache disque."""
    cpath = os.path.join(CACHE, f"{scen}_t{ti}_s{si}_{arm}_anchor.pt")
    if os.path.exists(cpath):
        return torch.load(cpath, map_location="cpu")
    src = os.path.join(S.DEC_ANCHOR_DIR, f"{scen}_t{ti}_s{si}_{arm}.f{switch_f}.pt")
    if not os.path.exists(src):
        alt = sorted(glob.glob(os.path.join(S.DEC_ANCHOR_DIR,
                                            f"{scen}_t{ti}_s{si}_{arm}*.pt")))
        assert alt, f"dump d'ancre manquant : {src}"
        src = alt[0]
    lat = torch.load(src, map_location="cpu")
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
    ap = argparse.ArgumentParser(description="GATE G2b : pre-ecran de l'ancre en situ")
    ap.add_argument("--arm", default="memdec_t0")
    ap.add_argument("--cells", default="S2:2,S2:4,S2:11,SS2:12,SS2:13,SS2:14")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--n_seg", type=int, default=16, help="frames echantillonnees par segment")
    ap.add_argument("--gate_a", type=float, default=0.15)
    ap.add_argument("--gate_cells", type=int, default=4)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    from run_reid_probe import sample_seg
    torch.set_grad_enabled(False)
    dev = torch.device("cuda")
    cells = [(c.split(":")[0], int(c.split(":")[1])) for c in args.cells.split(",")]
    rows = []
    for scen, ti in cells:
        si = args.seed
        clean = CLEAN_SCEN[scen]
        # A* PROPRE : segment 0 du rollout ou A ouvre (jamais contamine par le sink) ;
        # B (l'interlude, segment 2) et X (l'ouvreur epingle, segment 0) : dans le scenario.
        wA = S.score_segments(clean, S.SCENARIOS[clean]["frames"])[0]
        segs = S.score_segments(scen, S.SCENARIOS[scen]["frames"])
        wX, wB = segs[0], segs[2]
        pA = f"{VID}/{clean}_t{ti}_s{si}_memzero.mp4"
        pS = f"{VID}/{scen}_t{ti}_s{si}_memzero.mp4"
        if not (os.path.exists(pA) and os.path.exists(pS)):
            print(f"  [skip] t{ti} {scen}: mp4 de reference manquant", flush=True)
            continue
        anc = anchor_px_arm(scen, ti, si, args.arm, dev)[None]
        vA, _, _ = read_video(pA, pts_unit="sec", output_format="THWC")
        vS, _, _ = read_video(pS, pts_unit="sec", output_format="THWC")
        f = {"A": sample_seg(vA.permute(0, 3, 1, 2).float() / 255., *wA, args.n_seg),
             "B": sample_seg(vS.permute(0, 3, 1, 2).float() / 255., *wB, args.n_seg),
             "X": sample_seg(vS.permute(0, 3, 1, 2).float() / 255., *wX, args.n_seg)}
        la = lowband(anc, args.n)[0]
        r = {"scen": scen, "triple": ti, "seed": si}
        for nm, fr in f.items():
            r["lb_" + nm] = float((lowband(fr, args.n) @ la).mean())
        r["margin"] = r["lb_A"] - r["lb_B"]
        r["pass"] = bool(r["lb_A"] >= args.gate_a and r["lb_A"] > r["lb_B"])
        rows.append(r)
        print(f"  {scen} t{ti} s{si}: vs A* {r['lb_A']:+.4f} | vs B {r['lb_B']:+.4f} | "
              f"vs X {r['lb_X']:+.4f}  marge {r['margin']:+.4f}  "
              f"{'PASS' if r['pass'] else 'fail'}", flush=True)

    assert rows, "aucune cellule mesuree"
    n_pass = sum(1 for r in rows if r["pass"])
    n_ba = sum(1 for r in rows if r["lb_B"] > r["lb_A"])
    ok = n_pass >= args.gate_cells
    print("\n" + "=" * 92)
    print(f"GATE G2b ({args.arm}) : corr vs A* >= {args.gate_a:+.2f} ET > corr vs B sur "
          f">= {args.gate_cells}/{len(rows)} cellules")
    print(f"  moyennes : vs A* {sum(r['lb_A'] for r in rows) / len(rows):+.4f} | "
          f"vs B {sum(r['lb_B'] for r in rows) / len(rows):+.4f} | "
          f"vs X {sum(r['lb_X'] for r in rows) / len(rows):+.4f}   "
          f"[ancre courante publiee : +0.042 / +0.298 / +0.344]")
    print(f"  B encore plus proche que A* sur {n_ba}/{len(rows)} cellules "
          f"(sign-p {sign_p(n_ba, len(rows) - n_ba):.4f})")
    print(f"  VERDICT : {n_pass}/{len(rows)} => "
          f"{'PASS -- on score les videos (G3)' if ok else 'FAIL -- STOP, on ne score pas'}")
    print("=" * 92, flush=True)
    path = args.out or os.path.join(OUT, f"t0_prescreen_{args.arm}.json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as fh:
        json.dump({"arm": args.arm, "n": args.n, "rows": rows, "n_pass": n_pass,
                   "n_cells": len(rows), "gate": bool(ok),
                   "thresholds": {"a": args.gate_a, "cells": args.gate_cells}}, fh, indent=2)
    print(f"-> {path}", flush=True)
    sys.exit(0 if ok else 2)


if __name__ == "__main__":
    main()
