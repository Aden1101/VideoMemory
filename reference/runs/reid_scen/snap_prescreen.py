#!/usr/bin/env python
# ============================================================================================
# LEVIER 4 -- PRE-ECRAN SCELLE : l'ancre decodee du SNAPSHOT de fin de segment A est-elle enfin
# une image de A ?  Metrique = celle de T0 (correlation bande-basse 8x8), memes references, memes
# cellules, pour que le chiffre soit DIRECTEMENT comparable a celui de l'ancre courante :
#     ancre COURANTE  vs A* +0.042 | vs B +0.298 | vs X +0.344   (24 cellules, 16.1 quater)
# REGLE SCELLEE (registre) : corr(ancre_snap, A*) >= +0.15 ET > corr(ancre_snap, B) sur 2/2
# cellules du pilote => le colis est repare, on tire les 4 cellules restantes.
#
#   python snap_prescreen.py                       # toutes les cellules dont le dump existe
# ============================================================================================
import glob
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
from prescreen_offensif import VID, OUT, CACHE, sign_p, lowband          # noqa: E402
import prescreen_offensif as PR                                          # noqa: E402

SNAP_DIR = S.DEC_ANCHOR_DIR


def snap_px(ti, si, device):
    """Image RGB de l'ancre decodee du SNAPSHOT (cache disque, comme les ancres courantes)."""
    cpath = os.path.join(CACHE, f"S2_t{ti}_s{si}_snapanchor.pt")
    if os.path.exists(cpath):
        return torch.load(cpath, map_location="cpu")
    src = os.path.join(SNAP_DIR, f"S2_t{ti}_s{si}_snapanchor.f180.pt")
    assert os.path.exists(src), f"dump manquant {src} -- lancer la vague D0 d'abord"
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
    from run_reid_probe import sample_seg
    torch.set_grad_enabled(False)
    dev = torch.device("cuda")
    wA = S.score_segments("S3", 300)[0]
    segs = S.score_segments("S2", 240)
    wB, wX = segs[2], segs[0]
    cells = sorted({(int(os.path.basename(f).split("_t")[1].split("_")[0]),
                     int(os.path.basename(f).split("_s")[1].split("_")[0]))
                    for f in glob.glob(os.path.join(SNAP_DIR, "S2_t*_s*_snapanchor.f180.pt"))})
    assert cells, "aucun dump de snapshot: lancer `launch_offensif.py --waves D0`"
    rows = []
    for ti, si in cells:
        pA, pS = f"{VID}/S3_t{ti}_s{si}_memzero.mp4", f"{VID}/S2_t{ti}_s{si}_memzero.mp4"
        cur = os.path.join(CACHE, f"S2_t{ti}_s{si}_anchor.pt")
        if not all(map(os.path.exists, (pA, pS, cur))):
            continue
        vA, _, _ = read_video(pA, pts_unit="sec", output_format="THWC")
        vS, _, _ = read_video(pS, pts_unit="sec", output_format="THWC")
        ref = {"A": sample_seg(vA.permute(0, 3, 1, 2).float() / 255., *wA, 8),
               "B": sample_seg(vS.permute(0, 3, 1, 2).float() / 255., *wB, 8),
               "X": sample_seg(vS.permute(0, 3, 1, 2).float() / 255., *wX, 8)}
        anc = {"snap": lowband(snap_px(ti, si, dev)[None])[0],
               "cur": lowband(torch.load(cur, map_location="cpu")[None])[0]}
        r = {"triple": ti, "seed": si}
        for who, a in anc.items():
            for nm, f in ref.items():
                r[f"{who}_{nm}"] = float((lowband(f) @ a).mean())
        r["ok"] = bool(r["snap_A"] >= 0.15 and r["snap_A"] > r["snap_B"])
        rows.append(r)
        print(f"  t{ti} s{si}:  SNAP vs A* {r['snap_A']:+.3f} | vs B {r['snap_B']:+.3f} | "
              f"vs X {r['snap_X']:+.3f}      COURANTE vs A* {r['cur_A']:+.3f} | vs B "
              f"{r['cur_B']:+.3f} | vs X {r['cur_X']:+.3f}   -> "
              f"{'PASSE' if r['ok'] else 'echoue'}", flush=True)

    print("\n" + "=" * 100)
    print("LEVIER 4 -- PRE-ECRAN : le colis est-il repare ? (regle scellee : vs A* >= +0.15 ET "
          "> vs B)")
    print("=" * 100)
    n_ok = sum(r["ok"] for r in rows)
    for who, lab in (("snap", "ancre SNAPSHOT (fin du segment A)"),
                     ("cur", "ancre COURANTE (etat au retour)")):
        print(f"  {lab:36s}: vs A* {sum(r[who + '_A'] for r in rows) / len(rows):+.4f} | "
              f"vs B {sum(r[who + '_B'] for r in rows) / len(rows):+.4f} | "
              f"vs X {sum(r[who + '_X'] for r in rows) / len(rows):+.4f}")
    gain = sum(r["snap_A"] - r["cur_A"] for r in rows) / len(rows)
    npos = sum(1 for r in rows if r["snap_A"] > r["cur_A"])
    print(f"\n  gain apparie sur la ressemblance a A* : {gain:+.4f}  ({npos}/{len(rows)} cellules, "
          f"p {sign_p(npos, len(rows) - npos):.4f})")
    print(f"  REGLE SCELLEE : {n_ok}/{len(rows)} cellules passent -> "
          f"{'COLIS REPARE, on tire les cellules restantes' if n_ok == len(rows) else ('ZONE GRISE' if n_ok else 'le snapshot ne repare PAS le contenu : le probleme est dans l ECRITURE')}")
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "snap_prescreen.json"), "w") as f:
        json.dump({"cells": rows, "n_ok": n_ok, "n": len(rows), "gain_vs_A": gain}, f, indent=2)
    print(f"\n[snap] -> {OUT}/snap_prescreen.json")


if __name__ == "__main__":
    main()
