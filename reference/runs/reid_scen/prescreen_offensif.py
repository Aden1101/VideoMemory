#!/usr/bin/env python
# ============================================================================================
# LOT OFFENSIF 2026-08-04 -- LES DEUX ANALYSES A ~0 ROLLOUT, predictions scellees au registre
# (`memory2_probes_registry.md`, section "LOT OFFENSIF DE LA NUIT").
#
#   T0   correlation BANDE-BASSE (8x8) des retours `wronganchor` / `memdec20` deja sur disque
#        avec l'ancre ETRANGERE (celle reellement splicee par wronganchor) vs l'ancre PROPRE.
#        Design apparie 2x2 par cellule :
#            exces = [c(ret_wrong, anc_etr) - c(ret_wrong, anc_prop)]
#                  + [c(ret_dec,   anc_prop) - c(ret_dec,   anc_etr)]
#        > 0 = chaque retour ressemble davantage a l'ancre QU'IL A RECUE => copie de layout.
#        Si l'exces est net, "pas lu" (§11.5, mesure DINO) devient "lu dans une bande que le
#        juge ignore". Deux fenetres : `early` (juste apres le splice) et `scored`.
#
#   L1   PRE-ECRAN de la legende-VLM : notre ancre decodee porte-t-elle la COULEUR du sujet ?
#        avantage = d(ancre ETRANGERE, A*) - d(ancre PROPRE, A*)   (juge couleur de
#        score_textpoor2.py : histogramme teinte pondere S*V + moyennes S/V, sur le crop 60 %).
#        REGLE SCELLEE : avantage moyen <= 0 OU <= 12/24 signes => LEVIER 1 MORT, aucun rollout.
#
# Le seul calcul GPU est le VAE sur 24 ancres latentes (~0.02 GPU-h) ; tout le reste est du CPU
# sur des mp4 deja rendus. Les images d'ancres decodees sont mises en cache sur le disque projet.
#
#   python prescreen_offensif.py                 # tout
#   python prescreen_offensif.py --only t0       # une seule analyse
# ============================================================================================
import argparse
import json
import os
import sys

REPO = "/shared/ssd_14T/home/adenelmi/Anchor-Forcing"
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [REPO, os.path.join(REPO, "reference/runs/reid_probe_l1"), HERE]
os.chdir(REPO)                                   # le VAE resout wan_models/... en relatif

import torch                                                             # noqa: E402
import torch.nn.functional as F                                          # noqa: E402
from torchvision.io import read_video                                    # noqa: E402

VID = "/shared/ssd_14T_project/adenelmi/reid_scen/videos"
ANC = "/shared/ssd_14T_project/adenelmi/reid_scen/dec_anchors"
CACHE = "/shared/ssd_14T_project/adenelmi/reid_scen/anchor_px"           # images d'ancres (cache)
OUT = os.path.join(HERE, "results", "offensif")
CELLS = [(t, s) for t in range(12) for s in (0, 1)]                      # les 24 cellules S2
SWITCH_F = 180                                                          # frame latente du retour


# ----------------------------------------------------------------- ancres -> pixels ----------
def anchor_px(ti, si, device):
    """Image RGB (3, H, W) in [0,1] de l'ancre DECODEE de la cellule (ti, si). Cache disque."""
    cpath = os.path.join(CACHE, f"S2_t{ti}_s{si}_anchor.pt")
    if os.path.exists(cpath):
        return torch.load(cpath, map_location="cpu")
    lat = torch.load(os.path.join(ANC, f"S2_t{ti}_s{si}_decanchor.f{SWITCH_F}.pt"),
                     map_location="cpu")
    assert lat.shape[1] == 1, f"ancre attendue a 1 frame latente, {tuple(lat.shape)}"
    from utils.wan_wrapper import WanVAEWrapper
    global _VAE
    if "_VAE" not in globals() or _VAE is None:
        _VAE = WanVAEWrapper().to(device=device, dtype=torch.bfloat16).eval()
    px = _VAE.decode_to_pixel(lat.to(device=device, dtype=torch.bfloat16), use_cache=False)
    _VAE.model.clear_cache()
    im = (px[0].float() * 0.5 + 0.5).clamp(0, 1).mean(0).cpu()           # (3, H, W)
    os.makedirs(CACHE, exist_ok=True)
    torch.save(im, cpath)
    return im


# ----------------------------------------------------------------- juges ----------------------
def hsv_hist(frames01, bins=12):
    """Juge couleur de score_textpoor2.py, VERBATIM (histogramme teinte pondere S*V sur le crop
    central 60 %, + les moyennes S et V qui portent le noir/blanc)."""
    from run_reid_probe import center_crop
    x = center_crop(frames01)
    r, g, b = x[:, 0], x[:, 1], x[:, 2]
    mx, _ = torch.max(x, dim=1)
    mn, _ = torch.min(x, dim=1)
    v = mx
    s = torch.where(mx > 1e-6, (mx - mn) / mx.clamp_min(1e-6), torch.zeros_like(mx))
    d = (mx - mn).clamp_min(1e-6)
    h = torch.zeros_like(mx)
    h = torch.where(mx == r, ((g - b) / d) % 6, h)
    h = torch.where(mx == g, (b - r) / d + 2, h)
    h = torch.where(mx == b, (r - g) / d + 4, h)
    h = (h / 6.0).clamp(0, 1 - 1e-6)
    w = (s * v).flatten()
    idx = (h.flatten() * bins).long().clamp(0, bins - 1)
    hist = torch.zeros(bins).scatter_add_(0, idx, w)
    hist = hist / hist.sum().clamp_min(1e-6)
    return torch.cat([hist, s.mean().reshape(1), v.mean().reshape(1)])


def hist_dist(a, b):
    return float((a - b).abs().sum())


def lowband(frames01, n=8):
    """(T,3,H,W) -> (T, n*n) niveaux de gris moyennes en n x n, centres-reduits par frame.
    C'est LA bande basse : layout des masses + luminance, ce que DINO ne regarde pas."""
    gray = (0.299 * frames01[:, 0] + 0.587 * frames01[:, 1] + 0.114 * frames01[:, 2])[:, None]
    z = F.adaptive_avg_pool2d(gray, n).flatten(1)
    z = z - z.mean(1, keepdim=True)
    return z / z.norm(dim=1, keepdim=True).clamp_min(1e-6)


def lowband_rgb(frames01, n=8):
    z = F.adaptive_avg_pool2d(frames01, n).flatten(1)
    z = z - z.mean(1, keepdim=True)
    return z / z.norm(dim=1, keepdim=True).clamp_min(1e-6)


def corr(A, b):
    """Correlation moyenne de chaque frame de A (T,D) avec le vecteur b (D,)."""
    return float((A @ b).mean())


def sign_p(npos, nneg):
    from run_reid_probe import sign_test_p
    return sign_test_p(npos, nneg)


# ----------------------------------------------------------------- T0 -------------------------
def run_t0(device):
    """Design apparie 2x2 + DEUX controles ajoutes apres le premier passage (le resultat etait
    trop gros pour etre publie sans eux) :
      * ancre NULLE : une TROISIEME ancre (celle de t+2), que NI l'un NI l'autre bras n'a recue.
        Si la correlation avec elle est du niveau de l'ancre non-recue, le +0.52 est bien
        specifique a ce qui a ete splice et non "deux images floues se ressemblent".
      * ECHELLE de bande : n x n pour n = 4, 8, 16, 32, 64. Localise la bande ou la copie meurt,
        c'est-a-dire ce que le canal transmet vraiment.
    """
    from run_reid_probe import latent_to_pix
    import scenarios as S

    BANDS = (4, 8, 16, 32, 64)
    rows = []
    wins = {"early": (SWITCH_F, SWITCH_F + 12), "scored": S.score_segments("S2", 240)[3]}
    for ti, si in CELLS:
        other = S.wrong_anchor_triple(ti)
        null = S.wrong_anchor_triple(other)                          # t+2 : recue par personne
        p_w = f"{VID}/S2_t{ti}_s{si}_wronganchor.mp4"
        p_d = f"{VID}/S2_t{ti}_s{si}_memdec20.mp4"
        if not (os.path.exists(p_w) and os.path.exists(p_d)):
            print(f"  t{ti} s{si}: mp4 manquant, cellule sautee")
            continue
        anc = {"own": anchor_px(ti, si, device)[None],               # (1,3,H,W)
               "for": anchor_px(other, si, device)[None],
               "null": anchor_px(null, si, device)[None]}
        r = {"triple": ti, "seed": si, "source_triple": other, "null_triple": null}
        for name, path in (("wrong", p_w), ("dec", p_d)):
            v, _, _ = read_video(path, pts_unit="sec", output_format="THWC")
            vid = v.permute(0, 3, 1, 2).float() / 255.
            for tag, (lo, hi) in wins.items():
                plo, phi = latent_to_pix(lo, hi)
                f01 = vid[min(plo, vid.shape[0] - 1):min(phi, vid.shape[0] - 1) + 1:4]
                for band, fn in (("gray", lowband), ("rgb", lowband_rgb)):
                    A = fn(f01)
                    for who in ("own", "for", "null"):
                        r[f"{tag}_{name}_{who}_{band}"] = corr(A, fn(anc[who])[0])
                for n in BANDS:                                       # echelle de bande (gris)
                    A = lowband(f01, n)
                    for who in ("own", "for"):
                        r[f"{tag}_{name}_{who}_n{n}"] = corr(A, lowband(anc[who], n)[0])
        for tag in wins:
            for band in ("gray", "rgb"):
                r[f"excess_{tag}_{band}"] = (
                    (r[f"{tag}_wrong_for_{band}"] - r[f"{tag}_wrong_own_{band}"])
                    + (r[f"{tag}_dec_own_{band}"] - r[f"{tag}_dec_for_{band}"]))
            for n in BANDS:
                r[f"excess_{tag}_n{n}"] = (
                    (r[f"{tag}_wrong_for_n{n}"] - r[f"{tag}_wrong_own_n{n}"])
                    + (r[f"{tag}_dec_own_n{n}"] - r[f"{tag}_dec_for_n{n}"]))
        rows.append(r)
        print(f"  t{ti} s{si} (ancre etrangere = t{other}, nulle = t{null}) : exces early gray "
              f"{r['excess_early_gray']:+.4f} | scored gray {r['excess_scored_gray']:+.4f}",
              flush=True)

    print("\n" + "=" * 96)
    print("T0 -- CORRELATION BANDE-BASSE (8x8). exces > 0 = chaque retour copie le layout de "
          "L'ANCRE QU'IL A RECUE")
    print("=" * 96)
    summ = {"n": len(rows), "cells": rows, "verdict": {}}
    for tag in ("early", "scored"):
        for band in ("gray", "rgb"):
            v = [r[f"excess_{tag}_{band}"] for r in rows]
            m = sum(v) / len(v)
            sd = (sum((x - m) ** 2 for x in v) / max(1, len(v) - 1)) ** 0.5
            npos = sum(1 for x in v if x > 0)
            summ["verdict"][f"{tag}_{band}"] = {
                "mean": m, "sd": sd, "sem": sd / len(v) ** 0.5, "n_pos": npos,
                "n": len(v), "sign_p": sign_p(npos, len(v) - npos)}
            print(f"  {tag:7s} {band:5s}: exces moyen {m:+.4f}  sd {sd:.4f}  "
                  f"sem {sd / len(v) ** 0.5:.4f}  signes {npos}+/{len(v) - npos}-  "
                  f"p {sign_p(npos, len(v) - npos):.4f}")
    # niveaux absolus + CONTROLE NUL (ancre recue par personne)
    print("\n  [niveaux] correlation 8x8 du RETOUR avec chaque ancre (RECUE = celle que ce bras "
          "a effectivement splicee)")
    for tag in ("early", "scored"):
        for name in ("wrong", "dec"):
            recue = "for" if name == "wrong" else "own"
            pas = "own" if name == "wrong" else "for"
            a = sum(r[f"{tag}_{name}_{recue}_gray"] for r in rows) / len(rows)
            b = sum(r[f"{tag}_{name}_{pas}_gray"] for r in rows) / len(rows)
            c = sum(r[f"{tag}_{name}_null_gray"] for r in rows) / len(rows)
            summ["verdict"][f"levels_{tag}_{name}"] = {"recue": a, "non_recue": b, "nulle": c}
            print(f"    {tag:7s} retour {name:5s}: ancre RECUE {a:+.4f} | ancre NON RECUE "
                  f"{b:+.4f} | ancre NULLE (t+2, recue par personne) {c:+.4f}")
    print("\n  [echelle de bande] exces apparie a la resolution n x n (gris) -- ou meurt la copie")
    for tag in ("early", "scored"):
        line = f"    {tag:7s}: "
        for n in (4, 8, 16, 32, 64):
            v = [r[f"excess_{tag}_n{n}"] for r in rows]
            m = sum(v) / len(v)
            npos = sum(1 for x in v if x > 0)
            summ["verdict"][f"band_{tag}_n{n}"] = {"mean": m, "n_pos": npos, "n": len(v)}
            line += f"n={n:2d} {m:+.3f} ({npos}+)  "
        print(line)
    e = summ["verdict"]["early_gray"]
    ok = e["mean"] >= 0.10 and e["n_pos"] >= 18
    summ["regle_scellee"] = ("REQUALIFIE: lu dans une bande invisible a DINO" if ok
                             else "le negatif du §11 tient: pas de copie de bande basse non plus")
    print(f"\n  REGLE SCELLEE (exces early >= +0.10 ET >= 18/24 signes) -> {summ['regle_scellee']}")
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "t0_lowband.json"), "w") as f:
        json.dump(summ, f, indent=2)
    return summ


# ----------------------------------------------------------------- L1 pre-ecran ---------------
def run_l1(device):
    from run_reid_probe import sample_seg
    import scenarios as S

    w_clean = S.score_segments("S3", 300)[0]                  # A ouvre, rien avant lui
    rows = []
    for ti, si in CELLS:
        other = S.wrong_anchor_triple(ti)
        p_ref = f"{VID}/S3_t{ti}_s{si}_memzero.mp4"
        if not os.path.exists(p_ref):
            print(f"  t{ti} s{si}: A* manquant, cellule sautee")
            continue
        v, _, _ = read_video(p_ref, pts_unit="sec", output_format="THWC")
        ref = sample_seg(v.permute(0, 3, 1, 2).float() / 255., *w_clean, 8)
        h_ref = hsv_hist(ref)
        h_own = hsv_hist(anchor_px(ti, si, device)[None])
        h_for = hsv_hist(anchor_px(other, si, device)[None])
        d_own, d_for = hist_dist(h_own, h_ref), hist_dist(h_for, h_ref)
        rows.append({"triple": ti, "seed": si, "source_triple": other,
                     "d_own": d_own, "d_foreign": d_for, "advantage": d_for - d_own})
        print(f"  t{ti} s{si}: d(ancre propre, A*) {d_own:.4f} | d(ancre de t{other}, A*) "
              f"{d_for:.4f} | avantage {d_for - d_own:+.4f}", flush=True)

    v = [r["advantage"] for r in rows]
    m = sum(v) / len(v)
    sd = (sum((x - m) ** 2 for x in v) / max(1, len(v) - 1)) ** 0.5
    npos = sum(1 for x in v if x > 0)
    dead = (m <= 0) or (npos <= 12)
    print("\n" + "=" * 96)
    print("L1 PRE-ECRAN -- notre ancre decodee porte-t-elle la COULEUR du sujet ?")
    print("=" * 96)
    print(f"  avantage moyen {m:+.4f}  sd {sd:.4f}  sem {sd / len(v) ** 0.5:.4f}  "
          f"signes {npos}+/{len(v) - npos}-  p {sign_p(npos, len(v) - npos):.4f}")
    print(f"  d moyen ancre PROPRE {sum(r['d_own'] for r in rows) / len(rows):.4f} | "
          f"ancre ETRANGERE {sum(r['d_foreign'] for r in rows) / len(rows):.4f}")
    print(f"\n  REGLE SCELLEE (avantage <= 0 OU <= 12/24 signes => LEVIER 1 MORT) -> "
          f"{'LEVIER 1 MORT' if dead else 'LEVIER 1 VIVANT, on tire les 4 bras'}")
    summ = {"n": len(rows), "cells": rows, "mean_advantage": m, "sd": sd, "n_pos": npos,
            "sign_p": sign_p(npos, len(v) - npos), "dead": bool(dead)}
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "l1_prescreen.json"), "w") as f:
        json.dump(summ, f, indent=2)
    return summ


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="", choices=["", "t0", "l1"])
    args = ap.parse_args()
    torch.set_grad_enabled(False)
    device = torch.device("cuda")
    if args.only in ("", "l1"):
        run_l1(device)
    if args.only in ("", "t0"):
        run_t0(device)


if __name__ == "__main__":
    main()
