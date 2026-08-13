#!/usr/bin/env python
# ============================================================================================
# LEVIER 1 -- CONSTRUCTION DES LEGENDES (hors ligne, ~0.01 GPU-h : un VAE sur 8 ancres)
#
# ANTI-AFFABULATION, la regle du lot : PAS de VLM libre en v1. Une legende ne peut contenir que
#   * la CATEGORIE, qui vient du PROMPT (elle est connue, elle n'est pas devinee) ;
#   * TROIS COULEURS MESUREES sur l'image, par une procedure identique pour les 4 bras :
#       c1 = couleur moyenne du coeur du sujet   (boite centrale 30 %)
#       c2 = couleur moyenne de la peripherie    (anneau 30 % -> 60 %)
#       c3 = couleur moyenne du fond             (hors du crop 60 %)
#     nommees par PLUS PROCHE VOISIN dans une palette figee de 12 termes.
# Aucun degre de liberte n'est laisse au juge : changer d'image change la legende, rien d'autre.
#
# Les 4 bras ne different QUE par l'IMAGE sur laquelle les 3 couleurs sont mesurees :
#   caption_ours   notre ancre decodee de CETTE cellule
#   caption_wrong  l'ancre decodee de l'AUTRE cellule (celle que splice `wronganchor`), avec la
#                  categorie de CETTE cellule -> seules les COULEURS sont fausses
#   framebank_vlm  une VRAIE frame (derniere frame de la 1re apparition de A, = ce que la banque
#                  Echo stocke) -> le plafond du canal
#   padded         AUCUNE mesure : trois adjectifs neutres, meme gabarit, meme nombre de mots
#   caption_none   suffixe vide -> temoin bit a bit du memzero deja sur disque
#
#   python build_captions.py
# ============================================================================================
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
import prescreen_offensif as _PR                                       # noqa: E402,F401

VID = "/shared/ssd_14T_project/adenelmi/reid_scen/videos"
OUT = os.path.join(HERE, "results", "offensif")
CELLS = [("S2c", t, 0) for t in (0, 2, 3, 4, 11)] + [("SS2c", t, 0) for t in (12, 13, 14)]

# La CATEGORIE vient du prompt de A, mot pour mot (aucune invention) : le nom de tete du groupe
# nominal sujet. Ecrite a la main pour etre exacte, verifiee par assert contre le prompt.
CATEGORY = {0: "dog", 1: "woman", 2: "cat", 3: "man", 4: "horse", 5: "parrot", 6: "girl",
            7: "bicycle", 8: "bus", 9: "fisherman", 10: "collie", 11: "teapot",
            12: "man", 13: "retriever", 14: "scooter", 15: "car", 16: "parrot"}

# palette figee, 12 termes (RGB de reference dans [0,1])
PALETTE = {"black": (0.05, 0.05, 0.05), "white": (0.95, 0.95, 0.95), "grey": (0.5, 0.5, 0.5),
           "brown": (0.45, 0.28, 0.13), "tan": (0.76, 0.65, 0.48), "red": (0.75, 0.12, 0.12),
           "orange": (0.90, 0.50, 0.10), "yellow": (0.92, 0.86, 0.20),
           "green": (0.20, 0.55, 0.22), "blue": (0.16, 0.32, 0.72),
           "purple": (0.48, 0.20, 0.60), "pink": (0.92, 0.60, 0.68)}


def name_color(rgb):
    """Plus proche voisin dans la palette, en distance euclidienne RGB. Les couleurs peu saturees
    sont d'abord ramenees a l'axe noir/gris/blanc par leur VALEUR (sinon un gris bleute serait
    nomme 'blue' et la legende mentirait sur une teinte que l'image ne porte pas)."""
    r, g, b = [float(x) for x in rgb]
    mx, mn = max(r, g, b), min(r, g, b)
    sat = 0.0 if mx <= 1e-6 else (mx - mn) / mx
    if sat < 0.18:
        return "black" if mx < 0.25 else ("white" if mx > 0.75 else "grey")
    return min(PALETTE, key=lambda k: sum((PALETTE[k][i] - (r, g, b)[i]) ** 2 for i in range(3)))


def three_colors(img):
    """(3, H, W) in [0,1] -> (coeur, peripherie, fond) nommes."""
    _, H, W = img.shape
    def box(frac):
        h, w = int(H * frac), int(W * frac)
        t, l = (H - h) // 2, (W - w) // 2
        return t, t + h, l, l + w
    t3, b3, l3, r3 = box(0.30)
    t6, b6, l6, r6 = box(0.60)
    core = img[:, t3:b3, l3:r3].mean(dim=(1, 2))
    m = torch.zeros(H, W, dtype=torch.bool)
    m[t6:b6, l6:r6] = True
    m[t3:b3, l3:r3] = False
    peri = (img * m).sum(dim=(1, 2)) / m.sum().clamp_min(1)
    mb = torch.ones(H, W, dtype=torch.bool)
    mb[t6:b6, l6:r6] = False
    bg = (img * mb).sum(dim=(1, 2)) / mb.sum().clamp_min(1)
    return name_color(core), name_color(peri), name_color(bg)


def suffix(cat, c1, c2, c3):
    """15 mots, gabarit fixe. Le seul degre de liberte est le contenu des 3 couleurs."""
    return (f", and in this shot the {cat} is {c1}, with {c2} markings and a {c3} background")


PADDED = ", and in this shot the {cat} is visible, with clear markings and a plain background"


def real_frame(scen_base, ti, si):
    """La derniere frame PIXEL de la 1re apparition de A = ce que la banque Echo stocke."""
    from run_reid_probe import latent_to_pix
    fa = S.first_appearance(scen_base)[0]
    end = S.seg_bounds(scen_base)[fa][1]                      # frame latente exclusive
    p = f"{VID}/{scen_base}_t{ti}_s{si}_memzero.mp4"
    assert os.path.exists(p), f"il faut le memzero de la cellule pour la vraie frame: {p}"
    v, _, _ = read_video(p, pts_unit="sec", output_format="THWC")
    _, hi = latent_to_pix(end - 1, end)
    return v[min(hi, v.shape[0] - 1)].permute(2, 0, 1).float() / 255.


def main():
    torch.set_grad_enabled(False)
    device = torch.device("cuda")
    os.makedirs(OUT, exist_ok=True)
    caps, detail = {}, {}
    for scen, ti, si in CELLS:
        base = S.CAPTION_SCENARIOS[scen]
        cat = CATEGORY[ti]
        assert cat in S.TRIPLES[ti][0], \
            f"categorie '{cat}' absente du prompt de A de t{ti}: '{S.TRIPLES[ti][0]}'"
        other = S.wrong_anchor_triple(ti)
        # l'ancre est indexee par le scenario de BASE (c'est la campagne qui l'a dumpee)
        imgs = {"caption_ours": anchor_px_scen(base, ti, si, device),
                "caption_wrong": anchor_px_scen(base, other, si, device),
                "framebank_vlm": real_frame(base, ti, si)}
        for arm, im in imgs.items():
            c1, c2, c3 = three_colors(im)
            caps[f"{scen}_t{ti}_s{si}_{arm}"] = suffix(cat, c1, c2, c3)
            detail[f"{scen}_t{ti}_s{si}_{arm}"] = {"cat": cat, "colors": [c1, c2, c3]}
        caps[f"{scen}_t{ti}_s{si}_padded"] = PADDED.format(cat=cat)
        caps[f"{scen}_t{ti}_s{si}_caption_none"] = ""
        n = len(caps[f"{scen}_t{ti}_s{si}_caption_ours"].split())
        assert n == len(caps[f"{scen}_t{ti}_s{si}_padded"].split()), \
            "padded doit avoir EXACTEMENT le meme nombre de mots que caption_ours"
        assert 10 <= n <= 20, f"suffixe de {n} mots, la regle du lot demande 10-20"
        print(f"  {scen} t{ti} s{si} ({cat}, ancre etrangere = t{other}, {n} mots)")
        for arm in ("caption_ours", "caption_wrong", "framebank_vlm", "padded"):
            print(f"      {arm:14s} '{caps[f'{scen}_t{ti}_s{si}_{arm}'].strip()}'")
    with open(S.CAPTION_JSON, "w") as f:
        json.dump(caps, f, indent=2)
    with open(os.path.join(OUT, "captions_detail.json"), "w") as f:
        json.dump(detail, f, indent=2)
    # accord de couleur ours vs wrong sur le COEUR : le pre-ecran, en version discrete
    agree = [(detail[f"{s}_t{t}_s{q}_caption_ours"]["colors"][0]
              == detail[f"{s}_t{t}_s{q}_framebank_vlm"]["colors"][0],
              detail[f"{s}_t{t}_s{q}_caption_wrong"]["colors"][0]
              == detail[f"{s}_t{t}_s{q}_framebank_vlm"]["colors"][0])
             for s, t, q in CELLS]
    print(f"\n  accord de la couleur de COEUR avec la VRAIE frame : notre ancre "
          f"{sum(a for a, _ in agree)}/{len(agree)} | ancre etrangere "
          f"{sum(b for _, b in agree)}/{len(agree)}")
    print(f"[captions] {len(caps)} suffixes -> {S.CAPTION_JSON}")


def anchor_px_scen(scen_base, ti, si, device):
    """anchor_px de prescreen_offensif, mais pour un scenario quelconque (S2 ou SS2)."""
    import prescreen_offensif as PR
    cpath = os.path.join(PR.CACHE, f"{scen_base}_t{ti}_s{si}_anchor.pt")
    if os.path.exists(cpath):
        return torch.load(cpath, map_location="cpu")
    lat = torch.load(os.path.join(PR.ANC, f"{scen_base}_t{ti}_s{si}_decanchor.f180.pt"),
                     map_location="cpu")
    from utils.wan_wrapper import WanVAEWrapper
    if getattr(PR, "_VAE", None) is None:
        PR._VAE = WanVAEWrapper().to(device=device, dtype=torch.bfloat16).eval()
    px = PR._VAE.decode_to_pixel(lat.to(device=device, dtype=torch.bfloat16), use_cache=False)
    PR._VAE.model.clear_cache()
    im = (px[0].float() * 0.5 + 0.5).clamp(0, 1).mean(0).cpu()
    os.makedirs(PR.CACHE, exist_ok=True)
    torch.save(im, cpath)
    return im


if __name__ == "__main__":
    main()
