#!/usr/bin/env python
# ============================================================================================
# SCORING DU PILOTE "ANCRE COMPOSITE" (2026-08-02) -- les DEUX axes du verdict d'Aden.
#
# LE CONSTAT. Sur le tier meme-decor, notre splice pleine-frame re-rend TOUTE la scene au
# retour (l'auvent de SS2_t14 passe bleu -> rose) alors que la banque ne ramene que le sujet
# dans une scene intacte. Un bras ne peut donc plus etre juge sur le seul re-ID : il faut le
# COUPLE (sujet retrouve, fond continue).
#
# AXE 1 -- SUJET (verbatim rescore_ss2_clean.py) : DINO crop60 du retour de A contre A* PROPRE
#   = le segment 0 du rollout ou A OUVRE la video (SS1 pour SS2, S3 pour S2), meme triple, meme
#   seed. Reference propre obligatoire : la "1re apparition de A" de la campagne montre en fait
#   l'ouvreur X, epingle dans le sink. Chaque pipeline est score contre SON A* (stockoff a le
#   sien : sinon on confondrait pipeline et bras).
#
# AXE 2 -- FOND (esprit rejoin_border de reid_probe_l1/quadrant_score.py) : DINO sur le
#   COMPLEMENT du crop60 (zero_center = le centre 60 % mis a zero, donc "tout sauf le sujet"),
#   du retour contre LA SCENE COURANTE JUSTE AVANT LE RETOUR (la queue du segment B du MEME
#   rollout, que tous les bras MAG partagent bit a bit puisque aucun hook ne se declenche avant
#   le revisit). Plus haut = le monde a CONTINUE ; plus bas = le splice a rembobine le decor.
#     bg_cont_head : fenetre chaude (+3 a +20 frames apres le retour), la ou le splice pese
#     bg_cont_win  : toute la fenetre de scoring du retour, la ou l'effet persiste
#   Plafond de reference = memzero (il ne splice rien, le monde roule).
#
# BONUS gratuit -- cut2_ratio : brutalite visuelle du raccord au retour (max du lag-1 pixel dans
#   +/-8 frames autour du joint / lag-1 median intra-segment). Le composite doit couper MOINS.
#
#   python score_comp.py --scen SS2 --triples 12,13,14 --arms oracle1,oracle1comp,memdec20,...
# ============================================================================================
import argparse
import json
import os
import sys

REPO = "/shared/ssd_14T/home/adenelmi/Anchor-Forcing"
HERE = os.path.dirname(os.path.abspath(__file__))
PROBE_DIR = os.path.join(REPO, "reference/runs/reid_probe_l1")
sys.path[:0] = [REPO, PROBE_DIR, HERE]

VID = "/shared/ssd_14T_project/adenelmi/reid_scen/videos"
OUT = os.path.join(HERE, "results", "comp")

# scenario du retour -> scenario dont le segment 0 est le A* PROPRE de la meme cellule
CLEAN_REF = {"SS2": ("SS1", 240), "S2": ("S3", 300)}
LABELS = {"memdec21uf": "NOTRE 21uf", "memdec21u": "union nue",
          "stockoff": "LongLive", "memzero": "sans memoire", "framebank": "banque Echo",
          "memdec20": "notre mem", "memdec21c": "notre mem COMP",
          "memdec21c75": "notre mem C75",
          "oracle1": "oracle1", "oracle1comp": "oracle1 COMP",
          "oracle1comp75": "oracle1 C75"}


def pix(f):
    return 0 if f == 0 else 4 * f - 3


# ---------------------------------------------------------------------------------------------
# TRIPWIRES ajoutes le 2026-08-02 (campagne des gates). Un bras qui gagne du sujet en COLLANT une
# image fixe au centre doit etre rejete, pas discute -- d'ou:
#   dd_ell_*   MOUVEMENT DANS L'ELLIPSE : lag-1 pixel moyen a l'INTERIEUR du crop central (la
#              zone que le composite ecrit), sur la fenetre chaude et sur toute la fenetre de
#              retour. Reference = memzero (qui ne colle rien). Un effondrement = autocollant.
#   clip_text  ADHERENCE TEXTE du segment de retour : cosinus CLIP (ViT-B/32 openai) entre les
#              frames du retour et le PROMPT du retour. Une ancre qui ecrase le prompt fait
#              chuter ce nombre meme quand le re-ID monte.
_CLIP = None


def clip_text_score(frames01, prompt, device):
    """cos(CLIP image, CLIP texte) moyen sur les frames echantillonnees du segment."""
    global _CLIP
    import open_clip
    import torch
    import torch.nn.functional as F
    if _CLIP is None:
        m, _, _ = open_clip.create_model_and_transforms("ViT-B-32", pretrained="openai")
        m.eval().to(device)
        for p in m.parameters():
            p.requires_grad_(False)
        _CLIP = (m, open_clip.get_tokenizer("ViT-B-32"))
        print("[comp] CLIP ViT-B/32 (openai) charge pour l'adherence texte", flush=True)
    m, tok = _CLIP
    mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=device).view(1, 3, 1, 1)
    x = F.interpolate(frames01.to(device).float(), size=(224, 224), mode="bicubic",
                      align_corners=False).clamp(0, 1)
    x = (x - mean) / std
    with torch.no_grad():
        fi = F.normalize(m.encode_image(x).float(), dim=-1)
        ft = F.normalize(m.encode_text(tok([prompt]).to(device)).float(), dim=-1)
    return float((fi @ ft.T).mean())


# ---------------------------------------------------------------------------------------------
# METRIQUE FANTOME (ajoutee le 2026-08-03 apres le verdict oeil d'Aden sur memdec21u : "le sujet
# revient bien place MAIS il reste un fantome flou a l'ANCIENNE position du sujet d'interlude").
# Nos metriques ratent ca : le crop central voit le sujet rappele, le complement voit surtout le
# decor. Il faut REGARDER LA ZONE QUE LE MASQUE UNION A EFFACEE.
#
# Localisation : la boite est calculee UNE FOIS PAR CELLULE sur la video memzero (que tous les
# bras MAG partagent avant le retour, cf. plancher de bruit), sur la frame pixel juste avant le
# retour, par le MEME localiseur grossier que le pipeline (centroide de l'energie qui devie de la
# moyenne spatiale). Elle est donc identique pour tous les bras d'une cellule = comparable.
#   ghost_hf  : energie haute frequence moyenne (|laplacien|) dans la boite, fenetre chaude du
#               retour. Un fantome flou = HF EFFONDREE sous memzero. C'est LE detecteur direct.
#   ghost_dino: DINO de la boite au retour contre la MEME boite chez memzero (ou le monde a
#               simplement continue) -- a quel point cette zone est-elle etrangere.
def ghost_box(frame01, frac=0.75):
    """frame01 (3,H,W) -> (top, left, h, w) autour du centroide d'energie deviante."""
    import torch
    e = (frame01 - frame01.mean(dim=(1, 2), keepdim=True)).pow(2).sum(0).sqrt()
    w = (e - e.median()).clamp_min(0.0)
    H, W = w.shape
    if float(w.sum()) <= 0:
        cy, cx = (H - 1) / 2.0, (W - 1) / 2.0
    else:
        cy = float((w.sum(1) * torch.arange(H, dtype=torch.float32)).sum() / w.sum())
        cx = float((w.sum(0) * torch.arange(W, dtype=torch.float32)).sum() / w.sum())
    bh, bw = int(H * frac / 2), int(W * frac / 2)
    top, left = int(max(0, min(H - 2 * bh, cy - bh))), int(max(0, min(W - 2 * bw, cx - bw)))
    return top, left, 2 * bh, 2 * bw


def crop_box(frames01, box):
    t, l, h, w = box
    return frames01[:, :, t:t + h, l:l + w]


def hf_energy(frames01):
    """|laplacien| moyen : proxy de nettete. Un remplissage passe-bas ('fantome flou') le fait
    chuter, un contenu net (ou du bruit repeint proprement) non."""
    import torch
    import torch.nn.functional as F
    k = torch.tensor([[0., 1., 0.], [1., -4., 1.], [0., 1., 0.]]).view(1, 1, 3, 3)
    x = frames01.mean(dim=1, keepdim=True)
    return float(F.conv2d(x, k, padding=1).abs().mean())


def dd_center(frames01, frac=0.6):
    """Mouvement moyen (lag-1 pixel) a l'interieur du crop central: l'anti-autocollant."""
    import torch
    _, _, H, W = frames01.shape
    ch, cw = int(H * frac), int(W * frac)
    top, left = (H - ch) // 2, (W - cw) // 2
    c = frames01[:, :, top:top + ch, left:left + cw]
    return float((c[1:] - c[:-1]).abs().mean()) if c.shape[0] >= 2 else float("nan")


def main():
    import torch
    from torchvision.io import read_video
    from run_reid_probe import dino_embed, sample_seg, center_crop, seg_cos
    from quadrant_score import zero_center, lag1
    import scenarios as S

    ap = argparse.ArgumentParser()
    ap.add_argument("--scen", default="SS2")
    ap.add_argument("--triples", default="12,13,14")
    ap.add_argument("--seeds", default="0")
    ap.add_argument("--arms", default="stockoff,memzero,framebank,memdec20,oracle1")
    ap.add_argument("--out-tag", default="")
    args = ap.parse_args()

    torch.set_grad_enabled(False)
    device = torch.device("cuda")
    scen = args.scen
    triples = [int(x) for x in args.triples.split(",") if x != ""]
    seeds = [int(x) for x in args.seeds.split(",") if x != ""]
    arms = [a for a in args.arms.split(",") if a]
    ref_scen, ref_frames = CLEAN_REF[scen]

    frames = S.SCENARIOS[scen]["frames"]
    rev = S.revisit_segments(scen)[0]
    w_rev = S.score_segments(scen, frames)[rev]                 # (192, 239) pour SS2 / S2
    w_clean = S.score_segments(ref_scen, ref_frames)[0]         # A* propre, A ouvre
    sw_ret = S.SCENARIOS[scen]["schedule"][rev][0]              # 180 : frame latente du retour
    w_btail = (sw_ret - 12, sw_ret - 3)                         # (168, 177) queue de B
    w_head = (sw_ret + 3, sw_ret + 20)                          # (183, 200) juste apres le retour
    print(f"[comp] {scen}: retour @{sw_ret}, fenetres  retour {w_rev}  head {w_head}  "
          f"Bqueue {w_btail}  A*propre {ref_scen} seg0 {w_clean}", flush=True)

    # prompt effectivement demande au RETOUR, par triple (celui que le segment doit respecter)
    ret_prompt = {}
    for ti in triples:
        _pr, _sch, _ = S.schedule_for(scen, ti)
        ret_prompt[ti] = _pr[_sch[rev][1]]
    print(f"[comp] prompt du retour (t{triples[0]}): '{ret_prompt[triples[0]]}'", flush=True)

    vcache = {}

    def video(path):
        if path not in vcache:
            assert os.path.exists(path), f"video manquante {path}"
            v, _, _ = read_video(path, pts_unit="sec", output_format="THWC")
            vcache[path] = v.permute(0, 3, 1, 2).float() / 255.0
        return vcache[path]

    ecache = {}

    def emb(path, win, kind):
        """kind: full | crop50/60/70 | border (complement du crop60)"""
        key = (path, win, kind)
        if key not in ecache:
            seg = sample_seg(video(path), win[0], win[1], 8)
            x = {"full": lambda s: s,
                 "crop50": lambda s: center_crop(s, 0.5),
                 "crop60": center_crop,
                 "crop70": lambda s: center_crop(s, 0.7),
                 "border": zero_center}[kind](seg)
            ecache[key] = dino_embed(x, device)
        return ecache[key]

    def cut_ratio(path, latent_joint, win=8):
        vid = video(path)
        T = vid.shape[0]
        j = min(pix(latent_joint), T - 2)
        w = vid[max(0, j - win):min(T - 1, j + win) + 1]
        cut = float(lag1(w).max())
        intra = []
        for lo, hi in ((20, sw_ret - 60 - 5), (sw_ret - 55, sw_ret - 5), (sw_ret + 15, frames - 5)):
            s = vid[pix(lo):pix(hi):4]
            if s.shape[0] >= 2:
                intra.append(lag1(s))
        m = float(torch.cat(intra).median().clamp_min(1e-6))
        return cut / m

    res = {"scen": scen, "clean_reference": f"{ref_scen} segment 0 (A ouvre, meme triple/seed)",
           "windows": {"revisit": list(w_rev), "head": list(w_head), "b_tail": list(w_btail),
                       "clean": list(w_clean)},
           "cells": {}}
    for ti in triples:
        for si in seeds:
            cell = {}
            ref = {"mag": f"{VID}/{ref_scen}_t{ti}_s{si}_memzero.mp4",
                   "stock": f"{VID}/{ref_scen}_t{ti}_s{si}_stockoff.mp4"}
            # boite FANTOME de la cellule, calculee une seule fois sur memzero (partage par tous
            # les bras MAG avant le retour), sur la frame pixel juste avant le joint du retour
            _mz = f"{VID}/{scen}_t{ti}_s{si}_memzero.mp4"
            gbox = gref = None
            if os.path.exists(_mz):
                _v = video(_mz)
                gbox = ghost_box(_v[min(pix(sw_ret - 1), _v.shape[0] - 1)])
                gref = crop_box(sample_seg(_v, *w_head, 1), gbox)
                print(f"[comp] boite fantome t{ti}: top/left {gbox[0]},{gbox[1]} "
                      f"taille {gbox[2]}x{gbox[3]} | HF memzero {hf_energy(gref):.5f}", flush=True)
            for a in arms:
                p = f"{VID}/{scen}_t{ti}_s{si}_{a}.mp4"
                if not os.path.exists(p):
                    print(f"[comp] MANQUE {os.path.basename(p)} -- bras ignore", flush=True)
                    continue
                pipe = "stock" if a == "stockoff" else "mag"
                r = ref[pipe]
                cell[a] = {
                    "ref_pipeline": pipe,
                    # axe 1: sujet
                    "subject_crop60": seg_cos(emb(p, w_rev, "crop60"),
                                              emb(r, w_clean, "crop60")),
                    "subject_crop50": seg_cos(emb(p, w_rev, "crop50"),
                                              emb(r, w_clean, "crop50")),
                    "subject_crop70": seg_cos(emb(p, w_rev, "crop70"),
                                              emb(r, w_clean, "crop70")),
                    "subject_full": seg_cos(emb(p, w_rev, "full"), emb(r, w_clean, "full")),
                    # TRIPWIRES: mouvement dans l'ellipse (anti-autocollant) + adherence texte
                    "dd_ell_head": dd_center(sample_seg(video(p), *w_head, 1)),
                    "dd_ell_win": dd_center(sample_seg(video(p), *w_rev, 1)),
                    "clip_text": clip_text_score(sample_seg(video(p), *w_rev, 8),
                                                 ret_prompt[ti], device),
                    # FANTOME: netteté et étrangeté de la zone que le masque union a effacée
                    "ghost_hf": (hf_energy(crop_box(sample_seg(video(p), *w_head, 1), gbox))
                                 if gbox else float("nan")),
                    "ghost_dino": (seg_cos(dino_embed(crop_box(sample_seg(video(p), *w_head, 8),
                                                               gbox), device),
                                           dino_embed(crop_box(sample_seg(video(_mz), *w_head, 8),
                                                               gbox), device))
                                   if gbox else float("nan")),
                    # axe 2: fond (complement du crop60) vs la scene courante d'AVANT le retour
                    "bg_cont_head": seg_cos(emb(p, w_head, "border"),
                                            emb(p, w_btail, "border")),
                    "bg_cont_win": seg_cos(emb(p, w_rev, "border"), emb(p, w_btail, "border")),
                    # bonus: brutalite du raccord
                    "cut2_ratio": cut_ratio(p, sw_ret),
                }
                print(f"[comp] {scen} t{ti} s{si} {a:14s} sujet {cell[a]['subject_crop60']:.4f}  "
                      f"fond(head) {cell[a]['bg_cont_head']:.4f}  fond(win) "
                      f"{cell[a]['bg_cont_win']:.4f}  cut {cell[a]['cut2_ratio']:.2f}  "
                      f"DDell {cell[a]['dd_ell_head']:.4f}  CLIP {cell[a]['clip_text']:.4f}  "
                      f"FANTOME hf {cell[a]['ghost_hf']:.5f} dino {cell[a]['ghost_dino']:.4f}",
                      flush=True)
            res["cells"][f"t{ti}_s{si}"] = cell
            vcache.clear()
            ecache.clear()

    # ---- tableaux ---------------------------------------------------------------------------
    present = [a for a in arms if any(a in c for c in res["cells"].values())]
    for metric, title, arrow in (
            ("subject_crop60", "AXE 1  SUJET : reID crop60 du retour vs A* PROPRE", "haut = mieux"),
            ("bg_cont_head", "AXE 2  FOND : DINO complement-du-crop, retour(+3..+20) vs queue de B",
             "haut = le monde a continue"),
            ("bg_cont_win", "AXE 2b FOND : meme chose sur toute la fenetre de scoring du retour",
             "haut = mieux"),
            ("cut2_ratio", "BONUS  RACCORD : brutalite du joint au retour", "BAS = mieux"),
            ("dd_ell_head", "TRIPWIRE MOUVEMENT DANS L'ELLIPSE (+3..+20), lag-1 pixel",
             "s'effondrer sous memzero = AUTOCOLLANT = rejet"),
            ("dd_ell_win", "TRIPWIRE MOUVEMENT DANS L'ELLIPSE (toute la fenetre)",
             "vs memzero"),
            ("clip_text", "TRIPWIRE ADHERENCE TEXTE du retour (CLIP ViT-B/32)",
             "chuter sous memzero = le prompt n'est plus respecte"),
            ("ghost_hf", "FANTOME : nettete (|laplacien|) de la zone effacee, retour +3..+20",
             "CHUTER sous memzero = fantome flou"),
            ("ghost_dino", "FANTOME : DINO de cette zone vs memzero (monde qui continue)",
             "bas = zone etrangere"),
            ("subject_crop50", "AXE 1c SUJET, rayon 0.5", "haut = mieux"),
            ("subject_crop70", "AXE 1d SUJET, rayon 0.7", "haut = mieux"),
            ("subject_full", "AXE 1b SUJET (frame pleine, secondaire)", "haut = mieux")):
        print(f"\n===== {scen} -- {title}   [{arrow}] =====")
        print(f"{'cellule':>12} " + " ".join(f"{LABELS.get(a, a):>16}" for a in present))
        for k in sorted(res["cells"]):
            c = res["cells"][k]
            print(f"{k:>12} " + " ".join(
                f"{c[a][metric]:16.4f}" if a in c else f"{'--':>16}" for a in present))
        print(f"{'MOYENNE':>12} " + " ".join(
            f"{sum(c[a][metric] for c in res['cells'].values() if a in c) / max(1, len([c for c in res['cells'].values() if a in c])):16.4f}"
            for a in present))

    os.makedirs(OUT, exist_ok=True)
    tag = args.out_tag or scen
    p = os.path.join(OUT, f"comp_{tag}.json")
    with open(p, "w") as f:
        json.dump(res, f, indent=2)
    print(f"\n[comp] ecrit {p}")


if __name__ == "__main__":
    main()
