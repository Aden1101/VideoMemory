#!/usr/bin/env python
# ============================================================================================
# AGREGATION + REGLES SCELLEES de la MINI-BATTERIE TRAINING-FREE (2026-08-04).
#
# Les metriques par cellule viennent de score_comp.py (sujet crop60 vs A* PROPRE, fond, cut2,
# tripwires DD-ellipse / CLIP-texte) : ce script ne les recalcule pas, il les LIT et applique les
# regles pre-enregistrees dans reference/memory2_probes_registry.md :
#
#   R1  contraste `sdedit_ours` - `sdedit_wrong` (sujet crop60) >= +0.08 sur >= 4/6 cellules
#       => E1 confirmee (le contenu EST lisible une fois net). <= 1/6 => E1 ELIMINEE.
#   R2  `sinkpurge` - `memzero` >= +0.21 (la moitie du gain de memdec20) => l'objection du
#       "correctif minimal" est FONDEE et devient une ligne de baseline. < +0.05 => paree.
#
# + le test de CONTAMINATION (calcule ici, il n'est pas dans score_comp) : le retour d'un bras a
# ancre ETRANGERE ressemble-t-il au sujet SOURCE ? C'est la seule mesure qui distingue "le
# contenu n'est pas lu" de "le contenu est lu mais n'aide pas".
#
#   python score_tfree.py --arms-ours sdedit_ours06 --arms-wrong sdedit_wrong06
# ============================================================================================
import argparse
import json
import os
import sys

REPO = "/shared/ssd_14T/home/adenelmi/Anchor-Forcing"
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [REPO, os.path.join(REPO, "reference/runs/reid_probe_l1"), HERE]

VID = "/shared/ssd_14T_project/adenelmi/reid_scen/videos"
COMP = os.path.join(HERE, "results", "comp")
OUT = os.path.join(HERE, "results", "tfree")
CELLS = [("SS2", 12), ("SS2", 13), ("SS2", 14), ("S2", 2), ("S2", 4), ("S2", 11)]
CLEAN_REF = {"SS2": ("SS1", 240), "S2": ("S3", 300)}
SEED = 0
R1_THRESH = 0.08
R2_THRESH = 0.21


def sign_p(npos, nneg):
    from math import comb
    n = npos + nneg
    if n == 0:
        return 1.0
    k = min(npos, nneg)
    return min(1.0, 2.0 * sum(comb(n, i) for i in range(k + 1)) / 2.0 ** n)


def load_cells(tag_ss2, tag_s2):
    out = {}
    for scen, tag in (("SS2", tag_ss2), ("S2", tag_s2)):
        p = os.path.join(COMP, f"comp_{tag}.json")
        assert os.path.exists(p), f"manque {p} (lancer score_comp.py --scen {scen} --out-tag {tag})"
        with open(p) as f:
            d = json.load(f)
        for k, v in d["cells"].items():
            out[(scen, int(k.split("_")[0][1:]))] = v
    return out


def contamination(arms, device):
    """cos(retour, A* PROPRE du sujet SOURCE) -- le sujet dont sdedit_wrong/wronganchor a vole
    l'ancre. Un contenu LU laisserait une trace ; le §11 mesurait -0.0000 a 24 cellules."""
    import torch
    from torchvision.io import read_video
    from run_reid_probe import dino_embed, sample_seg, center_crop, seg_cos
    import scenarios as S
    cache = {}

    def emb(path, lo, hi):
        if (path, lo, hi) not in cache:
            v, _, _ = read_video(path, pts_unit="sec", output_format="THWC")
            seg = sample_seg(v.permute(0, 3, 1, 2).float() / 255., lo, hi, 8)
            cache[(path, lo, hi)] = dino_embed(center_crop(seg), device)
        return cache[(path, lo, hi)]

    res = {}
    for scen, ti in CELLS:
        ref_scen, ref_fr = CLEAN_REF[scen]
        w_clean = S.score_segments(ref_scen, ref_fr)[0]
        w_rev = S.score_segments(scen, S.SCENARIOS[scen]["frames"])[S.revisit_segments(scen)[0]]
        src = S.wrong_anchor_triple(ti)
        srcp = f"{VID}/{ref_scen}_t{src}_s{SEED}_memzero.mp4"
        if not os.path.exists(srcp):
            print(f"[contam] {scen} t{ti}: A* du SOURCE t{src} absent, cellule sautee")
            continue
        e_src = emb(srcp, *w_clean)
        row = {"source_triple": src}
        for a in arms:
            p = f"{VID}/{scen}_t{ti}_s{SEED}_{a}.mp4"
            if os.path.exists(p):
                row[a] = seg_cos(emb(p, *w_rev), e_src)
        res[f"{scen}_t{ti}"] = row
        print(f"[contam] {scen} t{ti} (source t{src}): " +
              "  ".join(f"{a} {row[a]:.4f}" for a in arms if a in row), flush=True)
    return res


def table(cells, arms, metric, title, arrow):
    print(f"\n===== {title}   [{arrow}] =====")
    print(f"{'cellule':>12} " + " ".join(f"{a:>16}" for a in arms))
    for scen, ti in CELLS:
        c = cells.get((scen, ti), {})
        print(f"{f'{scen} t{ti}':>12} " + " ".join(
            f"{c[a][metric]:16.4f}" if a in c else f"{'--':>16}" for a in arms))
    print(f"{'MOYENNE':>12} " + " ".join(
        f"{sum(cells[k][a][metric] for k in cells if a in cells[k]) / max(1, len([k for k in cells if a in cells[k]])):16.4f}"
        for a in arms))


def contrast(cells, a, b, metric="subject_crop60"):
    d = {}
    for scen, ti in CELLS:
        c = cells.get((scen, ti), {})
        if a in c and b in c:
            d[f"{scen}_t{ti}"] = c[a][metric] - c[b][metric]
    if not d:
        return None
    v = list(d.values())
    m = sum(v) / len(v)
    npos, nneg = sum(1 for x in v if x > 0), sum(1 for x in v if x < 0)
    return {"per_cell": d, "mean": m, "n": len(v), "n_pos": npos, "n_neg": nneg,
            "sign_p": sign_p(npos, nneg),
            "n_above_thresh": sum(1 for x in v if x >= R1_THRESH)}


def show(name, c):
    if c is None:
        print(f"  {name}: (bras absent)")
        return
    print(f"  {name:34s} moyenne {c['mean']:+.4f}  {c['n_pos']}+/{c['n_neg']}-  "
          f"sign-p {c['sign_p']:.3f}  cellules >= +{R1_THRESH}: {c['n_above_thresh']}/{c['n']}"
          f"   | " + " ".join(f"{k.split('_')[1]}:{v:+.3f}" for k, v in c["per_cell"].items()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ours", default="sdedit_ours06")
    ap.add_argument("--wrong", default="sdedit_wrong06")
    ap.add_argument("--tag-ss2", default="tfree_SS2")
    ap.add_argument("--tag-s2", default="tfree_S2")
    ap.add_argument("--no-contam", action="store_true")
    args = ap.parse_args()

    cells = load_cells(args.tag_ss2, args.tag_s2)
    arms = ["memzero", "sinkpurge", "memdec20", "wronganchor", args.ours, args.wrong]
    arms = [a for a in arms if any(a in c for c in cells.values())]

    for metric, title, arrow in (
            ("subject_crop60", "SUJET : reID crop60 du retour vs A* PROPRE", "haut = mieux"),
            ("bg_cont_head", "FOND : complement du crop, retour(+3..+20) vs queue de B",
             "haut = le monde a continue"),
            ("cut2_ratio", "RACCORD : brutalite du joint au retour", "BAS = mieux"),
            ("dd_ell_head", "TRIPWIRE MOUVEMENT dans l'ellipse", "s'effondrer = autocollant"),
            ("clip_text", "TRIPWIRE ADHERENCE TEXTE (CLIP)", "chuter sous memzero = mauvais"),
            ("subject_full", "SUJET (frame pleine, secondaire)", "haut = mieux")):
        table(cells, arms, metric, title, arrow)

    print("\n" + "=" * 78)
    print("CONTRASTES APPARIES (sujet crop60) -- les regles scellees du registre")
    print("=" * 78)
    c_key = contrast(cells, args.ours, args.wrong)
    show("*** LE CHIFFRE ***  ours - wrong", c_key)
    show("ours - memdec20", contrast(cells, args.ours, "memdec20"))
    show("wrong - wronganchor", contrast(cells, args.wrong, "wronganchor"))
    show("memdec20 - wronganchor (rappel §11)", contrast(cells, "memdec20", "wronganchor"))
    c_sp = contrast(cells, "sinkpurge", "memzero")
    show("sinkpurge - memzero", c_sp)
    c_md = contrast(cells, "memdec20", "memzero")
    show("memdec20 - memzero (le gain a battre)", c_md)

    verdict = {}
    if c_key:
        n = c_key["n_above_thresh"]
        verdict["R1"] = ("E1 CONFIRMEE (le contenu est lu une fois net)" if n >= 4
                         else "ZONE GRISE" if n >= 2 else "E1 ELIMINEE (le contenu n'est pas lu, "
                                                          "meme NET)")
        print(f"\nR1  contraste ours-wrong >= +{R1_THRESH} sur {n}/{c_key['n']} cellules "
              f"=> {verdict['R1']}")
    if c_sp:
        m = c_sp["mean"]
        verdict["R2"] = ("objection du CORRECTIF MINIMAL FONDEE" if m >= R2_THRESH
                         else "objection PAREE" if m < 0.05 else "plancher PARTIEL, a reporter")
        frac = m / c_md["mean"] if c_md and abs(c_md["mean"]) > 1e-9 else float("nan")
        print(f"R2  sinkpurge - memzero = {m:+.4f} ({100 * frac:.0f} % du gain de memdec20) "
              f"=> {verdict['R2']}")

    contam = {} if args.no_contam else contamination(
        [a for a in ("memzero", "memdec20", "wronganchor", args.ours, args.wrong) if a in arms],
        __import__("torch").device("cuda"))

    os.makedirs(OUT, exist_ok=True)
    p = os.path.join(OUT, "tfree_verdict.json")
    with open(p, "w") as f:
        json.dump({"arms": arms, "cells": {f"{s}_t{t}": cells.get((s, t), {}) for s, t in CELLS},
                   "contrasts": {"ours_minus_wrong": c_key,
                                 "ours_minus_memdec20": contrast(cells, args.ours, "memdec20"),
                                 "wrong_minus_wronganchor": contrast(cells, args.wrong,
                                                                     "wronganchor"),
                                 "sinkpurge_minus_memzero": c_sp,
                                 "memdec20_minus_memzero": c_md},
                   "verdict": verdict, "contamination": contam}, f, indent=2)
    print(f"\n[tfree] ecrit {p}")


if __name__ == "__main__":
    main()
