#!/usr/bin/env python
# ============================================================================================
# MINI-ROUND CORRIGE "texte insuffisant" (2026-08-03) -- SCORING + JUGE COULEUR
#
# Le pilote a montre que DINO ne separe pas les deux candidats d'une cellule ambigue (+0.0070
# entre "chien brun" et "chien blanc" sur la reference propre elle-meme) : il faut un juge de
# l'ATTRIBUT. Deux juges independants, l'un ancre au TEXTE, l'autre aux PIXELS :
#
#   attr_clip  cos(crop, "a <coulA> <categorie>") - cos(crop, "a <coulB> <categorie>")  (CLIP
#              ViT-B/32) : > 0 = l'attribut de A, < 0 = celui de B. Aucune reference video.
#   attr_hist  distance couleur (histogramme teinte-saturation-valeur du crop sujet) a la
#              reference A* PROPRE moins la distance a la reference B VU : > 0 = ressemble a A.
#
# GATE DE VALIDITE (avant de regarder le moindre bras) : le juge doit classer correctement les
# REFERENCES de chaque cellule -- A* propre -> attribut de A, segment B vu -> attribut de B.
# Une cellule ou il echoue est declaree VOID.
#
#   python score_textpoor2.py            # gate + tableaux + contrastes
#   python score_textpoor2.py --gate     # gate seul (selftest du juge)
# ============================================================================================
import argparse
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
OUT = os.path.join(HERE, "results", "textpoor2")
CELLS = [19, 18, 20, 21, 22]                # t19 = le cheval corrige, puis les 4 ambigues
ARMS = ["memzero", "memdec21un", "memdec20", "wronganchor", "noiseanchor", "framebank_gt"]
LABEL = {"memdec21un": "config-Aden", "memdec20": "notre-pur"}


# ------------------------------------------------------------------- juges ------------------
def hsv_hist(frames01, bins=12):
    """Histogramme TEINTE pondere par saturation*valeur, + les moyennes S et V, sur le crop
    sujet. Pondere parce qu'un pixel gris n'a pas de teinte fiable ; S et V portent le
    noir/blanc que la teinte ne peut pas coder."""
    import colorsys                                                     # noqa: F401
    x = center_crop(frames01)                                           # (T,3,H,W) in [0,1]
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


def attr_clip(frames01, ti, dev):
    from score_comp import clip_text_score
    ca, cb, cat = S.TEXTPOOR_ATTR[ti]
    return (clip_text_score(frames01, f"a {ca} {cat}", dev)
            - clip_text_score(frames01, f"a {cb} {cat}", dev))


# ------------------------------------------------------------------- main -------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gate", action="store_true")
    args = ap.parse_args()
    torch.set_grad_enabled(False)
    dev = torch.device("cuda")
    w_rev = S.score_segments("S2p", 240)[3]
    w_b = S.score_segments("S2p", 240)[2]
    w_clean = S.score_segments("SS1", 240)[0]
    cache = {}

    def frames(path, win):
        if (path, win) not in cache:
            assert os.path.exists(path), f"missing {path}"
            v, _, _ = read_video(path, pts_unit="sec", output_format="THWC")
            cache[(path, win)] = sample_seg(v.permute(0, 3, 1, 2).float() / 255., *win, 8)
        return cache[(path, win)]

    # ---- GATE : le juge classe-t-il les REFERENCES ? -------------------------------------
    gate, ok_cells = {}, []
    print("=" * 96)
    print("GATE DE VALIDITE DU JUGE (references propres : A* -> attribut de A, B vu -> attribut "
          "de B)")
    print("=" * 96)
    for ti in CELLS:
        if ti not in S.TEXTPOOR_ATTR:
            continue
        ca, cb, cat = S.TEXTPOOR_ATTR[ti]
        _pa, _pb = f"{VID}/SS1_t{ti}_s0_memzero.mp4", f"{VID}/S2p_t{ti}_s0_memzero.mp4"
        if not (os.path.exists(_pa) and os.path.exists(_pb)):
            print(f"  t{ti:2d} {cat:13s} references pas encore rendues, gate reporte")
            continue
        fa = frames(_pa, w_clean)
        fb = frames(_pb, w_b)
        va, vb = attr_clip(fa, ti, dev), attr_clip(fb, ti, dev)
        ha, hb = hsv_hist(fa), hsv_hist(fb)
        good = bool(va > 0 and vb < 0)
        sep = hist_dist(ha, hb)
        gate[ti] = {"clip_Aref": va, "clip_Bref": vb, "clip_ok": good, "hist_sep": sep,
                    "attr": [ca, cb, cat]}
        ok_cells += [ti] if good else []
        print(f"  t{ti:2d} {cat:13s} {ca:7s} vs {cb:7s} | CLIP(A*) {va:+.4f} "
              f"{'OK ' if va > 0 else 'RATE'} | CLIP(B vu) {vb:+.4f} "
              f"{'OK ' if vb < 0 else 'RATE'} | separation histogramme {sep:.3f} "
              f"-> {'VALIDE' if good else 'CELLULE VOID'}")
    n_ok = sum(1 for t in gate if gate[t]["clip_ok"])
    print(f"\n  juge valide sur {n_ok}/{len(gate)} cellules ambigues "
          f"({2 * n_ok}/{2 * len(gate)} references classees)")
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "gate_judge.json"), "w") as f:
        json.dump(gate, f, indent=2)
    if args.gate:
        return

    # ---- scores ---------------------------------------------------------------------------
    res = {}
    for ti in CELLS:
        cell = f"t{ti}_s0"
        clean = frames(f"{VID}/SS1_{cell}_memzero.mp4", w_clean)
        e_clean = (dino_embed(clean, dev), dino_embed(center_crop(clean), dev))
        h_clean = hsv_hist(clean)
        bseg = frames(f"{VID}/S2p_{cell}_memzero.mp4", w_b)
        h_b = hsv_hist(bseg)
        rec = {"attr": S.TEXTPOOR_ATTR.get(ti), "retour": S.PARAPHRASES[ti], "arms": {},
               "void": ti in S.TEXTPOOR_ATTR and not gate.get(ti, {}).get("clip_ok", False)}
        for a in ARMS:
            p = f"{VID}/S2p_{cell}_{a}.mp4"
            if not os.path.exists(p):
                continue
            f_ = frames(p, w_rev)
            row = {"reid_crop60": seg_cos(dino_embed(center_crop(f_), dev), e_clean[1]),
                   "reid_full": seg_cos(dino_embed(f_, dev), e_clean[0])}
            if ti in S.TEXTPOOR_ATTR:
                row["attr_clip"] = attr_clip(f_, ti, dev)
                h = hsv_hist(f_)
                row["attr_hist"] = hist_dist(h, h_b) - hist_dist(h, h_clean)
            rec["arms"][a] = row
        res[cell] = rec
        print(f"[tp2] {cell} score", flush=True)
    with open(os.path.join(OUT, "scores_textpoor2.json"), "w") as f:
        json.dump(res, f, indent=2)

    # ---- tableaux + contrastes -------------------------------------------------------------
    print("\n" + "=" * 96)
    for cell, rec in res.items():
        tag = "  [VOID: juge invalide]" if rec["void"] else ""
        print(f"\n== {cell}  retour='{rec['retour']}'  attributs={rec['attr']}{tag}")
        print(f"  {'bras':14s} {'reid crop60':>12s} {'reid full':>10s} {'attr_clip':>10s} "
              f"{'attr_hist':>10s}")
        for a, r in rec["arms"].items():
            print(f"  {a:14s} {r['reid_crop60']:12.4f} {r['reid_full']:10.4f} "
                  f"{r.get('attr_clip', float('nan')):10.4f} "
                  f"{r.get('attr_hist', float('nan')):10.4f}")
    print("\n" + "=" * 96)
    print("CONTRASTES (la signature d'une LECTURE du contenu) : bras - wronganchor")
    print(f"  {'cellule':10s} {'config-Aden attr':>18s} {'notre-pur attr':>16s} "
          f"{'config-Aden reid':>18s} {'notre-pur reid':>16s}")
    tally = {"attr_aden": 0, "reid_pur": 0, "n": 0}
    for cell, rec in res.items():
        A = rec["arms"]
        if "wronganchor" not in A:
            continue
        w = A["wronganchor"]
        line = f"  {cell:10s}"
        for key, arm in (("attr_clip", "memdec21un"), ("attr_clip", "memdec20"),
                         ("reid_crop60", "memdec21un"), ("reid_crop60", "memdec20")):
            d = (A[arm][key] - w[key]) if (arm in A and key in A[arm] and key in w) else float("nan")
            line += f" {d:+17.4f}" if key == "attr_clip" else f" {d:+17.4f}"
        print(line)
        if not rec["void"] and "memdec21un" in A and "attr_clip" in A["memdec21un"]:
            tally["n"] += 1
            tally["attr_aden"] += int(A["memdec21un"]["attr_clip"] - w["attr_clip"] >= 0.08)
            if "memdec20" in A:
                tally["reid_pur"] += int(A["memdec20"]["reid_crop60"] - w["reid_crop60"] >= 0.08)
    print(f"\n  REGLE SCELLEE : config-Aden - wronganchor sur attr_clip >= +0.08 sur "
          f"{tally['attr_aden']}/{tally['n']} cellules valides "
          f"(seuil : 4/5 = contenu lu ; 2-3 = zone grise ; <=1 = negatif general)")
    print(f"  (secondaire : notre-pur - wronganchor sur reid_crop60 >= +0.08 sur "
          f"{tally['reid_pur']}/{tally['n']})")
    with open(os.path.join(OUT, "verdict.json"), "w") as f:
        json.dump({"tally": tally, "gate": gate}, f, indent=2)
    print(f"\n[tp2] wrote {OUT}/")


if __name__ == "__main__":
    main()
