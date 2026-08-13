#!/usr/bin/env python
# ============================================================================================
# ATTRIBUTION "FENETRE vs FRAME" (question ouverte du §11.5) -- 6 cellules S2, seed 0.
#
# LE PROBLEME. Le §11 a montre que le CONTENU splice n'explique rien (wronganchor = 106 % du
# gain). Restent DEUX explications sans contenu, que rien ne separait :
#   (F1) "n'importe quelle frame lisse au slot de recence" casse le contexte de l'interlude ;
#   (F2) la CONVENTION DE FENETRE legacy de memdec20 (l'ancre est AJOUTEE apres la frame de
#        recence, donc les 11 vraies frames reculent d'un cran) -- alors qu'oracle1 REMPLACE la
#        frame de recence et ne vaut que +0.027 sur ce tier.
# Les bras `*_s0` (MEM_DECODE_SLOT_OFFSET=0) mettent notre ancre, l'ancre etrangere et le bruit
# dans la fenetre EXACTE d'oracle1. Croises avec les bras legacy, ils tranchent :
#   * si les bras s0 gardent le gain  => F1 (la frame lisse), la convention n'y est pour rien ;
#   * si les bras s0 s'effondrent vers +0.03 (le niveau d'oracle1) => F2 (la fenetre decalee),
#     et alors le "+0.43" du papier est un ARTEFACT DE CONVENTION, pas un rappel.
#
# Reference : A* PROPRE = segment 0 du rollout S3 de la meme cellule (identique a
# rescore_clean.py / rescore_controls.py, memes fenetres, meme DINO).
#
#   python score_attrib.py
# ============================================================================================
import json
import os
import statistics as st
import sys

REPO = "/shared/ssd_14T/home/adenelmi/Anchor-Forcing"
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [REPO, os.path.join(REPO, "reference/runs/reid_probe_l1"), HERE]

VID = "/shared/ssd_14T_project/adenelmi/reid_scen/videos"
OUT = os.path.join(HERE, "results", "attrib")
TRIPLES = [0, 1, 2, 3, 4, 5]
SEED = 0
LEGACY = ["memdec20", "wronganchor", "noiseanchor"]
S0 = ["memdec20s0", "wronganchor_s0", "noiseanchor_s0"]
ARMS = ["memzero"] + LEGACY + S0 + ["oracle1"]


def main():
    import torch
    from torchvision.io import read_video
    from run_reid_probe import dino_embed, sample_seg, center_crop, seg_cos, sign_test_p
    import scenarios as S

    torch.set_grad_enabled(False)
    dev = torch.device("cuda")
    w_rev = S.score_segments("S2", 240)[S.revisit_segments("S2")[0]]     # (192, 239)
    w_clean = S.score_segments("S3", 300)[0]                             # (20, 72), A ouvre
    cache = {}

    def emb(p, w):
        if (p, w) not in cache:
            assert os.path.exists(p), f"video manquante {p}"
            v, _, _ = read_video(p, pts_unit="sec", output_format="THWC")
            seg = sample_seg(v.permute(0, 3, 1, 2).float() / 255., *w, 8)
            cache[(p, w)] = (dino_embed(seg, dev), dino_embed(center_crop(seg), dev))
        return cache[(p, w)]

    res = {}
    for ti in TRIPLES:
        c = f"t{ti}_s{SEED}"
        clean = emb(f"{VID}/S3_{c}_memzero.mp4", w_clean)
        rec = {}
        for a in ARMS:
            p = f"{VID}/S2_{c}_{a}.mp4"
            if not os.path.exists(p):
                continue
            e = emb(p, w_rev)
            rec[a] = {"full": seg_cos(e[0], clean[0]), "crop60": seg_cos(e[1], clean[1])}
        res[c] = rec
        print(f"[attrib] {c}: {len(rec)} bras", flush=True)

    summary = {}
    for v in ("crop60", "full"):
        arms = [a for a in ARMS if any(a in r for r in res.values())]
        print(f"\n===== ATTRIBUTION fenetre-vs-frame -- S2, {len(res)} cellules  [{v}] =====")
        print(f"{'cellule':>9} " + " ".join(f"{a:>15}" for a in arms))
        for c, r in res.items():
            print(f"{c:>9} " + " ".join(f"{r[a][v]:15.4f}" if a in r else f"{'--':>15}"
                                        for a in arms))
        means = {a: st.mean([r[a][v] for r in res.values() if a in r])
                 for a in arms if any(a in r for r in res.values())}
        print(f"{'MOYENNE':>9} " + " ".join(f"{means.get(a, float('nan')):15.4f}" for a in arms))
        print(f"\n  gain sur memzero (le chiffre comparable au +0.434 du §11) :")
        gains = {}
        for a in arms:
            if a == "memzero":
                continue
            d = [r[a][v] - r["memzero"][v] for r in res.values() if a in r and "memzero" in r]
            if d:
                npos = sum(1 for x in d if x > 0)
                gains[a] = {"mean": st.mean(d), "n": len(d), "n_pos": npos,
                            "sign_p": sign_test_p(npos, len(d) - npos)}
                print(f"    {a:16s} {st.mean(d):+.4f}   {npos}+/{len(d) - npos}-  "
                      f"p={sign_test_p(npos, len(d) - npos):.4f}")
        print(f"\n  contrastes d'attribution :")
        contr = {}
        pairs = [("memdec20s0", "memdec20", "CONVENTION, contenu = le notre"),
                 ("wronganchor_s0", "wronganchor", "CONVENTION, contenu = faux"),
                 ("noiseanchor_s0", "noiseanchor", "CONVENTION, contenu = bruit"),
                 ("memdec20", "wronganchor", "CONTENU, convention legacy"),
                 ("memdec20s0", "wronganchor_s0", "CONTENU, convention oracle"),
                 ("memdec20s0", "noiseanchor_s0", "CONTENU vs bruit, convention oracle"),
                 ("memdec20s0", "oracle1", "notre ancre vs la VRAIE frame, meme fenetre")]
        for a, b, lab in pairs:
            d = {c: r[a][v] - r[b][v] for c, r in res.items() if a in r and b in r}
            if not d:
                continue
            npos = sum(1 for x in d.values() if x > 0)
            contr[f"{a}-{b}"] = {"mean": st.mean(d.values()), "n": len(d), "n_pos": npos,
                                 "sign_p": sign_test_p(npos, len(d) - npos), "per_cell": d}
            print(f"    {a:16s} - {b:16s} {st.mean(d.values()):+.4f}  "
                  f"{npos}+/{len(d) - npos}-  p={sign_test_p(npos, len(d) - npos):.4f}   ({lab})")
        summary[v] = {"means": means, "gains_vs_memzero": gains, "contrasts": contr}

    # ---- verdict ---------------------------------------------------------------------------
    g = summary["crop60"]["gains_vs_memzero"]
    leg = g.get("memdec20", {}).get("mean", float("nan"))
    s0 = g.get("memdec20s0", {}).get("mean", float("nan"))
    orc = g.get("oracle1", {}).get("mean", float("nan"))
    keep = s0 / leg if leg else float("nan")
    print(f"\n{'=' * 92}")
    print(f"VERDICT (crop60) : legacy {leg:+.4f} | convention oracle {s0:+.4f} "
          f"({100 * keep:.0f} % du gain) | oracle1 (vraie frame, meme fenetre) {orc:+.4f}")
    if keep >= 0.7:
        verdict = ("F1 : la CONVENTION n'y est pour rien -- ce qui compte est qu'une frame "
                   "plausible occupe le slot de recence")
    elif keep <= 0.3:
        verdict = ("F2 : c'est la FENETRE DECALEE (convention legacy) qui porte le gain -- le "
                   "+0.43 est un artefact de convention, pas un rappel")
    else:
        verdict = "MIXTE : les deux facteurs contribuent, aucun ne domine (30-70 %)"
    print(f"  => {verdict}")
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "attrib.json"), "w") as f:
        json.dump({"scores": res, "summary": summary, "verdict": verdict,
                   "keep_ratio_crop60": keep}, f, indent=2)
    print(f"[attrib] ecrit {OUT}/attrib.json")


if __name__ == "__main__":
    main()
