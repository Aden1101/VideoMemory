#!/usr/bin/env python
# ============================================================================================
# CONTROLE D'IDENTITE v3 -> v3h (exige par la chaine C2, etape 1). SANS LUI, RIEN NE PEUT
# ETRE SCORE.
#
# Pourquoi. Les masques-sujet d'Aden (et ses labels scelles) ont ete calcules sur les LATENTS
# de la PREMIERE generation (campagne --no-h). Les h viennent d'une SECONDE generation. Si les
# deux generations divergeaient d'un bit, les masques ne designeraient plus le meme pixel dans
# les memes frames que les h : le test entier serait invalide (masque d'une video, activations
# d'une autre).
#
#   I1  appariement : memes video_id des deux cotes, memes prompts, memes GRAINES
#   I2  LATENTS BIT A BIT (torch.equal) sur au moins 4 videos dont 2 twins -- le controle
#       demande ; par defaut TOUTES les videos sont verifiees (--n 0), c'est peu couteux
#   I3  la campagne h porte bien des h (h_seq non vide) la ou l'originale n'en a pas
#   I4  meme geometrie d'ecriture (write_chunks, frames, num_frame_per_block) : les ages
#       d'etat et les masques par chunk sont donc les memes indices
#
# Usage :  python reference/runs/probe1/selftest_v3h.py [--n 6]
# ============================================================================================
import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import probe1_common as C  # noqa: E402

import torch  # noqa: E402

import mem2_data as D  # noqa: E402

PASS, FAIL = "\033[32mPASS\033[0m", "\033[31mFAIL\033[0m"
_failed = []


def check(name, ok, detail=""):
    print(f"  [{PASS if ok else FAIL}] {name}" + (f" -- {detail}" if detail else ""), flush=True)
    if not ok:
        _failed.append(name)
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=0,
                    help="videos a comparer (0 = TOUTES ; le minimum exige est 4 dont 2 twins)")
    ap.add_argument("--corpora", default="v3", choices=("v3", "v4"),
                    help="quel materiau controler : v3 (sans h) vs v3h, ou v4 vs v4h")
    args = ap.parse_args()

    c0, c1 = args.corpora, args.corpora + "h"
    old = C.load_index(corpus=c0, rebuild=True, verbose=False)
    new = C.load_index(corpus=c1, rebuild=True, verbose=False)
    o_by, n_by = C.vids_by_id(old), C.vids_by_id(new)
    print(f"[{c1}] {c0} (sans h) : {len(old)} videos | {c1} (avec h) : {len(new)} videos")

    # ---- I1 appariement ------------------------------------------------------------------
    check("I1 memes video_id des deux cotes", sorted(o_by) == sorted(n_by),
          f"{len(set(o_by) ^ set(n_by))} en ecart")
    if sorted(o_by) != sorted(n_by):
        print("\nCONTROLE D'IDENTITE IMPOSSIBLE : corpus depareilles -> STOP")
        sys.exit(1)
    bad = [v for v in sorted(o_by)
           if o_by[v]["prompt"] != n_by[v]["prompt"] or o_by[v]["seed"] != n_by[v]["seed"]]
    check("I1 memes prompts ET memes graines", not bad, f"ecarts : {bad[:3]}")

    # ---- selection : au moins 4 videos dont 2 twins ---------------------------------------
    ids = sorted(o_by)
    mains = [v for v in ids if v.startswith("main:")]
    twins = [v for v in ids if v.startswith("twin:")]
    sel = ids if args.n <= 0 else (mains[:max(2, args.n // 2)] + twins[:max(2, args.n // 2)])
    n_tw = sum(1 for v in sel if v.startswith("twin:"))
    check("selection : >= 4 videos dont >= 2 twins", len(sel) >= 4 and n_tw >= 2,
          f"{len(sel)} videos, {n_tw} twins")

    # ---- I2/I3/I4 : le coeur --------------------------------------------------------------
    co, cn = D.PartCache(size=1), D.PartCache(size=1)
    n_eq, worst, n_h = 0, None, 0
    for vid in sel:
        ro = D.get_record(o_by[vid], co)
        rn = D.get_record(n_by[vid], cn)
        lo, ln = ro["latents"], rn["latents"]
        same = (tuple(lo.shape) == tuple(ln.shape) and lo.dtype == ln.dtype
                and torch.equal(lo, ln))
        n_eq += int(same)
        if not same and worst is None:
            worst = (vid, tuple(lo.shape), tuple(ln.shape),
                     float((lo.float() - ln.float()).abs().max())
                     if tuple(lo.shape) == tuple(ln.shape) else float("nan"))
        n_h += int(bool(rn.get("h_seq")))
        geo_ok = (o_by[vid]["write_chunks"] == n_by[vid]["write_chunks"]
                  and o_by[vid]["frames"] == n_by[vid]["frames"]
                  and o_by[vid]["num_frame_per_block"] == n_by[vid]["num_frame_per_block"])
        if not geo_ok:
            check(f"I4 {vid} meme geometrie d'ecriture", False)
        del ro, rn
    check(f"I2 LATENTS BIT A BIT sur {len(sel)} videos", n_eq == len(sel),
          f"{n_eq}/{len(sel)} identiques"
          + (f" ; 1er ecart {worst}" if worst else ""))
    check("I3 la campagne v3h porte bien des h", n_h == len(sel), f"{n_h}/{len(sel)}")
    check("I4 meme geometrie d'ecriture partout", True)

    print()
    if _failed:
        print(f"CONTROLE D'IDENTITE : {len(_failed)} ECHEC(S) : {_failed}\n"
              "STOP : les masques d'Aden ne correspondraient plus aux h -- ne RIEN scorer.")
        sys.exit(1)
    print("CONTROLE D'IDENTITE : PASSE. Les h de v3h decrivent EXACTEMENT les videos que "
          "l'oeil d'Aden a labellisees.")


if __name__ == "__main__":
    main()
