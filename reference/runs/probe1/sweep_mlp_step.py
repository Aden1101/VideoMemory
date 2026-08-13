#!/usr/bin/env python
# ============================================================================================
# PLAN FACTORIEL MLP -- CHOIX AVEUGLE DE LA TAILLE DE MICRO-LOT ET DU PAS
# (registre : "AMENDEMENT (Aden 07/08)", critere SCELLE AVANT execution).
#
# CE QUE CE SCRIPT DECIDE : les deux constantes `MLP_MICRO_BATCH` et `MLP_STEP_SCALE` de
# build_states.py, c'est-a-dire le calendrier sequentiel du bras A2 (PRIMARY).
# COMMENT : par le critere scelle, AVEUGLE A L'IDENTITE --
#     maximiser  R = moyenne_sur_les_ecritures  (||f_W(k)-v||_avant - ||f_W(k)-v||_apres)
#                                               / ||f_W(k)-v||_avant
#     sous contrainte de SANITE NUMERIQUE (fonction `sane` ci-dessous, et RIEN d'autre) :
#     etats finis, |max| dans la dynamique fp16 (65504), pas d'effondrement
#     (||etat||_F >= 1 % de la norme d'init).
# Aucun masque, aucune sonde, aucune reference de sujet ou de fond n'entre ici : il est
# IMPOSSIBLE de choisir une cellule "parce qu'elle donne un meilleur score d'identite".
# Mesure sur les 2 PREMIERES videos du corpus v2 (ordre de construction, pas un choix), sur
# les 3 blocs memoire, sur TOUTES les ecritures.
#
# Le pur sequentiel (micro-lot = 1) est un CONTROLE : son cout (~4680 pas par chunk et par
# bloc) est mesure d'abord ; s'il ne tient pas dans le budget, il est evalue sur un prefixe
# borne d'ecritures et compare aux autres micro-lots SUR LE MEME PREFIXE (table bornee).
#
# RESULTAT (07/08, 2 videos x 3 blocs x 25 ecritures ; artefacts runs/sweep_mlp_step*.json) :
#   - grille SCELLEE (pas 0.5 / 1 / 2, somme et moyenne) : les 18 cellules ont R < 0, de -0.45
#     (micro-lot 32, x1.0, moyenne) a -345 (micro-lot 512, x0.5, somme). Autrement dit AUCUN
#     reglage de la plage scellee ne fait descendre a A2 son propre objectif : le pas brut y est
#     1 a 2 ordres de grandeur trop grand, les etats intermediaires explosent et c'est le
#     plafond de norme qui les rabat.
#   - EXTENSION DECLAREE de la plage de pas vers le bas (0.0003 ... 0.25), meme critere aveugle :
#     R devient positif, maximum en (micro-lot 32, pas x0.01, moyenne) R = +0.0480 ;
#     micro-lot 128 x0.02 -> +0.0440 ; micro-lot 512 x0.05 -> +0.0398 ; PUR SEQUENTIEL
#     micro-lot 1 x0.01 -> +0.0749 (le meilleur, mais 464 s pour 2 videos : ~3.9 h pour les 60
#     videos d'un bras BRUT, et hors budget pour A3 qui paie Muon a chaque pas).
#   - references au meme critere : A1 (residu, moyenne, Muon) -0.1737 ; A0 (notre ecriture
#     actuelle : Hebb, moyenne, Muon) -0.3596 ; A3 (residu, sequentiel, Muon) +0.0251 en
#     micro-lot 512 et -0.0178 en 128 (son pas n'a pas de parametre libre : Muon renormalise).
#
# CE QUE LE CRITERE SCELLE NE VOIT PAS (mesure ajoutee le 07/08, elle aussi AVEUGLE) : la PART
# ECRITE de l'etat, ||etat - etat_sans_ecriture|| / ||etat||, ou "sans ecriture" = la meme
# trajectoire a gradient nul (decroissance + plafond seuls). Le critere borne le pas par le HAUT
# (l'explosion est ecartee par la sanite) mais RIEN ne le borne par le bas : maximiser R pousse
# vers des ecritures qui n'ecrivent presque rien. Table conjointe (2 videos, R | part ecrite) :
#     micro-lot 32 : x0.01 +0.0480|0.067   x0.05 +0.0267|0.169   x0.1 -3.33|0.224
#                    x0.25 -0.143|0.363    x0.5 -0.577|1.034     x1 -0.449|1.043
#     micro-lot 128: x0.01 +0.0416|0.042   x0.05 +0.0396|0.097   x0.1 +0.0225|0.141
#     micro-lot 512: x0.01 +0.0374|0.031   x0.05 +0.0398|0.061   x0.1 +0.0308|0.088
#     micro-lot 1  : x0.01 +0.0749|0.294   <- le seul reglage BRUT bon sur les DEUX axes
#     A0 -0.3596|0.883   A1 -0.1737|0.728   A3(512) +0.0251|0.952   A3(128) -0.0178|1.005
# LECTURE : pour le gradient BRUT par micro-lots, R > 0 seulement la ou l'ecriture pese < ~20 %
# de l'etat ; des qu'elle domine l'etat (>= 0.36) R passe negatif. Les deux seuls reglages qui
# ecrivent ET descendent sont le PUR sequentiel (brut) et A3 en micro-lot 512 (Muon). C'est un
# RESULTAT sur la cellule (decroissance + plafond de norme + lr appris sous Muon), pas un bug :
# il est rapporte tel quel et la ratification du reglage revient au coordinateur.
#
# Usage (GPU 2 uniquement : 0 et 1 = generation v4) :
#   cd /shared/ssd_14T/home/adenelmi/Anchor-Forcing && \
#   CUDA_DEVICE_ORDER=PCI_BUS_ID nvidia-smi --query-gpu=index,memory.free --format=csv && \
#   CUDA_VISIBLE_DEVICES=2 python reference/runs/probe1/sweep_mlp_step.py
# ============================================================================================
import argparse
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import probe1_common as C  # noqa: E402

import torch  # noqa: E402

import build_signatures as B  # noqa: E402
import build_states as BS  # noqa: E402
import mem2_data as D  # noqa: E402
from wan.modules.video_memory import (swiglu_apply,  # noqa: E402
                                      swiglu_fast_weight_gradient)

FP16_MAX = 65504.0
COLLAPSE_FRAC = 0.01          # "effondrement" : ||etat||_F < 1 % de la norme d'init


def fro(state):
    return float(sum(float(t.float().norm()) ** 2 for t in state) ** 0.5)


def amax(state):
    return max(float(t.float().abs().max()) for t in state)


def zero_grad_fn(k, v, w0, w1, w2, lr0, lr1, lr2):
    """Gradient identiquement NUL : la meme trajectoire d'ecritures, mais sans rien ecrire --
    il ne reste que la decroissance et le plafond de norme. Sert de reference pour la PART
    ECRITE de l'etat (garde-fou bas du pas ; cf. selftest_mlp_delta D6)."""
    return torch.zeros_like(w0), torch.zeros_like(w1), torch.zeros_like(w2)


@torch.no_grad()
def nowrite_final(cell, h_seq, max_writes=None):
    """Etat final de la trajectoire SANS ecriture (gradient nul), memes chunks, meme cellule."""
    hs = h_seq if max_writes is None else h_seq[:int(max_writes)]
    state = tuple(t.detach() for t in cell.init_state(int(hs[0].shape[0]), hs[0].device,
                                                      cell.qkv.weight.dtype))
    for h in hs:
        state = cell.write_only(h, state, grad_fn=zero_grad_fn)
    return state


@torch.no_grad()
def arm_curve(cell, h_seq, grad_fn, micro_batch, muon_steps, max_writes=None, nowrite=None):
    """Rejoue les ecritures d'UN bloc avec un reglage donne et mesure, ecriture par ecriture,
    l'erreur de reconstruction du chunk AVANT et APRES son ecriture (memes k, v : ils ne
    dependent pas de l'etat). Rend (reductions relatives, stats de sanite, secondes)."""
    saved = cell.muon_update_steps
    if muon_steps is not None:
        cell.muon_update_steps = int(muon_steps)
    try:
        hs = h_seq if max_writes is None else h_seq[:int(max_writes)]
        state = tuple(t.detach() for t in cell.init_state(int(hs[0].shape[0]), hs[0].device,
                                                          cell.qkv.weight.dtype))
        n0 = fro(state)
        red, worst_max, worst_fro = [], 0.0, float("inf")
        t0 = time.time()
        for h in hs:
            _, k, v, _l0, _l1, _l2, _B = cell._qkv_lr(h)
            kf, vf = k.float(), v.float()

            def err(st):
                w0, w1, w2 = [t.float() for t in st]
                return float((swiglu_apply(kf, w0, w1, w2) - vf).norm())

            e0 = err(state)
            state = cell.write_only(h, state, grad_fn=grad_fn, micro_batch=micro_batch)
            e1 = err(state)
            red.append((e0 - e1) / max(e0, 1e-12))
            worst_max = max(worst_max, amax(state))
            worst_fro = min(worst_fro, fro(state))
        dt = time.time() - t0
        finite = all(torch.isfinite(t).all() for t in state)
        wp = None
        if nowrite is not None:      # part ECRITE de l'etat final (0 = l'ecriture n'ecrit rien)
            num = float(sum(float((x.float() - y.float()).norm()) ** 2
                            for x, y in zip(state, nowrite)) ** 0.5)
            wp = num / max(fro(state), 1e-9)
        return red, {"max_abs": worst_max, "min_fro": worst_fro, "init_fro": n0,
                     "finite": bool(finite), "written_part": wp}, dt
    finally:
        cell.muon_update_steps = saved


def sane(st):
    return (st["finite"] and st["max_abs"] < FP16_MAX
            and st["min_fro"] >= COLLAPSE_FRAC * st["init_fro"])


def run_cell(cells, blocks, hs, rule, micro_batch, muon_steps, scale, max_writes=None,
             mean=False):
    """Une cellule de grille, moyennee sur les blocs et les videos. Rend
    (R, sanite, s, n_ecritures, courbe par ecriture).

    `courbe[j]` = reduction relative moyenne (sur les entrees video x bloc) de la j-eme
    ecriture. C'est ce qui distingue "le pas est trop grand des le debut" de "ca part en vrille
    quand l'etat s'est charge" : le critere scelle est la MOYENNE de cette courbe, la courbe
    elle-meme n'est que du diagnostic (aveugle a l'identite, comme tout ce fichier)."""
    base = BS.swiglu_delta_gradient if rule == "delta" else swiglu_fast_weight_gradient
    gfn = BS.scaled_gradient(base, scale, mean_over_tokens=mean)
    reds, ok, dt, per_entry, wps = [], True, 0.0, [], []
    for h in hs:                                   # une entree par (video, bloc)
        r, st, t = arm_curve(cells[h["block"]], h["seq"], gfn, micro_batch, muon_steps,
                             max_writes=max_writes, nowrite=h.get("nowrite"))
        reds += r
        per_entry.append(r)
        ok &= sane(st)
        dt += t
        if st["written_part"] is not None:
            wps.append(st["written_part"])
    nw = len(per_entry[0])
    curve = ([sum(e[j] for e in per_entry) / len(per_entry) for j in range(nw)]
             if all(len(e) == nw for e in per_entry) else [])
    return (sum(reds) / len(reds)), ok, dt, len(reds), curve, (min(wps) if wps else None)


def fmt_curve(curve):
    """Resume d'une courbe par ecriture : les 3 premieres, la mediane, les 3 dernieres."""
    if not curve:
        return ""
    s = sorted(curve)
    med = s[len(s) // 2] if len(s) % 2 else 0.5 * (s[len(s) // 2 - 1] + s[len(s) // 2])
    return (" | ecritures 1-3 " + ",".join(f"{c:+.3g}" for c in curve[:3])
            + f" med {med:+.3g} fin " + ",".join(f"{c:+.3g}" for c in curve[-3:]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="v2")
    ap.add_argument("--videos", type=int, default=2, help="scelle : 2 videos")
    ap.add_argument("--mbs", default="512,128,32")
    ap.add_argument("--scales", default="0.5,1.0,2.0")
    ap.add_argument("--modes", default="sum,mean", help="somme ou moyenne sur le micro-lot")
    ap.add_argument("--skip-refs", dest="skip_refs", action="store_true")
    ap.add_argument("--only-grid", dest="only_grid", action="store_true",
                    help="ne fait QUE la grille A2 (ni A3, ni A4, ni le controle borne) : sert a "
                         "COMPLETER la grille scellee (micro-lot 1 sur toute la longueur, ou une "
                         "plage de pas plus large) sans repayer les sections deja mesurees")
    ap.add_argument("--arms", default="",
                    help="mesure R et PART ECRITE pour EXACTEMENT les bras declares dans "
                         "build_states.MLP_ARMS (liste de noms separes par des virgules, ou "
                         "'all'), puis s'arrete. C'est le mode qui produit les deux chiffres "
                         "aveugles a l'identite rapportes a cote des scores de chaque bras.")
    ap.add_argument("--pure-writes", dest="pure_writes", type=int, default=3,
                    help="controle micro-lot=1 : nombre d'ecritures du prefixe borne")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    assert torch.cuda.is_available(), "GPU obligatoire"
    C.ensure_dirs()
    C.set_determinism(0)
    device = "cuda"
    blocks = [5, 16, 27]
    cells = B.build_cells_for_stream(C.DEFAULT_CONFIG, blocks, device)

    vids, pairs = C.pair_dataset(verbose=False, corpus=args.corpus)
    pair_ids = {p["main"] for p in pairs} | {p["twin"] for p in pairs}
    sel = sorted([v for v in vids if v["video_id"] in pair_ids],
                 key=lambda v: (v["part"], v["vi"]))[:args.videos]
    cache = D.PartCache(size=1)
    hs, ntok = [], 0
    for meta in sel:
        n = len(meta["write_chunks"])
        h = D.load_h(meta, cache, blocks, upto=n - 1, device=device, dtype=torch.bfloat16)
        for b in blocks:
            e = {"block": b, "seq": h[b], "video": meta["video_id"]}
            e["nowrite"] = nowrite_final(cells[b], h[b])   # reference "sans ecriture" du couple
            hs.append(e)
        ntok = int(h[blocks[0]][0].shape[-2])
    print(f"[sweep] {len(sel)} videos {[v['video_id'] for v in sel]}, {len(blocks)} blocs, "
          f"{len(hs[0]['seq'])} ecritures/video, {ntok} tokens/chunk", flush=True)

    # --- mode "bras declares" : les deux chiffres aveugles de chaque bras du plan -------------
    if args.arms:
        names = (sorted(BS.MLP_ARMS) if args.arms == "all"
                 else [n.strip() for n in args.arms.split(",")])
        out = {"corpus": args.corpus, "videos": [v["video_id"] for v in sel], "blocks": blocks,
               "criterion": "R (mean relative error reduction per write) + written_part "
                            "(||state - no-write state|| / ||state||), both identity-BLIND",
               "written_part_floor": 0.5, "arms": {}}
        print("[bras] R = reduction relative moyenne de l'erreur par ecriture ; part ecrite = "
              "poids de l'ecriture dans l'etat (plancher scelle 0.5)")
        # A0 = notre ecriture actuelle, la reference de la contrainte
        r0, ok0, dt0, _n0, cv0, wp0 = run_cell(cells, blocks, hs, "hebb", None, None, 1.0)
        out["arms"]["A0 state (ecriture actuelle)"] = {"R": r0, "written_part": wp0,
                                                       "sane": ok0, "s": dt0}
        print(f"      {'A0 state (actuelle)':<28} R = {r0:+.4f}   part ecrite {wp0:.3f}   "
              f"{dt0:.1f}s", flush=True)
        for n in names:
            rule, _mask, mb, muon, sc, mean = BS.MLP_ARMS[n]
            r, ok, dt, _n, cv, wp = run_cell(cells, blocks, hs, rule, mb, muon, sc, mean=mean)
            out["arms"][n] = {"rule": rule, "micro_batch": mb, "muon": muon, "scale": sc,
                              "mean": mean, "R": r, "written_part": wp, "sane": ok, "s": dt,
                              "R_by_write": cv, "meets_floor": bool(wp is not None
                                                                    and wp >= 0.5)}
            print(f"      {n:<28} R = {r:+.4f}   part ecrite {wp:.3f}   "
                  f"({rule}, micro-lot {mb}, Muon {muon}, pas x{sc}"
                  f"{', moyenne' if mean else ''})   {dt:.1f}s", flush=True)
        C.save_json(out, args.out or os.path.join(C.RUNS_DIR, "sweep_mlp_arms.json"))
        print(f"[bras] TERMINE -> {args.out or os.path.join(C.RUNS_DIR, 'sweep_mlp_arms.json')}")
        print("SWEEP_MLP_DONE")
        return

    mbs = [int(x) for x in args.mbs.split(",")]
    scales = [float(x) for x in args.scales.split(",")]
    res = {"corpus": args.corpus, "videos": [v["video_id"] for v in sel], "blocks": blocks,
           "n_tokens_per_chunk": ntok, "criterion":
           "mean relative reduction of ||f_W(k)-v|| per write (identity-BLIND, sealed)",
           "sanity": {"fp16_max": FP16_MAX, "collapse_frac": COLLAPSE_FRAC},
           "grid": [], "bounded": [], "reference": {}}

    # --- reference : le calendrier historique (A1 = un seul pas, Muon) ----------------------
    if not args.skip_refs:
        r, ok, dt, n, cv, wp = run_cell(cells, blocks, hs, "delta", None, None, 1.0)
        res["reference"]["A1_delta_avg_muon"] = {"R": r, "sane": ok, "s": dt, "n_writes": n,
                                                 "R_by_write": cv, "written_part": wp}
        print(f"[ref ] A1 residu/moyenne/Muon        : R = {r:+.4f}  sanite={ok}  "
              f"part ecrite {wp if wp is None else round(wp, 3)}  {dt:.1f}s"
              + fmt_curve(cv), flush=True)
        r0, ok0, dt0, _, cv0, wp0 = run_cell(cells, blocks, hs, "hebb", None, None, 1.0)
        res["reference"]["A0_hebb_avg_muon"] = {"R": r0, "sane": ok0, "s": dt0,
                                                "R_by_write": cv0, "written_part": wp0}
        print(f"[ref ] A0 Hebb/moyenne/Muon (actuel) : R = {r0:+.4f}  sanite={ok0}  "
              f"part ecrite {wp0 if wp0 is None else round(wp0, 3)}  {dt0:.1f}s"
              + fmt_curve(cv0), flush=True)

    # --- la grille scellee : A2 (residu, BRUT) x micro-lot x pas x (somme | moyenne) --------
    # L'axe somme/moyenne est une EXTENSION declaree du grillage scelle : le gradient d'un
    # micro-lot de m tokens est une SOMME, donc en "somme" le pas croit avec m et le facteur
    # calendrier serait confondu avec le facteur pas. Le critere reste le meme et reste aveugle.
    print("[grid] A2 = residu + sequentiel + BRUT (le bras dont le reglage est choisi ici)")
    best = None
    for mean in [m == "mean" for m in args.modes.split(",")]:
        for mb in mbs:
            for s in scales:
                r, ok, dt, n, cv, wp = run_cell(cells, blocks, hs, "delta", mb, 0, s,
                                                mean=mean)
                res["grid"].append({"micro_batch": mb, "scale": s, "mean": mean, "R": r,
                                    "sane": ok, "s": dt, "R_by_write": cv,
                                    "written_part": wp})
                print(f"       micro-lot {mb:>4}  pas x{s:<6g} "
                      f"{'moyenne' if mean else 'somme  '} : R = {r:+.4f}  sanite={ok}  "
                      f"part ecrite {wp if wp is None else round(wp, 3)}  "
                      f"{dt:.1f}s" + fmt_curve(cv), flush=True)
                if ok and (best is None or r > best["R"]):
                    best = {"micro_batch": mb, "scale": s, "mean": mean, "R": r}
    assert best is not None, "aucune cellule saine : le critere ne peut pas trancher"
    print(f"[grid] CELLULE RETENUE (critere aveugle) : micro-lot {best['micro_batch']}, "
          f"pas x{best['scale']}, {'moyenne' if best['mean'] else 'somme'}, "
          f"R = {best['R']:+.4f}", flush=True)
    res["selected"] = best

    # --- informatif : A3 (residu + sequentiel + MUON) au meme calendrier --------------------
    # Muon absorbe le pas (X/||X||) : A3 n'a donc pas de reglage libre. On mesure le meme
    # critere pour documenter l'effet de l'orthogonalisation a calendrier apparie.
    for mb in ([] if args.only_grid else mbs):
        r, ok, dt, n, cv, wp = run_cell(cells, blocks, hs, "delta", mb, None, 1.0)
        res.setdefault("A3", []).append({"micro_batch": mb, "R": r, "sane": ok, "s": dt,
                                         "R_by_write": cv, "written_part": wp})
        print(f"[A3  ] micro-lot {mb:>4} residu/sequentiel/Muon : R = {r:+.4f}  sanite={ok}  "
              f"part ecrite {wp if wp is None else round(wp, 3)}  {dt:.1f}s"
              + fmt_curve(cv), flush=True)
    # --- informatif : A4 (Hebb + sequentiel + brut) au reglage retenu de A2 -----------------
    if not args.only_grid:
        r, ok, dt, n, cv, wp = run_cell(cells, blocks, hs, "hebb", best["micro_batch"], 0,
                                    best["scale"], mean=best["mean"])
        res["A4"] = {"micro_batch": best["micro_batch"], "scale": best["scale"],
                     "mean": best["mean"], "R": r, "sane": ok, "s": dt, "R_by_write": cv,
                     "written_part": wp}
        print(f"[A4  ] Hebb/sequentiel/brut au reglage de A2 : R = {r:+.4f}  sanite={ok}  "
              f"{dt:.1f}s  (attendu NEGATIF : l'objectif Hebb n'optimise pas cette erreur)",
              flush=True)

    # --- controle : le pur sequentiel (micro-lot = 1) sur un prefixe borne ------------------
    print(f"[ctrl] pur sequentiel (micro-lot = 1) vs micro-lots, sur les {args.pure_writes} "
          f"premieres ecritures (meme prefixe pour tous : comparable)")
    for mb in ([] if args.only_grid else [1] + mbs):
        r, ok, dt, n, _cv, _wp = run_cell(cells, blocks, hs, "delta", mb, 0, best["scale"],
                                     max_writes=args.pure_writes, mean=best["mean"])
        res["bounded"].append({"micro_batch": mb, "scale": best["scale"], "mean": best["mean"],
                               "R": r, "sane": ok, "s": dt, "n_writes": n})
        # cout extrapole d'UNE video entiere (3 blocs), a partir du prefixe mesure
        full = dt / max(1, args.pure_writes) * len(hs[0]["seq"]) / max(1, len(hs) // 3)
        print(f"       micro-lot {mb:>4} : R = {r:+.4f}  sanite={ok}  {dt:.1f}s "
              f"(extrapolation : {full:.0f}s par video x 3 blocs, "
              f"{full * 60 / 60:.0f}s pour 60 videos = {full * 60 / 3600:.1f}h)", flush=True)

    out = args.out or os.path.join(C.RUNS_DIR, "sweep_mlp_step.json")
    C.save_json(res, out)
    print(f"[sweep] TERMINE -> {out}")
    print("SWEEP_MLP_DONE")


if __name__ == "__main__":
    main()
