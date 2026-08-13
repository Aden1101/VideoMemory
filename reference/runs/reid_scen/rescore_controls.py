#!/usr/bin/env python
# ============================================================================================
# RE-SCORING PROPRE des 2 CONTROLES REVIEWER du claim S2 (2026-08-03)
#
# Meme convention que rescore_clean.py, verbatim (memes fenetres, meme DINO, meme A* propre) :
#   * la reference A* d'une cellule = segment 0 du rollout S3 de LA MEME cellule (meme triple,
#     meme seed), ou A OUVRE la video et ou rien ne la precede -- la "premiere apparition" de S2
#     est contaminee par l'ouvreur epingle dans le sink (cos 0.165 d'un A propre) ;
#   * le segment note = le retour de A dans S2 (fenetre latente 192-239) ;
#   * les bras MAG sont tous references sur le A* MAG (memzero S3), qui est produit AVANT tout
#     switch donc identique d'un bras a l'autre du meme pipeline.
# Bras : memzero (jumeau apparie) | memdec20 (le claim) | noiseanchor | wronganchor.
#
# REGLE DE DECISION PRE-ENREGISTREE (reference/memory2_probes_registry.md, addendum 2026-08-03) :
#   si l'un des 2 controles rend >= la MOITIE du +0.434 de memdec20, le claim "c'est le CONTENU
#   de la memoire" tombe et le papier doit etre reformule en "recence generique".
#   Predictions : noiseanchor +0.02 a +0.08 ; wronganchor +0.00 a +0.06.
#
#   python rescore_controls.py --triples 0,1,2,3        # shard
#   python rescore_controls.py --mode aggregate         # tables
# ============================================================================================
import argparse
import glob
import json
import os
import sys

REPO = "/shared/ssd_14T/home/adenelmi/Anchor-Forcing"
HERE = os.path.dirname(os.path.abspath(__file__))
PROBE_DIR = os.path.join(REPO, "reference/runs/reid_probe_l1")
sys.path[:0] = [REPO, PROBE_DIR, HERE]

VID = "/shared/ssd_14T_project/adenelmi/reid_scen/videos"
OUT = os.path.join(HERE, "results", "controls_rescore")
SCEN = "S2"
ARMS = ("memzero", "memdec20", "noiseanchor", "wronganchor")
# le chiffre-titre re-scored du rapport (rescore_clean.py, 24 cellules) : la regle de decision
# est exprimee en fraction de celui-ci, et il est RECALCULE ici comme controle de coherence.
HEADLINE = {"full": 0.4342748633741091, "crop60": 0.4232193074033906}


def _shard_path(tag):
    return os.path.join(OUT, f"shard_{tag}.json")


# ----------------------------------------------------------------------------------------
def measure(triples, seeds, device):
    import torch                                                        # noqa: F401
    from torchvision.io import read_video
    from run_reid_probe import dino_embed, sample_seg, center_crop, seg_cos
    import scenarios as S

    w_rev = S.score_segments(SCEN, 240)[S.revisit_segments(SCEN)[0]]     # (192, 239)
    w_clean = S.score_segments("S3", 300)[0]                            # (20, 72), A ouvre
    cache = {}

    def emb(path, lo, hi):
        key = (path, lo, hi)
        if key in cache:
            return cache[key]
        assert os.path.exists(path), f"missing video {path}"
        v, _, _ = read_video(path, pts_unit="sec", output_format="THWC")
        vid = v.permute(0, 3, 1, 2).float() / 255.
        seg = sample_seg(vid, lo, hi, 8)
        out = (dino_embed(seg, device), dino_embed(center_crop(seg), device))
        cache[key] = out
        return out

    res = {"cells": {}}
    for ti in triples:
        for si in seeds:
            rec = {"triple": ti, "seed_idx": si}
            other = 1 - si if si in (0, 1) else None
            clean = emb(f"{VID}/S3_t{ti}_s{si}_memzero.mp4", *w_clean)
            clean_o = emb(f"{VID}/S3_t{ti}_s{other}_memzero.mp4", *w_clean) if other is not None \
                else None
            for a in ARMS:
                p = f"{VID}/{SCEN}_t{ti}_s{si}_{a}.mp4"
                if not os.path.exists(p):
                    print(f"[rescore] t{ti}_s{si}: no {a} video yet, skipped")
                    continue
                e = emb(p, *w_rev)
                row = {"full": seg_cos(e[0], clean[0]), "crop60": seg_cos(e[1], clean[1])}
                if clean_o is not None:
                    row["full_other_seed"] = seg_cos(e[0], clean_o[0])
                    row["crop60_other_seed"] = seg_cos(e[1], clean_o[1])
                rec[a] = row
            res["cells"][f"t{ti}_s{si}"] = rec
            print(f"[rescore] t{ti}_s{si} done ({len(cache)} embeddings cached)", flush=True)
    return res


# ----------------------------------------------------------------------------------------
def aggregate():
    import statistics as st
    from run_reid_probe import sign_test_p

    cells = {}
    for p in sorted(glob.glob(_shard_path("*"))):
        with open(p) as f:
            cells.update(json.load(f)["cells"])
    assert cells, f"no shard JSON in {OUT}"
    print(f"\n{len(cells)} cellules agregees depuis {len(glob.glob(_shard_path('*')))} shards")

    summary = {"n_cells": len(cells), "variants": {}, "headline_reference": HEADLINE}
    for variant in ("full", "crop60"):
        base = {k: c["memzero"][variant] for k, c in cells.items() if "memzero" in c}
        vs = {"arm_means": {}, "paired_vs_memzero": {}}
        print(f"\n================ S2 [{variant}] : re-ID vs A* PROPRE (segment 0 du S3 de la "
              f"meme cellule) ================")
        print(f"  {'bras':14s} {'score':>8s}   {'delta vs memzero':>18s}   {'signes':>9s} "
              f"{'p':>10s}   {'% du memdec20':>14s}")
        for a in ARMS:
            vals = {k: c[a][variant] for k, c in cells.items() if a in c}
            if not vals:
                continue
            m = st.mean(vals.values())
            vs["arm_means"][a] = m
            if a == "memzero":
                print(f"  {a:14s} {m:8.4f}   {'--':>18s}   {'--':>9s} {'--':>10s}   {'--':>14s}")
                continue
            d = [vals[k] - base[k] for k in vals if k in base]
            mm = st.mean(d)
            sd = st.stdev(d) if len(d) > 1 else 0.0
            npos = sum(1 for x in d if x > 0)
            p = sign_test_p(npos, len(d) - npos)
            frac = mm / HEADLINE[variant]
            vs["paired_vs_memzero"][a] = {
                "mean_delta": mm, "sd": sd, "sem": sd / max(1, len(d)) ** 0.5, "n": len(d),
                "n_pos": npos, "n_neg": len(d) - npos, "sign_p": p,
                "frac_of_headline": frac,
                "deltas": {k: vals[k] - base[k] for k in vals if k in base}}
            print(f"  {a:14s} {m:8.4f}   {mm:+9.4f} +-{sd / max(1, len(d)) ** 0.5:.4f}   "
                  f"{npos:3d}+/{len(d) - npos:2d}- {p:10.2e}   {100 * frac:13.1f}%")
        # controles apparies entre bras (le contraste qui compte vraiment)
        vs["paired_vs_memdec20"] = {}
        for a in ("noiseanchor", "wronganchor"):
            d = [cells[k][a][variant] - cells[k]["memdec20"][variant] for k in cells
                 if a in cells[k] and "memdec20" in cells[k]]
            if not d:
                continue
            npos = sum(1 for x in d if x > 0)
            vs["paired_vs_memdec20"][a] = {
                "mean_delta": st.mean(d), "n": len(d), "n_pos": npos, "n_neg": len(d) - npos,
                "sign_p": sign_test_p(npos, len(d) - npos)}
            r = vs["paired_vs_memdec20"][a]
            print(f"    [contraste] {a} - memdec20 : {r['mean_delta']:+.4f}  "
                  f"{r['n_pos']}+/{r['n_neg']}-  p = {r['sign_p']:.2e}")
        summary["variants"][variant] = vs

    # ---- test d'instance (meme graine vs autre graine du meme prompt) --------------------
    print(f"\n================ TEST D'INSTANCE (le A rendu est-il CELUI de cette cellule ?) "
          f"================")
    inst = {}
    for a in ARMS:
        for variant in ("full", "crop60"):
            rows = [(c[a][variant], c[a].get(variant + "_other_seed")) for c in cells.values()
                    if a in c and c[a].get(variant + "_other_seed") is not None]
            if not rows:
                continue
            d = [x - y for x, y in rows]
            npos = sum(1 for x in d if x > 0)
            sd = st.stdev(d) if len(d) > 1 else 0.0
            inst[f"{a}_{variant}"] = {
                "same_seed": st.mean(x for x, _ in rows), "other_seed": st.mean(y for _, y in rows),
                "mean_delta": st.mean(d), "sem": sd / max(1, len(d)) ** 0.5, "n": len(d),
                "n_pos": npos, "n_neg": len(d) - npos,
                "sign_p": sign_test_p(npos, len(d) - npos)}
            if variant == "full":
                r = inst[f"{a}_{variant}"]
                print(f"    {a:14s} meme graine {r['same_seed']:.4f}  autre graine "
                      f"{r['other_seed']:.4f}  delta {r['mean_delta']:+.4f} +-{r['sem']:.4f}  "
                      f"{r['n_pos']}+/{r['n_neg']}-  p = {r['sign_p']:.4f}")
    summary["instance_test"] = inst

    # ---- REGLE DE DECISION PRE-ENREGISTREE -----------------------------------------------
    print(f"\n================ REGLE DE DECISION (registre, addendum 2026-08-03) "
          f"================")
    verdicts = {}
    for a, lo, hi in (("noiseanchor", 0.02, 0.08), ("wronganchor", 0.00, 0.06)):
        r = summary["variants"]["full"]["paired_vs_memzero"].get(a)
        if not r:
            continue
        frac = r["frac_of_headline"]
        in_pred = lo - 1e-9 <= r["mean_delta"] <= hi + 1e-9
        verdicts[a] = {"mean_delta": r["mean_delta"], "frac_of_headline": frac,
                       "prediction": [lo, hi], "inside_prediction": in_pred,
                       "breaks_claim": frac >= 0.5}
        print(f"  {a:14s} delta {r['mean_delta']:+.4f} = {100 * frac:.1f} % du claim "
              f"(+{HEADLINE['full']:.4f})  | prediction [{lo:+.2f}, {hi:+.2f}] -> "
              f"{'DEDANS' if in_pred else 'DEHORS'} | seuil 50 % -> "
              f"{'CLAIM TOMBE' if frac >= 0.5 else 'attaque paree'}")
    summary["decision"] = verdicts
    broke = [a for a, v in verdicts.items() if v["breaks_claim"]]
    _verdict = (f"CLAIM REFORMULE (recence generique) -- {broke}" if broke
                else "le claim CONTENU tient, attaque n1 paree sur les 2 controles")
    summary["verdict"] = _verdict
    print(f"\n  VERDICT : {_verdict}")

    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "summary_controls.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[rescore] wrote {os.path.join(OUT, 'summary_controls.json')}")
    return summary


def contamination(triples, seeds, device):
    """TEST DE CONTAMINATION : le contenu de la MAUVAISE ancre laisse-t-il une trace ?

    Pour chaque cellule on mesure le retour de chaque bras contre le A* PROPRE DU TRIPLE SOURCE
    (celui dont wronganchor a splice l'ancre, meme seed). Si le contenu splice etait lu, le
    retour de `wronganchor` devrait ressembler au sujet SOURCE plus que celui des autres bras.
    S'il n'y a AUCUN exces, le contenu n'est pas lu du tout : le splice ne sert qu'a EFFACER."""
    import statistics as st
    import torch                                                        # noqa: F401
    from torchvision.io import read_video
    from run_reid_probe import dino_embed, sample_seg, center_crop, seg_cos, sign_test_p
    import scenarios as S

    w_rev = S.score_segments(SCEN, 240)[S.revisit_segments(SCEN)[0]]
    w_clean = S.score_segments("S3", 300)[0]
    cache = {}

    def emb(path, lo, hi):
        if (path, lo, hi) not in cache:
            v, _, _ = read_video(path, pts_unit="sec", output_format="THWC")
            seg = sample_seg(v.permute(0, 3, 1, 2).float() / 255., lo, hi, 8)
            cache[(path, lo, hi)] = (dino_embed(seg, device), dino_embed(center_crop(seg), device))
        return cache[(path, lo, hi)]

    rows = {a: [] for a in ARMS}
    per_cell = {}
    for ti in triples:
        for si in seeds:
            src_t = S.wrong_anchor_triple(ti)
            src = emb(f"{VID}/S3_t{src_t}_s{si}_memzero.mp4", *w_clean)
            per_cell[f"t{ti}_s{si}"] = {"source_triple": src_t}
            for a in ARMS:
                p = f"{VID}/{SCEN}_t{ti}_s{si}_{a}.mp4"
                if not os.path.exists(p):
                    continue
                c = seg_cos(emb(p, *w_rev)[0], src[0])
                rows[a].append(c)
                per_cell[f"t{ti}_s{si}"][a] = c
            print(f"[contam] t{ti}_s{si} (source t{src_t}) done", flush=True)
    print(f"\n================ CONTAMINATION : retour vs A* PROPRE DU TRIPLE SOURCE "
          f"================")
    out = {"per_cell": per_cell, "means": {}}
    for a in ARMS:
        if rows[a]:
            out["means"][a] = st.mean(rows[a])
            print(f"    {a:14s} cos(retour, A* du sujet SOURCE) = {st.mean(rows[a]):.4f}")
    if rows["wronganchor"] and rows["memdec20"]:
        d = [x - y for x, y in zip(rows["wronganchor"], rows["memdec20"])]
        npos = sum(1 for x in d if x > 0)
        out["excess_vs_memdec20"] = {"mean": st.mean(d), "n_pos": npos, "n_neg": len(d) - npos,
                                     "sign_p": sign_test_p(npos, len(d) - npos)}
        print(f"    EXCES de ressemblance au sujet SOURCE, wronganchor - memdec20 : "
              f"{st.mean(d):+.4f}  {npos}+/{len(d) - npos}-  p = "
              f"{sign_test_p(npos, len(d) - npos):.4f}")
    with open(os.path.join(OUT, "contamination.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(f"[rescore] wrote {os.path.join(OUT, 'contamination.json')}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["measure", "aggregate", "contam"], default="measure")
    ap.add_argument("--triples", type=str, default="0,1,2,3,4,5,6,7,8,9,10,11")
    ap.add_argument("--seeds", type=str, default="0,1")
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    if args.mode == "aggregate":
        aggregate()
        return
    import torch
    torch.set_grad_enabled(False)
    triples = [int(x) for x in args.triples.split(",") if x != ""]
    seeds = [int(x) for x in args.seeds.split(",") if x != ""]
    if args.mode == "contam":
        contamination(triples, seeds, torch.device("cuda"))
        return
    res = measure(triples, seeds, torch.device("cuda"))
    tag = "-".join(str(t) for t in triples)
    with open(_shard_path(tag), "w") as f:
        json.dump(res, f, indent=1)
    print(f"[rescore] wrote {_shard_path(tag)}")


if __name__ == "__main__":
    main()
