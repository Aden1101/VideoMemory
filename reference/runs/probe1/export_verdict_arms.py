#!/usr/bin/env python
# ============================================================================================
# EXPORT DES MARGES PAR PAIRE, TOUS BRAS CONFONDUS (demande du coordinateur, 07/08 : materiau
# de la planche que l'oeil d'Aden verifiera paire par paire).
#
# SORTIE 1 -- runs/verdict_arms_v2.json : EXACTEMENT {bras: {corpus_idx: marge}}, la marge
#   etant MOYENNEE SUR LES 3 GRAINES du sondage (per_seed[*]["pair_margins"]). Rien d'autre au
#   premier niveau : ce fichier se charge et s'itere sans precaution.
# SORTIE 2 -- runs/verdict_arms_v2_meta.json : tout le contexte (scores, p, permute, part
#   ecrite, R, chemin du json source, appartenance aux sous-ensembles strong19/valid25). Il est
#   SEPARE pour que la sortie 1 garde la forme exacte demandee.
#
# AUCUN CALCUL : les bras deja scores sont RELUS depuis leurs json de sondage, jamais recalcules.
# Une marge > 0 sur une paire = le sondeur a mis le bon jumeau devant ; c'est la quantite que la
# planche montre paire par paire.
#
# Usage : python reference/runs/probe1/export_verdict_arms.py [--out <json>]
# ============================================================================================
import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import probe1_common as C  # noqa: E402

# nom de bras (celui de la planche) -> fichier de sondage. Les 4 premiers sont les bras MLP de
# la ronde du 07/08 ; les suivants sont deja scores et ne sont QUE relus.
ARMS = [
    ("A3_mlp_full_muon",      "cv_mlp_full_muon_subject_pool_v2.json"),
    ("A1_mlp_delta_all",      "cv_mlp_delta_all_subject_pool_v2.json"),
    ("A2p_mlp_full",          "cv_mlp_full_subject_pool_v2.json"),
    ("A4_mlp_hebb_seq",       "cv_mlp_hebb_seq_subject_pool_v2.json"),
    ("A0_state",              "cv_state_subject_pool_v2.json"),
    ("lin_delta_all",         "cv_lin_delta_all_subject_pool_v2.json"),
    ("lin_mean_all",          "cv_lin_mean_all_subject_pool_v2.json"),
    ("lin_delta_subject",     "cv_lin_delta_subject_subject_pool_v2.json"),
    ("lin_mean_subject",      "cv_lin_mean_subject_subject_pool_v2.json"),
    ("hdirect_masked",        "cv_hdirect_masked_subject_pool_v2.json"),
    ("hdirect_masked_pre",    "cv_hdirect_masked_subject_pool_pre_v2.json"),
]
# Les deux chiffres AVEUGLES A L'IDENTITE de chaque bras d'ecriture (sweep_mlp_step.py --arms,
# runs/sweep_mlp_arms.json) : R = reduction relative moyenne de ||f_W(k)-v|| par ecriture,
# part ecrite = poids de l'ecriture dans l'etat. Ils sont recopies dans le meta pour que la
# planche puisse les afficher a cote du score ; les bras de LECTURE (hdirect*) n'en ont pas.
BLIND = {"A3_mlp_full_muon": {"R": 0.0251, "written_part": 0.952},
         "A1_mlp_delta_all": {"R": -0.1737, "written_part": 0.728},
         "A2p_mlp_full": {"R": -0.5772, "written_part": 1.034},
         "A4_mlp_hebb_seq": {"R": -4.6205, "written_part": 0.947},
         "A0_state": {"R": -0.3596, "written_part": 0.883}}


def mean_margins(d):
    """Marges par paire moyennees sur les graines. Rend {corpus_idx (str): marge}."""
    seeds = d["per_seed"]
    keys = set(seeds[0]["pair_margins"])
    for s in seeds:
        assert set(s["pair_margins"]) == keys, "les graines ne portent pas les memes paires"
    return {k: sum(float(s["pair_margins"][k]) for s in seeds) / len(seeds) for k in sorted(keys)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(C.RUNS_DIR, "verdict_arms_v2.json"))
    args = ap.parse_args()

    margins, meta, missing = {}, {}, []
    for name, fn in ARMS:
        path = os.path.join(C.RUNS_DIR, fn)
        if not os.path.exists(path):
            missing.append((name, fn))
            continue
        d = json.load(open(path))
        margins[name] = mean_margins(d)
        s = d["summary"]
        meta[name] = {
            "json": path, "source": d["source"], "target": d["target"],
            "features": d["features"], "corpus": d["corpus"], "pretrain": bool(d["pretrain"]),
            "seeds": d["seeds"], "n_folds": d["n_folds"], "minutes": d.get("minutes"),
            "strong19": s["strong19"], "valid25": s["valid25"], "all30": s["all"],
            "permuted": s["permuted"],
            "per_seed_strong19": [sd["subsets"]["strong19"] for sd in d["per_seed"]],
            "blind": BLIND.get(name),
        }
    # l'appartenance aux sous-ensembles est la MEME pour tous les bras (elle vient des labels
    # d'oeil scelles, pas du sondage) : on la sort une fois, verifiee identique partout.
    subsets, ref = None, None
    for name, fn in ARMS:
        p = os.path.join(C.RUNS_DIR, fn)
        if not os.path.exists(p):
            continue
        sub = json.load(open(p))["subsets_corpus_idx"]
        if subsets is None:
            subsets, ref = sub, name
        else:
            assert {k: sorted(v) for k, v in sub.items()} == \
                   {k: sorted(v) for k, v in subsets.items()}, \
                   f"{name} et {ref} n'ont pas les memes sous-ensembles"

    C.save_json(margins, args.out)                     # EXACTEMENT {bras: {idx: marge}}
    meta_path = args.out.replace(".json", "_meta.json")
    C.save_json({"subsets_corpus_idx": subsets, "arms": meta,
                 "note": "marges = moyenne des 3 graines ; > 0 = le bon jumeau est devant ; "
                         "les bras deja scores sont RELUS, jamais recalcules",
                 "missing": [f"{n} ({f})" for n, f in missing]}, meta_path)

    print(f"[export] {len(margins)} bras -> {args.out}")
    for name in margins:
        s = meta[name]["strong19"]
        b = meta[name]["blind"]
        print(f"   {name:<22} {len(margins[name])} paires | strong19 "
              f"{s['n_pairs_correct']:.2f}/19 p={s['wilcoxon_pair_p']:.3g}"
              + (f" | R {b['R']:+.4f} part {b['written_part']:.3f}" if b else ""))
    if missing:
        print("[export] ABSENTS (pas encore sondes) : "
              + ", ".join(f"{n} ({f})" for n, f in missing))
    print(f"[export] contexte -> {meta_path}")
    print("EXPORT_ARMS_DONE")


if __name__ == "__main__":
    main()
