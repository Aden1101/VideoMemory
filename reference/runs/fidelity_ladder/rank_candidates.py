#!/usr/bin/env python
# ============================================================================================
# SUPPLEMENT (3 nouvelles scenes, demande d'Aden 12/08) -- classement TRACABLE des candidats.
#
# Candidats = paires FORTES v2 (ADEN_V2_STRONG_19) NON utilisees par les scenarios 01-10.
# Statistiques DEJA CALCULEES (probe1/masks/{main,twin}_<cidx>__A.npz, 10 frames par video) :
#   area   = aire moyenne du masque sujet (fraction de l'image), frames valides
#   det    = score moyen du detecteur
#   valid  = nb de frames ou le suivi tient (sur 10)
#   sam    = IoU SAM moyen (stabilite du masque)
# Regle de classement (scellee ici) :
#   1. visibilite = min(main, twin) de l'aire moyenne  -- le plus faible des deux limite le
#      jugement (le sujet du rollout ET la frame jumelle inseree doivent etre visibles) ;
#   2. departage : det puis valid (min des deux videos) ;
#   3. la preference "sujets a forte variabilite d'apparence" (animaux / personnes, pas
#      d'objets inertes) est appliquee APRES coup et declaree : les objets sont classes mais
#      marques inertes.
# Sortie : classement complet imprime + refs JSON pour le rapport.
# ============================================================================================
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import fidelity_common as C1                                             # noqa: E402

MASKS = "/shared/ssd_14T_project/adenelmi/probe1/masks"
USED_BOARDS = {sc["board_no"] for sc in C1.SCENARIOS}
STRONG = (5, 6, 9, 11, 12, 13, 14, 15, 16, 17, 18, 22, 23, 24, 25, 26, 27, 28, 29)
CATEGORY = {9: ("nurse", "person", False), 16: ("owl", "bird", False),
            17: ("rabbit", "animal", False), 18: ("monkey", "animal", False),
            22: ("lamp", "object", True), 24: ("kettle", "object", True),
            25: ("suitcase", "object", True), 26: ("umbrella", "object", True),
            29: ("bus", "vehicle", True)}


def stats(cidx, kind):
    d = np.load(os.path.join(MASKS, f"{kind}_{cidx}__A.npz"), allow_pickle=True)
    v = d["valid"].astype(bool)
    return {"area": float(d["area_frac"][v].mean()) if v.any() else 0.0,
            "det": float(d["det_scores"][v].mean()) if v.any() else 0.0,
            "valid": int(v.sum()), "sam": float(d["sam_iou"][v].mean()) if v.any() else 0.0}


def main():
    cands = []
    for b in STRONG:
        if b in USED_BOARDS:
            continue
        cidx = 9000000 + b - 1
        head, cat, inert = CATEGORY[b]
        m, t = stats(cidx, "main"), stats(cidx, "twin")
        cands.append({"board_no": b, "cidx": cidx, "subject": head, "category": cat,
                      "inert": inert, "main": m, "twin": t,
                      "vis_min_area": min(m["area"], t["area"]),
                      "det_min": min(m["det"], t["det"]),
                      "valid_min": min(m["valid"], t["valid"])})
    cands.sort(key=lambda c: (-c["vis_min_area"], -c["det_min"], -c["valid_min"]))
    print(f"{'rang':<5}{'board':<7}{'sujet':<10}{'cat':<9}{'inerte':<8}"
          f"{'aire min':<10}{'aire m/t':<15}{'det min':<9}{'valid m/t':<10}{'sam m/t'}")
    for i, c in enumerate(cands, 1):
        print(f"{i:<5}{c['board_no']:<7}{c['subject']:<10}{c['category']:<9}"
              f"{'OUI' if c['inert'] else 'non':<8}{c['vis_min_area']:<10.4f}"
              f"{c['main']['area']:.4f}/{c['twin']['area']:.4f}   "
              f"{c['det_min']:<9.3f}{c['main']['valid']}/{c['twin']['valid']:<8}"
              f"{c['main']['sam']:.3f}/{c['twin']['sam']:.3f}")
    chosen = [c for c in cands if not c["inert"]][:4]
    print("\nCHOIX (4 meilleurs NON inertes, regle scellee) : "
          + ", ".join(f"board{c['board_no']:02d} {c['subject']}" for c in chosen))
    out = {"rule": "tri par min(main,twin) aire ; departage det puis valid ; inertes exclus "
                   "du choix final (declares au classement)",
           "ranking": cands, "chosen_boards": [c["board_no"] for c in chosen]}
    dst = os.path.join(HERE, "supplement_ranking.json")
    with open(dst, "w") as f:
        json.dump(out, f, indent=2)
    print(f"-> {dst}")


if __name__ == "__main__":
    main()
