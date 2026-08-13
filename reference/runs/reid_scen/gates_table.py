#!/usr/bin/env python
# ============================================================================================
# Assemble les JSON de score_comp.py en tableaux MARKDOWN pour reference/composite_gates_report.md
# (moyennes sur les cellules + delta vs memzero + fraction du chemin memzero -> oracle1).
#
#   python gates_table.py --tags SS2gates_g1,S2gates_g1 --arms memzero,oracle1,oracle1add1,...
# ============================================================================================
import argparse
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results", "comp")
KEYS = [("subject_crop60", "sujet crop60", 4), ("bg_cont_head", "fond head", 4),
        ("bg_cont_win", "fond win", 4), ("cut2_ratio", "cut2", 2),
        ("dd_ell_head", "DD ellipse", 4), ("clip_text", "CLIP texte", 4)]


def mean(xs):
    xs = [x for x in xs]
    return sum(xs) / len(xs) if xs else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", required=True, help="liste de out-tag (fichiers comp_<tag>.json)")
    ap.add_argument("--arms", required=True)
    ap.add_argument("--ref", default="memzero")
    ap.add_argument("--ceiling", default="oracle1")
    args = ap.parse_args()
    for tag in args.tags.split(","):
        p = os.path.join(OUT, f"comp_{tag}.json")
        if not os.path.exists(p):
            print(f"\n### {tag}: JSON absent ({p})")
            continue
        d = json.load(open(p))
        cells = d["cells"]
        arms = [a for a in args.arms.split(",") if any(a in c for c in cells.values())]
        print(f"\n### {tag}  ({d['scen']}, {len(cells)} cellules: {', '.join(sorted(cells))})\n")
        print("| bras | " + " | ".join(k[1] for k in KEYS) + " | n |")
        print("|" + "---|" * (len(KEYS) + 2))
        M = {}
        for a in arms:
            got = [c[a] for c in cells.values() if a in c]
            M[a] = {k: mean(g[k] for g in got) for k, _, _ in KEYS}
            print(f"| {a} | " + " | ".join(f"{M[a][k]:.{p_}f}" for k, _, p_ in KEYS)
                  + f" | {len(got)} |")
        r, c = args.ref, args.ceiling
        if r in M and c in M:
            gap = M[c]["subject_crop60"] - M[r]["subject_crop60"]
            print(f"\nChemin {r} -> {c} sur le sujet : {M[r]['subject_crop60']:.4f} -> "
                  f"{M[c]['subject_crop60']:.4f}  (ecart {gap:+.4f}, moitie du chemin = "
                  f"{M[r]['subject_crop60'] + gap / 2:.4f})\n")
            print("| bras | delta sujet vs " + r + " | fraction du chemin | signes (par cellule) |")
            print("|---|---|---|---|")
            for a in arms:
                if a in (r,):
                    continue
                ds = [cells[k][a]["subject_crop60"] - cells[k][r]["subject_crop60"]
                      for k in cells if a in cells[k] and r in cells[k]]
                if not ds:
                    continue
                frac = mean(ds) / gap if abs(gap) > 1e-9 else float("nan")
                print(f"| {a} | {mean(ds):+.4f} | {frac:+.0%} | "
                      f"{sum(1 for x in ds if x > 0)}+/{sum(1 for x in ds if x < 0)}- |")


if __name__ == "__main__":
    main()
