#!/usr/bin/env python
# ============================================================================================
# RESUME DE L'ANNEXE 10 GRAINES (annex_seeds10.sh). Rend, pour chaque bras, la distribution
# COMPLETE de strong19 sur les 10 graines : les 10 valeurs, mediane, min, max, ecart max-min,
# et le rappel du run 3 graines SCELLE (qui reste la reference, l'annexe l'accompagne).
# Aucun calcul de sondage ici : lecture seule.
# Usage : python reference/runs/probe1/summarize_annex.py
# ============================================================================================
import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import probe1_common as C  # noqa: E402

ARMS = [("A0 state", "state"), ("lin_delta_all", "lin_delta_all"),
        ("A3 mlp_full_muon (PRIMARY)", "mlp_full_muon")]


def binom_ge(k, n=19, p=0.5):
    k = math.ceil(k - 1e-9)
    return sum(math.comb(n, i) for i in range(k, n + 1)) * (p ** n)


def med(xs):
    s = sorted(xs)
    n = len(s)
    return s[n // 2] if n % 2 else 0.5 * (s[n // 2 - 1] + s[n // 2])


def main():
    for label, src in ARMS:
        pa = os.path.join(C.RUNS_DIR, f"annex_seeds10_{src}_subject_pool_v2.json")
        ps = os.path.join(C.RUNS_DIR, f"cv_{src}_subject_pool_v2.json")
        if not os.path.exists(pa):
            print(f"{label:<28} annexe pas encore deposee ({os.path.basename(pa)})")
            continue
        d = json.load(open(pa))
        v = [sd["subsets"]["strong19"]["n_pairs_correct"] for sd in d["per_seed"]]
        s = d["summary"]["strong19"]
        print(f"{label}  ({len(v)} graines, {d['minutes']:.0f} min)")
        print(f"   valeurs   {v}")
        print(f"   mediane {med(v):.1f}   min {min(v)}   max {max(v)}   etendue {max(v) - min(v)}"
              f"   moyenne {s['n_pairs_correct']:.2f}/19 (binom p {binom_ge(s['n_pairs_correct']):.4f})")
        if os.path.exists(ps):
            r = json.load(open(ps))
            rv = [sd["subsets"]["strong19"]["n_pairs_correct"] for sd in r["per_seed"]]
            print(f"   rappel run SCELLE 3 graines : {rv} -> "
                  f"{r['summary']['strong19']['n_pairs_correct']:.2f}/19 (reference, inchange)")
        print(f"   json {pa}")
    print("SUMMARIZE_ANNEX_DONE")


if __name__ == "__main__":
    main()
