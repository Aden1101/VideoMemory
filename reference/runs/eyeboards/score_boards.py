#!/usr/bin/env python
# ============================================================================================
# PLANCHES EN AVEUGLE -- scoreur. Prend le code renvoye par le juge + la cle scellee, et rend
# le tableau. Aucun torch, aucun GPU, aucun media : il ne lit que du texte.
#
#   python score_boards.py --key key_duel2.json --code "duel2~1~gdg...~a1b2c3"
#   python score_boards.py --key key_pair2.json --code_file reponses.txt
#
# CE QU'IL REFUSE DE FAIRE, et pourquoi (chaque garde-fou vient d'une erreur reelle du dossier)
#  - Il ne dilue JAMAIS les "impossible a dire" dans le hasard : il les compte a part et
#    calcule le score sur les seules cellules tranchees, en affichant les deux effectifs. Un
#    taux de 60 % sur 10 cellules tranchees sur 30 n'est pas le meme resultat qu'un 60 % sur 30.
#  - Il ne rend pas un p-value sans l'effectif ni l'intervalle : l'annexe 10 graines du
#    registre a montre qu'un point unique se presente toujours comme plus solide qu'il n'est.
#  - Il verifie la somme de controle : un code colle depuis la mauvaise planche est une erreur
#    bruyante, pas un score plausible.
# ============================================================================================
import argparse
import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from boards import CODE_VERSION, code_checksum  # noqa: E402

UNSURE = "?"
BLANK = "."


# ----------------------------- statistiques ---------------------------------------------------
def binom_p_two_sided(k, n, p0=0.5):
    """p exact bilateral. Pur python (pas de scipy sur la machine d'audit)."""
    if n == 0:
        return 1.0
    def pmf(i):
        return math.comb(n, i) * (p0 ** i) * ((1.0 - p0) ** (n - i))
    obs = pmf(k)
    return min(1.0, sum(pmf(i) for i in range(n + 1) if pmf(i) <= obs * (1.0 + 1e-12)))


def wilson(k, n, z=1.96):
    """Intervalle de Wilson a 95 % : honnete a petit n, contrairement a l'intervalle normal
    qui deborde de [0,1] des que le taux approche les bords -- et nos n sont toujours petits."""
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    r = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    # borne dure : aux extremes (k=0 ou k=n) l'arithmetique flottante sort de [0,1] d'un
    # 1e-17, et un intervalle affiche a -0.00 fait douter du reste du tableau.
    return (min(1.0, max(0.0, (c - r) / d)), min(1.0, max(0.0, (c + r) / d)))


# ----------------------------- lecture du code ------------------------------------------------
def parse_code(code, key):
    """`<board_id>~<version>~<payload>~<checksum>` -> liste de reponses, une par case."""
    code = code.strip().replace("\n", "").replace(" ", "")
    parts = code.split("~")
    assert len(parts) == 4, (
        f"code malforme : attendu 4 champs separes par ~, recu {len(parts)}. "
        f"Colle la ligne complete produite par la planche.")
    bid, ver, payload, chk = parts
    assert bid == key["board_id"], (
        f"ce code vient de la planche '{bid}', la cle est celle de '{key['board_id']}'")
    assert ver == CODE_VERSION, f"code version {ver}, scoreur version {CODE_VERSION}"
    assert chk == key["checksum"] == code_checksum(bid, key["kind"]), \
        "somme de controle invalide : code tronque, ou cle qui ne correspond pas a la planche"
    assert len(payload) == key["n_slots"], (
        f"{len(payload)} reponses pour {key['n_slots']} cases : code incomplet")
    allowed = {s["slot"]: set(s["answers"]) | {BLANK} for s in key["slots"]}
    for i, ch in enumerate(payload):
        assert ch in allowed[i], f"case {i} : reponse '{ch}' hors de {sorted(allowed[i])}"
    return list(payload)


# ----------------------------- les trois lectures ----------------------------------------------
def score_pair2(ans, key):
    """Un corrige existe : le vrai sujet est d'un cote connu. On rend le taux de reussite sur
    les cellules TRANCHEES, plus le detail par source (decodeur, flou leger, flou fort...) si
    la meta le porte -- c'est cette ventilation qui place la sortie du decodeur sur l'echelle
    de flou, la question meme de la planche du 22/08."""
    per_src, rows = {}, []
    hit = seen = unsure = 0
    for it in key["items"]:
        a = ans[it["slot"]]
        src = it.get("meta", {}).get("source", "(sans source)")
        d = per_src.setdefault(src, {"hit": 0, "n": 0, "unsure": 0})
        if a in (UNSURE, BLANK):
            unsure += 1
            d["unsure"] += 1
            rows.append((it["id"], src, "impossible"))
            continue
        ok = (a == it["true_side"])
        hit += ok; seen += 1
        d["hit"] += ok; d["n"] += 1
        rows.append((it["id"], src, "juste" if ok else "faux"))
    out = {"kind": "pair2", "n_total": len(key["items"]), "n_decided": seen,
           "n_unsure": unsure, "hits": hit,
           "acc": hit / seen if seen else None,
           "p": binom_p_two_sided(hit, seen),
           "ci": wilson(hit, seen), "rows": rows,
           "by_source": {k: {**v, "acc": (v["hit"] / v["n"]) if v["n"] else None,
                             "p": binom_p_two_sided(v["hit"], v["n"])}
                         for k, v in sorted(per_src.items())}}
    return out


def score_duel2(ans, key):
    """Pas de corrige : c'est une PREFERENCE entre deux bras. On compte les victoires par bras
    et on les teste contre le pile ou face. Le bras gagnant n'est pas 'le bon', c'est celui que
    l'oeil designe plus souvent -- et sans les deux effectifs la phrase n'a pas de sens."""
    wins, unsure, rows = {}, 0, []
    for it in key["items"]:
        a = ans[it["slot"]]
        if a in (UNSURE, BLANK):
            unsure += 1
            rows.append((it["id"], "impossible"))
            continue
        arm = it["sides"][a]
        wins[arm] = wins.get(arm, 0) + 1
        rows.append((it["id"], arm))
    for it in key["items"]:                       # tout bras jamais choisi doit apparaitre a 0
        for arm in it["sides"].values():
            wins.setdefault(arm, 0)
    seen = sum(wins.values())
    arms = sorted(wins)
    k = wins[arms[0]] if arms else 0
    return {"kind": "duel2", "n_total": len(key["items"]), "n_decided": seen,
            "n_unsure": unsure, "wins": wins, "arms": arms,
            "rate": {a: (wins[a] / seen if seen else None) for a in arms},
            "p": binom_p_two_sided(k, seen),
            "ci_first_arm": wilson(k, seen), "rows": rows}


def score_rank3(ans, key):
    """Deux lectures, comme la planche les demandait : le RANG moyen par variante (comparaison
    relative, qui survit meme si aucune variante ne convainc) et le jugement absolu."""
    ranks, judge, duels = {}, {}, {}
    for it in key["items"]:
        seen_here = {}
        for letter, meta in it["letters"].items():
            v = meta["variant"]
            r, j = ans[meta["rank_slot"]], ans[meta["judge_slot"]]
            if r != BLANK:
                ranks.setdefault(v, []).append(int(r))
                seen_here[v] = int(r)
            if j != BLANK:
                d = judge.setdefault(v, {"o": 0, "a": 0, "n": 0})
                d[j] += 1
        names = sorted(seen_here)
        for i in range(len(names)):
            for j2 in range(i + 1, len(names)):
                a_, b_ = names[i], names[j2]
                if seen_here[a_] == seen_here[b_]:
                    continue                       # ex aequo : ne compte pour personne
                w = a_ if seen_here[a_] < seen_here[b_] else b_   # rang 1 = meilleur
                pair = tuple(sorted((a_, b_)))
                d = duels.setdefault(pair, {pair[0]: 0, pair[1]: 0})
                d[w] += 1
    return {"kind": "rank3", "n_items": len(key["items"]),
            "mean_rank": {v: sum(r) / len(r) for v, r in sorted(ranks.items())},
            "n_ranked": {v: len(r) for v, r in sorted(ranks.items())},
            "judge": {v: d for v, d in sorted(judge.items())},
            "duels": {f"{a} vs {b}": d for (a, b), d in sorted(duels.items())}}


SCORERS = {"pair2": score_pair2, "duel2": score_duel2, "rank3": score_rank3}


def score(key, code):
    ans = parse_code(code, key)
    return SCORERS[key["kind"]](ans, key)


# ----------------------------- rendu -----------------------------------------------------------
def render(res):
    L = [f"planche {res['kind']}"]
    if res["kind"] == "pair2":
        L += [f"  tranchees {res['n_decided']}/{res['n_total']}  "
              f"(impossible a dire : {res['n_unsure']})"]
        if res["acc"] is None:
            L += ["  AUCUNE cellule tranchee : pas de score."]
        else:
            lo, hi = res["ci"]
            L += [f"  juste {res['hits']}/{res['n_decided']} = {res['acc']:.3f}  "
                  f"IC95 [{lo:.2f}, {hi:.2f}]  p={res['p']:.3g}  (hasard 0.5)"]
        for src, d in res["by_source"].items():
            a = f"{d['acc']:.3f}" if d["acc"] is not None else "n/a"
            L += [f"    {src:<16} {d['hit']}/{d['n']} = {a}  p={d['p']:.3g}  "
                  f"(impossible {d['unsure']})"]
    elif res["kind"] == "duel2":
        L += [f"  tranchees {res['n_decided']}/{res['n_total']}  "
              f"(impossible a dire : {res['n_unsure']})"]
        for a in res["arms"]:
            r = res["rate"][a]
            L += [f"    {a:<16} {res['wins'][a]} victoires" +
                  (f" = {r:.3f}" if r is not None else "")]
        if res["n_decided"]:
            lo, hi = res["ci_first_arm"]
            L += [f"  p={res['p']:.3g} contre le pile ou face  "
                  f"(IC95 de {res['arms'][0]} [{lo:.2f}, {hi:.2f}])"]
    else:
        L += [f"  {res['n_items']} lignes"]
        L += ["  rang moyen (1 = meilleur) :"]
        for v, r in sorted(res["mean_rank"].items(), key=lambda kv: kv[1]):
            L += [f"    {v:<20} {r:.2f}   (n={res['n_ranked'][v]})"]
        L += ["  jugement absolu :"]
        for v, d in res["judge"].items():
            L += [f"    {v:<20} oui {d['o']}  a peu pres {d['a']}  non {d['n']}"]
        if res["duels"]:
            L += ["  duels de rang :"]
            for name, d in res["duels"].items():
                L += [f"    {name:<34} " + "  ".join(f"{k} {v}" for k, v in d.items())]
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser(description="scoreur des planches en aveugle")
    ap.add_argument("--key", required=True)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--code")
    g.add_argument("--code_file")
    ap.add_argument("--json", action="store_true", help="sortie machine")
    a = ap.parse_args()
    with open(a.key, encoding="utf8") as f:
        key = json.load(f)
    code = a.code if a.code else open(a.code_file, encoding="utf8").read()
    res = score(key, code)
    print(json.dumps(res, indent=1, ensure_ascii=False) if a.json else render(res))


if __name__ == "__main__":
    main()
