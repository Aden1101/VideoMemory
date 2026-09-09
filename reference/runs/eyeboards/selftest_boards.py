#!/usr/bin/env python
# ============================================================================================
# SELFTESTS des planches -- CPU pur, aucun media reel, ~1 s. A passer AVANT de generer une
# campagne : une planche fausse coute une soiree de generation et, pire, un jugement de l'oeil
# qu'on ne peut pas redemander a l'identique.
#   python reference/runs/eyeboards/selftest_boards.py
# ============================================================================================
import json
import os
import shutil
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import boards  # noqa: E402
from boards import CODE_VERSION, build_duel2, build_pair2, build_rank3, perm  # noqa: E402
from score_boards import parse_code, score, wilson  # noqa: E402

OK = []


def check(name, cond, detail=""):
    assert cond, f"FAIL {name} {detail}"
    OK.append(name)
    print(f"  ok  {name}")


TMP = tempfile.mkdtemp(prefix="eyeboards_selftest_")
try:
    r_pair, r_rank, r_duel = boards._demo(os.path.join(TMP, "demo"))

    def load(p):
        with open(p, encoding="utf8") as f:
            return json.load(f)

    def page(p):
        with open(p, encoding="utf8") as f:
            return f.read()

    k_pair, k_rank, k_duel = (load(r["key"]) for r in (r_pair, r_rank, r_duel))

    # --- B1: la permutation est deterministe et depend de la question --------------------------
    check("B1 perm reproductible", perm("b", "s01", 3) == perm("b", "s01", 3))
    check("B1b perm change avec la question", perm("b", "s01", 3) != perm("b", "s02", 3)
          or perm("b", "s01", 3) != perm("b", "s03", 3))
    check("B1c perm est une vraie permutation",
          all(sorted(perm("b", f"s{i}", 3)) == [0, 1, 2] for i in range(50)))
    counts = {}
    for i in range(400):                       # elle doit aussi etre EQUILIBREE : un melange
        counts[perm("b", f"s{i}", 2)[0]] = counts.get(perm("b", f"s{i}", 2)[0], 0) + 1
    check("B1d perm equilibree sur 2 slots", 150 < counts.get(0, 0) < 250, f"{counts}")

    # --- B2: la cle ne fuit JAMAIS dans la page ------------------------------------------------
    # LE test qui compte. Une planche qui contient le nom du bras ou le chemin du fichier
    # n'est pas un jugement en aveugle, c'est une mesure perdue.
    for r, forbidden in ((r_duel, ["memon", "memoff"]),
                         (r_rank, ["v1_ancien", "v2_mixer", "v2_mixer_sharp"])):
        h = page(r["html"])
        leaked = [w for w in forbidden if w in h]
        check(f"B2 aucun nom de bras dans {os.path.basename(r['html'])}", not leaked, f"{leaked}")
    for r in (r_pair, r_rank, r_duel):
        h = page(r["html"])
        check(f"B2b aucun chemin de fichier dans {os.path.basename(r['html'])}",
              "/media/" not in h and ".png\"" not in h and TMP not in h)
        # 'true' seul apparait legitimement dans aria-pressed="true" : on cherche les
        # noms de champs de la CLE, qui eux ne doivent jamais traverser.
        check(f"B2c aucun champ de la cle dans {os.path.basename(r['html'])}",
              "true_side" not in h and "twin" not in h and "variant" not in h
              and "sides" not in h)

    # --- B3: le corrige est reellement reparti des deux cotes ----------------------------------
    sides = [it["true_side"] for it in k_pair["items"]]
    check("B3 le corrige n'est pas constant", len(set(sides)) == 2 or len(sides) < 4, f"{sides}")

    # --- B4: aller-retour code -> score, avec un juge PARFAIT ----------------------------------
    perfect = "".join(it["true_side"] for it in sorted(k_pair["items"], key=lambda x: x["slot"]))
    code = f"{k_pair['board_id']}~{CODE_VERSION}~{perfect}~{k_pair['checksum']}"
    res = score(k_pair, code)
    check("B4 juge parfait -> 1.0", res["acc"] == 1.0 and res["n_unsure"] == 0)
    inverse = "".join({"g": "d", "d": "g"}[c] for c in perfect)
    res0 = score(k_pair, f"{k_pair['board_id']}~{CODE_VERSION}~{inverse}~{k_pair['checksum']}")
    check("B4b juge systematiquement faux -> 0.0", res0["acc"] == 0.0)

    # --- B5: "impossible a dire" est compte a part, pas dilue ----------------------------------
    mixed = "?" + perfect[1:]
    res1 = score(k_pair, f"{k_pair['board_id']}~{CODE_VERSION}~{mixed}~{k_pair['checksum']}")
    check("B5 les doutes sortent du denominateur",
          res1["n_unsure"] == 1 and res1["n_decided"] == len(perfect) - 1 and res1["acc"] == 1.0)
    allq = "?" * len(perfect)
    res2 = score(k_pair, f"{k_pair['board_id']}~{CODE_VERSION}~{allq}~{k_pair['checksum']}")
    check("B5b tout en doute -> pas de score, pas un zero", res2["acc"] is None)

    # --- B6: les codes invalides sont refuses bruyamment ---------------------------------------
    for bad, why in ((f"{k_pair['board_id']}~{CODE_VERSION}~{perfect}~deadbe", "somme fausse"),
                     (f"autre~{CODE_VERSION}~{perfect}~{k_pair['checksum']}", "mauvaise planche"),
                     (f"{k_pair['board_id']}~{CODE_VERSION}~{perfect[:-1]}~{k_pair['checksum']}",
                      "code tronque"),
                     (f"{k_pair['board_id']}~{CODE_VERSION}~{'x'*len(perfect)}~"
                      f"{k_pair['checksum']}", "reponse hors options"),
                     ("nimportequoi", "format")):
        try:
            parse_code(bad, k_pair)
            raise SystemExit(f"FAIL B6: '{why}' a ete accepte")
        except AssertionError:
            pass
    check("B6 les 5 codes invalides sont refuses", True)

    # --- B7: duel2 compte les victoires par BRAS, pas par cote --------------------------------
    # Catches: le bug qui rendrait la permutation inutile (compter G/D au lieu du bras).
    always_g = "g" * k_duel["n_slots"]
    rd = score(k_duel, f"{k_duel['board_id']}~{CODE_VERSION}~{always_g}~{k_duel['checksum']}")
    left_arms = [it["sides"]["g"] for it in k_duel["items"]]
    expect = {a: left_arms.count(a) for a in set(left_arms)}
    check("B7 victoires attribuees au bras affiche a gauche",
          all(rd["wins"][a] == n for a, n in expect.items()), f"{rd['wins']} vs {expect}")
    check("B7b les deux bras apparaissent, meme a zero", len(rd["arms"]) == 2, f"{rd['arms']}")
    check("B7c le total des victoires = les cellules tranchees",
          sum(rd["wins"].values()) == rd["n_decided"])

    # --- B8: rank3 -- rang 1 = meilleur, duels coherents ---------------------------------------
    # On fabrique un juge qui met TOUJOURS la variante 'v2_mixer_sharp' en tete.
    payload = ["."] * k_rank["n_slots"]
    for it in k_rank["items"]:
        order = sorted(it["letters"].items(),
                       key=lambda kv: (kv[1]["variant"] != "v2_mixer_sharp", kv[1]["variant"]))
        for rank, (_letter, meta) in enumerate(order, start=1):
            payload[meta["rank_slot"]] = str(rank)
            payload[meta["judge_slot"]] = "o" if rank == 1 else "n"
    rr = score(k_rank, f"{k_rank['board_id']}~{CODE_VERSION}~{''.join(payload)}~"
                       f"{k_rank['checksum']}")
    check("B8 la variante preferee a le meilleur rang moyen",
          min(rr["mean_rank"], key=rr["mean_rank"].get) == "v2_mixer_sharp", f"{rr['mean_rank']}")
    check("B8b elle vaut exactement 1.0", abs(rr["mean_rank"]["v2_mixer_sharp"] - 1.0) < 1e-9)
    check("B8c elle gagne tous ses duels",
          all(d["v2_mixer_sharp"] == len(k_rank["items"])
              for name, d in rr["duels"].items() if "v2_mixer_sharp" in d), f"{rr['duels']}")
    check("B8d le jugement absolu est ventile",
          rr["judge"]["v2_mixer_sharp"]["o"] == len(k_rank["items"]))

    # --- B9: Wilson reste dans [0,1] aux bords ------------------------------------------------
    lo, hi = wilson(0, 5)
    check("B9 Wilson borne en bas", lo >= 0.0 and hi <= 1.0, f"{lo},{hi}")
    lo, hi = wilson(5, 5)
    check("B9b Wilson borne en haut", lo >= 0.0 and hi <= 1.0, f"{lo},{hi}")

    # --- B10: le plafond Artifact est verifie a la construction --------------------------------
    old = boards.ARTIFACT_MAX_BYTES
    boards.ARTIFACT_MAX_BYTES = 100
    try:
        build_duel2([{"id": "s0", "arms": {"a": r_duel["key"], "b": r_duel["key"]}}],
                    os.path.join(TMP, "toobig"))
        raise SystemExit("FAIL B10: une planche au-dessus du plafond a ete acceptee")
    except AssertionError as e:
        check("B10 le plafond 16 Mo est applique", "plafond" in str(e) or "Mo" in str(e))
    finally:
        boards.ARTIFACT_MAX_BYTES = old

    # --- B11: la page est autonome -------------------------------------------------------------
    for r in (r_pair, r_rank, r_duel):
        h = page(r["html"])
        check(f"B11 aucune ressource externe dans {os.path.basename(r['html'])}",
              "http://" not in h and "https://" not in h)

    print(f"\n{len(OK)}/{len(OK)} selftests planches PASS")
finally:
    shutil.rmtree(TMP, ignore_errors=True)
