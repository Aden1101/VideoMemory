#!/usr/bin/env python
"""ECHELLE de FIDELITE, selftest CPU (aucun GPU, aucune video) :
  (a) table des scenarios : 10 sid uniques, board_no tous dans ADEN_V2_STRONG_19 (copie
      litterale + verif contre probe1_common si importable), mapping board_no <-> corpus_idx
      verifie CONTRE L'INDEX v2 (ordre lexicographique des video_id = ordre numerique) ;
  (b) prompts de la table = prompts de l'index, mode single 90f, graines main/twin ;
  (c) donneurs : bijection, jamais soi-meme, TOUJOURS inter-categories ;
  (d) interludes : ne contiennent ni le mot-tete du sujet A ni celui du sujet donneur ;
  (e) table des bras : chaque bras remet toutes les cles de l'union OFF (lecon MEM_WRITER),
      `rien` ne splice pas, les 5 autres splicent 1 frame, les flous portent leur sigma ;
  (f) anonymisation : permutation deterministe (2 tirages identiques), bijective par
      scenario, et differente d'un scenario a l'autre (au moins 5 permutations distinctes
      sur 10 : une table constante serait un canal de fuite) ;
  (g) forme du scenario : 81/162 multiples de 3, ancre = 80, 240 frames.
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import fidelity_common as C                                              # noqa: E402

ADEN_V2_STRONG_19 = (5, 6, 9, 11, 12, 13, 14, 15, 16, 17, 18, 22, 23, 24, 25, 26, 27, 28, 29)

SUBJECT_HEAD = {"scenario01": "farmer", "scenario02": "fisherman", "scenario03": "parrot",
                "scenario04": "goat", "scenario05": "dog", "scenario06": "cat",
                "scenario07": "horse", "scenario08": "guitar", "scenario09": "scooter",
                "scenario10": "tractor"}
# familles interdites dans l'interlude si le sujet (A ou donneur) est de cette famille
FAMILY = {"parrot": {"bird", "parrot", "flamingo", "flamingos", "owl", "swan"},
          "farmer": {"man", "woman", "person", "farmer", "fisherman"},
          "fisherman": {"man", "woman", "person", "farmer", "fisherman"}}


def main():
    ok = 0

    # (a) table
    sids = [sc["sid"] for sc in C.SCENARIOS]
    assert len(sids) == 10 and len(set(sids)) == 10
    for sc in C.SCENARIOS:
        assert sc["board_no"] in ADEN_V2_STRONG_19, \
            f"{sc['sid']}: board {sc['board_no']} hors ADEN_V2_STRONG_19"
        assert sc["cidx"] == 9000000 + sc["board_no"] - 1, \
            f"{sc['sid']}: mapping board/cidx casse"
    try:
        sys.path.insert(0, os.path.join(C.REPO, "reference/runs/probe1"))
        from probe1_common import ADEN_V2_STRONG_19 as REF19
        assert tuple(REF19) == ADEN_V2_STRONG_19, "copie litterale STRONG_19 desynchronisee"
        print("[selftest] (a2) copie ADEN_V2_STRONG_19 == probe1_common : OK")
    except ImportError as e:
        print(f"[selftest] (a2) probe1_common non importable ici ({e}), copie non re-verifiee")
    # mapping verifie contre l'index : l'ordre lexicographique des main video_id est
    # l'ordre numerique des corpus_idx (tous "main:90000xx", meme longueur)
    with open(C.V2_INDEX) as f:
        idx = json.load(f)
    mains = sorted({v["video_id"] for v in idx if not v["twin"]})
    assert mains == [f"main:{9000000 + i}" for i in range(30)], \
        "ordre lexicographique != ordre numerique : la numerotation de planche serait fausse"
    ok += 1
    print("[selftest] (a) table scenarios + numerotation de planche : OK")

    # (b) prompts / graines / mode contre l'index
    by_vid = {v["video_id"]: v for v in idx}
    for sc in C.SCENARIOS:
        m, t = by_vid[f"main:{sc['cidx']}"], by_vid[f"twin:{sc['cidx']}"]
        assert m["prompt"] == sc["prompt"] == t["prompt"]
        assert m["mode"] == "single" and int(m["frames"]) == C.V2_FRAMES
        assert int(m["seed"]) == 42 + 500000 + sc["cidx"]
        assert int(t["seed"]) == int(m["seed"]) + 100000
    ok += 1
    print("[selftest] (b) prompts + graines + mode single 90f : OK")

    # (c) donneurs
    donors = [C.donor_of(i) for i in range(len(C.SCENARIOS))]
    assert sorted(donors) == list(range(10)), "donneurs non bijectifs"
    for i, d in enumerate(donors):
        assert d != i, f"scenario {i} serait son propre donneur"
        assert C.SCENARIOS[i]["category"] != C.SCENARIOS[d]["category"], \
            (f"{C.SCENARIOS[i]['sid']} ({C.SCENARIOS[i]['category']}) <- donneur "
             f"{C.SCENARIOS[d]['sid']} ({C.SCENARIOS[d]['category']}) : MEME categorie")
    ok += 1
    print("[selftest] (c) donneurs bijectifs, jamais soi-meme, inter-categories : OK")

    # (d) interludes sans le sujet A ni le sujet donneur
    for i, sc in enumerate(C.SCENARIOS):
        words = set(sc["interlude"].lower().replace(",", " ").split())
        for who in (sc["sid"], C.SCENARIOS[C.donor_of(i)]["sid"]):
            head = SUBJECT_HEAD[who]
            bad = FAMILY.get(head, {head})
            assert not (words & bad), \
                f"{sc['sid']}: interlude '{sc['interlude']}' contient {words & bad}"
    ok += 1
    print("[selftest] (d) interludes sans collision de sujet : OK")

    # (e) table des bras
    assert set(C.ARM_ENV) == set(C.ARMS)
    keys = set(C.OFF)
    for arm, env in C.ARM_ENV.items():
        assert set(env) == keys, f"{arm} ne remet pas toutes les cles de l'union"
        assert env["MEM_ZERO_ALL"] == "1", f"{arm}: la memoire doit etre muette"
    assert C.ARM_ENV["rien"]["ORACLE_SPLICE"] == "0"
    for arm in C.ARMS:
        if arm != "rien":
            assert C.ARM_ENV[arm]["ORACLE_SPLICE"] == "1"
            assert C.ARM_ENV[arm]["ORACLE_SPLICE_FRAMES"] == "1"
    assert float(C.ARM_ENV["flou_leger"]["ORACLE_BLUR_SIGMA"]) == C.BLUR_SIGMA_LIGHT
    assert float(C.ARM_ENV["flou_fort"]["ORACLE_BLUR_SIGMA"]) == C.BLUR_SIGMA_STRONG
    assert C.ARM_ENV["oracle"]["ORACLE_BLUR_SIGMA"] == ""
    for arm in ("oracle", "flou_leger", "flou_fort", "rien"):
        assert C.ARM_ENV[arm]["ORACLE_ANCHOR_SRC"] == ""
    ok += 1
    print("[selftest] (e) table des bras : OK")

    # (f) anonymisation
    perms = {}
    for sc in C.SCENARIOS:
        p1, p2 = C.letter_perm(sc["sid"]), C.letter_perm(sc["sid"])
        assert p1 == p2, "permutation non deterministe"
        assert sorted(p1.values()) == sorted(C.ARMS), "permutation non bijective"
        assert list(p1.keys()) == list(C.LETTERS)
        perms[sc["sid"]] = tuple(p1[c] for c in C.LETTERS)
    assert len(set(perms.values())) >= 5, \
        f"seulement {len(set(perms.values()))} permutations distinctes sur 10"
    ok += 1
    print(f"[selftest] (f) anonymisation ({len(set(perms.values()))}/10 permutations "
          f"distinctes) : OK")

    # (g) forme
    assert C.SW1 % 3 == 0 and C.SW2 % 3 == 0 and C.FRAMES % 3 == 0
    assert C.ANCHOR_FRAME == C.SW1 - 1 and C.FRAMES == 240
    assert 0 < C.BLUR_SIGMA_LIGHT < C.BLUR_SIGMA_STRONG
    ok += 1
    print("[selftest] (g) forme du scenario : OK")

    print(f"[selftest] {ok}/7 verts")


if __name__ == "__main__":
    main()
