#!/usr/bin/env python
# ============================================================================================
# ECHELLE DE FIDELITE, CAMPAGNE 2 (sujet non-ouvrant) -- selftest CPU :
#   (a) forme S2 : schedule verbatim scenarios.py (S2), frames multiples de 3, ancre = 119,
#       et ARITHMETIQUE D'EVICTION : retour - fin de A = 60 >= fenetre 12 + marge -> aucune
#       frame de A ne peut etre dans la fenetre au retour ; sink = frame 0 = C (pas A) ;
#   (b) schedule identique a reid_scen.SCENARIOS["S2"] si importable ;
#   (c) ouvreurs C : un par scenario, tous distincts, sans collision de categorie avec le
#       sujet A, l'interlude B, ni le sujet donneur ;
#   (d) memes sujets / graines / interludes / donneurs que la campagne 1 (comparabilite) ;
#   (e) table des bras inchangee (deleguee au selftest v1, re-verifiee par identite d'objet) ;
#   (f) anonymisation : sel DIFFERENT de v1, permutations deterministes, bijectives,
#       >= 5 distinctes, et table v2 != table v1 ;
#   (g) micro-decoupe : l'apercu tombe DANS le segment du sujet, le retour a 44.8125 s.
# ============================================================================================
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import fidelity_common as C1                                             # noqa: E402
import fidelity2_common as C2                                            # noqa: E402

SUBJECT_HEAD = {"scenario01": "farmer", "scenario02": "fisherman", "scenario03": "parrot",
                "scenario04": "goat", "scenario05": "dog", "scenario06": "cat",
                "scenario07": "horse", "scenario08": "guitar", "scenario09": "scooter",
                "scenario10": "tractor"}
FAMILY = {"parrot": {"bird", "parrot", "flamingo", "flamingos", "owl", "swan", "penguin",
                     "penguins", "sheep"} - {"sheep"},
          "farmer": {"man", "woman", "person", "farmer", "fisherman", "potter", "beekeeper"},
          "fisherman": {"man", "woman", "person", "farmer", "fisherman", "potter",
                        "beekeeper", "sailor"},
          "dog": {"dog"}, "cat": {"cat"}, "horse": {"horse"}, "goat": {"goat", "sheep"},
          "guitar": {"guitar"}, "scooter": {"scooter"}, "tractor": {"tractor", "truck"}}


def words_of(s):
    return set(s.lower().replace(",", " ").split())


def main():
    ok = 0

    # (a) forme + arithmetique d'eviction
    assert C2.SCHEDULE == [(0, 2), (60, 0), (120, 1), (180, 0)] and C2.FRAMES == 240
    for f, _ in C2.SCHEDULE:
        assert f % 3 == 0
    assert C2.A_START == 60 and C2.A_END == 120 and C2.ANCHOR_FRAME == 119
    assert C2.RETURN_FRAME - C2.A_END == 60 >= C2.LOCAL_ATTN + 12, \
        "le segment A doit etre sorti de la fenetre bien avant le retour"
    assert C2.SINK_FRAMES == 1, "sink = frame 0 = l'ouvreur C, jamais le sujet"
    assert C2.TWIN_SCHEDULE == [(0, 2), (60, 0)] and C2.TWIN_FRAMES == 120
    ok += 1
    print("[selftest2] (a) forme S2 + arithmetique d'eviction : OK")

    # (b) schedule verbatim reid_scen
    try:
        sys.path.insert(0, os.path.join(C2.REPO, "reference/runs/reid_scen"))
        import scenarios as S
        assert [tuple(x) for x in S.SCENARIOS["S2"]["schedule"]] == C2.SCHEDULE
        assert S.SCENARIOS["S2"]["frames"] == C2.FRAMES
        print("[selftest2] (b) schedule == reid_scen S2 verbatim : OK")
    except ImportError as e:
        print(f"[selftest2] (b) reid_scen non importable ici ({e}) -- verif differee")
    ok += 1

    # (c) ouvreurs
    ops = [sc["opener"] for sc in C2.SCENARIOS]
    assert len(set(ops)) == len(ops) == 10, "ouvreurs non distincts"
    for i, sc in enumerate(C2.SCENARIOS):
        w = words_of(sc["opener"])
        for who in (sc["sid"], C2.SCENARIOS[C2.donor_of(i)]["sid"]):
            bad = FAMILY.get(SUBJECT_HEAD[who], {SUBJECT_HEAD[who]})
            assert not (w & bad), \
                f"{sc['sid']}: ouvreur '{sc['opener']}' contient {w & bad}"
        assert not (w & words_of(sc["interlude"]) & set(SUBJECT_HEAD.values())), \
            f"{sc['sid']}: ouvreur et interlude partagent un sujet"
    ok += 1
    print("[selftest2] (c) ouvreurs distincts, sans collision A/B/donneur : OK")

    # (d) memes sujets / graines / donneurs que v1
    for sc1, sc2 in zip(C1.SCENARIOS, C2.SCENARIOS):
        assert sc1["sid"] == sc2["sid"] and sc1["cidx"] == sc2["cidx"]
        assert sc1["prompt"] == sc2["prompt"] and sc1["interlude"] == sc2["interlude"]
    assert C2.DONOR_SHIFT == C1.DONOR_SHIFT
    ok += 1
    print("[selftest2] (d) memes 10 sujets / interludes / donneurs que la campagne 1 : OK")

    # (e) table des bras partagee a l'identique
    assert C2.ARM_ENV is C1.ARM_ENV and C2.ARMS is C1.ARMS and C2.OFF is C1.OFF
    ok += 1
    print("[selftest2] (e) table des bras identique (objet partage) : OK")

    # (f) anonymisation independante de v1
    assert C2.ANON_SALT != C1.ANON_SALT
    assert C2.KEY_JSON != C1.KEY_JSON
    perms2, diff = {}, 0
    for sc in C2.SCENARIOS:
        p1, p2 = C2.letter_perm(sc["sid"]), C2.letter_perm(sc["sid"])
        assert p1 == p2 and sorted(p1.values()) == sorted(C2.ARMS)
        perms2[sc["sid"]] = tuple(p1[c] for c in C2.LETTERS)
        if p1 != C1.letter_perm(sc["sid"]):
            diff += 1
    assert len(set(perms2.values())) >= 5
    assert diff >= 8, f"permutations v2 trop proches de v1 ({10 - diff} identiques)"
    ok += 1
    print(f"[selftest2] (f) anonymisation independante ({len(set(perms2.values()))}/10 "
          f"distinctes, {diff}/10 differentes de v1) : OK")

    # (g) micro-decoupe dans le segment du sujet
    a_t = ((C2.A_START * 4 - 3) / 16.0, (C2.A_END * 4 - 3) / 16.0)
    assert a_t[0] < C2.MICRO_GLIMPSE[0] and C2.MICRO_GLIMPSE[1] < a_t[1], \
        f"apercu {C2.MICRO_GLIMPSE} hors du segment du sujet {a_t}"
    assert abs(C2.MICRO_RETURN_T - 44.8125) < 1e-9
    assert C2.RAW_DIR != C1.RAW_DIR and C2.REFS_DIR != C1.REFS_DIR \
        and C2.ANON_DIR != C1.ANON_DIR
    ok += 1
    print("[selftest2] (g) micro-decoupe + repertoires separes : OK")

    print(f"[selftest2] {ok}/7 verts")


if __name__ == "__main__":
    main()
