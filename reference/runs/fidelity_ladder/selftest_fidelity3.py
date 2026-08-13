#!/usr/bin/env python
# ============================================================================================
# SUPPLEMENT -- selftest CPU :
#  (a) les 4 sujets = exactement les 4 non-inertes du classement scelle
#      (supplement_ranking.json), dans l'ordre du classement, tous STRONG et non utilises ;
#  (b) prompts / graines contre l'index v2 (base_manifest ne leve pas) ;
#  (c) donneurs : jumeaux DEJA GENERES de la campagne 2, inter-categories, fichiers presents ;
#  (d) interludes et ouvreurs sans collision de categorie (sujet A et donneur), distincts ;
#  (e) forme et constantes partagees avec la campagne 2 (schedule, ancre, decoupe micro) ;
#  (f) anonymisation : sel independant, permutations deterministes, bijectives, >= 3
#      distinctes sur 4, differentes des tables v1/v2 pour les memes lettres.
# ============================================================================================
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import fidelity_common as C1                                             # noqa: E402
import fidelity2_common as C2                                            # noqa: E402
import fidelity3_common as C3                                            # noqa: E402

SUBJECT_HEAD = {"scenario11": "rabbit", "scenario12": "nurse", "scenario13": "monkey",
                "scenario14": "owl"}
DONOR_HEAD = {"scenario08": "guitar", "scenario09": "scooter", "scenario10": "tractor",
              "scenario01": "farmer"}
FAMILY = {"rabbit": {"rabbit"}, "monkey": {"monkey"},
          "nurse": {"man", "woman", "person", "nurse", "farmer", "fisherman", "worker"},
          "owl": {"owl", "bird", "parrot", "swan", "penguin", "penguins", "flamingo"},
          "guitar": {"guitar"}, "scooter": {"scooter"}, "tractor": {"tractor", "truck"},
          "farmer": {"man", "woman", "person", "farmer", "fisherman", "worker"}}


def words_of(s):
    return set(s.lower().replace(",", " ").split())


def main():
    ok = 0

    # (a) choix des sujets == classement scelle
    with open(os.path.join(HERE, "supplement_ranking.json")) as f:
        rk = json.load(f)
    non_inert = [c["board_no"] for c in rk["ranking"] if not c["inert"]]
    assert [sc["board_no"] for sc in C3.SCENARIOS] == non_inert[:4] == rk["chosen_boards"]
    used1 = {sc["board_no"] for sc in C1.SCENARIOS}
    strong = (5, 6, 9, 11, 12, 13, 14, 15, 16, 17, 18, 22, 23, 24, 25, 26, 27, 28, 29)
    for sc in C3.SCENARIOS:
        assert sc["board_no"] in strong and sc["board_no"] not in used1
        assert sc["cidx"] == 9000000 + sc["board_no"] - 1
    ok += 1
    print("[selftest3] (a) sujets == classement scelle (4 non-inertes, ordre) : OK")

    # (b) index v2
    C3.base_manifest()
    ok += 1
    print("[selftest3] (b) prompts + graines contre l'index v2 : OK")

    # (c) donneurs
    for sid, dn in C3.DONOR_SIDS.items():
        cat_a = next(sc["category"] for sc in C3.SCENARIOS if sc["sid"] == sid)
        assert C3.DONOR_CATEGORY[dn] != cat_a, f"{sid}: donneur {dn} de MEME categorie"
        assert os.path.exists(C3.DONOR_TWIN_SRC[sid]), \
            f"{sid}: jumeau donneur absent {C3.DONOR_TWIN_SRC[sid]}"
    assert len(set(C3.DONOR_SIDS.values())) == 4
    ok += 1
    print("[selftest3] (c) donneurs inter-categories, jumeaux presents : OK")

    # (d) interludes / ouvreurs
    seen = set()
    for sc in C3.SCENARIOS:
        for field in ("interlude", "opener"):
            txt = sc[field]
            assert txt not in seen, f"duplicat {field}: {txt}"
            seen.add(txt)
            w = words_of(txt)
            for head in (SUBJECT_HEAD[sc["sid"]], DONOR_HEAD[C3.DONOR_SIDS[sc["sid"]]]):
                bad = FAMILY.get(head, {head})
                assert not (w & bad), f"{sc['sid']}: {field} '{txt}' contient {w & bad}"
    ok += 1
    print("[selftest3] (d) interludes / ouvreurs sans collision : OK")

    # (e) constantes partagees
    assert C3.SCHEDULE is C2.SCHEDULE and C3.FRAMES == C2.FRAMES
    assert C3.ANCHOR_FRAME == 119 and C3.MICRO_RETURN_T == C2.MICRO_RETURN_T
    assert C3.ARM_ENV is C1.ARM_ENV and C3.WITNESS_SIDS == ("scenario11", "scenario13")
    assert C3.RAW_DIR != C2.RAW_DIR and C3.KEY_JSON not in (C1.KEY_JSON, C2.KEY_JSON)
    ok += 1
    print("[selftest3] (e) forme partagee campagne 2, repertoires/cle separes : OK")

    # (f) anonymisation
    assert C3.ANON_SALT not in (C1.ANON_SALT, C2.ANON_SALT)
    perms = {}
    for sc in C3.SCENARIOS:
        p1, p2 = C3.letter_perm(sc["sid"]), C3.letter_perm(sc["sid"])
        assert p1 == p2 and sorted(p1.values()) == sorted(C3.ARMS)
        perms[sc["sid"]] = tuple(p1[c] for c in C3.LETTERS)
    assert len(set(perms.values())) >= 3
    ok += 1
    print(f"[selftest3] (f) anonymisation ({len(set(perms.values()))}/4 distinctes) : OK")

    print(f"[selftest3] {ok}/6 verts")


if __name__ == "__main__":
    main()
