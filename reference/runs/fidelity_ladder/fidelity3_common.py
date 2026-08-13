#!/usr/bin/env python
# ============================================================================================
# SUPPLEMENT (demande d'Aden 12/08, ajustee a 4 scenes) : 4 NOUVEAUX SUJETS au protocole
# EXACT de la campagne 2 (regime non-ouvrant S2, memes 6 barreaux, meme canal, meme preuve
# d'eviction sur temoins).
#
# CHOIX DES SUJETS (classement scelle, rank_candidates.py -> supplement_ranking.json) :
# paires FORTES v2 non utilisees par les scenarios 01-10, classees par visibilite
# min(main, twin) de l'aire du masque, departage det/valid ; objets inertes exclus du choix
# final (regle 3 du coordinateur). Resultat : rabbit (board 17), nurse (9), monkey (18),
# owl (16) -- les 4 seuls candidats non inertes, dans l'ordre du classement. L'owl est
# dernier (aire 0.08) : signale au rapport.
#
# DONNEURS `autre_categorie` : les ancres JUMEAU deja generees de la campagne 2 (meme regime
# frame 119), affectation deterministe inter-categories :
#   scenario11 rabbit (animal)  <- jumeau scenario08 (guitare, objet)
#   scenario12 nurse  (person)  <- jumeau scenario09 (scooter, vehicule)
#   scenario13 monkey (animal)  <- jumeau scenario10 (tracteur, vehicule)
#   scenario14 owl    (bird)    <- jumeau scenario01 (fermier, personne)
# ============================================================================================
import json
import os

import fidelity_common as C1
import fidelity2_common as C2

REPO = C1.REPO
HERE = C1.HERE
HEAVY = os.path.join("/shared/ssd_14T_project/adenelmi/fidelity_ladder", "supplement")
REFS_DIR = os.path.join(HEAVY, "refs")
RAW_DIR = os.path.join(HEAVY, "videos_raw")
ANON_DIR = os.path.join(HEAVY, "anon")
MICRO_DIR = os.path.join(ANON_DIR, "micro")
CHECK_DIR = os.path.join(HEAVY, "check")
LOG_DIR = os.path.join(HEAVY, "logs")
KEY_JSON = os.path.join(HERE, "key_supplement.json")
REFS_MANIFEST = os.path.join(REFS_DIR, "manifest.json")
CONFIG_YAML = C1.CONFIG_YAML
PROBE_DIR = C1.PROBE_DIR
V2_INDEX = C1.V2_INDEX

# forme S2 identique a la campagne 2 (constantes partagees)
FRAMES, SCHEDULE = C2.FRAMES, C2.SCHEDULE
A_START, A_END, RETURN_FRAME = C2.A_START, C2.A_END, C2.RETURN_FRAME
ANCHOR_FRAME, TWIN_FRAMES, TWIN_SCHEDULE = C2.ANCHOR_FRAME, C2.TWIN_FRAMES, C2.TWIN_SCHEDULE
LOCAL_ATTN, SINK_FRAMES = C2.LOCAL_ATTN, C2.SINK_FRAMES
BLUR_SIGMA_LIGHT, BLUR_SIGMA_STRONG = C2.BLUR_SIGMA_LIGHT, C2.BLUR_SIGMA_STRONG

SCENARIOS = [
    {"sid": "scenario11", "board_no": 17, "cidx": 9000016, "category": "animal",
     "prompt": "a rabbit hops across a lawn",
     "interlude": "a pottery kiln glowing inside a workshop",
     "opener": "city fireworks bursting over a harbor at night"},
    {"sid": "scenario12", "board_no": 9, "cidx": 9000008, "category": "person",
     "prompt": "a nurse walks along a hospital corridor",
     "interlude": "a red cable car gliding over a mountain valley",
     "opener": "a fountain splashing in a quiet plaza"},
    {"sid": "scenario13", "board_no": 18, "cidx": 9000017, "category": "animal",
     "prompt": "a monkey swings between branches in a jungle",
     "interlude": "a lava flow glowing on a volcanic slope",
     "opener": "a rowboat tied to a wooden dock on a misty lake"},
    {"sid": "scenario14", "board_no": 16, "cidx": 9000015, "category": "bird",
     "prompt": "an owl perches on a branch in a forest",
     "interlude": "a desert canyon with towering red rock walls",
     "opener": "a steam locomotive crossing a stone viaduct"},
]
# donneur = ancre jumeau DEJA GENEREE de la campagne 2 (chemins materialises dans other_pt)
DONOR_SIDS = {"scenario11": "scenario08", "scenario12": "scenario09",
              "scenario13": "scenario10", "scenario14": "scenario01"}
DONOR_TWIN_SRC = {sid: C2.twin_pt(dn) for sid, dn in DONOR_SIDS.items()}
DONOR_CATEGORY = {"scenario08": "object", "scenario09": "vehicle",
                  "scenario10": "vehicle", "scenario01": "person"}

OFF, ARMS, ARM_ENV, SRC_ARMS = C1.OFF, C1.ARMS, C1.ARM_ENV, C1.SRC_ARMS
BLUR_ARM_ENV = {}                      # aucun barreau de flou fin dans ce paquet
WITNESS_SIDS = ("scenario11", "scenario13")   # 1er scenario de chaque moitie

ANON_SALT = "fidelity_ladder_supplement_2026-08-12"
LETTERS = C1.LETTERS

MICRO_GLIMPSE = C2.MICRO_GLIMPSE
MICRO_RETURN_T = C2.MICRO_RETURN_T
PLANCHE_TITLE = "Échelle de fidélité (scènes supplémentaires) — planche aveugle"
PLANCHE_NOTE = C2.PLANCHE_NOTE


def letter_perm(sid):
    import hashlib
    import random
    h = hashlib.sha1(f"{ANON_SALT}|{sid}".encode()).hexdigest()
    rng = random.Random(int(h[:16], 16))
    arms = list(ARMS)
    rng.shuffle(arms)
    return dict(zip(LETTERS, arms))


def base_manifest():
    """Graines et prompts des 4 nouveaux sujets, LUS dans l'index v2 (formule verifiee)."""
    with open(V2_INDEX) as f:
        idx = json.load(f)
    by_vid = {v["video_id"]: v for v in idx}
    out = {}
    for sc in SCENARIOS:
        m = by_vid[f"main:{sc['cidx']}"]
        t = by_vid[f"twin:{sc['cidx']}"]
        assert m["prompt"] == sc["prompt"] == t["prompt"], f"{sc['sid']}: prompt != index"
        assert int(m["seed"]) == 42 + 500000 + sc["cidx"], f"{sc['sid']}: graine main"
        assert int(t["seed"]) == int(m["seed"]) + 100000, f"{sc['sid']}: graine twin"
        out[sc["sid"]] = {"board_no": sc["board_no"], "cidx": sc["cidx"],
                          "category": sc["category"], "prompt": sc["prompt"],
                          "interlude": sc["interlude"], "opener": sc["opener"],
                          "seed_main": int(m["seed"]), "seed_twin": int(t["seed"])}
    return out


def anchor_pt(sid):
    return os.path.join(REFS_DIR, f"anchor_{sid}.pt")


def twin_pt(sid):
    return os.path.join(REFS_DIR, f"twin_{sid}.pt")


def other_pt(sid):
    return os.path.join(REFS_DIR, f"other_{sid}.pt")


def raw_mp4(sid, arm):
    return os.path.join(RAW_DIR, f"{sid}_{arm}.mp4")


def load_manifest():
    with open(REFS_MANIFEST) as f:
        return json.load(f)
