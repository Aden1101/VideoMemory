#!/usr/bin/env python
"""ECHELLE de FIDELITE (ETAPE 0 du plan reference/plan_entrainement_video_2026-08-07.md).

Scenario de revisite A -> B -> A (forme S1 du harnais reid_scen : A(0-81) B(81-162)
A(162-240), 240 frames latentes). Au retour (frame 162), UNE frame latente est inseree dans
la fenetre KV par le canal valide (+0.83) : ORACLE_SPLICE=1 / ORACLE_SPLICE_FRAMES=1, la
frame inseree REMPLACE le slot de recence. Les 6 barreaux ne different QUE par le CONTENU
de cette frame (meme prompt, meme graine, meme schedule, meme flux RNG) :

  Rien            aucune insertion (memzero verbatim)
  oracle          la vraie frame latente 80 du segment A de ce rollout (bras oracle1)
  flou_leger      la meme, VAE decode -> flou gaussien sigma=6 px -> VAE encode
                  (hook ORACLE_BLUR_SIGMA, bras oracle1blur de la batterie d'arbitrage)
  flou_fort       idem, sigma=24 px (= la fidelite documentee de nos decodeurs :
                  "the decoded anchor sits beyond sigma=32 px in DINO space", et le
                  bras oracle1blur historique tournait a sigma=24)
  meme_categorie  la frame latente 80 du JUMEAU v2 (meme prompt, graine +100000), lue
                  dans les shards du corpus v2 (hook ORACLE_ANCHOR_SRC, bras oracle1twin)
  autre_categorie la frame latente 80 de la video v2 PRINCIPALE d'un AUTRE scenario,
                  d'une autre categorie (tirage deterministe : donor = (i+5) % 10)

MATERIAU : 10 paires du corpus v2 (30 paires meme-prompt sans attribut), toutes dans la
liste scellee ADEN_V2_STRONG_19 (sujets juges DISTINGUABLES par l'oeil d'Aden, registre
05/08). Plage de numeros de planche v2 : board_no = corpus_idx - 9000000 + 1 (l'ordre
lexicographique des video_id "main:90000xx" est l'ordre numerique, verifie par selftest).

Reproduction bit A bit DU segment A : la graine du rollout = la graine v2 de la video
principale (config.seed 42 + 500000 + corpus_idx, lue dans l'index, jamais recalculee a
l'aveugle), et le bruit est tire en DEUX temps : d'abord torch.randn([1, 90, ...]) sur le
generateur global (exactement ce que collect_pairs.rollout_collect a consomme), puis les
150 frames restantes sur un torch.Generator SEPARE (le flux global n'est pas touche). Les
blocs 0..26 (frames 0..80) consomment ensuite les memes tirages add_noise que la collecte
-> segment A identique a la video v2 que l'oeil d'Aden a jugee (verifie contre les latents
stockes par run_fidelity, non fatal : en cas d'ecart residuel (autre carte GPU), on decode
les deux frames 80 et l'ecart est documente au rapport).

ANONYMISATION : fichiers scenarioNN_L.mp4 (L dans A..F), permutation des barreaux par
scenario tiree d'un hash deterministe (sel fige ci-dessous), table de correspondance dans
reference/runs/fidelity_ladder/key.json UNIQUEMENT.
"""
import hashlib
import json
import os
import random

REPO = "/shared/ssd_14T/home/adenelmi/Anchor-Forcing"
HERE = os.path.dirname(os.path.abspath(__file__))
HEAVY = "/shared/ssd_14T_project/adenelmi/fidelity_ladder"
REFS_DIR = os.path.join(HEAVY, "refs")
RAW_DIR = os.path.join(HEAVY, "videos_raw")
ANON_DIR = os.path.join(HEAVY, "anon")
MICRO_DIR = os.path.join(ANON_DIR, "micro")
CHECK_DIR = os.path.join(HEAVY, "check")
LOG_DIR = os.path.join(HEAVY, "logs")
KEY_JSON = os.path.join(HERE, "key.json")
REFS_MANIFEST = os.path.join(REFS_DIR, "manifest.json")

PROBE_DIR = os.path.join(REPO, "reference/runs/reid_probe_l1")
CONFIG_YAML = os.path.join(PROBE_DIR, "probe_l1.yaml")

V2_INDEX = "/shared/ssd_14T_project/adenelmi/probe1/index_v2.json"

# ---- forme du scenario (S1 du harnais reid_scen, la forme legacy validee) -------------------
FRAMES = 240
SW1 = 81            # A -> B
SW2 = 162           # B -> A (le retour : l'insertion se fait ICI)
ANCHOR_FRAME = 80   # la frame latente inseree par `oracle` = SW1 - 1
V2_FRAMES = 90      # les videos v2 sont des singles 90f : le bruit des 90 premieres frames
                    # est tire sur le generateur global pour reproduire la collecte

# ---- flous (documentes, mesures par measure_blur.py) ----------------------------------------
BLUR_SIGMA_LIGHT = 6.0    # px a 480x832 : adoucissement visible, details fins perdus
BLUR_SIGMA_STRONG = 24.0  # px : le niveau documente de nos decodeurs (bras oracle1blur)

# ---- les 10 scenarios ------------------------------------------------------------------------
# board_no dans ADEN_V2_STRONG_19 (labels scelles 05/08). Prompts verbatim du manifeste v2
# (verifies contre l'index par selftest). Interludes B choisis dans les prompts d'interlude
# eprouves du harnais reid_scen (scenarios.py), en evitant toute collision de categorie avec
# le sujet A ET avec le sujet du donneur `autre_categorie`.
# donor = scenario (i+5) % 10 : les 10 appariements sont tous inter-categories (selftest).
SCENARIOS = [
    {"sid": "scenario01", "board_no": 5, "cidx": 9000004, "category": "person",
     "prompt": "a farmer walks between rows of crops in a field",
     "interlude": "a swarm of jellyfish drifting in deep blue water"},
    {"sid": "scenario02", "board_no": 6, "cidx": 9000005, "category": "person",
     "prompt": "a fisherman pulls a net onto a boat",
     "interlude": "a snowy mountain village at dusk"},
    {"sid": "scenario03", "board_no": 11, "cidx": 9000010, "category": "bird",
     "prompt": "a parrot lands on a fence in a garden",
     "interlude": "a rally car kicking up dust on a desert track"},
    {"sid": "scenario04", "board_no": 12, "cidx": 9000011, "category": "animal",
     "prompt": "a goat climbs a slope above a valley",
     "interlude": "a flock of flamingos wading in a shallow lagoon"},
    {"sid": "scenario05", "board_no": 13, "cidx": 9000012, "category": "animal",
     "prompt": "a dog runs along a beach",
     "interlude": "a glass skyscraper reflecting the morning sky"},
    {"sid": "scenario06", "board_no": 14, "cidx": 9000013, "category": "animal",
     "prompt": "a cat jumps onto a counter in a kitchen",
     "interlude": "a wooden windmill turning slowly on a green hillside"},
    {"sid": "scenario07", "board_no": 15, "cidx": 9000014, "category": "animal",
     "prompt": "a horse trots across a paddock",
     "interlude": "a yellow taxi crossing a rain-soaked intersection"},
    {"sid": "scenario08", "board_no": 23, "cidx": 9000022, "category": "object",
     "prompt": "a guitar leans against a wall in a rehearsal room",
     "interlude": "a busy night market with glowing paper lanterns"},
    {"sid": "scenario09", "board_no": 27, "cidx": 9000026, "category": "vehicle",
     "prompt": "a scooter waits at a traffic light on a street",
     "interlude": "a field of sunflowers under a bright sky"},
    {"sid": "scenario10", "board_no": 28, "cidx": 9000027, "category": "vehicle",
     "prompt": "a tractor crosses a field at sunrise",
     "interlude": "a group of kayakers paddling down a river"},
]
DONOR_SHIFT = 5


def donor_of(i):
    """Index du scenario dont la frame `autre_categorie` du scenario i provient."""
    return (i + DONOR_SHIFT) % len(SCENARIOS)


# ---- table des bras --------------------------------------------------------------------------
# Copie LITTERALE de run_reid_probe._OFF (l'union des cles que chaque bras remet a zero,
# lecon MEM_WRITER) + les cles ajoutees par les campagnes ulterieures. Run_fidelity verifie a
# l'import que cette copie COUVRE run_reid_probe._OFF avec les memes valeurs (aucune derive
# silencieuse possible).
OFF = {"MEM_ZERO_ALL": "0", "MEM_EPISODIC": "0", "ORACLE_SPLICE": "0",
       "ORACLE_SPLICE_FRAMES": "0", "ORACLE_SPLICE_BLOCKS": "", "ORACLE_SPLICE_NOISE": "0",
       "ORACLE_SPLICE_POS": "", "ORACLE_SPLICE_SLOT": "-1",
       "MEM_DECODE": "", "MEM_DECODE_SLOTS": "1", "MEM_WRITER": "",
       "MEM_DECODE_THRU": "0", "ORACLE_ANCHOR_DUMP": "", "ORACLE_ANCHOR_SRC": "",
       "ORACLE_BLUR_SIGMA": "", "MEM_DECODE_DUMP": "",
       "MEM_DECODE_NOISE": "0", "MEM_DECODE_NOISE_SEED": "192837465", "MEM_DECODE_SRC": "",
       "FRAMEBANK_GT": "0",
       "MEM_SDEDIT": "0", "MEM_SDEDIT_SEED": "5150241", "MEM_SDEDIT_FRAMES": "0",
       "MEM_SINKPURGE": "0",
       # cles du chemin N-switch / campagnes 08 (toutes off ; on ne les activera jamais ici,
       # mais un env herite d'un shell precedent ne doit pas pouvoir les laisser allumees)
       "FRAMEBANK": "", "FRAMEBANK_TAU": "0.90", "FRAMEBANK_TIE": "recent",
       "MEM_DECODE_ROUTED": "0", "MEM_DECODE_COMPOSITE": "0", "MEM_ADDITIVE_KV": "0",
       "MEM_DECODE_SLOT_OFFSET": "-1", "MEM_ANCHOR_T": "0", "MEM_EVICT_DEBUG": "0"}

# ordre canonique des barreaux (celui de la permutation d'anonymisation, pas celui des
# fichiers livres : les lettres A..F sont permutees par scenario)
ARMS = ("oracle", "flou_leger", "flou_fort", "meme_categorie", "autre_categorie", "rien")

_SPLICE = dict(OFF, MEM_ZERO_ALL="1", ORACLE_SPLICE="1", ORACLE_SPLICE_FRAMES="1")
ARM_ENV = {
    "rien": dict(OFF, MEM_ZERO_ALL="1"),
    "oracle": dict(_SPLICE),
    "flou_leger": dict(_SPLICE, ORACLE_BLUR_SIGMA=str(BLUR_SIGMA_LIGHT)),
    "flou_fort": dict(_SPLICE, ORACLE_BLUR_SIGMA=str(BLUR_SIGMA_STRONG)),
    # ORACLE_ANCHOR_SRC injecte au run time (chemin du .pt du jumeau / du donneur)
    "meme_categorie": dict(_SPLICE),
    "autre_categorie": dict(_SPLICE),
}
SRC_ARMS = ("meme_categorie", "autre_categorie")

# ---- decoupe des micro-clips (anonymize) -----------------------------------------------------
MICRO_GLIMPSE = (8.0, 13.0)                 # 5 s du segment A (le sujet ouvre la video)
MICRO_RETURN_T = (SW2 * 4 - 3) / 16.0       # 40.3125 s : premiere frame pixel du retour
PLANCHE_TITLE = "Échelle de fidélité — planche aveugle"
PLANCHE_NOTE = ("Chaque vidéo montre : le sujet, puis une autre scène, puis le retour du "
                "sujet.")

# ---- anonymisation ---------------------------------------------------------------------------
ANON_SALT = "fidelity_ladder_v1_2026-08-10"
LETTERS = "ABCDEF"


def letter_perm(sid):
    """{lettre: barreau} deterministe par scenario, sel fige. La graine ne depend que du
    sel et du sid : re-executable a l'identique, jamais du contenu des videos."""
    h = hashlib.sha1(f"{ANON_SALT}|{sid}".encode()).hexdigest()
    rng = random.Random(int(h[:16], 16))
    arms = list(ARMS)
    rng.shuffle(arms)
    return dict(zip(LETTERS, arms))


# ---- refs ------------------------------------------------------------------------------------
def anchor_pt(sid):
    """Dump natif (bf16) de la frame 80 du rollout `rien` de ce scenario (hook
    ORACLE_ANCHOR_DUMP) ; re-verifie bit a bit par tous les autres bras."""
    return os.path.join(REFS_DIR, f"anchor_{sid}.pt")


def twin_pt(sid):
    """Frame latente 80 du JUMEAU v2 (fp16 des shards)."""
    return os.path.join(REFS_DIR, f"twin_{sid}.pt")


def other_pt(sid):
    """Frame latente 80 de la video v2 PRINCIPALE du scenario donneur (fp16 des shards)."""
    return os.path.join(REFS_DIR, f"other_{sid}.pt")


def main81_pt(sid):
    """Latents fp16 stockes [0:81] de la video v2 principale (verif bit a bit du segment A)."""
    return os.path.join(REFS_DIR, f"main81_{sid}.pt")


def raw_mp4(sid, arm):
    return os.path.join(RAW_DIR, f"{sid}_{arm}.mp4")


def load_manifest():
    with open(REFS_MANIFEST) as f:
        return json.load(f)
