#!/usr/bin/env python
"""ECHELLE de FIDELITE, CAMPAGNE 2 : regime DU SUJET non-OUVRANT (correction d'Aden 11/08).

DEFAUT de la CAMPAGNE 1 : forme S1 (A ouvre la video) -> la frame 0 du SUJET est epinglee
dans le sink (jamais evincee), donc le sujet peut revenir sans aucune insertion ; c'est le
terrain documente ou "le sink regne par le recache". La campagne 1 est CONSERVEE comme
reference "terrain facile".

Forme RETENUE : S2 du harnais reid_scen (scenarios.py), verbatim :
    C(0-60)  A(60-120)  B(120-180)  A(180-240)      240 frames latentes
Pourquoi S2 et pas S2p : (1) S2 est LE terrain non-ouvrant canonique de la matrice scellee
(celui ou stockoff s'effondre structurellement, gel 03/08) ; (2) la paraphrase de S2p ne
sert qu'au ROUTEUR de frame-bank, absent de cette campagne (insertion au planning) ; elle
changerait le conditionnement sans aucun gain ; (3) le prompt de retour verbatim et sans
attribut garde le meme conditionnement que la campagne 1 : seule la position du sujet dans
le planning change entre les deux campagnes.

Eviction PROUVEE (pas supposee) : sink_size=1 (frame 0 = C), local_attn_size=12 -> au
retour (frame 180) la fenetre couvre [169, 180) (queue de B) et le segment A [60, 120) est
sorti du cache depuis 48+ frames. La preuve RUNTIME est enregistree par run_fidelity2 sur
des cellules temoins : lecture des index du KV cache (global_end/local_end/sink) de CHAQUE
bloc a l'instant du recache de retour -> plus vieille frame non-sink retenue = 169 > 120.

BARREAUX : identiques a la campagne 1 (rien / oracle / flou_leger / flou_fort /
meme_categorie / autre_categorie), ancre = frame 119 (derniere frame latente du segment A).
JUMEAU : dans ce regime le segment A ne peut pas etre la video v2 jugee (le sujet emerge en
cours de rollout) ; le jumeau est donc REGENERE dans le MEME regime : rollout de preparation
C(0-60) A(60-120) en 120 frames avec la graine main+100000, frame 119 -> "meme prompt,
graine +100000, meme regime d'apparition". Le donneur `autre_categorie` = le jumeau du
scenario donneur ((i+5) % 10, inter-categories). Les labels ADEN_V2_STRONG_19 se
transferent ici comme propriete du prompt (forte variabilite intra-classe), pas des
instances exactes : ECART de PROTOCOLE documente au rapport, verifie a l'oeil sur les
planches de refs (sujet A du rollout vs jumeau insere).
"""
import os

import fidelity_common as C1

REPO = C1.REPO
HERE = C1.HERE
HEAVY = os.path.join("/shared/ssd_14T_project/adenelmi/fidelity_ladder", "nonopening")
REFS_DIR = os.path.join(HEAVY, "refs")
RAW_DIR = os.path.join(HEAVY, "videos_raw")
ANON_DIR = os.path.join(HEAVY, "anon")
MICRO_DIR = os.path.join(ANON_DIR, "micro")
CHECK_DIR = os.path.join(HEAVY, "check")
LOG_DIR = os.path.join(HEAVY, "logs")
KEY_JSON = os.path.join(HERE, "key_nonopening.json")
REFS_MANIFEST = os.path.join(REFS_DIR, "manifest.json")
CONFIG_YAML = C1.CONFIG_YAML
PROBE_DIR = C1.PROBE_DIR
V2_INDEX = C1.V2_INDEX

# ---- forme S2 (verbatim scenarios.py) --------------------------------------------------------
FRAMES = 240
SCHEDULE = [(0, 2), (60, 0), (120, 1), (180, 0)]     # (frame, prompt_idx) ; 0=A, 1=B, 2=C
A_START, A_END = 60, 120                              # 1re apparition du sujet
RETURN_FRAME = 180
ANCHOR_FRAME = A_END - 1                              # 119
TWIN_FRAMES = 120                                     # rollout de prep : C(0-60) A(60-120)
TWIN_SCHEDULE = [(0, 2), (60, 0)]
LOCAL_ATTN = 12
SINK_FRAMES = 1

BLUR_SIGMA_LIGHT = C1.BLUR_SIGMA_LIGHT
BLUR_SIGMA_STRONG = C1.BLUR_SIGMA_STRONG

# ---- scenarios : memes 10 sujets (A), memes interludes (B), memes graines ; + OUVREUR C -----
# C choisi sans collision de categorie avec A, avec B, ni avec le sujet DONNEUR.
OPENERS = {
    "scenario01": "a wooden sailboat drifting on a calm lake",
    "scenario02": "a freight train crossing a steel bridge at sunset",
    "scenario03": "a waterfall cascading in a tropical jungle",
    "scenario04": "ocean waves crashing on rocky cliffs",
    "scenario05": "a hot air balloon rising over mountains",
    "scenario06": "a fire truck speeding through a rainy street",
    "scenario07": "a lighthouse standing on a rocky shore at dusk",
    "scenario08": "a herd of sheep crossing a country road",
    "scenario09": "a group of penguins huddled on drifting ice",
    "scenario10": "a coral reef teeming with colorful fish",
}
SCENARIOS = [dict(sc, opener=OPENERS[sc["sid"]]) for sc in C1.SCENARIOS]
DONOR_SHIFT = C1.DONOR_SHIFT
donor_of = C1.donor_of

# ---- bras : memes 6 barreaux, memes env (les hooks sont identiques dans _switch_recache) ----
OFF = C1.OFF
ARMS = C1.ARMS
ARM_ENV = C1.ARM_ENV
SRC_ARMS = C1.SRC_ARMS

# ---- anonymisation : SEL different de la campagne 1 (permutations independantes) ------------
ANON_SALT = "fidelity_ladder_nonopening_2026-08-11"
LETTERS = C1.LETTERS


def letter_perm(sid):
    import hashlib
    import random
    h = hashlib.sha1(f"{ANON_SALT}|{sid}".encode()).hexdigest()
    rng = random.Random(int(h[:16], 16))
    arms = list(ARMS)
    rng.shuffle(arms)
    return dict(zip(LETTERS, arms))


# ---- decoupe des micro-clips (anonymize --campaign nonopening) ------------------------------
# apercu du segment DU SUJET (A : latents 60-120 -> t 14.8-29.8 s), puis le retour complet.
MICRO_GLIMPSE = (16.0, 21.0)
MICRO_RETURN_T = (RETURN_FRAME * 4 - 3) / 16.0        # 44.8125 s
PLANCHE_TITLE = "Échelle de fidélité (sujet non-ouvrant) — planche aveugle"
PLANCHE_NOTE = ("Chaque vidéo : une scène d'ouverture, PUIS le sujet, puis une autre scène, "
                "puis le retour. Le sujet à juger est celui du DEUXIÈME segment (celui de la "
                "phrase affichée).")


# ECHELLE de FLOU FINE (mission coordinateur 12/08, apres le verdict 7 scenarios) :
# le seuil decisif est peut-etre entre sigma=0 et sigma=6 -> 4 barreaux intermediaires
# sigma = 1, 2, 3, 4 sur les 3 scenarios non juges (08, 09, 10). Les barreaux sigma=0
# (oracle), 6 (flou_leger) et 24 (flou_fort) existent deja et sont REUTILISES.
# Paquet anonyme SEPARE : 3 scenarios x 7 niveaux, lettres A-G permutees par scenario
# (sel independant : l'ordre croissant du flou ne doit pas etre devinable), cle dans
# key_blur.json UNIQUEMENT.
BLUR_FINE_SIGMAS = {"flou_s1": 1.0, "flou_s2": 2.0, "flou_s3": 3.0, "flou_s4": 4.0}
BLUR_ARM_ENV = {name: dict(C1.ARM_ENV["oracle"], ORACLE_BLUR_SIGMA=str(s))
                for name, s in BLUR_FINE_SIGMAS.items()}
BLUR_SIDS = ("scenario08", "scenario09", "scenario10")
# barreau -> sigma effectif du paquet (0 = vraie frame, sans aller-retour VAE)
BLUR_LADDER = {"oracle": 0.0, "flou_s1": 1.0, "flou_s2": 2.0, "flou_s3": 3.0,
               "flou_s4": 4.0, "flou_leger": 6.0, "flou_fort": 24.0}
BLUR_LETTERS = "ABCDEFG"
BLUR_SALT = "fidelity_ladder_blurfine_2026-08-12"
ANON_BLUR_DIR = os.path.join(HEAVY, "anon_blur")
MICRO_BLUR_DIR = os.path.join(ANON_BLUR_DIR, "micro")
KEY_BLUR_JSON = os.path.join(HERE, "key_blur.json")


def blur_letter_perm(sid):
    import hashlib
    import random
    h = hashlib.sha1(f"{BLUR_SALT}|{sid}".encode()).hexdigest()
    rng = random.Random(int(h[:16], 16))
    arms = list(BLUR_LADDER)
    rng.shuffle(arms)
    return dict(zip(BLUR_LETTERS, arms))


# ---- refs ------------------------------------------------------------------------------------
def anchor_pt(sid):
    return os.path.join(REFS_DIR, f"anchor_{sid}.pt")


def twin_pt(sid):
    """Frame 119 du rollout de prep jumeau (graine main+100000, meme regime C->A)."""
    return os.path.join(REFS_DIR, f"twin_{sid}.pt")


def other_pt(sid):
    """= twin_pt du scenario donneur (autre categorie), chemin distinct pour l'audit."""
    return os.path.join(REFS_DIR, f"other_{sid}.pt")


def raw_mp4(sid, arm):
    return os.path.join(RAW_DIR, f"{sid}_{arm}.mp4")


def load_manifest():
    import json
    with open(REFS_MANIFEST) as f:
        return json.load(f)
