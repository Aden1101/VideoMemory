#!/usr/bin/env python
# ============================================================================================
# SCENARIO MATRIX for the re-ID revisit probe -- prompt TRIPLES, N-switch schedules, arm table
# and the FIRST-APPEARANCE scoring. Pure data + pure functions, no torch import at module load
# beyond what the scoring needs, so the held-out grep selftest runs on CPU in a second.
#
# WHY a matrix. The legacy campaign only ever ran S1 = A(0-81) -> B(81-162) -> A(162-240), i.e.
# the subject OPENS the video. That is the single best terrain for LongLive's attention sink
# (the sink holds frame 0 = the subject) and it is where stockoff scores 0.847 while our
# memdec20 reaches 0.812. The scenarios below move the subject OFF the opening (S2), stack more
# interference (S3) and ask for TWO resurrections (S4) -- the terrains where a first-frame sink
# structurally cannot help.
#
#   S1  A(0-81)   B(81-162)  A(162-240)                  legacy bridge, 240 frames
#   S2  C(0-60)   A(60-120)  B(120-180) A(180-240)       NON-OPENING subject, 240 frames
#   S3  A(0-75)   B(75-150)  C(150-225) A(225-300)       multi-switch, 2 interludes, 300 frames
#   S4  A(0-60)   B(60-120)  A(120-180) C(180-240) A(240-300)   DOUBLE revisit, 300 frames
#
# One TRIPLE (A, B, C) drives all four: A = the re-identified subject, B and C = interludes
# (in S2, C is the opener). Every start frame is a multiple of num_frame_per_block = 3.
# ============================================================================================
import json
import os
import re
import sys

REPO = "/shared/ssd_14T/home/adenelmi/Anchor-Forcing"
PROBE_DIR = os.path.join(REPO, "reference/runs/reid_probe_l1")
sys.path.insert(0, REPO)
sys.path.insert(0, PROBE_DIR)

from run_reid_probe import PAIRS, _OFF  # noqa: E402  (THE reference arm-env table)

# ------------------------------------------------------------------------------------------
# TRIPLES (A = re-identifiable subject, B and C = unrelated interludes)
# 0-7: the 8 legacy PAIRS, verbatim, + a third scene  -> S1 on triples 0-7 is the legacy
#      campaign bitwise (same prompts, same cell_seed formula), which makes the two campaigns
#      directly comparable and gives a free cross-campaign check.
# 8-11: new, chosen to widen the subject categories (vehicle / person / animal / object).
# ------------------------------------------------------------------------------------------
_EXTRA_C = [
    "a wooden sailboat drifting on a calm lake",
    "a busy night market with glowing paper lanterns",
    "a fire truck speeding through a rainy street",
    "a flock of flamingos wading in a shallow lagoon",
    "a potter shaping a clay bowl on a spinning wheel",
    "a snowy mountain village at dusk",
    "a yellow taxi crossing a rain-soaked intersection",
    "a beekeeper tending white hives in an orchard",
]
assert len(_EXTRA_C) == len(PAIRS)

TRIPLES = [(a, b, c) for (a, b), c in zip(PAIRS, _EXTRA_C)] + [
    # 8: vehicle subject
    ("a red double-decker bus driving past an old stone bridge",
     "a swarm of jellyfish drifting in deep blue water",
     "a violinist playing on a wooden stage"),
    # 9: person subject
    ("an old fisherman in a green woolen sweater mending nets on a wooden pier",
     "a rally car kicking up dust on a desert track",
     "a field of sunflowers under a bright sky"),
    # 10: animal subject
    ("a black and white border collie herding ducks on a farm",
     "a glass skyscraper reflecting the morning sky",
     "a street artist spray painting a colorful mural"),
    # 11: rigid object subject
    ("a copper teapot steaming on a marble kitchen counter",
     "a group of kayakers paddling down a river",
     "a marching band parading down a main street"),
]

# ------------------------------------------------------------------------------------------
# PARAPHRASE of A, one per triple (index 3 of the prompt list). Used ONLY by the "p" scenario
# variants (S2p/S3p/S4p): there the revisit is asked for with DIFFERENT WORDS, so
#   - every arm gets the same (paraphrased) conditioning at the revisit -> still a fair
#     comparison, and the re-ID reference stays the segment of the FIRST appearance;
#   - the frame-bank ROUTER must match A' to A among 2-4 banked scenes instead of hitting a
#     cosine of exactly 1.0. Measured (prompt_key_matrix.py): with the verbatim prompt the
#     correct match scores 1.0000 and the hardest in-rollout distractor 0.7218, i.e. selection
#     is trivially perfect -- the paraphrase variants are the only place where framebank and
#     framebank_oracle can actually differ by a SELECTION error.
# Rule for writing them: same subject and same discriminating attributes, different wording
# and a different setting clause.
# ------------------------------------------------------------------------------------------
PARAPHRASES = [
    "the brown dog with the white chest, trotting across the grass again",
    "the same woman in her yellow raincoat, back on the sand",
    "the orange tabby cat again, stretching on the wooden tabletop",
    "the red-bearded man in his blue denim jacket, back at the cafe table",
    "the white horse with the black mane, back among the tall grass",
    "the green and yellow parrot again, shuffling along the wooden rail",
    "the little girl in her purple dress with the red balloon, back on the playground",
    "the blue vintage bicycle again, still propped on the red bricks",
    "the red double-decker bus again, rolling back along the old bridge",
    "the old fisherman in his green woolen sweater, back at his nets",
    "the black and white border collie again, circling the ducks",
    "the copper teapot again, still steaming on the marble top",
]
assert len(PARAPHRASES) == len(TRIPLES)

# ------------------------------------------------------------------------------------------
# SAME-SCENE triples (scenarios SS2 / SS1). Everything above changes the WHOLE scene at a
# switch, so a revisit asks the model to bring back a subject AND its background at once, and
# the DINO re-ID is dominated by the background. Here the DECOR is held fixed and ONLY THE
# SUBJECT changes:
#
#   "the man in red, then the woman in blue, then the man in red: the camera keeps rolling."
#
# The three prompts of a same-scene triple share a VERBATIM common suffix (the scene clause) and
# differ only by the leading subject clause -- enforced by selftest (ss), >= 8 shared words.
# Role assignment follows the S2 convention: index 2 = C = the OPENER (a third subject of the
# same place: barista / gardener / delivery man), index 0 = A = the re-identified subject, index
# 1 = B = the confusable interlude. Consequences, and why this tier is the discriminating one:
#   * background consistency is no longer a free ride: every arm keeps the same decor, so the
#     re-ID delta between arms can only come from the SUBJECT;
#   * the confusable control ctrl_resA_B (revisit vs the interlude it follows) is now a REAL
#     distractor, same decor different subject -> reid - ctrl_resA_B is the margin that matters;
#   * crop60 (center 60 %, mostly the subject) is the primary variant here, full the secondary.
# 3 triples now, written so the list can grow to 12 without touching anything else.
# ------------------------------------------------------------------------------------------
SAMESCENE_TRIPLES = [
    # 12: person -- same cafe
    ("a man in a bright red jacket sitting at a wooden table in a warm Parisian cafe",
     "a woman in a long blue dress sitting at a wooden table in a warm Parisian cafe",
     "a barista in a white apron wiping a wooden table in a warm Parisian cafe"),
    # 13: animal -- same lawn and red bench
    ("a golden retriever lying on a green lawn beside a red wooden bench in a city park",
     "a grey cat lying on a green lawn beside a red wooden bench in a city park",
     "a gardener in blue overalls raking leaves on a green lawn beside a red wooden bench "
     "in a city park"),
    # 14: rigid object -- same bakery front
    ("a blue vintage scooter parked in front of a yellow bakery with striped awnings",
     "a black bicycle parked in front of a yellow bakery with striped awnings",
     "a delivery man in a green cap stacking bread crates in front of a yellow bakery "
     "with striped awnings"),
    # 15: vehicle -- same repair garage (elargissement 2026-08-03, comparaison a 5 cellules)
    ("a red vintage car parked inside a cluttered repair garage with tools hanging on the wall",
     "a blue bicycle leaning inside a cluttered repair garage with tools hanging on the wall",
     "a mechanic in orange overalls working inside a cluttered repair garage with tools "
     "hanging on the wall"),
    # 16: animal vs rigid object -- same greenhouse
    ("a green parrot perched on a wooden potting table in a sunlit greenhouse full of ferns",
     "a copper watering can standing on a wooden potting table in a sunlit greenhouse full "
     "of ferns",
     "a gardener in a straw hat leaning on a wooden potting table in a sunlit greenhouse full "
     "of ferns"),
]
SAMESCENE_PARAPHRASES = [
    "the man in the bright red jacket again, back at the same cafe table",
    "the golden retriever again, back on the lawn by the red bench",
    "the blue vintage scooter again, still outside the same yellow bakery",
    "the red vintage car again, still parked in the same repair garage",
    "the green parrot again, back on the potting table in the same greenhouse",
]
assert len(SAMESCENE_PARAPHRASES) == len(SAMESCENE_TRIPLES)
# indices 12, 13, 14 -- APPENDED, so triples 0-11 and every mp4/JSON already on disk keep their
# index, their prompts and their cell_seed (= config.seed + triple_idx + 1000 * seed_idx).
SAMESCENE_TRIPLE_IDS = list(range(len(TRIPLES), len(TRIPLES) + len(SAMESCENE_TRIPLES)))
TRIPLES = TRIPLES + SAMESCENE_TRIPLES
PARAPHRASES = PARAPHRASES + SAMESCENE_PARAPHRASES
assert len(PARAPHRASES) == len(TRIPLES)


# ------------------------------------------------------------------------------------------
# TIER "TEXTE INSUFFISANT" (pilote 2026-08-03, apres les controles reviewer du §11).
#
# LE POURQUOI. Sur le tier cross-scene, `wronganchor` rend 106 % du gain de `memdec20` sans
# laisser AUCUNE trace du sujet qu'il transportait : le contenu splice n'est pas lu, le PROMPT
# repeint A. Mais tous les prompts de retour testes jusqu'ici DECRIVENT le sujet en entier ("a
# brown dog with a white chest running in a green park") : le texte SUFFISAIT, la memoire n'avait
# rien a apporter. Hypothese d'Aden : le canal lit peut-etre le contenu quand le texte NE PEUT
# PAS repeindre le sujet. Deux cellules construites pour retirer cette suffisance :
#
#   17 "generique"  A = un cheval aux attributs ANTI-STEREOTYPES (robe noire de jais, criniere
#                   BLANCHE : l'inverse de la correlation apprise) ; le prompt de RETOUR ne
#                   contient AUCUN attribut ("the animal from earlier returns to the meadow").
#                   Si la robe noire et la criniere blanche reviennent, elles ne peuvent pas
#                   venir du texte du retour.
#   18 "ambigue"    A = un chien BRUN dans un parc, B = un chien BLANC dans LE MEME parc (meme
#                   categorie, meme decor, seule la couleur separe) ; retour = "the first dog
#                   comes back". Le texte ne peut pas choisir entre les deux chiens vus : seule
#                   une memoire du QUEL desambiguise. C'est un test a 2 alternatives, donc le
#                   hasard vaut 50 %.
#
# Les deux tournent sur la forme S2p (X -> A -> B -> A-paraphrase), donc le prompt de retour est
# l'entree 3 de la liste = la "paraphrase" (ici volontairement APPAUVRIE, pas enrichie).
# ------------------------------------------------------------------------------------------
TEXTPOOR_TRIPLES = [
    # 17: attributs saillants NON deductibles du texte de retour
    ("a horse with an unusual jet-black coat and a long snow-white mane standing in a meadow",
     "a wooden windmill turning slowly on a green hillside",
     "a shepherd in a grey coat leaning on a fence at the edge of the meadow"),
    # 18: interlude de la MEME categorie, seule la couleur distingue
    ("a brown dog with floppy ears sitting on the grass in a city park",
     "a white dog with pointed ears sitting on the grass in the same city park",
     "a jogger in an orange headband stretching on the grass in the same city park"),
]
TEXTPOOR_PARAPHRASES = [
    "the animal from earlier returns to the meadow",
    "the first dog comes back",
]
# ---- MINI-ROUND CORRIGE (2026-08-03, apres le verdict oeil d'Aden sur le pilote) ------------
# LA CORRECTION. t17 etait VOID : "the animal from earlier" retire la CATEGORIE, les 5 bras ont
# rendu une VACHE et il n'y avait aucun cheval a retrouver. Regle de design qui en sort :
# appauvrir l'ATTRIBUT, JAMAIS la categorie. t19 = t17 avec ce seul correctif (A, B, C
# VERBATIM identiques, seul le prompt de retour change) -- la duplication des 3 prompts est donc
# VOULUE et declaree (TEXTPOOR_RERUN), c'est ce qui rend les deux cellules comparables.
# LES 3 CELLULES AJOUTEES sont du type t18, le seul qui a produit un signal : ambiguite dans la
# MEME categorie, MEME decor, attributs OPPOSES et de couleur FRANCHE (pour le juge couleur), et
# un ouvreur qui partage le decor. Categories variees : chat, vehicule, oiseau.
TEXTPOOR_TRIPLES += [
    # 19 = 17 corrige : la CATEGORIE reste dans le prompt de retour
    ("a horse with an unusual jet-black coat and a long snow-white mane standing in a meadow",
     "a wooden windmill turning slowly on a green hillside",
     "a shepherd in a grey coat leaning on a fence at the edge of the meadow"),
    # 20: chats, noir vs blanc
    ("a black cat with green eyes sitting on a stone wall in a sunny garden",
     "a white cat with blue eyes sitting on the same stone wall in the same sunny garden",
     "a gardener in a straw hat watering flowers beside the same stone wall in the same garden"),
    # 21: pick-up, rouge vs bleu
    ("a red pickup truck parked on a gravel driveway in front of an old barn",
     "a blue pickup truck parked on the same gravel driveway in front of the same old barn",
     "a farmer in green overalls carrying a wooden crate across the same gravel driveway"),
    # 22: perroquets, jaune vs vert
    ("a yellow parrot perched on a metal railing on a city balcony",
     "a green parrot perched on the same metal railing on the same city balcony",
     "a woman in a red apron watering potted plants on the same city balcony"),
]
TEXTPOOR_PARAPHRASES += [
    "the horse comes back to the meadow",
    "the first cat comes back",
    "the first pickup truck comes back",
    "the first parrot comes back",
]
# cellule -> cellule dont elle est la reprise corrigee (duplication de prompts ASSUMEE)
TEXTPOOR_RERUN = {19: 17}
# cellules d'AMBIGUITE meme-categorie -> (attribut de A, attribut de B, nom de la categorie)
# = le juge couleur/attribut du mini-round (score_textpoor2.py) et son selftest.
TEXTPOOR_ATTR = {
    18: ("brown", "white", "dog"),
    20: ("black", "white", "cat"),
    21: ("red", "blue", "pickup truck"),
    22: ("yellow", "green", "parrot"),
}
assert len(TEXTPOOR_PARAPHRASES) == len(TEXTPOOR_TRIPLES)
TEXTPOOR_TRIPLE_IDS = list(range(len(TRIPLES), len(TRIPLES) + len(TEXTPOOR_TRIPLES)))
TRIPLES = TRIPLES + TEXTPOOR_TRIPLES
PARAPHRASES = PARAPHRASES + TEXTPOOR_PARAPHRASES
assert len(PARAPHRASES) == len(TRIPLES)


def samescene_suffix(triple_idx):
    """The VERBATIM scene clause shared by every prompt of a same-scene triple ('' if the triple
    is not a same-scene one). Used by selftest (ss) and by the eyepack captions."""
    if triple_idx not in SAMESCENE_TRIPLE_IDS:
        return ""
    words = [p.split() for p in TRIPLES[triple_idx]]
    k = 0
    while all(len(w) > k and w[-1 - k] == words[0][-1 - k] for w in words):
        k += 1
    return " ".join(words[0][len(words[0]) - k:])


# every prompt string the scenario campaign can put in front of the model = the level-1
# held-out set of the writer/decoder training corpus (checked by selftest (f)).
def scen_prompts():
    return [p for t in TRIPLES for p in t] + list(PARAPHRASES)


def _norm_text(s):
    return re.sub(r"\s+", " ", s.strip().lower())


# ------------------------------------------------------------------------------------------
# SCHEDULES.  schedule = [(start_frame, role), ...]  with role 0 = A, 1 = B, 2 = C.
# ------------------------------------------------------------------------------------------
SCENARIOS = {
    "S1": {"frames": 240, "schedule": [(0, 0), (81, 1), (162, 0)],
           "desc": "legacy A->B->A (subject OPENS the video); bridge to the l1 campaign"},
    "S2": {"frames": 240, "schedule": [(0, 2), (60, 0), (120, 1), (180, 0)],
           "desc": "NON-OPENING subject: C -> A -> B -> A, re-ID of A vs its 2nd segment"},
    "S3": {"frames": 300, "schedule": [(0, 0), (75, 1), (150, 2), (225, 0)],
           "desc": "multi-switch A -> B -> C -> A, two interludes before the revisit"},
    "S4": {"frames": 300, "schedule": [(0, 0), (60, 1), (120, 0), (180, 2), (240, 0)],
           "desc": "DOUBLE revisit A -> B -> A -> C -> A, re-ID scored at BOTH returns"},
}
# "p" variants: identical schedules, but every REVISIT is asked for with the PARAPHRASE of A
# (prompt index 3) while its ground-truth subject stays 0. Same conditioning for every arm, so
# the arm comparison is untouched; the only thing that changes is that the frame-bank router
# now has a non-trivial matching job (see PARAPHRASES above).
for _n in ("S2", "S3", "S4"):
    _sc = SCENARIOS[_n]
    _sub = [r for _, r in _sc["schedule"]]
    _fa = {}
    for _i, _r in enumerate(_sub):
        _fa.setdefault(_r, _i)
    SCENARIOS[_n + "p"] = {
        "frames": _sc["frames"],
        "schedule": [(f, (3 if (i > 0 and _fa[r] < i) else r))
                     for i, (f, r) in enumerate(_sc["schedule"])],
        "subjects": _sub,
        # measured by prompt_key_matrix.py: cos(A', A) = 0.665-0.780 while the wrong scenes of
        # the same triple reach 0.4690-0.7218. 0.90 (the verbatim default) would make the router
        # NEVER recall in a p-variant, so these scenarios carry their own threshold.
        "fb_tau": 0.68,
        "desc": _sc["desc"] + " -- revisits asked for with a PARAPHRASE (real routing job)",
    }
del _n, _sc, _sub, _fa, _i, _r

# ------------------------------------------------------------------------------------------
# SAME-SCENE scenarios. Schedules are the S2 / S1 shapes verbatim; what changes is the PROMPT
# TIER they are meant to be run on (SAMESCENE_TRIPLE_IDS), where all the segments share one
# decor and only the subject changes. Separate names (not just "S2 on triples 12-14") so the
# mp4s, the score JSONs and the summaries of the two tiers never mix.
#
#   SS2  X(0-60) A(60-120) B(120-180) A(180-240)   THE pilot: subject A does NOT open, and the
#        opener X is now the SAME PLACE with a third subject -> the attention sink can no longer
#        buy the score with the background, only the subject identity is at stake.
#   SS1  A(0-81) B(81-162) A(162-240)              same-scene version of the legacy S1 shape.
#        Its segment 0 is also the CLEAN A* of the SS2 cell of the same (triple, seed): A opens,
#        nothing precedes it, cell_seed does not depend on the scenario -- the same trick
#        rescore_clean.py uses on S2 with S3.
# ------------------------------------------------------------------------------------------
SCENARIOS["SS2"] = {
    "frames": 240, "schedule": [(0, 2), (60, 0), (120, 1), (180, 0)],
    "triples": SAMESCENE_TRIPLE_IDS,
    "desc": "SAME DECOR, subject-only switches: X -> A -> B -> A (A does NOT open), "
            "re-ID of A vs its 2nd segment",
}
SCENARIOS["SS1"] = {
    "frames": 240, "schedule": [(0, 0), (81, 1), (162, 0)],
    "triples": SAMESCENE_TRIPLE_IDS,
    "desc": "SAME DECOR, subject-only switches: A -> B -> A (A opens); segment 0 doubles as the "
            "clean A* reference of the SS2 cell with the same triple and seed",
}


# default routing threshold per scenario (overridable with --fb-tau)
def fb_tau(scen, override=None):
    if override is not None:
        return float(override)
    return float(SCENARIOS[scen].get("fb_tau", 0.90))


def seg_bounds(scen):
    """[(start, end)] latent-frame bounds of every segment of a scenario."""
    sc = SCENARIOS[scen]
    starts = [f for f, _ in sc["schedule"]]
    return [(starts[i], (starts[i + 1] if i + 1 < len(starts) else sc["frames"]))
            for i in range(len(starts))]


def seg_roles(scen):
    """GROUND-TRUTH subject per segment (== the prompt index unless the scenario paraphrases)."""
    sc = SCENARIOS[scen]
    return list(sc.get("subjects") or [r for _, r in sc["schedule"]])


def seg_prompt_idx(scen):
    return [p for _, p in SCENARIOS[scen]["schedule"]]


def first_appearance(scen):
    """subject -> index of the segment where it FIRST appears."""
    fa = {}
    for i, r in enumerate(seg_roles(scen)):
        fa.setdefault(r, i)
    return fa


def revisit_segments(scen):
    """Segment indices that re-enter an already-seen subject = where re-ID is measured."""
    fa = first_appearance(scen)
    return [i for i, r in enumerate(seg_roles(scen)) if fa[r] < i]


def schedule_for(scen, triple_idx):
    """(prompts, schedule, subjects) ready for
    SwitchCausalInferencePipeline.inference_schedule. `prompts` always carries the paraphrase
    at index 3, so the prompt indices mean the same thing in every scenario."""
    prompts = list(TRIPLES[triple_idx]) + [PARAPHRASES[triple_idx]]
    return prompts, [tuple(x) for x in SCENARIOS[scen]["schedule"]], seg_roles(scen)


# ------------------------------------------------------------------------------------------
# REVIEWER CONTROLS (2026-08-03): where the DECODED anchor of a cell is persisted, and which
# cell's anchor the `wronganchor` arm of a given cell reads back.
# The shift is a DETERMINISTIC +1 INSIDE the prompt tier (cross-scene triples 0-11 among
# themselves, same-scene triples among themselves), same seed: every cell gives its anchor to
# exactly one other cell and receives exactly one, no cell can ever read its own.
# ------------------------------------------------------------------------------------------
DEC_ANCHOR_DIR = "/shared/ssd_14T_project/adenelmi/reid_scen/dec_anchors"
N_CROSS_TRIPLES = 12       # les 12 triples cross-scene d'origine (les tiers suivants sont ajoutes
                           # a la fin de TRIPLES et ont chacun leur propre permutation)


def wrong_anchor_triple(triple_idx):
    """The triple whose DECODED anchor the `wronganchor` arm of `triple_idx` splices."""
    for tier in (SAMESCENE_TRIPLE_IDS, TEXTPOOR_TRIPLE_IDS):
        if triple_idx in tier:
            assert len(tier) >= 2, "a tier of 1 triple has no other cell to borrow from"
            return tier[(tier.index(triple_idx) + 1) % len(tier)]
    assert 0 <= triple_idx < N_CROSS_TRIPLES, f"unknown triple {triple_idx}"
    return (triple_idx + 1) % N_CROSS_TRIPLES


def revisit_start_frames(scen):
    """Latent start frame of every revisit segment = every frame where a recall hook fires."""
    starts = [f for f, _ in SCENARIOS[scen]["schedule"]]
    return [starts[i] for i in revisit_segments(scen)]


def dec_anchor_base(scen, triple_idx, seed_idx, dec_dir=None):
    """MEM_DECODE_DUMP base path of a cell (the hook appends '.f<switch frame>.pt')."""
    return os.path.join(dec_dir or DEC_ANCHOR_DIR,
                        f"{scen}_t{triple_idx}_s{seed_idx}_decanchor")


def wrong_anchor_src(scen, triple_idx, seed_idx, dec_dir=None):
    """The .pt the `wronganchor` arm of cell (scen, triple, seed) must splice."""
    fr = revisit_start_frames(scen)
    assert len(fr) == 1, \
        f"wronganchor is defined for a single-revisit scenario (got {len(fr)} in {scen})"
    other = wrong_anchor_triple(triple_idx)
    assert other != triple_idx, "wronganchor would read its own anchor"
    return f"{dec_anchor_base(scen, other, seed_idx, dec_dir)}.f{fr[0]}.pt"


# ------------------------------------------------------------------------------------------
# ARM TABLE.  Derived from run_reid_probe._OFF so it stays in sync: every arm re-sets EVERY
# key of the union, hence nothing can leak from one arm to the next (selftest (c)).
# ------------------------------------------------------------------------------------------
_OFF_SCEN = dict(_OFF, FRAMEBANK="", FRAMEBANK_TAU="0.90", FRAMEBANK_TIE="recent",
                 MEM_DECODE_ROUTED="0",
                 # composite splice (arms oracle1comp / memdec21c, pilot 2026-08-02)
                 MEM_DECODE_COMPOSITE="0", MEM_COMP_CROP="0.6", MEM_COMP_FEATHER="0.3",
                 MEM_COMP_DUMP="",
                 # --- 2026-08-02 gates -------------------------------------------------------
                 # gate 1: additive (no-eviction) k/v channel
                 MEM_ADDITIVE_KV="0", MEM_ADD_GAIN="1", MEM_ADD_POS="auto",
                 # gate 2: window convention / anchor placement of the MEM_DECODE splice
                 MEM_DECODE_SLOT_OFFSET="-1",
                 # gate 3: anti-illusion controls of the composite mask
                 MEM_COMP_UNIFORM="0", MEM_COMP_ANCHOR_NOISE="0", MEM_COMP_SHIFT="0",
                 MEM_COMP_INVERT="0",
                 # gate 4: corrected composite (union mask + prompt-cosine dose)
                 MEM_COMP_UNION="0", MEM_COMP_ALPHA="1", MEM_COMP_PROMPT_GATE="0",
                 MEM_COMP_GATE_LO="0.50", MEM_COMP_GATE_HI="0.80",
                 # remplissage de la region "sujet courant" du masque union (verdict oeil 03/08)
                 MEM_COMP_FILL="", MEM_COMP_FILL_SIGMA="1.0",
                 MEM_COMP_FILL_FLOOR="0.4",
                 # --- LOT OFFENSIF 2026-08-04 -----------------------------------------------
                 # I1 : l'ancre est bruitee a t* ET declaree a t* (fin du mensonge de metadonnee)
                 MEM_ANCHOR_T="0", MEM_ANCHOR_T_SEED="77400241",
                 # T1 : masse d'attention par source [sink | fenetre | ancre] du canal additif
                 MEM_ADD_MASS_SRC="0", MEM_ADD_MASS_DUMP="")

SCEN_ARM_ENV = {
    # paired causal twin: silent memory, no recall of any kind
    "memzero": dict(_OFF_SCEN, MEM_ZERO_ALL="1"),
    # memory 2.0 (writer A2 + decoder A2): the decoded anchor is spliced at EVERY revisit.
    # ckpt paths injected at run time (see run_reid_scen.run_arm).
    "memdec20": dict(_OFF_SCEN, MEM_ZERO_ALL="1"),
    # SAME arm, but WHEN it splices is decided by the framebank router instead of the schedule's
    # ground-truth revisit flag -> memdec20r and framebank differ ONLY by the spliced content.
    # Additive: memdec20 keeps its (verified correct) ground-truth gating, so the 144 mp4s of the
    # first campaign stay valid and comparable.
    "memdec20r": dict(_OFF_SCEN, MEM_ZERO_ALL="1", MEM_DECODE_ROUTED="1"),
    # Echo-Forcing-style REAL-content baseline: growing scene pool, prompt-cosine routing
    "framebank": dict(_OFF_SCEN, MEM_ZERO_ALL="1", FRAMEBANK="key"),
    # PILOTE TEXTE-INSUFFISANT (2026-08-03) : memes bras, mais le QUAND du splice vient de la
    # VERITE DU PLANNING (FRAMEBANK_GT=1) -- sinon, avec un prompt de retour appauvri, le cosinus
    # de prompt s'effondre, la banque ne se declenche jamais et le bras degenere en memzero SANS
    # LE DIRE. La SELECTION reste celle de la methode (argmax) pour `framebank_gt`, la verite
    # terrain pour `framebank_oracle_gt` (plafond). Le routeur tourne quand meme et sa decision
    # est loguee : c'est le resultat "le routage par mots meurt en regime generique".
    "framebank_gt": dict(_OFF_SCEN, MEM_ZERO_ALL="1", FRAMEBANK="key", FRAMEBANK_GT="1"),
    "framebank_oracle_gt": dict(_OFF_SCEN, MEM_ZERO_ALL="1", FRAMEBANK="oracle",
                                FRAMEBANK_GT="1"),
    # ceiling of the same pool: perfect (ground-truth) selection
    "framebank_oracle": dict(_OFF_SCEN, MEM_ZERO_ALL="1", FRAMEBANK="oracle"),
    # upper bound: the TRUE last frame of the FIRST-APPEARANCE segment
    "oracle1": dict(_OFF_SCEN, MEM_ZERO_ALL="1", ORACLE_SPLICE="1", ORACLE_SPLICE_FRAMES="1"),
    # ---- COMPOSITE SPLICE (pilot 2026-08-02, Aden's eye verdict on the same-scene tier) -------
    # oracle1comp = CEILING of the concept: the anchor is still the TRUE frame of A's first
    #   appearance (exactly what oracle1 splices) but only its SUBJECT region survives; the border
    #   comes from the last true frame of the current context. oracle1comp vs oracle1 answers
    #   "does compositing cost identity, and does it buy background continuity?" with the content
    #   held perfect -- i.e. it isolates the COMPOSITING from the decoder's imperfections.
    "oracle1comp": dict(_OFF_SCEN, MEM_ZERO_ALL="1", ORACLE_SPLICE="1", ORACLE_SPLICE_FRAMES="1",
                        MEM_DECODE_COMPOSITE="1"),
    # same, WIDER subject window (75 % instead of 60 %): the gate at 0.6 gives the anchor only
    # 29 % of the area, so a subject that overflows the ellipse loses identity. One sweep point
    # tells whether the identity cost of compositing is a mask-SIZE artifact or intrinsic.
    "oracle1comp75": dict(_OFF_SCEN, MEM_ZERO_ALL="1", ORACLE_SPLICE="1", ORACLE_SPLICE_FRAMES="1",
                          MEM_DECODE_COMPOSITE="1", MEM_COMP_CROP="0.75"),
    # memdec21c = OUR decoded anchor, composited the same way (memdec20 + the composite hook).
    "memdec21c": dict(_OFF_SCEN, MEM_ZERO_ALL="1", MEM_DECODE_COMPOSITE="1"),
    "memdec21c75": dict(_OFF_SCEN, MEM_ZERO_ALL="1", MEM_DECODE_COMPOSITE="1",
                        MEM_COMP_CROP="0.75"),
    # ---- GATE 1 (2026-08-02): the ADDITIVE recall channel, oracle CONTENT -------------------
    # oracle1add* splices NOTHING: the true anchor frame of the first appearance is injected as
    # 1560 EXTRA k/v tokens per block, in the free RoPE band just before the local window, so the
    # 12 true frames of the window survive untouched. Compared to oracle1 (which evicts one true
    # frame to place the same content), this separates "the anchor's CONTENT is readable" from
    # "the anchor OWNS the recency slot". Three attention-mass gains, one arm each, so every arm
    # keeps re-setting every key of the union.
    "oracle1add05": dict(_OFF_SCEN, MEM_ZERO_ALL="1", MEM_ADDITIVE_KV="1", MEM_ADD_GAIN="0.5"),
    "oracle1add1": dict(_OFF_SCEN, MEM_ZERO_ALL="1", MEM_ADDITIVE_KV="1", MEM_ADD_GAIN="1"),
    "oracle1add2": dict(_OFF_SCEN, MEM_ZERO_ALL="1", MEM_ADDITIVE_KV="1", MEM_ADD_GAIN="2"),
    # point de DOSE FORCEE: mesure (sonde de masse) -- a gain 1 les tokens d'ancre ne prennent que
    # 2.0 % de la masse d'attention (7.7 % s'ils etaient lus uniformement), donc {0.5, 1, 2} ne
    # balaie que 1-4 %. gain 8 monte a ~14 %, plus que la part de tokens qu'ils occupent : il
    # separe "canal mort parce que NON LU" de "canal mort parce que ILLISIBLE".
    "oracle1add8": dict(_OFF_SCEN, MEM_ZERO_ALL="1", MEM_ADDITIVE_KV="1", MEM_ADD_GAIN="8"),
    # ---- GATE 2 (2026-08-02): window convention + anchor placement --------------------------
    # memdec20s0 = memdec20 with the ORACLE window convention (the anchor REPLACES the most
    # recent true frame instead of being appended after it) -> memdec20s0 and oracle1 differ by
    # the anchor CONTENT only, at last. s1 / s2 push the anchor 1 / 2 slots back so the true
    # frames keep the recency (the model's immediate declared past stays real).
    "memdec20s0": dict(_OFF_SCEN, MEM_ZERO_ALL="1", MEM_DECODE_SLOT_OFFSET="0"),
    "memdec20s1": dict(_OFF_SCEN, MEM_ZERO_ALL="1", MEM_DECODE_SLOT_OFFSET="1"),
    "memdec20s2": dict(_OFF_SCEN, MEM_ZERO_ALL="1", MEM_DECODE_SLOT_OFFSET="2"),
    # ---- GATE 3 (2026-08-02): the anti-illusion controls of the composite mask --------------
    # All four are memdec21c + ONE control, so any difference is attributable to that control.
    #   unif  : no geometry, constant mask at the SAME mean weight (dose-matched)
    #   noise : mask kept, anchor CONTENT destroyed (RMS-preserving pure noise)
    #   shift : the ellipse moved a third of the width sideways (right shape, wrong place)
    #   inv   : the anchor keeps the BORDER and the context keeps the center (mask inverted)
    "memdec21c_unif": dict(_OFF_SCEN, MEM_ZERO_ALL="1", MEM_DECODE_COMPOSITE="1",
                           MEM_COMP_UNIFORM="1"),
    "memdec21c_noise": dict(_OFF_SCEN, MEM_ZERO_ALL="1", MEM_DECODE_COMPOSITE="1",
                            MEM_COMP_ANCHOR_NOISE="1.0"),
    "memdec21c_shift": dict(_OFF_SCEN, MEM_ZERO_ALL="1", MEM_DECODE_COMPOSITE="1",
                            MEM_COMP_SHIFT="0.3333"),
    "memdec21c_inv": dict(_OFF_SCEN, MEM_ZERO_ALL="1", MEM_DECODE_COMPOSITE="1",
                          MEM_COMP_INVERT="1"),
    # ---- GATE 4 (conditional): the CORRECTED composite --------------------------------------
    # union mask (anchor ellipse U current-subject ellipse), crop 0.75 pre-registered, composite
    # dose gated by the prompt cosine (unrelated scene -> alpha 0 -> bitwise memdec20).
    "memdec21u": dict(_OFF_SCEN, MEM_ZERO_ALL="1", MEM_DECODE_COMPOSITE="1", MEM_COMP_CROP="0.75",
                      MEM_COMP_UNION="1", MEM_COMP_PROMPT_GATE="1"),
    # memdec21un = memdec21u + la region du SUJET COURANT remplie de BRUIT au lieu du contenu
    # d'ancre. Verdict oeil d'Aden (03/08) sur memdec21u : le sujet revient bien placé, mais un
    # FANTOME FLOU reste a l'ancienne position du sujet d'interlude -- cette region recevait le
    # passe-bas de l'ancre et le debruiteur ne la nettoie qu'a moitie. Ici elle est effacee, et le
    # prompt + le contexte la repeignent librement.
    "memdec21un": dict(_OFF_SCEN, MEM_ZERO_ALL="1", MEM_DECODE_COMPOSITE="1", MEM_COMP_CROP="0.75",
                       MEM_COMP_UNION="1", MEM_COMP_PROMPT_GATE="1", MEM_COMP_FILL="noise"),
    # memdec21up = PRIORITE au bruit sur le recouvrement : w_n = m_b, w_a = max(0, m_a - m_b).
    # Toute la zone du sujet courant est effacee, Y COMPRIS sous l'ellipse de l'ancre (la ou
    # memdec21un laissait le passe-bas de A par-dessus l'endroit de B = le reste du fantome).
    # Risque a surveiller : si A et B se recouvrent beaucoup, on retire de l'ancre pile ou A doit
    # revenir -> tripwire "le sujet crop60 ne doit pas s'effondrer".
    "memdec21up": dict(_OFF_SCEN, MEM_ZERO_ALL="1", MEM_DECODE_COMPOSITE="1", MEM_COMP_CROP="0.75",
                       MEM_COMP_UNION="1", MEM_COMP_PROMPT_GATE="1", MEM_COMP_FILL="priority"),
    # memdec21uf = 'priority' avec un PLANCHER d'ancre f = 0.4 : efface le fantome sans jamais
    # couper completement le canal d'identite (f = 0 -> priority, f = 1 -> noise).
    "memdec21uf": dict(_OFF_SCEN, MEM_ZERO_ALL="1", MEM_DECODE_COMPOSITE="1", MEM_COMP_CROP="0.75",
                       MEM_COMP_UNION="1", MEM_COMP_PROMPT_GATE="1", MEM_COMP_FILL="floor",
                       MEM_COMP_FILL_FLOOR="0.4"),
    # ---- REVIEWER CONTROLS of the S2 claim (2026-08-03, risks.md attack 1) -------------------
    # Both are memdec20 with the anchor CONTENT replaced and NOTHING else changed: same
    # ground-truth revisit routing, same switch, same recency placement, same legacy window
    # convention, same writer A2, same decoder A2 (the decode still runs -- only its output is
    # discarded), same seed, same noise stream.
    #   noiseanchor : the anchor is PURE NOISE calibrated on the latent statistics of the true
    #                 window it is spliced into (dose 1.0 = zero anchor content). This is the
    #                 reviewer's "a noise splice would score the same" -- and it is literally the
    #                 "prompt-conditioned category prior" of attack 4, implemented.
    #   wronganchor : the anchor is the one decoded from ANOTHER CELL'S state (triple (t+1) % 12,
    #                 same seed; the .pt is dumped by MEM_DECODE_DUMP during that cell's run).
    #                 Same decoder, same fidelity, same statistics, WRONG memory: separates
    #                 "decoding THIS state" from "any plausible decoded frame". MEM_DECODE_SRC is
    #                 injected at run time (like the ckpt paths), so its table row is IDENTICAL to
    #                 memdec20's by construction -- checked by selftest_controls (a).
    "noiseanchor": dict(_OFF_SCEN, MEM_ZERO_ALL="1", MEM_DECODE_NOISE="1.0"),
    "wronganchor": dict(_OFF_SCEN, MEM_ZERO_ALL="1"),
    # ATTRIBUTION "fenetre vs frame" (question ouverte du §11.5) : les memes deux bras a la
    # convention de fenetre d'ORACLE1 (l'ancre REMPLACE la frame de recence au lieu d'etre
    # ajoutee apres elle). Croises avec memdec20s0, ils separent enfin les deux explications
    # sans contenu : "n'importe quelle frame lisse" vs "la fenetre decalee".
    "wronganchor_s0": dict(_OFF_SCEN, MEM_ZERO_ALL="1", MEM_DECODE_SLOT_OFFSET="0"),
    "noiseanchor_s0": dict(_OFF_SCEN, MEM_ZERO_ALL="1", MEM_DECODE_NOISE="1.0",
                           MEM_DECODE_SLOT_OFFSET="0"),
    # ---- MINI-BATTERIE TRAINING-FREE (2026-08-04) -------------------------------------------
    # sinkpurge = AUCUN splice, AUCUNE memoire (= memzero bit a bit jusqu'au retour) ; au recache
    # du retour les slots du sink EPINGLE (l'ouvreur, jamais rafraichi par LongLive) sont reecrits
    # avec la derniere frame vraie courante. C'est le "correctif minimal" qu'un reviewer opposera
    # a notre +0.43 sur les tiers ou le sujet n'ouvre pas la video.
    "sinkpurge": dict(_OFF_SCEN, MEM_ZERO_ALL="1", MEM_SINKPURGE="1"),
    # active memory read with the memory-2.0 writer (optional attribution column)
    "memon20": dict(_OFF_SCEN),
    # external references (own configs, no memory graft)
    "stock": dict(_OFF_SCEN),
    "stockoff": dict(_OFF_SCEN),
}
# ---- sdedit arms (mini-batterie training-free 2026-08-04) ---------------------------------
# `sdedit_ours<tag>` = memdec20 + l'ancre AFFUTEE par le prior du modele (SDEdit latent a sigma,
# mode 1 frame, prompt de RETOUR verbatim) ; `sdedit_wrong<tag>` = EXACTEMENT le meme traitement
# applique a l'ancre decodee d'une AUTRE cellule (= wronganchor affute). Le contraste apparie
# ours - wrong est LE chiffre de la batterie : les deux subissent le meme prior, seule change
# l'origine du contenu. Le sigma retenu vient de la planche d'oeil (regle scellee au registre) ;
# la table porte tous les points candidats pour que le choix soit un choix D'EXECUTION, pas de
# code (et pour qu'aucun bras ne puisse tourner avec un sigma non declare).
SDEDIT_SIGMAS = {"04": "0.4", "05": "0.5", "06": "0.6", "0625": "0.625", "0833": "0.833"}
SDEDIT_ARMS = []
for _tag, _sig in SDEDIT_SIGMAS.items():
    for _who in ("ours", "wrong"):
        _name = f"sdedit_{_who}{_tag}"
        SCEN_ARM_ENV[_name] = dict(_OFF_SCEN, MEM_ZERO_ALL="1", MEM_SDEDIT=_sig)
        SDEDIT_ARMS.append(_name)
del _tag, _sig, _who, _name
SDEDIT_ARMS = tuple(SDEDIT_ARMS)
# l'ancre etrangere de `sdedit_wrong*` vient du meme dump que `wronganchor` (MEM_DECODE_SRC)
SDEDIT_WRONG_ARMS = tuple(a for a in SDEDIT_ARMS if a.startswith("sdedit_wrong"))

# ============================================================================================
# LOT OFFENSIF (2026-08-04) -- predictions scellees dans memory2_probes_registry.md
# ============================================================================================
# ---- LEVIER 2 : I1, l'ancre DECLAREE BRUITEE ------------------------------------------------
# `i1_ours<t>` = memdec20 + l'ancre reellement bruitee a t* ET declaree a t* dans le
# context_timestep de SON slot (le champ est deja par frame). `i1_wrong<t>` = EXACTEMENT le meme
# traitement sur l'ancre decodee d'une AUTRE cellule (= wronganchor + la declaration) : le
# contraste apparie ours - wrong isole le CONTENU, les deux subissent la meme declaration.
I1_TSTARS = {"250": "250", "500": "500"}
I1_ARMS = []
for _tag, _t in I1_TSTARS.items():
    for _who in ("ours", "wrong"):
        _name = f"i1_{_who}{_tag}"
        SCEN_ARM_ENV[_name] = dict(_OFF_SCEN, MEM_ZERO_ALL="1", MEM_ANCHOR_T=_t)
        I1_ARMS.append(_name)
del _tag, _t, _who, _name
I1_ARMS = tuple(I1_ARMS)
I1_WRONG_ARMS = tuple(a for a in I1_ARMS if a.startswith("i1_wrong"))

# ---- LEVIER 3 : I3/T5, l'additif a DELTA = 1 ------------------------------------------------
# Le canal additif du gate 1 place l'ancre en bande RoPE 9 (auto = une frame avant la fenetre) et
# la requete est a 18 : Delta = 9. `MEM_ADD_POS=17` la met a Delta = 1, DANS la bande de recence
# (les positions 10-20 portent deja les 12 vraies frames : le partage de position est inherent au
# test "recence", et rien n'est evince pour autant). Contenu = la VRAIE frame d'ancre (borne du
# canal) ; `memdec20add8p17` y met NOTRE ancre decodee, si et seulement si la borne s'ouvre.
# MEM_ADD_MASS_SRC=1 partout : la masse par source [sink | fenetre | ancre] est LE tripwire.
ADDPOS_ARMS = ("oracle1add1p17", "oracle1add8p17", "oracle1add8p9", "memdec20add8p17")
for _n, _g, _p, _dec in (("oracle1add1p17", "1", "17", False),
                         ("oracle1add8p17", "8", "17", False),
                         # temoin de position a dose egale : le gain 8 deja mesure, a Delta = 9,
                         # re-tourne ICI avec la mesure de masse par source (sinon la comparaison
                         # de masse melangerait deux campagnes).
                         ("oracle1add8p9", "8", "9", False),
                         ("memdec20add8p17", "8", "17", True)):
    SCEN_ARM_ENV[_n] = dict(_OFF_SCEN, MEM_ZERO_ALL="1", MEM_ADDITIVE_KV="1", MEM_ADD_GAIN=_g,
                            MEM_ADD_POS=_p, MEM_ADD_MASS_SRC="1")
del _n, _g, _p, _dec

# ---- LEVIER 1 : la LEGENDE (canal TEXTE) ----------------------------------------------------
# Les 4 bras sont des `memzero` (memoire MUETTE, aucun splice, aucun decodeur) : la SEULE chose
# qui change est le PROMPT DE RETOUR, enrichi d'un suffixe de ~14 mots. C'est le canal prouve du
# projet (le prompt repeint A) qu'on essaie de charger avec le contenu de la memoire.
#   caption_ours   suffixe construit sur NOTRE ancre decodee   (categorie <- prompt, couleurs
#                  MESUREES sur le crop : pas de VLM libre, donc pas d'affabulation possible)
#   caption_wrong  meme suffixe construit sur l'ancre d'une AUTRE cellule  -> LE controle
#                  directionnel : son delta doit etre NEGATIF
#   padded         adjectifs neutres, meme nombre de mots      -> controle de longueur
#   framebank_vlm  suffixe construit sur une VRAIE frame de A  -> plafond
#   caption_none   suffixe VIDE -> prompts[3] == prompts[0] : doit reproduire `memzero` BIT A BIT
# ---- LEVIER 4 : DECODER LE SNAPSHOT DE FIN DE SEGMENT A -------------------------------------
# 16.1 quater a montre que notre ancre est une image de l'INTERLUDE : le decodeur reconstruit la
# frame qui vient d'etre EVINCEE, et au retour c'est B. `memdec_snap` decode l'etat SNAPSHOTTE a
# la fin du segment de A, ou le contenu le plus recemment evince EST A.
# AUCUN CODE NOUVEAU : MEM_EPISODIC=1 snapshotte deja cet etat (inference_schedule) et le RESTAURE
# au revisit AVANT le decodage (_switch_recache, bloc is_revisit) ; `_memdec_frame` lit l'etat
# vivant. Et sous MEM_ZERO_ALL=1 l'injection memoire est mise a ZERO dans les deux passes
# (causal_model.py), donc la restauration ne peut RIEN changer au rendu : son seul consommateur
# est le decodeur. `memdec_snap` est donc `memdec20` avec un autre ETAT en entree, rien d'autre.
SNAP_ARMS = ("memdec_snap",)
SCEN_ARM_ENV["memdec_snap"] = dict(_OFF_SCEN, MEM_ZERO_ALL="1", MEM_EPISODIC="1")

# ---- MEMOIRE 2.1 : `memdec_t0` -- memdec20 AU CARACTERE PRES, avec d'AUTRES POIDS ------------
# Le bras est `memdec20` bit pour bit cote code (meme env : MEM_ZERO_ALL=1, meme chemin de splice,
# meme convention de fenetre) ; la SEULE difference est le couple (ecrivain, decodeur) charge :
# celui re-entraine sur la metrique T0 (gates D0/G1/G2a scelles au registre). C'est exactement la
# comparaison qu'on veut : a mecanisme identique, un colis different.
# Ses checkpoints viennent de --mem2-t0-writer-ckpt / --mem2-t0-dec-ckpt (et NON des ckpts A2 de
# memdec20) pour que les deux bras puissent tourner dans la MEME campagne sans se marcher dessus.
# Son ancre est dumpee sous un nom PROPRE (`_t0anchor`) : ecrire dans `_decanchor` casserait la
# source que lit `wronganchor` (regle du dossier : un bras ne touche jamais le dump d'un autre).
T0_ARMS = ("memdec_t0",)
SCEN_ARM_ENV["memdec_t0"] = dict(_OFF_SCEN, MEM_ZERO_ALL="1")

CAPTION_ARMS = ("caption_ours", "caption_wrong", "padded", "framebank_vlm", "caption_none")
for _n in CAPTION_ARMS:
    SCEN_ARM_ENV[_n] = dict(_OFF_SCEN, MEM_ZERO_ALL="1")
del _n
CAPTION_JSON = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "offensif",
                            "captions.json")
# scenario de legende -> scenario dont il reprend la forme. Le retour pointe l'entree 3 de la
# liste de prompts, dont le texte est "prompt de A VERBATIM + suffixe" (et non la paraphrase).
CAPTION_SCENARIOS = {"S2c": "S2", "SS2c": "SS2"}


def caption_suffix(scen, triple_idx, seed_idx, arm):
    """Suffixe de legende d'une cellule, LU sur disque (mesure hors ligne par
    build_captions.py). Aucune generation de texte au moment du rollout."""
    with open(CAPTION_JSON) as f:
        d = json.load(f)
    k = f"{scen}_t{triple_idx}_s{seed_idx}_{arm}"
    assert k in d, f"legende manquante pour {k} -- lancer build_captions.py d'abord"
    return d[k]


def caption_return_prompt(triple_idx, suffix):
    base = TRIPLES[triple_idx][0]
    return base if not suffix else (base.rstrip(" .") + suffix)

# arms that run on the MAG (l1@2000-EMA) pipeline; the others build their own pipeline
MAG_SCEN_ARMS = ("memzero", "memdec20", "memdec20r", "framebank", "framebank_oracle",
                 "oracle1", "oracle1comp", "oracle1comp75", "memdec21c", "memdec21c75",
                 "memon20",
                 "oracle1add05", "oracle1add1", "oracle1add2", "oracle1add8",
                 "memdec20s0", "memdec20s1", "memdec20s2",
                 "memdec21c_unif", "memdec21c_noise", "memdec21c_shift", "memdec21c_inv",
                 "memdec21u", "memdec21un", "memdec21up", "memdec21uf",
                 "noiseanchor", "wronganchor", "framebank_gt", "framebank_oracle_gt",
                 "wronganchor_s0", "noiseanchor_s0",
                 "sinkpurge") + SDEDIT_ARMS + I1_ARMS + ADDPOS_ARMS + CAPTION_ARMS \
    + SNAP_ARMS + T0_ARMS
# arms that need the memory-2.0 writer + decoder checkpoints (the decode runs even when its
# output is thrown away, so the two control arms keep memdec20's exact code path).
# sinkpurge is NOT here on purpose: it splices nothing and decodes nothing.
MEM2_CKPT_ARMS = (("noiseanchor", "wronganchor", "wronganchor_s0", "noiseanchor_s0")
                  + SDEDIT_ARMS + I1_ARMS)
# les bras qui lisent l'ancre decodee d'une AUTRE cellule (MEM_DECODE_SRC injecte au run time)
WRONG_SRC_ARMS = ("wronganchor", "wronganchor_s0") + SDEDIT_WRONG_ARMS + I1_WRONG_ARMS
# composite arms -> the arm they must differ from by the composite flags ONLY (selftest)
COMPOSITE_BASE = {"oracle1comp": "oracle1", "oracle1comp75": "oracle1",
                  "memdec21c": "memdec20", "memdec21c75": "memdec20",
                  "memdec21c_unif": "memdec20", "memdec21c_noise": "memdec20",
                  "memdec21c_shift": "memdec20", "memdec21c_inv": "memdec20",
                  "memdec21u": "memdec20", "memdec21un": "memdec20", "memdec21up": "memdec20", "memdec21uf": "memdec20"}
STOCK_SCEN_ARMS = ("stock", "stockoff")

# THE recommended matrix arm list per scenario. framebank_oracle is dropped where it is
# provably identical to oracle1 (single revisit + a single prior occurrence of A => the bank
# ceiling and the first-appearance oracle pick the SAME frame; selftest (b) proves it bitwise),
# which saves ~1/6 of the campaign.
DEFAULT_ARMS = {
    # S1-S3 (verbatim revisit prompt): the router's correct match scores exactly 1.0 and the
    # hardest distractor 0.72, so framebank provably picks the first-appearance frame = the
    # frame oracle1 splices => framebank is BITWISE oracle1 there (selftest b). Running both
    # would burn a fifth of the campaign on a duplicate; framebank is kept as the arm actually
    # under test and oracle1 as the (identical) upper bound is dropped, EXCEPT in S4/p-variants.
    "S1": ("stockoff", "memzero", "memdec20", "framebank", "oracle1"),
    "S2": ("stockoff", "memzero", "memdec20", "framebank", "oracle1"),
    "S3": ("stockoff", "memzero", "memdec20", "framebank", "oracle1"),
    # S4: framebank/framebank_oracle recall the FRESHEST A (segment 2 = the first resurrection)
    # while oracle1 keeps the true first appearance -> the three arms genuinely differ.
    "S4": ("stockoff", "memzero", "memdec20", "framebank", "framebank_oracle", "oracle1"),
    # p-variants: the only place a SELECTION error is possible, so the ceiling arm earns its cost
    "S2p": ("stockoff", "memzero", "memdec20", "framebank", "framebank_oracle", "oracle1"),
    "S3p": ("stockoff", "memzero", "memdec20", "framebank", "framebank_oracle", "oracle1"),
    "S4p": ("stockoff", "memzero", "memdec20", "framebank", "framebank_oracle", "oracle1"),
    # SAME-SCENE tier: single revisit + verbatim revisit prompt, so framebank_oracle is again
    # provably framebank (selftest b) and is dropped. The 5 arms are the 5 lines of the report:
    # LongLive (stockoff), no memory (memzero), Echo-Forcing-like bank (framebank), our memory
    # 2.0 (memdec20), upper bound (oracle1).
    "SS2": ("stockoff", "memzero", "memdec20", "framebank", "oracle1"),
    "SS1": ("stockoff", "memzero", "memdec20", "framebank", "oracle1"),
    # LEVIER 1 : les 4 bras de legende + le temoin bit a bit
    "S2c": CAPTION_ARMS,
    "SS2c": CAPTION_ARMS,
}

# ---- SCENARIOS DE LEGENDE (levier 1) --------------------------------------------------------
# Meme forme, meme calendrier, memes sujets que S2 / SS2 : SEUL le segment de RETOUR pointe
# l'entree 3 de la liste de prompts au lieu de l'entree 0. Le texte de cette entree est construit
# au moment du run (`caption_return_prompt`) = prompt de A VERBATIM + suffixe de legende, si bien
# qu'avec un suffixe VIDE le rollout doit etre BIT A BIT le S2/SS2 memzero deja sur disque
# (l'encodeur de texte recoit exactement la meme chaine). C'est le selftest gratuit du levier.
for _c, _b in (("S2c", "S2"), ("SS2c", "SS2")):
    _sc = SCENARIOS[_b]
    _roles = seg_roles(_b)
    _fa = {}
    for _i, _r in enumerate(_roles):
        _fa.setdefault(_r, _i)
    SCENARIOS[_c] = {
        "frames": _sc["frames"],
        "schedule": [(f, (3 if (i > 0 and _fa[r] < i) else r))
                     for i, (f, r) in enumerate(_sc["schedule"])],
        "subjects": _roles,
        "desc": _sc["desc"] + " -- RETOUR conditionne par le prompt de A + une LEGENDE mesuree",
    }
    if "triples" in _sc:
        SCENARIOS[_c]["triples"] = _sc["triples"]
del _c, _b, _sc, _roles, _fa, _i, _r


# ------------------------------------------------------------------------------------------
# SCORING -- re-ID against the FIRST APPEARANCE of the subject (not necessarily the opening).
# ------------------------------------------------------------------------------------------
def score_segments(scen, frames, settle=20, resettle=12):
    """Latent-frame windows actually embedded, per segment. Segment 0 starts from scratch
    (settle), every later segment starts right after a switch (resettle). The last 3 latent
    frames of a segment are dropped (the model is already turning toward the next prompt),
    verbatim convention of run_reid_probe.score_video."""
    out = {}
    for i, (lo, hi) in enumerate(seg_bounds(scen)):
        s = lo + (settle if i == 0 else resettle)
        e = (hi - 3) if hi < frames else (frames - 1)
        assert e - s >= 8, f"{scen} segment {i}: window [{s}, {e}] too short to score"
        out[i] = (s, e)
    return out
