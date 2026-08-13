#!/usr/bin/env python
# ============================================================================================
# ETAPE 1 DU PLAN 04/08 -- SONDAGE DES POIDS RAPIDES : module commun.
#
# Protocole scelle : reference/plan_memoire_2026-08-04.md (sections "tache n 1/masques" et
# "ETAPE 1") + les seuils de fin de reference/memory2_probes_registry.md ("ETAPE 1 DU PLAN
# 04/08 : SONDAGE DES POIDS RAPIDES").
#
# Ce module ne fait AUCUN calcul lourd a l'import. Il definit UNE SEULE FOIS :
#   - les chemins (code ici, artefacts lourds sur le disque PROJET : le home est plein) ;
#   - le corpus (165 videos) et les 30 PAIRES meme-prompt (jumeaux, graine +100000) ;
#   - le SPLIT STRICT PAR PAIRE 18 train / 5 val / 7 test, scelle par une graine fixe ;
#   - l'extraction du NOM DU SUJET depuis le prompt (guide la segmentation, tache n 1) ;
#   - la definition des SEGMENTS-SUJET d'une video (A / B pour les switch, A seul pour les
#     single) et l'echantillonnage des frames latentes dans un segment ;
#   - les ~4 AGES d'etat (dont fin-du-segment-sujet et fin-de-sequence) ;
#   - le determinisme (la signature doit etre bitwise reproductible).
#
# ADDITIF : rien hors de reference/runs/probe1/ n'est modifie. mem2_data.py (machinerie de
# rejeu validee) et run_reid_probe.py (le DINO du harness) sont importes READ-ONLY.
# ============================================================================================
import hashlib
import json
import os
import re
import sys

REPO = "/shared/ssd_14T/home/adenelmi/Anchor-Forcing"
HERE = os.path.dirname(os.path.abspath(__file__))
MEM2_DIR = os.path.join(REPO, "reference/runs/mem2")
PROBE_L1_DIR = os.path.join(REPO, "reference/runs/reid_probe_l1")
FRAME_DEC_DIR = os.path.join(REPO, "reference/runs/frame_dec")
for _p in (REPO, MEM2_DIR, PROBE_L1_DIR, FRAME_DEC_DIR, HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)
os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")

# artefacts lourds : disque PROJET (le home est a 100 %)
HEAVY = "/shared/ssd_14T_project/adenelmi/probe1"
MASK_DIR = os.path.join(HEAVY, "masks")       # masques + crops sujet/fond, par (video, segment)
REFS_DIR = os.path.join(HEAVY, "refs")        # embeddings DINO des crops
STATE_DIR = os.path.join(HEAVY, "states")     # snapshots d'etat fp16
SIG_DIR = os.path.join(HEAVY, "sig")          # batterie de sondes + signatures
BOARD_DIR = os.path.join(HEAVY, "board")      # planches de validation a l'oeil
RUNS_DIR = os.path.join(HEAVY, "runs")        # sorties du sondeur
HF_HOME = os.path.join(HEAVY, "hf")           # modeles de segmentation (disque projet)

# ---------------------------------------------------------------------------------------------
# LES DEUX CORPUS DE PAIRES (v2 : ADDITIF, 2026-08-05)
#   v1 = les 30 paires de la campagne v3 (prompts vidprom, qui DECRIVENT l'individu) ;
#   v2 = les 30 paires a prompts SANS ATTRIBUT (reference/runs/probe1/prompts_noattr_v2.txt),
#        generees par collect_pairs.py --prompts-file (+ --twin-offset 100000).
# Les corpus_idx de v2 valent 9 000 000 + rang : ils ne peuvent JAMAIS entrer en collision avec
# un index de ligne vidprom, donc masques / etats / signatures des deux corpus cohabitent sans
# se marcher dessus dans les memes repertoires (la cle est le video_id).
# ---------------------------------------------------------------------------------------------
V2_DIR = "/shared/ssd_14T_project/adenelmi/prompts_noattr_v2"
V2_TWIN_DIR = V2_DIR + "_twins"
PROMPTS_V2 = os.path.join(HERE, "prompts_noattr_v2.txt")

# corpus v3 (AMENDEMENT C2, nuit 06-07/08) : 21 paires fraiches sans attribut, generees SANS
# h (--no-h : latents/conds/graines seulement ; la collecte h sera transitoire apres la
# labellisation). corpus_idx = 9 100 000 + rang (--prompts-base) : jamais en collision avec
# v2 (9 000 000..29) dans les repertoires partages masks/ sig/ states/ indexes par video_id.
V3P_DIR = "/shared/ssd_14T_project/adenelmi/prompts_noattr_v3"
V3P_TWIN_DIR = V3P_DIR + "_twins"
PROMPTS_V3 = os.path.join(REPO, "reference/runs/frame_dec/prompts_noattr_v3.txt")
# `v3h` = LE MEME corpus v3, re-collecte AVEC les h (transitoire, ~46 GB, purge apres les
# signatures). Memes prompts, meme base 9 100 000, memes graines -> memes video_id, donc les
# masques/refs deja calcules sur les latents de la 1re generation restent valides -- ce qui
# n'est vrai QUE si les latents re-generes sont bit a bit identiques : c'est le CONTROLE
# D'IDENTITE obligatoire (selftest_v3h.py), sans lequel rien ne doit etre score.
V3H_DIR = V3P_DIR + "_h"
V3H_TWIN_DIR = V3H_DIR + "_twins"

# corpus v4 (registre "C2 v3 : MATERIAU v4", scelle 07/08) : 24 paires, sujets ANIMAUX /
# VEHICULES / OBJETS, ZERO humain. base 9 200 000, planches 61..84. `v4h` = le meme corpus
# re-collecte AVEC les h (transitoire), exactement comme v3/v3h.
V4P_DIR = "/shared/ssd_14T_project/adenelmi/prompts_noattr_v4"
V4P_TWIN_DIR = V4P_DIR + "_twins"
V4H_DIR = V4P_DIR + "_h"
V4H_TWIN_DIR = V4H_DIR + "_twins"
PROMPTS_V4 = os.path.join(REPO, "reference/runs/frame_dec/prompts_noattr_v4.txt")
# Corpus dont le sujet n'est JAMAIS humain : le repli "a person" de subject_phrase y est coupe
# (il ne pourrait que poser le masque sur un figurant du decor).
NO_PERSON_CORPORA = ("v4", "v4h")

DEFAULT_CONFIG = os.path.join(PROBE_L1_DIR, "probe_l1.yaml")
DEC_V12 = os.path.join(FRAME_DEC_DIR, "frame_dec_v12/ckpt/dec_step4000.pt")
DEC_A2 = "/shared/ssd_14T_project/adenelmi/mem2_runs/writer/dec_A2_for_memdec.pt"
WRITER_A2 = "/shared/ssd_14T_project/adenelmi/mem2_runs/writer/mem2_writer_A2_final.pt"

# ---------------------------------------------------------------------------------------------
# CONSTANTES SCELLEES (aucune n'est un hyperparametre du sondeur : elles fixent le jeu de
# donnees, pas le modele, et sont donc figees avant tout entrainement).
# ---------------------------------------------------------------------------------------------
SPLIT_SEED = 20260805          # graine du split par paire (18/5/7)
N_TRAIN_PAIRS, N_VAL_PAIRS, N_TEST_PAIRS = 18, 5, 7
FRAMES_PER_SEGMENT = 10        # "~8-12 frames echantillonnees par segment-sujet"
N_AGES = 4                     # "snapshots d'etat a ~4 ages"
N_CENTROIDS = 192              # sondes : centroides k-means des activations h du corpus
N_DEC_QUERIES = 64             # sondes : requetes sous-echantillonnees du decodeur
N_PROBES = N_CENTROIDS + N_DEC_QUERIES
DIM = 1536
LATENT_H, LATENT_W = 60, 104
PIX_H, PIX_W = LATENT_H * 8, LATENT_W * 8      # 480 x 832
DINO_DIM = 768                 # dinov2_vitb14 CLS


# ---------------------------------------------------------------------------------------------
# determinisme
# ---------------------------------------------------------------------------------------------
def set_determinism(seed=0):
    """Le contrat 'la signature est deterministe (2 appels = bitwise)' n'exige pas des
    algorithmes deterministes au sens de torch (les matmuls cuBLAS a forme fixe le sont deja),
    mais il exige que RIEN d'aleatoire n'entre dans le chemin : on fige donc les RNG, on coupe
    TF32 (qui ne change pas le determinisme mais change la valeur selon la carte) et on desactive
    le benchmark cuDNN (qui peut selectionner un algo different d'un appel a l'autre)."""
    import random

    import numpy as np
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


# ---------------------------------------------------------------------------------------------
# corpus + paires + split
# ---------------------------------------------------------------------------------------------
def corpus_dirs(corpus):
    import mem2_data as D
    if corpus == "v1":
        return (D.V3_DIR, D.TWIN_DIR)
    if corpus == "v2":
        return (V2_DIR, V2_TWIN_DIR)
    if corpus == "v3":
        return (V3P_DIR, V3P_TWIN_DIR)
    if corpus == "v3h":
        return (V3H_DIR, V3H_TWIN_DIR)
    if corpus == "v4":
        return (V4P_DIR, V4P_TWIN_DIR)
    if corpus == "v4h":
        return (V4H_DIR, V4H_TWIN_DIR)
    raise ValueError(f"corpus inconnu {corpus!r} (v1 | v2 | v3 | v3h | v4 | v4h)")


def load_index(verbose=False, corpus="v1", rebuild=False):
    """Index des videos du corpus demande. v1 passe par le cache PARTAGE de mem2_data (le
    fichier que le trainer d'ecrivain utilise : on ne le touche pas) ; v2 a son propre cache
    dans le repertoire lourd de probe1, pour que la campagne v2 n'entre JAMAIS dans l'index du
    trainer."""
    import mem2_data as D
    if corpus == "v1":
        return D.load_index(verbose=verbose, rebuild=rebuild)
    path = os.path.join(HEAVY, f"index_{corpus}.json")
    return D.load_index(path=path, dirs=corpus_dirs(corpus), rebuild=rebuild, verbose=verbose)


def vids_by_id(vids):
    return {v["video_id"]: v for v in vids}


def build_pairs(vids):
    """Les PAIRES meme-prompt : (video principale, jumelle graine+100000). Une paire est la
    seule unite du split (un jumeau du bon cote suffirait a fuir la scene exacte, red-team
    F1.1b) et la seule unite de decision du sondeur (hasard = 50 %).
    Retourne une liste triee par corpus_idx : {pair_id, corpus_idx, main, twin, prompt}."""
    import mem2_data as D
    tidx = D.twin_index(vids)
    byid = vids_by_id(vids)
    pairs = []
    for cidx in sorted(tidx):
        e = tidx[cidx]
        if e["main"] is None or e["twin"] is None:
            continue
        m, t = byid[e["main"]], byid[e["twin"]]
        assert m["prompt"] == t["prompt"], f"paire {cidx}: prompts differents"
        assert t["seed"] != m["seed"], f"paire {cidx}: meme graine"
        pairs.append({"pair_id": len(pairs), "corpus_idx": int(cidx),
                      "main": m["video_id"], "twin": t["video_id"],
                      "mode": m["mode"], "prompt": m["prompt"]})
    return pairs


def assign_pair_split(pairs, seed=SPLIT_SEED):
    """Split STRICT PAR PAIRE, scelle par une graine fixe : 18 train / 5 val / 7 test.
    Deterministe et independant de l'ordre d'arrivee : la cle de tri est un hash de
    (graine, corpus_idx), donc ajouter une paire ne redistribue pas les autres au hasard --
    elle s'insere a sa place de hash. Aucun element du test n'est jamais vu ailleurs."""
    def key(p):
        h = hashlib.sha1(f"{seed}|{p['corpus_idx']}".encode()).hexdigest()[:12]
        return int(h, 16)
    order = sorted(pairs, key=key)
    n_need = N_TRAIN_PAIRS + N_VAL_PAIRS + N_TEST_PAIRS
    if len(order) < n_need:
        # corpus COURT (v3 : 21 paires) : le split fixe 18/5/7 n'existe pas pour lui. La
        # procedure PRIMAIRE est de toute facon la CV par paires (`set_fold` reecrit le split
        # pli par pli) ; le split fixe n'est qu'un heritage v1. Tout part en 'train' -- AUCUN
        # verdict ne se lit sur ce split-la. v1/v2 (30 paires) passent par le chemin historique
        # ci-dessous, inchange.
        return {int(p["corpus_idx"]): "train" for p in order}
    assert len(order) >= n_need, f"{len(order)} paires, il en faut {n_need}"
    split = {}
    for i, p in enumerate(order):
        if i < N_TRAIN_PAIRS:
            s = "train"
        elif i < N_TRAIN_PAIRS + N_VAL_PAIRS:
            s = "val"
        elif i < n_need:
            s = "test"
        else:
            s = "unused"
        split[p["corpus_idx"]] = s
    return split


def pair_dataset(verbose=False, corpus="v1"):
    """Le jeu de donnees du sondeur : paires + split + les deux video_id de chaque paire.
    Le split fixe 18/5/7 reste calcule (compatibilite / selftests), mais la procedure PRIMAIRE
    scellee le 04/08 au soir est la VALIDATION CROISEE PAR PAIRES : voir train_probe --stage cv.
    La numerotation de PLANCHE (celle des listes de l'oeil d'Aden) est posee ici, une fois."""
    vids = load_index(verbose=verbose, corpus=corpus)
    pairs = build_pairs(vids)
    split = assign_pair_split(pairs)
    # chaque corpus a sa plage de numeros de planche, en CONTINUITE, pour que l'oeil d'Aden ne
    # puisse jamais confondre deux planches : v1/v2 = 01-30, v3 = 31-51, v4 = 61-84. Un corpus
    # et sa variante `*h` (meme materiau re-collecte avec les h) partagent leur numerotation.
    start = 1
    if corpus in ("v3", "v3h"):
        start = 31
    elif corpus in ("v4", "v4h"):
        start = 61
    nums = board_numbers(pairs, start=start)
    for p in pairs:
        p["split"] = split[p["corpus_idx"]]
        p["board_no"] = nums[p["corpus_idx"]]
        p["corpus"] = corpus
    return vids, pairs


# ---------------------------------------------------------------------------------------------
# NUMEROTATION DE PLANCHE + SOUS-ENSEMBLES SCELLES (registre, 2026-08-04/05)
# ---------------------------------------------------------------------------------------------
def board_numbers(pairs, start=1):
    """{corpus_idx: numero start..} = l'ordre des cellules de `masks_board_pairs_numerote.jpg`,
    c'est-a-dire l'ordre LEXICOGRAPHIQUE de l'identifiant de la video principale ('main:10440',
    'main:111706', 'main:14984', ...). C'est cette numerotation, et elle seule, que designent
    les listes de l'oeil d'Aden scellees au registre ; elle est verifiee par selftest (T18 :
    01=pompier, 02=bouteille, 06=foule, 08=verrou, 11=oiseau, 12=athlete, 16=femme, 19=enfant+
    chat, 24=ange, 27=fillette -- les 10 paires qu'Aden a nommees).
    `start` (ADDITIF v3) : premier numero. v1 et v2 gardent start=1 (leurs planches et listes
    d'oeil sont scellees) ; le corpus v3 numerote 31..51, en CONTINUITE avec v2 01-30, pour
    que l'oeil d'Aden ne puisse jamais confondre deux planches."""
    return {p["corpus_idx"]: i for i, p in
            enumerate(sorted(pairs, key=lambda q: str(q["main"])), start=start)}


# LISTE OEIL D'ADEN (registre "LISTE OEIL D'ADEN", figee AVANT lecture de tout score) et
# PLAFOND HUMAIN (registre "PLAFOND HUMAIN", scelle le 05/08) -- en numeros de planche.
ADEN_GOOD_15 = (1, 2, 6, 8, 11, 12, 15, 16, 19, 20, 21, 22, 24, 27, 30)
ADEN_DECISIVE_6 = (1, 2, 11, 12, 19, 27)       # "CLAIREMENT DIFFERENTS" : la borne s'y juge
ADEN_SLIGHT_4 = (6, 8, 16, 24)                 # "LEGEREMENT differents"
ADEN_IDENTICAL_5 = (15, 20, 21, 22, 30)        # "QUASI IDENTIQUES" (plafond humain = 50 %)
# Liste SECONDAIRE "le masque porte bien le protagoniste" de l'agent des masques. Le rapport ne
# l'a PAS figee en indices : il la decrit par une REGLE ("~11 tombent sur un accessoire ou un
# objet de faible confiance : cup, small table, empty table, cozy sofa, glass door, camera,
# small bird, et les paires a score < 0,15"). On applique donc cette regle-la, MECANIQUEMENT,
# sur les .npz : elle rend 21 paires (et non ~19 : l'ecart est declare au rapport, la prose
# comptait aussi des cellules de la planche VARIEE du corpus, pas seulement les 30 paires).
AGENT_BAD_QUERIES = ("cup", "small table", "empty table", "cozy sofa", "glass door", "camera",
                     "small bird")
AGENT_MIN_DET_SCORE = 0.15

# LABELS OEIL D'ADEN, CORPUS v2 (registre "LABELS OEIL D'ADEN, CORPUS v2", scelles le 05/08
# AVANT tout run v2 : c'est la reponse a R2 de l'audit, qui refusait de scorer v2 sans verdict
# d'oeil). En numeros de planche de `masks_board_pairs_numerote_v2.jpg`.
ADEN_V2_INVALID_5 = (1, 3, 7, 8, 10)             # masque sur le mauvais objet -> ecartees
ADEN_V2_STRONG_19 = (5, 6, 9, 11, 12, 13, 14, 15, 16, 17, 18, 22, 23, 24, 25, 26, 27, 28, 29)
ADEN_V2_WEAK_6 = (2, 4, 19, 20, 21, 30)          # "faiblement differents"
ADEN_V2_VALID_25 = tuple(sorted(set(ADEN_V2_STRONG_19) | set(ADEN_V2_WEAK_6)))
assert len(ADEN_V2_VALID_25) == 25 and not (set(ADEN_V2_VALID_25) & set(ADEN_V2_INVALID_5))

# LABELS OEIL D'ADEN, CORPUS v3 (registre "LABELS v3 D'ADEN", scelles le 07/08 AVANT tout
# score v3). En numeros de planche de `masks_board_pairs_numerote_v3.jpg` (31..51).
# Regle scellee : D>=2 ET M>=1 ET Q>=1 = forte ; D1 = valide (faible) ; D0/M0/Q0 = exclue.
ADEN_V3_STRONG_15 = (31, 35, 36, 39, 40, 41, 42, 44, 45, 46, 47, 48, 49, 50, 51)
ADEN_V3_WEAK_3 = (33, 37, 43)                    # D1
ADEN_V3_INVALID_3 = (32, 34, 38)                 # M0 : le masque rate le sujet
ADEN_V3_VALID_18 = tuple(sorted(set(ADEN_V3_STRONG_15) | set(ADEN_V3_WEAK_3)))
assert len(ADEN_V3_VALID_18) == 18 and not (set(ADEN_V3_VALID_18) & set(ADEN_V3_INVALID_3))
assert len(ADEN_V3_STRONG_15) == 15 and len(set(ADEN_V3_STRONG_15) & set(ADEN_V3_WEAK_3)) == 0
# ANALYSE SECONDAIRE PRE-DECLAREE (registre) : re-score des FORTES en excluant les masques
# ASYMETRIQUES {43, 45, 51}. NOTA A ARBITRER (declare au rapport, jamais tranche ici) : le
# registre annonce "12 paires", mais 43 est classe FAIBLE par sa propre regle et n'est donc
# PAS dans les 15 fortes -- l'exclusion mecanique en rend 13. On calcule 13 et on rapporte
# l'ecart ; le seuil >= 10 est celui du registre, sa lecture appartient au coordinateur.
ADEN_V3_ASYM_3 = (43, 45, 51)
ADEN_V3_STRONG_NOASYM = tuple(sorted(set(ADEN_V3_STRONG_15) - set(ADEN_V3_ASYM_3)))

# LABELS OEIL D'ADEN, CORPUS v4 (registre "LABELS v4 D'ADEN", scelles le 07/08 AVANT tout
# score v4). Numeros de planche de `labels_board_v4.jpg` (61..84). Meme regle que v3 :
# D>=2 ET M>=1 ET Q>=1 = forte ; D1 = valide (faible) ; le reste = exclue.
# 5 des 6 exclusions sont des D0 : les deux jumeaux sont IDENTIQUES (le generateur CLONE les
# especes etroites) -- c'est le constat de materiau scelle au registre, pas un defaut de code.
ADEN_V4_STRONG_14 = (62, 65, 66, 68, 70, 72, 73, 74, 75, 78, 79, 80, 82, 84)
ADEN_V4_WEAK_4 = (61, 76, 77, 81)                       # D1
ADEN_V4_INVALID_6 = (63, 64, 67, 69, 71, 83)            # dont 5 D0 (jumeaux identiques)
ADEN_V4_VALID_18 = tuple(sorted(set(ADEN_V4_STRONG_14) | set(ADEN_V4_WEAK_4)))
assert len(ADEN_V4_STRONG_14) == 14 and len(ADEN_V4_VALID_18) == 18
assert not (set(ADEN_V4_VALID_18) & set(ADEN_V4_INVALID_6))
assert len(set(ADEN_V4_VALID_18) | set(ADEN_V4_INVALID_6)) == 24


def agent_subset(pairs):
    """Les paires 'le masque porte le protagoniste' selon la REGLE de l'agent des masques
    (ci-dessus), calculee sur les .npz. Rend un set de corpus_idx."""
    import numpy as np
    keep = set()
    for p in pairs:
        qs, lo = set(), 1.0
        ok = True
        for role in ("main", "twin"):
            f = os.path.join(MASK_DIR, seg_key(p[role], "A") + ".npz")
            if not os.path.exists(f):
                ok = False
                break
            r = np.load(f, allow_pickle=True)
            qs.add(str(r["query_chosen"]))
            lo = min(lo, float(np.median(r["det_scores"])))
        if ok and not (qs & set(AGENT_BAD_QUERIES)) and lo >= AGENT_MIN_DET_SCORE:
            keep.add(int(p["corpus_idx"]))
    return keep


def sealed_subsets(pairs, check_frozen=True):
    """Les sous-ensembles PRE-ENREGISTRES, en corpus_idx. `all` = l'analyse PRIMAIRE (30 paires
    sans filtre) ; les autres sont rapportes A COTE et ne rendent aucun verdict, sauf
    `decisive` sur lequel le registre a RE-ANCRE la borne de validite (>= 5/6).

    VERROU MECANIQUE (audit 2 du 05/08). `agent` est RECALCULE sur les .npz a chaque appel :
    si les masques etaient refaits, le sous-ensemble bougerait en silence APRES le gel. Quand
    `runs/subsets_<corpus>.json` existe (la trace horodatee du gel, ecrite AVANT tout score),
    le recalcul est donc CONFRONTE a ce fichier et toute divergence fait echouer le run --
    c'est l'equivalent, pour les sous-ensembles, du verrou de mtime de l'audit du 04/08."""
    by_no = {p["board_no"]: int(p["corpus_idx"]) for p in pairs}
    out = {"all": {int(p["corpus_idx"]) for p in pairs}}
    is_v1 = all(p.get("corpus", "v1") == "v1" for p in pairs)
    # les listes d'oeil sont des verdicts sur les PLANCHES v1 : elles n'ont aucun sens sur un
    # autre corpus (le corpus v2 aura sa propre planche numerotee et son propre verdict d'oeil).
    is_v2 = all(p.get("corpus", "v1") == "v2" for p in pairs)
    if is_v1 and len(pairs) == 30 and set(by_no) == set(range(1, 31)):
        out["aden15"] = {by_no[n] for n in ADEN_GOOD_15}
        out["decisive"] = {by_no[n] for n in ADEN_DECISIVE_6}
        out["agent"] = agent_subset(pairs)
    # corpus v2 : les listes d'oeil du v2, scellees au registre le 05/08 avant tout run v2.
    # `valid25` est la PRIMAIRE du corpus v2, `strong19` porte sa regle 1 (>= 16/19).
    if is_v2 and len(pairs) == 30 and set(by_no) == set(range(1, 31)):
        out["valid25"] = {by_no[n] for n in ADEN_V2_VALID_25}
        out["strong19"] = {by_no[n] for n in ADEN_V2_STRONG_19}
    # corpus v3 (= v3h, meme materiau) : labels d'oeil scelles le 07/08. `strong15` porte le
    # SEUIL DU VERDICT (>= 12/15) ; `valid18` et `all` sont secondaires ; `strong_noasym`
    # est l'analyse de robustesse pre-declaree (masques asymetriques retires).
    is_v3 = all(p.get("corpus", "v1") in ("v3", "v3h") for p in pairs)
    if is_v3 and len(pairs) == 21 and set(by_no) == set(range(31, 52)):
        out["strong15"] = {by_no[n] for n in ADEN_V3_STRONG_15}
        out["valid18"] = {by_no[n] for n in ADEN_V3_VALID_18}
        out["strong_noasym"] = {by_no[n] for n in ADEN_V3_STRONG_NOASYM}
    # corpus v4 : `strong14` porte LE SEUIL DU VERDICT (>= 11/14, binomial 0.0287).
    is_v4 = all(p.get("corpus", "v1") in ("v4", "v4h") for p in pairs)
    if is_v4 and len(pairs) == 24 and set(by_no) == set(range(61, 85)):
        out["strong14"] = {by_no[n] for n in ADEN_V4_STRONG_14}
        out["valid18"] = {by_no[n] for n in ADEN_V4_VALID_18}
    if check_frozen and pairs:
        corpus = pairs[0].get("corpus", "v1")
        fp = os.path.join(RUNS_DIR, f"subsets_{corpus}.json")
        if os.path.exists(fp):
            fr = load_json(fp)["corpus_idx"]
            assert set(fr) == set(out), \
                (f"{fp}: sous-ensembles geles {sorted(fr)} != recalcules {sorted(out)} -- le "
                 f"gel a ete rompu, RIEN ne doit etre score avant arbitrage")
            for k in sorted(out):
                assert sorted(out[k]) == sorted(fr[k]), \
                    (f"{fp}: le sous-ensemble '{k}' recalcule ({len(out[k])} paires) DIFFERE du "
                     f"gel ({len(fr[k])} paires) -- masques refaits apres le gel ?")
    return out


# ---------------------------------------------------------------------------------------------
# segments-sujet
# ---------------------------------------------------------------------------------------------
def segments(meta):
    """Les segments-sujet d'une video, en FRAMES LATENTES [f0, f1) :
      - single : un seul segment 'A' (tout le rollout), sujet decrit par `prompt` ;
      - switch : 'A' = [0, switch_frame) (sujet de `prompt`), 'B' = [switch_frame, frames)
        (sujet de `prompt_b`).
    Le segment 'A' est celui dont l'identite doit survivre a l'interference : c'est lui que le
    sondeur doit designer."""
    F = int(meta["frames"])
    sf = meta["switch_frame"]
    if sf is None:
        return [{"name": "A", "f0": 0, "f1": F, "prompt": meta["prompt"]}]
    sf = int(sf)
    assert 0 < sf < F, f"switch_frame {sf} hors de [0, {F})"
    return [{"name": "A", "f0": 0, "f1": sf, "prompt": meta["prompt"]},
            {"name": "B", "f0": sf, "f1": F, "prompt": meta["prompt_b"]}]


def sample_segment_frames(f0, f1, k=FRAMES_PER_SEGMENT, warmup=3):
    """k frames latentes reparties uniformement dans [f0+warmup, f1), bornes incluses.
    `warmup` saute le debut du segment (chez les switch le contenu du nouveau sujet met
    quelques frames a s'installer ; chez les single la frame 0 est la frame de bruit initiale
    la plus atypique). Deterministe : aucun RNG."""
    lo = min(f0 + warmup, max(f0, f1 - 1))
    hi = f1 - 1
    if hi <= lo:
        return [int(lo)]
    k = max(1, min(int(k), hi - lo + 1))
    if k == 1:
        return [int((lo + hi) // 2)]
    step = (hi - lo) / (k - 1)
    out = sorted({int(round(lo + i * step)) for i in range(k)})
    return [int(x) for x in out]


# ---------------------------------------------------------------------------------------------
# MASQUE-SUJET -> TOKENS h  (source `hdirect_masked`, CORRECTIF DE BORNE scelle le 05/08 nuit)
# ---------------------------------------------------------------------------------------------
# Pourquoi. La borne `hdirect` poole UNIFORMEMENT tous les tokens d'un chunk : le sujet, qui
# occupe 5-12 % de l'image, y est noye dans le fond. L'instrument reproduisait donc le defaut
# meme qu'il devait mesurer, et il echouait sur les deux corpus la ou un humain reussit. La
# source `hdirect_masked` ne poole QUE les tokens sous le masque-sujet -- masques deja
# calcules (tache n 1), aucun parametre nouveau, aucune ecriture.
#
# Geometrie, verifiee sur le code du modele : le patch d'entree est (1, 2, 2)
# (`causal_model.py` l. 949, Conv3d(kernel=patch, stride=patch)), et les tokens sont poses par
# `u.flatten(2).transpose(1, 2)` (l. 1366) -> index = frame * 1560 + ligne * 52 + colonne, avec
# une grille 30 x 52 = 1560 tokens par frame latente 60 x 104. Le masque, lui, est PIXEL
# (480 x 832) : exactement 16 x 16 pixels par cellule de grille.
GRID_H, GRID_W = 30, 52
TOKENS_PER_FRAME = GRID_H * GRID_W      # 1560
MASK_CELL_THR = 0.5                     # une cellule est "sous le masque" a la MAJORITE
MASK_MIN_FRAC = 0.03                    # plancher : le pooling n'est JAMAIS (quasi) vide


def mask_salience_grid(mask_hw):
    """Masque PIXEL (H, W) booleen -> SALIENCE (1560,) = fraction de pixels sous le masque dans
    chaque cellule de la grille de patches, dans l'ordre des tokens (ligne-majeur)."""
    import numpy as np
    H, W = mask_hw.shape
    assert H % GRID_H == 0 and W % GRID_W == 0, \
        f"masque {H}x{W} non divisible par la grille de tokens {GRID_H}x{GRID_W}"
    m = np.asarray(mask_hw, dtype=np.float32).reshape(GRID_H, H // GRID_H, GRID_W, W // GRID_W)
    return m.mean(axis=(1, 3)).reshape(-1)


_MASK_GRID_CACHE = {}


def video_mask_grids(video_id, use_cache=True):
    """[(frame latente echantillonnee, salience (1560,))] pour TOUS les segments-sujet d'une
    video, trie par frame. Chez les `single` (les 30 paires des deux corpus) il n'y a que le
    segment A ; chez les `switch` (le pre-entrainement) A et B sont pris tous les deux, si bien
    qu'un chunk post-switch recoit le masque du sujet qui est REELLEMENT dedans.
    Seules les frames VALIDES du masque comptent (meme regle que `load_refs`)."""
    if use_cache and video_id in _MASK_GRID_CACHE:
        return _MASK_GRID_CACHE[video_id]
    import numpy as np
    out = []
    for seg in ("A", "B"):
        f = os.path.join(MASK_DIR, seg_key(video_id, seg) + ".npz")
        if not os.path.exists(f):
            continue
        r = np.load(f, allow_pickle=True)
        H, W = [int(x) for x in r["mask_shape"]]
        frames = r["frames"]
        # meme depaquetage que `build_masks.unpack_masks` (np.packbits par ligne, count = H*W)
        masks = np.unpackbits(r["masks_packed"], axis=1, count=H * W)
        masks = masks.reshape(len(frames), H, W).astype(bool)
        valid = r["valid"].astype(bool) if "valid" in r.files else np.ones(len(frames), bool)
        for k in range(len(frames)):
            if not valid[k]:
                continue
            out.append((int(frames[k]), mask_salience_grid(masks[k])))
    assert out, f"aucun masque valide pour {video_id} (lance build_masks.py)"
    out.sort(key=lambda t: t[0])
    if use_cache:
        _MASK_GRID_CACHE[video_id] = out
    return out


def frame_token_mask(sal, min_frac=MASK_MIN_FRAC, thr=MASK_CELL_THR):
    """SALIENCE (1560,) -> (masque booleen (1560,), repli declenche ?).
    Regle : une cellule entre dans le pooling si la MAJORITE de ses pixels est sous le masque
    (c'est le sous-echantillonnage pixel -> grille qui conserve l'aire).
    REPLI, pour ne JAMAIS pooler le vide : si moins de `min_frac` des tokens sont retenus, on
    prend les `min_frac` les plus saillants ; les egalites sont tranchees par l'index, donc le
    repli est deterministe et ne depend d'aucun tirage."""
    import numpy as np
    sal = np.asarray(sal, dtype=np.float32)
    sel = sal >= thr
    k = int(np.ceil(min_frac * sal.size))
    if int(sel.sum()) >= k:
        return sel, False
    order = np.lexsort((np.arange(sal.size), -sal))[:k]
    out = np.zeros(sal.size, dtype=bool)
    out[order] = True
    return out, True


def chunk_frames(meta, i):
    """Les frames latentes du chunk evince par l'ecriture i (sa DERNIERE est `frame_of_write`,
    convention de mem2_data)."""
    import mem2_data as D
    F = int(meta["num_frame_per_block"])
    f1 = int(D.frame_of_write(meta, i))
    return list(range(f1 - F + 1, f1 + 1))


def chunk_token_mask(meta, i, grids, n_tokens=None, min_frac=MASK_MIN_FRAC):
    """Le masque de tokens du chunk evince par l'ecriture i.
    Le chunk porte F frames latentes ; on lui associe le masque de la frame echantillonnee la
    plus PROCHE de son centre (egalites tranchees par l'index de frame : deterministe), et on
    tuile ce masque sur les F frames -- un chunk, un masque, comme scelle au registre.
    Rend (masque booleen (F * 1560,), repli declenche ?).
    `min_frac` (ADDITIF, oracle d'ecriture masquee 06/08) est passe tel quel au repli de
    `frame_token_mask` ; le defaut reproduit l'existant A L'IDENTIQUE. `min_frac=0.0` coupe le
    repli top-k : la selection est alors le SEUIL PUR (MASK_CELL_THR), et un chunk sans token
    sujet rend un masque VIDE -- c'est la regle scellee du bras sujet-seul (ecriture SAUTEE),
    la ou le pooling de lecture, lui, ne doit jamais etre vide."""
    import numpy as np
    fs = chunk_frames(meta, i)
    centre = fs[len(fs) // 2]
    _, sal = min(grids, key=lambda t: (abs(t[0] - centre), t[0]))
    sel, fb = frame_token_mask(sal, min_frac=min_frac)
    nf = len(fs) if n_tokens is None else int(n_tokens) // TOKENS_PER_FRAME
    assert n_tokens is None or int(n_tokens) == nf * TOKENS_PER_FRAME, \
        (f"{meta.get('video_id')}: {n_tokens} tokens n'est pas un multiple de "
         f"{TOKENS_PER_FRAME} (grille {GRID_H}x{GRID_W})")
    return np.tile(sel, nf), fb


# ---------------------------------------------------------------------------------------------
# ages d'etat
# ---------------------------------------------------------------------------------------------
def state_ages(meta, n_ages=N_AGES):
    """Les indices d'ECRITURE ou l'etat est snapshotte, du plus jeune au plus vieux.
    Contrat du plan : "~4 ages, DONT fin-du-segment-sujet ET fin-de-sequence".
      - `end_seq`     = derniere ecriture du rollout. TOUJOURS present. Chez les switch c'est
        l'etat APRES l'interference du second sujet : la condition operationnelle reelle ;
      - `end_subject` = derniere ecriture dont le chunk evince est ENTIEREMENT dans le segment
        A (l'etat vient d'absorber le sujet, aucune interference posterieure). Present
        UNIQUEMENT pour les videos `switch` : chez les single, le sujet occupe tout le rollout,
        `end_subject` et `end_seq` seraient le MEME index et on ne stocke pas deux fois le
        meme etat. (Consequence a garder en tete : les 30 paires jumelles du corpus sont
        toutes `single`, donc la tache du sondeur ne comporte PAS de condition d'interference ;
        c'est une limite du corpus, pas du code, et elle est notee au rapport.)
      - les ages restants sont repartis uniformement (entre les deux reperes chez les switch,
        sur toute la sequence chez les single).
    Retourne une liste de dicts {i, name} avec i = index d'ecriture (0-based), triee, dedupliquee.
    """
    import mem2_data as D
    n = len(meta["write_chunks"])
    assert n > 0, "video sans ecriture"
    last = n - 1
    sf = meta["switch_frame"]
    named = {last: "end_seq"}
    lo = 0
    if sf is not None:
        cand = [i for i in range(n) if D.frame_of_write(meta, i) < int(sf)]
        i_end_subj = max(cand) if cand else 0
        if i_end_subj != last:
            named[i_end_subj] = "end_subject"
            lo = i_end_subj
    n_mid = max(0, int(n_ages) - len(named))
    for j in range(n_mid):
        frac = (j + 1) / (n_mid + 1)
        i = int(round(lo + frac * (last - lo)))
        if i not in named:
            named[i] = f"mid{j}"
    return [{"i": int(i), "name": named[i]} for i in sorted(named)]


# ---------------------------------------------------------------------------------------------
# extraction du NOM DU SUJET depuis le prompt (guide la segmentation, tache n 1)
# ---------------------------------------------------------------------------------------------
# Vocabulaire de TETES DE SYNTAGME : les categories qu'un detecteur ouvert-vocabulaire sait
# trouver. L'ordre du dict n'a aucun role : c'est la POSITION dans le prompt qui tranche
# (le sujet d'un prompt de generation video est quasi toujours le premier syntagme nominal).
HEAD_NOUNS = [
    # personnes
    "person", "people", "man", "men", "woman", "women", "boy", "boys", "girl", "girls",
    "child", "children", "kid", "kids", "baby", "toddler", "teenager", "teen", "lady",
    "gentleman", "guy", "figure", "silhouette", "soldier", "warrior", "knight", "chef",
    "dancer", "singer", "musician", "artist", "painter", "scientist", "doctor", "nurse",
    "farmer", "fisherman", "astronaut", "pilot", "athlete", "runner", "swimmer", "climber",
    "monk", "priest", "king", "queen", "princess", "prince", "wizard", "witch", "angel",
    "student", "teacher", "worker", "policeman", "firefighter", "cowboy", "samurai",
    "couple", "crowd", "family", "group", "elder", "grandmother", "grandfather", "mother",
    "father", "businessman", "businesswoman", "traveler", "hiker", "surfer", "skater",
    # animaux
    "dog", "dogs", "puppy", "cat", "cats", "kitten", "horse", "horses", "pony", "bird",
    "birds", "parrot", "eagle", "owl", "duck", "swan", "penguin", "chicken", "rooster",
    "cow", "cows", "bull", "sheep", "goat", "pig", "deer", "elephant", "lion", "tiger",
    "bear", "wolf", "fox", "rabbit", "squirrel", "monkey", "panda", "giraffe", "zebra",
    "camel", "dolphin", "whale", "shark", "fish", "turtle", "frog", "snake", "lizard",
    "butterfly", "bee", "dragon", "creature", "animal", "beast", "herd", "flock",
    # vehicules
    "car", "cars", "truck", "van", "bus", "motorcycle", "bike", "bicycle", "scooter",
    "train", "tram", "boat", "ship", "sailboat", "canoe", "kayak", "yacht", "plane",
    "airplane", "aircraft", "helicopter", "jet", "rocket", "spaceship", "tractor",
    "carriage", "wagon", "balloon", "drone", "submarine", "taxi", "ambulance",
    # objets
    "bottle", "cup", "mug", "teapot", "kettle", "glass", "plate", "bowl", "vase", "lamp",
    "lantern", "candle", "clock", "watch", "book", "notebook", "guitar", "violin", "piano",
    "drum", "camera", "phone", "laptop", "computer", "television", "radio", "chair",
    "table", "desk", "bench", "sofa", "bed", "door", "window", "mirror", "painting",
    "statue", "sculpture", "robot", "toy", "doll", "ball", "kite", "umbrella", "hat",
    "jacket", "coat", "dress", "shoe", "backpack", "bag", "suitcase", "box", "basket",
    "flower", "rose", "tree", "plant", "cake", "bread", "apple", "pizza", "burger",
    "house", "cabin", "tent", "castle", "tower", "bridge", "lighthouse", "windmill",
    "fountain", "sign", "flag", "banner", "swing", "ladder", "barrel", "crate",
]
HEAD_SET = set(HEAD_NOUNS)
# Les tetes qui designent deja un humain : si l'une d'elles est retenue, le repli "a person"
# n'est pas ajoute (il ferait doublon).
PERSON_SET = {
    "person", "people", "man", "men", "woman", "women", "boy", "boys", "girl", "girls",
    "child", "children", "kid", "kids", "baby", "toddler", "teenager", "teen", "lady",
    "gentleman", "guy", "figure", "silhouette", "soldier", "warrior", "knight", "chef",
    "dancer", "singer", "musician", "artist", "painter", "scientist", "doctor", "nurse",
    "farmer", "fisherman", "astronaut", "pilot", "athlete", "runner", "swimmer", "climber",
    "monk", "priest", "king", "queen", "princess", "prince", "wizard", "witch", "angel",
    "student", "teacher", "worker", "policeman", "firefighter", "cowboy", "samurai",
    "couple", "crowd", "family", "group", "elder", "grandmother", "grandfather", "mother",
    "father", "businessman", "businesswoman", "traveler", "hiker", "surfer", "skater",
    "woodcutter",
}
HEAD_NOUNS.append("woodcutter")
HEAD_SET.add("woodcutter")

# TETES v3 (AMENDEMENT C2, nuit 06-07/08) : les 21 prompts v3 portent des tetes absentes du
# vocabulaire initial ; sans elles, subject_phrase retombe sur le repli generique "a person"
# et la segmentation guidee par le texte n'a pas de requete (precedent : woodcutter, v1).
# Deja presents, VERIFIES contre la liste : monk, surfer, penguin, sheep, camel, balloon
# ("hot air balloon" -> tete balloon), boat ("fishing boat" -> tete boat), bus.
# Nota "bus driver" : la tete "bus" (position 1) gagnera sur "driver" (position 2) -- le
# detecteur masquera le bus ; borderline deja note au fichier de prompts, l'oeil tranchera.
_V3_PERSON_HEADS = ("performer", "carrier", "barista", "motorcyclist", "vendor", "officer",
                    "driver", "shepherd", "gondolier", "blacksmith", "diver")
_V3_ANIMAL_HEADS = ("peacock", "llama", "flamingo")
for _h in _V3_PERSON_HEADS + _V3_ANIMAL_HEADS:
    if _h not in HEAD_SET:
        HEAD_NOUNS.append(_h)
        HEAD_SET.add(_h)
PERSON_SET.update(_V3_PERSON_HEADS)

# mots qui NE PEUVENT PAS commencer un syntagme sujet (on remonte au plus jusqu'a eux)
_STOP_LEFT = {
    "a", "an", "the", "one", "this", "that", "these", "those", "his", "her", "its", "their",
    "of", "in", "on", "at", "with", "by", "for", "from", "to", "into", "onto", "over",
    "under", "near", "beside", "behind", "and", "or", "but", "as", "while", "when", "where",
    "then", ";", ",", ".", ":", "there", "here", "it", "he", "she", "they",
}
# adjectifs de scene a ne pas confondre avec un sujet quand ils precedent une tete generique
_TOKEN_RE = re.compile(r"[a-zA-Z][a-zA-Z\-']*")

# Un prompt video commence souvent par un COMPLEMENT DE LIEU ("In a luxurious house, a person
# walks...") : la tete qui s'y trouve decrit le DECOR, pas le sujet. On repere ce syntagme
# introductif (preposition en tete de prompt jusqu'a la premiere virgule) et on DEPRIORISE les
# tetes qui tombent dedans -- on ne les supprime pas : si le prompt n'a rien d'autre, elles
# restent le meilleur candidat.
_LEAD_PREPS = {"in", "on", "at", "inside", "within", "amid", "amidst", "against", "near",
               "under", "underneath", "above", "across", "through", "beneath", "beside",
               "between", "during", "over", "along", "around", "atop", "behind", "beyond",
               "throughout", "outside"}


def _tokens(text):
    """(mot_minuscule, mot_original, position_de_debut) pour chaque mot du texte."""
    return [(m.group(0).lower(), m.group(0), m.start()) for m in _TOKEN_RE.finditer(text)]


def _np_before(prompt, toks, k, max_mods=4):
    """Le syntagme nominal se terminant au token k : on remonte les modificateurs jusqu'a un
    determinant / une preposition / une PONCTUATION, au plus `max_mods` mots. La coupure par
    ponctuation est lue dans le TEXTE BRUT entre les deux mots (sans elle, "..., tall trees,
    numerous colorful paper airplanes" rend "trees numerous colorful paper airplanes")."""
    start = k
    for j in range(k - 1, max(-1, k - 1 - max_mods), -1):
        if toks[j][0] in _STOP_LEFT:
            break
        gap = prompt[toks[j][2] + len(toks[j][1]): toks[j + 1][2]]
        if any(c in ",.;:()\n" for c in gap):
            break
        start = j
    return " ".join(t[1] for t in toks[start:k + 1]).lower()


def _lead_scene_end(prompt, toks):
    """Fin (en offset caractere) du complement de lieu introductif, 0 s'il n'y en a pas."""
    if not toks or toks[0][0] not in _LEAD_PREPS:
        return 0
    cut = prompt.find(",")
    semi = prompt.find(";")
    if semi >= 0 and (cut < 0 or semi < cut):
        cut = semi
    return cut if cut > 0 else 0


def subject_phrase(prompt, max_candidates=3, person_fallback=True):
    """Nom du sujet extrait du prompt -> requetes texte pour le detecteur ouvert-vocabulaire.

    Regle (deterministe, sans modele de langue) : les TETES candidates = les occurrences dans
    le prompt d'un mot du vocabulaire de categories ; elles sont classees par (hors du
    complement de lieu introductif, puis position). Le syntagme = la tete plus ses
    modificateurs immediats. On rend jusqu'a `max_candidates` requetes (les 2 meilleures tetes
    DISTINCTES + la forme generique de la premiere) et c'est le DETECTEUR qui tranche (score
    median sur les frames echantillonnees) : le texte propose, l'image dispose.

    Retour : {'head', 'phrase', 'queries' (liste), 'found' (bool), 'in_scene' (bool)}.
    found=False => la video est SIGNALEE dans la planche de validation (repli generique)."""
    prompt = prompt or ""
    toks = _tokens(prompt)
    scene_end = _lead_scene_end(prompt, toks)
    hits = []
    for k, (w, _, pos) in enumerate(toks):
        base = w if w in HEAD_SET else (w[:-1] if (w.endswith("s") and w[:-1] in HEAD_SET)
                                        else None)
        if base is None:
            continue
        hits.append({"k": k, "head": w, "in_scene": bool(pos < scene_end)})
    if not hits:
        generic = ["a person", "the main subject", "an animal"] if person_fallback else \
            ["the main subject", "an animal", "an object"]
        return {"head": None, "phrase": None, "in_scene": False,
                "queries": generic, "found": False}
    hits.sort(key=lambda h: (h["in_scene"], h["k"]))
    best = hits[0]
    head, phrase = best["head"], _np_before(prompt, toks, best["k"])
    queries, seen_heads = [phrase], {head}
    for h in hits[1:]:
        if len(queries) >= max_candidates:
            break
        if h["head"] in seen_heads:
            continue
        seen_heads.add(h["head"])
        queries.append(_np_before(prompt, toks, h["k"]))
    queries.append(f"a {head}")
    # Repli ANIME arbitre par le detecteur : nos prompts nomment souvent le sujet par un nom
    # propre ("featuring Mustafa Kemal Ataturk", "capturing Joseph Stalin") que le vocabulaire
    # de categories ne peut pas voir ; la tete extraite tombe alors sur un accessoire ("raki
    # glass", "shiny black jacket"). On ajoute donc TOUJOURS "a person" en DERNIER candidat :
    # il ne gagne que si le detecteur le score franchement mieux (marge dans build_masks.py).
    # `person_fallback=False` (materiau v4, scelle : sujets ANIMAUX/VEHICULES/OBJETS, ZERO
    # humain) : le repli "a person" existait pour les prompts v1 qui nommaient l'individu par
    # un nom propre. Sur un materiau dont l'objet MEME est d'exclure les humains, il est
    # nuisible : `pick_query` peut le preferer (marge de bascule) et poser le masque sur un
    # figurant du decor -- exactement la collinearite des crops humains qui a rendu v3
    # illisible. On le coupe donc a la source pour ce corpus, jamais pour les autres.
    if person_fallback and not any(w in PERSON_SET for w in seen_heads):
        queries.append("a person")
    out, seen = [], set()
    for q in queries:
        q = (q or "").strip()
        if q and q not in seen:
            seen.add(q)
            out.append(q)
    return {"head": head, "phrase": phrase, "in_scene": best["in_scene"],
            "queries": out, "found": True}


# ---------------------------------------------------------------------------------------------
# VAE (decode des frames latentes) et DINO (le meme que le harness)
# ---------------------------------------------------------------------------------------------
def get_vae(device):
    """Le VAE de Wan SEUL (pas de pipeline, pas de T5, pas de 1.3B) : `WanVAEWrapper` charge
    wan_models/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth par un chemin RELATIF, d'ou le chdir."""
    import torch
    from utils.wan_wrapper import WanVAEWrapper
    cwd = os.getcwd()
    try:
        os.chdir(REPO)
        vae = WanVAEWrapper()
    finally:
        os.chdir(cwd)
    return vae.to(device=device, dtype=torch.bfloat16).eval().requires_grad_(False)


def decode_latent_frames(vae, latents, device, batch=4):
    """(T, 16, 60, 104) latents fp16 -> (T, 3, 480, 832) pixels dans [0, 1], cpu fp32.
    Chemin IDENTIQUE a build_bank.dino_audit_embed : chaque frame latente est decodee SEULE
    (use_cache=False), donc une frame latente = une frame pixel."""
    import torch
    out = []
    with torch.no_grad():
        for i in range(0, latents.shape[0], batch):
            lat = latents[i:i + batch].to(device=device, dtype=torch.bfloat16).unsqueeze(1)
            px = vae.decode_to_pixel(lat, use_cache=False, to_cpu=False).float()[:, 0]
            out.append(((px + 1.0) / 2.0).clamp(0, 1).cpu())
    return torch.cat(out, 0)


def dino_embed(frames01, device, batch=32):
    """Le DINO DU HARNESS, importe (jamais recopie) : dinov2_vitb14, 224x224, CLS, L2."""
    from run_reid_probe import dino_embed as _de
    return _de(frames01, device, batch=batch)


# ---------------------------------------------------------------------------------------------
# petits utilitaires
# ---------------------------------------------------------------------------------------------
def ensure_dirs():
    for d in (HEAVY, MASK_DIR, REFS_DIR, STATE_DIR, SIG_DIR, BOARD_DIR, RUNS_DIR, HF_HOME):
        os.makedirs(d, exist_ok=True)


def seg_key(video_id, seg_name):
    return f"{video_id.replace(':', '_')}__{seg_name}"


def save_json(obj, path):
    # le fichier temporaire porte le PID : deux processus qui ecrivent LE MEME resultat (deux
    # campagnes en parallele sur la meme carte) ne peuvent plus entrelacer leurs ecritures.
    # `os.replace` reste atomique, donc le lecteur voit toujours un JSON complet.
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = f"{path}.tmp.{os.getpid()}"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=1)
    os.replace(tmp, path)
    return path


def load_json(path):
    with open(path) as f:
        return json.load(f)


def gpu_precheck(min_free_gb=20.0, forbid=(6,)):
    """Regle de la maison : pre-check DANS la commande de lancement, jamais l'index 6 casse.
    Retourne la liste des index libres (memoire libre >= min_free_gb)."""
    import subprocess
    q = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.free,memory.used",
                        "--format=csv,noheader,nounits"], capture_output=True, text=True,
                       env={**os.environ, "CUDA_DEVICE_ORDER": "PCI_BUS_ID"})
    free = []
    for line in q.stdout.strip().splitlines():
        idx, mfree, mused = [int(x.strip()) for x in line.split(",")]
        if idx in forbid:
            continue
        if mfree / 1024.0 >= min_free_gb:
            free.append((idx, mfree / 1024.0, mused / 1024.0))
    return free


def freeze_subsets(corpus="v1", path=None):
    """Ecrit les sous-ensembles PRE-ENREGISTRES sur disque, avec leur date et la table
    numero-de-planche -> corpus_idx. C'est la trace horodatee du gel : le fichier doit etre
    ANTERIEUR a tout JSON de verdict (verrou repris de l'audit du 04/08)."""
    import time
    _, pairs = pair_dataset(corpus=corpus)
    sub = sealed_subsets(pairs, check_frozen=False)   # c'est CE geste qui pose le gel
    obj = {"corpus": corpus, "gele_le": time.strftime("%Y-%m-%d %H:%M:%S"),
           "source": "memory2_probes_registry.md, sections LISTE OEIL D'ADEN et PLAFOND HUMAIN",
           "numerotation": "ordre lexicographique de l'identifiant de la video principale "
                           "(= celui de masks_board_pairs_numerote.jpg)",
           "board_no_to_corpus_idx": {str(p["board_no"]): int(p["corpus_idx"]) for p in pairs},
           "board_no": {name: sorted(p["board_no"] for p in pairs
                                     if p["corpus_idx"] in keep)
                        for name, keep in sub.items()},
           "corpus_idx": {name: sorted(keep) for name, keep in sub.items()},
           "regle_agent": {"requetes_ecartees": list(AGENT_BAD_QUERIES),
                           "score_median_min": AGENT_MIN_DET_SCORE}}
    path = path or os.path.join(RUNS_DIR, f"subsets_{corpus}.json")
    save_json(obj, path)
    print(f"[probe1] sous-ensembles geles -> {path}")
    for k, v in obj["board_no"].items():
        print(f"    {k:9s} n={len(v):2d} : {v}")
    return obj


if __name__ == "__main__":
    ensure_dirs()
    if "--freeze-subsets" in sys.argv:
        c = "v2" if "--v2" in sys.argv else "v1"
        freeze_subsets(c)
        sys.exit(0)
    vids, pairs = pair_dataset(verbose=True)
    n_split = {}
    for p in pairs:
        n_split[p["split"]] = n_split.get(p["split"], 0) + 1
    print(f"[probe1] {len(vids)} videos, {len(pairs)} paires, split {n_split}")
    byid = vids_by_id(vids)
    n_seg = sum(len(segments(v)) for v in vids)
    print(f"[probe1] {n_seg} segments-sujet au total")
    nf = 0
    for v in vids[:5] + [byid[p['main']] for p in pairs[:5]]:
        for s in segments(v):
            sp = subject_phrase(s["prompt"])
            nf += int(not sp["found"])
            print(f"  {v['video_id']:>12} {s['name']} f[{s['f0']},{s['f1']}) "
                  f"-> {sp['queries'][0]!r}")
    miss = [v["video_id"] for v in vids for s in segments(v)
            if not subject_phrase(s["prompt"])["found"]]
    print(f"[probe1] segments sans tete reconnue : {len(miss)} / {n_seg}")
    v0 = vids[0]
    print(f"[probe1] ages de {v0['video_id']} ({len(v0['write_chunks'])} ecritures) : "
          f"{state_ages(v0)}")
