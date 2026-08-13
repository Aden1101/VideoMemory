#!/usr/bin/env python
# ============================================================================================
# ETAPE 1 -- LES SIGNATURES (lecture FONCTIONNELLE de l'etat) + LA BORNE SUPERIEURE.
#
# "On passe une batterie fixe d'entrees-sondes a travers le module a poids rapides et on
#  recueille ses sorties (sa signature comportementale : c'est ainsi que le decodeur le lit,
#  par requetes)." + renfort (a) : "BORNE SUPERIEURE obligatoire : le MEME pipeline applique
#  aux activations brutes (qui contiennent tout) doit etre quasi parfait : sinon c'est le
#  protocole qui est casse, pas la memoire : on construit sur ce cas facile d'abord, PUIS ON NE
#  CHANGE QUE L'ENTREE."
#
# C'est litteralement ce que fait ce fichier : UN SEUL chemin `signature()`, et un flag
# `--source {state,h}` qui ne change QUE le lecteur d'un bloc :
#     state -> swiglu_apply(sondes, w0, w1, w2)    (la lecture du decodeur, verbatim)
#     h     -> lecture associative LINEAIRE des memes sondes sur les tokens h bruts
# Les deux rendent (n_blocs, n_sondes, 1536) ; tout l'aval (blanchiment, ACP, MLP, evaluation)
# est identique et ne sait meme pas quelle source a produit le tenseur.
#
# Le blanchiment N'EST PAS fait ici : il depend du split (stats du TRAIN uniquement) et vit
# donc dans train_probe.py. Ici on ne stocke que la signature brute, en float16.
#
# Usage :
#   CUDA_VISIBLE_DEVICES=7 python reference/runs/probe1/build_signatures.py --source state
#   CUDA_VISIBLE_DEVICES=7 python reference/runs/probe1/build_signatures.py --source h
#   CUDA_VISIBLE_DEVICES=7 python .../build_signatures.py --source hdirect_masked   # borne v3
# ============================================================================================
import argparse
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import probe1_common as C  # noqa: E402

import torch  # noqa: E402

import mem2_data as D  # noqa: E402
from build_probes import PROBES_PT  # noqa: E402
from build_states import state_path  # noqa: E402
from wan.modules.video_memory import swiglu_apply  # noqa: E402

from build_states import LIN_SPEC, MLP_SPEC  # noqa: E402  (oracles ecriture, 06-07/08)

LIN_SOURCES = tuple(sorted(LIN_SPEC))
MLP_SOURCES = tuple(sorted(MLP_SPEC))       # ARM MLP-CORRECTEUR (07/08)
SOURCES = ("state", "h", "hdirect", "hdirect_masked") + LIN_SOURCES + MLP_SOURCES


# ---------------------------------------------------------------------------------------------
# les DEUX lecteurs (la seule chose que le flag --source change)
# ---------------------------------------------------------------------------------------------
def read_state(probes_b, payload_b):
    """probes_b (P, dim), payload_b = (w0, w1, w2) -> reponse (P, dim).
    C'est EXACTEMENT la lecture du decodeur (`swiglu_apply`, importe de video_memory, jamais
    recopie), en fp32 comme dans `_memdec_frame` du harness."""
    w0, w1, w2 = [t.to(torch.float32) for t in payload_b]
    if w0.dim() == 2:
        w0, w1, w2 = w0[None], w1[None], w2[None]
    return swiglu_apply(probes_b[None].to(torch.float32), w0, w1, w2)[0]


def read_h(probes_b, payload_b):
    """probes_b (P, dim), payload_b = h (Lw, dim) -> reponse (P, dim).

    La BORNE SUPERIEURE. L'etat des poids rapides EST une memoire associative NON LINEAIRE
    construite a partir de ces h ; la borne superieure lit donc la memoire associative
    LINEAIRE construite a partir des MEMES h avec les MEMES sondes :

        R(q) = (q h^T) h / (L * dim)   =   q (h^T h) / (L * dim)

    Meme forme fonctionnelle (une matrice-memoire lue par une requete), meme forme de sortie,
    et SURTOUT aucun hyperparametre. C'est la raison du choix : une lecture par attention
    softmax aurait exige une TEMPERATURE, donc un reglage libre au coeur de la mesure qui
    decide si le protocole est valide -- exactement ce qu'un juge ne doit pas avoir. Les deux
    constantes de division sont fixes et l'echelle est de toute facon absorbee par le
    blanchiment ; elles ne servent qu'a rester dans la dynamique du stockage."""
    h = payload_b.to(torch.float32)
    if h.dim() == 3:
        h = h[0]
    q = probes_b.to(torch.float32)
    return (q @ h.transpose(0, 1)) @ h / (h.shape[0] * q.shape[-1])


def read_hdirect(probes_b, payload_b):
    """LA BORNE DE VALIDITE v2 (mission 04/08, defaut n 2 du verdict PROTOCOLE INVALIDE) :
    les activations sont sondees DIRECTEMENT, SANS AUCUNE ECRITURE DANS LA BOUCLE.

    `read_h` (v1) lisait deja une memoire associative -- lineaire, mais une memoire quand meme
    (q h^T h) : si la forme d'ecriture etait le goulot, la borne le subissait AUSSI, et un echec
    ne separait pas "le materiau ne porte pas l'identite" de "la lecture par sonde ne la voit
    pas". Ici il n'y a plus de sonde du tout du cote de la charge utile : on POOLE les tokens h
    par chunk (moyenne et max sur les tokens), puis on agrege sur tous les chunks ecrits jusqu'a
    cet age -- ce qui est exactement l'ensemble des activations que l'etat a absorbees (audit
    A2 : l'ancienne borne ne voyait QUE le dernier write, l'etat les voit TOUS).

    payload_b = LISTE de tenseurs h (Lw, dim), un par chunk ecrit jusqu'a l'age -- ou, pour
    `hdirect_masked` (CORRECTIF DE BORNE du 05/08 nuit), une liste de couples (h, tokens
    retenus). Les DEUX sources partagent CE lecteur, a l'octet pres : la seule difference est
    QUELS TOKENS entrent dans le pooling, et elle est portee par la charge utile
    (`load_payload`), pas par une branche de lecture. Masque plein => sortie identique bit a
    bit a `hdirect`.
    Sortie : (dim, 2) -- MEME ORIENTATION que la matrice-reponse des sondes (une ligne = une
    unite de reponse, une colonne = une coordonnee) : ligne = canal, colonnes = [moyenne sur les
    tokens, max sur les tokens]. La compression generique (moyenne+max sur les colonnes) s'y
    applique donc TELLE QUELLE, sans savoir d'ou vient le tenseur -- et sans rien perdre : de
    ((m+M)/2, M) on retrouve (m, M) exactement.
    Aucun hyperparametre, aucune sonde, rien a regler."""
    hs = payload_b if isinstance(payload_b, (list, tuple)) else [payload_b]
    n, s, mx = 0, None, None
    for item in hs:
        h, keep = item if isinstance(item, tuple) else (item, None)
        h = h.to(torch.float32)
        if h.dim() == 3:
            h = h[0]
        if keep is not None:
            h = h[keep]
        s = h.sum(0) if s is None else s + h.sum(0)
        m = h.amax(0)
        mx = m if mx is None else torch.maximum(mx, m)
        n += h.shape[0]
    assert n > 0 and s is not None, "hdirect : aucun token"
    return torch.stack([s / n, mx], -1)


def read_lin(probes_b, payload_b):
    """ORACLE REGLE LINEAIRE : payload_b = W (d, d), fp16 du snapshot -> reponse (P, d).
    "Lecture de W pour les signatures : sorties = W @ sondes" (registre) : la ligne p de la
    reponse est W @ q_p, calculee en fp32 comme read_state (probes @ W^T == (W q)^T ligne a
    ligne). Meme forme de sortie que read_state, donc TOUT l'aval (compression moyenne+max,
    blanchiment, tete, evaluation) est bit a bit le meme code. LES QUATRE bras lineaires
    pointent sur CETTE fonction : seule l'ecriture (regle, masque) change, jamais la lecture."""
    W = payload_b.to(torch.float32)
    if W.dim() == 3:
        W = W[0]
    assert W.dim() == 2 and W.shape[0] == W.shape[1], f"W attendu (d, d), recu {tuple(W.shape)}"
    return probes_b.to(torch.float32) @ W.transpose(0, 1)


# `hdirect` et `hdirect_masked` pointent sur LA MEME fonction : un seul chemin de lecture,
# verifie par selftest (T5 : READERS["hdirect_masked"] is READERS["hdirect"]).
READERS = {"state": read_state, "h": read_h, "hdirect": read_hdirect,
           "hdirect_masked": read_hdirect}
READERS.update({s: read_lin for s in LIN_SOURCES})   # 4 bras, UN lecteur
# ARM MLP-CORRECTEUR : l'etat reste (w0, w1, w2) d'une cellule SwiGLU -- seule l'ECRITURE
# change. La LECTURE est donc, litteralement, celle du bras historique `state` : MEME objet
# fonction (selftest D0 : READERS["mlp_delta_all"] is READERS["state"]).
READERS.update({s: read_state for s in MLP_SOURCES})
# Les sources dont la charge utile est la SEQUENCE des h ecrits jusqu'a l'age (et non le seul
# dernier write) : c'est le correctif de l'asymetrie relevee par l'audit A2.
SEQ_SOURCES = ("hdirect", "hdirect_masked")
# Les sources dont la charge utile porte, EN PLUS, le masque-sujet par chunk.
MASKED_SOURCES = ("hdirect_masked",)


# ---------------------------------------------------------------------------------------------
# LA COMPRESSION (defaut n 1 du verdict PROTOCOLE INVALIDE : 1 179 648 dimensions pour 36
# videos d'entrainement, c'est-a-dire 8 000 dimensions par exemple).
# ---------------------------------------------------------------------------------------------
def compress_signature(sig):
    """(n_blocs, n_lignes, dim) -> (n_blocs, n_lignes, 2) : moyenne ET max sur les `dim`
    canaux de la reponse. Pour la signature d'etat (3 blocs x 256 sondes x 1536) cela fait
    3 x 256 x 2 = 1536 nombres : DEUX valeurs par sonde et par bloc, comme demande.

    Pourquoi moyenne ET max : la moyenne est le niveau de reponse de la sonde, le max sa
    saillance (quelle est la plus forte coordonnee qu'elle excite). Une sonde qui "voit"
    quelque chose de precis bouge le max sans bouger la moyenne.

    Deterministe, sans parametre, sans donnees : la compression ne peut pas fuir un split."""
    assert sig.dim() == 3, f"signature attendue (blocs, lignes, dim), recue {tuple(sig.shape)}"
    x = sig.to(torch.float32)
    return torch.stack([x.mean(-1), x.amax(-1)], dim=-1)


# ---------------------------------------------------------------------------------------------
# LE chemin unique
# ---------------------------------------------------------------------------------------------
@torch.no_grad()
def signature(source, probes, payload, blocks):
    """probes (n_blocs, P, dim) ; payload {bloc: charge utile} ; -> (n_blocs, lignes, cols) fp32.
    UN SEUL chemin pour la borne de validite et pour les etats : seul `source` change, et il ne
    selectionne qu'un lecteur par bloc. La matrice-reponse d'un bloc est toujours 2-D et de
    MEME forme pour tous les blocs ; l'aval (compression, blanchiment, tete, evaluation) ne
    sait pas quelle source l'a produite."""
    assert source in READERS, f"source inconnue {source!r}"
    assert probes.shape[0] == len(blocks), \
        f"batterie {probes.shape[0]} blocs != {len(blocks)} blocs memoire"
    reader = READERS[source]
    outs = []
    for k, bi in enumerate(blocks):
        r = reader(probes[k], payload[bi])
        assert r.dim() == 2, f"bloc {bi}: reponse {tuple(r.shape)} n'est pas 2-D"
        assert not outs or r.shape == outs[0].shape, \
            f"bloc {bi}: reponse {tuple(r.shape)} != {tuple(outs[0].shape)} des blocs precedents"
        outs.append(r)
    return torch.stack(outs, 0)


def sig_path(source, video_id, age_name):
    return os.path.join(C.SIG_DIR, source, f"{video_id.replace(':', '_')}__{age_name}.pt")


def load_sig(source, video_id, age_name, compressed=False):
    """La signature stockee ; `compressed=True` applique la compression moyenne+max par ligne
    (le chemin d'entree du sondeur v2). Un seul endroit dans tout le code lit un .pt de
    signature : celui-ci."""
    d = torch.load(sig_path(source, video_id, age_name), map_location="cpu", weights_only=False)
    s = d["sig"]
    return compress_signature(s).reshape(-1) if compressed else s.reshape(-1)


# ---------------------------------------------------------------------------------------------
def load_payload(source, meta, age, cache, blocks, device, states=None, state_dir=None):
    """La charge utile d'une (video, age), dans la source demandee.
    `states` (optionnel) = les etats deja rejoues en memoire (mode --stream : on ne passe pas
    par le disque, cf. `stream_states`).
    `state_dir` (optionnel, ADDITIF oracle 06/08) = repertoire d'etats alternatif
    (oracle_maskwrite/states_<bras>/) ; None = le repertoire historique, INCHANGE."""
    if source in LIN_SOURCES:
        # oracle regle lineaire : W vient TOUJOURS du rejeu en memoire (mode --stream,
        # lin_states) -- il n'existe aucun stockage disque de W.
        assert states is not None, f"source {source}: mode --stream obligatoire"
        return {b: states[int(age["i"])][b].to(device) for b in blocks}
    if source in MLP_SOURCES:
        # ARM MLP-CORRECTEUR : etat (w0, w1, w2) rejoue a la volee (--stream obligatoire :
        # aucun etat correcteur n'est materialise sur disque), charge utile IDENTIQUE a `state`.
        assert states is not None, f"source {source}: mode --stream obligatoire"
        return {b: tuple(t.to(device) for t in states[int(age["i"])][b]) for b in blocks}
    if source == "state":
        if states is not None:
            return {b: tuple(t.to(device) for t in states[int(age["i"])][b]) for b in blocks}
        p = state_path(meta["video_id"], age["name"], state_dir)
        assert os.path.exists(p), f"etat manquant : {p} (lance build_states.py d'abord)"
        ck = torch.load(p, map_location="cpu", weights_only=False)
        assert ck["write_index"] == int(age["i"]), \
            f"{p}: age {ck['write_index']} != {age['i']} (manifeste desynchronise)"
        return {b: tuple(t.to(device) for t in ck["state"][b]) for b in blocks}
    rec = D.get_record(meta, cache)
    if source in SEQ_SOURCES:
        # TOUS les h ecrits jusqu'a cet age (l'etat, lui, les a tous absorbes)
        seqs = {b: rec["h_seq"][str(b)][:int(age["i"]) + 1] for b in blocks}
        if source not in MASKED_SOURCES:
            return {b: [t.to(device) for t in s] for b, s in seqs.items()}
        # LE SEUL embranchement de `hdirect_masked` : on attache a chaque chunk les tokens
        # sous le masque-sujet. Le masque ne depend PAS du bloc (c'est une geometrie d'image) :
        # il est calcule une fois et partage, ce qui garantit que les 3 blocs poolent
        # exactement les memes positions.
        grids = C.video_mask_grids(meta["video_id"])
        ref = seqs[blocks[0]]
        masks, n_fb = [], 0
        for j, t in enumerate(ref):
            m, fb = C.chunk_token_mask(meta, j, grids, n_tokens=int(t.shape[-2]))
            masks.append(torch.as_tensor(m, device=device))
            n_fb += int(fb)
        if n_fb:
            print(f"[sig/masked] REPLI top-{C.MASK_MIN_FRAC:.0%} sur {n_fb}/{len(masks)} chunks "
                  f"({meta['video_id']} / {age['name']})", flush=True)
        return {b: [(t.to(device), masks[j]) for j, t in enumerate(s)]
                for b, s in seqs.items()}
    return {b: rec["h_seq"][str(b)][int(age["i"])].to(device) for b in blocks}


def stream_states(meta, ages, cache, blocks, device, cells=None, keep=None, grad_fn=None,
                  micro_batch=None):
    """Rejoue les ecritures d'UNE video et rend les etats aux ages demandes SANS RIEN ECRIRE SUR
    LE DISQUE (170 MB par etat : 4 ages x 165 videos = 112 GB qu'on n'a pas). Le rejeu est
    exactement `build_states.replay_video`, donc la trajectoire est celle de l'inference.
    `keep` (ADDITIF oracle 06/08) : liste de tokens gardes par ecriture (`maskwrite_keep`) --
    la variante zero-octet-d'etat des bras masques ; None = rejeu historique inchange.
    `grad_fn` / `micro_batch` (ADDITIFS, bras MLP 07/08) : objectif interne d'ecriture et
    calendrier de mise a jour dans le chunk ; None/None = le chemin par defaut, bit a bit
    l'existant."""
    from build_states import replay_video
    return replay_video(cells, blocks, meta, cache, ages, device, keep=keep, grad_fn=grad_fn,
                        micro_batch=micro_batch)


def build_cells_for_stream(config_path, blocks, device):
    """Les cellules l1@2000-EMA (contrat F2.3 : delta_l1 == 0), pour le mode --stream."""
    from omegaconf import OmegaConf

    from train_writer import build_cells, delta_l1, ref_theta_l1
    config = OmegaConf.load(config_path)
    cells, blocks_c, _ = build_cells(config, device, mode="direct")
    assert blocks_c == blocks, f"{blocks_c} != {blocks}"
    ref, src = ref_theta_l1(config, blocks, device)
    d = delta_l1(cells, ref)
    assert d == 0.0, f"contrat F2.3 viole : delta_l1 = {d}"
    print(f"[sig] cellules {blocks} depuis {src}, delta_l1 = {d}", flush=True)
    return cells


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True, choices=SOURCES)
    ap.add_argument("--probes", default=PROBES_PT)
    ap.add_argument("--all", action="store_true",
                    help="les 165 videos (defaut : les 60 videos des 30 paires, le seul "
                         "corpus dont le sondeur a besoin)")
    ap.add_argument("--switch_only", action="store_true",
                    help="les videos `switch` du corpus (aucune n'appartient a une paire) : "
                         "le jeu de PRE-ENTRAINEMENT contrastif de la tete")
    ap.add_argument("--corpus", default="v1",
                    choices=("v1", "v2", "v3", "v3h", "v4", "v4h"))
    ap.add_argument("--stream", action="store_true",
                    help="source=state : rejoue les ecritures a la volee au lieu de lire les "
                         "etats sur disque (170 MB/etat qu'on n'ecrit alors jamais)")
    ap.add_argument("--states-dir", dest="states_dir", default="",
                    help="oracle d'ecriture masquee : repertoire d'etats ALTERNATIF "
                         "(oracle_maskwrite/states_<bras>/). source=state, --sig-tag requis, "
                         "incompatible avec --stream. Vide = states/ historique, INCHANGE.")
    ap.add_argument("--write-mask", dest="write_mask", default="none",
                    choices=("none", "subject", "bg_matched"),
                    help="variante --stream de l'oracle : rejoue les ecritures MASQUEES a la "
                         "volee (zero octet d'etat sur disque). source=state, --stream et "
                         "--sig-tag requis.")
    ap.add_argument("--sig-tag", dest="sig_tag", default="",
                    help="sous-repertoire de sig/ et nom d'index pour CES signatures (defaut : "
                         "le nom de la source). OBLIGATOIRE pour les bras oracle, pour ne "
                         "jamais ecraser les signatures du bras original.")
    ap.add_argument("--config", default=C.DEFAULT_CONFIG)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--n_ages", type=int, default=C.N_AGES)
    ap.add_argument("--redo", action="store_true")
    args = ap.parse_args()

    oracle = bool(args.states_dir) or args.write_mask != "none"
    if oracle:
        assert args.source == "state", "les bras oracle ne concernent que --source state"
        assert args.sig_tag and args.sig_tag != "state", \
            "--sig-tag obligatoire (et != 'state') : les signatures du bras original ne " \
            "doivent JAMAIS etre ecrasees"
        assert not (args.states_dir and args.stream), \
            "--states-dir lit des etats deja materialises ; --stream les rejoue : choisis"
        if args.write_mask != "none":
            assert args.stream, "--write-mask sans --states-dir exige --stream"
    src_tag = args.sig_tag or args.source

    C.ensure_dirs()
    C.set_determinism(0)
    os.makedirs(os.path.join(C.SIG_DIR, src_tag), exist_ok=True)
    device = "cuda"

    pk = torch.load(args.probes, map_location="cpu", weights_only=False)
    probes, blocks = pk["probes"].to(device), list(pk["blocks"])
    print(f"[sig] batterie {tuple(probes.shape)} blocs {blocks} "
          f"({pk['n_centroids']} centroides + {pk['n_queries']} requetes)", flush=True)

    vids, pairs = C.pair_dataset(verbose=True, corpus=args.corpus)
    pair_ids = {p["main"] for p in pairs} | {p["twin"] for p in pairs}
    if args.switch_only:
        sel = [v for v in vids if v["mode"] == "switch"]
        leak = [v["video_id"] for v in sel if v["video_id"] in pair_ids]
        assert not leak, f"FUITE : {len(leak)} videos switch appartiennent a une paire {leak[:3]}"
    else:
        sel = [v for v in vids if (args.all or v["video_id"] in pair_ids)]
    sel.sort(key=lambda v: (v["part"], v["vi"]))
    if args.limit:
        sel = sel[:args.limit]
    print(f"[sig] {len(sel)} videos (corpus {args.corpus}, "
          f"{'switch/pre-entrainement' if args.switch_only else 'paires'})", flush=True)

    cells = None
    if args.source in LIN_SOURCES:
        assert args.stream, f"source {args.source}: mode --stream obligatoire (W n'est " \
                            "jamais materialise sur disque)"
        # AUCUNE cellule SwiGLU : les regles lineaires sont fixes, zero parametre appris.
    elif args.stream:
        assert args.source == "state" or args.source in MLP_SOURCES, \
            "--stream n'a de sens que pour --source state (ou les bras mlp_delta_*)"
        cells = build_cells_for_stream(args.config, blocks, device)
    elif args.source in MLP_SOURCES:
        raise SystemExit(f"source {args.source}: mode --stream obligatoire (aucun etat "
                         "correcteur n'est materialise sur disque)")

    cache = D.PartCache(size=1)
    t0, n_done, mx = time.time(), 0, 0.0
    index = []
    for n, meta in enumerate(sel):
        ages = C.state_ages(meta, args.n_ages)
        todo = [a for a in ages
                if args.redo or not os.path.exists(sig_path(src_tag, meta["video_id"],
                                                            a["name"]))]
        states, lin_stats = None, None
        if todo and args.source in LIN_SOURCES:
            from build_states import lin_states
            rule, masked = LIN_SPEC[args.source]
            states, lin_stats = lin_states(meta, cache, blocks, todo, device, rule, masked)
            kc = [k for k in lin_stats["kept_counts"] if k > 0]
            nrm = {i: {b: round(v, 2) for b, v in bl.items()}
                   for i, bl in sorted(lin_stats["norms"].items())}
            print(f"[lin/{args.source}] {meta['video_id']}: tokens/chunk moy "
                  f"{sum(kc) / max(1, len(kc)):.0f} min {min(kc) if kc else 0} "
                  f"max {max(kc) if kc else 0}, "
                  f"{lin_stats['kept_counts'].count(0)} sautee(s) | ||W||_F par age {nrm}",
                  flush=True)
        elif cells is not None and todo:
            keep = None
            if args.source in MLP_SOURCES:
                # BRAS MLP : le masque est porte par le NOM DU BRAS (comme les bras lineaires),
                # jamais par --write-mask ; meme chemin de masque que l'oracle masque
                # (maskwrite_keep -> C.chunk_token_mask, min_frac=0.0). L'objectif, le
                # calendrier et le pas viennent de la table MLP_ARMS, via `mlp_arm`.
                from build_states import maskwrite_keep, mlp_arm
                if MLP_SPEC[args.source]:
                    keep, _info = maskwrite_keep(meta, "subject")
                with mlp_arm(cells, args.source) as (grad_fn, micro_batch):
                    states = stream_states(meta, todo, cache, blocks, device, cells=cells,
                                           keep=keep, grad_fn=grad_fn, micro_batch=micro_batch)
                nrm = {i: {b: round(float(sum(float(t.float().norm()) ** 2
                                              for t in st[b]) ** 0.5), 2)
                           for b in blocks} for i, st in sorted(states.items())}
                nk = ("tous" if keep is None else
                      f"{sum(0 if k is None else int(k.size) for k in keep)} tokens gardes, "
                      f"{sum(1 for k in keep if k is None)} ecriture(s) sautee(s)")
                print(f"[mlp/{args.source}] {meta['video_id']}: {nk} | ||etat||_F par age "
                      f"{nrm}", flush=True)
            else:
                if args.write_mask != "none":
                    from build_states import maskwrite_keep
                    keep, _info = maskwrite_keep(meta, args.write_mask)
                states = stream_states(meta, todo, cache, blocks, device, cells=cells, keep=keep)
        for age in ages:
            p = sig_path(src_tag, meta["video_id"], age["name"])
            index.append({"video_id": meta["video_id"], "age": age["name"],
                          "write_index": int(age["i"]), "path": p,
                          "corpus_idx": int(meta["corpus_idx"]), "twin": bool(meta["twin"])})
            if os.path.exists(p) and not args.redo:
                continue
            payload = load_payload(args.source, meta, age, cache, blocks, device,
                                   states=states, state_dir=args.states_dir or None)
            sig = signature(args.source, probes, payload, blocks)
            amax = float(sig.abs().max())
            mx = max(mx, amax)
            # le stockage fp16 ne doit JAMAIS ecreter : si une reponse depasse la dynamique
            # fp16, on veut le savoir au moment de l'ecriture, pas dans les scores.
            sig16 = sig.to(torch.float16)
            assert torch.isfinite(sig16).all(), \
                (f"{meta['video_id']}/{age['name']}: reponse hors dynamique fp16 "
                 f"(|max| = {amax:.4g}) -- repasser le stockage en float32")
            os.makedirs(os.path.dirname(p), exist_ok=True)
            torch.save({"sig": sig16.cpu(), "source": args.source, "sig_tag": src_tag,
                        "video_id": meta["video_id"], "age": age["name"],
                        "write_index": int(age["i"]), "blocks": blocks,
                        "corpus": args.corpus, "write_mask": args.write_mask,
                        "states_dir": args.states_dir or None,
                        "w_fro": (lin_stats["norms"].get(int(age["i"]))
                                  if lin_stats else None),
                        "probes_file": os.path.abspath(args.probes)}, p)
            del payload, sig
            n_done += 1
        del states
        if (n + 1) % 10 == 0 or n == len(sel) - 1:
            el = time.time() - t0
            print(f"[sig/{src_tag}] {n + 1}/{len(sel)} videos, {n_done} signatures, "
                  f"{el / 60:.1f} min", flush=True)

    tag = src_tag + ("" if args.corpus == "v1" else f"_{args.corpus}") \
        + ("_switch" if args.switch_only else "")
    C.save_json({"source": args.source, "sig_tag": src_tag, "corpus": args.corpus,
                 "switch_only": bool(args.switch_only), "stream": bool(args.stream),
                 "states_dir": args.states_dir or None, "write_mask": args.write_mask,
                 "n_videos": len(sel), "n_signatures": len(index),
                 "probes": os.path.abspath(args.probes), "blocks": blocks,
                 "shape": [int(probes.shape[0]), int(probes.shape[1]), int(probes.shape[2])],
                 "index": index, "max_abs": mx},
                os.path.join(C.SIG_DIR, f"_index_{tag}.json"))
    print(f"[sig/{src_tag}] TERMINE : {len(index)} signatures ({n_done} recalculees), "
          f"|max| = {mx:.4g} (dynamique fp16 : 65504)")


if __name__ == "__main__":
    main()
