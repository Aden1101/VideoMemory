#!/usr/bin/env python
"""ETAPE 1, LE JEU de donnees D'ETATS (les poids rapides, reconstitues hors-ligne).

"Donnees : les etats des poids rapides, reconstitues hors-ligne en REJOUANT les ECRITURES sur
 les activations deja stockees sur disque (aucune generation video)."

Machinerie reutilisee TELLE QUELLE (rien de neuf, rien de recopie) :
  - train_writer.build_cells / ref_theta_l1 : les cellules l1@2000-EMA, exactement les poids
    qui ont ecrit les h du corpus (contrat F2.3 : delta_l1 == 0.0 au chargement, asserte ici) ;
  - wan.modules.video_memory.replay_writes : le rejeu differentiable valide au bit pres
    (selftest_replay.py S1) ; ici en pur avant (n_bptt_writes=0, detach_state=True).

Les etats sont snapshottes aux ~4 AGES definis par probe1_common.state_ages, dont
`end_subject` (fin du segment-sujet : l'etat vient d'absorber A, aucune interference) et
`end_seq` (fin de sequence : chez les switch, APRES l'interference du second sujet, la
condition operationnelle reelle). Stockage float16 sur le disque PROJET.

Usage :
  CUDA_DEVICE_ORDER=PCI_BUS_ID nvidia-smi --query-gpu=index,memory.free --format=csv && \
  CUDA_VISIBLE_DEVICES=7 python reference/runs/probe1/build_states.py --only_pairs
"""
import argparse
import contextlib
import hashlib
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import probe1_common as C  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402

import mem2_data as D  # noqa: E402
from train_writer import build_cells, delta_l1, ref_theta_l1  # noqa: E402
from wan.modules.video_memory import (replay_writes, swiglu_apply,  # noqa: E402
                                      swiglu_fast_weight_gradient)


def state_path(video_id, age_name, state_dir=None):
    """`state_dir=None` = le repertoire historique (bit a bit l'existant) ; sinon un
    repertoire d'etats alternatif (oracle d'ecriture masquee)."""
    return os.path.join(state_dir or C.STATE_DIR,
                        f"{video_id.replace(':', '_')}__{age_name}.pt")


# ---------------------------------------------------------------------------------------------
# oracle D'ECRITURE MASQUEE (registre : section "oracle D'ECRITURE MASQUEE", go 2026-08-06).
# Diagnostic par REJEU, zero entrainement : on refait les memes ecritures (meme cellule
# l1@2000-EMA, memes h) en filtrant quels TOKENS entrent dans chaque ecriture.
# ---------------------------------------------------------------------------------------------
MASKWRITE_ARMS = ("subject", "bg_matched")


def maskwrite_dir(arm):
    assert arm in MASKWRITE_ARMS, f"bras inconnu {arm!r}"
    return os.path.join(C.HEAVY, "oracle_maskwrite", f"states_{arm}")


def maskwrite_keep(meta, arm, n_tokens=None):
    """Les tokens gardes a CHAQUE ecriture, selon le bras. Rend (keep, info) :
        keep[j] = np.ndarray TRIEE d'indices de tokens (l'ecriture j n'ecrit qu'eux),
                  ou None = ecriture SAUTEE (etat inchange) ;
        info    = stats loggables (comptes par chunk, ecritures sautees).

    ALIGNEMENT masque -> token : la MEME fonction que le chemin `hdirect_masked`
    (`probe1_common.chunk_token_mask` : frame centrale du chunk -> grille de salience ->
    seuil MASK_CELL_THR a la majorite -> tuilage sur les F frames), JAMAIS recopiee. Seule
    difference, scellee au registre : `min_frac=0.0` coupe le repli top-3 %, le repli est
    une regle de lecture ("ne jamais pooler le vide") ; a l'ECRITURE la regle scellee est
    "chunk sans token sujet -> ecriture sautee", et un repli l'aurait rendue inatteignable.

      Subject    : les tokens dont la cellule passe le seuil d'appartenance au masque-sujet.
      bg_matched : pour chaque chunk, EXACTEMENT LE MEME NOMBRE de tokens, tires sans remise
                   parmi les tokens hors masque (le complement du bras subject), memes chunks
                   sautes -> calendriers d'ecriture apparies, seule la selection change.
                   Tirage deterministe : graine = sha1("oracle_maskwrite", video_id, chunk),
                   jamais d'horloge ni d'etat global.
                   cas LIMITE (constate au dry-check : sujet > 50 % du chunk, ex. twin:9000000
                   ecriture 2, 2418 > 2262) : l'appariement exact sans remise est impossible ;
                   le bras prend alors tout le complement et le deficit est enregistre dans
                   info["bg_deficit"] et logge, jamais silencieux, a declarer au rapport.

    Le masque ne depend pas du bloc (geometrie d'image) : une seule liste pour les 3 blocs,
    exactement comme dans `build_signatures.load_payload` (source hdirect_masked)."""
    assert arm in MASKWRITE_ARMS, f"bras inconnu {arm!r}"
    if n_tokens is None:
        n_tokens = int(meta["num_frame_per_block"]) * C.TOKENS_PER_FRAME
    grids = C.video_mask_grids(meta["video_id"])
    n = len(meta["write_chunks"])
    subj = []
    for j in range(n):
        m, fb = C.chunk_token_mask(meta, j, grids, n_tokens=n_tokens, min_frac=0.0)
        assert not fb, "min_frac=0.0 : le repli ne peut pas se declencher"
        subj.append(np.flatnonzero(m))          # trie par construction
    keep, bg_deficit = [], {}
    for j, s in enumerate(subj):
        if s.size == 0:
            keep.append(None)                   # ecriture SAUTEE (les DEUX bras)
            continue
        if arm == "subject":
            keep.append(s)
            continue
        comp = np.setdiff1d(np.arange(n_tokens, dtype=np.int64), s)
        assert comp.size > 0, f"{meta['video_id']} ecriture {j}: masque-sujet PLEIN"
        if comp.size < s.size:                  # cas limite documente ci-dessus
            bg_deficit[j] = int(s.size - comp.size)
            keep.append(comp)
            continue
        seed = int.from_bytes(hashlib.sha1(
            f"oracle_maskwrite|{meta['video_id']}|{int(meta['write_chunks'][j])}"
            .encode()).digest()[:4], "big")
        rng = np.random.RandomState(seed)
        pick = rng.choice(comp.size, size=int(s.size), replace=False)
        sel = np.sort(comp[pick])               # trie : l'ordre des tokens fixe l'arrondi
        assert sel.size == s.size and np.intersect1d(sel, s).size == 0
        keep.append(sel)
    info = {"arm": arm, "n_writes": n, "n_tokens": int(n_tokens),
            "subject_counts": [int(s.size) for s in subj],
            "kept_counts": [(0 if k is None else int(k.size)) for k in keep],
            "skipped_writes": [j for j, k in enumerate(keep) if k is None],
            "bg_deficit": bg_deficit}
    if bg_deficit:
        print(f"[maskwrite/{arm}] {meta['video_id']}: sujet > 50 % du chunk sur "
              f"{len(bg_deficit)} ecriture(s) {sorted(bg_deficit)} -> le fond-apparie prend "
              f"TOUT le complement (deficits {bg_deficit}) ; appariement exact impossible la",
              flush=True)
    return keep, info


# ---------------------------------------------------------------------------------------------
# oracle de la regle D'ECRITURE LINEAIRE (registre : section "oracle de la regle D'ECRITURE
# LINEAIRE", go 2026-08-06 soir). Etat par bloc : W (d x d), init 0, accumulation par produits
# exterieurs. Zero parametre appris, regles fixes : diagnostic pur, aucune cellule SwiGLU ici.
#   mean  : W = mean_t(v_t k_t^T) sur tous les tokens ecrits depuis le debut de la video
#           (moyenne cumulative, pas par chunk), k = h/||h|| (L2 par token), v = h ;
#   delta : W <- W + beta (v - W k) k^T, beta = 0.5, SEQUENTIEL token par token dans l'ordre
#           du chunk (= ordre naturel des indices, celui que rend maskwrite_keep, trie).
# Masquage : EXACTEMENT le meme chemin que l'oracle precedent (maskwrite_keep -> C.
# chunk_token_mask min_frac=0.0, chunk vide -> ecriture sautee). Accumulation fp32, snapshot
# fp16 avec l'assert de dynamique, memes ages que les etats SwiGLU.
# ---------------------------------------------------------------------------------------------
LIN_BETA = 0.5
# nom de source -> (regle, masque-sujet ?) ; les noms sont AUSSI les sous-repertoires de sig/
# et les --source de train_probe : une seule nomenclature partout.
LIN_SPEC = {"lin_mean_subject": ("mean", True),    # L1, primary
            "lin_mean_all": ("mean", False),       # L2
            "lin_delta_subject": ("delta", True),  # L3
            "lin_delta_all": ("delta", False)}     # L4


def lin_chunk_update(x, rule, S, N, W, beta=LIN_BETA):
    """UNE ecriture lineaire sur les tokens x (T, d) fp32. Rend (S, N, W) mis a jour.
    C'est LE coeur des deux regles, teste analytiquement par selftest_lin T1 :
      mean  : S <- S + somme_t v_t k_t^T (= x^T K), N <- N + T ; W du snapshot = S / N ;
      delta : W <- W + beta (v_t - W k_t) k_t^T, token par token, ordre des lignes de x."""
    assert x.dim() == 2 and x.dtype == torch.float32, f"tokens {tuple(x.shape)} {x.dtype}"
    nrm = x.norm(dim=-1, keepdim=True)
    assert torch.isfinite(nrm).all() and bool((nrm > 0).all()), "token de norme nulle"
    K = x / nrm                                   # k = h / ||h||, v = h
    if rule == "mean":
        return S + x.transpose(0, 1) @ K, N + int(x.shape[0]), W
    assert rule == "delta", f"regle inconnue {rule!r}"
    for t in range(x.shape[0]):                   # sequentiel, scelle au registre
        k = K[t]
        W.add_(torch.outer(x[t] - W @ k, k), alpha=beta)
    return S, N + int(x.shape[0]), W              # N = compteur de tokens (stats seulement ici)


@torch.no_grad()
def lin_states(meta, cache, blocks, ages, device, rule, masked):
    """Rejoue la regle d'ecriture LINEAIRE d'une video et snapshotte W aux ages demandes.
    Rend (out, stats) : out = {age_i: {bloc: W fp16 cpu}} (meme indexation que replay_video,
    donc le mode --stream de build_signatures les consomme tel quel) ; stats = normes de
    Frobenius fp32 par (age, bloc) + comptes de tokens (les stats loggees exigees).
    Le masque ne depend pas du bloc : maskwrite_keep une fois, partage par les 3 blocs."""
    assert rule in ("mean", "delta")
    idxs = sorted({int(a["i"]) for a in ages})
    n = len(meta["write_chunks"])
    assert idxs and idxs[-1] < n, f"{meta['video_id']}: age {idxs[-1]} >= {n} ecritures"
    keep = maskwrite_keep(meta, "subject")[0] if masked else None
    # fp16 -> fp32 est exact : les regles lineaires n'ont pas de dtype de deploiement a imiter
    # (aucune cellule), le registre scelle l'accumulation fp32.
    h = D.load_h(meta, cache, blocks, upto=idxs[-1], device=device, dtype=torch.float32)
    d = int(h[blocks[0]][0].shape[-1])
    out = {}
    stats = {"norms": {}, "kept_counts": [], "n_cum_tokens": {}}
    for bi in blocks:
        S = torch.zeros(d, d, dtype=torch.float32, device=device)
        W = torch.zeros(d, d, dtype=torch.float32, device=device)
        N, prev = 0, 0
        for i in idxs:
            for j in range(prev, i + 1):
                x = h[bi][j]
                x = x[0] if x.dim() == 3 else x
                if keep is not None:
                    if keep[j] is None:
                        if bi == blocks[0]:
                            stats["kept_counts"].append(0)
                        continue                  # ecriture SAUTEE (etat inchange)
                    x = x.index_select(0, torch.as_tensor(np.asarray(keep[j]),
                                                          dtype=torch.long, device=x.device))
                if bi == blocks[0]:
                    stats["kept_counts"].append(int(x.shape[0]))
                S, N, W = lin_chunk_update(x, rule, S, N, W)
            Wi = (S / N) if rule == "mean" and N > 0 else (W if rule == "delta" else S)
            W16 = Wi.to(torch.float16)
            assert torch.isfinite(W16).all(), \
                (f"{meta['video_id']} bloc {bi} age {i}: W hors dynamique fp16 "
                 f"(|max| = {float(Wi.abs().max()):.4g})")
            out.setdefault(i, {})[bi] = W16.cpu()
            stats["norms"].setdefault(i, {})[bi] = float(Wi.norm())
            stats["n_cum_tokens"][i] = N
            prev = i + 1
    del h
    return out, stats


# ---------------------------------------------------------------------------------------------
# arm MLP-CORRECTEUR (registre : section "arm MLP-CORRECTEUR", scelle 07/08 AVANT
# implementation). Diagnostic par REJEU, zero entrainement : meme cellule l1@2000-EMA, memes h,
# meme Muon / weight_norm_update / lr par token / decroissance / plafonds de norme / ages /
# sondes / pooling / plis / corpus. La SEULE chose qui change est l'OBJECTIF interne d'une
# ecriture.
#
# derivation (2 lignes). Pose f_W(k) = (silu(k w0) * (k w2)) w1 et J = df_W(k)/dW.
#   (1) L'existant monte le produit scalaire  dot = <f_W(k), v>, dont le gradient est J^T v :
#       c'est EXACTEMENT ce que rend `swiglu_fast_weight_gradient(k, v, ...)` (v traite comme
#       une constante), puis `weight_norm_update` fait  w <- decay*w + muon(J^T v)  (MONTEE).
#   (2) L'objectif correcteur est  L = 1/2 ||f_W(k) - v||^2, de gradient  dL/dW = J^T (f_W(k) - v).
#       Donc  -dL/dW = J^T (v - f_W(k))  =  `swiglu_fast_weight_gradient(k, v - f_W(k), ...)`.
#       Passer ce RESIDU a la place de v et GARDER la montee existante donne donc
#       w <- decay*w - muon(dL/dW) : la cellule descend l'erreur. Le signe est bien celui-la
#       parce que Newton-Schulz est impair (A = X X^T est paire en X, l'iteration
#       X <- aX + (bA + cA^2)X est donc impaire, et la normalisation X/||X|| aussi) :
#       muon(-G) = -muon(G), la descente ne peut pas etre absorbee par l'orthogonalisation.
#   Verifie numeriquement par selftest_mlp_delta.py D1 (autograd) et D2 (l'erreur baisse).
#
# plan FACTORIEL en UNE ronde (registre : "AMENDEMENT (Aden 07/08)", scelle AVANT execution).
# Trois facteurs, tous les bras sur tous les tokens :
#   (R) objectif  : residu v - f_W(k)      vs Hebb <f_W(k), v>
#   (S) calendrier: SEQUENTIEL par micro-lots dans le chunk vs un pas moyenne sur tout le chunk
#   (M) pas       : gradient brut (muon_update_steps = 0, "0 = plain sgd") vs Muon (5 pas NS)
# Les 5 cellules (A0 = notre ecriture actuelle, deja mesuree 8.00/19, JAMAIS relancee ici) :
#   A0  Hebb   + moyenne   + Muon   -> le bras `state` historique
#   A1  residu + moyenne   + Muon   -> mlp_delta_all
#   A2  residu + sequentiel + brut  -> mlp_full        (primary *initial*, cf. arbitrage)
#   A3  residu + sequentiel + Muon  -> mlp_full_muon   (isole l'orthogonalisation)
#   A4  Hebb   + sequentiel + brut  -> mlp_hebb_seq    (isole la correction d'erreur)
# ce plan A ETE amende LE 07/08 AVANT tout score (registre : "reparation DU CRITERE AVEUGLE") :
# le primary est desormais A3, A2 devient l'ablation A2' (brut, pas x0.5) et A4 passe sous Muon
# pour rester apparie au primary. La table MLP_ARMS plus bas fait foi.
# ce QUI ne change pas, dans tous les bras : cellule l1@2000-EMA, h du disque, _qkv_lr fusionne,
# lr appris par token, decroissance (UNE application par ecriture, cf. write_only), plafonds de
# norme, ages, sondes, compression, plis, corpus.
# approximation ASSUMEE : le "sequentiel" est par micro-lots (le pur token-par-token coute
# ~4680 pas par chunk et par bloc) ; la taille de micro-lot et le facteur de pas sont choisis
# AVANT tout score d'identite par le critere AVEUGLE scelle au registre (maximiser la reduction
# relative moyenne de ||f_W(k) - v|| par ecriture sous contrainte de sanite numerique), mesure
# par sweep_mlp_step.py sur 2 videos. Le pur sequentiel (micro-lot = 1) est mesure en controle.
# constantes CHOISIES par LE CRITERE AVEUGLE (sweep_mlp_step.py, 2 videos, 3 blocs, 25 ecritures,
# 07/08 ; artefacts : runs/sweep_mlp_step*.json). Aucun score d'identite n'entre dans ce choix.
# extension de la plage de pas : RATIFIEE (registre, "reparation DU CRITERE AVEUGLE"). Motif :
# la plage scellee {0.5, 1, 2} ne contient AUCUNE cellule ou le bras brut descend son propre
# objectif, ses 18 cellules ont une reduction relative R < 0 (de -0.45 a -345), c'est-a-dire
# que l'ecriture AUGMENTE l'erreur qu'elle est censee corriger : le pas brut y est 1 a 2 ordres
# de grandeur trop grand (les etats intermediaires explosent puis sont rabattus par le plafond
# de norme, selftest D2b : 2529 -> 3.06e4 -> 3.9e7 -> 1957). La plage a ete PROLONGEE VERS LE
# bas (0.0003 ... 0.25), meme critere, toujours aveugle a l'identite : R devient positif, mais
# SEULEMENT la ou l'ecriture ne pese presque plus rien, d'ou la contrainte de part ecrite.
# ARBITRAGE (registre, section "reparation DU CRITERE AVEUGLE", scelle 07/08 AVANT tout score) :
# le critere devient une optimisation sous CONTRAINTE, parmi les reglages dont la part ECRITE
# ||etat - etat_sans_ecriture|| / ||etat|| est >= 0.5 (l'ecriture domine l'initialisation ;
# ancrage non arbitraire : A0 = 0.88, A1 = 0.73), maximiser la reduction d'erreur. Les deux
# quantites sont internes a l'ecriture : le critere reste AVEUGLE A L'IDENTITE. Le maximum sans
# contrainte etait DEGENERE (n'ecris rien -> ne degrade rien -> R max, mais l'etat reste ~ son
# init, lequel est IDENTIQUE pour toutes les videos).
# SEULE cellule qui satisfait la contrainte ET descend : residu + sequentiel + MUON en
# micro-lot 512 (part 0.952, R = +0.0251) -> c'est le NOUVEAU primary. Sans Muon c'est l'un ou
# l'autre : soit l'ecriture domine et l'erreur explose (part 1.03, R = -0.45 a -48), soit
# l'erreur descend et l'ecriture est negligeable (part 0.29, R = +0.075). L'orthogonalisation
# borne la taille du pas, ce qui permet d'ecrire beaucoup sans diverger.
MLP_MB_MUON = 512        # calendrier des bras Muon (A3 primary et A4) : le SEUL apparie
MLP_MB_RAW = 32          # A2' : impose par sa physique (le brut ne tient pas le pas a 512)
MLP_RAW_SCALE = 0.5      # A2' : le pas ou l'ecriture domine l'etat (part 1.034, R = -0.577)
# DISSYMETRIE ASSUMEE ET DECLAREE : tous les bras sont en micro-lot 512 sauf A2', qui ne peut
# pas y etre (a 512 le gradient brut a part ecrite 1.04 et R = -124). A2'/A3 ne sont donc pas
# apparies sur le calendrier : c'est une contrainte physique du bras brut, pas un choix
# opportuniste, et elle est rapportee telle quelle a cote des scores.
# PLAFOND declare : le pur sequentiel (micro-lot 1, pas x0.01 : R +0.0749, part 0.294) est le
# meilleur reglage brut sur les deux axes mais il coute ~3.9 h par bras sur 60 videos : non
# couru. Il ne satisfait pas non plus la contrainte de part ecrite (0.294 < 0.5).
# Pourquoi la moyenne est un axe et pas un detail : le gradient d'un micro-lot de m tokens est
# une SOMME sur ces m tokens ; sans normalisation, la taille du pas serait proportionnelle a m
# et le facteur "calendrier" (S) serait confondu avec le facteur "pas" (M). En moyennant, le
# pas par token est le MEME quel que soit m, donc m ne regle plus que le calendrier, et
# m = 1 (pur sequentiel) devient exactement le cas limite de la meme famille. Le sweep mesure
# les deux (somme ET moyenne) et le critere aveugle tranche. Note d'algebre : "somme au pas s"
# EST "moyenne au pas s*m", l'axe somme/moyenne n'ajoute donc rien une fois la plage de pas
# assez large, il ne fait que deplacer la meme famille de reglages.

MLP_ARMS = {
    # nom                regle     masque micro-lot    Muon  pas             moyenne
    "mlp_delta_all":     ("delta", False, None,        None, 1.0,            False),  # A1
    "mlp_delta_subject": ("delta", True,  None,        None, 1.0,            False),  # A1 masque (hors plan)
    "mlp_full":          ("delta", False, MLP_MB_RAW,  0,    MLP_RAW_SCALE,  True),   # A2'
    "mlp_full_muon":     ("delta", False, MLP_MB_MUON, None, 1.0,            False),  # A3 primary
    "mlp_hebb_seq":      ("hebb",  False, MLP_MB_MUON, None, 1.0,            False),  # A4
}
# LE plan, apres l'arbitrage (A0 = notre ecriture actuelle, deja mesuree 8.00/19, jamais relancee) :
#   A3 mlp_full_muon  residu + sequentiel 512 + Muon  <- primary, seuil scelle >= 14/19
#   A1 mlp_delta_all  residu + moyenne (1 pas) + Muon  : isole le CALENDRIER (S) face a A3
#   A4 mlp_hebb_seq   Hebb   + sequentiel 512 + Muon   : isole l'OBJECTIF (R) face a A3, il
#       reprend donc EXACTEMENT le calendrier et le pas du primary, seul l'objectif change
#       (c'est la regle scellee de A4, reportee du bras brut vers le nouveau primary)
#   A2' mlp_full      residu + sequentiel 32 + brut x0.5 : isole MUON (M), ecrit (part 1.034)
#       mais ne descend pas (R = -0.577), son echec est attendu et informatif
# `micro-lot None` = un seul pas sur le chunk (le calendrier historique) ; `Muon None` = la
# valeur de la cellule (muon_update_steps = 5 du config), 0 = le chemin brut deja prevu par
# zeropower_via_newtonschulz5 ("0 = plain sgd, keep the raw grad").
# A4 reprend EXACTEMENT le micro-lot et le pas de A2 : entre A2 et A4, seul l'OBJECTIF change.
MLP_SPEC = {name: spec[1] for name, spec in MLP_ARMS.items()}   # nom -> masque (compat)


def swiglu_delta_gradient(k, v, w0, w1, w2, lr0, lr1, lr2):
    """Gradient interne de l'objectif CORRECTEUR D'ERREUR 1/2||f_W(k) - v||^2, au signe de la
    MONTEE (cf. derivation ci-dessus : ce qui est rendu est -dL/dW, exactement comme l'existant
    rend +d(dot)/dW, si bien que `weight_norm_update` est utilise TEL QUEL).

    Signature identique a `swiglu_fast_weight_gradient` : c'est le `grad_fn` que
    `write_only` accepte. Rien n'est recopie, la sortie f_W(k) vient de `swiglu_apply`
    (la fonction que `step()` utilise pour lire) et le gradient de
    `swiglu_fast_weight_gradient` (la fonction que l'ecriture historique utilise), toutes
    deux importees de wan.modules.video_memory."""
    resid = v - swiglu_apply(k, w0, w1, w2)          # = -(f_W(k) - v), etat AVANT mise a jour
    return swiglu_fast_weight_gradient(k, resid, w0, w1, w2, lr0, lr1, lr2)


def scaled_gradient(base, scale, mean_over_tokens=False):
    """Facteur de pas (facteur M du plan) : `base` mis a l'echelle. `scale = 1.0` sans moyenne
    rend `base` lui-MEME (aucun enrobage, donc aucun changement possible du chemin par defaut).
    `mean_over_tokens` divise par le NOMBRE de TOKENS du micro-lot (k.shape[-2]) : le gradient
    devient une moyenne au lieu d'une somme, donc le pas par token ne depend plus de la taille
    du micro-lot. Mettre le gradient a l'echelle avant `weight_norm_update` == mettre le pas a
    l'echelle sur le chemin brut ; sous Muon l'echelle est ABSORBEE par l'orthogonalisation
    (X / ||X||), c'est pourquoi les bras Muon gardent 1.0 sans moyenne."""
    if float(scale) == 1.0 and not mean_over_tokens:
        return base
    s = float(scale)

    def g(k, v, w0, w1, w2, lr0, lr1, lr2):
        g0, g1, g2 = base(k, v, w0, w1, w2, lr0, lr1, lr2)
        f = s / int(k.shape[-2]) if mean_over_tokens else s
        return g0 * f, g1 * f, g2 * f
    return g


def mlp_grad_fn(source):
    """Nom de bras -> `grad_fn` a passer au rejeu (objectif R + facteur de pas).
    Le bras Hebb rend `swiglu_fast_weight_gradient` lui-MEME quand son facteur vaut 1.0 sans
    moyenne : l'objectif historique, sans le moindre enrobage."""
    rule, _masked, _mb, _muon, scale, mean = MLP_ARMS[source]
    assert rule in ("delta", "hebb"), f"regle inconnue {rule!r}"
    base = swiglu_delta_gradient if rule == "delta" else swiglu_fast_weight_gradient
    return scaled_gradient(base, scale, mean_over_tokens=mean)


@contextlib.contextmanager
def mlp_arm(cells, source):
    """Le bras `source` monte sur les cellules, pour la duree du bloc `with`.

    Rend (grad_fn, micro_batch) et force `muon_update_steps` sur chaque cellule quand le bras
    demande le gradient brut (facteur M). C'est le reglage de cellule prevu par le code
    (`muon_update_steps: 0 = plain sgd, keep the raw grad`), pas un nouveau chemin. La valeur
    d'origine est TOUJOURS restauree (try/finally), y compris sur exception : selftest D5."""
    _rule, _masked, mb, muon, _scale, _mean = MLP_ARMS[source]
    saved = {b: c.muon_update_steps for b, c in cells.items()}
    try:
        if muon is not None:
            for c in cells.values():
                c.muon_update_steps = int(muon)
        yield mlp_grad_fn(source), mb
    finally:
        for b, c in cells.items():
            c.muon_update_steps = saved[b]


@torch.no_grad()
def replay_video(cells, blocks, meta, cache, ages, device, keep=None, grad_fn=None,
                 micro_batch=None):
    """Rejoue toute la sequence d'ecritures d'une video et rend l'etat aux indices demandes.

    On ne materialise pas les 35 etats intermediaires (3 blocs x 170 MB) : on avance par
    TRONCONS entre deux points de snapshot en repassant l'etat courant, exactement l'idiome de
    probe_p2p3_gain_steps.py. Le forward de replay_writes est inchange, donc la trajectoire est
    bit a bit celle de l'inference.

    `keep` (ADDITIF, oracle d'ecriture masquee) : None = bras `none`, le chemin HISTORIQUE
    ci-dessous, INTOUCHE (selftest S1 : bit a bit les etats du disque). Sinon, une liste
    indexee par ecriture (sortie de `maskwrite_keep`) : keep[j] = indices de tokens a ecrire
    (l'ecriture ne voit que h[:, keep[j], :]) ou None = ecriture SAUTEE (etat inchange).
    L'ecriture elle-meme reste `replay_writes` -> `write_only` : AUCUNE logique d'ecriture
    n'est dupliquee, on ne change que la LISTE de tokens presentee a chaque write.

    `grad_fn` (ADDITIF, arm MLP-CORRECTEUR 07/08) : l'objectif interne d'une ecriture.
    None = le chemin par DEFAUT, bit a bit l'existant (selftest D3 : egal aux etats du
    disque) ; `swiglu_delta_gradient` = le bras correcteur d'erreur.
    `micro_batch` (ADDITIF, plan FACTORIEL 07/08) : calendrier de mise a jour dans le chunk.
    None = un seul pas (historique) ; m > 0 = sequentiel par micro-lots de m tokens."""
    idxs = sorted({int(a["i"]) for a in ages})
    n = len(meta["write_chunks"])
    assert idxs and idxs[-1] < n, f"{meta['video_id']}: age {idxs[-1]} >= {n} ecritures"
    if keep is not None:
        assert len(keep) == n, f"{meta['video_id']}: keep de taille {len(keep)} != {n} ecritures"
    h = D.load_h(meta, cache, blocks, upto=idxs[-1], device=device, dtype=torch.bfloat16)
    out = {}
    for bi in blocks:
        state, prev = None, 0
        for i in idxs:
            if keep is None:
                state = replay_writes(cells[bi], h[bi][prev:i + 1], state=state,
                                      detach_state=True, detach_init=True, n_bptt_writes=0,
                                      grad_fn=grad_fn, micro_batch=micro_batch)
            else:
                if state is None:
                    # exactement l'init de replay_writes (init_state + detach_init), pose ICI
                    # parce qu'un prefixe entierement saute doit quand meme rendre l'etat init.
                    state = tuple(t.detach() for t in cells[bi].init_state(
                        int(h[bi][0].shape[0]), h[bi][0].device, cells[bi].qkv.weight.dtype))
                for j in range(prev, i + 1):
                    if keep[j] is None:
                        continue                                   # ecriture SAUTEE
                    kj = torch.as_tensor(np.asarray(keep[j]), dtype=torch.long,
                                         device=h[bi][j].device)
                    assert int(h[bi][j].shape[-2]) % C.TOKENS_PER_FRAME == 0 \
                        and int(kj.max()) < int(h[bi][j].shape[-2]), \
                        f"{meta['video_id']} ecriture {j}: indices hors du chunk"
                    state = replay_writes(cells[bi], [h[bi][j].index_select(1, kj)],
                                          state=state, detach_state=True, detach_init=True,
                                          n_bptt_writes=0, grad_fn=grad_fn,
                                          micro_batch=micro_batch)
            # bf16 -> fp16 est exact tant qu'on reste dans la dynamique fp16 (fp16 a plus de
            # bits de mantisse que bf16) : on l'asserte plutot que de l'esperer.
            st16 = tuple(t.detach().to(torch.float16).cpu() for t in state)
            assert all(torch.isfinite(t).all() for t in st16), \
                (f"{meta['video_id']} bloc {bi} age {i}: etat hors dynamique fp16 "
                 f"(|max| = {max(float(t.abs().max()) for t in state):.4g})")
            out.setdefault(i, {})[bi] = st16
            prev = i + 1
    del h
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=C.DEFAULT_CONFIG)
    ap.add_argument("--cells_mode", default="direct", choices=("direct", "pipeline"),
                    help="'direct' reconstruit la cellule depuis le ckpt (leger) ; 'pipeline' "
                         "passe par le harness complet. Les deux sont assertes egaux a theta_l1.")
    ap.add_argument("--corpus", default="v1", choices=("v1", "v2"))
    ap.add_argument("--only_pairs", action="store_true", help="les 60 videos des 30 paires")
    ap.add_argument("--all", action="store_true", help="les 165 videos (41 GB -> 112 GB !)")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--n_ages", type=int, default=C.N_AGES)
    ap.add_argument("--redo", action="store_true")
    ap.add_argument("--write-mask", dest="write_mask", default="none",
                    choices=("none",) + MASKWRITE_ARMS,
                    help="oracle d'ecriture masquee : `none` = l'existant (bit a bit, ecrit "
                         "dans states/) ; `subject` = chaque ecriture ne voit que les tokens "
                         "sous le masque-sujet (seuil hdirect_masked, chunk vide -> ecriture "
                         "sautee) ; `bg_matched` = meme NOMBRE de tokens hors masque, memes "
                         "chunks sautes. Sorties sous oracle_maskwrite/states_<bras>/")
    args = ap.parse_args()

    assert args.only_pairs or args.all or args.limit, "precise --only_pairs, --all ou --limit"
    C.ensure_dirs()
    C.set_determinism(0)
    device = "cuda"
    out_dir = C.STATE_DIR if args.write_mask == "none" else maskwrite_dir(args.write_mask)
    os.makedirs(out_dir, exist_ok=True)
    config = OmegaConf.load(args.config)
    blocks = sorted(int(x) for x in config.model_kwargs.memory_layers)

    cells, blocks_c, _vae = build_cells(config, device, mode=args.cells_mode)
    assert blocks_c == blocks, f"{blocks_c} != {blocks}"
    ref, src = ref_theta_l1(config, blocks, device)
    d = delta_l1(cells, ref)
    assert d == 0.0, f"contrat F2.3 viole : delta_l1 = {d} (les cellules ne sont pas theta_l1)"
    print(f"[states] cellules {blocks} depuis {src} de {config.lora_ckpt}, delta_l1 = {d}",
          flush=True)

    vids, pairs = C.pair_dataset(verbose=True, corpus=args.corpus)
    pair_ids = {p["main"] for p in pairs} | {p["twin"] for p in pairs}
    sel = [v for v in vids if (args.all or v["video_id"] in pair_ids)]
    sel.sort(key=lambda v: (v["part"], v["vi"]))
    if args.limit:
        sel = sel[:args.limit]

    cache = D.PartCache(size=1)
    manifest, t0, nb = {}, time.time(), 0
    for n, meta in enumerate(sel):
        ages = C.state_ages(meta, args.n_ages)
        paths = {a["name"]: state_path(meta["video_id"], a["name"], out_dir) for a in ages}
        keep, info = (None, None) if args.write_mask == "none" else \
            maskwrite_keep(meta, args.write_mask)
        if not args.redo and all(os.path.exists(p) for p in paths.values()):
            manifest[meta["video_id"]] = {"ages": ages, "paths": paths, "keep_info": info}
            continue
        got = replay_video(cells, blocks, meta, cache, ages, device, keep=keep)
        for a in ages:
            torch.save({"video_id": meta["video_id"], "age_name": a["name"],
                        "write_index": int(a["i"]), "n_writes": len(meta["write_chunks"]),
                        "blocks": blocks, "state": got[int(a["i"])],
                        "writer": os.path.abspath(config.lora_ckpt), "src": src,
                        "write_mask": args.write_mask, "keep_info": info},
                       paths[a["name"]])
            nb += sum(t.numel() * t.element_size() for st in got[int(a["i"])].values() for t in st)
        manifest[meta["video_id"]] = {"ages": ages, "paths": paths, "keep_info": info}
        if info is not None:
            kc = [k for k in info["kept_counts"] if k > 0]
            print(f"[states/{args.write_mask}] {meta['video_id']}: "
                  f"{info['n_writes']} ecritures, {len(info['skipped_writes'])} sautees "
                  f"({len(info['skipped_writes']) / info['n_writes']:.0%}), tokens/chunk "
                  f"garde moy {np.mean(kc) if kc else 0:.1f} "
                  f"min {min(kc) if kc else 0} max {max(kc) if kc else 0} / {info['n_tokens']}",
                  flush=True)
        if (n + 1) % 5 == 0 or n == len(sel) - 1:
            el = time.time() - t0
            print(f"[states] {n + 1}/{len(sel)} videos, {nb / 1e9:.1f} GB ecrits, "
                  f"{el / 60:.1f} min ({el / max(1, n + 1):.1f} s/video)", flush=True)

    C.save_json({"n_videos": len(sel), "blocks": blocks, "n_ages": args.n_ages,
                 "config": os.path.abspath(args.config), "cells_mode": args.cells_mode,
                 "write_mask": args.write_mask, "corpus": args.corpus,
                 "writer_ckpt": os.path.abspath(config.lora_ckpt), "writer_src": src,
                 "delta_l1": float(d), "manifest": manifest},
                os.path.join(out_dir, "_manifest.json"))
    print(f"[states] TERMINE : {len(sel)} videos x {args.n_ages} ages "
          f"(write_mask={args.write_mask}) -> {out_dir}")


if __name__ == "__main__":
    main()
