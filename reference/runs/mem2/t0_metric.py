#!/usr/bin/env python
# ============================================================================================
# MEMOIRE 2.1 -- LA METRIQUE T0 (correlation bande-basse), PARTAGEE par le builder de banque,
# le trainer, le pre-ecran et les selftests.  UNE seule definition, importee partout.
#
# POURQUOI T0 ET PAS DINO / LE COSINUS LATENT (les deux termes d'instance deja morts) :
#   - DINO (phase B) : gate pos_gap = -0.185, 10/10 val negatives, et la dose-reponse de flou
#     montre que meme la VERITE TERRAIN floutee a sigma=8 px echoue le gate. Le cosinus DINO
#     d'une image hors-variete mesure "a quel point c'est flou", pas "qui c'est".
#   - cosinus latent crop60 (J3) : pos_gap -0.19..-0.36 a tous les poolings, centre ou non.
#   - T0 est la SEULE metrique du dossier qui (a) VOIT le defaut (elle mesure ancre vs A* +0.042
#     contre vs B +0.298, B > A sur 21/24, p = 3e-4) et (b) TOLERE le flou par construction
#     (moyenne n x n : elle vit dans la bande que le decodeur atteint reellement).
#
# DEFINITION GELEE (identique a reference/runs/reid_scen/prescreen_offensif.lowband, verifiee
# BITWISE par selftest_t0.py test 1 -- la fonction de reference est importee, pas recopiee) :
#     frames (T,3,H,W) in [0,1] -> niveaux de gris 0.299/0.587/0.114
#                               -> adaptive_avg_pool2d(n) -> aplatir
#                               -> CENTRER par frame -> L2-normaliser
#     corr(ancre, ensemble) = moyenne sur les frames de l'ensemble du produit scalaire.
# Deux vecteurs unitaires centres : le produit scalaire EST la correlation de Pearson.
#
# VARIANTE LATENTE (l'espace a departager, regle D0 du registre) : memes operations sur le
# latent (B,16,60,104), avec deux reductions de canal possibles
#     'chan' : pooling n x n PAR CANAL   -> dim 16*n*n (garde la couleur/structure par canal)
#     'gray' : moyenne des 16 canaux     -> dim n*n    (l'analogue exact du niveau de gris)
# Le choix est fait par space_decision_t0.py (regle D0 scellee), pas par gout.
#
# TOUT EST DIFFERENTIABLE cote ancre (c'est le point : le gradient du terme T0 remonte par le
# decodeur jusqu'a l'ecrivain). Aucune operation en place, aucun no_grad ici.
# ============================================================================================
import math

import torch
import torch.nn.functional as Fn

DEFAULT_N = 8                 # la bande de tout le dossier (16.1 : la copie survit a n=4..64)
GRAY_W = (0.299, 0.587, 0.114)


# ----------------------------- le coeur ---------------------------------------------------------
def _finish(z, eps=1e-6, return_scale=False):
    """Aplatir -> centrer par frame -> L2-normaliser. Deux sorties de _finish ont un produit
    scalaire EGAL a la correlation de Pearson de leurs entrees.
    `return_scale` renvoie EN PLUS la norme AVANT normalisation (= le CONTRASTE de la bande
    basse, exactement ce que la normalisation efface). C'est le tripwire anti-effondrement de
    G1, et il doit vivre dans l'espace de la metrique -- AUDIT 2026-08-04."""
    z = z.flatten(1)
    z = z - z.mean(1, keepdim=True)
    s = z.norm(dim=1, keepdim=True).clamp_min(eps)
    zn = z / s
    return (zn, s[:, 0] / math.sqrt(z.shape[1])) if return_scale else zn


def lowband_px(frames01, n=DEFAULT_N, return_scale=False):
    """(T,3,H,W) in [0,1] -> (T, n*n) bande basse en niveaux de gris.
    IDENTIQUE a prescreen_offensif.lowband (selftest 1 le verifie bitwise)."""
    assert frames01.dim() == 4 and frames01.shape[1] == 3, \
        f"lowband_px attend (T,3,H,W), recu {tuple(frames01.shape)}"
    gray = (GRAY_W[0] * frames01[:, 0] + GRAY_W[1] * frames01[:, 1]
            + GRAY_W[2] * frames01[:, 2])[:, None]
    return _finish(Fn.adaptive_avg_pool2d(gray, n), return_scale=return_scale)


def lowband_lat(lat, n=DEFAULT_N, mode="chan", return_scale=False):
    """Latent (B,16,H,W) ou (B,1,16,H,W) -> (B, D) bande basse latente.
    mode 'chan' : D = C*n*n (pooling par canal)  |  'gray' : D = n*n (moyenne des canaux)."""
    if lat.dim() == 5:
        assert lat.shape[1] == 1, f"une frame latente a la fois, recu {tuple(lat.shape)}"
        lat = lat[:, 0]
    assert lat.dim() == 4, f"lowband_lat attend (B,C,H,W), recu {tuple(lat.shape)}"
    x = lat.float()
    if mode == "gray":
        x = x.mean(dim=1, keepdim=True)
    elif mode != "chan":
        raise ValueError(f"mode inconnu {mode!r} (chan|gray)")
    return _finish(Fn.adaptive_avg_pool2d(x, n), return_scale=return_scale)


def t0_embed(x, space, n=DEFAULT_N, lat_mode="chan", vae=None, use_checkpoint=False,
             return_scale=False):
    """L'entree unique du terme T0 et de la banque.
      space='latent' : x = frames latentes (B,16,60,104) ou (B,1,...)   -> lowband_lat
      space='pixel'  : x = frames latentes -> VAE decode (DIFFERENTIABLE) -> lowband_px
    Le VAE n'est jamais appele en espace latent (le terme reste gratuit).
    `return_scale` renvoie (z, contraste) ou le contraste est l'ecart-type spatial de la bande
    basse DANS L'ESPACE DE LA METRIQUE (tripwire G1)."""
    if space == "latent":
        return lowband_lat(x, n=n, mode=lat_mode, return_scale=return_scale)
    if space == "pixel":
        assert vae is not None, "space='pixel' exige un VAE (build_bank_dino.latent_to_pixels)"
        from build_bank_dino import latent_to_pixels
        px = latent_to_pixels(vae, x, use_checkpoint=use_checkpoint)
        return lowband_px(px.clamp(0, 1), n=n, return_scale=return_scale)
    raise ValueError(f"space inconnu {space!r} (latent|pixel)")


def t0_dim(space, n=DEFAULT_N, lat_mode="chan", C=16):
    return n * n if (space == "pixel" or lat_mode == "gray") else C * n * n


# ----------------------------- correlations -------------------------------------------------------
def corr_set(z, E):
    """z (K,D) ancres unitaires, E (M,D) references unitaires -> (K,) correlation MOYENNE de
    chaque ancre avec l'ensemble E. C'est exactement `corr()` de prescreen_offensif applique a
    un ensemble de frames (moyenne des produits scalaires)."""
    assert z.dim() == 2 and E.dim() == 2 and z.shape[1] == E.shape[1], \
        f"corr_set: z {tuple(z.shape)} vs E {tuple(E.shape)}"
    return (z @ E.t()).mean(dim=1)


def t0_margin(z, E_a, E_b):
    """LA quantite des gates : corr(ancre, A) - corr(ancre, B). Mesuree aujourd'hui a
    -0.256 (24 cellules S2 : +0.042 vs +0.298)."""
    return corr_set(z, E_a) - corr_set(z, E_b)


# ----------------------------- la perte -----------------------------------------------------------
def t0_loss(z, E_a, E_b, E_o, tau=0.10, m0=0.30, kind="both", w_margin=1.0, w_b=1.0):
    """Terme contrastif T0 sur l'ancre decodee.
        z   (K,D) ancres decodees (unitaires, DIFFERENTIABLES)
        E_a (Na,D) frames du segment A de la MEME video       -> POSITIFS
        E_b (Nb,D) frames du segment B de la MEME video       -> NEGATIFS DURS (l'interlude)
        E_o (No,D) frames des AUTRES videos (jumeau inclus)   -> NEGATIFS
    kind :
      'infonce' InfoNCE multi-positifs : -log( sum_p e^{s_p/tau} / sum_{p+n} e^{s/tau} ).
                Standard, borne, gradient stable ; pousse A vers le haut ET B/autres vers le bas.
      'margin'  hinge DIRECTEMENT sur la quantite des gates :
                w_b * relu(m0 - (corr_A - corr_B)) + relu(m0 - (corr_A - corr_O)).
                Sature une fois le gate franchi (pas de pression au-dela : anti-Goodhart).
      'both'    infonce + w_margin * margin  (DEFAUT : le terme standard porte l'optimisation,
                le hinge garantit qu'on optimise bien la quantite mesuree par le gate).
    Retourne (loss, logs) ; logs contient la marge, les trois correlations et l'ecart-type
    spatial de l'ancre AVANT normalisation (tripwire anti-effondrement du contraste, G1)."""
    assert kind in ("infonce", "margin", "both"), kind
    c_a = corr_set(z, E_a)
    c_b = corr_set(z, E_b)
    c_o = corr_set(z, E_o)
    loss = z.new_zeros(())
    if kind in ("infonce", "both"):
        # w_b sur-pondere l'interlude B dans le denominateur. La facon exacte de donner un poids
        # w a un ensemble dans un softmax est d'ajouter log(w) a ses logits (pas de multiplier la
        # similarite, ce qui changerait la temperature de ce seul groupe).
        l_pos = (z @ E_a.t()) / tau                                   # (K, Na)
        l_neg = torch.cat([(z @ E_b.t()) / tau + math.log(max(w_b, 1e-6)),
                           (z @ E_o.t()) / tau], dim=1)               # (K, Nb+No)
        num = torch.logsumexp(l_pos, dim=1)
        den = torch.logsumexp(torch.cat([l_pos, l_neg], dim=1), dim=1)
        loss = loss + (den - num).mean()
    if kind in ("margin", "both"):
        hinge = (Fn.relu(m0 - (c_a - c_b)) * w_b + Fn.relu(m0 - (c_a - c_o))).mean()
        loss = loss + (w_margin if kind == "both" else 1.0) * hinge
    with torch.no_grad():
        logs = {"t0_margin": float((c_a - c_b).mean()), "t0_corr_a": float(c_a.mean()),
                "t0_corr_b": float(c_b.mean()), "t0_corr_o": float(c_o.mean()),
                "t0_margin_o": float((c_a - c_o).mean()),
                "t0_n_pos": float(E_a.shape[0]), "t0_n_neg": float(E_b.shape[0] + E_o.shape[0]),
                "t0_frac_pos": float((c_a > c_b).float().mean())}
    return loss, logs


def anchor_spatial_std(x, space="latent"):
    """Ecart-type spatial du LATENT de l'ancre (diagnostic secondaire).
    AUDIT 2026-08-04 : cette fonction IGNORE `space` -- elle mesure toujours le latent. Le
    tripwire anti-Goodhart de G1 doit vivre dans l'espace de la METRIQUE (un latent bruite peut
    tres bien decoder en une image plate) ; il passe donc par `t0_embed(..., return_scale=True)`.
    Gardee pour le diagnostic latent uniquement."""
    if x.dim() == 5:
        x = x[:, 0]
    z = x.float()
    if z.shape[1] > 3:              # latent : moyenne des canaux
        z = z.mean(dim=1, keepdim=True)
    return float(z.flatten(1).std(dim=1).mean())
