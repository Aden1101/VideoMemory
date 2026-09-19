#!/usr/bin/env python
"""ETAPE 1, LE SONDEUR (l'instrument du verdict).

"Tache : une petite tete (MLP) apprend a projeter cette signature vers un espace d'embedding
 ou elle est plus proche de l'embedding visuel (DINO) du crop du sujet de sa video que de
 celui de la video jumelle du meme prompt, et des autres videos."
"Le piege identifie et son controle : deux videos de meme prompt different AUSSI par le fond
 [...] on mesure donc la precision contre des references 'crop du sujet' ET contre des
 references 'fond seul' : l'ecart entre les deux = ce qui est specifiquement extractible du
 sujet. Autres controles : etats permutes entre videos, split strict par video."

SEUILS SCELLES (memory2_probes_registry.md, derniere section), ils sont ECRITS ICI mais
JAMAIS appliques automatiquement a un run non audite :
  1. VALIDITE (borne superieure) : --source h doit atteindre >= 90 % de precision par paire
     sur le test. Sinon : protocole invalide, aucune conclusion sur la memoire.
  2. CONTROLE EFFONDREMENT : etats permutes entre videos -> precision <= 55 %.
  3. SUCCES de l'etat : precision-sujet test >= 65 % ET Wilcoxon p < 0.05 sur les marges ET
     (precision-sujet - precision-fond) >= +10 points.
  4. zone GRISE 55-75 % sans les 3 conditions -> +30 paires AVANT toute conclusion.
     ECHEC : < 55 % ou ecart sujet-fond nul -> ARRET du chantier memoire.

GARDE-fou D'execution : `--stage final` (la seule etape qui TOUCHE LE test) refuse de tourner
sans `--audited`. La recherche d'hyperparametres (`--stage hpsearch`) ne voit QUE le val.

Usage :
  python reference/runs/probe1/train_probe.py --stage hpsearch --source h     --target subject
  python reference/runs/probe1/train_probe.py --stage final --audited --source state ...
"""
import argparse
import itertools
import json
import math
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import probe1_common as C  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as Fn  # noqa: E402

from build_signatures import compress_signature, sig_path  # noqa: E402

TARGETS = ("subject", "background")
FEATURES = ("pool", "raw")      # pool = compression moyenne+max par sonde ; raw = signature brute


# ---------------------------------------------------------------------------------------------
# jeu de donnees
# ---------------------------------------------------------------------------------------------
def load_refs(video_id, target, seg="A"):
    """Embeddings DINO de reference d'une video : les crops-SUJET (l'individu) ou les
    crops-fond (le complement du masque). Seules les frames valides du masque comptent."""
    p = os.path.join(C.REFS_DIR, C.seg_key(video_id, seg) + ".pt")
    if not os.path.exists(p):
        return None
    r = torch.load(p, map_location="cpu", weights_only=False)
    e = r["subj" if target == "subject" else "bg"].float()
    v = r["valid"].bool() if "valid" in r else torch.ones(e.shape[0], dtype=torch.bool)
    e = e[v] if v.any() else e
    return Fn.normalize(e, dim=-1)


def load_features(source, video_id, age_name, features):
    """LE point d'entree unique du sondeur. `features` :
      pool = compression moyenne+max par ligne de reponse (correctif n 1 du verdict
             PROTOCOLE INVALIDE : 3 x 256 x 2 = 1536 nombres au lieu de 1 179 648) ;
      raw  = la signature brute, laissee a l'ACP (variante ACP de la mission).
    Aucune des deux ne regarde le split : la compression n'a ni parametre ni donnees."""
    d = torch.load(sig_path(source, video_id, age_name), map_location="cpu", weights_only=False)
    s = d["sig"]
    # `raw` rend le tenseur fp16 TEL QUEL : c'est exactement ce que faisait le chemin audite du
    # 04/08 (le blanchiment le repasse en fp32), donc la variante ACP reste bit a bit celle-la.
    return compress_signature(s).reshape(-1) if features == "pool" else s.reshape(-1)


def load_dataset(source, target, n_ages=C.N_AGES, verbose=True, features="pool", corpus="v1"):
    """Le jeu complet : une ligne par (video de paire, age). Tout l'aval est agnostique a la
    source ET a la representation : il ne voit qu'une matrice (n_lignes, dim)."""
    vids, pairs = C.pair_dataset(verbose=False, corpus=corpus)
    byid = C.vids_by_id(vids)
    videos, X, s_vid, s_age, dropped = [], [], [], [], []
    for p in pairs:
        rows, ok = {}, True
        for role in ("main", "twin"):
            vid = p[role]
            refs = load_refs(vid, target)
            if refs is None or refs.shape[0] == 0:
                ok = False
                break
            ages = C.state_ages(byid[vid], n_ages)
            sigs = [(a["name"], sig_path(source, vid, a["name"])) for a in ages]
            if not all(os.path.exists(sp) for _, sp in sigs):
                ok = False
                break
            rows[role] = {"video_id": vid, "refs": refs, "sigs": sigs,
                          "corpus_idx": p["corpus_idx"], "split": p["split"],
                          "pair_id": p["pair_id"], "board_no": p.get("board_no", 0)}
        if not ok:
            dropped.append(p["corpus_idx"])
            continue
        base = len(videos)                       # base = main, base + 1 = twin
        rows["main"]["twin_index"] = base + 1
        rows["twin"]["twin_index"] = base
        for off, role in enumerate(("main", "twin")):
            r = rows[role]
            videos.append(r)
            for name, _sp in r["sigs"]:
                X.append(load_features(source, r["video_id"], name, features))
                s_vid.append(base + off)
                s_age.append(name)
    assert videos, "jeu vide : lance build_masks / build_states / build_signatures d'abord"
    ds = {"videos": videos, "X": torch.stack(X, 0), "s_vid": torch.tensor(s_vid),
          "s_age": s_age, "source": source, "target": target, "dropped_pairs": dropped,
          "features": features, "corpus": corpus,
          "pairs": {int(p["corpus_idx"]): p for p in pairs}}
    if verbose:
        n = {s: sum(1 for v in videos if v["split"] == s) for s in ("train", "val", "test")}
        print(f"[probe] corpus={corpus} source={source} cible={target} features={features} : "
              f"{len(videos)} videos (train/val/test = {n['train']}/{n['val']}/{n['test']}), "
              f"{ds['X'].shape[0]} signatures de dim {ds['X'].shape[1]}, "
              f"{len(dropped)} paires ecartees {dropped}", flush=True)
    return ds


# ---------------------------------------------------------------------------------------------
# PRE-ENTRAINEMENT : les 105 videos `switch` du corpus (correctif n 1 de la mission)
# ---------------------------------------------------------------------------------------------
def load_pretrain(source, target, n_ages=C.N_AGES, features="pool", verbose=True):
    """Le jeu de PRE-ENTRAINEMENT contrastif de la tete : les videos `switch` du corpus v1.

    Pourquoi elles : leurs etats sont deja rejouables (memes h, meme ecrivain), elles
    n'appartiennent a AUCUNE paire (asserte ici, comme pour le k-means des sondes), et leur
    tache est la BONNE tache, retrouver le sujet du segment A, avec en prime la condition
    que les paires n'ont pas : l'interference du second sujet (les ages `end_seq` sont
    post-switch). 105 videos x 4 ages = 420 lignes, contre 144 : c'est la seule facon
    d'apprendre quoi que ce soit dans un espace qui, meme comprime, fait 1536 dimensions.

    La reference est le crop-sujet du segment A (le meme DINO, la meme construction).
    Les negatifs y sont des videos de prompts differents : c'est un pre-entrainement, pas
    l'epreuve ; l'epreuve (separer deux jumeaux du MEME prompt) reste celle des paires."""
    vids, pairs = C.pair_dataset(verbose=False, corpus="v1")
    pair_ids = {p["main"] for p in pairs} | {p["twin"] for p in pairs}
    videos, X, s_vid, s_age = [], [], [], []
    for v in vids:
        if v["mode"] != "switch":
            continue
        assert v["video_id"] not in pair_ids, f"FUITE : {v['video_id']} est dans une paire"
        refs = load_refs(v["video_id"], target, seg="A")
        if refs is None or refs.shape[0] == 0:
            continue
        ages = C.state_ages(v, n_ages)
        if not all(os.path.exists(sig_path(source, v["video_id"], a["name"])) for a in ages):
            continue
        vi = len(videos)
        videos.append({"video_id": v["video_id"], "refs": refs, "twin_index": -1,
                       "split": "train", "pair_id": -1, "corpus_idx": int(v["corpus_idx"]),
                       "board_no": 0})
        for a in ages:
            X.append(load_features(source, v["video_id"], a["name"], features))
            s_vid.append(vi)
            s_age.append(a["name"])
    assert videos, ("jeu de pre-entrainement vide : lance build_signatures.py --switch_only "
                    "(et build_masks --all pour les references)")
    ds = {"videos": videos, "X": torch.stack(X, 0), "s_vid": torch.tensor(s_vid),
          "s_age": s_age, "source": source, "target": target, "dropped_pairs": [],
          "features": features, "corpus": "v1-switch", "pairs": {}}
    if verbose:
        print(f"[probe] PRE-ENTRAINEMENT : {len(videos)} videos switch, {ds['X'].shape[0]} "
              f"signatures de dim {ds['X'].shape[1]} (0 video de paire)", flush=True)
    return ds


ALL_ROWS = "__all__"        # pseudo-split du mode inter-corpus : toutes les lignes du jeu


def split_rows(ds, split):
    if split == ALL_ROWS:
        # le corpus de test d'un run inter-corpus n'a pas de split interne : il est le test,
        # en entier. (Aucune video ne porte jamais ce nom de split : pas de collision.)
        return torch.arange(ds["X"].shape[0], dtype=torch.long)
    keep = {i for i, v in enumerate(ds["videos"]) if v["split"] == split}
    return torch.tensor([i for i, vi in enumerate(ds["s_vid"].tolist()) if vi in keep],
                        dtype=torch.long)


# ---------------------------------------------------------------------------------------------
# pretraitement : blanchiment par sonde + ACP, AJUSTES sur LE train SEUL
# ---------------------------------------------------------------------------------------------
class Preproc:
    """1) blanchiment par sonde = standardisation coordonnee par coordonnee (moyenne / ecart-type
          du train) : chaque sonde contribue alors a echelle egale, quelle que soit la norme
          brute de sa reponse ;
       2) ACP ajustee sur le train SEUL, calculee par le tour de Gram (n_train << dimension) :
          on ne materialise JAMAIS la base de 1.18 M x k, la projection d'une ligne z vaut
          (z . Z_train^T) U lambda^{-1/2}. Exact, et le test ne peut pas fuir dans la base.

    L'objet est construit UNE fois par jeu (l'eigendecomposition ne depend pas de k) ; `k` est
    tranche a la volee par `transform`, ce qui rend la recherche d'hyperparametres sur `sig_dim`
    gratuite et garantit que toutes les configs partagent EXACTEMENT la meme base.
    """

    def __init__(self, Xtr, eps=1e-5, device="cpu", extra=None):
        """`extra` (facultatif) = des lignes SUPPLEMENTAIRES, sans etiquette de test, qui
        entrent dans l'estimation de la moyenne / de l'ecart-type / de la base ACP : les
        signatures des videos `switch` du pre-entrainement. Elles n'appartiennent a aucune
        paire, donc ni au val ni au test, estimer un blanchiment de 1536 (ou 1,18 M)
        dimensions sur 144 lignes etait l'autre visage du defaut n 1."""
        Z = Xtr.to(device=device, dtype=torch.float32)
        if extra is not None and extra.shape[0] > 0:
            Z = torch.cat([Z, extra.to(device=device, dtype=torch.float32)], 0)
        assert Z.dim() == 2 and Z.shape[0] >= 3, f"train trop petit : {tuple(Z.shape)}"
        self.mu = Z.mean(0)
        self.sd = Z.std(0).clamp_min(eps)
        self.Ztr = (Z - self.mu) / self.sd                # (n_fit, Dfull), deja centree
        G = self.Ztr @ self.Ztr.transpose(0, 1)           # (n_fit, n_fit)
        lam, U = torch.linalg.eigh(G.double())
        order = torch.argsort(lam, descending=True)
        lam, U = lam[order], U[:, order]
        self.rank = int(max(2, min((lam > 1e-6 * lam[0].clamp_min(1e-12)).sum().item(),
                                   Z.shape[0] - 1)))
        self.W_full = (U[:, :self.rank] / lam[:self.rank].clamp_min(1e-9).sqrt()).float()
        self.n_train = int(Xtr.shape[0])
        self.n_fit = int(Z.shape[0])
        self.d_in = int(Z.shape[1])

    def dim(self, k):
        """k <= 0 = pas d'ACP du tout : les coordonnees standardisees telles quelles. C'est le
        regime naturel de la signature COMPRIMEE (1536 dimensions), ou l'ACP n'est plus une
        necessite mais un hyperparametre de plus."""
        return self.d_in if int(k) <= 0 else int(min(k, self.rank))

    def transform(self, X, k, device=None, batch=64):
        dev = device or self.mu.device
        no_pca = int(k) <= 0
        W = None if no_pca else self.W_full[:, :self.dim(k)]
        out = []
        for i in range(0, X.shape[0], batch):
            z = X[i:i + batch].to(device=dev, dtype=torch.float32)
            z = (z - self.mu) / self.sd
            out.append(z if no_pca else (z @ self.Ztr.transpose(0, 1)) @ W)
        return torch.cat(out, 0)


# ---------------------------------------------------------------------------------------------
# le MLP
# ---------------------------------------------------------------------------------------------
class ProbeMLP(nn.Module):
    """2-3 couches, signature (post-ACP) -> espace DINO. Sortie L2-normalisee : la tache est
    une comparaison d'angles, jamais de normes."""

    def __init__(self, d_in, hidden=512, depth=2, out=C.DINO_DIM, dropout=0.1):
        super().__init__()
        assert depth in (2, 3), "2 ou 3 couches (contrat du plan)"
        layers, d = [nn.LayerNorm(d_in)], d_in
        for _ in range(depth - 1):
            layers += [nn.Linear(d, hidden), nn.GELU(), nn.Dropout(dropout)]
            d = hidden
        layers += [nn.Linear(d, out)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return Fn.normalize(self.net(x), dim=-1)


# ---------------------------------------------------------------------------------------------
# entrainement contrastif
# ---------------------------------------------------------------------------------------------
def get_preproc(ds, device="cpu"):
    """L'ACP/blanchiment du jeu, ajuste UNE SEULE fois sur le train (+ le jeu de
    pre-entrainement s'il y en a un, qui ne contient aucune video de paire) et memorise."""
    if ds.get("_pre") is None:
        pre_ds = ds.get("_pretrain")
        if pre_ds is not None:
            assert pre_ds["X"].shape[1] == ds["X"].shape[1], \
                (f"pre-entrainement en dim {pre_ds['X'].shape[1]} != jeu en dim "
                 f"{ds['X'].shape[1]} : meme source ET memes features obligatoires")
            assert pre_ds["source"] == ds["source"] \
                and pre_ds.get("features") == ds.get("features"), \
                "le pre-entrainement doit venir de la MEME source et des MEMES features"
        extra = pre_ds["X"] if pre_ds is not None else None
        ds["_pre"] = Preproc(ds["X"][split_rows(ds, "train")], device=device, extra=extra)
    return ds["_pre"]


def transformed(ds, split, k, device="cpu"):
    """Les signatures d'un split, projetees, avec cache : la projection ne depend que de
    (split, k), pas des hyperparametres du MLP, la recherche n'en repaie donc jamais le prix,
    et toutes les configs partagent EXACTEMENT la meme entree."""
    pre = get_preproc(ds, device=device)
    cache = ds.setdefault("_Z", {})
    key = (split, pre.dim(k))
    if key not in cache:
        cache[key] = pre.transform(ds["X"][split_rows(ds, split)], k, device=device)
    return cache[key]


def _ref_bank(videos, vids_order, device):
    """Banque de references empaquetee (V, F_max, 768) + compte de frames valides : a chaque pas
    on tire UNE image de reference par video (le sujet bouge d'une frame a l'autre, c'est
    l'augmentation naturelle de la tache), en une seule operation vectorisee."""
    refs = [videos[v]["refs"].to(device) for v in vids_order]
    V, Fmax = len(refs), max(r.shape[0] for r in refs)
    R = torch.zeros(V, Fmax, C.DINO_DIM, device=device)
    cnt = torch.tensor([r.shape[0] for r in refs], dtype=torch.float32)
    for i, r in enumerate(refs):
        R[i, :r.shape[0]] = r
    return R, cnt, Fmax


def _infonce(model, Z, y, R, cnt, Fmax, cfg, seed, device, steps=None, lr=None, verbose=False):
    """La boucle contrastive, partagee mot pour mot par le pre-entrainement et l'entrainement :
    une seule definition de la perte, donc aucune divergence possible entre les deux."""
    g = torch.Generator().manual_seed(seed)
    steps = int(steps if steps is not None else cfg["steps"])
    opt = torch.optim.AdamW(model.parameters(), lr=(lr if lr is not None else cfg["lr"]),
                            weight_decay=cfg["wd"])
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, steps))
    V = R.shape[0]
    ar = torch.arange(V, device=device)
    n = Z.shape[0]
    bs = min(cfg["batch"], n)
    model.train()
    for step in range(steps):
        sel = torch.randperm(n, generator=g)[:bs].to(device)
        j = (torch.rand(V, generator=g) * cnt).long().clamp_(max=int(Fmax - 1))
        E = R[ar, j.to(device)]
        logits = (model(Z[sel]) @ E.transpose(0, 1)) / cfg["temp"]
        loss = Fn.cross_entropy(logits, y[sel])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sch.step()
        if verbose and (step % max(1, steps // 5) == 0):
            print(f"    step {step:4d} loss {loss.item():.4f}", flush=True)
    model.eval()
    return model


def pretrain_head(ds, cfg, seed=0, device="cpu", verbose=False):
    """Pre-entraine la tete sur les videos `switch` (jeu `ds['_pretrain']`), dans l'espace
    d'entree du fold courant (meme Preproc : le pre-entrainement ne peut donc rien apprendre
    d'un blanchiment qui aurait vu le test, il n'y en a pas)."""
    pds = ds["_pretrain"]
    pre = get_preproc(ds, device=device)
    Z = pre.transform(pds["X"], cfg["sig_dim"], device=device)
    vids_order = sorted({int(v) for v in pds["s_vid"].tolist()})
    pos = {v: k for k, v in enumerate(vids_order)}
    y = torch.tensor([pos[int(v)] for v in pds["s_vid"].tolist()], device=device)
    R, cnt, Fmax = _ref_bank(pds["videos"], vids_order, device)
    model = ProbeMLP(pre.dim(cfg["sig_dim"]), cfg["hidden"], cfg["depth"], C.DINO_DIM,
                     cfg["dropout"]).to(device)
    torch.manual_seed(seed)
    return _infonce(model, Z, y, R, cnt, Fmax, cfg, seed + 90000, device,
                    steps=cfg.get("pre_steps", 800), verbose=verbose)


def train_one(ds, cfg, seed=0, device="cpu", verbose=False):
    """Entraine le sondeur sur le train. Les negatifs du softmax = toutes les videos du train,
    ce qui inclut par construction le JUMEAU meme-prompt de chaque positif ; on l'ASSERTE.
    Si `ds['_pretrain']` existe, la tete part des poids PRE-entraines sur les videos switch
    (aucune n'est dans une paire) au lieu d'une initialisation aleatoire."""
    torch.manual_seed(seed)
    tr_rows = split_rows(ds, "train")
    tr_vids = sorted({int(ds["s_vid"][i]) for i in tr_rows.tolist()})
    vid_pos = {v: k for k, v in enumerate(tr_vids)}
    for v in tr_vids:
        tw = ds["videos"][v]["twin_index"]
        assert tw < 0 or tw in vid_pos, \
            f"video {v}: son jumeau n'est pas dans le softmax du train (split casse)"

    pre = get_preproc(ds, device=device)
    Ztr = transformed(ds, "train", cfg["sig_dim"], device=device)
    y = torch.tensor([vid_pos[int(ds["s_vid"][i])] for i in tr_rows.tolist()], device=device)
    R, cnt, Fmax = _ref_bank(ds["videos"], tr_vids, device)

    if ds.get("_pretrain") is not None:
        model = pretrain_head(ds, cfg, seed=seed, device=device, verbose=verbose)
    else:
        model = ProbeMLP(pre.dim(cfg["sig_dim"]), cfg["hidden"], cfg["depth"], C.DINO_DIM,
                         cfg["dropout"]).to(device)
    model = _infonce(model, Ztr, y, R, cnt, Fmax, cfg, seed, device, verbose=verbose)
    model._sig_dim = cfg["sig_dim"]
    return model, pre


# ---------------------------------------------------------------------------------------------
# evaluation : precision par paire + marges continues + Wilcoxon + controle etats permutes
# ---------------------------------------------------------------------------------------------
@torch.no_grad()
def evaluate(ds, model, pre, split, device="cpu", permute=False, seed=1234, n_perm=25,
             pre_ext=None):
    """Chaque signature (video v, age a) doit etre plus proche de la reference de v que de celle
    de son JUMEAU. La marge continue est moyennee sur (ages x images de reference).
    `permute` = LE CONTROLE D'EFFONDREMENT : les signatures sont permutees entre les videos du
    split (derangement), la precision doit retomber au hasard.

    Audit 2026-08-04 : le controle permute est desormais moyenne sur `n_perm` derangements et
    non plus estime sur un SEUL tirage. Avec 7 paires de test, un tirage unique est lui-meme une
    variable a ~14 degres de liberte : la regle scellee n 2 ("precision permutee <= 55 %")
    echouait ou passait au hasard du tirage. La quantite mesuree (la precision sous permutation)
    est inchangee ; seule sa variance d'estimation baisse. Les marges detaillees et le Wilcoxon
    restent ceux du premier derangement (le tirage historique, graine 1234)."""
    rows = split_rows(ds, split)
    assert rows.numel() > 0, f"split {split} vide"
    if pre_ext is None:
        Z = transformed(ds, split, getattr(model, "_sig_dim", pre.rank), device=device)
    else:
        # mode inter-corpus : le blanchiment et la base ACP viennent du corpus
        # D'ENTRAINEMENT (`pre_ext`) et n'ont JAMAIS vu une ligne de ce jeu-ci. C'est la
        # seule difference ; tout le calcul de score ci-dessous est le meme code.
        assert pre_ext.d_in == ds["X"].shape[1], \
            (f"dim {ds['X'].shape[1]} du corpus de test != {pre_ext.d_in} du corpus "
             f"d'entrainement : meme source ET memes features obligatoires")
        Z = pre_ext.transform(ds["X"][rows], getattr(model, "_sig_dim", pre_ext.rank),
                              device=device)
    P = model(Z)
    vids0 = [int(ds["s_vid"][i]) for i in rows.tolist()]

    def derangement(k):
        uniq = sorted(set(vids0))
        rng = np.random.RandomState(seed + k)
        perm = None
        for _ in range(200):
            cand = rng.permutation(len(uniq))
            if all(cand[i] != i for i in range(len(uniq))):
                perm = cand
                break
        assert perm is not None, "aucun derangement trouve"
        remap = {u: uniq[int(perm[i])] for i, u in enumerate(uniq)}
        return [remap[v] for v in vids0]

    def score(vids):
        pr, pv = [], {}
        for r in range(P.shape[0]):
            v = vids[r]
            tw = ds["videos"][v]["twin_index"]
            own = ds["videos"][v]["refs"].to(device)
            oth = ds["videos"][tw]["refs"].to(device)
            s_own = (P[r][None] @ own.transpose(0, 1)).mean().item()
            s_oth = (P[r][None] @ oth.transpose(0, 1)).mean().item()
            pr.append({"video": v, "video_id": ds["videos"][v]["video_id"],
                       "corpus_idx": int(ds["videos"][v]["corpus_idx"]),
                       "pair_id": int(ds["videos"][v]["pair_id"]),
                       "age": ds["s_age"][int(rows[r])],
                       "margin": s_own - s_oth, "correct": float(s_own > s_oth)})
            pv.setdefault(v, []).append(s_own - s_oth)
        return pr, pv

    variants = [derangement(k) for k in range(max(1, int(n_perm)))] if permute else [vids0]
    per_row, per_video, accs, pair_accs = None, None, [], []
    for k, vl in enumerate(variants):
        pr, pv = score(vl)
        if k == 0:
            per_row, per_video = pr, pv
        accs.append(float(np.mean([d["correct"] for d in pr])))
        # audit 2 (2026-08-05) : la precision par PAIRE de CHAQUE derangement, pas seulement du
        # premier. Le reglage final n 1 a fige la lecture des seuils sur "par paire" ; sans
        # cette ligne, la regle 2 (permute <= 55 %) retombait sur un SEUL tirage, exactement
        # le defaut que l'audit du 04/08 avait corrige pour la lecture par ligne. A 30 paires,
        # un tirage unique a ~29 % de chance de depasser 0,55 alors que la vraie valeur est 0,50.
        pm = {}
        for d in pr:
            pm.setdefault(int(d["corpus_idx"]), []).append(float(d["margin"]))
        pair_accs.append(float(np.mean([float(np.mean(m) > 0) for m in pm.values()])))

    acc = float(np.mean(accs))
    vmarg = {v: float(np.mean(m)) for v, m in per_video.items()}
    pmarg = {}
    for v, m in vmarg.items():
        pid = ds["videos"][v]["pair_id"]
        pmarg.setdefault(pid, []).append(m)
    pair_marg = {p: float(np.mean(m)) for p, m in pmarg.items()}

    from scipy.stats import wilcoxon
    out = {"acc": acc, "n_rows": len(per_row), "n_videos": len(vmarg),
           "n_pairs": len(pair_marg), "n_perm": len(variants) if permute else 0,
           "perm_accs": accs if permute else [],
           "acc_by_pair_allperm": float(np.mean(pair_accs)),
           "pair_corpus": {int(ds["videos"][v]["pair_id"]): int(ds["videos"][v]["corpus_idx"])
                           for v in vmarg},
           "margin_mean": float(np.mean(list(vmarg.values()))),
           "video_margins": vmarg, "pair_margins": pair_marg,
           "acc_by_video": float(np.mean([m > 0 for m in vmarg.values()])),
           "acc_by_pair": float(np.mean([m > 0 for m in pair_marg.values()]))}
    for name, arr in (("video", list(vmarg.values())), ("pair", list(pair_marg.values()))):
        a = np.asarray(arr, dtype=float)
        try:
            out[f"wilcoxon_{name}_p"] = float(wilcoxon(a, alternative="greater").pvalue) \
                if np.any(a != 0) else 1.0
        except Exception:
            out[f"wilcoxon_{name}_p"] = float("nan")
    out["per_row"] = per_row
    return out


# ---------------------------------------------------------------------------------------------
# recherche d'hyperparametres sur LE val UNIQUEMENT
# ---------------------------------------------------------------------------------------------
GRID = {"sig_dim": [64, 128], "hidden": [256, 512], "depth": [2, 3],
        "lr": [1e-3, 3e-4], "wd": [1e-2, 1e-1], "temp": [0.05, 0.1], "dropout": [0.1, 0.3]}
# Grille de la validation CROISEE : elle est parcourue 6 fois (une par pli), donc elle est
# reduite, et `sig_dim: 0` (pas d'ACP, les 1536 coordonnees comprimees telles quelles) y
# entre comme un choix a part entiere.
GRID_CV = {"sig_dim": [0, 64], "hidden": [256], "depth": [2, 3],
           "lr": [1e-3, 3e-4], "wd": [1e-2], "temp": [0.05, 0.1], "dropout": [0.1, 0.3]}
FIXED = {"steps": 600, "batch": 64, "pre_steps": 800}


def hpsearch(ds, device="cpu", seeds=(0, 1, 2), grid=None, verbose=True):
    grid = grid or GRID
    keys = sorted(grid)
    best, rows = None, []
    for combo in itertools.product(*[grid[k] for k in keys]):
        cfg = dict(zip(keys, combo))
        cfg.update(FIXED)
        accs, margs = [], []
        for s in seeds:
            m, pre = train_one(ds, cfg, seed=s, device=device)
            r = evaluate(ds, m, pre, "val", device=device)
            accs.append(r["acc"])
            margs.append(r["margin_mean"])
        row = {"cfg": cfg, "val_acc": float(np.mean(accs)), "val_acc_sd": float(np.std(accs)),
               "val_margin": float(np.mean(margs))}
        rows.append(row)
        if best is None or (row["val_acc"], row["val_margin"]) > (best["val_acc"],
                                                                 best["val_margin"]):
            best = row
            if verbose:
                print(f"[hp] nouveau meilleur val_acc={row['val_acc']:.3f} "
                      f"marge={row['val_margin']:+.4f} {cfg}", flush=True)
    rows.sort(key=lambda r: (-r["val_acc"], -r["val_margin"]))
    return best, rows


# ---------------------------------------------------------------------------------------------
# validation CROISEE par paires (procedure PRIMAIRE scellee le 04/08 au soir, reglage n 2)
#   "folds etanches, hyperparametres a l'interieur des folds, 30 paires au denominateur"
# ---------------------------------------------------------------------------------------------
def pair_folds(cidxs, n_folds=6, seed=C.SPLIT_SEED):
    """Les plis, deterministes et independants de l'ordre d'arrivee : meme cle de hash que le
    split fixe (sha1(graine, corpus_idx)), decoupee en `n_folds` blocs egaux."""
    import hashlib
    order = sorted(cidxs, key=lambda c: int(
        hashlib.sha1(f"{seed}|{c}".encode()).hexdigest()[:12], 16))
    return [sorted(order[i::n_folds]) for i in range(n_folds)]


def set_fold(ds, test_cidx, val_cidx):
    """Reecrit le split du jeu (par PAIRE) et invalide tous les caches derives : le
    blanchiment et la base ACP sont donc re-ajustes sur le train de ce pli, c'est la
    definition meme d'un pli etanche."""
    test_cidx, val_cidx = set(map(int, test_cidx)), set(map(int, val_cidx))
    assert not (test_cidx & val_cidx), "pli casse : val et test se recouvrent"
    for v in ds["videos"]:
        c = int(v["corpus_idx"])
        v["split"] = "test" if c in test_cidx else ("val" if c in val_cidx else "train")
    ds["_pre"] = None
    ds["_Z"] = {}
    return ds


def _agg_rows(rows_per_fold):
    """Concatene les `per_row` des plis et rend (acc par ligne, acc par PAIRE, marges par
    paire indexees par corpus_idx). Le denominateur est bien les 30 paires."""
    per_row = [r for fold in rows_per_fold for r in fold]
    by_pair = {}
    for r in per_row:
        by_pair.setdefault(int(r["corpus_idx"]), []).append(float(r["margin"]))
    pair_marg = {c: float(np.mean(m)) for c, m in by_pair.items()}
    return per_row, pair_marg


def subset_report(per_row, pair_marg, keep):
    """Precision (ligne et paire) + Wilcoxon restreints a un sous-ensemble de corpus_idx."""
    from scipy.stats import wilcoxon
    rows = [r for r in per_row if int(r["corpus_idx"]) in keep]
    marg = [m for c, m in sorted(pair_marg.items()) if c in keep]
    if not rows or not marg:
        return None
    a = np.asarray(marg, dtype=float)
    try:
        p = float(wilcoxon(a, alternative="greater").pvalue) if np.any(a != 0) else 1.0
    except Exception:
        p = float("nan")
    k = int(sum(1 for m in marg if m > 0))
    # p binomial unilateral P(X >= k | p = 0.5) : c'est lui que les seuils du registre citent
    # pour les corpus a paires (>= 12/15 -> 0.0176). Additif : aucune cle existante ne bouge.
    try:
        from scipy.stats import binomtest
        pb = float(binomtest(k, len(marg), 0.5, alternative="greater").pvalue)
    except Exception:
        pb = float("nan")
    return {"n_pairs": len(marg), "n_rows": len(rows),
            "acc_row": float(np.mean([r["correct"] for r in rows])),
            "acc_pair": float(np.mean([m > 0 for m in marg])),
            "n_pairs_correct": k,
            "margin_mean": float(a.mean()), "wilcoxon_pair_p": p, "binom_pair_p": pb}


# ---------------------------------------------------------------------------------------------
# mode inter-corpus (chaine C2, 07/08) : entraine sur un corpus, teste sur un AUTRE.
#   "entrainement sur les 30 paires v2 (toutes, y compris faibles/invalides : c'est de
#    l'ENTRAINEMENT, pas du test), selection d'hyperparametres par validation INTERNE a v2
#    (jamais sur v3), puis UNE SEULE evaluation sur v3."
# Zero fuite par construction : le corpus de test n'entre ni dans le blanchiment, ni dans la
# base ACP, ni dans la recherche d'hyperparametres, ni dans le softmax contrastif. Il n'est
# touche qu'a la derniere ligne, une fois par graine.
# ---------------------------------------------------------------------------------------------
def cross_run(ds_tr, ds_te, device="cpu", seeds=(0, 1, 2), grid=None, verbose=True):
    grid = grid or GRID_CV
    assert ds_tr["X"].shape[1] == ds_te["X"].shape[1], \
        f"dims {ds_tr['X'].shape[1]} != {ds_te['X'].shape[1]} : meme source ET memes features"
    assert ds_tr["source"] == ds_te["source"] and ds_tr["features"] == ds_te["features"]
    # aucune paire commune : les deux corpus vivent dans des plages de corpus_idx disjointes
    c_tr = {int(v["corpus_idx"]) for v in ds_tr["videos"]}
    c_te = {int(v["corpus_idx"]) for v in ds_te["videos"]}
    assert not (c_tr & c_te), f"FUITE : {len(c_tr & c_te)} paires communes aux deux corpus"

    # 1) hyperparametres : validation INTERNE au corpus d'entrainement (son split fixe
    #    18/5/7 ; le sondeur n'a jamais vu le corpus de test a ce stade).
    best, _rows = hpsearch(ds_tr, device=device, seeds=seeds[:2], grid=grid, verbose=False)
    if verbose:
        print(f"[cross] hyperparametres choisis sur le val INTERNE du corpus "
              f"{ds_tr['corpus']} : val_acc={best['val_acc']:.3f} {best['cfg']}", flush=True)

    # 2) re-entrainement sur toutes les paires du corpus d'entrainement (le val a fini son
    #    office ; le corpus de test, lui, n'a toujours rien vu).
    set_fold(ds_tr, [], [])
    n_tr = len({int(v["corpus_idx"]) for v in ds_tr["videos"] if v["split"] == "train"})
    assert n_tr == len(c_tr), f"{n_tr}/{len(c_tr)} paires en train apres set_fold"

    subsets = C.sealed_subsets(list(ds_te["pairs"].values())) if ds_te.get("pairs") else \
        {"all": set(c_te)}
    out = {"mode": "cross", "train_corpus": ds_tr["corpus"], "test_corpus": ds_te["corpus"],
           "cfg": best["cfg"], "val_acc_internal": best["val_acc"], "seeds": list(seeds),
           "n_train_pairs": len(c_tr), "n_test_pairs": len(c_te),
           "subsets_corpus_idx": {k: sorted(v) for k, v in subsets.items()}, "per_seed": []}
    for s in seeds:
        model, pre = train_one(ds_tr, best["cfg"], seed=s, device=device)
        # 3) L'unique evaluation sur le corpus de test (toutes ses paires), avec le
        #    blanchiment/ACP du corpus d'ENTRAINEMENT (pre_ext), jamais re-ajustes ici.
        r = evaluate(ds_te, model, pre, ALL_ROWS, device=device, pre_ext=pre)
        rp = evaluate(ds_te, model, pre, ALL_ROWS, device=device, permute=True, pre_ext=pre)
        per_row, pair_marg = _agg_rows([r["per_row"]])
        perm_row, perm_marg = _agg_rows([rp["per_row"]])
        out["per_seed"].append({
            "seed": s, "per_row": per_row,
            "pair_margins": {str(c): m for c, m in pair_marg.items()},
            "permuted": subset_report(perm_row, perm_marg, subsets["all"]),
            "perm_acc_pair_25": rp["acc_by_pair_allperm"],
            "subsets": {name: subset_report(per_row, pair_marg, keep)
                        for name, keep in subsets.items()}})
        if verbose:
            pr = out["per_seed"][-1]["subsets"]
            main = pr.get("strong15") or pr["all"]
            print(f"[cross] graine {s} : {main['n_pairs_correct']}/{main['n_pairs']} paires "
                  f"(p_binom={main['binom_pair_p']:.4g}) | permute pair "
                  f"{rp['acc_by_pair_allperm']:.3f}", flush=True)

    out["summary"] = {}
    for name in subsets:
        vals = [r["subsets"][name] for r in out["per_seed"] if r["subsets"][name]]
        if not vals:
            continue
        out["summary"][name] = {
            "n_pairs": vals[0]["n_pairs"], "n_rows": vals[0]["n_rows"],
            "acc_row": float(np.mean([v["acc_row"] for v in vals])),
            "acc_pair": float(np.mean([v["acc_pair"] for v in vals])),
            "acc_pair_sd": float(np.std([v["acc_pair"] for v in vals])),
            "n_pairs_correct": float(np.mean([v["n_pairs_correct"] for v in vals])),
            "n_pairs_correct_per_seed": [v["n_pairs_correct"] for v in vals],
            "margin_mean": float(np.mean([v["margin_mean"] for v in vals])),
            # p CONSERVATEURS : le PIRE des graines, jamais la moyenne des p
            "wilcoxon_pair_p": float(np.max([v["wilcoxon_pair_p"] for v in vals])),
            "binom_pair_p": float(np.max([v["binom_pair_p"] for v in vals]))}
    pv = [r["permuted"] for r in out["per_seed"] if r["permuted"]]
    out["summary"]["permuted"] = {
        "acc_row": float(np.mean([v["acc_row"] for v in pv])),
        "acc_pair": float(np.mean([v["acc_pair"] for v in pv])),
        "acc_pair_25perm": float(np.mean([r["perm_acc_pair_25"] for r in out["per_seed"]]))}
    return out


def cv_run(ds, device="cpu", seeds=(0, 1, 2), n_folds=6, grid=None, verbose=True):
    """Un tour complet de validation croisee par paires. Dans CHAQUE pli :
        test  = les 5 paires du pli (jamais vues, ni par le blanchiment, ni par l'ACP, ni par
                la recherche d'hyperparametres, ni par le pre-entrainement) ;
        val   = les 5 paires du pli suivant -> c'est la que les hyperparametres sont choisis ;
        train = les 20 restantes, puis les 25 (train + val) pour le re-entrainement final avec
                les hyperparametres retenus (le val a fini son office ; le test n'a rien vu).
    Rend les 30 paires au denominateur, plus le controle d'etats permutes, pli par pli."""
    grid = grid or GRID_CV
    cidxs = sorted({int(v["corpus_idx"]) for v in ds["videos"]})
    folds = pair_folds(cidxs, n_folds=n_folds)
    assert sum(len(f) for f in folds) == len(cidxs)
    out = {"n_folds": n_folds, "folds": folds, "seeds": list(seeds), "per_fold": [],
           "per_seed": []}
    rows_s = {s: [] for s in seeds}
    rows_perm_s = {s: [] for s in seeds}
    for k in range(n_folds):
        test_c, val_c = folds[k], folds[(k + 1) % n_folds]
        set_fold(ds, test_c, val_c)
        best, _rows = hpsearch(ds, device=device, seeds=seeds[:2], grid=grid, verbose=False)
        set_fold(ds, test_c, [])          # val reverse dans le train, test INCHANGE
        fold_rec = {"fold": k, "test": test_c, "val": val_c, "cfg": best["cfg"],
                    "val_acc": best["val_acc"], "test_acc": [], "test_acc_pair": []}
        for s in seeds:
            m, pre = train_one(ds, best["cfg"], seed=s, device=device)
            r = evaluate(ds, m, pre, "test", device=device)
            rp = evaluate(ds, m, pre, "test", device=device, permute=True)
            rows_s[s].append(r["per_row"])
            rows_perm_s[s].append(rp["per_row"])
            fold_rec["test_acc"].append(r["acc"])
            fold_rec["test_acc_pair"].append(r["acc_by_pair"])
            fold_rec.setdefault("perm_acc_25", []).append(rp["acc"])
            fold_rec.setdefault("perm_acc_pair_25", []).append(rp["acc_by_pair_allperm"])
        out["per_fold"].append(fold_rec)
        if verbose:
            print(f"[cv] pli {k + 1}/{n_folds} test={test_c} val_acc={best['val_acc']:.3f} "
                  f"-> test row={np.mean(fold_rec['test_acc']):.3f} "
                  f"pair={np.mean(fold_rec['test_acc_pair']):.3f}  cfg={best['cfg']}",
                  flush=True)
    subsets = C.sealed_subsets(list(ds["pairs"].values())) if ds.get("pairs") else \
        {"all": set(cidxs)}
    for s in seeds:
        per_row, pair_marg = _agg_rows(rows_s[s])
        perm_row, perm_marg = _agg_rows(rows_perm_s[s])
        rec = {"seed": s, "per_row": per_row,
               "pair_margins": {str(c): m for c, m in pair_marg.items()},
               "permuted": subset_report(perm_row, perm_marg, subsets["all"]),
               "subsets": {name: subset_report(per_row, pair_marg, keep)
                           for name, keep in subsets.items()}}
        out["per_seed"].append(rec)
    out["subsets_corpus_idx"] = {k: sorted(v) for k, v in subsets.items()}
    # moyennes sur les graines, par sous-ensemble : c'est ce que gates.py lit
    out["summary"] = {}
    for name in subsets:
        vals = [r["subsets"][name] for r in out["per_seed"] if r["subsets"][name]]
        if not vals:
            continue
        out["summary"][name] = {
            "n_pairs": vals[0]["n_pairs"], "n_rows": vals[0]["n_rows"],
            "acc_row": float(np.mean([v["acc_row"] for v in vals])),
            "acc_pair": float(np.mean([v["acc_pair"] for v in vals])),
            "acc_pair_sd": float(np.std([v["acc_pair"] for v in vals])),
            "n_pairs_correct": float(np.mean([v["n_pairs_correct"] for v in vals])),
            "margin_mean": float(np.mean([v["margin_mean"] for v in vals])),
            "wilcoxon_pair_p": float(np.max([v["wilcoxon_pair_p"] for v in vals]))}
    pv = [r["permuted"] for r in out["per_seed"] if r["permuted"]]
    out["summary"]["permuted"] = {
        "acc_row": float(np.mean([v["acc_row"] for v in pv])),
        "acc_pair": float(np.mean([v["acc_pair"] for v in pv])),
        # la quantite de la regle 2 telle que l'audit l'a corrigee : moyennee sur 25
        # derangements par pli (les `per_row` ci-dessus ne portent que le premier tirage).
        # `acc_pair_25perm` est la MEME correction dans la lecture par PAIRE, celle que
        # gates.py applique (les 5 paires de chaque pli pesent pareil : moyenner par pli
        # revient a moyenner sur les 30 paires).
        "acc_row_25perm": float(np.mean([a for f in out["per_fold"]
                                         for a in f["perm_acc_25"]])),
        "acc_pair_25perm": float(np.mean([a for f in out["per_fold"]
                                          for a in f["perm_acc_pair_25"]]))}
    return out


# ---------------------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=("hpsearch", "final", "cv", "cross"))
    ap.add_argument("--source", default="state",
                    choices=("state", "h", "hdirect", "hdirect_masked",
                             # oracle D'ECRITURE MASQUEE (06/08) : memes signatures d'ETAT
                             # (lecture swiglu_apply), seules les ECRITURES rejouees changent ;
                             # la "source" n'est ici que le sous-repertoire de sig/ (sig-tag).
                             "state_mw_subject", "state_mw_bg_matched",
                             # oracle regle LINEAIRE (06/08 soir) : etat W (d x d) accumule
                             # par produits exterieurs, lu par W @ sondes ; L1..L4 du registre.
                             "lin_mean_subject", "lin_mean_all",
                             "lin_delta_subject", "lin_delta_all",
                             # bras MLP (07/08) : MEME etat SwiGLU, MEME lecture (swiglu_apply)
                             # que `state` ; seule l'ECRITURE rejouee change (objectif residu,
                             # calendrier sequentiel, pas brut : plan factoriel A1..A4).
                             "mlp_delta_all", "mlp_delta_subject",
                             "mlp_full", "mlp_full_muon", "mlp_hebb_seq"))
    ap.add_argument("--target", default="subject", choices=TARGETS)
    ap.add_argument("--features", default="pool", choices=FEATURES,
                    help="pool = signature COMPRIMEE (moyenne+max par sonde, 1536 nombres) ; "
                         "raw = signature brute laissee a l'ACP")
    ap.add_argument("--corpus", default="v1", choices=("v1", "v2", "v3", "v3h", "v4", "v4h"))
    ap.add_argument("--train-corpus", dest="train_corpus", default="",
                    choices=("", "v1", "v2", "v3", "v3h", "v4", "v4h"),
                    help="MODE INTER-CORPUS (chaine C2) : entraine sur CE corpus. Exige "
                         "--test-corpus. Les hyperparametres sont choisis sur une validation "
                         "INTERNE au corpus d'entrainement ; le corpus de test n'est vu "
                         "qu'UNE fois, a la toute fin.")
    ap.add_argument("--test-corpus", dest="test_corpus", default="",
                    choices=("", "v1", "v2", "v3", "v3h", "v4", "v4h"),
                    help="MODE INTER-CORPUS : evalue sur CE corpus (toutes ses paires).")
    ap.add_argument("--pretrain", action="store_true",
                    help="pre-entraine la tete sur les 105 videos `switch` (aucune n'est dans "
                         "une paire) avant l'entrainement sur le train du pli")
    ap.add_argument("--folds", type=int, default=6)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--cfg", default="", help="JSON d'hyperparametres (etape finale)")
    ap.add_argument("--audited", action="store_true",
                    help="OBLIGATOIRE pour --stage final/cv : atteste que le code a passe "
                         "l'audit externe. Sans lui, le test n'est JAMAIS touche.")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    if args.stage in ("final", "cv", "cross") and not args.audited:
        raise SystemExit(
            f"REFUS : `--stage {args.stage}` touche le SPLIT DE TEST, c'est le run de VERDICT.\n"
            "Le protocole (plan 04/08) exige un audit externe du code AVANT. Relance avec\n"
            "--audited une fois l'audit fait.")

    C.ensure_dirs()
    C.set_determinism(0)
    seeds = tuple(int(x) for x in args.seeds.split(","))

    if args.stage == "cross":
        assert args.train_corpus and args.test_corpus, \
            "--stage cross exige --train-corpus ET --test-corpus"
        assert args.train_corpus != args.test_corpus, \
            "--train-corpus == --test-corpus : ce n'est pas un run inter-corpus"
        assert not args.pretrain, \
            "--pretrain n'a pas de sens ici (le pre-entrainement v1 switch n'est pas du " \
            "materiau du corpus d'entrainement demande)"
        t0 = time.time()
        ds_tr = load_dataset(args.source, args.target, features=args.features,
                             corpus=args.train_corpus)
        ds_te = load_dataset(args.source, args.target, features=args.features,
                             corpus=args.test_corpus)
        res = cross_run(ds_tr, ds_te, device=args.device, seeds=seeds)
        res.update({"source": args.source, "target": args.target, "features": args.features,
                    "minutes": (time.time() - t0) / 60})
        tag = f"{args.source}_{args.target}_{args.features}_" \
              f"{args.train_corpus}2{args.test_corpus}"
        out = args.out or os.path.join(C.RUNS_DIR, f"cross_{tag}.json")
        C.save_json(res, out)
        s = res["summary"]
        print(f"[probe] CROSS {tag} : entraine {args.train_corpus} "
              f"({res['n_train_pairs']} paires) -> teste {args.test_corpus} "
              f"({res['n_test_pairs']} paires)")
        for name in ("strong15", "valid18", "all", "strong_noasym", "strong19", "valid25"):
            if name in s:
                print(f"          {name:14s} n={s[name]['n_pairs']:2d} "
                      f"{s[name]['n_pairs_correct']:.2f}/{s[name]['n_pairs']} "
                      f"(pair={s[name]['acc_pair']:.3f}, p_binom={s[name]['binom_pair_p']:.4g}, "
                      f"par graine {s[name]['n_pairs_correct_per_seed']})")
        print(f"          permute        pair={s['permuted']['acc_pair_25perm']:.3f} -> {out}")
        return

    ds = load_dataset(args.source, args.target, features=args.features, corpus=args.corpus)
    if args.pretrain:
        ds["_pretrain"] = load_pretrain(args.source, args.target, features=args.features)
    t0 = time.time()
    tag = f"{args.source}_{args.target}_{args.features}" \
          + ("_pre" if args.pretrain else "") + ("" if args.corpus == "v1" else f"_{args.corpus}")

    if args.stage == "cv":
        res = cv_run(ds, device=args.device, seeds=seeds, n_folds=args.folds)
        res.update({"source": args.source, "target": args.target, "features": args.features,
                    "corpus": args.corpus, "pretrain": bool(args.pretrain),
                    "minutes": (time.time() - t0) / 60})
        out = args.out or os.path.join(C.RUNS_DIR, f"cv_{tag}.json")
        C.save_json(res, out)
        s = res["summary"]
        print(f"[probe] CV {tag} : 30 paires -> pair={s['all']['acc_pair']:.3f} "
              f"row={s['all']['acc_row']:.3f} p={s['all']['wilcoxon_pair_p']:.4g} | "
              f"permute row={s['permuted']['acc_row_25perm']:.3f} -> {out}")
        for name in ("agent", "aden15", "decisive", "valid25", "strong19"):
            if name in s:
                print(f"          {name:9s} n={s[name]['n_pairs']:2d} "
                      f"pair={s[name]['acc_pair']:.3f} "
                      f"({s[name]['n_pairs_correct']:.1f}/{s[name]['n_pairs']}) "
                      f"row={s[name]['acc_row']:.3f}")
        return

    if args.stage == "hpsearch":
        best, rows = hpsearch(ds, device=args.device, seeds=seeds)
        out = args.out or os.path.join(C.RUNS_DIR, f"hp_{tag}.json")
        C.save_json({"source": args.source, "target": args.target, "seeds": list(seeds),
                     "features": args.features, "corpus": args.corpus,
                     "pretrain": bool(args.pretrain),
                     "best": best, "top20": rows[:20], "n_configs": len(rows),
                     "minutes": (time.time() - t0) / 60}, out)
        print(f"[probe] hpsearch: meilleur val_acc={best['val_acc']:.3f} -> {out}")
        return

    cfg = json.loads(args.cfg) if args.cfg else None
    if cfg is None:
        hp = os.path.join(C.RUNS_DIR, f"hp_{tag}.json")
        assert os.path.exists(hp), f"pas de {hp} : fais d'abord --stage hpsearch"
        cfg = C.load_json(hp)["best"]["cfg"]
    res = {"cfg": cfg, "source": args.source, "target": args.target, "seeds": list(seeds),
           "features": args.features, "corpus": args.corpus, "pretrain": bool(args.pretrain)}
    got = {"test": [], "test_permuted": []}
    for s in seeds:
        # un SEUL modele par graine : le bras et son controle permute sont evalues sur
        # EXACTEMENT le meme sondeur, sinon le controle mesurerait aussi la variance d'init.
        m, pre = train_one(ds, cfg, seed=s, device=args.device)
        got["test"].append(evaluate(ds, m, pre, "test", device=args.device))
        got["test_permuted"].append(
            evaluate(ds, m, pre, "test", device=args.device, permute=True))
    for name, ps in got.items():
        accs = [p["acc"] for p in ps]
        # audit 2026-08-04 : `per_row` est conserve (il portait la seule trace permettant de
        # recalculer une precision sur un sous-ensemble de paires). Sans lui, l'analyse
        # SECONDAIRE pre-enregistree a l'addendum (le sous-ensemble "le masque porte bien le
        # protagoniste") etait litteralement incalculable a partir du JSON de verdict.
        res[name] = {"acc_mean": float(np.mean(accs)), "acc_sd": float(np.std(accs)),
                     "acc_by_pair_mean": float(np.mean([p["acc_by_pair"] for p in ps])),
                     "per_seed": ps}
    out = args.out or os.path.join(C.RUNS_DIR, f"final_{tag}.json")
    C.save_json(res, out)
    print(f"[probe] FINAL {tag}: test acc "
          f"{res['test']['acc_mean']:.3f} +- {res['test']['acc_sd']:.3f} | "
          f"permute {res['test_permuted']['acc_mean']:.3f} -> {out}")


if __name__ == "__main__":
    main()
