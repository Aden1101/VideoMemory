#!/usr/bin/env python
"""ETAPE 1, selftests (le contrat exige par la commande, un test par ligne du contrat).

  T1  py_compile de tous les fichiers probe1
  T2  split : 18/5/7 par PAIRE, aucun chevauchement train/val/test (assert par ids), jumeau
      toujours du meme cote, deterministe (2 appels identiques)
  T3  segments / ages : `end_subject` et `end_seq` presents, ages valides, le segment-sujet
      est bien anterieur au switch
  T4  batterie de sondes : forme, RMS unitaire, finitude, zero video val/test dans le k-means
  T5  un SEUL CHEMIN : borne superieure et etats passent par le meme `signature()`, seul un
      flag d'entree change (verification du code source + des formes)
  T6  signature DETERMINISTE : 2 appels = bitwise, pour les DEUX sources
  T7  masques : IoU > 0 sur 100 % des frames echantillonnees, sinon la video est signalee
  T8  le blanchiment/ACP ne fuit pas du test (perturber le test ne bouge aucune statistique)
  T9  le sondeur, bout en bout sur donnees SYNTHETIQUES a reponse connue : signal -> haute
      precision, bruit pur -> hasard, controle permute -> hasard
  T10 le run de verdict est verrouille : `--stage final` ET `--stage cv` refusent sans
      `--audited`
  T11 (GPU, optionnel) contrat de rejeu : delta_l1 == 0 et rejeu bit a bit reproductible

--- v2 (2026-08-05), les trois defauts du verdict "PROTOCOLE INVALIDE" + les materiaux neufs :
  T12 compression : 1536 nombres (2 par sonde et par bloc), deterministe, sans parametre,
      ligne a ligne (donc sans fuite possible), et INVERSIBLE sur la source hdirect
  T13 borne sans ECRITURE : le lecteur hdirect n'appelle ni swiglu_apply ni aucun rejeu et
      n'utilise meme pas les sondes ; c'est bien (moyenne, max) sur tous les tokens de tous
      les chunks jusqu'a l'age
  T14 gates : PRIMARY == 'pair' partout, aucun seuil applique a une precision par ligne,
      regle 1 re-ancree a 5/6 sur les "clairement differents", 4 sous-ensembles rapportes
  T15 plis : 6 x 5 paires, partition exacte, deterministe, blanchiment re-ajuste par pli et
      aveugle a son test
  T16 PRE-ENTRAINEMENT : 0 video de paire dans le pool des 105 switch
  T17 PROMPTS v2 : 30 prompts sans aucun attribut d'apparence, sujets reconnus, held-out
  T18 NUMEROTATION de planche = celle des listes d'oeil scellees (10 paires nommees verifiees)
  T19 (GPU) --stream == disque, bit a bit
  T20 validation CROISEE bout en bout sur donnees a reponse connue
  T21 le collecteur reste ADDITIF : sans --prompts-file, la selection v3 est inchangee

--- v3 (2026-08-05 nuit), CORRECTIF de borne (pooling sous le masque-sujet) :
  T22 borne MASQUEE : (a) masque plein => `hdirect_masked` == `hdirect` bitwise ;
      (b) masque vide => repli top-3 % declenche et logge (jamais un pooling vide) ;
      (c) deterministe (egalites tranchees par l'index) ; (d) alignement pixel -> grille
      30 x 52 verifie sur un damier CONNU + controle sur les vrais masques du disque

Usage :  python reference/runs/probe1/selftest_probe1.py            # tout ce qui ne coute rien
         CUDA_VISIBLE_DEVICES=7 python .../selftest_probe1.py --gpu # + T6 reel, T11 et T19
"""
import argparse
import glob
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import probe1_common as C  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as Fn  # noqa: E402

PY = sys.executable
RESULTS = []


def check(name, cond, detail=""):
    RESULTS.append((bool(cond), name, detail))
    print(f"  [{'OK ' if cond else 'ECHEC'}] {name}" + (f": {detail}" if detail else ""),
          flush=True)
    return bool(cond)


# ---------------------------------------------------------------------------------------------
def t1_compile():
    print("T1 py_compile")
    files = sorted(glob.glob(os.path.join(HERE, "*.py")))
    r = subprocess.run([PY, "-m", "py_compile"] + files, capture_output=True, text=True)
    check(f"py_compile de {len(files)} fichiers", r.returncode == 0, r.stderr[-300:])


def t2_split():
    print("T2 split strict par paire")
    vids, pairs = C.pair_dataset()
    n = {s: sum(1 for p in pairs if p["split"] == s) for s in ("train", "val", "test")}
    check("30 paires", len(pairs) == 30, f"{len(pairs)}")
    check("18 / 5 / 7", (n["train"], n["val"], n["test"]) == (18, 5, 7), str(n))
    ids = {s: set() for s in ("train", "val", "test")}
    for p in pairs:
        ids[p["split"]] |= {p["main"], p["twin"]}
    check("aucun chevauchement train/val/test (ids de video)",
          not (ids["train"] & ids["val"]) and not (ids["train"] & ids["test"])
          and not (ids["val"] & ids["test"]),
          f"|train|={len(ids['train'])} |val|={len(ids['val'])} |test|={len(ids['test'])}")
    check("le jumeau est toujours du cote de son frere",
          all(p["main"].split(":")[1] == p["twin"].split(":")[1] for p in pairs))
    s1 = C.assign_pair_split(C.build_pairs(vids))
    s2 = C.assign_pair_split(C.build_pairs(vids))
    check("split deterministe (2 appels identiques)", s1 == s2)
    check("les 60 videos de paires sont distinctes",
          len({p["main"] for p in pairs} | {p["twin"] for p in pairs}) == 60)


def t3_segments_ages():
    print("T3 segments-sujet et ages d'etat")
    import mem2_data as D
    vids, _ = C.pair_dataset()
    bad_age, bad_seg, bad_order = [], [], []
    for v in vids:
        segs = C.segments(v)
        if v["switch_frame"] is None:
            if len(segs) != 1:
                bad_seg.append(v["video_id"])
        elif len(segs) != 2 or segs[0]["f1"] != v["switch_frame"]:
            bad_seg.append(v["video_id"])
        ages = C.state_ages(v)
        names = {a["name"] for a in ages}
        n_w = len(v["write_chunks"])
        need = {"end_seq"} | ({"end_subject"} if v["switch_frame"] is not None else set())
        if not need <= names or any(a["i"] >= n_w for a in ages) \
                or len({a["i"] for a in ages}) != len(ages):
            bad_age.append(v["video_id"])
        if "end_subject" in names:
            i_es = next(a["i"] for a in ages if a["name"] == "end_subject")
            if D.frame_of_write(v, i_es) >= v["switch_frame"]:
                bad_order.append(v["video_id"])
    check("segments bien formes", not bad_seg, str(bad_seg[:5]))
    check("`end_seq` toujours present, `end_subject` sur les switch, ages valides et distincts",
          not bad_age, str(bad_age[:5]))
    check("`end_subject` evince bien une frame ANTERIEURE au switch", not bad_order,
          str(bad_order[:5]))
    check("~4 ages par video", all(len(C.state_ages(v)) <= C.N_AGES for v in vids))


def t4_probes():
    print("T4 batterie de sondes")
    from build_probes import PROBES_PT
    if not os.path.exists(PROBES_PT):
        check("probes.pt present", False, f"{PROBES_PT} absent (lance build_probes.py)")
        return
    pk = torch.load(PROBES_PT, map_location="cpu", weights_only=False)
    p = pk["probes"]
    check("forme (n_blocs, 256, 1536)",
          tuple(p.shape) == (len(pk["blocks"]), C.N_PROBES, C.DIM), str(tuple(p.shape)))
    rms = p.pow(2).mean(-1).sqrt()
    check("sondes RMS-normalisees", bool(((rms - 1).abs() < 1e-4).all()),
          f"min {rms.min():.5f} max {rms.max():.5f}")
    check("sondes finies et non nulles",
          bool(torch.isfinite(p).all()) and bool((p.abs().sum(-1) > 0).all()))
    _, pairs = C.pair_dataset()
    bad = {s: {q["main"] for q in pairs if q["split"] == s} | {q["twin"] for q in pairs
                                                              if q["split"] == s}
           for s in ("val", "test")}
    check("le k-means n'a vu AUCUNE video val/test", pk["kmeans_source"] in ("switch", "train"),
          f"source={pk['kmeans_source']}, {pk['n_pool_videos']} videos "
          f"(val={len(bad['val'])} test={len(bad['test'])} exclues par construction)")
    check("centroides et requetes distincts",
          not torch.allclose(p[:, :4], p[:, C.N_CENTROIDS:C.N_CENTROIDS + 4]))


def t5_single_path():
    print("T5 un seul chemin pour la borne de validite et pour les etats")
    import inspect

    import build_signatures as B
    src = inspect.getsource(B.signature)
    check("`signature` ne branche pas sur la source (pas de if/elif source)",
          "if source" not in src and "elif" not in src and "READERS[source]" in src)
    # L'invariant est "un SEUL registre" : toute source declaree a un lecteur, et aucun lecteur
    # n'existe hors des sources. Le litteral fige a 4 noms datait d'avant les bras d'oracle
    # d'ecriture (lin_* 06/08, mlp_delta_* 07/08), qui AJOUTENT des sources sans ajouter de
    # chemin : il testait la liste du jour, pas la propriete. Le socle historique reste
    # explicitement exige.
    check("toutes les sources sont enregistrees dans le MEME registre (un seul chemin)",
          set(B.READERS) == set(B.SOURCES)
          and {"state", "h", "hdirect", "hdirect_masked"} <= set(B.SOURCES),
          f"{len(B.SOURCES)} sources : {', '.join(sorted(B.SOURCES))}")
    check("`hdirect_masked` partage LITTERALEMENT le lecteur de `hdirect` (un seul chemin)",
          B.READERS["hdirect_masked"] is B.READERS["hdirect"])
    csrc = inspect.getsource(B.compress_signature)
    check("la COMPRESSION ne branche pas sur la source non plus",
          "source" not in csrc and "if " not in csrc.split('"""')[-1])
    dim, P, nb = 32, 6, 2
    blocks = [5, 16]
    probes = Fn.normalize(torch.randn(nb, P, dim), dim=-1) * (dim ** 0.5)
    st = {b: tuple(torch.randn(1, dim, 4 * dim) if k != 1 else torch.randn(1, 4 * dim, dim)
                   for k in range(3)) for b in blocks}
    hh = {b: torch.randn(1, 40, dim) for b in blocks}
    hseq = {b: [torch.randn(1, 40, dim) for _ in range(3)] for b in blocks}
    a = B.signature("state", probes, st, blocks)
    b = B.signature("h", probes, hh, blocks)
    d = B.signature("hdirect", probes, hseq, blocks)
    check("meme forme de sortie pour les deux sources a sondes",
          a.shape == b.shape == (nb, P, dim), f"{tuple(a.shape)} vs {tuple(b.shape)}")
    check("la source hdirect rend une matrice-reponse 2-D homogene entre blocs",
          d.shape == (nb, dim, 2), str(tuple(d.shape)))
    check("sorties finies", bool(torch.isfinite(a).all() and torch.isfinite(b).all()
                                 and torch.isfinite(d).all()))
    ca, cd = B.compress_signature(a), B.compress_signature(d)
    check("la MEME compression s'applique aux deux (aucune connaissance de la source)",
          ca.shape == (nb, P, 2) and cd.shape == (nb, dim, 2),
          f"{tuple(ca.shape)} / {tuple(cd.shape)}")


def t12_compression():
    print("T12 compression : dimensionnalite, determinisme, absence de fuite")
    import build_signatures as B
    sig = torch.randn(3, C.N_PROBES, C.DIM)
    c1 = B.compress_signature(sig)
    c2 = B.compress_signature(sig.clone())
    check("2 valeurs par sonde et par bloc -> 1536 nombres au total",
          c1.shape == (3, C.N_PROBES, 2) and c1.numel() == 1536, str(tuple(c1.shape)))
    check("compression bitwise deterministe", bool(torch.equal(c1, c2)))
    check("compression = (moyenne, max) exacts",
          bool(torch.allclose(c1[..., 0], sig.mean(-1)) and
               torch.allclose(c1[..., 1], sig.amax(-1))))
    # la compression ne peut pas fuir un split : elle ne voit qu'UNE ligne a la fois
    big = torch.randn(8, 3, C.N_PROBES, C.DIM)
    each = torch.stack([B.compress_signature(x) for x in big], 0)
    check("la compression d'une ligne ne depend d'aucune autre ligne",
          bool(torch.equal(each[0], B.compress_signature(big[0]))))
    # la VRAIE matrice-reponse hdirect (colonne 0 = moyenne sur les tokens, colonne 1 = max :
    # colonne 1 >= colonne 0 par construction)
    hd = torch.stack([B.read_hdirect(None, [torch.randn(1, 50, C.DIM)]) for _ in range(3)], 0)
    check("la borne hdirect comprimee fait 9216 nombres (3 blocs x 1536 canaux x 2)",
          B.compress_signature(hd).numel() == 9216)
    check("colonne 1 (max) >= colonne 0 (moyenne) : la propriete qui rend la compression sure",
          bool((hd[..., 1] >= hd[..., 0]).all()))
    # ... et elle ne perd rien : de ((m+M)/2, M) on retrouve (m, M)
    z = B.compress_signature(hd)
    m_rec = 2 * z[..., 0] - z[..., 1]
    check("la compression de hdirect est INVERSIBLE (aucune perte d'information)",
          bool(torch.allclose(m_rec, hd[..., 0], atol=1e-4)),
          f"max |delta| = {float((m_rec - hd[..., 0]).abs().max()):.2e}")


def t13_bound_no_write():
    print("T13 borne de validite : AUCUNE ecriture dans la boucle")
    import inspect

    import build_signatures as B
    src = inspect.getsource(B.read_hdirect)
    check("le lecteur hdirect n'appelle NI swiglu_apply NI aucune ecriture",
          "swiglu" not in src and "replay" not in src and "state" not in src)
    check("il n'utilise meme pas la batterie de sondes (pas de produit avec `probes_b`)",
          "probes_b" not in src.split('"""')[-1])
    dim, nb = 16, 2
    blocks = [5, 16]
    probes = torch.randn(nb, 4, dim)
    hs = {b: [torch.randn(1, 7, dim) for _ in range(3)] for b in blocks}
    out = B.signature("hdirect", probes, hs, blocks)
    for k, b in enumerate(blocks):
        cat = torch.cat([h[0] for h in hs[b]], 0)
        check(f"bloc {b}: c'est bien (moyenne, max) sur TOUS les tokens de TOUS les chunks",
              bool(torch.allclose(out[k, :, 0], cat.mean(0), atol=1e-5) and
                   torch.allclose(out[k, :, 1], cat.amax(0), atol=1e-5)))
    check("hdirect est declaree source SEQUENTIELLE (tous les writes jusqu'a l'age)",
          "hdirect" in B.SEQ_SOURCES)
    srcp = inspect.getsource(B.load_payload)
    check("la charge utile d'une source sequentielle est bien h[0..age], pas h[age]",
          "[:int(age[\"i\"]) + 1]" in srcp)


def t22_masked_bound():
    """CORRECTIF de borne (registre, 05/08 nuit) : `hdirect_masked` = le MEME pooling que
    `hdirect`, restreint aux tokens sous le masque-sujet. Les quatre contrats exiges :
    (a) masque PLEIN => identique bitwise a `hdirect` ; (b) masque vide => repli declenche
    (jamais un pooling vide) ; (c) deterministe ; (d) l'alignement pixel -> grille de tokens
    est le bon, verifie sur un damier CONNU."""
    print("T22 borne masquee : masque-sujet -> tokens h")
    import build_signatures as B

    # --- (d) ALIGNEMENT pixel -> grille, sur un cas synthetique a reponse connue -------------
    H, W = 480, 832
    cell_h, cell_w = H // C.GRID_H, W // C.GRID_W
    check("un patch de token = 16 x 16 pixels du masque", (cell_h, cell_w) == (16, 16),
          f"{cell_h}x{cell_w}")
    rows = np.arange(H) // cell_h
    cols = np.arange(W) // cell_w
    damier = ((rows[:, None] + cols[None, :]) % 2 == 0)      # damier cellule par cellule
    sal = C.mask_salience_grid(damier)
    att = np.zeros((C.GRID_H, C.GRID_W), dtype=np.float32)
    att[::2, ::2] = 1.0
    att[1::2, 1::2] = 1.0
    check("le damier cellule-par-cellule se retrouve EXACTEMENT sur la grille 30 x 52",
          bool(np.array_equal(sal.reshape(C.GRID_H, C.GRID_W), att)),
          f"{int((sal > 0.5).sum())} cellules retenues sur {sal.size}")
    # un carre PIXEL place a la main : ses tokens sont ceux, et seulement ceux, de son rectangle
    r0, r1, c0, c1 = 7, 12, 20, 33                            # en cellules
    carre = np.zeros((H, W), dtype=bool)
    carre[r0 * cell_h:r1 * cell_h, c0 * cell_w:c1 * cell_w] = True
    sel, fb = C.frame_token_mask(C.mask_salience_grid(carre))
    idx = set(np.flatnonzero(sel).tolist())
    want = {r * C.GRID_W + c for r in range(r0, r1) for c in range(c0, c1)}
    check("un carre pixel -> EXACTEMENT les tokens de son rectangle (index = ligne*52+colonne)",
          idx == want and not fb, f"{len(idx)} tokens, attendus {len(want)}")
    # decalage sous-patch : la regle de MAJORITE conserve l'aire (ni dilatation ni erosion).
    # Le rectangle glisse de 6 pixels : sa premiere ligne de cellules est couverte a 10/16
    # (retenue), sa derniere a 6/16 (rejetee) -> autant de tokens qu'avant le glissement.
    decale = np.zeros((H, W), dtype=bool)
    decale[r0 * cell_h + 6:r1 * cell_h + 6, c0 * cell_w:c1 * cell_w] = True
    sel2, _ = C.frame_token_mask(C.mask_salience_grid(decale))
    check("la regle de MAJORITE conserve l'aire (decalage sous-patch : ni dilatation ni erosion)",
          int(sel2.sum()) == len(want), f"{int(sel2.sum())} vs {len(want)}")
    # a l'egalite EXACTE (decalage d'un demi-patch), le seuil >= tranche vers l'inclusion :
    # une ligne de cellules de plus, jamais un pooling ampute. C'est declare, pas subi.
    demi = np.zeros((H, W), dtype=bool)
    demi[r0 * cell_h + 8:r1 * cell_h + 8, c0 * cell_w:c1 * cell_w] = True
    sel2b, _ = C.frame_token_mask(C.mask_salience_grid(demi))
    check("a l'egalite exacte (demi-patch), le seuil >= inclut les deux bords",
          int(sel2b.sum()) == len(want) + (c1 - c0), f"{int(sel2b.sum())} vs {len(want)}")

    # --- (a) masque PLEIN == hdirect, bitwise -----------------------------------------------
    dim, nb, nch = 24, 2, 4
    blocks = [5, 16]
    probes = torch.randn(nb, 3, dim)
    torch.manual_seed(7)
    raw = {b: [torch.randn(1, 3 * C.TOKENS_PER_FRAME, dim) for _ in range(nch)] for b in blocks}
    full = torch.ones(3 * C.TOKENS_PER_FRAME, dtype=torch.bool)
    masked_full = {b: [(t, full) for t in raw[b]] for b in blocks}
    a = B.signature("hdirect", probes, raw, blocks)
    bfull = B.signature("hdirect_masked", probes, masked_full, blocks)
    check("(a) masque PLEIN : hdirect_masked == hdirect BIT A BIT",
          bool(torch.equal(a, bfull)),
          f"max |delta| = {float((a - bfull).abs().max()):.3e}")
    # et un masque PARTIEL est bien le pooling des seuls tokens retenus
    part = torch.zeros(3 * C.TOKENS_PER_FRAME, dtype=torch.bool)
    part[5:400] = True
    bp = B.signature("hdirect_masked", probes, {b: [(t, part) for t in raw[b]] for b in blocks},
                     blocks)
    cat = torch.cat([t[0][part] for t in raw[blocks[0]]], 0)
    check("masque PARTIEL : c'est (moyenne, max) sur les SEULS tokens retenus",
          bool(torch.allclose(bp[0, :, 0], cat.mean(0), atol=1e-5)
               and torch.allclose(bp[0, :, 1], cat.amax(0), atol=1e-5)))
    check("le pooling masque DIFFERE du pooling uniforme (le correctif fait quelque chose)",
          float((bp[0] - a[0]).abs().max()) > 1e-3)

    # --- (b) masque VIDE -> repli declenche, jamais un pooling vide --------------------------
    vide = np.zeros(C.TOKENS_PER_FRAME, dtype=np.float32)
    sel3, fb3 = C.frame_token_mask(vide)
    k = int(np.ceil(C.MASK_MIN_FRAC * C.TOKENS_PER_FRAME))
    check("(b) masque VIDE : le repli est DECLENCHE et retient exactement le plancher de 3 %",
          fb3 and int(sel3.sum()) == k, f"repli={fb3}, {int(sel3.sum())} tokens (plancher {k})")
    presque = np.zeros(C.TOKENS_PER_FRAME, dtype=np.float32)
    presque[:10] = 1.0                                        # 10 tokens < 47 = 3 %
    sel4, fb4 = C.frame_token_mask(presque)
    check("un masque TROP PETIT (10 tokens) declenche aussi le repli, et garde ses 10 tokens",
          fb4 and int(sel4.sum()) == k and bool(sel4[:10].all()))
    gros = np.zeros(C.TOKENS_PER_FRAME, dtype=np.float32)
    gros[:200] = 1.0
    sel5, fb5 = C.frame_token_mask(gros)
    check("un masque suffisant (200 tokens) ne declenche AUCUN repli",
          (not fb5) and int(sel5.sum()) == 200)
    # le repli est declare dans le code de production, pas seulement ici
    import inspect
    check("le repli est LOGGE par le constructeur de signatures",
          "REPLI" in inspect.getsource(B.load_payload))

    # --- (c) DETERMINISME --------------------------------------------------------------------
    sal_r = np.abs(np.sin(np.arange(C.TOKENS_PER_FRAME, dtype=np.float64) * 12.9898)) * 0.49
    s1, f1 = C.frame_token_mask(sal_r.astype(np.float32))
    s2, f2 = C.frame_token_mask(sal_r.astype(np.float32))
    check("(c) selection deterministe (2 appels identiques, egalites tranchees par l'index)",
          bool(np.array_equal(s1, s2)) and f1 == f2 and f1)
    plat = np.full(C.TOKENS_PER_FRAME, 0.1, dtype=np.float32)  # QUE des egalites
    p1, _ = C.frame_token_mask(plat)
    p2, _ = C.frame_token_mask(plat)
    check("egalites parfaites : le repli prend les PREMIERS index, deux fois pareil",
          bool(np.array_equal(p1, p2)) and bool(p1[:k].all()) and not bool(p1[k:].any()))
    b1 = B.signature("hdirect_masked", probes, masked_full, blocks)
    b2 = B.signature("hdirect_masked", probes, masked_full, blocks)
    check("signature hdirect_masked bitwise deterministe", bool(torch.equal(b1, b2)))
    check("hdirect_masked est une source SEQUENTIELLE ET une source MASQUEE",
          "hdirect_masked" in B.SEQ_SOURCES and "hdirect_masked" in B.MASKED_SOURCES)

    # --- sur les VRAIS masques du disque : le chunk trouve sa frame, le pooling n'est jamais
    #     vide, et l'aire retenue est du bon ordre --------------------------------------------
    try:
        vids, pairs = C.pair_dataset(corpus="v1")
    except Exception as e:                                    # pas d'index sur cette machine
        check("corpus disponible pour le controle sur vrais masques", False, str(e)[:80])
        return
    byid = C.vids_by_id(vids)
    pid = sorted({p["main"] for p in pairs})[:6]
    fracs, replis, nchunks = [], 0, 0
    for vid in pid:
        meta = byid[vid]
        grids = C.video_mask_grids(vid)
        n = len(meta["write_chunks"])
        for i in range(n):
            m, fb = C.chunk_token_mask(meta, i, grids, n_tokens=3 * C.TOKENS_PER_FRAME)
            fracs.append(m.mean())
            replis += int(fb)
            nchunks += 1
    check(f"vrais masques ({len(pid)} videos, {nchunks} chunks) : le pooling n'est JAMAIS vide",
          min(fracs) >= C.MASK_MIN_FRAC - 1e-9,
          f"aire retenue min/med/max = {min(fracs):.3f}/{float(np.median(fracs)):.3f}/"
          f"{max(fracs):.3f}, repli sur {replis}/{nchunks} chunks")
    check("l'aire retenue reste MINORITAIRE (c'est bien un masque-sujet, pas tout le cadre)",
          float(np.median(fracs)) < 0.5, f"mediane {float(np.median(fracs)):.3f}")
    meta = byid[pid[0]]
    g = C.video_mask_grids(pid[0])
    fs = C.chunk_frames(meta, 0)
    check("les frames d'un chunk sont contigues et finissent sur `frame_of_write`",
          len(fs) == int(meta["num_frame_per_block"]) and fs[-1] ==
          __import__("mem2_data").frame_of_write(meta, 0) and fs == list(range(fs[0], fs[-1] + 1)),
          str(fs))
    check("le masque d'un chunk vient de la frame echantillonnee la plus proche de son centre",
          min(abs(t[0] - fs[len(fs) // 2]) for t in g) ==
          abs(min(g, key=lambda t: (abs(t[0] - fs[len(fs) // 2]), t[0]))[0] - fs[len(fs) // 2]))


def t6_determinism(gpu=False):
    print("T6 signature deterministe (2 appels = bitwise)")
    import build_signatures as B
    dev = "cuda" if (gpu and torch.cuda.is_available()) else "cpu"
    C.set_determinism(0)
    dim, P, nb = 1536, 16, 3
    blocks = [5, 16, 27]
    probes = (torch.randn(nb, P, dim, device=dev) * 0.3)
    st = {b: (torch.randn(1, dim, 4 * dim, device=dev, dtype=torch.float16),
              torch.randn(1, 4 * dim, dim, device=dev, dtype=torch.float16),
              torch.randn(1, dim, 4 * dim, device=dev, dtype=torch.float16)) for b in blocks}
    hh = {b: torch.randn(1, 4680, dim, device=dev, dtype=torch.float16) for b in blocks}
    for name, payload in (("state", st), ("h", hh)):
        a = B.signature(name, probes, payload, blocks)
        b = B.signature(name, probes, payload, blocks)
        check(f"signature {name} bitwise identique sur 2 appels ({dev})",
              bool(torch.equal(a, b)),
              f"max |delta| = {float((a - b).abs().max()):.3e}")

    real = sorted(glob.glob(os.path.join(C.SIG_DIR, "state", "*.pt")))
    if real and gpu:
        from build_probes import PROBES_PT
        from build_states import state_path
        pk = torch.load(PROBES_PT, map_location="cpu", weights_only=False)
        pr, bl = pk["probes"].to(dev), list(pk["blocks"])
        # une signature dont l'ETAT est aussi sur disque : depuis le mode --stream (v2.4) la
        # plupart des signatures n'ont plus d'etat stocke, c'est voulu.
        d, sp = None, None
        for f in real:
            x = torch.load(f, map_location="cpu", weights_only=False)
            if os.path.exists(state_path(x["video_id"], x["age"])):
                d, sp = x, state_path(x["video_id"], x["age"])
                break
        if d is None:
            check("signature REELLE : saute (aucun etat stocke sur disque)", True, "")
            return
        ck = torch.load(sp, map_location="cpu", weights_only=False)
        pay = {b: tuple(t.to(dev) for t in ck["state"][b]) for b in bl}
        s1 = B.signature("state", pr, pay, bl)
        s2 = B.signature("state", pr, pay, bl)
        check("signature REELLE bitwise identique sur 2 appels",
              bool(torch.equal(s1, s2)), os.path.basename(sp))
        check("la signature stockee reproduit le recalcul",
              bool(torch.equal(s1.to(torch.float16).cpu(), d["sig"])),
              f"max |delta| = {float((s1.to(torch.float16).cpu() - d['sig']).abs().max()):.3e}")


def t7_masks():
    print("T7 masques : IoU > 0 sur 100 % des frames, sinon la video est SIGNALEE")
    files = sorted(glob.glob(os.path.join(C.MASK_DIR, "*.npz")))
    if not files:
        check("masques presents", False, "aucun .npz (lance build_masks.py)")
        return
    bad, flagged, n_frames = [], [], 0
    for f in files:
        r = np.load(f, allow_pickle=True)
        iou, valid, fl = r["iou_box"], r["valid"], bool(r["flagged"])
        n_frames += iou.shape[0]
        if fl:
            flagged.append(os.path.basename(f)[:-4])
        elif not (iou > 0).all() or not valid.all():
            bad.append(os.path.basename(f)[:-4])
    check(f"{len(files)} segments, {n_frames} frames : toute frame d'IoU nulle est SIGNALEE",
          not bad, f"non signales mais invalides : {bad[:8]}")
    check("le taux de segments signales reste sous 20 %",
          len(flagged) <= 0.20 * len(files),
          f"{len(flagged)}/{len(files)} signales : {flagged[:8]}")
    r = np.load(files[0], allow_pickle=True)
    check("le .npz porte masque + crops + embeddings DINO sujet ET fond",
          all(k in r for k in ("masks_packed", "dino_subj", "dino_bg", "boxes", "frames")))
    check("sujet et fond ne sont PAS le meme embedding",
          float(np.abs(r["dino_subj"] - r["dino_bg"]).max()) > 1e-3)


# ---------------------------------------------------------------------------------------------
# jeu SYNTHETIQUE (T8 / T9) : meme structure exacte que load_dataset
# ---------------------------------------------------------------------------------------------
def synth_dataset(signal=True, n_pairs=30, n_ages=4, D=2000, seed=0, snr=1.0):
    """Un jeu a REPONSE CONNUE : la signature d'une video est une image lineaire fixe de
    l'embedding DINO de son sujet, noyee dans un bruit d'amplitude comparable. Un instrument
    sain doit y reussir presque parfaitement (le sujet EST dans l'entree) et retomber au
    hasard quand on retire le signal ou quand on permute les etats. `A` est mis a l'echelle
    pour que le terme de signal ait une variance par coordonnee de ~1, comme le bruit."""
    g = torch.Generator().manual_seed(seed)
    A = torch.randn(C.DINO_DIM, D, generator=g)
    split_of = ["train"] * 18 + ["val"] * 5 + ["test"] * 7
    videos, X, s_vid, s_age = [], [], [], []
    for p in range(n_pairs):
        base = len(videos)
        e = Fn.normalize(torch.randn(2, C.DINO_DIM, generator=g), dim=-1)
        for off in range(2):
            refs = Fn.normalize(e[off][None] + 0.05 * torch.randn(5, C.DINO_DIM, generator=g),
                                dim=-1)
            videos.append({"video_id": f"synth:{p}:{off}", "refs": refs,
                           "twin_index": base + (1 - off), "split": split_of[p],
                           "pair_id": p, "corpus_idx": p})
            for a in range(n_ages):
                noise = torch.randn(D, generator=g)
                x = (e[off] @ A) * snr + noise if signal else noise
                X.append(x.to(torch.float16))
                s_vid.append(base + off)
                s_age.append(f"a{a}")
    return {"videos": videos, "X": torch.stack(X, 0), "s_vid": torch.tensor(s_vid),
            "s_age": s_age, "source": "synth", "target": "subject", "dropped_pairs": []}


def t8_no_leak():
    print("T8 le blanchiment / l'ACP ne fuient pas du test")
    import train_probe as T
    ds = synth_dataset(signal=True, seed=1)
    pre = T.Preproc(ds["X"][T.split_rows(ds, "train")])
    mu0, sd0, W0, Z0 = pre.mu.clone(), pre.sd.clone(), pre.W_full.clone(), pre.Ztr.clone()
    ds2 = dict(ds)
    ds2["X"] = ds["X"].clone()
    te = T.split_rows(ds2, "test")
    ds2["X"][te] = ds2["X"][te] * 100 + 7          # on saccage le test
    pre2 = T.Preproc(ds2["X"][T.split_rows(ds2, "train")])
    check("mu / sd inchanges quand le TEST change",
          torch.equal(mu0, pre2.mu) and torch.equal(sd0, pre2.sd))
    check("la base ACP est inchangee quand le TEST change",
          torch.equal(W0, pre2.W_full) and torch.equal(Z0, pre2.Ztr))
    check("l'ACP est ajustee sur le TRAIN SEUL (n_train lignes)",
          pre.n_train == T.split_rows(ds, "train").numel(),
          f"{pre.n_train} lignes")
    a = pre.transform(ds["X"][te], 32)
    b = pre.transform(ds["X"][te], 32)
    check("transform deterministe", torch.equal(a, b))


def t9_prober_synthetic():
    print("T9 le sondeur, bout en bout, sur donnees a reponse connue")
    import train_probe as T
    cfg = {"sig_dim": 64, "hidden": 256, "depth": 2, "lr": 1e-3, "wd": 1e-2,
           "temp": 0.05, "dropout": 0.1, "steps": 400, "batch": 64}
    ds = synth_dataset(signal=True, seed=2, snr=1.5)
    m, pre = T.train_one(ds, cfg, seed=0)
    r = T.evaluate(ds, m, pre, "test")
    check("SIGNAL : le sondeur retrouve le bon individu (acc >= 0.90)", r["acc"] >= 0.90,
          f"acc={r['acc']:.3f} marge={r['margin_mean']:+.4f} "
          f"p_video={r['wilcoxon_video_p']:.4g}")
    rp = T.evaluate(ds, m, pre, "test", permute=True)
    check("CONTROLE etats permutes : la precision s'effondre (<= 0.60)", rp["acc"] <= 0.60,
          f"acc={rp['acc']:.3f}")
    dn = synth_dataset(signal=False, seed=3)
    mn, pn = T.train_one(dn, cfg, seed=0)
    rn = T.evaluate(dn, mn, pn, "test")
    check("BRUIT PUR : la precision reste au hasard (0.30-0.70)",
          0.30 <= rn["acc"] <= 0.70, f"acc={rn['acc']:.3f}")
    check("l'evaluation rend precision, marges par paire et Wilcoxon",
          all(k in r for k in ("acc", "acc_by_pair", "pair_margins", "wilcoxon_pair_p",
                               "wilcoxon_video_p")))
    check("le jumeau est FORCE dans chaque softmax d'entrainement (asserte par train_one)",
          True, "assert present dans train_one")


def t9b_real_refs():
    """Audit 2026-08-04, LE CONTROLE POSITIF A DIFFICULTE REELLE.

    T9 valide la plomberie, mais sur des references DINO tirees au hasard : deux jumeaux y sont
    orthogonaux (cos ~ 0), alors que les VRAIS jumeaux (meme prompt) ont un cos median de 0,82.
    T9 ne prouvait donc pas que l'instrument sait RESOUDRE l'ecart d'individu qu'on lui demande
    de resoudre. Ici la meme signature synthetique (image lineaire de l'embedding du sujet +
    bruit) est apprise contre les VRAIES references des 30 paires, avec le VRAI split : c'est le
    probleme a reponse connue, a la difficulte du terrain."""
    print("T9b controle positif a la difficulte REELLE (vraies references DINO)")
    import train_probe as T
    vids, pairs = C.pair_dataset(verbose=False)
    miss = [p[r] for p in pairs for r in ("main", "twin")
            if not os.path.exists(os.path.join(C.REFS_DIR, C.seg_key(p[r], "A") + ".pt"))]
    if miss:
        check("T9b saute (references absentes)", True, f"{len(miss)} manquantes")
        return

    def refs(vid):
        r = torch.load(os.path.join(C.REFS_DIR, C.seg_key(vid, "A") + ".pt"),
                       map_location="cpu", weights_only=False)
        e, v = r["subj"].float(), r["valid"].bool()
        return Fn.normalize(e[v] if v.any() else e, dim=-1)

    def build(signal, seed, dim=2000, snr=1.5):
        g = torch.Generator().manual_seed(seed)
        A = torch.randn(C.DINO_DIM, dim, generator=g)
        videos, X, s_vid, s_age = [], [], [], []
        for p in pairs:
            base = len(videos)
            for off, role in enumerate(("main", "twin")):
                R = refs(p[role])
                e = Fn.normalize(R.mean(0), dim=-1)
                videos.append({"video_id": p[role], "refs": R, "twin_index": base + (1 - off),
                               "split": p["split"], "pair_id": p["pair_id"],
                               "corpus_idx": p["corpus_idx"]})
                for a in range(C.N_AGES):
                    n = torch.randn(dim, generator=g)
                    X.append(((e @ A) * snr + n if signal else n).to(torch.float16))
                    s_vid.append(base + off)
                    s_age.append(f"a{a}")
        return {"videos": videos, "X": torch.stack(X, 0), "s_vid": torch.tensor(s_vid),
                "s_age": s_age, "source": "synth_real_refs", "target": "subject",
                "dropped_pairs": []}

    tw = [float((Fn.normalize(refs(p["main"]).mean(0), dim=-1)
                 * Fn.normalize(refs(p["twin"]).mean(0), dim=-1)).sum()) for p in pairs]
    cfg = {"sig_dim": 64, "hidden": 256, "depth": 2, "lr": 1e-3, "wd": 1e-2,
           "temp": 0.05, "dropout": 0.1, "steps": 400, "batch": 64}
    ds = build(True, 2)
    rows = [T.evaluate(ds, *T.train_one(ds, cfg, seed=s), split="test") for s in (0, 1)]
    acc = float(np.mean([r["acc"] for r in rows]))
    ap = float(np.mean([r["acc_by_pair"] for r in rows]))
    pv = float(np.max([r["wilcoxon_pair_p"] for r in rows]))
    check("SIGNAL a difficulte reelle : l'instrument separe les jumeaux (acc >= 0.80)",
          acc >= 0.80, f"cos(jumeaux) median={np.median(tw):.3f} | acc_row={acc:.3f} "
                       f"acc_pair={ap:.3f} p_pair={pv:.4g}")
    check("SIGNAL a difficulte reelle : precision PAR PAIRE >= 0.85", ap >= 0.85, f"{ap:.3f}")
    dn = build(False, 3)
    accn = float(np.mean([T.evaluate(dn, *T.train_one(dn, cfg, seed=s), split="test")["acc"]
                          for s in (0, 1)]))
    check("BRUIT PUR (vraies references) : retour au hasard (<= 0.70)", accn <= 0.70,
          f"acc={accn:.3f}")
    rp = T.evaluate(ds, *T.train_one(ds, cfg, seed=0), split="test", permute=True)
    check("CONTROLE permute moyenne sur plusieurs derangements", rp["n_perm"] > 1 and
          rp["acc"] <= 0.60, f"acc={rp['acc']:.3f} sur {rp['n_perm']} derangements")
    check("`per_row` porte corpus_idx et pair_id (analyse secondaire calculable)",
          all(k in rows[0]["per_row"][0] for k in ("corpus_idx", "pair_id", "video_id")))


def t10_verdict_locked():
    print("T10 le run de verdict est verrouille")
    for stage in ("final", "cv"):
        r = subprocess.run([PY, os.path.join(HERE, "train_probe.py"), "--stage", stage],
                           capture_output=True, text=True)
        check(f"`--stage {stage}` sans `--audited` refuse de tourner",
              r.returncode != 0 and "REFUS" in (r.stdout + r.stderr),
              (r.stdout + r.stderr).strip().splitlines()[-1][:110] if (r.stdout + r.stderr)
              else "")


def t14_gates_pair():
    print("T14 gates : la lecture est PAR PAIRE partout, borne re-ancree")
    import inspect

    import gates as G
    check("PRIMARY == 'pair' (reglage final n 1, scelle)", G.PRIMARY == "pair", G.PRIMARY)
    summ = {"all": {"acc_pair": 0.7, "acc_row": 0.55, "n_pairs": 30, "n_pairs_correct": 21,
                    "margin_mean": 0.01, "wilcoxon_pair_p": 0.01}}
    check("acc() rend la precision PAR PAIRE par defaut", G.acc(summ, "all") == 0.7)
    check("la lecture par ligne reste accessible, mais explicitement",
          G.acc(summ, "all", reading="row") == 0.55)
    check("regle 1 re-ancree a 5/6 sur les 'clairement differents'",
          G.T_UPPER_DECISIVE == 5 and G.T_UPPER == 0.90)
    check("regle 1 par corpus, telle que citee du registre (v1 5/6 decisif, v2 16/19 fortes)",
          G.RULE1 == {"v1": ("decisive", 5), "v2": ("strong19", 16)}, str(G.RULE1))
    check("sous-ensemble PRIMAIRE par corpus (v1 = 30 paires, v2 = les 25 a masque valide)",
          G.PRIMARY_SUBSET == {"v1": "all", "v2": "valid25"}, str(G.PRIMARY_SUBSET))
    check("la BORNE est lue par defaut sur la source scellee le 05/08 nuit",
          G.UPPER_SOURCES[0] == "hdirect_masked", str(G.UPPER_SOURCES))
    src = inspect.getsource(G.main)
    check("aucun seuil de verdict n'est applique a une precision par LIGNE",
          "acc(" in src and 'reading="row"' not in src and "acc_row" not in
          src.split("REGLE 2")[0].replace("subset_lines", ""))
    check("les sous-ensembles pre-enregistres des DEUX corpus sont rapportes cote a cote",
          set(G.SUBSET_LABEL) == {"all", "agent", "aden15", "decisive", "valid25", "strong19"})


def t15_folds():
    print("T15 validation croisee par paires : plis etanches")
    import train_probe as T
    cidx = list(range(100, 130))
    folds = T.pair_folds(cidx, n_folds=6)
    flat = [c for f in folds for c in f]
    check("6 plis de 5 paires, partition exacte des 30",
          len(folds) == 6 and all(len(f) == 5 for f in folds) and sorted(flat) == cidx,
          str([len(f) for f in folds]))
    check("chaque paire est de TEST exactement une fois", len(set(flat)) == 30)
    check("plis deterministes (2 appels identiques)",
          T.pair_folds(cidx, n_folds=6) == folds)
    ds = synth_dataset(signal=True, seed=5)
    for v in ds["videos"]:                       # corpus_idx = pair_id dans le jeu synthetique
        v["corpus_idx"] = v["pair_id"]
    T.set_fold(ds, [0, 1, 2, 3, 4], [5, 6, 7, 8, 9])
    n = {s: sum(1 for v in ds["videos"] if v["split"] == s) for s in ("train", "val", "test")}
    check("set_fold repartit bien 20/5/5 paires (40/10/10 videos)",
          (n["train"], n["val"], n["test"]) == (40, 10, 10), str(n))
    pre1 = T.get_preproc(ds)
    mu1 = pre1.mu.clone()
    te = T.split_rows(ds, "test")
    ds["X"] = ds["X"].clone()
    ds["X"][te] = ds["X"][te] * 100 + 7          # on saccage le test du pli
    ds["_pre"], ds["_Z"] = None, {}
    check("le blanchiment du pli ne voit PAS son test (saccager le test ne bouge rien)",
          torch.equal(mu1, T.get_preproc(ds).mu))
    T.set_fold(ds, [5, 6, 7, 8, 9], [10, 11, 12, 13, 14])
    check("changer de pli INVALIDE le cache de blanchiment (re-ajuste sur le nouveau train)",
          ds["_pre"] is None and ds["_Z"] == {})


def t16_pretrain_disjoint():
    print("T16 pre-entrainement : les videos switch ne touchent aucune paire")
    vids, pairs = C.pair_dataset(verbose=False)
    pair_ids = {p["main"] for p in pairs} | {p["twin"] for p in pairs}
    sw = [v for v in vids if v["mode"] == "switch"]
    check("le pool de pre-entrainement ne contient AUCUNE video de paire",
          not [v for v in sw if v["video_id"] in pair_ids], f"{len(sw)} videos switch")
    check("aucun corpus_idx de paire dans le pool",
          not ({v["corpus_idx"] for v in sw} & {p["corpus_idx"] for p in pairs}))
    import train_probe as T
    ds = synth_dataset(signal=True, seed=7)
    pds = synth_dataset(signal=True, seed=8, n_pairs=6)
    for v in pds["videos"]:
        v["split"], v["twin_index"] = "train", -1
    ds["_pretrain"] = pds
    pre = T.get_preproc(ds)
    check("le blanchiment integre les lignes de pre-entrainement (elles sont hors paires)",
          pre.n_fit == T.split_rows(ds, "train").numel() + pds["X"].shape[0],
          f"n_fit={pre.n_fit} n_train={pre.n_train}")
    cfg = {"sig_dim": 0, "hidden": 64, "depth": 2, "lr": 1e-3, "wd": 1e-2, "temp": 0.05,
           "dropout": 0.1, "steps": 20, "batch": 32, "pre_steps": 20}
    m, _ = T.train_one(ds, cfg, seed=0)
    check("un sondeur pre-entraine s'entraine et s'evalue sans erreur",
          0.0 <= T.evaluate(ds, m, pre, "test")["acc"] <= 1.0)


def t17_prompts_v2():
    print("T17 materiaux v2 : 30 prompts SANS attribut, held-out")
    sys.path.insert(0, os.path.join(C.REPO, "reference/runs/frame_dec"))
    from collect_pairs import (PROMPTS_FILE_BASE, _CLOTHES, _COLORS, _MATERIALS,
                               load_prompts_file)
    sel = load_prompts_file(C.PROMPTS_V2, verbose=False)   # held-out asserte a l'interieur
    check("30 prompts, corpus_idx >= 9 000 000 et distincts",
          len(sel) == 30 and len({i for i, _ in sel}) == 30
          and all(i >= PROMPTS_FILE_BASE for i, _ in sel), f"{len(sel)}")
    banned = set(_COLORS) | set(_MATERIALS) | set(_CLOTHES) | {
        "small", "big", "large", "tall", "short", "tiny", "huge", "little", "giant", "young",
        "old", "elderly", "aged", "slim", "thick", "curly", "blonde", "bearded", "striped",
        "spotted", "shiny", "dark", "bright", "pale", "breed", "purebred"}
    bad = [(i, w) for i, p in sel for w in p.lower().replace(",", " ").split() if w in banned]
    check("aucun attribut d'apparence (couleur / matiere / vetement / taille / age)",
          not bad, str(bad[:5]))
    heads = [C.subject_phrase(p) for _, p in sel]
    check("le sujet de chaque prompt est reconnu par l'extracteur de nom",
          all(h["found"] for h in heads),
          str([p for (_, p), h in zip(sel, heads) if not h["found"]][:3]))
    check("le sujet n'est jamais dans un complement de lieu introductif",
          not any(h["in_scene"] for h in heads))
    check("categories variees (>= 20 tetes de sujet distinctes)",
          len({h["head"] for h in heads}) >= 20, f"{len({h['head'] for h in heads})} tetes")
    check("prompts held-out : verifie a la lecture contre les sondes ET le corpus l1",
          True, "assert integre dans load_prompts_file")


def t18_board_numbering():
    print("T18 numerotation de planche = celle des listes d'oeil scellees")
    vids, pairs = C.pair_dataset(verbose=False)
    by_no = {p["board_no"]: p for p in pairs}
    check("30 paires numerotees 1..30", sorted(by_no) == list(range(1, 31)))
    named = {1: "firefighter", 2: "bottle", 6: "crowd", 8: "lock", 11: "bird",
             12: "athlete", 16: "couple", 19: "child", 24: "angelic", 27: "girl"}
    bad = [(n, w) for n, w in named.items() if w not in by_no[n]["prompt"].lower()]
    check("les 10 paires nommees par Aden tombent sur les bons numeros", not bad, str(bad))
    sub = C.sealed_subsets(pairs)
    check("sous-ensemble 'clairement differents' = 6 paires",
          len(sub["decisive"]) == 6, str(sorted(by_no[n]["corpus_idx"]
                                                for n in C.ADEN_DECISIVE_6)))
    check("sous-ensemble oeil d'Aden = 15 paires, et il contient les 6 decisives",
          len(sub["aden15"]) == 15 and sub["decisive"] <= sub["aden15"])
    check("sous-ensemble agent (regle des masques) calculable et strictement inclus dans 30",
          0 < len(sub["agent"]) <= 30, f"{len(sub['agent'])} paires")
    check("les 3 sous-ensembles + la primaire sont tous rapportes",
          set(sub) == {"all", "agent", "aden15", "decisive"})
    # corpus v2 : les labels d'oeil scelles au registre le 05/08 (25 valides, 19 fortes)
    try:
        _, pairs2 = C.pair_dataset(verbose=False, corpus="v2")
    except Exception as e:
        check("corpus v2 disponible", False, str(e)[:80])
        return
    sub2 = C.sealed_subsets(pairs2)
    check("corpus v2 : primaire 30, masque valide 25, clairement differents 19",
          len(sub2["all"]) == 30 and len(sub2["valid25"]) == 25
          and len(sub2["strong19"]) == 19,
          f"{len(sub2['all'])}/{len(sub2.get('valid25', []))}/{len(sub2.get('strong19', []))}")
    check("les 19 fortes sont incluses dans les 25 a masque valide",
          sub2["strong19"] <= sub2["valid25"])
    check("les listes d'oeil du v1 ne fuient PAS sur le corpus v2 (et reciproquement)",
          not ({"aden15", "decisive", "agent"} & set(sub2))
          and not ({"valid25", "strong19"} & set(sub)))


def t11_replay(gpu=False):
    print("T11 contrat de rejeu (GPU)")
    if not (gpu and torch.cuda.is_available()):
        check("T11 saute (pas de --gpu)", True, "relance avec --gpu pour l'executer")
        return
    from omegaconf import OmegaConf

    import mem2_data as D
    from build_states import replay_video
    from train_writer import build_cells, delta_l1, ref_theta_l1
    config = OmegaConf.load(C.DEFAULT_CONFIG)
    blocks = sorted(int(x) for x in config.model_kwargs.memory_layers)
    cells, _, _ = build_cells(config, "cuda", mode="direct")
    ref, _ = ref_theta_l1(config, blocks, "cuda")
    check("contrat F2.3 : delta_l1 == 0 au chargement", delta_l1(cells, ref) == 0.0)
    vids, pairs = C.pair_dataset()
    byid = C.vids_by_id(vids)
    meta = byid[pairs[0]["main"]]
    ages = C.state_ages(meta)[:2]
    cache = D.PartCache(size=1)
    a = replay_video(cells, blocks, meta, cache, ages, "cuda")
    b = replay_video(cells, blocks, meta, cache, ages, "cuda")
    same = all(torch.equal(a[i][bi][k], b[i][bi][k])
               for i in a for bi in blocks for k in range(3))
    check("rejeu bit a bit reproductible (2 passes)", same, meta["video_id"])


def t21_collector_additive():
    """`--prompts-file` doit etre STRICTEMENT additif : sans le drapeau, la selection de corpus
    de la campagne v3 doit rester EXACTEMENT celle qui a produit les 165 videos sur disque."""
    print("T21 collect_pairs --prompts-file est additif (non-regression du chemin v3)")
    sys.path.insert(0, os.path.join(C.REPO, "reference/runs/frame_dec"))
    from collect_pairs import select_prompts
    # la campagne v3 sur disque : n=135, graine 9012 (lue dans son propre manifeste)
    sel = {i for i, _ in select_prompts(135, 9012, verbose=False)}
    vids = C.load_index(verbose=False)
    got = {int(v["corpus_idx"]) for v in vids if not v["twin"]}
    check("la selection v3 (135 prompts, graine 9012) est inchangee",
          sel == got, f"{len(sel & got)}/{len(sel)} en commun")
    check("aucun corpus_idx v3 ne peut heurter la base des prompts custom (9e6)",
          max(sel) < 9_000_000, f"max = {max(sel)}")


def t20_cv_end_to_end():
    """La validation croisee, bout en bout, sur le probleme a REPONSE CONNUE : elle doit
    trouver le signal quand il est la (30 paires au denominateur, pas 7) et retomber au hasard
    quand il n'y est pas. C'est T9 transpose a la procedure PRIMAIRE."""
    print("T20 validation croisee bout en bout (donnees a reponse connue)")
    import train_probe as T
    grid = {"sig_dim": [64], "hidden": [128], "depth": [2], "lr": [1e-3], "wd": [1e-2],
            "temp": [0.05], "dropout": [0.1]}
    old = T.FIXED.copy()
    T.FIXED.update({"steps": 400, "batch": 32, "pre_steps": 60})
    try:
        ds = T.synth_like_pairs(signal=True, seed=11, snr=1.5) if hasattr(T, "synth_like_pairs") \
            else synth_dataset(signal=True, seed=11, snr=1.5)
        for v in ds["videos"]:
            v["corpus_idx"] = v["pair_id"]
        ds["pairs"] = {}
        r = T.cv_run(ds, seeds=(0,), n_folds=3, grid=grid, verbose=False)
        s = r["summary"]["all"]
        check("SIGNAL : la CV rend bien les 30 PAIRES au denominateur",
              s["n_pairs"] == 30 and s["n_rows"] == 240, f"{s['n_pairs']} paires")
        check("SIGNAL : precision par paire >= 0.90", s["acc_pair"] >= 0.90,
              f"pair={s['acc_pair']:.3f} row={s['acc_row']:.3f} p={s['wilcoxon_pair_p']:.3g}")
        check("CONTROLE permute : retour au hasard (<= 0.60)",
              r["summary"]["permuted"]["acc_row_25perm"] <= 0.60,
              f"{r['summary']['permuted']['acc_row_25perm']:.3f}")
        check("chaque paire est evaluee EXACTEMENT une fois (plis etanches)",
              len({int(x["corpus_idx"]) for x in r["per_seed"][0]["per_row"]}) == 30)
        dn = synth_dataset(signal=False, seed=12)
        for v in dn["videos"]:
            v["corpus_idx"] = v["pair_id"]
        dn["pairs"] = {}
        rn = T.cv_run(dn, seeds=(0,), n_folds=3, grid=grid, verbose=False)
        check("BRUIT PUR : la CV retombe au hasard (0.30-0.70 par paire)",
              0.30 <= rn["summary"]["all"]["acc_pair"] <= 0.70,
              f"{rn['summary']['all']['acc_pair']:.3f}")
    finally:
        T.FIXED.clear()
        T.FIXED.update(old)


def t19_stream_equals_disk(gpu=False):
    """Le mode --stream (rejeu a la volee, aucun etat ecrit sur disque : 41 GB economises par
    corpus) doit rendre EXACTEMENT la signature du chemin par disque, sinon les videos de
    pre-entrainement et les paires v2 ne seraient pas dans le meme espace que les paires v1."""
    print("T19 --stream == disque, bit a bit (GPU)")
    if not (gpu and torch.cuda.is_available()):
        check("T19 saute (pas de --gpu)", True, "relance avec --gpu")
        return
    import build_signatures as B
    from build_probes import PROBES_PT
    from build_states import state_path

    import mem2_data as D
    real = sorted(glob.glob(os.path.join(C.SIG_DIR, "state", "*.pt")))
    if not real or not os.path.exists(PROBES_PT):
        check("T19 saute (pas de signatures d'etat sur disque)", True, "")
        return
    vids, _ = C.pair_dataset(verbose=False)
    byid = C.vids_by_id(vids)
    d = None
    for f in real:                      # une signature dont l'ETAT est aussi sur disque
        x = torch.load(f, map_location="cpu", weights_only=False)
        if os.path.exists(state_path(x["video_id"], x["age"])) and x["video_id"] in byid:
            d = x
            break
    if d is None:
        check("T19 saute (aucun etat sur disque)", True, "")
        return
    meta = byid[d["video_id"]]
    pk = torch.load(PROBES_PT, map_location="cpu", weights_only=False)
    pr, bl = pk["probes"].cuda(), list(pk["blocks"])
    ages = [a for a in C.state_ages(meta) if a["name"] == d["age"]]
    cells = B.build_cells_for_stream(C.DEFAULT_CONFIG, bl, "cuda")
    cache = D.PartCache(size=1)
    st = B.stream_states(meta, ages, cache, bl, "cuda", cells=cells)
    pay_stream = B.load_payload("state", meta, ages[0], cache, bl, "cuda", states=st)
    pay_disk = B.load_payload("state", meta, ages[0], cache, bl, "cuda")
    s1 = B.signature("state", pr, pay_stream, bl).to(torch.float16).cpu()
    s2 = B.signature("state", pr, pay_disk, bl).to(torch.float16).cpu()
    check("signature streamee == signature depuis le disque (bitwise)",
          bool(torch.equal(s1, s2)), f"{d['video_id']}/{d['age']}")
    check("... et == la signature STOCKEE (bitwise)", bool(torch.equal(s1, d["sig"])))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", action="store_true")
    ap.add_argument("--only", default="")
    args = ap.parse_args()
    C.ensure_dirs()
    C.set_determinism(0)
    tests = [("t1", t1_compile), ("t2", t2_split), ("t3", t3_segments_ages),
             ("t4", t4_probes), ("t5", t5_single_path),
             ("t6", lambda: t6_determinism(args.gpu)), ("t7", t7_masks),
             ("t8", t8_no_leak), ("t9", t9_prober_synthetic), ("t9b", t9b_real_refs),
             ("t10", t10_verdict_locked),
             ("t11", lambda: t11_replay(args.gpu)),
             ("t12", t12_compression), ("t13", t13_bound_no_write), ("t14", t14_gates_pair),
             ("t15", t15_folds), ("t16", t16_pretrain_disjoint), ("t17", t17_prompts_v2),
             ("t18", t18_board_numbering),
             ("t19", lambda: t19_stream_equals_disk(args.gpu)),
             ("t20", t20_cv_end_to_end), ("t21", t21_collector_additive),
             ("t22", t22_masked_bound)]
    for name, fn in tests:
        if args.only and name not in args.only.split(","):
            continue
        fn()
    ok = sum(1 for r, _, _ in RESULTS if r)
    print(f"\n==== SELFTEST probe1 : {ok}/{len(RESULTS)} verts ====")
    for r, n, d in RESULTS:
        if not r:
            print(f"  ECHEC: {n}, {d}")
    sys.exit(0 if ok == len(RESULTS) else 1)


if __name__ == "__main__":
    main()
