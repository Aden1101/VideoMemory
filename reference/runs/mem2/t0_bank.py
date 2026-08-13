#!/usr/bin/env python
# ============================================================================================
# MEMOIRE 2.1 -- la BANQUE de references du terme T0 cote runtime, et le forcage de la position
# de RETOUR.  Ce module n'importe QUE torch + mem2_data : il est testable sur CPU sans GPU
# (importer `train_writer` ou `wan` initialise un contexte CUDA, cf. wan/modules/t5.py).
# C'est la seule raison de son existence : garder les deux briques les plus faciles a se
# tromper (quels negatifs ? quel etat ?) sous des selftests qui tournent partout.
# ============================================================================================
import os

import torch

import mem2_data as D


class T0Bank:
    """Les references PRE-CALCULEES du terme T0 (build_bank_t0.py), residentes sur `device`.

    Par video switch : E_a (positifs = frames du segment A) et E_b (negatifs DURS = l'interlude).
    Le POOL "autres videos" est materialise UNE fois par cote du split (concatenation des A et
    des B de toutes les videos de ce cote) avec, pour chaque ligne, l'index de la video dont
    elle vient -- masquer la video courante coute alors un masque booleen et rien d'autre.
    Le JUMEAU est dans le pool par construction (autre video, meme prompt, graine +100000 : le
    negatif dur valide empiriquement au J2, 93/145 sondes) et sa correlation est loggee a part."""

    KEY = {("latent", "chan"): "lat", ("latent", "gray"): "latg", ("pixel", "chan"): "px",
           ("pixel", "gray"): "px"}

    def __init__(self, path, device, space="latent", lat_mode="chan", verbose=True):
        d = torch.load(path, map_location="cpu", weights_only=False)
        assert d.get("format") == "mem2_bank_t0_v1", f"{path}: mauvais format {d.get('format')!r}"
        if space == "pixel":
            assert d["with_pixel"], \
                (f"{path} a ete construit SANS le cote pixel : relancer build_bank_t0.py "
                 f"--with_pixel (le cote reference n'a pas de gradient, il se pre-calcule)")
        key = self.KEY[(space, lat_mode)]
        self.space, self.lat_mode, self.n = space, lat_mode, int(d["n"])
        self.rows = d["rows"]
        self.by_id = {r["video_id"]: i for i, r in enumerate(self.rows)}
        self.E_a = [r[f"{key}_a"].to(device=device, dtype=torch.float32) for r in self.rows]
        self.E_b = [r[f"{key}_b"].to(device=device, dtype=torch.float32) for r in self.rows]
        self.dim = self.E_a[0].shape[1]
        self.device = device
        self.pool, self.pool_vid = {}, {}
        for side in ("train", "val"):
            mats, owner = [], []
            for i, r in enumerate(self.rows):
                if r["split"] != side:
                    continue
                mats += [self.E_a[i], self.E_b[i]]
                owner += [i] * (self.E_a[i].shape[0] + self.E_b[i].shape[0])
            assert mats, f"T0Bank: aucune video du cote {side}"
            self.pool[side] = torch.cat(mats, 0)
            self.pool_vid[side] = torch.tensor(owner, device=device)
        self.twin = {}
        by_c = {}
        for i, r in enumerate(self.rows):
            by_c.setdefault(r["corpus_idx"], []).append(i)
        for idxs in by_c.values():
            if len(idxs) == 2:
                self.twin[idxs[0]], self.twin[idxs[1]] = idxs[1], idxs[0]
        if verbose:
            print(f"[T0] bank {os.path.basename(path)}: {len(self.rows)} videos switch, "
                  f"espace {space}/{lat_mode} n={self.n} dim {self.dim}, pool "
                  f"{self.pool['train'].shape[0]} train / {self.pool['val'].shape[0]} val, "
                  f"{len(self.twin) // 2} paires de jumeaux", flush=True)

    def has(self, meta):
        return meta["video_id"] in self.by_id

    def refs(self, meta, side):
        """(E_a, E_b, E_o, E_twin|None). E_o EXCLUT la video courante (ses frames sont deja les
        positifs et les negatifs durs) mais GARDE son jumeau -- c'est tout l'interet."""
        i = self.by_id[meta["video_id"]]
        assert self.rows[i]["split"] == side, \
            f"{meta['video_id']} est du cote {self.rows[i]['split']}, pas {side} (split incoherent)"
        keep = self.pool_vid[side] != i
        E_o = self.pool[side][keep]
        j = self.twin.get(i)
        E_tw = torch.cat([self.E_a[j], self.E_b[j]], 0) if j is not None else None
        return self.E_a[i], self.E_b[i], E_o, E_tw


def force_end_pos(meta, pos, enabled=True):
    """La position de RETOUR la plus profonde du corpus = l'ETAT DE FIN DE SEQUENCE.
    On force la PREMIERE position 'anchor' a cet etat ; les autres restent tirees au sort parmi
    les etats post-switch (le "et 2-3 positions post-switch echantillonnees" du mandat).
    AUCUNE position n'est ajoutee : K, la VRAM et le temps par pas sont inchanges.
    `enabled=False` est un NO-OP strict : le tirage de la recette A2 est reproduit a l'identique."""
    if not enabled:
        return pos
    aj = D.anchor_write_index(meta)
    elig = D.anchor_eligible_states(meta)
    if aj is None or not elig:
        return pos
    i_end = max(elig)
    for p in pos:
        if p["kind"] != "anchor":
            continue
        p["i"] = int(i_end)
        p["age"] = int(i_end - aj + 1)
        assert p["j"] == aj and p["frame"] == D.frame_of_write(meta, aj) and p["cond"] == "a", \
            f"position anchor malformee: {p}"
        break
    return pos
