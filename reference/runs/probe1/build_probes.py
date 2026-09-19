#!/usr/bin/env python
"""ETAPE 1, la BATTERIE de sondes (fixe, ecrite une fois, jamais re-tiree).

"Lecture de l'etat : FONCTIONNELLE, pas directe : on passe une batterie fixe d'entrees-sondes
 a travers le module a poids rapides et on recueille ses sorties."
Renfort (b) du 04/08 : "sondes ALIGNEES sur les donnees (centroides k-means des activations
 du corpus + sous-echantillon des requetes apprises du decodeur), pas de vecteurs aleatoires
 (recouvrement quasi nul avec les directions sujet en dim 1536)."

La batterie est par bloc (5 / 16 / 27) : les etats de blocs differents vivent dans des espaces
de features differents, une sonde unique ne parlerait pas les trois. Chaque bloc recoit :
  - 192 centroides k-means des activations h de ce bloc ;
  - 64 requetes du decodeur de ce bloc, sous-echantillonnees sur la grille 30x52 (8x8
    positions equidistantes) et prises APRES le q_norm du decodeur, c'est-a-dire exactement
    dans l'etat ou le decodeur les envoie a swiglu_apply.
Toutes les sondes sont RMS-normalisees a la fin : la batterie est homogene en echelle, ce qui
rend le blanchiment par sonde (build_signatures) interpretable.

Anti-FUITE : le k-means est ajuste UNIQUEMENT sur les videos `switch` du corpus (105), qui
n'appartiennent a AUCUNE paire, donc ni au train, ni au val, ni au test du sondeur. La
batterie ne peut pas avoir vu une video de test. (--kmeans_source train restreint en plus aux
paires d'entrainement, si l'auditeur veut la version la plus stricte.)

Usage :
  CUDA_VISIBLE_DEVICES=7 python reference/runs/probe1/build_probes.py
"""
import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import probe1_common as C  # noqa: E402

import torch  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402

import mem2_data as D  # noqa: E402

PROBES_PT = os.path.join(C.SIG_DIR, "probes.pt")


def rms_normalize(x, eps=1e-6):
    """p <- p / rms(p) : ||p||_2 = sqrt(dim). Meme convention que les RMSNorm du modele."""
    return x / x.pow(2).mean(dim=-1, keepdim=True).clamp_min(eps).sqrt()


# ---------------------------------------------------------------------------------------------
# 192 centroides k-means des activations h
# ---------------------------------------------------------------------------------------------
def sample_h_tokens(vids, blocks, n_videos, n_writes, n_tokens, device, seed=0):
    """Echantillon deterministe de tokens h, par bloc. Aucun RNG global n'est consomme :
    les indices sont tires d'un generateur local seede."""
    g = torch.Generator().manual_seed(seed)
    order = sorted(vids, key=lambda v: v["video_id"])
    pick = order if len(order) <= n_videos else [
        order[int(i)] for i in torch.linspace(0, len(order) - 1, n_videos).round().long()]
    pick.sort(key=lambda v: (v["part"], v["vi"]))
    cache = D.PartCache(size=1)
    out = {b: [] for b in blocks}
    for v in pick:
        n = len(v["write_chunks"])
        idxs = sorted({int(round(i * (n - 1) / max(1, n_writes - 1)))
                       for i in range(min(n_writes, n))})
        rec = D.get_record(v, cache)
        for b in blocks:
            seq = rec["h_seq"][str(b)]
            for i in idxs:
                t = seq[i]
                t = t[0] if t.dim() == 3 else t                 # (Lw, dim)
                sel = torch.randperm(t.shape[0], generator=g)[:n_tokens]
                out[b].append(t[sel].to(torch.float32))
        del rec
    return {b: torch.cat(v, 0).to(device) for b, v in out.items()}


def kmeans(x, k, iters=60, seed=0, device="cuda"):
    """k-means++ (init deterministe a graine fixe) + Lloyd. X: (N, dim) fp32."""
    g = torch.Generator(device=x.device).manual_seed(seed)
    N = x.shape[0]
    cent = x[torch.randint(N, (1,), generator=g, device=x.device)]
    d2 = (x - cent[0]).pow(2).sum(1)
    for _ in range(k - 1):
        p = d2.clamp_min(0)
        p = p / p.sum().clamp_min(1e-12)
        j = torch.multinomial(p, 1, generator=g)
        cent = torch.cat([cent, x[j]], 0)
        d2 = torch.minimum(d2, (x - x[j]).pow(2).sum(1))
    for _ in range(iters):
        a = torch.cdist(x, cent).argmin(1)
        new = torch.zeros_like(cent)
        cnt = torch.zeros(k, device=x.device)
        new.index_add_(0, a, x)
        cnt.index_add_(0, a, torch.ones(N, device=x.device))
        empty = cnt == 0
        cent = torch.where(empty[:, None], cent, new / cnt.clamp_min(1)[:, None])
    return cent


# ---------------------------------------------------------------------------------------------
# 64 requetes du decodeur
# ---------------------------------------------------------------------------------------------
def decoder_queries(dec_ckpt, n_queries, device):
    """Les requetes apprises du decodeur, sous-echantillonnees. On charge le decodeur par le
    chemin exact du harness (`_memdec_get_decoder`, aucune recopie) et on prend
    q_norms[b](queries[b] + pos) : la requete telle qu'elle part vers swiglu_apply."""
    from pipeline.switch_causal_inference import _memdec_get_decoder
    dec = _memdec_get_decoder(dec_ckpt, torch.device(device))
    gh, gw, dim = dec.grid_h, dec.grid_w, dec.dim
    side = int(round(n_queries ** 0.5))
    assert side * side == n_queries, f"n_queries {n_queries} doit etre un carre parfait"
    rows = torch.linspace(0, gh - 1, side).round().long()
    cols = torch.linspace(0, gw - 1, side).round().long()
    sel = (rows[:, None] * gw + cols[None, :]).reshape(-1)          # (n_queries,)
    pos = (dec.pos_row[:, None, :] + dec.pos_col[None, :, :]).reshape(gh * gw, dim)
    out = []
    with torch.no_grad():
        for b in range(dec.num_blocks):
            q = dec.q_norms[b](dec.queries[b] + pos)                 # (1560, dim)
            out.append(q[sel.to(q.device)].float().cpu())
    return torch.stack(out, 0), {"grid": [int(gh), int(gw)], "side": side,
                                 "sel": [int(x) for x in sel]}


# ---------------------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=C.DEFAULT_CONFIG)
    ap.add_argument("--dec_ckpt", default=C.DEC_V12,
                    help=f"decodeur dont on sous-echantillonne les requetes "
                         f"(defaut v1.2 = celui de memdec20 ; {C.DEC_A2} pour A2)")
    ap.add_argument("--n_centroids", type=int, default=C.N_CENTROIDS)
    ap.add_argument("--n_queries", type=int, default=C.N_DEC_QUERIES)
    ap.add_argument("--kmeans_source", default="switch", choices=("switch", "train", "all"))
    ap.add_argument("--km_videos", type=int, default=60)
    ap.add_argument("--km_writes", type=int, default=8)
    ap.add_argument("--km_tokens", type=int, default=128)
    ap.add_argument("--km_iters", type=int, default=60)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=PROBES_PT)
    args = ap.parse_args()

    C.ensure_dirs()
    C.set_determinism(args.seed)
    device = "cuda"
    config = OmegaConf.load(args.config)
    blocks = sorted(int(x) for x in config.model_kwargs.memory_layers)
    vids, pairs = C.pair_dataset(verbose=True)

    if args.kmeans_source == "switch":
        pool = [v for v in vids if v["mode"] == "switch"]
    elif args.kmeans_source == "train":
        keep = {p["main"] for p in pairs if p["split"] == "train"} | \
               {p["twin"] for p in pairs if p["split"] == "train"}
        pool = [v for v in vids if v["video_id"] in keep]
    else:
        pool = list(vids)
    test_ids = {p["main"] for p in pairs if p["split"] == "test"} | \
               {p["twin"] for p in pairs if p["split"] == "test"}
    val_ids = {p["main"] for p in pairs if p["split"] == "val"} | \
              {p["twin"] for p in pairs if p["split"] == "val"}
    leak = [v["video_id"] for v in pool if v["video_id"] in (test_ids | val_ids)]
    assert not leak, f"FUITE : le k-means verrait {len(leak)} videos val/test : {leak[:5]}"
    print(f"[probes] k-means sur {len(pool)} videos ({args.kmeans_source}), "
          f"0 video val/test", flush=True)

    toks = sample_h_tokens(pool, blocks, args.km_videos, args.km_writes, args.km_tokens,
                           device, seed=args.seed)
    cents = []
    for bi in blocks:
        x = toks[bi]
        c = kmeans(x, args.n_centroids, iters=args.km_iters, seed=args.seed, device=device)
        cents.append(c.cpu())
        print(f"[probes] bloc {bi}: {tuple(x.shape)} tokens -> {tuple(c.shape)} centroides",
              flush=True)
    cents = torch.stack(cents, 0)                                   # (nb, 192, dim)
    del toks
    torch.cuda.empty_cache()

    qs, qmeta = decoder_queries(args.dec_ckpt, args.n_queries, device)   # (nb, 64, dim)
    assert qs.shape[0] == cents.shape[0], "nb de blocs decodeur != nb de blocs memoire"

    probes = rms_normalize(torch.cat([cents, qs], dim=1).float())       # (nb, 256, dim)
    assert probes.shape[1] == args.n_centroids + args.n_queries
    assert torch.isfinite(probes).all(), "sondes non finies"

    torch.save({"probes": probes, "blocks": blocks,
                "n_centroids": args.n_centroids, "n_queries": args.n_queries,
                "kmeans_source": args.kmeans_source, "n_pool_videos": len(pool),
                "dec_ckpt": os.path.abspath(args.dec_ckpt), "dec_meta": qmeta,
                "seed": args.seed, "config": os.path.abspath(args.config)}, args.out)
    print(f"[probes] batterie {tuple(probes.shape)} -> {args.out}")


if __name__ == "__main__":
    main()
