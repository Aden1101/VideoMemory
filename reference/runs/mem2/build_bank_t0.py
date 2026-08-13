#!/usr/bin/env python
# ============================================================================================
# MEMOIRE 2.1 -- LA BANQUE DE REFERENCES DU TERME T0 (positifs A / negatifs B / negatifs autres).
#
# LE POINT D'INGENIERIE QUI REND LE TERME PIXEL ABORDABLE : le cote REFERENCE de la perte T0
# n'a JAMAIS besoin de gradient (ce sont des frames collectees, des donnees). On peut donc le
# pre-calculer UNE fois, hors ligne, dans les DEUX espaces (latent gratuit + pixel VAE), et ne
# payer un decode VAE differentiable QUE sur l'ancre (1 frame par position) pendant le train.
# C'est exactement l'economie de la phase B (10.9 GB mesures) appliquee a T0.
#
# CONTENU (par video du corpus, switch UNIQUEMENT -- une video single n'a pas de segment B) :
#   A = n_a frames latentes reparties uniformement sur le segment A (frames < switch_frame)
#   B = n_b frames latentes reparties uniformement sur le segment B (frames >= switch_frame)
# et pour chacune son embedding bande-basse (t0_metric.t0_embed) en latent ET (option) en pixel.
# Le POOL de negatifs "autres videos" n'est pas stocke separement : il est assemble au vol par
# le trainer comme la concatenation des A et B de TOUTES les autres videos du bon cote du split
# (le JUMEAU y est par construction -- meme prompt, graine +100000, le negatif dur valide
# empiriquement : 93/145 sondes du J2 -- et il est loggue a part).
#
# ECHANTILLONNAGE DES REFERENCES : uniforme sur le segment, DETERMINISTE (linspace), pour que
# la banque soit reproductible et que le gate ne depende pas d'un tirage. Le segment A exclut
# les `--skip_head` premieres frames (l'ouvreur X pollue tout le rollout : 16.1 quater le mesure
# a +0.344, et on ne veut pas que le "positif" soit en partie l'ouvreur).
#
# Usage (pre-check GPU dans la MEME commande, regle maison ; --space latent ne touche pas le GPU) :
#   CUDA_DEVICE_ORDER=PCI_BUS_ID nvidia-smi --query-gpu=index,memory.free --format=csv && \
#   CUDA_VISIBLE_DEVICES=1 python reference/runs/mem2/build_bank_t0.py --with_pixel
#   ... --selftest_only          # re-verifie une banque existante (CPU)
# ============================================================================================
import argparse
import os
import sys
import time

REPO = "/shared/ssd_14T/home/adenelmi/Anchor-Forcing"
PROBE_DIR = os.path.join(REPO, "reference/runs/reid_probe_l1")
FRAME_DEC = os.path.join(REPO, "reference/runs/frame_dec")
HERE = os.path.dirname(os.path.abspath(__file__))
for p in (REPO, PROBE_DIR, FRAME_DEC, HERE):
    sys.path.insert(0, p)
os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")

import torch  # noqa: E402

import mem2_data as D  # noqa: E402
import t0_metric as T0  # noqa: E402
from build_bank import BANK_DIR, check, _FAILS  # noqa: E402

BANK_T0 = os.path.join(BANK_DIR, "bank_t0.pt")


# ----------------------------- selection des frames de reference ------------------------------
def ref_frames(meta, n_a=8, n_b=8, skip_head=8, min_gap=2):
    """(frames_A, frames_B) : indices de frames latentes, deterministes, uniformement repartis.
      A = [skip_head, switch_frame - min_gap)   -- on saute l'ouvreur (X pollue tout, 16.1 quater)
                                                   et les toutes dernieres frames de A (elles sont
                                                   la CIBLE de reconstruction : les garder comme
                                                   positifs ferait du terme T0 une copie du terme
                                                   de reco, pas une pression d'INSTANCE)
      B = [switch_frame + min_gap, n_frames)    -- l'interlude, le negatif dur
    Renvoie (None, None) pour une video sans switch."""
    sf = meta["switch_frame"]
    if sf is None:
        return None, None
    n_lat = meta["frames"]
    lo_a, hi_a = skip_head, sf - min_gap
    lo_b, hi_b = sf + min_gap, n_lat
    if hi_a - lo_a < n_a or hi_b - lo_b < n_b:
        return None, None
    fa = torch.linspace(lo_a, hi_a - 1, n_a).round().long().unique().tolist()
    fb = torch.linspace(lo_b, hi_b - 1, n_b).round().long().unique().tolist()
    return fa, fb


# ----------------------------- build ------------------------------------------------------------
def build(args):
    vids = D.load_index(verbose=True)
    split = D.assign_split(vids, args.n_val_switch, args.n_val_single)
    cache = D.PartCache(size=2)
    vae = None
    if args.with_pixel:
        os.chdir(REPO)                              # WanVAEWrapper lit des chemins relatifs
        from train_decoder import build_vae
        vae = build_vae(torch.device(args.device))

    rows, t_start = [], time.time()
    n_skip = 0
    for k, meta in enumerate(vids):
        fa, fb = ref_frames(meta, args.n_a, args.n_b, args.skip_head, args.min_gap)
        if fa is None:
            n_skip += 1
            continue
        n_lat = int(D.get_record(meta, cache)["latents"].shape[0])
        assert n_lat == meta["frames"], \
            (f"{meta['video_id']}: meta['frames']={meta['frames']} != latents {n_lat} -- la "
             f"convention de frame (LATENTE) doit tenir, sinon les segments A/B sont faux")
        lat_a = D.load_frames(meta, cache, fa)              # (n_a, 16, 60, 104) float
        lat_b = D.load_frames(meta, cache, fb)
        rec = {"video_id": meta["video_id"], "corpus_idx": meta["corpus_idx"],
               "twin": meta["twin"], "mode": meta["mode"],
               "split": split[meta["video_id"]], "frames_a": fa, "frames_b": fb,
               "switch_frame": meta["switch_frame"]}
        for tag, lat in (("a", lat_a), ("b", lat_b)):
            rec[f"lat_{tag}"] = T0.lowband_lat(lat, n=args.n, mode="chan").to(torch.float32)
            rec[f"latg_{tag}"] = T0.lowband_lat(lat, n=args.n, mode="gray").to(torch.float32)
            if vae is not None:
                with torch.no_grad():
                    from build_bank_dino import latent_to_pixels
                    px = []
                    for i in range(0, lat.shape[0], args.batch):
                        z = lat[i:i + args.batch].to(device=args.device, dtype=torch.bfloat16)
                        px.append(latent_to_pixels(vae, z).clamp(0, 1).float().cpu())
                    px = torch.cat(px, 0)
                rec[f"px_{tag}"] = T0.lowband_px(px, n=args.n).to(torch.float32)
        rows.append(rec)
        if k % 20 == 0:
            print(f"[t0bank] {k + 1}/{len(vids)} videos ({len(rows)} retenues, {n_skip} sautees) "
                  f"{time.time() - t_start:.0f} s", flush=True)
    payload = {
        "format": "mem2_bank_t0_v1",
        "n": args.n, "n_a": args.n_a, "n_b": args.n_b, "skip_head": args.skip_head,
        "min_gap": args.min_gap, "with_pixel": bool(vae is not None),
        "rows": rows,
        "meta": {"n_videos": len(rows), "n_skipped": n_skip,
                 "n_val_switch": args.n_val_switch, "n_val_single": args.n_val_single,
                 "created": time.strftime("%Y-%m-%d %H:%M:%S"),
                 "dim_lat_chan": T0.t0_dim("latent", args.n, "chan"),
                 "dim_lat_gray": T0.t0_dim("latent", args.n, "gray"),
                 "dim_px": T0.t0_dim("pixel", args.n)},
    }
    os.makedirs(BANK_DIR, exist_ok=True)
    tmp = BANK_T0 + ".tmp"
    torch.save(payload, tmp)
    os.replace(tmp, BANK_T0)
    print(f"[t0bank] -> {BANK_T0} ({os.path.getsize(BANK_T0) / 1e6:.1f} MB, {len(rows)} videos, "
          f"{(time.time() - t_start) / 60:.1f} min)", flush=True)
    if vae is not None:
        del vae
        torch.cuda.empty_cache()
    return payload


# ----------------------------- selftests (CPU) ---------------------------------------------------
def selftests(payload=None):
    d = payload if payload is not None else torch.load(BANK_T0, map_location="cpu",
                                                       weights_only=False)
    rows = d["rows"]
    check("banque non vide", len(rows) > 0, f"{len(rows)} videos")
    check("uniquement des videos switch (une single n'a pas d'interlude)",
          all(r["mode"] == "switch" for r in rows),
          f"{sum(1 for r in rows if r['mode'] != 'switch')} non-switch")
    keys = ["lat_a", "lat_b", "latg_a", "latg_b"] + (["px_a", "px_b"] if d["with_pixel"] else [])
    ok_unit, ok_fin, ok_center = True, True, True
    for r in rows:
        for k in keys:
            e = r[k]
            ok_unit &= bool(((e.norm(dim=1) - 1).abs() < 2e-3).all())
            ok_fin &= bool(torch.isfinite(e).all())
            ok_center &= bool((e.mean(dim=1).abs() < 1e-4).all())
    check("embeddings unitaires", ok_unit)
    check("embeddings finis", ok_fin)
    check("embeddings centres (la correlation EST un produit scalaire)", ok_center)
    check("aucune frame de reference A ne touche le segment B",
          all(max(r["frames_a"]) < r["switch_frame"] for r in rows))
    check("aucune frame de reference B ne touche le segment A",
          all(min(r["frames_b"]) >= r["switch_frame"] for r in rows))
    check("l'ouvreur est saute au debut de A (skip_head)",
          all(min(r["frames_a"]) >= d["skip_head"] for r in rows))
    check("les dernieres frames de A (la CIBLE de reco) ne sont pas des positifs T0",
          all(max(r["frames_a"]) <= r["switch_frame"] - 1 - d["min_gap"] for r in rows),
          f"min_gap = {d['min_gap']}")
    # les jumeaux restent du meme cote du split (F1.1b) et sont donc des negatifs LEGITIMES
    by_c = {}
    for r in rows:
        by_c.setdefault(r["corpus_idx"], []).append(r)
    pairs = [v for v in by_c.values() if len(v) == 2]
    check("jumeaux du meme cote du split (F1.1b)",
          bool(pairs) and all(a["split"] == b["split"] for a, b in pairs),
          f"{len(pairs)} paires")
    # sanite du signe : une VRAIE frame de A ressemble plus a A qu'a B (D0.b, en latent)
    n_ok, n_tot = 0, 0
    for r in rows:
        Ea, Eb = r["lat_a"], r["lat_b"]
        z = Ea[len(Ea) // 2:len(Ea) // 2 + 1]                # une vraie frame de A
        others = torch.cat([Ea[:len(Ea) // 2], Ea[len(Ea) // 2 + 1:]], 0)
        n_tot += 1
        n_ok += int(float(T0.corr_set(z, others)) > float(T0.corr_set(z, Eb)))
    check("D0.b (sanite) : une VRAIE frame de A est plus proche de A que de B, en latent",
          n_ok >= 0.90 * n_tot, f"{n_ok}/{n_tot}")
    return not _FAILS


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="banque de references du terme T0 (memoire 2.1)")
    ap.add_argument("--n", type=int, default=T0.DEFAULT_N, help="bande n x n (8 = le dossier)")
    ap.add_argument("--n_a", type=int, default=8, help="frames de reference du segment A")
    ap.add_argument("--n_b", type=int, default=8, help="frames de reference du segment B")
    ap.add_argument("--skip_head", type=int, default=8,
                    help="frames sautees au debut de A (l'ouvreur X pollue, 16.1 quater)")
    ap.add_argument("--min_gap", type=int, default=2,
                    help="frames exclues de part et d'autre du switch")
    ap.add_argument("--with_pixel", action="store_true",
                    help="pre-calcule AUSSI la bande basse en pixels (VAE, GPU)")
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--n_val_switch", type=int, default=18)
    ap.add_argument("--n_val_single", type=int, default=4)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--selftest_only", action="store_true")
    return ap.parse_args(argv)


def main():
    args = parse_args()
    payload = None if args.selftest_only else build(args)
    ok = selftests(payload)
    print(f"\n[t0bank] SELFTESTS: {'ALL PASS' if ok else 'FAILURES: ' + ', '.join(_FAILS)}",
          flush=True)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
