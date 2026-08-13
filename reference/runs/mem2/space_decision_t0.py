#!/usr/bin/env python
# ============================================================================================
# MEMOIRE 2.1 -- REGLE D0 : LE TERME T0 VIT-IL EN LATENT POOLE (gratuit) OU EN PIXEL (VAE) ?
#
# La question n'est pas esthetique : elle a deja tue deux termes d'instance. Le cosinus latent
# crop60 (J3) demarrait a pos_gap -0.19..-0.36 et le cosinus DINO (phase B) a -0.185, tous deux
# du MAUVAIS cote -- entrainer dessus, c'est pousser sur l'artefact "frame decodee = passe-bas".
# On ne prend donc l'espace latent QUE s'il classe les cas CONNUS comme le pixel le fait.
#
# REGLE SCELLEE (memory2_probes_registry.md, section D0). Latent retenu SSI les trois :
#   D0.a  accord de GEOMETRIE (CPU, 0 GPU) : sur les 30 ancres decodees dont le LATENT
#         (reid_scen/dec_anchors) ET le PIXEL (reid_scen/anchor_px) sont sur disque, la matrice
#         de similarite bande-basse latente et la pixel doivent s'accorder a Spearman >= 0.70
#         sur les paires croisees ET >= 50 % de plus-proche-voisin identique.
#   D0.b  cas connu "une VRAIE frame de A" (CPU) : sur >= 15 videos switch val, une vraie frame
#         latente de A doit correler davantage avec A qu'avec B, en latent, sur >= 90 % des cas.
#   D0.c  cas connu "l'ancre ACTUELLE" (~5 GPU-min, LE test qui compte) : l'ancre decodee par
#         l'ecrivain+decodeur A2 depuis l'etat de FIN de sequence doit reproduire en latent le
#         classement mesure en pixel -- B > A -- sur >= 70 % des videos, et les deux espaces
#         doivent s'accorder video par video sur le SIGNE de corr(A) - corr(B) dans >= 75 % des cas.
# Sinon => PIXEL (banque de references pre-decodee, VAE differentiable seulement cote ancre).
#
#   python space_decision_t0.py --cpu_only          # D0.a + D0.b, aucun GPU
#   CUDA_VISIBLE_DEVICES=1 python space_decision_t0.py     # + D0.c (VAE + replay A2)
# ============================================================================================
import argparse
import glob
import json
import os
import random
import sys

REPO = "/shared/ssd_14T/home/adenelmi/Anchor-Forcing"
PROBE_DIR = os.path.join(REPO, "reference/runs/reid_probe_l1")
FRAME_DEC = os.path.join(REPO, "reference/runs/frame_dec")
SCEN_DIR = os.path.join(REPO, "reference/runs/reid_scen")
HERE = os.path.dirname(os.path.abspath(__file__))
for p in (REPO, PROBE_DIR, FRAME_DEC, SCEN_DIR, HERE):
    sys.path.insert(0, p)
os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")

import torch  # noqa: E402

import mem2_data as D  # noqa: E402
import t0_metric as T0  # noqa: E402

DEC_ANC = "/shared/ssd_14T_project/adenelmi/reid_scen/dec_anchors"
ANC_PX = "/shared/ssd_14T_project/adenelmi/reid_scen/anchor_px"
OUT = os.path.join(D.RUNS_DIR, "writer", "space_decision_t0.json")


# ----------------------------- petites stats sans scipy ---------------------------------------
def spearman(a, b):
    """Correlation de rang de deux vecteurs 1-D (moyenne des rangs sur les ex aequo)."""
    def ranks(x):
        order = torch.argsort(x)
        r = torch.empty_like(x)
        r[order] = torch.arange(x.numel(), dtype=x.dtype, device=x.device)
        # ex aequo : moyenne des rangs (rare ici, mais gratuit et correct)
        vals, inv, cnt = torch.unique(x, return_inverse=True, return_counts=True)
        if int(cnt.max()) > 1:
            sums = torch.zeros_like(vals, dtype=x.dtype).scatter_add_(0, inv, r)
            r = (sums / cnt.to(x.dtype))[inv]
        return r
    ra, rb = ranks(a.double()), ranks(b.double())
    ra = ra - ra.mean()
    rb = rb - rb.mean()
    return float((ra @ rb) / (ra.norm() * rb.norm()).clamp_min(1e-12))


# ----------------------------- D0.a : accord de geometrie (CPU) --------------------------------
def d0a(args):
    """Les 30 ancres decodees dont on a LE MEME OBJET dans les deux representations."""
    pairs = []
    # cache pixel : "<cell>_anchor.pt" (ancre courante) et "<cell>_snapanchor.pt" (levier 4).
    # Le dump LATENT correspondant est "<cell>_decanchor.f180.pt" / "<cell>_snapanchor.f180.pt".
    for px_path in sorted(glob.glob(os.path.join(ANC_PX, "*_anchor.pt"))):
        base = os.path.basename(px_path)[:-len(".pt")]                # ex: S2_t2_s0_snapanchor
        if base.endswith("_snapanchor"):
            cell, lat_name = base[:-len("_snapanchor")], f"{base}.f180.pt"
        else:
            cell = base[:-len("_anchor")]
            lat_name = f"{cell}_decanchor.f180.pt"
        lat_path = os.path.join(DEC_ANC, lat_name)
        if not os.path.exists(lat_path):
            continue
        stem = base
        px = torch.load(px_path, map_location="cpu").float()          # (3, H, W) in [0,1]
        lat = torch.load(lat_path, map_location="cpu").float()        # (1, 1, 16, 60, 104)
        pairs.append((stem, px, lat))
    assert len(pairs) >= 10, f"D0.a : seulement {len(pairs)} paires (latent, pixel) sur disque"
    Zp = torch.cat([T0.lowband_px(px[None], n=args.n) for _, px, _ in pairs], 0)
    out = {"n_pairs": len(pairs), "names": [p[0] for p in pairs], "modes": {}}
    Sp = Zp @ Zp.t()
    off = ~torch.eye(len(pairs), dtype=torch.bool)
    for mode in ("chan", "gray"):
        Zl = torch.cat([T0.lowband_lat(lat, n=args.n, mode=mode) for _, _, lat in pairs], 0)
        Sl = Zl @ Zl.t()
        rho = spearman(Sp[off], Sl[off])
        nn_p = (Sp - 3 * torch.eye(len(pairs))).argmax(dim=1)
        nn_l = (Sl - 3 * torch.eye(len(pairs))).argmax(dim=1)
        agree = float((nn_p == nn_l).float().mean())
        out["modes"][mode] = {"spearman": rho, "nn_agree": agree,
                              "pass": bool(rho >= args.a_rho and agree >= args.a_nn)}
        print(f"[D0.a] latent/{mode:4s} vs pixel : Spearman des similarites {rho:+.3f} "
              f"(seuil {args.a_rho:+.2f}) | plus-proche-voisin identique "
              f"{100 * agree:.0f} % (seuil {100 * args.a_nn:.0f} %) : "
              f"{'PASS' if out['modes'][mode]['pass'] else 'FAIL'}", flush=True)
    return out


# ----------------------------- D0.b : "une vraie frame de A" (CPU) -----------------------------
def d0b(args):
    vids = D.load_index(verbose=False)
    split = D.assign_split(vids)
    cache = D.PartCache(size=2)
    from build_bank_t0 import ref_frames
    rows = {"chan": [], "gray": []}
    n_done = 0
    for meta in vids:
        if meta["mode"] != "switch" or split[meta["video_id"]] != "val":
            continue
        fa, fb = ref_frames(meta, args.n_a, args.n_b)
        if fa is None:
            continue
        lat_a = D.load_frames(meta, cache, fa)
        lat_b = D.load_frames(meta, cache, fb)
        mid = len(fa) // 2
        for mode in ("chan", "gray"):
            Ea = T0.lowband_lat(lat_a, n=args.n, mode=mode)
            Eb = T0.lowband_lat(lat_b, n=args.n, mode=mode)
            z = Ea[mid:mid + 1]
            others = torch.cat([Ea[:mid], Ea[mid + 1:]], 0)
            rows[mode].append(float(T0.corr_set(z, others) - T0.corr_set(z, Eb)))
        n_done += 1
        if n_done >= args.b_videos:
            break
    out = {"n_videos": n_done, "modes": {}}
    for mode, v in rows.items():
        frac = sum(1 for x in v if x > 0) / max(1, len(v))
        out["modes"][mode] = {"frac_a_gt_b": frac, "mean_margin": sum(v) / max(1, len(v)),
                              "pass": bool(frac >= args.b_frac and n_done >= 15)}
        print(f"[D0.b] latent/{mode:4s} : une VRAIE frame de A est plus proche de A que de B "
              f"sur {100 * frac:.0f} % des {n_done} videos val (seuil {100 * args.b_frac:.0f} %), "
              f"marge moyenne {out['modes'][mode]['mean_margin']:+.4f} : "
              f"{'PASS' if out['modes'][mode]['pass'] else 'FAIL'}", flush=True)
    return out


# ----------------------------- D0.c : "l'ancre actuelle" (GPU, ~5 min) -------------------------
def d0c(args):
    """Le test qui compte : l'ancre decodee par A2 depuis l'etat de FIN de sequence, mesuree
    dans les DEUX espaces sur les MEMES videos val -- le latent doit reproduire le classement
    B > A du pixel (le fait publie : B plus proche que A sur 21/24 cellules, p = 3e-4)."""
    import train_writer as TW
    argv = ["--load", args.load, "--dec_ckpt", args.dec_ckpt, "--t0_log",
            "--t0_space", "latent", "--t0_bank", args.t0_bank, "--t0_k", str(args.t0_k),
            "--device", args.device]
    if args.writer_ckpt:
        argv += ["--writer_ckpt", args.writer_ckpt]
    a = TW.parse_args(argv)
    tr = TW.WriterTrainer(a)
    vids = [v for v in tr.val_vids if v["mode"] == "switch"][:args.c_videos]
    from build_bank_dino import latent_to_pixels
    if tr.vae is None:
        from train_decoder import build_vae
        tr.vae = build_vae(tr.device)
    bank = tr.t0bank
    rows = []
    with torch.no_grad():
        for meta in vids:
            if not bank.has(meta):
                continue
            pos = tr.positions_for(meta, k=args.t0_k, anchor_frac=1.0,
                                   rng=random.Random(20260804))
            sel = [t for t, p in enumerate(pos) if p["kind"] == "anchor"][:args.t0_k]
            if not sel:
                continue
            pred, _, _, _ = tr.forward_video(meta, pos, n_bptt=0, force_cond=True)
            x = pred[sel]
            i = bank.by_id[meta["video_id"]]
            r = {"video_id": meta["video_id"]}
            # latent (les deux reductions)
            for mode in ("chan", "gray"):
                key = "lat" if mode == "chan" else "latg"
                Ea = bank.rows[i][f"{key}_a"].to(tr.device)
                Eb = bank.rows[i][f"{key}_b"].to(tr.device)
                z = T0.lowband_lat(x, n=args.n, mode=mode)
                r[f"lat_{mode}_a"] = float(T0.corr_set(z, Ea).mean())
                r[f"lat_{mode}_b"] = float(T0.corr_set(z, Eb).mean())
            # pixel : le cote reference DOIT venir du VAE lui aussi (meme espace des deux cotes)
            assert "px_a" in bank.rows[i], \
                "D0.c a besoin d'une banque construite avec --with_pixel"
            Ea = bank.rows[i]["px_a"].to(tr.device)
            Eb = bank.rows[i]["px_b"].to(tr.device)
            zp = T0.lowband_px(latent_to_pixels(tr.vae, x).clamp(0, 1), n=args.n)
            r["px_a"] = float(T0.corr_set(zp, Ea).mean())
            r["px_b"] = float(T0.corr_set(zp, Eb).mean())
            rows.append(r)
            del pred
    assert rows, "D0.c : aucune video val exploitable"
    out = {"n_videos": len(rows), "rows": rows, "modes": {}}
    n = len(rows)
    px_sign = [(r["px_a"] - r["px_b"]) > 0 for r in rows]
    out["px_frac_b_gt_a"] = sum(1 for s in px_sign if not s) / n
    print(f"[D0.c] PIXEL (reference) : B > A sur {100 * out['px_frac_b_gt_a']:.0f} % des {n} "
          f"videos val  [fait publie sur les cellules S2 : 21/24, p = 3e-4]", flush=True)
    for mode in ("chan", "gray"):
        lat_sign = [(r[f"lat_{mode}_a"] - r[f"lat_{mode}_b"]) > 0 for r in rows]
        frac_b = sum(1 for s in lat_sign if not s) / n
        agree = sum(1 for i in range(n) if lat_sign[i] == px_sign[i]) / n
        out["modes"][mode] = {"frac_b_gt_a": frac_b, "sign_agree": agree,
                              "pass": bool(frac_b >= args.c_frac and agree >= args.c_agree)}
        print(f"[D0.c] latent/{mode:4s} : B > A sur {100 * frac_b:.0f} % (seuil "
              f"{100 * args.c_frac:.0f} %) | accord de SIGNE avec le pixel "
              f"{100 * agree:.0f} % (seuil {100 * args.c_agree:.0f} %) : "
              f"{'PASS' if out['modes'][mode]['pass'] else 'FAIL'}", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser(description="regle D0 : espace du terme T0 (latent vs pixel)")
    ap.add_argument("--n", type=int, default=T0.DEFAULT_N)
    ap.add_argument("--n_a", type=int, default=8)
    ap.add_argument("--n_b", type=int, default=8)
    ap.add_argument("--cpu_only", action="store_true", help="D0.a + D0.b seulement (0 GPU)")
    ap.add_argument("--a_rho", type=float, default=0.70)
    ap.add_argument("--a_nn", type=float, default=0.50)
    ap.add_argument("--b_videos", type=int, default=26)
    ap.add_argument("--b_frac", type=float, default=0.90)
    ap.add_argument("--c_videos", type=int, default=18)
    ap.add_argument("--c_frac", type=float, default=0.70)
    ap.add_argument("--c_agree", type=float, default=0.75)
    ap.add_argument("--t0_bank", default=None)
    ap.add_argument("--t0_k", type=int, default=1)
    ap.add_argument("--load", default="pipeline")
    ap.add_argument("--dec_ckpt", default=os.path.join(D.RUNS_DIR, "writer",
                                                       "dec_A2_for_memdec.pt"))
    ap.add_argument("--writer_ckpt", default=os.path.join(D.RUNS_DIR, "writer",
                                                          "mem2_writer_A2_final.pt"))
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    if args.t0_bank is None:
        from build_bank_t0 import BANK_T0
        args.t0_bank = BANK_T0

    res = {"n": args.n, "thresholds": {"a_rho": args.a_rho, "a_nn": args.a_nn,
                                       "b_frac": args.b_frac, "c_frac": args.c_frac,
                                       "c_agree": args.c_agree}}
    print("\n=== D0.a  accord de geometrie latent vs pixel (CPU) ===", flush=True)
    res["a"] = d0a(args)
    print("\n=== D0.b  cas connu : une VRAIE frame de A (CPU) ===", flush=True)
    res["b"] = d0b(args)
    if not args.cpu_only:
        print("\n=== D0.c  cas connu : l'ancre ACTUELLE (A2), pixel vs latent (GPU) ===",
              flush=True)
        res["c"] = d0c(args)

    print("\n=== VERDICT D0 ===", flush=True)
    decided = "pixel"
    for mode in ("chan", "gray"):
        ok = res["a"]["modes"][mode]["pass"] and res["b"]["modes"][mode]["pass"]
        if "c" in res:
            ok = ok and res["c"]["modes"][mode]["pass"]
        print(f"  latent/{mode:4s} : {'RETENU' if ok else 'REJETE'}"
              + ("" if "c" in res else "  (D0.c non evalue : verdict PROVISOIRE)"), flush=True)
        if ok and decided == "pixel":
            decided = f"latent/{mode}"
    res["decision"] = decided
    res["provisional"] = "c" not in res
    print(f"  => ESPACE RETENU : {decided}"
          + ("  (PROVISOIRE tant que D0.c n'a pas tourne)" if "c" not in res else ""), flush=True)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(res, f, indent=2)
    print(f"  -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
