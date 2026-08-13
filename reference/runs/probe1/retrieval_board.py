#!/usr/bin/env python
# ============================================================================================
# PHASE 1 (2026-08-10) -- "QU'EST-CE QUE LA MEMOIRE A RETENU ?" EN IMAGES, SANS GENERATION.
#
# Pour chaque video du corpus v2 (60 videos, 30 paires), on projette l'etat de memoire dans
# l'espace des references DINO avec la TETE DE SONDAGE DU PLI OU LA VIDEO EST EN TEST :
#   - les plis, les configs par pli, les graines viennent des JSON DE VERDICT SCELLES
#     (runs/cv_{source}_subject_pool_v2.json) : AUCUNE recherche d'hyperparametres n'est
#     relancee, AUCUN choix nouveau ; on refait exactement le geste de cv_run apres hpsearch
#     (set_fold(ds, test, []) puis train_one(cfg, seed)), qui est deterministe ;
#   - ZERO fuite : chaque video n'est projetee que par le modele du pli ou sa paire est en
#     test (ni blanchiment, ni ACP, ni softmax n'ont vu sa paire) ;
#   - CONTROLE D'IDENTITE : les marges recalculees sont confrontees aux marges stockees dans
#     le JSON scelle (per_row), graine par graine ; l'ecart max est rapporte.
#
# RECUPERATION a 3 niveaux (galerie = crops-sujet DINO, 1 reference moyenne par video) :
#   (a) 1 parmi 2  : {sa video, sa jumelle}                      -- ancre = mesure existante
#   (b) 1 parmi 10 : + 8 distracteurs tires DETERMINISTEMENT (hash sha1, graine du split)
#   (c) 1 parmi 60 : toute la galerie, sa propre vignette INCLUSE ("retrouve TON sujet")
# La cible est toujours la reference de la video elle-meme ; les distracteurs excluent la
# video et sa jumelle (la jumelle, elle, est TOUJOURS dans la galerie : c'est le piege).
# Controle permute : 25 derangements des etats entre les 60 videos (graines 1234+k, comme
# evaluate()).
#
# Sorties :
#   runs/retrieval_gallery_v2.json         (tous les chiffres bruts)
#   board/retrieval_board_v2.jpg           (la planche pour l'oeil d'Aden, < 1.5 MB)
# ============================================================================================
import argparse
import hashlib
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import probe1_common as C                     # noqa: E402

import numpy as np                            # noqa: E402
import torch                                  # noqa: E402

import train_probe as TP                      # noqa: E402

SOURCES = [("state", "ecriture actuelle"),
           ("mlp_full_muon", "MLP corr. (A3)"),
           ("lin_delta_all", "delta (L4)")]
# BRAS DE DIAGNOSTIC (demande coordinateur 10/08) : la lecture directe des activations
# masquees, DANS SA CONFIGURATION VALIDEE (avec pre-entrainement, 15.67/19) et sans
# pre-entrainement (armes egales avec les etats, 12.00/19). Chiffres seulement, pas de
# colonne de planche : ils departagent "goulot = memoire" vs "goulot = reference/tete".
ARMS = [(src, src, False, lbl) for src, lbl in SOURCES] + [
    ("hdirect_masked_pre", "hdirect_masked", True, "lecture directe PRE (ref. validee)"),
    ("hdirect_masked", "hdirect_masked", False, "lecture directe sans pre"),
]
SEEDS = (0, 1, 2)
N_DISTRACTORS = 8
N_PERM = 25
PERM_SEED = 1234                              # la graine historique d'evaluate()

OUT_JSON = os.path.join(C.RUNS_DIR, "retrieval_gallery_v2.json")
OUT_BOARD = os.path.join(C.BOARD_DIR, "retrieval_board_v2.jpg")


# ---------------------------------------------------------------------------------------------
# 1) reconstruire les tetes de pli et projeter chaque video par SON pli de test
# ---------------------------------------------------------------------------------------------
def project_source(source, device, pretrain=False):
    """Rend (ds, P) avec P[(video_index, age_name, seed)] = projection L2-normalisee (768,),
    calculee par le modele du pli ou la video est en TEST. Verifie les marges contre le JSON
    scelle et rend aussi le rapport de cet audit. `pretrain=True` relit le JSON `_pre` et
    pre-entraine la tete sur les 105 videos switch, EXACTEMENT comme le run scelle."""
    cvp = os.path.join(C.RUNS_DIR,
                       f"cv_{source}_subject_pool{'_pre' if pretrain else ''}_v2.json")
    cv = C.load_json(cvp)
    assert cv["corpus"] == "v2" and cv["features"] == "pool" \
        and bool(cv["pretrain"]) == bool(pretrain)
    assert list(cv["seeds"]) == list(SEEDS), f"graines du JSON {cv['seeds']} != {SEEDS}"

    ds = TP.load_dataset(source, "subject", features="pool", corpus="v2")
    assert len(ds["videos"]) == 60 and not ds["dropped_pairs"], \
        f"{source}: {len(ds['videos'])} videos, dropped={ds['dropped_pairs']}"
    if pretrain:
        ds["_pretrain"] = TP.load_pretrain(source, "subject", features="pool")

    # les plis du JSON scelle doivent etre EXACTEMENT ceux que pair_folds recalcule
    cidxs = sorted({int(v["corpus_idx"]) for v in ds["videos"]})
    assert TP.pair_folds(cidxs, n_folds=cv["n_folds"]) == [sorted(f) for f in cv["folds"]], \
        f"{source}: plis recalcules != plis scelles"

    # marges scellees, indexees (seed, video_id, age)
    sealed = {}
    for rec in cv["per_seed"]:
        for r in rec["per_row"]:
            sealed[(int(rec["seed"]), r["video_id"], r["age"])] = float(r["margin"])

    tag = source + ("+pre" if pretrain else "")
    P = {}
    audits = []
    for k, fold in enumerate(cv["folds"]):
        cfg = cv["per_fold"][k]["cfg"]
        TP.set_fold(ds, fold, [])             # val reverse dans le train, test = le pli
        rows = TP.split_rows(ds, "test")
        row_meta = [(int(ds["s_vid"][i]), ds["s_age"][int(i)]) for i in rows.tolist()]
        for seed in SEEDS:
            model, pre = TP.train_one(ds, cfg, seed=seed, device=device)
            with torch.no_grad():
                Z = TP.transformed(ds, "test", cfg["sig_dim"], device=device)
                proj = model(Z).float().cpu()
            for r, (vi, age) in enumerate(row_meta):
                P[(vi, age, seed)] = proj[r]
                # audit contre la marge scellee
                v = ds["videos"][vi]
                own = v["refs"].mean(0)
                oth = ds["videos"][v["twin_index"]]["refs"].mean(0)
                marg = float(proj[r] @ own - proj[r] @ oth)
                audits.append({"seed": seed, "video_id": v["video_id"], "age": age,
                               "margin_new": marg,
                               "margin_sealed": sealed[(seed, v["video_id"], age)]})
        print(f"[{tag}] pli {k + 1}/6 projete (cfg sig_dim={cfg['sig_dim']} "
              f"depth={cfg['depth']} lr={cfg['lr']})", flush=True)

    diffs = np.array([a["margin_new"] - a["margin_sealed"] for a in audits])
    flips = sum(1 for a in audits
                if (a["margin_new"] > 0) != (a["margin_sealed"] > 0)
                and abs(a["margin_sealed"]) > 1e-6)
    audit = {"n_rows": len(audits), "max_abs_diff": float(np.abs(diffs).max()),
             "mean_abs_diff": float(np.abs(diffs).mean()), "n_sign_flips": int(flips)}
    print(f"[{tag}] AUDIT marges vs JSON scelle : max|d|={audit['max_abs_diff']:.3e} "
          f"flips={flips}/{len(audits)}", flush=True)
    return ds, P, audit


# ---------------------------------------------------------------------------------------------
# 2) galerie + niveaux de difficulte
# ---------------------------------------------------------------------------------------------
def gallery_scores(ds, P):
    """S[seed] = matrice (60, 60) : S[v, g] = sim moyenne (sur les ages) entre la projection de
    l'etat de v et la reference moyenne du sujet de g. Rend aussi S_avg (moyenne des graines)."""
    n = len(ds["videos"])
    R = torch.stack([v["refs"].mean(0) for v in ds["videos"]], 0)          # (60, 768)
    ages = sorted({a for (_vi, a, _s) in P})
    S = {}
    for seed in SEEDS:
        M = torch.zeros(n, n)
        for vi in range(n):
            pr = torch.stack([P[(vi, a, seed)] for a in ages if (vi, a, seed) in P], 0)
            M[vi] = (pr @ R.transpose(0, 1)).mean(0)
        S[seed] = M
    S_avg = torch.stack([S[s] for s in SEEDS], 0).mean(0)
    return S, S_avg


def distractors_for(ds, vi):
    """Les 8 distracteurs du niveau (b), tires DETERMINISTEMENT : hash sha1 de
    (graine du split, 'retr10', video, candidat) ; excluent la video et sa jumelle."""
    v = ds["videos"][vi]
    cands = [g for g in range(len(ds["videos"])) if g != vi and g != v["twin_index"]]
    key = lambda g: hashlib.sha1(  # noqa: E731
        f"{C.SPLIT_SEED}|retr10|{v['video_id']}|{ds['videos'][g]['video_id']}".encode()
    ).hexdigest()
    return sorted(cands, key=key)[:N_DISTRACTORS]


def levels_for(ds, vi):
    tw = ds["videos"][vi]["twin_index"]
    return {"a2": [vi, tw],
            "b10": [vi, tw] + distractors_for(ds, vi),
            "c60": list(range(len(ds["videos"])))}


def rank_of(row_scores, cand, target):
    """Rang (1-based) de `target` parmi `cand` selon les scores (plus grand = mieux).
    Egalites tranchees par l'index : deterministe."""
    s = [(float(row_scores[g]), -g) for g in cand]
    st = (float(row_scores[target]), -target)
    return 1 + sum(1 for x in s if x > st)


def retrieval_metrics(ds, S_by_seed, S_avg):
    """top-1 et rang median du vrai sujet aux 3 niveaux ; par graine + sur la moyenne des
    graines ; + controle permute (25 derangements des etats entre videos, sur S_avg)."""
    n = len(ds["videos"])
    lv = {vi: levels_for(ds, vi) for vi in range(n)}
    out = {}
    for name in ("a2", "b10", "c60"):
        per_seed = {}
        for seed in SEEDS:
            rk = [rank_of(S_by_seed[seed][vi], lv[vi][name], vi) for vi in range(n)]
            per_seed[seed] = rk
        rk_avg = [rank_of(S_avg[vi], lv[vi][name], vi) for vi in range(n)]

        # permute : l'etat d'une AUTRE video (derangement), scores S_avg
        perm_top1, perm_med = [], []
        for k in range(N_PERM):
            rng = np.random.RandomState(PERM_SEED + k)
            perm = None
            for _ in range(200):
                cand = rng.permutation(n)
                if all(cand[i] != i for i in range(n)):
                    perm = cand
                    break
            assert perm is not None
            rkp = [rank_of(S_avg[int(perm[vi])], lv[vi][name], vi) for vi in range(n)]
            perm_top1.append(float(np.mean([r == 1 for r in rkp])))
            perm_med.append(float(np.median(rkp)))

        # rang de la JUMELLE dans la meme galerie (diagnostic de voisinage, 10/08) :
        # si le vrai sujet ET la jumelle sont tous deux tres loin, la projection ne vit
        # pas du tout dans le bon voisinage de l'espace des references.
        rk_twin = [rank_of(S_avg[vi], lv[vi][name], ds["videos"][vi]["twin_index"])
                   for vi in range(n)] if name == "c60" else None

        out[name] = {
            "n_gallery": len(lv[0][name]),
            "top1_per_seed": [float(np.mean([r == 1 for r in per_seed[s]])) for s in SEEDS],
            "top1_mean_seeds": float(np.mean(
                [np.mean([r == 1 for r in per_seed[s]]) for s in SEEDS])),
            "top1_seedavg_scores": float(np.mean([r == 1 for r in rk_avg])),
            "median_rank_per_seed": [float(np.median(per_seed[s])) for s in SEEDS],
            "median_rank_seedavg": float(np.median(rk_avg)),
            "mean_rank_seedavg": float(np.mean(rk_avg)),
            "permuted_top1_mean": float(np.mean(perm_top1)),
            "permuted_median_rank_mean": float(np.mean(perm_med)),
            "ranks_seedavg": rk_avg,
        }
        if rk_twin is not None:
            out[name]["twin_median_rank_seedavg"] = float(np.median(rk_twin))
            out[name]["twin_mean_rank_seedavg"] = float(np.mean(rk_twin))
            out[name]["twin_ranks_seedavg"] = rk_twin
    return out


# ---------------------------------------------------------------------------------------------
# 3) la planche
# ---------------------------------------------------------------------------------------------
def crop_path(video_id):
    """Le crop-sujet du milieu des frames VALIDES (meme regle de validite que les refs)."""
    d = os.path.join(C.MASK_DIR, "crops", C.seg_key(video_id, "A"))
    files = sorted(f for f in os.listdir(d) if f.endswith("_subj.jpg"))
    assert files, f"aucun crop sujet pour {video_id}"
    npz = os.path.join(C.MASK_DIR, C.seg_key(video_id, "A") + ".npz")
    try:
        r = np.load(npz, allow_pickle=True)
        frames = [int(x) for x in r["frames"]]
        valid = r["valid"].astype(bool) if "valid" in r.files else np.ones(len(frames), bool)
        ok = [f for f, v in zip(frames, valid) if v]
        if ok:
            want = f"f{ok[len(ok) // 2]:03d}_subj.jpg"
            if want in files:
                return os.path.join(d, want)
    except Exception:
        pass
    return os.path.join(d, files[len(files) // 2])


def build_board(ds, results, path, thumb=160):
    from PIL import Image, ImageDraw, ImageFont
    try:
        font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 22)
        font_s = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 17)
    except Exception:
        font = font_s = ImageFont.load_default()

    strata = {}
    for no in C.ADEN_V2_INVALID_5:
        strata[no] = "x"
    for no in C.ADEN_V2_WEAK_6:
        strata[no] = "~"

    order = sorted(range(len(ds["videos"])),
                   key=lambda vi: (ds["videos"][vi]["board_no"],
                                   0 if ds["videos"][vi]["video_id"].startswith("main") else 1))
    lab_w, pad, txt_h = 84, 5, 26
    n_cols = 1 + len(SOURCES)
    W = lab_w + n_cols * (thumb + pad) + pad
    header_h = 96
    row_h = thumb + txt_h + 2 * pad
    Hh = header_h + len(order) * row_h + pad
    img = Image.new("RGB", (W, Hh), (18, 18, 18))
    dr = ImageDraw.Draw(img)

    cols = ["vrai sujet"] + [lbl for _s, lbl in SOURCES]
    dr.text((6, 6), "v2 : sujet retrouve dans la galerie des 60",
            fill=(255, 255, 255), font=font_s)
    dr.text((6, 28), "vert = sa video (r=1) ; orange = sa jumelle ; rouge = autre",
            fill=(200, 200, 200), font=font_s)
    dr.text((6, 50), "NNa/b = planche Aden + principale/jumelle ; x = invalide, ~ = faible",
            fill=(200, 200, 200), font=font_s)
    for c, name in enumerate(cols):
        x = lab_w + c * (thumb + pad) + pad
        dr.text((x + 2, header_h - 24), name[:22], fill=(255, 255, 0), font=font_s)

    def paste(pth, x, y, border):
        im = Image.open(pth).convert("RGB")
        im.thumbnail((thumb - 8, thumb - 8))
        cell = Image.new("RGB", (thumb, thumb), (18, 18, 18))
        cx, cy = (thumb - im.width) // 2, (thumb - im.height) // 2
        cell.paste(im, (cx, cy))
        if border:
            d2 = ImageDraw.Draw(cell)
            for t in range(4):
                d2.rectangle([cx - 1 - t, cy - 1 - t, cx + im.width + t, cy + im.height + t],
                             outline=border)
        img.paste(cell, (x, y))

    for rix, vi in enumerate(order):
        v = ds["videos"][vi]
        y = header_h + rix * row_h
        no = v["board_no"]
        role = "a" if v["video_id"].startswith("main") else "b"
        lab = f"{no:02d}{role}{strata.get(no, '')}"
        col = (130, 130, 130) if strata.get(no) == "x" else (255, 255, 255)
        dr.text((6, y + thumb // 2 - 10), lab, fill=col, font=font)

        paste(crop_path(v["video_id"]), lab_w + pad, y, None)
        for c, (src, _lbl) in enumerate(SOURCES, start=1):
            det = results[src]["per_video"][vi]
            g = det["top1_c60_video"]
            rk = det["rank_c60"]
            if g == vi:
                border = (0, 200, 0)
            elif g == v["twin_index"]:
                border = (255, 160, 0)
            else:
                border = (220, 40, 40)
            x = lab_w + c * (thumb + pad) + pad
            paste(crop_path(ds["videos"][g]["video_id"]), x, y, border)
            dr.text((x + 2, y + thumb + 2), f"r={rk}/60", fill=(255, 255, 255), font=font_s)

    # < 1.5 MB : on descend la qualite puis l'echelle
    target = 1.5 * 1024 * 1024
    for scale in (1.0, 0.85, 0.72, 0.6):
        im2 = img if scale == 1.0 else img.resize(
            (int(W * scale), int(Hh * scale)), Image.LANCZOS)
        for q in (77, 68, 60, 52, 45):
            im2.save(path, "JPEG", quality=q, optimize=True)
            if os.path.getsize(path) <= target:
                print(f"[board] {path} : {os.path.getsize(path) / 1e6:.2f} MB "
                      f"(scale={scale}, q={q}, {im2.width}x{im2.height})", flush=True)
                return
    print(f"[board] ATTENTION : {os.path.getsize(path) / 1e6:.2f} MB > 1.5 MB", flush=True)


# ---------------------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    C.ensure_dirs()
    C.set_determinism(0)
    t0 = time.time()

    out = {"date": time.strftime("%Y-%m-%d %H:%M:%S"), "corpus": "v2", "seeds": list(SEEDS),
           "n_perm": N_PERM, "perm_seed": PERM_SEED, "n_distractors": N_DISTRACTORS,
           "protocole": "tetes re-entrainees a l'identique des runs CV scelles (plis, cfg "
                        "par pli, graines relus du JSON de verdict ; hpsearch NON relance) ; "
                        "chaque video projetee par le modele du pli ou elle est en TEST",
           "sources": {}}
    results = {}
    ds_keep = None
    for key, src, pretrain, _lbl in ARMS:
        ds, P, audit = project_source(src, args.device, pretrain=pretrain)
        S_by_seed, S_avg = gallery_scores(ds, P)
        met = retrieval_metrics(ds, S_by_seed, S_avg)
        per_video = []
        for vi in range(len(ds["videos"])):
            lvv = levels_for(ds, vi)
            top1 = int(max(lvv["c60"], key=lambda g: (float(S_avg[vi][g]), -g)))
            per_video.append({
                "video_id": ds["videos"][vi]["video_id"],
                "board_no": int(ds["videos"][vi]["board_no"]),
                "rank_a2": rank_of(S_avg[vi], lvv["a2"], vi),
                "rank_b10": rank_of(S_avg[vi], lvv["b10"], vi),
                "rank_c60": rank_of(S_avg[vi], lvv["c60"], vi),
                "rank_twin_c60": rank_of(S_avg[vi], lvv["c60"],
                                         ds["videos"][vi]["twin_index"]),
                "top1_c60_video": top1,
                "top1_c60_video_id": ds["videos"][top1]["video_id"],
                "distractors_b10": [ds["videos"][g]["video_id"]
                                    for g in lvv["b10"][2:]],
            })
        results[key] = {"metrics": met, "per_video": per_video, "audit": audit}
        out["sources"][key] = {"source": src, "pretrain": bool(pretrain),
                               "audit_marges": audit,
                               "metrics": met,
                               "per_video": per_video}
        if key == SOURCES[-1][0]:
            ds_keep = ds

    build_board(ds_keep, results, OUT_BOARD)
    out["board"] = OUT_BOARD
    out["minutes"] = (time.time() - t0) / 60
    C.save_json(out, OUT_JSON)

    print("\n================ RESUME ================")
    for key, _src, _pre, lbl in ARMS:
        m = results[key]["metrics"]
        a = results[key]["audit"]
        print(f"-- {key} ({lbl})  [audit marges max|d|={a['max_abs_diff']:.2e} "
              f"flips={a['n_sign_flips']}]")
        for name, nice in (("a2", "1 parmi 2 "), ("b10", "1 parmi 10"), ("c60", "1 parmi 60")):
            mm = m[name]
            print(f"   {nice}: top1={mm['top1_mean_seeds']:.3f} (graines "
                  f"{[round(x, 3) for x in mm['top1_per_seed']]}, "
                  f"seedavg {mm['top1_seedavg_scores']:.3f}) "
                  f"rang median={mm['median_rank_seedavg']:.1f} | "
                  f"permute top1={mm['permuted_top1_mean']:.3f} "
                  f"rang med={mm['permuted_median_rank_mean']:.1f}"
                  + (f" | rang med JUMELLE={mm['twin_median_rank_seedavg']:.1f}"
                     if 'twin_median_rank_seedavg' in mm else ""))
    print(f"\n[out] {OUT_JSON}\n[out] {OUT_BOARD}\nPHASE1_DONE ({out['minutes']:.1f} min)")


if __name__ == "__main__":
    main()
