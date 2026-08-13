#!/usr/bin/env python
# ============================================================================================
# MEMOIRE 2.1 -- SELFTESTS DU TERME T0.  100 % CPU, aucun GPU, aucun poids de 1.3B.
#
# Ce qu'ils garantissent, dans l'ordre ou un auditeur voudra les lire :
#   1-4   la METRIQUE est bien celle du dossier (identite BITWISE avec prescreen_offensif.lowband),
#         c'est bien une correlation de Pearson, elle est invariante par affine positive.
#   5-8   la PERTE fait ce qu'elle dit : InfoNCE mono-positif == cross-entropy, le hinge sature,
#         le gradient pointe vers A, une descente sur donnees synthetiques retourne la marge.
#   9-10  les NEGATIFS sont les bons : la video courante est exclue du pool, le jumeau y reste.
#   11-12 les POSITIONS sont les bonnes : `--t0_end_pos` force l'etat de FIN de sequence sans
#         toucher a la cible ni au cond, et il est INERTE quand on ne le demande pas.
#   13    BYTE-SAFETY : lam_t0 = 0 et pas de --t0_log => la perte est EXACTEMENT celle de A2.
#   14-15 les references de la banque respectent les frontieres A/B, et sur le VRAI corpus une
#         vraie frame de A est plus proche de A que de B en latent (le sens de D0.b).
#   16    l'aide statistique (Spearman) est correcte sur un cas connu.
#
#   python reference/runs/mem2/selftest_t0.py
# ============================================================================================
import os
import random
import sys
import types

REPO = "/shared/ssd_14T/home/adenelmi/Anchor-Forcing"
PROBE_DIR = os.path.join(REPO, "reference/runs/reid_probe_l1")
FRAME_DEC = os.path.join(REPO, "reference/runs/frame_dec")
SCEN_DIR = os.path.join(REPO, "reference/runs/reid_scen")
HERE = os.path.dirname(os.path.abspath(__file__))
for p in (REPO, PROBE_DIR, FRAME_DEC, SCEN_DIR, HERE):
    sys.path.insert(0, p)
os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")        # ceinture: ce fichier ne touche AUCUN GPU

import torch  # noqa: E402
import torch.nn.functional as Fn  # noqa: E402

import mem2_data as D  # noqa: E402
import t0_metric as T0  # noqa: E402

_FAILS = []
_N = [0]


def check(name, ok, detail=""):
    _N[0] += 1
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"   ({detail})" if detail else ""),
          flush=True)
    if not ok:
        _FAILS.append(name)
    return ok


# ============================================================================================
def test_metric():
    print("\n--- 1-4 : la metrique ---")
    torch.manual_seed(0)
    x = torch.rand(5, 3, 47, 83)
    from prescreen_offensif import lowband as ref_lowband       # LA fonction du dossier
    a = T0.lowband_px(x, n=8)
    b = ref_lowband(x, n=8)
    check("1. lowband_px est BITWISE prescreen_offensif.lowband",
          bool(torch.equal(a, b)), f"max|delta| = {float((a - b).abs().max()):.3e}")

    # 2. le produit scalaire de deux sorties EST la correlation de Pearson des entrees poolees
    g = (0.299 * x[:, 0] + 0.587 * x[:, 1] + 0.114 * x[:, 2])[:, None]
    p = Fn.adaptive_avg_pool2d(g, 8).flatten(1)
    u, v = p[0], p[1]
    pear = float(((u - u.mean()) @ (v - v.mean()))
                 / ((u - u.mean()).norm() * (v - v.mean()).norm()))
    check("2. <lowband(u), lowband(v)> == correlation de Pearson",
          abs(float(a[0] @ a[1]) - pear) < 1e-5, f"{float(a[0] @ a[1]):.6f} vs {pear:.6f}")

    # 3. invariance par affine positive (le contraste global de l'ancre ne doit pas compter)
    y = 2.5 * x + 0.3
    check("3. invariance par affine positive (a>0)",
          bool((T0.lowband_px(y, n=8) - a).abs().max() < 1e-4),
          f"max|delta| = {float((T0.lowband_px(y, n=8) - a).abs().max()):.3e}")

    # 4. la variante latente : dims annoncees, unitaire, centree, differentiable
    lat = torch.randn(4, 16, 60, 104, requires_grad=True)
    for mode in ("chan", "gray"):
        z = T0.lowband_lat(lat, n=8, mode=mode)
        ok = (z.shape == (4, T0.t0_dim("latent", 8, mode))
              and bool(((z.norm(dim=1) - 1).abs() < 1e-4).all())
              and bool((z.mean(dim=1).abs() < 1e-6).all()))
        z.sum().backward(retain_graph=True)
        ok &= lat.grad is not None and float(lat.grad.abs().sum()) > 0
        lat.grad = None
        check(f"4. lowband_lat/{mode}: dim {tuple(z.shape)}, unitaire, centre, differentiable",
              ok)
    z5 = T0.lowband_lat(torch.randn(2, 1, 16, 60, 104), n=8)
    check("4b. lowband_lat accepte (B,1,C,H,W)", z5.shape[0] == 2)


def test_loss():
    print("\n--- 5-8 : la perte ---")
    torch.manual_seed(1)
    Dd = 64
    z = Fn.normalize(torch.randn(3, Dd), dim=1)
    Ea = Fn.normalize(torch.randn(1, Dd), dim=1)
    Eb = Fn.normalize(torch.randn(5, Dd), dim=1)
    Eo = Fn.normalize(torch.randn(20, Dd), dim=1)

    # 5. InfoNCE mono-positif == cross_entropy sur [pos, negs]
    tau = 0.1
    li, _ = T0.t0_loss(z, Ea, Eb, Eo, tau=tau, kind="infonce", w_b=1.0)
    logits = torch.cat([(z @ Ea.t()) / tau, (z @ Eb.t()) / tau, (z @ Eo.t()) / tau], 1)
    ce = Fn.cross_entropy(logits, torch.zeros(3, dtype=torch.long))
    check("5. InfoNCE mono-positif == cross_entropy", abs(float(li) - float(ce)) < 1e-5,
          f"{float(li):.6f} vs {float(ce):.6f}")

    # 6. le hinge sature une fois la marge atteinte (anti-Goodhart : pas de pression au-dela)
    zi = Ea.repeat(1, 1)                                  # ancre = exactement le positif
    Eb2 = Fn.normalize(-Ea + 1e-3 * torch.randn(1, Dd), dim=1)
    Eo2 = Eb2.clone()
    lm, lg = T0.t0_loss(zi, Ea, Eb2, Eo2, kind="margin", m0=0.30)
    check("6. hinge sature quand la marge depasse m0", float(lm) == 0.0,
          f"loss {float(lm):.6f}, marge mesuree {lg['t0_margin']:+.3f}")
    lm2, _ = T0.t0_loss(Fn.normalize(Eb2.clone(), dim=1), Ea, Eb2, Eo2, kind="margin", m0=0.30)
    check("6b. hinge actif quand la marge est du mauvais cote", float(lm2) > 0.0,
          f"loss {float(lm2):.4f}")

    # 7. le gradient existe, est fini, et pointe vers A
    zz = Fn.normalize(torch.randn(2, Dd), dim=1).requires_grad_(True)
    ll, _ = T0.t0_loss(zz, Ea, Eb, Eo, kind="both")
    ll.backward()
    g = zz.grad
    ok = g is not None and torch.isfinite(g).all() and float(g.abs().sum()) > 0
    # descente : -grad doit augmenter corr(z, A)
    with torch.no_grad():
        c0 = T0.corr_set(zz.detach(), Ea).mean()
        z1 = Fn.normalize(zz.detach() - 0.05 * g, dim=1)
        c1 = T0.corr_set(z1, Ea).mean()
    check("7. gradient fini, non nul, et -grad augmente corr(ancre, A)",
          bool(ok and c1 > c0), f"corr_A {float(c0):+.4f} -> {float(c1):+.4f}")

    # 8. descente complete sur donnees synthetiques : la marge se RETOURNE
    torch.manual_seed(2)
    Ea3 = Fn.normalize(torch.randn(8, Dd), dim=1)
    Eb3 = Fn.normalize(torch.randn(8, Dd), dim=1)
    Eo3 = Fn.normalize(torch.randn(64, Dd), dim=1)
    w = (Eb3.mean(0, keepdim=True) + 0.05 * torch.randn(1, Dd)).requires_grad_(True)  # part de B
    opt = torch.optim.SGD([w], lr=0.5)
    m_start = None
    for it in range(200):
        opt.zero_grad()
        zz = Fn.normalize(w, dim=1)
        loss, lg = T0.t0_loss(zz, Ea3, Eb3, Eo3, kind="both", m0=0.30)
        if m_start is None:
            m_start = lg["t0_margin"]
        loss.backward()
        opt.step()
    check("8. une descente retourne la marge (marge < 0 -> marge > 0)",
          m_start < 0 < lg["t0_margin"], f"{m_start:+.4f} -> {lg['t0_margin']:+.4f}")


# ============================================================================================
def _fake_bank_payload(n_videos=6, dim_lat=16 * 8 * 8, n_a=4, n_b=4):
    """Une banque T0 synthetique minimale, avec une paire de jumeaux (corpus_idx partage)."""
    torch.manual_seed(3)
    rows = []
    for i in range(n_videos):
        cidx = i // 2 if i < 4 else 100 + i          # videos 0/1 et 2/3 sont deux paires jumelles
        rows.append({
            "video_id": f"v{i}", "corpus_idx": cidx, "twin": bool(i % 2) and i < 4,
            "mode": "switch", "split": "train" if i < 4 else "val",
            "frames_a": list(range(n_a)), "frames_b": list(range(50, 50 + n_b)),
            "switch_frame": 50,
            "lat_a": Fn.normalize(torch.randn(n_a, dim_lat), dim=1),
            "lat_b": Fn.normalize(torch.randn(n_b, dim_lat), dim=1),
            "latg_a": Fn.normalize(torch.randn(n_a, 64), dim=1),
            "latg_b": Fn.normalize(torch.randn(n_b, 64), dim=1),
        })
    return {"format": "mem2_bank_t0_v1", "n": 8, "n_a": n_a, "n_b": n_b, "skip_head": 8,
            "min_gap": 2, "with_pixel": False, "rows": rows, "meta": {}}


def test_bank(tmp):
    print("\n--- 9-10 : les negatifs ---")
    from t0_bank import T0Bank
    payload = _fake_bank_payload()
    path = os.path.join(tmp, "fake_bank_t0.pt")
    torch.save(payload, path)
    bank = T0Bank(path, torch.device("cpu"), space="latent", lat_mode="chan", verbose=False)
    meta = {"video_id": "v0"}
    Ea, Eb, Eo, Etw = bank.refs(meta, "train")
    check("9. E_a / E_b sont bien ceux de la video courante",
          bool(torch.equal(Ea, bank.E_a[0])) and bool(torch.equal(Eb, bank.E_b[0])))
    # la video courante ne doit apparaitre NULLE PART dans le pool
    own = torch.cat([bank.E_a[0], bank.E_b[0]], 0)
    hit = (Eo @ own.t()).max()
    check("9b. la video courante est EXCLUE du pool de negatifs",
          Eo.shape[0] == bank.pool["train"].shape[0] - own.shape[0] and float(hit) < 0.999,
          f"pool {bank.pool['train'].shape[0]} -> {Eo.shape[0]} lignes")
    # le jumeau (v1, meme corpus_idx) DOIT rester dans le pool: c'est LE negatif dur
    tw = torch.cat([bank.E_a[1], bank.E_b[1]], 0)
    in_pool = float((Eo @ tw.t()).max()) > 0.999
    check("10. le JUMEAU reste dans le pool de negatifs (negatif dur, F1.1b)",
          in_pool and Etw is not None and bool(torch.equal(Etw, tw)))
    ok_side = True
    try:
        bank.refs({"video_id": "v4"}, "train")        # v4 est du cote val
        ok_side = False
    except AssertionError:
        pass
    check("10b. un cote de split incoherent leve une assertion", ok_side)


def test_positions():
    print("\n--- 11-12 : les positions de retour ---")
    from t0_bank import force_end_pos
    vids = D.load_index(verbose=False)
    sw = [v for v in vids if v["mode"] == "switch"][:5]
    check("11.0 des videos switch existent dans l'index", len(sw) == 5, f"{len(sw)}")

    def draw(meta, enabled, seed=7):
        pos = D.sample_positions(meta, random.Random(seed), k=3, ages=(1, 2, 5, 10, 20),
                                 anchor_frac=0.5, min_age_writes=1)
        for p in pos:
            p["chunk_i"] = meta["write_chunks"][p["j"]]
        return force_end_pos(meta, pos, enabled=enabled)

    ok_force, ok_intact, ok_elig = True, True, True
    for meta in sw:
        pos = draw(meta, True)
        aj = D.anchor_write_index(meta)
        elig = D.anchor_eligible_states(meta)
        anc = [p for p in pos if p["kind"] == "anchor"]
        if not anc:
            continue
        p0 = anc[0]
        ok_force &= (p0["i"] == max(elig))
        ok_force &= (p0["i"] == len(meta["write_chunks"]) - 1)
        ok_intact &= (p0["j"] == aj and p0["frame"] == D.frame_of_write(meta, aj)
                      and p0["cond"] == "a" and p0["age"] == p0["i"] - aj + 1)
        ok_elig &= all(p["i"] in elig for p in anc)
    check("11. --t0_end_pos force la 1re ancre a l'ETAT DE FIN DE SEQUENCE", ok_force)
    check("11b. la cible/le cond/l'age de cette position restent coherents", ok_intact)
    check("11c. toutes les positions 'anchor' sont post-switch (regle min_gap)", ok_elig)

    same = True
    for meta in sw:
        a = draw(meta, False)
        b = D.sample_positions(meta, random.Random(7), k=3, ages=(1, 2, 5, 10, 20),
                               anchor_frac=0.5, min_age_writes=1)
        same &= [(p["i"], p["j"]) for p in a] == [(p["i"], p["j"]) for p in b]
    check("12. --t0_end_pos 0 est INERTE (tirage A2 reproduit exactement)", same)

    # determinisme du RNG dedie du gate
    d1 = D.sample_positions(sw[0], random.Random(20260804), k=2, anchor_frac=1.0)
    d2 = D.sample_positions(sw[0], random.Random(20260804), k=2, anchor_frac=1.0)
    check("12b. le RNG dedie du gate est deterministe", d1 == d2)


def test_byte_safety_static():
    """13a : verification SOURCE, sans importer train_writer (dont l'import de `wan` initialise
    un contexte CUDA). Elle garantit que les trois gardes de byte-safety sont bien en place."""
    print("\n--- 13a : byte-safety, verification source (0 GPU) ---")
    src = open(os.path.join(HERE, "train_writer.py")).read()
    check("13a.1 la banque T0 n'est chargee QUE si on la demande",
          "if args.lam_t0 > 0 or args.t0_log or args.t0_probe:" in src)
    check("13a.2 le terme T0 n'est construit QUE si lam_t0 > 0 ou --t0_log",
          "if self.t0bank is not None and (lam_t0 > 0 or a.t0_log):" in src)
    check("13a.3 le terme T0 sort immediatement si la banque est absente",
          "if self.t0bank is None or not self.t0bank.has(meta):" in src)
    check("13a.4 --rec_w ne pondere QUE la reconstruction",
          "loss = per.mean() * w * a.rec_w" in src)
    check("13a.5 --t0_end_pos delegue au helper sans CUDA",
          "return force_end_pos(meta, pos, enabled=enabled)" in src)
    # AUDIT 2026-08-04 : le defaut --t0_end_pos 1 modifiait le tirage MEME a lam_t0 = 0
    # (positions_for appelle _force_end_pos sans regarder lam_t0) => byte-safety cassee.
    check("13a.7 le forcage de la position de retour est conditionne au terme T0",
          "enabled = bool(self.args.t0_end_pos) and self.t0bank is not None" in src)
    check("13a.8 le tripwire anti-Goodhart vit dans l'espace de la METRIQUE",
          'logs["t0_anchor_std"] = float(band.detach().mean())' in src
          and "return_scale=True" in src)
    check("13a.6 le defaut de --lam_t0 est 0 et celui de --rec_w est 1.0",
          '"--lam_t0", type=float, default=0.0' in src
          and '"--rec_w", type=float, default=1.0' in src)


def test_byte_safety():
    print("\n--- 13b : byte-safety, verification FONCTIONNELLE (importe train_writer) ---")
    import train_writer as TW

    class _StubBank:
        def negatives(self, side, meta, chunks, n_neg=0, mask_chunks=2, rng=None):
            torch.manual_seed(11)
            neg = Fn.normalize(torch.randn(32, D.latent_embed_dim()), dim=1)
            return neg, torch.zeros(len(chunks), 32, dtype=torch.bool), None

        def twin_entries(self, meta):
            return None

    args = types.SimpleNamespace(subject_weight=6.0, single_w=0.3, rec_w=1.0, crop_frac=0.6,
                                 pool=2, tau=0.07, n_neg=0, mask_chunks=2, dino_log=False,
                                 t0_log=False)
    stub = types.SimpleNamespace(args=args, bank=_StubBank(), dino_emb=None, t0bank=None)
    torch.manual_seed(5)
    pred = torch.randn(2, 1, 16, 60, 104)
    tgt = torch.randn(2, 1, 16, 60, 104)
    pos = [{"chunk_i": 5, "kind": "anchor"}, {"chunk_i": 9, "kind": "recency"}]
    meta = {"video_id": "v0", "mode": "switch"}
    loss, per, logs, _ = TW.WriterTrainer.video_losses(stub, meta, pos, pred, tgt, "train",
                                                       0.3, lam_t0=0.0)
    # la valeur de reference: EXACTEMENT la recette A2 (reco + lam * InfoNCE latent)
    ref_per = TW.norm_mse_per(pred, tgt, subject_weight=6.0)
    z = D.latent_embed(pred, frac=0.6, pool=2)
    pe = D.latent_embed(tgt, frac=0.6, pool=2).detach()
    neg, mask, _ = _StubBank().negatives("train", meta, [5, 9])
    li, _ = TW.infonce(z, pe, neg, 0.07, mask)
    ref = ref_per.mean() * 1.0 + 0.3 * li * 1.0
    check("13b.1 lam_t0 = 0 (banque absente) => perte EXACTEMENT la recette A2",
          abs(float(loss) - float(ref)) < 1e-6, f"{float(loss):.8f} vs {float(ref):.8f}")
    check("13b.2 aucune cle t0/* n'est loggee quand le terme est inactif",
          not any(k == "t0" or k.startswith("t0_") for k in logs), f"{sorted(logs)}")
    args.rec_w = 0.3
    loss2, _, _, _ = TW.WriterTrainer.video_losses(stub, meta, pos, pred, tgt, "train", 0.3,
                                                   lam_t0=0.0)
    check("13b.3 --rec_w ne pondere QUE la reconstruction",
          abs(float(loss2) - float(0.3 * ref_per.mean() + 0.3 * li)) < 1e-6,
          f"{float(loss2):.6f}")


def test_ref_frames():
    print("\n--- 14-15 : les references de la banque, sur le VRAI corpus ---")
    from build_bank_t0 import ref_frames
    vids = D.load_index(verbose=False)
    sw = [v for v in vids if v["mode"] == "switch"]
    ok_a, ok_b, ok_head, ok_gap, ok_det, n = True, True, True, True, True, 0
    for meta in sw[:40]:
        fa, fb = ref_frames(meta, 8, 8, skip_head=8, min_gap=2)
        if fa is None:
            continue
        n += 1
        sf = meta["switch_frame"]
        ok_a &= max(fa) < sf
        ok_b &= min(fb) >= sf
        ok_head &= min(fa) >= 8
        ok_gap &= max(fa) <= sf - 1 - 2                # la CIBLE de reco (sf-1) n'est pas positif
        ok_det &= (ref_frames(meta, 8, 8, 8, 2) == (fa, fb))
    check("14. A entierement AVANT le switch", ok_a, f"{n} videos")
    check("14b. B entierement APRES le switch", ok_b)
    check("14c. l'ouvreur est saute (skip_head)", ok_head)
    check("14d. la frame CIBLE de la reco n'est pas un positif T0 (min_gap)", ok_gap)
    check("14e. la selection est deterministe", ok_det)

    # 15. D0.b en miniature : une VRAIE frame de A est plus proche de A que de B, en latent
    cache = D.PartCache(size=2)
    res = {"chan": [], "gray": []}
    done = 0
    for meta in sw:
        fa, fb = ref_frames(meta, 8, 8)
        if fa is None:
            continue
        la = D.load_frames(meta, cache, fa)
        lb = D.load_frames(meta, cache, fb)
        for mode in ("chan", "gray"):
            Ea = T0.lowband_lat(la, n=8, mode=mode)
            Eb = T0.lowband_lat(lb, n=8, mode=mode)
            mid = Ea.shape[0] // 2
            z = Ea[mid:mid + 1]
            others = torch.cat([Ea[:mid], Ea[mid + 1:]], 0)
            res[mode].append(float(T0.corr_set(z, others) - T0.corr_set(z, Eb)))
        done += 1
        if done >= 8:
            break
    for mode, v in res.items():
        frac = sum(1 for x in v if x > 0) / max(1, len(v))
        check(f"15. latent/{mode}: une VRAIE frame de A est plus proche de A que de B",
              frac >= 0.75, f"{sum(1 for x in v if x > 0)}/{len(v)}, "
                            f"marge moyenne {sum(v) / max(1, len(v)):+.4f}")


def test_arm():
    """17-18 : le bras `memdec_t0` du harness. CPU (scenarios.py n'importe pas de CUDA)."""
    print("\n--- 17-18 : le bras memdec_t0 et la campagne G3 ---")
    import scenarios as S
    check("17. memdec_t0 a EXACTEMENT l'environnement de memdec20 (meme chemin de code)",
          S.SCEN_ARM_ENV["memdec_t0"] == S.SCEN_ARM_ENV["memdec20"],
          "seuls les POIDS changent")
    check("17b. memdec_t0 tourne sur le pipeline MAG", "memdec_t0" in S.MAG_SCEN_ARMS)
    dump_t0 = f"{S.DEC_ANCHOR_DIR}/S2_t2_s0_memdec_t0"
    check("17c. le dump d'ancre de memdec_t0 n'ECRASE PAS la source de wronganchor",
          dump_t0 != S.dec_anchor_base("S2", 2, 0)
          and not S.wrong_anchor_src("S2", 2, 0).startswith(dump_t0))
    src = open(os.path.join(SCEN_DIR, "run_reid_scen.py")).read()
    check("17d. les bras T0 lisent LEURS ckpts (jamais de repli silencieux sur A2)",
          '_is_t0 = arm in S.T0_ARMS' in src
          and 'getattr(config, "_mem2_t0_dec_ckpt", "") if _is_t0' in src
          and 'getattr(config, "_mem2_t0_writer_ckpt", "") if _is_t0' in src)
    check("17e. le dump d'ancre de memdec_t0 est arme dans le driver",
          'if arm in S.T0_ARMS:' in src and '_{arm}")' in src)

    # 18 : les comparateurs de G3 sont deja sur disque -> seul memdec_t0 est a tirer (6 rollouts)
    vid = "/shared/ssd_14T_project/adenelmi/reid_scen/videos"
    cells = [("S2", 2), ("S2", 4), ("S2", 11), ("SS2", 12), ("SS2", 13), ("SS2", 14)]
    miss = [f"{sc}_t{t}_s0_{a}.mp4" for sc, t in cells for a in ("memdec20", "wronganchor")
            if not os.path.exists(os.path.join(vid, f"{sc}_t{t}_s0_{a}.mp4"))]
    check("18. les 12 mp4 comparateurs de G3 (memdec20 + wronganchor) sont sur disque",
          not miss, f"manquants: {miss[:4]}" if miss else "6 cellules x 2 bras")
    clean = [f"{c}_t{t}_s0_memzero.mp4" for (sc, t) in cells
             for c in [{"S2": "S3", "SS2": "SS1"}[sc]]
             if not os.path.exists(os.path.join(vid, f"{c}_t{t}_s0_memzero.mp4"))]
    check("18b. les references A* PROPRES (S3/SS1 memzero) sont sur disque",
          not clean, f"manquants: {clean[:4]}" if clean else "6 cellules")


def test_spearman():
    print("\n--- 16 : l'aide statistique ---")
    from space_decision_t0 import spearman
    a = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
    check("16. Spearman(x, x) == 1", abs(spearman(a, a) - 1.0) < 1e-9)
    check("16b. Spearman(x, -x) == -1", abs(spearman(a, -a) + 1.0) < 1e-9)
    # cas ferme : d = (0,-1,1,-1,1), sum d^2 = 4, rho = 1 - 6*4/(5*(25-1)) = 0.8
    b = torch.tensor([1.0, 3.0, 2.0, 5.0, 4.0])
    check("16c. Spearman sur un cas connu (rho = 0.8)", abs(spearman(a, b) - 0.8) < 1e-6,
          f"{spearman(a, b):.6f}")
    c = torch.tensor([1.0, 1.0, 2.0, 2.0, 3.0])          # ex aequo : rangs moyennes
    check("16d. Spearman gere les ex aequo sans NaN",
          abs(spearman(c, c) - 1.0) < 1e-9 and spearman(a, c) == spearman(a, c))


def main():
    import tempfile
    # --with_train_writer : importe train_writer (donc `wan`, donc un CONTEXTE CUDA sur la carte
    # visible ~0.3 GB). Hors de ce drapeau, ce fichier ne touche AUCUN GPU et le meme contrat de
    # byte-safety est verifie au niveau SOURCE (13a).
    with_tw = "--with_train_writer" in sys.argv
    print("=" * 92)
    print("SELFTESTS T0 (memoire 2.1) -- CPU uniquement"
          + ("  (+ import train_writer : contexte CUDA)" if with_tw else ""))
    print("=" * 92)
    with tempfile.TemporaryDirectory() as tmp:
        test_metric()
        test_loss()
        test_bank(tmp)
        test_positions()
        test_byte_safety_static()
        if with_tw:
            test_byte_safety()
        else:
            print("\n--- 13b : SAUTE (relancer avec --with_train_writer et un GPU visible) ---")
        test_ref_frames()
        test_arm()
        test_spearman()
    print("\n" + "=" * 92)
    print(f"{_N[0] - len(_FAILS)}/{_N[0]} PASS"
          + ("" if not _FAILS else f"   ECHECS: {', '.join(_FAILS)}"))
    print("=" * 92, flush=True)
    sys.exit(0 if not _FAILS else 1)


if __name__ == "__main__":
    main()
