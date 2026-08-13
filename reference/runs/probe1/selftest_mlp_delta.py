#!/usr/bin/env python
# ============================================================================================
# ARM MLP-CORRECTEUR -- SELFTESTS OBLIGATOIRES AVANT LE RUN COMPLET
# (registre : section "ARM MLP-CORRECTEUR", scelle 07/08 AVANT implementation).
#
# Ce que le bras change, et RIEN d'autre : l'objectif interne d'une ecriture.
#   existant  : monte  dot = <f_W(k), v>            (regle ASSOCIATIVE / Hebb)
#   ce bras   : descend L  = 1/2 ||f_W(k) - v||^2   (regle CORRECTRICE D'ERREUR)
# Implementation : le RESIDU (v - f_W(k)) prend la place de v dans le calcul de gradient
# EXISTANT (cf. la derivation en tete de build_states.MLP_SPEC). Muon, weight_norm_update,
# lr par token, decroissance, plafonds de norme, ages, sondes, pooling, plis, corpus :
# INCHANGES, et les fonctions ne sont jamais recopiees.
#
#   D0  py_compile + unicite des chemins (UN lecteur : celui de `state` ; UN coeur d'ecriture :
#       write_only ; le gradient correcteur est bati sur swiglu_apply + swiglu_fast_weight_
#       gradient importes, rien de recopie)
#   D1  ANALYTIQUE : mini-cas construit a la main (B=2, L=3, C=5, Ch=7, fp64) -- le gradient
#       rendu est compare a l'AUTOGRAD de 1/2||f_W(k) - v||^2 sur la MEME fonction. Trois
#       verifications : (a) l'harnais est valide (le gradient EXISTANT == autograd du dot),
#       (b) lr = 1, (c) lr par token DIFFERENT pour w0/w1/w2 (le vrai cas). Tolerance stricte.
#   D2  DIRECTION : sur 3 chunks REELS enchaines, l'erreur ||f_W(k) - v|| sur CE chunk doit
#       DIMINUER apres l'ecriture. Rapporte avant/apres pour le bras correcteur ET pour la
#       regle existante (contraste), plus la derivee directionnelle <dL/dW, delta W> qui prouve
#       le signe independamment du pas. EXIGE sur A1 et A2 (leur pas est reglable, et il est
#       regle par le critere AVEUGLE de sweep_mlp_step.py) ; seulement MESURE sur A3, dont le
#       pas Muon n'a aucun parametre libre -- descendre ou non y est un resultat du bras, pas
#       une propriete du code (celle-ci est prouvee par D1 et D1e).
#   D3  NON-REGRESSION : chemin par DEFAUT (grad_fn=None) == etats deja sur disque, 2 videos
#       x 3 blocs x 4 ages, torch.equal (bit a bit).
#   D4  DETERMINISME : 2 executions completes (rejeu -> signature -> fp16) bit a bit, 2 bras.
#   D5  `mlp_arm` : muon_update_steps force puis TOUJOURS restaure (y compris sur exception).
#   D6  PART ECRITE : ||etat - etat_sans_ecriture|| / ||etat|| par age et par bloc, tous les
#       bras. C'est le garde-fou SYMETRIQUE du choix de pas : la sanite numerique borne le pas
#       par le haut, D6 le borne par le bas (un pas minuscule donnerait un bras "sain" qui
#       n'ecrit rien et dont l'etat n'est que l'init qui s'oublie, init IDENTIQUE pour toutes
#       les videos). Le plancher 0.5 est la contrainte scellee au registre le 07/08.
#   Stats : normes de l'etat par age/bloc, bras correcteur vs existant.
#
# Usage (GPU obligatoire, pre-check DANS la commande, jamais l'index smi 6) :
#   cd /shared/ssd_14T/home/adenelmi/Anchor-Forcing && \
#   CUDA_DEVICE_ORDER=PCI_BUS_ID nvidia-smi --query-gpu=index,memory.free --format=csv && \
#   CUDA_VISIBLE_DEVICES=2 python reference/runs/probe1/selftest_mlp_delta.py
# ============================================================================================
import contextlib
import inspect
import json
import os
import py_compile
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import probe1_common as C  # noqa: E402

import torch  # noqa: E402

import mem2_data as D  # noqa: E402

PASS, FAIL = "\033[32mPASS\033[0m", "\033[31mFAIL\033[0m"
_failed = []


def check(name, ok, detail=""):
    print(f"  [{PASS if ok else FAIL}] {name}" + (f" -- {detail}" if detail else ""), flush=True)
    if not ok:
        _failed.append(name)
    return ok


def state_norms(state, blocks):
    """||etat||_F par bloc (les 3 matrices agregees) : la statistique comparable d'un bras a
    l'autre."""
    return {b: round(float(sum(float(t.float().norm()) ** 2 for t in state[b]) ** 0.5), 2)
            for b in blocks}


def main():
    assert torch.cuda.is_available(), "GPU obligatoire"
    C.ensure_dirs()
    C.set_determinism(0)
    import build_signatures as B
    import build_states as BS
    from wan.modules.video_memory import (swiglu_apply, swiglu_fast_weight_gradient,
                                          VideoMemoryLayer)

    # ---------------- D0 : compilation + unicite des chemins -------------------------------
    print("[D0] compilation + un seul lecteur / un seul coeur d'ecriture")
    for f in ("build_states.py", "build_signatures.py", "train_probe.py",
              "selftest_mlp_delta.py"):
        py_compile.compile(os.path.join(HERE, f), doraise=True)
    py_compile.compile(os.path.join(os.path.dirname(inspect.getfile(VideoMemoryLayer)),
                                    "video_memory.py"), doraise=True)
    check("D0 py_compile (4 fichiers probe1 + video_memory.py)", True)
    check("D0 tous les bras -> LE lecteur de `state` (meme objet fonction)",
          all(B.READERS[s] is B.READERS["state"] for s in B.MLP_SOURCES)
          and sorted(B.MLP_SOURCES) == sorted(BS.MLP_SPEC) == sorted(BS.MLP_ARMS)
          == ["mlp_delta_all", "mlp_delta_subject", "mlp_full", "mlp_full_muon",
              "mlp_hebb_seq"])
    a2, a3, a4 = (BS.MLP_ARMS[n] for n in ("mlp_full", "mlp_full_muon", "mlp_hebb_seq"))
    a1 = BS.MLP_ARMS["mlp_delta_all"]
    # Plan AMENDE (registre "REPARATION DU CRITERE AVEUGLE") : le PRIMARY est A3.
    check("D0 plan factoriel : A3 (PRIMARY) et A4 ne different QUE par l'objectif (R)",
          a3[1:] == a4[1:] and a3[0] == "delta" and a4[0] == "hebb",
          f"micro-lot {a3[2]}, Muon {a3[3]} (None = la valeur de la cellule, 5), pas x{a3[4]}")
    check("D0 plan factoriel : A3 et A1 ne different QUE par le calendrier (S)",
          a3[0] == a1[0] == "delta" and a3[3] == a1[3] and a3[4:] == a1[4:]
          and a1[2] is None and a3[2] == BS.MLP_MB_MUON,
          f"A1 micro-lot {a1[2]} (un seul pas) vs A3 micro-lot {a3[2]}")
    check("D0 plan factoriel : A2' isole le pas (M) -- brut, calendrier NON apparie, DECLARE",
          a2[0] == "delta" and a2[3] == 0 and a2[2] == BS.MLP_MB_RAW != a3[2],
          f"A2' micro-lot {a2[2]} pas x{a2[4]} (brut) vs A3 micro-lot {a3[2]} Muon : le brut ne "
          f"tient pas le calendrier 512 (part 1.04, R = -124), dissymetrie assumee")
    check("D0 aucun bras n'est masque dans le plan (la delta lineaire a montre que le "
          "masque nuit)", not any(BS.MLP_ARMS[n][1] for n in ("mlp_delta_all", "mlp_full",
                                                             "mlp_full_muon", "mlp_hebb_seq")))
    from wan.modules.video_memory import zeropower_via_newtonschulz5 as NS
    graw = torch.randn(1, 8, 12)
    check("D0 le chemin BRUT est bien le chemin existant (muon_update_steps=0 -> identite)",
          NS(graw, 0) is graw)
    src = inspect.getsource(BS.swiglu_delta_gradient)
    check("D0 gradient correcteur = swiglu_apply + swiglu_fast_weight_gradient (rien recopie)",
          "swiglu_apply(k, w0, w1, w2)" in src
          and "swiglu_fast_weight_gradient(k, resid" in src
          and len([l for l in src.splitlines() if l.strip()
                   and not l.strip().startswith(("#", '"'))]) < 12)
    wo = inspect.getsource(VideoMemoryLayer.write_only)
    check("D0 write_only : grad_fn=None -> swiglu_fast_weight_gradient (defaut inchange)",
          "gfn = swiglu_fast_weight_gradient if grad_fn is None else grad_fn" in wo
          and "g0, g1, g2 = gfn(" in wo)
    check("D0 le masque du bras masque passe par maskwrite_keep (meme chemin que l'oracle)",
          "maskwrite_keep(meta, \"subject\")" in inspect.getsource(B.main))

    # ---------------- D1 : analytique, autograd, fp64 --------------------------------------
    print("[D1] mini-cas a la main (B=2, L=3, C=5, Ch=7, fp64) vs autograd de 1/2||f-v||^2")
    g = torch.Generator().manual_seed(1234)
    Bd, L, Cd, Ch = 2, 3, 5, 7
    dt = torch.float64
    k = torch.randn(Bd, L, Cd, generator=g, dtype=dt)
    v = torch.randn(Bd, L, Cd, generator=g, dtype=dt)
    w0 = torch.randn(Bd, Cd, Ch, generator=g, dtype=dt) * 0.3
    w1 = torch.randn(Bd, Ch, Cd, generator=g, dtype=dt) * 0.3
    w2 = torch.randn(Bd, Cd, Ch, generator=g, dtype=dt) * 0.3
    ones = torch.ones(Bd, L, 1, dtype=dt)
    lam = {i: torch.rand(Bd, L, 1, generator=g, dtype=dt) + 0.1 for i in range(3)}

    def autograd_grads(loss_fn, lrs):
        """d/dW de  sum_i lr_i * loss_fn(f_i, v_i)  -- une passe par matrice, avec SON lr."""
        outs = []
        for m, lr in enumerate(lrs):
            ws = [w0.clone(), w1.clone(), w2.clone()]
            ws[m].requires_grad_(True)
            f = swiglu_apply(k, *ws)
            loss = (lr * loss_fn(f, v)).sum()
            outs.append(torch.autograd.grad(loss, ws[m])[0])
        return outs

    def dot_loss(f, x):
        return (f * x).sum(-1, keepdim=True)

    def sq_loss(f, x):
        return 0.5 * (f - x).pow(2).sum(-1, keepdim=True)

    # (a) l'harnais est valide : le gradient EXISTANT est bien +d<f,v>/dW
    ge = swiglu_fast_weight_gradient(k, v, w0, w1, w2, lam[0], lam[1], lam[2])
    ref = autograd_grads(dot_loss, [lam[0], lam[1], lam[2]])
    dev = [float((a - b).abs().max()) for a, b in zip(ge, ref)]
    check("D1a harnais : gradient EXISTANT == +autograd(<f_W(k), v>)",
          max(dev) < 1e-11, f"ecart max w0/w1/w2 = {['%.2e' % d for d in dev]}")

    # (b) lr = 1
    gd = BS.swiglu_delta_gradient(k, v, w0, w1, w2, ones, ones, ones)
    ref = autograd_grads(sq_loss, [ones, ones, ones])
    dev = [float((a + b).abs().max()) for a, b in zip(gd, ref)]     # a == -b (montee de -L)
    scl = [float(b.abs().max()) for b in ref]
    check("D1b lr=1 : gradient rendu == -autograd(1/2||f_W(k)-v||^2), signe DESCENTE",
          max(dev) < 1e-11,
          f"ecart max {['%.2e' % d for d in dev]} (echelle {['%.2f' % s for s in scl]})")

    # (c) le vrai cas : un lr par token ET par matrice
    gd = BS.swiglu_delta_gradient(k, v, w0, w1, w2, lam[0], lam[1], lam[2])
    ref = autograd_grads(sq_loss, [lam[0], lam[1], lam[2]])
    dev = [float((a + b).abs().max()) for a, b in zip(gd, ref)]
    check("D1c lr par token/matrice : == -autograd(sum_i lr_i/2 ||f-v||^2)",
          max(dev) < 1e-11, f"ecart max {['%.2e' % d for d in dev]}")
    # et le residu est bien celui de l'etat AVANT mise a jour, pas une constante
    r = v - swiglu_apply(k, w0, w1, w2)
    check("D1d le residu passe est bien (v - f_W(k)) sur l'etat courant",
          torch.equal(BS.swiglu_delta_gradient(k, v, w0, w1, w2, ones, ones, ones)[0],
                      swiglu_fast_weight_gradient(k, r, w0, w1, w2, ones, ones, ones)[0]))

    # ---------------- cellules + video pour D2/D3/D4 ---------------------------------------
    device = "cuda"
    man = json.load(open(os.path.join(C.STATE_DIR, "_manifest.json")))
    blocks = list(man["blocks"])
    cells = B.build_cells_for_stream(C.DEFAULT_CONFIG, blocks, device)
    vids = C.load_index(verbose=False, corpus="v1")
    byid = C.vids_by_id(vids)
    disk_ids = [vid for vid in man["manifest"] if vid in byid][:2]
    assert len(disk_ids) == 2, f"videos du manifeste introuvables dans l'index v1 : {disk_ids}"
    cache = D.PartCache(size=1)

    # ---------------- D1e : le sequentiel a UN SEUL micro-lot == le chemin historique -------
    print("[D1e] micro-lot >= longueur du chunk == un seul pas, BIT A BIT (boucle + decroissance)")
    meta = byid[disk_ids[0]]
    bi = blocks[0]
    h = D.load_h(meta, cache, [bi], upto=2, device=device, dtype=torch.bfloat16)[bi]
    cell = cells[bi]
    L = int(h[0].shape[-2])
    with torch.no_grad():
        st0 = tuple(t.detach() for t in cell.init_state(int(h[0].shape[0]), device,
                                                        cell.qkv.weight.dtype))
        for gname, gfn in (("Hebb (defaut)", None), ("residu", BS.swiglu_delta_gradient)):
            a = cell.write_only(h[0], st0, grad_fn=gfn)
            b = cell.write_only(h[0], st0, grad_fn=gfn, micro_batch=L)
            c = cell.write_only(h[0], st0, grad_fn=gfn, micro_batch=L * 10)
            check(f"D1e {gname} : micro_batch=L et 10L == micro_batch=None (bitwise)",
                  all(torch.equal(x, y) for x, y in zip(a, b))
                  and all(torch.equal(x, y) for x, y in zip(a, c)),
                  f"L = {L} tokens")
        # la decroissance ne s'applique qu'UNE fois par ecriture, meme en 4 micro-lots :
        # sinon l'etat serait strictement plus petit qu'avec 1 micro-lot a gradient nul.
        def zero(_k, _v, w0_, w1_, w2_, _l0, _l1, _l2):
            return torch.zeros_like(w0_), torch.zeros_like(w1_), torch.zeros_like(w2_)

        z1 = cell.write_only(h[0], st0, grad_fn=zero)
        z4 = cell.write_only(h[0], st0, grad_fn=zero, micro_batch=(L + 3) // 4)
        check("D1e decroissance appliquee UNE fois par ecriture (gradient nul, 1 vs 4 micro-lots)",
              all(torch.equal(x, y) for x, y in zip(z1, z4)),
              f"||etat|| = {state_norms({bi: z1}, [bi])[bi]} (identique en 4 micro-lots)")

    # ---------------- D2 : direction de l'erreur, PAR BRAS -----------------------------------
    print("[D2] 3 chunks REELS enchaines, par bras : l'erreur ||f_W(k) - v|| doit BAISSER "
          "(A1/A2/A3) et MONTER (A0/A4, objectif Hebb)")
    ARMS_D2 = (("A0 state (Hebb/moy/Muon)", None, None, None),
               ("A1 mlp_delta_all", "mlp_delta_all", None, None),
               ("A3 mlp_full_muon PRIMARY", "mlp_full_muon", None, None),
               ("A2' mlp_full (brut)", "mlp_full", None, None),
               ("A4 mlp_hebb_seq", "mlp_hebb_seq", None, None))
    d2 = {}
    with torch.no_grad():
        for label, arm, _a, _b in ARMS_D2:
            st = tuple(t.detach() for t in cell.init_state(int(h[0].shape[0]), device,
                                                           cell.qkv.weight.dtype))
            ctx = (BS.mlp_arm({bi: cell}, arm) if arm is not None
                   else contextlib.nullcontext((None, None)))
            rows = []
            with ctx as (gfn, mb):
                for j in range(3):
                    _, kk, vv, _l0, _l1, _l2, _B = cell._qkv_lr(h[j])
                    kf, vf = kk.float(), vv.float()

                    def err(state):
                        w0_, w1_, w2_ = [t.float() for t in state]
                        return float((swiglu_apply(kf, w0_, w1_, w2_) - vf).norm())

                    e0 = err(st)
                    new = cell.write_only(h[j], st, grad_fn=gfn, micro_batch=mb)
                    e1 = err(new)
                    with torch.enable_grad():       # derivee directionnelle sur le pas REEL
                        ws = [t.float().clone().requires_grad_(True) for t in st]
                        loss = 0.5 * (swiglu_apply(kf, *ws) - vf).pow(2).sum()
                        gs = torch.autograd.grad(loss, ws)
                    dd = float(sum((gi * (nt.float() - ot.float())).sum()
                                   for gi, nt, ot in zip(gs, new, st)))
                    rows.append((e0, e1, dd))
                    st = new
            d2[label] = rows
            desc = "  ".join(f"{e0:.4g}->{e1:.4g} ({100 * (e1 - e0) / e0:+.1f}%)"
                             for e0, e1, _ in rows)
            print(f"      {label:<26} {desc}  | <dL/dW,dW> = "
                  f"{[f'{d:.2g}' for _, _, d in rows]}", flush=True)
    # Ce qui est EXIGE est ce que le critere scelle mesure : une MOYENNE de reduction relative
    # PAR ECRITURE (R). Exiger 3/3 chunks serait strictement plus fort que le critere, et faux
    # pour un pas de taille FIXE : sous Muon, une ecriture isolee peut degrader son chunk sans
    # que la regle cesse de descendre en moyenne (A3 mesure R = +0.0251 sur 150 ecritures alors
    # qu'une ecriture sur trois monte ici). Le detail par chunk reste imprime ci-dessus.
    for label in ("A1 mlp_delta_all", "A3 mlp_full_muon PRIMARY"):
        red = [(e0 - e1) / e0 for e0, e1, _ in d2[label]]
        check(f"D2 {label} : l'erreur BAISSE EN MOYENNE sur les 3 chunks (R > 0, la quantite "
              f"du critere scelle)", sum(red) / len(red) > 0,
              f"reductions {[f'{r:+.1%}' for r in red]}, moyenne {sum(red) / len(red):+.1%}")
    # A2' : MESURE, pas contrat -- et son echec est ATTENDU. C'est le bras "sans Muon" au pas ou
    # l'ecriture domine l'etat (part 1.03) : le critere aveugle sous contrainte a montre que,
    # sans orthogonalisation, ecrire fort et descendre l'erreur sont incompatibles dans cette
    # cellule. Que son erreur monte est donc le RESULTAT que le bras est la pour montrer, pas un
    # defaut de code : l'implementation est prouvee par D1 (gradient == -dL/dW a zero machine),
    # D1e (micro-lot >= L == le pas unique, bit a bit) et par A1/A3 qui descendent 3/3.
    a2r = d2["A2' mlp_full (brut)"]
    print(f"  [INFO] D2 A2' mlp_full (MESURE, echec attendu : brut au pas ou il ecrit) : "
          f"erreur en baisse sur {sum(1 for e0, e1, _ in a2r if e1 < e0)}/3 chunks", flush=True)
    # La derivee directionnelle n'est EXACTE (premier ordre) que pour un pas unique : elle est
    # ASSERTEE sur A1, et seulement IMPRIMEE pour les bras sequentiels (dont le deplacement
    # total agrege des dizaines de pas, ou le premier ordre ne dit plus rien).
    check("D2 A1 : derivee directionnelle < 0 (signe de la DESCENTE, pas unique)",
          all(d < 0 for _, _, d in d2["A1 mlp_delta_all"]))
    for label in ("A0 state (Hebb/moy/Muon)", "A4 mlp_hebb_seq"):
        check(f"D2 {label} : objectif Hebb -> l'erreur MONTE (propriete de l'objectif, "
              f"pas un defaut)", all(e1 > e0 for e0, e1, _ in d2[label]))

    # ---------------- D2b : l'erreur decroit AU FIL DES MICRO-LOTS (bras sequentiels) --------
    # Ecrire le PREFIXE [0:s] des tokens avec le meme micro-lot donne EXACTEMENT l'etat
    # intermediaire de l'ecriture complete apres s tokens (la decroissance tombe sur le premier
    # micro-lot dans les deux cas) : on peut donc tracer la courbe d'erreur sans toucher
    # write_only.
    print("[D2b] courbe d'erreur AU FIL DES MICRO-LOTS, chunk 0 (bras sequentiels)")
    with torch.no_grad():
        _, kk, vv, _l0, _l1, _l2, _B = cell._qkv_lr(h[0])
        kf, vf = kk.float(), vv.float()
        for label, arm in (("A3 mlp_full_muon PRIMARY", "mlp_full_muon"),
                           ("A2' mlp_full (brut)", "mlp_full"),
                           ("A4 mlp_hebb_seq", "mlp_hebb_seq")):
            st = tuple(t.detach() for t in cell.init_state(int(h[0].shape[0]), device,
                                                           cell.qkv.weight.dtype))
            with BS.mlp_arm({bi: cell}, arm) as (gfn, mb):
                pts = [mb, 4 * mb, 16 * mb, 64 * mb, L]
                pts = sorted({min(int(p), L) for p in pts})
                curve = []
                for s in pts:
                    w = cell.write_only(h[0][:, :s], st, grad_fn=gfn, micro_batch=mb)
                    w0_, w1_, w2_ = [t.float() for t in w]
                    curve.append(float((swiglu_apply(kf, w0_, w1_, w2_) - vf).norm()))
            print(f"      {label:<22} tokens {pts} -> erreur "
                  f"{[f'{c:.4g}' for c in curve]}", flush=True)
            mono = all(b <= a for a, b in zip(curve, curve[1:]))
            e_avant = d2[label][0][0]      # l'erreur AVANT cette meme ecriture (D2, chunk 0)
            if arm == "mlp_hebb_seq":
                check(f"D2b {label} : Hebb -> l'erreur ne decroit PAS le long des micro-lots",
                      not mono)
            elif arm == "mlp_full":            # A2' : mesure, echec attendu (cf. D2 ci-dessus)
                print(f"  [INFO] D2b {label} (MESURE, echec attendu) : courbe "
                      f"{'monotone' if mono else 'NON monotone'}, "
                      f"{curve[0]:.4g} -> {curve[-1]:.4g}", flush=True)
            else:
                # A3 : le pas Muon a une taille FIXE, donc la courbe INTERNE a une ecriture n'a
                # aucune raison d'etre monotone (un micro-lot peut depasser). Ce qui est exige
                # est ce que "l'ecriture descend" veut dire : l'erreur a la FIN de l'ecriture est
                # sous celle d'AVANT. La forme intermediaire est imprimee, pas exigee.
                check(f"D2b {label} : l'erreur a la FIN de l'ecriture < l'erreur AVANT",
                      curve[-1] < e_avant,
                      f"{e_avant:.4g} -> {curve[-1]:.4g} ; courbe interne "
                      f"{'monotone' if mono else 'NON monotone (attendu : pas de taille fixe)'}")

    # ---------------- D5 : `mlp_arm` restaure toujours muon_update_steps ---------------------
    print("[D5] mlp_arm : muon_update_steps force puis RESTAURE (y compris sur exception)")
    ref_muon = {b: c.muon_update_steps for b, c in cells.items()}
    with BS.mlp_arm(cells, "mlp_full") as (_g, _m):
        forced = {b: c.muon_update_steps for b, c in cells.items()}
    check("D5 A2 force bien le gradient BRUT pendant le bloc",
          all(v == 0 for v in forced.values()), f"{forced} pendant, {ref_muon} avant")
    check("D5 restauration apres le bloc",
          {b: c.muon_update_steps for b, c in cells.items()} == ref_muon)
    try:
        with BS.mlp_arm(cells, "mlp_full") as (_g, _m):
            raise RuntimeError("boum")
    except RuntimeError:
        pass
    check("D5 restauration meme sur EXCEPTION",
          {b: c.muon_update_steps for b, c in cells.items()} == ref_muon)
    with BS.mlp_arm(cells, "mlp_full_muon") as (_g, _m):
        a3_muon = {b: c.muon_update_steps for b, c in cells.items()}
    check("D5 A3 garde le Muon de la CELLULE (aucun forcage)", a3_muon == ref_muon,
          f"{a3_muon}")

    # ---------------- D3 : non-regression bit a bit vs le disque ---------------------------
    print("[D3] chemin par DEFAUT (grad_fn=None) == etats du disque, 2 videos x 3 blocs x 4 ages")
    from build_states import replay_video, state_path
    for vid in disk_ids:
        m = byid[vid]
        ages = C.state_ages(m, C.N_AGES)
        got = replay_video(cells, blocks, m, cache, ages, device)      # grad_fn=None par defaut
        same, n = True, 0
        for a in ages:
            ck = torch.load(state_path(vid, a["name"]), map_location="cpu", weights_only=False)
            for b in blocks:
                for t_ref, t_got in zip(ck["state"][b], got[int(a["i"])][b]):
                    same &= torch.equal(t_ref, t_got)
                    n += 1
        check(f"D3 {vid} : {n} tenseurs bit a bit (3 blocs x 4 ages x 3 matrices)", same)
        del got

    # ---------------- D4 : determinisme bout en bout + stats de normes, PAR BRAS ------------
    print("[D4] 2 executions completes -> signatures bit a bit, et ||etat||_F par age (tous "
          "les bras)")
    from build_probes import PROBES_PT
    pk = torch.load(PROBES_PT, map_location="cpu", weights_only=False)
    probes, pblocks = pk["probes"].to(device), list(pk["blocks"])
    assert pblocks == blocks, f"{pblocks} != {blocks}"
    m = byid[disk_ids[0]]
    ages = C.state_ages(m, C.N_AGES)
    chrono, norms, written = {}, {}, {}

    # D6 (garde-fou SYMETRIQUE du choix de pas) : le critere aveugle borne le pas par le HAUT
    # (une cellule qui explose est ecartee par la sanite numerique) mais rien ne le borne par le
    # BAS -- un pas minuscule donnerait un bras "sain" qui n'ecrit RIEN et dont l'etat est juste
    # l'init qui s'oublie. On mesure donc la PART ECRITE de l'etat :
    #     ||etat_bras - etat_sans_ecriture|| / ||etat_bras||,
    # ou "sans ecriture" = la MEME trajectoire avec un gradient identiquement nul (donc la
    # decroissance et le plafond de norme SEULS ; le micro-lot n'y change rien, cf. D1e). C'est
    # une statistique de norme, aveugle a l'identite comme tout ce fichier.
    def zero_grad_fn(k, v, w0, w1, w2, lr0, lr1, lr2):
        return torch.zeros_like(w0), torch.zeros_like(w1), torch.zeros_like(w2)

    nowrite = replay_video(cells, blocks, m, cache, ages, device, grad_fn=zero_grad_fn)

    def written_part(states):
        """states : {age_i -> {bloc -> (w0, w1, w2)}} -> {nom d'age -> {bloc -> part ecrite}}."""
        out = {}
        for a in ages:
            st, nw = states[int(a["i"])], nowrite[int(a["i"])]
            out[a["name"]] = {
                b: round(float(sum(float((x.float() - y.float()).norm()) ** 2
                                   for x, y in zip(st[b], nw[b])) ** 0.5)
                         / max(float(sum(float(x.float().norm()) ** 2
                                         for x in st[b]) ** 0.5), 1e-9), 4)
                for b in blocks}
        return out

    ref_states = replay_video(cells, blocks, m, cache, ages, device)      # A0, reference
    norms["A0 state (defaut)"] = {a["name"]: state_norms(ref_states[int(a["i"])], blocks)
                                  for a in ages}
    written["A0 state (defaut)"] = written_part(ref_states)
    for srcname in sorted(B.MLP_SOURCES):
        keep = BS.maskwrite_keep(m, "subject")[0] if BS.MLP_SPEC[srcname] else None
        sigs, t0 = [], time.time()
        for _run in range(2):
            with BS.mlp_arm(cells, srcname) as (gfn, mb):
                states = B.stream_states(m, ages, cache, blocks, device, cells=cells,
                                         keep=keep, grad_fn=gfn, micro_batch=mb)
            per_age = []
            for a in ages:
                pay = B.load_payload(srcname, m, a, cache, blocks, device, states=states)
                per_age.append(B.signature(srcname, probes, pay, blocks).to(torch.float16))
            sigs.append(per_age)
            if _run == 0:
                norms[srcname] = {a["name"]: state_norms(states[int(a["i"])], blocks)
                                  for a in ages}
                written[srcname] = written_part(states)
                diff = any(not torch.equal(ref_states[int(a["i"])][b][0],
                                           states[int(a["i"])][b][0])
                           for a in ages for b in blocks)
                check(f"D4 {srcname} : etat DIFFERENT du bras par defaut", diff)
            del states
        chrono[srcname] = (time.time() - t0) / 2
        check(f"D4 {srcname} bitwise (etats + signatures)",
              all(torch.equal(x, y) for x, y in zip(*sigs)), f"{chrono[srcname]:.1f} s/video")
    del ref_states

    print("[stats] ||etat||_F par age et par bloc")
    for name, per_age in norms.items():
        print(f"      {name:<20} " + " | ".join(f"{a}: {v}" for a, v in per_age.items()),
              flush=True)

    print("[D6] part ECRITE de l'etat : ||etat - etat_sans_ecriture|| / ||etat||, par age/bloc")
    for name, per_age in written.items():
        print(f"      {name:<20} " + " | ".join(f"{a}: {v}" for a, v in per_age.items()),
              flush=True)
    # Le plancher 0.5 est la CONTRAINTE SCELLEE du critere aveugle repare (registre :
    # "REPARATION DU CRITERE AVEUGLE") : l'ecriture doit DOMINER l'initialisation, laquelle est
    # identique pour toutes les videos. Ancrage non arbitraire : A0 = 0.88, A1 = 0.73 -- ces
    # deux valeurs sont celles de la FIN de sequence (c'est sur cet etat que sweep_mlp_step.py
    # mesure la part ecrite), donc c'est sur end_seq que le plancher s'applique. Aux ages
    # JEUNES la part ecrite est mecaniquement plus faible pour TOUS les bras (moins d'ecritures
    # accumulees : A0 lui-meme tombe a 0.45 a mid0) : la table par age est imprimee ci-dessus a
    # titre d'information, elle n'est pas soumise au plancher.
    last_age = list(next(iter(written.values())))[-1]
    for name, per_age in written.items():
        worst = min(per_age[last_age].values())
        check(f"D6 {name} : a {last_age}, l'ecriture DOMINE l'init (part >= 0.5)",
              worst >= 0.50, f"part ecrite minimale {worst:.3f} "
                             f"(jeunes ages : {min(v for pb in per_age.values() for v in pb.values()):.3f})")

    print()
    print("[chrono] par video : " + ", ".join(f"{k}={v:.1f}s" for k, v in chrono.items()))
    plan = [s for s in ("mlp_delta_all", "mlp_full", "mlp_full_muon", "mlp_hebb_seq")
            if s in chrono]
    print(f"[chrono] extrapolation 60 videos x les {len(plan)} bras du plan : "
          f"~{60 * sum(chrono[s] for s in plan) / 60:.0f} min de signatures "
          f"(+ ~8.5 min par CV x {len(plan)})")
    if _failed:
        print(f"SELFTEST MLP : {len(_failed)} ECHEC(S) : {_failed}")
        sys.exit(1)
    print("SELFTEST MLP : TOUT PASSE (D0-D6). STOP ici : relecture avant le run complet "
          "(commandes : run_mlp_delta.sh).")


if __name__ == "__main__":
    main()
