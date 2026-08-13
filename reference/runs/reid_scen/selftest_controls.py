#!/usr/bin/env python
# ============================================================================================
# SELFTEST des 2 CONTROLES REVIEWER du claim S2 (2026-08-03, reference/paper/risks.md attaque 1)
#
#   noiseanchor  MEM_DECODE_NOISE=1.0 : l'ancre decodee est remplacee par du BRUIT PUR calibre
#                sur les statistiques latentes de la fenetre vraie ou elle est splicee.
#   wronganchor  MEM_DECODE_SRC=<.pt> : l'ancre est celle decodee de l'etat d'une AUTRE cellule
#                (triple (t+1) % 12, meme seed), persistee par MEM_DECODE_DUMP.
#
# Ce que le selftest prouve (tout sur CPU, ~2 s) :
#   (a) tout ce qui est touche compile ;
#   (b) TABLE DES BRAS : les deux bras re-settent TOUTES les cles de l'union (aucune fuite d'un
#       bras au suivant), noiseanchor ne differe de memdec20 que par MEM_DECODE_NOISE, et la
#       ligne de wronganchor est IDENTIQUE a celle de memdec20 (son contenu est injecte au run) ;
#   (c) INERTIE LEGACY : hooks eteints => les deux helpers renvoient l'OBJET ancre lui-meme
#       (pas une copie), donc memdec20 et tous les bras existants sont bit a bit inchanges ;
#   (d) CALIBRATION du bruit : a la dose 1.0 la frame splicee a les mean/std PAR CANAL de la
#       fenetre vraie, ne contient plus rien de l'ancre (cos ~ 0), est deterministe par graine
#       de cellule et differe d'une cellule a l'autre ;
#   (e) ANCRE ETRANGERE : le .pt est splice VERBATIM, l'exclusion mutuelle avec le bruit est
#       gardee, et un fichier manquant leve ;
#   (f) MEME FENETRE : memdec20 / noiseanchor / wronganchor ne different QUE par le dernier slot
#       (meme moment, meme position de recence, meme convention de fenetre legacy) ;
#   (g) DECALAGE de cellule : permutation sans point fixe, dans le tier, meme seed ;
#   (h) STRUCTURE : les deux hooks sont bien DANS le bloc MEM_DECODE (donc au meme routage /
#       moment que memdec20), APRES le dump (le dump reste l'ancre brute de la cellule) et
#       AVANT le composite ; inference() legacy les refuse.
#
#   python selftest_controls.py
# ============================================================================================
import os
import sys

REPO = "/shared/ssd_14T/home/adenelmi/Anchor-Forcing"
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [REPO, os.path.join(REPO, "reference/runs/reid_probe_l1"), HERE]

import torch                                                            # noqa: E402

FAILS = []
NEW_KEYS = ("MEM_DECODE_NOISE", "MEM_DECODE_NOISE_SEED", "MEM_DECODE_SRC")


def check(name, ok, detail=""):
    print(f"[{'OK ' if ok else 'FAIL'}] {name}" + (f"  -- {detail}" if detail else ""), flush=True)
    if not ok:
        FAILS.append(name)


def _clean_env():
    for k in NEW_KEYS:
        os.environ.pop(k, None)


# ---------------------------------------------------------------------------------------------
def test_a_compile():
    import py_compile
    ok = True
    for p in ("pipeline/switch_causal_inference.py",
              "reference/runs/reid_scen/scenarios.py",
              "reference/runs/reid_scen/run_reid_scen.py",
              "reference/runs/reid_probe_l1/run_reid_probe.py",
              "reference/runs/reid_scen/rescore_controls.py"):
        f = os.path.join(REPO, p)
        try:
            py_compile.compile(f, doraise=True)
        except Exception as e:                                           # noqa: BLE001
            ok = False
            print(f"   {p}: {e}")
    check("(a) py_compile de tout ce qui est touche", ok)


# ---------------------------------------------------------------------------------------------
def test_b_arm_table():
    import scenarios as S
    union = set().union(*(set(v) for v in S.SCEN_ARM_ENV.values()))
    missing = {a: sorted(union - set(S.SCEN_ARM_ENV[a])) for a in ("noiseanchor", "wronganchor")}
    check("(b1) les 2 bras re-settent TOUTES les cles de l'union",
          not any(missing.values()), str(missing))
    check("(b2) les 3 nouvelles cles sont dans l'union (donc tous les AUTRES bras les eteignent)",
          all(k in union for k in NEW_KEYS),
          str([k for k in NEW_KEYS if k not in union]))
    off = [a for a, e in S.SCEN_ARM_ENV.items()
           if a != "noiseanchor" and (e.get("MEM_DECODE_NOISE", "0") != "0"
                                      or e.get("MEM_DECODE_SRC", "") != "")]
    check("(b3) aucun autre bras n'allume les controles", not off, str(off))
    d = {k for k in union
         if S.SCEN_ARM_ENV["noiseanchor"].get(k) != S.SCEN_ARM_ENV["memdec20"].get(k)}
    check("(b4) noiseanchor - memdec20 = {MEM_DECODE_NOISE} exactement",
          d == {"MEM_DECODE_NOISE"}, str(sorted(d)))
    check("(b5) wronganchor a la ligne EXACTE de memdec20 (contenu injecte au run)",
          S.SCEN_ARM_ENV["wronganchor"] == S.SCEN_ARM_ENV["memdec20"])
    check("(b6) les 2 bras tournent sur le pipeline MAG et exigent writer+decodeur",
          all(a in S.MAG_SCEN_ARMS and a in S.MEM2_CKPT_ARMS
              for a in ("noiseanchor", "wronganchor")))


# ---------------------------------------------------------------------------------------------
def _fake(cur=180, n=12, C=16, H=6, W=10, F=240, seed=7):
    g = torch.Generator().manual_seed(seed)
    out = torch.randn(1, F, C, H, W, generator=g)
    # des statistiques latentes REALISTES et par canal (mesurees sur nos dumps : std 0.75-1.30,
    # mean jusqu'a +-0.5) : une calibration qui ignore les canaux echouerait ici.
    mu = torch.linspace(-0.5, 0.5, C).view(1, 1, C, 1, 1)
    sd = torch.linspace(0.75, 1.30, C).view(1, 1, C, 1, 1)
    out = out * sd + mu
    anchor = torch.randn(1, 1, C, H, W, generator=g) * 0.58              # decodeur = passe-bas
    return out, anchor, cur, n


def test_c_inert():
    from pipeline.switch_causal_inference import SwitchCausalInferencePipeline as P
    _clean_env()
    out, anchor, cur, n = _fake()
    check("(c1) MEM_DECODE_NOISE absent => l'ancre elle-meme (meme objet)",
          P._maybe_noise_anchor(None, anchor, out, cur, n) is anchor)
    check("(c2) MEM_DECODE_SRC absent => l'ancre elle-meme (meme objet)",
          P._maybe_wrong_anchor(None, anchor) is anchor)
    os.environ["MEM_DECODE_NOISE"] = "0"
    os.environ["MEM_DECODE_SRC"] = ""
    check("(c3) et explicitement eteints (la valeur du _OFF) aussi",
          P._maybe_noise_anchor(None, anchor, out, cur, n) is anchor
          and P._maybe_wrong_anchor(None, anchor) is anchor)
    _clean_env()


def test_d_noise():
    from pipeline.switch_causal_inference import SwitchCausalInferencePipeline as P
    _clean_env()
    # taille SPATIALE reelle (60 x 104 = 6240 echantillons par canal latent) : la calibration se
    # verifie sur les statistiques empiriques, un jouet 6 x 10 n'aurait que 60 echantillons et le
    # test mesurerait le bruit d'echantillonnage au lieu de la calibration.
    out, anchor, cur, n = _fake(cur=24, n=12, H=60, W=104, F=30)
    ref = out[:, cur - n:cur].float()
    mu_r = ref.mean(dim=(0, 1, 3, 4))
    sd_r = ref.std(dim=(0, 1, 3, 4))

    os.environ["MEM_DECODE_NOISE"] = "1.0"
    os.environ["MEM_DECODE_NOISE_SEED"] = "1234"
    f1 = P._maybe_noise_anchor(None, anchor, out, cur, n)
    check("(d1) forme et dtype inchanges",
          f1.shape == anchor.shape and f1.dtype == anchor.dtype)
    a = f1.float().flatten()
    b = anchor.float().flatten()
    cos = float(torch.dot(a, b) / (a.norm() * b.norm()))
    check("(d2) dose 1.0 : plus RIEN de l'ancre (cos ~ 0)", abs(cos) < 0.02, f"cos = {cos:+.4f}")
    mu_f = f1.float().mean(dim=(0, 1, 3, 4))
    sd_f = f1.float().std(dim=(0, 1, 3, 4))
    dmu = float((mu_f - mu_r).abs().max())
    dsd = float((sd_f - sd_r).abs().max())
    check("(d3) calibre PAR CANAL sur la fenetre vraie (mean)", dmu < 0.06, f"max|dmu| = {dmu:.3f}")
    check("(d4) calibre PAR CANAL sur la fenetre vraie (std)", dsd < 0.05, f"max|dsd| = {dsd:.3f}")
    naive = float((sd_f - 1.0).abs().max())
    check("(d5) et ce n'est PAS du N(0,1) naif (la calibration fait une difference mesurable)",
          naive > 0.15, f"max|std - 1| = {naive:.3f}")

    f1b = P._maybe_noise_anchor(None, anchor, out, cur, n)
    check("(d6) deterministe a graine de cellule fixee", torch.equal(f1, f1b))
    os.environ["MEM_DECODE_NOISE_SEED"] = "5678"
    f2 = P._maybe_noise_anchor(None, anchor, out, cur, n)
    c12 = float(torch.dot(f1.float().flatten(), f2.float().flatten())
                / (f1.float().norm() * f2.float().norm()))
    # le cos residuel (~0.08) n'est PAS de la dependance entre tirages : c'est la moyenne par
    # canal, commune aux deux parce que les deux sont calibres sur la MEME fenetre (mesure :
    # ||mu||^2 / (||mu||^2 + ||sigma||^2) ~ 0.08). La partie aleatoire, elle, est independante.
    c12eps = float(torch.dot((f1.float() - f2.float()).flatten(),
                             (f1.float() - f2.float()).flatten()))
    check("(d7) 2 cellules = 2 tirages independants (pas UNE realisation pour les 24)",
          not torch.equal(f1, f2) and abs(c12) < 0.15 and c12eps > 0, f"cos = {c12:+.4f}")
    os.environ["MEM_DECODE_NOISE_SEED"] = "1234"
    f3 = P._maybe_noise_anchor(None, anchor, out, cur + 3, n)
    check("(d8) 2 retours dans un meme rollout = 2 tirages", not torch.equal(f1, f3))

    os.environ["MEM_DECODE_NOISE"] = "0.5"
    fh = P._maybe_noise_anchor(None, anchor, out, cur, n).float()
    mix = ((1 - 0.25) ** 0.5) * anchor.float() + 0.5 * (f1.float())
    # (meme eps, meme graine : la dose interpole exactement)
    check("(d9) la dose interpole exactement (formule RMS-preservante)",
          float((fh - mix).abs().max()) < 1e-5, f"max|d| = {float((fh - mix).abs().max()):.2e}")
    _clean_env()


def test_e_wrong():
    from pipeline.switch_causal_inference import SwitchCausalInferencePipeline as P
    import tempfile
    _clean_env()
    out, anchor, cur, n = _fake()
    other = torch.randn(anchor.shape, generator=torch.Generator().manual_seed(99)) * 0.58
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, "other.f180.pt")
        torch.save(other.to(torch.float16), p)
        os.environ["MEM_DECODE_SRC"] = p
        got = P._maybe_wrong_anchor(None, anchor)
        check("(e1) l'ancre etrangere est splicee VERBATIM (au cast pres)",
              torch.equal(got.float(), other.to(torch.float16).float()))
        check("(e2) dtype/device de la cellule courante conserves",
              got.dtype == anchor.dtype and got.device == anchor.device)
        # format {'anchor': t} (celui du hook ORACLE_ANCHOR_DUMP) accepte aussi
        torch.save({"anchor": other, "frame": 179}, p)
        check("(e3) format dict {'anchor': ...} accepte",
              torch.equal(P._maybe_wrong_anchor(None, anchor).float(), other.float()))
        os.environ["MEM_DECODE_NOISE"] = "1.0"
        try:
            P._maybe_wrong_anchor(None, anchor)
            ok = False
        except AssertionError:
            ok = True
        check("(e4) exclusion mutuelle bruit / ancre etrangere", ok)
        os.environ.pop("MEM_DECODE_NOISE")
        # mauvaise forme = refus (une ancre d'un autre SCENARIO ne peut pas passer en silence)
        torch.save(torch.randn(1, 1, 16, 7, 10), p)
        try:
            P._maybe_wrong_anchor(None, anchor)
            ok = False
        except AssertionError:
            ok = True
        check("(e5) forme incompatible = refus", ok)
    os.environ["MEM_DECODE_SRC"] = os.path.join(td, "nope.pt")
    try:
        P._maybe_wrong_anchor(None, anchor)
        ok = False
    except AssertionError:
        ok = True
    check("(e6) fichier manquant = refus (jamais de silence)", ok)
    _clean_env()


def test_f_same_window():
    """Les 3 bras ne different QUE par le contenu du dernier slot."""
    from pipeline.switch_causal_inference import SwitchCausalInferencePipeline as P
    import tempfile
    _clean_env()
    out, anchor, cur, n = _fake()
    wins = {}
    wins["memdec20"] = P._memdec_window(out, cur, n, anchor, 1, -1)
    os.environ["MEM_DECODE_NOISE"] = "1.0"
    os.environ["MEM_DECODE_NOISE_SEED"] = "1234"
    wins["noiseanchor"] = P._memdec_window(
        out, cur, n, P._maybe_noise_anchor(None, anchor, out, cur, n), 1, -1)
    _clean_env()
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, "o.pt")
        other = torch.randn(anchor.shape, generator=torch.Generator().manual_seed(5)) * 0.58
        torch.save(other, p)
        os.environ["MEM_DECODE_SRC"] = p
        wins["wronganchor"] = P._memdec_window(
            out, cur, n, P._maybe_wrong_anchor(None, anchor), 1, -1)
    _clean_env()
    base = wins["memdec20"]
    for a in ("noiseanchor", "wronganchor"):
        check(f"(f1.{a}) les {n - 1} slots de contexte sont BIT A BIT ceux de memdec20",
              torch.equal(wins[a][:, :-1], base[:, :-1]))
        check(f"(f2.{a}) seul le slot de recence differe",
              not torch.equal(wins[a][:, -1:], base[:, -1:]))
        check(f"(f3.{a}) et ce contexte est la VRAIE queue recente",
              torch.equal(wins[a][:, :-1], out[:, cur - (n - 1):cur]))


def test_g_shift():
    import scenarios as S
    tier = list(range(S.N_CROSS_TRIPLES))
    img = [S.wrong_anchor_triple(t) for t in tier]
    check("(g1) permutation du tier cross-scene, sans point fixe",
          sorted(img) == tier and all(a != b for a, b in zip(tier, img)), str(img))
    check("(g2) +1 modulo 12, deterministe", img == [(t + 1) % 12 for t in tier])
    ss = S.SAMESCENE_TRIPLE_IDS
    check("(g3) le tier meme-decor reste dans son tier",
          all(S.wrong_anchor_triple(t) in ss and S.wrong_anchor_triple(t) != t for t in ss))
    check("(g4) S2 : un seul retour, a la frame 180",
          S.revisit_segments("S2") == [3] and S.revisit_start_frames("S2") == [180])
    src = S.wrong_anchor_src("S2", 11, 1, "/tmp/d")
    check("(g5) le chemin lu = le dump de (t+1)%12, MEME SEED, a la frame du retour",
          src == "/tmp/d/S2_t0_s1_decanchor.f180.pt", src)
    check("(g6) le dump ecrit par noiseanchor porte bien ce nom",
          S.dec_anchor_base("S2", 0, 1, "/tmp/d") + ".f180.pt" == src)
    try:
        S.wrong_anchor_src("S4", 0, 0)                                   # 2 retours
        ok = False
    except AssertionError:
        ok = True
    check("(g7) refus explicite sur un scenario a 2 retours", ok)


def test_h_structure():
    src = open(os.path.join(REPO, "pipeline/switch_causal_inference.py")).read()
    blk = src[src.index("_dec_ckpt = os.environ.get(\"MEM_DECODE\", \"\").strip() if"):]
    blk = blk[:blk.index("_blocks_env = os.environ.get")]
    i_dump = blk.index("MEM_DECODE_DUMP")
    i_wrong = blk.index("_maybe_wrong_anchor")
    i_noise = blk.index("_maybe_noise_anchor")
    i_comp = blk.index("_maybe_composite")
    check("(h1) les 2 hooks sont DANS le bloc MEM_DECODE (meme routage, meme moment)",
          i_wrong > 0 and i_noise > 0)
    check("(h2) APRES le dump (le dump reste l'ancre BRUTE de la cellule)",
          i_dump < i_wrong < i_comp and i_dump < i_noise < i_comp)
    check("(h3) le splice se fait avec la MEME convention de fenetre (slot_offset legacy)",
          "MEM_DECODE_SLOT_OFFSET" in blk and "_memdec_window(" in blk)
    check("(h4) inference() legacy refuse les 2 controles",
          "MEM_DECODE_NOISE / MEM_DECODE_SRC (reviewer controls" in src)


def main():
    print("=" * 92)
    print("SELFTEST controles reviewer S2 : noiseanchor (MEM_DECODE_NOISE) / wronganchor "
          "(MEM_DECODE_SRC)")
    print("=" * 92)
    for t in (test_a_compile, test_b_arm_table, test_c_inert, test_d_noise, test_e_wrong,
              test_f_same_window, test_g_shift, test_h_structure):
        print(f"\n--- {t.__name__} ---")
        t()
    print("\n" + "=" * 92)
    if FAILS:
        print(f"{len(FAILS)} FAIL: {FAILS}")
        sys.exit(1)
    print("TOUT PASSE")


if __name__ == "__main__":
    main()
