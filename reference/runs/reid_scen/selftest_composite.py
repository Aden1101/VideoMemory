#!/usr/bin/env python
# ============================================================================================
# SELFTEST du hook MEM_DECODE_COMPOSITE (arms oracle1comp / memdec21c, pilote 2026-08-02).
# CPU seul, quelques secondes -- aucun GPU, aucun modele charge.
#
#   (a) py_compile de tout ce qui est touche
#   (b) MASQUE : forme, bornes, somme des poids = 1 partout, centre = 1, coins = 0, symetrie,
#       niveau 0.5 exactement sur l'ellipse MEM_COMP_CROP, monotonie radiale
#   (c) DOUCEUR : le plus grand saut du masque entre deux pixels voisins reste petit (proxy
#       numerique du "pas de couture visible"); et le masque est C1 (smoothstep) donc la derivee
#       s'annule aux deux bords de l'anneau
#   (d) MATH du composite : comp == m * ancre + (1 - m) * fond, exactement ; le centre vient de
#       l'ancre, le bord du fond ; broadcast sur plusieurs frames d'ancre
#   (e) LEGACY : hook eteint => _maybe_composite renvoie l'OBJET ancre lui-meme (identite), donc
#       le chemin de code existant est inchange ; et rallume, le compteur _comp_applied bouge
#   (f) TABLE DES BRAS : oracle1comp/memdec21c re-settent bien toutes les cles de l'union
#       (invariant "rien ne fuit d'un bras a l'autre") et n'allument le composite que chez eux
#
#   python selftest_composite.py
# ============================================================================================
import os
import sys

REPO = "/shared/ssd_14T/home/adenelmi/Anchor-Forcing"
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [REPO, os.path.join(REPO, "reference/runs/reid_probe_l1"), HERE]

import torch                                                            # noqa: E402

FAILS = []


def check(name, ok, detail=""):
    print(f"[{'OK ' if ok else 'FAIL'}] {name}" + (f"  -- {detail}" if detail else ""),
          flush=True)
    if not ok:
        FAILS.append(name)


# ---------------------------------------------------------------------------------------------
def test_a_compile():
    import py_compile
    ok = True
    for p in ("pipeline/switch_causal_inference.py",
              "reference/runs/reid_scen/scenarios.py",
              "reference/runs/reid_scen/run_reid_scen.py",
              "reference/runs/reid_scen/score_comp.py"):
        f = os.path.join(REPO, p)
        if not os.path.exists(f):
            continue
        try:
            py_compile.compile(f, doraise=True)
        except Exception as e:                                          # noqa: BLE001
            ok = False
            print(f"   {p}: {e}")
    check("(a) py_compile", ok)


def _mask(H=60, W=104, crop=0.6, feather=0.3):
    from pipeline.switch_causal_inference import SwitchCausalInferencePipeline as P
    return P._composite_mask(H, W, crop, feather, torch.device("cpu"), torch.float32)


def test_b_mask():
    H, W, crop = 60, 104, 0.6
    m5 = _mask(H, W, crop)
    check("(b1) shape (1,1,1,H,W)", tuple(m5.shape) == (1, 1, 1, H, W), str(tuple(m5.shape)))
    m = m5[0, 0, 0]
    check("(b2) bornes [0,1]", bool((m >= 0).all() and (m <= 1).all()),
          f"min {float(m.min()):.4f} max {float(m.max()):.4f}")
    w = m + (1.0 - m)
    check("(b3) somme des poids = 1 partout", float((w - 1.0).abs().max()) == 0.0,
          f"max|m + (1-m) - 1| = {float((w - 1.0).abs().max()):.2e}")
    check("(b4) centre = 1 (sujet pur)", float(m[H // 2, W // 2]) == 1.0,
          f"m[centre] = {float(m[H // 2, W // 2]):.6f}")
    corners = [float(m[0, 0]), float(m[0, -1]), float(m[-1, 0]), float(m[-1, -1])]
    check("(b5) coins = 0 (fond pur)", max(corners) == 0.0, f"max coin = {max(corners):.2e}")
    check("(b6) symetrie H et V", float((m - m.flip(0)).abs().max()) < 1e-6
          and float((m - m.flip(1)).abs().max()) < 1e-6)
    # niveau 0.5 exactement sur l'ellipse: sur l'axe horizontal, a |x - cx| = crop*W/2
    cx, cy = (W - 1) / 2.0, (H - 1) / 2.0
    for ax, half, row_col in (("horizontal", crop * W / 2.0, ("row", cx)),
                              ("vertical", crop * H / 2.0, ("col", cy))):
        c = row_col[1]
        i0, i1 = int(c - half), int(c - half) + 1
        line = m[H // 2] if ax == "horizontal" else m[:, W // 2]
        t = (c - half) - i0
        v = float(line[i0] * (1 - t) + line[i1] * t)      # interpolation lineaire au point exact
        # tolerance = la courbure du smoothstep sur un pas de pixel (l'interpolation est lineaire)
        check(f"(b7 {ax}) m = 0.5 sur l'ellipse crop={crop}", abs(v - 0.5) < 5e-3,
              f"m({ax} @ {crop*100:.0f}% de la dimension) = {v:.6f}")
    # monotonie radiale sur l'axe horizontal, du centre vers le bord
    line = m[H // 2, W // 2:]
    check("(b8) decroissance monotone du centre vers le bord",
          bool((line[1:] - line[:-1] <= 1e-7).all()))
    # le plateau (m = 1) couvre bien (1 - feather) * crop et zero au-dela de (1 + feather) * crop
    frac1 = float((m[H // 2] >= 1.0 - 1e-6).sum()) / W
    frac0 = float((m[H // 2] > 0).sum()) / W
    check("(b9) geometrie plateau / support", abs(frac1 - 0.7 * crop) < 0.02
          and abs(frac0 - 1.3 * crop) < 0.02,
          f"plateau {frac1:.3f} de la largeur (attendu {0.7*crop:.3f}), "
          f"support {frac0:.3f} (attendu {1.3*crop:.3f})")


def test_c_smooth():
    m = _mask()[0, 0, 0]
    dh = (m[:, 1:] - m[:, :-1]).abs().max()
    dv = (m[1:] - m[:-1]).abs().max()
    check("(c1) pas de marche dans le masque (proxy 'pas de couture')",
          float(max(dh, dv)) < 0.15, f"plus grand saut voisin-a-voisin = {float(max(dh, dv)):.4f}")
    # C1 (smoothstep) : la pente s'annule aux DEUX bords de l'anneau, donc pas de cassure de
    # derivee la ou le masque atteint 0 et 1 -> la courbure reste bornee partout (un raccord
    # lineaire, lui, ferait un pic de courbure egal au saut de pente).
    line = m[m.shape[0] // 2]
    d2 = (line[2:] - 2 * line[1:-1] + line[:-2]).abs().max()
    check("(c2) pas de cassure de pente (courbure bornee, raccord C1)", float(d2) < 0.05,
          f"max|d2m| = {float(d2):.4f}")
    flat = ((line[:-1] == 1) & (line[1:] == 1)) | ((line[:-1] == 0) & (line[1:] == 0))
    d = (line[1:] - line[:-1]).abs()
    check("(c3) plateau et exterieur strictement plats", float(d[flat].max()) == 0.0)


def test_d_math():
    from pipeline.switch_causal_inference import SwitchCausalInferencePipeline as P

    class Fake:                                       # juste le porteur des methodes
        _composite_mask = staticmethod(P._composite_mask)
        _energy_centroid = staticmethod(P._energy_centroid)
        _comp_alpha = P._comp_alpha
        _maybe_composite = P._maybe_composite

    f = Fake()
    B, F, C, H, W = 1, 200, 16, 60, 104
    g = torch.Generator().manual_seed(7)
    output = torch.randn(B, F, C, H, W, generator=g, dtype=torch.float32)
    anchor = torch.randn(B, 1, C, H, W, generator=g, dtype=torch.float32)
    cur = 180
    os.environ["MEM_DECODE_COMPOSITE"] = "1"
    os.environ["MEM_COMP_CROP"], os.environ["MEM_COMP_FEATHER"] = "0.6", "0.3"
    os.environ.pop("MEM_COMP_DUMP", None)
    os.environ["ORACLE_SPLICE_POS"] = ""
    f._comp_applied = 0
    comp = f._maybe_composite(anchor, output, cur)
    m = _mask(H, W)
    bg = output[:, cur - 1:cur]
    ref = m * anchor + (1.0 - m) * bg
    check("(d1) comp == m*ancre + (1-m)*fond (exact)",
          float((comp - ref).abs().max()) == 0.0, f"max|diff| = {float((comp - ref).abs().max()):.2e}")
    check("(d2) le centre vient de l'ANCRE",
          float((comp[..., H // 2, W // 2] - anchor[..., H // 2, W // 2]).abs().max()) == 0.0)
    check("(d3) les coins viennent du FOND",
          float((comp[..., 0, 0] - bg[..., 0, 0]).abs().max()) == 0.0)
    check("(d4) le fond est bien la DERNIERE VRAIE frame du contexte (cur-1)",
          float((bg - output[:, cur - 1:cur]).abs().max()) == 0.0)
    check("(d5) shape preservee", tuple(comp.shape) == tuple(anchor.shape))
    # broadcast sur plusieurs frames d'ancre (MEM_DECODE_SLOTS > 1)
    a3 = torch.randn(B, 3, C, H, W, generator=g, dtype=torch.float32)
    c3 = f._maybe_composite(a3, output, cur)
    check("(d6) broadcast du fond sur k frames d'ancre", tuple(c3.shape) == (B, 3, C, H, W)
          and float((c3 - (m * a3 + (1.0 - m) * bg)).abs().max()) == 0.0)
    check("(d7) compteur _comp_applied", f._comp_applied == 2, str(f._comp_applied))
    # placements non supportes -> assert
    os.environ["ORACLE_SPLICE_POS"] = "sink0"
    try:
        f._maybe_composite(anchor, output, cur)
        ok = False
    except AssertionError:
        ok = True
    os.environ["ORACLE_SPLICE_POS"] = ""
    check("(d8) refus explicite du placement sink0/head", ok)


def test_e_legacy():
    from pipeline.switch_causal_inference import SwitchCausalInferencePipeline as P

    class Fake:
        _composite_mask = staticmethod(P._composite_mask)
        _energy_centroid = staticmethod(P._energy_centroid)
        _comp_alpha = P._comp_alpha
        _maybe_composite = P._maybe_composite

    f = Fake()
    f._comp_applied = 0
    output = torch.randn(1, 200, 16, 60, 104)
    anchor = torch.randn(1, 1, 16, 60, 104)
    for val in ("0", "", None):
        if val is None:
            os.environ.pop("MEM_DECODE_COMPOSITE", None)
        else:
            os.environ["MEM_DECODE_COMPOSITE"] = val
        out = f._maybe_composite(anchor, output, 180)
        check(f"(e) hook OFF (MEM_DECODE_COMPOSITE={val!r}) -> ancre renvoyee TELLE QUELLE",
              out is anchor and f._comp_applied == 0)


def test_f_armtable():
    import scenarios as S
    keys = set()
    for env in S.SCEN_ARM_ENV.values():
        keys |= set(env)
    bad = {a: sorted(keys - set(env)) for a, env in S.SCEN_ARM_ENV.items() if keys - set(env)}
    check("(f1) chaque bras re-set TOUTES les cles de l'union (rien ne fuit)", not bad, str(bad))
    on = {a for a, e in S.SCEN_ARM_ENV.items() if e.get("MEM_DECODE_COMPOSITE", "0") not in ("0", "")}
    check("(f2) composite allume UNIQUEMENT sur les bras declares composites",
          on == set(S.COMPOSITE_BASE), str(sorted(on ^ set(S.COMPOSITE_BASE))))
    for comp, base in sorted(S.COMPOSITE_BASE.items()):
        e_c, e_b = S.SCEN_ARM_ENV[comp], S.SCEN_ARM_ENV[base]
        d = {k for k in keys if e_c.get(k) != e_b.get(k)}
        check(f"(f3) {comp} = {base} + les seuls flags composite",
              all(k == "MEM_DECODE_COMPOSITE" or k.startswith("MEM_COMP_") for k in d)
              and "MEM_DECODE_COMPOSITE" in d, str(sorted(d)))
    check("(f4) tous les bras composites sont dans MAG_SCEN_ARMS",
          set(S.COMPOSITE_BASE) <= set(S.MAG_SCEN_ARMS))


if __name__ == "__main__":
    torch.set_grad_enabled(False)
    test_a_compile()
    test_b_mask()
    test_c_smooth()
    test_d_math()
    test_e_legacy()
    test_f_armtable()
    print("\n" + ("ALL COMPOSITE SELFTESTS PASSED" if not FAILS
                  else f"FAILED: {FAILS}"))
    sys.exit(1 if FAILS else 0)
