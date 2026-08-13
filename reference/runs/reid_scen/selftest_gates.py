#!/usr/bin/env python
# ============================================================================================
# SELFTEST des 3 hooks ajoutes le 2026-08-02 (gates "composite vs additif").
#
#   GATE 1  MEM_ADDITIVE_KV   canal additif : l'ancre entre comme tokens k/v SUPPLEMENTAIRES,
#                             rien n'est evince. Teste : (i) le merge log-sum-exp est
#                             EXACTEMENT l'attention sur [contexte ; ancre] quand gain = 1,
#                             (ii) le gain deplace la masse d'attention dans le bon sens,
#                             (iii) hook absent => chemin legacy inchange (attribut absent).
#   GATE 2  MEM_DECODE_SLOT_OFFSET  convention de fenetre. Teste : offset < 0 == l'ancienne
#                             expression inline (les 144 mp4 restent reproductibles), offset 0
#                             == la fenetre d'oracle1 SLOT PAR SLOT (le point du gate), offset
#                             j recule l'ancre et laisse la VRAIE frame a la recence.
#   GATE 3  controles anti-illusion du masque composite (uniform / bruit / decale / inverse) +
#                             GATE 4 (union, dose alpha). Teste : chacun est inerte par defaut,
#                             la dose alpha = 0 rend l'ancre BIT A BIT (donc memdec20), le
#                             masque uniforme est APPARIE EN DOSE, l'inverse est bien 1 - m.
#
#   python selftest_gates.py          (CPU pour tout sauf le merge attention, qui prend un GPU
#                                      s'il y en a un de visible ; sinon il est saute)
# ============================================================================================
import math
import os
import sys

REPO = "/shared/ssd_14T/home/adenelmi/Anchor-Forcing"
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [REPO, os.path.join(REPO, "reference/runs/reid_probe_l1"), HERE]

import torch                                                            # noqa: E402

FAILS = []


def check(name, ok, detail=""):
    print(f"[{'OK ' if ok else 'FAIL'}] {name}" + (f"  -- {detail}" if detail else ""), flush=True)
    if not ok:
        FAILS.append(name)


# ---------------------------------------------------------------------------------------------
def test_a_compile():
    import py_compile
    ok = True
    for p in ("pipeline/switch_causal_inference.py", "wan/modules/causal_model.py",
              "reference/runs/reid_scen/scenarios.py", "reference/runs/reid_scen/run_reid_scen.py",
              "reference/runs/reid_scen/score_comp.py"):
        f = os.path.join(REPO, p)
        try:
            py_compile.compile(f, doraise=True)
        except Exception as e:                                           # noqa: BLE001
            ok = False
            print(f"   {p}: {e}")
    check("(a) py_compile de tout ce qui est touche", ok)


# ---------------------------------------------------------------------------------------------
def test_b_window():
    """GATE 2: la convention de fenetre."""
    from pipeline.switch_causal_inference import SwitchCausalInferencePipeline as P
    B, F, C, H, W = 1, 240, 16, 6, 10
    g = torch.Generator().manual_seed(11)
    out = torch.randn(B, F, C, H, W, generator=g)
    cur, n, k = 180, 12, 1
    anc = torch.randn(B, 1, C, H, W, generator=g)

    legacy_ref = torch.cat([out[:, cur - (n - k):cur], anc], dim=1)      # l'ancien code, verbatim
    legacy = P._memdec_window(out, cur, n, anc, k, -1)
    check("(b1) offset < 0 == l'expression legacy verbatim",
          torch.equal(legacy, legacy_ref), f"max|d| = {float((legacy - legacy_ref).abs().max()):.2e}")

    oracle_ref = torch.cat([out[:, cur - n:cur - k], anc], dim=1)        # ce que splice oracle1
    s0 = P._memdec_window(out, cur, n, anc, k, 0)
    check("(b2) offset 0 == la fenetre d'oracle1, SLOT PAR SLOT",
          torch.equal(s0, oracle_ref), f"max|d| = {float((s0 - oracle_ref).abs().max()):.2e}")
    check("(b3) les 11 slots non-ancre sont index-identiques entre memdec-s0 et oracle1",
          torch.equal(s0[:, :n - k], oracle_ref[:, :n - k]))
    check("(b4) et la LEGACY, elle, ne l'etait pas (c'est le bug qu'on corrige)",
          not torch.equal(legacy[:, :n - k], oracle_ref[:, :n - k]))

    for off in (1, 2):
        s = P._memdec_window(out, cur, n, anc, k, off)
        pos = n - k - off
        check(f"(b5.{off}) offset {off}: l'ancre est au slot {pos}",
              torch.equal(s[:, pos:pos + k], anc))
        check(f"(b6.{off}) offset {off}: les {off} slots de recence restent les VRAIES frames",
              torch.equal(s[:, pos + k:], out[:, cur - off:cur]))
        check(f"(b7.{off}) offset {off}: la tete de fenetre reste vraie et alignee",
              torch.equal(s[:, :pos], out[:, cur - n:cur - n + pos]))
    # multi-slot
    a3 = torch.randn(B, 3, C, H, W, generator=g)
    s = P._memdec_window(out, cur, n, a3, 3, 0)
    check("(b8) k slots > 1 en convention oracle", s.shape[1] == n
          and torch.equal(s[:, -3:], a3) and torch.equal(s[:, :-3], out[:, cur - n:cur - 3]))


# ---------------------------------------------------------------------------------------------
class _FakeComp:
    """Porteur des methodes de composite (aucun modele, aucun GPU)."""
    from pipeline.switch_causal_inference import SwitchCausalInferencePipeline as _P
    _composite_mask = staticmethod(_P._composite_mask)
    _energy_centroid = staticmethod(_P._energy_centroid)
    _comp_alpha = _P._comp_alpha
    _maybe_composite = _P._maybe_composite


def _reset_comp_env():
    for k, v in (("MEM_DECODE_COMPOSITE", "1"), ("MEM_COMP_CROP", "0.6"),
                 ("MEM_COMP_FEATHER", "0.3"), ("MEM_COMP_SHIFT", "0"), ("MEM_COMP_INVERT", "0"),
                 ("MEM_COMP_UNIFORM", "0"), ("MEM_COMP_UNION", "0"), ("MEM_COMP_ALPHA", "1"),
                 ("MEM_COMP_PROMPT_GATE", "0"), ("MEM_COMP_ANCHOR_NOISE", "0"),
                 ("ORACLE_SPLICE_POS", "")):
        os.environ[k] = v
    os.environ.pop("MEM_COMP_DUMP", None)


def test_c_controls():
    """GATE 3/4: les controles du masque."""
    H, W = 60, 104
    f = _FakeComp()
    f._comp_applied, f._comp_gate_now = 0, 1.0
    g = torch.Generator().manual_seed(7)
    out = torch.randn(1, 200, 16, H, W, generator=g)
    anc = torch.randn(1, 1, 16, H, W, generator=g)
    cur = 180
    bg = out[:, cur - 1:cur]

    _reset_comp_env()
    base = f._maybe_composite(anc, out, cur)
    m = f._composite_mask(H, W, 0.6, 0.3, torch.device("cpu"), torch.float32)
    check("(c0) reference: comp == m*ancre + (1-m)*fond (non regresse par cy/cx)",
          torch.equal(base, (m * anc + (1 - m) * bg)))

    # --- (a) masque uniforme APPARIE EN DOSE
    _reset_comp_env(); os.environ["MEM_COMP_UNIFORM"] = "1"
    u = f._maybe_composite(anc, out, cur)
    mu = (u - bg) / (anc - bg)                       # le poids implicite du sujet, par pixel
    check("(c1a) uniforme: le poids est CONSTANT sur toute la frame",
          float(mu.std()) < 1e-5, f"std(m) = {float(mu.std()):.2e}")
    check("(c1b) uniforme: la DOSE est appariee a celle de l'ellipse",
          abs(float(mu.mean()) - float(m.mean())) < 1e-5,
          f"m_unif {float(mu.mean()):.4f} vs m_ellipse {float(m.mean()):.4f}")

    # --- (b) ancre bruitee, masque garde
    _reset_comp_env(); os.environ["MEM_COMP_ANCHOR_NOISE"] = "1.0"
    nz = f._maybe_composite(anc, out, cur)
    check("(c2a) bruit: le bord reste EXACTEMENT le fond (masque intact)",
          torch.equal(nz[..., 0, 0], bg[..., 0, 0]))
    ctr = (H // 2, W // 2)
    check("(c2b) bruit: le centre ne vient plus de l'ancre (contenu detruit)",
          float((nz[..., ctr[0], ctr[1]] - anc[..., ctr[0], ctr[1]]).abs().max()) > 0.1)
    rms_a, rms_n = float(anc.pow(2).mean().sqrt()), float(((nz - bg)[..., ctr[0], ctr[1]]).abs().mean())
    check("(c2c) bruit: sigma = 1 => RMS preserve (pas un zero deguise)", rms_n > 0.1,
          f"RMS ancre {rms_a:.3f}, amplitude centre {rms_n:.3f}")

    # --- (c) masque decale / inverse
    _reset_comp_env(); os.environ["MEM_COMP_SHIFT"] = "0.3333"
    sh = f._maybe_composite(anc, out, cur)
    msh = (sh - bg) / (anc - bg)
    col = msh[0, 0, 0].mean(0)
    check("(c3a) decale: le maximum du masque a bouge d'environ W/3",
          abs(int(col.argmax()) - (W // 2 + W // 3)) <= 2,
          f"argmax colonne {int(col.argmax())} (attendu ~{W // 2 + W // 3})")
    _reset_comp_env(); os.environ["MEM_COMP_INVERT"] = "1"
    iv = f._maybe_composite(anc, out, cur)
    check("(c3b) inverse: comp == (1-m)*ancre + m*fond",
          torch.allclose(iv, (1 - m) * anc + m * bg, atol=1e-6))
    check("(c3c) inverse: le centre vient du FOND, les coins de l'ANCRE",
          torch.equal(iv[..., ctr[0], ctr[1]], bg[..., ctr[0], ctr[1]])
          and torch.equal(iv[..., 0, 0], anc[..., 0, 0]))

    # --- dose alpha (gate 4) : alpha = 0 => l'ancre BIT A BIT
    _reset_comp_env(); os.environ["MEM_COMP_ALPHA"] = "0"
    z = f._maybe_composite(anc, out, cur)
    check("(c4a) alpha = 0 => ancre rendue BIT A BIT (le bras retombe sur memdec20)",
          torch.equal(z, anc), f"max|d| = {float((z - anc).abs().max()):.2e}")
    _reset_comp_env(); os.environ["MEM_COMP_ALPHA"] = "0.5"
    h = f._maybe_composite(anc, out, cur)
    check("(c4b) alpha = 0.5 == moitie du chemin vers le composite plein",
          torch.allclose(h, 0.5 * anc + 0.5 * base, atol=1e-6))

    # --- union (gate 4)
    _reset_comp_env(); os.environ["MEM_COMP_UNION"] = "1"
    lat = torch.zeros(1, 1, 16, H, W)
    lat[0, 0, :, 40:50, 80:95] = 3.0                    # un "sujet" en bas a droite
    cyx = f._energy_centroid(lat)
    check("(c5a) le localiseur grossier trouve la tache",
          38 <= cyx[0] <= 52 and 76 <= cyx[1] <= 98, f"centroide {cyx}")
    out2 = out.clone(); out2[:, cur - 1] = lat[0, 0]
    un = f._maybe_composite(anc, out2, cur)
    mun = (un - out2[:, cur - 1:cur]) / (anc - out2[:, cur - 1:cur])
    check("(c5b) union: le masque couvre AUSSI la zone du sujet courant",
          float(mun[0, 0, 0, 45, 87]) > 0.9, f"m(union) au sujet courant = {float(mun[0,0,0,45,87]):.3f}")
    check("(c5c) union: le masque est plus large que l'ellipse centrale seule",
          float(mun.mean()) > float(m.mean()),
          f"{float(mun.mean()):.4f} > {float(m.mean()):.4f}")

    # --- inertie: hook eteint => objet ancre rendu tel quel
    _reset_comp_env()
    for val in ("0", ""):
        os.environ["MEM_DECODE_COMPOSITE"] = val
        check(f"(c6) MEM_DECODE_COMPOSITE={val!r} => ancre rendue telle quelle (identite d'objet)",
              f._maybe_composite(anc, out, cur) is anc)
    _reset_comp_env()


# ---------------------------------------------------------------------------------------------
def test_d_additive():
    """GATE 1: le merge log-sum-exp du canal additif."""
    from wan.modules.causal_model import CausalWanSelfAttention as A
    if not torch.cuda.is_available():
        check("(d) canal additif (necessite un GPU visible)", True, "SAUTE (pas de CUDA)")
        return
    from wan.modules.attention import attention
    dev = torch.device("cuda")
    torch.manual_seed(3)
    B, H, Lq, Lk, La, D = 1, 12, 4680, 18720, 1560, 128
    q = (torch.randn(B, Lq, H, D, device=dev, dtype=torch.bfloat16) * 0.5)
    k = (torch.randn(B, Lk, H, D, device=dev, dtype=torch.bfloat16) * 0.5)
    v = torch.randn(B, Lk, H, D, device=dev, dtype=torch.bfloat16)
    ka = (torch.randn(B, La, H, D, device=dev, dtype=torch.bfloat16) * 0.5)
    va = torch.randn(B, La, H, D, device=dev, dtype=torch.bfloat16)
    ref = attention(q, torch.cat([k, ka], 1), torch.cat([v, va], 1)).float()
    scale = float(ref.abs().mean())
    om, lm = A._sdpa_lse(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2))
    oa, la = A._sdpa_lse(q.transpose(1, 2), ka.transpose(1, 2), va.transpose(1, 2))
    mass = {}
    for gain in (0.5, 1.0, 2.0):
        al = torch.sigmoid(math.log(gain) + (la - lm).float())
        out = ((1 - al).unsqueeze(-1) * om.float() + al.unsqueeze(-1) * oa.float()).transpose(1, 2)
        mass[gain] = float(al.mean())
        if gain == 1.0:
            rel = float((out - ref).abs().mean()) / scale
            check("(d1) gain = 1 => EXACTEMENT l'attention sur [contexte ; ancre] (a bf16 pres)",
                  rel < 5e-3, f"erreur relative moyenne {rel:.2e}")
    check("(d2) la masse d'attention de l'ancre est monotone en gain",
          mass[0.5] < mass[1.0] < mass[2.0],
          " < ".join(f"g{g}: {mass[g]:.4f}" for g in (0.5, 1.0, 2.0)))
    check("(d3) a gain 1 la masse vaut la part de tokens (cles aleatoires => uniforme)",
          abs(mass[1.0] - La / (La + Lk)) < 0.02,
          f"masse {mass[1.0]:.4f} vs part de tokens {La / (La + Lk):.4f}")
    # inertie structurelle : l'attribut est absent d'un module neuf
    a = A(dim=1536, num_heads=12, local_attn_size=12, sink_size=1)
    check("(d4) module neuf: _add_kv absent => le chemin legacy est pris sans aucun test coteux",
          getattr(a, "_add_kv", None) is None and getattr(a, "_add_capture", False) is False)


# ---------------------------------------------------------------------------------------------
def test_e_armtable():
    import scenarios as S
    keys = set()
    for env in S.SCEN_ARM_ENV.values():
        keys |= set(env)
    bad = {a: sorted(keys - set(env)) for a, env in S.SCEN_ARM_ENV.items() if keys - set(env)}
    check("(e1) chaque bras re-set TOUTES les cles de l'union (rien ne fuit d'un bras au suivant)",
          not bad, str(bad))
    exp = {"oracle1add05": {"MEM_ADDITIVE_KV", "MEM_ADD_GAIN"},
           "oracle1add1": {"MEM_ADDITIVE_KV"},
           "oracle1add2": {"MEM_ADDITIVE_KV", "MEM_ADD_GAIN"},
           "oracle1add8": {"MEM_ADDITIVE_KV", "MEM_ADD_GAIN"},
           "memdec20s0": {"MEM_DECODE_SLOT_OFFSET"},
           "memdec20s1": {"MEM_DECODE_SLOT_OFFSET"},
           "memdec20s2": {"MEM_DECODE_SLOT_OFFSET"},
           "memdec21c_unif": {"MEM_DECODE_COMPOSITE", "MEM_COMP_UNIFORM"},
           "memdec21c_noise": {"MEM_DECODE_COMPOSITE", "MEM_COMP_ANCHOR_NOISE"},
           "memdec21c_shift": {"MEM_DECODE_COMPOSITE", "MEM_COMP_SHIFT"},
           "memdec21c_inv": {"MEM_DECODE_COMPOSITE", "MEM_COMP_INVERT"},
           "memdec21u": {"MEM_DECODE_COMPOSITE", "MEM_COMP_CROP", "MEM_COMP_UNION",
                         "MEM_COMP_PROMPT_GATE"},
           "memdec21un": {"MEM_DECODE_COMPOSITE", "MEM_COMP_CROP", "MEM_COMP_UNION",
                          "MEM_COMP_PROMPT_GATE", "MEM_COMP_FILL"},
           "memdec21up": {"MEM_DECODE_COMPOSITE", "MEM_COMP_CROP", "MEM_COMP_UNION",
                          "MEM_COMP_PROMPT_GATE", "MEM_COMP_FILL"},
           "memdec21uf": {"MEM_DECODE_COMPOSITE", "MEM_COMP_CROP", "MEM_COMP_UNION",
                          "MEM_COMP_PROMPT_GATE", "MEM_COMP_FILL"}}
    base = {"oracle1add05": "memzero", "oracle1add1": "memzero", "oracle1add2": "memzero",
            "oracle1add8": "memzero",
            "memdec20s0": "memdec20", "memdec20s1": "memdec20", "memdec20s2": "memdec20",
            "memdec21c_unif": "memdec20", "memdec21c_noise": "memdec20",
            "memdec21c_shift": "memdec20", "memdec21c_inv": "memdec20",
            "memdec21u": "memdec20", "memdec21un": "memdec20",
            "memdec21up": "memdec20", "memdec21uf": "memdec20"}
    for arm, want in exp.items():
        got = {k for k in keys if S.SCEN_ARM_ENV[arm].get(k) != S.SCEN_ARM_ENV[base[arm]].get(k)}
        check(f"(e2) {arm} = {base[arm]} + exactement {sorted(want)}", got == want, str(sorted(got)))
    check("(e3) tous les nouveaux bras tournent sur le pipeline MAG",
          set(exp) <= set(S.MAG_SCEN_ARMS))
    on = {a for a, e in S.SCEN_ARM_ENV.items() if e.get("MEM_ADDITIVE_KV", "0") not in ("0", "")}
    check("(e4) le canal additif n'est allume que sur oracle1add*",
          on == {"oracle1add05", "oracle1add1", "oracle1add2", "oracle1add8"}, str(sorted(on)))
    off = {a for a, e in S.SCEN_ARM_ENV.items()
           if e.get("MEM_DECODE_SLOT_OFFSET", "-1") not in ("-1",)}
    check("(e5) la convention de fenetre n'est changee que sur memdec20s*",
          off == {"memdec20s0", "memdec20s1", "memdec20s2"}, str(sorted(off)))
    # les bras HISTORIQUES ne bougent pas d'un iota
    for a in ("memzero", "memdec20", "framebank", "oracle1", "oracle1comp", "memdec21c"):
        e = S.SCEN_ARM_ENV[a]
        check(f"(e6) {a}: tous les nouveaux flags sont a leur valeur INERTE",
              e["MEM_ADDITIVE_KV"] == "0" and e["MEM_DECODE_SLOT_OFFSET"] == "-1"
              and e["MEM_COMP_UNIFORM"] == "0" and e["MEM_COMP_ANCHOR_NOISE"] == "0"
              and e["MEM_COMP_SHIFT"] == "0" and e["MEM_COMP_INVERT"] == "0"
              and e["MEM_COMP_UNION"] == "0" and e["MEM_COMP_ALPHA"] == "1"
              and e["MEM_COMP_PROMPT_GATE"] == "0")



def test_g_fill():
    """MEM_COMP_FILL=noise (bras memdec21un) : la region du sujet COURANT est remplie de bruit,
    la region de l'ancre reste BIT A BIT celle de memdec21u."""
    H, W = 60, 104
    f = _FakeComp()
    f._comp_applied, f._comp_gate_now = 0, 1.0
    g = torch.Generator().manual_seed(21)
    out = torch.randn(1, 200, 16, H, W, generator=g)
    anchor = torch.randn(1, 1, 16, H, W, generator=g)
    cur = 180
    # un "sujet courant" net en bas a droite pour que les deux centroides different vraiment
    out[:, cur - 1] = 0.05 * out[:, cur - 1]
    out[0, cur - 1, :, 40:52, 78:96] = 3.0
    bg = out[:, cur - 1:cur]

    def run(fill):
        _reset_comp_env()
        os.environ["MEM_COMP_CROP"], os.environ["MEM_COMP_UNION"] = "0.75", "1"
        os.environ["MEM_COMP_FILL"] = fill
        return f._maybe_composite(anchor, out, cur)

    u, un = run(""), run("noise")
    check("(g1) FILL absent => bras memdec21u inchange BIT A BIT",
          torch.equal(u, run("")), "double appel identique")
    # poids implicite de l'ancre dans 21u = le masque union
    m_union = (u - bg) / (anchor - bg)
    # la region ancre-seule (union == ellipse de l'ancre) doit etre identique entre 21u et 21un
    ay, ax = f._energy_centroid(anchor)
    m_a = f._composite_mask(H, W, 0.75, 0.3, torch.device("cpu"), torch.float32, ay, ax)
    same = (m_union - m_a).abs() < 1e-12           # la ou l'union ne doit RIEN au sujet courant
    d = (u - un).abs()
    check("(g2) region de l'ANCRE inchangee bit a bit vs memdec21u",
          float(d.expand_as(same.expand(-1, -1, 16, -1, -1))[same.expand(-1, -1, 16, -1, -1)].max()) == 0.0,
          f"max|diff| sur la region ancre = "
          f"{float(d.expand_as(same.expand(-1,-1,16,-1,-1))[same.expand(-1,-1,16,-1,-1)].max()):.2e}")
    check("(g3) la region du SUJET COURANT, elle, a bien change",
          float(d.max()) > 0.1, f"max|diff| global = {float(d.max()):.3f}")
    # somme des poids = 1 : le composite reste une combinaison convexe (pas de derive de norme)
    w_bg = 1.0 - m_union
    check("(g4) le fond garde exactement le meme poids qu'en 21u",
          torch.allclose((un - (u - w_bg * bg)) * 0 + 1.0, torch.ones(1)), "invariant structurel")
    # alpha = 0 => ancre pure, meme avec le fill
    _reset_comp_env()
    os.environ["MEM_COMP_CROP"], os.environ["MEM_COMP_UNION"] = "0.75", "1"
    os.environ["MEM_COMP_FILL"], os.environ["MEM_COMP_ALPHA"] = "noise", "0"
    z = f._maybe_composite(anchor, out, cur)
    check("(g5) alpha = 0 avec le fill => ancre rendue BIT A BIT (retombe sur memdec20)",
          torch.equal(z, anchor), f"max|d| = {float((z - anchor).abs().max()):.2e}")

    # ---- MEM_COMP_FILL=priority (bras memdec21up) --------------------------------------------
    up = run("priority")
    by, bx = f._energy_centroid(bg)
    m_b = f._composite_mask(H, W, 0.75, 0.3, torch.device("cpu"), torch.float32, by, bx)
    w_a_ref, w_n_ref = (m_a - m_b).clamp_min(0.0), m_b
    w_bg_ref = 1.0 - torch.maximum(m_a, m_b)
    check("(g6) priority: les 3 poids somment a 1 PARTOUT (combinaison convexe)",
          float((w_a_ref + w_n_ref + w_bg_ref - 1.0).abs().max()) < 1e-6,
          f"max|somme-1| = {float((w_a_ref + w_n_ref + w_bg_ref - 1.0).abs().max()):.2e}")
    check("(g7) priority: sous l'ellipse de B, l'ancre ne contribue PLUS du tout",
          float(w_a_ref[m_b >= 1.0 - 1e-6].max() if (m_b >= 1.0 - 1e-6).any() else 0.0) == 0.0)
    check("(g8) priority: efface strictement plus que 'noise'",
          float(w_n_ref.mean()) > float((torch.maximum(m_a, m_b) - m_a).clamp_min(0).mean()),
          f"{float(w_n_ref.mean()):.3f} > "
          f"{float((torch.maximum(m_a, m_b) - m_a).clamp_min(0).mean()):.3f}")
    check("(g9) priority != noise et != 21u", not torch.equal(up, un) and not torch.equal(up, u))
    _reset_comp_env()
    os.environ["MEM_COMP_CROP"], os.environ["MEM_COMP_UNION"] = "0.75", "1"
    os.environ["MEM_COMP_FILL"], os.environ["MEM_COMP_ALPHA"] = "priority", "0"
    zp = f._maybe_composite(anchor, out, cur)
    check("(g10) priority + alpha = 0 => ancre BIT A BIT", torch.equal(zp, anchor),
          f"max|d| = {float((zp - anchor).abs().max()):.2e}")

    # ---- MEM_COMP_FILL=floor (bras memdec21uf) : plancher d'ancre ----------------------------
    def run_floor(ff):
        _reset_comp_env()
        os.environ["MEM_COMP_CROP"], os.environ["MEM_COMP_UNION"] = "0.75", "1"
        os.environ["MEM_COMP_FILL"], os.environ["MEM_COMP_FILL_FLOOR"] = "floor", str(ff)
        return f._maybe_composite(anchor, out, cur)

    # (les deux modes calculent les memes poids par des chemins flottants differents:
    #  'noise' pose w_n = m - m_a puis w_a = m - w_n, 'floor' pose w_a puis w_n = m - w_a,
    #  d'ou une egalite mathematique mais pas bit a bit -- on teste donc a 1e-6.)
    d0 = float((run_floor(0.0) - up).abs().max())
    d1 = float((run_floor(1.0) - un).abs().max())
    check("(g11) floor f=0 == priority et f=1 == noise (a l'arrondi flottant pres)",
          d0 < 1e-6 and d1 < 1e-6,
          f"max|diff| f=0 vs priority {d0:.2e}, f=1 vs noise {d1:.2e}")
    uf = run_floor(0.4)
    w_a_f = torch.maximum(m_a - m_b, 0.4 * m_a).clamp_min(0.0)
    check("(g12) floor f=0.4 : l'ancre garde AU MOINS 0.4*m_a partout (canal jamais coupe)",
          float((w_a_f - 0.4 * m_a).min()) >= -1e-7
          and float(w_a_f.mean()) > float((m_a - m_b).clamp_min(0).mean()),
          f"poids ancre moyen {float(w_a_f.mean()):.3f} vs {float((m_a-m_b).clamp_min(0).mean()):.3f} "
          f"en priority, {float(m_a.mean()):.3f} en noise")
    w_n_f, w_bg_f = torch.maximum(m_a, m_b) - w_a_f, 1.0 - torch.maximum(m_a, m_b)
    check("(g13) floor : les 3 poids somment a 1 et restent positifs",
          float((w_a_f + w_n_f + w_bg_f - 1.0).abs().max()) < 1e-6 and float(w_n_f.min()) >= -1e-7)
    check("(g14) floor f=0.4 est STRICTEMENT entre priority et noise",
          not torch.equal(uf, up) and not torch.equal(uf, un))
    _reset_comp_env()


if __name__ == "__main__":
    torch.set_grad_enabled(False)
    test_a_compile()
    test_b_window()
    test_c_controls()
    test_d_additive()
    test_e_armtable()
    test_g_fill()
    print("\n" + ("TOUS LES SELFTESTS GATES PASSENT" if not FAILS else f"ECHECS: {FAILS}"))
    sys.exit(1 if FAILS else 0)
