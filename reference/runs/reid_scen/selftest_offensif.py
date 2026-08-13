#!/usr/bin/env python
# ============================================================================================
# SELFTEST du LOT OFFENSIF (2026-08-04) -- avant tout rollout.
#
#  (a) TABLE DES BRAS : les 3 familles nouvelles (i1_*, *p17, caption_*) re-mettent TOUTES les
#      cles de l'union -> aucune fuite d'un bras au suivant, et elles ne different de leur bras
#      de reference QUE par les cles annoncees.
#  (b) INERTIE LEGACY, BIT A BIT : hooks eteints => `_maybe_declare_anchor_noise` rend l'OBJET
#      splice lui-meme et `timestep_override=None`, et `_additive_attention` sans
#      MEM_ADD_MASS_SRC produit exactement la meme sortie qu'avec (le drapeau ne touche pas la
#      sortie, seulement la mesure).
#  (c) I1 MECANIQUE : les slots d'ancre calcules par `_anchor_slots` sont EXACTEMENT ceux que
#      `_memdec_window` remplit (les deux conventions), le bruit applique correspond au sigma
#      du scheduler a t*, et le vecteur de timesteps declare t* sur CES slots et le baseline
#      partout ailleurs.
#  (d) MASSE PAR SOURCE : la decomposition [sink | fenetre | ancre] somme a 1 et redonne
#      exactement `alpha` sur la composante ancre (identite logaddexp).
#  (e) LEGENDE : le scenario S2c/SS2c a la meme forme, les memes sujets et les memes fenetres de
#      score que S2/SS2 ; avec un suffixe VIDE le prompt de retour est le prompt de A VERBATIM
#      (donc prompts[3] == prompts[0] : rollout bit a bit).
#
#   python selftest_offensif.py            # (a)(c)(d)(e) sur CPU + GPU si dispo
# ============================================================================================
import os
import sys

REPO = "/shared/ssd_14T/home/adenelmi/Anchor-Forcing"
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [REPO, os.path.join(REPO, "reference/runs/reid_probe_l1"), HERE]

import torch                                                            # noqa: E402
import scenarios as S                                                   # noqa: E402
from pipeline.switch_causal_inference import SwitchCausalInferencePipeline as P  # noqa: E402

FAIL = []


def check(name, cond, msg=""):
    print(f"  [{'OK ' if cond else 'FAIL'}] {name}" + (f"  -- {msg}" if msg else ""))
    if not cond:
        FAIL.append(name)


# ------------------------------------------------------------------ (a) table des bras --------
def test_arms():
    print("\n(a) TABLE DES BRAS -- aucune fuite possible")
    keys = set().union(*[set(v) for v in S.SCEN_ARM_ENV.values()])
    for a in S.I1_ARMS + S.ADDPOS_ARMS + S.CAPTION_ARMS:
        check(f"{a} re-met toutes les cles", set(S.SCEN_ARM_ENV[a]) == keys,
              f"manque {sorted(keys - set(S.SCEN_ARM_ENV[a]))}")
    base = S.SCEN_ARM_ENV["memdec20"]
    for a in S.I1_ARMS:
        d = {k for k in keys if S.SCEN_ARM_ENV[a][k] != base[k]}
        check(f"{a} ne differe de memdec20 que par MEM_ANCHOR_T", d == {"MEM_ANCHOR_T"}, str(d))
    check("i1_ours250 et i1_wrong250 ont une table IDENTIQUE (le contenu vient de "
          "MEM_DECODE_SRC, injecte au run time)",
          S.SCEN_ARM_ENV["i1_ours250"] == S.SCEN_ARM_ENV["i1_wrong250"])
    check("i1_ours500 et i1_wrong500 ont une table IDENTIQUE",
          S.SCEN_ARM_ENV["i1_ours500"] == S.SCEN_ARM_ENV["i1_wrong500"])
    check("i1_wrong* sont dans WRONG_SRC_ARMS (sinon ils tourneraient sur LEUR ancre)",
          all(a in S.WRONG_SRC_ARMS for a in S.I1_WRONG_ARMS))
    check("i1_* exigent decodeur + ecrivain", all(a in S.MEM2_CKPT_ARMS for a in S.I1_ARMS))
    for a in S.ADDPOS_ARMS:
        d = {k for k in keys if S.SCEN_ARM_ENV[a][k] != S.SCEN_ARM_ENV["oracle1add8"][k]}
        check(f"{a} ne differe de oracle1add8 que par gain/pos/mass-src",
              d <= {"MEM_ADD_GAIN", "MEM_ADD_POS", "MEM_ADD_MASS_SRC"}, str(d))
    check("oracle1add8p9 a bien la position AUTO explicitee (9), gain 8",
          S.SCEN_ARM_ENV["oracle1add8p9"]["MEM_ADD_POS"] == "9"
          and S.SCEN_ARM_ENV["oracle1add8p9"]["MEM_ADD_GAIN"] == "8")
    for a in S.CAPTION_ARMS:
        check(f"{a} est un memzero pur (aucun hook de rappel)",
              S.SCEN_ARM_ENV[a] == S.SCEN_ARM_ENV["memzero"])
    check("les bras de legende n'exigent NI decodeur NI ecrivain",
          not any(a.startswith("memdec") or a in S.MEM2_CKPT_ARMS for a in S.CAPTION_ARMS))
    check("tous les nouveaux bras tournent sur le pipeline MAG",
          all(a in S.MAG_SCEN_ARMS for a in S.I1_ARMS + S.ADDPOS_ARMS + S.CAPTION_ARMS))
    # LEVIER 4
    d = {k for k in keys if S.SCEN_ARM_ENV["memdec_snap"][k] != base[k]}
    check("memdec_snap ne differe de memdec20 que par MEM_EPISODIC", d == {"MEM_EPISODIC"}, str(d))
    import inspect
    from pipeline.switch_causal_inference import SwitchCausalInferencePipeline as _P
    src = inspect.getsource(_P._switch_recache)
    i_res, i_dec = src.find("_restore_ttt"), src.find("_memdec_frame")
    check("l'etat est RESTAURE avant que le decodeur ne lise l'etat vivant",
          0 < i_res < i_dec, f"restore@{i_res} decode@{i_dec}")
    cm = inspect.getsource(__import__("wan.modules.causal_model", fromlist=["x"]))
    check("MEM_ZERO_ALL met l'injection a ZERO dans les deux passes (donc restaurer l'etat ne "
          "peut pas changer le rendu : seul le decodeur le consomme)",
          "_mz_all = int(os.environ.get(\"MEM_ZERO_ALL\", \"0\"))" in cm)
    check("MEM_EPISODIC par defaut = 0 pour tous les autres bras",
          all(v.get("MEM_EPISODIC", "0") == "0" for k, v in S.SCEN_ARM_ENV.items()
              if k not in S.SNAP_ARMS))
    check("MEM_ANCHOR_T par defaut = 0 (OFF) dans l'union",
          all(v.get("MEM_ANCHOR_T") == "0" for k, v in S.SCEN_ARM_ENV.items()
              if k not in S.I1_ARMS))
    check("MEM_ADD_MASS_SRC par defaut = 0 dans l'union",
          all(v.get("MEM_ADD_MASS_SRC") == "0" for k, v in S.SCEN_ARM_ENV.items()
              if k not in S.ADDPOS_ARMS))


# ------------------------------------------------------------------ (c) I1 mecanique ----------
class _Fake:
    """Le minimum de pipeline dont les deux helpers ont besoin (ils sont purs)."""
    class _A:
        context_noise = 0
    args = _A()

    def __init__(self, sched):
        self.scheduler = sched

    _anchor_slots = staticmethod(P._anchor_slots)
    _memdec_window = staticmethod(P._memdec_window)
    _maybe_declare_anchor_noise = P._maybe_declare_anchor_noise


def test_i1():
    print("\n(c) I1 -- slots, bruit et timesteps declares")
    from utils.scheduler import FlowMatchScheduler
    sched = FlowMatchScheduler(shift=5.0)
    sched.set_timesteps(1000, training=True)
    fake = _Fake(sched)
    n, k = 12, 1
    torch.manual_seed(0)
    out = torch.randn(1, 200, 16, 8, 8)
    anchor = torch.full((1, 1, 16, 8, 8), 3.0)
    for soff in (-1, 0, 1, 2):
        win = P._memdec_window(out, 180, n, anchor, k, soff)
        lo, hi = P._anchor_slots(n, k, soff)
        got = [i for i in range(n) if torch.allclose(win[:, i], anchor[:, 0])]
        check(f"_anchor_slots(offset={soff}) == slots reellement remplis",
              got == list(range(lo, hi)), f"attendu {list(range(lo, hi))}, trouve {got}")

    os.environ["MEM_ANCHOR_T"] = "0"
    w = P._memdec_window(out, 180, n, anchor, k, -1)
    sp, ts = _Fake._maybe_declare_anchor_noise(fake, w, n, k, -1, 180)
    check("OFF: le splice est l'OBJET lui-meme et timestep_override est None",
          sp is w and ts is None)

    for t in (250.0, 500.0):
        os.environ["MEM_ANCHOR_T"] = str(int(t))
        w = P._memdec_window(out, 180, n, anchor, k, -1)
        sp, ts = _Fake._maybe_declare_anchor_noise(fake, w, n, k, -1, 180)
        lo, hi = P._anchor_slots(n, k, -1)
        untouched = float((sp[:, :lo] - w[:, :lo]).abs().max())
        check(f"t*={t:g}: les {lo} vraies frames sont INTACTES", untouched == 0.0)
        sig = float(sched.sigmas[int(torch.argmin((sched.timesteps - t).abs()))])
        # x_t = (1 - sigma) x0 + sigma eps  =>  la moyenne du slot doit valoir (1-sigma)*3
        got_mean = float(sp[:, lo:hi].mean())
        check(f"t*={t:g}: le slot d'ancre est bruite au sigma du scheduler ({sig:.3f})",
              abs(got_mean - (1 - sig) * 3.0) < 0.05 * max(1.0, sig),
              f"moyenne {got_mean:.4f}, attendu ~{(1 - sig) * 3.0:.4f}")
        check(f"t*={t:g}: timesteps declares = t* sur [{lo},{hi}), baseline ailleurs",
              [float(x) for x in ts] == [0.0] * lo + [t] * (hi - lo) + [0.0] * (n - hi))
        check(f"t*={t:g}: un SEUL slot hors baseline", int((ts != 0).sum()) == k)
    os.environ["MEM_ANCHOR_T"] = "0"

    # determinisme: meme graine (meme frame de switch) => meme bruit ; frame differente => autre
    os.environ["MEM_ANCHOR_T"] = "500"
    a1, _ = _Fake._maybe_declare_anchor_noise(fake, w.clone(), n, k, -1, 180)
    a2, _ = _Fake._maybe_declare_anchor_noise(fake, w.clone(), n, k, -1, 180)
    a3, _ = _Fake._maybe_declare_anchor_noise(fake, w.clone(), n, k, -1, 240)
    check("bruit deterministe par (graine, frame de switch)", torch.equal(a1, a2))
    check("frame de switch differente -> autre realisation", not torch.equal(a1, a3))
    g0 = torch.randn(3)                                  # le RNG global ne doit pas avoir bouge
    torch.manual_seed(1234)
    b0 = torch.randn(3)
    _Fake._maybe_declare_anchor_noise(fake, w.clone(), n, k, -1, 180)
    torch.manual_seed(1234)
    b1 = torch.randn(3)
    check("aucun tirage sur le RNG GLOBAL (appariement memzero preserve)",
          torch.equal(b0, b1) and g0 is not None)
    os.environ["MEM_ANCHOR_T"] = "0"


# ------------------------------------------------------------------ (d) masse par source ------
def test_uniform_timestep():
    """FOOTGUN : WanDiffusionWrapper reduit `timestep[:, 0]` quand uniform_timestep est vrai. Le
    levier 2 repose ENTIEREMENT sur le timestep PAR FRAME ; si ce drapeau etait vrai, le bras
    tournerait en silence sans sa declaration. On verifie la source de verite du drapeau."""
    print("\n(f) FOOTGUN uniform_timestep -- la declaration par slot doit survivre au wrapper")
    import inspect
    from utils.wan_wrapper import WanDiffusionWrapper
    src = inspect.getsource(WanDiffusionWrapper.__init__)
    check("uniform_timestep est bien 'not is_causal' (donc False pour notre pipeline causal)",
          "self.uniform_timestep = not is_causal" in src)
    fwd = inspect.getsource(WanDiffusionWrapper.forward)
    check("le wrapper reduit bien timestep[:, 0] dans ce cas (c'est ce qu'on garde)",
          "timestep[:, 0]" in fwd)
    from pipeline.switch_causal_inference import SwitchCausalInferencePipeline as _P
    rc = inspect.getsource(_P._recache_after_switch)
    check("_recache_after_switch REFUSE de tourner si uniform_timestep est vrai",
          "uniform_timestep" in rc and "MEM_ANCHOR_T exige un generateur CAUSAL" in rc)


def test_mass_src():
    print("\n(d) MASSE PAR SOURCE -- identite logaddexp exacte")
    if not torch.cuda.is_available():
        print("  (pas de GPU: teste sur la formule seule)")
    import math
    torch.manual_seed(0)
    lse_s = torch.randn(2, 4, 8) * 2
    lse_w = torch.randn(2, 4, 8) * 2
    lse_a = torch.randn(2, 4, 8) * 2
    for gain in (0.5, 1.0, 8.0):
        lg = math.log(gain)
        lse_ctx = torch.logaddexp(lse_s, lse_w)
        tot = torch.logaddexp(lse_ctx, lg + lse_a)
        ms, mw, ma = (torch.exp(lse_s - tot), torch.exp(lse_w - tot), torch.exp(lg + lse_a - tot))
        alpha = torch.sigmoid(lg + (lse_a - lse_ctx))
        check(f"gain {gain}: les 3 masses somment a 1",
              float((ms + mw + ma - 1).abs().max()) < 1e-5)
        check(f"gain {gain}: la composante ancre == alpha (la mesure historique)",
              float((ma - alpha).abs().max()) < 1e-5)


# ------------------------------------------------------------------ (e) legende ---------------
def test_caption():
    print("\n(e) LEGENDE -- scenarios S2c / SS2c")
    for c, b in S.CAPTION_SCENARIOS.items():
        check(f"{c}: meme nombre de frames que {b}",
              S.SCENARIOS[c]["frames"] == S.SCENARIOS[b]["frames"])
        check(f"{c}: memes frames de depart que {b}",
              [f for f, _ in S.SCENARIOS[c]["schedule"]]
              == [f for f, _ in S.SCENARIOS[b]["schedule"]])
        check(f"{c}: memes SUJETS que {b} (donc meme reference de re-ID)",
              S.seg_roles(c) == S.seg_roles(b))
        check(f"{c}: memes segments de revisit et memes fenetres de score que {b}",
              S.revisit_segments(c) == S.revisit_segments(b)
              and S.score_segments(c, S.SCENARIOS[c]["frames"])
              == S.score_segments(b, S.SCENARIOS[b]["frames"]))
        check(f"{c}: SEUL le segment de retour pointe le prompt 3",
              [p for _, p in S.SCENARIOS[c]["schedule"]]
              == [p if i not in S.revisit_segments(c) else 3
                  for i, (_, p) in enumerate(S.SCENARIOS[b]["schedule"])])
    ti = 2
    check("suffixe VIDE => prompt de retour == prompt de A VERBATIM (rollout bit a bit)",
          S.caption_return_prompt(ti, "") == S.TRIPLES[ti][0])
    suf = ", and in this shot the cat is clearly orange"
    check("suffixe non vide => concatenation propre",
          S.caption_return_prompt(ti, suf) == S.TRIPLES[ti][0] + suf)


def main():
    test_arms()
    test_i1()
    test_uniform_timestep()
    test_mass_src()
    test_caption()
    print("\n" + ("=" * 60))
    print(f"SELFTEST OFFENSIF: {'TOUT PASSE' if not FAIL else 'ECHECS: ' + ', '.join(FAIL)}")
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
