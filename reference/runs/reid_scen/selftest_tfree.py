#!/usr/bin/env python
# ============================================================================================
# SELFTEST de la MINI-BATTERIE TRAINING-FREE (2026-08-04)
#
#   sdedit_ours<t>   MEM_SDEDIT=sigma : l'ancre DECODEE de notre etat est AFFUTEE par le prior du
#                    modele (SDEdit latent, mode 1 frame, prompt de RETOUR verbatim) avant splice.
#   sdedit_wrong<t>  IDENTIQUE, sur l'ancre decodee d'une AUTRE cellule (MEM_DECODE_SRC) : le
#                    contraste apparie ours - wrong est LE chiffre de la batterie.
#   sinkpurge        MEM_SINKPURGE=1 : AUCUN splice, AUCUNE memoire ; au recache du retour les
#                    slots du sink EPINGLE (l'ouvreur) sont reecrits avec la frame vraie courante.
#
# Ce que le selftest prouve :
#   (a) tout ce qui est touche compile ;
#   (b) TABLE DES BRAS : les nouveaux bras re-settent TOUTES les cles de l'union, sdedit_* ne
#       differe de memdec20 que par MEM_SDEDIT, sdedit_wrong<t> a une ligne IDENTIQUE a
#       sdedit_ours<t> (son contenu est injecte au run, comme wronganchor), sinkpurge ne differe
#       de memzero que par MEM_SINKPURGE, et aucun autre bras n'allume les 2 hooks ;
#   (c) INERTIE LEGACY (le point qui protege les 1000+ mp4 deja sur disque) : hooks eteints =>
#       _maybe_sdedit renvoie l'OBJET ancre lui-meme (pas une copie) et _maybe_sinkpurge ne
#       touche a rien (0 appel a _write_sink_anchor) ;
#   (d) ARITHMETIQUE SDEdit : la trajectoire = [1000*sigma] + les pas du sampler strictement en
#       dessous, et sigma est bien le sigma flow-matching (add_noise a 1000*s donne
#       (1-s) x0 + s eps) -- verifie contre le scheduler REEL de la config de campagne ;
#   (e) STRUCTURE / EXCLUSIONS : _maybe_sdedit est appele DANS le bloc MEM_DECODE, APRES le dump
#       et APRES les 2 controles de contenu (donc sdedit_wrong = wronganchor + le meme affutage),
#       et AVANT le composite ; _maybe_sinkpurge est appele APRES le recache ; sinkpurge refuse
#       de se combiner a un bras de rappel ; inference() legacy refuse les 2 hooks ;
#   (f) SEEDS : la graine SDEdit depend de la CELLULE (les 6 cellules appariees ne partagent pas
#       une realisation) et de la frame du switch ;
#   (g) [--gpu] MECANIQUE REELLE sur une ancre decodee vraie : l'affutage change l'ancre, garde
#       shape/dtype/device, ne consomme AUCUN RNG global (l'appariement memzero est preserve),
#       ne touche NI le kv cache reel NI l'etat memoire, et monte la nettete latente ;
#   (h) [--audit-logs] les logs des rollouts REELS : exactement 1 affutage / 1 purge par cellule,
#       au bon instant, et les tripwires mecaniques du hook sont passes.
#
#   python selftest_tfree.py                     # CPU, ~3 s
#   python selftest_tfree.py --gpu               # + la mecanique reelle (~2 min, 1 GPU)
#   python selftest_tfree.py --audit-logs logs/tfree_*.log
# ============================================================================================
import argparse
import glob
import inspect
import os
import re
import sys

REPO = "/shared/ssd_14T/home/adenelmi/Anchor-Forcing"
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [REPO, os.path.join(REPO, "reference/runs/reid_probe_l1"), HERE]

import torch                                                            # noqa: E402

FAILS = []
NEW_KEYS = ("MEM_SDEDIT", "MEM_SDEDIT_SEED", "MEM_SDEDIT_FRAMES", "MEM_SINKPURGE")


def check(name, ok, detail=""):
    print(f"[{'OK ' if ok else 'FAIL'}] {name}" + (f"  -- {detail}" if detail else ""), flush=True)
    if not ok:
        FAILS.append(name)


def _clean_env():
    for k in NEW_KEYS + ("MEM_DECODE", "MEM_DECODE_SRC", "MEM_DECODE_NOISE", "ORACLE_SPLICE",
                         "ORACLE_SPLICE_POS", "FRAMEBANK"):
        os.environ.pop(k, None)


# ---------------------------------------------------------------------------------------------
def test_a_compile():
    import py_compile
    ok = True
    for p in ("pipeline/switch_causal_inference.py",
              "reference/runs/reid_scen/scenarios.py",
              "reference/runs/reid_scen/run_reid_scen.py",
              "reference/runs/reid_probe_l1/run_reid_probe.py",
              "reference/runs/reid_scen/score_comp.py"):
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
    new = list(S.SDEDIT_ARMS) + ["sinkpurge"]
    union = set().union(*(set(v) for v in S.SCEN_ARM_ENV.values()))
    missing = {a: sorted(union - set(S.SCEN_ARM_ENV[a])) for a in new}
    check("(b1) les nouveaux bras re-settent TOUTES les cles de l'union",
          not any(missing.values()), str({k: v for k, v in missing.items() if v}))
    check("(b2) les 3 nouvelles cles sont dans l'union (tous les AUTRES bras les eteignent)",
          all(k in union for k in NEW_KEYS),
          str([k for k in NEW_KEYS if k not in union]))
    on = [a for a, e in S.SCEN_ARM_ENV.items()
          if a not in new and (e.get("MEM_SDEDIT", "0") not in ("0", "")
                               or e.get("MEM_SINKPURGE", "0") != "0")]
    check("(b3) aucun autre bras n'allume les 2 hooks", not on, str(on))
    bad = []
    for tag, sig in S.SDEDIT_SIGMAS.items():
        for who in ("ours", "wrong"):
            a = f"sdedit_{who}{tag}"
            d = {k for k in union if S.SCEN_ARM_ENV[a].get(k)
                 != S.SCEN_ARM_ENV["memdec20"].get(k)}
            if d != {"MEM_SDEDIT"} or S.SCEN_ARM_ENV[a]["MEM_SDEDIT"] != sig:
                bad.append((a, sorted(d), S.SCEN_ARM_ENV[a].get("MEM_SDEDIT")))
    check("(b4) sdedit_* - memdec20 = {MEM_SDEDIT} exactement, au bon sigma", not bad, str(bad))
    diff = [t for t in S.SDEDIT_SIGMAS
            if S.SCEN_ARM_ENV[f"sdedit_ours{t}"] != S.SCEN_ARM_ENV[f"sdedit_wrong{t}"]]
    check("(b5) la ligne de sdedit_wrong<t> est IDENTIQUE a sdedit_ours<t> (contenu injecte au "
          "run) => les deux bras ne different QUE par l'origine de l'ancre", not diff, str(diff))
    d = {k for k in union
         if S.SCEN_ARM_ENV["sinkpurge"].get(k) != S.SCEN_ARM_ENV["memzero"].get(k)}
    check("(b6) sinkpurge - memzero = {MEM_SINKPURGE} exactement", d == {"MEM_SINKPURGE"},
          str(sorted(d)))
    check("(b7) sinkpurge ne demande NI decodeur NI ecrivain (aucun splice, aucune memoire)",
          "sinkpurge" not in S.MEM2_CKPT_ARMS and not S.SCEN_ARM_ENV["sinkpurge"]["MEM_DECODE"])
    check("(b8) les sdedit_* sont bien dans MAG_SCEN_ARMS et MEM2_CKPT_ARMS (ils decodent)",
          all(a in S.MAG_SCEN_ARMS and a in S.MEM2_CKPT_ARMS for a in S.SDEDIT_ARMS)
          and "sinkpurge" in S.MAG_SCEN_ARMS)


# ---------------------------------------------------------------------------------------------
class _Dummy:
    """Le strict minimum pour appeler les helpers sans construire de pipeline."""
    from pipeline.switch_causal_inference import SwitchCausalInferencePipeline as _P
    _maybe_sdedit = _P._maybe_sdedit
    _maybe_sinkpurge = _P._maybe_sinkpurge
    _sdedit_steps = _P._sdedit_steps
    _lat_hf = _P._lat_hf
    _fb_mode = _P._fb_mode
    global_sink = True

    def __init__(self):
        self.denoising_step_list = torch.tensor([1000.0, 937.5, 833.3333, 625.0])
        self.n_sink_writes = 0

    def _write_sink_anchor(self, *a, **k):
        self.n_sink_writes += 1


def test_c_inertia():
    _clean_env()
    d = _Dummy()
    x = torch.randn(1, 1, 16, 60, 104)
    out = d._maybe_sdedit(x, None, 180)
    check("(c1) MEM_SDEDIT eteint => l'OBJET ancre lui-meme est renvoye (aucune copie, aucun "
          "changement de flux) => tous les bras existants sont bit a bit inchanges", out is x)
    os.environ["MEM_SDEDIT"] = "0"
    check("(c2) MEM_SDEDIT='0' idem", d._maybe_sdedit(x, None, 180) is x)
    _clean_env()
    d._maybe_sinkpurge(torch.randn(1, 200, 16, 60, 104), 180, None, True)
    check("(c3) MEM_SINKPURGE eteint => 0 ecriture de sink", d.n_sink_writes == 0)
    os.environ["MEM_SINKPURGE"] = "1"
    d._maybe_sinkpurge(torch.randn(1, 200, 16, 60, 104), 180, None, False)
    check("(c4) MEM_SINKPURGE=1 sur un switch qui n'est PAS un retour => 0 ecriture "
          "(meme instant que les bras de splice)", d.n_sink_writes == 0)
    _clean_env()


# ---------------------------------------------------------------------------------------------
def test_d_sdedit_arithmetic():
    d = _Dummy()
    grid = [1000.0, 937.5, 833.3333, 625.0]
    cases = {0.4: [400.0], 0.5: [500.0], 0.6: [600.0], 0.625: [625.0],
             0.7: [700.0, 625.0], 0.833: [833.0, 625.0],
             0.9: [900.0, 833.3333, 625.0], 1.0: [1000.0, 937.5, 833.3333, 625.0]}
    bad = []
    for s, want in cases.items():
        got = [round(t, 4) for t in d._sdedit_steps(s)]
        if got != [round(t, 4) for t in want]:
            bad.append((s, got, want))
    check("(d1) trajectoire = [1000*sigma] + les pas du sampler STRICTEMENT en dessous", not bad,
          str(bad))
    check("(d2) sigma <= 0.625 => UN seul appel modele (sous la grille distillee) ; sigma = 1 "
          "=> le sampler complet (= generation pure par le prompt, plancher de la batterie)",
          len(d._sdedit_steps(0.6)) == 1
          and [round(t, 3) for t in d._sdedit_steps(1.0)] == [round(t, 3) for t in grid],
          str([round(t, 2) for t in d._sdedit_steps(1.0)]))
    # le sigma DEMANDE est bien le sigma flow-matching du scheduler REEL de la campagne
    from utils.scheduler import FlowMatchScheduler
    sch = FlowMatchScheduler(shift=5.0, sigma_min=0.0, extra_one_step=True)
    sch.set_timesteps(1000, training=True)
    x0 = torch.zeros(1, 4, 4, 4)
    eps = torch.ones(1, 4, 4, 4)
    err = []
    for s in (0.4, 0.5, 0.6, 0.833):
        xt = sch.add_noise(x0, eps, torch.tensor([s * 1000.0]))
        err.append(abs(float(xt.mean()) - s))
    check("(d3) MEM_SDEDIT=s donne bien x_t = (1-s) x0 + s eps (timesteps = sigmas*1000)",
          max(err) < 2e-3, f"max|erreur| = {max(err):.2e}")
    tl = torch.cat((sch.timesteps.cpu(), torch.tensor([0.0])))
    warped = [float(x) for x in tl[1000 - torch.tensor([1000, 750, 500, 250])]]
    check("(d4) la grille du sampler de la campagne est bien ~[1000, 937.5, 833.3, 625]",
          max(abs(a - b) for a, b in zip(warped, grid)) < 0.1, str([round(w, 2) for w in warped]))


# ---------------------------------------------------------------------------------------------
def test_e_structure():
    from pipeline.switch_causal_inference import SwitchCausalInferencePipeline as P
    src = inspect.getsource(P._switch_recache)
    i_dump = src.index("MEM_DECODE_DUMP")
    i_wrong = src.index("_maybe_wrong_anchor")
    i_noise = src.index("_maybe_noise_anchor")
    i_sde = src.index("_maybe_sdedit")
    i_comp = src.rindex("_maybe_composite")        # celui du bloc MEM_DECODE (le 1er est ORACLE)
    i_dec = src.index('_dec_ckpt = os.environ.get("MEM_DECODE"')
    check("(e1) _maybe_sdedit est DANS le bloc MEM_DECODE, APRES le dump et les 2 controles de "
          "contenu, et AVANT le composite (donc sdedit_wrong == wronganchor + le meme affutage)",
          i_dec < i_dump < i_wrong < i_noise < i_sde < i_comp,
          f"dec {i_dec} dump {i_dump} wrong {i_wrong} noise {i_noise} sdedit {i_sde} comp {i_comp}")
    i_rec = src.rindex("_recache_after_switch(")
    i_sp = src.index("_maybe_sinkpurge")
    check("(e2) _maybe_sinkpurge est appele APRES le recache (il reecrit le sink du cache que le "
          "retour va lire)", i_rec < i_sp, f"recache {i_rec} sinkpurge {i_sp}")
    leg = inspect.getsource(P.inference)
    check("(e3) inference() legacy REFUSE les 2 hooks (ils n'existent que dans le driver N-switch)",
          "MEM_SDEDIT" in leg and "MEM_SINKPURGE" in leg)
    sp = inspect.getsource(P._maybe_sinkpurge)
    check("(e4) sinkpurge refuse de se combiner a un bras de rappel (MEM_DECODE / ORACLE_SPLICE "
          "/ FRAMEBANK) et a ORACLE_SPLICE_POS=sink0",
          all(k in sp for k in ("MEM_DECODE", "ORACLE_SPLICE", "_fb_mode", "sink0")))
    check("(e5) sinkpurge porte ses tripwires mecaniques DANS le hook (le sink DOIT changer, la "
          "fenetre locale NE DOIT PAS bouger)",
          "did not change" in sp and "must not" in sp)
    sd = inspect.getsource(P._sdedit_frame)
    check("(e6) l'affutage tourne sur un cache KV FRAIS, de la TAILLE DE BLOC NATIVE du modele "
          "(le bloc de 1 frame est OOD : prouve par la planche, sigma=1 rendait du bruit), et ne "
          "touche pas le cache reel",
          "_real = self.kv_cache1" in sd and "self.kv_cache1 = _real" in sd
          and "kv_cache_size_override=nf * self.frame_seq_length" in sd
          and "self.num_frame_per_block" in sd and "MEM_SDEDIT_FRAMES" in sd)
    check("(e7) l'affutage utilise un generateur DEDIE (aucun tirage sur le flux global)",
          "torch.Generator(" in sd and "torch.randn(" in sd and "generator=_g" in sd)
    check("(e8) l'affutage ne met a jour AUCUNE memoire (ttt_update=False)",
          "ttt_update=False" in sd and "ttt_update=True" not in sd)


# ---------------------------------------------------------------------------------------------
def test_f_seeds():
    import scenarios as S
    from run_reid_probe import cell_seed
    from omegaconf import OmegaConf
    cfg = OmegaConf.create({"seed": 42})
    seeds = {(sc, t): 5150241 + cell_seed(cfg, t, 0)
             for sc in ("SS2", "S2") for t in (12, 13, 14, 2, 4, 11)}
    check("(f1) la graine SDEdit est DISTINCTE par triple (6 cellules, 6 realisations)",
          len(set(seeds.values())) == len({t for _, t in seeds}) == 6, str(sorted(set(seeds.values()))))
    src = inspect.getsource(
        __import__("pipeline.switch_causal_inference", fromlist=["x"])
        .SwitchCausalInferencePipeline._sdedit_frame)
    check("(f2) la graine incorpore aussi la frame du switch (un scenario a 2 retours tire deux "
          "fois)", "int(current_start_frame)" in src)
    drv = open(os.path.join(HERE, "run_reid_scen.py")).read()
    # NB (2026-08-04) : un autre agent du projet a generalise la branche "ancre etrangere" en
    # S.WRONG_SRC_ARMS. On teste donc le SENS (les bras sdedit_wrong* sont bien dans la liste qui
    # declenche MEM_DECODE_SRC) et non le nom litteral, sinon ce selftest casse a chaque refonte.
    src_arms = set(getattr(S, "WRONG_SRC_ARMS", ()) or ()) | set(getattr(S, "SDEDIT_WRONG_ARMS", ()))
    check("(f3) le driver injecte MEM_SDEDIT_SEED par cellule, et les bras sdedit_wrong* sont "
          "dans la liste qui declenche MEM_DECODE_SRC",
          "MEM_SDEDIT_SEED" in drv and "MEM_DECODE_SRC" in drv
          and set(S.SDEDIT_WRONG_ARMS) <= src_arms,
          str(sorted(set(S.SDEDIT_WRONG_ARMS) - src_arms)))
    check("(f4) sdedit_wrong lit le MEME dump que wronganchor (permutation sans point fixe)",
          all(S.wrong_anchor_triple(t) != t for t in (12, 13, 14, 2, 4, 11)))


# ---------------------------------------------------------------------------------------------
def test_g_gpu(anchor_pt, sigmas=(0.4, 0.6)):
    """Mecanique reelle : ancre decodee vraie -> affutage -> tripwires."""
    from omegaconf import OmegaConf
    from run_reid_probe import build_pipeline, MEM2_DEC_DEFAULT
    import scenarios as S

    torch.set_grad_enabled(False)
    device = torch.device("cuda")
    cfg = OmegaConf.load(os.path.join(REPO, "reference/runs/reid_probe_l1/probe_l1.yaml"))
    cfg.distributed = False
    pipe = build_pipeline(cfg, device)
    pipe._initialize_kv_cache(1, dtype=torch.bfloat16, device=device,
                              kv_cache_size_override=12 * pipe.frame_seq_length)
    pipe._initialize_crossattn_cache(batch_size=1, dtype=torch.bfloat16, device=device)
    from wan.modules.video_memory import TTTBlockCache
    pipe.ttt_cache = [TTTBlockCache() for _ in range(pipe.num_transformer_blocks)]
    for c in pipe.kv_cache1:                       # un cache reel NON trivial, pour le tripwire
        c["k"].normal_()
        c["v"].normal_()
    kv_before = [(c["k"].clone(), c["v"].clone()) for c in pipe.kv_cache1]

    prompts, sched, _ = S.schedule_for("S2", 2)
    cond = pipe.text_encoder(text_prompts=[prompts[sched[S.revisit_segments("S2")[0]][1]]])
    a = torch.load(anchor_pt, map_location="cpu", weights_only=False)
    a = a["anchor"] if isinstance(a, dict) and "anchor" in a else a
    a = a.to(torch.bfloat16)
    print(f"[selftest g] ancre {tuple(a.shape)} {a.dtype} <- {os.path.basename(anchor_pt)}")

    os.environ["MEM_DECODE"] = MEM2_DEC_DEFAULT
    os.environ["MEM_SDEDIT_SEED"] = "5150241"
    torch.manual_seed(1234)
    rng_ref = torch.randn(4).tolist()
    outs = {}
    for s in sigmas:
        os.environ["MEM_SDEDIT"] = str(s)
        torch.manual_seed(1234)
        outs[s] = pipe._maybe_sdedit(a.clone(), cond, 180)
    torch.manual_seed(1234)
    check("(g1) l'affutage ne consomme AUCUN RNG global (l'appariement memzero est preserve)",
          torch.randn(4).tolist() == rng_ref)
    for s, o in outs.items():
        check(f"(g2 s={s}) shape / dtype / device preserves",
              tuple(o.shape) == tuple(a.shape) and o.dtype == a.dtype and o.device == a.device)
        check(f"(g3 s={s}) la sortie est finie et DIFFERE de l'ancre brute",
              bool(torch.isfinite(o.float()).all()) and float((o.float() - a.float()).abs().max()) > 0)
        hf0 = _Dummy._lat_hf(a)
        hf1 = _Dummy._lat_hf(o)
        cos = float(torch.nn.functional.cosine_similarity(
            o.float().flatten()[None], a.float().flatten()[None]).item())
        print(f"   s={s}: cos(affutee, brute) {cos:.4f} | |laplacien| latent {hf0:.4f} -> "
              f"{hf1:.4f} (x{hf1 / max(hf0, 1e-9):.2f})")
    check("(g4) le cache KV REEL est bit a bit inchange par l'affutage",
          all(bool((c["k"] == b[0]).all()) and bool((c["v"] == b[1]).all())
              for c, b in zip(pipe.kv_cache1, kv_before)))
    check("(g5) aucun etat memoire n'a ete ecrit par l'affutage",
          all(c.state is None for c in pipe.ttt_cache))
    check("(g6) le cache de cross-attention est remis a zero apres le passage fantome",
          all(not blk["is_init"] and float(blk["k"].abs().max()) == 0.0
              for blk in pipe.crossattn_cache))
    _clean_env()
    del pipe
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    return outs


# ---------------------------------------------------------------------------------------------
def test_h_audit_logs(patterns):
    logs = sorted(set(p for pat in patterns for p in glob.glob(pat)))
    check("(h0) des logs de rollouts reels a auditer", bool(logs), f"{len(logs)} fichiers")
    per = {}
    for p in logs:
        txt = open(p, errors="ignore").read()
        for m in re.finditer(r"\[scen\] (\S+) t(\d+) s(\d+) arm=(\S+)", txt):
            key = (m.group(1), int(m.group(2)), int(m.group(3)), m.group(4))
            rest = txt[m.end():]
            nxt = re.search(r"\[scen\] \S+ t\d+ s\d+ arm=", rest)
            body = rest[:nxt.start()] if nxt else rest
            # un rollout TUE par un OOM de co-run n'a rien a auditer : il n'a pas fini. On ne
            # compte que les rollouts qui ont produit leur mp4 (le "saved" du driver), sinon la
            # saturation de la machine ferait echouer un selftest de CODE.
            if "saved /shared" not in body:
                continue
            per[key] = {
                "sdedit": len(re.findall(r"\[MEM_SDEDIT\] anchor SHARPENED", body)),
                "purge": len(re.findall(r"\[MEM_SINKPURGE\] the \d+ pinned sink", body)),
                "splice": len(re.findall(r"\[MEM_DECODE\] decoded anchor spliced", body)),
                "frames": re.findall(r"at frame (\d+) \(mode", body),
            }
    bad = []
    for (sc, t, s, arm), r in sorted(per.items()):
        want_sd = 1 if arm.startswith("sdedit_") else 0
        want_sp = 1 if arm == "sinkpurge" else 0
        want_spl = 1 if (arm.startswith("sdedit_") or arm in ("memdec20", "wronganchor",
                                                              "noiseanchor")) else 0
        if (r["sdedit"], r["purge"], r["splice"]) != (want_sd, want_sp, want_spl):
            bad.append((sc, t, s, arm, r))
    check("(h1) exactement 1 affutage par cellule sdedit_*, 1 purge par cellule sinkpurge, "
          "0 splice chez sinkpurge, 1 splice chez les bras d'ancre", not bad, str(bad[:4]))
    fr = {f for r in per.values() for f in r["frames"]}
    check("(h2) toutes les purges tombent au retour (frame 180)", fr <= {"180"}, str(sorted(fr)))
    print(f"   {len(per)} rollouts audites")


# ---------------------------------------------------------------------------------------------
def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", action="store_true")
    ap.add_argument("--anchor", default="/shared/ssd_14T_project/adenelmi/reid_scen/dec_anchors/"
                                        "S2_t2_s0_decanchor.f180.pt")
    ap.add_argument("--audit-logs", nargs="*", default=None)
    args = ap.parse_args(argv)

    test_a_compile()
    test_b_arm_table()
    test_c_inertia()
    test_d_sdedit_arithmetic()
    test_e_structure()
    test_f_seeds()
    if args.gpu:
        test_g_gpu(args.anchor)
    if args.audit_logs:
        test_h_audit_logs(args.audit_logs)

    print("\n" + ("=" * 70))
    if FAILS:
        print(f"ECHECS ({len(FAILS)}): " + ", ".join(FAILS))
        sys.exit(1)
    print("TOUT PASSE")


if __name__ == "__main__":
    main()
