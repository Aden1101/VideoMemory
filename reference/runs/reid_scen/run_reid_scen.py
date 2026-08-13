#!/usr/bin/env python
# ============================================================================================
# SCENARIO-MATRIX re-ID revisit probe (N-switch). Same machinery as
# reference/runs/reid_probe_l1/run_reid_probe.py (imported, never modified) but driven by
# reference/runs/reid_scen/scenarios.py: several revisit SCHEDULES, prompt TRIPLES, and the
# re-ID scored against the segment of the subject's FIRST APPEARANCE (not the opening).
#
# ADDITIVE: the legacy campaign keeps running through run_reid_probe.py + pipeline.inference();
# this driver calls the new pipeline.inference_schedule(), proven bitwise identical on the
# legacy schedule by selftest_scenarios.py (a).
#
#   python run_reid_scen.py --mode selftest                       # tiny end-to-end smoke
#   python run_reid_scen.py --mode run   --scen S2 --triples 0,1 --seeds 0 --arms memzero,framebank
#   python run_reid_scen.py --mode score --scen S2 --triples 0,1 --seeds 0 --arms ...
#   python run_reid_scen.py --mode summary --scen S2,S3,S4
# ============================================================================================
import argparse
import gc
import json
import os
import sys

REPO = "/shared/ssd_14T/home/adenelmi/Anchor-Forcing"
HERE = os.path.dirname(os.path.abspath(__file__))
PROBE_DIR = os.path.join(REPO, "reference/runs/reid_probe_l1")
sys.path.insert(0, REPO)
sys.path.insert(0, PROBE_DIR)
sys.path.insert(0, HERE)

import torch                                                             # noqa: E402
from omegaconf import OmegaConf                                          # noqa: E402
from einops import rearrange                                             # noqa: E402
from torchvision.io import write_video, read_video                       # noqa: E402

from run_reid_probe import (build_pipeline, release_vram, cell_seed, dino_embed,   # noqa: E402
                            sign_test_p, sample_seg, center_crop, seg_cos,
                            MEM2_WRITER_DEFAULT, MEM2_DEC_DEFAULT)
import scenarios as S                                                    # noqa: E402

# home is 100 % full: the mp4s (25-35 MB each) live on the project disk, only the JSONs stay here
DEFAULT_VID_DIR = "/shared/ssd_14T_project/adenelmi/reid_scen/videos"
RES_DIR = os.path.join(HERE, "results")


# ----------------------------- generation ---------------------------------------------------
def run_arm(pipeline, config, scen, triple_idx, seed_idx, arm, device, out_mp4, fb_log=None):
    """One N-switch generation. Returns (latents_cpu, blocks_with_written_ttt_state)."""
    from utils.misc import set_seed
    prompts, schedule, subjects = S.schedule_for(scen, triple_idx)
    # LEVIER 1 (legende, 2026-08-04): les scenarios S2c / SS2c font pointer le RETOUR sur
    # l'entree 3 de la liste, dont le texte est construit ICI = prompt de A VERBATIM + le suffixe
    # de legende MESURE hors ligne. Suffixe vide (`caption_none`) => prompts[3] == prompts[0]
    # => rollout bit a bit identique au memzero deja sur disque (le selftest gratuit du levier).
    if arm in S.CAPTION_ARMS:
        assert scen in S.CAPTION_SCENARIOS, \
            f"le bras {arm} n'a de sens que sur un scenario de legende {list(S.CAPTION_SCENARIOS)}"
        _suf = S.caption_suffix(scen, triple_idx, seed_idx, arm)
        prompts[3] = S.caption_return_prompt(triple_idx, _suf)
        print(f"[caption][{arm}] retour = '{prompts[3]}'", flush=True)
    else:
        assert scen not in S.CAPTION_SCENARIOS, \
            f"le scenario {scen} est reserve aux bras de legende (le retour y pointe prompts[3])"
    frames = S.SCENARIOS[scen]["frames"]
    for k, v in S.SCEN_ARM_ENV[arm].items():
        os.environ[k] = v
    # T1 (lot offensif): trace de la masse d'attention par source, une par cellule et par bras.
    # APRES la table des bras (qui re-met TOUTES les cles de l'union, MEM_ADD_MASS_DUMP compris).
    if arm in S.ADDPOS_ARMS:
        _md = os.path.join(getattr(config, "_res_dir", "") or RES_DIR, "add_mass",
                           f"{scen}_t{triple_idx}_s{seed_idx}_{arm}.json")
        os.makedirs(os.path.dirname(_md), exist_ok=True)
        os.environ["MEM_ADD_MASS_DUMP"] = _md
    # PREFIX rule (2026-08-02): every memdec* variant (memdec20, memdec20s*, memdec21c*,
    # memdec21u, ...) needs the decoder, and every one of them plus memon20 needs the writer.
    # A prefix test so a new variant can never silently run without its checkpoints.
    # MEMOIRE 2.1 : les bras T0 chargent LEURS PROPRES poids (ecrivain + decodeur re-entraines
    # sur la metrique T0). Ils ne partagent pas les ckpts A2 de memdec20 -- sans quoi les deux
    # bras d'une meme campagne seraient le MEME modele et le contraste serait vide par
    # construction. Un ckpt T0 manquant est une ERREUR, jamais un repli silencieux sur A2.
    _is_t0 = arm in S.T0_ARMS
    if arm.startswith("memdec") or arm in S.MEM2_CKPT_ARMS:
        _dk = (getattr(config, "_mem2_t0_dec_ckpt", "") if _is_t0
               else getattr(config, "_mem2_dec_ckpt", "")) or ""
        assert _dk and os.path.exists(_dk), (
            f"arm {arm} requires --mem2-{'t0-' if _is_t0 else ''}dec-ckpt (got '{_dk}')")
        os.environ["MEM_DECODE"] = _dk
    if arm.startswith("memdec") or arm in S.MEM2_CKPT_ARMS or arm == "memon20":
        _wk = (getattr(config, "_mem2_t0_writer_ckpt", "") if _is_t0
               else getattr(config, "_mem2_writer_ckpt", "")) or ""
        assert _wk and os.path.exists(_wk), (
            f"arm {arm} requires --mem2-{'t0-' if _is_t0 else ''}writer-ckpt (got '{_wk}')")
        os.environ["MEM_WRITER"] = _wk
    # ---- REVIEWER CONTROLS (2026-08-03) -----------------------------------------------------
    # noiseanchor PERSISTS the anchor it decodes (and then throws away): that dump is what the
    # wronganchor arm of the PREVIOUS triple splices, so the two waves cost 48 rollouts and not
    # 72. The dump is written BEFORE any content substitution (pipeline hook order), so it is the
    # raw memdec20 anchor of this cell -- the rollout is identical to memdec20 up to the splice.
    _dec_dir = getattr(config, "_dec_anchor_dir", "") or S.DEC_ANCHOR_DIR
    # --dump-decanchor: memdec20 persiste AUSSI l'ancre qu'il decode (effet de bord pur, aucun
    # tirage RNG, le rollout est inchange) -- utile quand la campagne ne passe pas par
    # noiseanchor et que wronganchor a quand meme besoin d'une source.
    # LEVIER 4 : `memdec_snap` persiste TOUJOURS l'ancre qu'il decode -- c'est elle le pre-ecran
    # (metrique T0 : correlation bande-basse contre A*, B et X). Effet de bord pur, aucun RNG.
    if arm in S.SNAP_ARMS:
        os.makedirs(_dec_dir, exist_ok=True)
        os.environ["MEM_DECODE_DUMP"] = os.path.join(
            _dec_dir, f"{scen}_t{triple_idx}_s{seed_idx}_snapanchor")
    # MEMOIRE 2.1 : `memdec_t0` persiste TOUJOURS l'ancre qu'il decode -- c'est elle le gate G2b
    # (metrique T0 contre A*, B, X). Nom PROPRE : ecrire dans `_decanchor` ecraserait la source
    # que lit `wronganchor`. Effet de bord pur, aucun tirage RNG, rollout inchange.
    if arm in S.T0_ARMS:
        os.makedirs(_dec_dir, exist_ok=True)
        os.environ["MEM_DECODE_DUMP"] = os.path.join(
            _dec_dir, f"{scen}_t{triple_idx}_s{seed_idx}_{arm}")
    if arm == "memdec20" and getattr(config, "_dump_decanchor", False):
        os.makedirs(_dec_dir, exist_ok=True)
        os.environ["MEM_DECODE_DUMP"] = S.dec_anchor_base(scen, triple_idx, seed_idx, _dec_dir)
    if arm.startswith("noiseanchor"):
        os.makedirs(_dec_dir, exist_ok=True)
        os.environ["MEM_DECODE_DUMP"] = S.dec_anchor_base(scen, triple_idx, seed_idx, _dec_dir)
        os.environ["MEM_DECODE_NOISE_SEED"] = str(cell_seed(config, triple_idx, seed_idx))
    if arm.startswith("wronganchor") or arm in S.WRONG_SRC_ARMS:
        _src = S.wrong_anchor_src(scen, triple_idx, seed_idx, _dec_dir)
        assert os.path.exists(_src), (
            f"arm {arm} needs the DECODED anchor of triple "
            f"{S.wrong_anchor_triple(triple_idx)} seed {seed_idx} ({_src}) -- run the "
            f"noiseanchor wave (or memdec20 --dump-decanchor) on that cell first")
        os.environ["MEM_DECODE_SRC"] = _src
        print(f"[scen][{arm}] {scen} t{triple_idx} s{seed_idx} <- decoded anchor of triple "
              f"{S.wrong_anchor_triple(triple_idx)} (same seed)", flush=True)
    # mini-batterie training-free: graine par CELLULE du bruit SDEdit (les 6 cellules appariees ne
    # doivent pas partager une seule realisation), + la frame du switch (ajoutee dans le hook).
    if arm in S.SDEDIT_ARMS:
        os.environ["MEM_SDEDIT_SEED"] = str(5150241 + cell_seed(config, triple_idx, seed_idx))
    # composite arms: optional dump of {bg, anchor, composite, mask} for the eye board. Set here
    # (not from the caller's environment) because the arm table above re-sets EVERY key.
    if arm in S.COMPOSITE_BASE and getattr(config, "_comp_dump_dir", ""):
        os.makedirs(config._comp_dump_dir, exist_ok=True)
        os.environ["MEM_COMP_DUMP"] = os.path.join(
            config._comp_dump_dir, f"{scen}_t{triple_idx}_s{seed_idx}_{arm}")
    # routing threshold: per-scenario default (0.90 verbatim, 0.68 paraphrase), --fb-tau wins
    os.environ["FRAMEBANK_TAU"] = str(S.fb_tau(scen, getattr(config, "_fb_tau", None)))
    if getattr(config, "_fb_tie", None):
        os.environ["FRAMEBANK_TIE"] = str(config._fb_tie)

    seed = cell_seed(config, triple_idx, seed_idx)
    set_seed(seed)
    noise = torch.randn([1, frames, 16, 60, 104], device=device, dtype=torch.bfloat16)
    print(f"[scen] {scen} t{triple_idx} s{seed_idx} arm={arm} seed={seed} frames={frames}\n"
          f"       schedule={schedule} subjects={subjects}\n"
          f"       A='{prompts[0]}'\n       B='{prompts[1]}'\n       C='{prompts[2]}'\n"
          f"       A'='{prompts[3]}'", flush=True)
    video, latents = pipeline.inference_schedule(
        noise=noise, prompts=prompts, schedule=schedule, subjects=subjects,
        return_latents=True, low_memory=True)
    written = [bi for bi in range(pipeline.num_transformer_blocks)
               if pipeline.ttt_cache[bi].state is not None]
    print(f"[scen][{arm}] ttt_cache.state written on blocks: {written}", flush=True)

    # FRAMEBANK routing log (error analysis of the selection)
    events = list(getattr(pipeline, "_fb_events", []) or [])
    if fb_log and events:
        os.makedirs(os.path.dirname(fb_log), exist_ok=True)
        with open(fb_log, "w") as f:
            json.dump({"scen": scen, "triple": triple_idx, "seed_idx": seed_idx, "arm": arm,
                       "prompts": prompts, "schedule": schedule, "events": events}, f, indent=2)
        print(f"[scen][{arm}] framebank routing log -> {fb_log}", flush=True)

    vid = (rearrange(video[0].cpu().float(), "t c h w -> t h w c") * 255.0)
    os.makedirs(os.path.dirname(out_mp4), exist_ok=True)
    write_video(out_mp4, vid.clamp(0, 255).to(torch.uint8), fps=16)
    print(f"[scen][{arm}] saved {out_mp4} ({vid.shape[0]} pixel frames)", flush=True)

    lat = latents.float().cpu()
    pipeline.vae.model.clear_cache()
    pipeline.kv_cache1 = None
    pipeline.crossattn_cache = None
    pipeline.ttt_cache = None
    gc.collect()
    torch.cuda.empty_cache()
    return lat, written


# ----------------------------- scoring ------------------------------------------------------
def score_video_scen(path, scen, device, settle=20, resettle=12, stride=8):
    """re-ID of every REVISIT segment against the segment of the subject's FIRST APPEARANCE.

    reid_resA_origA = mean over the revisit segments (the KEY number, same name as the legacy
    campaign so the two are directly comparable); reid_by_revisit keeps the per-return values
    (S4 has two). Controls: the interlude that immediately precedes each revisit vs the first
    appearance (ctrl_B_origA) and the revisit vs that interlude (ctrl_resA_B)."""
    frames = S.SCENARIOS[scen]["frames"]
    v, _, _ = read_video(path, pts_unit="sec", output_format="THWC")
    vid = v.permute(0, 3, 1, 2).float() / 255.0
    wins = S.score_segments(scen, frames, settle=settle, resettle=resettle)
    fa = S.first_appearance(scen)
    roles = S.seg_roles(scen)
    revisits = S.revisit_segments(scen)
    assert revisits, f"{scen} has no revisit segment"
    out = {"scen": scen, "segments_latent": {str(k): list(v) for k, v in wins.items()},
           "roles": roles, "revisits": revisits,
           "first_appearance": {str(k): v for k, v in fa.items()}}
    for variant, xform in (("full", lambda f: f), ("crop60", center_crop)):
        E = {i: dino_embed(xform(sample_seg(vid, *wins[i], stride)), device) for i in wins}
        by_rev, ctrl_prev, ctrl_rb = {}, {}, {}
        for r in revisits:
            src = fa[roles[r]]
            by_rev[str(r)] = seg_cos(E[r], E[src])
            ctrl_rb[str(r)] = seg_cos(E[r], E[r - 1])
            ctrl_prev[str(r)] = seg_cos(E[r - 1], E[src])
        half = max(1, E[fa[roles[revisits[0]]]].shape[0] // 2)
        Efa = E[fa[roles[revisits[0]]]]
        out[variant] = {
            "reid_resA_origA": sum(by_rev.values()) / len(by_rev),
            "reid_by_revisit": by_rev,
            "ctrl_resA_B": sum(ctrl_rb.values()) / len(ctrl_rb),
            "ctrl_B_origA": sum(ctrl_prev.values()) / len(ctrl_prev),
            "self_origA": seg_cos(Efa[:half], Efa[half:]),
            "n_frames": {str(i): int(E[i].shape[0]) for i in E},
        }
    return out


# ----------------------------- summary ------------------------------------------------------
def summarize(res_dir, scens, triple_ids, seed_ids, out_json=None):
    summary = {"scenarios": {}}
    for scen in scens:
        cells = {}
        for ti in triple_ids:
            for si in seed_ids:
                p = os.path.join(res_dir, f"{scen}_t{ti}_s{si}_scores.json")
                if os.path.exists(p):
                    with open(p) as f:
                        cells[(ti, si)] = json.load(f)
        if not cells:
            print(f"[summary] {scen}: no result JSONs yet")
            continue
        arms = [a for a in S.SCEN_ARM_ENV if any(a in c for c in cells.values())]
        revisits = S.revisit_segments(scen)
        print(f"\n===== {scen} ({S.SCENARIOS[scen]['desc']}) -- {len(cells)} cells, "
              f"arms: {','.join(arms)} =====")
        ssum = {"n_cells": len(cells), "revisit_segments": revisits, "variants": {}}
        for variant in ("full", "crop60"):
            vsum = {"arm_means": {}, "paired_vs_memzero": {}, "per_revisit_means": {}}
            line = []
            for a in arms:
                vals = [c[a][variant]["reid_resA_origA"] for c in cells.values() if a in c]
                if vals:
                    vsum["arm_means"][a] = sum(vals) / len(vals)
                    line.append(f"{a} {sum(vals)/len(vals):.4f} (n={len(vals)})")
                    vsum["per_revisit_means"][a] = {
                        str(r): sum(c[a][variant]["reid_by_revisit"][str(r)]
                                    for c in cells.values() if a in c)
                        / max(1, len([c for c in cells.values() if a in c]))
                        for r in revisits}
            print(f"\n[{variant}] mean reid(revisit vs FIRST appearance): " + "  ".join(line))
            if len(revisits) > 1:
                for a in arms:
                    if a in vsum["per_revisit_means"]:
                        print("    per-revisit " + a + ": " + "  ".join(
                            f"seg{r} {vsum['per_revisit_means'][a][str(r)]:.4f}" for r in revisits))
            for a in arms:
                if a == "memzero":
                    continue
                deltas, per_t, cd = [], {}, {}
                for (ti, si), c in sorted(cells.items()):
                    if a in c and "memzero" in c:
                        d = c[a][variant]["reid_resA_origA"] - c["memzero"][variant]["reid_resA_origA"]
                        deltas.append(d)
                        per_t.setdefault(ti, []).append(d)
                        cd[f"{ti}_{si}"] = d
                if not deltas:
                    continue
                m = sum(deltas) / len(deltas)
                sd = (sum((x - m) ** 2 for x in deltas) / max(1, len(deltas) - 1)) ** 0.5
                npos = sum(1 for x in deltas if x > 0)
                nneg = sum(1 for x in deltas if x < 0)
                p = sign_test_p(npos, nneg)
                print(f"  delta({a} - memzero), {len(deltas)} paired cells: mean {m:+.4f}  "
                      f"sd {sd:.4f}  sem {sd / max(1, len(deltas)) ** 0.5:.4f}  "
                      f"signs {npos}+/{nneg}-  sign-p {p:.4f}")
                vsum["paired_vs_memzero"][a] = {"mean_delta": m, "sd": sd, "n_pos": npos,
                                                "n_neg": nneg, "sign_p": p, "deltas": cd,
                                                "per_triple_mean": {str(t): sum(v) / len(v)
                                                                    for t, v in per_t.items()}}
            ssum["variants"][variant] = vsum
        # framebank selection accuracy (error analysis)
        fb = {}
        for arm in ("framebank", "framebank_oracle"):
            recs = []
            for ti in triple_ids:
                for si in seed_ids:
                    lp = os.path.join(res_dir, "fb_logs", f"{scen}_t{ti}_s{si}_{arm}.json")
                    if os.path.exists(lp):
                        with open(lp) as f:
                            recs += json.load(f)["events"]
            if recs:
                at_rev = [e for e in recs if e["seg"] in revisits]
                fb[arm] = {
                    "n_routing_events": len(recs),
                    "n_at_revisit": len(at_rev),
                    "recall_rate_at_revisit": sum(e["recalled"] for e in at_rev) / max(1, len(at_rev)),
                    "correct_rate_at_revisit": sum(e["correct"] for e in at_rev) / max(1, len(at_rev)),
                    "false_recall_off_revisit": sum(e["recalled"] for e in recs
                                                    if e["seg"] not in revisits),
                    "n_off_revisit": len(recs) - len(at_rev),
                }
                print(f"  [{arm}] routing: {fb[arm]['n_at_revisit']} revisit events, recall "
                      f"{fb[arm]['recall_rate_at_revisit']:.2%}, CORRECT subject "
                      f"{fb[arm]['correct_rate_at_revisit']:.2%}; false recalls on "
                      f"{fb[arm]['n_off_revisit']} non-revisit switches: "
                      f"{fb[arm]['false_recall_off_revisit']}")
        ssum["framebank_routing"] = fb
        summary["scenarios"][scen] = ssum
    out = out_json or os.path.join(res_dir, "summary_scen.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[summary] wrote {out}")


# ----------------------------- main ---------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["run", "score", "summary", "selftest"], default="run")
    ap.add_argument("--scen", type=str, default="S2", help="comma list, e.g. S2,S3,S4")
    ap.add_argument("--triples", type=str, default="0")
    ap.add_argument("--seeds", type=str, default="0,1,2")
    ap.add_argument("--arms", type=str, default="",
                    help="comma list; empty = scenarios.DEFAULT_ARMS of each scenario")
    ap.add_argument("--config", type=str, default=os.path.join(PROBE_DIR, "probe_l1.yaml"))
    ap.add_argument("--stock-config", type=str, default=os.path.join(PROBE_DIR, "stock_probe.yaml"))
    ap.add_argument("--stockoff-config", type=str,
                    default=os.path.join(PROBE_DIR, "stock_official.yaml"))
    ap.add_argument("--mem2-writer-ckpt", type=str, default=MEM2_WRITER_DEFAULT)
    ap.add_argument("--mem2-dec-ckpt", type=str, default=MEM2_DEC_DEFAULT)
    # MEMOIRE 2.1 : les poids du bras `memdec_t0` (re-entraines sur la metrique T0). Vides par
    # defaut : le bras leve une assertion explicite plutot que de retomber en silence sur A2.
    ap.add_argument("--mem2-t0-writer-ckpt", type=str, default="")
    ap.add_argument("--mem2-t0-dec-ckpt", type=str, default="")
    ap.add_argument("--vid-dir", type=str, default=DEFAULT_VID_DIR)
    ap.add_argument("--res-dir", type=str, default=RES_DIR)
    ap.add_argument("--fb-tau", type=float, default=None,
                    help="routing threshold; default = per-scenario (0.90 verbatim, 0.68 'p')")
    ap.add_argument("--fb-tie", type=str, default="recent", choices=["recent", "first"])
    ap.add_argument("--dump-frames", type=str, default="",
                    help="smoke: directory where 4 PNG frames per segment are dumped")
    ap.add_argument("--dec-anchor-dir", type=str, default=S.DEC_ANCHOR_DIR,
                    help="where the DECODED anchor of a cell is persisted (arm noiseanchor) and "
                         "read back from another cell (arm wronganchor)")
    ap.add_argument("--dump-decanchor", action="store_true",
                    help="memdec20 persiste aussi l'ancre decodee (source du bras wronganchor)")
    ap.add_argument("--comp-dump-dir", type=str, default="",
                    help="composite arms: directory where {bg, anchor, composite, mask} latents "
                         "are dumped (eye board of the mechanism)")
    args = ap.parse_args()

    device = torch.device("cuda")
    torch.set_grad_enabled(False)
    config = OmegaConf.load(args.config)
    config.distributed = False
    config._mem2_writer_ckpt = args.mem2_writer_ckpt
    config._mem2_dec_ckpt = args.mem2_dec_ckpt
    config._mem2_t0_writer_ckpt = args.mem2_t0_writer_ckpt
    config._mem2_t0_dec_ckpt = args.mem2_t0_dec_ckpt
    config._fb_tau = args.fb_tau
    config._fb_tie = args.fb_tie
    config._comp_dump_dir = args.comp_dump_dir
    config._dec_anchor_dir = args.dec_anchor_dir
    config._dump_decanchor = args.dump_decanchor
    config._res_dir = args.res_dir

    scens = [s for s in args.scen.split(",") if s]
    for s in scens:
        assert s in S.SCENARIOS, f"unknown scenario {s} (have {list(S.SCENARIOS)})"
    triple_ids = [int(x) for x in args.triples.split(",") if x != ""]
    seed_ids = [int(x) for x in args.seeds.split(",") if x != ""]
    os.makedirs(args.res_dir, exist_ok=True)
    os.makedirs(args.vid_dir, exist_ok=True)

    def mp4_path(scen, ti, si, arm):
        return os.path.join(args.vid_dir, f"{scen}_t{ti}_s{si}_{arm}.mp4")

    def fb_path(scen, ti, si, arm):
        return os.path.join(args.res_dir, "fb_logs", f"{scen}_t{ti}_s{si}_{arm}.json")

    def arms_of(scen):
        return [a for a in (args.arms.split(",") if args.arms else S.DEFAULT_ARMS[scen]) if a]

    if args.mode == "summary":
        summarize(args.res_dir, scens, triple_ids or range(len(S.TRIPLES)), seed_ids)
        return
    if args.mode == "selftest":
        from selftest_scenarios import main as st_main
        st_main([])
        return

    if args.mode == "run":
        stock_cfgs = {}
        mag_todo = [(sc, ti, si, a) for sc in scens for ti in triple_ids for si in seed_ids
                    for a in arms_of(sc)
                    if a in S.MAG_SCEN_ARMS and not os.path.exists(mp4_path(sc, ti, si, a))]
        if mag_todo:
            pipeline = build_pipeline(config, device)
            for sc, ti, si, a in mag_todo:
                run_arm(pipeline, config, sc, ti, si, a, device, mp4_path(sc, ti, si, a),
                        fb_log=fb_path(sc, ti, si, a))
            del pipeline
            release_vram()
        for sarm, cfg_path in (("stock", args.stock_config), ("stockoff", args.stockoff_config)):
            todo = [(sc, ti, si) for sc in scens for ti in triple_ids for si in seed_ids
                    if sarm in arms_of(sc) and not os.path.exists(mp4_path(sc, ti, si, sarm))]
            if not todo:
                continue
            cfg = OmegaConf.load(cfg_path)
            cfg.distributed = False
            cfg._mem2_writer_ckpt = args.mem2_writer_ckpt
            cfg._mem2_dec_ckpt = args.mem2_dec_ckpt
            cfg._mem2_t0_writer_ckpt = args.mem2_t0_writer_ckpt
            cfg._mem2_t0_dec_ckpt = args.mem2_t0_dec_ckpt
            cfg._fb_tau, cfg._fb_tie = args.fb_tau, args.fb_tie
            cfg._comp_dump_dir = args.comp_dump_dir
            stock = build_pipeline(cfg, device)
            for sc, ti, si in todo:
                run_arm(stock, cfg, sc, ti, si, sarm, device, mp4_path(sc, ti, si, sarm))
            del stock
            release_vram()

    # ---- score (merge into the existing cell JSON, never clobber other arms) ----------------
    for sc in scens:
        for ti in triple_ids:
            for si in seed_ids:
                out_json = os.path.join(args.res_dir, f"{sc}_t{ti}_s{si}_scores.json")
                cell = {}
                if os.path.exists(out_json):
                    with open(out_json) as f:
                        cell = json.load(f)
                cell.update({"scen": sc, "triple": list(S.TRIPLES[ti]),
                             "paraphrase": S.PARAPHRASES[ti],
                             "seed": cell_seed(config, ti, si),
                             "frames": S.SCENARIOS[sc]["frames"],
                             "schedule": S.SCENARIOS[sc]["schedule"],
                             "subjects": S.seg_roles(sc),
                             "checkpoint": str(config.lora_ckpt)})
                missing = [a for a in arms_of(sc) if not os.path.exists(mp4_path(sc, ti, si, a))]
                if missing:
                    print(f"[scen] {sc} t{ti} s{si}: missing videos for {missing}, skipping cell")
                    continue
                for a in arms_of(sc):
                    cell[a] = score_video_scen(mp4_path(sc, ti, si, a), sc, device)
                with open(out_json, "w") as f:
                    json.dump(cell, f, indent=2)
                print(f"[scen] wrote {out_json}", flush=True)
                if args.dump_frames:
                    dump_segment_frames(mp4_path(sc, ti, si, arms_of(sc)[0]), sc,
                                        args.dump_frames, f"{sc}_t{ti}_s{si}",
                                        arms=[(a, mp4_path(sc, ti, si, a)) for a in arms_of(sc)])

    summarize(args.res_dir, scens, triple_ids, seed_ids)


def dump_segment_frames(_unused, scen, out_dir, tag, arms, per_seg=4):
    """Smoke eye-check: a PNG per arm, rows = segments, columns = per_seg evenly spaced pixel
    frames of that segment -> the switches and the revisits are visible at a glance."""
    from run_reid_probe import latent_to_pix
    import torchvision.utils as vutils
    os.makedirs(out_dir, exist_ok=True)
    frames = S.SCENARIOS[scen]["frames"]
    bounds = S.seg_bounds(scen)
    for arm, path in arms:
        if not os.path.exists(path):
            continue
        v, _, _ = read_video(path, pts_unit="sec", output_format="THWC")
        vid = v.permute(0, 3, 1, 2).float() / 255.0
        rows = []
        for (lo, hi) in bounds:
            plo, phi = latent_to_pix(lo, min(hi, frames) - 1)
            phi = min(phi, vid.shape[0] - 1)
            idx = [int(plo + (phi - plo) * k / max(1, per_seg - 1)) for k in range(per_seg)]
            rows += [vid[i] for i in idx]
        grid = vutils.make_grid(torch.stack(rows), nrow=per_seg, padding=4)
        out = os.path.join(out_dir, f"{tag}_{arm}.png")
        vutils.save_image(grid, out)
        print(f"[eye] {out}  ({len(bounds)} segments x {per_seg} frames)", flush=True)


if __name__ == "__main__":
    main()
