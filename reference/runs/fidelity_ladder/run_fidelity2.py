#!/usr/bin/env python
# ============================================================================================
# ECHELLE DE FIDELITE, CAMPAGNE 2 (sujet NON-OUVRANT) -- driver de generation (GPU).
#
# Forme S2 verbatim : C(0-60) A(60-120) B(120-180) A(180-240), via inference_schedule (le
# chemin N-switch du harnais reid_scen ; les hooks ORACLE_SPLICE / ORACLE_ANCHOR_SRC /
# ORACLE_BLUR_SIGMA / ORACLE_ANCHOR_DUMP y sont natifs dans _switch_recache).
#
# PHASE 0 (une seule fois, verrou repertoire) : 10 rollouts de preparation C(0-60) A(60-120)
# en 120 frames, graine main+100000 -> frame 119 = l'ancre JUMEAU (meme prompt, autre graine,
# MEME regime d'apparition mi-rollout). Le donneur `autre_categorie` du scenario i = le
# jumeau du scenario (i+5) % 10.
#
# TEMOIN D'EVICTION (exigence du coordinateur : prouve, pas suppose) : sur le bras `rien`
# des scenarios temoins (le 1er de chaque moitie), _switch_recache est enveloppe pour LIRE
# les index du KV cache de chaque bloc a CHAQUE switch : plus vieille frame non-sink retenue
# = global_end - (local_end - sink_tokens). Au retour elle doit etre > 120 (fin du segment
# A) -> aucune frame de A dans la fenetre ; le sink = frame 0 (C). Ecrit dans
# check/<sid>_witness.json et ASSERTE.
#
#   python run_fidelity2.py --half A|B|all [--scenarios ...] [--arms ...]
# ============================================================================================
import argparse
import gc
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import fidelity_common as C1                                             # noqa: E402
import fidelity2_common as C                                             # noqa: E402

sys.path.insert(0, C.REPO)
sys.path.insert(0, C.PROBE_DIR)
os.chdir(C.REPO)

import torch                                                             # noqa: E402
from omegaconf import OmegaConf                                          # noqa: E402
from einops import rearrange                                             # noqa: E402
from torchvision.io import write_video                                   # noqa: E402
import torchvision.utils as vutils                                       # noqa: E402

from run_reid_probe import build_pipeline, _OFF                          # noqa: E402

for _k, _v in _OFF.items():
    assert _k in C.OFF and C.OFF[_k] == _v, \
        f"OFF derive de run_reid_probe._OFF sur {_k!r}"

FRAME_SEQLEN = 1560


def witness_sids():
    return getattr(C, "WITNESS_SIDS", ("scenario01", "scenario06"))


def draw_noise(seed, frames, device):
    """Tirage simple (set_seed puis randn) : la reproduction bit a bit des videos v2 n'existe
    pas dans ce regime (le sujet emerge en cours de rollout), donc pas de tirage en 2 temps.
    Meme graine par scenario que la campagne 1 (consigne coordinateur)."""
    from utils.misc import set_seed
    set_seed(seed)
    return torch.randn([1, frames, 16, 60, 104], device=device, dtype=torch.bfloat16)


def prompts_of(meta):
    return [meta["prompt"], meta["interlude"], meta["opener"]]


def decode_one(pipeline, lat_1f, device):
    px = pipeline.vae.decode_to_pixel(
        lat_1f.to(device=device, dtype=torch.bfloat16), use_cache=False)
    pipeline.vae.model.clear_cache()
    return (px[:, 0].float().cpu() * 0.5 + 0.5).clamp(0, 1)


def free_caches(pipeline):
    pipeline.vae.model.clear_cache()
    pipeline.kv_cache1 = None
    pipeline.crossattn_cache = None
    pipeline.ttt_cache = None
    gc.collect()
    torch.cuda.empty_cache()


# ---------------------------------- phase 0 : jumeaux ----------------------------------------
def gen_twin_refs(pipeline, manifest, device):
    """Genere les refs jumeau manquantes (verrou repertoire : UN processus fait tout, l'autre
    attend le manifeste). Idempotent : les .pt existants ne sont jamais regeneres."""
    os.makedirs(C.REFS_DIR, exist_ok=True)
    lock = os.path.join(C.REFS_DIR, "gen_lock")
    need = [sc["sid"] for sc in C.SCENARIOS if not os.path.exists(C.twin_pt(sc["sid"]))]
    if not need and os.path.exists(C.REFS_MANIFEST):
        return
    try:
        os.makedirs(lock)
        holder = True
    except FileExistsError:
        holder = False
    if not holder:
        print("[twins] un autre processus genere les refs jumeau, attente du manifeste...",
              flush=True)
        t0 = time.time()
        while not os.path.exists(C.REFS_MANIFEST):
            assert time.time() - t0 < 3 * 3600, \
                (f"refs jumeau toujours absentes apres 3 h -- verrou {lock} probablement "
                 f"orphelin (processus createur mort) : le supprimer et relancer")
            time.sleep(60)
        return
    try:
        for sc in C.SCENARIOS:
            sid = sc["sid"]
            if os.path.exists(C.twin_pt(sid)):
                continue
            meta = dict(sc, **{k: manifest[sid][k] for k in ("seed_main", "seed_twin")})
            for k, v in C.ARM_ENV["rien"].items():        # memoire muette, aucun hook
                os.environ[k] = v
            seed = int(meta["seed_twin"])
            noise = draw_noise(seed, C.TWIN_FRAMES, device)
            print(f"[twins] {sid} prep jumeau seed={seed} frames={C.TWIN_FRAMES} "
                  f"schedule={C.TWIN_SCHEDULE}", flush=True)
            t0 = time.time()
            _, lat = pipeline.inference_schedule(
                noise=noise, prompts=prompts_of(meta), schedule=C.TWIN_SCHEDULE,
                return_latents=True, low_memory=True)
            fr = lat[0, C.ANCHOR_FRAME][None, None].detach().float().cpu()
            tmp = C.twin_pt(sid) + ".tmp"
            torch.save({"anchor": fr.to(torch.float16),
                        "src": f"twin_prep:{sid}", "seed": seed,
                        "frame": C.ANCHOR_FRAME}, tmp)
            os.replace(tmp, C.twin_pt(sid))
            print(f"[twins] {sid} -> {C.twin_pt(sid)} ({time.time() - t0:.0f}s)", flush=True)
            del lat
            free_caches(pipeline)
        # `other_<sid>.pt` = lien materialise vers le jumeau du donneur (audit lisible).
        # Campagne supplement : le donneur est un jumeau DEJA GENERE d'une autre campagne
        # (C.DONOR_TWIN_SRC) ; sinon, le jumeau du scenario (i+5) % n de la meme campagne.
        def donor_of_sid(i, sid):
            if hasattr(C, "DONOR_TWIN_SRC"):
                return C.DONOR_TWIN_SRC[sid], C.DONOR_SIDS[sid]
            dn = C.SCENARIOS[C.donor_of(i)]["sid"]
            return C.twin_pt(dn), dn

        for i, sc in enumerate(C.SCENARIOS):
            src, dn = donor_of_sid(i, sc["sid"])
            assert os.path.exists(src), f"jumeau donneur manquant: {src}"
            if not os.path.exists(C.other_pt(sc["sid"])):
                d = torch.load(src, map_location="cpu", weights_only=False)
                d["donor_of"] = sc["sid"]
                d["donor_sid"] = dn
                tmp = C.other_pt(sc["sid"]) + ".tmp"
                torch.save(d, tmp)
                os.replace(tmp, C.other_pt(sc["sid"]))
        man2 = {}
        for i, sc in enumerate(C.SCENARIOS):
            sid = sc["sid"]
            man2[sid] = dict(manifest[sid])
            man2[sid].update({"opener": sc["opener"], "schedule": C.SCHEDULE,
                              "anchor_frame": C.ANCHOR_FRAME,
                              "twin_pt": C.twin_pt(sid), "other_pt": C.other_pt(sid),
                              "donor_sid": donor_of_sid(i, sid)[1]})
            man2[sid].pop("main81_pt", None)
        tmp = C.REFS_MANIFEST + ".tmp"
        with open(tmp, "w") as f:
            json.dump(man2, f, indent=2)
        os.replace(tmp, C.REFS_MANIFEST)
        print(f"[twins] manifeste -> {C.REFS_MANIFEST}", flush=True)
    finally:
        os.rmdir(lock)


# ---------------------------------- temoin d'eviction ----------------------------------------
def install_witness(pipeline, sid):
    """Enveloppe _switch_recache : a CHAQUE switch, lit les index du KV cache de chaque bloc
    AVANT le recache. Preuve runtime que le segment A est sorti du contexte au retour."""
    records = []
    orig = pipeline._switch_recache
    sink_tokens = C.SINK_FRAMES * FRAME_SEQLEN

    def wrapped(output, current_start_frame, cond_new, **kw):
        per_block = []
        for bi in range(pipeline.num_transformer_blocks):
            kc = pipeline.kv_cache1[bi]
            g = int(kc["global_end_index"].item())
            l = int(kc["local_end_index"].item())
            per_block.append({"block": bi, "global_end_frame": g / FRAME_SEQLEN,
                              "cache_frames": l / FRAME_SEQLEN,
                              "oldest_nonsink_frame": (g - (l - sink_tokens)) / FRAME_SEQLEN})
        records.append({"switch_frame": int(current_start_frame),
                        "is_revisit": bool(kw.get("is_revisit")),
                        "sink_frames": C.SINK_FRAMES,
                        "per_block": per_block})
        return orig(output, current_start_frame, cond_new, **kw)

    pipeline._switch_recache = wrapped
    return records, orig


def check_witness(sid, records):
    ret = [r for r in records if r["switch_frame"] == C.RETURN_FRAME]
    assert len(ret) == 1 and ret[0]["is_revisit"], \
        f"{sid}: pas de switch de retour a la frame {C.RETURN_FRAME} dans le temoin"
    oldest = [b["oldest_nonsink_frame"] for b in ret[0]["per_block"]]
    verdict = {
        "sid": sid, "return_frame": C.RETURN_FRAME,
        "a_segment": [C.A_START, C.A_END],
        "sink_frames": C.SINK_FRAMES, "sink_content": "frame 0 = ouvreur C (jamais reecrit)",
        "oldest_nonsink_frame_min": min(oldest), "oldest_nonsink_frame_max": max(oldest),
        "a_fully_evicted": bool(min(oldest) >= C.A_END),
        "gap_frames": min(oldest) - C.A_END,
        "n_blocks": len(oldest),
        "all_switches": records,
    }
    os.makedirs(C.CHECK_DIR, exist_ok=True)
    out = os.path.join(C.CHECK_DIR, f"{sid}_witness.json")
    with open(out, "w") as f:
        json.dump(verdict, f, indent=2)
    print(f"[witness] {sid}: au retour (frame {C.RETURN_FRAME}), plus vieille frame non-sink "
          f"en cache = {min(oldest):.0f} (sur {len(oldest)} blocs) ; segment A = "
          f"[{C.A_START}, {C.A_END}) ; A evince = {verdict['a_fully_evicted']} -> {out}",
          flush=True)
    assert verdict["a_fully_evicted"], \
        f"{sid}: le segment A n'est PAS entierement evince au retour (min oldest {min(oldest)})"
    return verdict


# ---------------------------------- bras -----------------------------------------------------
def arm_env(arm):
    """Table des 6 barreaux + les barreaux de flou FIN (flou_s1..s4, echelle 12/08)."""
    env = C.ARM_ENV.get(arm) or C.BLUR_ARM_ENV.get(arm)
    assert env is not None, f"bras inconnu {arm}"
    return env


def run_arm(pipeline, sid, arm, meta, device):
    out_mp4 = C.raw_mp4(sid, arm)
    if os.path.exists(out_mp4):
        print(f"[fidelity2] {sid} {arm}: deja fait", flush=True)
        return None
    for k, v in arm_env(arm).items():
        os.environ[k] = v
    os.environ["ORACLE_ANCHOR_DUMP"] = C.anchor_pt(sid)
    if arm == "meme_categorie":
        os.environ["ORACLE_ANCHOR_SRC"] = meta["twin_pt"]
    elif arm == "autre_categorie":
        os.environ["ORACLE_ANCHOR_SRC"] = meta["other_pt"]

    witness = None
    if arm == "rien" and sid in witness_sids():
        witness, orig = install_witness(pipeline, sid)

    seed = int(meta["seed_main"])
    noise = draw_noise(seed, C.FRAMES, device)
    print(f"[fidelity2] {sid} arm={arm} seed={seed} frames={C.FRAMES} schedule={C.SCHEDULE}\n"
          f"            A='{meta['prompt']}'\n            B='{meta['interlude']}'\n"
          f"            C='{meta['opener']}'", flush=True)
    t0 = time.time()
    try:
        video, latents = pipeline.inference_schedule(
            noise=noise, prompts=prompts_of(meta), schedule=C.SCHEDULE,
            return_latents=True, low_memory=True)
    finally:
        if witness is not None:
            pipeline._switch_recache = orig
    vid = (rearrange(video[0].cpu().float(), "t c h w -> t h w c") * 255.0)
    os.makedirs(os.path.dirname(out_mp4), exist_ok=True)
    write_video(out_mp4, vid.clamp(0, 255).to(torch.uint8), fps=16)
    print(f"[fidelity2] {sid} {arm}: {vid.shape[0]} frames pixel -> {out_mp4} "
          f"({time.time() - t0:.0f}s)", flush=True)
    if witness is not None:
        check_witness(sid, witness)

    lat = latents.float().cpu()
    del video, latents
    free_caches(pipeline)
    return lat


def refs_board(pipeline, sid, lat, meta, device):
    """Planche d'oeil : sujet du rollout (frame 119) vs jumeau insere vs donneur -- verifie
    que le sujet EMERGE bien au segment 2 et que le jumeau est un autre individu."""
    tiles = []
    tiles.append(decode_one(pipeline, lat[0, C.ANCHOR_FRAME][None, None].float(), device)[0])
    for p in (meta["twin_pt"], meta["other_pt"]):
        t = torch.load(p, map_location="cpu", weights_only=False)["anchor"].float()
        tiles.append(decode_one(pipeline, t, device)[0])
    # + la frame 59 (fin de l'ouvreur C) : le sujet ne doit PAS y etre
    tiles.insert(0, decode_one(pipeline, lat[0, C.A_START - 1][None, None].float(), device)[0])
    grid = vutils.make_grid(torch.stack(tiles), nrow=2, padding=6)
    os.makedirs(C.CHECK_DIR, exist_ok=True)
    out = os.path.join(C.CHECK_DIR, f"{sid}_refs_endC_oursf119_twin_other.jpg")
    vutils.save_image(grid, out)
    print(f"[fidelity2] planche refs -> {out}", flush=True)


def main():
    global C
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenarios", type=str, default="")
    ap.add_argument("--half", type=str, default="all", choices=["A", "B", "all"])
    ap.add_argument("--arms", type=str, default="")
    ap.add_argument("--config", type=str, default="")
    ap.add_argument("--campaign", type=str, default="nonopening",
                    choices=["nonopening", "supplement"])
    args = ap.parse_args()
    if args.campaign == "supplement":
        import fidelity3_common as C_sup
        C = C_sup
    args.arms = args.arms or ",".join(C.ARMS)
    args.config = args.config or C.CONFIG_YAML

    # graines et prompts de reference : campagne 1 (memes sujets) ou index v2 (supplement)
    manifest1 = C.base_manifest() if hasattr(C, "base_manifest") else C1.load_manifest()
    half = len(C.SCENARIOS) // 2
    if args.scenarios:
        sids = [s for s in args.scenarios.split(",") if s]
    elif args.half == "A":
        sids = [sc["sid"] for sc in C.SCENARIOS[:half]]
    elif args.half == "B":
        sids = [sc["sid"] for sc in C.SCENARIOS[half:]]
    else:
        sids = [sc["sid"] for sc in C.SCENARIOS]
    arms = sorted([a for a in args.arms.split(",") if a], key=lambda a: a != "rien")
    for a in arms:
        arm_env(a)          # leve si le bras est inconnu

    device = torch.device("cuda")
    torch.set_grad_enabled(False)
    config = OmegaConf.load(args.config)
    config.distributed = False
    pipeline = build_pipeline(config, device)
    print(f"[fidelity2] pipeline pret ; scenarios {sids} ; bras {arms}", flush=True)

    gen_twin_refs(pipeline, manifest1, device)
    manifest = C.load_manifest()
    for sid in sids:
        for p in ("twin_pt", "other_pt"):
            assert os.path.exists(manifest[sid][p]), f"{sid}: {manifest[sid][p]} manquant"

    t_all = time.time()
    for sid in sids:
        meta = manifest[sid]
        for arm in arms:
            lat = run_arm(pipeline, sid, arm, meta, device)
            if arm == "rien" and lat is not None:
                refs_board(pipeline, sid, lat, meta, device)
            del lat
            gc.collect()
    print(f"[fidelity2] TERMINE : {len(sids)} scenarios x {len(arms)} bras en "
          f"{(time.time() - t_all) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
