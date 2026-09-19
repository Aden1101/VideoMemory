#!/usr/bin/env python
"""ECHELLE de FIDELITE, driver de generation (GPU).

6 barreaux x 10 scenarios, forme A(0-81) B(81-162) A(162-240) via le chemin legacy
pipeline.inference() (celui des bras oracle1 / oracle1twin / oracle1blur de la batterie
d'arbitrage, hooks valides). Les 6 bras d'un scenario partagent la MEME graine, le MEME
bruit et le MEME flux RNG : ils ne different que par le contenu insere au retour.

  python run_fidelity.py --scenarios scenario01,scenario02      # ou --half A|B|all

Prerequis : extract_refs.py (refs/manifest.json + twin/other/main81 .pt).
GPU : env longlive, CUDA_VISIBLE_DEVICES=<une carte>, CUDA_DEVICE_ORDER=PCI_BUS_ID.
"""
import argparse
import gc
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import fidelity_common as C                                              # noqa: E402

sys.path.insert(0, C.REPO)
sys.path.insert(0, C.PROBE_DIR)
os.chdir(C.REPO)                     # wan_models/ est un chemin relatif au depot

import torch                                                             # noqa: E402
from omegaconf import OmegaConf                                          # noqa: E402
from einops import rearrange                                             # noqa: E402
from torchvision.io import write_video                                   # noqa: E402
import torchvision.utils as vutils                                       # noqa: E402

from run_reid_probe import build_pipeline, _OFF                          # noqa: E402

# la copie locale de l'union d'env doit COUVRIR la table de reference (aucune derive)
for _k, _v in _OFF.items():
    assert _k in C.OFF and C.OFF[_k] == _v, \
        f"fidelity_common.OFF derive de run_reid_probe._OFF sur {_k!r}: {C.OFF.get(_k)!r} != {_v!r}"


def two_step_noise(seed, device):
    """Le bruit de la collecte v2 ([1, 90, ...] sur le generateur global apres set_seed),
    puis les 150 frames restantes sur un generateur SEPARE : le flux RNG global au premier
    bloc est exactement celui de collect_pairs.rollout_collect -> segment A bit a bit."""
    from utils.misc import set_seed
    set_seed(seed)
    n90 = torch.randn([1, C.V2_FRAMES, 16, 60, 104], device=device, dtype=torch.bfloat16)
    g = torch.Generator(device=device)
    g.manual_seed(seed + 424242)
    rest = torch.randn([1, C.FRAMES - C.V2_FRAMES, 16, 60, 104],
                       generator=g, device=device, dtype=torch.bfloat16)
    return torch.cat([n90, rest], dim=1)


def decode_one(pipeline, lat_1f, device):
    """Decode UNE frame latente -> tenseur pixel (1,3,480,832) dans [0,1] (verifs d'oeil)."""
    px = pipeline.vae.decode_to_pixel(
        lat_1f.to(device=device, dtype=torch.bfloat16), use_cache=False)
    pipeline.vae.model.clear_cache()
    return (px[:, 0].float().cpu() * 0.5 + 0.5).clamp(0, 1)


def bitcheck_segment_a(pipeline, sid, latents, meta, device):
    """Compare le segment A genere aux latents v2 stockes (fp16). Non fatal : un ecart
    (autre carte / autre librairie) est documente + planche de decodage pour l'oeil."""
    stored = torch.load(C.main81_pt(sid), map_location="cpu", weights_only=False)["latents"]
    ours = latents[0, :C.SW1].to(torch.float16).cpu()
    diff = (ours.float() - stored.float()).abs()
    n_neq = int((ours != stored).sum().item())
    res = {"sid": sid, "n_elems": int(ours.numel()), "n_diff_fp16": n_neq,
           "max_abs_diff": float(diff.max()), "mean_abs_diff": float(diff.mean()),
           "frame80_max_abs_diff": float(diff[C.ANCHOR_FRAME].max()),
           "bitwise": bool(n_neq == 0)}
    os.makedirs(C.CHECK_DIR, exist_ok=True)
    with open(os.path.join(C.CHECK_DIR, f"{sid}_bitcheck.json"), "w") as f:
        json.dump(res, f, indent=2)
    print(f"[bitcheck] {sid}: segment A vs latents v2 stockes -> "
          f"{'BIT A BIT IDENTIQUE' if res['bitwise'] else 'ECART'} "
          f"(n_diff {n_neq}/{res['n_elems']}, max|d| {res['max_abs_diff']:.3e})", flush=True)
    # planche d'oeil : notre frame 80 vs la frame 80 stockee vs jumeau vs donneur
    tiles, names = [], []
    for tag, lat in (("ours_f80", ours[C.ANCHOR_FRAME][None, None].float()),
                     ("v2_f80", stored[C.ANCHOR_FRAME][None, None].float()),
                     ("twin_f80", torch.load(C.twin_pt(sid), map_location="cpu",
                                             weights_only=False)["anchor"].float()),
                     ("other_f80", torch.load(C.other_pt(sid), map_location="cpu",
                                              weights_only=False)["anchor"].float())):
        tiles.append(decode_one(pipeline, lat, device)[0])
        names.append(tag)
    grid = vutils.make_grid(torch.stack(tiles), nrow=2, padding=6)
    out = os.path.join(C.CHECK_DIR, f"{sid}_refs_{'_'.join(names)}.jpg")
    vutils.save_image(grid, out)
    print(f"[bitcheck] planche refs -> {out}", flush=True)
    return res


def run_arm(pipeline, sid, arm, meta, device):
    prompt, interlude = meta["prompt"], meta["interlude"]
    out_mp4 = C.raw_mp4(sid, arm)
    if os.path.exists(out_mp4):
        print(f"[fidelity] {sid} {arm}: deja fait ({out_mp4})", flush=True)
        return None
    for k, v in C.ARM_ENV[arm].items():
        os.environ[k] = v
    # dump/verif de l'ancre : tout bras dumpe (1er arrive) puis re-VERIFIE bit a bit que son
    # segment A est le meme (assert du hook) -> integrite "seule l'insertion change"
    os.environ["ORACLE_ANCHOR_DUMP"] = C.anchor_pt(sid)
    if arm == "meme_categorie":
        os.environ["ORACLE_ANCHOR_SRC"] = meta["twin_pt"]
    elif arm == "autre_categorie":
        os.environ["ORACLE_ANCHOR_SRC"] = meta["other_pt"]

    seed = int(meta["seed_main"])
    noise = two_step_noise(seed, device)
    print(f"[fidelity] {sid} arm={arm} seed={seed} frames={C.FRAMES} "
          f"sw1={C.SW1} sw2={C.SW2}\n           A='{prompt}'\n           B='{interlude}'",
          flush=True)
    t0 = time.time()
    video, latents = pipeline.inference(
        noise=noise,
        text_prompts_first=[prompt],
        text_prompts_second=[interlude],
        switch_frame_index=C.SW1,
        text_prompts_third=None,               # None -> retour au prompt A
        second_switch_frame_index=C.SW2,
        return_latents=True,
        low_memory=True,
    )
    vid = (rearrange(video[0].cpu().float(), "t c h w -> t h w c") * 255.0)
    os.makedirs(os.path.dirname(out_mp4), exist_ok=True)
    write_video(out_mp4, vid.clamp(0, 255).to(torch.uint8), fps=16)
    print(f"[fidelity] {sid} {arm}: {vid.shape[0]} frames pixel -> {out_mp4} "
          f"({time.time() - t0:.0f}s)", flush=True)

    lat = latents.float().cpu()
    del video, latents
    pipeline.vae.model.clear_cache()
    pipeline.kv_cache1 = None
    pipeline.crossattn_cache = None
    pipeline.ttt_cache = None
    gc.collect()
    torch.cuda.empty_cache()
    return lat


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenarios", type=str, default="",
                    help="liste sid separes par des virgules ; vide = --half")
    ap.add_argument("--half", type=str, default="all", choices=["A", "B", "all"],
                    help="A = scenarios 1-5, B = 6-10 (2 GPUs max, un job par carte)")
    ap.add_argument("--arms", type=str, default=",".join(C.ARMS))
    ap.add_argument("--config", type=str, default=C.CONFIG_YAML)
    args = ap.parse_args()

    manifest = C.load_manifest()
    if args.scenarios:
        sids = [s for s in args.scenarios.split(",") if s]
    elif args.half == "A":
        sids = [sc["sid"] for sc in C.SCENARIOS[:5]]
    elif args.half == "B":
        sids = [sc["sid"] for sc in C.SCENARIOS[5:]]
    else:
        sids = [sc["sid"] for sc in C.SCENARIOS]
    for sid in sids:
        assert sid in manifest, f"{sid} absent de refs/manifest.json (lancer extract_refs.py)"
        for p in ("twin_pt", "other_pt", "main81_pt"):
            assert os.path.exists(manifest[sid][p]), f"{sid}: {manifest[sid][p]} manquant"
    arms = [a for a in args.arms.split(",") if a]
    for a in arms:
        assert a in C.ARM_ENV, f"bras inconnu {a}"
    # `rien` d'abord : c'est lui qui pose le dump d'ancre et la verif bit a bit
    arms = sorted(arms, key=lambda a: a != "rien")

    device = torch.device("cuda")
    torch.set_grad_enabled(False)
    config = OmegaConf.load(args.config)
    config.distributed = False
    pipeline = build_pipeline(config, device)
    print(f"[fidelity] pipeline pret ; scenarios {sids} ; bras {arms}", flush=True)

    t_all = time.time()
    for sid in sids:
        meta = manifest[sid]
        for arm in arms:
            lat = run_arm(pipeline, sid, arm, meta, device)
            if arm == "rien" and lat is not None:
                bitcheck_segment_a(pipeline, sid, lat, meta, device)
            del lat
            gc.collect()
    print(f"[fidelity] TERMINE : {len(sids)} scenarios x {len(arms)} bras en "
          f"{(time.time() - t_all) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
