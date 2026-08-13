#!/usr/bin/env python
# ============================================================================================
# Re-ID REVISIT probe, L1 CAMPAIGN (LongLive base, statistical): does the fast-weight memory
# help SUBJECT RESURRECTION on an A->B->A schedule, measured with enough pairs x seeds to
# distinguish signal from seed lottery? (The c3 probe gave n=2 with OPPOSITE signs.)
#
#   8 pairs x 3 seeds x 3 arms:
#       memon   = l1@2000 EMA, memory injection active          (probe_l1.yaml, sink 1)
#       memzero = SAME ckpt + seed, MEM_ZERO_ALL=1               (paired causal twin)
#       stock   = released LongLive (lora.pt, no memory, sink 3) (stock_probe.yaml)
#   SAME seed/noise across arms within a (pair, seed) cell; seed = cfg.seed + pair + 1000*sidx.
#
#   Scoring identical to the c3 probe (DINOv2-vitb14, full + crop60):
#       reid_resA_origA = KEY number, cos(resurrected-A, original-A)
#   Paired stats over the 24 (pair, seed) cells: mean delta(memon - memzero), sign counts,
#   exact two-sided sign test. Stock column = external reference level.
#
# Usage (see run_all.sh for the 4-GPU sharded launch):
#   python run_reid_probe.py --mode selftest
#   python run_reid_probe.py --mode run --pairs 0,1 --seeds 0,1,2
#   python run_reid_probe.py --mode score --pairs 0,1,2,3,4,5,6,7 --seeds 0,1,2
#   python run_reid_probe.py --mode summary          # aggregate all results/*.json
# ============================================================================================
import argparse
import gc
import json
import math
import os
import sys

REPO = "/shared/ssd_14T/home/adenelmi/Anchor-Forcing"
sys.path.insert(0, REPO)
HERE = os.path.dirname(os.path.abspath(__file__))

import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from einops import rearrange
from torchvision.io import write_video, read_video

# (A, B) pairs: A = distinctive re-identifiable subject, B = unrelated interlude.
# Pairs 0-2 = the c3 probe pairs (kept for cross-campaign comparability); 3-7 new, chosen to
# spread subject categories (people / animals / rigid object) so a category effect is visible.
PAIRS = [
    ("a brown dog with a white chest running in a green park",
     "a red vintage car driving down a city street"),
    ("a woman in a yellow raincoat walking on a beach",
     "a group of penguins on ice"),
    ("an orange tabby cat sitting on a wooden table",
     "a hot air balloon rising over mountains"),
    ("a man with a red beard wearing a blue denim jacket drinking coffee at an outdoor cafe",
     "a waterfall cascading in a tropical jungle"),
    ("a white horse with a black mane grazing in a sunny meadow",
     "a freight train crossing a steel bridge at sunset"),
    ("a green and yellow parrot perched on a wooden fence",
     "ocean waves crashing on rocky cliffs"),
    ("a young girl in a purple dress holding a red balloon in a playground",
     "a chef tossing noodles in a busy restaurant kitchen"),
    ("a blue vintage bicycle leaning against a red brick wall",
     "a herd of sheep crossing a country road"),
]

# arms that run on the MAG (l1@2000-EMA) pipeline; "stock" runs on the LongLive pipeline.
#   memsnap = memory ON + MEM_EPISODIC (state snapshotted at A->B, restored at return of A)
#   oracle  = memory OFF + ORACLE_SPLICE (KV window recached from the ORIGINAL A tail at the
#             return switch) = the pure self-oracle upper bound RSD would distill from
MAG_ARMS = ("memon", "memzero", "memsnap", "oracle", "oracle3", "oracle1", "oracle3b",
            "oracle6b", "oracle10b", "oracle15b", "dose03", "dose06", "dose09", "oracle1sink", "oracle1s0", "oracle1k3", "oracle1k6", "memdec", "memdec10", "memdec3", "memdec12",
            "memdec20", "memdec20d",
            # arbiter battery (reference/memory2_probes_registry.md, 2026-07-28)
            "oracle1thru", "oracle1twin", "oracle1blur", "oracle1blur0", "memon20", "anchordump")
_OFF = {"MEM_ZERO_ALL": "0", "MEM_EPISODIC": "0", "ORACLE_SPLICE": "0", "ORACLE_SPLICE_FRAMES": "0",
        "ORACLE_SPLICE_BLOCKS": "", "ORACLE_SPLICE_NOISE": "0", "ORACLE_SPLICE_POS": "", "ORACLE_SPLICE_SLOT": "-1",
        "MEM_DECODE": "", "MEM_DECODE_SLOTS": "1", "MEM_WRITER": "",
        # arbiter battery hooks (all default-off; every arm re-sets EVERY key, so nothing leaks
        # from one arm to the next -- the MEM_WRITER lesson, re-checked by selftest_probes (d))
        "MEM_DECODE_THRU": "0", "ORACLE_ANCHOR_DUMP": "", "ORACLE_ANCHOR_SRC": "",
        "ORACLE_BLUR_SIGMA": "", "MEM_DECODE_DUMP": "",
        # reviewer controls of the S2 claim (2026-08-03, arms noiseanchor / wronganchor of the
        # scenario campaign): anchor CONTENT replaced by calibrated noise / by the anchor decoded
        # from another cell's state. Listed here so EVERY arm of both campaigns re-sets them.
        "MEM_DECODE_NOISE": "0", "MEM_DECODE_NOISE_SEED": "192837465", "MEM_DECODE_SRC": "",
        # pilote "texte insuffisant" (2026-08-03): le QUAND du rappel framebank vient du planning
        "FRAMEBANK_GT": "0",
        # mini-batterie training-free (2026-08-04): l'ancre decodee est AFFUTEE par le prior du
        # modele avant splice (arms sdedit_ours / sdedit_wrong), et le sink epingle de l'ouvreur
        # est RAFRAICHI au retour sans aucun splice ni memoire (arm sinkpurge, le "correctif
        # minimal" qu'un reviewer nous opposera). Listes ici pour que TOUT bras les re-mette a 0.
        "MEM_SDEDIT": "0", "MEM_SDEDIT_SEED": "5150241", "MEM_SDEDIT_FRAMES": "0", "MEM_SINKPURGE": "0"}
ARM_ENV = {
    "memon":   dict(_OFF),
    "memzero": dict(_OFF, MEM_ZERO_ALL="1"),
    "memsnap": dict(_OFF, MEM_EPISODIC="1"),
    "oracle":  dict(_OFF, MEM_ZERO_ALL="1", ORACLE_SPLICE="1"),
    # compression floor: only the last 3 (resp. 1) window slots carry original-A KV
    "oracle3": dict(_OFF, MEM_ZERO_ALL="1", ORACLE_SPLICE="1", ORACLE_SPLICE_FRAMES="3"),
    "oracle1": dict(_OFF, MEM_ZERO_ALL="1", ORACLE_SPLICE="1", ORACLE_SPLICE_FRAMES="1"),
    # gating hole (localplan_A §3.3): oracle restricted to the 3 memory blocks only
    "oracle3b": dict(_OFF, MEM_ZERO_ALL="1", ORACLE_SPLICE="1", ORACLE_SPLICE_BLOCKS="5,16,27"),
    # depth sweep: how many (evenly spaced) blocks must receive A-KV to recover the oracle delta
    "oracle6b": dict(_OFF, MEM_ZERO_ALL="1", ORACLE_SPLICE="1",
                     ORACLE_SPLICE_BLOCKS="0,5,10,15,20,25"),
    "oracle10b": dict(_OFF, MEM_ZERO_ALL="1", ORACLE_SPLICE="1",
                      ORACLE_SPLICE_BLOCKS="0,3,6,9,12,15,18,21,24,27"),
    "oracle15b": dict(_OFF, MEM_ZERO_ALL="1", ORACLE_SPLICE="1",
                      ORACLE_SPLICE_BLOCKS="0,2,4,6,8,10,12,14,16,18,20,22,24,26,28"),
    # tolerance probe (oracle1 + degraded anchor frame): how imperfect may the decoded recall
    # frame be before resurrection collapses?
    # position probe: the single A anchor at the OLDEST (sink-adjacent) window slot
    "oracle1sink": dict(_OFF, MEM_ZERO_ALL="1", ORACLE_SPLICE="1", ORACLE_SPLICE_FRAMES="1",
                        ORACLE_SPLICE_POS="head"),
    # position curve: anchor at window slot k frames before the most recent
    "oracle1k3": dict(_OFF, MEM_ZERO_ALL="1", ORACLE_SPLICE="1", ORACLE_SPLICE_FRAMES="1",
                      ORACLE_SPLICE_SLOT="3"),
    "oracle1k6": dict(_OFF, MEM_ZERO_ALL="1", ORACLE_SPLICE="1", ORACLE_SPLICE_FRAMES="1",
                      ORACLE_SPLICE_SLOT="6"),
    # persistent position: anchor OVERWRITES the sink slot (read forever), window stays normal
    "oracle1s0": dict(_OFF, MEM_ZERO_ALL="1", ORACLE_SPLICE="1", ORACLE_SPLICE_FRAMES="1",
                      ORACLE_SPLICE_POS="sink0"),
    "dose03": dict(_OFF, MEM_ZERO_ALL="1", ORACLE_SPLICE="1", ORACLE_SPLICE_FRAMES="1",
                   ORACLE_SPLICE_NOISE="0.3"),
    "dose06": dict(_OFF, MEM_ZERO_ALL="1", ORACLE_SPLICE="1", ORACLE_SPLICE_FRAMES="1",
                   ORACLE_SPLICE_NOISE="0.6"),
    "dose09": dict(_OFF, MEM_ZERO_ALL="1", ORACLE_SPLICE="1", ORACLE_SPLICE_FRAMES="1",
                   ORACLE_SPLICE_NOISE="0.9"),
    # decoded-anchor arm (design v1 verdict): MEM_DECODE ckpt path is injected at
    # run time from --memdec-ckpt (see run_arm); MEM_ZERO_ALL=1 = residual injection off,
    # states still written (the decoder needs them live)
    "memdec":  dict(_OFF, MEM_ZERO_ALL="1"),
    "memdec3": dict(_OFF, MEM_ZERO_ALL="1", MEM_DECODE_SLOTS="3"),
    "memdec12": dict(_OFF, MEM_ZERO_ALL="1"),
    # memory 2.0 phase A (co-trained WRITER + decoder ADDRESSING, offline write-replay):
    #   memdec20  = MEM_WRITER (theta_w A2) + MEM_DECODE (decoder A2)  -> the full mem2.0 arm
    #   memdec20d = MEM_DECODE (decoder A2) ALONE, writer stays stock l1@2000-EMA
    #               -> attribution control: writer gain vs decoder-addressing gain
    # Both ckpt paths are injected at run time (see run_arm: --mem2-writer-ckpt/--mem2-dec-ckpt).
    "memdec20":  dict(_OFF, MEM_ZERO_ALL="1"),
    "memdec20d": dict(_OFF, MEM_ZERO_ALL="1"),
    # ---- ARBITER BATTERY (memory2_probes_registry.md) ---------------------------------------
    # oracle1thru: the REAL anchor frame goes through the REAL write, the state is decoded AT
    #   THE EVICTION (write age 1) and the cached frame is spliced at the return = "perfect
    #   content through the write+read channel, without the forgetting between write and
    #   revisit". Writer A2 + decoder A2 injected at run time.
    "oracle1thru": dict(_OFF, MEM_ZERO_ALL="1", MEM_DECODE_THRU="1"),
    # oracle1twin: oracle1 with the anchor frame of ANOTHER SEED of the same pair -- same
    #   prompt, perfect sharpness, WRONG instance (tests the "content is everything" model).
    "oracle1twin": dict(_OFF, MEM_ZERO_ALL="1", ORACLE_SPLICE="1", ORACLE_SPLICE_FRAMES="1"),
    # oracle1blur: oracle1 with the TRUE frame blurred to our decoders' fidelity (sigma px,
    #   VAE decode -> gaussian blur -> VAE encode). ORACLE_BLUR_SIGMA injected at run time.
    "oracle1blur": dict(_OFF, MEM_ZERO_ALL="1", ORACLE_SPLICE="1", ORACLE_SPLICE_FRAMES="1"),
    # oracle1blur0: CALIBRATION of the blur arm -- the same VAE decode->encode round trip with
    #   NO blur. selftest (b) measured that an isolated 1-frame round trip already costs
    #   rel-L2 0.49 in latent space, so this arm is what "sigma = 0" is worth on the re-ID
    #   scale; oracle1blur minus oracle1blur0 = the part actually due to the blur.
    "oracle1blur0": dict(_OFF, MEM_ZERO_ALL="1", ORACLE_SPLICE="1", ORACLE_SPLICE_FRAMES="1",
                         ORACLE_BLUR_SIGMA="0"),
    # memon20: ACTIVE memory read (no MEM_ZERO_ALL) with the memory-2.0 writer.
    "memon20": dict(_OFF),
    # anchordump: memzero rollout whose only purpose is to persist the original-A anchor latent
    #   of a cell (auto-scheduled when the twin arm needs a seed nobody else generates).
    "anchordump": dict(_OFF, MEM_ZERO_ALL="1"),
    "stock":   dict(_OFF),
    # LongLive OFFICIEL (rope_enhance false : sans le Tri-Region RoPE d'AF, audit 2026-07-24) ;
    # la colonne "stock" (avec tri-region) est conservée : l'écart = la part positionnelle d'AF
    "stockoff": dict(_OFF),
}

# memory 2.0 phase A artefacts (train_writer.py --phase A, run mem2_writer_A2)
MEM2_RUNS = "/shared/ssd_14T_project/adenelmi/mem2_runs/writer"
MEM2_WRITER_DEFAULT = os.path.join(MEM2_RUNS, "mem2_writer_A2_final.pt")
MEM2_DEC_DEFAULT = os.path.join(MEM2_RUNS, "dec_A2_for_memdec.pt")

# original-A anchor latents, persisted per (pair, seed) by the ORACLE_ANCHOR_DUMP hook: the twin
# arm of cell (p, s) splices the anchor of cell (p, (s + offset) % n_seeds). Project disk (home
# is full); ~200 kB per cell.
ANCHOR_DIR = "/shared/ssd_14T_project/adenelmi/mem2_runs/anchors"
# arms whose segment A is the canonical silent-memory rollout, hence may write the shared anchor
# file (the hook re-checks bitwise equality when the file already exists). memon20 is NOT here:
# with the injection live its segment A is a different rollout.
ANCHOR_DUMP_ARMS = ("memzero", "oracle1", "oracle1thru", "oracle1blur", "anchordump")


def anchor_path(pair_idx, seed_idx):
    return os.path.join(ANCHOR_DIR, f"pair{pair_idx}_s{seed_idx}.pt")

# ----------------------------- DINO (copied verbatim from the c3 probe) ---------------------
DINO_MEAN = (0.485, 0.456, 0.406)
DINO_STD = (0.229, 0.224, 0.225)
_DINO = None


def get_dino(device):
    global _DINO
    if _DINO is None:
        hub = os.path.expanduser("~/.cache/torch/hub")
        torch.hub.set_dir(hub)
        local = os.path.join(hub, "facebookresearch_dinov2_main")
        m = torch.hub.load(local, "dinov2_vitb14", source="local", pretrained=True)
        m.eval().to(device)
        for p in m.parameters():
            p.requires_grad_(False)
        _DINO = m
        print(f"[probe] loaded frozen DINOv2-vitb14 on {device}")
    return _DINO


@torch.no_grad()
def dino_embed(frames01, device, batch=64):
    """frames01 [T,3,H,W] in [0,1] -> L2-normalized DINO CLS embeddings [T,D] (cpu fp32)."""
    dino = get_dino(device)
    mean = torch.tensor(DINO_MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(DINO_STD, device=device).view(1, 3, 1, 1)
    out = []
    for i in range(0, frames01.shape[0], batch):
        x = frames01[i:i + batch].to(device).float()
        x = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)
        x = (x - mean) / std
        f = dino(x)
        out.append(F.normalize(f, dim=-1).cpu())
    return torch.cat(out, 0)


# ----------------------------- pipeline build (mirrors inference/inference.py) --------------
def build_pipeline(config, device):
    """SwitchCausalInferencePipeline with the trained weights:
    base.pt -> LoRA adapters -> generator_lora -> generator_memory -> optional EMA overlay.
    Configs with memory_layers [] (stock arm) skip the memory-load asserts."""
    from pipeline.switch_causal_inference import SwitchCausalInferencePipeline
    from utils.lora_utils import configure_lora_for_model
    from utils.memory import DynamicSwapInstaller
    import peft

    has_memory = bool(list(config.model_kwargs.memory_layers or []))
    pipeline = SwitchCausalInferencePipeline(config, device=device)

    # base weights (inference.py non-EMA branch)
    state_dict = torch.load(config.generator_ckpt, map_location="cpu")
    if "generator" in state_dict:
        raw = state_dict["generator"]
    elif "model" in state_dict:
        raw = state_dict["model"]
    else:
        raise ValueError(f"Generator state dict not found in {config.generator_ckpt}")
    missing, unexpected = pipeline.generator.load_state_dict(raw, strict=False)
    non_memory_missing = [k for k in missing if "memory" not in k]
    assert not non_memory_missing, f"Missing non-memory keys: {non_memory_missing[:5]}"
    assert not unexpected, f"Unexpected keys: {list(unexpected)[:5]}"

    # LoRA wrap + trained adapters
    pipeline.generator.model = configure_lora_for_model(
        pipeline.generator.model, model_name="generator",
        lora_config=config.adapter, is_main_process=True)
    lora_checkpoint = torch.load(config.lora_ckpt, map_location="cpu")
    assert isinstance(lora_checkpoint, dict) and "generator_lora" in lora_checkpoint, \
        f"no generator_lora in {config.lora_ckpt}"
    peft.set_peft_model_state_dict(pipeline.generator.model, lora_checkpoint["generator_lora"])
    print(f"[probe] LoRA weights loaded from {config.lora_ckpt}")

    if has_memory:
        # trained MAG memory (full-rank) -- must load, else the probe runs a random memory
        assert "generator_memory" in lora_checkpoint, \
            f"no generator_memory in {config.lora_ckpt} -- refusing to probe a random-init memory"
        mem_sd = lora_checkpoint["generator_memory"]
        result = pipeline.generator.load_state_dict(mem_sd, strict=False)
        assert not result.unexpected_keys, f"Unexpected memory keys: {list(result.unexpected_keys)[:5]}"
        still_missing_mem = [k for k in result.missing_keys if "memory" in k]
        assert not still_missing_mem, f"Memory NOT loaded: {still_missing_mem[:5]}"
        print(f"[probe] {len(mem_sd)} trained memory keys loaded")
    else:
        assert "generator_memory" not in lora_checkpoint, \
            "checkpoint HAS memory weights but config grafts no memory layers -- config mismatch"
        print("[probe] no memory graft (stock arm)")

    # EMA overlay (copied from inference.py c3 block): replace the trainable subset (LoRA +
    # memory) with the smoothed shadow. config use_ema:true REFUSES to fall back to raw.
    if bool(getattr(config, "use_ema", False)):
        assert isinstance(lora_checkpoint, dict) and "generator_ema" in lora_checkpoint, \
            f"use_ema:true but no generator_ema shadow in {config.lora_ckpt}"

        def _clean_ema_key(name):
            for pfx in ("_fsdp_wrapped_module.", "_checkpoint_wrapped_module.", "_orig_mod."):
                name = name.replace(pfx, "")
            return name

        shadow = {_clean_ema_key(k): v for k, v in lora_checkpoint["generator_ema"].items()}
        ema_res = pipeline.generator.load_state_dict(shadow, strict=False)
        n_matched = len(shadow) - len(ema_res.unexpected_keys)
        assert not ema_res.unexpected_keys, \
            f"[EMA] unexpected shadow keys (naming mismatch): {list(ema_res.unexpected_keys)[:5]}"
        assert n_matched > 0, "[EMA] ZERO shadow keys matched -- refusing silent raw-weight probe"
        print(f"[probe] EMA overlay: {n_matched}/{len(shadow)} trainable shadow keys (smoothed LoRA+memory)")

    pipeline = pipeline.to(dtype=torch.bfloat16)
    DynamicSwapInstaller.install_model(pipeline.text_encoder, device=device)
    pipeline.generator.to(device=device)
    pipeline.vae.to(device=device)
    return pipeline


def release_vram():
    """Callers must `del`/rebind their own pipeline reference FIRST (a del inside a helper only
    drops the helper's local ref -- lesson from the selftest2 OOM), then call this."""
    gc.collect()
    torch.cuda.empty_cache()


# ----------------------------- generation ---------------------------------------------------
def cell_seed(config, pair_idx, seed_idx):
    """Same seed across ARMS within a (pair, seed_idx) cell; distinct per pair and per seed."""
    return int(config.seed) + pair_idx + 1000 * seed_idx


def run_arm(pipeline, config, pair_idx, seed_idx, arm, frames, sw1, sw2, device, out_mp4):
    """One A->B->A generation. Returns (latents_cpu, blocks_with_written_ttt_state)."""
    from utils.misc import set_seed
    A, B = PAIRS[pair_idx]
    for k, v in ARM_ENV[arm].items():
        os.environ[k] = v
    if arm in ("memdec", "memdec3", "memdec12"):
        _ck = getattr(config, "_memdec_ckpt", "") or ""
        assert _ck, "arm memdec requires --memdec-ckpt"
        os.environ["MEM_DECODE"] = _ck
    if arm in ("memdec20", "memdec20d", "oracle1thru"):
        # memory 2.0: the A2 decoder for both, the A2 writer for memdec20 only (the "d" arm is
        # the attribution control = new addressing on the STOCK l1 states).
        _dk = getattr(config, "_mem2_dec_ckpt", "") or ""
        assert _dk and os.path.exists(_dk), f"arm {arm} requires --mem2-dec-ckpt (got '{_dk}')"
        os.environ["MEM_DECODE"] = _dk
    if arm in ("memdec20", "oracle1thru", "memon20"):
        _wk = getattr(config, "_mem2_writer_ckpt", "") or ""
        assert _wk and os.path.exists(_wk), \
            f"arm {arm} requires --mem2-writer-ckpt (got '{_wk}')"
        os.environ["MEM_WRITER"] = _wk
    # ---- arbiter battery: anchor dump / twin source / blur sigma ------------------------------
    if arm == "oracle1thru" and getattr(config, "_anchor_dir", ""):
        # eye-pack: persist the frame the writer+decoder actually re-read at the eviction
        # (200 kB per cell) so the PNG can put it next to the true anchor.
        os.environ["MEM_DECODE_DUMP"] = os.path.join(
            config._anchor_dir, f"thru_pair{pair_idx}_s{seed_idx}")
    if arm in ANCHOR_DUMP_ARMS and getattr(config, "_anchor_dir", ""):
        os.environ["ORACLE_ANCHOR_DUMP"] = os.path.join(
            config._anchor_dir, f"pair{pair_idx}_s{seed_idx}.pt")
    if arm == "oracle1twin":
        _ns = int(getattr(config, "_n_seeds", 3) or 3)
        _off = int(getattr(config, "_twin_offset", 1))
        _tw = (seed_idx + _off) % _ns
        _tp = os.path.join(getattr(config, "_anchor_dir", ANCHOR_DIR),
                           f"pair{pair_idx}_s{_tw}.pt")
        assert os.path.exists(_tp), \
            (f"arm oracle1twin needs the original-A anchor of pair{pair_idx} seed{_tw} ({_tp}) "
             f"-- run a dumping arm ({'/'.join(ANCHOR_DUMP_ARMS)}) on that cell first")
        os.environ["ORACLE_ANCHOR_SRC"] = _tp
        print(f"[probe][oracle1twin] pair{pair_idx} s{seed_idx} <- anchor of seed {_tw} "
              f"(offset {_off})", flush=True)
    if arm == "oracle1blur":
        os.environ["ORACLE_BLUR_SIGMA"] = str(getattr(config, "_blur_sigma", 24.0))
    seed = cell_seed(config, pair_idx, seed_idx)
    set_seed(seed)
    noise = torch.randn([1, frames, 16, 60, 104], device=device, dtype=torch.bfloat16)
    print(f"[probe] pair{pair_idx} s{seed_idx} arm={arm} seed={seed} frames={frames} "
          f"switch@{sw1} switchback@{sw2}\n        A='{A}'\n        B='{B}'", flush=True)
    video, latents = pipeline.inference(
        noise=noise,
        text_prompts_first=[A],
        text_prompts_second=[B],
        switch_frame_index=sw1,
        text_prompts_third=None,          # None -> defaults back to prompt A
        second_switch_frame_index=sw2,
        return_latents=True,
        low_memory=True,
    )
    written = [bi for bi in range(pipeline.num_transformer_blocks)
               if pipeline.ttt_cache[bi].state is not None]
    print(f"[probe][{arm}] ttt_cache.state written on blocks: {written}", flush=True)

    vid = (rearrange(video[0].cpu().float(), "t c h w -> t h w c") * 255.0)
    write_video(out_mp4, vid.clamp(0, 255).to(torch.uint8), fps=16)
    print(f"[probe][{arm}] saved {out_mp4} ({vid.shape[0]} pixel frames)", flush=True)

    lat = latents.float().cpu()
    pipeline.vae.model.clear_cache()
    pipeline.kv_cache1 = None
    pipeline.crossattn_cache = None
    pipeline.ttt_cache = None
    gc.collect()
    torch.cuda.empty_cache()
    return lat, written


# ----------------------------- scoring (verbatim from the c3 probe) -------------------------
def latent_to_pix(f_lo, f_hi):
    lo = 0 if f_lo == 0 else 4 * f_lo - 3
    hi = 4 * f_hi
    return lo, hi


def sample_seg(vid01, f_lo, f_hi, stride):
    lo, hi = latent_to_pix(f_lo, f_hi)
    hi = min(hi, vid01.shape[0] - 1)
    return vid01[lo:hi + 1:stride]


def center_crop(frames, frac=0.6):
    _, _, H, W = frames.shape
    ch, cw = int(H * frac), int(W * frac)
    top, left = (H - ch) // 2, (W - cw) // 2
    return frames[:, :, top:top + ch, left:left + cw]


def seg_cos(Ea, Eb):
    return float((Ea @ Eb.T).mean())


def score_video(path, frames, sw1, sw2, device, settle=20, resettle=12, stride=8):
    v, _, _ = read_video(path, pts_unit="sec", output_format="THWC")
    vid = v.permute(0, 3, 1, 2).float() / 255.0
    segs = {
        "origA": (settle, sw1 - 3),
        "B": (sw1 + resettle, sw2 - 3),
        "resA": (sw2 + resettle, frames - 1),
    }
    out = {"segments_latent": {k: list(rng) for k, rng in segs.items()}}
    for variant, xform in (("full", lambda f: f), ("crop60", center_crop)):
        E = {k: dino_embed(xform(sample_seg(vid, *rng, stride)), device) for k, rng in segs.items()}
        half = max(1, E["origA"].shape[0] // 2)
        out[variant] = {
            "reid_resA_origA": seg_cos(E["resA"], E["origA"]),
            "ctrl_B_origA": seg_cos(E["B"], E["origA"]),
            "ctrl_resA_B": seg_cos(E["resA"], E["B"]),
            "self_origA": seg_cos(E["origA"][:half], E["origA"][half:]),
            "n_frames": {k: int(E[k].shape[0]) for k in E},
        }
    return out


# ----------------------------- selftest -----------------------------------------------------
def selftest(config, stock_config, device):
    """Tiny end-to-end check on BOTH pipelines: (1) MAG A->B->A runs, ttt written exactly on
    memory_layers, MEM_ZERO_ALL changes latents; (2) EMA overlay asserted in build; (3) stock
    pipeline (no memory) runs A->B->A and writes NO ttt state; (4) memsnap and oracle paths
    execute and diverge from their references. sw1=12 so the oracle splice has a full 12-frame
    A window."""
    frames, sw1, sw2 = 27, 12, 21
    os.makedirs(os.path.join(HERE, "videos"), exist_ok=True)

    pipeline = build_pipeline(config, device)
    os.environ["MEM_EVICT_DEBUG"] = "1"
    mem_layers = sorted(int(x) for x in config.model_kwargs.memory_layers)
    lat = {}
    for arm in ("memon", "memzero"):
        vid_path = os.path.join(HERE, "videos", f"selftest_{arm}.mp4")
        lat[arm], written = run_arm(pipeline, config, 0, 0, arm, frames, sw1, sw2, device, vid_path)
        if arm == "memon":
            assert written == mem_layers, f"ttt written on {written}, expected {mem_layers}"
    d = (lat["memon"] - lat["memzero"]).abs().max().item()
    m = (lat["memon"] - lat["memzero"]).abs().mean().item()
    print(f"[selftest] latent diff mem-ON vs mem-OFF: max={d:.6f} mean={m:.6f}")
    assert d > 0, "MEM_ZERO_ALL had NO effect -> memory injection NOT live"
    os.environ["MEM_EVICT_DEBUG"] = "0"

    # new-arm smoke: each new env path must run AND change the post-return rollout vs its
    # reference (same seed): memsnap vs memon (state restore), oracle vs memzero (KV splice)
    for arm, ref in (("memsnap", "memon"), ("oracle", "memzero")):
        vid_path = os.path.join(HERE, "videos", f"selftest_{arm}.mp4")
        lat[arm], _ = run_arm(pipeline, config, 0, 0, arm, frames, sw1, sw2, device, vid_path)
        d = (lat[arm] - lat[ref]).abs().max().item()
        print(f"[selftest] latent diff {arm} vs {ref}: max={d:.6f}")
        assert d > 0, f"{arm} had NO effect vs {ref} -> env hook not live"
    del pipeline
    release_vram()

    stock = build_pipeline(stock_config, device)
    vid_path = os.path.join(HERE, "videos", "selftest_stock.mp4")
    _, written = run_arm(stock, stock_config, 0, 0, "stock", frames, sw1, sw2, device, vid_path)
    assert written == [], f"stock arm wrote ttt state on {written} -- it must have NO memory"
    del stock
    release_vram()
    print("[selftest] PASS")


# ----------------------------- summary ------------------------------------------------------
def sign_test_p(n_pos, n_neg):
    """Exact two-sided sign test p-value (ties dropped)."""
    n = n_pos + n_neg
    if n == 0:
        return 1.0
    k = min(n_pos, n_neg)
    tail = sum(math.comb(n, i) for i in range(0, k + 1)) / (2 ** n)
    return min(1.0, 2.0 * tail)


def summarize(res_dir, pair_ids, seed_ids):
    cells = {}
    for pi in pair_ids:
        for si in seed_ids:
            p = os.path.join(res_dir, f"pair{pi}_s{si}_scores.json")
            if os.path.exists(p):
                with open(p) as f:
                    cells[(pi, si)] = json.load(f)
    if not cells:
        print("[summary] no result JSONs found yet")
        return
    all_arms = [a for a in ("memon", "memzero", "memsnap", "oracle", "oracle3", "oracle1",
                            "oracle3b", "oracle6b", "oracle10b", "oracle15b",
                            "dose03", "dose06", "dose09", "oracle1sink", "oracle1s0", "oracle1k3", "oracle1k6",
                            "memdec", "memdec3", "memdec10", "memdec12", "memdec20", "memdec20d",
                            "oracle1thru", "oracle1twin", "oracle1blur", "oracle1blur0", "memon20",
                            "stock", "stockoff")
                if any(a in ps for ps in cells.values())]
    print(f"\n========== L1 RE-ID REVISIT SUMMARY ({len(cells)} cells, arms: {','.join(all_arms)}) ==========")
    summary = {"n_cells": len(cells), "variants": {}}
    for variant in ("full", "crop60"):
        vsum = {"arm_means": {}, "paired_vs_memzero": {}}
        means_line = []
        for arm in all_arms:
            vals = [ps[arm][variant]["reid_resA_origA"] for ps in cells.values() if arm in ps]
            if vals:
                vsum["arm_means"][arm] = sum(vals) / len(vals)
                means_line.append(f"{arm} {sum(vals)/len(vals):.4f} (n={len(vals)})")
        print(f"\n[{variant}] mean reid_resA_origA: " + "  ".join(means_line))
        for probe_arm in ("memon", "memsnap", "oracle", "oracle3", "oracle1", "oracle3b",
                          "oracle6b", "oracle10b", "oracle15b", "dose03", "dose06", "dose09", "oracle1sink", "oracle1s0", "oracle1k3", "oracle1k6", "memdec", "memdec10", "memdec3", "memdec12",
                          "memdec20", "memdec20d",
                          "oracle1thru", "oracle1twin", "oracle1blur", "oracle1blur0", "memon20"):
            deltas, per_pair, cell_deltas = [], {}, {}
            for (pi, si), ps in sorted(cells.items()):
                if probe_arm in ps and "memzero" in ps:
                    d = ps[probe_arm][variant]["reid_resA_origA"] - ps["memzero"][variant]["reid_resA_origA"]
                    deltas.append(d)
                    per_pair.setdefault(pi, []).append(d)
                    cell_deltas[f"{pi}_{si}"] = d
            if not deltas:
                continue
            mean_d = sum(deltas) / len(deltas)
            sd = (sum((x - mean_d) ** 2 for x in deltas) / max(1, len(deltas) - 1)) ** 0.5
            n_pos = sum(1 for x in deltas if x > 0)
            n_neg = sum(1 for x in deltas if x < 0)
            p = sign_test_p(n_pos, n_neg)
            print(f"  delta({probe_arm} - memzero), {len(deltas)} paired cells: "
                  f"mean {mean_d:+.4f}  sd {sd:.4f}  sem {sd / max(1, len(deltas)) ** 0.5:.4f}  "
                  f"signs {n_pos}+/{n_neg}-  sign-p {p:.4f}")
            for pi in sorted(per_pair):
                ds = per_pair[pi]
                print(f"    pair{pi} ({PAIRS[pi][0][:52]}...): {sum(ds)/len(ds):+.4f}  " +
                      " ".join(f"{x:+.3f}" for x in ds))
            vsum["paired_vs_memzero"][probe_arm] = {
                "mean_delta": mean_d, "sd": sd, "n_pos": n_pos, "n_neg": n_neg, "sign_p": p,
                "per_pair_mean": {str(pi): sum(ds) / len(ds) for pi, ds in per_pair.items()},
                "deltas": cell_deltas,
            }
        summary["variants"][variant] = vsum
    out = os.path.join(res_dir, "summary.json")
    with open(out, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[summary] wrote {out}")
    print("=================================================================")


# ----------------------------- main ---------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["selftest", "run", "score", "summary"], default="run")
    ap.add_argument("--pairs", type=str, default="0", help="comma list of pair indices")
    ap.add_argument("--seeds", type=str, default="0,1,2", help="comma list of seed indices")
    ap.add_argument("--arms", type=str, default="memon,memzero,stock")
    ap.add_argument("--frames", type=int, default=240)
    ap.add_argument("--switch1", type=int, default=81)
    ap.add_argument("--switch2", type=int, default=162)
    ap.add_argument("--config", type=str, default=os.path.join(HERE, "probe_l1.yaml"))
    ap.add_argument("--stock-config", type=str, default=os.path.join(HERE, "stock_probe.yaml"))
    ap.add_argument("--memdec-ckpt", type=str, default="", help="AnchorFrameDecoder ckpt for the memdec arm")
    ap.add_argument("--mem2-writer-ckpt", type=str, default=MEM2_WRITER_DEFAULT,
                    help="memory 2.0 writer ckpt (MEM_WRITER) for the memdec20 arm")
    ap.add_argument("--mem2-dec-ckpt", type=str, default=MEM2_DEC_DEFAULT,
                    help="memory 2.0 co-trained decoder (MEM_DECODE) for memdec20/memdec20d")
    # ---- arbiter battery ---------------------------------------------------------------------
    ap.add_argument("--anchor-dir", type=str, default=ANCHOR_DIR,
                    help="where the original-A anchor latents are dumped/read (twin arm)")
    ap.add_argument("--twin-offset", type=int, default=1,
                    help="oracle1twin takes the anchor of seed (s + offset) %% n_seeds; "
                         "0 == oracle1 bitwise (selftest a)")
    ap.add_argument("--blur-sigma", type=float, default=24.0,
                    help="oracle1blur gaussian sigma in PIXELS (0 = pure VAE round trip)")
    args = ap.parse_args()

    assert args.switch1 % 3 == 0 and args.switch2 % 3 == 0, "switches must be multiples of 3"
    device = torch.device("cuda")
    config = OmegaConf.load(args.config)
    config.distributed = False
    config._memdec_ckpt = args.memdec_ckpt
    config._mem2_writer_ckpt = args.mem2_writer_ckpt
    config._mem2_dec_ckpt = args.mem2_dec_ckpt
    config._anchor_dir = args.anchor_dir
    config._twin_offset = args.twin_offset
    config._blur_sigma = args.blur_sigma
    config._n_seeds = len([x for x in args.seeds.split(",") if x != ""])
    os.makedirs(args.anchor_dir, exist_ok=True)
    stock_config = OmegaConf.load(args.stock_config)
    stock_config.distributed = False
    torch.set_grad_enabled(False)

    pair_ids = [int(x) for x in args.pairs.split(",") if x != ""]
    seed_ids = [int(x) for x in args.seeds.split(",") if x != ""]
    arms = [a for a in args.arms.split(",") if a != ""]
    vid_dir = os.path.join(HERE, "videos")
    res_dir = os.path.join(HERE, "results")
    os.makedirs(vid_dir, exist_ok=True)
    os.makedirs(res_dir, exist_ok=True)

    if args.mode == "summary":
        summarize(res_dir, pair_ids if args.pairs != "0" else range(len(PAIRS)), seed_ids)
        return
    if args.mode == "selftest":
        selftest(config, stock_config, device)
        return

    def mp4_path(pi, si, arm):
        return os.path.join(vid_dir, f"pair{pi}_s{si}_{arm}.mp4")

    if args.mode == "run":
        # all MAG videos first with ONE pipeline build, then all stock videos with the other
        # (stock uses the same seeds -> same noise; MEM_ZERO_ALL is irrelevant there but reset).
        mag_todo = [(pi, si, a) for pi in pair_ids for si in seed_ids
                    for a in arms if a in MAG_ARMS and not os.path.exists(mp4_path(pi, si, a))]
        # arbiter battery ORDER: the twin arm CONSUMES the original-A anchor of another seed of
        # the same pair, so every dumping arm must run first. Stable sort -> the rest of the
        # order (and every existing campaign) is unchanged.
        if any(a == "oracle1twin" for _, _, a in mag_todo):
            need = set()
            for pi in pair_ids:
                for si in seed_ids:
                    tw = (si + args.twin_offset) % max(1, len(seed_ids))
                    if os.path.exists(anchor_path(pi, tw)):
                        continue
                    if any((pi, tw, a) in mag_todo for a in ANCHOR_DUMP_ARMS):
                        continue          # a scheduled arm of that cell will dump it
                    need.add((pi, tw))
            if need:
                print(f"[probe] auto-scheduling {len(need)} anchordump cell(s) for the twin arm: "
                      f"{sorted(need)}", flush=True)
                mag_todo = [(pi, si, "anchordump") for pi, si in sorted(need)] + mag_todo
        mag_todo.sort(key=lambda t: t[2] == "oracle1twin")
        stock_cfgs = {"stock": stock_config}
        if "stockoff" in arms:
            _c = OmegaConf.load(os.path.join(HERE, "stock_official.yaml"))
            _c.distributed = False
            stock_cfgs["stockoff"] = _c
        stock_todo = [(pi, si) for pi in pair_ids for si in seed_ids
                      if "stock" in arms and not os.path.exists(mp4_path(pi, si, "stock"))]
        if mag_todo:
            pipeline = build_pipeline(config, device)
            for pi, si, arm in mag_todo:
                run_arm(pipeline, config, pi, si, arm, args.frames, args.switch1, args.switch2,
                        device, mp4_path(pi, si, arm))
            del pipeline
            release_vram()
        for _sarm, _scfg in stock_cfgs.items():
            _todo = [(pi, si) for pi in pair_ids for si in seed_ids
                     if _sarm in arms and not os.path.exists(mp4_path(pi, si, _sarm))]
            if not _todo:
                continue
            stock = build_pipeline(_scfg, device)
            for pi, si in _todo:
                run_arm(stock, _scfg, pi, si, _sarm, args.frames, args.switch1,
                        args.switch2, device, mp4_path(pi, si, _sarm))
            del stock
            release_vram()

    # score everything requested; MERGE into any existing cell JSON so later arm campaigns
    # (memsnap/oracle) extend the memon/memzero/stock scores instead of clobbering them
    for pi in pair_ids:
        for si in seed_ids:
            out_json = os.path.join(res_dir, f"pair{pi}_s{si}_scores.json")
            cell = {}
            if os.path.exists(out_json):
                with open(out_json) as f:
                    cell = json.load(f)
            cell.update({"pair": PAIRS[pi], "seed": cell_seed(config, pi, si),
                         "frames": args.frames, "switch1": args.switch1, "switch2": args.switch2,
                         "checkpoint": str(config.lora_ckpt), "use_ema": bool(config.use_ema)})
            missing = [a for a in arms if not os.path.exists(mp4_path(pi, si, a))]
            if missing:
                print(f"[probe] pair{pi} s{si}: missing videos for {missing}, skipping cell")
                continue
            for arm in arms:
                cell[arm] = score_video(mp4_path(pi, si, arm), args.frames, args.switch1,
                                        args.switch2, device)
            with open(out_json, "w") as f:
                json.dump(cell, f, indent=2)
            print(f"[probe] wrote {out_json}", flush=True)

    summarize(res_dir, pair_ids, seed_ids)


if __name__ == "__main__":
    main()
