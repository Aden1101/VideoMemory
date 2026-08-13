#!/usr/bin/env python
# ============================================================================================
# MEMORY 2.0 -- PHASE B deliverable 2a: the DINO side of the InfoNCE bank.
#
# WHY a second bank. The J3 acc_probe verdict (memory2_j3_report 5.1): in LATENT crop60 space
# a decoded anchor is SYSTEMATICALLY closer to another video's frame than to its own target
# (top-1 = 0.000, pos_gap -0.19..-0.36 at every pooling) -- the latent cosine is dominated by
# the "decoded frame = low-pass" artefact, so an InfoNCE there would push on sharpness, not on
# instance. Phase B moves the instance term to the space the VERDICT is measured in: DINOv2,
# through a differentiable VAE decode of ONE frame.
#
# This script embeds the bank TARGETS (data, they never move) ONCE, offline, with EXACTLY the
# harness re-ID pipeline (reference/runs/reid_probe_l1/run_reid_probe.py):
#     VAE decode -> [0,1] -> centre crop 60% -> bilinear 224x224 -> ImageNet norm
#     -> DINOv2 ViT-B/14 CLS -> L2 normalize
# The model itself is imported from run_reid_probe.get_dino, so "same DINO as the harness" is
# a code identity, not a comment. (build_bank.py's optional --dino field is ViT-S/14 AUDIT
# space, deliberately different; this one is the training space and is stored separately.)
#
# Output: bank/bank_dino.pt = {"emb": (N, 768) fp16 L2-normalized, aligned ROW BY ROW with
# bank.pt["entries"], "frame_key" (video_id, frame) list, "meta"}.
#
# Usage (GPU pre-check in the same command, house rule):
#   CUDA_DEVICE_ORDER=PCI_BUS_ID nvidia-smi --query-gpu=index,memory.free --format=csv && \
#   CUDA_VISIBLE_DEVICES=0 python reference/runs/mem2/build_bank_dino.py
#   ... --selftest_only            # re-check an existing bank_dino.pt
# ============================================================================================
import argparse
import os
import sys
import time

REPO = "/shared/ssd_14T/home/adenelmi/Anchor-Forcing"
PROBE_DIR = os.path.join(REPO, "reference/runs/reid_probe_l1")
FRAME_DEC = os.path.join(REPO, "reference/runs/frame_dec")
HERE = os.path.dirname(os.path.abspath(__file__))
for p in (REPO, PROBE_DIR, FRAME_DEC, HERE):
    sys.path.insert(0, p)
os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")

import torch  # noqa: E402
import torch.nn.functional as Fn  # noqa: E402

import mem2_data as D  # noqa: E402
from build_bank import BANK_DIR, BANK_LAT, BANK_PT, check, _FAILS  # noqa: E402

BANK_DINO = os.path.join(BANK_DIR, "bank_dino.pt")
DINO_MEAN = (0.485, 0.456, 0.406)
DINO_STD = (0.229, 0.224, 0.225)


# ----------------------------- the shared (differentiable) preprocessing --------------------
def pixels_to_dino_input(px01, crop_frac=0.6, size=224, device=None):
    """px01 (B, 3, H, W) in [0,1] -> normalized 224x224 DINO input. DIFFERENTIABLE (the phase-B
    instance term backprops through this exact function; the bank uses it under no_grad, so the
    two sides of the InfoNCE can never drift apart)."""
    H, W = px01.shape[-2:]
    ch, cw = int(H * crop_frac), int(W * crop_frac)
    top, left = (H - ch) // 2, (W - cw) // 2
    x = px01[..., top:top + ch, left:left + cw]
    x = Fn.interpolate(x, size=(size, size), mode="bilinear", align_corners=False)
    dev = device if device is not None else x.device
    mean = torch.tensor(DINO_MEAN, device=dev, dtype=x.dtype).view(1, 3, 1, 1)
    std = torch.tensor(DINO_STD, device=dev, dtype=x.dtype).view(1, 3, 1, 1)
    return (x - mean) / std


def latent_to_pixels(vae, lat, use_checkpoint=False):
    """lat (B, 16, 60, 104) -> (B, 3, 480, 832) in [0,1]. ONE latent frame -> ONE pixel frame
    (Wan VAE: T_pix = 4*(T_lat-1)+1). Differentiable: WanVAEWrapper.decode_to_pixel contains no
    no_grad and the VAE params are frozen (requires_grad_(False)), so the gradient flows to the
    latent -- i.e. to the decoder -- and to nothing else."""
    if lat.dim() == 4:
        lat = lat.unsqueeze(1)                         # (B, 1, 16, 60, 104)
    assert lat.shape[1] == 1, f"one frame at a time, got {tuple(lat.shape)}"

    def _dec(z):
        return vae.decode_to_pixel(z, use_cache=False, to_cpu=False)

    if use_checkpoint and torch.is_grad_enabled() and lat.requires_grad:
        from torch.utils.checkpoint import checkpoint
        px = checkpoint(_dec, lat, use_reentrant=False)
    else:
        px = _dec(lat)
    px = px[:, 0].float()                              # (B, 3, H, W) in [-1, 1]
    return (px + 1.0) / 2.0


# ----------------------------- build --------------------------------------------------------
def build(args):
    device = torch.device(args.device)
    os.chdir(REPO)                                     # WanVAEWrapper reads repo-relative paths
    bank = torch.load(BANK_PT, map_location="cpu", weights_only=False)
    entries = bank["entries"]
    lat = torch.load(BANK_LAT, map_location="cpu", weights_only=False)["latents"]
    assert lat.shape[0] == len(entries), \
        f"bank_latents {lat.shape[0]} != {len(entries)} entries -- rebuild the bank first"
    print(f"[dino] {len(entries)} bank entries, latents {tuple(lat.shape)}", flush=True)

    from train_decoder import build_vae
    from run_reid_probe import get_dino
    vae = build_vae(device)
    dino = get_dino(device)

    out = []
    t0 = time.time()
    with torch.no_grad():
        for i in range(0, lat.shape[0], args.batch):
            z = lat[i:i + args.batch].to(device=device, dtype=torch.bfloat16)
            px = latent_to_pixels(vae, z).clamp(0, 1)
            x = pixels_to_dino_input(px, crop_frac=args.crop_frac, size=args.size)
            out.append(Fn.normalize(dino(x).float(), dim=-1).cpu())
            if (i // args.batch) % 25 == 0:
                done = i + z.shape[0]
                rate = done / max(1e-6, time.time() - t0)
                print(f"[dino] {done}/{lat.shape[0]} ({rate:.1f} frames/s, "
                      f"eta {(lat.shape[0] - done) / max(rate, 1e-6) / 60:.1f} min)", flush=True)
    emb = torch.cat(out, 0)
    print(f"[dino] embeddings {tuple(emb.shape)} in {(time.time() - t0) / 60:.1f} min", flush=True)

    payload = {
        "format": "mem2_bank_dino_v1",
        "emb": emb.to(torch.float16),                  # (N, 768) L2-normalized, row-aligned
        "frame_key": [(e["video_id"], int(e["frame"])) for e in entries],
        "meta": {"model": "dinov2_vitb14 (run_reid_probe.get_dino)", "dim": int(emb.shape[1]),
                 "crop_frac": args.crop_frac, "size": args.size, "n": int(emb.shape[0]),
                 "bank": BANK_PT, "created": time.strftime("%Y-%m-%d %H:%M:%S")},
    }
    tmp = BANK_DINO + ".tmp"
    torch.save(payload, tmp)
    os.replace(tmp, BANK_DINO)
    print(f"[dino] -> {BANK_DINO} ({os.path.getsize(BANK_DINO) / 1e6:.0f} MB)", flush=True)
    del vae, dino
    torch.cuda.empty_cache()
    return payload


# ----------------------------- selftests ----------------------------------------------------
def selftests(payload=None):
    bank = torch.load(BANK_PT, map_location="cpu", weights_only=False)
    entries = bank["entries"]
    d = payload if payload is not None else torch.load(BANK_DINO, map_location="cpu",
                                                       weights_only=False)
    emb = d["emb"].float()
    check("row alignment: one DINO embedding per bank entry",
          emb.shape[0] == len(entries), f"{emb.shape[0]} vs {len(entries)}")
    check("dim = 768 (ViT-B/14, the harness model)", emb.shape[1] == 768, str(emb.shape[1]))
    n = emb.norm(dim=1)
    check("L2-normalized", bool(((n - 1).abs() < 2e-3).all()),
          f"norm min {float(n.min()):.4f} max {float(n.max()):.4f}")
    check("finite", bool(torch.isfinite(emb).all()))
    check("frame_key matches the entries",
          all(k == (e["video_id"], int(e["frame"])) for k, e in zip(d["frame_key"], entries)))

    # twins: same corpus_idx, opposite `twin` flag -> MUST be on the same split side (F1.1b)
    vids = bank["videos"]
    split = bank["split"]
    tidx = D.twin_index(vids)
    pairs = [(e["main"], e["twin"]) for e in tidx.values() if e["main"] and e["twin"]]
    same_side = [split[a] == split[b] for a, b in pairs]
    check("twins paired by corpus_idx AND on the same split side",
          bool(pairs) and all(same_side), f"{len(pairs)} pairs, {sum(same_side)} same-side")

    # a twin pair must be a HARD negative in DINO space, not a duplicate: report the margin
    by_vid = {}
    for i, e in enumerate(entries):
        by_vid.setdefault(e["video_id"], []).append(i)
    sims, self_sims = [], []
    for a, b in pairs[:30]:
        ia = [i for i in by_vid.get(a, []) if entries[i]["is_anchor"]]
        ib = [i for i in by_vid.get(b, []) if entries[i]["is_anchor"]]
        if not ia or not ib:
            ia, ib = by_vid.get(a, [])[:1], by_vid.get(b, [])[:1]
        if ia and ib:
            sims.append(float(emb[ia[0]] @ emb[ib[0]]))
            self_sims.append(float(emb[ia[0]] @ emb[ia[0]]))
    if sims:
        check("twin similarity is high but < 1 (hard negative, not a duplicate)",
              max(sims) < 0.999, f"cos(twin) mean {sum(sims) / len(sims):.3f} "
              f"max {max(sims):.3f} over {len(sims)} pairs")

    # anchors: every switch video contributes exactly one anchor entry
    sw = {v["video_id"] for v in vids if v["mode"] == "switch"}
    anc = {}
    for e in entries:
        if e["is_anchor"]:
            anc[e["video_id"]] = anc.get(e["video_id"], 0) + 1
    check("one anchor entry per switch video",
          set(anc) == sw and set(anc.values()) <= {1},
          f"{len(anc)} anchors / {len(sw)} switch videos")

    # retrieval sanity in DINO space: a target retrieves ITSELF against the whole bank
    idx = torch.randperm(emb.shape[0])[:200]
    sim = emb[idx] @ emb.t()
    top1 = sim.argmax(dim=1)
    check("self-retrieval 200/200 in DINO space", bool((top1 == idx).all()),
          f"{int((top1 == idx).sum())}/200")
    return not _FAILS


def main():
    ap = argparse.ArgumentParser(description="DINO bank for the phase-B instance term")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--crop_frac", type=float, default=0.6)
    ap.add_argument("--size", type=int, default=224)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--selftest_only", action="store_true")
    args = ap.parse_args()
    payload = None if args.selftest_only else build(args)
    ok = selftests(payload)
    print(f"\n[dino] SELFTESTS: {'ALL PASS' if ok else 'FAILURES: ' + ', '.join(_FAILS)}",
          flush=True)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
