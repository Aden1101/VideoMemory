#!/usr/bin/env python
# ============================================================================================
# MEMORY 2.0 -- PHASE B diagnostic: WHY does the DINO instance gate start on the wrong side?
#
# Measured by train_writer.py --dino_probe on the A2 writer + A2 decoder (2026-07-28):
#     top-1 0.000 | cos(decoded anchor, its own target) 0.113 vs best negative 0.298
#     -> pos_gap = -0.185, negative for 10/10 val videos.
# Two candidate explanations, and they imply OPPOSITE next moves:
#   (H1) RENDERING: the decoded frame is off the natural-image manifold (low-pass + 2x2
#        checkerboard), so a DINO cosine against REAL frames measures "how blurry", not "who".
#        -> the instance term is unusable UNTIL the rendering is fixed (B1 first, B2 after).
#   (H2) CONTENT: the state simply does not carry the right instance, and no rendering fix
#        would help.  -> phase B cannot deliver, the bottleneck is upstream.
# Three cheap measurements separate them:
#   A  BLUR DOSE-RESPONSE on the TRUE targets: blur the ground truth by sigma px, re-embed,
#      retrieve. If the true frame ALSO stops retrieving itself around the cos_pos the decoder
#      reaches, H1 is sufficient on its own (the metric dies of blur, not of identity).
#   B  CENTERING: remove the population mean of each side (decoded anchors vs bank) before the
#      cosine -- the standard fix when one side sits in a common off-manifold direction.
#   C  SELF-CONSISTENCY: does a decoded anchor at least retrieve its own video AMONG DECODED
#      frames (decoded-vs-decoded)? That is identity information surviving the rendering.
#
# Usage: CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 \
#        python reference/runs/mem2/probe_dino_render.py --device cuda:0 --videos 16
# ============================================================================================
import argparse
import json
import os
import sys

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
import train_writer as T  # noqa: E402
from build_bank_dino import BANK_DINO, latent_to_pixels, pixels_to_dino_input  # noqa: E402


def gaussian_blur(x, sigma):
    if sigma <= 0:
        return x
    k = int(2 * round(3 * sigma) + 1)
    g = torch.arange(k, device=x.device, dtype=x.dtype) - (k - 1) / 2
    g = torch.exp(-g.pow(2) / (2 * sigma ** 2))
    g = g / g.sum()
    C = x.shape[1]
    x = Fn.conv2d(Fn.pad(x, (k // 2,) * 2 + (0, 0), mode="reflect"),
                  g.view(1, 1, 1, k).expand(C, 1, 1, k), groups=C)
    x = Fn.conv2d(Fn.pad(x, (0, 0) + (k // 2,) * 2, mode="reflect"),
                  g.view(1, 1, k, 1).expand(C, 1, k, 1), groups=C)
    return x


def retrieval(z, rows, bank, mask):
    """z (n,D) anchors, rows (n,) their positive bank rows, bank (M,D), mask (n,M) True=masked.
    -> (top1, mean gap, mean cos_pos, mean cos_best_neg)."""
    pos = (z * bank[rows]).sum(-1)
    sim = (z @ bank.t()).masked_fill(mask, -2.0)
    best = sim.max(dim=1).values
    top1 = float((pos > best).float().mean())
    return top1, float((pos - best).mean()), float(pos.mean()), float(best.mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--videos", type=int, default=16)
    ap.add_argument("--writer_ckpt", default=T.WRITER_A2)
    ap.add_argument("--sigmas", type=float, nargs="+", default=[0, 1, 2, 4, 8, 16, 32])
    ap.add_argument("--out", default=os.path.join(D.RUNS_DIR, "logs/dino_render_probe.json"))
    args_p = ap.parse_args()

    argv = ["--phase", "B", "--load", "direct", "--device", args_p.device,
            "--writer_ckpt", args_p.writer_ckpt, "--freeze_writer", "--mixer",
            "--dec_ckpt", args_p.writer_ckpt, "--dino_log",
            "--ages", "5", "10", "20", "35", "--min_age", "5", "--anchor_frac", "1.0"]
    tr = T.WriterTrainer(T.parse_args(argv))
    dev = tr.device
    bank = tr.dino_emb                                   # (N, 768) fp32, row-aligned
    entries = tr.bank.entries
    val_rows = tr.bank.sides["val"]

    # ---- collect val ANCHORS: decoded (A2) + their true bank rows -----------------------
    zs, rows, masks, vids = [], [], [], []
    for meta in tr.val_vids[:args_p.videos]:
        pos = tr.positions_for(meta, k=1, anchor_frac=1.0)
        sel = tr.dino_positions(meta, pos, k=1)
        if not sel:
            continue
        with torch.no_grad():
            pred, _, _, _ = tr.forward_video(meta, pos, n_bptt=0, force_cond=True)
            z = tr.dino_z(pred[sel])
        r = tr.dino_key2row[(meta["video_id"], int(pos[sel[0]]["frame"]))]
        _, m, pool = tr.bank.negatives("val", meta, [p["chunk_i"] for p in pos],
                                       n_neg=0, mask_chunks=tr.args.mask_chunks)
        assert torch.equal(pool, val_rows), "negatives pool != the whole val side"
        zs.append(z[0])
        rows.append(r)
        masks.append(m[sel[0]])
        vids.append((meta["video_id"], meta["mode"]))
        del pred
    Z = torch.stack(zs)                                  # (n, 768) decoded anchors
    R = torch.tensor(rows, device=dev)
    M = torch.stack(masks)                               # (n, |val|)
    NEG = bank[val_rows]
    # row index of the positive INSIDE the val pool (for the masks / centering bookkeeping)
    pos_in_pool = torch.tensor([int((val_rows == r).nonzero()[0]) for r in rows], device=dev)
    print(f"\n[render] {Z.shape[0]} val anchors, val pool {NEG.shape[0]} entries", flush=True)

    res = {}
    pos_cos = (Z * bank[R]).sum(-1)            # exactly the trainer's computation
    sim = (Z @ NEG.t()).masked_fill(M, -2.0)
    best = sim.max(dim=1).values
    res["decoded_raw"] = dict(top1=float((pos_cos > best).float().mean()),
                              gap=float((pos_cos - best).mean()),
                              cos_pos=float(pos_cos.mean()), cos_neg=float(best.mean()))
    print(f"[render] A: decoded anchor (A2, RAW)      top1 {res['decoded_raw']['top1']:.3f} "
          f"gap {res['decoded_raw']['gap']:+.4f}  cos_pos {res['decoded_raw']['cos_pos']:.3f} "
          f"vs neg {res['decoded_raw']['cos_neg']:.3f}", flush=True)

    # ---- A: blur dose-response on the TRUE targets --------------------------------------
    from build_bank import BANK_LAT
    lat_all = torch.load(BANK_LAT, map_location="cpu", weights_only=False)["latents"]
    print("[render] --- blur dose-response on the GROUND-TRUTH target frames ---", flush=True)
    for s in args_p.sigmas:
        zb = []
        with torch.no_grad():
            for r in rows:
                px = latent_to_pixels(tr.vae, lat_all[r][None].to(dev, torch.bfloat16))
                px = gaussian_blur(px.float(), s).clamp(0, 1)
                x = pixels_to_dino_input(px, crop_frac=tr.args.dino_crop, size=tr.args.dino_size)
                zb.append(Fn.normalize(tr.dino(x), dim=-1)[0])
        ZB = torch.stack(zb)
        pc = (ZB * bank[R]).sum(-1)
        bn = (ZB @ NEG.t()).masked_fill(M, -2.0).max(dim=1).values
        res[f"blur_sigma{int(s)}"] = dict(top1=float((pc > bn).float().mean()),
                                          gap=float((pc - bn).mean()),
                                          cos_pos=float(pc.mean()), cos_neg=float(bn.mean()))
        print(f"[render]   sigma={s:>4.0f}px: top1 {res[f'blur_sigma{int(s)}']['top1']:.3f} "
              f"gap {res[f'blur_sigma{int(s)}']['gap']:+.4f}  cos_pos {float(pc.mean()):.3f} "
              f"vs neg {float(bn.mean()):.3f}", flush=True)

    # ---- B: centering (remove each side's population mean) -------------------------------
    mu_d = Fn.normalize(Z.mean(0, keepdim=True), dim=1)
    mu_b = Fn.normalize(bank[val_rows].mean(0, keepdim=True), dim=1)
    Zc = Fn.normalize(Z - mu_d, dim=1)
    Bc = Fn.normalize(bank[val_rows] - mu_b, dim=1)
    Pc = Fn.normalize(bank[R] - mu_b, dim=1)
    pc = (Zc * Pc).sum(-1)
    bn = (Zc @ Bc.t()).masked_fill(M, -2.0).max(dim=1).values
    res["decoded_centered"] = dict(top1=float((pc > bn).float().mean()),
                                   gap=float((pc - bn).mean()),
                                   cos_pos=float(pc.mean()), cos_neg=float(bn.mean()))
    print(f"[render] B: decoded anchor, CENTERED      top1 {res['decoded_centered']['top1']:.3f} "
          f"gap {res['decoded_centered']['gap']:+.4f}  cos_pos {float(pc.mean()):.3f} "
          f"vs neg {float(bn.mean()):.3f}", flush=True)

    # ---- C: decoded-vs-decoded (is identity information present at all?) -----------------
    # positive = the SAME video's decoded anchor is trivially itself, so instead ask: is the
    # decoded anchor's nearest DECODED neighbour a video whose TRUE anchor is also its nearest
    # true neighbour? (rank correlation of the two similarity matrices = does the rendering
    # preserve the metric structure at all?)
    Sd = Z @ Z.t()
    St = bank[R] @ bank[R].t()
    n = Z.shape[0]
    off = ~torch.eye(n, dtype=torch.bool, device=dev)
    sd, st = Sd[off].flatten(), St[off].flatten()
    rank = lambda v: v.argsort().argsort().float()          # noqa: E731
    rs = float(torch.corrcoef(torch.stack([rank(sd), rank(st)]))[0, 1])
    nn_d, nn_t = Sd.masked_fill(~off, -2).argmax(1), St.masked_fill(~off, -2).argmax(1)
    res["structure"] = dict(spearman=rs, nn_agree=float((nn_d == nn_t).float().mean()))
    print(f"[render] C: decoded-vs-true metric structure: Spearman {rs:+.3f}, "
          f"nearest-neighbour agreement {res['structure']['nn_agree']:.2f} "
          f"(chance {1 / (n - 1):.2f})", flush=True)

    res["videos"] = vids
    with open(args_p.out, "w") as f:
        json.dump(res, f, indent=2)
    print(f"\n[render] -> {args_p.out}", flush=True)


if __name__ == "__main__":
    main()
