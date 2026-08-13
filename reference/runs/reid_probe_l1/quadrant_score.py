#!/usr/bin/env python
# ============================================================================================
# PROBE 0 (brainstorm rejoin-vs-rewind, volet 2): rescore the EXISTING A->B->A videos with the
# transition-quadrant metrics -- no generation, pure scoring.
#
# Axis 1, cut_score: how visually brutal is the joint at the RETURN switch (switch2)?
#   cut2_ratio = max lag-1 pixel diff in a +/-8-frame window around the switch2 joint,
#                divided by the median intra-segment lag-1 (video's own motion scale).
#   cut_vs_cut1 = same measure at switch2 divided by same at switch1 (per-video normalizer:
#                switch1 is a "legitimate" prompt cut every arm shares).
#   d2_ratio   = same construction with the second temporal difference (impulse detector).
#
# Axis 2, rejoin_index (border world): does the WORLD around the subject continue or rewind?
#   DINO embeddings of the BORDER region (center 60% crop zeroed out) for:
#     B_tail   = last ~10 latent frames of B before switch2 (the ongoing world)
#     resA_head= first latent frames after the return (what the world became)
#     origA    = the original A segment (the old world)
#   rejoin_border = cos(resA_head, B_tail) - cos(resA_head, origA)
#     > 0: world continued from B (rejoin);  < 0: world snapped back to A's past (rewind).
#   NOTE on reading: for CROSS-SCENE pairs (current 8), the prompt itself orders the world back
#   to A's scene, so rewind-negative is EXPECTED; the value here is metric CALIBRATION against
#   human eyes + arm-vs-arm ordering (does the oracle cut harder than memzero?).
#   id_center_head = cos(center-crop resA_head, center-crop origA): identity axis, early window.
#
# Usage: python quadrant_score.py [--arms memzero,memon,oracle,oracle1,oracle3b,stock]
#                                 [--pairs 0-7] [--seeds 0,1,2]
# Output: results/quadrant_scores.json + printed extremes table.
# ============================================================================================
import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, "/shared/ssd_14T/home/adenelmi/Anchor-Forcing")

import torch
from torchvision.io import read_video

from run_reid_probe import dino_embed, latent_to_pix, PAIRS  # reuse the probe's utilities


def pix(f):
    return 0 if f == 0 else 4 * f - 3


def lag1(frames):
    """frames [T,3,H,W] float01 -> [T-1] mean abs consecutive diff."""
    return (frames[1:] - frames[:-1]).abs().mean(dim=(1, 2, 3))


def zero_center(frames, frac=0.6):
    _, _, H, W = frames.shape
    ch, cw = int(H * frac), int(W * frac)
    top, left = (H - ch) // 2, (W - cw) // 2
    out = frames.clone()
    out[:, :, top:top + ch, left:left + cw] = 0.0
    return out


def center_crop(frames, frac=0.6):
    _, _, H, W = frames.shape
    ch, cw = int(H * frac), int(W * frac)
    top, left = (H - ch) // 2, (W - cw) // 2
    return frames[:, :, top:top + ch, left:left + cw]


def seg_cos(Ea, Eb):
    return float((Ea @ Eb.T).mean())


def score_one(path, device, sw1=81, sw2=162, frames=240, joint_win=8, stride=8):
    v, _, _ = read_video(path, pts_unit="sec", output_format="THWC")
    vid = v.permute(0, 3, 1, 2).float() / 255.0
    T = vid.shape[0]

    # ---- axis 1: cut scores at the two joints, normalized by intra-segment motion ----
    def joint_stats(latent_f):
        j = min(pix(latent_f), T - 2)
        lo, hi = max(0, j - joint_win), min(T - 1, j + joint_win)
        w = vid[lo:hi + 1]
        l1 = lag1(w)
        d2 = (w[2:] - 2 * w[1:-1] + w[:-2]).abs().mean(dim=(1, 2, 3)) if w.shape[0] >= 3 else l1
        return float(l1.max()), float(d2.max())

    cut1, d2_1 = joint_stats(sw1)
    cut2, d2_2 = joint_stats(sw2)
    # intra-segment motion scale: sampled lag-1 well inside each segment
    intra = []
    for lo_f, hi_f in ((20, 75), (95, 155), (175, 235)):
        seg = vid[pix(lo_f):pix(hi_f):4]
        if seg.shape[0] >= 2:
            intra.append(lag1(seg))
    intra = torch.cat(intra)
    m_intra = float(intra.median().clamp_min(1e-6))

    # ---- axis 2: border-world rejoin + center identity, DINO ----
    segs = {
        "B_tail": vid[pix(150):pix(159) + 1:stride],
        "resA_head": vid[pix(165):pix(182) + 1:stride],
        "origA": vid[pix(20):pix(78) + 1:stride],
    }
    Eb = {k: dino_embed(zero_center(s), device) for k, s in segs.items()}
    Ec = {k: dino_embed(center_crop(s), device) for k, s in segs.items()}

    return {
        "cut2_ratio": cut2 / m_intra,
        "cut_vs_cut1": cut2 / max(cut1, 1e-6),
        "d2_ratio": d2_2 / max(d2_1, 1e-6),
        "rejoin_border": seg_cos(Eb["resA_head"], Eb["B_tail"]) - seg_cos(Eb["resA_head"], Eb["origA"]),
        "id_center_head": seg_cos(Ec["resA_head"], Ec["origA"]),
        "intra_motion": m_intra, "cut1_abs": cut1, "cut2_abs": cut2,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", default="memzero,memon,oracle,oracle1,oracle3b,stock")
    ap.add_argument("--pairs", default="0,1,2,3,4,5,6,7")
    ap.add_argument("--seeds", default="0,1,2")
    args = ap.parse_args()
    device = torch.device("cuda")
    torch.set_grad_enabled(False)

    vid_dir = os.path.join(HERE, "videos")
    out_path = os.path.join(HERE, "results", "quadrant_scores.json")
    scores = {}
    if os.path.exists(out_path):
        with open(out_path) as f:
            scores = json.load(f)

    arms = args.arms.split(",")
    for pi in [int(x) for x in args.pairs.split(",")]:
        for si in [int(x) for x in args.seeds.split(",")]:
            for arm in arms:
                key = f"pair{pi}_s{si}_{arm}"
                mp4 = os.path.join(vid_dir, key + ".mp4")
                if key in scores or not os.path.exists(mp4):
                    continue
                try:
                    scores[key] = score_one(mp4, device)
                    print(f"[quadrant] {key}: cut2r={scores[key]['cut2_ratio']:.2f} "
                          f"cutvs1={scores[key]['cut_vs_cut1']:.2f} "
                          f"rejoin={scores[key]['rejoin_border']:+.3f} "
                          f"id={scores[key]['id_center_head']:.3f}", flush=True)
                except Exception as e:
                    print(f"[quadrant] FAILED {key}: {e}", flush=True)
    with open(out_path, "w") as f:
        json.dump(scores, f, indent=2)
    print(f"[quadrant] wrote {out_path} ({len(scores)} videos)")

    # per-arm means + extremes for the human calibration pass
    import collections
    by_arm = collections.defaultdict(list)
    for k, s in scores.items():
        by_arm[k.rsplit("_", 1)[-1]].append((k, s))
    print("\n===== QUADRANT SUMMARY (per arm) =====")
    print(f"{'arm':<10}{'n':>3}{'cut2_ratio':>12}{'cut_vs_cut1':>13}{'rejoin_border':>15}{'id_center':>11}")
    for arm, items in sorted(by_arm.items()):
        n = len(items)
        mean = lambda f: sum(f(s) for _, s in items) / n
        print(f"{arm:<10}{n:>3}{mean(lambda s: s['cut2_ratio']):>12.2f}"
              f"{mean(lambda s: s['cut_vs_cut1']):>13.2f}"
              f"{mean(lambda s: s['rejoin_border']):>15.3f}"
              f"{mean(lambda s: s['id_center_head']):>11.3f}")
    allitems = [(k, s) for items in by_arm.values() for k, s in items]
    for metric in ("cut_vs_cut1", "rejoin_border"):
        ranked = sorted(allitems, key=lambda ks: ks[1][metric])
        print(f"\n-- {metric}: 3 plus BAS --")
        for k, s in ranked[:3]:
            print(f"   {k}: {s[metric]:+.3f}")
        print(f"-- {metric}: 3 plus HAUTS --")
        for k, s in ranked[-3:]:
            print(f"   {k}: {s[metric]:+.3f}")


if __name__ == "__main__":
    main()
