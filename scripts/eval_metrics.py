#!/usr/bin/env python
"""Three-way evaluation: the scoring code behind every results/*/eval*/metrics.txt.

Input: the videos of three arms generated on the same prompts and the same seed (memory active,
memory zeroed with MEM_ZERO_ALL=1, unmodified model) and their VBench outputs. Output: the table.
The title line carries no run name: the tables under results/ were relabelled from their folder.

Per video and per arm:
  motion    mean |x[t] - x[t-2]| in pixels (lag 2), which a period-2 flicker cannot inflate, and
            lag 1 next to it: both near zero is a freeze, lag 2 near zero with lag 1 high is a
            flicker. --optical_flow adds the Farneback flow magnitude as a cross-check.
  identity  cos(DINO(frame 0), DINO(last frame)), and the whole curve against frame 0. DINOv2
            ViT-B/14, loaded as model/twin_objective.py loads it.
Per pair of arms, frame by frame on the same prompt and seed, so that only the arm differs:
  memon vs memzero   what the memory injection changes, against the 0.18 noise floor
  memzero vs stock   what the LoRA co-training alone changes
VBench SC / BC / DD are collated for the three arms, then the thresholds below give a verdict.

Standalone: it imports nothing from the training stack.
"""
import argparse
import glob
import json
import math
import os
import sys

import torch
import torch.nn.functional as F
from torchvision.io import read_video

# ----------------------------- PRE-REGISTERED THRESHOLDS -----------------------------
NOISE_FLOOR = 0.18       # DINO cosine-distance: delta(memon,memzero) below this => memory made ~no difference
FREEZE_LAG = 0.010       # normalized-[0,1] mean-abs pixel diff below this => "~0" motion
VBENCH_PARITY_TOL = 0.02 # |arm - stock| on a VBench dimension within this => parity
DRIFT_PARITY_TOL = 0.03  # |id_cos_arm - id_cos_stock| within this => identity-drift parity

DINO_MEAN = (0.485, 0.456, 0.406)
DINO_STD = (0.229, 0.224, 0.225)

VB_DIMS = {"subject_consistency": "SC", "background_consistency": "BC", "dynamic_degree": "DD"}
ARMS = ("memon", "memzero", "stock")


# ----------------------------- DINO (matches _get_dino in model/twin_objective.py) -----------------------------
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
        print(f"[metrics] loaded frozen DINOv2-vitb14 on {device}")
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


# ----------------------------- IO -----------------------------
def load_video(path, device, small_hw=(120, 208)):
    """Returns:
       big01   [T,3,H,W] float in [0,1] on `device` (native res, for DINO),
       small01 [T,3,h,w] float16 in [0,1] on CPU (downsampled, for pixel metrics / pairing)."""
    v, _, _ = read_video(path, pts_unit="sec", output_format="THWC")  # [T,H,W,3] uint8
    big = v.permute(0, 3, 1, 2).contiguous().to(device).float().div_(255.0)  # [T,3,H,W]
    small = F.interpolate(big, size=small_hw, mode="bilinear", align_corners=False).half().cpu()
    return big, small


def lag_motion(small01):
    """small01 [T,3,h,w] fp16 cpu in [0,1] -> (lag1, lag2) mean-abs temporal diff."""
    x = small01.float()
    lag1 = (x[1:] - x[:-1]).abs().mean().item() if x.shape[0] > 1 else 0.0
    lag2 = (x[2:] - x[:-2]).abs().mean().item() if x.shape[0] > 2 else 0.0
    return lag1, lag2


def optical_flow_mag(small01):
    """Mean Farneback optical-flow magnitude between consecutive frames (cross-check freeze signal)."""
    import cv2
    import numpy as np
    x = (small01.float().clamp(0, 1) * 255).to(torch.uint8).permute(0, 2, 3, 1).numpy()  # [T,h,w,3]
    gray = [cv2.cvtColor(f, cv2.COLOR_RGB2GRAY) for f in x]
    mags = []
    for a, b in zip(gray[:-1], gray[1:]):
        flow = cv2.calcOpticalFlowFarneback(a, b, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        mags.append(float(np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2).mean()))
    return float(sum(mags) / len(mags)) if mags else 0.0


def key_of(path):
    return os.path.splitext(os.path.basename(path))[0]  # e.g. "3-0"


def list_videos(d):
    return sorted(glob.glob(os.path.join(d, "*-*.mp4")), key=lambda p: _prompt_order(key_of(p)))


# ----------------------------- VBench collation -----------------------------
def collate_vbench(vbench_dir):
    """Newest results_*_<dim>_eval_results.json -> {SC,BC,DD: aggregate score or None}."""
    scores = {"SC": None, "BC": None, "DD": None}
    if not vbench_dir or not os.path.isdir(vbench_dir):
        return scores
    for dim, short in VB_DIMS.items():
        matches = sorted(glob.glob(os.path.join(vbench_dir, f"results_*_{dim}_eval_results.json")))
        if not matches:
            continue
        try:
            d = json.load(open(matches[-1]))
            scores[short] = round(float(d[dim][0]), 4)
        except Exception as e:
            print(f"[metrics] WARN could not parse {matches[-1]}: {e}")
    return scores


# ----------------------------- per-arm processing -----------------------------
def process_arm(name, vdir, device, small_hw, do_flow):
    """Returns dict: key -> {lag1,lag2,flow,id_cos,drift_curve,emb(cpu),small(cpu fp16)}."""
    res = {}
    paths = list_videos(vdir)
    print(f"[metrics] arm={name} dir={vdir} -> {len(paths)} videos")
    for p in paths:
        k = key_of(p)
        big, small = load_video(p, device, small_hw)
        lag1, lag2 = lag_motion(small)
        emb = dino_embed(big, device)                       # [T,D] cpu
        c = (emb @ emb[0:1].T).squeeze(1)                   # cos(f0, ft)
        entry = {
            "lag1": lag1, "lag2": lag2,
            "id_cos": float(c[-1].item()),                  # first->last identity retention
            "drift_curve": [round(float(x), 4) for x in c.tolist()],
            "emb": emb, "small": small,
        }
        if do_flow:
            entry["flow"] = optical_flow_mag(small)
        res[k] = entry
        del big
        torch.cuda.empty_cache()
        fl = f" flow={entry.get('flow'):.4f}" if do_flow else ""
        print(f"    {k}: lag1={lag1:.4f} lag2={lag2:.4f}{fl} id_cos(f0,fL)={entry['id_cos']:.3f} T={emb.shape[0]}")
    return res


def _mean(xs):
    return sum(xs) / len(xs) if xs else float("nan")


def _prompt_order(key):
    return [int(t) for t in key.split("-")]


def paired_delta(arm_a, arm_b):
    """Distance between two arms on the videos they share, frame by frame. Returns the mean DINO
    cosine distance, the mean pixel L2, the number of shared videos and the DINO distance of
    each video."""
    keys = sorted(set(arm_a) & set(arm_b), key=_prompt_order)
    dds, pxs = [], []
    for k in keys:
        ea, eb = arm_a[k]["emb"], arm_b[k]["emb"]
        n = min(ea.shape[0], eb.shape[0])
        cos = (ea[:n] * eb[:n]).sum(-1).clamp(-1, 1)        # per-frame cos
        dds.append(float((1.0 - cos).mean().item()))         # cosine DISTANCE
        sa, sb = arm_a[k]["small"].float(), arm_b[k]["small"].float()
        m = min(sa.shape[0], sb.shape[0])
        pxs.append(float((sa[:m] - sb[:m]).pow(2).sum(1).sqrt().mean().item()))  # per-pixel L2, meaned
    return _mean(dds), _mean(pxs), len(keys), {k: round(d, 4) for k, d in zip(keys, dds)}


def arm_summary(arm):
    entries = list(arm.values())
    return {
        "n": len(entries),
        "lag1": _mean([e["lag1"] for e in entries]),
        "lag2": _mean([e["lag2"] for e in entries]),
        "id_cos": _mean([e["id_cos"] for e in entries]),
        "flow": _mean([e["flow"] for e in entries]) if entries and "flow" in entries[0] else None,
    }


# ----------------------------- decision (pre-registered) -----------------------------
# Everything from here to main() is plain Python on the numbers computed above, so
# tests/test_eval_report.py replays it against the tables published under results/.
def compare_to_stock(s_on, s_stock, vb):
    """memon minus stock: per VBench dimension (None where a score is missing), whether every
    available dimension is within tolerance, the identity gap, and whether it is within
    tolerance. A parity is None when there is nothing to compare."""
    vb_diffs = {}
    for dim in VB_DIMS.values():
        a, b = vb["memon"][dim], vb["stock"][dim]
        vb_diffs[dim] = None if (a is None or b is None) else round(a - b, 4)
    within = [abs(v) <= VBENCH_PARITY_TOL for v in vb_diffs.values() if v is not None]
    vb_par = all(within) if within else None
    if math.isnan(s_on["id_cos"]) or math.isnan(s_stock["id_cos"]):
        return vb_diffs, vb_par, None, None
    drift_diff = round(s_on["id_cos"] - s_stock["id_cos"], 4)
    return vb_diffs, vb_par, drift_diff, abs(drift_diff) <= DRIFT_PARITY_TOL


def verdict_of(s_on, d_on_zero, noise_floor, vb_diffs, vb_par, drift_diff, drift_par):
    """The pre-registered reading, tried in this order: frozen, flickering, memory without
    effect, then memory active and at parity with, better than or worse than the stock model."""
    still = s_on["lag2"] < FREEZE_LAG
    if still and s_on["lag1"] < FREEZE_LAG:
        return "FREEZE  (lag-2 ~0 AND lag-1 ~0: memon output is static)"
    if still and s_on["lag1"] >= FREEZE_LAG:
        return "FLICKER (lag-2 ~0 but lag-1 >>0: period-2 phase-swap, not real motion)"
    if not d_on_zero >= noise_floor:        # a NaN distance, no shared video, reads as inert
        return (f"INERT   (delta(memon,memzero)={d_on_zero:.4f} < floor {noise_floor}: memory made "
                f"~no difference to the output)\n"
                f"          NOTE: cannot distinguish INERT from CANCELLED without injection-dose telemetry. "
                f"If a telemetry probe shows injection dose > 0 while this delta < floor => CANCELLED.")
    if vb_par and drift_par:
        return ("ACTIVE-NEUTRAL (the honest win): memory changes the output (delta > floor) yet "
                "VBench SC/BC/DD AND identity-drift are at parity with AF-stock.")
    if vb_par is None or drift_par is None:
        return ("ACTIVE (delta > floor) but VBench/drift parity INCONCLUSIVE (missing a VBench dim or "
                "stock drift). Fill the gap, then re-read. Flag for human eye.")
    better = sum(1 for v in vb_diffs.values() if v and v > 0) >= 2 or (drift_diff and drift_diff > 0)
    return (f"ACTIVE-{'POSITIVE' if better else 'NEGATIVE'} (delta > floor + a VBench/drift difference "
            f"vs AF-stock). FLAG FOR HUMAN EYE (ship côte-à-côte).")


def render_report(summaries, vb, vb_diffs, on_zero, zero_stock, drift_diff, drift_par, verdict,
                  noise_floor):
    """The text table. `summaries` and `vb` are keyed by arm; `on_zero` and `zero_stock` are the
    (DINO distance, pixel L2, n, per-video distances) that paired_delta returns."""
    score = lambda x: f"{x:.4f}" if x is not None else "  n/a"
    rule = "=" * 92
    lines = [rule, "3-way memory eval  (memon | memzero=LoRA-drift-only | AF-stock)", rule,
             f"pre-registered thresholds: NOISE_FLOOR(DINO cos-dist)={noise_floor}  FREEZE_LAG={FREEZE_LAG}  "
             f"VBENCH_PARITY_TOL={VBENCH_PARITY_TOL}  DRIFT_PARITY_TOL={DRIFT_PARITY_TOL}",
             "",
             f"{'arm':<10}{'n':>3}  {'lag1':>8}{'lag2(flkblind)':>16}{'flow':>9}  {'id_cos(f0,fL)':>14}"]
    for arm in ARMS:
        s = summaries[arm]
        lines.append(f"{arm:<10}{s['n']:>3}  {s['lag1']:>8.4f}{s['lag2']:>16.4f}{score(s['flow']):>9}"
                     f"  {s['id_cos']:>14.3f}")
    lines += ["", "VBench (aggregate; higher=better for SC/BC/DD):",
              f"{'dim':<6}{'memon':>9}{'memzero':>9}{'stock':>9}{'memon-stock':>13}"]
    for dim in VB_DIMS.values():
        diff = vb_diffs[dim]
        lines.append(f"{dim:<6}" + "".join(f"{score(vb[arm][dim]):>9}" for arm in ARMS)
                     + f"{(f'{diff:+.4f}' if diff is not None else '  n/a'):>13}")
    lines += ["", "PAIRED deltas (same seed 42, per-frame):",
              f"  delta(memon, memzero)  DINO cos-dist = {on_zero[0]:.4f}   pixel-L2 = {on_zero[1]:.4f}   "
              f"(n={on_zero[2]} paired)   [PRIMARY vs floor {noise_floor}]",
              f"      per-prompt DINO cos-dist: {on_zero[3]}",
              f"  delta(memzero, AF-stock) DINO cos-dist = {zero_stock[0]:.4f}   pixel-L2 = {zero_stock[1]:.4f}   "
              f"(n={zero_stock[2]} paired)   [pure LoRA co-train drift, attribution guard]",
              "", f"identity-drift(memon) - drift(stock) = {drift_diff}   (parity={drift_par})", "",
              "-" * 92, f"VERDICT: {verdict}", "-" * 92]
    return "\n".join(lines)


# ----------------------------- main -----------------------------
def parse_args():
    ap = argparse.ArgumentParser()
    for arm in ARMS:
        ap.add_argument(f"--{arm}_dir", required=True)
    for arm in ARMS:
        ap.add_argument(f"--vbench_{arm}", default="")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--noise_floor", type=float, default=NOISE_FLOOR)
    ap.add_argument("--optical_flow", action="store_true", help="also compute Farneback flow magnitude (slow)")
    ap.add_argument("--small_h", type=int, default=120)
    ap.add_argument("--small_w", type=int, default=208)
    ap.add_argument("--out", default="")
    return ap.parse_args()


def main():
    args = parse_args()
    device = args.device if torch.cuda.is_available() else "cpu"
    small_hw = (args.small_h, args.small_w)

    arms = {arm: process_arm(arm, getattr(args, f"{arm}_dir"), device, small_hw, args.optical_flow)
            for arm in ARMS}
    on_zero = paired_delta(arms["memon"], arms["memzero"])
    zero_stock = paired_delta(arms["memzero"], arms["stock"])
    summaries = {arm: arm_summary(videos) for arm, videos in arms.items()}
    vb = {arm: collate_vbench(getattr(args, f"vbench_{arm}")) for arm in ARMS}

    vb_diffs, vb_par, drift_diff, drift_par = compare_to_stock(summaries["memon"], summaries["stock"], vb)
    verdict = verdict_of(summaries["memon"], on_zero[0], args.noise_floor,
                         vb_diffs, vb_par, drift_diff, drift_par)
    report = render_report(summaries, vb, vb_diffs, on_zero, zero_stock, drift_diff, drift_par,
                           verdict, args.noise_floor)
    print(report)

    if args.out:
        dump = {
            "thresholds": {"noise_floor": args.noise_floor, "freeze_lag": FREEZE_LAG,
                           "vbench_parity_tol": VBENCH_PARITY_TOL, "drift_parity_tol": DRIFT_PARITY_TOL},
            "arms": summaries,
            "vbench": vb, "vbench_memon_minus_stock": vb_diffs,
            "delta_memon_memzero": {"dino_cos_dist": on_zero[0], "pixel_l2": on_zero[1], "n": on_zero[2],
                                    "per_prompt": on_zero[3]},
            "delta_memzero_stock": {"dino_cos_dist": zero_stock[0], "pixel_l2": zero_stock[1],
                                    "n": zero_stock[2]},
            "drift_memon_minus_stock": drift_diff,
            "drift_curves": {arm: {k: v["drift_curve"] for k, v in videos.items()}
                             for arm, videos in arms.items()},
            "verdict": verdict,
        }
        text_path = os.path.splitext(args.out)[0] + ".txt"
        with open(args.out, "w") as f:
            json.dump(dump, f, indent=2)
        with open(text_path, "w") as f:
            f.write(report + "\n")
        print(f"\n[metrics] wrote {args.out} and {text_path}")


if __name__ == "__main__":
    sys.exit(main())
