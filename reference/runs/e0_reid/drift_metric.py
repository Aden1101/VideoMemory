#!/usr/bin/env python
"""
E0 identity-drift + motion ruler (Fable spec, 2026-07-02).

Given a manifest of generated videos (each tagged with a prompt id + an arm label),
compute per (prompt, arm):
  - identity-drift curve over time = 1 - cos( DINOv2(frame_t), anchor )
    anchor = mean DINOv2 embedding over the first `anchor_sec` seconds (chunk-0 window)
  - motion curve = mean RAFT optical-flow magnitude between adjacent sampled frames,
    aggregated in `win_sec` windows
Oracle normalization: reference distance = mean pairwise (1-cos) between the chunk-0
anchors of DIFFERENT prompts (i.e. different subjects). normalized_drift = drift / ref.
A drift of 1.0 ~ "as different as a different subject".

Freeze coupling: a segment whose motion < motion_floor is flagged FROZEN (its identity
score is not trustworthy -- a still frame trivially has ~0 drift).

Everything runs offline from the torch.hub / torchvision caches.

Usage:
  python drift_metric.py --manifest manifest.json --out out/ [--fps 4] [--anchor_sec 2]
                         [--win_sec 10] [--motion_floor 0.5] [--device cuda:0]

manifest.json = [ {"path": ".../v.mp4", "prompt": "jogging", "arm": "stock"}, ... ]
"""
import os, sys, json, argparse, subprocess, tempfile, glob, math
import numpy as np

def log(*a):
    print(*a, flush=True)

# ---------------------------------------------------------------- frame IO
def extract_frames(video, fps, tmpdir, max_w=832):
    os.makedirs(tmpdir, exist_ok=True)
    # native aspect, cap width for speed; fps subsample
    vf = f"fps={fps},scale='min({max_w},iw)':-2"
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-i", video,
           "-vf", vf, os.path.join(tmpdir, "f_%05d.png")]
    subprocess.run(cmd, check=True)
    return sorted(glob.glob(os.path.join(tmpdir, "f_*.png")))

# ---------------------------------------------------------------- models
def load_dino(device):
    import torch
    hub = os.path.expanduser("~/.cache/torch/hub")
    torch.hub.set_dir(hub)
    local = os.path.join(hub, "facebookresearch_dinov2_main")
    model = torch.hub.load(local, "dinov2_vitb14", source="local", pretrained=True)
    model.eval().to(device)
    return model

def load_raft(device):
    from torchvision.models.optical_flow import raft_small, Raft_Small_Weights
    model = raft_small(weights=Raft_Small_Weights.C_T_V2, progress=False)
    model.eval().to(device)
    return model

# ---------------------------------------------------------------- embeddings
def dino_embed(model, paths, device, bs=32):
    import torch
    from PIL import Image
    import torchvision.transforms.functional as TF
    mean = [0.485, 0.456, 0.406]; std = [0.229, 0.224, 0.225]
    embs = []
    with torch.no_grad():
        for i in range(0, len(paths), bs):
            batch = []
            for p in paths[i:i+bs]:
                im = Image.open(p).convert("RGB")
                # center square -> 224 (14*16). DINOv2 patch14 needs mult of 14; 224 ok.
                w, h = im.size; s = min(w, h)
                im = im.crop(((w-s)//2, (h-s)//2, (w-s)//2+s, (h-s)//2+s)).resize((224, 224))
                t = TF.to_tensor(im); t = TF.normalize(t, mean, std)
                batch.append(t)
            x = torch.stack(batch).to(device)
            f = model(x)                       # [B, 768] CLS embedding
            f = torch.nn.functional.normalize(f, dim=-1)
            embs.append(f.cpu().numpy())
    return np.concatenate(embs, 0)             # [T, 768], L2-normalized

def raft_motion(model, paths, device, stride=1, max_pairs=None, size=(256, 448)):
    import torch
    from PIL import Image
    import torchvision.transforms.functional as TF
    mags = []
    idxs = list(range(0, len(paths)-stride, stride))
    if max_pairs and len(idxs) > max_pairs:
        idxs = idxs[:: max(1, len(idxs)//max_pairs)]
    with torch.no_grad():
        for i in idxs:
            def ld(p):
                im = Image.open(p).convert("RGB").resize((size[1], size[0]))
                t = TF.to_tensor(im) * 2 - 1   # [-1,1]
                return t
            a = ld(paths[i]).unsqueeze(0).to(device)
            b = ld(paths[i+stride]).unsqueeze(0).to(device)
            flow = model(a, b)[-1]             # [1,2,H,W]
            m = torch.sqrt((flow**2).sum(1)).mean().item()
            mags.append((i, m))
    return mags                                # list of (frame_idx, mean_mag)

# ---------------------------------------------------------------- per video
def process_video(paths, fps, dino, raft, device, anchor_sec):
    emb = dino_embed(dino, paths, device)      # [T,768]
    n_anchor = max(1, int(anchor_sec * fps))
    anchor = emb[:n_anchor].mean(0)
    anchor = anchor / (np.linalg.norm(anchor) + 1e-8)
    drift = 1.0 - emb @ anchor                 # [T], raw (0..~2)
    t = np.arange(len(paths)) / fps
    motion = raft_motion(raft, paths, device, stride=1)
    return {
        "anchor": anchor.tolist(),
        "t": t.tolist(),
        "drift_raw": drift.tolist(),
        "motion_idx": [i for i, _ in motion],
        "motion_mag": [m for _, m in motion],
    }

# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--fps", type=float, default=4.0)
    ap.add_argument("--anchor_sec", type=float, default=2.0)
    ap.add_argument("--win_sec", type=float, default=10.0)
    ap.add_argument("--motion_floor", type=float, default=0.5)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    items = json.load(open(args.manifest))
    log(f"[e0] {len(items)} videos, fps={args.fps}, device={args.device}")

    dino = load_dino(args.device)
    raft = load_raft(args.device)
    log("[e0] models loaded (DINOv2-vitb14, RAFT-small)")

    jobtmp = os.environ.get("CLAUDE_JOB_DIR", "/tmp")
    jobtmp = os.path.join(jobtmp, "tmp", "e0frames")
    results = {}
    for it in items:
        key = f"{it['prompt']}::{it['arm']}"
        vp = it["path"]
        if not os.path.exists(vp):
            log(f"[e0] MISSING {vp}"); continue
        with tempfile.TemporaryDirectory(dir=jobtmp if os.path.isdir(jobtmp) else None,
                                         prefix="e0_") as td:
            try:
                paths = extract_frames(vp, args.fps, td)
            except Exception as e:
                log(f"[e0] ffmpeg fail {vp}: {e}"); continue
            if len(paths) < 4:
                log(f"[e0] too few frames {vp}"); continue
            log(f"[e0] {key}: {len(paths)} frames")
            r = process_video(paths, args.fps, dino, raft, args.device, args.anchor_sec)
            r["path"] = vp; r["prompt"] = it["prompt"]; r["arm"] = it["arm"]
            results[key] = r

    # oracle reference: cross-PROMPT anchor distance (different subjects), use stock arm if present
    anchors = {}
    for k, r in results.items():
        anchors.setdefault(r["prompt"], {})[r["arm"]] = np.array(r["anchor"])
    ref_arm_pref = ["stock", "baseline", "mem_off", "mem_on"]
    prompt_anchor = {}
    for p, arms in anchors.items():
        a = None
        for pref in ref_arm_pref:
            if pref in arms: a = arms[pref]; break
        if a is None: a = list(arms.values())[0]
        prompt_anchor[p] = a
    plist = list(prompt_anchor.values())
    dists = []
    for i in range(len(plist)):
        for j in range(i+1, len(plist)):
            dists.append(1.0 - float(plist[i] @ plist[j]))
    oracle_ref = float(np.mean(dists)) if dists else 1.0
    log(f"[e0] oracle cross-subject reference distance = {oracle_ref:.4f} (n_prompts={len(plist)})")

    # summaries
    summary = []
    for k, r in results.items():
        t = np.array(r["t"]); drift = np.array(r["drift_raw"])
        norm_drift = (drift / oracle_ref).tolist()
        r["drift_norm"] = norm_drift
        r["oracle_ref"] = oracle_ref
        # windowed motion
        mi = np.array(r["motion_idx"]) / args.fps
        mm = np.array(r["motion_mag"])
        mean_motion = float(mm.mean()) if len(mm) else 0.0
        # late drift = mean over last third
        late = drift[len(drift)*2//3:]
        summary.append({
            "prompt": r["prompt"], "arm": r["arm"],
            "mean_motion": round(mean_motion, 3),
            "drift_late_raw": round(float(late.mean()), 4),
            "drift_late_norm": round(float(late.mean()/oracle_ref), 4),
            "drift_end_norm": round(float(norm_drift[-1]), 4),
            "frozen": bool(mean_motion < args.motion_floor),
            "n_frames": len(t),
            "dur_s": round(float(t[-1]), 1) if len(t) else 0.0,
        })

    json.dump({"oracle_ref": oracle_ref, "results": results, "summary": summary},
              open(os.path.join(args.out, "drift_results.json"), "w"))
    # print table
    log("\n=== E0 drift + motion summary ===")
    log(f"{'prompt':18s} {'arm':10s} {'motion':>8s} {'drift_late':>11s} {'drift_end':>10s} {'frozen':>7s}")
    for s in sorted(summary, key=lambda x: (x["prompt"], x["arm"])):
        log(f"{s['prompt']:18s} {s['arm']:10s} {s['mean_motion']:8.2f} "
            f"{s['drift_late_norm']:11.3f} {s['drift_end_norm']:10.3f} {str(s['frozen']):>7s}")

    # plot drift curves (arms overlaid per prompt)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        prompts = sorted(set(r["prompt"] for r in results.values()))
        ncol = 3; nrow = math.ceil(len(prompts)/ncol)
        fig, axes = plt.subplots(nrow, ncol, figsize=(5*ncol, 3.2*nrow), squeeze=False)
        for idx, p in enumerate(prompts):
            ax = axes[idx//ncol][idx%ncol]
            for k, r in results.items():
                if r["prompt"] != p: continue
                ax.plot(r["t"], r["drift_norm"], label=r["arm"], lw=1.4)
            ax.set_title(p, fontsize=10); ax.set_xlabel("s"); ax.set_ylabel("norm drift")
            ax.axhline(1.0, color="grey", ls=":", lw=0.8); ax.legend(fontsize=7)
        for idx in range(len(prompts), nrow*ncol): axes[idx//ncol][idx%ncol].axis("off")
        fig.tight_layout()
        fig.savefig(os.path.join(args.out, "drift_curves.png"), dpi=90)
        log(f"[e0] plot -> {os.path.join(args.out,'drift_curves.png')}")
    except Exception as e:
        log(f"[e0] plot skipped: {e}")

    log("[e0] done.")

if __name__ == "__main__":
    main()
