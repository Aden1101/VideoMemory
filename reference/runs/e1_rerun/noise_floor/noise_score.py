#!/usr/bin/env python
"""Noise-floor measurement: is the mem-DC on-vs-off effect ABOVE the seed-to-seed noise?
Defines do-no-harm (error bar over seeds) and active (effect vs seed floor) for the first time."""
import sys, os, glob, tempfile
import numpy as np
sys.path.insert(0, "reference/runs/e0_reid")
from drift_metric import load_dino, dino_embed, extract_frames

ND = "reference/runs/e1_rerun/noise_floor"
EVAL = "reference/runs/e1_rerun/eval1000/videos"
SEEDS = [42, 43, 44, 45]
LAB = ["jog", "cycle", "ballet", "boy", "skate", "elder", "surf", "martial"]
dev = "cuda:0"
dino = load_dino(dev)

def vpath(arm, s, idx):                       # arm in {off,dc}
    if s == 42:
        return os.path.join(EVAL, "memoff" if arm == "off" else "memdc", f"{idx}-0.mp4")
    return os.path.join(ND, "videos", f"mem{arm}_s{s}", f"{idx}-0.mp4")

emb = {}
jt = os.path.join(os.environ.get("CLAUDE_JOB_DIR", "/tmp"), "tmp", "nf"); os.makedirs(jt, exist_ok=True)
for arm in ["off", "dc"]:
    for s in SEEDS:
        for idx in range(8):
            p = vpath(arm, s, idx)
            if not os.path.exists(p): continue
            with tempfile.TemporaryDirectory(dir=jt) as td:
                fr = extract_frames(p, 4, td)
                if len(fr) >= 4:
                    emb[(arm, s, idx)] = dino_embed(dino, fr, dev)   # [T,768] L2-normalized

def cdist(a, b):                              # mean per-frame cosine distance
    n = min(len(a), len(b))
    return float((1 - (a[:n] * b[:n]).sum(1)).mean())

print(f"{'prompt':8s} {'seed-floor':>10s} {'memDC-eff':>10s} {'ratio':>6s}  {'verdict':>14s}")
floor, effect = [], []
for idx in range(8):
    offs = [emb[("off", s, idx)] for s in SEEDS if ("off", s, idx) in emb]
    fp = [cdist(offs[i], offs[j]) for i in range(len(offs)) for j in range(i + 1, len(offs))]
    ep = [cdist(emb[("dc", s, idx)], emb[("off", s, idx)]) for s in SEEDS
          if ("dc", s, idx) in emb and ("off", s, idx) in emb]
    fi = float(np.mean(fp)) if fp else float("nan")
    ei = float(np.mean(ep)) if ep else float("nan")
    floor.append(fi); effect.append(ei)
    r = ei / fi if fi else float("nan")
    v = "ABOVE (active)" if r > 1.5 else ("~floor (hollow)" if r > 0.7 else "below")
    print(f"{LAB[idx]:8s} {fi:10.4f} {ei:10.4f} {r:6.2f}  {v:>14s}")

mf, me = np.nanmean(floor), np.nanmean(effect)
print(f"\nMEAN  seed-floor={mf:.4f}  memDC-effect={me:.4f}  ratio={me/mf:.2f}  (#off-seeds/prompt={len([s for s in SEEDS if ('off',s,0) in emb])})")
print("VERDICT: mem-DC effect is", "ABOVE the seed noise floor (REAL active, hollow=seed-equiv ruled out)"
      if me > 1.5 * mf else "WITHIN the seed noise floor (indistinguishable from a re-seed => INERT/hollow)")
