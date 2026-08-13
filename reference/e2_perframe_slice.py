#!/usr/bin/env python
"""Unconfounded test: per-frame DINO subject-consistency PROFILE (early->late) on the EXISTING 6s videos.
No video modification, no discontinuity. Replicates vbench/subject_consistency.py exactly (dino_vitb16,
cur_sim = (cos_to_prev + cos_to_first)/2), but reports the mean in N bins across each video instead of one
average. If the mem-vs-base gap is concentrated in the EARLY bins -> damage is front-loaded (cold-start).
If the gap is uniform across bins -> the memory read is harmful throughout."""
import os, glob, sys, statistics
import torch, torch.nn.functional as F
sys.path.insert(0, "/shared/ssd_14T/home/adenelmi/LongLive/vbench_repo")
from vbench.utils import load_video, dino_transform

DEV = "cuda"
NB = 4
OUT = "/shared/ssd_14T_project/adenelmi/e2_write3L_eval"
model = torch.hub.load('facebookresearch/dino:main', 'dino_vitb16').to(DEV).eval()
tf = dino_transform(224)


def perframe(video_path):
    imgs = tf(load_video(video_path))
    sims = []
    first = prev = None
    for i in range(len(imgs)):
        with torch.no_grad():
            f = F.normalize(model(imgs[i].unsqueeze(0).to(DEV)), dim=-1, p=2)
        if i == 0:
            first = f
        else:
            sp = max(0.0, F.cosine_similarity(prev, f).item())
            sf = max(0.0, F.cosine_similarity(first, f).item())
            sims.append((sp + sf) / 2)
        prev = f
    return sims


def binmean(sims, nbins=NB):
    n = len(sims)
    return [round(sum(sims[b*n//nbins:(b+1)*n//nbins]) / max(1, (b+1)*n//nbins - b*n//nbins), 4) for b in range(nbins)]


for tag, d in [("base", "base_H24"), ("mem", "mem_H24"), ("mem_coldzero", "mem_H24_coldzero")]:
    vids = sorted(glob.glob(f"{OUT}/{d}/*.mp4"))
    allb = [binmean(perframe(v)) for v in vids]
    avg = [round(statistics.mean(x[b] for x in allb), 4) for b in range(NB)]
    overall = round(statistics.mean(sum(x, []) if False else [v for x in allb for v in x]), 4)
    print(f"{tag:14s} bins(early->late)={avg}  overall={overall}", flush=True)
print("DONE", flush=True)
