#!/usr/bin/env python
# Eye-check: rerun 2 memdec cells dumping the DECODED ANCHOR latent, VAE-decode it to PNG.
# Answers: is the anchor clean-A or an A/B fusion? Usage: dump_anchor.py <dec_ckpt> <outdir>
import os, sys
REPO = "/shared/ssd_14T/home/adenelmi/Anchor-Forcing"
sys.path.insert(0, REPO); sys.path.insert(0, os.path.join(REPO, "reference/runs/reid_probe_l1"))
import torch
from omegaconf import OmegaConf
import run_reid_probe
from run_reid_probe import build_pipeline, run_arm
from torchvision.utils import save_image

CKPT, OUT = sys.argv[1], sys.argv[2]
os.makedirs(OUT, exist_ok=True)
device = torch.device("cuda")
torch.set_grad_enabled(False)
cfg = OmegaConf.load(os.path.join(REPO, "reference/runs/reid_probe_l1/probe_l1.yaml"))
cfg.distributed = False
cfg._memdec_ckpt = CKPT
# DISCRIMINANT: B = A (no real switch) -> the state at the return contains ONLY
# horse content; if the decoded anchor is STILL brown, the failure is category-regression
# (instance detail never stored), not B-contamination.
run_reid_probe.PAIRS[4] = (run_reid_probe.PAIRS[4][0], run_reid_probe.PAIRS[4][0])
pipe = build_pipeline(cfg, device)
os.environ["MEM_DECODE_DUMP"] = os.path.join(OUT, "anchor")
for pi in (4,):
    os.environ["MEM_DECODE_DUMP"] = os.path.join(OUT, f"anchor_single_p{pi}")
    run_arm(pipe, cfg, pi, 0, "memdec", 240, 81, 162, device, os.path.join(OUT, f"single_p{pi}_memdec.mp4"))
os.environ.pop("MEM_DECODE_DUMP", None)
# VAE-decode the dumped anchors
vae = pipe.vae
for pi in (4,):
    lat = torch.load(os.path.join(OUT, f"anchor_single_p{pi}.f162.pt"), map_location="cpu").to(device, torch.bfloat16)
    px = vae.decode_to_pixel(lat, use_cache=False)          # (1, T, 3, H, W) in [-1,1]
    img = (px[0, 0].float() * 0.5 + 0.5).clamp(0, 1)
    save_image(img, os.path.join(OUT, f"anchor_single_p{pi}.png"))
    print(f"[dump] anchor_single_p{pi}.png saved")
print("[dump] DONE")
