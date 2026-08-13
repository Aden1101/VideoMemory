#!/usr/bin/env python
# Eye-check memoire 2.0: rerun the HORSE cell (pair4, seed0) on arms memdec12 / memdec20 /
# memdec20d, dump the decoded anchor latent, VAE-decode to PNG. Answers Aden's discriminant
# question: is the 2.0 anchor WHITE (instance stored) or still brown (category regression)?
import os
import sys

REPO = "/shared/ssd_14T/home/adenelmi/Anchor-Forcing"
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "reference/runs/reid_probe_l1"))
import torch
from omegaconf import OmegaConf
from torchvision.utils import save_image

from run_reid_probe import MEM2_DEC_DEFAULT, MEM2_WRITER_DEFAULT, build_pipeline, run_arm

V12_DEC = os.path.join(REPO, "reference/runs/frame_dec/frame_dec_v12/ckpt/dec_step4000.pt")
OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(REPO, "reference/runs/mem2/eyepack20")
PAIRS = [int(x) for x in (sys.argv[2].split(",") if len(sys.argv) > 2 else ["4"])]
SEED = 0
os.makedirs(OUT, exist_ok=True)
device = torch.device("cuda")
torch.set_grad_enabled(False)
cfg = OmegaConf.load(os.path.join(REPO, "reference/runs/reid_probe_l1/probe_l1.yaml"))
cfg.distributed = False
cfg._memdec_ckpt = V12_DEC
cfg._mem2_writer_ckpt = MEM2_WRITER_DEFAULT
cfg._mem2_dec_ckpt = MEM2_DEC_DEFAULT
pipe = build_pipeline(cfg, device)
for PAIR in PAIRS:
    for arm in ("memdec12", "memdec20", "memdec20d"):
        os.environ["MEM_DECODE_DUMP"] = os.path.join(OUT, f"anchor_p{PAIR}_{arm}")
        run_arm(pipe, cfg, PAIR, SEED, arm, 240, 81, 162, device,
                os.path.join(OUT, f"p{PAIR}_{arm}.mp4"))
os.environ.pop("MEM_DECODE_DUMP", None)
vae = pipe.vae
for PAIR in PAIRS:
    for arm in ("memdec12", "memdec20", "memdec20d"):
        lat = torch.load(os.path.join(OUT, f"anchor_p{PAIR}_{arm}.f162.pt"),
                         map_location="cpu").to(device, torch.bfloat16)
        px = vae.decode_to_pixel(lat, use_cache=False)
        img = (px[0, 0].float() * 0.5 + 0.5).clamp(0, 1)
        save_image(img, os.path.join(OUT, f"anchor_p{PAIR}_{arm}.png"))
        print(f"[dump20] anchor_p{PAIR}_{arm}.png saved")
print("[dump20] DONE")
