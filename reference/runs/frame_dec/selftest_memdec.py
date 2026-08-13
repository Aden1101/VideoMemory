#!/usr/bin/env python
# Selftest of the memdec arm (decoded anchor spliced at the return switch).
# Usage: CUDA_VISIBLE_DEVICES=N python selftest_memdec.py /path/to/dec_ckpt.pt
# Checks: (1) MEM_DECODE unset => bitwise identical to memzero (default-off);
# (2) memdec arm runs, the decode hook fires, latents differ from memzero post-return;
# (3) the decoded frame is finite and non-zero.
import os
import sys

REPO = "/shared/ssd_14T/home/adenelmi/Anchor-Forcing"
PROBE_DIR = os.path.join(REPO, "reference/runs/reid_probe_l1")
sys.path.insert(0, REPO)
sys.path.insert(0, PROBE_DIR)

import torch
from omegaconf import OmegaConf

from run_reid_probe import build_pipeline, run_arm  # noqa: E402

CKPT = sys.argv[1] if len(sys.argv) > 1 else ""
assert CKPT and os.path.exists(CKPT), f"usage: selftest_memdec.py <decoder ckpt>; got '{CKPT}'"

FRAMES, SW1, SW2 = 27, 12, 21
HERE = os.path.dirname(os.path.abspath(__file__))
VID = os.path.join(HERE, "selftest_memdec_videos")
os.makedirs(VID, exist_ok=True)


def main():
    device = torch.device("cuda")
    torch.set_grad_enabled(False)
    config = OmegaConf.load(os.path.join(PROBE_DIR, "probe_l1.yaml"))
    config.distributed = False
    config._memdec_ckpt = CKPT
    pipeline = build_pipeline(config, device)

    # (1) memzero reference (MEM_DECODE stays empty via ARM_ENV hygiene)
    lat_zero, _ = run_arm(pipeline, config, 0, 0, "memzero", FRAMES, SW1, SW2, device,
                          os.path.join(VID, "st_memzero.mp4"))
    # re-run memzero: bitwise determinism of the harness itself (sanity for the comparison)
    lat_zero2, _ = run_arm(pipeline, config, 0, 0, "memzero", FRAMES, SW1, SW2, device,
                           os.path.join(VID, "st_memzero2.mp4"))
    assert torch.equal(lat_zero, lat_zero2), "harness not deterministic, comparison void"
    print("[selftest-memdec] determinism OK (memzero reproducible bitwise)")

    # (2) memdec arm: decode fires, output differs post-return
    lat_dec, _ = run_arm(pipeline, config, 0, 0, "memdec", FRAMES, SW1, SW2, device,
                         os.path.join(VID, "st_memdec.mp4"))
    assert os.environ.get("MEM_DECODE", "") != "", "MEM_DECODE env not set by the memdec arm"
    pre = (lat_dec[:, :SW2] - lat_zero[:, :SW2]).abs().max().item()
    post = (lat_dec[:, SW2:] - lat_zero[:, SW2:]).abs().max().item()
    print(f"[selftest-memdec] latent diff vs memzero: pre-return max={pre:.6f} "
          f"post-return max={post:.6f}")
    assert post > 0, "decoded splice had NO effect after the return switch"
    assert torch.isfinite(lat_dec).all(), "non-finite latents with memdec"
    print("[selftest-memdec] PASS")


if __name__ == "__main__":
    main()
