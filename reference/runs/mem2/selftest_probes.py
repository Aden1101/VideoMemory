#!/usr/bin/env python
# ============================================================================================
# SELFTESTS of the ARBITER BATTERY hooks (reference/memory2_probes_registry.md) -- MANDATORY
# before the campaign. Everything runs on ONE short A->B->A rollout (36 frames, switch 12,
# return 27) so the whole file takes a few minutes on one GPU.
#
#   (a) oracle1twin with a seed offset of 0 == oracle1 BITWISE
#       (the twin path re-splices the very frame oracle1 splices, read back from disk).
#   (b) oracle1blur with sigma = 0 ~= oracle1: the residual IS the VAE round-trip error, and
#       it is MEASURED here (latent-space rel-L2 on the anchor frame + rollout divergence),
#       so the sigma=24 arm can be read against a known floor.
#   (c) oracle1thru decodes the anchor state AT THE EVICTION of the anchor chunk (age 1 write,
#       logged) and the return splice USES that cached frame (no decode at the return).
#   (d) NO ARM LEAKS ENV ONTO THE NEXT: a memzero run AFTER the writer/decoder/blur/twin arms
#       is bitwise identical to the memzero run made BEFORE any of them (the MEM_WRITER lesson).
#
# Usage (GPU pre-check in the SAME command, house rule):
#   CUDA_DEVICE_ORDER=PCI_BUS_ID nvidia-smi --query-gpu=index,memory.free --format=csv && \
#   CUDA_VISIBLE_DEVICES=1 python reference/runs/mem2/selftest_probes.py
# ============================================================================================
import argparse
import os
import shutil
import sys

REPO = "/shared/ssd_14T/home/adenelmi/Anchor-Forcing"
PROBE_DIR = os.path.join(REPO, "reference/runs/reid_probe_l1")
HERE = os.path.dirname(os.path.abspath(__file__))
for p in (REPO, PROBE_DIR, HERE):
    sys.path.insert(0, p)
os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")

import torch  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402

from run_reid_probe import (ANCHOR_DIR, MEM2_DEC_DEFAULT, MEM2_WRITER_DEFAULT,  # noqa: E402
                            build_pipeline, run_arm)

OUT = os.path.join(HERE, "selftest_probes_out")


def diff(a, b):
    d = (a - b).abs()
    return float(d.max()), float(d.mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=36)
    ap.add_argument("--switch1", type=int, default=12)
    ap.add_argument("--switch2", type=int, default=27)
    ap.add_argument("--config", default=os.path.join(PROBE_DIR, "probe_l1.yaml"))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--keep", action="store_true", help="keep the scratch dir")
    args = ap.parse_args()

    os.chdir(REPO)
    if os.path.isdir(OUT) and not args.keep:
        shutil.rmtree(OUT)
    os.makedirs(os.path.join(OUT, "anchors"), exist_ok=True)
    device = torch.device(args.device)
    torch.set_grad_enabled(False)
    config = OmegaConf.load(args.config)
    config.distributed = False
    config._memdec_ckpt = ""
    config._mem2_writer_ckpt = MEM2_WRITER_DEFAULT
    config._mem2_dec_ckpt = MEM2_DEC_DEFAULT
    config._anchor_dir = os.path.join(OUT, "anchors")
    config._n_seeds = 3
    config._twin_offset = 1
    config._blur_sigma = 24.0
    F, S1, S2 = args.frames, args.switch1, args.switch2
    Fb = int(config.num_frame_per_block)
    law = int(config.model_kwargs.local_attn_size)
    wc = max(1, -(-law // Fb))
    anchor_chunk = (S1 - 1) // Fb
    print(f"[selftest] frames {F} switch {S1} return {S2} | num_frame_per_block {Fb}, "
          f"local_attn {law} -> window_chunks {wc}; anchor frame {S1 - 1} (chunk {anchor_chunk}) "
          f"is evicted/written at chunk {anchor_chunk + wc} (frame {(anchor_chunk + wc) * Fb}), "
          f"return at chunk {S2 // Fb}", flush=True)
    assert (anchor_chunk + wc) * Fb < S2, \
        "this schedule evicts the anchor chunk AFTER the return: pick a larger --switch2"

    pipeline = build_pipeline(config, device)

    def go(arm, tag=None):
        mp4 = os.path.join(OUT, f"{tag or arm}.mp4")
        lat, _ = run_arm(pipeline, config, 0, 0, arm, F, S1, S2, device, mp4)
        return lat

    fails = []

    # ---------- reference rollouts -----------------------------------------------------------
    lat_mz0 = go("memzero")                    # dumps the original-A anchor of (pair0, s0)
    apath = os.path.join(OUT, "anchors", "pair0_s0.pt")
    assert os.path.exists(apath), "ORACLE_ANCHOR_DUMP produced nothing"
    lat_o1 = go("oracle1")                     # re-checks the dumped anchor BITWISE (hook assert)
    d_max, d_mean = diff(lat_o1, lat_mz0)
    print(f"[selftest] oracle1 vs memzero: max {d_max:.4e} mean {d_mean:.4e} (must be > 0)")
    if d_max <= 0:
        fails.append("oracle1 did not change the rollout")

    # ---------- (a) twin with offset 0 == oracle1 bitwise --------------------------------------
    config._twin_offset = 0
    lat_tw0 = go("oracle1twin", tag="oracle1twin_off0")
    d_max, d_mean = diff(lat_tw0, lat_o1)
    ok = (d_max == 0.0)
    print(f"[selftest] (a) oracle1twin(offset 0) vs oracle1: max|diff| {d_max:.3e} -> "
          f"{'PASS (bitwise)' if ok else 'FAIL'}")
    if not ok:
        fails.append("(a) twin offset 0 is not bitwise oracle1")
    config._twin_offset = 1

    # ---------- (b) blur sigma = 0 ~= oracle1 (VAE round-trip tolerance) -----------------------
    anchor = torch.load(apath, map_location="cpu", weights_only=False)["anchor"]
    rt = pipeline._blur_latent(anchor.to(device), 0.0).cpu()
    rel = float((rt.float() - anchor.float()).norm() / anchor.float().norm())
    mx, mn = diff(rt.float(), anchor.float())
    print(f"[selftest] (b) VAE round trip on the anchor LATENT: rel-L2 {rel:.4f}, "
          f"max|diff| {mx:.4f}, mean|diff| {mn:.4f}")
    config._blur_sigma = 0.0
    lat_b0 = go("oracle1blur", tag="oracle1blur_s0")
    d0_max, d0_mean = diff(lat_b0, lat_o1)
    config._blur_sigma = 24.0
    lat_b24 = go("oracle1blur", tag="oracle1blur_s24")
    d24_max, d24_mean = diff(lat_b24, lat_o1)
    ok = d24_mean > d0_mean
    print(f"[selftest] (b) rollout divergence vs oracle1: sigma=0 mean {d0_mean:.4e} "
          f"(max {d0_max:.4e}) | sigma=24 mean {d24_mean:.4e} (max {d24_max:.4e}) -> "
          f"{'PASS (blur >> round-trip floor)' if ok else 'FAIL'}")
    if not ok:
        fails.append("(b) sigma=24 does not diverge more than the sigma=0 round trip")

    # ---------- (c) thru: decode at eviction, splice at return ---------------------------------
    lat_thru = go("oracle1thru")
    info = getattr(pipeline, "_thru_info", None)
    print(f"[selftest] (c) thru info: {info}")
    ok = (info is not None and info["anchor_frame"] == S1 - 1
          and info["evict_chunk"] == anchor_chunk + wc and info["window_chunks"] == wc)
    age = None if info is None else (S2 // Fb - info["evict_chunk"])
    print(f"[selftest] (c) anchor decoded at chunk {None if info is None else info['evict_chunk']} "
          f"= {age} writes before the return -> {'PASS' if ok else 'FAIL'}")
    if not ok:
        fails.append("(c) thru did not decode at the eviction of the anchor chunk")
    d_max, _ = diff(lat_thru, lat_mz0)
    if d_max <= 0:
        fails.append("(c) thru rollout identical to memzero")

    # ---------- memon20 (writer overlay + live injection) --------------------------------------
    lat_m20 = go("memon20")
    d_max, d_mean = diff(lat_m20, lat_mz0)
    print(f"[selftest] memon20 vs memzero: max {d_max:.4e} mean {d_mean:.4e} (must be > 0)")
    if d_max <= 0:
        fails.append("memon20 identical to memzero -> injection or writer not live")

    # ---------- (d) no env / writer leak onto the next arm --------------------------------------
    lat_mz1 = go("memzero", tag="memzero_after")
    d_max, d_mean = diff(lat_mz1, lat_mz0)
    ok = (d_max == 0.0)
    print(f"[selftest] (d) memzero AFTER every probe arm vs the first memzero: max|diff| "
          f"{d_max:.3e} -> {'PASS (no leak)' if ok else 'FAIL'}")
    if not ok:
        fails.append("(d) an arm leaked state/env onto the next")
    from pipeline.switch_causal_inference import _find_causal_model
    cm = _find_causal_model(pipeline.generator.model)
    applied = getattr(cm, "_memwriter_applied", None)
    print(f"[selftest] (d) MEM_WRITER overlay after the last arm: {applied} (must be None)")
    if applied is not None:
        fails.append("(d) the MEM_WRITER overlay was not restored")

    print("\n[selftest] " + ("ALL PASS" if not fails else "FAILURES: " + "; ".join(fails)),
          flush=True)
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
