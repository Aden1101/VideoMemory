#!/usr/bin/env python
"""Selftest of the MEM_WRITER hook (memory 2.0 phase A, arm memdec20).

  T1 DEFAULT-OFF / trivial overlay: a writer ckpt whose theta_w is copied from the live
      l1@2000-EMA cells (delta 0 by construction) must give a rollout BYTE-identical to the
      same rollout without MEM_WRITER, latents and the three ttt states. This is the
      founding no-op: any hidden side effect of the hook (RNG draw, dtype churn, an extra
      parameter touched) breaks it.
  T2 real overlay: MEM_WRITER=<A2 ckpt> logs a per-block L1 delta > 0 and the resulting
      memory states differ from the stock-writer states (the writer actually drives the
      write chain at inference).
  T3 restore: turning MEM_WRITER back off restores the stock write params bitwise, and the
      rollout is byte-identical to T1 again (no leak into the next arm of the same process:
      run_reid_probe builds one pipeline for all MAG arms).
  T4 decoder: the exported A2 decoder loads through the MEM_DECODE hook loader and decodes
      the live states into a finite (B,1,16,60,104) latent frame.

Usage (GPU pre-check in the same command, house rule):
  CUDA_DEVICE_ORDER=PCI_BUS_ID nvidia-smi --query-gpu=index,memory.free --format=csv && \
  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 \
    python reference/runs/mem2/selftest_memwriter.py
Optional args: --writer <ckpt> --decoder <ckpt> --frames 27 --sw1 12 --sw2 21
"""
import argparse
import gc
import os
import sys

REPO = "/shared/ssd_14T/home/adenelmi/Anchor-Forcing"
PROBE_DIR = os.path.join(REPO, "reference/runs/reid_probe_l1")
for p in (REPO, PROBE_DIR):
    sys.path.insert(0, p)
os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")

import torch  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402

from run_reid_probe import (MEM2_DEC_DEFAULT, MEM2_WRITER_DEFAULT, PAIRS, _OFF,  # noqa: E402
                            build_pipeline, cell_seed)
from pipeline.switch_causal_inference import (_find_causal_model, _memdec_get_decoder,  # noqa: E402
                                              _memdec_pooled_cond, _MEMWRITER_PARAMS)

BLOCKS = (5, 16, 27)
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "selftest_memwriter_out")


def set_env(over):
    """Same env hygiene as run_reid_probe.run_arm: every knob reset, then the arm's own."""
    for k, v in dict(_OFF, **{"MEM_ZERO_ALL": "1", **over}).items():
        os.environ[k] = v


def rollout(pipeline, config, device, frames, sw1, sw2, tag, **env):
    """One A->B->A rollout in the memdec-style regime (MEM_ZERO_ALL=1). Returns
    (latents fp32 cpu, {block: (w0,w1,w2) fp32 cpu})."""
    from utils.misc import set_seed
    set_env(env)
    A, B = PAIRS[0]
    seed = cell_seed(config, 0, 0)
    set_seed(seed)
    noise = torch.randn([1, frames, 16, 60, 104], device=device, dtype=torch.bfloat16)
    print(f"\n[selftest-memwriter] --- rollout '{tag}' (MEM_WRITER='{os.environ.get('MEM_WRITER','')}', "
          f"MEM_DECODE='{os.environ.get('MEM_DECODE','')}') ---", flush=True)
    _video, latents = pipeline.inference(
        noise=noise,
        text_prompts_first=[A],
        text_prompts_second=[B],
        switch_frame_index=sw1,
        text_prompts_third=None,
        second_switch_frame_index=sw2,
        return_latents=True,
        low_memory=True,
    )
    states = {}
    for bi in BLOCKS:
        st = pipeline.ttt_cache[bi].state
        assert st is not None, f"rollout '{tag}': block {bi} never wrote a state (rollout too short?)"
        states[bi] = tuple(t.detach().float().cpu().clone() for t in st)
    lat = latents.float().cpu()
    pipeline.vae.model.clear_cache()
    pipeline.kv_cache1 = None
    pipeline.crossattn_cache = None
    pipeline.ttt_cache = None
    gc.collect()
    torch.cuda.empty_cache()
    return lat, states


def theta_w_now(cm):
    return {(bi, n): cm.blocks[bi].memory.get_parameter(n).detach().cpu().clone()
            for bi in BLOCKS for n in _MEMWRITER_PARAMS}


def state_diff(sa, sb):
    """(max abs diff, all-equal-bitwise) over the three (w0,w1,w2) triples."""
    mx, eq = 0.0, True
    for bi in BLOCKS:
        for a, b in zip(sa[bi], sb[bi]):
            mx = max(mx, float((a - b).abs().max()))
            eq = eq and torch.equal(a, b)
    return mx, eq


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--writer", default=MEM2_WRITER_DEFAULT)
    ap.add_argument("--decoder", default=MEM2_DEC_DEFAULT)
    ap.add_argument("--config", default=os.path.join(PROBE_DIR, "probe_l1.yaml"))
    ap.add_argument("--frames", type=int, default=27)
    ap.add_argument("--sw1", type=int, default=12)
    ap.add_argument("--sw2", type=int, default=21)
    args = ap.parse_args()
    assert os.path.exists(args.writer), f"no writer ckpt {args.writer}"
    assert os.path.exists(args.decoder), f"no decoder ckpt {args.decoder}"
    os.makedirs(OUT_DIR, exist_ok=True)
    os.chdir(REPO)                                   # WanVAEWrapper reads repo-relative paths

    device = torch.device("cuda")
    torch.set_grad_enabled(False)
    config = OmegaConf.load(args.config)
    config.distributed = False
    config._mem2_writer_ckpt, config._mem2_dec_ckpt = args.writer, args.decoder
    pipeline = build_pipeline(config, device)
    cm = _find_causal_model(pipeline.generator.model)
    assert cm is not None, "CausalWanModel not found under the peft wrappers"

    # ---- the trivial writer ckpt: theta_w read back from the live l1@2000-EMA cells --------
    stock = theta_w_now(cm)
    trivial_path = os.path.join(OUT_DIR, "writer_trivial_l1.pt")
    torch.save({"writer": {str(bi): {n: stock[(bi, n)] for n in _MEMWRITER_PARAMS}
                           for bi in BLOCKS},
                "step": "trivial(l1@2000-EMA)"}, trivial_path)
    print(f"[selftest-memwriter] trivial writer ckpt -> {trivial_path}", flush=True)

    # ---- T1: no MEM_WRITER  vs  trivial MEM_WRITER  (must be BYTE-identical) ---------------
    lat_off, st_off = rollout(pipeline, config, device, args.frames, args.sw1, args.sw2, "no-writer")
    assert getattr(cm, "_memwriter_applied", None) is None, "hook fired with MEM_WRITER unset"
    lat_triv, st_triv = rollout(pipeline, config, device, args.frames, args.sw1, args.sw2,
                                "trivial-writer", MEM_WRITER=trivial_path)
    assert getattr(cm, "_memwriter_applied", None) == os.path.abspath(trivial_path), \
        "trivial overlay was not applied"
    mx, eq = state_diff(st_off, st_triv)
    assert torch.equal(lat_off, lat_triv), \
        f"T1 FAIL: latents differ under a delta-0 overlay (max {float((lat_off-lat_triv).abs().max()):.3e})"
    assert eq, f"T1 FAIL: ttt states differ under a delta-0 overlay (max {mx:.3e})"
    print(f"[selftest-memwriter] T1 PASS: trivial overlay is BYTE-IDENTICAL "
          f"(latents equal, states equal, max state diff {mx:.1e})", flush=True)

    # ---- T2: the real A2 writer -----------------------------------------------------------
    lat_a2, st_a2 = rollout(pipeline, config, device, args.frames, args.sw1, args.sw2,
                            "A2-writer", MEM_WRITER=args.writer)
    now = theta_w_now(cm)
    tot = 0.0
    for bi in BLOCKS:
        num = den = 0.0
        for n in _MEMWRITER_PARAMS:
            num += float((now[(bi, n)].float() - stock[(bi, n)].float()).abs().sum())
            den += float(stock[(bi, n)].float().abs().sum())
        tot += num
        print(f"[selftest-memwriter] T2 block {bi}: theta_w L1 delta {num:.4e} "
              f"(rel {num / max(den, 1e-12):.3e})", flush=True)
        assert num > 0, f"T2 FAIL: block {bi} writer identical to l1 (overlay did nothing)"
    mx_a2, eq_a2 = state_diff(st_off, st_a2)
    assert not eq_a2, "T2 FAIL: the A2 writer produced BITWISE-IDENTICAL memory states"
    lat_mx = float((lat_off - lat_a2).abs().max())
    print(f"[selftest-memwriter] T2 PASS: total theta_w L1 {tot:.4e}, states differ "
          f"(max {mx_a2:.3e}); latents differ by max {lat_mx:.3e} "
          f"({'expected ~0 under MEM_ZERO_ALL=1' if lat_mx == 0 else 'nonzero'})", flush=True)
    assert torch.isfinite(lat_a2).all(), "T2 FAIL: non-finite latents with the A2 writer"
    for bi in BLOCKS:
        assert all(torch.isfinite(t).all() for t in st_a2[bi]), \
            f"T2 FAIL: non-finite state on block {bi} with the A2 writer"

    # ---- T3: MEM_WRITER back off => stock params restored, rollout identical to T1 ---------
    lat_back, st_back = rollout(pipeline, config, device, args.frames, args.sw1, args.sw2, "writer-off-again")
    assert getattr(cm, "_memwriter_applied", None) is None, "restore did not clear the marker"
    back = theta_w_now(cm)
    assert all(torch.equal(back[k], stock[k]) for k in stock), \
        "T3 FAIL: stock write params NOT restored bitwise"
    _, eq3 = state_diff(st_off, st_back)
    assert torch.equal(lat_off, lat_back) and eq3, \
        "T3 FAIL: rollout after restore differs from the no-writer rollout"
    print("[selftest-memwriter] T3 PASS: overlay fully reverted (params + rollout bitwise)",
          flush=True)

    # ---- T4: the exported A2 decoder loads through the MEM_DECODE loader -------------------
    dec = _memdec_get_decoder(args.decoder, device)
    assert getattr(dec, "cond_dim", 0) == 4096, f"cond_dim changed: {getattr(dec, 'cond_dim', 0)}"
    cond_dict = pipeline.text_encoder(text_prompts=[PAIRS[0][0]])
    cond = _memdec_pooled_cond(cond_dict).to(device, torch.float32)[None]
    sts = [tuple(t.to(device, torch.float32) for t in st_a2[bi]) for bi in BLOCKS]
    frame = dec(sts, cond=cond)
    assert frame.shape[-3:] == (16, 60, 104) and frame.shape[0] == 1, \
        f"T4 FAIL: decoded frame shape {tuple(frame.shape)}"
    assert torch.isfinite(frame).all(), "T4 FAIL: non-finite decoded frame"
    print(f"[selftest-memwriter] T4 PASS: decoder {os.path.basename(args.decoder)} -> "
          f"{tuple(frame.shape)}, finite, absmean {float(frame.abs().mean()):.4f}, "
          f"std {float(frame.std()):.4f}", flush=True)

    print("\n[selftest-memwriter] ALL PASS (T1 byte-identity, T2 real overlay, T3 restore, "
          "T4 decoder)", flush=True)


if __name__ == "__main__":
    main()
