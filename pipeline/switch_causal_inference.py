# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES
#
# Licensed under the Apache License, Version 2.0 (the "License").
# You may not use this file except in compliance with the License.
# To view a copy of this license, visit http://www.apache.org/licenses/LICENSE-2.0
#
# No warranties are given. The work is provided "AS IS", without warranty of any kind, express or implied.
#
# SPDX-License-Identifier: Apache-2.0
import json
import math
import os
from typing import List, Optional
import torch

from utils.wan_wrapper import WanDiffusionWrapper, WanTextEncoder, WanVAEWrapper
from utils.memory import gpu, get_cuda_free_memory_gb, move_model_to_device_with_memory_preservation
from pipeline.causal_inference import CausalInferencePipeline  # reuse init utilities
from wan.modules.video_memory import TTTBlockCache
import torch.distributed as dist
from utils.debug_option import DEBUG


# ---------------------------------------------------------------------------------------------
# MEM_DECODE (eval arm "memdec", design v1 plan_final_sink_virtuel.md): at the return of A in
# an A->B->A rollout, the spliced anchor frame is decoded from the live fast-weight memory
# states by the offline-trained AnchorFrameDecoder, instead of the oracle's real frame.
# Env-gated, default off (MEM_DECODE unset/empty = byte-identical).
_MEMDEC_CACHE = {}   # (abs ckpt path, device str) -> loaded AnchorFrameDecoder, once per process


def _memdec_pooled_cond(cond_dict):
    """Verbatim logic copy of reference/runs/frame_dec/collect_pairs.py::pooled_cond_emb --
    must stay identical, fp16 cpu round-trip included: the conditioned decoder was trained on
    'cond_emb' produced by exactly this pooling. WanTextEncoder zeroes the padding positions:
    a plain mean over all 512 rows would scale with prompt length, so we average over the
    non-zero rows only (sum / count of rows with nonzero norm)."""
    emb = cond_dict["prompt_embeds"]
    assert emb.dim() == 3 and emb.shape[0] == 1, f"prompt_embeds shape {tuple(emb.shape)}"
    nz = emb[0].float().norm(dim=-1) > 0                      # (512,) non-padding rows
    assert nz.any(), "prompt_embeds entirely zero?"
    pooled = (emb[0][nz].mean(dim=0)).detach().to("cpu", torch.float16)
    assert pooled.shape == (4096,), f"pooled cond_emb shape {tuple(pooled.shape)}"   # umt5-xxl
    assert torch.isfinite(pooled).all(), "non-finite pooled cond_emb"
    return pooled


def _memdec_get_decoder(ckpt_path, device):
    """Lazy-load (once per process+device) the offline-trained AnchorFrameDecoder from a
    train_decoder.py checkpoint ({'decoder': state_dict, 'step': ...} or a bare state_dict).
    cond_dim/cond_hidden are detected from the state_dict keys: 'cond_in.weight' of shape
    (cond_hidden, cond_dim) exists only on v1.1 conditioned decoders; absent = v1.0
    unconditioned (cond_dim=0 creates no cond modules, state_dict matches strictly).
    Kept in fp32 on the device (training kept fp32 params; one decode per video, speed moot).
    RNG hygiene: AnchorFrameDecoder.__init__ draws from the global torch RNG (queries/pos/
    linear inits) and this lazy load runs mid-ROLLOUT at the return switch, without the
    fork_rng, every post-return randn would desync vs the paired memzero arm."""
    key = (os.path.abspath(ckpt_path), str(device))
    if key in _MEMDEC_CACHE:
        return _MEMDEC_CACHE[key]
    from wan.modules.video_memory import AnchorFrameDecoder
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = ck["decoder"] if isinstance(ck, dict) and "decoder" in ck else ck
    assert isinstance(sd, dict) and "head.weight" in sd, \
        f"MEM_DECODE ckpt {ckpt_path} does not look like an AnchorFrameDecoder state_dict " \
        f"(keys: {list(sd)[:5]})"
    if "cond_in.weight" in sd:
        cond_hidden, cond_dim = (int(x) for x in sd["cond_in.weight"].shape)
    else:
        cond_dim, cond_hidden = 0, 512
    _devs = [device] if device.type == "cuda" else []
    with torch.random.fork_rng(devices=_devs):
        decoder = AnchorFrameDecoder(cond_dim=cond_dim, cond_hidden=cond_hidden)
        decoder.load_state_dict(sd, strict=True)
        decoder.eval().to(device=device, dtype=torch.float32)
    for p in decoder.parameters():
        p.requires_grad_(False)
    _MEMDEC_CACHE[key] = decoder
    step = ck.get("step", "?") if isinstance(ck, dict) else "?"
    print(f"[MEM_DECODE] AnchorFrameDecoder loaded from {ckpt_path} (step {step}, "
          f"cond_dim={cond_dim}, cond_hidden={cond_hidden}, {decoder.num_params():,} params, "
          f"fp32 on {device})", flush=True)
    return decoder


# ---------------------------------------------------------------------------------------------
# MEM_WRITER (eval arm "memdec20", memory 2.0 phase A): the write parameters of the memory cells
# (blocks 5/16/27) are replaced by the ones trained offline by write-replay
# (reference/runs/mem2/train_writer.py). Strictly symmetric to MEM_DECODE above: env-gated,
# default off, MEM_WRITER unset/empty leaves the checkpoint's own theta_l1 untouched, so every
# existing arm is byte-identical. The overlay changes nothing else: the rollout still runs under
# MEM_ZERO_ALL=1 (silent memory) and only the writes that build the state use the new weights.
#
# Writer ckpt format (train_writer.Trainer.save): {"writer": {"5": {"qkv.weight": t, ...},
# "16": {...}, "27": {...}}, "decoder": state_dict, "step": int, ...}. Only the tensors under
# "writer" are read here; the co-trained decoder travels through MEM_DECODE, separately.
_MEMWRITER_CACHE = {}    # abs ckpt path -> ({block int -> {param name -> cpu tensor}}, step)
# == train_writer.WRITE_PARAMS: theta_w and nothing else (the read path stays stock).
_MEMWRITER_PARAMS = ("qkv.weight", "qkv.bias", "lrs.weight", "lrs.bias",
                     "k_norm.weight", "decay_logit")


def _find_causal_model(m):
    """Walk peft/wrapper layers down to the raw CausalWanModel (verbatim idiom of
    reference/runs/frame_dec/collect_pairs.py::_find_causal_model). Type check, not hasattr:
    peft wrappers delegate attribute reads but setattr lands on the wrapper, so the backup
    bookkeeping must be attached to the real model."""
    seen = 0
    while m is not None and seen < 8:
        if type(m).__name__ == "CausalWanModel":
            return m
        m = getattr(m, "model", None) or getattr(m, "base_model", None)
        seen += 1
    return None


def _memwriter_load(ckpt_path):
    """Lazy-load (once per process) the writer tensors of a train_writer.py checkpoint."""
    key = os.path.abspath(ckpt_path)
    if key in _MEMWRITER_CACHE:
        return _MEMWRITER_CACHE[key]
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    assert isinstance(ck, dict) and "writer" in ck, \
        f"MEM_WRITER ckpt {ckpt_path} has no 'writer' entry (keys: {list(ck)[:6]})"
    writer = {}
    for bi, params in ck["writer"].items():
        assert isinstance(params, dict) and params, f"MEM_WRITER: empty writer block {bi}"
        unknown = set(params) - set(_MEMWRITER_PARAMS)
        assert not unknown, \
            f"MEM_WRITER: block {bi} carries non-write parameters {sorted(unknown)}, refusing " \
            f"to overlay anything outside theta_w {list(_MEMWRITER_PARAMS)}"
        writer[int(bi)] = {n: t.detach().clone() for n, t in params.items()}
    assert writer, f"MEM_WRITER ckpt {ckpt_path}: no blocks in 'writer'"
    step = ck.get("step", "?")
    _MEMWRITER_CACHE[key] = (writer, step)
    del ck
    return writer, step


def _memwriter_restore(cm):
    """Put the stock (checkpoint) write parameters back: an overlay must never leak into the
    next arm of the same process (run_reid_probe builds one pipeline for every arm)."""
    backup = getattr(cm, "_memwriter_backup", None)
    if not backup:
        return
    with torch.no_grad():
        for (bi, name), t in backup.items():
            cm.blocks[bi].memory.get_parameter(name).data.copy_(t)
    cm._memwriter_backup = None
    prev = getattr(cm, "_memwriter_applied", None)
    cm._memwriter_applied = None
    print(f"[MEM_WRITER] stock write parameters RESTORED (was {prev})", flush=True)


def _memwriter_sync(model):
    """Called once per inference() call, before any write happens. MEM_WRITER empty/unset =>
    no-op (or a restore, if a previous arm of this process had overlaid a writer)."""
    path = os.environ.get("MEM_WRITER", "").strip()
    cm = _find_causal_model(model)
    applied = getattr(cm, "_memwriter_applied", None) if cm is not None else None
    if not path:
        if applied is not None:
            _memwriter_restore(cm)
        return
    assert cm is not None, "MEM_WRITER set but no CausalWanModel found under the wrappers"
    if applied == os.path.abspath(path):
        print(f"[MEM_WRITER] already applied ({path}), cells unchanged", flush=True)
        return
    if applied is not None:                    # switching writers: always measure vs the stock
        _memwriter_restore(cm)
    writer, step = _memwriter_load(path)
    backup, report = {}, []
    with torch.no_grad():
        for bi in sorted(writer):
            cell = getattr(cm.blocks[bi], "memory", None)
            assert cell is not None, \
                f"MEM_WRITER: block {bi} has no memory cell (memory_layers mismatch with the ckpt)"
            have = dict(cell.named_parameters())
            num = den = 0.0
            for name, t in writer[bi].items():
                assert name in have, f"MEM_WRITER: block {bi} memory cell has no parameter {name}"
                p = have[name]
                assert tuple(p.shape) == tuple(t.shape), \
                    f"MEM_WRITER: block {bi}.{name} shape {tuple(t.shape)} != {tuple(p.shape)}"
                new = t.to(device=p.device, dtype=p.dtype)
                backup[(bi, name)] = p.detach().clone()
                num += float((new.float() - p.detach().float()).abs().sum())
                den += float(p.detach().float().abs().sum())
                p.data.copy_(new)
            report.append(f"b{bi} dL1={num:.4e} (rel {num / max(den, 1e-12):.3e})")
    cm._memwriter_backup = backup
    cm._memwriter_applied = os.path.abspath(path)
    print(f"[MEM_WRITER] writer overlay from {path} (step {step}) on blocks {sorted(writer)}, "
          f"{len(backup)} tensors | " + " | ".join(report), flush=True)


class SwitchCausalInferencePipeline(CausalInferencePipeline):

    def __init__(
        self,
        args,
        device,
        *,
        generator: WanDiffusionWrapper | None = None,
        text_encoder: WanTextEncoder | None = None,
        vae: WanVAEWrapper | None = None,
    ):
        super().__init__(args, device, generator=generator, text_encoder=text_encoder, vae=vae)
        self.global_sink = getattr(args, "global_sink", False)

    # Internal helpers
    def _snapshot_ttt(self):
        """MEM_EPISODIC probe: freeze the full memory state (fast weights + eviction fifo + injection ema)
        as it stands right now. Entries are replaced, never mutated in place, so shallow copies of
        the containers suffice (same invariant as the trainer/twin snapshots)."""
        return [
            {
                "state": None if c.state is None else tuple(c.state),
                "fifo": list(getattr(c, "fifo", []) or []),
                "ema": dict(getattr(c, "ema", {}) or {}),
            }
            for c in self.ttt_cache
        ]

    def _restore_ttt(self, snap):
        for c, s in zip(self.ttt_cache, snap):
            c.state = s["state"]
            c.fifo = list(s["fifo"])
            c.ema = dict(s["ema"])

    # ---- memory-2.0 arbiter probes (2026-07-28): all env-gated, default off ------------------
    def _memdec_frame(self, ckpt_path, cond_dict, out_dtype, out_device):
        """Decode one anchor latent frame from the live memory states with the offline-trained
        AnchorFrameDecoder. Factored verbatim out of the MEM_DECODE block below so the
        THRU probe (decode at eviction) runs the exact same code path as the decode at the
        return switch, same op order, same dtypes, same fork_rng-protected lazy load."""
        _dev = next(self.generator.parameters()).device
        _dec = _memdec_get_decoder(ckpt_path, _dev)
        for _bi in (5, 16, 27):
            assert self.ttt_cache[_bi].state is not None, \
                f"MEM_DECODE: no live memory state on block {_bi}"
        _sts = [tuple(t.detach().to(_dev, torch.float32)
                      for t in self.ttt_cache[_bi].state) for _bi in (5, 16, 27)]
        _cond = None
        if getattr(_dec, "cond_dim", 0):
            _cond = _memdec_pooled_cond(cond_dict).to(_dev, torch.float32)[None]
        with torch.no_grad():
            _frame = _dec(_sts, cond=_cond)
        return _frame[:, 0:1].to(dtype=out_dtype, device=out_device), _cond is not None

    def _thru_maybe_decode(self, current_start_frame, cond_third, out_dtype, out_device):
        """MEM_DECODE_THRU (arm oracle1thru): decode the memory state at the moment the chunk
        carrying the anchor frame is evicted (i.e. the write that stores it has just happened,
        write age = 1, the decoder's best regime) and CACHE the frame; the return switch then
        splices that frame instead of decoding a ~20-writes-old state. Isolates 'the write+read
        channel at age 1' from 'the state's forgetting between the write and the revisit'.
        Eviction arithmetic = causal_model.py's FIFO rule: chunk c is written window_chunks =
        ceil(local_attn_size / num_frame_per_block) chunks after it was produced."""
        if not int(os.environ.get("MEM_DECODE_THRU", "0") or 0):
            return
        if self._thru_frame is not None or self._first_switch_frame is None:
            return
        _F = self.num_frame_per_block
        assert isinstance(self.local_attn_size, int) and self.local_attn_size > 0, \
            "MEM_DECODE_THRU needs local attention (the eviction FIFO)"
        _wc = max(1, math.ceil(self.local_attn_size / _F))
        _anchor_frame = self._first_switch_frame - 1
        _anchor_chunk = _anchor_frame // _F
        _cur_chunk = current_start_frame // _F
        if _cur_chunk != _anchor_chunk + _wc:
            return
        _ck = os.environ.get("MEM_DECODE", "").strip()
        assert _ck, "MEM_DECODE_THRU=1 requires MEM_DECODE=<AnchorFrameDecoder ckpt>"
        self._thru_frame, _cond_on = self._memdec_frame(_ck, cond_third, out_dtype, out_device)
        self._thru_info = {"anchor_frame": int(_anchor_frame), "anchor_chunk": int(_anchor_chunk),
                           "evict_chunk": int(_cur_chunk), "window_chunks": int(_wc)}
        print(f"[MEM_DECODE_THRU] anchor chunk {_anchor_chunk} (frame {_anchor_frame}) EVICTED "
              f"and WRITTEN at chunk {_cur_chunk} (window_chunks {_wc}) -> state decoded NOW "
              f"(write age 1), frame cached for the return splice; "
              f"cond={'ON' if _cond_on else 'off'}", flush=True)

    def _blur_latent(self, lat, sigma):
        """ORACLE_BLUR_SIGMA probe: send a real latent frame through VAE decode -> gaussian blur
        of `sigma` PIXELS -> VAE re-encode, i.e. degrade the true anchor down to the spatial
        fidelity our decoders reach (phase-B dose-response: the decoded anchor sits beyond
        sigma=32 px in DINO space). sigma=0 = pure VAE round trip = the tolerance floor of the
        probe (selftest b measures it)."""
        import torchvision.transforms.functional as TVF
        _dev = next(self.generator.parameters()).device
        assert lat.shape[1] == 1, \
            f"ORACLE_BLUR_SIGMA supports ONE latent frame (VAE temporal compression), got {lat.shape[1]}"
        px = self.vae.decode_to_pixel(lat.to(device=_dev, dtype=torch.bfloat16),
                                      use_cache=False)          # (B,1,3,H,W) in [-1,1]
        if sigma > 0:
            k = int(6 * sigma) | 1                              # odd kernel ~ +-3 sigma
            px = TVF.gaussian_blur(px[:, 0], kernel_size=[k, k], sigma=[sigma, sigma])[:, None]
        out = self.vae.encode_to_latent(px.permute(0, 2, 1, 3, 4).to(torch.bfloat16))
        self.vae.model.clear_cache()
        print(f"[ORACLE_BLUR] anchor frame VAE round trip with sigma={sigma} px "
              f"(kernel {int(6 * sigma) | 1 if sigma > 0 else 0})", flush=True)
        return out.to(dtype=lat.dtype, device=lat.device)

    def _anchor_content(self, output, lo, hi):
        """The original-A content that gets spliced at the return: output[:, lo:hi] by default.
        ORACLE_ANCHOR_SRC (arm oracle1twin) replaces it with a latent frame loaded from disk --
        the original-A anchor of ANOTHER SEED of the same prompt pair: same prompt, same scene
        grammar, PERFECT sharpness, WRONG instance. ORACLE_BLUR_SIGMA (arm oracle1blur) blurs
        the true frame instead. Both are off by default => byte-identical to every existing arm."""
        real = output[:, lo:hi]
        src = os.environ.get("ORACLE_ANCHOR_SRC", "").strip()
        sig = os.environ.get("ORACLE_BLUR_SIGMA", "").strip()
        assert not (src and sig), "ORACLE_ANCHOR_SRC and ORACLE_BLUR_SIGMA are mutually exclusive"
        if src:
            assert hi - lo == 1, f"ORACLE_ANCHOR_SRC replaces ONE frame, asked for {hi - lo}"
            t = torch.load(src, map_location="cpu", weights_only=False)
            t = t["anchor"] if isinstance(t, dict) and "anchor" in t else t
            assert tuple(t.shape) == tuple(real.shape), \
                f"ORACLE_ANCHOR_SRC {src}: shape {tuple(t.shape)} != {tuple(real.shape)}"
            print(f"[ORACLE_ANCHOR_SRC] anchor content REPLACED by {src} "
                  f"(other seed's original-A frame)", flush=True)
            return t.to(dtype=real.dtype, device=real.device)
        if sig:
            return self._blur_latent(real, float(sig))
        return real

    def _dump_anchor(self, output, a_end):
        """ORACLE_ANCHOR_DUMP: save the original-A anchor latent (the exact frame oracle1
        splices) so another cell can splice it (twin probe). Free side-effect, no RNG draw.
        If the file already exists it is checked bitwise instead of rewritten: every
        MEM_ZERO_ALL arm of a given (pair, seed) must produce the same segment A."""
        path = os.environ.get("ORACLE_ANCHOR_DUMP", "").strip()
        if not path:
            return
        assert a_end is not None and a_end >= 1, "ORACLE_ANCHOR_DUMP before the first switch"
        # native dtype (no fp16 cast): the twin arm must be able to splice it back bitwise,
        # which is what selftest (a) with a seed offset of 0 checks.
        cur = output[:, a_end - 1:a_end].detach().cpu().clone()
        if os.path.exists(path):
            old = torch.load(path, map_location="cpu", weights_only=False)
            old = old["anchor"] if isinstance(old, dict) and "anchor" in old else old
            d = float((old.float() - cur.float()).abs().max())
            print(f"[ORACLE_ANCHOR_DUMP] existing {os.path.basename(path)} re-checked: "
                  f"max|diff| = {d:.3e}", flush=True)
            assert d == 0.0, \
                f"ORACLE_ANCHOR_DUMP: segment A differs from the stored anchor ({path}, {d:.3e})"
            return
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({"anchor": cur, "frame": int(a_end - 1)}, path + ".tmp")
        os.replace(path + ".tmp", path)
        print(f"[ORACLE_ANCHOR_DUMP] original-A anchor latent (frame {a_end - 1}) -> {path}",
              flush=True)

    # ---- reviewer controls of the S2 claim (2026-08-03, reference/paper/risks.md attack 1) ----
    # Both replace the content of the MEM_DECODE anchor and change nothing else: same routing
    # (the schedule's revisit flag), same moment (the return switch), same placement (recency,
    # legacy window convention), same writer, same decoder call. Default off = every existing arm
    # is byte-identical (the helpers return the anchor object itself, not a copy).
    #
    #   MEM_DECODE_NOISE=s     arm `noiseanchor`: the decoded anchor is mixed with pure NOISE
    #                          calibrated on the latent statistics of the current local window
    #                          (per-channel mean and std of the true frames the splice sits in),
    #                          x' = sqrt(1 - s^2) x + s (mu_c + sigma_c eps). s = 1 (the dose of
    #                          the campaign) = pure calibrated noise, zero anchor content: what
    #                          survives is exactly "erase the recency slot, let the prompt
    #                          repaint". The gate-3 arm memdec21c_noise did this at an effective
    #                          dose of 0.288 (the composite mask's mean weight) on 3 cells.
    #   MEM_DECODE_SRC=path    arm `wronganchor`: the anchor is the one decoded from another
    #                          cell's state (dumped by MEM_DECODE_DUMP during that cell's run),
    #                          i.e. a decoder output of the same kind, same fidelity, same
    #                          statistics, but carrying another video's memory. Same mechanic as
    #                          the legacy oracle1twin (ORACLE_ANCHOR_DUMP -> ORACLE_ANCHOR_SRC),
    #                          one level later in the chain: there the wrong content was a real
    #                          frame, here it is a wrong decoded state.
    def _maybe_wrong_anchor(self, frame):
        src = os.environ.get("MEM_DECODE_SRC", "").strip()
        if not src:
            return frame
        assert float(os.environ.get("MEM_DECODE_NOISE", "0") or 0) <= 0.0, \
            "MEM_DECODE_SRC and MEM_DECODE_NOISE are two different control arms, not a stack"
        assert os.path.exists(src), f"MEM_DECODE_SRC: no such anchor dump {src}"
        t = torch.load(src, map_location="cpu", weights_only=False)
        t = t["anchor"] if isinstance(t, dict) and "anchor" in t else t
        assert tuple(t.shape) == tuple(frame.shape), \
            f"MEM_DECODE_SRC {src}: shape {tuple(t.shape)} != this cell's anchor {tuple(frame.shape)}"
        out = t.to(dtype=frame.dtype, device=frame.device)
        a, b = out.float().flatten(), frame.float().flatten()
        cos = float(torch.dot(a, b) / (a.norm() * b.norm() + 1e-12))
        rel = float((a - b).norm() / (b.norm() + 1e-12))
        print(f"[MEM_DECODE_SRC] anchor REPLACED by the one decoded from another cell's state "
              f"({os.path.basename(src)}): cos with this cell's own anchor {cos:.4f}, "
              f"rel-L2 {rel:.4f}", flush=True)
        return out

    def _maybe_noise_anchor(self, frame, output, current_start_frame, n):
        s = float(os.environ.get("MEM_DECODE_NOISE", "0") or 0)
        if s <= 0.0:
            return frame
        s = min(s, 1.0)
        assert current_start_frame >= 1, "MEM_DECODE_NOISE before any frame was produced"
        ref = output[:, max(0, current_start_frame - n):current_start_frame].to(
            device=frame.device, dtype=torch.float32)
        mu = ref.mean(dim=(0, 1, 3, 4), keepdim=True)      # per latent channel (dim 2)
        sd = ref.std(dim=(0, 1, 3, 4), keepdim=True)
        _g = torch.Generator(device=frame.device)
        # per-cell seed (injected by the driver) so the 24 paired cells do not all share one
        # noise realization; + the switch frame so a scenario with 2 revisits draws twice.
        _g.manual_seed(int(os.environ.get("MEM_DECODE_NOISE_SEED", "192837465") or 192837465)
                       + current_start_frame)
        eps = torch.randn(frame.shape, generator=_g, device=frame.device, dtype=torch.float32)
        noise = mu + sd * eps
        out = (((1.0 - s ** 2) ** 0.5) * frame.float() + s * noise).to(frame.dtype)
        print(f"[MEM_DECODE_NOISE] anchor content REPLACED by noise calibrated on the {ref.shape[1]} "
              f"true window frames (dose {s}, per-channel mu in [{float(mu.min()):+.3f}, "
              f"{float(mu.max()):+.3f}], sigma in [{float(sd.min()):.3f}, {float(sd.max()):.3f}]); "
              f"anchor std {float(frame.float().std()):.3f} -> spliced std "
              f"{float(out.float().std()):.3f}", flush=True)
        return out

    # ---- MEM_SDEDIT (arms sdedit_ours / sdedit_wrong, mini-batterie training-free 2026-08-04) --
    # the question. Two explanations survive the reviewer controls of the S2 claim (report §11):
    #   E1 "the anchor is illegible": our decoded anchor is blurry (past sigma = 32 px in DINO
    #      space) and the model only reads sharp frames, so its content is below the reading
    #      threshold, not absent. The Echo frame-bank splices a real frame and its content is
    #      read, which is exactly the fidelity difference.
    #   E2 "the SLOT does everything": what matters is erasing the context at the recency slot
    #      (gate 2: pushing the anchor back by one slot destroys 98 % of the gain) and the prompt
    #      repaints A; content is never read, sharp or blurry.
    # MEM_SDEDIT sharpens the decoded anchor with the model's own prior before splicing it, so
    # that E1's premise is removed: content from the memory, sharpness from the prior.
    #
    # mechanics (SDEdit in latent space, the denoiser is the sharpener; no VAE round trip, which
    # would only re-project, never re-sharpen):
    #   1. x_t = (1 - sigma) x0 + sigma eps  via the pipeline's own scheduler (FlowMatch,
    #      timesteps = sigmas * 1000, so MEM_SDEDIT=s asks for timestep t = 1000 s exactly);
    #   2. denoise along [t] + [the sampler's own remaining steps below t], the same loop as the
    #      main generation (model call -> re-noise to the next step -> ... -> x0);
    #   3. 1-frame mode: a fresh 1-frame kv cache at current_start = 0 (same construction as
    #      _install_add_kv / _write_sink_anchor: nothing is written into the real cache), so the
    #      sharpening sees no context, only the anchor and the return prompt, verbatim.
    # s = 1.0 is the degenerate end (pure prompt generation, zero anchor content); the eye board
    # picks the LOWEST s that sharpens without replacing the subject (rule sealed in the registry).
    #
    #   MEM_SDEDIT=s        0/unset = OFF -> the helper returns the anchor object itself, so every
    #                       existing arm stays byte-identical.
    #   MEM_SDEDIT_SEED=n   per-cell seed of the sdedit noise (injected by the driver), so the 6
    #                       paired cells do not share one realization; + the switch frame.
    def _sdedit_steps(self, sigma):
        """[t_start] + the sampler's own remaining timesteps strictly below it. With the campaign
        config (denoising_step_list [1000, 750, 500, 250], warp_denoising_step, shift 5) the warped
        sampler grid is ~[1000, 937.5, 833.3, 625], so s <= 0.625 gives one model call and larger s
        walks the tail of the real schedule."""
        t0 = float(sigma) * 1000.0
        rest = [float(t) for t in self.denoising_step_list.tolist() if float(t) < t0 - 1e-6]
        return [t0] + rest

    def _sdedit_frame(self, frame, cond, sigma, current_start_frame):
        """SDEdit the anchor latent (B, 1, C, H, W). Returns a new tensor in `frame`'s dtype and
        device. Draws no global RNG (a dedicated generator, like _maybe_noise_anchor), so the
        paired memzero noise stream is untouched.

        Block size (MEM_SDEDIT_FRAMES, default = num_frame_per_block). The first implementation
        denoised a block of one latent frame and the eye board (2026-08-04) showed it produces
        high-frequency mosaic, not sharp content, including at sigma = 1.0, which is the model's
        own full 4-step sampler and must therefore render a clean image. That isolates the cause:
        a 1-frame block is out of distribution for this backbone (it is trained and cached with
        num_frame_per_block = 3, and frames inside a block attend to each other), not the noise
        level. So the anchor is replicated to a native-size block and the middle frame is read
        back. MEM_SDEDIT_FRAMES=1 restores the broken variant, on purpose, as a control."""
        dev = next(self.generator.parameters()).device
        assert frame.shape[1] == 1, \
            f"MEM_SDEDIT sharpens ONE latent frame, got {frame.shape[1]}"
        nf = int(os.environ.get("MEM_SDEDIT_FRAMES", "0") or 0) or self.num_frame_per_block
        x0 = frame.to(device=dev, dtype=torch.bfloat16).repeat(1, nf, 1, 1, 1)
        B = x0.shape[0]
        _g = torch.Generator(device=dev)
        _g.manual_seed(int(os.environ.get("MEM_SDEDIT_SEED", "5150241") or 5150241)
                       + int(current_start_frame))
        steps = self._sdedit_steps(sigma)

        def _renoise(x, t):
            e = torch.randn(x.shape, generator=_g, device=dev, dtype=torch.float32).to(x.dtype)
            tt = torch.ones([x.shape[0] * x.shape[1]], device=dev, dtype=torch.float32) * t
            return self.scheduler.add_noise(x.flatten(0, 1), e.flatten(0, 1), tt) \
                       .unflatten(0, x.shape[:2])

        xt = _renoise(x0, steps[0])
        _real = self.kv_cache1
        self._initialize_kv_cache(batch_size=B, dtype=x0.dtype, device=dev,
                                  kv_cache_size_override=nf * self.frame_seq_length)
        _tmp = self.kv_cache1
        self.kv_cache1 = _real                     # the real cache is never touched here
        self.generator.model.block_mask = self.generator.model._prepare_blockwise_causal_attn_mask(
            device=dev, num_frames=nf, frame_seqlen=self.frame_seq_length,
            num_frame_per_block=self.num_frame_per_block, local_attn_size=self.local_attn_size)
        for blk in self.crossattn_cache:           # the return prompt, freshly cross-attended
            blk["k"].zero_()
            blk["v"].zero_()
            blk["is_init"] = False
        with torch.no_grad():
            for i, t in enumerate(steps):
                for c in _tmp:                     # every call is an independent 1-frame denoise
                    c["k"].zero_()
                    c["v"].zero_()
                    c["global_end_index"].zero_()
                    c["local_end_index"].zero_()
                _, x0_pred = self.generator(
                    noisy_image_or_video=xt,
                    conditional_dict=cond,
                    timestep=torch.ones([B, nf], device=dev, dtype=torch.float32) * t,
                    kv_cache=_tmp,
                    crossattn_cache=self.crossattn_cache,
                    current_start=0,
                    rope_enhance=self.rope_enhance,
                    ttt_cache=getattr(self, "ttt_cache", None),
                    ttt_update=False,
                )
                if i + 1 < len(steps):
                    xt = _renoise(x0_pred, steps[i + 1])
        del _tmp
        for blk in self.crossattn_cache:           # nothing of the phantom pass may survive
            blk["k"].zero_()
            blk["v"].zero_()
            blk["is_init"] = False
        # the middle frame of the block: frames inside a block attend to each other, and latent
        # frame 0 of a Wan clip is the odd one out (1 pixel frame vs 4 for the others).
        _pick = nf // 2
        return x0_pred[:, _pick:_pick + 1].to(dtype=frame.dtype, device=frame.device)

    @staticmethod
    def _lat_hf(x):
        """|laplacian| of the latent, averaged: the cheap sharpness proxy of score_comp.py, in
        latent space. A blurry anchor has a low value; the sharpening must raise it."""
        import torch.nn.functional as F
        k = torch.tensor([[0., 1., 0.], [1., -4., 1.], [0., 1., 0.]],
                         device=x.device).view(1, 1, 3, 3)
        v = x.float().flatten(0, 2)[:, None]                       # (B*T*C, 1, H, W)
        return float(F.conv2d(v, k, padding=1).abs().mean())

    def _maybe_sdedit(self, frame, cond, current_start_frame):
        s = float(os.environ.get("MEM_SDEDIT", "0") or 0)
        if s <= 0.0:
            return frame
        assert 0.0 < s <= 1.0, f"MEM_SDEDIT must be in (0, 1], got {s}"
        assert os.environ.get("MEM_DECODE", "").strip(), \
            "MEM_SDEDIT sharpens the MEM_DECODE anchor: it needs MEM_DECODE=<decoder ckpt>"
        out = self._sdedit_frame(frame, cond, s, current_start_frame)
        a, b = out.float().flatten(), frame.float().flatten()
        cos = float(torch.dot(a, b) / (a.norm() * b.norm() + 1e-12))
        rel = float((a - b).norm() / (b.norm() + 1e-12))
        hf0, hf1 = self._lat_hf(frame), self._lat_hf(out)
        print(f"[MEM_SDEDIT] anchor SHARPENED with the model prior (sigma={s}, trajectory "
              f"{[round(t, 1) for t in self._sdedit_steps(s)]}, 1-frame mode, return prompt): "
              f"cos with the raw anchor {cos:.4f}, rel-L2 {rel:.4f}, latent |laplacian| "
              f"{hf0:.4f} -> {hf1:.4f} (x{hf1 / max(hf0, 1e-9):.2f}), std "
              f"{float(frame.float().std()):.3f} -> {float(out.float().std()):.3f}", flush=True)
        return out

    # ---- MEM_ANCHOR_T (arms i1_*, lot OFFENSIF 2026-08-04) -----------------------------------
    # LE DEFAUT NOMME. Le splice declare l'ancre a t = 0 (`context_timestep = ones *
    # context_noise`, et context_noise = 0 dans toutes nos configs) alors qu'elle est FLOUE et
    # hors-variete : le modele la lit comme une frame PROPRE de son passe immediat, la RMSNorm
    # l'attenue et le contexte B l'absorbe. C'est un MENSONGE de METADONNEE, et c'est le seul
    # correctif qui nomme un defaut mesure (rapport du mecaniste : queries formees a SNR <= -4.4 dB).
    # LE CORRECTIF, en deux gestes indissociables :
    #   1. l'ancre est REELLEMENT bruitee au niveau t* (scheduler FlowMatch du pipeline lui-meme,
    #      x_t = (1 - sigma) x0 + sigma eps, sigma = t*/1000 sur la grille fine) ;
    #   2. le `context_timestep` de ce SLOT vaut t* (le champ est deja par frame).
    # L'ancre devient un latent bruite HONNETE, c'est-a-dire un objet que le modele a vu des
    # millions de fois pendant l'entrainement, au lieu d'une frame propre impossible.
    #   MEM_ANCHOR_T=0/unset   OFF (defaut ; tout bras existant est bit a bit inchange)
    #   MEM_ANCHOR_T=t         t in (0, 1000] : le niveau declare ET applique
    #   MEM_ANCHOR_T_SEED=n    graine du bruit (injectee par le driver, par cellule)
    @staticmethod
    def _anchor_slots(n, kslots, slot_offset):
        """[lo, hi) : les slots de la fenetre de recache occupes par l'ancre, dans les DEUX
        conventions de `_memdec_window` (legacy: les k derniers ; offset >= 0: pousses de
        `offset` vers le passe). Une seule source de verite pour le hook et pour son selftest."""
        if slot_offset < 0:
            return n - kslots, n
        lo = max(0, n - kslots - slot_offset)
        return lo, lo + kslots

    def _maybe_declare_anchor_noise(self, splice, n, kslots, slot_offset, current_start_frame):
        """(splice, timestep_override), l'objet splice lui-MEME et None quand le hook est OFF,
        donc tout bras deja sur disque reste bit a bit reproductible. Ne tire AUCUN RNG global
        (generateur dedie, comme _maybe_noise_anchor)."""
        t = float(os.environ.get("MEM_ANCHOR_T", "0") or 0)
        if t <= 0.0:
            return splice, None
        assert 0.0 < t <= 1000.0, f"MEM_ANCHOR_T must be in (0, 1000], got {t}"
        assert splice is not None and splice.shape[1] == n, \
            f"MEM_ANCHOR_T: window of {n} slots expected, got {None if splice is None else splice.shape[1]}"
        lo, hi = self._anchor_slots(n, kslots, slot_offset)
        part = splice[:, lo:hi]
        dev = part.device
        _g = torch.Generator(device=dev)
        _g.manual_seed(int(os.environ.get("MEM_ANCHOR_T_SEED", "77400241") or 77400241)
                       + int(current_start_frame))
        eps = torch.randn(part.shape, generator=_g, device=dev, dtype=torch.float32)
        tt = torch.ones([part.shape[0] * part.shape[1]], device=dev, dtype=torch.float32) * t
        noised = self.scheduler.add_noise(part.flatten(0, 1).float(), eps.flatten(0, 1), tt) \
            .unflatten(0, part.shape[:2]).to(part.dtype)
        out = splice.clone()
        out[:, lo:hi] = noised
        ts = torch.full((n,), float(self.args.context_noise), dtype=torch.float32)
        ts[lo:hi] = t
        _rel = float((noised.float() - part.float()).norm() / part.float().norm().clamp_min(1e-6))
        assert _rel > 1e-3, "MEM_ANCHOR_T fired but the anchor content did not change"
        print(f"[MEM_ANCHOR_T] anchor slot(s) [{lo}, {hi}) of {n} NOISED to t={t:g} "
              f"(sigma~{t / 1000:.3f}, rel-L2 {_rel:.3f}) AND declared at t={t:g}; the {n - (hi - lo)} "
              f"true frames keep t={int(self.args.context_noise)}", flush=True)
        return out, ts

    # ---- MEM_SINKPURGE (arm sinkpurge, mini-batterie training-free 2026-08-04) ----------------
    # the reviewer's minimal fix. With global_sink the pinned sink (sink_size latent frames of the
    # opener, 1 here, 3 in LongLive's stock config) is never rewritten: _recache_after_switch
    # neither clears the cache nor sets sink_recache_after_switch, and causal_model.py protects
    # [0, sink_tokens) on every recompute. So in S2/SS2 (where the subject does not open) the
    # opener keeps a permanent, never-refreshed grip on every later segment, the documented
    # pathology of report §13.1. The obvious objection to our +0.43 is therefore: "that is just
    # the opener's grip being lifted; refresh the sink at the switch and you get it for free,
    # with no memory at all". This arm is that objection, implemented:
    #   no splice, no memory, no decoder, at the return recache the sink slots are overwritten
    #   with the last true current frame(s), through _write_sink_anchor (the exact mechanism the
    #   ORACLE_SPLICE_POS=sink0 probe already uses; the window is left completely alone).
    #   MEM_SINKPURGE=0/unset  OFF (default; every existing arm byte-identical)
    #   MEM_SINKPURGE=1        purge at revisit switches only (paired with the splice arms: same
    #                          instant, same cell, so the comparison is apples to apples)
    #   MEM_SINKPURGE=2        purge at every switch (the "always refresh" variant)
    def _sink_frames(self):
        """Runtime sink size in latent FRAMES, read from the model (set_runtime_sink_size aware)."""
        cm = _find_causal_model(self.generator.model)
        assert cm is not None, "MEM_SINKPURGE: no CausalWanModel found under the wrappers"
        return int(getattr(cm.blocks[0].self_attn, "_sink_runtime",
                           getattr(cm.blocks[0].self_attn, "sink_size", 0)))

    def _maybe_sinkpurge(self, output, current_start_frame, cond_new, is_revisit):
        mode = int(os.environ.get("MEM_SINKPURGE", "0") or 0)
        if not mode or not (is_revisit or mode >= 2):
            return
        assert os.environ.get("ORACLE_SPLICE_POS", "") != "sink0", \
            "MEM_SINKPURGE and ORACLE_SPLICE_POS=sink0 both rewrite the sink: not a stack"
        assert not os.environ.get("MEM_DECODE", "").strip() \
            and not int(os.environ.get("ORACLE_SPLICE", "0") or 0) and not self._fb_mode(), \
            "MEM_SINKPURGE is the NO-SPLICE, NO-MEMORY control: it must not be combined with a " \
            "recall arm (MEM_DECODE / ORACLE_SPLICE / FRAMEBANK)"
        assert getattr(self, "global_sink", False), \
            "MEM_SINKPURGE targets the PINNED (global) sink; global_sink is off in this config"
        k = self._sink_frames()
        assert k >= 1, "MEM_SINKPURGE: sink_size is 0, there is nothing pinned to purge"
        assert current_start_frame >= k, \
            f"MEM_SINKPURGE needs {k} true frame(s) before the switch (have {current_start_frame})"
        n_tok = k * self.frame_seq_length
        before = self.kv_cache1[0]["k"][:, :n_tok].detach().float().clone()
        tail = self.kv_cache1[0]["k"][:, n_tok:].detach().float().clone()
        cur = output[:, current_start_frame - k:current_start_frame]
        self._write_sink_anchor(cur, cond_new)
        after = self.kv_cache1[0]["k"][:, :n_tok].detach().float()
        rel = float((after - before).norm() / (before.norm() + 1e-12))
        untouched = float((self.kv_cache1[0]["k"][:, n_tok:].detach().float() - tail).abs().max())
        print(f"[MEM_SINKPURGE] the {k} pinned sink frame(s) of the OPENER were overwritten with "
              f"the current true frame(s) [{current_start_frame - k}, {current_start_frame}) at "
              f"frame {current_start_frame} (mode {mode}): sink k rel-L2 change {rel:.4f}, rest of "
              f"the window max|diff| {untouched:.3e} (must be 0)", flush=True)
        assert rel > 1e-3, "MEM_SINKPURGE fired but the sink k/v did not change"
        assert untouched == 0.0, "MEM_SINKPURGE touched the local window, it must not"

    # ---- MEM_DECODE_COMPOSITE (arms oracle1comp / memdec21c, 2026-08-02) ---------------------
    # the problem (Aden's eye verdict on the same-scene tier). Our splice replaces a whole latent
    # frame, so the return re-renders the entire scene from the past: on SS2_t14 the bakery awning
    # flips blue -> pink at the return, while the frame-bank baseline brings the subject back into
    # an intact scene. The metric that saw this is the border/complement-of-crop one, not the
    # crop60 re-ID.
    # the fix. Composite the spliced latent instead of overwriting it:
    #     composite = m * ANCHOR (the subject, decoded or real) + (1 - m) * background
    # with background = the last true latent frame of the current context (the frame produced just
    # before the return switch, i.e. exactly the frame oracle1 evicts) and m a soft elliptical
    # center mask. The recalled identity keeps the center; the world keeps rolling at the border.
    #   MEM_DECODE_COMPOSITE=1  turn it on   (default 0/unset = every existing arm byte-identical)
    #   MEM_COMP_CROP=0.6       fraction of H and W covered by the ellipse at m = 0.5 (the
    #                           center-60 % of the re-ID crop by default)
    #   MEM_COMP_FEATHER=0.3    relative half-width of the smoothstep ring around the ellipse
    #   MEM_COMP_DUMP=<prefix>  debug: save {bg, anchor, composite, mask} for the eye board
    @staticmethod
    def _composite_mask(H, W, crop, feather, device, dtype, cy=None, cx=None):
        """Soft elliptical mask on the latent grid: 1 = take the anchor (the subject),
        0 = keep the current context (the background). The blend weights are the pair
        (m, 1 - m), so they sum to 1 EVERYWHERE by construction. smoothstep => C1 at both ends
        of the ring, hence no seam in the decoded pixels (selftest_composite.py).
        (cy, cx) = ellipse center in PIXELS of the latent grid; None = the geometric center
        (the historical behaviour, used by every existing composite arm)."""
        assert 0.0 < crop <= 1.0, f"MEM_COMP_CROP must be in (0, 1], got {crop}"
        assert 0.0 <= feather < 1.0, f"MEM_COMP_FEATHER must be in [0, 1), got {feather}"
        cy = (H - 1) / 2.0 if cy is None else float(cy)
        cx = (W - 1) / 2.0 if cx is None else float(cx)
        yy = (torch.arange(H, device=device, dtype=torch.float32) - cy) / (crop * H / 2.0)
        xx = (torch.arange(W, device=device, dtype=torch.float32) - cx) / (crop * W / 2.0)
        r = torch.sqrt(yy[:, None] ** 2 + xx[None, :] ** 2)
        r0, r1 = 1.0 - feather, 1.0 + feather
        t = ((r1 - r) / max(r1 - r0, 1e-6)).clamp(0.0, 1.0)
        m = t * t * (3.0 - 2.0 * t)                       # smoothstep
        return m.to(dtype)[None, None, None]              # (1,1,1,H,W), broadcasts over B,F,C

    @staticmethod
    def _energy_centroid(lat):
        """Crude subject localizer on a latent frame, used by the UNION mask (gate 4): the
        centroid of the spatial energy that deviates from the frame's own spatial mean, i.e.
        'where does this frame stop looking like a flat backdrop'. No cross-attention, no
        detector: a 5-line saliency whose only job is to be better than 'always the center'.
        lat (B,1,C,H,W) -> (cy, cx) in latent pixels."""
        x = lat[0, 0].float()                                     # (C,H,W)
        e = (x - x.mean(dim=(1, 2), keepdim=True)).pow(2).sum(0).sqrt()      # (H,W)
        w = (e - e.median()).clamp_min(0.0)                       # keep the salient half
        if float(w.sum()) <= 0:
            return None, None
        H, W = w.shape
        ys = torch.arange(H, device=w.device, dtype=torch.float32)
        xs = torch.arange(W, device=w.device, dtype=torch.float32)
        cy = float((w.sum(1) * ys).sum() / w.sum())
        cx = float((w.sum(0) * xs).sum() / w.sum())
        return cy, cx

    def _comp_alpha(self):
        """Composite DOSE in [0, 1]. alpha = 1 (default) = the full composite; alpha = 0 leaves
        the anchor STRICTLY untouched, i.e. the arm falls back to its non-composite twin, which
        is what makes the prompt gate safe to ship (selftest).
          MEM_COMP_ALPHA=a          fixed dose
          MEM_COMP_PROMPT_GATE=1    a is multiplied by a ramp on the cosine between the prompt
                                    ENTERING at this revisit and the prompt of the segment being
                                    left: same decor asked twice (high cosine) => composite, so
                                    the world must keep rolling; unrelated scene (low cosine) =>
                                    alpha -> 0, the plain full-frame splice, because there a cut
                                    is legitimate montage grammar (design v1 §2, per-tier rule).
          MEM_COMP_GATE_LO/HI       the ramp bounds (defaults 0.50 / 0.80)"""
        a = float(os.environ.get("MEM_COMP_ALPHA", "1") or 1)
        if int(os.environ.get("MEM_COMP_PROMPT_GATE", "0") or 0):
            a *= float(getattr(self, "_comp_gate_now", 1.0))
        return max(0.0, min(1.0, a))

    def _comp_gate_from_prompts(self, cond_new, cond_prev):
        """Prompt-cosine ramp, computed once per revisit switch (no RNG draw, CPU only)."""
        lo = float(os.environ.get("MEM_COMP_GATE_LO", "0.50") or 0.50)
        hi = float(os.environ.get("MEM_COMP_GATE_HI", "0.80") or 0.80)
        if cond_prev is None:
            self._comp_gate_now = 1.0
            return
        c = float(torch.dot(self._fb_key(cond_new), self._fb_key(cond_prev)))
        self._comp_gate_now = max(0.0, min(1.0, (c - lo) / max(hi - lo, 1e-6)))
        print(f"[MEM_COMP_PROMPT_GATE] cos(prompt entering, prompt being left) = {c:.4f} "
              f"-> composite dose alpha = {self._comp_gate_now:.3f} (ramp {lo}..{hi})", flush=True)

    def _maybe_composite(self, anchor, output, current_start_frame):
        """Blend the recalled ANCHOR (subject) onto the last true latent frame of the current
        context (background). Returns `anchor` itself (same object) when the hook is off, so the
        legacy code path is untouched."""
        if not int(os.environ.get("MEM_DECODE_COMPOSITE", "0") or 0):
            return anchor
        assert os.environ.get("ORACLE_SPLICE_POS", "") not in ("head", "sink0"), \
            "MEM_DECODE_COMPOSITE is defined for the RECENCY placement only (the background it " \
            "blends into is the CURRENT context, which only makes sense at the recency slot)"
        assert current_start_frame >= 1, "MEM_DECODE_COMPOSITE before any frame was produced"
        bg = output[:, current_start_frame - 1:current_start_frame].to(
            device=anchor.device, dtype=anchor.dtype)
        assert bg.shape[0] == anchor.shape[0] and bg.shape[2:] == anchor.shape[2:], \
            f"composite shape mismatch: bg {tuple(bg.shape)} vs anchor {tuple(anchor.shape)}"
        crop = float(os.environ.get("MEM_COMP_CROP", "0.6") or 0.6)
        feather = float(os.environ.get("MEM_COMP_FEATHER", "0.3") or 0.3)
        H, W = anchor.shape[-2], anchor.shape[-1]
        # ---- anti-illusion controls (gate 3, 2026-08-02), every one of them defaults to OFF,
        # so the composite arms already on disk are reproduced bitwise.
        #   MEM_COMP_SHIFT=f       ellipse center moved by f * W horizontally (wrong place)
        #   MEM_COMP_INVERT=1      m -> 1 - m (anchor at the border, context at the center)
        #   MEM_COMP_UNIFORM=1     no geometry at all: constant m = mean of the elliptical mask
        #                          (dose-matched: same average subject weight, no shape)
        #   MEM_COMP_ANCHOR_NOISE  RMS-preserving noise mix on the anchor before the blend
        #                          (mask kept, content destroyed)
        shift = float(os.environ.get("MEM_COMP_SHIFT", "0") or 0)
        invert = int(os.environ.get("MEM_COMP_INVERT", "0") or 0)
        uniform = int(os.environ.get("MEM_COMP_UNIFORM", "0") or 0)
        union = int(os.environ.get("MEM_COMP_UNION", "0") or 0)
        cy = cx = None
        w_noise = w_anchor = None   # MEM_COMP_FILL: poids de la region "sujet courant" / de l'ancre
        if shift:
            cx = (W - 1) / 2.0 + shift * W
        m = self._composite_mask(H, W, crop, feather, anchor.device, torch.float32, cy, cx)
        if union:
            # gate 4: the mask must also cover the current subject, otherwise the composite
            # pastes the recalled subject ON top of the interlude's subject (two heads in the
            # frame). No cross-attention available -> union of the ellipse around the anchor's
            # energy centroid and the ellipse around the background's energy centroid.
            ay, ax = self._energy_centroid(anchor)
            by, bx = self._energy_centroid(bg)
            m_a = self._composite_mask(H, W, crop, feather, anchor.device, torch.float32, ay, ax)
            m_b = self._composite_mask(H, W, crop, feather, anchor.device, torch.float32, by, bx)
            m = torch.maximum(m_a, m_b)
            print(f"[MEM_COMP_UNION] anchor centroid ({ay}, {ax}) U background centroid "
                  f"({by}, {bx}), union mask mean {float(m.mean()):.3f}", flush=True)
            # MEM_COMP_FILL=noise (bras memdec21un, verdict oeil d'Aden 2026-08-03) : dans la part
            # du masque union qui vient de l'ellipse du sujet courant et pas de celle de l'ancre,
            # le composite collait le contenu de l'ancre, c'est-a-dire du passe-bas de A a
            # l'ancienne place de B : le FANTOME FLOU vu a l'oeil, que le debruiteur ne nettoie
            # qu'a moitie. On y met du bruit a la place (meme calibration que le bras
            # memdec21c_noise du gate 3) : la region est effacee, et le prompt + le contexte la
            # repeignent librement au lieu d'avoir a effacer une image.
            # MEM_COMP_FILL=priority (bras memdec21up) : le correctif "noise" ci-dessus ne touche
            # QUE le croissant "B seul" (poids moyen 0.11 sur le tier meme-decor, parce que A et B
            # y occupent quasi la meme place) ; l'essentiel du fantome est sous l'ellipse de
            # l'ancre, la ou le composite colle le passe-bas de A par-dessus l'endroit ou etait B.
            # 'priority' donne la priorite au bruit sur le recouvrement :
            #     w_n = m_b,  w_a = max(0, m_a - m_b),  w_bg = 1 - max(m_a, m_b)
            # (somme = 1 partout, dans les deux ordres). Toute la zone du sujet courant est
            # effacee, y compris sous l'ancre : le prompt et le contexte la repeignent.
            _fill = (os.environ.get("MEM_COMP_FILL", "") or "").strip().lower()
            if _fill == "noise":
                w_noise = (m - m_a).clamp_min(0.0)
                w_anchor = m - w_noise
            elif _fill == "priority":
                w_noise = m_b
                w_anchor = (m_a - m_b).clamp_min(0.0)
            elif _fill == "floor":
                # plancher D'ANCRE (bras memdec21uf) : 'priority' repare le fantome mais coupe le
                # canal d'identite la ou A et B se recouvrent (tripwire sujet tire sur 2/3
                # cellules). On garde donc au minimum une fraction f de l'ancre PARTOUT :
                #     w_a = max(m_a - m_b, f * m_a),  w_n = m_union - w_a,  w_bg = 1 - m_union
                # f = 0 redonne EXACTEMENT 'priority', f = 1 redonne EXACTEMENT 'noise' : le
                # floor interpolated between the two already-measured extremes (selftest g11).
                _ff = float(os.environ.get("MEM_COMP_FILL_FLOOR", "0.4") or 0.4)
                w_anchor = torch.maximum(m_a - m_b, _ff * m_a).clamp_min(0.0)
                w_noise = m - w_anchor
            elif _fill:
                raise AssertionError(
                    f"MEM_COMP_FILL={_fill!r} (attendu ''|'noise'|'priority'|'floor')")
        if invert:
            m = 1.0 - m
        if uniform:
            m = torch.full_like(m, float(m.mean()))
        # MEM_COMP_ALPHA / prompt gate: composite dose. Alpha = 0 leaves the anchor untouched
        # (bitwise the non-composite arm), alpha = 1 is the full composite.
        alpha = self._comp_alpha()
        if alpha != 1.0 and w_noise is None:
            # (le chemin MEM_COMP_FILL applique alpha lui-meme, sur ses trois poids)
            m = 1.0 - alpha * (1.0 - m)
        a_use = anchor
        sig = float(os.environ.get("MEM_COMP_ANCHOR_NOISE", "0") or 0)
        if sig > 0:
            _g = torch.Generator(device=anchor.device)
            _g.manual_seed(192837465 + current_start_frame)
            _eps = torch.randn(anchor.shape, generator=_g, device=anchor.device,
                               dtype=torch.float32).to(anchor.dtype)
            a_use = ((1.0 - min(sig, 1.0) ** 2) ** 0.5) * anchor + min(sig, 1.0) * _eps
            print(f"[MEM_COMP_ANCHOR_NOISE] anchor content degraded with sigma={sig} "
                  f"(mask untouched)", flush=True)
        if w_noise is None:
            comp = (m * a_use.float() + (1.0 - m) * bg.float()).to(anchor.dtype)
        else:
            # trois poids qui somment a 1 PARTOUT par construction (w_a + w_n + w_bg = 1) :
            #   w_a  = m_a          l'ancre, dans sa propre ellipse (region inchangee vs 21u)
            #   w_n  = m - m_a      la region du sujet courant, remplie de bruit
            #   w_bg = 1 - m        le monde, inchange
            # La dose alpha porte sur les deux poids non-ancre, donc alpha = 0 rend encore
            # l'ancre pure (= memdec20 bit a bit) exactement comme dans le chemin normal.
            assert not (invert or uniform), \
                "MEM_COMP_FILL=noise est defini pour le masque UNION nu (pas avec invert/uniform)"
            assert w_anchor is not None
            _fs = min(float(os.environ.get("MEM_COMP_FILL_SIGMA", "1.0") or 1.0), 1.0)
            _gf = torch.Generator(device=anchor.device)
            _gf.manual_seed(192837465 + current_start_frame)      # meme graine que le gate 3
            _eps_fill = torch.randn(anchor.shape, generator=_gf, device=anchor.device,
                                    dtype=torch.float32).to(anchor.dtype)
            _eps_fill = ((1.0 - _fs ** 2) ** 0.5) * a_use + _fs * _eps_fill
            w_a, w_n, w_bg = w_anchor, w_noise, 1.0 - m
            if alpha != 1.0:
                w_n, w_bg = alpha * w_n, alpha * w_bg
                w_a = 1.0 - alpha * (1.0 - w_a)
            comp = (w_a * a_use.float() + w_n * _eps_fill.float()
                    + w_bg * bg.float()).to(anchor.dtype)
            print(f"[MEM_COMP_FILL={_fill}] region 'sujet courant' (poids moyen {float(w_n.mean()):.3f}) "
                  f"remplie de BRUIT sigma={_fs} au lieu du contenu d'ancre ; ancre {float(w_a.mean()):.3f}, "
                  f"fond {float(w_bg.mean()):.3f}", flush=True)
        self._comp_applied = getattr(self, "_comp_applied", 0) + 1
        _dump = os.environ.get("MEM_COMP_DUMP", "").strip()
        if _dump:
            os.makedirs(os.path.dirname(_dump) or ".", exist_ok=True)
            torch.save({"bg": bg.detach().cpu(), "anchor": anchor.detach().cpu(),
                        "composite": comp.detach().cpu(), "mask": m.detach().cpu(),
                        "crop": crop, "feather": feather,
                        "bg_frame": int(current_start_frame - 1)},
                       f"{_dump}.f{current_start_frame}.pt")
            print(f"[MEM_COMP_DUMP] composite parts -> {_dump}.f{current_start_frame}.pt")
        print(f"[MEM_DECODE_COMPOSITE] anchor ({anchor.shape[1]} frame(s)) composited onto the "
              f"background of true frame {current_start_frame - 1}: elliptical mask crop={crop} "
              f"feather={feather}, subject weight mean {float(m.mean()):.3f}"
              + (f" | CONTROLS shift={shift} invert={invert} uniform={uniform} union={union} "
                 f"alpha={alpha:.3f} anchor_noise={sig}"
                 if (shift or invert or uniform or union or sig or alpha != 1.0) else ""),
              flush=True)
        return comp

    # ---- MEM_ADDITIVE_KV (arm oracle1add, GATE 1 of 2026-08-02) -----------------------------
    # Every recall arm measured so far splices: the recalled frame takes a slot of the 12-frame
    # local window, so a true frame of the current context is evicted and the model's immediate
    # declared past becomes partly a foreign world (same-scene tier: bg_cont 0.46 vs 0.90 for the
    # bank, cut2 20.0 vs 7.7). This channel is ADDITIVE instead: the anchor becomes 1560 extra
    # k/v tokens per block, concatenated at attention time in the free RoPE band between the sink
    # and the window (see CausalWanSelfAttention._additive_attention). Nothing is evicted.
    #   MEM_ADDITIVE_KV=1   turn it on (default 0/unset = every existing arm byte-identical)
    #   MEM_ADD_GAIN=g      attention-mass gain of the anchor tokens (1 = plain concatenation)
    #   MEM_ADD_POS=k|auto  RoPE band start of the anchor tokens; auto = one frame before the
    #                       local window (the distances the model already reads every step)
    def _add_blocks(self):
        cm = _find_causal_model(self.generator.model)
        assert cm is not None, "MEM_ADDITIVE_KV: no CausalWanModel found under the wrappers"
        return cm.blocks

    def _clear_add_kv(self):
        """Called once per rollout, unconditionally: an additive channel installed by a previous
        arm of the same process must never leak into the next one."""
        cm = _find_causal_model(self.generator.model)
        if cm is None:
            return
        for blk in cm.blocks:
            sa = blk.self_attn
            sa._add_kv = None
            sa._add_capture = False
            sa._add_captured = None

    def _install_add_kv(self, anchor_frames, conditional_dict):
        """Phantom forward of the anchor frame through a fresh 1-frame cache: every block sees a
        clean position-0 write and we capture the raw (pre-RoPE) k/v it produces. Same
        construction as _write_sink_anchor, except nothing is written into the real cache, the
        tokens are read at attention time only. Draws no RNG (the memzero pairing is preserved).
        """
        device = next(self.generator.parameters()).device
        anchor_frames = anchor_frames.to(device)
        B, nf = anchor_frames.shape[0], anchor_frames.shape[1]
        blocks = self._add_blocks()
        for blk in blocks:
            blk.self_attn._add_kv = None            # the phantom pass itself must be clean
            blk.self_attn._add_capture = True
            blk.self_attn._add_captured = None
        _real = self.kv_cache1
        self._initialize_kv_cache(batch_size=B, dtype=anchor_frames.dtype, device=device,
                                  kv_cache_size_override=nf * self.frame_seq_length)
        _tmp = self.kv_cache1
        self.kv_cache1 = _real
        block_mask = self.generator.model._prepare_blockwise_causal_attn_mask(
            device=device, num_frames=nf, frame_seqlen=self.frame_seq_length,
            num_frame_per_block=self.num_frame_per_block, local_attn_size=self.local_attn_size)
        context_timestep = torch.ones([B, nf], device=device,
                                      dtype=torch.int64) * self.args.context_noise
        self.generator.model.block_mask = block_mask
        with torch.no_grad():
            self.generator(
                noisy_image_or_video=anchor_frames,
                conditional_dict=conditional_dict,
                timestep=context_timestep,
                kv_cache=_tmp,
                crossattn_cache=self.crossattn_cache,
                current_start=0,
                rope_enhance=self.rope_enhance,
                ttt_cache=getattr(self, "ttt_cache", None),
                ttt_update=False,
            )
        del _tmp
        gain = float(os.environ.get("MEM_ADD_GAIN", "1") or 1)
        _p = (os.environ.get("MEM_ADD_POS", "") or "").strip().lower()
        pos = -1 if _p in ("", "auto") else int(_p)
        n_tok = nf * self.frame_seq_length
        for bi, blk in enumerate(blocks):
            sa = blk.self_attn
            assert sa._add_captured is not None, \
                f"MEM_ADDITIVE_KV: the phantom pass captured nothing on block {bi}"
            k, v = sa._add_captured
            assert k.shape[1] == n_tok and v.shape[1] == n_tok, \
                f"MEM_ADDITIVE_KV: block {bi} captured {k.shape[1]} tokens, expected {n_tok}"
            sa._add_kv = (k, v)
            sa._add_gain = gain
            sa._add_pos = pos
            sa._add_capture = False
            sa._add_captured = None
        for blk in self.crossattn_cache:
            blk["k"].zero_()
            blk["v"].zero_()
            blk["is_init"] = False
        self._add_installed = True
        print(f"[MEM_ADDITIVE_KV] anchor frame installed as {n_tok} EXTRA k/v tokens on "
              f"{len(blocks)} blocks (gain={gain}, rope band "
              f"{'auto (one frame before the window)' if pos < 0 else pos}); the local window "
              f"keeps its 12 TRUE frames, nothing evicted", flush=True)

    def _add_mass_report(self):
        """Mean attention mass the anchor tokens took, per block, the anti-inertia tripwire of
        the design plan (a channel that reads 0 % is dead by construction)."""
        cm = _find_causal_model(self.generator.model)
        if cm is None or not getattr(self, "_add_installed", False):
            return None
        return {bi: float(getattr(blk.self_attn, "_add_mass", float("nan")))
                for bi, blk in enumerate(cm.blocks)}

    def _add_mass_src_report(self):
        """T1 (lot offensif 2026-08-04): attention mass by source [sink | window | anchor] per
        block, at the last forward. `window` = everything in the context that is not the sink
        (the prompt cache, when active, is counted there). None unless MEM_ADD_MASS_SRC=1."""
        cm = _find_causal_model(self.generator.model)
        if cm is None or not getattr(self, "_add_installed", False):
            return None
        out = {}
        for bi, blk in enumerate(cm.blocks):
            m = getattr(blk.self_attn, "_add_mass_src", None)
            if m is not None:
                out[bi] = {"sink": m[0], "window": m[1], "anchor": m[2],
                           "n_tok": getattr(blk.self_attn, "_add_ntok_src", None)}
        return out or None

    def _write_sink_anchor(self, anchor_frames, conditional_dict):
        """ORACLE_SPLICE_POS=sink0 probe: overwrite the sink slot k/v with a fresh anchor frame
        (the persistent-position variant: the anchor lives where the stale first-frame anchor
        lived, read by every future query, never evicted). Runs a 1-frame forward at
        current_start=0 with sink_recache_after_switch=True so the protected sink slots are
        legitimately rewritten (the flag exists for exactly this)."""
        device = anchor_frames.device if anchor_frames.device.type == "cuda" \
            else next(self.generator.parameters()).device
        anchor_frames = anchor_frames.to(device)
        B, nf = anchor_frames.shape[0], anchor_frames.shape[1]
        # The main cache is in rolled/recompute state: a forward at current_start=0 cannot
        # address the sink slots through the window arithmetic (empty write span). Instead:
        # prefill the anchor into a fresh temporary 1-frame cache (clean position-0 write),
        # then copy its k/v into the sink slots of the real cache (buffer layout [sink|window],
        # sink = first sink_tokens entries, stored UNroped under rope_enhance like any key).
        _real = self.kv_cache1
        self._initialize_kv_cache(batch_size=B, dtype=anchor_frames.dtype, device=device,
                                  kv_cache_size_override=nf * self.frame_seq_length)
        _tmp = self.kv_cache1
        self.kv_cache1 = _real
        block_mask = self.generator.model._prepare_blockwise_causal_attn_mask(
            device=device, num_frames=nf, frame_seqlen=self.frame_seq_length,
            num_frame_per_block=self.num_frame_per_block, local_attn_size=self.local_attn_size)
        context_timestep = torch.ones([B, nf], device=device,
                                      dtype=torch.int64) * self.args.context_noise
        self.generator.model.block_mask = block_mask
        with torch.no_grad():
            self.generator(
                noisy_image_or_video=anchor_frames,
                conditional_dict=conditional_dict,
                timestep=context_timestep,
                kv_cache=_tmp,
                crossattn_cache=self.crossattn_cache,
                current_start=0,
                rope_enhance=self.rope_enhance,
                ttt_cache=getattr(self, "ttt_cache", None),
                ttt_update=False,
            )
            n_tok = nf * self.frame_seq_length
            for bi in range(self.num_transformer_blocks):
                self.kv_cache1[bi]["k"][:, :n_tok].copy_(_tmp[bi]["k"][:, :n_tok])
                self.kv_cache1[bi]["v"][:, :n_tok].copy_(_tmp[bi]["v"][:, :n_tok])
        del _tmp
        for blk in self.crossattn_cache:
            blk["k"].zero_()
            blk["v"].zero_()
            blk["is_init"] = False

    # ---- FRAMEBANK (Echo-Forcing-style real-content scene recall baseline) --------------------
    # Echo-Forcing (2605.16003, training-free) keeps a "scene memory pool" of compressed real KV
    # per past scene, routes at a scene transition by cosine similarity between the current
    # prompt and every previous prompt, and recalls only when the best match is a non-adjacent
    # scene above a threshold. Our faithful-in-spirit port on this substrate: the pool holds the
    # last latent frame of every completed segment (one entry per segment, so the bank grows),
    # keyed by the pooled prompt embedding of that segment; on a switch the entering prompt is
    # matched against the pool and the winning frame is spliced at the recency window slot --
    # the exact mechanism of oracle1, only the content source differs.
    #   FRAMEBANK=""       off (default; every existing arm is byte-identical)
    #   FRAMEBANK="key"    real routing: argmax cosine over non-adjacent entries, recall iff >= tau
    #   FRAMEBANK="oracle" ceiling of the same bank: ground-truth prompt identity, no cosine
    #   FRAMEBANK_TAU      routing threshold (default 0.90)
    #   FRAMEBANK_TIE      "recent" (default, freshest true observation) | "first" (1st appearance)
    def _fb_mode(self):
        return (os.environ.get("FRAMEBANK", "") or "").strip().lower()

    def _route_on(self):
        """Is the Echo-Forcing ROUTER (bank + cosine decision) needed at all this rollout?
        True for the framebank arms (which also splice the banked frame) and for
        MEM_DECODE_ROUTED (arm memdec20r), which borrows the router's DECISION only. Default
        off on both counts => `_fb_store`/`_fb_select` stay no-ops and every existing arm is
        byte-identical."""
        return bool(self._fb_mode()) or bool(int(os.environ.get("MEM_DECODE_ROUTED", "0") or 0))

    def _fb_reset(self):
        self._fb_bank = []       # list of {seg, prompt_idx, end_frame, key, frame}
        self._fb_events = []     # per-switch routing log (dumped by the driver for error analysis)

    @staticmethod
    def _fb_key(cond):
        """L2-normalized pooled prompt embedding = the scene key. Same pooling as the decoder's
        cond (non-padding rows only), fp32, CPU: deterministic, draws no RNG (the paired-arm
        noise stream must stay identical)."""
        k = _memdec_pooled_cond(cond).float()
        return k / k.norm().clamp_min(1e-12)

    def _fb_store(self, output, current_start_frame, seg_idx, prompt_idx, cond, subject=None):
        """Scene lifecycle 'preserve': the segment that just ended enters the pool.
        `subject` = the ground-truth identity of the segment (which may be described by several
        different prompt strings, cf. the paraphrase scenarios); it is used only by the oracle
        selection and by the correctness bookkeeping, the real router never sees it."""
        if not self._route_on():
            return
        assert current_start_frame >= 1, "FRAMEBANK store before any frame was produced"
        self._fb_bank.append({
            "seg": int(seg_idx), "prompt_idx": int(prompt_idx),
            "subject": int(prompt_idx if subject is None else subject),
            "end_frame": int(current_start_frame),
            "key": self._fb_key(cond),
            "frame": output[:, current_start_frame - 1:current_start_frame].detach().clone(),
        })
        print(f"[FRAMEBANK] preserve: segment {seg_idx} (prompt {prompt_idx}, subject "
              f"{self._fb_bank[-1]['subject']}), last latent frame {current_start_frame - 1} "
              f"-> pool size {len(self._fb_bank)}", flush=True)

    def _fb_select(self, seg_idx, prompt_idx, cond, subject=None, mode=None, force=None):
        """Scene lifecycle 'recall'. Returns (frame or None, event dict). The event dict logs
        every candidate similarity and whether the pick was the right subject -> the error
        analysis of the selection (easy at bank size 1-2 in S1/S2, 2-4 in S3/S4).

        `mode` overrides FRAMEBANK: MEM_DECODE_ROUTED calls this with mode='key' to obtain the
        routing decision ('is the entering prompt a revisit of a non-adjacent banked scene?')
        while FRAMEBANK stays off and the recalled frame is thrown away, same QUAND as the
        framebank arm, a different QUOI.

        `force` (FRAMEBANK_GT, pilote 'texte insuffisant' 2026-08-03) replaces the router's
        decision by the schedule's ground-truth revisit flag, without touching the selection:
        force=True recalls the argmax candidate even below tau, force=False recalls nothing.
        Default None = the published behaviour, bitwise. Why: with an impoverished return prompt
        ('the animal from earlier returns') the prompt cosine collapses, the bank never fires and
        the arm silently degenerates into memzero, the comparison with memdec20 would then be
        'a splice vs no splice', not 'which content'. What the router would have done is kept in
        the event (router_recalled / router_chosen_seg / router_s_max): it is a result of its own,
        not a detail (word routing dies in the generic regime)."""
        mode = mode or self._fb_mode()
        tau = float(os.environ.get("FRAMEBANK_TAU", "0.90") or 0.90)
        tie = (os.environ.get("FRAMEBANK_TIE", "recent") or "recent").strip().lower()
        assert tie in ("recent", "first"), f"FRAMEBANK_TIE must be recent|first, got {tie!r}"
        subj = int(prompt_idx if subject is None else subject)
        q = self._fb_key(cond)
        # Echo-Forcing routing rule: the immediately preceding scene is the recent window, never
        # a "recall" -> it is excluded from the candidate pool.
        cands = [e for e in self._fb_bank if e["seg"] != seg_idx - 1]
        sims = [float(torch.dot(q, e["key"])) for e in cands]
        ev = {"seg": int(seg_idx), "prompt_idx": int(prompt_idx), "subject": subj, "mode": mode,
              "tau": tau, "tie": tie, "bank_size": len(self._fb_bank),
              "candidates": [{"seg": e["seg"], "prompt_idx": e["prompt_idx"],
                              "subject": e["subject"], "end_frame": e["end_frame"], "sim": s}
                             for e, s in zip(cands, sims)]}
        pick = None
        if mode == "oracle":
            ok = [e for e in cands if e["subject"] == subj]
            if ok:
                pick = ok[-1] if tie == "recent" else ok[0]
            ev["rule"] = "ground-truth subject identity (bank ceiling)"
        elif mode == "key":
            if cands:
                smax = max(sims)
                best = [e for e, s in zip(cands, sims) if s >= smax - 1e-6]
                ev["s_max"] = smax
                if smax >= tau:
                    pick = best[-1] if tie == "recent" else best[0]
            ev["rule"] = f"argmax cosine over non-adjacent scenes, recall iff s_max >= {tau}"
        else:
            raise AssertionError(f"FRAMEBANK={mode!r} (expected ''|'key'|'oracle')")
        if force is not None:
            # ce que le ROUTEUR aurait fait est conserve tel quel (resultat a part entiere)
            ev["router_recalled"] = pick is not None
            ev["router_chosen_seg"] = None if pick is None else int(pick["seg"])
            ev["router_chosen_subject"] = None if pick is None else int(pick["subject"])
            ev["router_correct"] = bool(pick is not None and pick["subject"] == subj)
            ev["forced"] = bool(force)
            ev["rule"] += " | FRAMEBANK_GT: le QUAND vient du planning, la SELECTION reste celle "
            ev["rule"] += "de la methode (argmax meme sous tau)" if mode == "key" else "du plafond"
            pick = None
            if force and cands:
                if mode == "oracle":
                    ok = [e for e in cands if e["subject"] == subj]
                    pick = (ok[-1] if tie == "recent" else ok[0]) if ok else None
                else:
                    smax = max(sims)
                    best = [e for e, s in zip(cands, sims) if s >= smax - 1e-6]
                    pick = best[-1] if tie == "recent" else best[0]
        ev["recalled"] = pick is not None
        ev["chosen_seg"] = None if pick is None else int(pick["seg"])
        ev["chosen_prompt_idx"] = None if pick is None else int(pick["prompt_idx"])
        ev["chosen_subject"] = None if pick is None else int(pick["subject"])
        ev["correct"] = bool(pick is not None and pick["subject"] == subj)
        self._fb_events.append(ev)
        print(f"[FRAMEBANK] recall @segment {seg_idx} (prompt {prompt_idx}, subject {subj}): " +
              " ".join(f"seg{c['seg']}(p{c['prompt_idx']}):{c['sim']:.4f}" for c in ev["candidates"])
              + f" -> {'segment ' + str(ev['chosen_seg']) if ev['recalled'] else 'NO RECALL'}"
              + (" [CORRECT]" if ev["correct"] else (" [WRONG SUBJECT]" if ev["recalled"] else "")),
              flush=True)
        return (None if pick is None else pick["frame"]), ev

    def _recache_after_switch(self, output, current_start_frame, new_conditional_dict,
                              frames_override=None, timestep_override=None):
        if not self.global_sink:
            # reset kv cache
            for block_idx in range(self.num_transformer_blocks):
                cache = self.kv_cache1[block_idx]
                cache["k"].zero_()
                cache["v"].zero_()
                # cache["global_end_index"].zero_()
                # cache["local_end_index"].zero_()
            
        # reset cross-attention cache
        for blk in self.crossattn_cache:
            blk["k"].zero_()
            blk["v"].zero_()
            blk["is_init"] = False
        
        # recache
        if current_start_frame == 0:
            return

        num_recache_frames = current_start_frame if self.local_attn_size == -1 else min(self.local_attn_size, current_start_frame)
        recache_start_frame = current_start_frame - num_recache_frames

        if frames_override is not None:
            # ORACLE_SPLICE probe: recache content from elsewhere in the rollout (e.g. the original
            # A tail) at the current positions (current_start unchanged below), i.e. a KV splice.
            assert frames_override.shape[1] == num_recache_frames, \
                f"frames_override has {frames_override.shape[1]} frames, recache needs {num_recache_frames}"
            frames_to_recache = frames_override
        else:
            frames_to_recache = output[:, recache_start_frame:current_start_frame]
        # move to gpu
        if frames_to_recache.device.type == 'cpu':
            target_device = next(self.generator.parameters()).device
            frames_to_recache = frames_to_recache.to(target_device)
        batch_size = frames_to_recache.shape[0]
        print(f"num_recache_frames: {num_recache_frames}, recache_start_frame: {recache_start_frame}, current_start_frame: {current_start_frame}")
        
        # prepare blockwise causal mask
        device = frames_to_recache.device
        block_mask = self.generator.model._prepare_blockwise_causal_attn_mask(
            device=device,
            num_frames=num_recache_frames,
            frame_seqlen=self.frame_seq_length,
            num_frame_per_block=self.num_frame_per_block,
            local_attn_size=self.local_attn_size
        )

        context_timestep = torch.ones([batch_size, num_recache_frames],
                                    device=device, dtype=torch.int64) * self.args.context_noise
        # MEM_ANCHOR_T (arm i1_*, 2026-08-04): the declared noise level of the recached frames is
        # per frame already (e0 has shape (B, F, 6, C) -> one modulation per frame), so a single
        # slot can be declared at t* while the 11 true frames keep context_noise. Default None =
        # every existing arm byte-identical.
        if timestep_override is not None:
            # FOOTGUN (verifie 2026-08-04) : WanDiffusionWrapper.forward fait
            # `input_timestep = timestep[:, 0]` quand `uniform_timestep` est vrai, ce qui
            # ECRASERAIT silencieusement la declaration par-slot avec celle du slot 0 (une vraie
            # frame, t = 0), le hook deviendrait un simple bruitage, sans declaration, et le
            # bras i1 mesurerait autre chose que ce qu'il annonce. Ici is_causal=True donc
            # uniform_timestep est False ; on refuse de tourner dans le cas contraire.
            assert not getattr(self.generator, "uniform_timestep", False), \
                "MEM_ANCHOR_T exige un generateur CAUSAL (uniform_timestep=False) : sinon le " \
                "timestep par frame est reduit a celui du slot 0 et la declaration est perdue"
            assert timestep_override.shape[-1] == num_recache_frames, \
                f"timestep_override has {timestep_override.shape[-1]} entries, recache needs " \
                f"{num_recache_frames}"
            context_timestep = timestep_override.to(device=device).round().to(torch.int64) \
                .reshape(1, num_recache_frames).expand(batch_size, -1).contiguous()
            _d = [(i, int(x)) for i, x in enumerate(context_timestep[0].tolist())
                  if int(x) != int(self.args.context_noise)]
            print(f"[MEM_ANCHOR_T] recache timesteps: {len(_d)} slot(s) declared off-baseline "
                  f"{_d} (baseline {int(self.args.context_noise)}, {num_recache_frames} slots)",
                  flush=True)

        self.generator.model.block_mask = block_mask

        with torch.no_grad():
            self.generator(
                noisy_image_or_video=frames_to_recache,
                conditional_dict=new_conditional_dict,
                timestep=context_timestep,
                kv_cache=self.kv_cache1,
                crossattn_cache=self.crossattn_cache,
                current_start=recache_start_frame * self.frame_seq_length,
                rope_enhance=self.rope_enhance,
                sink_recache_after_switch=not self.global_sink,
                ttt_cache=getattr(self, "ttt_cache", None),
                ttt_update=False,
            )

        # reset cross-attention cache
        for blk in self.crossattn_cache:
            blk["k"].zero_()
            blk["v"].zero_()
            blk["is_init"] = False

    def inference(
        self,
        noise: torch.Tensor,
        *,
        text_prompts_first: List[str],
        text_prompts_second: List[str],
        switch_frame_index: int,
        text_prompts_third: Optional[List[str]] = None,
        second_switch_frame_index: Optional[int] = None,
        initial_latent: Optional[torch.Tensor] = None,
        return_latents: bool = False,
        low_memory: bool = False,
    ):
        """Generate video with prompt switch.

        Notes (differences from base inference):
            text_prompts_first: prompt list used before switching (aligned with batch)
            text_prompts_second: prompt list used after switching
            switch_frame_index: 0-based frame index; frames >= this value use the second prompt
            text_prompts_third: optional prompt list for a second switch (A->B->A revisit);
                defaults to text_prompts_first when second_switch_frame_index is set
            second_switch_frame_index: optional 0-based frame index; frames >= this value use
                the third prompt (must be > switch_frame_index). None = single-switch A->B
                behaviour, byte-identical to before.
        """
        batch_size, num_output_frames, num_channels, height, width = noise.shape
        assert num_output_frames % self.num_frame_per_block == 0
        num_blocks = num_output_frames // self.num_frame_per_block

        assert not int(os.environ.get("MEM_DECODE_COMPOSITE", "0") or 0), \
            "MEM_DECODE_COMPOSITE (arms oracle1comp / memdec21c) is implemented in the N-switch " \
            "driver only, run those arms through inference_schedule()"
        assert not int(os.environ.get("MEM_ADDITIVE_KV", "0") or 0), \
            "MEM_ADDITIVE_KV (arm oracle1add) is implemented in the N-switch driver only, " \
            "run it through inference_schedule()"
        assert int(os.environ.get("MEM_DECODE_SLOT_OFFSET", "-1") or -1) < 0, \
            "MEM_DECODE_SLOT_OFFSET (gate 2) is implemented in the N-switch driver only"
        assert float(os.environ.get("MEM_ANCHOR_T", "0") or 0) <= 0.0, \
            "MEM_ANCHOR_T (lot offensif 2026-08-04, arms i1_*) is implemented in the N-switch " \
            "driver only, run it through inference_schedule()"
        assert float(os.environ.get("MEM_SDEDIT", "0") or 0) <= 0.0 \
            and not int(os.environ.get("MEM_SINKPURGE", "0") or 0), \
            "MEM_SDEDIT / MEM_SINKPURGE (mini-batterie training-free 2026-08-04, arms " \
            "sdedit_ours / sdedit_wrong / sinkpurge) are implemented in the N-switch driver only"
        assert float(os.environ.get("MEM_DECODE_NOISE", "0") or 0) <= 0.0 \
            and not os.environ.get("MEM_DECODE_SRC", "").strip(), \
            "MEM_DECODE_NOISE / MEM_DECODE_SRC (reviewer controls noiseanchor / wronganchor, " \
            "2026-08-03) are implemented in the N-switch driver only"

        # Encode both prompts upfront
        cond_first = self.text_encoder(text_prompts=text_prompts_first)
        cond_second = self.text_encoder(text_prompts=text_prompts_second)
        # Optional A->B->A revisit: encode the third prompt (defaults to the first prompt)
        cond_third = None
        if second_switch_frame_index is not None:
            assert second_switch_frame_index > switch_frame_index, \
                "second_switch_frame_index must come after switch_frame_index"
            if text_prompts_third is None:
                text_prompts_third = text_prompts_first
            cond_third = self.text_encoder(text_prompts=text_prompts_third)

        if low_memory:
            gpu_memory_preservation = get_cuda_free_memory_gb(gpu) + 5
            move_model_to_device_with_memory_preservation(
                self.text_encoder, target_device=gpu, preserved_memory_gb=gpu_memory_preservation
            )

        # Place output on CPU to save GPU memory
        output_device = torch.device('cpu') if low_memory else noise.device
        output = torch.zeros(
            [batch_size, num_output_frames, num_channels, height, width],
            device=output_device,
            dtype=noise.dtype
        )

        local_attn_cfg = getattr(self.args.model_kwargs, "local_attn_size", -1)
        kv_policy = ""
        if local_attn_cfg != -1:
            # local attention
            kv_cache_size = local_attn_cfg * self.frame_seq_length
            kv_policy = f"int->local, size={local_attn_cfg}"
        else:
            # global attention
            kv_cache_size = num_output_frames * self.frame_seq_length
            kv_policy = "global (-1)"
        print(f"kv_cache_size: {kv_cache_size} (policy: {kv_policy}, frame_seq_length: {self.frame_seq_length}, num_output_frames: {num_output_frames})")

        self._initialize_kv_cache(
            batch_size,
            dtype=noise.dtype,
            device=noise.device,
            kv_cache_size_override=kv_cache_size
        )
        self._initialize_crossattn_cache(
            batch_size=batch_size,
            dtype=noise.dtype,
            device=noise.device
        )
        # fast-weight memory state, fresh per sequence (was missing, so switch inference ran the
        # stock model with the memory never read or written)
        self.ttt_cache = [TTTBlockCache() for _ in range(self.num_transformer_blocks)]
        # MEM_WRITER (arm memdec20, memory 2.0): overlay the offline-trained write parameters on
        # the memory cells before the first write of this rollout. Default off = no-op; consumes
        # no RNG draw, so the paired arms keep identical noise streams.
        _memwriter_sync(self.generator.model)

        current_start_frame = 0
        num_input_frames = 0
        self._switch_global_frame = None
        self._first_switch_frame = None
        self._ttt_snap_A = None
        self._thru_frame = None          # MEM_DECODE_THRU cache (arm oracle1thru), per rollout
        self._thru_info = None
        self._comp_gate_now = 1.0
        self._add_installed = False
        self._add_mass_log = []          # T1: masse par source, par chunk et par bloc
        self._clear_add_kv()             # no additive channel may survive from a previous arm
        self.generator.model.local_attn_size = self.local_attn_size
        print(f"[inference] local_attn_size set on model: {self.generator.model.local_attn_size}")
        self._set_all_modules_max_attention_size(self.local_attn_size)

        # Temporal denoising by blocks
        all_num_frames = [self.num_frame_per_block] * num_blocks

        using_second = False
        using_third = False
        for current_num_frames in all_num_frames:
            if (second_switch_frame_index is not None) and using_second and (not using_third) \
                    and (current_start_frame >= second_switch_frame_index):
                # MEM_EPISODIC probe (default off = byte-identical): restore the memory state
                # snapshotted at the A->B switch, before the recache, so the return-of-A rollout
                # (recache reads included) runs on the A-era state instead of B's recency.
                if int(os.environ.get("MEM_EPISODIC", "0")) and getattr(self, "_ttt_snap_A", None):
                    self._restore_ttt(self._ttt_snap_A)
                    print(f"[MEM_EPISODIC] restored A-era memory state at frame {current_start_frame}")
                # ORACLE_SPLICE probe (default off): recache the KV window from the original A tail
                # (the model's own generated past) instead of the B tail = self-oracle upper bound.
                _splice = None
                # arbiter probes: dump the original-A anchor latent for the twin arm (free,
                # no RNG draw, off by default) before any splice touches `output`.
                self._dump_anchor(output, self._first_switch_frame)
                if int(os.environ.get("ORACLE_SPLICE", "0")):
                    _n = current_start_frame if self.local_attn_size == -1 \
                        else min(self.local_attn_size, current_start_frame)
                    _a_end = self._first_switch_frame
                    assert _a_end is not None and _a_end - _n >= 0, \
                        f"oracle splice needs {_n} frames of original A before frame {_a_end}"
                    # ORACLE_SPLICE_FRAMES (compression-floor probe, default 0 = full window):
                    # only the last nf slots of the recached window carry original-A content, the
                    # rest keeps the true recent (B) tail -> measures how much A-KV is needed.
                    _nf = int(os.environ.get("ORACLE_SPLICE_FRAMES", "0") or 0)
                    # ORACLE_SPLICE_SLOT=k (position curve probe): the single A anchor replaces
                    # the window slot k frames before the most recent one (k=0 == oracle1's last
                    # slot; k~3-6 = readable mid-window, "a few moments ago"). -1/unset = off.
                    _slot_k = int(os.environ.get("ORACLE_SPLICE_SLOT", "-1") or -1)
                    if _slot_k >= 0:
                        _splice = output[:, current_start_frame - _n:current_start_frame].clone()
                        _pos = _n - 1 - min(_slot_k, _n - 1)
                        _splice[:, _pos:_pos + 1] = self._anchor_content(output, _a_end - 1, _a_end)
                        print(f"[ORACLE_SPLICE] slot probe: anchor at window slot {_pos}/{_n} "
                              f"({_slot_k} frames before most-recent)")
                    elif 0 < _nf < _n:
                        # ORACLE_SPLICE_POS=head (position probe): the A anchor occupies the
                        # OLDEST window slots (sink-adjacent band) instead of the most recent
                        # ones -> "distant past reference" instead of "the frame from a moment
                        # ago". Decisive for the cut-at-return question (quadrant).
                        if os.environ.get("ORACLE_SPLICE_POS", "") == "head":
                            _splice = torch.cat([
                                self._anchor_content(output, _a_end - _nf, _a_end),
                                output[:, current_start_frame - (_n - _nf):current_start_frame]],
                                dim=1)
                            print(f"[ORACLE_SPLICE] partial HEAD: first {_nf}/{_n} window slots "
                                  f"from original A [{_a_end - _nf}, {_a_end}), rest = recent tail")
                        else:
                            _splice = torch.cat([
                                output[:, current_start_frame - _n:current_start_frame - _nf],
                                self._anchor_content(output, _a_end - _nf, _a_end)], dim=1)
                            print(f"[ORACLE_SPLICE] partial: last {_nf}/{_n} window slots from original A "
                                  f"[{_a_end - _nf}, {_a_end}), rest = true recent tail")
                    else:
                        _splice = self._anchor_content(output, _a_end - _n, _a_end)
                        print(f"[ORACLE_SPLICE] recaching frames [{_a_end - _n}, {_a_end}) of original A "
                              f"at positions [{current_start_frame - _n}, {current_start_frame})")
                # ORACLE_SPLICE_NOISE (tolerance probe, default 0 = off): degrade the A-content
                # part of the splice with RMS-preserving noise mixing x' = sqrt(1-s^2)x + s*eps,
                # simulating an imperfect decoded recall frame. Dedicated RNG generator so the
                # main stream is untouched (the memzero pairing depends on identical draws).
                _dose = float(os.environ.get("ORACLE_SPLICE_NOISE", "0") or 0)
                if _dose > 0.0 and _splice is not None:
                    _nf_env = int(os.environ.get("ORACLE_SPLICE_FRAMES", "0") or 0)
                    _na = _nf_env if 0 < _nf_env < _splice.shape[1] else _splice.shape[1]
                    _g = torch.Generator(device=_splice.device)
                    _g.manual_seed(987654321 + current_start_frame)
                    _a_part = _splice[:, -_na:]
                    _eps = torch.randn(_a_part.shape, generator=_g, device=_a_part.device,
                                       dtype=torch.float32).to(_a_part.dtype)
                    _a_deg = ((1.0 - _dose ** 2) ** 0.5) * _a_part + _dose * _eps
                    _splice = torch.cat([_splice[:, :-_na], _a_deg], dim=1)
                    print(f"[ORACLE_SPLICE_NOISE] degraded last {_na} splice frame(s), sigma={_dose}")
                # MEM_DECODE (arm memdec, design v1): the anchor frame is decoded from the live
                # memory states by the offline-trained AnchorFrameDecoder, then spliced at the
                # recency position (the only one that works, position curve 2026-07-24) --
                # same mechanism as oracle1 but the anchor is synthesized, not the real past.
                _dec_ckpt = os.environ.get("MEM_DECODE", "").strip()
                if _dec_ckpt:
                    assert _splice is None, "MEM_DECODE and ORACLE_SPLICE are mutually exclusive"
                    _n = current_start_frame if self.local_attn_size == -1 \
                        else min(self.local_attn_size, current_start_frame)
                    if int(os.environ.get("MEM_DECODE_THRU", "0") or 0):
                        # arm oracle1thru: the frame was already decoded at the eviction of the
                        # anchor chunk (write age 1); nothing is decoded here.
                        assert self._thru_frame is not None, \
                            "MEM_DECODE_THRU: the anchor chunk was never evicted before the " \
                            "return switch (check switch indices / local_attn_size)"
                        _frame, _cond_on = self._thru_frame, None
                        print(f"[MEM_DECODE_THRU] splicing the frame decoded at eviction "
                              f"{self._thru_info}", flush=True)
                    else:
                        _frame, _cond_on = self._memdec_frame(_dec_ckpt, cond_third,
                                                              output.dtype, output.device)
                    _cond = _cond_on
                    # MEM_DECODE_SLOTS (default 1): repeat the decoded anchor over the last k
                    # window slots, evicts B from the readable context k times faster (window-
                    # mixing fix; oracle3 proved multi-slot anchors keep the same re-ID score).
                    _kslots = max(1, int(os.environ.get("MEM_DECODE_SLOTS", "1") or 1))
                    # MEM_DECODE_DUMP (debug/eye-check): save the decoded anchor latent so it
                    # can be VAE-decoded and inspected (is the anchor clean-A or an A/B fusion?)
                    _dump = os.environ.get("MEM_DECODE_DUMP", "").strip()
                    if _dump:
                        torch.save(_frame.detach().to("cpu", torch.float16),
                                   f"{_dump}.f{current_start_frame}.pt")
                        print(f"[MEM_DECODE_DUMP] anchor latent saved -> {_dump}.f{current_start_frame}.pt")
                    _splice = torch.cat(
                        [output[:, current_start_frame - (_n - _kslots):current_start_frame]]
                        + [_frame] * _kslots, dim=1)
                    print(f"[MEM_DECODE] decoded anchor spliced at recency ({_kslots}/{_n} slots), "
                          f"cond={'thru' if _cond is None else ('ON' if _cond else 'off')}",
                          flush=True)
                _blocks_env = os.environ.get("ORACLE_SPLICE_BLOCKS", "").strip()
                if _splice is not None and _blocks_env:
                    # oracle3b probe: the ORACLE recache is kept only on the listed blocks;
                    # every other block keeps a normal (recent-tail) recache. Measures whether
                    # KV-of-A at the memory blocks alone carries the resurrection (the full
                    # oracle splices all 30 blocks, the virtual sink would only feed [5,16,27]).
                    # Both recache passes process the same frames at the same positions, so the
                    # cache index tensors evolve identically: restoring k/v alone is exact.
                    _keep = {int(b) for b in _blocks_env.split(",") if b != ""}
                    self._recache_after_switch(output, current_start_frame, cond_third)
                    _snap_kv = {bi: {kk: self.kv_cache1[bi][kk].detach().to("cpu", copy=True)
                                     for kk in ("k", "v")}
                                for bi in range(self.num_transformer_blocks) if bi not in _keep}
                    self._recache_after_switch(output, current_start_frame, cond_third,
                                               frames_override=_splice)
                    with torch.no_grad():
                        for bi, kv in _snap_kv.items():
                            self.kv_cache1[bi]["k"].copy_(kv["k"].to(self.kv_cache1[bi]["k"].device))
                            self.kv_cache1[bi]["v"].copy_(kv["v"].to(self.kv_cache1[bi]["v"].device))
                    del _snap_kv
                    print(f"[ORACLE_SPLICE_BLOCKS] oracle KV kept ONLY on blocks {sorted(_keep)}, "
                          f"other blocks restored to the normal recache")
                elif os.environ.get("ORACLE_SPLICE_POS", "") == "sink0" and _splice is not None:
                    # persistent-position probe: normal recache (window = true recent tail),
                    # then the anchor frame overwrites the sink slot (read forever, never evicted)
                    self._recache_after_switch(output, current_start_frame, cond_third)
                    self._write_sink_anchor(_splice[:, -1:], cond_third)
                    print(f"[ORACLE_SPLICE] sink0: anchor frame written into the SINK slot "
                          f"(persistent), window = normal recent tail")
                else:
                    self._recache_after_switch(output, current_start_frame, cond_third,
                                               frames_override=_splice)
                cond_in_use = cond_third
                using_third = True
                self._switch_global_frame = current_start_frame
                print("second_switch_frame_index", second_switch_frame_index)
                print("current_start_frame", current_start_frame)
            elif (not using_second) and (current_start_frame >= switch_frame_index):
                # MEM_EPISODIC probe: freeze the A-era memory state before B overwrites it.
                if int(os.environ.get("MEM_EPISODIC", "0")):
                    self._ttt_snap_A = self._snapshot_ttt()
                    print(f"[MEM_EPISODIC] snapshotted A-era memory state at frame {current_start_frame}")
                self._first_switch_frame = current_start_frame
                self._recache_after_switch(output, current_start_frame, cond_second)
                cond_in_use = cond_second
                using_second = True
                self._switch_global_frame = current_start_frame
                print("switch_frame_index", switch_frame_index)
                print("current_start_frame", current_start_frame)
            else:
                cond_in_use = (cond_third if using_third
                               else (cond_second if using_second else cond_first))
            
            noisy_input = noise[:, current_start_frame - num_input_frames : current_start_frame + current_num_frames - num_input_frames]

            # Spatial denoising loop (same as parent but uses cond_in_use)
            for index, current_timestep in enumerate(self.denoising_step_list):
                timestep = torch.ones([batch_size, current_num_frames], device=noise.device, dtype=torch.int64) * current_timestep
                if index < len(self.denoising_step_list) - 1:
                    _, denoised_pred = self.generator(
                        noisy_image_or_video=noisy_input,
                        conditional_dict=cond_in_use,
                        timestep=timestep,
                        kv_cache=self.kv_cache1,
                        crossattn_cache=self.crossattn_cache,
                        current_start=current_start_frame * self.frame_seq_length,
                        rope_enhance=self.rope_enhance,
                        ttt_cache=self.ttt_cache,
                        ttt_update=False,
                        switch_global_frame=self._switch_global_frame,
                    )
                    next_timestep = self.denoising_step_list[index + 1]
                    noisy_input = self.scheduler.add_noise(
                        denoised_pred.flatten(0, 1),
                        torch.randn_like(denoised_pred.flatten(0, 1)),
                        next_timestep * torch.ones([batch_size * current_num_frames], device=noise.device, dtype=torch.long),
                    ).unflatten(0, denoised_pred.shape[:2])
                else:
                    _, denoised_pred = self.generator(
                        noisy_image_or_video=noisy_input,
                        conditional_dict=cond_in_use,
                        timestep=timestep,
                        kv_cache=self.kv_cache1,
                        crossattn_cache=self.crossattn_cache,
                        current_start=current_start_frame * self.frame_seq_length,
                        rope_enhance=self.rope_enhance,
                        ttt_cache=self.ttt_cache,
                        ttt_update=False,
                        switch_global_frame=self._switch_global_frame,
                    )

            output[:, current_start_frame : current_start_frame + current_num_frames] = denoised_pred.to(output.device)

            # rerun with clean context noise for cache update
            context_timestep = torch.ones_like(timestep) * self.args.context_noise
            self.generator(
                noisy_image_or_video=denoised_pred,
                conditional_dict=cond_in_use,
                timestep=context_timestep,
                kv_cache=self.kv_cache1,
                crossattn_cache=self.crossattn_cache,
                current_start=current_start_frame * self.frame_seq_length,
                rope_enhance=self.rope_enhance,
                ttt_cache=self.ttt_cache,
                ttt_update=True,
                switch_global_frame=self._switch_global_frame,
            )

            # MEM_DECODE_THRU (arm oracle1thru, default off = no-op): the write that stores the
            # anchor chunk has just happened -> decode and cache the frame now (age 1 write).
            self._thru_maybe_decode(current_start_frame, cond_third, output.dtype, output.device)

            current_start_frame += current_num_frames

        # Standard decoding (to_cpu mirrors causal_inference.py: stack decoded chunks in RAM
        # instead of VRAM for long videos)
        video = self.vae.decode_to_pixel(output.to(noise.device), use_cache=False, to_cpu=low_memory)
        video = (video * 0.5 + 0.5).clamp(0, 1)

        if return_latents:
            return video, output
        return video

    # ==========================================================================================
    # N-switch driver (scenario matrix, reference/runs/reid_scen/), ADDITIVE.
    # `inference()` above is not touched: every existing campaign stays bitwise reproducible by
    # construction. `inference_schedule()` is the generalization, and selftest (a) of
    # reference/runs/reid_scen/selftest_scenarios.py proves the two produce bitwise identical
    # latents when the schedule is the legacy [(0,A), (sw1,B), (sw2,A)] one, on every arm whose
    # hook lives in the switch block (memzero / oracle / oracle1 / oracle3b / memdec20).
    # ==========================================================================================
    @staticmethod
    def _memdec_window(output, current_start_frame, n, anchor, kslots, slot_offset):
        """The 12-frame recache window carrying the recalled anchor.

        Window convention (fixed 2026-08-02, gate 2). The historical MEM_DECODE path built
            [output[cur - (n - k) : cur]] + [anchor] * k
        i.e. it kept the true most recent frame cur-1 and appended the anchor after it, so every
        true frame of the window sat one slot EARLIER than in the oracle path, which builds
            [output[cur - n : cur - k]] + [anchor] * k
        (the anchor replaces the most recent true frame). The two arms were therefore not
        measuring the same thing at all: memdec20's window duplicated the recency while
        oracle1's replaced it. Slot_offset >= 0 selects the oracle convention:
            slot_offset = 0  anchor at the recency slot, the 11 other slots = output[cur-12:cur-1]
                             -> index-identical to the oracle1 splice (selftest)
            slot_offset = j  the anchor is pushed j slots back and the true frames keep the
                             recency (the model's immediate past stays real)
        slot_offset < 0 keeps the legacy (asymmetric) construction so the 144 mp4s already on
        disk stay reproducible."""
        if slot_offset < 0:
            return torch.cat(
                [output[:, current_start_frame - (n - kslots):current_start_frame]]
                + [anchor] * kslots, dim=1)
        win = output[:, current_start_frame - n:current_start_frame].clone()
        pos = max(0, n - kslots - slot_offset)
        win[:, pos:pos + kslots] = anchor.to(dtype=win.dtype, device=win.device)
        return win

    def _switch_recache(self, output, current_start_frame, cond_new, *, a_end=None,
                        seg_idx=None, prompt_idx=None, is_revisit=False, subject=None,
                        cond_prev=None):
        """Cache surgery at one schedule boundary.

        is_revisit=False -> plain switch: no ORACLE/MEM_DECODE hook fires and, with FRAMEBANK
        off, the call is byte-identical to the legacy first switch.
        is_revisit=True  -> the entering prompt has been seen before: the recall hooks fire.
        `a_end` = end latent frame (exclusive) of the segment where the entering subject first
        appeared, i.e. the anchor frame is `a_end - 1`. In the legacy A->B->A rollout that is
        `self._first_switch_frame`, so this is a strict generalization of the block in
        `inference()` (same statements, same order, same tensors).

        Note on FRAMEBANK: its router runs at every transition, revisit or not, Echo-Forcing
        has no oracle telling it which switch is a revisit, so a false recall on a fresh scene
        is a real failure mode of the baseline and must be measurable, not hidden.
        """
        assert not int(os.environ.get("MEM_DECODE_THRU", "0") or 0), \
            "MEM_DECODE_THRU (arm oracle1thru) is not supported by the N-switch driver: its " \
            "eviction arithmetic assumes the single legacy anchor, run it through inference()"
        _splice = None
        _ts_over = None                  # MEM_ANCHOR_T: per-slot declared timestep of the recache
        _comp_before = getattr(self, "_comp_applied", 0)
        if is_revisit:
            assert a_end is not None and a_end >= 1, "revisit without a first-appearance segment"
            # MEM_EPISODIC: restore the state snapshotted when this subject was last left.
            _snaps = getattr(self, "_ttt_snap_by_prompt", None) or {}
            _sk = prompt_idx if subject is None else subject
            if int(os.environ.get("MEM_EPISODIC", "0")) and _snaps.get(_sk):
                self._restore_ttt(_snaps[_sk])
                print(f"[MEM_EPISODIC] restored the memory state of subject {_sk} at frame "
                      f"{current_start_frame}")
            self._dump_anchor(output, a_end)
            if int(os.environ.get("MEM_COMP_PROMPT_GATE", "0") or 0):
                self._comp_gate_from_prompts(cond_new, cond_prev)
        # ---- MEM_ADDITIVE_KV (gate 1): the anchor enters as extra k/v tokens, no splice ------
        _addk = bool(is_revisit and int(os.environ.get("MEM_ADDITIVE_KV", "0") or 0))
        if _addk:
            assert not int(os.environ.get("ORACLE_SPLICE", "0") or 0) and not self._fb_mode(), \
                "MEM_ADDITIVE_KV is the NO-EVICTION channel: it must not be combined with a " \
                "splice arm (ORACLE_SPLICE / FRAMEBANK)"
            _ck = os.environ.get("MEM_DECODE", "").strip()
            if _ck:                       # future arm memdec20add: the decoded anchor, additive
                _content, _ = self._memdec_frame(_ck, cond_new, output.dtype, output.device)
            else:                         # arm oracle1add: the true anchor frame, additive
                _content = self._anchor_content(output, a_end - 1, a_end)
            self._install_add_kv(_content, cond_new)
        if is_revisit and int(os.environ.get("ORACLE_SPLICE", "0")):
            _n = current_start_frame if self.local_attn_size == -1 \
                else min(self.local_attn_size, current_start_frame)
            _a_end = a_end
            _nf = int(os.environ.get("ORACLE_SPLICE_FRAMES", "0") or 0)
            _need = _nf if 0 < _nf < _n else _n
            assert _a_end - _need >= 0, \
                f"oracle splice needs {_need} frames of the first appearance before frame {_a_end}"
            _slot_k = int(os.environ.get("ORACLE_SPLICE_SLOT", "-1") or -1)
            if _slot_k >= 0:
                _splice = output[:, current_start_frame - _n:current_start_frame].clone()
                _pos = _n - 1 - min(_slot_k, _n - 1)
                _splice[:, _pos:_pos + 1] = self._anchor_content(output, _a_end - 1, _a_end)
                print(f"[ORACLE_SPLICE] slot probe: anchor at window slot {_pos}/{_n} "
                      f"({_slot_k} frames before most-recent)")
            elif 0 < _nf < _n:
                if os.environ.get("ORACLE_SPLICE_POS", "") == "head":
                    _splice = torch.cat([
                        self._anchor_content(output, _a_end - _nf, _a_end),
                        output[:, current_start_frame - (_n - _nf):current_start_frame]], dim=1)
                    print(f"[ORACLE_SPLICE] partial HEAD: first {_nf}/{_n} window slots "
                          f"from first appearance [{_a_end - _nf}, {_a_end}), rest = recent tail")
                else:
                    _splice = torch.cat([
                        output[:, current_start_frame - _n:current_start_frame - _nf],
                        self._maybe_composite(
                            self._anchor_content(output, _a_end - _nf, _a_end),
                            output, current_start_frame)], dim=1)
                    print(f"[ORACLE_SPLICE] partial: last {_nf}/{_n} window slots from the first "
                          f"appearance [{_a_end - _nf}, {_a_end}), rest = true recent tail")
            else:
                _splice = self._anchor_content(output, _a_end - _n, _a_end)
                print(f"[ORACLE_SPLICE] recaching frames [{_a_end - _n}, {_a_end}) of the first "
                      f"appearance at positions [{current_start_frame - _n}, {current_start_frame})")
        _dose = float(os.environ.get("ORACLE_SPLICE_NOISE", "0") or 0)
        if _dose > 0.0 and _splice is not None:
            _nf_env = int(os.environ.get("ORACLE_SPLICE_FRAMES", "0") or 0)
            _na = _nf_env if 0 < _nf_env < _splice.shape[1] else _splice.shape[1]
            _g = torch.Generator(device=_splice.device)
            _g.manual_seed(987654321 + current_start_frame)
            _a_part = _splice[:, -_na:]
            _eps = torch.randn(_a_part.shape, generator=_g, device=_a_part.device,
                               dtype=torch.float32).to(_a_part.dtype)
            _a_deg = ((1.0 - _dose ** 2) ** 0.5) * _a_part + _dose * _eps
            _splice = torch.cat([_splice[:, :-_na], _a_deg], dim=1)
            print(f"[ORACLE_SPLICE_NOISE] degraded last {_na} splice frame(s), sigma={_dose}")
        # ---- FRAMEBANK (Echo-Forcing-style real-content recall), routes at every switch ------
        if self._fb_mode():
            assert _splice is None and not os.environ.get("MEM_DECODE", "").strip(), \
                "FRAMEBANK, ORACLE_SPLICE and MEM_DECODE are mutually exclusive"
            _n = current_start_frame if self.local_attn_size == -1 \
                else min(self.local_attn_size, current_start_frame)
            # FRAMEBANK_GT (pilote texte-insuffisant): le QUAND = la verite du planning, pour que
            # framebank, memdec20 et wronganchor splicent EXACTEMENT au meme instant (egalite
            # stricte). Le routeur tourne quand meme et sa decision est loguee.
            _gt = bool(int(os.environ.get("FRAMEBANK_GT", "0") or 0))
            _fb_frame, _ = self._fb_select(seg_idx, prompt_idx, cond_new, subject=subject,
                                           force=(bool(is_revisit) if _gt else None))
            if _gt and not is_revisit:
                _fb_frame = None
            if _fb_frame is not None:
                # same placement as oracle1 (ORACLE_SPLICE_FRAMES=1): the recalled frame takes
                # the most-recent window slot, the rest of the window stays the true recent tail.
                _splice = torch.cat(
                    [output[:, current_start_frame - _n:current_start_frame - 1],
                     _fb_frame.to(dtype=output.dtype, device=output.device)], dim=1)
                print(f"[FRAMEBANK] recalled frame spliced at recency (1/{_n} slots)", flush=True)
        # ---- MEM_DECODE (memory 2.0 decoded anchor) ------------------------------------------
        # when it fires. Default (MEM_DECODE_ROUTED unset): the schedule's ground-truth revisit
        # flag, correct on the matrix (verified: exactly 1 splice in S1/S2/S3, 2 in S4) but an
        # ORACLE signal that the framebank baseline does not get. MEM_DECODE_ROUTED=1 (arm
        # memdec20r) replaces it with the framebank router's own decision (prompt cosine, tau,
        # non-adjacency, same FRAMEBANK_TAU/FRAMEBANK_TIE defaults), so the two arms differ only
        # by what they splice: a decoded anchor vs a stored real frame.
        _fires = is_revisit
        if int(os.environ.get("MEM_DECODE_ROUTED", "0") or 0):
            assert not self._fb_mode(), \
                "MEM_DECODE_ROUTED and FRAMEBANK are mutually exclusive (one router, one arm)"
            _, _rev = self._fb_select(seg_idx, prompt_idx, cond_new, subject=subject, mode="key")
            _fires = bool(_rev["recalled"])
            print(f"[MEM_DECODE_ROUTED] segment {seg_idx}: router says "
                  f"{'REVISIT -> splice' if _fires else 'fresh scene -> no splice'} "
                  f"(ground truth was {'revisit' if is_revisit else 'fresh'})", flush=True)
        _dec_ckpt = os.environ.get("MEM_DECODE", "").strip() if (_fires and not _addk) else ""
        if _dec_ckpt:
            assert _splice is None, "MEM_DECODE and ORACLE_SPLICE are mutually exclusive"
            _n = current_start_frame if self.local_attn_size == -1 \
                else min(self.local_attn_size, current_start_frame)
            _frame, _cond_on = self._memdec_frame(_dec_ckpt, cond_new, output.dtype, output.device)
            _cond = _cond_on
            _kslots = max(1, int(os.environ.get("MEM_DECODE_SLOTS", "1") or 1))
            _dump = os.environ.get("MEM_DECODE_DUMP", "").strip()
            if _dump:
                # ECRITURE ATOMIQUE (2026-08-04) : un autre bras/agent peut RELIRE ce fichier au
                # meme instant (wronganchor lit le dump d'une autre cellule). torch.save direct
                # laisse une fenetre ou le .pt est tronque -> ecriture dans .tmp puis os.replace,
                # qui est atomique sur le meme systeme de fichiers.
                _dp = f"{_dump}.f{current_start_frame}.pt"
                torch.save(_frame.detach().to("cpu", torch.float16), _dp + ".tmp")
                os.replace(_dp + ".tmp", _dp)
                print(f"[MEM_DECODE_DUMP] anchor latent saved -> {_dp}")
            # reviewer controls (2026-08-03), both after the dump so MEM_DECODE_DUMP always
            # persists the raw anchor decoded from this cell's state (that is what the
            # `wronganchor` arm of another cell reads back), and before the composite so the
            # controls compose with it exactly like the real content would.
            _frame = self._maybe_wrong_anchor(_frame)
            _frame = self._maybe_noise_anchor(_frame, output, current_start_frame, _n)
            # MEM_SDEDIT (arms sdedit_ours / sdedit_wrong): sharpen the anchor with the model's
            # own prior. After the content controls on purpose, sdedit_wrong is exactly
            # wronganchor + the same sharpening, so the paired contrast isolates the content.
            _frame = self._maybe_sdedit(_frame, cond_new, current_start_frame)
            # MEM_DECODE_COMPOSITE (arm memdec21c): the decoded anchor keeps only the subject
            # region; the border comes from the current context. Applied after MEM_DECODE_DUMP so
            # that hook keeps dumping the raw decoder output.
            _frame = self._maybe_composite(_frame, output, current_start_frame)
            _soff = int(os.environ.get("MEM_DECODE_SLOT_OFFSET", "-1") or -1)
            _splice = self._memdec_window(output, current_start_frame, _n, _frame, _kslots, _soff)
            # MEM_ANCHOR_T (arms i1_*): stop lying about the anchor's noise level. OFF by default,
            # and it returns the splice object itself then, so every arm on disk is unchanged.
            _splice, _ts_over = self._maybe_declare_anchor_noise(
                _splice, _n, _kslots, _soff, current_start_frame)
            print(f"[MEM_DECODE] decoded anchor spliced ({_kslots}/{_n} slots, window convention "
                  f"{'LEGACY (recency duplicated)' if _soff < 0 else f'oracle, slot_offset={_soff}'}"
                  f"), cond={'thru' if _cond is None else ('ON' if _cond else 'off')}", flush=True)
        if int(os.environ.get("MEM_DECODE_COMPOSITE", "0") or 0) and _splice is not None:
            assert getattr(self, "_comp_applied", 0) > _comp_before, \
                "MEM_DECODE_COMPOSITE=1 but the splice built at this switch carries NO " \
                "composited anchor, unsupported placement (SLOT / POS=head / full window / " \
                "FRAMEBANK content)"
        _blocks_env = os.environ.get("ORACLE_SPLICE_BLOCKS", "").strip()
        assert _ts_over is None or not (_blocks_env
                                        or os.environ.get("ORACLE_SPLICE_POS", "") == "sink0"), \
            "MEM_ANCHOR_T is defined for the plain recency splice (it declares the noise level " \
            "of the slot the recache writes), not for ORACLE_SPLICE_BLOCKS / POS=sink0"
        if _splice is not None and _blocks_env:
            _keep = {int(b) for b in _blocks_env.split(",") if b != ""}
            self._recache_after_switch(output, current_start_frame, cond_new)
            _snap_kv = {bi: {kk: self.kv_cache1[bi][kk].detach().to("cpu", copy=True)
                             for kk in ("k", "v")}
                        for bi in range(self.num_transformer_blocks) if bi not in _keep}
            self._recache_after_switch(output, current_start_frame, cond_new,
                                       frames_override=_splice)
            with torch.no_grad():
                for bi, kv in _snap_kv.items():
                    self.kv_cache1[bi]["k"].copy_(kv["k"].to(self.kv_cache1[bi]["k"].device))
                    self.kv_cache1[bi]["v"].copy_(kv["v"].to(self.kv_cache1[bi]["v"].device))
            del _snap_kv
            print(f"[ORACLE_SPLICE_BLOCKS] oracle KV kept ONLY on blocks {sorted(_keep)}, "
                  f"other blocks restored to the normal recache")
        elif os.environ.get("ORACLE_SPLICE_POS", "") == "sink0" and _splice is not None:
            self._recache_after_switch(output, current_start_frame, cond_new)
            self._write_sink_anchor(_splice[:, -1:], cond_new)
            print(f"[ORACLE_SPLICE] sink0: anchor frame written into the SINK slot "
                  f"(persistent), window = normal recent tail")
        else:
            self._recache_after_switch(output, current_start_frame, cond_new,
                                       frames_override=_splice, timestep_override=_ts_over)
        # MEM_SINKPURGE (arm sinkpurge): after the recache, so it rewrites the sink slots of the
        # cache the return will actually read. No-op unless the hook is on (default off).
        self._maybe_sinkpurge(output, current_start_frame, cond_new, is_revisit)

    def inference_schedule(
        self,
        noise: torch.Tensor,
        *,
        prompts: List[str],
        schedule: List,
        subjects: Optional[List[int]] = None,
        return_latents: bool = False,
        low_memory: bool = False,
    ):
        """N-switch generation.

        Prompts  : the distinct prompt strings, indexed by the schedule.
        schedule : [(start_frame, prompt_idx), ...], strictly increasing start frames, the first
                   one 0, every start frame a multiple of num_frame_per_block. A segment whose
                   subject already appeared earlier is a revisit: the recall hooks fire and the
                   anchor is taken from the segment of that subject's first appearance.
        subjects : optional ground-truth identity per segment (same length as schedule).
                   Defaults to the prompt indices, i.e. "same prompt string = same subject".
                   Give it explicitly when a revisit is described by a paraphrase: then the
                   conditioning differs but the subject (hence the re-ID reference and the
                   frame-bank ceiling) is the same, and the router has a real matching job.

        Legacy equivalence: schedule [(0, 0), (sw1, 1), (sw2, 0)] with prompts [A, B] reproduces
        `inference(noise, text_prompts_first=[A], text_prompts_second=[B], switch_frame_index=sw1,
        second_switch_frame_index=sw2)` bitwise (selftest a). The text encoder is called once per
        schedule entry, in schedule order, exactly the legacy call sequence (A, B, A).
        """
        batch_size, num_output_frames, num_channels, height, width = noise.shape
        assert num_output_frames % self.num_frame_per_block == 0
        num_blocks = num_output_frames // self.num_frame_per_block

        assert not int(os.environ.get("MEM_DECODE_THRU", "0") or 0), \
            "MEM_DECODE_THRU (arm oracle1thru) is not supported by the N-switch driver: its " \
            "eviction arithmetic assumes the single legacy anchor, run it through inference()"
        schedule = [(int(f), int(p)) for f, p in schedule]
        assert schedule and schedule[0][0] == 0, "schedule must start at frame 0"
        for i, (f, p) in enumerate(schedule):
            assert 0 <= p < len(prompts), f"schedule[{i}] prompt index {p} out of range"
            assert f % self.num_frame_per_block == 0, \
                f"schedule[{i}] start frame {f} is not a multiple of {self.num_frame_per_block}"
            if i:
                assert f > schedule[i - 1][0], "schedule start frames must be strictly increasing"
        assert schedule[-1][0] < num_output_frames, "last segment is empty"
        seg_prompt = [p for _, p in schedule]
        seg_subj = [int(x) for x in (subjects if subjects is not None else seg_prompt)]
        assert len(seg_subj) == len(schedule), \
            f"subjects has {len(seg_subj)} entries for {len(schedule)} segments"
        # first appearance of each subject + the end frame of that segment (= anchor frame + 1)
        seg_end = [schedule[i + 1][0] for i in range(len(schedule) - 1)] + [num_output_frames]
        first_app = {}
        for i, s in enumerate(seg_subj):
            first_app.setdefault(s, i)
        is_revisit = [i > 0 and first_app[s] < i for i, s in enumerate(seg_subj)]
        print(f"[schedule] {len(schedule)} segments, {len(prompts)} distinct prompts, subjects "
              f"{seg_subj}, revisits at segments "
              f"{[i for i, r in enumerate(is_revisit) if r]}", flush=True)
        for i, (f, p) in enumerate(schedule):
            print(f"[schedule]   seg{i}: frames [{f}, {seg_end[i]})  prompt{p} subject{seg_subj[i]}"
                  f"{'  REVISIT of seg' + str(first_app[seg_subj[i]]) if is_revisit[i] else ''}"
                  f"  '{prompts[p][:60]}'", flush=True)

        # one text-encoder call per schedule entry, in schedule order (legacy call sequence).
        conds = [self.text_encoder(text_prompts=[prompts[p]]) for _, p in schedule]

        if low_memory:
            gpu_memory_preservation = get_cuda_free_memory_gb(gpu) + 5
            move_model_to_device_with_memory_preservation(
                self.text_encoder, target_device=gpu, preserved_memory_gb=gpu_memory_preservation
            )

        output_device = torch.device('cpu') if low_memory else noise.device
        output = torch.zeros(
            [batch_size, num_output_frames, num_channels, height, width],
            device=output_device,
            dtype=noise.dtype
        )

        local_attn_cfg = getattr(self.args.model_kwargs, "local_attn_size", -1)
        kv_policy = ""
        if local_attn_cfg != -1:
            kv_cache_size = local_attn_cfg * self.frame_seq_length
            kv_policy = f"int->local, size={local_attn_cfg}"
        else:
            kv_cache_size = num_output_frames * self.frame_seq_length
            kv_policy = "global (-1)"
        print(f"kv_cache_size: {kv_cache_size} (policy: {kv_policy}, frame_seq_length: {self.frame_seq_length}, num_output_frames: {num_output_frames})")

        self._initialize_kv_cache(
            batch_size, dtype=noise.dtype, device=noise.device,
            kv_cache_size_override=kv_cache_size)
        self._initialize_crossattn_cache(
            batch_size=batch_size, dtype=noise.dtype, device=noise.device)
        self.ttt_cache = [TTTBlockCache() for _ in range(self.num_transformer_blocks)]
        _memwriter_sync(self.generator.model)

        current_start_frame = 0
        num_input_frames = 0
        self._switch_global_frame = None
        self._first_switch_frame = schedule[1][0] if len(schedule) > 1 else None
        self._ttt_snap_A = None
        self._ttt_snap_by_prompt = {}
        self._thru_frame = None
        self._thru_info = None
        self._comp_applied = 0           # MEM_DECODE_COMPOSITE bookkeeping, per rollout
        self._comp_gate_now = 1.0
        self._add_installed = False
        self._add_mass_log = []          # T1: masse par source, par chunk et par bloc
        self._clear_add_kv()             # no additive channel may survive from a previous arm
        self._fb_reset()
        self.generator.model.local_attn_size = self.local_attn_size
        print(f"[inference] local_attn_size set on model: {self.generator.model.local_attn_size}")
        self._set_all_modules_max_attention_size(self.local_attn_size)

        all_num_frames = [self.num_frame_per_block] * num_blocks

        seg = 0
        for current_num_frames in all_num_frames:
            if seg + 1 < len(schedule) and current_start_frame >= schedule[seg + 1][0]:
                prev_seg, seg = seg, seg + 1
                # FRAMEBANK 'preserve': the segment that just ended enters the scene pool
                # (before the recall, so the just-left scene is in the bank and excluded as the
                # adjacent one, exactly Echo-Forcing's routing rule).
                self._fb_store(output, current_start_frame, prev_seg, seg_prompt[prev_seg],
                               conds[prev_seg], subject=seg_subj[prev_seg])
                # MEM_EPISODIC: freeze the state of a first appearance that will be revisited.
                if int(os.environ.get("MEM_EPISODIC", "0")) \
                        and first_app[seg_subj[prev_seg]] == prev_seg \
                        and seg_subj[prev_seg] in seg_subj[prev_seg + 1:]:
                    self._ttt_snap_by_prompt[seg_subj[prev_seg]] = self._snapshot_ttt()
                    print(f"[MEM_EPISODIC] snapshotted the state of subject "
                          f"{seg_subj[prev_seg]} at frame {current_start_frame}")
                self._switch_recache(
                    output, current_start_frame, conds[seg],
                    a_end=seg_end[first_app[seg_subj[seg]]],
                    seg_idx=seg, prompt_idx=seg_prompt[seg], subject=seg_subj[seg],
                    is_revisit=is_revisit[seg], cond_prev=conds[prev_seg])
                self._switch_global_frame = current_start_frame
                print(f"[schedule] switch to segment {seg} (prompt {seg_prompt[seg]}) at frame "
                      f"{current_start_frame}", flush=True)
            cond_in_use = conds[seg]

            noisy_input = noise[:, current_start_frame - num_input_frames : current_start_frame + current_num_frames - num_input_frames]

            for index, current_timestep in enumerate(self.denoising_step_list):
                timestep = torch.ones([batch_size, current_num_frames], device=noise.device, dtype=torch.int64) * current_timestep
                if index < len(self.denoising_step_list) - 1:
                    _, denoised_pred = self.generator(
                        noisy_image_or_video=noisy_input,
                        conditional_dict=cond_in_use,
                        timestep=timestep,
                        kv_cache=self.kv_cache1,
                        crossattn_cache=self.crossattn_cache,
                        current_start=current_start_frame * self.frame_seq_length,
                        rope_enhance=self.rope_enhance,
                        ttt_cache=self.ttt_cache,
                        ttt_update=False,
                        switch_global_frame=self._switch_global_frame,
                    )
                    next_timestep = self.denoising_step_list[index + 1]
                    noisy_input = self.scheduler.add_noise(
                        denoised_pred.flatten(0, 1),
                        torch.randn_like(denoised_pred.flatten(0, 1)),
                        next_timestep * torch.ones([batch_size * current_num_frames], device=noise.device, dtype=torch.long),
                    ).unflatten(0, denoised_pred.shape[:2])
                else:
                    _, denoised_pred = self.generator(
                        noisy_image_or_video=noisy_input,
                        conditional_dict=cond_in_use,
                        timestep=timestep,
                        kv_cache=self.kv_cache1,
                        crossattn_cache=self.crossattn_cache,
                        current_start=current_start_frame * self.frame_seq_length,
                        rope_enhance=self.rope_enhance,
                        ttt_cache=self.ttt_cache,
                        ttt_update=False,
                        switch_global_frame=self._switch_global_frame,
                    )

            output[:, current_start_frame : current_start_frame + current_num_frames] = denoised_pred.to(output.device)

            context_timestep = torch.ones_like(timestep) * self.args.context_noise
            self.generator(
                noisy_image_or_video=denoised_pred,
                conditional_dict=cond_in_use,
                timestep=context_timestep,
                kv_cache=self.kv_cache1,
                crossattn_cache=self.crossattn_cache,
                current_start=current_start_frame * self.frame_seq_length,
                rope_enhance=self.rope_enhance,
                ttt_cache=self.ttt_cache,
                ttt_update=True,
                switch_global_frame=self._switch_global_frame,
            )

            # T1 (lot offensif 2026-08-04): trace de la masse d'attention par source, un point par
            # chunk genere, par bloc. Sans MEM_ADD_MASS_SRC=1 le report vaut None et rien n'est
            # accumule (aucun cout, aucun changement de sortie).
            if getattr(self, "_add_installed", False):
                _src = self._add_mass_src_report()
                if _src is not None:
                    self._add_mass_log.append({"frame": int(current_start_frame),
                                               "segment": int(seg), "per_block": _src})

            current_start_frame += current_num_frames

        # MEM_ADDITIVE_KV tripwire: how much attention mass did the extra tokens actually take on
        # the last forward? A channel read at ~0 % is dead by construction, not by content.
        _mass = self._add_mass_report()
        if _mass:
            _v = [m for m in _mass.values() if m == m]
            print(f"[MEM_ADDITIVE_KV] attention mass of the anchor tokens on the last forward: "
                  f"mean {sum(_v) / max(1, len(_v)):.4f}, min {min(_v):.4f}, max {max(_v):.4f} "
                  f"| per block " + " ".join(f"{b}:{m:.3f}" for b, m in sorted(_mass.items())),
                  flush=True)
        # T1: masse par source [sink | fenetre | ancre]. Resume + dump complet (bloc x pas).
        _log = getattr(self, "_add_mass_log", None)
        if _log:
            _post = [e for e in _log if e["frame"] > (self._switch_global_frame or 0)]
            _use = _post or _log
            _agg = {k: sum(b[k] for e in _use for b in e["per_block"].values())
                    / max(1, sum(len(e["per_block"]) for e in _use))
                    for k in ("sink", "window", "anchor")}
            _ntok = next(iter(_use[-1]["per_block"].values()))["n_tok"]
            print(f"[MEM_ADD_MASS_SRC] masse d'attention par SOURCE apres le retour "
                  f"({len(_use)} pas x {len(_use[-1]['per_block'])} blocs) : "
                  f"sink {_agg['sink']:.4f} | fenetre {_agg['window']:.4f} | "
                  f"ANCRE {_agg['anchor']:.4f}   (tokens {_ntok} = sink/fenetre/ancre, "
                  f"part uniforme de l'ancre "
                  f"{_ntok[2] / max(1, sum(_ntok)):.4f})", flush=True)
            _dump = os.environ.get("MEM_ADD_MASS_DUMP", "").strip()
            if _dump:
                os.makedirs(os.path.dirname(_dump) or ".", exist_ok=True)
                with open(_dump, "w") as _f:
                    json.dump({"summary_after_return": _agg, "n_tok": _ntok, "steps": _log}, _f)
                print(f"[MEM_ADD_MASS_SRC] trace complete -> {_dump}", flush=True)

        # VRAM (2026-08-04): the KV and cross-attention caches are dead here, the generation loop
        # is over and nothing reads them again (the driver only inspects `ttt_cache`, left intact).
        # Releasing them returns ~3.5 GB (30 blocks x 2 x 18720 x 12 x 128 bf16) right before the
        # single biggest allocation of the rollout, the full-clip VAE decode, exactly where a
        # co-run on the same card makes us OOM. numerically inert by construction: freeing memory
        # that is never read again cannot change a single output bit.
        if low_memory:
            import gc
            self.kv_cache1 = None
            self.crossattn_cache = None
            self._clear_add_kv()
            gc.collect()
            torch.cuda.empty_cache()
        video = self.vae.decode_to_pixel(output.to(noise.device), use_cache=False, to_cpu=low_memory)
        video = (video * 0.5 + 0.5).clamp(0, 1)

        if return_latents:
            return video, output
        return video 