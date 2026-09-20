# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES
#
# Licensed under the Apache License, Version 2.0 (the "License").
# You may not use this file except in compliance with the License.
# To view a copy of this license, visit http://www.apache.org/licenses/LICENSE-2.0
#
# No warranties are given. The work is provided "AS IS", without warranty of any kind, express or implied.
#
# SPDX-License-Identifier: Apache-2.0
"""Bringing back only the subject of the anchor, and keeping the current background."""
import os

import torch




class CompositeMixin:

    # ---- bringing back the subject without the old scene -------------------------------------
    # The splice replaces a whole latent frame, so the return re-renders the entire scene from
    # the past: a shop awning flips colour at the return, while the frame-bank baseline brings
    # the subject back into an intact scene. The border metric saw it, the re-ID crop did not.
    # The fix is to composite the spliced latent instead of overwriting it:
    #     composite = m * anchor (the subject, decoded or real) + (1 - m) * background
    # with background = the last true latent frame of the current context (the frame produced just
    # before the return switch, i.e. exactly the frame the replay evicts) and m a soft elliptical
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
        of the ring, hence no seam in the decoded pixels.
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
        is what makes the prompt gate safe to ship.
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
        """Blend the recalled anchor (the subject) onto the last true latent frame of the current
        context (background). Returns `anchor` itself (same object) when the hook is off, so the
        legacy code path is untouched."""
        if not int(os.environ.get("MEM_DECODE_COMPOSITE", "0") or 0):
            return anchor
        assert os.environ.get("REPLAY_PAST_POS", "") not in ("head", "sink0"), \
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
        # ---- controls: is it the mask's geometry that helps, or the anchor's content? --------
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
        w_noise = w_anchor = None   # MEM_COMP_FILL: weights of the current-subject region and
                                    # of the anchor region
        if shift:
            cx = (W - 1) / 2.0 + shift * W
        m = self._composite_mask(H, W, crop, feather, anchor.device, torch.float32, cy, cx)
        if union:
            # the mask must cover the current subject too, otherwise the recalled one is pasted
            # on top of it and the frame ends up with two heads. Without cross-attention, take the
            # union of the two ellipses, one around each energy centroid.
            ay, ax = self._energy_centroid(anchor)
            by, bx = self._energy_centroid(bg)
            m_a = self._composite_mask(H, W, crop, feather, anchor.device, torch.float32, ay, ax)
            m_b = self._composite_mask(H, W, crop, feather, anchor.device, torch.float32, by, bx)
            m = torch.maximum(m_a, m_b)
            print(f"[MEM_COMP_UNION] anchor centroid ({ay}, {ax}) U background centroid "
                  f"({by}, {bx}), union mask mean {float(m.mean()):.3f}", flush=True)
            # Where the union mask covers the current subject but not the anchor, pasting the
            # anchor leaves a blurred ghost of the old scene that the denoiser only half cleans.
            # MEM_COMP_FILL=noise erases that region instead, so the prompt repaints it freely.
            # MEM_COMP_FILL=priority goes further: the ghost mostly sits UNDER the anchor's own
            # ellipse, where the two subjects overlap, so noise takes priority there too:
            #     w_n = m_b,  w_a = max(0, m_a - m_b),  w_bg = 1 - max(m_a, m_b)   (they sum to 1)
            _fill = (os.environ.get("MEM_COMP_FILL", "") or "").strip().lower()
            if _fill == "noise":
                w_noise = (m - m_a).clamp_min(0.0)
                w_anchor = m - w_noise
            elif _fill == "priority":
                w_noise = m_b
                w_anchor = (m_a - m_b).clamp_min(0.0)
            elif _fill == "floor":
                # 'priority' kills the ghost but also cuts the identity channel exactly where the
                # two subjects overlap. Keep at least a fraction f of the anchor everywhere:
                #     w_a = max(m_a - m_b, f * m_a),  w_n = m_union - w_a,  w_bg = 1 - m_union
                # f = 0 is exactly 'priority', f = 1 is exactly 'noise'.
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
            # the MEM_COMP_FILL path applies alpha itself, on its three weights
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
            # three weights that sum to 1 everywhere by construction:
            #   w_a  = m_a          the anchor, inside its own ellipse
            #   w_n  = m - m_a      the current subject's region, filled with noise
            #   w_bg = 1 - m        the world, untouched
            # alpha scales the two non-anchor weights, so alpha = 0 still gives the pure anchor.
            assert not (invert or uniform), \
                "MEM_COMP_FILL=noise is defined for the plain union mask, not with invert or uniform"
            assert w_anchor is not None
            _fs = min(float(os.environ.get("MEM_COMP_FILL_SIGMA", "1.0") or 1.0), 1.0)
            _gf = torch.Generator(device=anchor.device)
            _gf.manual_seed(192837465 + current_start_frame)      # same seed as the noise above
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
