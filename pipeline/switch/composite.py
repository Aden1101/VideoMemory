"""Bringing back only the subject of the anchor, and keeping the current background."""
import os

import torch

from pipeline.switch.noise import private_noise
from utils.env import env_float, env_int, env_str


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
    #   MEM_DECODE_COMPOSITE=1  turn it on   (0 or unset: the anchor is spliced whole)
    #   MEM_COMP_CROP=0.6       fraction of H and W covered by the ellipse at m = 0.5 (the
    #                           center-60 % of the re-ID crop by default)
    #   MEM_COMP_FEATHER=0.3    relative half-width of the smoothstep ring around the ellipse
    #   MEM_COMP_DUMP=<prefix>  debug: save {bg, anchor, composite, mask} for visual inspection
    @staticmethod
    def _composite_mask(H, W, crop, feather, device, dtype, cy=None, cx=None):
        """Soft elliptical mask on the latent grid: 1 = take the anchor (the subject),
        0 = keep the current context (the background). The blend weights are the pair
        (m, 1 - m), so they sum to 1 EVERYWHERE by construction. smoothstep => C1 at both ends
        of the ring, hence no seam in the decoded pixels.
        (cy, cx) = ellipse center in PIXELS of the latent grid; None = the geometric center
        (the default, which every composite arm uses unless it passes a center)."""
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
        """Crude subject localizer on a latent frame, used by the UNION mask: the
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

    @staticmethod
    def _seeded_noise(like, current_start_frame):
        """Unit noise shaped like `like`, from its own generator seeded by the switch frame: the
        main random stream is not touched, so the paired arms keep identical noise."""
        return private_noise(like, 192837465 + current_start_frame).to(like.dtype)

    @staticmethod
    def _fill_weights(mode, m, m_a, m_b, floor):
        """(w_anchor, w_noise) under the union mask m = max(m_a, m_b), the background taking
        1 - m, so the three weights sum to 1 everywhere. (None, None) when mode is empty.

        Where the union covers the current subject but not the anchor, pasting the anchor
        leaves a blurred ghost of the old scene that the denoiser only half cleans.
          noise     erase that region with noise, so the prompt repaints it freely
          priority  the ghost mostly sits under the anchor's own ellipse, where the two
                    subjects overlap, so noise takes priority there too:
                    w_n = m_b, w_a = max(0, m_a - m_b)
          floor     priority also cuts the identity channel exactly where the subjects
                    overlap, so keep at least a fraction f of the anchor everywhere:
                    w_a = max(m_a - m_b, f * m_a), w_n = m - w_a. f = 0 is priority, f = 1 noise.
        """
        if mode == "noise":
            w_noise = (m - m_a).clamp_min(0.0)
            return m - w_noise, w_noise
        if mode == "priority":
            return (m_a - m_b).clamp_min(0.0), m_b
        if mode == "floor":
            w_anchor = torch.maximum(m_a - m_b, floor * m_a).clamp_min(0.0)
            return w_anchor, m - w_anchor
        if mode:
            raise ValueError(
                f"MEM_COMP_FILL={mode!r} (expected ''|'noise'|'priority'|'floor')")
        return None, None

    def _reset_composite(self):
        """Per rollout: how many composites were applied, which the driver checks against the
        splices it built, and the dose the prompt gate last computed."""
        self._comp_applied = 0
        self._comp_gate_now = 1.0

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
                                    is legitimate montage grammar.
          MEM_COMP_GATE_LO/HI       the ramp bounds (defaults 0.50 / 0.80)"""
        a = env_float("MEM_COMP_ALPHA", 1)
        if env_int("MEM_COMP_PROMPT_GATE"):
            a *= float(getattr(self, "_comp_gate_now", 1.0))
        return max(0.0, min(1.0, a))

    def _comp_gate_from_prompts(self, cond_new, cond_prev):
        """Prompt-cosine ramp, computed once per revisit switch (no RNG draw, CPU only)."""
        lo = env_float("MEM_COMP_GATE_LO", 0.50)
        hi = env_float("MEM_COMP_GATE_HI", 0.80)
        if cond_prev is None:
            self._comp_gate_now = 1.0
            return
        c = float(torch.dot(self._fb_key(cond_new), self._fb_key(cond_prev)))
        self._comp_gate_now = max(0.0, min(1.0, (c - lo) / max(hi - lo, 1e-6)))
        print(f"[MEM_COMP_PROMPT_GATE] cos(prompt entering, prompt being left) = {c:.4f} "
              f"-> composite dose alpha = {self._comp_gate_now:.3f} (ramp {lo}..{hi})", flush=True)

    def _maybe_composite(self, anchor, output, current_start_frame):
        """Blend the recalled anchor (the subject) onto the last true latent frame of the current
        context (the background). Returns `anchor` itself, the same object, when the hook is off.

        Controls, to tell whether the geometry of the mask helps or the content of the anchor:
          MEM_COMP_SHIFT=f         ellipse centre moved by f * W: the wrong place
          MEM_COMP_INVERT=1        m -> 1 - m: anchor at the border, context at the centre
          MEM_COMP_UNIFORM=1       constant m, the mean of the ellipse: same dose, no shape
          MEM_COMP_ANCHOR_NOISE=s  RMS-preserving noise mixed into the anchor: mask kept,
                                   content destroyed
          MEM_COMP_UNION=1         mask = union of an ellipse on the anchor's subject and one on
                                   the current subject, so the recalled subject is not pasted
                                   over the current one; MEM_COMP_FILL says what goes where the
                                   two differ (see _fill_weights)"""
        if not env_int("MEM_DECODE_COMPOSITE"):
            return anchor
        assert os.environ.get("REPLAY_PAST_POS", "") not in ("head", "sink0"), \
            "MEM_DECODE_COMPOSITE is defined for the RECENCY placement only (the background it " \
            "blends into is the CURRENT context, which only makes sense at the recency slot)"
        assert current_start_frame >= 1, "MEM_DECODE_COMPOSITE before any frame was produced"
        bg = output[:, current_start_frame - 1:current_start_frame].to(
            device=anchor.device, dtype=anchor.dtype)
        assert bg.shape[0] == anchor.shape[0] and bg.shape[2:] == anchor.shape[2:], \
            f"composite shape mismatch: bg {tuple(bg.shape)} vs anchor {tuple(anchor.shape)}"
        crop, feather = env_float("MEM_COMP_CROP", 0.6), env_float("MEM_COMP_FEATHER", 0.3)
        shift, invert = env_float("MEM_COMP_SHIFT"), env_int("MEM_COMP_INVERT")
        uniform, union = env_int("MEM_COMP_UNIFORM"), env_int("MEM_COMP_UNION")
        sig = env_float("MEM_COMP_ANCHOR_NOISE")
        H, W = anchor.shape[-2:]

        def ellipse(cy=None, cx=None):
            return self._composite_mask(H, W, crop, feather, anchor.device, torch.float32, cy, cx)

        m = ellipse(cx=(W - 1) / 2.0 + shift * W if shift else None)
        w_anchor = w_noise = None
        fill = ""
        if union:
            ay, ax = self._energy_centroid(anchor)
            by, bx = self._energy_centroid(bg)
            m_a, m_b = ellipse(ay, ax), ellipse(by, bx)
            m = torch.maximum(m_a, m_b)
            print(f"[MEM_COMP_UNION] anchor centroid ({ay}, {ax}) U background centroid "
                  f"({by}, {bx}), union mask mean {float(m.mean()):.3f}", flush=True)
            fill = (os.environ.get("MEM_COMP_FILL", "") or "").strip().lower()
            w_anchor, w_noise = self._fill_weights(fill, m, m_a, m_b,
                                                   env_float("MEM_COMP_FILL_FLOOR", 0.4))
        if invert:
            m = 1.0 - m
        if uniform:
            m = torch.full_like(m, float(m.mean()))

        alpha = self._comp_alpha()      # the dose: 0 gives back the anchor bitwise, 1 the full composite
        a_use = anchor
        if sig > 0:
            a_use = ((1.0 - min(sig, 1.0) ** 2) ** 0.5) * anchor \
                + min(sig, 1.0) * self._seeded_noise(anchor, current_start_frame)
            print(f"[MEM_COMP_ANCHOR_NOISE] anchor content degraded with sigma={sig} "
                  f"(mask untouched)", flush=True)
        if w_noise is None:
            if alpha != 1.0:
                m = 1.0 - alpha * (1.0 - m)
            comp = (m * a_use.float() + (1.0 - m) * bg.float()).to(anchor.dtype)
        else:
            assert not (invert or uniform), \
                "MEM_COMP_FILL is defined for the plain union mask, not with invert or uniform"
            comp = self._blend_with_fill(a_use, bg, m, w_anchor, w_noise, alpha, fill,
                                         current_start_frame)

        self._comp_applied = getattr(self, "_comp_applied", 0) + 1
        dump = env_str("MEM_COMP_DUMP")
        if dump:
            os.makedirs(os.path.dirname(dump) or ".", exist_ok=True)
            torch.save({"bg": bg.detach().cpu(), "anchor": anchor.detach().cpu(),
                        "composite": comp.detach().cpu(), "mask": m.detach().cpu(),
                        "crop": crop, "feather": feather,
                        "bg_frame": int(current_start_frame - 1)},
                       f"{dump}.f{current_start_frame}.pt")
            print(f"[MEM_COMP_DUMP] composite parts -> {dump}.f{current_start_frame}.pt", flush=True)
        controls = (f" | CONTROLS shift={shift} invert={invert} uniform={uniform} union={union} "
                    f"alpha={alpha:.3f} anchor_noise={sig}"
                    if (shift or invert or uniform or union or sig or alpha != 1.0) else "")
        print(f"[MEM_DECODE_COMPOSITE] anchor ({anchor.shape[1]} frame(s)) composited onto the "
              f"background of true frame {current_start_frame - 1}: elliptical mask crop={crop} "
              f"feather={feather}, subject weight mean {float(m.mean()):.3f}{controls}", flush=True)
        return comp

    def _blend_with_fill(self, anchor, bg, m, w_anchor, w_noise, alpha, fill, current_start_frame):
        """anchor, noise and background under three weights that sum to 1 everywhere:
        w_anchor, w_noise and 1 - m. alpha scales the two that are not the anchor, so alpha = 0
        still gives back the pure anchor. The noise is the draw MEM_COMP_ANCHOR_NOISE uses,
        mixed with the anchor by MEM_COMP_FILL_SIGMA (1 = pure noise)."""
        sigma = min(env_float("MEM_COMP_FILL_SIGMA", 1.0), 1.0)
        filler = ((1.0 - sigma ** 2) ** 0.5) * anchor \
            + sigma * self._seeded_noise(anchor, current_start_frame)
        w_a, w_n, w_bg = w_anchor, w_noise, 1.0 - m
        if alpha != 1.0:
            w_n, w_bg = alpha * w_n, alpha * w_bg
            w_a = 1.0 - alpha * (1.0 - w_a)
        print(f"[MEM_COMP_FILL={fill}] current-subject region (mean weight {float(w_n.mean()):.3f}) "
              f"filled with noise sigma={sigma} instead of anchor content; anchor "
              f"{float(w_a.mean()):.3f}, background {float(w_bg.mean()):.3f}", flush=True)
        return (w_a * anchor.float() + w_n * filler.float() + w_bg * bg.float()).to(anchor.dtype)
