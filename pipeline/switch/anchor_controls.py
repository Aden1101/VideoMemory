"""Degrading or replacing the anchor, to find out what about it actually helps."""
import os

import torch

from pipeline.switch.noise import private_noise
from utils.env import env_float, env_int, env_str


def _load_anchor(path):
    """An anchor latent from disk: either the {"anchor": tensor, "frame": i} dict that
    ANCHOR_DUMP writes, or a bare tensor as MEM_DECODE_DUMP writes."""
    t = torch.load(path, map_location="cpu", weights_only=False)
    return t["anchor"] if isinstance(t, dict) and "anchor" in t else t


class AnchorControlsMixin:
    def _blur_latent(self, lat, sigma):
        """Blur a real frame: decode it to pixels, blur, re-encode. Used to find how much detail
        the returning anchor actually needs. ANCHOR_BLUR_SIGMA sets the blur radius.
        The point is to degrade a true anchor down to the fidelity our own decoder reaches,
        which sits beyond a 32-pixel blur. sigma=0 is a pure encode-decode round trip, so it
        measures what the round trip alone costs."""
        import torchvision.transforms.functional as TVF
        dev = self._generator_device
        assert lat.shape[1] == 1, \
            f"ANCHOR_BLUR_SIGMA supports ONE latent frame (VAE temporal compression), got {lat.shape[1]}"
        px = self.vae.decode_to_pixel(lat.to(device=dev, dtype=torch.bfloat16),
                                      use_cache=False)          # (B,1,3,H,W) in [-1,1]
        if sigma > 0:
            k = int(6 * sigma) | 1                              # odd kernel ~ +-3 sigma
            px = TVF.gaussian_blur(px[:, 0], kernel_size=[k, k], sigma=[sigma, sigma])[:, None]
        out = self.vae.encode_to_latent(px.permute(0, 2, 1, 3, 4).to(torch.bfloat16))
        self.vae.model.clear_cache()
        print(f"[ANCHOR_BLUR] anchor frame VAE round trip with sigma={sigma} px "
              f"(kernel {int(6 * sigma) | 1 if sigma > 0 else 0})", flush=True)
        return out.to(dtype=lat.dtype, device=lat.device)

    def _anchor_content(self, output, lo, hi):
        """The original-A content that gets spliced at the return: output[:, lo:hi] by default.
        ANCHOR_FROM_FILE replaces it with a latent frame loaded from disk:
        the original-A anchor of another seed of the same prompt pair: same prompt, same scene
        grammar, same sharpness, wrong instance. ANCHOR_BLUR_SIGMA blurs
        the true frame instead. With both off, the true frame is returned as it is."""
        real = output[:, lo:hi]
        src = env_str("ANCHOR_FROM_FILE")
        sig = env_str("ANCHOR_BLUR_SIGMA")
        assert not (src and sig), "ANCHOR_FROM_FILE and ANCHOR_BLUR_SIGMA are mutually exclusive"
        if src:
            assert hi - lo == 1, f"ANCHOR_FROM_FILE replaces ONE frame, asked for {hi - lo}"
            t = _load_anchor(src)
            assert tuple(t.shape) == tuple(real.shape), \
                f"ANCHOR_FROM_FILE {src}: shape {tuple(t.shape)} != {tuple(real.shape)}"
            print(f"[ANCHOR_FROM_FILE] anchor content REPLACED by {src} "
                  f"(other seed's original-A frame)", flush=True)
            return t.to(dtype=real.dtype, device=real.device)
        if sig:
            return self._blur_latent(real, float(sig))
        return real

    def _dump_anchor(self, output, a_end):
        """ANCHOR_DUMP: save the anchor latent of the first appearance (the exact frame the
        replay splices) so another cell can splice the same frame. Free side-effect, no RNG draw.
        If the file already exists it is checked bitwise instead of rewritten: every
        MEM_ZERO_ALL arm of a given (pair, seed) must produce the same segment A."""
        path = env_str("ANCHOR_DUMP")
        if not path:
            return
        assert a_end is not None and a_end >= 1, "ANCHOR_DUMP before the first switch"
        # native dtype (no fp16 cast): the twin arm must be able to splice it back bitwise,
        # so that the same frame can be spliced back unchanged.
        cur = output[:, a_end - 1:a_end].detach().cpu().clone()
        if os.path.exists(path):
            old = _load_anchor(path)
            d = float((old.float() - cur.float()).abs().max())
            print(f"[ANCHOR_DUMP] existing {os.path.basename(path)} re-checked: "
                  f"max|diff| = {d:.3e}", flush=True)
            assert d == 0.0, \
                f"ANCHOR_DUMP: segment A differs from the stored anchor ({path}, {d:.3e})"
            return
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({"anchor": cur, "frame": int(a_end - 1)}, path + ".tmp")
        os.replace(path + ".tmp", path)
        print(f"[ANCHOR_DUMP] original-A anchor latent (frame {a_end - 1}) -> {path}",
              flush=True)

    # ---- controls: does the anchor help because of what it contains, or just by being there? ----
    # Both replace the content of the anchor and change nothing else: same moment, same slot,
    # same decoder call. Off by default, and the helpers then return the anchor unchanged.
    #
    #   MEM_DECODE_NOISE=s     replace the anchor with noise calibrated on the current window's
    #                          own statistics. At s = 1 nothing of the anchor is left, so what
    #                          still helps is only "erase that slot and let the prompt repaint".
    #   MEM_DECODE_SRC=path    use the anchor decoded from another video's memory: same decoder,
    #                          same fidelity, same statistics, wrong content.
    def _maybe_wrong_anchor(self, frame):
        """MEM_DECODE_SRC: the anchor decoded from another video's memory instead of this one,
        with its cosine and relative distance to the right one logged. `frame` when off."""
        src = env_str("MEM_DECODE_SRC")
        if not src:
            return frame
        assert env_float("MEM_DECODE_NOISE") <= 0.0, \
            "MEM_DECODE_SRC and MEM_DECODE_NOISE are two different control arms, not a stack"
        assert os.path.exists(src), f"MEM_DECODE_SRC: no such anchor dump {src}"
        t = _load_anchor(src)
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
        """MEM_DECODE_NOISE=s: mix the anchor with noise drawn at the per-channel mean and
        standard deviation of the n true window frames, keeping its RMS. `frame` when off."""
        s = env_float("MEM_DECODE_NOISE")
        if s <= 0.0:
            return frame
        s = min(s, 1.0)
        assert current_start_frame >= 1, "MEM_DECODE_NOISE before any frame was produced"
        ref = output[:, max(0, current_start_frame - n):current_start_frame].to(
            device=frame.device, dtype=torch.float32)
        mu = ref.mean(dim=(0, 1, 3, 4), keepdim=True)      # per latent channel (dim 2)
        sd = ref.std(dim=(0, 1, 3, 4), keepdim=True)
        # MEM_DECODE_NOISE_SEED is set per run, so paired runs do not all share one realisation
        eps = private_noise(frame, env_int("MEM_DECODE_NOISE_SEED", 192837465) + current_start_frame)
        noise = mu + sd * eps
        out = (((1.0 - s ** 2) ** 0.5) * frame.float() + s * noise).to(frame.dtype)
        print(f"[MEM_DECODE_NOISE] anchor content REPLACED by noise calibrated on the {ref.shape[1]} "
              f"true window frames (dose {s}, per-channel mu in [{float(mu.min()):+.3f}, "
              f"{float(mu.max()):+.3f}], sigma in [{float(sd.min()):.3f}, {float(sd.max()):.3f}]); "
              f"anchor std {float(frame.float().std()):.3f} -> spliced std "
              f"{float(out.float().std()):.3f}", flush=True)
        return out

    # ---- declaring the anchor's real noise level ------------------------------------------
    # A splice declares the anchor as a clean frame (t = 0) although it is blurry and off the
    # data manifold: the model reads it as a clean frame of its immediate past, and the current
    # context absorbs it. Here the anchor is noised to a level t and that same t is declared
    # for its slots, two steps that only make sense together. The anchor is then an ordinary
    # noisy latent, which the model saw throughout training.
    #
    #   MEM_ANCHOR_T=0/unset   off, and every arm is unchanged
    #   MEM_ANCHOR_T=t         t in (0, 1000], the level both applied and declared
    #   MEM_ANCHOR_T_SEED=n    seed of that noise
    @staticmethod
    def _anchor_slots(n, kslots, slot_offset):
        """[lo, hi): the slots of the recache window that the anchor occupies, in both
        conventions of `_memdec_window`: the last k slots when slot_offset < 0, or those k
        slots pushed `slot_offset` further into the past otherwise."""
        if slot_offset < 0:
            return n - kslots, n
        lo = max(0, n - kslots - slot_offset)
        return lo, lo + kslots

    def _maybe_declare_anchor_noise(self, splice, n, kslots, slot_offset, current_start_frame):
        """Returns (window, timestep_override). When the hook is off it returns the window object
        itself and None, so anything already rendered stays reproducible. Draws from its own
        generator, never from the global random stream."""
        t = env_float("MEM_ANCHOR_T")
        if t <= 0.0:
            return splice, None
        assert 0.0 < t <= 1000.0, f"MEM_ANCHOR_T must be in (0, 1000], got {t}"
        assert splice is not None and splice.shape[1] == n, \
            f"MEM_ANCHOR_T: window of {n} slots expected, got {None if splice is None else splice.shape[1]}"
        lo, hi = self._anchor_slots(n, kslots, slot_offset)
        part = splice[:, lo:hi]
        dev = part.device
        eps = private_noise(part, env_int("MEM_ANCHOR_T_SEED", 77400241) + int(current_start_frame))
        tt = torch.ones([part.shape[0] * part.shape[1]], device=dev, dtype=torch.float32) * t
        noised = self.scheduler.add_noise(part.flatten(0, 1).float(), eps.flatten(0, 1), tt) \
            .unflatten(0, part.shape[:2]).to(part.dtype)
        out = splice.clone()
        out[:, lo:hi] = noised
        ts = torch.full((n,), float(self.args.context_noise), dtype=torch.float32)
        ts[lo:hi] = t
        rel = float((noised.float() - part.float()).norm() / part.float().norm().clamp_min(1e-6))
        assert rel > 1e-3, "MEM_ANCHOR_T fired but the anchor content did not change"
        print(f"[MEM_ANCHOR_T] anchor slot(s) [{lo}, {hi}) of {n} NOISED to t={t:g} "
              f"(sigma~{t / 1000:.3f}, rel-L2 {rel:.3f}) AND declared at t={t:g}; the {n - (hi - lo)} "
              f"true frames keep t={int(self.args.context_noise)}", flush=True)
        return out, ts
