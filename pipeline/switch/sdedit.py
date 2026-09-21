"""Re-noising the anchor and denoising it again, to sharpen what the decoder produced."""

import torch

from pipeline.switch.noise import SDEDIT_SEED
from utils.env import env_float, env_int, env_str


class SdeditMixin:

    # ---- sharpening the decoded anchor ---------------------------------------------------
    # Two explanations survive the controls. Either the anchor is too blurry for the model to
    # read, or only the slot matters and the content is never read at all. Sharpening the anchor
    # with the model's own prior removes the first: content from the memory, sharpness from the
    # prior. If it still does not help, the content was never the point.
    #
    # Done in latent space, the denoiser being the sharpener: add noise to level s with the
    # pipeline's own scheduler, then denoise along the sampler's remaining steps. The sharpening
    # runs on a fresh one-frame cache, so it sees only the anchor and the return prompt.
    # s = 1.0 is pure prompt generation with no anchor content left.
    #
    #   MEM_SDEDIT=s        0 or unset = off, and the anchor comes back untouched
    #   MEM_SDEDIT_SEED=n   per-run seed, so paired runs do not share one noise draw
    def _sdedit_steps(self, sigma):
        """[t_start] + the sampler's own remaining timesteps strictly below it. With the
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

        Block size (MEM_SDEDIT_FRAMES, default = num_frame_per_block). A block of one latent
        frame comes out as high-frequency mosaic, even at sigma = 1.0, which is the model's own
        full sampler and must render a clean image: a 1-frame block is out of distribution for
        this backbone, trained and cached with num_frame_per_block = 3, where frames inside a
        block attend to each other. So the anchor is replicated to a native-size block and the
        middle frame is read back. MEM_SDEDIT_FRAMES=1 is kept as the control that shows it."""
        dev = self._generator_device
        assert frame.shape[1] == 1, \
            f"MEM_SDEDIT sharpens ONE latent frame, got {frame.shape[1]}"
        nf = env_int("MEM_SDEDIT_FRAMES") or self.num_frame_per_block
        x0 = frame.to(device=dev, dtype=torch.bfloat16).repeat(1, nf, 1, 1, 1)
        B = x0.shape[0]
        g = torch.Generator(device=dev)
        g.manual_seed(env_int("MEM_SDEDIT_SEED", SDEDIT_SEED)
                       + int(current_start_frame))
        steps = self._sdedit_steps(sigma)

        def _renoise(x, t):
            e = torch.randn(x.shape, generator=g, device=dev, dtype=torch.float32).to(x.dtype)
            tt = torch.ones([x.shape[0] * x.shape[1]], device=dev, dtype=torch.float32) * t
            return self.scheduler.add_noise(x.flatten(0, 1), e.flatten(0, 1), tt) \
                       .unflatten(0, x.shape[:2])

        xt = _renoise(x0, steps[0])
        tmp = self._scratch_kv_cache(B, nf, x0.dtype, dev)
        self._reset_crossattn_cache()     # the return prompt, freshly cross-attended
        with torch.no_grad():
            for i, t in enumerate(steps):
                for c in tmp:                     # every call is an independent 1-frame denoise
                    c["k"].zero_()
                    c["v"].zero_()
                    c["global_end_index"].zero_()
                    c["local_end_index"].zero_()
                _, x0_pred = self.generator(
                    noisy_image_or_video=xt,
                    conditional_dict=cond,
                    timestep=torch.ones([B, nf], device=dev, dtype=torch.float32) * t,
                    kv_cache=tmp,
                    crossattn_cache=self.crossattn_cache,
                    current_start=0,
                    rope_enhance=self.rope_enhance,
                    ttt_cache=self.ttt_cache,
                    ttt_update=False,
                )
                if i + 1 < len(steps):
                    xt = _renoise(x0_pred, steps[i + 1])
        del tmp
        self._reset_crossattn_cache()     # nothing of the phantom pass may survive
        # the middle frame of the block: frames inside a block attend to each other, and latent
        # frame 0 of a Wan clip is the odd one out (1 pixel frame vs 4 for the others).
        pick = nf // 2
        return x0_pred[:, pick:pick + 1].to(dtype=frame.dtype, device=frame.device)

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
        """MEM_SDEDIT=s: the anchor sharpened by _sdedit_frame, with how far it moved and how
        much high-frequency content it gained logged. `frame` when off."""
        s = env_float("MEM_SDEDIT")
        if s <= 0.0:
            return frame
        assert 0.0 < s <= 1.0, f"MEM_SDEDIT must be in (0, 1], got {s}"
        assert env_str("MEM_DECODE"), \
            "MEM_SDEDIT sharpens the MEM_DECODE anchor: it needs MEM_DECODE=<decoder ckpt>"
        out = self._sdedit_frame(frame, cond, s, current_start_frame)
        a, b = out.float().flatten(), frame.float().flatten()
        cos = float(torch.dot(a, b) / (a.norm() * b.norm() + 1e-12))
        rel = float((a - b).norm() / (b.norm() + 1e-12))
        hf0, hf1 = self._lat_hf(frame), self._lat_hf(out)
        print(f"[MEM_SDEDIT] anchor SHARPENED with the model prior (sigma={s}, trajectory "
              f"{[round(t, 1) for t in self._sdedit_steps(s)]}, "
              f"{env_int('MEM_SDEDIT_FRAMES') or self.num_frame_per_block}-frame block, return prompt): "
              f"cos with the raw anchor {cos:.4f}, rel-L2 {rel:.4f}, latent |laplacian| "
              f"{hf0:.4f} -> {hf1:.4f} (x{hf1 / max(hf0, 1e-9):.2f}), std "
              f"{float(frame.float().std()):.3f} -> {float(out.float().std()):.3f}", flush=True)
        return out
