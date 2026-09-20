"""Re-noising the anchor and denoising it again, to sharpen what the decoder produced."""

import torch

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
        nf = env_int("MEM_SDEDIT_FRAMES") or self.num_frame_per_block
        x0 = frame.to(device=dev, dtype=torch.bfloat16).repeat(1, nf, 1, 1, 1)
        B = x0.shape[0]
        _g = torch.Generator(device=dev)
        _g.manual_seed(env_int("MEM_SDEDIT_SEED", 5150241)
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
              f"{[round(t, 1) for t in self._sdedit_steps(s)]}, 1-frame mode, return prompt): "
              f"cos with the raw anchor {cos:.4f}, rel-L2 {rel:.4f}, latent |laplacian| "
              f"{hf0:.4f} -> {hf1:.4f} (x{hf1 / max(hf0, 1e-9):.2f}), std "
              f"{float(frame.float().std()):.3f} -> {float(out.float().std()):.3f}", flush=True)
        return out
