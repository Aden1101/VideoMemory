"""The counterfactual-twin objective: regenerate a chunk with the memory switched off, sharing
history, KV cache and noise, and score the real chunk against it.

Mixed into StreamingTrainingModel. With every twin weight at 0, which is the case of the four
reported runs, none of this runs and training is plain DMD.
"""
from dataclasses import dataclass, fields
import hashlib
import json
import os
import random
import torch
import torch.distributed as dist
import torch.nn.functional as F


@dataclass
class TwinObjective:
    """Hyperparameters of the counterfactual-twin objective (see _compute_advantage_loss for
    the terms). Every weight defaults to 0, which leaves plain DMD and builds no twin."""
    adv_loss_weight: float = 0.0     # bonus mode: reward for beating the twin on identity
    adv_clamp_pos: float = 0.05      # that reward saturates here (cosine units)
    adv_clamp_neg: float = 0.20      # and its do-no-harm side stays linear 4x deeper
    adv_gate_sigma: float = 0.405    # ln(1.5): width of the log-motion validity band
    adv_floor_frac: float = 0.5      # the external floor fires below this fraction of stock motion
    mm_rel_weight: float = 0.0       # two-sided log-motion match to the twin
    mm_ext_weight: float = 0.0       # one-sided log-motion floor against the stock curve
    dnh_weight: float = 0.0          # do-no-harm hinge: identity may not be worse than the twin
    dnh_deadband: float = 0.01       # cosine units of tolerated noise
    act_weight: float = 0.0          # activity floor on the injection's dose; a relu, so a
    act_floor: float = 0.05          # floor and never a maximiser (relative AC-dose)
    motion_lag2: bool = False        # lag-2 motion statistic, blind to frame-rate shimmer
    dose_tmean_remove: bool = False  # a static overlay then earns zero dose
    flick_weight: float = 0.0        # second-temporal-difference hinge against the twin
    flick_deadband: float = 0.25     # relative to the twin's own roughness

    @classmethod
    def from_config(cls, config):
        return cls(**{f.name: f.type(getattr(config, f.name, f.default)) if f.type is bool
                      else getattr(config, f.name, f.default) for f in fields(cls)})

    @property
    def needs_anchor(self):
        """The two terms that build the chunk-0 identity anchor every other term is scored next to."""
        return self.adv_loss_weight > 0.0 or self.dnh_weight > 0.0

    @property
    def on(self):
        return (self.needs_anchor or self.act_weight > 0.0 or self.mm_rel_weight > 0.0
                or self.mm_ext_weight > 0.0 or self.flick_weight > 0.0)


class TwinObjectiveMixin:
    # ---- counterfactual-twin machinery: snapshot/restore everything one chunk mutates ----
    _DINO_MEAN = (0.485, 0.456, 0.406)
    _DINO_STD = (0.229, 0.224, 0.225)

    def _init_twin_objective(self, config):
        """Read the identity and twin options. Every weight defaults to 0: nothing here then
        runs during training, and the DINO encoder is never loaded."""
        # identity to the chunk-0 anchor, on id_decode_frames decoded frames per chunk, and a
        # floor on the motion between adjacent latents
        self.id_loss_weight = getattr(config, "id_loss_weight", 0.0)
        self.id_decode_frames = getattr(config, "id_decode_frames", 2)
        self.tv_loss_weight = getattr(config, "tv_loss_weight", 0.0)
        self.tv_motion_floor = getattr(config, "tv_motion_floor", 0.0)
        self.twin = TwinObjective.from_config(config)
        # every twin term is scored next to the identity anchor of chunk 0, which only an
        # identity term builds; without one the whole objective returns zero
        if self.twin.on and not self.twin.needs_anchor and self.id_loss_weight <= 0.0:
            raise ValueError("act_weight, mm_rel_weight, mm_ext_weight and flick_weight need "
                             "dnh_weight, adv_loss_weight or id_loss_weight > 0: the twin "
                             "objective returns zero without the identity anchor those build")
        self._dino = None             # frozen DINOv2, loaded on first use
        self._id_anchor = None        # identity of the subject in chunk 0, per sequence
        # {sha1(prompt)[:16]: motion of the unmodified model per chunk, "_global": the same, pooled}
        self._motion_targets = None
        path = getattr(config, "motion_target_path", None)
        if path and self.twin.on:
            with open(path) as f:
                self._motion_targets = json.load(f)
            if not dist.is_initialized() or dist.get_rank() == 0:
                print(f"[StreamingTrain-Model] twin objective on: {self.twin}, "
                      f"targets={path} ({len(self._motion_targets)} keys)")

    def _twin_snapshot(self):
        """Everything generating a chunk mutates: the KV, cross-attention, flow and prompt
        caches, the memory state, the three RNG streams and the switch flag. The tensor copies
        go to the CPU to spare VRAM. Whether the list is complete is checked by TWIN_SELFTEST=1:
        a twin generated with the memory on must equal the real chunk bitwise."""
        p = self.inference_pipeline
        tensors = []

        def copy_tensors(obj):
            if torch.is_tensor(obj):
                tensors.append((obj, obj.detach().to("cpu", copy=True)))
            elif isinstance(obj, (dict, list, tuple)):
                for v in (obj.values() if isinstance(obj, dict) else obj):
                    copy_tensors(v)

        for name in ("kv_cache1", "crossattn_cache", "flow_cache", "prompt_cache"):
            copy_tensors(getattr(p, name, None))
        # fast weights, injection EMA buffers and eviction FIFO: the twin's writes to any of
        # them would leak into the real rollout and break the counterfactual
        ttt = [c.snapshot() for c in p.ttt_cache] if getattr(p, "ttt_cache", None) else None
        return {
            "tensors": tensors, "ttt": ttt,
            "has_switched": self.state.get("has_switched", False),
            "rng_cpu": torch.get_rng_state(),
            "rng_cuda": torch.cuda.get_rng_state(self.device),
            "rng_py": random.getstate(),
        }

    def _generate_twin(self, noise_chunk, chunk_start_frame):
        """The memory-off twin of the chunk about to be generated: same state before the chunk,
        same noise, same RNG stream, injection zeroed, so the memory's injection during this
        chunk is the only difference. Generated first; every state it touched is then restored,
        which leaves the real rollout bitwise what it is without a twin.

        TWIN_SELFTEST=1 keeps the memory on in the twin; _twin_selftest then checks that the twin
        equals the real chunk, which validates that the snapshot is complete and the RNG paired."""
        snap = self._twin_snapshot()
        previous = os.environ.get("MEM_ZERO_ALL")
        if os.environ.get("TWIN_SELFTEST", "0") != "1":
            os.environ["MEM_ZERO_ALL"] = "1"
        try:
            with torch.no_grad():
                twin, _, _ = self._generate_chunk(
                    noise_chunk=noise_chunk, chunk_start_frame=chunk_start_frame,
                    requires_grad=False, detach_state=True, keep_last_write_grad=False)
            return twin.detach()
        finally:
            if previous is None:
                os.environ.pop("MEM_ZERO_ALL", None)
            else:
                os.environ["MEM_ZERO_ALL"] = previous
            self._twin_restore(snap)

    @staticmethod
    def _twin_selftest(real, twin):
        if twin is None or os.environ.get("TWIN_SELFTEST", "0") != "1":
            return
        diff = (real.detach() - twin).abs().max().item()
        print(f"[TWIN-SELFTEST] rank{dist.get_rank() if dist.is_initialized() else 0} "
              f"max|real - twin(mem ON)| = {diff:.3e} (MUST be 0.0)", flush=True)

    def _twin_restore(self, snap):
        p = self.inference_pipeline
        with torch.no_grad():
            for ref, saved in snap["tensors"]:
                ref.copy_(saved.to(ref.device))
        if snap["ttt"] is not None and getattr(p, "ttt_cache", None):
            for c, s in zip(p.ttt_cache, snap["ttt"]):
                c.restore(s)
        self.state["has_switched"] = snap["has_switched"]
        torch.set_rng_state(snap["rng_cpu"])
        torch.cuda.set_rng_state(snap["rng_cuda"], self.device)
        random.setstate(snap["rng_py"])

    def _get_dino(self):
        """Lazily load a frozen DINOv2-ViT-B/14 (offline from the torch.hub cache)."""
        if self._dino is None:
            hub = os.path.expanduser("~/.cache/torch/hub")
            torch.hub.set_dir(hub)
            local = os.path.join(hub, "facebookresearch_dinov2_main")
            m = torch.hub.load(local, "dinov2_vitb14", source="local", pretrained=True)
            m.eval().to(self.device)
            for p in m.parameters():
                p.requires_grad_(False)
            self._dino = m
            if (not dist.is_initialized() or dist.get_rank() == 0):
                print("[StreamingTrain-Model] loaded frozen DINOv2-vitb14 for identity loss")
        return self._dino

    def _dino_embed_pixels(self, pixels):
        """Pixels [B,T,C,H,W] in [-1,1] -> L2-normalized DINO CLS embeddings [B*T, D] (fp32)."""
        dino = self._get_dino()
        b, t, c, h, w = pixels.shape
        x = pixels.reshape(b * t, c, h, w).float()
        x = (x * 0.5 + 0.5).clamp(0, 1)                       # [-1,1] -> [0,1]
        x = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)
        mean = torch.tensor(self._DINO_MEAN, device=x.device).view(1, 3, 1, 1)
        std = torch.tensor(self._DINO_STD, device=x.device).view(1, 3, 1, 1)
        x = (x - mean) / std
        f = dino(x)                                           # [B*T, D] CLS embedding
        return F.normalize(f, dim=-1)

    # -----------------------------------------------------------------------------------------
    # The extra loss terms, and why the runs this repo presents have all of them off.
    #
    # Eight were built, in three generations, each closing the channel the previous one left
    # open. Every one is a hinge with a deadband rather than a maximiser: an unbounded "be more
    # active" term is always satisfied by the cheapest route, which is noise or a constant bias.
    #
    #   id, tv            identity to the chunk-0 anchor, and a floor on latent motion. Used by
    #                     the first generation of runs, then dropped: the identity term is a
    #                     freeze ally, since the cheapest way to keep a face is to stop moving,
    #                     and the motion floor is subsumed by the mm terms below.
    #   dnh, act,         do-no-harm on identity; a floor on how much the injection changes the
    #   mm_rel, mm_ext    frame; a two-sided motion match to the twin; and an absolute floor
    #                     against the unmodified model's own motion curve, which is what catches
    #                     a twin that froze too. Used by the second generation of runs.
    #   flick,            a second-temporal-difference hinge against the twin, plus two switches,
    #   motion_lag2,      added after the model learned to counterfeit motion with frame-rate
    #   dose_tmean_remove shimmer and to satisfy the activity floor with a static overlay. Each
    #                     switch closes one of those two routes.
    #   adv               a clamped reward for beating the twin on identity. Written, bounded,
    #                     gated on motion validity, and never enabled in any run of this project:
    #                     it was judged a freeze risk. adv_floor_frac is still used, by mm_ext.
    #
    # The four runs behind the README figure turn all of this off and keep DMD alone, because the
    # stack fought the distillation and swept the model between freezing and flickering. With
    # every twin weight at 0, twin.on is false and the trainer never even builds the twin
    # (make_twin in trainer/memory_training.py), so none of the code below runs.
    # -----------------------------------------------------------------------------------------
    def _set_id_anchor(self, chunk):
        """Cache the chunk-0 subject identity anchor (no grad), called from the rollout on the first
        chunk. Uses the head frames (highest prompt fidelity, last to degrade under collapse) so the
        anchor is a real subject; supervised chunks compare their tail frames, so id_cos=1.0 becomes a
        genuine static-video alarm instead of a vacuous garbage-vs-garbage fixed point."""
        if self.id_loss_weight <= 0.0 and not self.twin.needs_anchor:
            return
        k = max(1, int(self.id_decode_frames))
        with torch.no_grad():
            px = self.base_model.vae.decode_to_pixel(chunk[:, :k, ...], use_cache=False)  # head, pre-collapse
            emb = self._dino_embed_pixels(px)
            self._id_anchor = F.normalize(emb.mean(0, keepdim=True), dim=-1).detach()

    def _compute_identity_tv_loss(self, chunk, chunk_start_frame):
        """Frame-level identity anchor loss + temporal-variance floor. Returns (id_loss, tv_loss, log).
        All log values are 0-dim tensors present on every chunk (chunk-0 defaults) so merge_dict_list
        (which keys off the first chunk's dict) keeps them and stacks them across chunks."""
        device = chunk.device
        id_loss = torch.zeros((), device=device)
        tv_loss = torch.zeros((), device=device)
        # consistent keys every chunk (chunk-0 identity is perfect by construction -> cos=1, loss=0)
        log = {
            "identity_latent_motion": torch.zeros((), device=device),
            "identity_tv_loss": torch.zeros((), device=device),
            "identity_id_cos": torch.ones((), device=device),
            "identity_id_loss": torch.zeros((), device=device),
        }

        # temporal-variance floor (anti-freeze): hinge on adjacent-latent motion; penalize only if frozen
        if self.tv_loss_weight > 0.0 and chunk.shape[1] >= 2:
            motion = (chunk[:, 1:] - chunk[:, :-1]).abs().mean()
            tv_loss = F.relu(self.tv_motion_floor - motion)
            log["identity_latent_motion"] = motion.detach()
            log["identity_tv_loss"] = tv_loss.detach()

        # frame-level identity: decode a few frames -> DINO -> cosine to the chunk-0 anchor, on later
        # chunks only, with grad through decode -> gate -> read -> write (anchor cached in the rollout).
        # Skip after a real prompt switch: the anchor is the pre-switch subject, penalizing the new one.
        if (self.id_loss_weight > 0.0 and chunk_start_frame > 0 and self._id_anchor is not None
                and not self.state.get("has_switched", False)):
            k = max(1, int(self.id_decode_frames))
            px = self.base_model.vae.decode_to_pixel(chunk[:, -k:, ...], use_cache=False)
            emb = self._dino_embed_pixels(px)
            cos = (emb * self._id_anchor).sum(-1)            # [k]
            id_loss = (1.0 - cos).mean()
            log["identity_id_cos"] = cos.mean().detach()
            log["identity_id_loss"] = id_loss.detach()
        return id_loss, tv_loss, log

    # ---- counterfactual, motion-gated, clamped identity advantage over the mem-OFF twin ----
    def _motion_target_index(self, chunk_start_frame):
        """chunk_start (0, 21, 39, 57, ...) -> index (0, 1, 2, 3, ...) into the motion-target curve."""
        if chunk_start_frame <= 0:
            return 0
        return (chunk_start_frame - self.chunk_size) // self.min_new_frame + 1

    def _get_motion_target(self, chunk_start_frame):
        """Per-prompt (fallback: _global) stock-AF latent-motion target for this chunk position.
        Keys = sha1(prompt.strip())[:16]; the trainer stores the prompt in state["current_prompt"]."""
        if self._motion_targets is None:
            return None
        p = self.state.get("current_prompt")
        key = hashlib.sha1(p.strip().encode()).hexdigest()[:16] if isinstance(p, str) else None
        curve = self._motion_targets.get(key) if key else None
        if curve is None:
            curve = self._motion_targets.get("_global")
        if not curve:
            return None
        i = min(self._motion_target_index(chunk_start_frame), len(curve) - 1)
        return torch.tensor(float(curve[i]), device=self.device)

    def _compute_advantage_loss(self, chunk, chunk_info):
        """The twin objective, on the newly generated frames (ON) against their memory-off
        twin (CF):

            L = w_dnh  * relu(cos_CF - cos_ON - deadband)        identity may not get worse
              + w_act  * relu(act_floor - dose)                  the injection may not be inert
              + w_rel  * (log m_ON - log m_CF)^2                 motion must match the twin's
              + w_ext  * relu(log(frac * m_stock) - log m_ON)^2  and may not co-freeze with it
              + w_flick * flicker added beyond the twin's
              - w_adv  * g * clamp(cos_ON - cos_CF, -neg, +pos)  bonus mode, off by default

        Why this shape: the twin is no-grad and shares history, KV cache and noise, so the
        injection during this chunk is the only lever and nothing is earned by degrading the
        baseline. Nothing pays for self-similarity, so freezing buys nothing that staying
        neutral does not, at a higher motion cost. The activity term is a relu: zero gradient
        once the dose clears the floor, a floor and never a maximiser. What is left to game is
        a static texture overlay, judged at evaluation; dose_tmean_remove closes it.

        Returns (loss, log); every log key is present on every chunk, because merge_dict_list
        takes its keys from the first one."""
        device = chunk.device
        zero = lambda: torch.zeros((), device=device)
        one = lambda: torch.ones((), device=device)
        log = {k: zero() for k in (
            "twin_adv_loss", "twin_adv_raw", "twin_gate", "twin_dnh", "twin_act", "twin_dose",
            "twin_mm_rel", "twin_mm_ext", "twin_flick", "twin_motion_on", "twin_motion_cf",
            "twin_motion_tgt", "twin_active", "twin_oom_skip")}
        log.update({"twin_cos_on": one(), "twin_cos_cf": one()})
        twin = chunk_info.get("twin_new_frames")
        chunk_start_frame = chunk_info["chunk_start_frame"]
        if twin is None or chunk_start_frame <= 0 or self._id_anchor is None:
            return zero(), log
        cfg, eps = self.twin, 1e-6
        # the newly generated frames only: the twin spans the same global frames, and leaving
        # the overlap out avoids comparing re-encoded first frames
        real = chunk[:, int(chunk_info.get("overlap_frames_used", 0)):]

        m_on = self._motion(real)
        with torch.no_grad():
            m_cf = self._motion(twin)
        m_tgt = self._get_motion_target(chunk_start_frame)
        mm_rel = (torch.log(m_on + eps) - torch.log(m_cf + eps)) ** 2
        mm_ext = zero() if m_tgt is None else F.relu(
            torch.log(cfg.adv_floor_frac * m_tgt + eps) - torch.log(m_on + eps)) ** 2
        dose = self._dose(real, twin, eps)
        act = F.relu(cfg.act_floor - dose)
        flick = self._added_flicker(real, twin, eps) if cfg.flick_weight > 0.0 else zero()
        cos_on, cos_cf = self._paired_identity(real, twin)
        advantage = cos_on - cos_cf
        dnh = F.relu(-advantage - cfg.dnh_deadband)
        # bonus mode: the reward counts only while motion stays valid; the gate is detached,
        # a condition and never a lever
        with torch.no_grad():
            ref = m_tgt if m_tgt is not None else m_cf
            gate = torch.exp(-((torch.log(m_on.detach() + eps) - torch.log(ref + eps)) ** 2)
                             / (2.0 * cfg.adv_gate_sigma ** 2))
        adv = gate * torch.clamp(advantage, min=-cfg.adv_clamp_neg, max=cfg.adv_clamp_pos)

        loss = (cfg.dnh_weight * dnh + cfg.act_weight * act + cfg.mm_rel_weight * mm_rel
                + cfg.mm_ext_weight * mm_ext + cfg.flick_weight * flick
                + cfg.adv_loss_weight * (-adv))
        log.update({
            "twin_adv_loss": (cfg.adv_loss_weight * (-adv)).detach(),
            "twin_adv_raw": advantage.detach(), "twin_gate": gate.detach(),
            "twin_dnh": dnh.detach(), "twin_act": act.detach(), "twin_dose": dose.detach(),
            "twin_mm_rel": mm_rel.detach(), "twin_mm_ext": mm_ext.detach(),
            "twin_flick": flick.detach(),
            "twin_cos_on": cos_on.detach(), "twin_cos_cf": cos_cf.detach(),
            "twin_motion_on": m_on.detach(), "twin_motion_cf": m_cf.detach(),
            "twin_motion_tgt": (m_tgt.detach() if m_tgt is not None else m_cf.detach()),
            "twin_active": one(),
        })
        return loss, log

    def _motion(self, frames):
        """Mean absolute latent difference between frames. With motion_lag2, two frames apart
        and halved: a frame-rate alternation contributes almost nothing to it while smooth
        motion contributes fully, so shimmer cannot pass for motion. motion_targets.json has
        to be generated with the same choice."""
        if self.twin.motion_lag2 and frames.shape[1] >= 3:
            return (frames[:, 2:] - frames[:, :-2]).abs().mean() * 0.5
        return (frames[:, 1:] - frames[:, :-1]).abs().mean()

    def _dose(self, real, twin, eps):
        """How much the injection changed the chunk: RMS of (real - twin) over the RMS of the
        twin, both with the per-channel mean removed so a global bias earns nothing. With
        dose_tmean_remove the per-site temporal mean goes too: a static overlay earns nothing,
        only a change that evolves in time and has spatial structure does."""
        d = real - twin                                                 # (B, T, C, H, W)
        d = d - d.mean(dim=(1, 3, 4), keepdim=True)
        with torch.no_grad():
            ref = twin - twin.mean(dim=(1, 3, 4), keepdim=True)
            if self.twin.dose_tmean_remove:
                ref = ref - ref.mean(dim=1, keepdim=True)
            ref_rms = ref.float().pow(2).mean().sqrt().clamp_min(eps)
        if self.twin.dose_tmean_remove:
            d = d - d.mean(dim=1, keepdim=True)
        return d.float().pow(2).mean().sqrt() / ref_rms

    def _added_flicker(self, real, twin, eps):
        """Per-site roughness, the second temporal difference, that the injection adds beyond
        the twin's own and beyond a deadband, relative to the twin's mean roughness. The
        second difference peaks on a frame-rate alternation and is about zero for constant
        velocity, so the twin's real accelerations are already priced in."""
        if real.shape[1] < 3:
            return torch.zeros((), device=real.device)

        def roughness(x):
            d2 = x[:, 2:] - 2 * x[:, 1:-1] + x[:, :-2]
            return d2.float().pow(2).mean(dim=2).clamp_min(1e-12).sqrt()         # (B, T-2, H, W)

        r_on = roughness(real)
        with torch.no_grad():
            r_cf = roughness(twin)
            r_ref = r_cf.mean().clamp_min(eps)
        return F.relu(r_on - r_cf - self.twin.flick_deadband * r_ref).mean() / r_ref

    def _paired_identity(self, real, twin):
        """Cosine to the chunk-0 identity anchor of the same decoded frames in the real chunk
        (with grad) and in the twin. The k frames are drawn at a random offset in the last six,
        so the read cannot lock onto a position."""
        k = max(1, int(self.id_decode_frames))
        span = min(6, real.shape[1])
        off = int(torch.randint(0, max(1, span - k + 1), (1,)).item())
        s0, s1 = real.shape[1] - off - k, real.shape[1] - off
        px_on = self.base_model.vae.decode_to_pixel(real[:, s0:s1], use_cache=False)
        cos_on = (self._dino_embed_pixels(px_on) * self._id_anchor).sum(-1).mean()
        with torch.no_grad():
            px_cf = self.base_model.vae.decode_to_pixel(twin[:, s0:s1], use_cache=False)
            cos_cf = (self._dino_embed_pixels(px_cf) * self._id_anchor).sum(-1).mean()
        return cos_on, cos_cf

