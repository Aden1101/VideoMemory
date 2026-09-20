"""Rebuilding an image of the subject out of the memory state alone."""
import math
import os

import torch

from utils.env import env_int, env_str


# MEM_DECODE: when a subject comes back, the frame fed into the attention context is decoded
# from the live memory state by a decoder trained offline, instead of being the real past frame.
# Off unless MEM_DECODE names that decoder's checkpoint.
_MEMDEC_CACHE = {}   # (abs ckpt path, device str) -> loaded AnchorFrameDecoder, once per process
# the blocks whose fast-weight state the decoder reads: the memory_layers it was trained on
_MEMDEC_BLOCKS = (5, 16, 27)


def _memdec_pooled_cond(cond_dict):
    """Verbatim logic copy of the pooling used when the decoder training pairs were collected,
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
    (cond_hidden, cond_dim) exists only on prompt-conditioned decoders; absent = an
    unconditioned one (cond_dim=0 creates no cond modules, state_dict matches strictly).
    Kept in fp32 on the device (training kept fp32 params; one decode per video, speed moot).
    RNG hygiene: AnchorFrameDecoder.__init__ draws from the global torch RNG (queries/pos/
    linear inits) and this lazy load runs mid-ROLLOUT at the return switch, without the
    fork_rng, every post-return randn would desync vs the paired memzero arm."""
    key = (os.path.abspath(ckpt_path), str(device))
    if key in _MEMDEC_CACHE:
        return _MEMDEC_CACHE[key]
    from wan.modules.anchor_decoder import AnchorFrameDecoder
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


class AnchorDecodeMixin:
    # ---- ways of feeding an anchor frame back at the return; each is off unless its
    # environment variable is set ----------------------------------------------------------
    def _memdec_frame(self, ckpt_path, cond_dict, out_dtype, out_device):
        """Decode one anchor latent frame out of the live memory states, with the decoder trained
        offline for that. Kept as a single function so that decoding early (at eviction) and
        decoding at the return go through exactly the same path: same op order, same dtypes,
        same lazy load protected by fork_rng."""
        _dev = next(self.generator.parameters()).device
        _dec = _memdec_get_decoder(ckpt_path, _dev)
        for _bi in _MEMDEC_BLOCKS:
            assert self.ttt_cache[_bi].state is not None, \
                f"MEM_DECODE: no live memory state on block {_bi}"
        _sts = [tuple(t.detach().to(_dev, torch.float32)
                      for t in self.ttt_cache[_bi].state) for _bi in _MEMDEC_BLOCKS]
        _cond = None
        if getattr(_dec, "cond_dim", 0):
            _cond = _memdec_pooled_cond(cond_dict).to(_dev, torch.float32)[None]
        with torch.no_grad():
            _frame = _dec(_sts, cond=_cond)
        return _frame[:, 0:1].to(dtype=out_dtype, device=out_device), _cond is not None

    def _thru_maybe_decode(self, current_start_frame, cond_third, out_dtype, out_device):
        """MEM_DECODE_THRU: decode the memory state at the moment the chunk
        carrying the anchor frame is evicted (i.e. the write that stores it has just happened,
        write age = 1, the decoder's best regime) and CACHE the frame; the return switch then
        splices that frame instead of decoding a ~20-writes-old state. Isolates 'the write+read
        channel at age 1' from 'the state's forgetting between the write and the revisit'.
        Eviction arithmetic = causal_model.py's FIFO rule: chunk c is written window_chunks =
        ceil(local_attn_size / num_frame_per_block) chunks after it was produced."""
        if not env_int("MEM_DECODE_THRU"):
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
        _ck = env_str("MEM_DECODE")
        assert _ck, "MEM_DECODE_THRU=1 requires MEM_DECODE=<AnchorFrameDecoder ckpt>"
        self._thru_frame, _cond_on = self._memdec_frame(_ck, cond_third, out_dtype, out_device)
        self._thru_info = {"anchor_frame": int(_anchor_frame), "anchor_chunk": int(_anchor_chunk),
                           "evict_chunk": int(_cur_chunk), "window_chunks": int(_wc)}
        print(f"[MEM_DECODE_THRU] anchor chunk {_anchor_chunk} (frame {_anchor_frame}) evicted "
              f"and written at chunk {_cur_chunk} (window_chunks {_wc}), state decoded now "
              f"(write age 1), frame cached for the return splice; "
              f"cond={'ON' if _cond_on else 'off'}", flush=True)

    # Both drivers build their window here. `inference_schedule()` generalises `inference()`,
    # and the two are checked to produce bitwise identical latents on the single-switch
    # schedule, for every arm.
    @staticmethod
    def _memdec_window(output, current_start_frame, n, anchor, kslots, slot_offset):
        """The 12-frame recache window carrying the recalled anchor.

        Two conventions, because the original MEM_DECODE path built
            [output[cur - (n - k) : cur]] + [anchor] * k
        which keeps the true most recent frame cur-1 and appends the anchor after it, while the
        real-frame replay builds
            [output[cur - n : cur - k]] + [anchor] * k
        where the anchor replaces that most recent true frame. The first duplicates the recency,
        the second replaces it, so the two are not measuring the same thing. slot_offset >= 0
        selects the second:
            slot_offset = 0  anchor at the recency slot, the 11 others = output[cur-12:cur-1],
                             so the indices match the real-frame replay exactly
            slot_offset = j  the anchor is pushed j slots back and the true frames keep the
                             recency, so the model's immediate past stays real
        slot_offset < 0 keeps the first construction, so videos already rendered with it stay
        reproducible."""
        if slot_offset < 0:
            return torch.cat(
                [output[:, current_start_frame - (n - kslots):current_start_frame]]
                + [anchor] * kslots, dim=1)
        win = output[:, current_start_frame - n:current_start_frame].clone()
        pos = max(0, n - kslots - slot_offset)
        win[:, pos:pos + kslots] = anchor.to(dtype=win.dtype, device=win.device)
        return win
