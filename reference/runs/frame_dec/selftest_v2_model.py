#!/usr/bin/env python
"""Selftests v2, model side, the AnchorFrameDecoder contracts that the three fixes rely on.
CPU only, tiny dims, ~10 s. Loads wan/modules/video_memory.py directly by path (the `wan`
package __init__ pulls diffusers/flash-attn, which no audit box needs).
  python reference/runs/frame_dec/selftest_v2_model.py
"""
import importlib.util
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, HERE)
from decoder_v2 import (assert_mixer_warm_start, grad_match_loss,  # noqa: E402
                        pair_discrimination)

_spec = importlib.util.spec_from_file_location(
    "vmem_standalone", os.path.join(REPO, "wan", "modules", "video_memory.py"))
vmem = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(vmem)
AnchorFrameDecoder = vmem.AnchorFrameDecoder

OK = []


def check(name, cond, detail=""):
    assert cond, f"FAIL {name} {detail}"
    OK.append(name)
    print(f"  ok  {name}")


# Tiny geometry so the whole thing runs on a laptop; the contracts are shape-independent.
KW = dict(dim=32, inter_multi=2, num_blocks=3, grid_h=6, grid_w=8, fuse_hidden=64)
B = 6
torch.manual_seed(0)


def make_states(b=B):
    dh = KW["dim"] * KW["inter_multi"]
    return [(torch.randn(b, KW["dim"], dh), torch.randn(b, dh, KW["dim"]),
             torch.randn(b, KW["dim"], dh)) for _ in range(KW["num_blocks"])]


# --- M1: mixer=False is byte-identical to v1 --------------------------------------------------
# Catches: the v2 flag silently changing the module even when it is off.
d1 = AnchorFrameDecoder(**KW)
d1b = AnchorFrameDecoder(mixer=False, **KW)
check("M1 mixer=False creates no module", set(d1.state_dict()) == set(d1b.state_dict()))
check("M1b no `mixer.` key exists when off",
      not [k for k in d1.state_dict() if k.startswith("mixer.")])

# --- M2: zero-init head emits an exactly null frame -------------------------------------------
# The rule the project wrote in blood: never xavier noise into the cache.
st = make_states()
with torch.no_grad():
    y0 = d1(st)
check("M2 output is exactly zero at init", float(y0.abs().max()) == 0.0)
check("M2b output shape is one latent frame",
      tuple(y0.shape) == (B, 1, 16, KW["grid_h"] * 2, KW["grid_w"] * 2), f"got {tuple(y0.shape)}")

# --- M3: the mixer is the identity at init, so a v1 ckpt warm-starts with no functional change -
# the contract of cause 2. If this fails, "--mixer" means "throw the decoder away".
d2 = AnchorFrameDecoder(mixer=True, mixer_blocks=2, mixer_hidden=48, **KW)
missing, unexpected = d2.load_state_dict(d1.state_dict(), strict=False)
new_keys = assert_mixer_warm_start(list(missing), list(unexpected))
check("M3 only `mixer.*` is new", len(new_keys) > 0 and len(new_keys) == len(missing))
# make the head non-trivial so "identical outputs" is a real test, not 0 == 0
with torch.no_grad():
    d1.head.weight.normal_(0, 0.05)
    d1.head.bias.normal_(0, 0.05)
d2.load_state_dict(d1.state_dict(), strict=False)
with torch.no_grad():
    ya, yb = d1(st), d2(st)
check("M3b mixer at init reproduces the v1 output exactly",
      torch.equal(ya, yb), f"max|diff| = {float((ya - yb).abs().max())}")

# --- M4: once trained, the mixer can mix (it is not dead code) ---------------------------------
# Catches the opposite failure: a mixer that is identity for ever, so the mosaic never goes.
with torch.no_grad():
    for blk in d2.mixer:
        for p in blk.parameters():
            p.add_(0.1 * torch.randn_like(p))
    yc = d2(st)
check("M4 a perturbed mixer changes the output", not torch.equal(ya, yc))

# --- M5: the two loss terms are commensurate at init ------------------------------------------
# Both are exactly 1.0 for the null output, which is what makes --sharp_w a readable ratio.
tgt = torch.randn(B, 1, 16, KW["grid_h"] * 2, KW["grid_w"] * 2)
# import train_decoder for its loss without dragging in the `wan` package (diffusers,
# flash-attn): register the standalone module under the name train_decoder expects.
import types  # noqa: E402
_pkg = types.ModuleType("wan"); _pkg.__path__ = []
_mods = types.ModuleType("wan.modules"); _mods.__path__ = []
sys.modules.setdefault("wan", _pkg)
sys.modules.setdefault("wan.modules", _mods)
sys.modules.setdefault("wan.modules.video_memory", vmem)
os.environ.setdefault("AF_REPO", REPO)
from train_decoder import norm_mse  # noqa: E402
d3 = AnchorFrameDecoder(**KW)
with torch.no_grad():
    z = d3(st)
check("M5 norm_mse == 1.0 at the zero-init head", abs(float(norm_mse(z, tgt)) - 1.0) < 1e-5)
check("M5b grad_match == 1.0 at the zero-init head",
      abs(float(grad_match_loss(z, tgt)) - 1.0) < 1e-5)

# --- M6: the combined objective actually trains, and the sharpness term pulls the gradients ----
# Overfit a 2-sample batch: if this does not go down, the wiring is wrong, GPU or not.
def overfit(sharp_w, steps=120, seed=0):
    torch.manual_seed(seed)
    d = AnchorFrameDecoder(mixer=True, mixer_blocks=1, mixer_hidden=48, **KW)
    s = make_states(2)
    t = torch.randn(2, 1, 16, KW["grid_h"] * 2, KW["grid_w"] * 2)
    opt = torch.optim.AdamW(d.parameters(), lr=3e-3)
    for _ in range(steps):
        p = d(s)
        loss = norm_mse(p, t) + sharp_w * grad_match_loss(p, t)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    with torch.no_grad():
        p = d(s)
        return float(norm_mse(p, t)), float(grad_match_loss(p, t))


rec_off, sharp_off = overfit(0.0)
rec_on, sharp_on = overfit(0.5)
check("M6 the objective trains (reconstruction falls well below 1.0)",
      rec_off < 0.5, f"rec {rec_off:.3f}")
check("M6b the sharpness term improves gradient fidelity",
      sharp_on < sharp_off, f"sharp with {sharp_on:.3f} vs without {sharp_off:.3f}")

# --- M7: the pair metric on real decoder outputs -----------------------------------------------
# A decoder that has memorized its targets must score 1.0; the metric must not be fooled by
# the fact that all outputs come from the same network.
prompts = ["p0", "p0", "p1", "p1", "p2", "p2"]
check("M7 identity decoder scores 1.0", pair_discrimination(tgt, tgt, prompts)["acc"] == 1.0)
check("M7b a null decoder is uncomputable-safe, not a fake win",
      pair_discrimination(torch.zeros_like(tgt) + 1e-9, tgt, prompts)["acc"] <= 0.5 + 1e-9)

print(f"\n{len(OK)}/{len(OK)} model selftests PASS")
