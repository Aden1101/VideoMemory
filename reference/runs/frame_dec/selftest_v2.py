#!/usr/bin/env python
# ============================================================================================
# SELFTESTS for decoder_v2.py -- pure CPU, no GPU, no shard, no checkpoint. Runs in ~2 s.
# House rule: the fixes are audited BEFORE any GPU is spent, and each test states the failure
# it is there to catch.
#   python reference/runs/frame_dec/selftest_v2.py
# ============================================================================================
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from decoder_v2 import (BestTracker, EmaWeights, assert_mixer_warm_start,  # noqa: E402
                        binom_p_two_sided, grad_match_loss, lowband,
                        pair_discrimination, twin_groups)

OK = []


def check(name, cond, detail=""):
    assert cond, f"FAIL {name} {detail}"
    OK.append(name)
    print(f"  ok  {name}")


torch.manual_seed(0)
B, C, H, W = 6, 16, 60, 104
tgt = torch.randn(B, 1, C, H, W)

# --- T1: the sharpness term starts at exactly 1.0 for a constant output ----------------------
# Catches: a scale mismatch between norm_mse and the new term, which would make --sharp_w
# unreadable (is 0.1 a tenth of the reconstruction term, or a thousandth?).
flat = torch.zeros_like(tgt)
check("T1 grad_match == 1.0 on a constant output",
      abs(float(grad_match_loss(flat, tgt)) - 1.0) < 1e-5,
      f"got {float(grad_match_loss(flat, tgt))}")

# --- T2: it is a MATCHER, not a maximizer ----------------------------------------------------
# Catches the e2_amp degeneracy: a term that rewards gradient ENERGY is satisfied by noise.
noisy = tgt + 3.0 * torch.randn_like(tgt)
check("T2 noise scores worse than the exact target",
      float(grad_match_loss(noisy, tgt)) > float(grad_match_loss(tgt, tgt)),
      f"noise {float(grad_match_loss(noisy, tgt)):.3f} vs exact {float(grad_match_loss(tgt, tgt)):.3f}")
check("T2b exact target scores 0", float(grad_match_loss(tgt, tgt)) < 1e-6)

# --- T3: blind to DC and to a smooth ramp ----------------------------------------------------
# Catches: the term being satisfiable by the token-constant bias the read learned in e1/e2.
shifted = tgt + 7.0
check("T3 DC shift is free", abs(float(grad_match_loss(shifted, tgt))) < 1e-6)

# --- T4: a blurred output scores WORSE than a sharp one --------------------------------------
# This is the whole point of cause 3: the v1 loss was happy with the low-pass output.
blur = torch.nn.functional.avg_pool2d(tgt.view(B, C, H, W), 5, 1, 2).view(B, 1, C, H, W)
check("T4 blur is penalized",
      float(grad_match_loss(blur, tgt)) > 0.5,
      f"got {float(grad_match_loss(blur, tgt)):.3f}")

# --- T5: BestTracker keeps the 4200/12000 point ----------------------------------------------
# Catches cause 1 verbatim: a run that peaks early then decays must ship the peak.
bt = BestTracker(higher_is_better=True)
curve = [(0.52, 1000), (0.61, 2000), (0.67, 4200), (0.63, 8000), (0.55, 12000)]
for v, s in curve:
    bt.update(v, s)
check("T5 best step is the peak, not the last", bt.best_step == 4200 and abs(bt.best - 0.67) < 1e-9,
      f"got step {bt.best_step} value {bt.best}")
check("T5b staleness counts since the peak", bt.since_best == 2, f"got {bt.since_best}")
check("T5c patience 0 never stops", not bt.should_stop(0))
check("T5d patience 2 stops", bt.should_stop(2))

bl = BestTracker(higher_is_better=False)
for v, s in [(1.0, 1), (0.4, 2), (0.9, 3)]:
    bl.update(v, s)
check("T5e loss mode picks the minimum", bl.best_step == 2)

# --- T6: EMA is a real average and does not touch integer buffers ----------------------------
m = torch.nn.Linear(4, 4)
torch.nn.init.constant_(m.weight, 0.0)
torch.nn.init.constant_(m.bias, 0.0)
ema = EmaWeights(m, decay=0.9)
with torch.no_grad():
    m.weight.fill_(1.0)
ema.update(m)
sd = ema.state_dict(m)
w = float(sd["weight"].mean())
check("T6 EMA lags the live weights", 0.0 < w < 1.0, f"got {w}")
check("T6b EMA returns every key", set(sd) == set(m.state_dict()))

# --- T7: the mixer warm-start contract -------------------------------------------------------
# Catches the worst silent failure available here: "--mixer" quietly re-randomising the decoder.
got = assert_mixer_warm_start(["mixer.0.pw1.weight", "mixer.1.norm.weight"], [])
check("T7 mixer-only missing keys accepted", got == ["mixer.0.pw1.weight", "mixer.1.norm.weight"])
try:
    assert_mixer_warm_start(["mixer.0.pw1.weight", "queries"], [])
    raise SystemExit("FAIL T7b: a missing `queries` was accepted")
except AssertionError:
    check("T7b a missing non-mixer key is refused", True)
try:
    assert_mixer_warm_start([], ["ghost.weight"])
    raise SystemExit("FAIL T7c: an unexpected key was accepted")
except AssertionError:
    check("T7c an unexpected key is refused", True)

# --- T8: lowband is a correlation ------------------------------------------------------------
lb = lowband(tgt)
check("T8 lowband rows are unit-norm", torch.allclose(lb.norm(dim=1), torch.ones(B), atol=1e-5))
check("T8b lowband rows are zero-mean", float(lb.mean(dim=1).abs().max()) < 1e-5)
check("T8c self-correlation is 1", abs(float(torch.dot(lb[0], lb[0])) - 1.0) < 1e-5)

# --- T9: twin grouping ------------------------------------------------------------------------
check("T9 ordered pairs per prompt",
      sorted(twin_groups(["a", "a", "b", "b", "c"])) == [(0, 1), (1, 0), (2, 3), (3, 2)])
check("T9b no pair when every prompt is unique", twin_groups(["a", "b", "c"]) == [])

# --- T10: pair_discrimination is 1.0 for a perfect decoder, ~chance for a blind one -----------
prompts = ["p0", "p0", "p1", "p1", "p2", "p2"]
check("T10 perfect reconstruction scores 1.0",
      pair_discrimination(tgt, tgt, prompts)["acc"] == 1.0)
const = torch.randn(1, 1, C, H, W).expand(B, 1, C, H, W).contiguous()
acc_blind = pair_discrimination(const, tgt, prompts)["acc"]
check("T10b a content-free decoder does not win", acc_blind <= 0.5 + 1e-9, f"got {acc_blind}")
check("T10c margin is positive when the decoder is right",
      pair_discrimination(tgt, tgt, prompts)["margin"] > 0)

# --- T11: the setup error is reported as None, never as a score -------------------------------
# Catches: a holdout cut that separates twins would otherwise read as "the memory learned
# nothing" instead of "the measurement was never possible".
check("T11 no twin in the holdout -> acc is None",
      pair_discrimination(tgt, tgt, [f"u{i}" for i in range(B)])["acc"] is None)

# --- T12: the binomial p-value ----------------------------------------------------------------
check("T12 chance result is not significant", binom_p_two_sided(15, 30) > 0.9)
check("T12b 24/30 is significant", binom_p_two_sided(24, 30) < 0.01,
      f"got {binom_p_two_sided(24, 30)}")
check("T12c symmetric", abs(binom_p_two_sided(24, 30) - binom_p_two_sided(6, 30)) < 1e-12)
check("T12d n=0 is 1.0", binom_p_two_sided(0, 0) == 1.0)

print(f"\n{len(OK)}/{len(OK)} selftests PASS")
