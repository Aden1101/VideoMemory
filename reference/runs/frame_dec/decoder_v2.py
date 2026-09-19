#!/usr/bin/env python
"""Decoder v2, the three fixes named by the 2026-08-17 eye board "Ce que produit la memoire",
rebuilt 2026-09-09 after the loss of the post-13/08 working tree.

What that board measured (the only surviving numbers of the campaign, quoted verbatim from
the published page):
  - from the memory state alone, the decoder picks the right subject over its twin in
    62-68 % of cases (chance 50 %), on subjects never seen, confirmed on two independent
    trainings  =>  the identity information is in the state;
  - but the rendering is unusable: it carries the palette and the coarse layout, the subject
    is not recognizable, and it sits below the strong-blur rung (which already fails the eye).

The three causes the board named, and the fix each one gets here:
  1. "the best point is reached at 4 200 steps out of 12 000, and quality degrades after"
     -> BestTracker + EmaWeights: select on validation, never on the last step, and average
        the weights so a late-training excursion cannot be what we ship. Not a new schedule:
        the v1 cosine schedule is fine, what was missing is that nothing kept the best point.
  2. "the output is made of independent blocks: nothing enforces coherence between
     neighbouring blocks, hence the mosaic"
     -> the mixer already exists in AnchorFrameDecoder(mixer=True) (FrameMixerBlock, zero-init
        => identity at init => a v1 checkpoint warm-starts bitwise). It was never wired into
        the trainer. `assert_mixer_warm_start` enforces the contract that the only missing
        keys are `mixer.*`, so "I turned the mixer on" can never silently mean "I reset the
        decoder".
  3. "no term of the objective pushes toward sharpness"
     -> grad_match_loss: match the target's spatial finite differences, normalized by the
        target's own gradient energy. Matching, not maximizing: a maximizer is gamed by
        noise (the e2_amp lesson, an unbounded "be more active" term always degenerates),
        a matcher is not. Scale-free and exactly 1.0 for a constant output, like norm_mse,
        so the zero-init head still starts every term at 1.0.

The metric that was missing from the trainer. The 62-68 % is a pair discrimination, and the
v1 trainer never computed it, it selected on reconstruction loss, which the board proved is
not the quantity we care about (loss went on improving while the eye verdict went negative).
`pair_discrimination` computes it on the holdout: for each item whose same-prompt twin is
also in the holdout, is the decoded frame closer to its own target than to the twin's? It is
scored in the low-band (8x8 centred-reduced correlation), the same statistic the project used
for T0 in composite_gates_report §16.1, so the number is comparable to the sealed campaign.

Everything here is pure torch on CPU-sized tensors and is covered by selftest_v2.py.
nothing in this file changes v1 behaviour: train_decoder.py calls it only behind new flags.
"""
import copy
import math

import torch


# ----------------------------- fix 3: the sharpness term -------------------------------------
def grad_match_loss(pred, target, eps=1e-8):
    """Match the target's spatial finite differences. Scale-free, batch-averaged.

    Pred/target: (B, ..., H, W). Returns a scalar.

    Properties that matter, and why (each one blocks a degeneracy we have already been bitten
    by elsewhere in this project):
      - a constant output scores exactly 1.0, like norm_mse at the zero-init head, so the two
        terms start commensurate and `--sharp_w` is a real ratio, not a scale guess;
      - it is a matching term, so pouring high-frequency noise into the output makes it worse,
        not better (an energy-maximizing sharpness term is the same trap as e2_amp's rel_norm);
      - it is blind to the DC and to any smooth ramp, so it cannot be satisfied by the
        token-constant bias that the memory read learned in the e1/e2 era.
    """
    pf, tf = pred.float(), target.float()
    assert pf.shape == tf.shape, f"shape mismatch {tuple(pf.shape)} vs {tuple(tf.shape)}"
    dxp = pf[..., :, 1:] - pf[..., :, :-1]
    dyp = pf[..., 1:, :] - pf[..., :-1, :]
    dxt = tf[..., :, 1:] - tf[..., :, :-1]
    dyt = tf[..., 1:, :] - tf[..., :-1, :]
    num = (dxp - dxt).flatten(1).pow(2).mean(dim=1) + (dyp - dyt).flatten(1).pow(2).mean(dim=1)
    den = dxt.flatten(1).pow(2).mean(dim=1) + dyt.flatten(1).pow(2).mean(dim=1)
    return (num / den.clamp_min(eps)).mean()


# ----------------------------- fix 1: keep the best point ------------------------------------
class BestTracker:
    """Remember the best validation point and how long ago it was.

    The v1 trainer saved `latest` and `final` only. The board's cause 1 is exactly the failure
    that produces: the run's best point was at step 4 200 of 12 000 and what got shipped was
    step 12 000. `higher_is_better=True` is the right mode for the pair-discrimination metric,
    False for a loss.
    """

    def __init__(self, higher_is_better=True, min_delta=0.0):
        self.higher_is_better = bool(higher_is_better)
        self.min_delta = float(min_delta)
        self.best = None
        self.best_step = -1
        self.since_best = 0

    def update(self, value, step):
        """Returns True when `value` is a new best (and resets the staleness counter)."""
        v = float(value)
        if self.best is None:
            improved = True
        elif self.higher_is_better:
            improved = v > self.best + self.min_delta
        else:
            improved = v < self.best - self.min_delta
        if improved:
            self.best, self.best_step, self.since_best = v, int(step), 0
        else:
            self.since_best += 1
        return improved

    def should_stop(self, patience):
        """Patience <= 0 disables early stopping (the default: we prefer a full run plus a
        best-checkpoint over a truncated run, and the two are not the same experiment)."""
        return patience > 0 and self.since_best >= patience


class EmaWeights:
    """Exponential moving average of the decoder weights, for the same reason as BestTracker:
    a single late step must not decide what we ship. Buffers are copied, not averaged."""

    def __init__(self, module, decay=0.999):
        assert 0.0 < decay < 1.0, f"decay must be in (0,1), got {decay}"
        self.decay = float(decay)
        self.shadow = {k: v.detach().clone().float()
                       for k, v in module.state_dict().items() if v.is_floating_point()}
        self.n = 0

    @torch.no_grad()
    def update(self, module):
        self.n += 1
        # debiased warmup: early on, follow the weights instead of the (arbitrary) init
        d = min(self.decay, (1.0 + self.n) / (10.0 + self.n))
        for k, v in module.state_dict().items():
            if k in self.shadow:
                self.shadow[k].mul_(d).add_(v.detach().float(), alpha=1.0 - d)

    def state_dict(self, module):
        """A full state_dict: EMA for the float tensors, live values for everything else."""
        out = copy.deepcopy(module.state_dict())
        for k, v in self.shadow.items():
            out[k] = v.to(dtype=out[k].dtype)
        return out


# ----------------------------- fix 2: the mixer, wired safely --------------------------------
def assert_mixer_warm_start(missing, unexpected):
    """Contract for `decoder.load_state_dict(v1_ckpt, strict=False)` when --mixer is on.

    The mixer is zero-init on its last pointwise, so it is exactly the identity at init and a
    v1 checkpoint must warm-start into it with no functional change. The only acceptable
    missing keys are therefore `mixer.*`. Anything else means the checkpoint does not match
    the architecture and the run would silently start from a partly random decoder, the
    class of mistake that cost this project two campaigns.
    """
    bad_missing = [k for k in missing if not k.startswith("mixer.")]
    assert not bad_missing, (
        f"warm-start into the mixer expects ONLY `mixer.*` to be missing, but these are also "
        f"missing: {bad_missing[:8]}{' ...' if len(bad_missing) > 8 else ''}")
    assert not unexpected, (
        f"checkpoint carries keys the model does not have: "
        f"{list(unexpected)[:8]}{' ...' if len(unexpected) > 8 else ''}")
    return sorted(k for k in missing if k.startswith("mixer."))


# ----------------------------- the metric that was missing ------------------------------------
def lowband(x, n=8, eps=1e-8):
    """Centred-reduced n x n low-band of a latent frame, the T0 statistic of
    composite_gates_report §16.1 (VAE-free variant: we stay in latent space here).

    x: (B, C, H, W) or (B, 1, C, H, W). Returns (B, n*n), each row zero-mean unit-norm, so a
    dot product between two rows is their correlation.
    """
    if x.dim() == 5:
        assert x.shape[1] == 1, f"expected a single frame, got {tuple(x.shape)}"
        x = x[:, 0]
    assert x.dim() == 4, f"expected (B,C,H,W), got {tuple(x.shape)}"
    g = x.float().mean(dim=1, keepdim=True)                       # channel mean = "grey"
    g = torch.nn.functional.adaptive_avg_pool2d(g, (n, n)).flatten(1)
    g = g - g.mean(dim=1, keepdim=True)
    return g / g.norm(dim=1, keepdim=True).clamp_min(eps)


def twin_groups(prompts):
    """Group holdout indices by prompt. Returns the list of (i, j) ordered pairs such that i
    and j share a prompt and i != j, i.e. every (subject, its twin) query the board's
    "le bon sujet plutot que son jumeau" question is asked on.

    A prompt with k > 2 members contributes every ordered pair, which is what we want: the
    corpus v2 twins come in 2s, but nothing in the collector guarantees it.
    """
    by = {}
    for i, p in enumerate(prompts):
        by.setdefault(p, []).append(i)
    pairs = []
    for idx in by.values():
        for i in idx:
            for j in idx:
                if i != j:
                    pairs.append((i, j))
    return pairs


@torch.no_grad()
def pair_discrimination(pred, target, prompts, n=8):
    """The board metric: does the decoded frame look more like its own target than like the
    same-prompt twin's target?

    Pred/target: (B, 1, C, H, W) latent frames. Prompts: list of B strings.
    Returns dict(acc, n_pairs, margin), acc is the 62-68 % number, chance is 0.5, and
    `margin` is the mean (own correlation - twin correlation), which says how far from the
    coin flip the wins are rather than just how many there were.

    Returns acc=None when the holdout contains no twin pair at all: that is a setup error
    (the holdout was cut without keeping both members of a pair together), not a score of
    zero, and the trainer must not select a checkpoint on it.
    """
    pairs = twin_groups(prompts)
    if not pairs:
        return {"acc": None, "n_pairs": 0, "margin": None}
    p = lowband(pred, n=n)
    t = lowband(target, n=n)
    wins, margins = 0, []
    for i, j in pairs:
        own = float(torch.dot(p[i], t[i]))
        twin = float(torch.dot(p[i], t[j]))
        wins += int(own > twin)
        margins.append(own - twin)
    return {"acc": wins / len(pairs),
            "n_pairs": len(pairs),
            "margin": sum(margins) / len(margins)}


def binom_p_two_sided(k, n, p0=0.5):
    """Exact two-sided binomial p-value, for reporting the pair score honestly (the project's
    own rule after the 10-seed annex: never present a single point as if it were the finding).
    Pure python, no scipy."""
    if n == 0:
        return 1.0
    def pmf(i):
        return math.comb(n, i) * (p0 ** i) * ((1.0 - p0) ** (n - i))
    obs = pmf(k)
    return min(1.0, sum(pmf(i) for i in range(n + 1) if pmf(i) <= obs * (1.0 + 1e-12)))
