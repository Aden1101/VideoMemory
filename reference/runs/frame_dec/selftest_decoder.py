#!/usr/bin/env python
# ============================================================================================
# Self-test for AnchorFrameDecoder (design v1 "frame de rappel decodee"). No data, no wandb,
# no VAE -- light enough for GPU 0 alongside a training run (a few GB, <1 min).
#
#   T1  shapes: 3 synthetic block states (B=2) -> (2, 1, 16, 60, 104)
#   T2  EXACT zero output at init (zero-init head: a null frame, never xavier noise)
#   T3  gradient routing:
#         (a) at zero-init, only the head gets gradient (documented zero-init consequence);
#         (b) with a nudged head, queries / pos emb / fusion all get non-zero gradient;
#         (a+b) the input STATES get NO gradient even with requires_grad=True (they are
#               INPUTS, detached inside the decoder by default).
#   T4  overfit sanity: 8 synthetic (states, frame) pairs, AdamW, 200 steps,
#       normalized loss must drop below 10% of its init value (init == 1.0 by construction).
#   T5  v1.1 FiLM conditioning (cond_dim>0): a v1.0 state_dict warm-starts it (strict=False,
#       missing keys == cond modules only), and AT INIT the conditioned decoder's output is
#       IDENTICAL to the v1.0 model, both with cond=None and with a random cond (zero-init
#       FiLM = exact identity); cond on a cond_dim=0 decoder must raise; param budget.
#   T6  cond gradient routing: at zero-init FiLM only cond_out gets gradient (cond_in gets
#       exact zeros); after nudging cond_out, gradient reaches cond_in and cond_norm; the
#       cond INPUT never receives gradient (data, detached inside like the states).
#
# Run:  python reference/runs/frame_dec/selftest_decoder.py
# ============================================================================================
import math
import os
import sys

REPO = "/shared/ssd_14T/home/adenelmi/Anchor-Forcing"
sys.path.insert(0, REPO)
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")

import torch

from wan.modules.video_memory import AnchorFrameDecoder
from train_decoder import norm_mse  # the EXACT training loss, not a reimplementation

DIM, DH = 1536, 6144  # cell head_dim (num_heads=1) and hidden (inter_multi=4)


def synth_states(B, device, num_blocks=3, seed=0, requires_grad=False):
    """Random states at the VideoMemoryLayer init scale (randn / sqrt(fan_in)), leading axis
    B*nh with nh=1, exactly the shapes init_state()/step() produce."""
    g = torch.Generator().manual_seed(seed)
    states = []
    for _ in range(num_blocks):
        w0 = (torch.randn(B, DIM, DH, generator=g) / math.sqrt(DIM))
        w1 = (torch.randn(B, DH, DIM, generator=g) / math.sqrt(DH))
        w2 = (torch.randn(B, DIM, DH, generator=g) / math.sqrt(DIM))
        ws = tuple(t.to(device).requires_grad_(requires_grad) for t in (w0, w1, w2))
        states.append(ws)
    return states


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        print("[selftest] WARNING: no CUDA visible -- T4 (overfit) will be very slow on CPU")
    torch.manual_seed(0)

    # ---------------- T0: param budget ----------------
    dec = AnchorFrameDecoder().to(device)
    n = dec.num_params()
    assert n < 40_000_000, f"param budget blown: {n:,} >= 40M"
    assert n == 26_304_064, f"param count drifted from the documented 26,304,064: {n:,}"
    print(f"[selftest] T0 PASS  params = {n:,} (~{n/1e6:.1f}M, budget 40M)")

    # ---------------- T1: shapes ----------------
    states = synth_states(2, device, seed=1)
    with torch.no_grad():
        out = dec(states)
    assert out.shape == (2, 1, 16, 60, 104), f"bad output shape {tuple(out.shape)}"
    # unbatched (di, dh) convenience path
    states1 = [tuple(w[0] for w in st) for st in synth_states(1, device, seed=2)]
    with torch.no_grad():
        out1 = dec(states1)
    assert out1.shape == (1, 1, 16, 60, 104), f"bad unbatched output shape {tuple(out1.shape)}"
    print("[selftest] T1 PASS  shapes (B=2 and unbatched) -> (B, 1, 16, 60, 104)")

    # ---------------- T2: exact zero at init ----------------
    assert out.abs().max().item() == 0.0, "init output is NOT exactly zero (head zero-init broken)"
    print("[selftest] T2 PASS  init output exactly 0 (null frame, no xavier noise)")

    # ---------------- T3: gradient routing ----------------
    states = synth_states(2, device, seed=3, requires_grad=True)
    target = torch.randn(2, 1, 16, 60, 104, device=device)

    # (a) zero-init head: only the head can receive gradient at step 0
    out = dec(states)
    loss = norm_mse(out, target)
    assert abs(loss.item() - 1.0) < 1e-4, f"init normalized loss should be 1.0, got {loss.item()}"
    loss.backward()
    assert dec.head.weight.grad is not None and dec.head.weight.grad.abs().sum() > 0, \
        "head.weight got no gradient"
    assert dec.head.bias.grad is not None and dec.head.bias.grad.abs().sum() > 0, \
        "head.bias got no gradient"
    assert dec.queries.grad is None or dec.queries.grad.abs().sum() == 0, \
        "queries got gradient THROUGH a zero head (impossible -> graph is wrong)"

    # (b) nudge the head so gradient flows everywhere upstream; states must STILL get none
    with torch.no_grad():
        dec.head.weight.normal_(0, 0.02)
    dec.zero_grad(set_to_none=True)
    out = dec(states)
    loss = norm_mse(out, target)
    loss.backward()
    for name in ("queries", "pos_row", "pos_col"):
        g = getattr(dec, name).grad
        assert g is not None and g.abs().sum() > 0, f"{name} got no gradient"
    for name, mod in (("fuse1", dec.fuse1), ("fuse2", dec.fuse2), ("head", dec.head)):
        assert mod.weight.grad is not None and mod.weight.grad.abs().sum() > 0, \
            f"{name}.weight got no gradient"
    for b, st in enumerate(states):
        for j, w in enumerate(st):
            assert w.grad is None, \
                f"state block {b} w{j} received gradient -- states are INPUTS, must stay detached"
    print("[selftest] T3 PASS  grads: head-only at zero-init; queries/pos/fusion/head after "
          "nudge; states ALWAYS grad-free")

    # ---------------- T4: overfit 8 synthetic pairs ----------------
    torch.manual_seed(4)
    dec = AnchorFrameDecoder().to(device)   # fresh, honest zero-init start
    states = synth_states(8, device, seed=5)
    target = torch.randn(8, 1, 16, 60, 104, device=device)
    opt = torch.optim.AdamW(dec.parameters(), lr=2e-3, weight_decay=0.0, betas=(0.9, 0.95))
    with torch.no_grad():
        init_loss = norm_mse(dec(states), target).item()   # == 1.0 by construction
    last = init_loss
    for it in range(200):
        loss = norm_mse(dec(states), target)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(dec.parameters(), 1.0)
        opt.step()
        last = loss.item()
        if (it + 1) % 25 == 0:
            print(f"[selftest]   overfit step {it+1:3d}: loss {last:.4f} (init {init_loss:.4f})")
    assert last < 0.10 * init_loss, \
        f"overfit sanity FAILED: loss {last:.4f} did not reach <10% of init {init_loss:.4f} in 200 steps"
    print(f"[selftest] T4 PASS  overfit 8 pairs: {init_loss:.4f} -> {last:.4f} "
          f"({100*last/init_loss:.1f}% of init) in 200 steps")

    # ---------------- T5: FiLM conditioning == EXACT v1.0 at init (warm-start) ----------------
    COND_DIM = 4096
    base = AnchorFrameDecoder().to(device)
    with torch.no_grad():                       # non-trivial output: a zero head would make
        base.head.weight.normal_(0, 0.02)       # the identity checks pass vacuously
        base.head.bias.normal_(0, 0.02)
    cdec = AnchorFrameDecoder(cond_dim=COND_DIM).to(device)
    missing, unexpected = cdec.load_state_dict(base.state_dict(), strict=False)
    assert not unexpected, f"v1.0 ckpt has keys unknown to the cond decoder: {unexpected[:5]}"
    assert missing and all(k.startswith(("cond_norm.", "cond_in.", "cond_out.")) for k in missing), \
        f"warm-start missing keys must be the cond modules ONLY, got {missing[:8]}"
    n_c = cdec.num_params()
    assert n_c == 33_137_728, \
        f"cond param count drifted from the documented 33,137,728: {n_c:,}"
    assert n_c < 40_000_000, f"cond variant blows the 40M budget: {n_c:,}"
    states = synth_states(2, device, seed=7)
    cond_vec = torch.randn(2, COND_DIM, device=device)
    with torch.no_grad():
        out_base = base(states)
        out_none = cdec(states)                  # cond omitted
        out_cond = cdec(states, cond=cond_vec)   # random cond through zero-init FiLM
    assert out_base.abs().max().item() > 0, "nudged base output is zero -- test is vacuous"
    assert torch.equal(out_none, out_base), "cond decoder (cond=None) != v1.0 output at init"
    assert torch.equal(out_cond, out_base), \
        "cond decoder WITH cond != v1.0 output at init (zero-init FiLM broken)"
    try:
        base(states, cond=cond_vec)
        raise SystemExit("cond on a cond_dim=0 decoder must raise")
    except AssertionError:
        pass
    print(f"[selftest] T5 PASS  warm-start (missing = {len(missing)} cond keys only), "
          f"init output == v1.0 bitwise (cond=None AND random cond), params {n_c:,} < 40M")

    # ---------------- T6: cond gradient routing ----------------
    target = torch.randn(2, 1, 16, 60, 104, device=device)
    cond_vec = torch.randn(2, COND_DIM, device=device, requires_grad=True)
    # (a) zero-init FiLM: cond_out gets gradient (same mechanism as the zero head in T3a),
    #     cond_in can only get EXACT ZERO (chain through the zero cond_out weight)
    out = cdec(states, cond=cond_vec)
    loss = norm_mse(out, target)
    loss.backward()
    assert cdec.cond_out.weight.grad is not None and \
        cdec.cond_out.weight.grad.abs().sum() > 0, "cond_out.weight got no gradient"
    assert cdec.cond_out.bias.grad is not None and \
        cdec.cond_out.bias.grad.abs().sum() > 0, "cond_out.bias got no gradient"
    g_in = cdec.cond_in.weight.grad
    assert g_in is None or g_in.abs().sum() == 0, \
        "cond_in got gradient THROUGH a zero cond_out (impossible -> graph is wrong)"
    assert cond_vec.grad is None, \
        "the cond INPUT received gradient -- cond is DATA, must stay detached inside"
    # (b) nudge cond_out -> gradient reaches the whole cond MLP; cond input still none
    with torch.no_grad():
        cdec.cond_out.weight.normal_(0, 0.02)
    cdec.zero_grad(set_to_none=True)
    out = cdec(states, cond=cond_vec)
    norm_mse(out, target).backward()
    for name, p in (("cond_in.weight", cdec.cond_in.weight),
                    ("cond_in.bias", cdec.cond_in.bias),
                    ("cond_norm.weight", cdec.cond_norm.weight)):
        assert p.grad is not None and p.grad.abs().sum() > 0, f"{name} got no gradient"
    assert cond_vec.grad is None, "cond input received gradient after nudge"
    # nudged FiLM must now actually CHANGE the output vs cond=None
    with torch.no_grad():
        d = (cdec(states, cond=cond_vec) - cdec(states)).abs().max().item()
    assert d > 0, "nudged FiLM does not affect the output -- conditioning path dead"
    print(f"[selftest] T6 PASS  grads: cond_out-only at zero-init; full cond MLP after nudge; "
          f"cond input ALWAYS grad-free; nudged FiLM changes output (max |d| = {d:.4f})")

    print("[selftest] ALL SELFTESTS PASSED")


if __name__ == "__main__":
    main()
