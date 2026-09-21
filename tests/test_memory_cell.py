"""The memory cell itself, through the code the model calls: VideoMemoryLayer.step, write_only
and TTTBlockCache. Small, CPU, float32, no weights.

tests/test_grad.py checks the hand-derived write gradient against autograd; this file checks
the behaviour the README claims for the cell.

    python -m pytest tests/test_memory_cell.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import torch

from wan.modules.video_memory import TTTBlockCache, VideoMemoryLayer, linear_delta_chunk_update

DIM, TOKENS = 32, 12


def cell(**options):
    torch.manual_seed(0)
    return VideoMemoryLayer(dim=DIM, **options).eval()


def chunks(n, seed=1):
    g = torch.Generator().manual_seed(seed)
    return [torch.randn(1, TOKENS, DIM, generator=g) for _ in range(n)]


def test_a_zero_gate_makes_the_branch_exactly_inert():
    out, _ = cell(g_init=0.0, bounded_gate=True, rel_norm=True).step(chunks(1)[0])
    assert torch.equal(out, torch.zeros_like(out))


def test_the_read_uses_the_state_from_before_this_write():
    c, (x,) = cell(g_init=0.1), chunks(1)
    state = c.init_state(1, x.device, x.dtype)
    read_only, _ = c.step(x, state=state, do_update=False)
    read_then_write, new_state = c.step(x, state=state, do_update=True)
    assert torch.equal(read_only, read_then_write)
    assert not torch.equal(new_state[0], state[0])              # and the write did happen


def test_a_later_chunk_cannot_change_an_earlier_output():
    def rollout(xs):
        c, state, outs = cell(g_init=0.1), None, []
        for x in xs:
            out, state = c.step(x, state=state)
            outs.append(out)
        return outs

    xs = chunks(3)
    changed = xs[:2] + [xs[2] + 1.0]
    a, b = rollout(xs), rollout(changed)
    assert torch.equal(a[0], b[0]) and torch.equal(a[1], b[1])
    assert not torch.equal(a[2], b[2])


def test_a_write_rotates_the_state_without_changing_its_column_norms():
    c, xs = cell(g_init=0.1), chunks(4)
    state = c.init_state(1, xs[0].device, xs[0].dtype)
    before = [w.norm(dim=1) for w in state]
    for x in xs:
        _, state = c.step(x, state=state)
    for w, norm in zip(state, before):
        assert torch.allclose(w.norm(dim=1), norm, rtol=1e-3)


def test_with_decay_the_column_norms_never_exceed_their_init_value():
    c, xs = cell(g_init=0.1, use_decay=True), chunks(4)
    state = c.init_state(1, xs[0].device, xs[0].dtype)
    ceiling = [w.norm(dim=1) for w in state]
    for x in xs:
        _, state = c.step(x, state=state)
        for w, norm in zip(state, ceiling):
            assert bool((w.norm(dim=1) <= norm * (1 + 1e-3)).all())


def test_the_eviction_feed_writes_the_evicted_chunk_not_the_current_one():
    c = cell(g_init=0.1)
    current_a, current_b, evicted = chunks(3)
    state = c.init_state(1, evicted.device, evicted.dtype)
    _, after_a = c.step(current_a, state=state, x_write=evicted)
    _, after_b = c.step(current_b, state=state, x_write=evicted)
    _, wrote_current = c.step(current_a, state=state)
    for wa, wb, wc in zip(after_a, after_b, wrote_current):
        assert torch.equal(wa, wb)                  # the current chunk plays no part in the write
        assert not torch.equal(wa, wc)


@pytest.mark.parametrize("options", [{}, {"use_decay": True}])
def test_write_only_builds_the_state_that_step_builds(options):
    c = cell(g_init=0.1, **options)
    current, evicted = chunks(2)
    state = c.init_state(1, evicted.device, evicted.dtype)
    with torch.no_grad():
        _, from_step = c.step(current, state=state, x_write=evicted)
        replayed = c.write_only(evicted, state)
    for a, b in zip(from_step, replayed):
        assert torch.equal(a, b)


def test_the_delta_rule_stores_a_token_under_its_own_direction():
    x = torch.randn(1, DIM)
    W = linear_delta_chunk_update(x, torch.zeros(DIM, DIM), beta=1.0)
    assert torch.allclose(W @ (x[0] / x[0].norm()), x[0], atol=1e-5)


def test_the_linear_state_is_read_and_written_through_step():
    c = cell(g_init=0.1, state_type="linear_delta")
    current, evicted = chunks(2)
    out, state = c.step(current, x_write=evicted)
    assert len(state) == 1 and state[0].dtype == torch.float32
    assert state[0].shape == (1, DIM, DIM) and float(state[0].abs().max()) > 0
    assert out.shape == current.shape


def test_a_snapshot_survives_later_writes_and_can_be_restored_twice():
    cache = TTTBlockCache()
    cache.state, cache.ema, cache.fifo = ("w0", "w1", "w2"), {1000: "e"}, [(0, "h0")]
    snap = cache.snapshot()
    cache.state = ("x",)
    cache.ema[750] = "f"
    cache.fifo.pop(0)
    for _ in range(2):
        cache.restore(snap)
        assert (cache.state, cache.ema, cache.fifo) == (("w0", "w1", "w2"), {1000: "e"}, [(0, "h0")])
        cache.ema[500] = "g"                        # writing after a restore leaves the snapshot intact
        cache.fifo.append((1, "h1"))
