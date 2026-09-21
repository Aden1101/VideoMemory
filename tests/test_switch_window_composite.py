"""The recache window that carries a recalled anchor, and the subject/background composite.

Both are pure tensor functions of the switch pipeline, so they are tested without a model:
what each window convention puts in each slot, and that every control of the composite mask
changes what it claims to change and nothing else.

    python -m pytest tests/test_switch_window_composite.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import torch

from pipeline.switch.anchor_decode import AnchorDecodeMixin
from pipeline.switch.composite import CompositeMixin

H, W, CUR = 60, 104, 180


# ---- the recache window --------------------------------------------------------------------
def _video_and_anchor(k=1):
    g = torch.Generator().manual_seed(11)
    out = torch.randn(1, 240, 16, 6, 10, generator=g)
    return out, torch.randn(1, k, 16, 6, 10, generator=g)


def test_negative_offset_appends_the_anchor_after_the_true_recency():
    out, anchor = _video_and_anchor()
    n = 12
    window = AnchorDecodeMixin._memdec_window(out, CUR, n, anchor, 1, -1)
    assert torch.equal(window, torch.cat([out[:, CUR - (n - 1):CUR], anchor], dim=1))


def test_offset_zero_replaces_the_recency_slot_like_the_real_frame_replay():
    out, anchor = _video_and_anchor()
    n = 12
    window = AnchorDecodeMixin._memdec_window(out, CUR, n, anchor, 1, 0)
    assert torch.equal(window, torch.cat([out[:, CUR - n:CUR - 1], anchor], dim=1))
    appended = AnchorDecodeMixin._memdec_window(out, CUR, n, anchor, 1, -1)
    # the two conventions disagree on the 11 true slots, which is why both exist
    assert not torch.equal(appended[:, :n - 1], window[:, :n - 1])


@pytest.mark.parametrize("offset", [1, 2])
def test_positive_offset_pushes_the_anchor_back_and_keeps_true_frames_at_recency(offset):
    out, anchor = _video_and_anchor()
    n = 12
    window = AnchorDecodeMixin._memdec_window(out, CUR, n, anchor, 1, offset)
    pos = n - 1 - offset
    assert torch.equal(window[:, pos:pos + 1], anchor)
    assert torch.equal(window[:, pos + 1:], out[:, CUR - offset:CUR])
    assert torch.equal(window[:, :pos], out[:, CUR - n:CUR - n + pos])


def test_several_anchor_slots():
    out, anchor = _video_and_anchor(k=3)
    window = AnchorDecodeMixin._memdec_window(out, CUR, 12, anchor, 3, 0)
    assert window.shape[1] == 12
    assert torch.equal(window[:, -3:], anchor)
    assert torch.equal(window[:, :-3], out[:, CUR - 12:CUR - 3])


# ---- the composite -------------------------------------------------------------------------
class _Composite(CompositeMixin):
    _comp_applied, _comp_gate_now = 0, 1.0


@pytest.fixture
def scene(monkeypatch):
    """(composite(**env) -> tensor, anchor, background, plain elliptical mask)."""
    for k in [k for k in os.environ if k.startswith(("MEM_COMP", "REPLAY_PAST"))]:
        monkeypatch.delenv(k)
    g = torch.Generator().manual_seed(7)
    out = torch.randn(1, 200, 16, H, W, generator=g)
    anchor = torch.randn(1, 1, 16, H, W, generator=g)

    def composite(video=out, **env):
        monkeypatch.setenv("MEM_DECODE_COMPOSITE", "1")
        for k in [k for k in os.environ if k.startswith("MEM_COMP")]:
            monkeypatch.delenv(k)
        for k, v in env.items():
            monkeypatch.setenv(k, v)
        return _Composite()._maybe_composite(anchor, video, CUR)

    mask = CompositeMixin._composite_mask(H, W, 0.6, 0.3, torch.device("cpu"), torch.float32)
    return composite, anchor, out, mask


def test_plain_composite_is_the_masked_blend(scene):
    composite, anchor, out, m = scene
    bg = out[:, CUR - 1:CUR]
    assert torch.equal(composite(), m * anchor + (1 - m) * bg)


def test_uniform_mask_is_flat_and_dose_matched(scene):
    composite, anchor, out, m = scene
    bg = out[:, CUR - 1:CUR]
    weight = (composite(MEM_COMP_UNIFORM="1") - bg) / (anchor - bg)   # implied subject weight
    assert float(weight.std()) < 1e-5
    assert abs(float(weight.mean()) - float(m.mean())) < 1e-5


def test_anchor_noise_destroys_the_content_and_keeps_the_mask(scene):
    composite, anchor, out, _ = scene
    bg = out[:, CUR - 1:CUR]
    noised = composite(MEM_COMP_ANCHOR_NOISE="1.0")
    cy, cx = H // 2, W // 2
    assert torch.equal(noised[..., 0, 0], bg[..., 0, 0])                     # border: background
    assert float((noised[..., cy, cx] - anchor[..., cy, cx]).abs().max()) > 0.1
    assert float((noised - bg)[..., cy, cx].abs().mean()) > 0.1              # RMS kept, not zeroed


def test_shifted_and_inverted_masks(scene):
    composite, anchor, out, m = scene
    bg = out[:, CUR - 1:CUR]
    weight = (composite(MEM_COMP_SHIFT="0.3333") - bg) / (anchor - bg)
    assert abs(int(weight[0, 0, 0].mean(0).argmax()) - (W // 2 + W // 3)) <= 2
    inverted = composite(MEM_COMP_INVERT="1")
    assert torch.allclose(inverted, (1 - m) * anchor + m * bg, atol=1e-6)
    cy, cx = H // 2, W // 2
    assert torch.equal(inverted[..., cy, cx], bg[..., cy, cx])
    assert torch.equal(inverted[..., 0, 0], anchor[..., 0, 0])


def test_alpha_is_the_dose(scene):
    composite, anchor, _, _ = scene
    assert torch.equal(composite(MEM_COMP_ALPHA="0"), anchor)               # bitwise the anchor
    assert torch.allclose(composite(MEM_COMP_ALPHA="0.5"), 0.5 * anchor + 0.5 * composite(),
                          atol=1e-6)


def test_union_mask_also_covers_the_current_subject(scene):
    composite, anchor, out, m = scene
    blob = torch.zeros(1, 1, 16, H, W)
    blob[0, 0, :, 40:50, 80:95] = 3.0                                       # a subject, bottom right
    cy, cx = CompositeMixin._energy_centroid(blob)
    assert 38 <= cy <= 52 and 76 <= cx <= 98
    video = out.clone()
    video[:, CUR - 1] = blob[0, 0]
    bg = video[:, CUR - 1:CUR]
    weight = (composite(video, MEM_COMP_UNION="1") - bg) / (anchor - bg)
    assert float(weight[0, 0, 0, 45, 87]) > 0.9
    assert float(weight.mean()) > float(m.mean())


@pytest.mark.parametrize("mode", ["noise", "priority", "floor"])
def test_fill_weights_leave_the_background_outside_the_union_untouched(scene, mode):
    composite, _, out, _ = scene
    blob = torch.zeros(1, 1, 16, H, W)
    blob[0, 0, :, 40:50, 80:95] = 3.0
    video = out.clone()
    video[:, CUR - 1] = blob[0, 0]
    filled = composite(video, MEM_COMP_UNION="1", MEM_COMP_FILL=mode)
    # top-left corner: outside both ellipses, so anchor and noise weights are 0 there
    assert torch.equal(filled[..., 0, 0], video[:, CUR - 1:CUR][..., 0, 0])


@pytest.mark.parametrize("value", ["0", ""])
def test_off_returns_the_anchor_object_itself(scene, monkeypatch, value):
    _, anchor, out, _ = scene
    monkeypatch.setenv("MEM_DECODE_COMPOSITE", value)
    assert _Composite()._maybe_composite(anchor, out, CUR) is anchor


@pytest.mark.parametrize("mode", ["noise", "priority", "floor"])
def test_fill_weights_sum_to_one_with_the_background(mode):
    dev, dt = torch.device("cpu"), torch.float32
    m_a = CompositeMixin._composite_mask(H, W, 0.6, 0.3, dev, dt)                  # centred
    m_b = CompositeMixin._composite_mask(H, W, 0.6, 0.3, dev, dt, cy=45.0, cx=87.0)
    m = torch.maximum(m_a, m_b)
    w_anchor, w_noise = CompositeMixin._fill_weights(mode, m, m_a, m_b, 0.4)
    assert float(w_anchor.min()) >= 0.0 and float(w_noise.min()) >= -1e-6
    assert torch.allclose(w_anchor + w_noise + (1.0 - m), torch.ones_like(m), atol=1e-6)


def test_fill_floor_interpolates_between_priority_and_noise():
    dev, dt = torch.device("cpu"), torch.float32
    m_a = CompositeMixin._composite_mask(H, W, 0.6, 0.3, dev, dt)
    m_b = CompositeMixin._composite_mask(H, W, 0.6, 0.3, dev, dt, cy=45.0, cx=87.0)
    m = torch.maximum(m_a, m_b)
    fw = CompositeMixin._fill_weights
    assert torch.allclose(fw("floor", m, m_a, m_b, 0.0)[0], fw("priority", m, m_a, m_b, 0.0)[0])
    assert torch.allclose(fw("floor", m, m_a, m_b, 1.0)[0], fw("noise", m, m_a, m_b, 0.0)[0])
    assert CompositeMixin._fill_weights("", m, m_a, m_b, 0.4) == (None, None)
    with pytest.raises(ValueError):
        CompositeMixin._fill_weights("typo", m, m_a, m_b, 0.4)
