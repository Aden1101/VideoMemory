"""The recache window that carries a recalled anchor: what each window convention puts in
each slot. A pure tensor function of the switch pipeline, so it is tested without a model.

    python -m pytest tests/test_switch_window.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import torch

from pipeline.switch.decoded_anchor import AnchorDecodeMixin

CUR = 180          # the frame generation has reached when the switch happens

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
