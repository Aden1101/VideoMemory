# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES
#
# Licensed under the Apache License, Version 2.0 (the "License").
# You may not use this file except in compliance with the License.
# To view a copy of this license, visit http://www.apache.org/licenses/LICENSE-2.0
#
# No warranties are given. The work is provided "AS IS", without warranty of any kind, express or implied.
#
# SPDX-License-Identifier: Apache-2.0
"""Saving and restoring the memory itself: one slot per subject, and the offline writer overlay."""
import json
import math
import os

import torch
import torch.distributed as dist

from utils.debug_option import DEBUG


# MEM_WRITER: replace the write parameters of the memory cells with ones trained offline, on
# replayed writes. Off unless the variable names a checkpoint, and nothing else changes: the
# rollout still runs with the injection silenced, only the writes that build the state differ.
# The checkpoint also holds the co-trained decoder, which travels through MEM_DECODE instead.
_MEMWRITER_CACHE = {}    # abs ckpt path -> ({block int -> {param name -> cpu tensor}}, step)
# == train_writer.WRITE_PARAMS: theta_w and nothing else (the read path stays stock).
_MEMWRITER_PARAMS = ("qkv.weight", "qkv.bias", "lrs.weight", "lrs.bias",
                     "k_norm.weight", "decay_logit")


def _find_causal_model(m):
    """Walk peft/wrapper layers down to the raw CausalWanModel (verbatim idiom of
    the pair-collection code). Type check, not hasattr:
    peft wrappers delegate attribute reads but setattr lands on the wrapper, so the backup
    bookkeeping must be attached to the real model."""
    seen = 0
    while m is not None and seen < 8:
        if type(m).__name__ == "CausalWanModel":
            return m
        m = getattr(m, "model", None) or getattr(m, "base_model", None)
        seen += 1
    return None


def _memwriter_load(ckpt_path):
    """Lazy-load (once per process) the writer tensors of a train_writer.py checkpoint."""
    key = os.path.abspath(ckpt_path)
    if key in _MEMWRITER_CACHE:
        return _MEMWRITER_CACHE[key]
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    assert isinstance(ck, dict) and "writer" in ck, \
        f"MEM_WRITER ckpt {ckpt_path} has no 'writer' entry (keys: {list(ck)[:6]})"
    writer = {}
    for bi, params in ck["writer"].items():
        assert isinstance(params, dict) and params, f"MEM_WRITER: empty writer block {bi}"
        unknown = set(params) - set(_MEMWRITER_PARAMS)
        assert not unknown, \
            f"MEM_WRITER: block {bi} carries non-write parameters {sorted(unknown)}, refusing " \
            f"to overlay anything outside theta_w {list(_MEMWRITER_PARAMS)}"
        writer[int(bi)] = {n: t.detach().clone() for n, t in params.items()}
    assert writer, f"MEM_WRITER ckpt {ckpt_path}: no blocks in 'writer'"
    step = ck.get("step", "?")
    _MEMWRITER_CACHE[key] = (writer, step)
    del ck
    return writer, step


def _memwriter_restore(cm):
    """Put the stock (checkpoint) write parameters back: an overlay must never leak into the
    next arm of the same process (one pipeline is built for every arm)."""
    backup = getattr(cm, "_memwriter_backup", None)
    if not backup:
        return
    with torch.no_grad():
        for (bi, name), t in backup.items():
            cm.blocks[bi].memory.get_parameter(name).data.copy_(t)
    cm._memwriter_backup = None
    prev = getattr(cm, "_memwriter_applied", None)
    cm._memwriter_applied = None
    print(f"[MEM_WRITER] stock write parameters RESTORED (was {prev})", flush=True)


def _memwriter_sync(model):
    """Called once per inference() call, before any write happens. MEM_WRITER empty/unset =>
    no-op (or a restore, if a previous arm of this process had overlaid a writer)."""
    path = os.environ.get("MEM_WRITER", "").strip()
    cm = _find_causal_model(model)
    applied = getattr(cm, "_memwriter_applied", None) if cm is not None else None
    if not path:
        if applied is not None:
            _memwriter_restore(cm)
        return
    assert cm is not None, "MEM_WRITER set but no CausalWanModel found under the wrappers"
    if applied == os.path.abspath(path):
        print(f"[MEM_WRITER] already applied ({path}), cells unchanged", flush=True)
        return
    if applied is not None:                    # switching writers: always measure vs the stock
        _memwriter_restore(cm)
    writer, step = _memwriter_load(path)
    backup, report = {}, []
    with torch.no_grad():
        for bi in sorted(writer):
            cell = getattr(cm.blocks[bi], "memory", None)
            assert cell is not None, \
                f"MEM_WRITER: block {bi} has no memory cell (memory_layers mismatch with the ckpt)"
            have = dict(cell.named_parameters())
            num = den = 0.0
            for name, t in writer[bi].items():
                assert name in have, f"MEM_WRITER: block {bi} memory cell has no parameter {name}"
                p = have[name]
                assert tuple(p.shape) == tuple(t.shape), \
                    f"MEM_WRITER: block {bi}.{name} shape {tuple(t.shape)} != {tuple(p.shape)}"
                new = t.to(device=p.device, dtype=p.dtype)
                backup[(bi, name)] = p.detach().clone()
                num += float((new.float() - p.detach().float()).abs().sum())
                den += float(p.detach().float().abs().sum())
                p.data.copy_(new)
            report.append(f"b{bi} dL1={num:.4e} (rel {num / max(den, 1e-12):.3e})")
    cm._memwriter_backup = backup
    cm._memwriter_applied = os.path.abspath(path)
    print(f"[MEM_WRITER] writer overlay from {path} (step {step}) on blocks {sorted(writer)}, "
          f"{len(backup)} tensors | " + " | ".join(report), flush=True)


class MemoryStateMixin:
    # Internal helpers
    def _snapshot_ttt(self):
        """Freeze the memory as it stands: the fast weights, the queue of evicted chunks waiting
        to be written, and the injection smoothing buffers. Used to give the memory one slot per
        subject, frozen when a subject leaves the video and restored when it comes back.

        The copies are shallow because the memory always replaces its state rather than editing
        it in place, so the tensors held here cannot change under us."""
        return [
            {
                "state": None if c.state is None else tuple(c.state),
                "fifo": list(getattr(c, "fifo", []) or []),
                "ema": dict(getattr(c, "ema", {}) or {}),
            }
            for c in self.ttt_cache
        ]

    def _restore_ttt(self, snap):
        for c, s in zip(self.ttt_cache, snap):
            c.state = s["state"]
            c.fifo = list(s["fifo"])
            c.ema = dict(s["ema"])
