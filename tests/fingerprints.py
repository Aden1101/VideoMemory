#!/usr/bin/env python
"""Bitwise fingerprints of the memory path, to tell a refactor from a behaviour change.

A refactor must leave every number unchanged, and "it still runs" does not show that. This
script runs two small deterministic computations and hashes what comes out: the memory cell
alone, and a miniature CausalWanModel rolled out with the memory branch live (read pass, then
the write pass that feeds the evicted chunk). It needs no weights.

    python tests/fingerprints.py --record   # on the code you trust: writes tests/fingerprints.json
    python tests/fingerprints.py            # on the code you changed: compares, exits 1 on a diff

To compare two revisions, copy this file onto the older one first
(`git checkout <newer> -- tests/fingerprints.py`); the recorded file is ignored by git, so it
survives the checkout back. Hashes depend on the device, the kernels and the torch version:
record and compare on the same machine. AF_DEVICE=cpu makes that easy.
"""
import argparse
import hashlib
import json
import os
import sys

import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils.device import device  # noqa: E402
from wan.modules.causal_model import CausalWanModel  # noqa: E402
from wan.modules.video_memory import TTTBlockCache, VideoMemoryLayer  # noqa: E402

RECORDED = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fingerprints.json")
# the cell options of the reported runs (configs/runs/run3_write_evicted_stable.yaml)
CELL = dict(bounded_gate=True, bounded_mod=True, rel_norm=True, dc_remove=True, use_decay=True)


def digest(*tensors):
    h = hashlib.sha256()
    for t in tensors:
        t = t.detach().to("cpu", torch.float32).contiguous()
        h.update(str(tuple(t.shape)).encode())
        h.update(t.numpy().tobytes())
    return h.hexdigest()[:16]


def cell_alone():
    """Four chunks through VideoMemoryLayer.step, the write fed the chunk from two steps back."""
    torch.manual_seed(0)
    cell = VideoMemoryLayer(dim=64, g_init=0.1, **CELL).eval()
    chunks = [torch.randn(1, 12, 64) for _ in range(4)]
    state, outs = None, []
    with torch.no_grad():
        for i, x in enumerate(chunks):
            out, state = cell.step(x, state=state, do_update=i >= 2,
                                   x_write=chunks[i - 2] if i >= 2 else None, num_frames=3)
            outs.append(out)
    return digest(*outs, *state)


def _build_model(**kwargs):
    """The cell options travel as one `memory_kwargs` dict, or as flat keyword arguments in the
    revisions that predate it: this script has to run on both sides of such a change."""
    cell = dict(gate_init=0.1, **CELL)
    try:
        return CausalWanModel(memory_kwargs=cell, **kwargs)
    except TypeError:
        return CausalWanModel(**cell, **kwargs)


def model_rollout():
    """A miniature model, six chunks, memory branch live: per chunk one read pass at t=1000
    and one write pass at t=0, as the inference pipeline does. The eviction delay of a 12-frame
    window is four chunks and chunk 0 is the sink, which is never written, so chunk 1 reaches
    the memory on the sixth write pass: the shortest rollout in which the memory writes."""
    heads, head_dim, layers, F, H, W, local = 12, 128, 2, 3, 16, 16, 12
    frame_seq = (H // 2) * (W // 2)
    dt = torch.bfloat16
    torch.manual_seed(0)
    m = _build_model(dim=heads * head_dim, ffn_dim=2048, num_heads=heads, num_layers=layers,
                     text_dim=64, text_len=16, local_attn_size=local, sink_size=3,
                     memory_layers=(1,), in_dim=16, out_dim=16,
                     mem_stream_scale=0.5, eviction_feed=True)
    m = m.to(device=device, dtype=dt).eval()
    with torch.no_grad():
        m.head.head.weight.normal_(0, 0.02)     # the output head is zero-init: perturb it
    kv = [{"k": torch.zeros(1, local * frame_seq, heads, head_dim, dtype=dt, device=device),
           "v": torch.zeros(1, local * frame_seq, heads, head_dim, dtype=dt, device=device),
           "global_end_index": torch.tensor([0], dtype=torch.long, device=device),
           "local_end_index": torch.tensor([0], dtype=torch.long, device=device)}
          for _ in range(layers)]
    xa = [{"k": torch.zeros(1, 512, heads, head_dim, dtype=dt, device=device),
           "v": torch.zeros(1, 512, heads, head_dim, dtype=dt, device=device),
           "is_init": False} for _ in range(layers)]
    ttt = [TTTBlockCache() for _ in range(layers)]
    ctx = [torch.randn(16, 64, device=device, dtype=dt)]
    outs = []
    with torch.no_grad():
        for c in range(6):
            x = [torch.randn(16, F, H, W, device=device, dtype=dt)]
            common = dict(context=ctx, seq_len=F * frame_seq, kv_cache=kv, crossattn_cache=xa,
                          current_start=c * F * frame_seq, ttt_cache=ttt)
            out = m(x, t=torch.full((1, F), 1000.0, device=device, dtype=dt),
                    ttt_update=False, **common)
            m(out, t=torch.zeros(1, F, device=device, dtype=dt), ttt_update=True, **common)
            outs.append(torch.stack(list(out)))
    init = m.blocks[1].memory.init_state(1, device, dt)
    assert any(not torch.equal(w.float(), w0.float()) for w, w0 in zip(ttt[1].state, init)), \
        "the memory state never left its init: the rollout is too short to write"
    return digest(*outs, *ttt[1].state)


CASES = {"cell_alone": cell_alone, "model_rollout": model_rollout}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--record", action="store_true", help="write the reference hashes")
    args = ap.parse_args()
    got = {name: fn() for name, fn in CASES.items()}
    got["_where"] = f"{device} torch {torch.__version__}"
    if args.record:
        with open(RECORDED, "w") as f:
            json.dump(got, f, indent=2)
        print(f"recorded {RECORDED}\n{json.dumps(got, indent=2)}")
        return
    with open(RECORDED) as f:
        want = json.load(f)
    if want["_where"] != got["_where"]:
        print(f"warning: recorded on {want['_where']}, running on {got['_where']}")
    bad = [k for k in CASES if want.get(k) != got[k]]
    for k in CASES:
        print(f"  {'DIFF' if k in bad else 'same'}  {k}  {want.get(k)} -> {got[k]}")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
