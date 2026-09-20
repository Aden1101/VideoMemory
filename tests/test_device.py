#!/usr/bin/env python
"""The device layer. The check that matters reimplements upstream's RoPE and demands bitwise
equality with the device-agnostic one, so no number measured on CUDA can move.

    python tests/test_device.py
"""
import os
import sys

import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils.device import (Timer, describe, device, free_memory_gb,  # noqa: E402
                          phase_dtype, supports_float64, synchronize, total_memory_gb)
from wan.modules.causal_model import causal_rope_apply  # noqa: E402
from wan.modules.model import rope_freqs_to, rope_params, rope_phase_dtype  # noqa: E402

OK = []


def check(name, cond, detail=""):
    assert cond, f"FAIL {name} {detail}"
    OK.append(name)
    print(f"  ok  {name}")


print(describe())


# ----------------------------- the reference: upstream's code --------------------------------
def causal_rope_apply_original(x, grid_sizes, freqs, start_frame=0):
    """Verbatim copy of the version before the device pass (float64 hardcoded)."""
    n, c = x.size(2), x.size(3) // 2
    freqs = freqs.split([c - 2 * (c // 3), c // 3, c // 3], dim=1)
    output = []
    for i, (f, h, w) in enumerate(grid_sizes.tolist()):
        seq_len = f * h * w
        x_i = torch.view_as_complex(x[i, :seq_len].to(torch.float64).reshape(seq_len, n, -1, 2))
        freqs_i = torch.cat([
            freqs[0][start_frame:start_frame + f].view(f, 1, 1, -1).expand(f, h, w, -1),
            freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
            freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1)], dim=-1).reshape(seq_len, 1, -1)
        x_i = torch.view_as_real(x_i * freqs_i).flatten(2)
        x_i = torch.cat([x_i, x[i, seq_len:]])
        output.append(x_i)
    return torch.stack(output).type_as(x)


HEADS, HEAD_DIM = 2, 16
F, H, W = 3, 4, 5
SEQ = F * H * W
torch.manual_seed(0)
d = HEAD_DIM
freqs_cpu = torch.cat([rope_params(1024, d - 4 * (d // 6)),
                       rope_params(1024, 2 * (d // 6)),
                       rope_params(1024, 2 * (d // 6))], dim=1)
x_cpu = torch.randn(1, SEQ, HEADS, HEAD_DIM)
grid = torch.tensor([[F, H, W]])

# --- the device resolves without ever querying an absent driver --------------------------
check("device resolved", device.type in ("cuda", "mps", "cpu"), device.type)
check("memory is readable", total_memory_gb() > 0 and free_memory_gb() >= 0,
      f"{free_memory_gb():.1f}/{total_memory_gb():.1f}")

# --- float64 is kept wherever it exists ---------------------------------------------------
check("CPU keeps float64", phase_dtype(torch.device("cpu")) is torch.float64)
check("CUDA would keep float64", supports_float64(torch.device("cuda")))
check("only Metal falls back to float32", phase_dtype(torch.device("mps")) is torch.float32)
check("the CPU table stays complex128",
      rope_freqs_to(freqs_cpu, torch.device("cpu")).dtype is torch.complex128)
check("rope_phase_dtype follows the table",
      rope_phase_dtype(freqs_cpu) is torch.float64
      and rope_phase_dtype(freqs_cpu.to(torch.complex64)) is torch.float32)

# --- bitwise equality with upstream's code, on a float64 device --------------------------
ref = causal_rope_apply_original(x_cpu, grid, freqs_cpu, start_frame=2)
new = causal_rope_apply(x_cpu, grid, freqs_cpu, start_frame=2)
check("RoPE bitwise identical to the original code (CPU)", torch.equal(ref, new),
      f"max|diff| = {float((ref - new).abs().max())}")
ref0 = causal_rope_apply_original(x_cpu, grid, freqs_cpu)
check("same without a frame offset", torch.equal(ref0, causal_rope_apply(x_cpu, grid, freqs_cpu)))

# --- on Metal it runs, and lands where it should ------------------------------------------
if device.type == "mps":
    fq = rope_freqs_to(freqs_cpu, device)
    check("the table is downcast to complex64 for Metal", fq.dtype is torch.complex64)
    got = causal_rope_apply(x_cpu.to(device), grid, fq, start_frame=2).cpu()
    err = float((got - ref).abs().max())
    check("RoPE executes on Metal", got.shape == ref.shape)
    # Tolerance: this is a float32-vs-float64 gap on a unit-norm rotation, so of the order of
    # float32 epsilon relative to the amplitude of x, not a difference of method.
    check("gap consistent with plain precision loss", err < 1e-4, f"max|diff| = {err:.2e}")
    check("norm is preserved (it is a rotation)",
          abs(float(got.norm()) - float(ref.norm())) < 1e-3)
    check("float64 refused on Metal, as expected", not supports_float64(device))
else:
    print("  ..  Metal checks skipped (no Metal device here)")

# --- the stopwatch returns milliseconds whichever path it takes ---------------------------
a, b = Timer(), Timer()
a.record()
_ = torch.randn(600, 600, device=device) @ torch.randn(600, 600, device=device)
b.record()
synchronize()
ms = a.elapsed_time(b)
check("stopwatch in milliseconds", 0.0 <= ms < 60000.0, f"{ms} ms")

# --- video writing has a usable path -------------------------------------------------------
from utils.video_io import backend, write_video  # noqa: E402
import tempfile  # noqa: E402
with tempfile.TemporaryDirectory() as tmp:
    p = os.path.join(tmp, "t.mp4")
    write_video(p, torch.randint(0, 255, (6, 32, 32, 3), dtype=torch.uint8), fps=8)
    check(f"video written via {backend()}", os.path.getsize(p) > 0)

print(f"\n{len(OK)}/{len(OK)} device tests PASS")
