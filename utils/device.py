"""Single source of truth for the accelerator.

The repo assumed everywhere that the accelerator was an NVIDIA card: the device object was
built by querying the CUDA driver at import time (utils/memory.py), the timers were CUDA
events, AMP came from torch.cuda.amp, and RoPE went through float64. None of those
assumptions is necessary; they were just scattered across six files.

The contract is that on CUDA every function below delegates to the original torch.cuda call,
identically, so no number already measured can move. This is not an abstraction layer over
accelerators: it exposes the five things this repo asked of the driver and nothing more.

Manual override: AF_DEVICE=cpu|mps|cuda|cuda:1 in the environment.
"""
import os
import time

import torch

cpu = torch.device("cpu")
_GB = 1024 ** 3


def _detect():
    forced = os.environ.get("AF_DEVICE", "").strip()
    if forced:
        return torch.device(forced)
    if torch.cuda.is_available():
        return torch.device(f"cuda:{torch.cuda.current_device()}")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return cpu


# Resolved once at import, without querying a driver that may not be there: on a machine
# with no CUDA, asking torch.cuda for the current device raises.
device = _detect()
gpu = device                      # upstream's name for it, kept so its imports still work


def kind(dev=None):
    return (dev or device).type


def supports_float64(dev=None):
    """Metal has no float64 at all: the conversion raises TypeError. Wan's RoPE computes its
    phases in float64, so on CUDA we keep that dtype exactly and fall back to float32
    elsewhere."""
    return kind(dev) != "mps"


def phase_dtype(dev=None):
    """Dtype for RoPE phase computations. Float64 wherever it exists."""
    return torch.float64 if supports_float64(dev) else torch.float32


def high_precision_dtype(x=None):
    """The repo's "high precision" dtype: float64 wherever it exists, float32 on Metal.

    Accepts a tensor, a torch.device, a string, or nothing (the current device). The code
    calls `.double()` in several places where precision genuinely matters (flow/x0 conversion,
    scheduler noise addition, DMD losses), because a rounding error there gets multiplied by
    1/sigma when sigma is small. On CUDA and CPU the computation is therefore the original
    one; on Metal it degrades to float32.
    """
    if isinstance(x, torch.Tensor):
        dev = x.device
    elif x is None:
        dev = device
    else:
        dev = torch.device(x)
    return torch.float64 if dev.type != "mps" else torch.float32


def hp(x, dev=None):
    """`x` moved to `dev` (default: its own device) and cast to high precision.

    Order matters: casting then moving would fail, because a float64 tensor cannot be
    transferred to Metal. Move first, cast second.
    """
    target = torch.device(dev) if dev is not None else x.device
    return x.to(device=target, dtype=high_precision_dtype(target))


def synchronize(dev=None):
    k = kind(dev)
    if k == "cuda":
        torch.cuda.synchronize(dev)
    elif k == "mps":
        torch.mps.synchronize()


def empty_cache(dev=None):
    k = kind(dev)
    if k == "cuda":
        torch.cuda.empty_cache()
    elif k == "mps":
        torch.mps.empty_cache()


def total_memory_gb(dev=None):
    k = kind(dev)
    if k == "cuda":
        return torch.cuda.get_device_properties(dev).total_memory / _GB
    if k == "mps":
        # Metal's recommended working-set ceiling, not total RAM: on unified memory this is the
        # amount a process can actually expect to use.
        return torch.mps.recommended_max_memory() / _GB
    return float(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")) / _GB


def free_memory_gb(dev=None):
    """Available memory in GB.

    CUDA: the original computation, verbatim (driver-free bytes plus PyTorch's inactive
    reserved pool, reusable without going back to the driver).
    Metal: `mem_get_info` does not exist, so we take the recommended ceiling minus what the
    allocator holds. More conservative than the truth, and it serves the same purpose:
    deciding whether modules need to be moved.
    """
    k = kind(dev)
    if k == "cuda":
        stats = torch.cuda.memory_stats(dev)
        inactive_reserved = stats["reserved_bytes.all.current"] - stats["active_bytes.all.current"]
        free_driver, _ = torch.cuda.mem_get_info(dev)
        return (free_driver + inactive_reserved) / _GB
    if k == "mps":
        used = torch.mps.driver_allocated_memory() if hasattr(torch.mps, "driver_allocated_memory") \
            else torch.mps.current_allocated_memory()
        return max(0.0, torch.mps.recommended_max_memory() - used) / _GB
    return total_memory_gb(dev)


class Timer:
    """Stopwatch with the CUDA-event interface (record / elapsed_time in ms).

    CUDA and Metal both have real events; elsewhere we fall back to the wall clock after a
    synchronize. `elapsed_time` returns milliseconds on every path: a stopwatch that changes
    unit depending on the machine is worse than no stopwatch.
    """

    __slots__ = ("_dev", "_impl", "_t")

    def __init__(self, enable_timing=True, dev=None):
        self._dev = dev or device
        k = self._dev.type
        if k == "cuda":
            self._impl = torch.cuda.Event(enable_timing=enable_timing)
        elif k == "mps" and hasattr(torch.mps, "Event"):
            self._impl = torch.mps.Event(enable_timing=enable_timing)
        else:
            self._impl = None
        self._t = None

    def record(self):
        if self._impl is not None:
            self._impl.record()
        else:
            synchronize(self._dev)
            self._t = time.perf_counter()

    def elapsed_time(self, end):
        """Milliseconds between this stopwatch and `end` (same direction as torch.cuda.Event)."""
        if self._impl is not None:
            return self._impl.elapsed_time(end._impl)
        return (end._t - self._t) * 1000.0


def describe():
    """One line for the logs: what was found and what will be used."""
    k = kind()
    if k == "cuda":
        name = torch.cuda.get_device_name(device)
    elif k == "mps":
        name = "Apple GPU (Metal)"
    else:
        name = "CPU"
    return (f"device={device} ({name}) | memory {free_memory_gb():.1f}/"
            f"{total_memory_gb():.1f} GB | float64 "
            f"{'yes' if supports_float64() else 'no (RoPE in float32)'}")
