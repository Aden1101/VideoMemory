# ============================================================================================
# ON-DISK PROMPT-EMBEDDING CACHE -- pass 1 of the two-pass split.
#
# WHY (2026-09-09). The umT5-XXL text encoder weighs 11.4 GB, the generator with its LoRA
# 3.6 GB, the KV cache 3.2 GB. On a 24 GB unified-memory machine shared with the OS, holding
# them TOGETHER does not fit: measured, the OS killed the process (SIGKILL) after saturating
# 15.5 GB of swap. Putting the encoder on the accelerator starves the generator; putting it in
# RAM overflows the system. There is no good side.
#
# The way out is to never hold them at the same time. One process encodes the prompts and
# EXITS -- that is the only way to actually return the memory; an in-process offload always
# leaves the weights somewhere in the shared pool. A second process generates, without ever
# constructing the encoder.
#
# WHAT IT DOES NOT COST. The prompt schedule is always known ahead of time in this repo: even
# the so-called "interactive" path reads its scenarios from a file, there is no keyboard input
# anywhere. Precomputing therefore removes no capability.
#
# IDENTITY GUARANTEE. We store the tensor EXACTLY as the encoder produced it, same dtype, no
# quantization. A cache read therefore returns the same tensor bit for bit, and a cached run is
# indistinguishable from an uncached one. This is non-negotiable: this project compares effects
# the size of its own chain noise, so a rounding error introduced here would be invisible and
# wrong.
#
# KEY = sha256 of the prompt text AND the dtype. The cache is therefore reusable across runs
# and across campaigns, and identical prompts are encoded only once.
# ============================================================================================
import hashlib
import os

import torch

DEFAULT_DIR = os.environ.get("AF_PROMPT_CACHE", "cache/prompt_embeds")


def dtype_tag(dtype) -> str:
    """Short, stable name of a torch dtype, for the key."""
    return str(dtype).replace("torch.", "")


def key_of(prompt: str, dtype=None) -> str:
    """The key of a prompt, DTYPE INCLUDED.

    The text is taken VERBATIM (no strip, no normalization): two prompts differing by one space
    are two different conditionings.

    The dtype is part of the key because the inference script casts the WHOLE pipeline to
    bfloat16, encoder included (inference.py). An embedding computed in float32 and then cast is
    NOT the one a bfloat16 encoder would have produced: weights and activations both differ.
    Mixing them would give a subtly wrong conditioning with no error raised -- exactly the class
    of bug this project has already paid two campaigns for.
    """
    tag = dtype_tag(dtype) if dtype is not None else "float32"
    return hashlib.sha256(f"{tag}|{prompt}".encode("utf-8")).hexdigest()


def path_of(prompt: str, cache_dir: str = None, dtype=None) -> str:
    k = key_of(prompt, dtype)
    d = cache_dir or DEFAULT_DIR
    return os.path.join(d, k[:2], f"{k}.pt")            # 2 levels: at most 256 directories


def has(prompt: str, cache_dir: str = None, dtype=None) -> bool:
    return os.path.exists(path_of(prompt, cache_dir, dtype))


def put(prompt: str, embed: torch.Tensor, cache_dir: str = None) -> str:
    """Write ONE prompt's embedding. Atomic write (temp file + rename): a process killed
    mid-write must not leave a half-written cache entry that a later read would take for valid
    -- exactly the fix this project already had to apply to its dump hook back in August."""
    p = path_of(prompt, cache_dir, embed.dtype)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    tmp = p + f".tmp{os.getpid()}"
    torch.save({"prompt": prompt, "embed": embed.detach().cpu(),
                "dtype": str(embed.dtype), "shape": list(embed.shape)}, tmp)
    os.replace(tmp, p)
    return p


def get(prompt: str, cache_dir: str = None, dtype=None):
    """Return the stored embedding, or None. Checks that the recorded prompt is the one asked
    for: a sha256 collision is impossible in practice, but a hand-copied cache or a corrupt
    file is not."""
    p = path_of(prompt, cache_dir, dtype)
    if not os.path.exists(p):
        return None
    try:
        obj = torch.load(p, map_location="cpu", weights_only=False)
    except Exception:
        return None                                      # damaged entry -> re-encode
    if obj.get("prompt") != prompt:
        raise RuntimeError(f"inconsistent cache: {p} holds a different prompt than requested")
    return obj["embed"]


def get_many(prompts, cache_dir: str = None, dtype=None):
    """Return (list of embeddings or None, indices of the misses)."""
    out = [get(p, cache_dir, dtype) for p in prompts]
    missing = [i for i, e in enumerate(out) if e is None]
    return out, missing


def stats(cache_dir: str = None):
    d = cache_dir or DEFAULT_DIR
    n = tot = 0
    for root, _, files in os.walk(d):
        for f in files:
            if f.endswith(".pt"):
                n += 1
                tot += os.path.getsize(os.path.join(root, f))
    return {"dir": d, "n": n, "bytes": tot}
