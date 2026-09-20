"""On-disk prompt-embedding cache, pass 1 of the two-pass split.

The umT5-XXL text encoder weighs 11.4 GB, the generator with its LoRA 3.6 GB, the KV cache
3.2 GB. On a 24 GB unified-memory machine shared with the OS, holding them together does not
fit: measured, the OS killed the process after saturating 15.5 GB of swap. Putting the
encoder on the accelerator starves the generator; putting it in RAM overflows the system.

The way out is to never hold them at the same time. One process encodes the prompts and
exits, which is the only way to actually return the memory (an in-process offload always
leaves the weights somewhere in the shared pool). A second process generates, without ever
constructing the encoder.

That costs no capability here: the prompt schedule is always known ahead of time, even on
the so-called interactive path, which reads its scenarios from a file.

Embeddings are stored exactly as the encoder produced them, same dtype, no quantization, so
a cached run is indistinguishable from an uncached one. This project compares effects the
size of its own chain noise, so a rounding error introduced here would be invisible and
wrong.

The key is the sha256 of the prompt text and the dtype, so the cache is reusable across runs
and evaluation runs, so identical prompts are encoded once.
"""
import hashlib
import os

import torch

DEFAULT_DIR = os.environ.get("AF_PROMPT_CACHE", "cache/prompt_embeds")


def dtype_tag(dtype) -> str:
    """Short, stable name of a torch dtype, for the key."""
    return str(dtype).replace("torch.", "")


def key_of(prompt: str, dtype=None) -> str:
    """The key of a prompt, dtype included.

    The text is taken verbatim, no strip and no normalization: two prompts differing by one
    space are two different conditionings.

    The dtype belongs in the key because inference.py casts the whole pipeline to bfloat16,
    encoder included. An embedding computed in float32 and then cast is not the one a bfloat16
    encoder would have produced, since weights and activations both differ. Mixing them gives
    a subtly wrong conditioning with no error raised, which is the class of bug this project
    has already paid for twice.
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
    """Write one prompt's embedding, through a temp file and a rename. A process killed
    mid-write must not leave a half-written entry that a later read would take for valid."""
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
