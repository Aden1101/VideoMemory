#!/usr/bin/env python
"""Pass 1: encode the prompts, write the cache, and exit.

The exit is the point. An in-process offload always leaves the 11.4 GB somewhere in the
shared pool of a unified-memory machine; only the end of the process returns them to the OS.
Pass 2 (inference/inference.py with AF_LAZY_TEXT_ENCODER=1) then never constructs the
encoder and keeps the memory for the generator.

    python scripts/encode_prompts.py --file ckpt/prompts/bench/vbench_all_dimension_extended.txt --limit 1
    python scripts/encode_prompts.py --file inference/interactive_case.jsonl        # switch scenarios
    python scripts/encode_prompts.py --stats                                        # cache state

Then:
    AF_LAZY_TEXT_ENCODER=1 PYTHONPATH=. python inference/inference.py --config_path ...

Accepted formats: .txt (one prompt per line) and .jsonl (key "prompts", one list per line,
the revisit-scenario format). Duplicates are encoded once.
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def read_prompts(path):
    """The prompts of a .txt file, one per line, or of a .jsonl file, whose lines hold either
    one prompt object or a {"prompts": [...]} list."""
    if path.endswith(".jsonl"):
        out = []
        with open(path, encoding="utf8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                out.extend(obj["prompts"] if isinstance(obj.get("prompts"), list) else [obj])
        return out
    # A full `.rstrip()`, not just the newline, which is what utils.dataset.TextDataset does.
    # The cache key is the verbatim text, so a trailing space left in here would produce a
    # silent cache miss on every run.
    with open(path, encoding="utf-8") as f:
        return [l.rstrip() for l in f if l.strip()]


def cache_summary(cache_dir):
    from utils import prompt_cache
    st = prompt_cache.stats(cache_dir)
    return f"cache {st['dir']}: {st['n']} prompts, {st['bytes'] / 1e6:.0f} MB"


def main():
    ap = argparse.ArgumentParser(description="pass 1: prompt-embedding cache")
    ap.add_argument("--file", action="append", default=[],
                    help="prompt file (.txt or .jsonl); repeatable")
    ap.add_argument("--prompt", action="append", default=[], help="a literal prompt; repeatable")
    ap.add_argument("--limit", type=int, default=0, help="take only the first N")
    ap.add_argument("--batch", type=int, default=4, help="prompts encoded per batch")
    ap.add_argument("--cache_dir", default=None)
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float32", "float16"],
                    help="encoding precision. bfloat16 by default because inference.py does "
                         "`pipeline.to(dtype=torch.bfloat16)`: pass 1 must encode in the same "
                         "precision as pass 2, otherwise the conditioning differs.")
    ap.add_argument("--stats", action="store_true", help="print the cache state and exit")
    a = ap.parse_args()

    from utils import prompt_cache

    if a.stats:
        print(cache_summary(a.cache_dir))
        return 0

    prompts = list(a.prompt)
    for f in a.file:
        prompts += read_prompts(f)
    if a.limit:
        prompts = prompts[:a.limit]
    if not prompts:
        ap.error("nothing to encode: give --file or --prompt")

    uniq = list(dict.fromkeys(prompts))          # de-duplicated, still in file order
    import torch                                 # imported here, after argparse
    want = getattr(torch, a.dtype)
    todo = [p for p in uniq if not prompt_cache.has(p, a.cache_dir, want)]
    print(f"{len(prompts)} prompts read, {len(uniq)} unique, {len(uniq)-len(todo)} already cached, "
          f"{len(todo)} to encode", flush=True)
    if not todo:
        print(f"nothing to do. {cache_summary(a.cache_dir)}")
        return 0

    from utils.device import describe
    from utils.wan_wrapper import WanTextEncoder

    print(describe(), flush=True)
    t0 = time.time()
    te = WanTextEncoder()                      # eager load here, which is the point of this script
    te = te.to(dtype=want)                     # the cast inference.py applies to the pipeline
    print(f"encoder loaded in {time.time()-t0:.0f} s, precision {a.dtype}", flush=True)

    done, t0 = 0, time.time()
    with torch.no_grad():
        for i in range(0, len(todo), a.batch):
            chunk = todo[i:i + a.batch]
            emb = te.encode(chunk).to(want)    # encode() bypasses the cache, deliberately
            for p, e in zip(chunk, emb):
                prompt_cache.put(p, e, a.cache_dir)
            done += len(chunk)
            el = time.time() - t0
            print(f"  {done}/{len(todo)}  ({el/done:.1f} s/prompt, ~"
                  f"{el/done*(len(todo)-done)/60:.1f} min left)", flush=True)

    print(f"\n{cache_summary(a.cache_dir)}")
    print("pass 1 done. Launch pass 2 with AF_LAZY_TEXT_ENCODER=1.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
