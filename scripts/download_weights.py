#!/usr/bin/env python
"""Weight download, to the paths the code expects.

The README gives the two huggingface-cli commands, but the exact paths (and the fact that the
tokenizer lives in a `google/umt5-xxl/` subdirectory) can only be guessed by reading
utils/wan_wrapper.py. This script pins them down.

    python download_weights.py            # both repos, resumes automatically
    python download_weights.py --only af  # only the Anchor-Forcing checkpoints

huggingface_hub picks an interrupted file back up, so re-running is free and safe. Nothing is
overwritten if the file is already complete and its hash matches.
"""
import argparse
import os
import sys
import time

REPOS = {
    "wan": ("Wan-AI/Wan2.1-T2V-1.3B", "wan_models/Wan2.1-T2V-1.3B", 17.6),
    "af": ("young98/AnchorForcing", "ckpt", 8.8),
}


def human(n):
    return f"{n/1e9:.1f} GB"


def local_size(path):
    tot = 0
    for root, _, files in os.walk(path):
        for f in files:
            try:
                tot += os.path.getsize(os.path.join(root, f))
            except OSError:
                pass
    return tot


def main():
    ap = argparse.ArgumentParser(description="Wan2.1 + Anchor-Forcing weights")
    ap.add_argument("--only", choices=sorted(REPOS), default=None)
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()

    from huggingface_hub import snapshot_download

    keys = [a.only] if a.only else list(REPOS)
    t0 = time.time()
    for k in keys:
        repo, dest, gb = REPOS[k]
        os.makedirs(dest, exist_ok=True)
        print(f"\n=== {repo} -> {dest}/  (~{gb} GB) ===", flush=True)
        before = local_size(dest)
        snapshot_download(repo_id=repo, local_dir=dest, max_workers=a.workers,
                          resume_download=True)
        after = local_size(dest)
        print(f"    {human(after)} on disk (+{human(after - before)} this time)", flush=True)

    print(f"\nDone in {(time.time()-t0)/60:.1f} min.", flush=True)
    for k in keys:
        _, dest, _ = REPOS[k]
        print(f"  {dest:<34} {human(local_size(dest))}")


if __name__ == "__main__":
    sys.exit(main())
