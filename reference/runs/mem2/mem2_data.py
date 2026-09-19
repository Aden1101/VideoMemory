#!/usr/bin/env python
"""Memory 2.0 (write-replay), shared data layer for J2/J3.
Reads the v3 "h_replay" shards written by reference/runs/frame_dec/collect_pairs.py --v3
(format documented in that file's header + memory2_design.md 2.5) and gives the bank
builder / h-oracle probe / writer trainer one definition of:
  - the video INDEX (metadata only, no big tensor materialized)          -> build_index()
  - the per-video train/val split (red-team F4.1: >= 20 val videos)      -> assign_split()
  - the twin pairing (same prompt, seed + 100000: red-team F1.1b)        -> twin_of()
  - target frames, their AGE in writes, and the cond that goes with them -> targets_for()
  - the latent crop60 embedding used by the (VAE-free) InfoNCE           -> latent_embed()

Additive: nothing outside reference/runs/mem2/ is imported-and-modified; collect_pairs.py is
imported read-only (corpus/held-out helpers) so the held-out grep selftest uses the very same
code that filtered the corpus.

Shard access: torch.load(..., mmap=True), the 11-14 GB parts are memory-mapped, so reading
one video's h (1.5 GB) never materializes its 5-8 neighbours. Index building touches metadata
only and takes ~1 s for the whole 191 GB campaign.

AGE convention (used everywhere, including wandb bucket names):
  write index i = position in write_chunks (0-based). After write i the state has just
  absorbed evicted chunk write_chunks[i] - window_chunks, whose last latent frame is
  frame_of_write(i). A target of AGE a decoded from the state at index i is the frame
  evicted at write index j = i - (a - 1):  age 1 = "your latest write" (the trivial recency
  target the red-team F1.6 warns about), age 20 = 20 writes ago.
"""
import glob
import hashlib
import json
import math
import os
import sys

REPO = "/shared/ssd_14T/home/adenelmi/Anchor-Forcing"
FRAME_DEC = os.path.join(REPO, "reference/runs/frame_dec")
sys.path.insert(0, REPO)
sys.path.insert(0, FRAME_DEC)

import torch  # noqa: E402

V3_DIR = "/shared/ssd_14T_project/adenelmi/mem2_h_v3"
TWIN_DIR = "/shared/ssd_14T_project/adenelmi/mem2_h_v3_twins"
RUNS_DIR = "/shared/ssd_14T_project/adenelmi/mem2_runs"
INDEX_JSON = os.path.join(RUNS_DIR, "video_index.json")
TWIN_SEED_OFFSET = 100000          # collect_pairs.py --twin-offset used for the twin campaign
FRAME_SEQLEN = 1560                # tokens per latent frame (60/2 * 104/2)
MIN_GAP_CHUNKS = 5                 # collector rule: anchor positions at chunk >= switch + 5


# ----------------------------- shard listing / loading --------------------------------------
def list_parts(dirs=(V3_DIR, TWIN_DIR)):
    """All hreplay_*.pt parts of the given campaign dirs, sorted, existing dirs only."""
    out = []
    for d in dirs:
        if not os.path.isdir(d):
            continue
        out += sorted(glob.glob(os.path.join(d, "hreplay_*.pt")))
    return out


def load_part(path, mmap=True):
    """Memory-mapped shard load. Big tensors stay on disk until touched."""
    obj = torch.load(path, map_location="cpu", weights_only=False, mmap=mmap)
    assert isinstance(obj, dict) and obj.get("format_version") == 3 \
        and obj.get("kind") == "h_replay", f"{path}: not a v3 h_replay shard"
    assert obj.get("mem_zero_all") is True, \
        (f"{path}: mem_zero_all is {obj.get('mem_zero_all')!r}, the write-replay premise "
         f"(h = data, independent of theta_w) only holds for MEM_ZERO_ALL=1 rollouts")
    return obj


# ----------------------------- index ---------------------------------------------------------
def _video_meta(rec, part, vi, twin):
    return {
        "video_id": ("twin:" if twin else "main:") + str(int(rec["corpus_idx"])),
        "part": part, "vi": int(vi), "twin": bool(twin),
        "corpus_idx": int(rec["corpus_idx"]),
        "corpus_idx_b": int(rec.get("corpus_idx_b", -1)),
        "seed": int(rec["seed"]), "mode": str(rec["mode"]),
        "frames": int(rec["frames"]),
        "write_chunks": [int(c) for c in rec["write_chunks"]],
        "switch_frame": (int(rec["switch_frame"]) if rec.get("switch_frame") is not None
                         else None),
        "switch_chunk": (int(rec["switch_chunk"]) if rec.get("switch_chunk") is not None
                         else None),
        "window_chunks": int(rec["window_chunks"]), "sink_chunks": int(rec["sink_chunks"]),
        "num_frame_per_block": int(rec["num_frame_per_block"]),
        "prompt": str(rec["prompt"]),
        "prompt_b": (str(rec["prompt_b"]) if rec.get("prompt_b") else None),
        "has_check_states": rec.get("check_states") is not None,
    }


def build_index(dirs=(V3_DIR, TWIN_DIR), verbose=True):
    """Scan every part's metadata (no h/latent materialized) -> list of video meta dicts."""
    vids = []
    for p in list_parts(dirs):
        twin = "_twins/" in p or os.path.basename(os.path.dirname(p)).endswith("_twins")
        obj = load_part(p)
        for vi, rec in enumerate(obj["videos"]):
            vids.append(_video_meta(rec, p, vi, twin))
        del obj
    vids.sort(key=lambda v: (v["twin"], v["corpus_idx"]))
    if verbose:
        n_sw = sum(1 for v in vids if v["mode"] == "switch")
        n_tw = sum(1 for v in vids if v["twin"])
        print(f"[mem2_data] index: {len(vids)} videos ({n_sw} switch / {len(vids) - n_sw} "
              f"single, {n_tw} twins) over {len(list_parts(dirs))} parts", flush=True)
    return vids


def save_index(vids, path=INDEX_JSON):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(vids, f)
    os.replace(tmp, path)
    return path


def load_index(path=INDEX_JSON, dirs=(V3_DIR, TWIN_DIR), rebuild=False, verbose=True):
    if not rebuild and os.path.exists(path):
        with open(path) as f:
            vids = json.load(f)
        if verbose:
            print(f"[mem2_data] index loaded from {path}: {len(vids)} videos", flush=True)
        return vids
    vids = build_index(dirs, verbose=verbose)
    save_index(vids, path)
    return vids


# ----------------------------- held-out selftest (F. brief: grep the manifests) --------------
def held_out_hits(vids):
    """Re-verify (grep) that no selected prompt hits one of the 16 probe prompts, in both
    directions, the collector asserted it at selection time, this re-checks the shards that
    actually landed on disk (prompt and prompt_b of every video)."""
    from collect_pairs import held_out_prompts, _hits_held_out, _norm_text
    held = held_out_prompts()
    hits = []
    for v in vids:
        for key in ("prompt", "prompt_b"):
            p = v.get(key)
            if p and _hits_held_out(_norm_text(p), held):
                hits.append((v["video_id"], key, p[:80]))
    return hits, len(held)


# ----------------------------- split (F4.1: per video, >= 20 val videos) ---------------------
def _hash01(s):
    return int(hashlib.sha1(s.encode()).hexdigest()[:12], 16) / float(1 << 48)


def assign_split(vids, n_val_switch=18, n_val_single=4, salt="mem2-v1"):
    """Deterministic per-video split. Rules:
      - stratified by mode so val contains both populations (>= 20 videos total);
      - a TWIN always lands on the side of its sibling (key = corpus_idx), red-team F1.1b:
        a twin in val while its sibling trains would leak the exact scene.
    Returns {video_id: 'train'|'val'} and mutates nothing."""
    main = [v for v in vids if not v["twin"]]
    by_mode = {}
    for v in main:
        by_mode.setdefault(v["mode"], []).append(v)
    val_cidx = set()
    for mode, n in (("switch", n_val_switch), ("single", n_val_single)):
        pool = sorted(by_mode.get(mode, []), key=lambda v: _hash01(salt + v["video_id"]))
        assert len(pool) >= n, f"only {len(pool)} {mode} videos, need {n} for val"
        val_cidx.update(v["corpus_idx"] for v in pool[:n])
    split = {v["video_id"]: ("val" if v["corpus_idx"] in val_cidx else "train") for v in vids}
    return split


def twin_index(vids):
    """corpus_idx -> {'main': video_id | None, 'twin': video_id | None} (F1.1b pairing key)."""
    idx = {}
    for v in vids:
        e = idx.setdefault(v["corpus_idx"], {"main": None, "twin": None})
        e["twin" if v["twin"] else "main"] = v["video_id"]
    return idx


def twin_of(video_id, tidx, vids_by_id):
    v = vids_by_id[video_id]
    e = tidx.get(v["corpus_idx"], {})
    other = e["main"] if v["twin"] else e["twin"]
    return other if other != video_id else None


# ----------------------------- targets / ages -------------------------------------------------
def frame_of_write(meta, i):
    """Latent frame index of the target evicted by write i (its chunk's last frame)."""
    wc, F = meta["window_chunks"], meta["num_frame_per_block"]
    c = meta["write_chunks"][i]
    return (c - wc) * F + (F - 1)


def anchor_write_index(meta):
    """Write index j whose evicted frame is the anchor (last frame of A, switch_frame - 1).
    None for single videos."""
    if meta["switch_chunk"] is None:
        return None
    c = meta["switch_chunk"] - 1 + meta["window_chunks"]
    wcs = meta["write_chunks"]
    return wcs.index(c) if c in wcs else None


def anchor_eligible_states(meta):
    """State (write) indices where the anchor target is the revisit task: A fully evicted from
    the KV window and >= 2 pure-B chunks written (the collector's min_gap rule)."""
    if meta["switch_chunk"] is None:
        return []
    lo = meta["switch_chunk"] + MIN_GAP_CHUNKS
    return [i for i, c in enumerate(meta["write_chunks"]) if c >= lo]


def cond_for_frame(meta, frame):
    """'a' or 'b': which pooled prompt embedding describes the content of that latent frame."""
    sf = meta["switch_frame"]
    return "a" if (sf is None or frame < sf) else "b"


def sample_positions(meta, rng, k=3, ages=(1, 2, 5, 10, 20), anchor_frac=0.5,
                     min_age_writes=1):
    """K supervised positions for one video, as dicts:
        {'i': state index, 'j': target write index, 'age': i-j+1, 'frame': latent frame,
         'cond': 'a'|'b', 'kind': 'anchor'|'recency'}
    'anchor' positions (switch videos only) are the revisit task: target = last frame of A,
    state = post-switch, cond = A. 'recency' positions implement red-team F1.6 (aged targets
    sampled in {1,2,5,10,20} writes, never only age 1)."""
    n = len(meta["write_chunks"])
    out = []
    aj = anchor_write_index(meta)
    elig = anchor_eligible_states(meta)
    n_anchor = int(round(k * anchor_frac)) if (aj is not None and elig) else 0
    ok_anchor = [i for i in elig if (i - aj + 1) >= min_age_writes] if n_anchor else []
    for _ in range(n_anchor):
        if not ok_anchor:      # min_age not reachable for this switch position -> no anchor,
            break              # the remaining slots become aged recency targets (never a
        i = rng.choice(ok_anchor)                        # target younger than min_age_writes)
        out.append({"i": int(i), "j": int(aj), "age": int(i - aj + 1),
                    "frame": int(frame_of_write(meta, aj)), "cond": "a", "kind": "anchor"})
    ages = [a for a in ages if a >= min_age_writes] or [min_age_writes]
    guard = 0
    while len(out) < k and guard < 100:
        guard += 1
        a = rng.choice(list(ages))
        if a > n:
            continue
        i = rng.randrange(a - 1, n)
        j = i - (a - 1)
        f = frame_of_write(meta, j)
        out.append({"i": int(i), "j": int(j), "age": int(a), "frame": int(f),
                    "cond": cond_for_frame(meta, f), "kind": "recency"})
    return out


AGE_BUCKETS = (1, 2, 5, 10, 20)


def age_bucket(age):
    """Bucket name for wandb (loss_by_age, red-team F3.2)."""
    for a in AGE_BUCKETS:
        if age <= a:
            return f"a{a}"
    return "a20p"


# ----------------------------- video payload access --------------------------------------------
class PartCache:
    """Keeps the last `size` memory-mapped parts open (the trainer draws several videos from
    the same part in a row; re-opening a 12 GB zip per sample costs ~0.3 s)."""

    def __init__(self, size=2):
        self.size = size
        self._d = {}
        self._order = []

    def get(self, path):
        if path not in self._d:
            self._d[path] = load_part(path)
            self._order.append(path)
            while len(self._order) > self.size:
                self._d.pop(self._order.pop(0), None)
        return self._d[path]

    def clear(self):
        self._d.clear()
        self._order.clear()


def get_record(meta, cache):
    return cache.get(meta["part"])["videos"][meta["vi"]]


def load_h(meta, cache, blocks, upto=None, device=None, dtype=torch.bfloat16):
    """h_seq[block][0..upto] -> list per block. Stored fp16 -> cast to the deployment dtype
    (bf16) exactly like selftest_replay's test 4 (round-trip exact for in-range values)."""
    rec = get_record(meta, cache)
    out = {}
    for b in blocks:
        seq = rec["h_seq"][str(b)]
        n = len(seq) if upto is None else int(upto) + 1
        hs = []
        for t in seq[:n]:
            t = t.to(device=device, non_blocking=True) if device is not None else t
            hs.append(t.to(dtype))
        out[b] = hs
    return out


def load_frames(meta, cache, frames):
    """Target latent frames (fp16 cpu clones, safe to keep after the mmap is dropped)."""
    rec = get_record(meta, cache)
    lat = rec["latents"]
    return torch.stack([lat[int(f)].clone() for f in frames]).float()


def load_conds(meta, cache):
    rec = get_record(meta, cache)
    cb = rec.get("cond_emb_b")
    return (rec["cond_emb_a"].clone().float(),
            cb.clone().float() if cb is not None else None)


# ----------------------------- latent embedding (VAE-free InfoNCE space) ----------------------
def latent_embed(x, frac=0.6, pool=2, eps=1e-6):
    """(B, 16, 60, 104) or (B, 1, 16, 60, 104) latent frames -> L2-normalized crop60 embedding.
    The mandate keeps the VAE out of the loss (viz only), so the instance term lives in latent
    space: centre crop60 (the region re-ID scores), 2x2 average pool (kills the pixel-adversarial
    high-frequency directions of red-team F1.5), per-sample RMS normalization then L2 norm ->
    plain cosine similarity. Differentiable (the anchor side carries the gradient)."""
    if x.dim() == 5:
        x = x[:, 0]
    assert x.dim() == 4, f"latent_embed expects (B, C, H, W), got {tuple(x.shape)}"
    B, C, H, W = x.shape
    ch, cw = int(H * frac), int(W * frac)
    top, left = (H - ch) // 2, (W - cw) // 2
    z = x[..., top:top + ch, left:left + cw].float()
    if pool and pool > 1:
        z = torch.nn.functional.avg_pool2d(z, pool)
    z = z.flatten(1)
    z = z / z.pow(2).mean(dim=1, keepdim=True).clamp_min(eps).sqrt()
    return torch.nn.functional.normalize(z, dim=1)


def latent_embed_dim(frac=0.6, pool=2, C=16, H=60, W=104):
    ch, cw = int(H * frac), int(W * frac)
    if pool and pool > 1:
        ch, cw = ch // pool, cw // pool
    return C * ch * cw


# ----------------------------- misc ------------------------------------------------------------
def frame_token_slice(frame_in_chunk, F=3, seqlen=FRAME_SEQLEN):
    """h is (1, F*seqlen, dim) laid out frame-major (causal_model unflattens (num_frames,
    frame_seqlen)) -> the tokens of the chunk's k-th frame."""
    assert 0 <= frame_in_chunk < F
    return slice(frame_in_chunk * seqlen, (frame_in_chunk + 1) * seqlen)


def gb(nbytes):
    return nbytes / 1e9


def describe(vids):
    n_w = sum(len(v["write_chunks"]) for v in vids)
    return (f"{len(vids)} videos, {n_w} writes, "
            f"{sum(1 for v in vids if v['mode'] == 'switch')} switch, "
            f"{sum(1 for v in vids if v['twin'])} twins")


if __name__ == "__main__":
    vids = load_index(rebuild="--rebuild" in sys.argv)
    print("[mem2_data]", describe(vids))
    hits, n_held = held_out_hits(vids)
    print(f"[mem2_data] held-out grep: {len(hits)} hit(s) on {n_held} probe prompts")
    split = assign_split(vids)
    n_val = sum(1 for k, v in split.items() if v == "val")
    print(f"[mem2_data] split: {len(split) - n_val} train / {n_val} val videos")
