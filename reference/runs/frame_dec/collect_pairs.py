#!/usr/bin/env python
"""frame_dec data collection (design v1 "frame de rappel decodee", plan_final_sink_virtuel.md):
collect (memory state, evicted latent frame) pairs from self-generated single-prompt
rollouts of the l1@2000-EMA checkpoint, to train the offline state->frame decoder.

Mechanism (matches the c2 eviction-feed exactly, wan/modules/causal_model.py ~L1342-1385):
  window_chunks = ceil(local_attn_size / F) = ceil(12/3) = 4. On the write pass
  (ttt_update=True) of chunk c, the model pops chunk c-4 from each memory block's fifo and
  writes it into the fast weight (unless it is a sink chunk, popped < sink_chunks). So right
  after the write pass of chunk c, the state has just absorbed the evicted chunk c-4, whose
  most recent latent frame (the one that "just left the 12-frame window") is frame
  3*(c-4)+2. That frame is the reconstruction target for the decoder.

Shard format v2 (.pt, for the decoder-training agent), torch.load(path) gives:
  {
    "format_version": 2,                     # v2 = v1 + per-pair "cond_emb"/"mode"/"target_frame"
                                             #      (+ switch bookkeeping in switch pairs);
                                             #      v1 shards (no cond_emb) stay trainer-readable
    "memory_layers": [5, 16, 27],            # block order of the "states" list
    "state_keys":   ["w0", "w1", "w2"],      # order inside each per-block tuple
    "state_shapes": {"w0": [1, 1536, 6144],  # (B*num_heads, head_dim, head_dim*4)
                     "w1": [1, 6144, 1536],
                     "w2": [1, 1536, 6144]},
    "target_shape": [16, 60, 104],           # one latent frame [C, H, W]
    "cond_dim": 4096,                        # T5 umt5-xxl channel dim of "cond_emb"
    "cond_key": "cond_emb",
    "cond_pooling": "masked mean over non-padding rows", fp16 cpu
    "collect_mode": "single" | "switch" | "mix",
    "pairs": [                               # ~200 per shard part (--shard-size)
      {
        "states": [ (w0, w1, w2), (w0, w1, w2), (w0, w1, w2) ],  # fp16 cpu, blocks 5/16/27
        "target": Tensor[16, 60, 104] fp16 cpu,  # single: the just-evicted latent frame;
                                             # switch: the last latent frame of A before the
                                             # switch (the anchor-frame candidate)
        "cond_emb": Tensor[4096] fp16 cpu,   # pooled T5 embedding of the prompt whose content
                                             # the decoder must retrieve: the rollout prompt in
                                             # single mode, prompt A in switch mode (new in v2)
        "mode": "single" | "switch",         # new in v2
        "target_frame": int,                 # new in v2: latent index of "target" in the rollout
                                             # (single: == evicted_frame; switch: switch_frame-1)
        "prompt": str,                       # single: the rollout prompt; switch: prompt A
        "chunk_idx": int,                    # chunk whose write pass just ran (0-based)
        "evicted_chunk": int,                # = chunk_idx - window_chunks (bookkeeping; in
        "evicted_frame": int,                #   switch mode this is a B chunk, not the target)
        "corpus_idx": int,                   # line index in the vidprom corpus file
        "seed": int,                         # torch/python seed of this rollout
        "num_frame_per_block": int,          # F = 3
        "window_chunks": int,                # 4
        "sink_chunks": int,                  # 1
       , switch pairs only (mode == "switch"):
        "prompt_b": str,                     # the interlude prompt B (another corpus line)
        "corpus_idx_b": int,
        "switch_frame": int,                 # frames >= this used prompt B (in {60, 81, 102})
        "switch_chunk": int,                 # = switch_frame // F; collection only at
                                             #   chunk_idx >= switch_chunk + min_gap_chunks (5)
                                             #   => A fully evicted, states are post-B-writes
      }, ...
    ],
    "meta": { config paths, ckpt, frames, ... }
  }
  Per-pair size ~170 MB (3 blocks x 3 fp16 state tensors of 1536x6144 = 169.9 MB + 0.2 MB
  target). 8 positions/video -> ~1.36 GB/video, 400 videos -> ~544 GB: over the 150 GB
  ceiling, the projection guard will abort after video 1. To fit under 150 GB use e.g.
  --positions 2 (400 videos ~ 136 GB) or --n 110 --positions 8 (~150 GB).

Usage (Aden launches; one process per GPU, see run_collect.sh):
  python collect_pairs.py --mode corpus                      # dry-run: selection stats, no GPU
  python collect_pairs.py --mode run --shard 0/4 [--n 400 --positions 8]
  python collect_pairs.py --mode run --shard 0/4 --switch-mode   # all rollouts A->B (v1.1)
  python collect_pairs.py --mode run --shard 0/4 --mix           # ~50/50 single/switch
  python collect_pairs.py --mode manifest                    # merge shard manifests

--------------------------------------------------------------------------------------------
shard format v3 "h_replay" (--v3, memory 2.0 J1/J2, reference/memory2_design.md 2.5-2.6):
per-video records for the offline write-replay training of the writer, states are not
stored anymore (the replay recomputes them); what is stored is the write input sequence:
  {
    "format_version": 3, "kind": "h_replay", "memory_layers": [5, 16, 27],
    "h_dtype": "float16",            # bf16 h round-tripped through fp16 (see below)
    "mem_zero_all": True,            # rollouts run MEM_ZERO_ALL=1: h = data (independent
                                     # of the writer params) = the replay-exactness premise,
                                     # and train == eval regime (memdec arms run memzero)
    "writer_ckpt": {...}, "cond_dim": 4096, "cond_pooling": <v2 masked mean>,
    "videos": [{
       "h_seq": {"5": [T(1,4680,1536) fp16, ...], "16": [...], "27": [...]},
                                     # h_seq[b][i] = the mem_x_write consumed by the write
                                     # pass of chunk write_chunks[i] (fifo peek, not recomputed)
       "write_chunks": [5, ...],     # replay order, exhaustive (all genuine writes)
       "evicted_chunks": [...],      # = write_chunks - window_chunks (bookkeeping)
       "latents": T(F,16,60,104) fp16,   # the whole rollout: every target is available
       "cond_emb_a"/"cond_emb_b", prompt/prompt_b/corpus_idx*/seed/mode/switch_*,
       "check_states": {"5": [(chunk, (w0,w1,w2) bf16), ...], ...} | None,
                                     # first --check-videos videos of shard 0 only:
                                     # trainer-side bitwise cross-check (selftest S5)
       "h_stats": {...},             # fp16 round-trip + subnormal counters (selftest S4)
    }, ...], "meta": {...}
  }
fp16 storage exactness: h lives in bf16; bf16 -> fp16 is exact in-range except |x| < ~2^-17
(fp16 subnormal precision floor). To make the stored bytes authoritative anyway, the fifo
entry is replaced by its fp16 round-trip before the write pass consumes it (a script-level
poke, zero core-code change): under MEM_ZERO_ALL=1 this provably cannot touch the latents
(the state feeds nothing but the zeroed injection), and it guarantees
in-situ state == replay(bf16(stored fp16)) bitwise whatever the tails do. The number of
elements actually changed by the poke is counted in h_stats (expected ~0-100 per tensor).
  python collect_pairs.py --v3 --mode corpus --corpus-seed 20260727
  python collect_pairs.py --v3 --mode run --shard 0/4 --corpus-seed 20260727
  python collect_pairs.py --v3 --mode manifest
Defaults under --v3: n=135 (110 switch@120f + 25 single@90f), switch-frames 45,60,75,
out .../mem2_h_v3, max-gb 200. Founder selftest: selftest_replay.py (S1 replay-bitwise).
--------------------------------------------------------------------------------------------

Additive code only: nothing outside reference/runs/frame_dec/ and the output dir is touched.
"""
import argparse
import gc
import glob
import json
import math
import os
import random
import re
import sys
import time

REPO = "/shared/ssd_14T/home/adenelmi/Anchor-Forcing"
PROBE_DIR = os.path.join(REPO, "reference/runs/reid_probe_l1")
sys.path.insert(0, REPO)
sys.path.insert(0, PROBE_DIR)
HERE = os.path.dirname(os.path.abspath(__file__))

import torch
from omegaconf import OmegaConf

# the reference loader of the l1@2000-EMA ckpt + the held-out probe pairs + the env reset
from run_reid_probe import build_pipeline, PAIRS, _OFF, release_vram  # noqa: E402

L1_YAML = os.path.join(REPO, "reference/runs/l1_longlive/l1_longlive.yaml")
DEFAULT_CONFIG = os.path.join(PROBE_DIR, "probe_l1.yaml")
DEFAULT_OUT = "/shared/ssd_14T_project/adenelmi/frame_dec_data"   # home is full, project disk


# ----------------------------- corpus selection ---------------------------------------------
# "re-identifiable subject" heuristic (localplan_C 2.1): a noun phrase carrying >= 1
# discriminating attribute (color / clothing / material), e.g. "a brown dog with a white
# chest", "a woman in a yellow raincoat", "a blue vintage bicycle".
_COLORS = ("red", "blue", "green", "yellow", "orange", "purple", "pink", "brown", "black",
           "white", "gray", "grey", "golden", "silver", "turquoise", "violet", "crimson",
           "scarlet", "beige", "teal", "maroon", "navy", "emerald", "ivory", "amber")
_MATERIALS = ("wooden", "leather", "denim", "brick", "steel", "metallic", "velvet", "silk",
              "copper", "bronze", "marble", "straw", "woolen", "plastic", "bamboo", "wicker",
              "rusty", "chrome", "porcelain")
_CLOTHES = ("jacket", "coat", "raincoat", "overcoat", "dress", "hat", "cap", "shirt",
            "t-shirt", "suit", "scarf", "uniform", "sweater", "hoodie", "cape", "gown",
            "kimono", "vest", "robe", "apron", "helmet", "boots", "gloves", "skirt", "jeans",
            "overalls", "sneakers", "bandana")
_ATTR = "|".join(_COLORS + _MATERIALS)
# scenery/lighting nouns right after the attribute -> not a re-identifiable subject
_NOUN_STOP = {"sky", "skies", "light", "lights", "glow", "sun", "sunset", "sunrise",
              "sunlight", "background", "backdrop", "air", "water", "ocean", "sea",
              "horizon", "hue", "hues", "tone", "tones", "color", "colors", "colour",
              "colours", "lighting", "atmosphere", "mist", "fog", "haze", "sand", "snow",
              "grass", "waves", "clouds", "cloud", "landscape", "scene", "scenery"}

# "a/an/the [adj ]{0,2} ATTR <noun>"  (noun captured for the stoplist check)
RE_NP_ATTR = re.compile(
    r"\b(?:a|an|the)\s+(?:[a-z][a-z-]*\s+){0,2}?(?:%s)\s+(?P<noun>[a-z][a-z-]{2,})" % _ATTR,
    re.IGNORECASE)
# attributed clothing, article optional: "wearing red jackets", "in yellow raincoats",
# "with brown leather boots" (safe without a noun stoplist: the tail requires a clothes noun)
RE_WEAR = re.compile(
    r"\b(?:wearing|dressed in|in|with)\s+(?:[a-z][a-z-]*\s+){0,2}?(?:%s)\s+(?:[a-z][a-z-]*\s+){0,1}?(?:%s)s?\b"
    % (_ATTR, "|".join(_CLOTHES)), re.IGNORECASE)


def is_reidentifiable(prompt: str) -> bool:
    for m in RE_NP_ATTR.finditer(prompt):
        if m.group("noun").lower() not in _NOUN_STOP:
            return True
    return bool(RE_WEAR.search(prompt))


def _norm_text(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip().lower())


def held_out_prompts():
    """The 16 prompts of the 8 (A, B) probe pairs = the level-1 held-out set."""
    return [_norm_text(p) for pair in PAIRS for p in pair]


def _hits_held_out(prompt_norm: str, held) -> bool:
    # substring match in both directions (a corpus line embedding a probe prompt, or the
    # probe prompt embedding a short corpus line, are both leaks)
    return any(h in prompt_norm or prompt_norm in h for h in held)


def find_l1_corpus_path() -> str:
    """The vidprom prompt file the l1 training consumed: data_path / switch_prompt_path in
    l1_longlive.yaml (they must agree, the training used one file for both)."""
    cfg = OmegaConf.load(L1_YAML)
    data_path = str(cfg.data_path)
    switch_path = str(cfg.switch_prompt_path)
    assert data_path == switch_path, \
        f"l1 corpus ambiguity: data_path={data_path} != switch_prompt_path={switch_path}"
    assert os.path.exists(data_path), f"l1 corpus file not found: {data_path}"
    return data_path


def select_prompts(n: int, corpus_seed: int, corpus_path: str = None, verbose: bool = True):
    """Deterministic selection of n re-identifiable prompts from the l1 vidprom corpus,
    with the 16 probe prompts excluded (substring, both directions).
    Returns a list of (corpus_idx, prompt), sorted by corpus_idx, identical on every
    shard/process for the same (n, corpus_seed): sharding partitions this list.
    Integrated selftest: asserts 0 held-out hits on the selected set."""
    corpus_path = corpus_path or find_l1_corpus_path()
    held = held_out_prompts()

    lines, seen = [], set()
    with open(corpus_path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            p = line.strip()
            if not p:
                continue
            key = _norm_text(p)
            if key in seen:          # dedupe exact repeats, keep first occurrence
                continue
            seen.add(key)
            lines.append((idx, p))

    filtered = [(i, p) for i, p in lines if is_reidentifiable(p)]
    clean = [(i, p) for i, p in filtered if not _hits_held_out(_norm_text(p), held)]
    n_excluded = len(filtered) - len(clean)

    if len(clean) < n:
        raise RuntimeError(f"corpus too small after filtering: {len(clean)} < n={n}")
    rng = random.Random(corpus_seed)
    selected = sorted(rng.sample(clean, n), key=lambda t: t[0])

    # ---- integrated held-out selftest: 0 hit required on the selected prompts ----
    hits = [(i, p) for i, p in selected if _hits_held_out(_norm_text(p), held)]
    assert not hits, f"HELD-OUT LEAK: {len(hits)} selected prompts hit the 16 probe prompts: " \
                     f"{[i for i, _ in hits][:5]}"
    if verbose:
        print(f"[corpus] {corpus_path}")
        print(f"[corpus] {len(lines)} unique lines -> {len(filtered)} re-identifiable "
              f"({100.0 * len(filtered) / max(1, len(lines)):.1f}%) -> {n_excluded} excluded "
              f"by held-out substring -> sampled {len(selected)} (seed {corpus_seed})")
        print(f"[corpus][selftest] held-out exclusion: 0 hit on {len(selected)} selected prompts")
    return selected


# ----------------------------- v1.1 switch-mode helpers -------------------------------------
DEFAULT_SWITCH_FRAMES = (60, 81, 102)   # chunk-aligned (F=3): switch chunks 20 / 27 / 34
COND_DIM = 4096                         # umt5-xxl channel dim

# ----------------------------- v3 (memory 2.0 h-replay) constants ---------------------------
V3_SWITCH_FRAMES = (45, 60, 75)         # 120f videos: switch chunks 15 / 20 / 25 (design 2.2)
V3_DEFAULT_OUT = "/shared/ssd_14T_project/adenelmi/mem2_h_v3"   # new dir, never mixed with v1
V3_FRAME_SEQLEN = 1560                  # tokens per latent frame (60/2 * 104/2)
V1_CORPUS_SEED = 1234                   # the v1 campaign seed, v3 must use a fresh one


def v3_assign_modes(corpus_indices, n_switch, mix_seed):
    """Deterministic exact-count switch/single split (design: 110 switch + 25 single).
    Rank every corpus_idx by a private hash-random keyed on (cidx, mix_seed), identical on
    every shard/process and resume-safe (same guarantees as choose_mode), and take the
    first n_switch as switch videos. Returns the set of switch corpus_idx."""
    ranked = sorted(corpus_indices,
                    key=lambda cidx: random.Random(1000003 * cidx + mix_seed).random())
    return set(ranked[:n_switch])


def v3_video_gb(frames, F, wc, sc, mem_layers, h_bytes=2):
    """A-priori size of one v3 video record (h sequences + full latents + conds), in GB.
    writes = chunks - (wc + sc): the first wc chunks pop nothing, the next sc pops are
    sink-pinned (never written)."""
    n_writes = max(0, frames // F - (wc + sc))
    h = n_writes * len(mem_layers) * (F * V3_FRAME_SEQLEN) * 1536 * h_bytes
    lat = frames * 16 * 60 * 104 * 2                       # fp16 latents, whole rollout
    return (h + lat + 2 * COND_DIM * 2) / 1e9


def v3_video_nbytes(rec):
    """Measured bytes of one v3 video record (h + latents + conds + optional check_states)."""
    n = rec["latents"].numel() * rec["latents"].element_size()
    for lst in rec["h_seq"].values():
        for t in lst:
            n += t.numel() * t.element_size()
    for k in ("cond_emb_a", "cond_emb_b"):
        if rec.get(k) is not None:
            n += rec[k].numel() * rec[k].element_size()
    if rec.get("check_states"):
        for lst in rec["check_states"].values():
            for _, st in lst:
                for t in st:
                    n += t.numel() * t.element_size()
    return n


def pooled_cond_emb(cond_dict):
    """v2 'cond_emb': masked-mean pooled T5 embedding of a prompt, fp16 cpu, shape (4096,).
    WanTextEncoder zeroes the padding positions: a plain mean over all 512 rows would scale
    with prompt length (parasitic feature flagged at v1.1 review), so we average over the
    non-ZERO rows only (sum / count of rows with nonzero norm). Any code computing cond at
    inference time (memdec) must use this same pooling."""
    emb = cond_dict["prompt_embeds"]
    assert emb.dim() == 3 and emb.shape[0] == 1, f"prompt_embeds shape {tuple(emb.shape)}"
    nz = emb[0].float().norm(dim=-1) > 0                      # (512,) non-padding rows
    assert nz.any(), "prompt_embeds entirely zero?"
    pooled = (emb[0][nz].mean(dim=0)).detach().to("cpu", torch.float16)
    assert pooled.shape == (COND_DIM,), f"pooled cond_emb shape {tuple(pooled.shape)}"
    assert torch.isfinite(pooled).all(), "non-finite pooled cond_emb"
    return pooled


# ----------------------------- custom prompt list (--prompts-file) --------------------------
# ADDITIF (2026-08-05, materiaux v2 du sondage) : une campagne dont les prompts ne viennent pas
# du corpus vidprom mais d'un FICHIER. Tout le reste de la machinerie v3 est reutilise tel quel
# (meme rollout, meme capture h, meme sharding, meme manifeste, meme --twin-offset) : ce bloc ne
# fait que remplacer `select_prompts` par une lecture de fichier, avec le MEME contrat de sortie
# [(corpus_idx, prompt)] trie par corpus_idx.
PROMPTS_FILE_BASE = 9_000_000   # corpus_idx = base + rang de ligne : au-dessus de tout index de
                                # ligne vidprom (le corpus fait ~1M lignes), donc AUCUNE collision
                                # possible avec un corpus_idx de la campagne v3.


def load_prompts_file(path, base=PROMPTS_FILE_BASE, check_corpus=True, verbose=True):
    """Lit une liste de prompts custom (une par ligne ; '#' et lignes vides ignorees) et rend
    EXACTEMENT la meme structure que select_prompts() : [(corpus_idx, prompt)] trie.

    corpus_idx = base + rang de la ligne (pas le numero de ligne du fichier) : l'ordre du
    fichier est la cle, ajouter un commentaire ne renumerote rien.

    Selftest de held-out integre, DEUX references et DEUX directions :
      (a) les 16 prompts des paires de sonde (exactement le test de select_prompts) ;
      (b) les lignes du corpus vidprom que l'entrainement l1 a consommees, un prompt custom
          qui reproduirait une ligne du corpus ne serait pas un materiau NEUF.
    """
    with open(path, "r", encoding="utf-8") as f:
        raw = [ln.strip() for ln in f]
    prompts = [p for p in raw if p and not p.startswith("#")]
    assert prompts, f"{path}: aucun prompt (lignes vides / commentaires seulement)"
    norm = [_norm_text(p) for p in prompts]
    dup = {p for p in norm if norm.count(p) > 1}
    assert not dup, f"{path}: prompts dupliques {sorted(dup)[:3]}"

    held = held_out_prompts()
    hits = [p for p in norm if _hits_held_out(p, held)]
    assert not hits, f"HELD-OUT LEAK (prompts de sonde): {hits[:3]}"

    n_corpus = 0
    if check_corpus:
        corpus_path = find_l1_corpus_path()
        clash = []
        with open(corpus_path, "r", encoding="utf-8") as f:
            for line in f:
                c = _norm_text(line)
                if not c:
                    continue
                n_corpus += 1
                for p in norm:
                    if p in c or c in p:
                        clash.append((p[:60], c[:60]))
        assert not clash, (f"HELD-OUT LEAK (corpus vidprom): {len(clash)} collision(s), "
                           f"p.ex. {clash[:2]}")

    selected = [(base + i, p) for i, p in enumerate(prompts)]
    if verbose:
        print(f"[prompts-file] {path}: {len(selected)} prompts, corpus_idx "
              f"{selected[0][0]}..{selected[-1][0]}")
        print(f"[prompts-file][selftest] held-out : 0 hit sur les {len(held)} prompts de sonde "
              f"et 0 collision sur les {n_corpus} lignes du corpus l1")
    return selected


def choose_mode(cidx, switch_mode, mix, mix_seed):
    """Per-video single/switch decision, keyed only on corpus_idx (+ mix_seed) so it is
    identical across shards/processes and resume-safe. --mix gives ~50/50 (not index parity:
    with shard stride 4, parity would send whole shards to one mode)."""
    if switch_mode:
        return "switch"
    if not mix:
        return "single"
    return "switch" if random.Random(1000003 * cidx + mix_seed).random() < 0.5 else "single"


def pick_switch(cidx, selected, switch_frames, switch_seed):
    """Deterministic (switch_frame, corpus_idx_b, prompt_b) for a switch rollout of prompt
    A = selected[cidx]: switch position random among switch_frames, prompt B = another line
    of the (already filtered + held-out-clean) selected corpus list."""
    assert len(selected) >= 2, "need >= 2 selected prompts to draw a distinct prompt B"
    prng = random.Random(switch_seed * 1000003 + cidx)
    switch_frame = int(prng.choice(list(switch_frames)))
    while True:
        cidx_b, prompt_b = selected[prng.randrange(len(selected))]
        if cidx_b != cidx:
            break
    return switch_frame, int(cidx_b), prompt_b


# ----------------------------- rollout + collection -----------------------------------------
def _find_causal_model(m):
    """Walk peft/wrapper layers down to the raw CausalWanModel (for _evict_dbg access)."""
    seen = 0
    while m is not None and seen < 8:
        # type check, not hasattr: peft wrappers delegate attribute reads (hasattr passes on the
        # wrapper) but setattr does not, resetting _evict_dbg on the wrapper shadows the real
        # model's list and the post-rollout read sees an empty list (bug caught by selftest).
        if type(m).__name__ == "CausalWanModel":
            return m
        m = getattr(m, "model", None) or getattr(m, "base_model", None)
        seen += 1
    return None


@torch.no_grad()
def rollout_collect(pipeline, config, prompt, corpus_idx, frames, n_positions, device,
                    pos_seed=777, forced_positions=None, debug_evict=False,
                    switch_frame=None, prompt_b=None, corpus_idx_b=None, min_gap_chunks=5,
                    v3_capture=False, v3_fp16_roundtrip=True, v3_state_snap_every=0,
                    seed_offset=0):
    """One rollout + pair collection after each write pass.

    v3_capture=False (default): byte-identical to the v1/v2 collector (the three v3_ kwargs
    add code only inside `if v3_capture:` branches).

    v3_capture=True (memory 2.0 h-replay, design 2.6): the whole rollout runs under
    MEM_ZERO_ALL=1 (h = data + train==eval regime); no pairs are built; instead, before the
    write pass of every chunk that genuinely writes, the fifo head (the tensor the model is
    about to pop as mem_x_write) is peeked and saved for each memory block. With
    v3_fp16_roundtrip=True (campaign mode) the fifo entry is first replaced by its
    bf16->fp16->bf16 round-trip so the in-situ write consumes exactly the bytes the shard
    stores (bitwise replay guarantee even for fp16-subnormal tails; a no-op when the
    round-trip is exact, counted in h_rt_changed); False keeps h in lossless bf16 (selftest
    S1 mode). v3_state_snap_every=N>0 additionally snapshots the post-write states of every
    N-th write (+ the last one) to cpu bf16, the trainer-side bitwise cross-check (S5).

    switch_frame=None (default): single-prompt rollout, RNG/byte-identical to the v1.0
    collector (the v2 additions, cond_emb pooling, consume no random numbers).

    switch_frame set (v1.1 recency-bias attack): one A->B switch at that (chunk-aligned)
    frame, exactly the single-switch branch of SwitchCausalInferencePipeline.inference
    (_recache_after_switch + cond swap + switch_global_frame). Collection happens only at
    post-switch chunks c >= switch_chunk + min_gap_chunks, where A is totally evicted from
    the 12-frame window (window covers chunks [c-3, c], all > switch_chunk; enforced by a
    fifo assert) and >= 2 pure-B chunks have been written into the state. The pair target is
    the last latent frame of A (switch_frame - 1, the anchor-frame candidate), the condition
    is prompt A's pooled T5 embedding = the exact task of the revisit, data-free.

    Returns a dict with pairs, latents, positions, and the list of blocks whose ttt state
    was written."""
    from utils.misc import set_seed

    # env hygiene: all probe knobs OFF (memory injection live = the state we snapshot is the
    # one the trained model actually uses)
    for k, v in _OFF.items():
        os.environ[k] = v
    # memory 2.0 regime change (design 1.2, the difference vs v1): h-replay collection runs
    # the whole rollout under MEM_ZERO_ALL=1, the injection is zeroed on every pass, so
    # (a) the h feeding the writes are pure functions of the frozen backbone = data,
    # independent of the writer params (the replay-exactness premise, design 1.1), and
    # (b) the latents match the harness memzero arm (train == eval alignment, selftest S2).
    if v3_capture:
        os.environ["MEM_ZERO_ALL"] = "1"
    os.environ["MEM_EVICT_DEBUG"] = "1" if debug_evict else "0"

    F = pipeline.num_frame_per_block
    assert frames % F == 0, f"frames={frames} not a multiple of num_frame_per_block={F}"
    num_blocks_t = frames // F
    law = int(config.model_kwargs.local_attn_size)
    assert law > 0, "eviction-feed collection requires local attention"
    wc = max(1, math.ceil(law / F))                                    # window_chunks (4)
    sc = math.ceil(int(config.model_kwargs.sink_size) / F)             # sink_chunks (1)
    mem_layers = sorted(int(x) for x in config.model_kwargs.memory_layers)

    switch_chunk = None
    if switch_frame is not None:
        assert prompt_b is not None and prompt_b != prompt, \
            "switch rollout needs a DIFFERENT prompt B"
        assert switch_frame % F == 0, \
            f"switch_frame {switch_frame} not chunk-aligned (F={F})"
        switch_chunk = switch_frame // F
        assert 0 < switch_chunk < num_blocks_t, \
            f"switch_frame {switch_frame} outside the {frames}-frame rollout"

    # eligible chunk positions: the popped chunk (c - wc) exists and is past the sink, i.e.
    # a chunk has genuinely been evicted from the 12-frame window and written to the memory
    eligible = [c for c in range(num_blocks_t) if c - wc >= sc]
    if switch_chunk is not None:
        # post-switch only, A totally evicted: the window of chunk c covers chunks
        # [c-wc+1, c]; c >= switch_chunk + min_gap_chunks (5 > wc-1 = 3) guarantees every
        # window chunk > switch_chunk and >= 2 pure-B chunks already written into the state
        assert min_gap_chunks >= wc + 1, \
            (f"min_gap_chunks {min_gap_chunks} < window_chunks+1 = {wc + 1}: gap == wc barely "
             f"evicts A; the spec (chunk >= switch+5) wants >= 2 pure-B WRITES in the state")
        eligible = [c for c in eligible if c >= switch_chunk + min_gap_chunks]
    assert eligible, (f"no eligible collection chunk: frames={frames}, "
                      f"switch_frame={switch_frame}")
    if v3_capture:
        # v3 stores the full write sequence + all latents; the per-pair "positions"
        # machinery is irrelevant (loss positions are sampled at training time, for free)
        positions = []
    elif forced_positions is not None:
        positions = sorted(forced_positions)
        assert all(c in eligible for c in positions), \
            f"forced positions {positions} not all eligible {eligible[:5]}..."
    else:
        # rank/shard-stable: depends only on (pos_seed, corpus_idx), not on the process
        prng = random.Random(pos_seed * 1000003 + corpus_idx)
        positions = sorted(prng.sample(eligible, min(n_positions, len(eligible))))
    pos_set = set(positions)

    # distinct seed stream per rollout kind: a corpus_idx collected in both a single and a
    # switch campaign must not share its noise (single keeps the exact v1.0 formula)
    seed = int(config.seed) + (500000 if switch_chunk is None else 700000) + corpus_idx \
        + seed_offset
    set_seed(seed)
    noise = torch.randn([1, frames, 16, 60, 104], device=device, dtype=torch.bfloat16)

    cond = pipeline.text_encoder(text_prompts=[prompt])
    # T5 forwards consume no torch RNG -> the single-prompt stream stays v1.0-identical
    cond_b = pipeline.text_encoder(text_prompts=[prompt_b]) if switch_chunk is not None else None
    # v2 condition = the prompt whose content the decoder must retrieve: the rollout prompt
    # in single mode, prompt A (the return prompt of a revisit) in switch mode
    cond_emb = pooled_cond_emb(cond)
    # v3 stores both pooled conds (the trainer picks per position); no torch RNG consumed
    cond_emb_b = pooled_cond_emb(cond_b) if (v3_capture and cond_b is not None) else None
    # low_memory text-encoder handling, verbatim from the pipeline's low_memory branch
    from utils.memory import gpu, get_cuda_free_memory_gb, \
        move_model_to_device_with_memory_preservation
    move_model_to_device_with_memory_preservation(
        pipeline.text_encoder, target_device=gpu,
        preserved_memory_gb=get_cuda_free_memory_gb(gpu) + 5)

    output = torch.zeros([1, frames, 16, 60, 104], device="cpu", dtype=noise.dtype)

    kv_cache_size = law * pipeline.frame_seq_length
    pipeline._initialize_kv_cache(1, dtype=noise.dtype, device=device,
                                  kv_cache_size_override=kv_cache_size)
    pipeline._initialize_crossattn_cache(batch_size=1, dtype=noise.dtype, device=device)
    from wan.modules.video_memory import TTTBlockCache
    pipeline.ttt_cache = [TTTBlockCache() for _ in range(pipeline.num_transformer_blocks)]

    pipeline.generator.model.local_attn_size = pipeline.local_attn_size
    pipeline._set_all_modules_max_attention_size(pipeline.local_attn_size)

    if debug_evict:
        _cm = _find_causal_model(pipeline.generator.model)
        if _cm is not None:
            _cm._evict_dbg = []

    pairs = []
    # ---- v3 h-replay accumulators (empty/unused unless v3_capture) ----
    v3_h_seq = {bi: [] for bi in mem_layers} if v3_capture else None
    v3_write_chunks = [] if v3_capture else None
    v3_check = ({str(bi): [] for bi in mem_layers}
                if (v3_capture and v3_state_snap_every > 0) else None)
    v3_rt_changed = 0        # elements actually altered by the fp16 round-trip poke
    v3_subnormal = 0         # |h| < fp16 min-normal (6.1e-5), nonzero (design 2.6 counter)
    v3_h_elems = 0
    v3_h_absmax = 0.0
    current_start_frame = 0
    cond_in_use = cond
    switch_global = None
    switched = False
    for chunk_idx in range(num_blocks_t):
        current_num_frames = F

        # v1.1 A->B switch: the single-switch branch of SwitchCausalInferencePipeline.inference
        # (recache the window under the new cond, then generate with prompt B)
        if switch_chunk is not None and not switched and chunk_idx >= switch_chunk:
            assert current_start_frame == switch_frame, \
                f"switch misaligned: chunk {chunk_idx} starts at {current_start_frame} != {switch_frame}"
            pipeline._recache_after_switch(output, current_start_frame, cond_b)
            cond_in_use = cond_b
            switch_global = current_start_frame
            switched = True

        noisy_input = noise[:, current_start_frame:current_start_frame + current_num_frames]

        # spatial denoising loop (copied from the switch pipeline)
        for index, current_timestep in enumerate(pipeline.denoising_step_list):
            timestep = torch.ones([1, current_num_frames], device=device,
                                  dtype=torch.int64) * current_timestep
            _, denoised_pred = pipeline.generator(
                noisy_image_or_video=noisy_input,
                conditional_dict=cond_in_use,
                timestep=timestep,
                kv_cache=pipeline.kv_cache1,
                crossattn_cache=pipeline.crossattn_cache,
                current_start=current_start_frame * pipeline.frame_seq_length,
                rope_enhance=pipeline.rope_enhance,
                ttt_cache=pipeline.ttt_cache,
                ttt_update=False,
                switch_global_frame=switch_global,
            )
            if index < len(pipeline.denoising_step_list) - 1:
                next_timestep = pipeline.denoising_step_list[index + 1]
                noisy_input = pipeline.scheduler.add_noise(
                    denoised_pred.flatten(0, 1),
                    torch.randn_like(denoised_pred.flatten(0, 1)),
                    next_timestep * torch.ones([1 * current_num_frames], device=device,
                                               dtype=torch.long),
                ).unflatten(0, denoised_pred.shape[:2])

        output[:, current_start_frame:current_start_frame + current_num_frames] = \
            denoised_pred.to(output.device)

        # ---- v3 h-capture: peek the fifo element the write pass below is about to pop ----
        # (the pop happens at model level, causal_model.py ~L1371, and the popped h is the
        # mem_x_write the cell writes: we save that tensor, nothing is recomputed). Genuine
        # write iff the popped chunk (chunk_idx - wc) exists and is past the sink.
        if v3_capture and chunk_idx >= wc:
            will_write = (chunk_idx - wc) >= sc
            for bi in mem_layers:
                fifo = pipeline.ttt_cache[bi].fifo
                assert len(fifo) == wc, \
                    (f"block {bi}: fifo len {len(fifo)} != window_chunks {wc} before the "
                     f"write pass of chunk {chunk_idx}")
                pchunk, h_ev = fifo[0]
                assert pchunk == chunk_idx - wc, \
                    f"block {bi}: fifo head chunk {pchunk} != {chunk_idx - wc} at chunk {chunk_idx}"
                if not will_write:
                    continue           # sink chunk: popped but pinned in KV, never written
                assert h_ev.dim() == 3 and h_ev.dtype == torch.bfloat16, \
                    f"block {bi}: unexpected fifo h {tuple(h_ev.shape)} {h_ev.dtype}"
                v3_h_elems += h_ev.numel()
                _amax = float(h_ev.detach().abs().max().item())
                v3_h_absmax = max(v3_h_absmax, _amax)
                assert _amax < 65504.0, \
                    f"block {bi} chunk {chunk_idx}: |h| max {_amax} out of fp16 range"
                v3_subnormal += int(((h_ev.abs() < 6.1e-5) & (h_ev != 0)).sum().item())
                if v3_fp16_roundtrip:
                    _h16 = h_ev.to(torch.float16)
                    assert torch.isfinite(_h16).all(), \
                        f"block {bi} chunk {chunk_idx}: non-finite fp16 h"
                    _hrt = _h16.to(torch.bfloat16)
                    _nd = int((_hrt != h_ev).sum().item())
                    if _nd:
                        v3_rt_changed += _nd
                        # poke: the write consumes exactly the bytes the shard stores.
                        # Under MEM_ZERO_ALL=1 this cannot touch the latents (the state
                        # feeds nothing but the zeroed injection), asserted by the
                        # latents-equality check in selftest_replay.py.
                        fifo[0] = (pchunk, _hrt)
                    v3_h_seq[bi].append(_h16.to("cpu"))
                else:
                    v3_h_seq[bi].append(h_ev.detach().to("cpu"))   # lossless bf16 (S1 mode)
            if will_write:
                v3_write_chunks.append(chunk_idx)

        # write pass (clean context noise; eviction-feed pops chunk_idx - wc and writes it)
        context_timestep = torch.ones_like(timestep) * pipeline.args.context_noise
        pipeline.generator(
            noisy_image_or_video=denoised_pred,
            conditional_dict=cond_in_use,
            timestep=context_timestep,
            kv_cache=pipeline.kv_cache1,
            crossattn_cache=pipeline.crossattn_cache,
            current_start=current_start_frame * pipeline.frame_seq_length,
            rope_enhance=pipeline.rope_enhance,
            ttt_cache=pipeline.ttt_cache,
            ttt_update=True,
            switch_global_frame=switch_global,
        )

        # ---- v3 state snapshot: right after the write pass of every Nth write (+ the last
        # one), for the trainer-side bitwise cross-check (selftest S5). bf16 GPU -> bf16 CPU
        # is bit-identical.
        if (v3_check is not None and v3_write_chunks and
                v3_write_chunks[-1] == chunk_idx):
            _widx = len(v3_write_chunks) - 1
            if _widx % v3_state_snap_every == 0 or chunk_idx == num_blocks_t - 1:
                for bi in mem_layers:
                    _st = pipeline.ttt_cache[bi].state
                    assert _st is not None, \
                        f"block {bi}: no ttt state after write pass of chunk {chunk_idx}"
                    # dtype-preserving snapshot (P0, 2026-08-10): the historical SwiGLU
                    # states are bf16, so `.to("cpu", t.dtype)` is bitwise the previous
                    # `.to("cpu", torch.bfloat16)` for them (identity change); it only
                    # matters for the fp32 linear-delta W (state_type=linear_delta), whose
                    # equivalence selftest needs lossless per-write snapshots.
                    v3_check[str(bi)].append(
                        (int(chunk_idx),
                         tuple(t.detach().to("cpu", t.dtype) for t in _st)))

        # ---- collection: right after the write pass, state has just absorbed chunk_idx-wc ----
        if chunk_idx in pos_set:
            evicted_chunk = chunk_idx - wc
            evicted_frame = evicted_chunk * F + (F - 1)   # most recent frame that left the window
            for bi in mem_layers:
                cache = pipeline.ttt_cache[bi]
                assert cache.state is not None, \
                    f"block {bi}: ttt state is None after write pass of chunk {chunk_idx}"
                fifo_idx = [c for c, _ in cache.fifo]
                assert fifo_idx == list(range(chunk_idx - wc + 1, chunk_idx + 1)), \
                    (f"block {bi} fifo out of sync at chunk {chunk_idx}: {fifo_idx} "
                     f"(expected the last {wc} chunks -> popped chunk must be {evicted_chunk})")
                if switch_chunk is not None:
                    # A totally evicted from the window: every chunk still in the fifo is B's
                    assert min(fifo_idx) > switch_chunk, \
                        (f"block {bi}: A NOT fully evicted at chunk {chunk_idx}: window chunks "
                         f"{fifo_idx} reach into A (switch_chunk {switch_chunk})")
            # target: single = the just-evicted frame; switch = the last frame of A (anchor)
            target_frame = evicted_frame if switch_chunk is None else switch_frame - 1
            pair = {
                "states": [tuple(t.detach().to("cpu", torch.float16)
                                 for t in pipeline.ttt_cache[bi].state) for bi in mem_layers],
                "target": output[0, target_frame].detach().to("cpu", torch.float16).clone(),
                "cond_emb": cond_emb.clone(),           # v2: pooled T5 of the retrieval prompt
                "mode": "single" if switch_chunk is None else "switch",
                "target_frame": int(target_frame),
                "prompt": prompt,
                "chunk_idx": int(chunk_idx),
                "evicted_chunk": int(evicted_chunk),
                "evicted_frame": int(evicted_frame),
                "corpus_idx": int(corpus_idx),
                "seed": int(seed),
                "num_frame_per_block": int(F),
                "window_chunks": int(wc),
                "sink_chunks": int(sc),
            }
            if switch_chunk is not None:
                pair.update({
                    "prompt_b": prompt_b,
                    "corpus_idx_b": int(corpus_idx_b) if corpus_idx_b is not None else -1,
                    "switch_frame": int(switch_frame),
                    "switch_chunk": int(switch_chunk),
                })
            pairs.append(pair)

        current_start_frame += current_num_frames

    written = [bi for bi in range(pipeline.num_transformer_blocks)
               if pipeline.ttt_cache[bi].state is not None]

    evict_dbg = None
    if debug_evict:
        _cm = _find_causal_model(pipeline.generator.model)
        evict_dbg = list(getattr(_cm, "_evict_dbg", []) or []) if _cm is not None else None

    latents = output.clone()   # cpu bf16 [1, frames, 16, 60, 104]

    # per-video cleanup (mirrors run_arm; no VAE was used)
    pipeline.kv_cache1 = None
    pipeline.crossattn_cache = None
    pipeline.ttt_cache = None
    gc.collect()
    torch.cuda.empty_cache()

    result = {"pairs": pairs, "latents": latents, "positions": positions,
              "eligible": eligible, "written": written, "seed": seed,
              "window_chunks": wc, "sink_chunks": sc, "evict_dbg": evict_dbg,
              "mode": "single" if switch_chunk is None else "switch",
              "switch_frame": switch_frame, "switch_chunk": switch_chunk}
    if v3_capture:
        result.update({
            "h_seq": v3_h_seq, "write_chunks": v3_write_chunks, "check_states": v3_check,
            "cond_emb_a": cond_emb.clone(), "cond_emb_b": cond_emb_b,
            "h_dtype": "float16" if v3_fp16_roundtrip else "bfloat16",
            "h_rt_changed": int(v3_rt_changed), "h_subnormals": int(v3_subnormal),
            "h_elems": int(v3_h_elems), "h_absmax": float(v3_h_absmax)})
    return result


# ----------------------------- sharding / persistence ---------------------------------------
def pair_nbytes(pair):
    n = pair["target"].numel() * pair["target"].element_size()
    ce = pair.get("cond_emb")
    if ce is not None:                 # v2: +8 KB/pair, negligible vs the 170 MB of states
        n += ce.numel() * ce.element_size()
    for st in pair["states"]:
        for t in st:
            n += t.numel() * t.element_size()
    return n


def shard_header(config, config_path, mem_layers, frames, positions_per_video,
                 collect_mode="single"):
    return {
        "format_version": 2,   # v2 = v1 + per-pair cond_emb/mode/target_frame (+ switch fields)
        "memory_layers": mem_layers,
        "state_keys": ["w0", "w1", "w2"],
        "state_shapes": {"w0": [1, 1536, 6144], "w1": [1, 6144, 1536],
                         "w2": [1, 1536, 6144]},
        "target_shape": [16, 60, 104],
        "cond_dim": COND_DIM,
        "cond_key": "cond_emb",
        "cond_pooling": "MASKED mean over non-zero (non-padding) token rows "
                        "(padding rows are zero), fp16 cpu",
        "collect_mode": collect_mode,
        "meta": {
            "config": str(config_path),
            "generator_ckpt": str(config.generator_ckpt),
            "lora_ckpt": str(config.lora_ckpt),
            "use_ema": bool(getattr(config, "use_ema", False)),
            "frames": int(frames),
            "positions_per_video": int(positions_per_video),
            "collect_mode": collect_mode,
            "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        },
    }


def flush_part(out_dir, shard_tag, part_idx, buffer, header, key="pairs", prefix="pairs"):
    fname = f"{prefix}_{shard_tag}_part{part_idx:03d}.pt"
    path = os.path.join(out_dir, fname)
    obj = dict(header)
    obj[key] = buffer
    tmp = path + ".tmp"
    torch.save(obj, tmp)
    os.replace(tmp, path)
    nbytes = os.path.getsize(path)
    print(f"[collect] flushed {fname}: {len(buffer)} {key}, {nbytes / 1e9:.2f} GB", flush=True)
    return {"file": fname, "n_pairs": len(buffer), "bytes": nbytes}


def shard_header_v3(config, config_path, mem_layers, args, switch_frames, h_dtype):
    """v3 'h_replay' shard header (design 2.5). No per-pair states anymore: the replay
    recomputes them from h_seq; mem_zero_all must be asserted True by the trainer."""
    return {
        "format_version": 3,
        "kind": "h_replay",
        "memory_layers": mem_layers,
        "h_dtype": h_dtype,                    # "float16" (campaign) | "bfloat16" (--h-bf16)
        "h_shape_per_write": [1, 3 * V3_FRAME_SEQLEN, 1536],
        "mem_zero_all": True,                  # rollouts ran MEM_ZERO_ALL=1 (h = data)
        "writer_ckpt": {"lora_ckpt": str(config.lora_ckpt),
                        "use_ema": bool(getattr(config, "use_ema", False))},
        "cond_dim": COND_DIM,
        "cond_key": "cond_emb_a/cond_emb_b",
        "cond_pooling": "MASKED mean over non-zero (non-padding) token rows "
                        "(padding rows are zero), fp16 cpu",
        "latents_dtype": "float16",
        "collect_mode": "v3",
        "meta": {
            "config": str(config_path),
            "generator_ckpt": str(config.generator_ckpt),
            "lora_ckpt": str(config.lora_ckpt),
            "use_ema": bool(getattr(config, "use_ema", False)),
            "frames_switch": int(args.frames_switch),
            "frames_single": int(args.frames_single),
            "n": int(args.n), "n_switch": int(args.n_switch),
            "switch_frames": list(switch_frames),
            "corpus_seed": int(args.corpus_seed),
            "check_videos": int(args.check_videos), "check_every": int(args.check_every),
            "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        },
    }


def write_manifest(out_dir, shard_tag, manifest):
    path = os.path.join(out_dir, f"manifest_{shard_tag}.json")
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(manifest, f, indent=2)
    os.replace(tmp, path)


def merge_manifests(out_dir):
    files = sorted(glob.glob(os.path.join(out_dir, "manifest_shard*.json")))
    if not files:
        print(f"[manifest] no shard manifests in {out_dir}")
        return
    total = {"shards": [], "n_pairs": 0, "n_videos": 0, "bytes_total": 0, "prompts": {},
             "video_modes": {}}
    for fp in files:
        with open(fp) as f:
            m = json.load(f)
        total["shards"].append(os.path.basename(fp))
        total["n_pairs"] += m["n_pairs"]
        total["n_videos"] += m["n_videos_done"]
        total["bytes_total"] += m["bytes_written"]
        total["prompts"].update(m["prompts"])
        total["video_modes"].update(m.get("video_modes", {}))
    total["gb_total"] = total["bytes_total"] / 1e9
    out = os.path.join(out_dir, "manifest.json")
    with open(out, "w") as f:
        json.dump(total, f, indent=2)
    print(f"[manifest] merged {len(files)} shard manifests -> {out}: "
          f"{total['n_pairs']} pairs, {total['n_videos']} videos, {total['gb_total']:.1f} GB")


# ----------------------------- main ----------------------------------------------------------
def parse_shard(s):
    m = re.fullmatch(r"(\d+)/(\d+)", s)
    assert m, f"--shard must look like i/n, got {s!r}"
    i, n = int(m.group(1)), int(m.group(2))
    assert 0 <= i < n, f"--shard {s}: need 0 <= i < n"
    return i, n


def main_v3(args):
    """Memory 2.0 h-replay campaign (--v3, format v3). Reuses the v1 machinery wholesale:
    same corpus selection (held-out grep selftest inside select_prompts), same seed formulas
    (500000/700000 + corpus_idx), same pick_switch, same shard/manifest/resume pattern.
    Differences: per-video records (h sequences + full latents), MEM_ZERO_ALL=1 rollouts
    (set inside rollout_collect(v3_capture=True)), exact-count switch/single mix, per-mode
    video lengths."""
    # v3-tuned defaults when the v1 defaults were left untouched
    if args.prompts_file:
        # campagne a prompts custom : tout single, corpus = le fichier, repertoire dedie.
        if args.out == DEFAULT_OUT:
            stem = os.path.splitext(os.path.basename(args.prompts_file))[0]
            args.out = f"/shared/ssd_14T_project/adenelmi/{stem}"
            print(f"[v3] --out -> {args.out} (campagne --prompts-file, jamais melangee aux "
                  f"shards v3)")
        if args.n == 400:
            args.n = 0            # fixe par le fichier, pas par --n
    if args.out == DEFAULT_OUT:
        args.out = V3_DEFAULT_OUT
        print(f"[v3] --out -> {args.out} (v3 default, never mixed with v1 pair shards)")
    if args.twin_offset:
        # twin campaign (red-team F1.1): same corpus selection + same single prompts, only
        # the generation seed moves -> same-prompt/different-seed hard negatives for the
        # InfoNCE bank. Twins must share the train/val side of their sibling (bank builder's
        # job, keyed on corpus_idx). Switch videos are skipped; check-states are pointless.
        assert args.twin_offset > 0, "--twin-offset must be a positive seed shift"
        args.check_videos = 0
        if not args.out.rstrip("/").endswith("_twins"):
            args.out = args.out.rstrip("/") + "_twins"
            print(f"[v3] --out -> {args.out} (twin campaign kept apart from the main shards)")
    if args.max_gb == 150.0:
        args.max_gb = 200.0
        print("[v3] --max-gb -> 200 (design ceiling)")
    if args.n == 400 and not args.prompts_file:
        args.n = 135
        print(f"[v3] --n -> 135 ({args.n_switch} switch @{args.frames_switch}f + "
              f"{135 - args.n_switch} single @{args.frames_single}f)")
    if args.switch_frames == "60,81,102":
        switch_frames = V3_SWITCH_FRAMES
        print(f"[v3] --switch-frames -> {switch_frames} (chunk-aligned for "
              f"{args.frames_switch}f videos)")
    else:
        switch_frames = tuple(int(x) for x in args.switch_frames.split(",") if x != "")
    assert args.prompts_file or 0 < args.n_switch <= args.n, \
        f"--n-switch {args.n_switch} not in (0, {args.n}]"
    h_dtype = "bfloat16" if args.h_bf16 else "float16"

    if args.mode == "manifest":
        merge_manifests(args.out)
        return

    if args.corpus_seed == V1_CORPUS_SEED and not args.allow_old_corpus_seed \
            and not args.prompts_file:
        sys.exit("[v3] --corpus-seed is still the v1 seed 1234; the 2.0 campaign wants a "
                 "FRESH corpus (design decision 6.4), e.g. --corpus-seed 20260727. "
                 "Pass --allow-old-corpus-seed to reuse 1234 deliberately.")

    config = OmegaConf.load(args.config)
    config.distributed = False
    mem_layers = sorted(int(x) for x in config.model_kwargs.memory_layers)
    assert mem_layers, "config has no memory_layers, nothing to collect"
    F = int(config.num_frame_per_block)
    law = int(config.model_kwargs.local_attn_size)
    wc = max(1, math.ceil(law / F))
    sc = math.ceil(int(config.model_kwargs.sink_size) / F)
    for sf in switch_frames:
        assert sf % F == 0 and 0 < sf < args.frames_switch, \
            f"switch frame {sf}: not chunk-aligned or outside the {args.frames_switch}f video"
    assert args.frames_switch % F == 0 and args.frames_single % F == 0

    if args.prompts_file:
        # corpus = le FICHIER. Aucune video switch : une paire meme-prompt n'a de sens que sur
        # un rollout single (le jumeau doit partager TOUTE la trajectoire de conditionnement).
        selected = load_prompts_file(args.prompts_file, base=args.prompts_base)
        args.n, args.n_switch = len(selected), 0
        switch_set = set()
    else:
        selected = select_prompts(args.n, args.corpus_seed)
        switch_set = v3_assign_modes([i for i, _ in selected], args.n_switch, args.mix_seed)

    # a-priori disk projection: pure arithmetic, checked before any GPU work (guard n°1)
    gb_sw = v3_video_gb(args.frames_switch, F, wc, sc, mem_layers)
    gb_si = v3_video_gb(args.frames_single, F, wc, sc, mem_layers)
    n_sw, n_si = args.n_switch, args.n - args.n_switch
    if args.twin_offset:
        n_sw = 0                         # twins re-run the single prompts only
    proj_gb = n_sw * gb_sw + n_si * gb_si   # fp16 and bf16 h are both 2 bytes: same size
    n_writes_sw = args.frames_switch // F - (wc + sc)
    n_snaps = max(1, math.ceil(n_writes_sw / max(1, args.check_every))) + 1  # +1: last write
    check_gb = (args.check_videos * n_snaps * len(mem_layers)
                * (2 * 1536 * 6144 + 6144 * 1536) * 2) / 1e9   # bf16 states, shard 0 only
    print(f"[v3] projection: {n_sw} switch x {gb_sw:.2f} GB + {n_si} single x {gb_si:.2f} GB "
          f"= {proj_gb:.1f} GB (+ ~{check_gb:.1f} GB check-states on shard 0) "
          f"(ceiling {args.max_gb:.0f} GB)")

    if args.mode == "corpus":
        print("[v3][corpus] first 5 selected prompts:")
        for i, p in selected[:5]:
            tag = "switch" if i in switch_set else "single"
            print(f"    [{i}][{tag}] {p[:110]}{'...' if len(p) > 110 else ''}")
        print(f"[v3][corpus] modes: {len(switch_set)} switch / "
              f"{len(selected) - len(switch_set)} single (exact counts), "
              f"switch_frames={switch_frames}, h_dtype={h_dtype}")
        return

    if proj_gb + check_gb > args.max_gb:
        sys.exit(f"[v3] ABORT: a-priori projection {proj_gb + check_gb:.1f} GB > ceiling "
                 f"{args.max_gb:.0f} GB. Reduce --n / --n-switch or raise --max-gb.")

    shard_i, shard_n = parse_shard(args.shard)
    mine = selected[shard_i::shard_n]
    shard_tag = f"shard{shard_i}of{shard_n}"
    os.makedirs(args.out, exist_ok=True)

    manifest_path = os.path.join(args.out, f"manifest_{shard_tag}.json")
    parts, completed, bytes_written, n_writes_flushed = [], [], 0, 0
    modes_done = {}
    rt_total, sub_total = 0, 0
    if os.path.exists(manifest_path):
        if not args.resume:
            sys.exit(f"[v3] {manifest_path} exists. Pass --resume to continue this shard, "
                     f"or delete its parts+manifest to start fresh.")
        with open(manifest_path) as f:
            prev = json.load(f)
        assert prev.get("collect_mode") == "v3", \
            f"resume mode mismatch: manifest says {prev.get('collect_mode')!r}, CLI says 'v3'"
        parts = prev["parts"]
        completed = prev["completed_corpus_idx"]
        bytes_written = prev["bytes_written"]
        n_writes_flushed = prev["n_pairs"]
        modes_done = prev.get("video_modes", {})
        rt_total = prev.get("h_rt_changed_total", 0)
        sub_total = prev.get("h_subnormals_total", 0)
        print(f"[v3] resuming {shard_tag}: {len(completed)} videos already flushed")

    device = torch.device("cuda")
    torch.set_grad_enabled(False)
    pipeline = build_pipeline(config, device)
    assert int(pipeline.num_frame_per_block) == F
    header = shard_header_v3(config, args.config, mem_layers, args, switch_frames, h_dtype)

    def current_manifest(per_video_bytes=None, projected_gb=None):
        return {
            "shard": args.shard, "created": time.strftime("%Y-%m-%d %H:%M:%S"),
            "config": args.config, "lora_ckpt": str(config.lora_ckpt),
            "use_ema": bool(getattr(config, "use_ema", False)),
            "collect_mode": "v3", "format_version": 3, "h_dtype": h_dtype,
            "frames_switch": args.frames_switch, "frames_single": args.frames_single,
            "n_total_prompts_campaign": len(selected), "n_prompts_shard": len(mine),
            "n_videos_done": len(completed), "n_pairs": n_writes_flushed,
            "bytes_written": bytes_written, "parts": parts,
            "completed_corpus_idx": completed,
            "prompts": {str(i): p for i, p in mine if i in set(completed)},
            "per_video_bytes": per_video_bytes, "projected_campaign_gb": projected_gb,
            "corpus_seed": args.corpus_seed, "pos_seed": args.pos_seed,
            "video_modes": modes_done, "switch_seed": args.switch_seed,
            "switch_frames": list(switch_frames), "mix_seed": args.mix_seed,
            "min_gap_chunks": args.min_gap_chunks, "n_switch": args.n_switch,
            "videos_per_part": args.videos_per_part,
            "check_videos": args.check_videos, "check_every": args.check_every,
            "h_rt_changed_total": rt_total, "h_subnormals_total": sub_total,
            "twin_offset": args.twin_offset,
            "prompts_base": int(args.prompts_base), "no_h": bool(args.no_h),
        }

    buffer = []
    per_video_bytes = None
    first_collected = not completed      # evict-debug cross-check on the 1st video collected
    done_set = set(completed)
    t0 = time.time()
    for vnum, (cidx, prompt) in enumerate(mine):
        if cidx in done_set:
            continue
        t_vid = time.time()
        vmode = "switch" if cidx in switch_set else "single"
        if args.twin_offset and vmode == "switch":
            continue
        frames = args.frames_switch if vmode == "switch" else args.frames_single
        sw_frame, cidx_b, prompt_b = (None, None, None)
        if vmode == "switch":
            sw_frame, cidx_b, prompt_b = pick_switch(cidx, selected, switch_frames,
                                                     args.switch_seed)
        want_check = (shard_i == 0 and vnum < args.check_videos)
        print(f"[v3] {shard_tag} video {vnum + 1}/{len(mine)} corpus_idx={cidx} mode={vmode} "
              f"frames={frames}"
              + (f" switch_frame={sw_frame} corpus_idx_b={cidx_b}" if sw_frame else "")
              + (" +check_states" if want_check else "") + "\n"
              f"     prompt: {prompt[:110]}{'...' if len(prompt) > 110 else ''}", flush=True)
        if prompt_b is not None:
            print(f"     prompt_b: {prompt_b[:108]}{'...' if len(prompt_b) > 108 else ''}",
                  flush=True)

        res = rollout_collect(pipeline, config, prompt, cidx, frames, 0, device,
                              pos_seed=args.pos_seed, debug_evict=first_collected,
                              switch_frame=sw_frame, prompt_b=prompt_b, corpus_idx_b=cidx_b,
                              min_gap_chunks=args.min_gap_chunks,
                              v3_capture=True, v3_fp16_roundtrip=not args.h_bf16,
                              v3_state_snap_every=(args.check_every if want_check else 0),
                              seed_offset=args.twin_offset)

        assert res["written"] == mem_layers, \
            f"ttt state written on {res['written']}, expected {mem_layers}"
        expected_writes = [c for c in range(frames // F) if c >= wc + sc]
        assert res["write_chunks"] == expected_writes, \
            (f"write_chunks {res['write_chunks'][:6]}... != expected "
             f"{expected_writes[:6]}... (wc={wc}, sc={sc})")
        for bi in mem_layers:
            assert len(res["h_seq"][bi]) == len(expected_writes), \
                f"block {bi}: {len(res['h_seq'][bi])} h != {len(expected_writes)} writes"
        if first_collected:
            # cross-check our fifo-peek capture against the model's own eviction log
            # (MEM_EVICT_DEBUG=1 on the first video, design 2.6)
            assert res["evict_dbg"], "MEM_EVICT_DEBUG produced no entries on video 1"
            for bi in mem_layers:
                wrote = [d["chunk"] for d in res["evict_dbg"]
                         if d["block"] == bi and d["wrote"]]
                assert wrote == expected_writes, \
                    f"block {bi}: evict-debug wrote-chunks {wrote[:6]}... != capture"
            print(f"[v3] evict-debug cross-check OK on video 1 "
                  f"({len(expected_writes)} writes x {len(mem_layers)} blocks)", flush=True)
            first_collected = False

        rec = {
            # --no-h : les h sont captures (les asserts de parite ci-dessus ont tourne) mais
            # pas stockes ; rec garde tout ce que la labellisation et la re-collecte seedee
            # exigent (latents, conds, graine, write_chunks).
            "h_seq": ({} if args.no_h else {str(bi): res["h_seq"][bi] for bi in mem_layers}),
            "no_h": bool(args.no_h),
            "write_chunks": [int(c) for c in res["write_chunks"]],
            "evicted_chunks": [int(c) - wc for c in res["write_chunks"]],
            "latents": res["latents"][0].to(torch.float16),
            "cond_emb_a": res["cond_emb_a"], "cond_emb_b": res["cond_emb_b"],
            "prompt": prompt, "prompt_b": prompt_b,
            "corpus_idx": int(cidx),
            "corpus_idx_b": int(cidx_b) if cidx_b is not None else -1,
            "seed": int(res["seed"]), "mode": vmode, "frames": int(frames),
            "switch_frame": int(sw_frame) if sw_frame is not None else None,
            "switch_chunk": int(res["switch_chunk"]) if res["switch_chunk"] is not None else None,
            "window_chunks": int(wc), "sink_chunks": int(sc),
            "num_frame_per_block": int(F),
            "check_states": res["check_states"],
            "h_stats": {"rt_changed": res["h_rt_changed"],
                        "subnormals": res["h_subnormals"],
                        "elems": res["h_elems"], "absmax": res["h_absmax"]},
        }
        rt_total += res["h_rt_changed"]
        sub_total += res["h_subnormals"]
        buffer.append(rec)
        completed.append(cidx)
        modes_done[str(cidx)] = vmode
        print(f"[v3] corpus_idx={cidx}: {len(expected_writes)} writes x {len(mem_layers)} "
              f"blocks captured ({res['h_dtype']}), rt_changed={res['h_rt_changed']} "
              f"subnormals={res['h_subnormals']} absmax={res['h_absmax']:.1f} "
              f"in {time.time() - t_vid:.0f}s", flush=True)

        if per_video_bytes is None:
            # guard n°2: measured size of the first video vs the analytic projection
            per_video_bytes = v3_video_nbytes(rec)
            base_bytes = v3_video_nbytes({**rec, "check_states": None})
            analytic = (gb_sw if vmode == "switch" else gb_si) * 1e9
            ratio = base_bytes / analytic
            proj_measured = (proj_gb * ratio) + check_gb
            print(f"[v3] ===== DISK PROJECTION (after first video) =====\n"
                  f"     measured {base_bytes / 1e9:.2f} GB vs analytic {analytic / 1e9:.2f} "
                  f"GB (x{ratio:.3f}) -> projected campaign {proj_measured:.1f} GB "
                  f"(ceiling {args.max_gb:.0f} GB)", flush=True)
            if proj_measured > args.max_gb:
                part = flush_part(args.out, shard_tag, len(parts), buffer, header,
                                  key="videos", prefix="hreplay")
                parts.append(part)
                bytes_written += part["bytes"]
                n_writes_flushed += sum(len(r["write_chunks"]) for r in buffer)
                write_manifest(args.out, shard_tag,
                               current_manifest(per_video_bytes, proj_measured))
                sys.exit(f"[v3] ABORT: measured projection {proj_measured:.1f} GB > "
                         f"{args.max_gb:.0f} GB ceiling. First video kept for inspection.")

        if len(buffer) >= args.videos_per_part:
            part = flush_part(args.out, shard_tag, len(parts), buffer, header,
                              key="videos", prefix="hreplay")
            parts.append(part)
            bytes_written += part["bytes"]
            n_writes_flushed += sum(len(r["write_chunks"]) for r in buffer)
            buffer = []
            write_manifest(args.out, shard_tag, current_manifest(per_video_bytes))

    if buffer:
        part = flush_part(args.out, shard_tag, len(parts), buffer, header,
                          key="videos", prefix="hreplay")
        parts.append(part)
        bytes_written += part["bytes"]
        n_writes_flushed += sum(len(r["write_chunks"]) for r in buffer)
    write_manifest(args.out, shard_tag, current_manifest(per_video_bytes))
    print(f"[v3] {shard_tag} DONE: {len(completed)} videos, {n_writes_flushed} writes, "
          f"{bytes_written / 1e9:.1f} GB in {(time.time() - t0) / 3600:.1f} h "
          f"(rt_changed={rt_total}, subnormals={sub_total})", flush=True)
    del pipeline
    release_vram()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["run", "corpus", "manifest"], default="run")
    ap.add_argument("--n", type=int, default=400, help="total prompts across ALL shards")
    ap.add_argument("--positions", type=int, default=8, help="chunk positions kept per video")
    ap.add_argument("--shard", type=str, default="0/1", help="i/n partition of the prompts")
    ap.add_argument("--frames", type=int, default=240)
    ap.add_argument("--config", type=str, default=DEFAULT_CONFIG)
    ap.add_argument("--out", type=str, default=DEFAULT_OUT)
    ap.add_argument("--shard-size", type=int, default=200, help="pairs per .pt part "
                    "(200 x ~170MB ~ 34 GB buffered in RAM before each flush)")
    ap.add_argument("--max-gb", type=float, default=150.0, help="abort if the projected "
                    "TOTAL campaign size exceeds this")
    ap.add_argument("--corpus-seed", type=int, default=1234, help="prompt sampling seed "
                    "(must be IDENTICAL across shards)")
    ap.add_argument("--pos-seed", type=int, default=777, help="chunk-position sampling seed")
    ap.add_argument("--resume", action="store_true", help="continue a shard from its manifest")
    # ---- v1.1 switch-mode (recency-bias attack) ----
    ap.add_argument("--switch-mode", action="store_true", dest="switch_mode",
                    help="ALL rollouts are A->B switch rollouts: target = last frame of A, "
                         "cond = prompt A, states = post-B-writes")
    ap.add_argument("--mix", action="store_true",
                    help="~50/50 single/switch per video (deterministic on corpus_idx)")
    ap.add_argument("--switch-frames", type=str, default="60,81,102",
                    help="comma list of chunk-aligned switch positions (frames)")
    ap.add_argument("--switch-seed", type=int, default=4242,
                    help="switch position + prompt-B sampling seed (IDENTICAL across shards)")
    ap.add_argument("--mix-seed", type=int, default=31337,
                    help="--mix single/switch assignment seed (IDENTICAL across shards)")
    ap.add_argument("--min-gap-chunks", type=int, default=5,
                    help="collect only at chunk_idx >= switch_chunk + this (A fully evicted)")
    # ---- v3 (memory 2.0 h-replay collection, design 2.5/2.6) ----
    ap.add_argument("--v3", action="store_true",
                    help="h-replay collection: per-video h sequences under MEM_ZERO_ALL=1 "
                         "(format v3); see the v3 block of the module docstring")
    ap.add_argument("--n-switch", type=int, default=110,
                    help="v3: EXACT number of switch videos among --n (rest = single)")
    ap.add_argument("--frames-switch", type=int, default=120, help="v3 switch-video length")
    ap.add_argument("--frames-single", type=int, default=90, help="v3 single-video length")
    ap.add_argument("--videos-per-part", type=int, default=8,
                    help="v3: videos per .pt part (~12 GB fp16)")
    ap.add_argument("--check-videos", type=int, default=4,
                    help="v3: first N videos of shard 0 also store post-write state "
                         "snapshots (trainer bitwise cross-check, selftest S5)")
    ap.add_argument("--check-every", type=int, default=12,
                    help="v3: snapshot every Nth write (+ the last) on check videos "
                         "(12 keeps the default 135-video campaign under the 200 GB ceiling)")
    ap.add_argument("--h-bf16", action="store_true",
                    help="v3 debug: store h in bf16 WITHOUT the fp16 round-trip poke")
    ap.add_argument("--twin-offset", type=int, default=0,
                    help="v3 twins: re-run the SINGLE prompts of this corpus selection with "
                         "generation seed shifted by this amount (same-prompt/different-seed "
                         "negatives, red-team F1.1); skips switch videos, separate --out")
    ap.add_argument("--prompts-file", type=str, default="",
                    help="v3 : les prompts viennent de ce FICHIER (une par ligne, '#' ignore) "
                         "au lieu du corpus vidprom. Force le mode single, corpus_idx = "
                         "9000000 + rang, --out dedie ; combiner avec --twin-offset pour la "
                         "2e graine de chaque paire. Held-out verifie a la lecture.")
    ap.add_argument("--prompts-base", type=int, default=PROMPTS_FILE_BASE,
                    help="v3 --prompts-file : base des corpus_idx (defaut 9 000 000). Chaque "
                         "campagne custom DOIT avoir sa base (v2 = 9 000 000, v3 = 9 100 000) :"
                         " les repertoires probe1 (masks/, sig/, states/) sont indexes par "
                         "video_id 'main:<corpus_idx>', deux campagnes de meme base s'y "
                         "ecraseraient mutuellement.")
    ap.add_argument("--no-h", action="store_true", dest="no_h",
                    help="v3 : capture le rollout SANS stocker les sequences h (h_seq = {}) : "
                         "on ne garde que latents/conds/graines/stats, ~20 MB/video au lieu de "
                         "~1.1 GB. Pour une nuit de generation-labellisation ; la collecte h "
                         "se refait a l'identique (rollouts seedes) sur les paires retenues.")
    ap.add_argument("--allow-old-corpus-seed", action="store_true",
                    help="v3: allow reusing the v1 corpus seed 1234 (default: refuse; the "
                         "design wants a FRESH corpus, decision 6.4)")
    args = ap.parse_args()
    assert not (args.switch_mode and args.mix), "--switch-mode and --mix are exclusive"
    if args.v3:
        assert not args.switch_mode and not args.mix, \
            "--v3 has its own exact-count mode assignment (--n-switch); drop --switch-mode/--mix"
        main_v3(args)
        return
    switch_frames = tuple(int(x) for x in args.switch_frames.split(",") if x != "")
    collect_mode = "switch" if args.switch_mode else ("mix" if args.mix else "single")

    if args.mode == "manifest":
        merge_manifests(args.out)
        return

    selected = select_prompts(args.n, args.corpus_seed)
    if args.mode == "corpus":
        print("[corpus] first 5 selected prompts:")
        for i, p in selected[:5]:
            print(f"    [{i}] {p[:120]}{'...' if len(p) > 120 else ''}")
        per_pair = 3 * (2 * 1536 * 6144 + 2 * 6144 * 1536 + 2 * 1536 * 6144) \
            + 2 * 16 * 60 * 104
        proj = per_pair * args.positions * len(selected)
        print(f"[corpus] a-priori projection: {per_pair / 1e6:.1f} MB/pair x {args.positions} "
              f"pos/video x {len(selected)} videos = {proj / 1e9:.1f} GB "
              f"(ceiling {args.max_gb:.0f} GB)")
        if collect_mode != "single":
            modes = [choose_mode(i, args.switch_mode, args.mix, args.mix_seed)
                     for i, _ in selected]
            n_sw = modes.count("switch")
            print(f"[corpus] collect_mode={collect_mode}: {n_sw}/{len(modes)} switch videos "
                  f"({100.0 * n_sw / max(1, len(modes)):.1f}%), switch_frames={switch_frames}")
        return

    shard_i, shard_n = parse_shard(args.shard)
    mine = selected[shard_i::shard_n]
    shard_tag = f"shard{shard_i}of{shard_n}"
    os.makedirs(args.out, exist_ok=True)

    manifest_path = os.path.join(args.out, f"manifest_{shard_tag}.json")
    parts, completed, bytes_written, n_pairs_flushed = [], [], 0, 0
    modes_done = {}
    if os.path.exists(manifest_path):
        if not args.resume:
            sys.exit(f"[collect] {manifest_path} exists. Pass --resume to continue this "
                     f"shard, or delete its parts+manifest to start fresh.")
        with open(manifest_path) as f:
            prev = json.load(f)
        parts = prev["parts"]
        completed = prev["completed_corpus_idx"]
        bytes_written = prev["bytes_written"]
        n_pairs_flushed = prev["n_pairs"]
        modes_done = prev.get("video_modes", {})
        prev_mode = prev.get("collect_mode", "single")
        assert prev_mode == collect_mode, \
            f"resume mode mismatch: manifest says {prev_mode!r}, CLI says {collect_mode!r}"
        print(f"[collect] resuming {shard_tag}: {len(completed)} videos already flushed")

    device = torch.device("cuda")
    torch.set_grad_enabled(False)
    config = OmegaConf.load(args.config)
    config.distributed = False
    mem_layers = sorted(int(x) for x in config.model_kwargs.memory_layers)
    assert mem_layers, "config has no memory_layers, nothing to collect"

    pipeline = build_pipeline(config, device)
    header = shard_header(config, args.config, mem_layers, args.frames, args.positions,
                          collect_mode=collect_mode)

    def current_manifest(per_pair_bytes=None, projected_gb=None):
        return {
            "shard": args.shard, "created": time.strftime("%Y-%m-%d %H:%M:%S"),
            "config": args.config, "lora_ckpt": str(config.lora_ckpt),
            "use_ema": bool(getattr(config, "use_ema", False)),
            "frames": args.frames, "positions_per_video": args.positions,
            "n_total_prompts_campaign": len(selected), "n_prompts_shard": len(mine),
            "n_videos_done": len(completed), "n_pairs": n_pairs_flushed,
            "bytes_written": bytes_written, "parts": parts,
            "completed_corpus_idx": completed,
            "prompts": {str(i): p for i, p in mine if i in set(completed)},
            "per_pair_bytes": per_pair_bytes,
            "projected_campaign_gb": projected_gb,
            "corpus_seed": args.corpus_seed, "pos_seed": args.pos_seed,
            "collect_mode": collect_mode, "video_modes": modes_done,
            "switch_seed": args.switch_seed, "switch_frames": list(switch_frames),
            "mix_seed": args.mix_seed, "min_gap_chunks": args.min_gap_chunks,
        }

    buffer = []
    first_video_checked = bool(completed)   # projection check only on the very first video
    per_pair_bytes = None
    done_set = set(completed)
    t0 = time.time()
    for vnum, (cidx, prompt) in enumerate(mine):
        if cidx in done_set:
            continue
        t_vid = time.time()
        vmode = choose_mode(cidx, args.switch_mode, args.mix, args.mix_seed)
        sw_frame, cidx_b, prompt_b = (None, None, None)
        if vmode == "switch":
            sw_frame, cidx_b, prompt_b = pick_switch(cidx, selected, switch_frames,
                                                     args.switch_seed)
        print(f"[collect] {shard_tag} video {vnum + 1}/{len(mine)} corpus_idx={cidx} "
              f"mode={vmode}"
              + (f" switch_frame={sw_frame} corpus_idx_b={cidx_b}" if sw_frame else "") + "\n"
              f"          prompt: {prompt[:110]}{'...' if len(prompt) > 110 else ''}", flush=True)
        if prompt_b is not None:
            print(f"          prompt_b: {prompt_b[:108]}{'...' if len(prompt_b) > 108 else ''}",
                  flush=True)
        res = rollout_collect(pipeline, config, prompt, cidx, args.frames, args.positions,
                              device, pos_seed=args.pos_seed,
                              switch_frame=sw_frame, prompt_b=prompt_b, corpus_idx_b=cidx_b,
                              min_gap_chunks=args.min_gap_chunks)
        assert res["written"] == mem_layers, \
            f"ttt state written on {res['written']}, expected {mem_layers}"
        buffer.extend(res["pairs"])
        completed.append(cidx)
        modes_done[str(cidx)] = vmode
        print(f"[collect] corpus_idx={cidx}: {len(res['pairs'])} {vmode} pairs at chunks "
              f"{res['positions']} in {time.time() - t_vid:.0f}s", flush=True)

        if not first_video_checked:
            first_video_checked = True
            per_pair_bytes = sum(pair_nbytes(p) for p in res["pairs"]) / max(1, len(res["pairs"]))
            projected = per_pair_bytes * args.positions * len(selected)
            print(f"[collect] ===== DISK PROJECTION (after first video) =====\n"
                  f"          {per_pair_bytes / 1e6:.1f} MB/pair x {args.positions} pos/video "
                  f"x {len(selected)} videos (all shards) = {projected / 1e9:.1f} GB\n"
                  f"          ceiling: {args.max_gb:.0f} GB", flush=True)
            if projected > args.max_gb * 1e9:
                part = flush_part(args.out, shard_tag, len(parts), buffer, header)
                parts.append(part)
                bytes_written += part["bytes"]
                n_pairs_flushed += part["n_pairs"]
                write_manifest(args.out, shard_tag,
                               current_manifest(per_pair_bytes, projected / 1e9))
                sys.exit(f"[collect] ABORT: projection {projected / 1e9:.1f} GB > "
                         f"{args.max_gb:.0f} GB ceiling. First video kept for inspection. "
                         f"Reduce --positions (e.g. 2 -> "
                         f"{per_pair_bytes * 2 * len(selected) / 1e9:.0f} GB) or --n (e.g. "
                         f"{int(args.max_gb * 1e9 / (per_pair_bytes * args.positions))} videos "
                         f"at {args.positions} pos/video).")

        # flush between videos only, so a video's pairs are atomic within one part (resume-safe)
        if len(buffer) >= args.shard_size:
            part = flush_part(args.out, shard_tag, len(parts), buffer, header)
            parts.append(part)
            bytes_written += part["bytes"]
            n_pairs_flushed += part["n_pairs"]
            buffer = []
            write_manifest(args.out, shard_tag, current_manifest(per_pair_bytes))

    if buffer:
        part = flush_part(args.out, shard_tag, len(parts), buffer, header)
        parts.append(part)
        bytes_written += part["bytes"]
        n_pairs_flushed += part["n_pairs"]
    write_manifest(args.out, shard_tag, current_manifest(per_pair_bytes))
    print(f"[collect] {shard_tag} DONE: {n_pairs_flushed} pairs, {len(completed)} videos, "
          f"{bytes_written / 1e9:.1f} GB in {(time.time() - t0) / 3600:.1f} h", flush=True)
    del pipeline
    release_vram()


if __name__ == "__main__":
    main()
