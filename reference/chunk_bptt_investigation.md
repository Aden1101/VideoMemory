# Chunk / BPTT / leak — code investigation (2026-06-27)

Goal of this check: answer "can we decouple chunk-size from the window and put several
write/read cycles in one backward (smaller chunks, faster reuse)?" + the gradient-checkpoint
gotcha. Reading the actual code overturned several assumptions we'd been working from. This is
the corrected picture.

## TL;DR (the correction)

**The memory write does NOT happen once per chunk. It happens once per BLOCK (3 frames).**
A "chunk" (21 frames) is generated as `21/3 = 7 blocks`, and the memory is read+written **per
block**, all inside ONE `generate_chunk_with_cache` call, under ONE autograd graph, freed by ONE
`backward()`.

Consequences:
- **Within-chunk cross-block write training already exists.** On generator steps the state is
  carried block->block WITH grad, and `backward()` flows from later blocks back to earlier blocks'
  writes. So the write IS being trained, over ~6 block-hops (~18 frames ~4.5s). My earlier claim
  "the write is not trained at all / K=0" was WRONG.
- The "make chunks smaller so cycles fit in one backward" idea is **already realized at the block
  level** (block = 3 frames = the recurrence unit). We don't need to shrink the chunk to get
  multi-cycle training in one backward; it's already there.
- The **C3 detach** ([[phase2-c3-blocker-fixed]], `model/streaming_training.py:288`) only cuts the
  carry **across chunks** (chunk->chunk), at chunk start. It does NOT touch the within-chunk
  block chain. So it costs less than we thought.

## How the write actually works (verified)

`pipeline/streaming_training.py :: generate_chunk_with_cache`:
- L128-130: `num_blocks = chunk_frames // num_frame_per_block` (21//3 = 7 blocks/chunk).
- L174 `for block_index ...:` per-block loop:
  - L184-261: denoising read passes, `ttt_update=False`. Intermediate steps `no_grad`; the final
    (exit) step is `enable_grad` iff `local_start_frame >= start_gradient_frame_index`.
  - L277-291: **the write**, `with self._write_context(): generator(..., ttt_update=True)` — a WHOLE
    extra full-model forward (the "clean rerun") that updates the fast-weight state.
- `start_gradient_frame_index` (L159-162): `requires_grad=True -> 0` (all blocks grad-enabled);
  `requires_grad=False -> chunk_frames` (no read ever grad-enabled).

`wan/modules/video_memory.py :: step` (per block):
- L374 `o = swiglu_apply(q, w0, w1, w2)` — APPLY-FIRST read, uses the state from the PREVIOUS
  block's write.
- L376-388 the write: `new_state = update(w, grad(k,v))`; under `train_write` `detach_state=False`
  so `new_state` keeps grad_fn -> the block chain carries gradient.

So per generator step the graph is: `output_block_{n+1} -> read -> state(after block n write) ->
write_block_n -> ... -> w0 (leaf)`. `backward()` trains writes of blocks 1..6 (block 7's write is
dangling: nothing reads it in this chunk).

## The real leak (high-confidence prime suspect)

`_write_context()` (`pipeline/streaming_training.py:79-80`) returns `enable_grad` whenever
`train_write=True` — it **ignores `requires_grad`**. But the write sits inside the block loop and
runs on EVERY step, including critic steps.

Training loop (`trainer/distillation.py`, `dfake_gen_update_ratio=5`):
- generator step (1 of 6, `step % 5 == 0`): `generate_next_chunk(requires_grad=True)` (L1115/1123)
  -> reads grad + write grad -> `backward()` (L1143) frees the graph. OK.
- critic step (5 of 6): `generate_next_chunk(requires_grad=False)` (L1168) -> reads `no_grad`, BUT
  the write still runs under `enable_grad` -> builds the full clean-rerun graph (7 blocks x 32
  transformer blocks). Then the chunk is **detached** (L1177-1178) and the critic `backward()`
  (L1191) goes through the fake_score only -> **the write graph is NEVER backpropped, never freed
  by backward.** = the `(1,4680,1536)` / `causal_model.py:608` retained blocks measured as the leak.

So on 5 of every 6 steps we build a ~3.7 GB autograd graph for nothing. That matches the
measurement (the steps profiled, 1/2/3, are all critic steps; each adds ~3.7 GB).

### The clean fix (low-risk, principled)
Gate the write's grad on `requires_grad`:
```python
# pipeline/streaming_training.py
def _write_context(self, requires_grad=True):
    return torch.enable_grad() if (getattr(self, "train_write", False) and requires_grad) else torch.no_grad()
...
with self._write_context(requires_grad):     # pass the per-chunk flag through
    self.generator(..., ttt_update=True)
```
- On critic steps the write still RUNS (state still adapts -> generation quality unchanged) but
  under `no_grad` -> **no graph built -> nothing to leak.**
- On generator steps unchanged -> the write still trains (within-chunk).
- This does not change what is learned (the write is only ever trained on generator steps anyway).
- Expected: removes the dominant per-step graph on 5/6 steps -> the OOM should disappear and
  2 layers + window 24 should fit. MUST verify by measurement (mem_breakdown) before trusting it.

This is much smaller and safer than the "K=2 cross-chunk truncated BPTT" I was proposing earlier.

## On the user's chunk-size question

- "Smaller chunks so the memory reuses writes faster" -> already true: reuse latency is **1 block
  = 3 frames ~0.75s**, not 1 chunk ~5s.
- "Put several write/read cycles in one backward" -> already true: 7 cycles per chunk in one graph.
- The actual lever that chunk-size controls is the **within-chunk BPTT depth** = `chunk_size/3`
  blocks. BIGGER chunk = MORE blocks = DEEPER credit assignment for the write (and more VRAM).
  SMALLER chunk = shallower BPTT (and less VRAM, faster reuse). For training the write this is the
  OPPOSITE of "smaller is better": smaller chunks give the write LESS gradient horizon, not more.
- For our goal (slow identity info), ~4.5s of within-chunk horizon is already a real signal.
  Recommendation: do NOT shrink the chunk for training reasons. Keep 21 (or even consider larger
  once the leak is gone and VRAM allows), and only revisit cross-chunk carry if within-chunk proves
  insufficient.

## Gradient-checkpoint gotcha (for the record)

`slice_last_frames` (`configs/ttt_memory.yaml:86 = 9`) sets the KV cache window
(`kv_cache_size = (local_attn_size + slice_last_frames) * frame_seq_length`,
`pipeline/streaming_training.py:59-60`). With `gradient_checkpointing: true` (config L59), changing
the KV slice changes what the recompute pass sees vs the forward pass; the 24->8 experiment tripped
a "recomputed metadata mismatch" and hung at step 0. Lesson: KV-slice / chunk-size changes are not
free under gradient checkpointing -> change one knob at a time and smoke-test step 0. This is
independent of `streaming_chunk_size`.

## Recommended staged plan

1. **Fix the leak** = gate the write's `enable_grad` on `requires_grad` (above). Measure with
   `mem_breakdown` that static mem is now FLAT across steps (like read-only).
2. With the leak gone, **launch write training** (within-chunk BPTT, layers [5], window 24/chunk 21,
   gate combo) on gpu1/2/7 and confirm it trains the write (gate/mem grads non-zero, loss moves),
   wandb-logged.
3. Only if within-chunk (~4.5s) horizon proves too short for identity: extend horizon via larger
   chunk (more blocks, more VRAM) OR proper cross-chunk truncated BPTT (replace C3's blanket detach).

## What I got wrong before this check (honesty log)
- "The write is not trained at all (K=0), C3 cuts the only gradient." -> WRONG. Within-chunk
  block-chain BPTT trains the write on generator steps; C3 only cuts cross-chunk.
- "Fix needs cross-chunk truncated BPTT (K=2), heavy." -> Not first. The leak is a critic-step
  enable_grad bug; the targeted fix is one-line-ish and low-risk.
- Lesson (again): measure/read before theorizing. The per-block write structure was in the code the
  whole time.
