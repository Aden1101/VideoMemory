# Write-training VRAM leak — investigation (2026-06-27, overnight)

## TL;DR
Write training (`train_write=true`) leaks **~3.7 GB of LIVE VRAM per training step**, linearly, until OOM
(~step 9-10, peak hits the 48 GB card limit). Read-only training does **not** leak (flat). This is the
real reason "the write doesn't run anymore" — not a config/headroom issue, an actual memory leak.
**It was always there; it only bites now because the config sits near the 48 GB edge.**

## Proven by measurement (not theory)
Instrumented `trainer/distillation.py` (gated by config flags, all default-off):
`mem_breakdown`, `mem_snapshot`, `gc_each_step`.

```
              static (persistent mem at step start)
read-only :   2.5 -> 12.46 -> 12.46 -> 12.46     = FLAT, no leak  (runs to step 84+)
write     :   2.6 -> 17.57 -> 21.24 -> 24.90 -> 28.57   = +3.7 GB/step LINEAR -> OOM
```
- `gc` live cuda-tensor count grows **+346/step** (write); flat for read-only.
- shape histogram: the growers are **`(1,4680,1536)`** tensors (one `num_frame_per_block`=3-frame block
  activation) **+210/step**, each with a `grad_fn` -> they are saved activations of a **retained autograd graph**.
- CUDA memory snapshot (`_record_memory_history(stacks="python")`, dumped at step 3, parsed):
  the leaking 13.7 MiB blocks are allocated at **`wan/modules/causal_model.py:608` in `cross_attn_ffn`**
  (the block-forward FFN residual) — i.e. the **clean-rerun's full model forward graph**.

## What it is
`train_write=true` makes `_write_context()` (`pipeline/streaming_training.py:79-80`) return `enable_grad`
for the **clean-rerun** (the `ttt_update=True` pass that writes the chunk into memory). That runs the WHOLE
generator forward (32 blocks) under `enable_grad`, building a full block-forward graph each step. That graph
is **never freed** -> +3.7 GB/step.

## What it is NOT (ruled out by experiment)
- NOT cyclic garbage: forcing `gc.collect()` every step did **not** stop it.
- NOT fragmentation: `expandable_segments` / `garbage_collection_threshold` don't help (it's LIVE mem).
- NOT `mem_out` flowing downstream: detaching it (`causal_model.py:593`, applied) didn't change the leak.
- NOT the KV/crossattn/flow/ttt caches: all are detached each step AND `reset_state` clears them ~every
  step (video len = `streaming_max_length`=24 ~= 1 chunk), yet the leak persists -> root is OUTSIDE these.
- NOT in `self` / `model` / `streaming_model.__dict__` (scanned depth 3): 0 grad-roots found there.
  -> the graph is held by the C++ autograd engine; its root reference is subtle / inside a big submodule.

## Why it matters / why now
The leak adds ~3.7 GB/step. The config peaks ~33 GB by itself, so it OOMs at ~step 10. June fit because...
actually June would have leaked too, but likely ran short videos / fewer steps before reset. The headroom
work (window 24, chunk 21, combo, 2 layers) all sits near 48 GB, so the leak tips it over fast.

## Fix direction (needs the user's design intent — NOT a safe blind change)
The crux: **is the write meant to be trained cross-chunk (state written at step N, read at step N+1) or
within-chunk?**
- The **C3 fix** (`model/streaming_training.py:288-292`) detaches the TTT state at each chunk boundary.
  If the write is meant to be trained cross-chunk, C3 **cuts that gradient** -> the clean-rerun graph is
  built but never backpropped -> never freed -> leak. (So C3 may be both "fixing the crash" and "causing
  the leak + not actually training the write cross-chunk".)
- **Fix A (if within-chunk training is intended):** run the clean-rerun's BACKBONE under `no_grad`, and only
  the memory module's write update under `enable_grad` (with its input as a detached leaf). This removes the
  ~15 GB block-forward graph (the leak) while still training the write params. This is the principled fix and
  should make 2 layers + window 24 + combo fit on 3 GPUs.
- **Fix B (if cross-chunk training is intended):** replace C3's blanket detach with a proper truncated-BPTT
  that backprops + releases the write graph each step (sliding window), instead of building-and-leaking it.

## What's running now (overnight)
- `read-only` 1-layer on gpu1/2/7 (`/tmp/_read1L.yaml`, wandb `read_1L_3gpu`): stable, no leak, trains the
  gate + read-path (Phase-1). Held by `reference/keepalive.sh` (relaunches if it ever dies -> never idle).
- gpu6 = dead (poison, excluded). gpu0/3/4/5 = other users. So gpu1/2/7 are our only usable cards.

## Diagnostic instrumentation left in place (all gated off by default)
`trainer/distillation.py`: `mem_breakdown` (static-vs-peak + gc shape histogram + grad-root scan),
`mem_snapshot` (records `_record_memory_history` python stacks, dumps `/tmp/memsnap.pickle` at step 3),
`gc_each_step`. Parser: `/tmp/parse_snap.py`. Turn on via the config to re-measure.
