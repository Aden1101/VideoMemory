# Cross-chunk write-training horizon: exhaustive solution analysis (2026-06-28)

5 parallel agents analyzed how to extend the write-training horizon past ~5s (1 chunk). Goal: train the
memory WRITE so info written in chunk k stays useful when read in chunk k+1..N (60-240s identity). Below
is the synthesis; each section cites the agent's code-grounded findings.

## The problem (recap)
Within-chunk BPTT trains the write only ~5s (7 blocks/chunk). Chunk hard-capped at 21 latent frames
(model seq_len 32760). Naive cross-chunk BPTT (carry the TTT state graph across chunks, sum K losses, one
backward) CRASHES: `CheckpointError: recomputed values different metadata` — gradient checkpointing
recomputes chunk-1's forward against the ROLLING KV cache that chunk-2 advanced. No-checkpoint OOMs.

## CONVERGENT FINDINGS

### 1. My "snapshot/restore the cache" idea is INFEASIBLE — drop it (Agent A)
Content-faithful snapshots = storing the no-checkpoint attention tape across the window ≈ **~90 GiB for
K=2** (un-shardable by FSDP). Indices-only snapshot kills the crash but gives **silently WRONG gradients**.
Useful by-product: the crash is precisely an **eviction/offset artifact** — in the append (no-eviction)
regime the recompute index formula (`causal_model.py:309`) self-reconstructs correctly, so recompute is
faithful for free. Sizing `kv_cache_size` so eviction never fires within the window (raise
`slice_last_frames`) removes the CRASH cheaply — but you still hit the ~24-frame activation wall.

### 2. THE KEY INSIGHT (Agents B + C converge): the cross-chunk gradient is CACHE-FREE
The cross-chunk credit path is `loss(chunk k+1) -> read of carried fast-weight STATE -> chunk k's WRITE`.
`memory.step` (video_memory.py:366-395) NEVER reads the KV cache — the write (`swiglu_fast_weight_gradient`
+ `weight_norm_update`) only touches `h=norm1(x)` and the prior state `w`. So `w_k`'s grad_fn depends only
on `{h (value only), qkv/lrs params, w_{k-1}}`. The KV cache is dragged in ONLY by two *incidental*
couplings: (A) the write shares the per-block `checkpoint(...)` with `self_attn`, and (B) `h` carries a
graph into the frozen backbone (lr=0 -> that gradient trains NOTHING).

**Implementable fix (NOT snapshot):** isolate the cache-free state path —
- detach the write's input `h` (free: backbone is frozen, lr=0, so zero gradient is lost);
- lift the write OUT of the self_attn checkpoint (run the write-pass backbone under no_grad, only
  `memory.step`'s update under enable_grad -> not checkpointed -> no recompute of attention -> no cache conflict);
- carry ONLY the cache-free state grad across chunks (per-chunk immediate backward for the read path +
  a leaf-bottleneck `s_k = w_{k-1}.detach().requires_grad_()` to inject the cross-chunk grad), OR the
  equivalent "last-chunk-loss + detached-h write bridge".
This keeps EXACTLY the write-training long-horizon gradient (qkv/lrs/fast-weight-init), drops only the
zero-valued frozen-backbone terms, **fits memory** (~44 GiB/3L, scales to K>2 cheaply — Agent B), and
**structurally dodges the crash** (the cross-chunk subgraph never touches the cache). Effort: M-L.
Files: video_memory.py:366-388, causal_model.py:556-596 + 1196-1221, pipeline write call + _write_context,
model/streaming_training.py C3 detach, trainer/distillation.py:1117-1137.

### 3. LITERATURE REFRAME (Agent D): the field does NOT do cross-chunk BPTT over a rolling cache
TTT-Video (2504.05298) and LaCT (2505.23884) do FULL BPTT through the fast-weight recurrence WITHIN ONE
forward over one in-context sequence, state kept FUNCTIONAL (a scan input/output, never an in-place cache),
memory bought back with REMATERIALIZATION of the scan + BIG chunks. Both RESET the fast-weight to a learned
init per sequence — never carry across cache-mutating calls. Our crash is the textbook
"`use_cache=True` incompatible with gradient checkpointing". Long horizon is reached via:
(a) **CURRICULUM on in-context length** (TTT-Video staged 3s->9->18->30->63s, training only TTT+gates each
stage) — the single highest-value portable mechanism; our scaffold is commented at
model/streaming_training.py:391-396. (b) **BIGGER chunks** (LaCT Eq3: bigger chunk = higher MFU AND more
write credit-assignment per backward — matches our reference/chunk_bptt_investigation.md). Also: LaCT's
released VIDEO config sets `use_moun=false` (Muon off) — de-risks our Muon arm; and all-reduce raw grads
BEFORE the Muon step for context-parallel (answers our allreduce_fn=None). KEY: a 5s-trained write needs a
curriculum to reach 60-240s; do NOT expect 40x zero-shot extrapolation.

### 4. CHALLENGE THE PREMISE (Agent E): EVAL FIRST; teacher-distillation may beat cross-chunk entirely
"Identity drift" has 4 failure modes needing OPPOSITE fixes: (A) write doesn't store identity = horizon
(needs cross-chunk); (B) gate/read don't pull it back at range (within-chunk fixable); (C) state decays
too fast = a scalar lr/renorm knob (video_memory.py:185-187,260-262); (D) gate~=0, memory contributes
nothing. With gate_init 0.1 + lr_mem 1e-4, the likely bottleneck is (D)/(B)/(C), NOT horizon. So measure
first. STRONGEST BPTT-FREE alternative: **teacher feature-distillation** — the frozen BIDIRECTIONAL teacher
`real_score` (already instantiated, model/dmd.py:448) sees full/long context; hook its block-5 features
over past+current frames as the WRITE's TARGET (`MSE(student post-memory feat, teacher feat.detach())`,
attach in model/streaming_training.py:710). Within-chunk differentiable (no carried graph, no crash, no
OOM), and it STRUCTURALLY fixes the within-chunk blind spot (the last-block/boundary write that currently
gets zero gradient now gets a target). The long-range info lives in the TARGET, not the backward. Caveat:
feed the teacher only past+current (no future leak).

## RECOMMENDED ORDER (what to actually do)
1. **EVAL FIRST** (the 800-1000-step 3L run -> VBench 60/120/240s + per-length identity score + the
   gate/drift logs). Decide the failure mode. Cheap knob sweeps (lr_mem/renorm, gate, #layers) in parallel.
   [Agents A, D, E all: don't build cross-chunk until the eval proves it's horizon-limited.]
2. If NOT horizon-limited -> fix the cheap thing (gate/decay/capacity). Done.
3. If horizon-limited -> try the CHEAPER long-horizon routes before cross-chunk BPTT:
   (a) **curriculum on in-context length** [Agent D, the field's actual method], and/or
   (b) **teacher feature-distillation** [Agent E, BPTT-free, fixes the boundary-write gap].
4. Only if (3) underperforms -> implement the **cache-free-state-gradient cross-chunk BPTT** [Agents B+C],
   NOT the snapshot (infeasible). Effort M-L.

Bottom line: cross-chunk BPTT IS doable (via the cache-free state path, not the cache snapshot), but the
eval comes first and the curriculum / teacher-distillation routes are likely cheaper AND as effective.
Full agent reports are in this session's transcript.
