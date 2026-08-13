# Fable code audit -- MAG memory (2026-07-03)

Method: 4 Fable (`claude-fable-5`) agents, one per subsystem (memory core / injection+inference / E1 objective+rollout / training loop+plumbing). Each read its files in full, hand-verified the math, and cross-checked call sites. Central question: is the "memory makes identity worse" negative a real result or a code artifact, plus any correctness bug.

## Headline verdict

- The **plain-inference attribution** ("the memory is what hurts identity", via `MEM_ZERO_ALL`) is **SOLID**: the read path is clean, the memory is invoked end-to-end on `causal_inference.py`, and the zero-injection ablation is a valid no-injection baseline. One caveat: re-confirm the `MEM_ZERO_ALL` value actually used (it is a frame threshold, not a boolean -- see MAJ-6).
- The **read-only negative** ("read is toxic") is **clean**.
- But the stronger claim "we fairly tested the full mechanism (trained write, cross-chunk, switch) and it still fails" is **NOT clean**: the write-training arms are confounded (CRIT-3, MAJ-1), cross-chunk is void (CRIT-2), switch evals are void (CRIT-4, MAJ-4, MAJ-7), and the default config ran with the write untrained (MAJ-2).
- No sign flip, no gate mis-init, no renorm error, no injection/detach bug on the read path. The core write math (Muon ascent + renorm-to-init + gated-AdaLN) is a faithful LaCT/Scal3R transcription.

Net: the negative is real for read-only and for the memory as actually run, and the attribution is trustworthy. This does not resurrect the mechanism (Fable's objective-level argument stands on its own), but it means the empirical "trained write doesn't help" evidence is weaker than we had been treating it, and several experiments need a clean re-run before any strong claim.

## CRITICAL

**CRIT-1 -- `id_loss_weight` crash in the BPTT branch (confirmed by 2 agents).**
`trainer/distillation.py:1150` reads `self.id_loss_weight`, which is never assigned on `Trainer` (only on `StreamingTrainingModel`, `model/streaming_training.py:87`). Any `bptt_window > 1` run raises `AttributeError` on the first generator step, caught by the catch-all (CRIT/MAJ-5 below), exits 0, wandb closes clean -> a dead run looks "completed". One-line fix: `self.streaming_model.id_loss_weight` or `getattr(self.config, "id_loss_weight", 0.0)`.

**CRIT-2 -- cross-chunk BPTT silently severed.**
`pipeline/streaming_training.py:284` (and the switch twin `:254`) set `_write_grad = requires_grad and (block_index < last)`, so the last temporal block's write runs under `no_grad`. But the state that crosses every chunk boundary is produced by exactly that last-block write. So with `grad_tail_W > 1` the cross-chunk gradient is exactly zero by construction: the summed-window backward is gradient-identical to bptt=1 at K x the cost. Every cross-chunk experiment (realk2 lineage) is void as a test of cross-chunk write training. Fix: thread a `keep_state_graph` flag so the last block keeps grad on tail chunks, still no_grad on the window's final chunk (retains the leak fix).

**CRIT-3 -- checkpoint-recompute clobbers the fast-weight state during write-training backward.**
`wan/modules/causal_model.py:603-605` does `ttt_cache.state = new_state` as a side effect inside a gradient-checkpointed block (`use_reentrant=False`, checkpointing on). At backward, non-reentrant checkpoint re-runs the block body, so the assignment re-fires; because backward recomputes blocks in reverse, the first sub-block's recompute fires last -> after backward the live state = state after only the chunk's first write. On every generator step ~6/7 of that chunk's writes are dropped from the carried state at train time, while inference accumulates all of them -> systematic train/eval state mismatch. Applies only to `train_write=True` (read-only runs take the non-checkpoint branch, so it does NOT explain the read-only negatives), but it confounds e2_write3L and realk2. NOTE: agent A independently flagged the same mechanism but judged it benign in the current schedule; agent B's reverse-recompute-ordering argument says it fires. This is the one finding worth a 5-minute empirical check before acting: snapshot `[c.state for c in ttt_cache]` right after generation and again after `backward()`, compare. Fix: return `new_state` as a block output and assign it outside the checkpointed region (or snapshot+restore in the trainer).

**CRIT-4 -- switch and interactive inference bypass the memory entirely (known bug, still unfixed).**
`pipeline/switch_causal_inference.py` and `interactive_causal_inference.py` never pass `ttt_cache`, so `causal_model.py:565` skips the memory branch with no error. Memory-on vs memory-off are identical on these paths -> all switch/interactive evals are void. Fix: mirror `causal_inference.py` (fresh `ttt_cache` per sequence, `ttt_update` per pass, `switch_global_frame` threading).

## MAJOR

**MAJ-1 -- multi-GPU trains world_size different, never-synchronized memory modules.**
`set_seed(seed + rank)` before model construction -> divergent memory init per rank; base ckpt loaded `strict=False` with memory keys tolerated -> nothing overwrites it; memory in FSDP `ignored_modules` with `sync_module_states=False`; no gradient all-reduce anywhere for the ignored params; `save()` gathers rank 0 only. So every >=2-GPU memory run: the saved memory saw only rank 0's local batch, from a random init no other rank shares, and ranks 1+ pushed different memory outputs through the shared critic/LoRA grads (extra noise). This alone can produce a spurious "memory ON is worse" at fixed step budget. Fix: broadcast memory params from rank 0 after init, all-reduce their grads before `optimizer.step()`.

**MAJ-2 -- the default config tests nothing it was meant to.**
`configs/ttt_memory.yaml` leaves `train_write`, `bptt_window`, `id_loss_weight`, `tv_loss_weight`, `text_dropout_prob`, `lora_unfreeze_from_block` all unset (off), and `lr: 0.0` makes LoRA inert. So the headline default-config "memory ON worse" was measured with an untrained write and no identity term -- the code ran as configured, but the config does not exercise the mechanism. Dead keys in the yaml: `total_batch_size`, `load_raw_video`, `auto_resume` (overridden, MAJ-5), `ema_weight`/`ema_start_step` (EMA force-disabled in LoRA mode).

**MAJ-3 -- text-dropout largely bypassed by the cross-attn cache.**
`model/streaming_training.py:252-261` swaps to a null text dict, but the cross-attn cache was already filled with the true prompt on the first no-grad forward (chunk 0 never dropped), and every later `no_grad` denoising pass reads the cache and ignores the null text (`model/model.py:174-183`). Only the final grad exit-step sees null text; critic-step rollouts see it never. So "text-dropout doesn't un-starve the memory" is not trustworthy. Fix: bypass the cross-attn cache for the whole dropped chunk.

**MAJ-4 -- `global_sink` read off the empty `*args` tuple -> always False in switch training.**
`pipeline/streaming_switch_training.py:31` does `getattr(args, "global_sink", False)` where `args` is a tuple (the pipeline is built kwargs-only), so config `global_sink: true` is silently discarded. Training zeroes the entire KV cache at every switch; inference keeps the sink -> train/eval cache-regime mismatch around every switch. Fix: `kwargs.get("global_sink", False)`.

**MAJ-5 -- `train()` swallows every exception.**
`trainer/distillation.py:1509` catches all, prints on rank 0 only, no re-raise, no exit(1), no barrier. A non-main-rank OOM/NaN prints nothing and exits 0 while other ranks hang in the next NCCL collective until `PG_TIMEOUT_MIN` (30 min). It also defeats the fail-loud drift-guard assert. Fix: re-raise (print on all ranks) and let torchrun propagate a non-zero exit.

**MAJ-6 -- `MEM_ZERO_ALL` is a frame threshold, not a boolean.**
`causal_model.py:599` zeroes only chunks with `start // seqlen < int(MEM_ZERO_ALL)`. Setting `=1` (the natural boolean) ablates only the frame-0 chunk, leaving memory active for ~97% of a long video. Nothing in the repo sets the var, so the value used for the "100% of the damage" attribution cannot be audited from code -> re-confirm it was set >> total latent frames. Fix: treat any nonzero as zero-everywhere, print a loud banner when active.

**MAJ-7 -- `frames_since_switch` gate signal only wired inside the switch chunk.**
The chunk after the switch snaps `fss` back to frames-since-video-start (`causal_model.py:585`), so the gate cannot learn persistent post-switch behavior. Inert with `gate_extra: []`, active in e3_overfit14. Fix: persist the global switch frame and pass it on every later chunk, the recache pass, and inference.

**MAJ-8 -- `auto_resume` override + `muon=0` mislabel.**
(a) `train.py:31` overwrites the yaml `auto_resume: False` to True unless `--no-auto-resume` is passed -> relaunching into a logdir with checkpoints silently resumes the old LoRA ckpt (a "fresh" ablation that isn't). (b) `video_memory.py` runs Newton-Schulz unconditionally, which unit-normalizes the grad before the `steps` loop, so a `muon_update_steps=0` ablation actually tests unit-Frobenius-normalized SGD, not SGD -- the gradient magnitude is erased regardless. Any past `muon=0` comparison is mislabeled. Fix: `if steps == 0: return G`; only override auto_resume when the flag is given.

## Design note (not a bug)

**Write magnitude is fixed and unlearnable.** Newton-Schulz drives all singular values of the update to ~1, so the per-chunk write strength is constant; the learned `lrs` only reweight the token mixture. Every clean chunk rotates each state column by a fixed ~26 degrees (effective horizon a few chunks), and a numerically tiny grad (near-static / degenerate chunk) is amplified to the same full-strength write. The model cannot learn to write less on garbage chunks. This is faithful to LaCT/Scal3R (verified layout+dims+renorm), not a transcription bug, but it is a concrete mechanistic candidate for "memory injects harm" and it aligns with the objective-level diagnosis. A fix would be a design change: a learned scalar step-size applied after Muon.

## MINOR (latent, mostly inert under the current config)

- OOM guard in the identity loss can `KeyError` via `merge_dict_list` and desync the objective across ranks (`model/streaming_training.py:813`); fix: fill the zero-default log dict in the `except`.
- Identity anchor blends batch samples if `batch_size > 1` (`:717`); inert at batch 1.
- tv-floor motion is diluted by overlap frames + the first-frame boundary spike (`:735`), so the anti-freeze hinge can silently never fire; `gradient_mask` is accepted but unused.
- i2v runs compare the first chunk to itself (`:658` vs `:743`); T2V unaffected.
- windowed path without force-fresh (`id_loss_weight==0`) can log silent zero-loss generator steps (`trainer:1153`).
- `_log_memory_stats` drift/statenorm baselines collide across memory layers (`trainer:1655`) -> the 3-layer runs' drift curves were junk (logging only).
- `denoising_step` gate signal is the raw warped timestep (~1000/750/...), aliases the sinusoid; fine unless the step list changes at eval.
- `forward_sequence` in `video_memory.py:436` is dead code and crashes with any `gate_extra`.
- renorm target norm is detached (biases, does not kill, the write grad); q_norm/k_norm built without eps (~0.4% bias).

## Verified clean (audited, no defect)

- Ascent sign, renorm norm-conservation, gate init (do-no-harm at g_init=0), apply-first ordering, (b h) fold consistency, Newton-Schulz coefficients.
- Detach on the write pass (blocks no loss grad; the write grad flows via new_state -> next sub-block's read; read grad flows on the exit-step read).
- E1 loss plumbing: anchor set from the rollout path (the earlier structural no-op is genuinely fixed), set once per sequence, on chunk-0's last k frames, decoded no-grad + detached; differentiable end-to-end through decode -> gate -> read -> write; cosine and tv-floor signs correct; grad reaches both read and write; grad probe confirms it empirically.
- `_write_context` VRAM-leak fix (both call sites), selective LoRA unfreeze ordering (before FSDP, use_orig_params), BPTT window indexing, the dmdtrain guard, logging try/except.
- Plain inference invokes the memory end-to-end (read every pass, written once per sub-block on the clean re-cache).

## Config-lineage implications (memory notes to revisit)

- `e2-3lwrite-eval-not-horizon` (write worse than baseline) -> confounded by CRIT-3 (+ MAJ-1 if multi-GPU). The conclusion "bottleneck = objective not gradient path" drew on this run and is weakened, not overturned.
- `realk2-unblocked` cross-chunk runs -> void by CRIT-2.
- Switch/overfit14 switch evals -> void by CRIT-4 / MAJ-4 / MAJ-7.
- `muon-decay-verdict` muon comparisons -> mislabeled by MAJ-8b.
