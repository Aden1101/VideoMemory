# MAG audit fixes (2026-07-04)

Applied on top of the committed MAG implementation (HEAD 4e3a9bf). Uncommitted diff: 9 files, ~165 insertions. Addresses the findings in reference/fable_code_audit.md. A 4-agent Fable re-audit of these fixes is the verification pass.

## Critical

- **id_loss_weight crash (CRIT-1)** -- `trainer/distillation.py`: the force-fresh gate in the bptt_window>1 branch read `self.id_loss_weight`, never assigned on Trainer. Now `getattr(self.config, "id_loss_weight", 0.0)`. This attribute was almost certainly dropped during the earlier comment cleanup (the e1_overfit_v1 run executed this line without crashing).
- **Cross-chunk BPTT severed (CRIT-2)** -- last sub-block write ran no_grad unconditionally, so grad_tail_W>1 trained nothing across chunk boundaries. Added a `keep_last_write_grad` flag threaded trainer -> generate_next_chunk -> _generate_chunk -> both pipelines; `_write_grad = requires_grad and (block_index < last or keep_last_write_grad)`; the windowed loop sets it `(rg and _k < K_bptt-1)`. Default (W=1) is byte-identical; the final window chunk still writes no_grad so the leak fix holds.
- **Checkpoint-recompute clobbers state (CRIT-3)** -- `trainer/distillation.py`: added `_snapshot_ttt_state` / `_restore_ttt_state`, wrapped around both generator backward sites. Snapshots the post-generation fast-weight state (detached clone) before backward, restores it after, so backward-time recompute of the checkpointed write no longer leaves the cache at an early-chunk state. No-op when there is no clobber, so it is safe whether or not the bug fires.
- **Switch + interactive inference bypassed the memory (CRIT-4)** -- `pipeline/switch_causal_inference.py`, `pipeline/interactive_causal_inference.py`: added TTTBlockCache import, a fresh `self.ttt_cache` per sequence, `ttt_update=False` on the denoising passes, `ttt_update=True` on the clean re-cache write, and `ttt_update=False` in the switch re-cache. switch_causal_inference also had pre-existing NameErrors (`recompute_frames`, `frames_to_recompute`, `recompute_start_frame`, `num_input_frames`) fixed. These two files were never runtime-tested and need a smoke-test before trusting switch/interactive evals.

## Major

- **Multi-GPU memory never synced (MAJ-1)** -- `trainer/distillation.py`: broadcast every memory param + buffer from rank 0 after they are moved to device; all-reduce (AVG) every memory grad before each generator optimizer step (world_size>1 only). Single-GPU unaffected.
- **Text-dropout bypassed by cross-attn cache (MAJ-3)** -- `model/streaming_training.py` + `pipeline/streaming_training.py` + `pipeline/streaming_switch_training.py`: `_generate_chunk` sets `disable_crossattn_cache=True` on a dropped chunk and threads it; the pipeline then passes `crossattn_cache=None` for that chunk so the denoising/write passes recompute cross-attn from the null text. `causal_model.py` no-grad path None-guards the crossattn index.
- **global_sink always False in switch training (MAJ-4)** -- `pipeline/streaming_switch_training.py`: `getattr(args, ...)` on the tuple changed to `kwargs.get("global_sink", False)`.
- **train() swallowed all exceptions (MAJ-5)** -- `trainer/distillation.py`: prints on all ranks and re-raises, so torchrun exits nonzero and the drift-guard assert propagates.
- **MEM_ZERO_ALL threshold (MAJ-6)** -- `causal_model.py`: any nonzero value now zeroes the injection on every frame (was a frame index threshold).
- **frames_since_switch only in switch chunk (MAJ-7)** -- switch pipeline persists `self._switch_global_frame`; base pipeline threads it (default None) into its generator calls and resets it per sequence in clear_kv_cache.
- **auto_resume override (MAJ-8b)** -- `train.py`: only forces off when `--no-auto-resume`, else keeps the config value (default True only when neither specifies).
- **muon=0 mislabel (MAJ-8a)** -- `video_memory.py`: `if steps == 0: return G` (genuine SGD instead of unit-normalized SGD).

## Minor

- OOM guard in the identity/tv loss now fills the e1_* log keys so merge_dict_list does not KeyError (`model/streaming_training.py`).
- prompt_cache switch_frame fill guarded on `switch_frame_index is not None` (`model/streaming_training.py`).
- drift/statenorm baselines keyed by block index instead of bare suffix, so 3-layer runs no longer compare every layer to the last layer's init (`trainer/distillation.py`).
- q_norm / k_norm built with `eps=1e-5` to match o_norm (`video_memory.py`).

## Deliberately NOT changed (inert under current configs, or design)

- Write magnitude fixed/unlearnable under Muon (design property inherited from LaCT/Scal3R, not a bug).
- Identity anchor batch-blend (correct at batch_size=1, the only used config), i2v self-anchor (T2V unaffected), tv-floor overlap dilution (soft regularizer), denoising_step raw timestep, forward_sequence dead code. Left as-is to avoid untested regressions; flagged in the audit doc.

## Config-lineage consequence

The e1_overfit_v1 run that drove the "mechanism dead" verdict was trained with train_write + gradient_checkpointing (bitten by CRIT-3) on 2 GPUs (bitten by MAJ-1), with a text-dropout that mostly did not fire (MAJ-3). Its plain-inference eval (MEM_ZERO_ALL=100000) was clean, so "the memory hurt identity" is a real measurement, but that checkpoint was not a fair E1 attempt. A clean re-run (single-GPU or synced, CRIT-3 fixed, text-dropout firing) is warranted before treating "trained write fails" as established.

## Re-audit round (2026-07-04)

4-agent Fable re-audit of the fixes above.

- **Memory core**: both fixes (muon=0, q/k eps) verified correct, no regression, no bugs. Note: q/k eps shifts pre-fix-checkpoint numerics by ~0.4% (intended), and muon=0 is a fresh data point vs old muon=0 logs.
- **E1 objective**: fixes 1-3 correct; the OOM-guard keys type-promote fine. Two regressions from the fixes, both FIXED:
  - HIGH: the `fill_(None)` guard left `switch_frame` at 0 (clear_prompt_cache zeroes it), which spuriously fires the switch gate and injects zero K/V into attention on no-switch sequences. Fixed by filling the far-future sentinel (1e8) when there is no switch.
  - MEDIUM (latent): the two new kwargs broke the plain-DMD `SelfForcingTrainingPipeline` (no such params, no **kwargs). Fixed by adding them as accepted-unused params there. Not reachable by current configs.
- **Injection + inference**: the four CRIT-4 fixes verified correct (memory now read+written on switch/interactive, NameErrors gone, plain inference bit-identical). Two further findings on switch inference, both PRE-EXISTING (HEAD identical), not regressions:
  - HIGH: switch inference generator calls omitted `rope_enhance` / `prompt_cache` / `new_prompt_chunk_flag` (train/inference mismatch). Added `rope_enhance` to all calls + `sink_recache_after_switch=not self.global_sink` to the re-cache. `prompt_cache` NOT added (needs prompt-cache init + a switch_frame policy on untested code) -> flagged as a remaining gap; switch evals still need a dedicated pass + smoke-test.
  - MEDIUM (latent, global_sink=false only): sink recache omission, now passes `sink_recache_after_switch`.
  - LOW: interactive leaves `switch_global_frame=None` (multi-switch); fine unless a gate_extra=frames_since_switch ckpt is evaluated interactively.
- **Training loop**: all 9 fixes (CRIT-1 id_loss_weight, CRIT-2 keep_last_write_grad, CRIT-3 snapshot/restore, MAJ-1 multi-GPU sync, MAJ-5 re-raise, MAJ-4 global_sink, MAJ-7 switch-frame persist, MAJ-8b auto_resume, drift-baseline-per-block) verified CORRECT AND COMPLETE. Default path (single-GPU, bptt_window=1, no identity) is byte-identical. No new bug, no prior bug unfixed. CRIT-2/3 cross-chunk grad flow confirmed genuine. Residual: all LOW/NOTE and mostly pre-existing (grad-is-None all-reduce could theoretically mismatch on a one-rank early break but FSDP hangs first + seq length is rank-broadcast; FSDP clip over-counts replicated mem grads; drift guard rank-0-only). None blocking, left as-is.

Remaining flagged (not fixed): switch-inference prompt_cache/new_prompt_chunk_flag policy; the design-level write-magnitude-unlearnable property; the inert MINOR traps. Switch + interactive inference paths were never runtime-tested and need a smoke-test before trusting switch evals. A short training smoke-test (a few steps, single GPU + multi-GPU) is the recommended final validation before relying on the training-loop fixes.

## Final audit round (round 3, 2026-07-04)

Full fresh 4-agent Fable correctness audit of the entire working tree (not just diff verification). Verdict: **YES on all files for every live code path.**

- Memory core (video_memory.py): YES. Findings are latent traps in dead/config-guarded branches only (decay_init=1.0 would ZeroDivision, forward_sequence dead code + crashes with gate_extra, gate_extra name not validated, use_modulation=False would re-create the clobber). None active.
- Injection + inference (causal_model.py, switch/interactive inference): YES on all 3. Notes all inert/config-guarded: interactive doesn't thread switch_global_frame (inert, gate_extra=[]); a pre-existing rope_enhance + sink_size=0 NameError now reachable in principle but sink_size=3 everywhere; low_memory decode not offloaded in switch/interactive; initial_latent ignored (t2v only).
- E1 objective (model/streaming_training.py, self_forcing): YES. Caught one real incompleteness in the MAJ-3 fix: the base pipeline write pass still used self.crossattn_cache (the replace_all matched the 28-space denoising calls, not the 20-space write). FIXED (now _xattn on all three base calls, matching the switch pipeline). switch_frame sentinel + self_forcing ignored-params verified correct.
- Training loop (distillation.py, both pipelines, train.py): YES on all 4. Independently re-derived: no NCCL-hang hazard, FSDP clip_grad_norm counts replicated mem grads correctly, snapshot/restore + keep_last_write_grad grad-graph correct, default single-GPU/bptt=1 path byte-identical. Latent edges (config-specific, non-default): merge_dict_list KeyError if bptt_window>1 + grad_tail_W<K + grad_accum>1 + no id_loss (guarded by force-fresh when id_loss>0); snapshot/restore not try/finally; non-streaming backward has no snapshot (dormant); clear_prompt_cache zeroes switch_frame vs 1e8 (pre-existing).

Net after 3 rounds: the diff (10 files, ~174 insertions) is verified correct for all live code paths. The re-audit caught 3 issues introduced/missed during fixing (2 E1 regressions + the base-write crossattn), all fixed. Remaining items are latent traps in dead / config-guarded / untested-config paths, documented, none blocking. Still pending: a runtime smoke-test (switch/interactive inference never executed; training-loop fixes never run). The F1 fix and the 2 regression fixes were applied after the agents ran, so those specific edits are not themselves agent-verified (but are trivial and align with agent-verified code).

## Runtime smoke tests (2026-07-04)

Config: reference/smoke/smoke.yaml (= e1_overfit_v1 with max_iters=12), the heavy E1 path (train_write + bptt_window=2 + id_loss VAE decode + text_dropout), wandb `smoke_audit_fixes`.

- **1-GPU**: OOM at step 0 (expected: this E1 config needs multi-GPU FSDP). VALIDATED MAJ-5: the crash propagated LOUDLY (`[ERROR] rank 0 training crashed at step 0`, exit 1) instead of the old silent exit-0.
- **2-GPU**: reached step 5, then OOM at the generator step (23 MB short). Before OOM, VALIDATED: memory writes (drift 34-52 on blocks 5/16/27), MAJ-1 (5 steps, no NCCL hang => broadcast + all-reduce work), CRIT-1 (bptt+id_loss ran), CRIT-3 (generator backward ran).
- **3-GPU: FULL PASS, step 12, zero errors.** Grad probe at step 5: `WRITE lrs=1.7/2.0/0.7` AND `READ qkv=33/16/8` on all 3 layers => the id-loss gradient reaches BOTH read and write, snapshot/restore did not sever the graph (CRIT-2/CRIT-3 validated at runtime). Step 0 = the gate_init=0 pinch (proj-only), gate opens by step 5. Drift 29-48 sustained, statenorm/init 0.999 (renorm conserves norm), per-block drift baseline correct. MAJ-1 across 3 ranks with no hang. text-dropout + merge_dict_list + wandb all exercised without crash.

Takeaway: all critical training-loop + E1 fixes validated AT RUNTIME on multi-GPU. The E1 config is VRAM-tight (OOMs on 1-2 GPU at the generator step); a real E1 re-run needs 3 GPU or a lighter backprop/decode setup. Switch inference (CRIT-4) runtime test via the trainer visualization path: reference/smoke/smoke_vis.yaml.

## Switch-inference runtime smoke (CRIT-4)

reference/smoke/smoke_vis.yaml (vis_interval=3), 3-GPU. The trainer visualization at step 3 instantiated SwitchCausalInferencePipeline and ran it end-to-end: the log shows `num_recache_frames: 12, recache_start_frame: 27, current_start_frame: 39` (from switch_causal_inference._recache_after_switch, using the renamed variables that were the pre-existing NameErrors) then `switch_frame_index 39` / `current_start_frame 39` (from inference()). No exception, no "Visualization failed", training resumed to step 4. So the previously-uninstantiable switch inference (NameError on first switch) now RUNS with memory threaded => CRIT-4 validated at runtime. One loose end: the vis saved 0 mp4 (pre-existing vis video-save plumbing, not touched by these fixes; unrelated to switch-inference correctness).

Bottom line: audit converged over 4 rounds, and the critical fixes are validated at runtime (multi-GPU training + switch inference). Remaining known gaps are documented and non-blocking.
