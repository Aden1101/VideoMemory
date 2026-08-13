# Runs archive -- what each meaningful run taught (2026-07-02)

Curated by Opus. wandb project = `anchor-forcing-memory`. Local wandb logs + checkpoints live on the
project disk (`/shared/ssd_14T_project/adenelmi/<exp>/`), even for runs removed from the wandb cloud.
Nothing is lost -- "removed from cloud" != "deleted from disk"; any run is restorable with `wandb sync <dir>`.

Note: GPUs were BLOCKED 22-25 June (khoutaibi), so there is NO run in that window -- the "first real runs"
are 18-21 June (read-only), then the write-training milestones start 21 + 26-27 June.

---

## KEEP on wandb cloud -- the milestones (9 runs)

Each carries a real, documented finding across the project arc: read-only -> write -> decay -> unfreeze ->
muon -> cross-chunk -> gate-opens -> overfit.

| run | date | what it was | what we learned | findings doc |
|---|---|---|---|---|
| **e1_ro_5L_metrics** | 19 Jun | first substantive read-only 5-layer memory training | read-only fits on 1 GPU; the read injection is **uniformly toxic** (degrades vs baseline) -- seeded the whole "why ours fails" thread | `why_ours_fails_vs_papers.md`, `vram_recap.md` |
| **e2_write_phase2** | 21 Jun | **FIRST write-training** (Phase-2, write lifted to enable_grad) | write trains within-chunk; found+fixed the **C3 blocker** (TTT state grad carried across chunks) and the **VRAM leak** | `write_path_research_report.md`, `leak_investigation.md` |
| **e2_write3L** | 27 Jun | 3-layer write-training (the real write eval) | eval **WORSE than baseline at ALL lengths**, worst at the training horizon -> NOT horizon-limited; bottleneck = the **objective**, not the gradient path | `why_ours_fails_vs_papers.md` |
| **e3_run1_decay** | 29 Jun | decay / gated-delta cap-to-init experiment | the scalar decay is **our graft** (not in LaCT/Scal3R/TTT-Video); scale mismatch -> state collapse, cap never binds | `muon_decay_review.md` |
| **e3_run2_full** | 29-30 Jun | **FULL LoRA unfreeze** | full-unfreeze fits on clean GPUs (~20-22GB); the OOM scare was GPU **contention**, not the unfreeze cost; still needs text-dropout for the gate to open | `muon_decay_review.md` |
| **e3_run3_muon2** | 30 Jun | restore muon=2 + decay | muon value is **irrelevant while the gate ~0**; the bottleneck is the objective, not muon | `muon_decay_review.md` |
| **e3_realk2** | 30 Jun-1 Jul | real K=2 cross-chunk (write trained cross-chunk) | it RUNS (config-only fix: slice_last_frames=27, no eviction); DMD-pure; predicted not to beat baseline | `crosschunk_consensus.md` |
| **e3_real3_gate01_scal3r** | 1 Jul | gate_init=0.01 + Scal3R revert (decay OFF) | **the gate OPENS** data-dependently, memory contributes ~0.5 (not the do-no-harm story). Source of the FREEZE finding + the eval videos | `fable_brainstorm_verdict.md` (Sec 8) |
| **e3_overfit14** | 1 Jul | overfit: 7 no-switch + 7 single-switch prompts | collapses to **garbage from frame 0**; VBench subject_consistency is **blind** (0.94 on a collapsed video). The visual/metric-blindness diagnosis | `fable_brainstorm_verdict.md` (Sec 8) |

---

## Ranked by importance (impact on the CURRENT conclusion, not chronology)

**Tier S -- the runs the current strategy rests on:**
1. **e3_real3_gate01_scal3r** (1 Jul) -- the pivot's anchor: the gate opens + the robust FREEZE/Goodhart finding + all the eval videos. Everything downstream (E0 ruler, E1 objective fix) exists because of this run.
2. **e2_write3L** (27 Jun) -- the decisive negative: write-training worse than baseline at ALL lengths -> bottleneck is the OBJECTIVE, not the gradient/horizon. Redirected the whole program (killed cross-chunk/horizon as the fix).
3. **e3_overfit14** (1 Jul) -- proved VBench is BLIND (0.94 on a collapsed video) + the collapse-from-frame-0. This is why we now build a DINO identity ruler instead of trusting VBench.

**Tier A -- foundational facts:**
4. **e1_ro_5L_metrics** (19 Jun) -- the origin: first memory training, "read is uniformly toxic." Seeded the entire "why ours fails" thread (conclusion later refined).
5. **e2_write_phase2** (21 Jun) -- the enabler: first write-training, unblocked the write path (C3 fix + leak fix) that every later write run depends on.

**Tier B -- design-space narrowing (supporting):**
6. **e3_run2_full** (29-30 Jun) -- full LoRA unfreeze fits; OOM was contention not cost.
7. **e3_realk2** (30 Jun-1 Jul) -- refuted "cross-chunk infeasible" for K=2 (but predicted not to help).
8. **e3_run1_decay** (29 Jun) -- decay is our graft + scale mismatch -> collapse.
9. **e3_run3_muon2** (30 Jun) -- muon value irrelevant while the gate ~0.

---

## LEAVE OFF cloud -- infra / probes / tests / aborted (recommend NOT restoring)

These are already off the cloud; keep them off (they add no finding and would clutter). Data stays on disk.

- **GPU holders / fillers** (conserve GPUs, no experiment): `ro_fill_g1/g2/g7`, `hold_ro_gpu7`, `e3_readonly_hold`,
  `membd_readonly`, `_isolate_nogpu7`, `_isolate2_nogpu7`. (Note ro_fill_* are 29-31MB but are HOLDERS, not runs.)
- **Write-VRAM feasibility sweep** (26-27 Jun): `read_1L_3gpu`, `write_1L/2L_*` (many variants),
  `diag_write_2gpu_1L`, `e2_writegate_1L/2L/3L`, `e2_probe_1L/2L`. The finding (read+write fits on 2x48GB,
  max window 24 latent) is already captured in `vram_recap.md` -> no need for the individual ladder runs.
- **Tests / aborted / superseded**: `_sweep_3L/4L/5L`, `_xchunk_1L_K2*`, `e3_bptt2_test`, `e3_k3_test`,
  `e3_iso_ml57`, `e3_wdmd_k4/nockpt/squeeze`, `_chunk_27`, `_reorder_test`, `e2_write1L_leakfix*`,
  `e2_write2L*`, `e2_ctrl_nowrite`, `e2_write_nobd`, `e3_run1b_decay`, `e3_run2_unfreeze5`,
  `e1_ro_2L_1gpu_postfix`, `e1_ro_5L_postfix`, `e1_ro_5L_v2`, `membd_2L_full`, `e3_ro_long120`, `e3_overfit10`.
- **Optional (borderline, left off)**: `e1_ro_5L_combo652` (37MB) + `e1_ro_5L_agreement` (19MB) = read-only
  gating combos (idea-6 agreement, combo 652). Superseded by the gate01 work; restore only if we want the
  gating-exploration history on record.

## wandb cloud status (2026-07-02)
Runs deleted from the cloud leave a **tombstone**: their id cannot be re-created (`wandb sync` -> HTTP 409).
So re-upload requires a NEW id (`wandb sync --id <new>`). Decision after review:
- **Restored under new ids** (historical value): `e1_ro_5L_metrics` -> `e1_ro_5L_metrics_restored`,
  `e2_write_phase2` -> `e2_write_phase2_restored`.
- **NOT restored** (Tier B, subsumed): `e3_run1_decay`, `e3_run2_full`, `e3_run3_muon2` -- their lesson
  ("muon/decay don't matter while the gate is starved; the bottleneck is the objective") is already stated
  better by `e2_write3L` + `e3_real3_gate01_scal3r` and fully written in `muon_decay_review.md`. Documented
  here, curves not worth re-uploading.
- **Already in cloud** (kept): `e2_write3L`, `e3_realk2`, `e3_real3_gate01_scal3r`, `e3_overfit14`.
Final cloud set = 6 runs (4 kept + 2 restored).

## Recommendation on removal
- The 9 above are the right cloud set (arc-complete, each with a finding). Nothing among the 4 already in
  cloud should be removed -- all carry findings.
- Do NOT restore the ~50 infra/probe/test runs. They can even be deleted from disk if space is ever needed
  (total ~0.3GB, not worth it now -- keep as a disk archive).
