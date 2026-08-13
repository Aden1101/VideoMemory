# Anchor-Forcing MAG memory — Research Status (living doc)

_Last updated: 2026-07-07. **Read this first when you come back.** Terse by design; the full detail is in `reference/fable_brainstorm_verdict.md` (append-only source of truth)._

## Goal — HARDENED ACCEPTANCE CRITERION (user, 2026-07-07, non-negotiable, BEFORE any re-ID pivot)
The memory must **IMPROVE the baseline by +1-2% on PLAIN drift, clean** — concretely mem-ON late-drift ≤ mem-OFF − 0.01..0.02 on the MOTION-VALID prompts (not a freeze artifact), zero visual artifact. This is stronger than the earlier criterion: NOT do-no-harm (0% = mem-DC's +0.004), a REAL positive gain. Requires a term that REWARDS beating mem-OFF on identity — the hard part, because a naive "lower drift" reward is gamed by freeze. Pre-registered: if the corrected run can't hit +1% clean motion-valid → plain drift is EMPIRICALLY CLOSED, pivot. (Superseded criterion below kept for reference.)
1. Visually CLEAN — zero artifact. 2. do-no-harm OR small improvement (NOW: must be a real +1-2% improvement). 3. Mechanism does something VISIBLE (not inert). Runs so far each fail one: gate=0/mem-DC → fails the improvement; e2_amp/streamgate active → artifact (dissolution/freeze).

**No run has hit all 3.** Each fails one: gate=0 → fails #3; mem-DC (DC removed) → clean+do-no-harm but effectively inert (#3); mem-ON DC / e2_amp rel_norm → active but artifact (#1). The needle: an ACTIVE injection (visible effect) that is STILL clean AND not-worse. Implication: the injection must be CONTENT-ALIGNED (identity-correlated), not generic — a strong generic signal → artifact, a weak one → inert. That likely needs a mechanism where injection is inherently identity-aligned, or an aux objective that trains the read to encode useful identity (DMD-only doesn't reward it).

## TL;DR (where we are)
- The old "memory is TOXIC" story was a **DC-injection artifact**, not the mechanism. The memory's raw projection was ~90% a token-constant bias vector that darkened the video + printed a cross-hatch. Proven by a 1-line DC-removal probe.
- With the DC artifact removed, the memory content is **INERT / do-no-harm** on plain drift (mem-DC 0.135 ≈ mem-OFF 0.132), NOT toxic. Matches Fable's original prediction.
- The **amplitude fix** (run `e2_amp`: `rel_norm` + `dc_remove` + long horizon bptt4) at step 250 **DISSOLVES** the video at long inference (mem-ON DINO 0.75 vs mem-OFF 0.27, motion collapses) — worse, not better. Plateau (step 1000) re-eval pending.
- **Why plain drift resists** (structural, not bugs): (1) supervised horizon sits inside the KV window → KV+sink already carries identity → no gradient reason for the memory to help; (2) the Muon+renorm substrate has no usable relevance-magnitude signal (Muon erases write magnitude; every amplitude workaround either flattens → inert, or unbounds → collapse/dissolve); (3) **VBench is blind** — it scores a dissolving video at 0.95, so "improve VBench" is a trap (do-no-harm already reads ~0.95).
- The **one untested task** where the memory could beat KV+sink: **re-ID / revisit** (subject A leaves / is evicted from the KV window, content B fills it, A returns — KV+sink already evicted A; a fast-weight memory could carry A). Positive control shows the substrate CAN store+retrieve past a 13-write gap.

## Currently running
- **Noise-floor measurement** (free, inference-only, GPUs 0/1/3/4/5): mem-OFF + mem-DC at seeds 43/44/45 (seed 42 reused) → the seed-vs-seed DINO error bar (DEFINES do-no-harm for the first time — all prior evals were single-seed 42, so "active"/"within noise" were UNDEFINED) + is the mem-DC on-vs-off effect ABOVE the seed floor (real active) or WITHIN it (hollow = a re-seed = INERT). BOTH Fable agents (2026-07-07) converge on this as the mandatory next step; the e3 loss agent made it pre-launch gate #2 (act_floor calibration vs the mem-DC inert reference). Scripts: reference/runs/e1_rerun/noise_floor/.
- **`e3_antifreeze` — IMPLEMENTED (not launched), behind flags.** Retargeted to the RELAXED criterion (clean+active+do-no-harm+dynamic, +1-2% is bonus/off). "Active without a reward" = an AC-DOSE FLOOR (inert costs w·0.05 so the gate must open to change latents measurably; a relu floor, not a maximizer = anti-e2_amp). Deep supervision at the freeze horizon (randomized-K 4..14, W=1 at proven VRAM), id_loss REMOVED (freeze-ally), counterfactual do-no-harm + motion-match to a fixed external stock-AF curve. Odds: clean active-not-inert win ~25-35%, new (mild) degeneracy ~30-35% modal, inert ~15-20%. GATED on the noise-floor measurement (calibrate act_floor + twin selftest). Artifacts: reference/runs/e3_antifreeze/. STOP rule: plateau inert or any degeneracy → plain drift CLOSED (iteration 7), pivot E-REID/autopsy.
- (streamgate STOPPED at step 1391 — froze, self-inflicted; GPUs freed.)

### KEY (both Fable agents, 2026-07-07): "active + do-no-harm on plain drift" is likely hollow-or-contradiction; INERT is the ceiling; and we CANNOT SCORE it without the noise floor (all evals single-seed). The relaxed criterion may ALREADY be met (hollowly) by mem-DC. The measurement decides: already-met(hollow/real) vs strictly-inert(closed). Adversarial's info argument: with global_sink the frame-0 anchor is retained forever + middle frames redundant with anchor+window+text → the memory's conditional info ~0 → any injection is a perturbation → invisible/reseed/bias(degenerate). Attractor structure, not bad luck.
- Diagnostics tb channels: `ampdisp` (per-token coupling, >0.2), `dcfrac` (raw proj DC), `gatemean` (real tanh gate), `dbg_inj_stream` (injection-to-stream ratio, want 0.05-0.3 bounded).

### STREAMGATE step-1000 RESULT (2026-07-07) — FREEZE degeneracy (Goodhart #2)
mem-ON is CLEAN (no dissolution, envelope-in-training works) BUT the memory converged to FREEZE: mem-ON motion 0.9 vs mem-OFF 7.1 (8x less), 3/8 prompts (ballet/boy/elder) flagged FROZEN. DINO drift mem-ON 0.073 < mem-OFF 0.086 (-0.013) LOOKS like do-no-harm+ but is a FREEZE ARTIFACT (a near-static video trivially has low drift; the freeze-motion gate catches it). β-sweep: mem-ON(β1.0) 0.073 vs mem-β0.5 0.059 (delta +0.014, injection causally active). VERDICT on the 3 criteria: #1 clean = clean-BY-FREEZING (degenerate); #2 do-no-harm = FAIL (video degraded/frozen, motion 8x lower); #3 active = yes but WRONG direction (freezes). This is the 2nd degeneracy (dissolution→freeze), both Goodhart — the envelope closed dissolution, the optimizer found freeze; the existing tv_loss 0.5 / tv_motion_floor 0.3 anti-freeze did NOT prevent it. The freeze-motion gate WORKED (without it, the drift number would have falsely read as a win). Artifacts: reference/runs/e2_amp2_streamgate/eval1000/.

### ENVELOPE PROBE RESULT (free, 2026-07-07) — leg #1 VALIDATED
`MEM_STREAM_SCALE=1.0` on the e2_amp step-250 ckpt (no retrain): **STOPS the dissolution** — mem-ENV video is clean (visual: ballet/skate coherent, match mem-OFF). Drift: mem-ON 0.75 → mem-ENV 0.55 → mem-OFF 0.27 (envelope recovers half the gap). BUT mem-ENV still WORSE than off (+0.28) + motion suppressed (2.6 vs 5.5) — EXPECTED, since this ckpt was trained WITHOUT the envelope/force-identity, so the bounded injection is still un-helpful content. => envelope = the confirmed #1 fix (no more artifact); the retrain (streamgate) must train the memory to inject USEFUL content within the bound for #2/#3.

## THE plan for the user's #1+#2+#3 criterion (Opus drift-fix agent)
The dissolution = fixed per-chunk injection magnitude × OOD write-count (train ≤4 writes, infer 20-80). The competitor is the 3-frame SINK (carries chunk-0 identity free) → the memory's only headroom is the RESIDUAL drift; so the dominant risk is clean-but-INERT (#3), not dissolution.
- **Fix = `e2_amp2_streamgate`**: keep e2_amp content (rel_norm+dc_remove+bounded gate, already active) + **(a) stream-RMS envelope** (causal_model.py:608, `mem_out *= β·stream_rms`, env MEM_STREAM_SCALE / config mem_stream_scale) so active injection can't accumulate → clean #1 + not-worse #2; + **(c1) deep supervision** (bptt_window 16, grad_tail_W 4 = same VRAM/backward cost, grad window past the 39-frame KV window) + text_dropout 0.3 + id_loss 2.0 → force the MEMORY (not sink/text) to carry residual identity → active #3.
- **Escalation if clean-but-inert (~35%, code-ready before launch): (e) counterfactual identity-on-read loss** — decode tail-k with vs without the injection, penalize when REMOVING the memory helps identity → makes "active" and "helpful" the same objective; then (ii) training-time sink dropout.
- **Odds (honest): full win ~20-30%, clean-inert ~35%, still-dissolves ~10%.** Verify all 3: clean=visual+dbg_inj_stream bounded; not-worse=DINO mem-ON vs mem-OFF; active=β-sweep (MEM_STREAM_SCALE 0/0.5/1.0, drift must differ measurably & not-worse) + gatemean>0.05.
- Free zero-GPU probes gate the spend: (1) envelope-as-inference-fix (RUNNING); (2) periodic state-reset every 4 writes (proves OOD-write-count if it fixes dissolution).

## Iteration log (each closed a failure mode = progress)
| Run | Tried | Result / what it eliminated |
|-----|-------|-----------------------------|
| 01 readonly-mem | read-only memory | "read toxic" — later understood as the DC artifact |
| 03 write-3L | train the write, 3 layers | worse than baseline |
| 04 crosschunk-K2 | cross-chunk BPTT | runs, no gain |
| 05 gate01-scal3r | gate_init 0.01 + Scal3R revert | "gate opens" was a WEIGHT-norm misread; not the gate |
| 06 overfit14 | overfit sanity | collapses; **VBench is blind** (key methodology finding) |
| 07 objective-fixed | id-loss objective | "still toxic" = still the DC artifact |
| e1_rerun2 | clean re-run, all code bugs fixed | mem "hurts" 8/8 → **revealed as DC artifact**; content INERT |
| fable_darkprobe | instrument the ckpt | measured dc_frac 0.88-0.99 → the DC diagnosis |
| **e2_amp** | amplitude fix (rel_norm+dc_remove+bptt4) | step-250: **DISSOLVES** at long horizon, worse; plateau pending |

## Open questions / next candidates
- Does e2_amp recover by the plateau (step 1000+), or keep dissolving? (watcher running)
- Amplitude fix v2 that avoids the fixed-cross-chunk-magnitude dissolution: **scale injection by the local stream RMS** (couples to the stream, bounded) instead of the chunk-scalar read norm; or **cap/anneal** the injection; or match inference horizon to training.
- The re-ID **free inference probe** (A→B→A, eviction gap curve, instance-vs-class DINO metric) on stock AF + a checkpoint — gates whether ANY memory retrain is worth it. All 3 agents recommended this first.
- Fundamental: is the substrate wrong (Muon+renorm → no relevance signal)? gated-delta = VideoSSM (works but novelty 0).

## Strategic fork (the real decision)
- **A. Keep grinding plain drift** — low odds (structural no-job), but each iteration eliminates a failure mode (real progress). Next: rel_norm→stream-RMS variant.
- **B. Re-ID pivot** — 0 GPU-train probe first; the one task with a possible job.
- **C. Write the negative-result autopsy** — "DC artifact fixed → under DMD self-rollout the fast-weight memory has no job on plain long video because KV+sink suffices" is a clean, real contribution.

## KEY INSIGHT — the Goodhart trap (2026-07-07, Opus agent)
The training objective (DMD, chunk-local temporal self-similarity) AND the standard eval metrics (VBench-SC, naive DINO-drift) reward the **same wrong thing: temporal self-similarity**. So a learnable memory converges to a **freeze / DC-injection degeneracy** the dashboard scores as *healthy* (VBench 0.94 on a dissolving clip; the DC-bias artifact read as "hurts 8/8"). "Consistent because frozen." => This is WHY the acceptance criterion's #1 (clean) and #3 (active) fight: any active injection gets pushed by the optimizer toward the freeze/DC degeneracy that scores well. To hit #1+#2+#3 the injection must be CONTENT-ALIGNED and the eval must not reward freeze (motion-gated). This Goodhart finding is demonstrated, un-preempted, and is likely the project's real contribution (turns every negative into an asset).

## Hard truth (Opus completeness critic)
No version of "our fast-weight TTT memory beats KV+sink" is both NOVEL and WINNABLE: every fix that would work (reconstruction/history-loss = MemGen; gated-delta = VideoSSM) is already published (novelty≈0). The honest paper: spine = the audited clean negative (+ "3 confounds nearly shipped a false narrative" methods lesson), headline = the Goodhart/metric-integrity finding, residual = the re-ID measurement decomposed by the sink. The one sanctioned METHOD shot (only after E-REID, memory-AGNOSTIC): long-range identity anchoring via no-grad rollout on the LoRA (attacks train-horizon-21 ≪ eval-horizon-480, no BPTT, reuse E1 DINO-identity-loss).

## Re-ID probe (E-REID) — GO, free, parallel, decisive
Inference-only, 0 training GPU (idle GPUs 0/2/7), runs alongside e2_amp. PRIMARY subject = stock AF baseline (`ckpt/final.pt`) — answers "does re-ID-after-eviction even exist as a problem". Design: A→B→A′ (B = different-subject-same-class, span > local_attn12+sink3), VERIFIED eviction (global_sink=false first cut, or late-entry construction + anchor-in-sink control), metric = instance-vs-class (DINO cos of A′-return vs A-original, normalized by class floor = same-prompt-diff-seed, cross-subject floor, intra-A oracle), MANDATORY freeze-adversary motion gate. Arms: stock AF (primary) / mem-ON / mem-OFF / mem-DC (secondary canaries — mem-ON unlikely to win, query-retrieval path never trained). Code (~1 day, inference-side): (1) two-switch A→B→A′ in pipeline/switch_causal_inference.py (~40 lines), (2) switch runner script (~120), (3) re-ID scorer extending drift_metric.py (~100), (4) metric validation (never-leaves vs substituted >3σ, freeze adversaries must rank below stock). Pre-registered: baselines PASS→no job, autopsy; FAIL→real problem + one honest retrain; MID (bet: 0.4-0.7, text-reliance)→"text masquerades as memory".

## Ops notes
- Agents: use **Opus 4.8** (`model: "opus"`) — Fable being deprecated.
- GPUs: `CUDA_DEVICE_ORDER=PCI_BUS_ID`; smi 6 is BROKEN (exclude); training on 1,3,4,5.
- Smoke-test gotcha: a sibling repo `infinity-rope/utils/wan_wrapper.py` shadows the real one unless `PYTHONPATH=/shared/ssd_14T/home/adenelmi/Anchor-Forcing` is set (training via torchrun is fine).
- ALWAYS wandb (project `anchor-forcing-memory`). All generated files in `reference/` only.

## 2026-07-08 — Baseline audit + the 3-way motion finding + e3 launched
**Aden doubted the mem-OFF baseline. 3-agent audit (static / upstream / empirical) verdict:**
- CODE is CLEAN: base commit 392b3f8 byte-identical to upstream vivoCameraResearch/Anchor-Forcing@25bdfba (full tree); ckpts base.pt/final.pt sha256-match HF young98/AnchorForcing; mem-OFF code path == AF code path with ZERO bias (empirical: memOFF-vs-stockAF MSE 11.5/DINO .997 == stockAF-vs-ITSELF 11.7 = bf16/flex-attn nondeterminism; RNG contamination refuted). No code bug corrupts any comparison.
- BUT mem-OFF WEIGHTS = our TRAINED checkpoint (memory zeroed), NOT stock AF. mem-ON vs mem-OFF = valid injection ablation (freeze/dissolve/inert HOLD); it is NOT "vs Anchor-Forcing". Real AF = `memory_layers:[]`+`ckpt/final.pt` (stockaf.yaml). Footguns: []-fallback CRASHES on trained-with-mem ckpt; wrapper default runs random mem if config omits memory_layers+MEM_ZERO_ALL; 480f evals = OOD (AF max 240).

**NEW standing rule (Aden): ALWAYS compare 3 arms — mem-ON | mem-OFF | AF stock.** Stock-AF videos (8 e0_dynamic prompts, 240f, seed42) are reusable as the reference column. Deliverables: reference/vbcheck_viewer.html, reference/baseline_3way_viewer.html.

**THE 3-way finding (streamgate + e1_rerun2, VBench Dynamic Degree):**
| DD (motion) | mem-ON | mem-OFF | AF stock |
| streamgate | 0.054 | 0.488 | 1.000 |
| e1_rerun2  | 0.675 | 0.567 | 1.000 |
Stock AF is 100% dynamic; its SC 0.936 is the healthy price of motion. **Our TRAINING halves AF's motion BEFORE the memory injects** (mem-OFF DD ~0.49-0.57 vs 1.0), in BOTH runs → GENERAL, not run-specific. The VBench SC/BC "wins" of mem-ON over mem-OFF were a freeze mirage (frozen = trivially consistent). Real bottleneck = the DMD-self-rollout+LoRA TRAINING degrades dynamism, not (only) the memory. Framing = a motion/consistency tradeoff; target = top-right (AF-level motion + higher SC). e1 mem-ON DD 0.675>mem-OFF likely the DC-artifact flicker counted as flow, not healthy motion.

**mem_stream_scale plumbing BUG fixed:** e3 config put `mem_stream_scale:1.0` in model_kwargs but it was never threaded through wan_wrapper→CausalWanModel→block (only the MEM_STREAM_SCALE env worked). Would have crashed launch. Wired properly through all 3 levels (block/model/wrapper __init__ + block attr). Caught by the mandatory twin-selftest gate.

**e3_antifreeze LAUNCHED (2026-07-08)** on GPU 1,3,4,5, wandb run `e3_antifreeze`, from stock AF (final.pt)+fresh memory. All gates PASS: motion_targets(9 keys) ✓; twin selftest max|real-twin(memON)|=0.000e+00 exactly ✓; grad probe step0 read qkv/proj + write lrs all nonzero ✓ (b5/16/27). Objective: w_dnh=10, w_act=1(floor 0.05, calibrate in-run), w_rel=1, w_ext=1(motion match to STOCK-AF curve). ~142s/step → step250 eval ≈10h, full 2000 ≈3.3d. Bar for e3: lift DD ~0.5→~1.0 without breaking identity. First eval @250 goes through the 3-way + Aden's eye.
