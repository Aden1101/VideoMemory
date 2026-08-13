# Fable full re-brainstorm on the project (2026-07-03)

Fresh literature pass + strategic synthesis after the mechanism bet was declared dead. Source-of-truth = `fable_brainstorm_verdict.md`. This is the go-forward strategic artifact.

## 0. One-paragraph honest read

The mechanism bet is dead. What the fresh literature pass changes is the *fallback*: the two cells held open (a re-ID-after-eviction benchmark, a GRPO-on-horizon method) both had strong entrants land in the last ~8 weeks. The subfield (streaming long-video memory/consistency) is now one of the most saturated in video generation, moving monthly, and we are ~6 months + one model-scale behind the frontier. The realistic, high-EV output is **not** a competitive method or benchmark paper. It is a **diagnosis + metric-integrity paper** that converts our unusually clean negatives into a positive, explanatory contribution, with one cheap memory-agnostic method experiment held as the upgrade path.

## 1. Updated competitive landscape (read this session, post-cutoff)

- **MBench (2606.00793, Jun 2026)** -- comprehensive memory benchmark for video world models (entity/environment/causal consistency, 12 sub-dims, real long videos, rule-based + VLM scoring). The "memory benchmark" flag is planted. => standalone "streaming memory benchmark" is a RACE ENTRY, not an open cell.
- **IAMFlow / NarraStream-Bench (2605.18733, May 2026)** -- training-free identity-aware memory for multi-prompt narrative long video; targets identity drift / character duplication / attribute loss on prompt shifts; own 324-script 60s benchmark. => owns "identity across prompt transitions". Seam left: rigorous embedding-based re-ID after TRUE eviction (theirs is LLM-judged, training-free).
- **RAVEN (2605.15190, May 2026)** -- CM-GRPO on causal AR video diffusion, real-time streaming, distilled, self-rollout repack. OUR exact stack with GRPO already applied (reward = extrapolation/consistency, not identity). => "GRPO on a streaming distilled student" is DONE.
- **Identity-GRPO (2510.14256, Oct 2025)** -- GRPO w/ ArcFace identity reward, but OFFLINE multi-human, not streaming/eviction.
- **VDE-Subject** -- a subject identity-drift metric already exists (part of the drift-metric contribution preempted).
- Mechanism death confirmed: VideoSSM (DD 50.5 vs LongLive 37.5), MemFlow (2512.14699), linear-attn cross-frame memory.

Board: mechanism CLOSED; benchmark CLOSING; method/GRPO largely OCCUPIED; metric partially TAKEN.

## 2. Candidate papers, ranked by honest EV

- **P1 -- Diagnosis + metric-integrity: "Goodhart in long-video generation." RECOMMENDED, ~65-70%.** Two legs: (a) *mechanism diagnosis* we uniquely have -- a bolt-on fast-weight memory with a correct pixel-level identity objective, proven gradient reach, bounded read is STILL net-negative (E1 drift 1.1 > random subject, MEM_ZERO-attributed, LoRA-alone = baseline) + E0 (sink hurts) + train<<eval horizon; nobody published this isolation. (b) *metric integrity* (the differentiator vs MBench/VDE) -- temporal-consistency metrics (VBench-SC, DINO-drift) are gamed by stasis; we give a freeze-adversarial, motion-conditioned identity score (RAFT floor + DINO/ArcFace-on-crop, oracle-norm) that RERANKS released models. Venue: CVPR eval workshop / TMLR / NeurIPS workshop. Inference-heavy, fits 3-4wk.
- **P2 -- Standalone re-ID-after-eviction benchmark. DOWNGRADED ~25%.** MBench + NarraStream crowd it. Keep the re-ID probe as a diagnostic INSIDE P1, not a product.
- **P3 -- Method via horizon (GRPO-on-re-ID). HIGH VARIANCE ~15-20%.** Now a merge of RAVEN + Identity-GRPO = incremental; GRPO in 3-4wk = compute/tuning gamble. Lower EV than it looked.
- **P4 -- Pure negative-results. FALLBACK framing of P1, ~50%.**

## 3. Is leave-return / re-ID-after-eviction genuinely open?

Problem real (LoL + SF++ concede leave-and-return; VideoSSM = plain-long only; NarraStream/IAMFlow = prompt-transition, training-free, LLM-judged, NOT true-eviction embedding re-ID). But the benchmark cell is closing -- the defensible unit is a PROBE, not a product. Sharpest problem statement:

> Subject re-identification after TRUE context eviction in real-time distilled streaming generation: subject present in A, absent for a span strictly longer than (local window + sink) during B, prompted back in A'; scored by ArcFace(face)/DINO(general) on SAM-cropped subjects, oracle-normalized to an intra-A ceiling, gated by a RAFT motion floor so a freeze can't win.

Defensible seams vs the crowd: (a) freeze-adversarial gate (our gate01/overfit ckpts are the adversary the metric must not rank above baselines), (b) true-eviction control (B > window+sink), (c) oracle normalization. Ship as a SECTION of P1.

## 4. The GRPO shot -- and a better memory-agnostic attack

GRPO-on-re-ID is mostly a mirage for a 3-4wk novelty win (RAVEN + Identity-GRPO cover both halves; breaks on reward hacking -> freeze, 480-frame credit assignment, tuning cost). The better, cheaper, more novel-shaped attack on the actual binding constraint (train-horizon << eval-horizon, BPTT OOMs) is a **supervised drift-consistency objective, no BPTT, no memory**:
- Roll the student under `no_grad` to a distant frame N (cheap: no graph/checkpoint -- RAVEN's repack minus RL).
- Cache a clean identity target from an early chunk (crop -> ArcFace/DINO), detached.
- Generate ONE grad chunk at frame N, pull its subject identity toward the early target through the frozen decoder.
- Trains the BACKBONE LoRA (memory-agnostic) at a LONG horizon we otherwise can't reach; sidesteps BPTT via the stop-grad long rollout.

Worth exactly one shot: E1 mem-ZERO showed a SHORT-range id loss on the backbone did nothing; this tests a LONG-range anchor. Win => simple novel positive ("long-range identity anchoring via truncated self-rollout, no BPTT, no memory"), upgrades P1 to a method paper. Loss => proves compounding drift not anchor-fixable at any horizon, strengthens the diagnosis. Check RAVEN's supervised baselines first (looks unclaimed -- they went straight to CM-GRPO).

## 5. What we might still be WRONG about / reframes

- We may be over-invested in "identity/memory" when the real object is **compounding error under distilled self-rollout** (E0: drift is dynamical, not referential). Reframe: our whole body of work is evidence that *the streaming-distillation objective itself is the drift source*, and memory is a category error as the fix. This turns every negative into a coherent thesis (P1).
- DINO-cos-to-frame-0 conflates identity with pose/motion -- a weakness of our training target AND a finding for the metric-integrity paper (SC metrics inherit the confound).
- Strengthening reframe: position freeze/Goodhart as a GENERAL evaluation-integrity claim -- "temporal-consistency metrics across long-video are satisfiable by motion suppression; we quantify and correct the exploit." Bigger tent than "our memory failed."

## 6. Contrarian / left-field

- **Evaluation-integrity as the headline** (not diagnosis-as-headline): a "Goodhart's law in long-video generation" position+evidence paper re-scoring the released streaming zoo under a freeze-adversarial metric. Cheap, contrarian, citable, inference-only. Possibly HIGHER EV than the mechanism diagnosis (generalizes beyond our stack).
- **Distillation-induced drift probe:** teacher-forced vs self-rollout drift on the same student -- isolates how much drift the causal distillation introduces vs inherits. Reframes the "add memory" reflex as symptom-treating.
- **Repurpose the instrumentation as a detector:** invert the identity-drift + freeze + off-manifold pipeline into an AI-video temporal-artifact / identity-drift detector (forensics community). Uncertain EV, real asset.
- **Wind-down option (honest):** if the goal is a strong main-conference METHOD paper, highest-EV might be to stop competing in this saturated subfield and redeploy the eval/ablation harness to a less-crowded question. On the table, not the recommendation.

## 7. The bet -- single highest-EV path, 3-4 weeks

**Bet: P1 (diagnosis + metric-integrity), with the section-4 long-range anchoring experiment as the upgrade path, re-ID probe folded in as a diagnostic.** Only option that (a) uses assets we have, (b) is robust to crowding (explanatory + measurement, not a scoreboard), (c) fits compute/time, (d) has a cheap method-upgrade path.

Ordered plan:
1. **Days 1-3 (inference-only, decides the shape):** (a) leave-return A-B-A re-ID probe on BASELINES (stock AF / LongLive / Self-Forcing; VideoSSM if ckpt public) -- is the problem real, how bad. (b) freeze-adversarial RERANK: score the released streaming zoo on VBench-SC vs our motion-conditioned identity metric -- does the ranking CHANGE? (this rerank is the paper's spine; null => pivot to P4). (c) o_hat/state-recall probe to nail the "state never stored" sentence.
2. **Days 4-10 (the one positive shot):** the section-4 long-range drift-consistency run (no BPTT, no memory, backbone LoRA). One config, kill fast if drift >= baseline.
3. **Days 8-14 (parallel):** lock the 3 diagnosis figures (E1 corrupts + MEM_ZERO; E0 sink-hurts; freeze/Goodhart) + the corrected-metric definition + validation gates (>3sigma freeze-adversary separation, Spearman vs human).
4. **Days 14-24:** write. Target CVPR-2027 eval workshop / TMLR; if the shot hit, aim main-track method framing.

**Single cheapest experiment that most changes the picture:** the freeze-adversarial RERANK of released streaming models (step 1b), ~1-2 days, pure inference. Ranking changes under motion-gating => citable evaluation-integrity result, P1 real. Null => weak metric-integrity leg, fall back to P4, lower the ceiling. That one result tells you whether there's a POSITIVE paper here or only a negative one.

## Bottom line

Stop treating this as a memory project -- the memory story is over and the adjacent method/benchmark cells filled in while we were heads-down. The genuine defensible asset built is a clean, instrumented ability to *show the field is measuring the wrong thing and that the obvious fixes backfire*. Point that at an evaluation-integrity + diagnosis paper, spend 1-2 days on the rerank to confirm there's a positive result, take exactly one no-BPTT anchoring shot for a possible method upgrade, and don't spend another GPU-week on the fast-weight.
