# Independent assessment: is the learnable-memory bet still alive? (2026-07-02)

## 0. Context for an outside reader (skip if you know the project)

**The project.** We work on streaming, real-time, autoregressive video diffusion: a
Wan2.1-T2V-1.3B student distilled into a 4-step causal generator via Distribution Matching
Distillation (DMD) with self-rollout (the student trains on its own generated frames) and a
rolling KV cache. Teacher: Wan2.1-T2V-14B (frozen, bidirectional, memoryless). Our repo is a
fork of Anchor-Forcing, itself built on LongLive, itself built on Self-Forcing. Goal: fix
long-horizon subject/identity drift, especially when a subject leaves the KV window and must
return with the same identity (prompt switches, revisits).

**Our method (internal codename "MAG", to be renamed).** A learnable fast-weight /
test-time-training memory module (`VideoMemoryLayer`, `wan/modules/video_memory.py`, adapted
from the 3D-reconstruction work Scal3R) grafted at 1-3 transformer blocks. Per generator
call it READS the fast-weight state through a SwiGLU apply and injects the result into the
residual stream through a gated AdaLN output; it WRITES via an inner-loop gradient step with
Muon/Newton-Schulz orthogonalization and a weight-norm renorm. Video chunks are 21 latent
frames processed as 7 sub-blocks of 3; the KV attention window is 12 frames + 3 sink frames;
backprop is within-chunk only (state detached at chunk boundaries).

**Run names used below.**
- `e2_write3L`: the main trained-write run (3 memory blocks [5,16,27], gate_init 0.1,
  write trained within-chunk, checkpoint 1200), fully evaluated on VBench at 6/60/120s.
- `e3_real3_gate01_scal3r` (short: "gate01"): Scal3R-faithful run (gate_init 0.01, decay OFF,
  norm-conserving renorm, Muon 5 steps), 795+ steps, checkpoints 450/600/750, NEVER evaluated.
- "The gate": the AdaLN output gate of the memory read; zero or near-zero init is the
  do-no-harm design. `gate(first)` in logs is the pinned bias; `gatew_b*` in tensorboard is
  the data-dependent gate norm per block.

**Where to resume from.** Code entry points: `configs/ttt_memory.yaml` (WARNING: the committed
config differs from what the real runs used; real runs used /tmp configs with 3 memory layers
and train_write on), `wan/modules/video_memory.py` (module), `wan/modules/causal_model.py`
(integration), `model/streaming_training.py` + `pipeline/streaming_switch_training.py`
(rollout/write path), `trainer/distillation.py` (DMD loss). Key internal docs, all in
`reference/`: `why_ours_fails_vs_papers.md` (failure analysis vs published TTT works),
`ultra_memory_program.md` and `final_research_plan.md` (prior plans),
`crosschunk_consensus.md` + `chunk_bptt_investigation.md` (why cross-chunk BPTT is
infeasible), `muon_decay_review.md` (update-rule review), `kvcache_landscape_map.md`
(metric landscape). Tensorboard events for gate01:
`/shared/ssd_14T_project/adenelmi/e3_real3_gate01_scal3r/`.

---

Method: 4 independent evidence tracks. (1) Full digest of internal `reference/` docs; (2) exact
mechanism spec extracted from the working-tree code; (3) adversarial preemption sweep of
Dec 2025 to Jun 2026 (~20 searches, 15 full fetches); (4) deep-research workflow: 20 sources,
100 extracted claims, 3-vote adversarial verification: **13 confirmed 3-0, 0 refuted**, 15
initially unverified because of session quota. A second pass on 2026-07-02 re-verified all 15
against the primary sources (raw HTML fetched, quotes checked verbatim); several wordings in
this doc were tightened as a result, see section 7 for the full outcome table. Plus one
direct log check: tensorboard of `e3_real3_gate01_scal3r`.

---

## 1. Verdict

**The mechanism bet is dead. The problem bet is alive, narrow, and closing fast. Pivot the
contribution from "our TTT module" to "the re-identification problem, its benchmark, and the
minimal recipe that solves it." Conditional GO, gated on one cheap inference-only probe (E0).**

Three independent kill shots on the original bet (multi-step Muon TTT fast-weight, frozen
backbone, DMD-only objective, judged on VBench):

1. **Our own data.** e2_write3L is worse than baseline at every length and worst AT the training
   horizon (-0.163 subject consistency @6s, gap shrinking with length). The gate drifted 0.1 to
   1.26 under DMD while identity collapsed: DMD actively rewards a read that hurts identity.
2. **The field converged elsewhere.** Every trained memory in this lineage that works is a
   closed-form or attention-based update, co-trained with the backbone: VideoSSM = gated
   delta rule, full fine-tune, pure DMD e2e (confirmed 3-0); ARL2 = gated DeltaNet state, stage-1
   dense per-layer alignment loss + LoRA co-train; Echo-Infinity = query tokens updated by
   attention+gating on eviction, end-to-end; MemGen = KV-compression transformer trained
   OFFLINE with a reconstruction loss, then frozen (confirmed 3-0). Nobody runs a multi-step
   gradient inner loop, and nobody trains a memory with DMD alone on a frozen backbone.
3. **The unclaimed cell is unclaimed for a reason.** "Multi-step TTT write trained through
   self-rollout distillation" remains unpublished (verified by explicit search), but the most
   plausible explanation is that it is a negative result: harder to train, with no demonstrated
   payoff over the 1-step closed-form update anywhere in the literature.

Why the problem bet is alive, now with primary-source confirmation:

- **LoL, the 12-hour RoPE-jitter paper (2601.16914), concedes it verbatim** [VERIFIED against
  the raw HTML, Limitations appendix]: "the model has no mechanism for maintaining long-term
  memory. As a result, if an object leaves the frame and later reappears, or if it becomes
  occluded for an extended period, subject consistency may fail to hold." This is the
  strongest primary-source statement of exactly our niche (leave-and-return), coming from a
  training-free method that otherwise reaches 12 hours of continuous generation.
- **Self-Forcing++ (arXiv 2510.02283) concedes the occlusion failure** [VERIFIED, wording
  weaker than a re-ID claim]. Paper, Limitations: "a lack of long-term memory, which can
  cause content divergence in regions occluded for extended periods." Project page: "The
  model lacks long-term memory, causing occluded objects to change after being blocked for
  an extended period of time," and "unoccluded objects may also gradually change due to
  underlying continuous value drifts." Future work: "we aim to incorporate long-term memory
  mechanisms... crucial for achieving true long-range temporal coherence." Caution when
  citing: they never use the words identity or re-identification.
- **LongLive's own ablation (Table 4)** [VERIFIED, attribution corrected]: subject
  consistency is 89.59 with NO KV cache, 93.69 with a plain KV cache, 94.04 with KV
  re-cache. Cache-level mechanisms already carry switch consistency to ~94; re-cache itself
  adds only +0.35 there (its distinctive win is prompt compliance, CLIP 25.92 to 27.87).
  Conclusion unchanged: switch consistency is largely handled at the cache level;
  return-after-eviction is not addressed at all.
- **Context Forcing's Table 1 puts LongLive's effective context at 3.0 seconds** [VERIFIED,
  with a caveat] while LongLive's own paper advertises 240s generation. Caveat: the 3.0s is
  the authors' characterization; the paper defines no explicit measurement protocol for
  "effective context," and its own eval is 60s. That protocol gap is itself an opportunity:
  a rigorous effective-context probe would be a small extra contribution of our benchmark.
- **VBench subject consistency is confirmed near-saturated** (VideoSSM 92.51 vs LongLive 91.09,
  a +1.4 margin, while dynamic degree differs by +13) [confirmed 3-0]. The standard metric is
  blind exactly where the residual problem lives.
- **No published benchmark scores subject-level A-B-A return in streaming generation** with
  re-ID embeddings: MAG-Bench is camera/scene revisit with pixel metrics (PSNR/SSIM/LPIPS, no
  subjects, no ArcFace); EntityBench is multi-shot keyframe pipelines, not streaming KV
  eviction; NarraStream-Bench is closest but training-free symbolic tracking, LLM-judged.
  Three adjacent benchmarks landed May-June 2026: the window is months, not years.

## 2. Anti-anchoring: our negative results do NOT prove the mechanism cannot work

The e2/e3 failures are over-determined. Five defects were simultaneously active, so "the TTT MLP
is the wrong substrate" is NOT established by our data:

1. Frozen backbone (lr 0.0 on everything but memory): no layer ever learned to consume the read.
2. No identity-bearing term in the objective: the only gradient to the memory is the DMD
   distribution-matching gradient, scored by a memoryless bidirectional teacher; nothing
   rewards recall. This root-cause diagnosis is now independently published as MemGen's
   "degenerate text-reliance" finding, in nearly our words (confirmed 3-0).
3. Read magnitude decoupled from relevance: unbounded gate + double normalization
   (o_norm RMSNorm then a_norm LayerNorm) means only direction survives; no path to silence.
4. Apply-first cold reads from a generic init state on the never-supervised first chunk.
5. **Newly found in this review: the switch inference pipeline bypasses the memory entirely.**
   `SwitchCausalInferencePipeline` never passes `ttt_cache`
   (`pipeline/switch_causal_inference.py:82,186,201,214`), so the guard in
   `causal_model.py:563` short-circuits and every switch-mode eval of a memory checkpoint ran
   with the memory OFF. Any switch-benchmark conclusion about the memory is void until fixed.
   This is also exactly the failure mode RAD (2511.12940) blames for prior diffusion-RNN
   failures: the training-inference gap of the memory update (plus missing window overlap);
   their fix is frame-wise autoregressive memory update run identically at train and
   inference. Their second finding [VERIFIED, Table 1, Memory Maze]: chunk-wise, a plain
   LSTM-with-attention block (PSNR 15.64) beats TTT (14.36) and Mamba2 (13.80), both of
   which fall BELOW the no-memory baseline (14.73); frame-wise, the substrates converge
   (TTT-f 15.50 vs LSTM-f 15.30). Substrate choice matters less than update consistency.

Also settled here: the gate01 contradiction. The log scalar `gate(first)` is the pinned bias
(flat 0.0100 for 795 steps); the tensorboard data-dependent series shows the gate genuinely
opening under pure DMD (gatew_b5/16/27: 0.15 to 1.19/1.38/1.48; memout RMS 0.01 to 0.49-0.92).
Combined with e2 (gate 1.26 + identity worse), the consistent reading: **DMD rewards the read
for something (local quality/smoothness), and that something is not identity.** "The gate
opens" is not evidence of usefulness. Checkpoints 450/600/750 exist and were never evaluated.

## 3. What the working recipe looks like (from the landscape)

- **Mechanism**: 1-step gated delta rule (decay gate carries retention architecturally;
  VideoSSM shows a truncated ~5s DMD window then suffices), or retrieval over stored latents
  (LongLive-RAG), or compressed-KV encoder (MemGen). If a fast-weight stays in the design, it
  should be reachable as the `use_decay=True`, 1-step, no-Muon corner of our existing module.
- **Objective**: never DMD alone for the memory. MemGen: memory trained offline with
  flow-matching reconstruction, generator with history loss = the same DMD-form loss with an
  EMPTY text condition for chunks i>1; lambda is "implemented via random sampling" with
  empty-text probability 0.6 [VERIFIED: v1 and v2 texts are identical on this, checked by
  full diff; 0.6 is a sampling probability, never a fixed per-step coefficient]. ARL2:
  stage-1 dense MSE alignment to softmax-attention outputs. VideoSSM is the only pure-DMD
  case and it co-trains the FULL backbone after a stage-1 ODE-regression, with retention
  carried by the decay gate, not by gradient reach.
- **Co-training**: every working system trains at least LoRA-level backbone. The frozen-backbone
  cell is open (DiT-Mem is adjacent: frozen DiT + trained memory encoder, but bidirectional
  T2V guidance, no self-rollout) and doubles as our deployability angle, but our own failure
  analysis found the frozen backbone to be one of the killers (layers never learn to consume
  or attenuate the injected read); treat "frozen + LoRA" as the compromise.

## 4. The defensible contribution (the paper)

Working title shape: **"Subjects that return: re-identification across memory eviction in
streaming video diffusion. A benchmark, a diagnosis, and a minimal recipe."**

1. **Benchmark** (rename internal MAG first; the name is taken by 2512.18741): A-B-A
   prompt-switch revisit protocol on the Self-Forcing lineage. Subject present in A, absent
   for a span strictly longer than KV window + sink during B, prompted to return in A'.
   Score identity of returned vs first appearance on SAM-cropped subjects (ArcFace for
   humans, DINOv2/DreamSim general), normalized to an intra-A oracle. Half same-scene, half
   scene-changed. Validation gates: >3 sigma separation between "never leaves" and
   "substituted subject"; Spearman >0.6 vs human. Note EntityBench's own warning: embedding
   similarity rewards uniformity, not identity; report the oracle-normalized score.
2. **Diagnosis** (mostly free, from existing logs): (a) VBench SC saturated while re-ID
   collapses on the same videos; (b) DMD-identity misalignment: the gate-opening curves
   (gatew/memout tensorboard) against identity metrics; (c) effective-context blindness:
   Context Forcing asserts 3.0s for LongLive but defines no measurement protocol, so we
   define one and measure it on our stack. Our negative results become
   evidence, not embarrassment. Caveat: the diagnosis ALONE is no longer novel (MemGen
   published the text-reliance shortcut); the novelty is the mechanism-level demonstration
   plus the re-ID measurement.
3. **Method**: the cheapest memory that closes the re-ID gap, trained with an identity-bearing
   objective: history loss (text-dropout, the flag already exists in
   `model/streaming_training.py:81-89`, prob currently 0.0) + revisit training rollouts
   (A-B-A needs code; current switch training does one switch, no return) + an identity
   anchor term on the returned subject (feature-match pre-eviction subject crop to
   post-return, through self-rollout). Then THE TABLE, same objective, equal parameter
   budget, substrates: {none (KV+sink), bigger anchor-KV, retrieval, gated-delta 1-step,
   TTT-MLP multi-step}. Whichever wins, the table is the contribution; nobody has it.
4. **Optional technical novelty** if the storage probe says the state cannot even encode the
   evicted subject: reward-style write training (GRPO on the re-ID score at return). Honest
   framing: you cannot backprop through KV eviction at scale (measured: naive cross-chunk
   BPTT OOMs at ~46.6 GiB even at minimal scale K=2, gradient checkpointing crashes because
   the recompute runs against a rolling KV cache the next chunk already advanced, and a
   content-faithful cache snapshot costs ~90 GiB and cannot be sharded); reward-based
   training sidesteps the whole problem. Only if E2/E3 plateau.

**Venue reality**: CVPR 2026 is over (June). Realistic targets: ICLR 2027 (deadline typically
late Sept 2026, ~12 weeks away), CVPR 2027 (~Nov 2026), NeurIPS 2026 workshops (Aug-Sept).
Honest calibration: benchmark + diagnosis, executed cleanly, is a near-certain workshop paper
(the CVPR eval workshop explicitly lists object permanence / multi-shot consistency). A main
conference paper additionally requires the method to actually win the table. That is not
guaranteed; it is a real experiment.

## 5. Ranked experiment plan (3x48GB)

- **E0, week 1, inference-only, THE GATE.**
  (a) Fix the switch-inference memory bypass (thread `ttt_cache` through
  `SwitchCausalInferencePipeline`); without it no memory eval is valid.
  (b) Build the revisit probe (30-50 A-B-A triples, B-span > KV window + sink) + metric
  validation gates. (c) Run: stock Anchor-Forcing, LongLive, Self-Forcing, MemGen if a
  checkpoint is public, plus our e2/e3 checkpoints. Runs under GPU contention.
  **Decision rule: if baselines re-identify fine, there is no job; go to pivots. If they fail
  catastrophically (predicted by MAG-Bench 19-21 PSNR, EntityMem face-acc 0.406, and the
  SF++/RoPE-jitter concessions), GO.**
- **E1, parallel, 1-2 days.** Evaluate gate01 checkpoints 450/600/750 vs baseline (VBench +
  probe). Settles helpful-vs-toxic; either outcome feeds the diagnosis section.
- **E2, weeks 2-3.** Objective MVP on the existing architecture: text-dropout history loss
  (0 to ~0.5), A-B-A revisit rollouts in switch training (new code), identity anchor on
  return; read-path hygiene (bounded tanh gate init 0, single norm); LoRA unfreeze.
  Success metric = probe re-ID delta, never VBench SC.
- **E3, weeks 3-5.** The substrate table under the fixed objective, equal params:
  none / anchor-KV / retrieval / gated-delta (use_decay=True, 1 step, no Muon) / TTT-MLP
  (Muon on). Explicit kill criterion: if anchor-KV or retrieval matches the learned states,
  the learnable memory has no job and the paper ships as benchmark + diagnosis + "cheap
  tricks suffice," which is still a paper.
- **E4, conditional.** Decodability probe of the state after eviction; if storage-limited and
  E2/E3 plateau, GRPO write training on the re-ID reward. Precedent on this exact stack:
  Self-Forcing++ already runs a GRPO stage on top of its DMD student (for long-horizon
  smoothness), so the plumbing is proven; ours changes the reward to subject re-ID at return.
- **E5.** Write-up; diagnosis figures directly from existing logs.

## 6. Pivots if E0 says "no job"

- **P1 (best)**: analysis paper, "what do streaming students actually remember": effective
  context, text-reliance, revisit probes across the whole released-checkpoint zoo; aimed at
  the eval workshop track. Cheap, mostly inference.
- **P2**: pure benchmark race entry (the streaming re-ID cell is still open even as
  measurement-only).
- **P3**: interactive world-model revisit (crowded: WorldMem, VMem, Echo-Memory, MBench).
- **P4**: efficiency/linear-attention conversion (crowded: ARL2, VideoSSM).

## 7. Verification status (second pass completed 2026-07-02)

First pass (workflow, 3-vote adversarial): VideoSSM backbone/mechanism/training/scores,
ARL2 mechanism/motivation, MemGen stack/mechanism/diagnosis/history-loss/not-end-to-end:
all confirmed 3-0, none refuted.

Second pass: every claim left unverified by the quota was re-verified against the primary
sources by four dedicated agents (raw HTML fetched, quotes checked verbatim, MemGen v1/v2
diffed in full). Raw source copies archived in `reference/verification_sources/`.

CONFIRMED exact: LoL 12h + its leave-and-return concession (quoted verbatim in section 1);
SF++ 4min15s with zero memory modules and "incorporate long-term memory mechanisms" as
stated future work; LongLive 240s attention/KV-only mechanism (note: the model itself is
LoRA-tuned during streaming long tuning; "no learnable memory" refers to the mechanism, not
to training); LongLive Table-4 numbers; Context Forcing diagnosis wording, the 3.0s figure,
and the non-learnable cache details (tau=0.95, sink 3 / slow 12 / fast 6, bounded RoPE);
TetherCache training-free, best VBench-Long Total at 30/60/240s (baselines: CausVid,
Self-Forcing, Infinity-RoPE, MemRoPE, Deep Forcing; LongLive NOT compared); RAD Table-1
numbers and the training-inference-gap diagnosis; MemGen history-loss mechanics.

CORRECTED in this doc accordingly:
- SF++ concession wording: "occluded objects change" / "content divergence in occluded
  regions", never "identity" or "re-identification". SF++ also runs a GRPO stage on top of
  DMD (relevant precedent for E4).
- LongLive ablation attribution: 89.59 is the no-cache row; re-cache adds +0.35 subject
  consistency over a plain cache; its distinctive gain is CLIP/prompt compliance.
- Context Forcing: 3.0s is asserted without a measurement protocol; its eval is 60s (the
  240s figure belongs to LongLive's own paper).
- TetherCache 300s: qualitative claim only (Figure 1); quantitative metrics stop at 240s.
- LongLive-RAG: not purely non-parametric. A retrieval-embedding autoencoder is trained
  OFFLINE (reconstruction + window temporal delta loss); the generator stays frozen with no
  memory layers in the denoiser; retrieved latents re-enter via the attention context. It
  evaluates VBench-Long only: no revisit or re-ID experiment (searched for revisit/return/
  re-identification: none), so the gap stands even against this paper.
- MemGen lambda: "implemented via random sampling", empty-text probability 0.6, identical
  in v1 and v2 (no version discrepancy; our earlier caveat was wrong).
- RAD: two failure causes cited (training-inference gap AND missing window overlap);
  frame-wise variants converge, so the LSTM advantage is chunk-wise specific.

STILL UNVERIFIED (do not cite without checking): VideoSSM and MemGen venue/acceptance
status; Echo-Infinity (2606.04527) training objective and backbone (abstract-only, HTML
unavailable); the CVPR 2026 accepted-titles list (secondary blog source).

---

## 8. Execution findings log (Opus 4.8, appended per "keep this doc updated")

**2026-07-02 -- overfit14 (Aden's 14-prompt overfit; 7 no-switch + 7 single-switch; fresh memory from ckpt/final.pt;
gate_init 0.01, decay off, Muon 5, K=2; resumed at 400 -> ~1400 actual steps at ckpt "1000") VBench + frame inspection.**
- VBench subject_consistency: baseline_af (stock AF) 0.978 / memory-on 0.938 / memory-ablated (MEM_ZERO) 0.870.
  memory-on LOSES to stock AF (like e2); memory-on > ablate (+0.068) but confounded (the LoRA co-adapted to the memory,
  so ablating it breaks the model). NB: this eval used single-prompt inference.py (NOT the switch pipeline), so it is
  not touched by the switch-bypass bug, but it also does not exercise re-ID.
- FRAME INSPECTION (decisive, visual): memory-on videos are FULL ARTEFACT FROM FRAME 0 (background ~2/3 garbled
  colour-speckle mush; text-pinned subject partly survives early), collapsing to uniform brown mush by 15s; the baseline
  stays sharp and coherent to the end. So the memory actively DESTROYS the generation across the self-rollout. This
  visually PROVES the VBench-blindness claim: the collapsed-but-uniform memory video still scores 0.938 because
  subject_consistency = adjacent-frame DINO similarity and uniform mush has near-identical adjacent frames.
- lr-vs-objective (Aden asked "lr too high?"): NO, not a blow-up -- training was numerically stable (state norm
  conserved ~1.0, no NaN, clean finish). lr_mem=1e-4 (10x the LoRA lr) is a SEVERITY knob (opens the gate fast/wide ->
  bigger injection -> more garbage), but lowering it only TRADES toxic for inert (do-no-harm-but-useless); it does not
  fix the root cause = the objective (DMD-only rewards the off-manifold injection). Consistent with sections 1-2.
- Why DMD gives coherent baselines but garbage with our memory: DMD shapes a PRETRAINED, text-anchored, FEED-FORWARD
  backbone (on-manifold by construction); our memory is a random add-on = an extra weakly-constrained injection channel
  + a RECURRENT self-feedback loop (no analog in LongLive/SF) + a train/inference gap (cold frame-0 state) -> an
  off-manifold injection that compounds. Matches the 5-defect anti-anchor list (section 2) and the RAD train-inference-gap.
- gate01 ckpt 450 vs 750 (same arch, diverse stream, not overfit) test done: less training = less garbage CONFIRMED, a
  clean SEVERITY GRADIENT. VBench subject_consistency (same 4 prompts): baseline 0.991 > gate01-450 0.987 > gate01-750
  0.984 > overfit 0.940. Ranking is correct (VBench not blind to direction) BUT compressed near saturation: a full
  collapse-to-mush only drops to 0.940 (reads as "fine"), and background_consistency is uniformity-blind (overfit 0.960
  == gate01-750 0.960). So VBench captures direction, not severity. gate01 checkpoints are CLEAN at frame 0 (the
  overfit's frame-0 garbage was an overfit-only extreme, not the general memory behavior). Frames in reference/runs/*/frames/.

**2026-07-02 -- THE MEMORY FREEZES MOTION (new mechanistic finding, Aden-prompted).** Aden noticed the memory videos
"do not advance". Measured mean inter-frame difference on a walking prompt (high motion): baseline 15.9 (woman) / 23.2
(man); memory normal-training (gate01-750) 5.9 / 5.7; memory overfit 2.4 / 2.1. The memory CAPS motion at a low value
regardless of prompt (~6 normal, ~2 overfit = 7-11x less dynamic than baseline) -- it acts as a motion limiter/freezer.
INTERPRETATION: trained with DMD alone, the memory learns to inject a near-constant per-frame signal that FREEZES the
clip; freezing is a degenerate optimum because a static clip has near-identical adjacent frames (trivial for the local
DMD loss, and it directly inflates VBench subject_consistency, which literally rewards frame-to-frame similarity). So
the memory's "high VBench / clean-looking" score is a FROZEN degenerate clip, not quality. This strengthens two doc
claims at once: (a) DMD-only objective is the root cause (it rewards a degenerate freeze), and (b) VBench is inadequate
(it rewards exactly that freeze). Demo: reference/runs/e3_overfit14/eval/walk_freeze.html.

**2026-07-02 -- Fable consult on the freeze finding (targeted, lit-checked).** Phenomenon CONFIRMED; causal claim
split in two, with plan consequences.
- CAUSE SPLIT: the freeze INCENTIVE is not ours; static-motion mode collapse is a documented property of the DMD
  loss family: Adaptive Video Distillation (2603.21864) verbatim "mode collapse extends into the temporal dimension,
  resulting in videos with limited or even static motion" (their fix: temporal-variance regularizer, DD 72.22 -> 100.0);
  SF++ (2510.02283) calls motion stagnation "a common failure mode in Self-Forcing"; Infinity-RoPE (2511.20649)
  quantifies it in our lineage (Self-Forcing DD 0.60 @5s -> 0.32 @60/120s) and its Table-1 caption states baselines
  "report high temporal-quality metrics largely due to stagnation effects reflected in their low dynamic degree".
  What the memory adds is the CAPABILITY: double norm pins read magnitude, likely-collinear reads make
  `x = x + mem_out` (causal_model.py:607) a shared per-frame additive bias = a temporal low-pass channel; DMD only
  has to open the gate. Strong universal form REFUTED by VideoSSM: pure-DMD e2e learnable memory with DD 50.50 vs
  LongLive 37.50 / Self-Forcing 35.00 (2512.04519 Table 2), so memory+DMD does not intrinsically freeze; our
  channel + frozen backbone do. Alternatives sorted: recurrent low-freq bias = the proximate mechanism (compatible);
  cold read explains frame-0 artefacts not a 15s cap (reject); train/inference gap cannot give a monotone severity
  gradient with training steps (reject as primary).
- LITERATURE: documented in 3 forms, none exactly ours: (1) static gaming consistency = VBench's own DD rationale
  ("a static video can score well in consistency metrics") + the Infinity-RoPE caption; (2) motion stagnation in
  memoryless DMD students (SF++, VideoSSM: "CausVid ... succumbs to motion stagnation, with the child becoming
  nearly static"); (3) NON-learnable memory freezing: VideoSSM on sinks, "by repeatedly attending to static early
  frames, they freeze global memory, which encourages scene looping and repetition". Guards used: VideoSSM =
  architectural (continuously updated state) + reports DD + user study, NO motion loss; SF++ = horizon training +
  GRPO; 2603.21864 = explicit temporal-variance loss. NOT FOUND anywhere: a learnable memory under DMD reported as
  converging to a motion limiter -> our finding is a new quantified INSTANCE of a documented FAMILY (frame it so).
- PLAN CHANGES (verdict unchanged): (1) E0 probe is FREEZE-GAMEABLE as designed (a frozen subject re-IDs
  perfectly): add DD-conditioned re-ID scoring + a freeze adversary as a validation gate (our own gate01-750/overfit
  ckpts are the adversary; metric must NOT rank them above baselines); condition GO/NO-GO on matched DD (baselines
  themselves stagnate over horizon). (2) E1: add RAFT DD next to VBench+probe. (3) E2: the three proposed terms do
  NOT remove the freeze incentive -- history loss is EASIER when frozen (history=present), identity anchor is
  MAXIMIZED by a frozen subject, revisit rollouts fight freeze only via the same DMD term that permitted it. So:
  read-path hygiene (bounded tanh gate + kill the double norm) PROMOTES from hygiene to load-bearing anti-freeze
  fix; if freezing persists add the 2603.21864-style temporal-variance term (paired with motion smoothness to
  block jitter-gaming); log a motion metric as kill-switch in EVERY run. (4) E3: DD becomes a mandatory column of
  THE TABLE, and the injection interface must be held fixed + non-degenerate across substrates or the table
  measures the channel, not the substrate.
- MECHANISM: the finding cuts on the READ/INJECTION channel, not the write rule (Muon vs delta vs retrieval is
  orthogonal); the additive-into-residual channel IS implicated. Composes with RAD: update/injection consistency
  dominates substrate choice.
- CAVEATS before the claim hardens: inter-frame pixel diff conflates motion with blur (memory clips also degrade);
  n=2 prompts, baseline spread 15.9 vs 23.2; LoRA co-adaptation confound. Cheap discriminators, all inference-only:
  mem_out cross-frame cosine (~1 = DC injection), read-on/write-frozen rollout, MEM_ZERO on gate01-750 (motion
  should return to ~16-23), state SVD for rank collapse; recompute motion with RAFT (VBench DD protocol) for
  anything citable. New paper angle: "Goodhart without seeing the metric" (DMD's chunk-local objective and VBench
  SC both reward temporal self-similarity, so the memory converges to the metric's documented blind spot without
  ever touching it); also note MAG-Bench's PSNR revisit scores are freeze-gameable too.

**2026-07-02 -- Fable follow-up: RAFT + MEM_ZERO probes ran; freeze caveats CLOSED; MEM_ZERO control found LEAKY
(code-verified).**
- Both discriminators requested above came back as predicted. RAFT flow, 6 diverse high-motion prompts: baseline
  mean 11.3 (range 4.0-32.1) vs gate01-750 mem-on 1.6 (0.5-2.8), ~7x less on 6/6 prompts -> the freeze is real
  motion loss, not blur; the pixel-diff caveat is closed. MEM_ZERO on the same checkpoint: 9.3 ~ baseline -> the
  additive read injection is causally the freezer, not the co-trained LoRA; the LoRA-confound caveat is closed for
  gate01-750 (overfit14's mem-on > mem-zero gap was the overfit extreme, not general memory behavior).
- BUT code check: `MEM_ZERO_COLD_FRAMES` zeroes the read ONLY on denoising passes (guard `not ttt_update`,
  `causal_model.py:603-606`); the clean KV-cache rerun (`causal_inference.py:226-238`, `ttt_update=True`) STILL adds
  `mem_out` (detached) into the activations that populate the KV cache (`causal_model.py:607`). So the control removed
  the denoising-path read only; the residual 9.3-vs-11.3 gap is attributable to this KV-path leak as much as to the
  LoRA. One-line fix (also zero on the write pass, e.g. `MEM_ZERO_ALL`) required before any E0/E3 mem-zero arm.
- "Freezing memory as motion regularizer, then discard" lead: DEAD, twice over. Empirically: mem-zero mean 9.3 is
  slightly BELOW baseline 11.3, overshoot on only 2/6 prompts; the one 2.6x overshoot clip was judged over-driven
  (a second degenerate mode), and its motion is sustained, not cut-spam. Theoretically: a tug-of-war requires an
  objective that pulls AGAINST the freeze, and DMD demonstrably does not (static collapse is an optimum of the loss
  family, see the entry above) -- the LoRA had no gradient incentive to compensate, so no counter-bias was learned.
  Lit check: the specific phenomenon (in-forward suppressive module, co-trained net over-compensates, module
  discarded for a net gain) found NOWHERE; nearest neighbors are mechanistically different: loss-side adversarial
  modules discarded at inference (MoGAN 2511.21592 motion-GAN alongside DMD; APT 2501.08316; DANN gradient
  reversal), inference-time exploitation of a degraded model's bias (Autoguidance 2406.02507), and co-adaptation
  where ablation HURTS (dropout rationale, 1207.0580) = the opposite sign.
- Established vs open, stated precisely: THIS memory (multi-step Muon TTT, additive read, DMD-only, near-frozen
  backbone) is now strictly net-negative on every within-window axis measured (VBench SC, visual coherence, motion),
  causally via the read channel; removing it merely restores baseline. NOT established: whether any memory has a
  positive re-ID job -- no probe so far exercises leave-and-return, and VideoSSM still refutes the universal "no job
  under pure DMD" (different channel + full co-train). The E0 gate is UNCHANGED: it tests the problem (does the
  baseline fail re-ID), not our mechanism.
- Instrumentation deltas: (1) full-kill MEM_ZERO first. (2) E0: three arms per memory checkpoint (mem-on /
  mem-zero(full) / stock baseline; on-minus-zero = the memory's causal re-ID delta, zero-minus-stock = LoRA-drift
  canary, currently ~0); RAFT flow computed on the SAME probe videos, reported as (re-ID, flow) pairs; motion
  validity floor (e.g. flow >= 0.5x stock on the matched prompt) below which re-ID is excluded from the GO/NO-GO
  statistic; gate01-750 stays the canonical freeze adversary; A' segments must themselves demand motion. (3) E3:
  RAFT flow is a mandatory column (in-house pipeline now validated, finer than VBench DD); add a per-row
  mem-zero-delta column (substrate contribution isolated within-checkpoint, immune to run-to-run LoRA variance);
  any row below the motion floor is marked FROZEN-INVALID regardless of its other scores; the 6-prompt RAFT battery
  is the standard early kill-switch for every E2/E3 run. Note the limit of what MEM_ZERO proves: the channel
  CARRIES the freeze, not that a bounded channel RESISTS it -- the DMD incentive is untouched, so read hygiene stays
  load-bearing-but-unproven-sufficient and the temporal-variance backup + per-run motion logging stay non-negotiable.

### Log 2026-07-02 -- clean MEM_ZERO_ALL re-eval on the full 6-prompt dynamic battery (Opus)
- After the MEM_ZERO_ALL fix (zero mem_out on BOTH the denoising READ pass and the clean-rerun WRITE/KV pass),
  re-ran the clean memzero-full arm on all 6 dynamic prompts. RAFT flow (baseline / memzero-FULL / ratio):
  jogging 5.61/2.98/0.53x, cycling 12.10/19.11/1.58x, breakdance 4.03/0.97/0.24x, sprint 7.70/7.13/0.93x,
  spin-dress 6.34/1.83/0.29x, skateboard 32.12/31.10/0.97x. MEAN 11.32 / 10.52 = **0.93x**.
- Conclusion: the clean memzero-full arm is ~baseline and HIGHLY variable (0.24x-1.58x); NOT systematically more
  dynamic. The earlier "more dynamic model" lead (the 2 walking clips at ~1.9x) was prompt-specific and does NOT
  generalize. Fable's "mem-zero ~= baseline" holds. No Fable re-consult warranted -- this CONFIRMS the verdict,
  it is not a reversal.
- What survives as the genuinely interesting result: the FREEZE + metric-gaming diagnosis (memory-ON mean 1.55 vs
  baseline 11.32, ~7x less, 6/6 prompts, causal via the additive read; frozen video games VBench subject-consistency
  = Goodhart-without-seeing-the-metric). Keep as a diagnosis figure for the pivot paper, not a standalone result.
- Viewer redeployed: clean_dynamic.html (Artifact 3cc3c81e), 6 side-by-sides + the RAFT table above.

**2026-07-02 -- Fable consult: E0-gate dropped for plain-long-first (AGREE-WITH-CHANGES).** PI decision ratified with
corrections. (A/B/C) Plain-long 120s+ E2-first APPROVED; leave/return benchmark DEMOTED to follow-up (novelty reserve,
switch-bypass fix must land before it); PREMISE CORRECTED: at 120s the first frames are NOT evicted (sink_size 3,
global_sink true, anchor KV-reorg) -- plain-long drift = retained-anchor-not-used + compounding value drift, so the
memory's job is "always-local RoPE-free identity anchor", and the mandatory cheap competitor is pinned-KV/bigger-sink.
STRATEGIC COST accepted knowingly: plain-long = VideoSSM's turf; differentiation = freeze/Goodhart diagnosis +
identity-objective recipe + substrate table + leave/return follow-up. GATE NOT DROPPED, replaced: 1-2 day
inference-only gate = baseline drift curve (stock AF, 8-10 dynamic subject prompts, 120-240s, DINO/ArcFace-on-crop vs
chunk-0 anchor, oracle-normalized, RAFT per segment; freeze-adversary validation vs gate01-750) + ANCHORING PROBE
(sink 3->9->21 / pinned subject-KV, same seeds): if pinning flattens drift -> anchoring-limited, job exists, pinned-KV
= the bar; if not -> compounding-dominated, read-injection bet weakens. E2 GO iff headroom + metric valid.
FALSE-NULL guard: evaluate drift CURVES not endpoints (write trained ~5-10s, tested over ~24-48 detached renorm
writes -> predicted early-help-late-fade = retention problem = cue for use_decay/gated-delta, NOT a no-job proof).
FALSE-POSITIVE guard: 3 inference arms necessary but NOT sufficient (identity objective deepens LoRA co-adaptation ->
on-minus-zero overstates); clean attribution = NO-MEMORY TWIN (same recipe, no memory) = the "none" row of THE TABLE;
run twin only after mem-on beats stock. E2 LEVER CORRECTED: pre-gate o_hat aux alone WILL fail silently (zero gradient
to the gate; gate set by DMD alone = freeze attractor; pooled head can absorb the aux) -> load-bearing term = FRAME-
LEVEL identity anchor (chunk-0 subject-crop DINO/ArcFace embedding cached, 1-2 decoded frames/chunk, cosine, backprop
through frozen decoder through gate+read+write); o_hat aux demoted to optional probe; post-gate supervision REJECTED
(forces gate open = toxic shock). INGREDIENTS: +read-path hygiene load-bearing (bounded_gate tanh init 0 exists in
_project_output; kill double norm); gate-warmup schedule DROPPED (bounded learnable gate + through-gate identity term
subsumes it; forced ramp = toxic shock); temporal-variance = differentiable TRAIN term (2603.21864-style, + motion
smoothness vs jitter) vs RAFT = non-differentiable EVAL kill-switch (roles split); watched degenerate mode = moving
background + frozen subject (escalate variance term to subject crop only if it appears). E2 v1 recipe = frame-level
identity anchor + temporal variance + text-dropout 0.3-0.5 (impl already MemGen-shaped, chunk_start_frame>0 guard) +
bounded gate init 0 + single norm + LoRA unfreeze; success = drift-vs-time vs pinned-KV bar + twin, RAFT floor, never
VBench SC.

### Log 2026-07-02 -- E0 LAUNCHED + E1 code/config PREPARED (Opus, executing Fable's ratified plan)
- PI ruling: skip E0 verdict-as-gate but keep the measurement; drop the re-ID leave/return build for now; go
  plain-long-first (Fable ratified with corrections, above). Fable = strategy, Opus = execution.
- E0 (running, reference/runs/e0_reid/): stock-AF baseline drift + anchoring probe. 8 subject-centric DYNAMIC
  prompts, ~120s (num_output_frames 480). 4 arms, 1 GPU each (CVD 0/3/4/5): stock sink3 (baseline), stock sink9
  (local18), stock sink21 (local30), gate01-750 memory-ON sink3 (freeze adversary for metric validation).
  Ruler = reference/runs/e0_reid/drift_metric.py: DINOv2-vitb14 identity-drift curve (1-cos vs chunk-0 anchor,
  oracle-normalized by cross-subject anchor distance) + RAFT per segment + frozen flag. All offline (hub cache).
  FINDING (pre-results): sink21/local30 CRASHES the prompt_cache (_apply_cache_updates copy_ size 4680 vs 0) =
  the OOD instability Fable predicted for sink inflation; worked around with use_prompt_cache:false for that arm.
- E1 ("Fable E2", code done, NOT launched -- gated on E0 headroom + clean GPU + tv-floor calibration):
  reference/runs/e1_identity/configs/e1_identity_v1.yaml. Implemented, all behind flags (defaults OFF = baseline
  unchanged): (1) read hygiene = single_norm flag (drop redundant o_norm; plumbed video_memory->causal_model->
  wan_wrapper) + bounded_gate:true + gate_init:0.0; (2) FRAME-LEVEL identity anchor loss in
  StreamingTrainingModel.compute_generator_loss -- decode k=2 latent frames (grad) -> frozen DINOv2 CLS ->
  cosine to chunk-0 anchor; grad-flow through decode VERIFIED (CPU smoke test). o_hat pre-gate aux NOT used
  (Fable demoted it). (3) differentiable temporal-variance FLOOR (hinge on adjacent-latent motion, anti-freeze,
  ships with id term). (4) text_dropout_prob 0.4, (5) lora_unfreeze_from_block 0 + train_write true, lr 1e-5 /
  lr_mem 1e-4, use_decay false. OPEN before launch: calibrate tv_motion_floor to ~0.5x baseline e1_latent_motion
  (read from first ~20 steps' log); full-model 1-step smoke test on a free GPU (confirm decode_to_pixel signature
  + param groups). Not yet run: no-memory TWIN (Fable's clean-attribution arm) -- deferred until mem-on beats stock.

### Log 2026-07-02 -- Fable code audit of E1: identity term was a STRUCTURAL NO-OP (caught pre-trust)
- PI asked Fable to audit the E1 code. Verdict: loss wiring / single_norm / anchor cache / dtype all CORRECT,
  BUT the identity term was a structural no-op: empirically `id_loss=0.0000` every generator step (all `chunk@0`).
  The observed gate opening (0->0.265, gate_init=0) was **pure DMD** (the SAME freeze-driver as gate01), NOT the
  identity term. Do not misread that 0.265 as "identity bites."
- Root cause (2 compounding defects): (1) anchor set only in the generator-loss path (chunk_start_frame==0), and
  (2) under dfake_gen_update_ratio=5 + max_length=60 the generator phase-locks to frame 0 every step, so the
  `elif chunk>0` id-loss branch NEVER runs. id gradient identically zero. (tv also inactive: floor=0.0.)
- FIX APPLIED (both required): Fix 1 = set the anchor in the ROLLOUT (`_set_id_anchor` called from
  `generate_next_chunk` when previous_frames is None), not the loss path -> populated regardless of gen/critic.
  Fix 2 = windowed path `bptt_window:2, grad_tail_W:1` so ONE generator step spans chunk-0 (no-grad, sets anchor)
  -> chunk-18 (grad) -> id_loss fires deterministically; also re-enables the grad probe. + `tv_motion_floor:0.2`
  (~0.5x observed chunk-0 latent motion 0.43), `slice_last_frames:27` (proven K=2 KV span). Relaunched CVD 0,3,5.
- VALIDATION GATE before trusting ANY overfit signal: confirm `e1_id_loss > 0` on chunk@18/36 prints. Only then
  read gate/motion/id_cos. Distinguish id-driven vs DMD-driven gate: id_loss_weight=0 ablation (if gate still
  opens -> DMD). This is exactly the do-not-fool-yourself guard.
- Meta: the PI's instinct to have Fable review the code BEFORE trusting results was correct -- it caught a bug that
  would have made us misread a pure-DMD (freeze-driver) gate opening as "the identity mechanism works".

### Log 2026-07-02 -- E0 RESULT: baseline drift + anchoring probe (DINOv2 ruler, 8 dynamic prompts, ~120s)
Per-arm means (motion=RAFT, drift=oracle-normalized identity drift late-third, ref=0.865):
  stock_sink3 (baseline): motion 26.2, drift 0.237, 0 frozen
  stock_sink9:            motion 10.3, drift 0.429, 0 frozen
  stock_sink21:           motion  1.2, drift 0.402, 0 frozen
  memon (memory ON):      motion  1.1, drift 0.089, 2/8 frozen
Verdict:
- HEADROOM = YES: baseline drifts 0.237 while moving (motion 26) -> E1 has a real target. GO for E1.
- ANCHORING PROBE = a bigger sink does NOT flatten drift, it WORSENS it (sink3->9 +0.193, sink3->21 +0.165)
  and sink21 collapses motion (26->1.2, the OOD instability). So drift is NOT anchoring-limited -> looks
  COMPOUNDING-ERROR-DOMINATED. Cheap-trick (sink) FAILS; but it also means pure read re-injection is a weak
  lever -> E1's load-bearing part is the identity OBJECTIVE + backbone co-train, not injection alone.
- The memory FREEZE-GAMES the identity metric exactly like VBench: memon drift 0.089 looks best but motion is
  4% of baseline, 2/8 frozen -> invalid. Re-confirms the freeze diagnosis on the identity axis + validates the
  mandatory motion floor.
- E1 success bar (sharpened): reduce drift BELOW ~0.24 WHILE keeping motion comparable to baseline (never a
  frozen "win"). Report = reference/runs/e0_reid/out/e0_report.html (Artifact 2ee1b6c4), drift_results.json.

### Log 2026-07-02 -- Fable 2nd code+log audit: 80% wasted-step bug + grad-probe interpretation
After Fix1/Fix2 the run trained + id_loss fired, but PI spotted intermittent firing (~3/7 gen steps). Fable audit:
- Q1 INTERMITTENT FIRING = REAL BUG, worse than "id skips": on a broken window (_k=1 breaks on
  can_generate_more before the only rg=True chunk) `_losses` is empty -> else branch -> generator_loss=0,
  NO backward. ~80% of generator steps trained NOTHING (not even DMD). Cause: windowed generator shares the
  critic-advanced streaming state -> starts mid-sequence; max_length 60 = only 3 chunks (0,21,39), K=2 window
  completes only from frame 0 or 21. FIX Q1 = force fresh sequence before the windowed loop
  (`self.streaming_active=False; self.start_new_sequence()` at distillation.py ~1156). CAVEAT: sanity-only,
  always trains frame-0->21, NEVER frame-39 (worst horizon); for the BROAD run raise streaming_max_length to
  120+ AND use a headroom-guard reset (reset only if current_length + K*min_new_frame > temp_max_length),
  NOT force-fresh. Do NOT let the window cross a sequence boundary (would pair anchor prompt A w/ grad prompt B).
- Q2 GRAD PROBE step-0 `WRITE lrs=0 READ qkv=0 proj=466` is CORRECT, not a bug: proj zero-init + gate_init 0 +
  bounded -> gate=tanh(0)=0 -> mem_out=0 -> dL/do=0 so qkv/lrs get exactly 0 at step 0; only the gate weight can
  move. Probe was gated `step<4` so never re-fired post-open. FIX Q2 = probe at steps {0,5,10,15,20,30};
  PREDICTION qkv>0 & lrs>0 by step ~10 once gate opens. NOT YET PROVEN that id-grad reaches read+write until this.
- Q3 id_cos swing 0.34-0.86 = EXPECTED (real drift; anchor & grad chunk always same prompt, rolled back-to-back;
  wrong-prompt anchor impossible; chunk@21 close vs chunk@39 further-horizon). Q4 floor 0.2 permissive
  (0.44x motion) -> bumped to 0.3; calibrate vs STOCK AF baseline motion (not this run) for broad run.
- GOOD NEWS: memout_b5 ~0.055, gatew ~0.19 = a FRACTION of gate01's runaway (gatew->1.0, memout->0.5). The
  bounded-tanh gate + single_norm + gate_init=0 read hygiene HOLDS the injection small/controlled, as intended.
- Applied Q1+Q2+Q4, relaunched. Validation gates: id fires every gen step; step-10 probe shows qkv>0/lrs>0;
  then id_loss_weight=0 CONTROL run to attribute the gate opening (still opens with id off => DMD; only with
  id on => id-driven). Do NOT conclude "memory has a job" from the current overfit yet.

### Log 2026-07-03 -- E1 OVERFIT EVAL: DECISIVE NEGATIVE. The additive read channel is TOXIC even with a correct objective.
Clean 5-arm eval (8 prompts, 120s, seed 42, DINOv2 drift oracle-normalized ref 0.865; MEM_ZERO_ALL attribution):
  baseline (stock AF):  drift 0.237  motion 26.2  frozen 0/8
  ck150 mem-ON:         drift 1.106  motion 11.1  frozen 0/8
  ck150 mem-ZERO:       drift 0.234  motion 17.4  frozen 0/8
  ck300 mem-ON:         drift 1.173  motion 17.5  frozen 0/8
  ck300 mem-ZERO:       drift 0.246  motion 13.5  frozen 0/8
VERDICT:
- The memory DESTROYS identity: mem-ON drift 1.1-1.2 (>1.0 = subject ends FURTHER from its own frame-0 than a
  random different subject), ~5x worse than baseline 0.237. Not a preservation, an active corruption.
- Clean MEM_ZERO attribution: mem-ZERO (memory off, SAME co-trained LoRA) = 0.234 = baseline. So the LoRA alone
  is fine; the toxicity is 100% the MEMORY injection (on-minus-zero = +0.9 at both ckpts). NOT the LoRA.
- NOT a freeze this time (0/8 frozen, motion 40-67% of baseline). A distinct failure = active identity corruption
  via the additive read. More training = worse (150->300: 1.106->1.173). Confirms the memout->0.5 toxic-branch read.
- KEY: objective FIXED (frame-level id term, id_loss fires), gradient FIXED (reaches read+write, grad-probe proven),
  hygiene FIXED (bounded tanh gate init0 + single_norm). It STILL fails -> the culprit is isolated to the
  ADDITIVE-READ INJECTION CHANNEL itself, not the objective/gradient. VideoSSM uses a different channel
  (gated-delta state recurrence, no additive frame injection).
- This is a clean, publishable diagnosis: "additive gated-read injection is toxic for identity on this DMD stack
  even with a correct identity objective + gradient + read hygiene." Isolates the channel as the failure.
- STRATEGIC FORK (Fable consult pending): additive channel dead -> switch substrate to gated-delta state
  recurrence (VideoSSM-style) vs pivot fully to the diagnosis+benchmark paper. Awaiting Fable.
- Data: reference/runs/e1_identity/eval/out/ruler/drift_results.json; checkpoints
  /shared/ssd_14T_project/adenelmi/e1_overfit_v1/checkpoint_model_{000150,000300}.

**2026-07-03 -- E1 clean eval: MECHANISM BET DEAD (decisive, ablation-clean). Fable strategic call.**
Setup: objective FIXED (frame-level pixel identity term, id fires every gen step), gradient FIXED (grad-probe:
id reaches read qkv~0.4 + write lrs~0.05 on all 3 layers once gate cracks), read hygiene FIXED (bounded tanh
gate init 0 + single_norm; memout controlled ~0.055 early). Eval: 8 prompts, 120s, seed42, DINOv2 drift
oracle-normalized (ref 0.865), MEM_ZERO_ALL attribution.
RESULT: baseline drift 0.237 / ck150 mem-ON 1.106 / ck150 mem-ZERO 0.234 / ck300 mem-ON 1.173 / ck300
mem-ZERO 0.246. mem-ON drift >1.0 = subject ends FURTHER from its own frame-0 than a random different subject
= ACTIVE identity corruption, ~5x baseline; on-minus-zero = +0.9 (100% attributable to the additive read;
LoRA-only = baseline); MONOTONE worse with training (150->300); NOT a freeze this time (0/8 frozen, motion
40-67% of baseline) = read hygiene converted the earlier FREEZE mode into a CORRUPTION mode without fixing
net-negativity. Confirms the memout-vs-id_cos toxic-branch prediction (memout ran to ~0.5 while training id_cos
FELL 0.585->0.455 = id gradient lost the tug-of-war THROUGH this channel).
INTERPRETATION: this retires the Section-2 anti-anchor defense -- 3 of 5 defects cleared (objective, gradient,
magnitude) and it STILL fails toxically. Culprit isolated to the ADDITIVE-READ CHANNEL x FROZEN BACKBONE x DMD
interaction (NOT the objective or gradient path). SCOPE (keep, or TTT-Video rebuts): additive residual is NOT
toxic universally -- TTT-Video's additive TTT block works WITH full backbone fine-tuning; it is toxic in OUR
frozen-backbone+DMD cell. single_norm keeps a_norm -> injection is direction-only fixed-magnitude = a
perturbation, not a content-faithful edit.
VERDICT: mechanism bet DEAD for any NOVEL positive. Novelty-preserving salvages (subspace/site tweaks) = ~do-no-
harm at best (inert, not a paper); winning salvages (state-recurrence / full co-train) = VideoSSM/TTT-Video =
preempted. DO NOT swap substrate as a method play; gated-delta belongs only as the "minimal recipe" ROW of the
benchmark substrate table. PIVOT FULLY to diagnosis + benchmark (problem bet). This negative is the CENTERPIECE:
"additive fast-weight read corrupts identity even with a correct identity objective, proven gradient, and bounded
magnitude, on a frozen-backbone DMD student -- ablation-isolated to the channel"; explains why the field converged
on state-recurrence + co-training (VideoSSM) and offline-reconstruction/history-loss (MemGen).
CHEAP CONFIRMATIONS before write-up (inference-only, ck300, ~1h, no training): (1) injection dose-response --
scale gate/mem_out alpha in {0,0.1,0.25,0.5,1.0}, expect drift monotone 0.24->1.17 = channel toxic by dose,
objective-independent; (2) pre-gate o_hat identity-decodability probe -- if o_hat IS identity-bearing but additive
injection destroys identity => clean channel-form diagnosis; if not => compound (write can't store + channel toxic).
Differentiator still alive: re-ID / leave-return A-B-A (VideoSSM tested plain-long only). Mechanism as a novel
contribution is CLOSED.

**2026-07-03 -- Fable open re-brainstorm (fresh code re-read, supersedes the narrow "additive-channel" fork).**
RE-DIAGNOSIS (code-grounded, corrects prior "additive channel toxic"): the failure is a STACK, not the channel.
(1) The Muon+renorm-to-init write (video_memory.py:184-192) only ROTATES a fixed-norm shell -> logs show drift~110
@ statenorm/init~1.0 = the state CHURNS, never accumulates -> `o=swiglu_apply(q,state)` is a per-frame random-ish
projection of the query = a FEATURE INJECTOR, not memory (no identity is ever stored to inject). (2) That off-
manifold per-frame feature is added into a FROZEN backbone that can't consume it. (3) DMD steers the open gate to a
non-identity use (memout->0.5 while train id_cos FELL). Tell: mem-ON is LESS dynamic AND MORE drifted at once =
off-manifold degradation, not clean freeze/drift. BINDING CONSTRAINT underneath all of it (the real story): TRAIN
horizon ~21 frames << EVAL horizon 120s(~480), and cross-chunk BPTT OOMs -> a 21-frame objective can't teach 480-
frame drift-resistance through ANY module. Proof it's horizon not memory: E0 = plain-long drift is compounding-error-
dominated (bigger sink WORSE, frame-0 still in sink so re-anchoring has no job); E1 mem-ZERO=0.234~=baseline 0.237 =
the id gradient failed through the BACKBONE path too. Field escapes the horizon trap 3 ways (TTT-Video curriculum
3->63s; SF++ GRPO on the long rollout; VideoSSM closed-form recurrence = retention w/o BPTT) -- we do none; gated-
delta "fix" just reproduces VideoSSM. Target caveat: DINO-cos-to-frame-0 conflates identity w/ pose/motion (anti-
motion pull), confounded but not the main story. VERDICT unchanged & firmer: mechanism bet DEAD for a novel positive.
MOVES: (1) PIVOT to diagnosis+benchmark; 3 core figures = E1 clean negative (correct-objective-still-toxic, ablation-
clean) + E0 compounding-drift/sink-hurts + freeze/Goodhart metric-blindness. (2) ONE positive shot, aimed at the
HORIZON not the memory = GRPO on a long-range re-ID reward (SF++ plumbing exists, no BPTT, trains LoRA directly,
memory-agnostic); do NOT gate the paper on it. (3) drop the fast-weight as novelty; gated-delta only as a table row.
CHEAPEST MIND-CHANGER (do first, ~1 day, inference-only): leave-return A-B-A re-ID probe on BASELINES (subject evicted
> KV+sink, returns; DINO/ArcFace on crop, oracle-norm, RAFT floor). Catastrophic baseline failure => real open problem
=> benchmark+GRPO justified; baselines fine => pure diagnosis/measurement paper. Second (~1h): o_hat decodability +
state-recall probe -- predict the state does NOT encode frame-0 subject (churn) = crisp diagnosis sentence.

**2026-07-03 -- Fable full re-brainstorm + fresh literature pass (post-cutoff). Landscape moved; fallback cells closing.**
NEW PREEMPTIONS read this session: MBench (2606.00793, Jun26) = comprehensive video-world-model MEMORY benchmark
(entity/environment/causal, 12 sub-dims) -> standalone "streaming memory benchmark" is now a RACE ENTRY not an open
cell. IAMFlow/NarraStream-Bench (2605.18733, May26) = TRAINING-FREE identity-aware memory across prompt transitions +
own 60s benchmark -> owns "identity across prompt shifts in narrative long video". RAVEN (2605.15190, May26) = CM-GRPO
on OUR exact stack (causal AR diffusion, distilled, real-time streaming, self-rollout repack) -> "GRPO on a streaming
distilled student" is DONE (reward = extrapolation/consistency, not identity). Identity-GRPO (2510.14256, Oct25) =
ArcFace-reward GRPO but OFFLINE multi-human. VDE-Subject = a subject identity-drift metric already exists. => mechanism
CLOSED, benchmark CLOSING, GRPO-on-re-ID = merge of two published ideas (low novelty), drift-metric partially taken.
REVISED EV RANKING: P1 diagnosis + METRIC-INTEGRITY ("Goodhart in long-video gen") = RECOMMENDED ~65-70% (E1+MEM_ZERO
clean negative + E0 sink-hurts + a freeze-adversarial motion-conditioned identity metric that RERANKS released models;
differentiates from MBench/VDE by explanation+measurement-integrity not a scoreboard; inference-heavy, fits 3-4wk).
P2 standalone re-ID benchmark = DOWNGRADED ~25% (MBench/NarraStream crowd) -> keep re-ID A-B-A as a PROBE inside P1.
P3 GRPO-on-re-ID = mostly mirage ~15-20% (RAVEN+Identity-GRPO). P4 negative-results = fallback framing of P1.
ONE POSITIVE SHOT (memory-AGNOSTIC, attacks train-horizon<<eval-horizon, no BPTT): supervised DRIFT-CONSISTENCY /
long-range identity anchoring -- roll student under no_grad to frame N (RAVEN repack minus RL), cache detached early
identity, 1 grad chunk at N pulled toward it through the frozen decoder = trains BACKBONE LoRA at long horizon. Win =>
novel positive, upgrades P1 to method; loss => proves compounding drift not anchor-fixable, strengthens diagnosis.
CHEAPEST DECIDER (~1-2 days, inference): freeze-adversarial RERANK of released streaming zoo -- if ranking CHANGES
under motion-gating, P1 is real; if null, fall back to P4. CONTRARIAN: frame freeze as GENERAL evaluation-integrity
("temporal-consistency metrics are satisfiable by motion suppression"), bigger tent. HONEST: subfield saturated +
moving monthly + ~6mo behind; win = convert clean negatives into a diagnosis/metric-integrity paper. Full doc =
reference/fable_project_rebrainstorm.md

**2026-07-03 -- Fable TTT head-to-head vs LaCT/Scal3R/TTT-Video (primary sources). CORRECTS our substrate narrative.**
Read our inner loop (video_memory.py:129-155 ascend <f_W(k),v>, :184-192 renorm) vs the papers' actual equations:
- INNER LOSS: NOT a difference. LaCT Eq.7 L=-f_W(k)^T v and Scal3R L=sum -f_W(k_i)^T v_i are the SAME dot-ascent
  we use (both descend the negative dot = ascend the dot). Our earlier "degenerate inner objective / switch to
  ||v-f_W(k)||^2" idea is WRONG -- MSE would make us UNLIKE the working papers. why_ours_fails #4/#6 overstated this.
- NORM RULE: NOT a difference. LaCT update = L2-Normalize(W - Muon(g)) -- it ALSO projects onto a fixed-norm shell
  every step; norm-conservation/no-accumulation is STANDARD in the TTT-linear-attn line, not our Scal3R graft. The
  "churn statenorm/init~1.0" is exactly LaCT's design. So the SUBSTRATE (dot-ascent + norm-conserving + Muon +
  additive residual) is IDENTICAL to systems that WORK; it is NOT our bug.
- WHAT IS DIFFERENT (ranked): (#1, load-bearing) the OUTER OBJECTIVE. LaCT/Scal3R/TTT-Video supervise with
  RECONSTRUCTION of the REAL signal (real video / 3D geom) over LONG horizons -> storing the input is how you lower
  the loss -> the same fixed-norm rotation encodes RECALL. We supervise with DMD (distribution-matching, per-frame,
  MEMORYLESS bidirectional teacher, ~21f) which has NO per-sample target -> storage is never rewarded, and the read
  can only nudge the per-frame loss via local realism = the identity corruption we measured. Same shell, opposite
  content, because supervision is opposite. (#2) backbone co-trained WITH TTT in the loop in all 3 (LaCT e2e; Scal3R
  joint bb 1e-5/fast 1e-4; TTT-Video fine-tunes self-attn+gates); ours near-frozen AND DMD gives even the LoRA no
  storage gradient (E1 mem-ZERO=baseline proves the LoRA learned nothing memory-useful). (#3) horizon (their recon
  spans long seq / TTT-Video 3->63s curriculum; ours 21f). Additive channel / read-write noise mismatch = secondary.
- VERDICT: NO distinct fix. The mechanism is already correct (=LaCT); what's missing is a storage-REWARDING outer
  objective + co-training + horizon. DMD STRUCTURALLY cannot provide the reconstruction target (no per-sample target),
  so any fix must ADD a recon/recall term (=MemGen history-loss / TTT-Video denoise) + unfreeze read-consumers
  (=co-train) + long horizon (=VideoSSM/TTT-Video). All three = the published recipe. The only novel-shaped ingredient
  (on-policy self-supervised recall on the student's own clean-x0 latents inside the DMD rollout) is insufficient alone
  and still needs co-train+horizon -> collapses to MemGen+VideoSSM. TTT-under-DMD-on-frozen-backbone is STRUCTURALLY
  dead; the contribution is the diagnosis (correct substrate + WRONG OBJECTIVE), not a rescue.
Sources: LaCT arXiv 2505.23884 (Eq.7, update rule); Scal3R 2604.08542 (loss + joint bb/GCM train); TTT-Video 2504.05298.

## Section 8 log -- Fable code audit (2026-07-03)

4-agent correctness audit of the whole MAG codebase. Full doc: reference/fable_code_audit.md.

Verdict on "is the negative real or an artifact":
- Plain-inference attribution (memory hurts identity; MEM_ZERO ablation) = SOLID (read path clean, memory invoked end-to-end). Modulo re-confirming the MEM_ZERO_ALL value used (it is a frame threshold, not a boolean).
- Read-only negative = clean.
- Write-training / cross-chunk / switch arms = confounded or void. NOT clean evidence that a trained write can't work.
- No sign flip / gate mis-init / renorm / injection / detach bug on the read path. Core write math = faithful LaCT/Scal3R.

Critical bugs: (1) id_loss_weight crash in bptt>1 branch (2 agents); (2) cross-chunk BPTT silently severed (last-block write no_grad) -> realk2 void; (3) checkpoint-recompute clobbers fast-weight state during write-training backward -> e2_write3L/realk2 confounded (contested A-vs-B, needs a 5-min state-snapshot check); (4) switch+interactive inference bypass memory -> all switch evals void.
Major: multi-GPU memory never synced across ranks; default config ran train_write OFF; text-dropout bypassed by cross-attn cache; global_sink always False in switch training; train() swallows all exceptions; MEM_ZERO_ALL is a threshold; frames_since_switch only in switch chunk; auto_resume overridden; muon=0 mislabeled (NS normalizes unconditionally).
Design note (not a bug): write magnitude fixed/unlearnable under Muon -> can't write less on garbage chunks -> a concrete mechanistic candidate for "memory injects harm", aligns with the objective-level diagnosis.

Implication for the mechanism verdict: unchanged at the theory level (objective can't reward storage on a frozen/light backbone), but the empirical "trained write doesn't help" is weaker than treated -- e2_write3L/realk2 need a clean re-run (fix CRIT-2/3, MAJ-1) before any strong "mechanism dead" claim rests on them. The read-only + plain-inference negatives remain citable.

## Section 9 -- Clean E1 re-run result (2026-07-05, DECISIVE)

After the 4-round audit + fixes (all runtime-validated), the clean E1 re-run (e1_rerun_clean, byte-identical config to the confounded e1_overfit_v1 except the fixes + 3 GPU: synced memory MAJ-1, un-clobbered fast-weight state CRIT-3, live text-dropout MAJ-3; bptt_window=2, grad_tail_W=1, train_write, id_loss=1.0, text_dropout=0.4). 350 steps, 3 GPU, clean finish, ck150+ck300.

Eval = mem-ON vs mem-ZERO identity drift (DINOv2 ruler, 480-frame videos, 8 e0_dynamic prompts, MEM_ZERO_ALL=100000 ablation):
- ck150: mem-ON drift = 1.168, mem-ZERO = 0.226
- ck300: mem-ON drift = 1.161, mem-ZERO = 0.229

The memory STILL destroys identity (~1.16 drift ~ a fully different subject, vs ~0.23 without), IDENTICAL to the confounded original (1.1-1.2). So the "mechanism dead" verdict is NOT a code artifact -- it is a real property of the mechanism. The fixes were necessary to make the test valid (rule out the 3 confounds), and the valid test confirms: the trained-write Muon fast-weight memory under DMD is toxic to identity, even with a live id_loss=1.0 objective (the toxic injection overwhelms the identity term). Consistent with reference/why_ours_fails_vs_papers.md (DMD can't reward storage, decoupled read magnitude, near-frozen backbone).

Last untested variable: cross-chunk write training (grad_tail_W=2, newly enabled by CRIT-2) -- launched as e1_rerun_crosschunk. Tests whether training the write for its actual long-range purpose (across chunk boundaries) helps. If it also fails, the mechanism is conclusively dead.
UPDATE: the cross-chunk arm OOMs even on 4x48GB (activation-bound) as configured -- untested, see Section 10 for its disposition.

## Section 10 -- Fable mechanism-life re-brainstorm on the clean negative (2026-07-05)

Question posed: with the confounds removed and the negative confirmed clean (Section 9), is any single
fast-weight variant worth a final decisive shot? Full candidate-by-candidate reasoning delivered in-session;
summary of record:

Candidates assessed (flip mechanism / P(flip) / novelty-if-flipped):
- (A) gated-delta closed-form swap: fixes NS-bf16 + no-abstention + BPTT-free retention, ~15% alone
  (VideoSSM's success needs their full co-train recipe), novelty ~0 (= VideoSSM). Table row only.
- (B) contrastive re-ID on pre-gate o_hat: gives memory a DMD-untradeable storage job, ~10% (gate/injection
  still steered by DMD alone -> predicted silent failure: stored-but-corrupted-in-delivery). Run the 1h
  o_hat decodability PROBE, not the training run.
- (C) GRPO identity reward on the memory: only candidate aimed at the horizon constraint, ~10-15% (reward
  hacking -> freeze; 480-frame credit onto a net-toxic module; RAVEN x Identity-GRPO = low novelty). If GRPO
  ever runs, it is on the LoRA WITHOUT the fast-weight (= the memory-agnostic shot, not a rescue).
- (D) full backbone co-train: highest P(~30-40%) removal of net-toxicity, novelty exactly 0 (= TTT-Video/
  VideoSSM recipe), compute marginal. Reject.
- (E) hard-bound read magnitude: cannot flip (bounded tanh already live in the clean run; converts failure
  mode, doesn't remove it), ~2%. Run only as the 1h dose-response diagnosis (alpha sweep 0..1 on ck300).
- (F) learned write abstention: real defect (code-confirmed: NS spectrum-flattening + renorm make write
  magnitude content-independent), ~5% (toxicity is read/injection-dominated from chunk 0; collapses into A).
- (G) cross-chunk grad tail: 5-10%, OOM-blocked; keep ONLY as kill-completeness, hard-capped at 2 days of
  un-OOM engineering (Muon-off tail + ckpt_write + 1 write/chunk); else the measured OOM is the citable line.
- (H) attention-token injection (memory emits K/V; optionally MemGen-style offline recon pretrain):
  highest genuine flip odds ~25-35%, but the probability comes precisely from becoming retrieval/compressed-KV
  (= MemGen with a TTT encoder; if it works, plain retrieval matches and the fast-weight has no job).

THE PATTERN (the actual finding): P(flip) rises monotonically with how much of the published recipe is
adopted (co-train, reconstruction-family objective, closed-form retention, attention injection) and novelty
falls at the same rate. The intersection "plausibly flips" AND "still ours" is EMPTY. Three independent kill
layers now: (1) our ablation-clean negative with objective+gradient+hygiene fixed and code confounds removed
(drift 1.16 vs 0.23, invariant to every fix); (2) theory (DMD has no per-sample target -> storage never
rewarded; frozen consumer); (3) field convergence (all working systems = one of three published escapes).

VERDICT: mechanism settled-DEAD for any novel positive. No single-shot variant clears the bar. Stop grafting
memory; posture is forensic (autopsy), not clinical (rescue).

RANKED MOVES: (1) diagnosis-hardening this week, ~2-3 GPU-days: dose-response alpha sweep + o_hat
decodability/state-recall probe on ck300 (bulletproofs the centerpiece figure; gate01-750 stays the freeze
adversary). (2) optional cross-chunk coffin nail, capped 2 days engineering. (3) the ONE positive shot stays
memory-agnostic (long-range identity anchoring via no-grad rollout on LoRA; GRPO flavor only if supervised
shows signal). (4) gated-delta / attention-token = substrate-table rows at most; prefer citing VideoSSM/
MemGen over reimplementing. (5) never run: full co-train as method, GRPO-on-fast-weight, contrastive-o_hat
training, write-abstention, further hygiene tuning.

## Section 11 -- Fable seam verdict: is re-ID-after-eviction still a distinct seam? (2026-07-05)

Consult question: after the clean E1 re-run confirmed the memory is toxic (drift 1.16 vs 0.23, real property,
not artifact), is "subject re-ID after eviction" still a distinct, defensible, un-preempted research seam, and
what is the minimal decisive experiment? Fresh checks this session: VideoSSM (2512.04519) and MemGen (2512.18741)
have NO public code/checkpoints as of 2026-07-05 (both "to be released"); no new true-eviction re-ID benchmark
landed since the 07-03 literature pass.

### Adversarial preemption map (cell by cell)

- VideoSSM: owns "learned memory METHOD for streaming long video" on our exact stack. Evaluates PLAIN-LONG only
  (VBench + DD + user study). No leave-and-return protocol, no eviction control, no embedding re-ID. Whether its
  state performs RECALL (vs context smoothing) is unmeasured -- and our objective-level diagnosis (truncated ~5s
  DMD window rewards nothing beyond it) predicts it does not. Not measurable by us (no ckpt).
- MemGen/MAG-Bench: owns "revisit benchmark + text-reliance diagnosis". But MAG-Bench = CAMERA/SCENE revisit,
  pixel metrics (PSNR/SSIM/LPIPS), geometric, no subjects, no embedding identity, no motion gating --
  freeze/loop-gameable by construction. Subject re-ID after a scene change is non-geometric content addressing;
  pixel metrics cannot even score it (subject returns in a new pose/position).
- NarraStream/IAMFlow: closest TASK overlap ("identity across prompt transitions"). LLM-judged, training-free,
  60s, no eviction verification, no motion gate. They own the headline; they do not own the rigor cell.
- MBench: memory benchmark for INTERACTIVE WORLD MODELS (VLM+rule scoring). Different regime; plants the
  "memory benchmark" flag; reviewers will conflate -- positioning must differentiate explicitly.
- RAVEN / Identity-GRPO: close the GRPO method cells. World-model revisit line (WorldMem/VMem/CaM): geometry-keyed.

### What is literally still unclaimed (and its honest size)

The unclaimed cell: **verified-eviction, embedding-scored, motion-and-absence-gated subject re-ID in streaming
distilled T2V, with a sink-retention decomposition.** Real, but NARROW, and it is a MEASUREMENT cell, not a
method or benchmark-product cell. Three adjacent benchmarks landed within ~6 weeks; as a standalone product we
lose the race (6 months behind, 1.3B, 3 GPUs). It survives ONLY as the spine instrument of P1
(diagnosis + metric-integrity), where the differentiators are rigor-shaped, not scale-shaped:
1. Eviction VERIFIED by construction AND detection. Key point others miss: with global_sink, the sink holds the
   first frames FOREVER, so "B-span > window" does NOT guarantee eviction. Two arms: (i) late-entry/true-eviction
   (subject-free establishing shot fills the sink; subject enters after; B evicts all subject evidence) vs
   (ii) anchor-in-sink (subject in frame 0, retained forever) -- decomposes "evidence gone" from "evidence
   retained but unused" (E0 already showed retained-anchor-not-used in plain-long).
2. Freeze-adversary validation: our toxic checkpoints (gate01-750, e1 ck300) are a stable of in-house Goodharters
   the metric must rank BELOW baselines. Nobody else has such adversaries; our negatives become assets.
3. Oracle normalization + substituted-subject >3sigma separation + return-rate reported separately from re-ID.

Ceiling accepted: even if the gap is confirmed, FIXING it collapses into the published recipe (recall-rewarding
objective + co-train = MemGen/VideoSSM), so our ceiling is measurement + diagnosis + the one memory-agnostic
anchoring shot. Workshop/TMLR grade unless the shot hits. Mechanism stays closed; no more fast-weight GPU-weeks
(e1_rerun_crosschunk may finish as the final nail; nothing waits on it, nothing follows it).

### E-REID: the minimal decisive experiment (~10-12 days, inference-only)

Construction: 36 A-B-A prompt-switch triples, switch pipeline (native for baselines; ttt_cache bypass fix needed
only for adversary arms). Factors: eviction arm (late-entry vs anchor-in-sink) x return scene (same vs changed),
9 triples/cell; B = different-subject-same-class (hard case); B-span >= 2x(window+sink); A' prompts demand motion.
Metric: GroundingDINO/SAM crop -> ArcFace (faces) / DINOv2 (general); score = cos(A' return, A first-appearance
centroid), normalized (s - cross_subject_floor)/(intra_A_oracle - floor).
Gates (per-triple, exclusions reported): absence gate (subject-A detector quiet through B), RAFT motion floor
(>=0.5x matched stock), return gate (return-rate = its own statistic).
Metric validation BEFORE reading results: never-leaves vs substituted-subject >3sigma; freeze adversaries must
not outrank stock; ~40-pair human Spearman >=0.6.
Table: stock AF (sink3 + sink9 rows = cheap-trick), LongLive, Self-Forcing (public ckpts), our memory ckpts as
adversary rows; VideoSSM/MemGen noted unmeasurable (no public ckpts, checked 2026-07-05).
Free co-deliverable: score the SAME videos with VBench-SC vs the gated metric -> the rank-flip figure
(the 07-03 "cheapest decider" for the metric-integrity leg, zero extra generation).

### Pre-registered decision rule (no post-hoc arbitration)

- Baselines FAIL hard on late-entry (score ~ substituted-subject floor), anchor-in-sink also fails ->
  capability gap + evidence-unused finding -> GO: ship P1 with the re-ID spine; long-range anchoring shot
  (section 4 of the 07-03 rebrainstorm) as the upgrade path.
- Baselines PASS (~oracle) -> no job for memory in this regime -> ABANDON memory-for-video entirely, no
  re-litigation; salvage = short metric-integrity/Goodhart paper or redeploy the harness.
- MID (predicted most likely: text re-generation lands 0.4-0.7) -> the finding is "text-reliance masquerades
  as memory, quantified in embedding space on the lineage, with the sink decomposition"; ceiling
  workshop/TMLR; take the anchoring shot to try to lift it.

VERDICT: conditional GO -- the seam survives as a probe-spine inside P1, not as a benchmark product, never as
a method bet. E-REID is the last gate; every branch of it ships something, which is the only reason not to
abandon today.

## Section 12 -- Synthesis of the 4-agent re-brainstorm (2026-07-05)

Four Fable agents (mechanism-life, honest-paper, distinct-seam, assets/methodology) brainstormed on the clean decisive negative. They CONVERGE, hard.

**Consensus: the mechanism is settled-dead. Zero more training GPU-hours on the fast-weight. The project's asset is the autopsy, not the corpse.** The clean E1 negative (drift 1.16 vs 0.23, objective+gradient+hygiene+code all cleared, injection-attributed, monotone-worse, replicated across regimes) is a structural property, not a tuning failure. P(flip) of any variant rises exactly as fast as its novelty falls (co-train=TTT-Video, gated-delta=VideoSSM, recon=MemGen, GRPO=RAVEN); the "plausibly flips AND still ours" set is empty. Three independent kill layers: ablation-clean negative, theory (DMD never rewards storage on a frozen consumer), field convergence.

**The paper (P1) -- one combined paper, not three.** Spine = the audited clean negative (rigor). Headline = emergent Goodhart (the memory freezes video -> VBench-SC 0.94 + naive drift ruler both score it "better", a metric it never observed; then, once the freeze channel is closed, identity corruption those metrics equally cannot see). Credibility layer = the audit (3 confounds nearly shipped a false narrative; fixed + runtime-validated; negative reproduced exactly) -> methods appendix + a generalizable "your negative might be a bug" checklist. Unified thesis: *in distilled streaming video generation, the training objective and the evaluation metrics agree on the wrong thing (both reward temporal self-similarity), so a learnable memory converges to degeneracy the dashboard scores as health; the binding constraint is evaluation integrity, not memory architecture.* Draft title: "Consistent Because It's Frozen: Emergent Metric Gaming and an Audited Negative Result for Fast-Weight Memory in Distilled Streaming Video Generation." Venue: TMLR primary (rigorous negatives welcome, appendix room) + a condensed Goodhart leg to a NeurIPS 2026 eval/benchmark workshop for a timestamp.

**Cheap diagnosis-hardening this week (~2-3 GPU-days, mostly inference), locks the centerpiece:** (1) freeze-adversarial rerank of the released streaming zoo (stock AF, LongLive, Self-Forcing + our adversary ckpts) under VBench-SC vs a motion-gated identity metric -- THE experiment that decides headline order (rank-flip => Goodhart headline; null => negative headline, same spine); (2) dose-response alpha sweep {0,0.1,0.25,0.5,1.0} -> toxicity as a curve; (3) state-decodability / o_hat probe on ck300 -> "there is no memory in the memory" sentence.

**E-REID (~10-12 days, inference-only): the re-ID-after-eviction seam survives ONLY as the spine instrument of P1, never again a method/benchmark bet** (we lose the standalone-benchmark race from 6 months behind at 1.3B/3GPU vs VideoSSM/MemGen/NarraStream). Our three un-preempted differentiators: verified eviction DECOMPOSED (late-entry that truly evicts vs anchor-in-sink retained-but-unused -- nobody has this), in-house Goodhart adversary checkpoints, oracle-normalized + substituted-subject scoring. A-B-A triples, ArcFace/DINO on SAM crops, absence+motion+return gates, metric validated (>3 sigma freeze-adversary separation, human Spearman >=0.6) BEFORE reading results, pre-registered decision rule (baselines fail hard=GO; pass=abandon memory-for-video, no appeal; mid=text-reliance-masquerades-as-memory paper, most likely).

**The one positive shot stays memory-agnostic:** long-range identity anchoring via no-grad rollout on the LoRA (reuse the E1 DINO-identity-loss code, which outlives the dead memory), one grad chunk at frame N pulled toward the cached chunk-0 identity. Timeboxed; the paper must NOT depend on it; it is the only path diagnosis -> method.

**Drop / never run:** the mechanism and all substrate swaps (gated-delta/attention-token = table rows at most), full co-train as method, GRPO-on-fast-weight, contrastive-o_hat training, write-abstention, more hygiene tuning, the standalone re-ID benchmark race. **W=2 cross-chunk = a DATUM, not a rescue:** once CRIT-2 makes the gradient real, the graph OOMs at 4x48GB (the old "realk2 fits 24GB" was the severed-gradient artifact) -> this is the citable line explaining why the field escaped via closed-form/RL not BPTT; one cheap alloc-flag retry at most, else report the OOM.

**Process lesson for next project:** week-zero inference-only; every metric gets an adversary it must fail before it gates anything; module-invocation + ablation-fires + grad-reach proofs at integration day; fail loud; max one departure from a published working recipe; positive control (toy write-evict-read) before grafting; monthly preemption sweep as a pre-build gate.

## Section 13 -- The confounded-negative reversal + the clean-run fix suite (2026-07-05)

A 4-agent adversarial re-audit of the E1-clean negative (requested after the strategy brainstorm) found it CONFOUNDED, overturning "ship the paper now":
- It is CATASTROPHIC COLLAPSE (video -> black noise at ~2s / latent frame ~5-9, after 1-3 writes), NOT graded identity drift; 1.16 is the ruler's ceiling on a dead video.
- The memory CORE w0/w1/w2 was NEVER trained (bit-identical across ckpts; the seed grad path is severed by chunk-0 no_grad in the bptt window). So it was a RANDOM fast-weight, not a trained memory.
- The id_loss was VACUOUS: anchored on chunk-0's post-collapse TAIL frames (id_cos saturated 1.0), and gated off ~56% by the no-op same-prompt switch (has_switched).
- Toxicity mechanism: single_norm dropped o_norm -> uncapped read magnitude (a write-count integrator, never a relevance signal) -> runaway feedback -> black attractor.

POSITIVE CONTROL (reference/e1_positive_control.py, associative store+retrieve, read-at-key): the Muon+renorm substrate CAN store and retrieve past the 13-write eviction gap (retrieval 0.59 vs chance 0.006 at gap 13; decays 0.92->0.51 over gap 1->20 = a clean memory-horizon curve). So the mechanism is NOT dead at the storage level; the confounds (untrained seed, vacuous id_loss, uncapped-read collapse) are fixable => the clean re-run is genuinely warranted.

FIX SUITE (implemented + a 2-round Fable re-audit reached CONSENSUS "safe to launch"):
- Anti-collapse: `bounded_mod` flag tanh-bounds scale/shift (video_memory + threading), config single_norm=false (restore o_norm). Injection amplitude now state-independent.
- Seed training: `train_seed_every` seed steps put chunk 0 in-graph (init_state differentiable) so w0/w1/w2 train; per-chunk backward with `_clear_cache_gradients()` severing the KV grad link; normalized by _n_grad. Default (train_seed_every=0) byte-identical.
- Id objective: anchor = chunk-0 HEAD (pre-collapse) not tail; switch_choices=[1000] beyond the 81-frame video => switch never fires => id_loss every chunk (chooser no longer crashes on all-out-of-range).
- Config e1_rerun2.yaml: plateau 2000 steps, vis_interval=50, assert_memory_drift=true, text_dropout 0.1, streaming_max_length 81, gate_init 0.01.
- Deferred (noted, not blockers): fp32 master weights (q_norm/k_norm bf16 quantization), randomized-K horizon (id supervises ~chunk 1 only), frozen-seed control arm (needed to attribute any gain to the seed vs the extra chunk-0 supervision).

Prediction (Fable, DMD-only recipe): bounded+trained memory is most likely (i) do-no-harm/inert ~0.80, (iii) helps ~0.12. So the paper story likely shifts from "toxic" to "inert/no-job" -- weaker but honest and un-confounded.

## Section 14 -- CLEAN ablation result, e1_rerun2 step-250 checkpoint (2026-07-06)

The clean run RAN (both OOMs fixed: seed-step grad-residency + vis persistent-cache, both config-preserving; see memory e1-seedstep-oom-fix). No collapse -- video stays coherent through 60-240s (unlike the confounded run). Training healthy to step 500+, id/tv trains (0 skips), gate opens 0.15->1.0, memout 0.01->0.34 (moderate, not the toxic 1.26).

FIRST un-confounded mem-ON vs mem-OFF ablation (same step-250 ckpt, MEM_ZERO_ALL toggle, 36 trained memory keys VERIFIED loaded, plain CausalInferencePipeline, 8 e0_dynamic prompts x 240 frames, DINOv2 drift ruler + RAFT motion):
- mem-ON mean late-drift = 0.501 vs mem-OFF = 0.204 (drift 1.0 = as different as another subject). Memory HELPS 0 / HURTS 8 / tie 0.
- NOT a motion artifact: mem-ON moves ~2x LESS than mem-OFF on every prompt (jog 2.6 vs 6.6, skate 5.1 vs 53) yet drifts 2x MORE. Lower motion should lower drift; it is higher => real appearance drift (slow global wash-out/style-shift RAFT misses, DINO catches). No video frozen (motion_floor guard clean).
- Visual (montage) agrees: mem-OFF = stable subject + dynamic scene; mem-ON = appearance drift, a 2nd subject appears (boy), less motion.

VERDICT: the memory READ mildly HURTS identity at step 250 -- reproduces the "read toxic / no-job" finding WITHOUT the earlier code confounds and WITHOUT collapse. Contradicts the "inert do-no-harm ~0.80" prediction: it is do-HARM (mild), not inert. Caveats: (1) step 250 = early (12.5%), re-eval at plateau step 1000+ pending (watcher running); direction is unambiguous and consistent with all prior evidence. (2) This is PLAIN DRIFT only; the re-ID/revisit niche (subject leaves+returns) is untested -- the last live card. Files: reference/runs/e1_rerun/eval250/ (configs, videos, drift_out/drift_results.json + drift_curves.png).

CONFIRMED at step 1000 (2026-07-06, same clean ablation, reference/runs/e1_rerun/eval1000/): mem-ON mean late-drift 0.340 vs mem-OFF 0.132, HURTS 8/8 again. Cleaner than step 250: motion is now MIXED (mem-ON often moves MORE), and the killer cases have mem-OFF moving much more yet drifting less (skate OFF mot 46.8 drift 0.125 vs ON mot 9.4 drift 0.389; martial OFF 23.8/0.145 vs ON 5.6/0.450) => mem-OFF is both more dynamic AND more identity-stable, motion artifact fully ruled out. Both arms' absolute drift DROPPED 250->1000 (ON 0.50->0.34, OFF 0.20->0.13: training improves overall identity retention) but the mem-ON-worse gap does NOT close. CONCLUSION for plain drift: memory has no job, it hurts -- clean, confirmed at two checkpoints. Only the re-ID/revisit probe remains untested.

## Section 15 -- REVISED: the "hurts" was a near-DC injection artifact; content is INERT (2026-07-06)

The mem-ON video is dark + has a diagonal cross-hatch/woven mesh (mem-OFF is clean). 3-agent Fable code audit of the injection path:
- NO code/layout/reshape bug. Layout agent traced every tensor op: memory read is pointwise per token, write sums over l (order-invariant), grid never materialized -> a layout bug is impossible. Config e1_rerun2 == eval1000 verified. 36 memory keys loaded.
- Root cause (magnitude agent INSTRUMENTED the step-1000 ckpt at inference, darkprobe): the injection is ~90% a token-CONSTANT near-DC bias (dc_frac 0.88-0.99), fixed-RMS (o_norm+a_norm pin it regardless of read relevance), worth 25-45% of the residual stream at 3 depths. DC component = the darkening; the coherent direction printed through the 30x52 patch grid at latent Nyquist = the +-45deg cross-hatch after VAE 8x decode.
- TWO metric corrections: (1) `gatew_b*` telemetry = Frobenius norm of proj.weight's gate third (a WEIGHT norm), NOT the gate value; real tanh(gate) = 0.08-0.18 UNSATURATED (gate does NOT open to 1.0; bounded_gate works). (2) `memout_b*` "contribution" was measuring this DC bias, not content. So all prior "gate opens, memory contributes ~0.5" readings (incl. [[gate01-scal3r-gate-opens]]) were the DC bias.

DECISIVE TEST (env MEM_DC_REMOVE=1, one line `out - out.mean(dim=1)` in video_memory.py _project_output): removing the token-mean fixes the artifact ENTIRELY -- mem-DC video is clean, matches mem-OFF. 3-arm DINOv2 drift (eval1000, 8 prompts x 240f): mem-ON 0.340, mem-DC 0.135, mem-OFF 0.132. So the +0.208 "hurts" was ~97% the DC artifact; mem-DC vs mem-OFF = +0.004 (4 better/2 worse/2 tie) = NEUTRAL.

## Section 16 -- e2_amp amplitude fix (rel_norm+dc_remove+bptt4), step-250 eval (2026-07-06)

User overrode the "no-job / probe-first" consensus to try to improve PLAIN drift, minimally. Built the best defensible shot: `rel_norm` (chunk-scalar RMS norm on the read -> per-token relative magnitude survives, write-count growth cancels/chunk; carrier = raw read, a_norm dropped) + `dc_remove` (subtract token-mean, train+inference) + long horizon (bptt_window 4, id supervises chunk-3 = frames 57-74). Config reference/runs/e2_amp/e2_amp.yaml. 2 fresh Fable code audits: ALL CLEAN (threading runtime-proven all 3 layers rel_norm=True; ablation active by cmp; 1 low-sev eps-outside-sqrt bug fixed). Diagnostics (step 250): ampdisp ~0.25-0.42 (amplitude DOES couple per-token, the mechanism works), dcfrac ~0.84-0.93 (raw proj STILL ~90% DC, stripped by dc_remove), gatemean 0.01->~0.2 (real tanh gate, opens modestly).

RESULT (step-250 eval, mem-ON vs mem-OFF, 240f, 8 prompts): mem-ON DINO late-drift 0.753 vs mem-OFF 0.271, WORSE 8/8. mem-ON motion collapses (0.8-2 vs 3-16). VISUAL: mem-ON DISSOLVES the subject into abstract color blobs over the 240 frames; mem-OFF clean. VBench SC 0.952(on)/0.966(off), BC 0.956/0.963 -- barely distinguishes a DISSOLVING video from a clean one, re-confirming VBench is blind here. MECHANISM (math agent's flagged design-risk biting): rel_norm pins injection to fixed chunk-RMS across ALL chunks, so at 240f inference (>> 75f train horizon, ~80 OOD writes) the memory injects fixed-magnitude every chunk regardless of relevance -> accumulates -> dissolution. The amplitude fix injects MORE and that MORE melts long video.

CAVEAT: step 250 = early (12%); gate could learn to suppress; plateau (step 1000+) re-eval pending (watcher). But this is a WORSE early state than e1_rerun2 (0.75 vs 0.34) and the dissolution mechanism is structural. Leans toward confirming no-job + the rel_norm long-inference trade-off. Artifacts: reference/runs/e2_amp/eval250/.

REVISED VERDICT: the memory READ is do-no-harm / INERT on plain drift once the DC-bias artifact is removed -- NOT toxic. This matches Fable's original "do-no-harm ~0.80" prediction; the earlier "hurts 8/8" was the DC confound, and the "toxic collapse" of the confounded run was a different (code-bug) story. Real fix if pursued = retrain with amplitude coupled to the stream (drop a_norm / scale by stream RMS) or a magnitude-bearing gated-delta update; single_norm=True alone is insufficient (a_norm still renormalizes); the step-1000 proj can't be rescued at inference. The one untested niche is unchanged: re-ID/revisit. Artifacts: reference/runs/e1_rerun/eval1000/ (memdc arm, drift_out3, ablation_dcfix_viewer.html, darkprobe).


## Section 17 -- Fable verdict on the RELAXED criterion: is "active + do-no-harm" on plain drift achievable? (2026-07-07)

Question (user, after streamgate freeze): target relaxed from "+1-2% improvement" to CLEAN + ACTIVE (real
measurable on-vs-off effect, not mem-DC's +0.004) + DO-NO-HARM (drift <= off within noise) + DYNAMIC (motion
~ baseline). Is a stable active-neutral operating point achievable, or a contradiction?

VERDICT: not a logical contradiction, but every NON-HOLLOW version of it is. Inert is the honest ceiling on
plain drift; the relaxed bar is either already met vacuously (by mem-DC) or unmeetable. Next move = a
measurement, not a training run.

1. DECOMPOSITION OF "ACTIVE" (3 levels):
   - Level 0, trajectory divergence: the rollout is chaotic; ANY nonzero injection yields a different video,
     exactly like a seed change. mem-DC already injects a nonzero AC read (gate 0.08-0.18) with drift 0.135
     vs 0.132 -> the LETTER of the relaxed criterion is ALREADY satisfied by the existing checkpoint, and
     equally by re-seeding. Hollow by equivalence.
   - Level 1, consistent-direction style/detail effect: achievable, but nothing makes it state-dependent; a
     frozen-state / reset-every-chunk control would show the same effect = static adapter, not memory.
   - Level 2, state-dependent (the only sense in which the MEMORY is active): a state-dependent APPEARANCE
     effect either accumulates over the 80 OOD writes (e2_amp mechanism -> registers on the drift ruler; the
     DC experiment PROVED the DINO ruler sees pure style/color shifts, 0.13->0.34) or is temporally constant
     (= not state-dependent = the DC/artifact family). A state-dependent CONTENT effect is neutral only while
     indistinguishable from chaos; once its history-correlation is measurable it sits on the consistency
     axis = help (structurally absent) or harm. "Provably neutral" would require knowing the null space of
     DINO+RAFT+human through 20+ blocks + VAE over 80 writes; the last neutrality assumption (norm hygiene)
     leaked a 25-45%-of-stream bias.

2. THE ACTIVE-NEUTRAL LOSS IS GOODHART-BY-CONSTRUCTION: paired on/off rollouts + hinge reward on a minimum
   on-vs-off delta + identity/motion-matching penalties = "differ, but only where my rulers don't look" =
   training the model to find the blind spots of DINO+RAFT, i.e., the project's own headline finding
   weaponized against itself (the optimizer found DC/dissolution/freeze WITHOUT being paid to hide). Plus the
   neutrality terms are scored inside the 75f train window while failure lives at 240-480f -> e2_amp
   resurrected by construction. Even a win is hollow: perturbation-without-harm is true of a noise injector
   and a seed change; any strengthening of "active" that excludes them re-imports help-or-harm.

3. WHY INERT IS THE CEILING (3 layers): (a) trichotomy: on the measured axes an injection helps, harms, or is
   inert; active+neutral demands the rulers' null space, which above the noise floor contains only chaos.
   (b) information: with global_sink the memory's conditional information given (KV, sink, text) ~ 0 on plain
   drift (evicted middles redundant with anchor+window+text) -> the injection is a perturbation; perturbations
   in a chaotic rollout = invisible, reseeding (hollow), or bias -- and the only learned consistent bias
   directions the objective offers are the degenerate corners (DC / dissolve / freeze), which is exactly the
   6-iteration empirical record. (c) the user's OWN pre-registration: "corrected run can't hit +1% clean ->
   plain drift closed"; streamgate WAS the corrected run and froze. The relaxed bar moves goalposts downward
   to a twitch that is free. Active+useful exists only under an information monopoly = re-ID after TRUE
   eviction (E-REID, unchanged).

4. NEXT MOVE = SEED-NOISE-FLOOR CONTROL (zero training, ~1 GPU-day inference). Fact established this consult:
   every eval arm ever run is seed 42, single seed -> "measurable" and "within noise" are UNDEFINED; the
   +0.004 (inert call) and streamgate beta-delta +0.014 (active call) have no error bars. Run mem-OFF on the
   8 e0_dynamic prompts x 240f x 4-5 seeds; score existing mem-DC videos against them. Deliver (i) the error
   bar of the 8-prompt mean drift/motion (defines do-no-harm), (ii) the trajectory-divergence floor
   (seed-vs-seed DINO/LPIPS frame distance vs mem-DC-vs-off at matched seed). Pre-registered: (a) on-vs-off
   <= seed floor AND drift delta inside the band -> mem-DC strictly inert, relaxed criterion CLOSED on plain
   drift, pivot (E-REID or autopsy); (b) on-vs-off >> seed floor with drift+motion inside the band -> relaxed
   criterion ALREADY MET by the existing checkpoint, zero new training; then the reset-every-chunk control;
   prediction: effect survives reset = static adapter = hollow -> pivot anyway. Both branches terminate the
   plain-drift grind; NO seventh training iteration is justified before this measurement exists.
