# Review: muon_update_steps=0 + decay+cap (3-agent report, 2026-06-30)

## The change under review (active in run2_full / proposed for run3)
`muon_update_steps` 5 -> 0 (Muon/Newton-Schulz orthogonalization OFF; gradient only Frobenius-normalized at video_memory.py:107) + `use_decay=True` (decay_init=0.95, gated-delta `w=decay*w+grad` then cap row-norm to init) + `bounded_gate=True`, replacing the original Muon-5 + renorm-to-init (norm-conserving). The fast weight is a nonlinear SwiGLU MLP (w0,w1,w2); inner objective = ascend `<f_W(k),v>` (swiglu_fast_weight_gradient).

## Agent A — Muon lens
- Muon (Newton-Schulz, video_memory.py:82-119) produces ~US'V^T, S'~U(0.5,1.5): spectral whitening -> equalizes singular directions so the MLP write fills its hidden capacity.
- muon=0 -> Frobenius-normalized raw grad: top-singular-direction-dominated, step magnitude anisotropy-coupled -> practically rank-deficient write inside the fixed-norm shell.
- References: Scal3R (our source) default muon=5; LaCT Muon variant is their best; TTT-Video ADDS Muon "to improve stability and effectiveness" (else magnitude explosion / decayed memory); VideoSSM/Gated-DeltaNet have NO Muon BUT use a closed-form rank-1 delta on a MATRIX, not raw-grad SGD on an MLP -> "VideoSSM has no Muon" does not license muon=0 here.
- VERDICT: muon=0 is a real (mild-moderate) degradation, the one config no MLP-TTT reference uses; justified only as a temporary BPTT-tractability/speed crutch. Recommend muon=1 (try 2): one non-convergent NS step recovers most whitening at 1/5 the cost, far cleaner/shallower backward than 5 (avoids the bf16 2-4% gradient-error FIXME at line 103).

## Agent B — decay+cap lens
- Cap `w/(w.norm/(init+eps)).clamp(min=1.0)` math is CORRECT & safe: allows shrink (forget), prevents blow-up (pins at init), direction unchanged, no NaN/div0.
- Single shared SCALAR decay (sigmoid(decay_logit), broadcast to all columns + all 3 matrices) is far weaker than VideoSSM's per-channel, input-dependent gate `exp(g_t(x))` (Eq.8). It is fixed-rate EMA, not selective forgetting.
- decay_init=0.95 -> effective horizon ~1/(1-0.95)=20 chunks, too conservative for our 2-3 chunk horizon -> want ~0.6-0.8.
- CRITICAL Muon interaction: with muon=0, grad column-norm ~1/sqrt(dh)~0.013 vs init col-norm ~1 -> fixed point of `0.95*w+grad` ~ 0.013/0.05 = 0.26*init -> STATE NORM COLLAPSES to ~1/4 init and the CAP NEVER BINDS (forgetting becomes a scale accident, not learned). With muon=5, orthogonalized grad ~O(1) -> cap binds at init -> decay = mild ~5% leaky renorm.
- The smoke test ran muon=0 (decay_smoke_test.py:13) -> only exercised the collapse regime; "norm 74->52" = collapse in progress, not healthy forgetting.
- VERDICT: right primitive, correct math, but neutral-to-risky as configured. Lower decay_init (~0.6-0.8), make decay per-column/input-gated, scale-match the write, re-validate at muon=5 (not muon=0).

## Agent C — combination/coherence lens
- With muon off + decay+cap, the recurrence `w_t = d*w_{t-1} + Ghat_t` IS structurally a gated-delta memory with a nonlinear (SwiGLU) read. Coherent, NOT self-defeating. Still genuine (regularized) TTT: each chunk still ascends `<f_W(k),v>`; decay discounts PAST writes, added at full strength AFTER decay so it never subtracts the current step -> over-damped, realistic failure is UNDER-writing/inertness, not sabotage.
- Numerically stable: 4 independent bound points (unit-Frobenius increment, decay+cap, o_norm RMSNorm, a_norm+tanh gate). Smoke test proves only "doesn't blow up," not "works" (random inputs, no recall test, no outer loss, a norm can't see rank collapse).
- CORRECTION to Agent A: LaCT-VIDEO config runs Muon OFF (per our own reference/mlp_fastweight_plan.md note); TTT-Video DOES use Muon. So "all MLP-TTT use Muon" is half-right; muon=0 MLP is PARTIALLY precedented by LaCT-video. BUT the one published proof that MLP>linear (TTT-Video) uses Muon -> muon=0 biases the MLP arm toward the linear matrix it is meant to beat -> a muon=0 MLP-vs-matrix ablation would be CONFOUNDED (testing a rank-collapsed MLP, not an MLP).
- Wiring note: committed configs/ttt_memory.yaml lacks use_decay/muon_update_steps/bounded_gate -> defaults to Muon-5 / no-decay / unbounded gate. Our RUN config (_run2_full.yaml) DOES set them.
- Recommended config: use_decay=True, decay_init=0.95, muon_update_steps=2, bounded_gate=True, g_init=0.0 (truly inert do-no-harm), ckpt_write=True; wire all flags into YAML; optionally make decay input-dependent.

## Convergent recommendation (all 3 agents agree)
KEEP decay (sound primitive), RESTORE muon=1-2 (fixes write conditioning AND makes the decay cap bind at init instead of collapsing the state), LOWER decay_init to ~0.6-0.8 (real forgetting at our horizon), bounded gate g_init=0. This is a LOCAL refinement: it does NOT fix the deeper diagnosed blockers (no identity term in the DMD target; KV-cache+global-sink already supply identity; see reference/why_ours_fails_vs_papers.md and the "memory has no clear job" verdict).

## OPEN QUESTIONS for the consensus panel
1. FACT: does LaCT-VIDEO actually run Muon OFF (Agent C) or do all MLP-TTT use Muon (Agent A)? Verify against the LaCT repo (github a1600012888/LaCT) + TTT-Video. Is muon=0 precedented?
2. muon value for run3: 0, 1, 2, or 5?
3. decay_init: keep 0.95 or lower to 0.6-0.8?
4. Decision-relevance: does the muon value even matter NOW given gate~0 (do-no-harm) + the deeper blockers, or is it polishing a part that doesn't gate the result?

---

## CONSENSUS PANEL (2026-06-30, 3 agents: fact-checker / first-principles / contrarian)
All three vote **CONDITIONAL** on "muon 0 -> 1-2". They AGREE:

**Decision-relevance NOW = low.** With gate~0 (do-no-harm) the o_norm/a_norm strip the read magnitude and the gate multiplies by ~0, so muon=0 vs 2 changes NO current metric. Leave run2_full at muon=0.

**Corrections to the report above (important):**
1. muon=0 is PRECEDENTED, not unprecedented. Fact-checked from the repos: LaCT-video (`use_moun: false` in lact_ar_chunk3f_1.3B.yaml; ARFastWeightSwiGLU defaults use_moun=False) AND TTT-Video (plain GD `W<-W-eta*grad`, no Newton-Schulz) BOTH run Muon OFF. Scal3R (our source, muon=5) is 3D recon, not AR video. So Agent A's "all MLP-TTT use Muon" is wrong (2 of 3), and the "MLP>linear was proven WITH Muon" confound collapses (TTT-Video proved it at muon=0).
2. The real issue is a SCALE MISMATCH, not Muon-absence: we kept Scal3R's base_lr=1e-2 (calibrated for muon=5 O(1) grad) but switched to muon=0 (grad col-norm ~1/sqrt(dh)~0.013 vs init~1) AND added cap-to-init -> fixed point of `0.95*w+grad` ~ 0.04-0.26*init -> state collapses, cap NEVER binds. Two equivalent fixes: (a) muon=2 (grad back to O(1), cap binds), or (b) keep muon=0 + adopt LaCT recipe (drop cap-to-init, recalibrate lr/ttt_scale to no-Muon magnitude).
3. "Rank collapse" overstated: Frobenius-norm is rank-neutral; Newton-Schulz fixes CONDITIONING (equalizes singular values), not rank (can't create rank). The real harm of the collapse: small w0/w2 push silu into its near-linear regime -> the SwiGLU degenerates toward a linear matrix (expressivity loss = undercuts the MLP-vs-matrix novelty). o_norm RMS-normalizes the read so collapse does NOT shrink output amplitude directly.
4. muon=1 whitens only the bulk spectrum (quintic p(0.3..1.0)->0.7..1.2) and is BORDERLINE for binding the cap; muon=2 clears the ill-conditioned tail and reliably binds. So muon=2 if restored.
5. decay_init=0.95 -> ~20-chunk horizon, too high for our 2-3 chunk horizon -> lower to 0.6-0.8.

**Trigger to restore muon=2 + decay 0.6-0.8:** the FIRST run where the read carries signal -- whichever of (i) an identity-bearing/recall loss term is added to the DMD target, (ii) the MLP-vs-matrix expressivity ablation launches (else confounded by a rank/conditioning-collapsed MLP), (iii) the gate is force-opened (g_init>0 / clamp). NOT before; and NOT as a "fix" -- the bottleneck is the objective (no identity term) + the job (KV-cache 12 + global-sink already supply identity), per reference/why_ours_fails_vs_papers.md and the "memory has no clear job" verdict.

---

## (a) MUON=2 vs (b) ttt_scale/LaCT-recipe — 3-agent deliberation (2026-06-30)
Steelman-A pulled none; steelman-B pulled the ACTUAL LaCT repo code and corrected the framing. Key findings:

**The real bug = the UNCONDITIONAL Frobenius-normalize, not Muon-absence.** zeropower_via_newtonschulz5 is called
unconditionally (video_memory.py:180-182) and Frobenius-normalizes at :107 BEFORE the NS loop. So at muon=0 we STILL
wash out the lr-scaled magnitude, then don't orthogonalize -> collapse. LaCT-video (`use_moun: false`) simply NEVER
calls zeropower, so its lr-scaled grad keeps its natural magnitude. So muon=0 in OUR code != LaCT's muon=0.

**LaCT's `ttt_scale` is the READ GATE, not an update multiplier** (verified: ar_lact_swa_repeat.py ~977-985, a
zero-init Linear(dim,1)->SiLU per-token gate on the read) = EXACTLY our AdaLN gate (video_memory.py:373-380, already
have it). LaCT sets update magnitude via a learned per-token lr (lr_proj, base_lr=0.001, softplus) = our `self.lrs`
(:283-285, already have it) + renorm-to-pre-update (= our decay=None branch :184-192, already have it). So option (b)
imports NOTHING new -- it REMOVES Scal3R machinery (unconditional Frobenius-norm + decay + cap).

**Faithful (b1) = 3-line fix:** gate the NS call on `muon_update_steps>0` (bypass zeropower when 0 -> grad keeps its
lr-scaled magnitude, no collapse) + use the decay=None renorm-to-pre-update branch + recalibrate base_lr 1e-2 -> ~1e-3
+ keep the AdaLN read gate. Zero new params, removes the bf16 NS gradient-error (video_memory.py:103, which corrupts
exactly the write gradient), shallower BPTT, = the proven AR-video config.

**(a) vs (b) verdict:**
- (a) muon=2: 1 line, deterministic; steelman-A's true point = only Muon equalizes the singular spectrum (a scalar
  can't), spreading the write across the MLP hidden width. BUT spectrum-equalization is NOT necessary for an
  expressive MLP -- LaCT-video + TTT-Video prove it at muon=0 (magnitude via per-token lr, not orthogonalization).
  muon=2 is the 3D-recon (Scal3R) recipe.
- (b1): the AR-video recipe, cheaper, exact gradient, zero new params.
- DECISIVE for the MLP-vs-matrix novelty ablation: the matrix baseline (VideoSSM) is no-Muon, so the MLP arm MUST be
  no-Muon to isolate the functional form (else MLP-vs-matrix is confounded with Muon-vs-no-Muon). -> (b). Matches
  TTT-Video (the published MLP>linear proof, no Muon).

**CONSENSUS (all 3): defer-until-trigger, then (b1) faithful-LaCT as the default.** Now (gate~0) it changes no metric;
the bottleneck is the objective (no identity term) + KV+sink already supplies identity. At the first run where the
read carries signal (identity/recall loss / MLP-vs-matrix ablation / gate force-opened): use (b1). (a) muon=2 only as
a throwaway 1-line de-collapse for a quick general run. run3-muon2 (the current probe) tests (a) -- fine as a
"does de-collapsing open the gate" probe; if gate stays ~0 it confirms the objective (not read-conditioning) is the
blocker.
