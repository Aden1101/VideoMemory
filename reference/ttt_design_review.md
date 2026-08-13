# Verification addendum (main agent, 2026-06-25)

Produced by the `ttt-design-review` ultracode workflow (run wf_8f61a63b-2d0, 10 agents, ~650k tok). The 3 EXTERNAL sources each passed an adversarial citation-verifier (1 claim rejected total: a Scal3R "multi-head" detail). The OURS reader had NO separate verifier, so the main agent re-checked the load-bearing LOCAL claims by hand:

- **C3 (cross-chunk freed-graph) -- CONFIRMED real launch blocker.** `model/streaming_training.py::_clear_cache_gradients` (lines 700-724) detaches `kv_cache1` and `crossattn_cache` k/v but NEVER the TTT fast-weight state (the file has zero `ttt_cache` references). Under `train_write` the write runs in `enable_grad` with `detach_state=False` (`pipeline/streaming_training.py:76`), so the TTT state carries a grad_fn across chunks; `backward()` is once per chunk/step (`trainer/distillation.py:1139`; one `generate_next_chunk` per step, see comment :1119). => from the 2nd grad-bearing chunk, backward traverses the previous (already-freed) graph => RuntimeError. Never triggered before because read-only writes under `no_grad` (no graph). FIX = detach the TTT state at the chunk boundary (mirror the KV detach); this truncates BPTT to within-chunk (the intended window 24).
- **H2 (grad-norm blind) -- OVERSTATED by the report.** `generator_grad_norm` IS computed and logged: `trainer/distillation.py:1267-1269` and `:1328-1330` (`clip_grad_norm_`) overwrite the `0.0` placeholder at `:1145`, and `:1369` writes it. No fix needed (only confirm it reaches wandb, not just the SummaryWriter).
- **Scal3R inference-only** correction (training hard-disabled) -- accepted; it is the verifier-grounded core finding and removes any "Scal3R proves the write trains/generalizes" argument.

URLs (web-verified, see memory ttt-reference-sources): LaCT arXiv 2505.23884 / github a1600012888/LaCT ; TTT-Video arXiv 2504.05298 / github test-time-training/ttt-video-dit ; Scal3R arXiv 2604.08542 / github zju3dv/Scal3R.

---

# TTT memory design review (LaCT / TTT-Video / Scal3R vs ours)

Reconciled from the synthesis and the hostile critique. Where they disagreed, the hostile correction wins on facts (it re-verified the load-bearing code). Every external claim is cited by paper+location or by file:line. Anything not grounded in the verified findings is tagged `[UNVERIFIED]`.

---

## 1. Sources read (trust note)

The verified findings did **not** carry resolvable URLs, only paper names + section/line citations and our own `file:line`. I will not fabricate links.

```
SOURCE        | TYPE        | TRUST                                   | URL
--------------+-------------+-----------------------------------------+--------------------
Ours (code)   | local repo  | CONFIRMED (re-read by both reviewers)   | local file:line
LaCT (AR-vid) | paper+cfg   | CONFIRMED via cited sec/cfg lines       | [UNVERIFIED]
TTT-Video     | paper+code  | CONFIRMED via cited sec/Table2/ttt_mlp  | [UNVERIFIED]
Scal3R        | paper+code  | CONFIRMED; key correction below         | [UNVERIFIED]
```

Local files re-inspected (both reviewers): `wan/modules/video_memory.py`, `model/dmd.py`, `pipeline/streaming_training.py`, `model/streaming_training.py`, `trainer/distillation.py`, `configs/*phase2*`.

**Load-bearing correction (hostile beats synthesis):** Scal3R is **inference-only**; its training is hard-disabled — `forward` raises `RuntimeError('Release TTT only supports inference/eval mode.')` (Scal3R ttt_utils.py:475-476) and the aggregator asserts `not self.training` (Scal3R:640/673/700). Scal3R therefore provides **zero** evidence that BPTT-through-write *trains* stably or generalizes; it only shows the inference substrate runs, and its 990-frame number is carried by loop-closure + pose-graph optimization (Scal3R Sec4.4), not by the fast weight. Any argument that leans on "Scal3R trains this at 1-32 chunks and evals 150-990" is a **factual error** and is struck from this report.

---

## 2. Comparison matrix (cited)

```
DIM | OURS                          | LaCT (AR-video)            | TTT-Video                  | Scal3R (INFERENCE-ONLY)
----+-------------------------------+----------------------------+----------------------------+----------------------------
1   | fix AR identity DRIFT vs t;   | <5% FLOPs giant chunks;    | long-range global ctx,     | long-video 3D recon drift;
Obj | SwiGLU fast-wt @blk5          | fast-wt @inference         | local attn per 3s seg      | adapted sub-nets @TEST
    | vmem.py:199-207               | LaCT abs/Sec1              | TTTv README/3.2            | Scal3R abstract
----+-------------------------------+----------------------------+----------------------------+----------------------------
2   | (silu(q@w0)*(q@w2))@w1,       | w1@(silu(w0@x)*(w2@x))     | dual-form Z1->gelu->Z2     | (silu(q@w0)*(q@w2))@w1
READ| APPLY-FIRST, nh=1 hd=1536     | apply o=f_W(q)             | f=x+LN(fLin(x))            | apply o=f_W(q) Eq.4
    | vmem.py:122-126,374           | LaCT 162-65/Eq2,6          | TTTv ttt_mlp compute_mb    | Scal3R ttt_utils:232
----+-------------------------------+----------------------------+----------------------------+----------------------------
3   | dot-prod ascend, per-tok lr   | dot-prod L=-f(k)^T v;      | recon XV-XK; W-=eta*grad;  | dot-prod; raw->Muon->renorm
WRT | softplus base 1e-2; Muon NS=5 | renorm-init; Muon CODE     | LN-bwd; eta per-tok;       | Muon=5; base_lr 1e-2
    | renorm-to-init                | but use_moun=FALSE         | NO Muon                    |
    | vmem.py:129-188              | LaCT 116-153/Eq4,7,8       | TTTv Eq.1, ttt_mlp         | Scal3R 290-96,155-223
----+-------------------------------+----------------------------+----------------------------+----------------------------
4   | META: qkv,lr-net,proj,oqk     | META: q/k/v/o,lr,init w*   | META: W1/b1/W2/b2 init     | META: backbone,qkv,lr,proj
I/O | norm,INIT w0/w1/w2 ADAPT:state| ADAPT: per-batch state     | ADAPT: state               | RELEASE=inference-only
    | non-mem lr=0; vmem236-65;dist | LaCT 1011-12,863-65        | TTTv ttt_layer; train.md   | Scal3R 377-91,519
----+-------------------------------+----------------------------+----------------------------+----------------------------
5   | YES enable_grad+detach=False  | YES full BPTT, no detach   | YES grads-of-grads, scan,  | HARD-DISABLED (RuntimeError
BPTT| +ckpt; *cross-chunk NOT       | @compile (full seq)        | remat ckpt (full BPTT)     | training; inference only)
THRU| detached -> see §3 BUG        | LaCT 22,116-153            | TTTv 2.1; ttt_mlp scan     | Scal3R 475-76; aggr 640
----+-------------------------------+----------------------------+----------------------------+----------------------------
6   | write blk=3lf; chunk21/new18; | AR chunk=3lf; n_lat=21;    | mb b=64; CURRICULUM        | chunk60 ov30; TRAIN 1-32ch;
HOR | TRAIN cap 24lf(~5.8s);        | ONE fwd full seq           | 3/9/18/30/63s              | EVAL 150-990 fr (PGO-aided)
    | EVAL 240lf; phase2:49,66-78   | LaCT cfg 5,38,39,53        | TTTv Table2                | Scal3R base.yaml 11-12
----+-------------------------------+----------------------------+----------------------------+----------------------------
7   | AdaLN gate*(a_norm(1+sc)+sh)  | scalar gate ttt*silu(proj) | tanh(a)*TTT+X; 4 gates AdaLN| AdaLN gate; g_init 0.1
GATE| +zero proj_extra(agree/since/t)| +x_window; RMSNorm first  |                            |
    | vmem.py:344-55,277-310        | LaCT 978,983,916           | TTTv dit SSMGating;Eq6     | Scal3R 457-58; Eq.5
----+-------------------------------+----------------------------+----------------------------+----------------------------
8   | w*=1/sqrt(fan_in) NO sqrt2;   | randn/sqrt(fan_in) gain1;  | W~N(0,.02); alpha 0.1      | sqrt(2)/sqrt(fan_in);
INIT| gate-bias 0.1; proj zero      | ttt_scale_proj ZERO        | (tanh~.0997)               | g_init 0.1; proj zero
    | vmem.py:254-71,312-19         | LaCT 631-48,695-97         | TTTv configs; 3.1          | Scal3R 377-91,411-15
----+-------------------------------+----------------------------+----------------------------+----------------------------
9   | DMD distill 4-step; SELF-     | flow-match MSE; TEACHER-   | diffusion v-pred 1000stp;  | geometry regr L=lam*Lcam+
LOSS| ROLLOUT write (own output)    | FORCED clean-chunk write   | TEACHER-FORCED non-causal  | Ldpt+Lxyz; inf-only
    | dmd.py:161-227; pipe switch   | LaCT vid_flow 361-70,221   | TTTv App.A/4               | Scal3R Eq.11
----+-------------------------------+----------------------------+----------------------------+----------------------------
10  | fast-wt persist+renorm; gate  | renorm-init; o_norm; gate  | global TTT + BIDIRECTIONAL  | renorm + LOOP-CLOSURE +
DFT | since_switch/agreement        | LaCT Sec3.2                | + curric; TTTv Eq.7        | POSE-GRAPH OPT (geom)
    | vmem.py:374-88                |                            |                            | Scal3R Sec4.4 loop/pgo/
----+-------------------------------+----------------------------+----------------------------+----------------------------
11  | NOT_FOUND (no metric/numbers  | match full-attn val loss;  | +34 Elo vs 2nd (LMSys)     | beats MASt3R-SLAM; RRE/RTE/
RES | in code); baseline=mem-OFF    | beat SWA & Mamba+SWA       |                            | ATE KITTI/Spires
    | dbg_*+wandb only              | LaCT 5.3/Fig6              | TTTv 4.3                   | Scal3R Tbl1
----+-------------------------------+----------------------------+----------------------------+----------------------------
12  | CAUSAL AR; apply-first; TTT   | CAUSAL AR; interleave      | NON-causal bidir; local 3s | NON-causal; update-then-
CAU | cache pytree-LEAF for FSDP    | noisy/clean; write clean   | attn; chunked scan         | apply; GCS all-reduce
    | vmem.py:44-57; cm:556-90      | LaCT Eq12/4.3              | TTTv Eq12; ttt_mlp         | Scal3R scal3r.py:43-61
```

---

## 3. Where we diverge — JUSTIFIED / RISKY / LIKELY-BUG

**vs LaCT**
- Write on SELF-ROLLOUT output vs teacher-forced GT clean chunks — **RISKY (central novelty+risk)**. LaCT writes GT clean (vid_flow:221); we write our own 4-step denoised chunk (switch:243-263). Generation error is written into the fast weight and re-read = positive-feedback drift channel.
- Muon ON (5 iters) under BPTT vs LaCT-video `use_moun=false` (cfg:46-47) — **RISKY**. We backprop through bf16 Newton-Schulz whose own comment warns 2-4% grad error (vmem.py:103,107).
- Truncated cross-chunk BPTT vs full single-forward seq loop (cfg:53) — **RISKY / LIKELY-BUG** (§ below).
- Loss = DMD vs flow-matching — **JUSTIFIED** (base is a 4-step DMD student, dmd.py:161).
- gate_init 0.1 vs LaCT inert SiLU(0)=0 (695-97) — **RISKY (minor)**; step-0 ≠ baseline.

**vs TTT-Video**
- Fixed 5.8s train horizon vs CURRICULUM 3→63s (Table2) — **RISKY (central)**.
- Truncated/detached cross-chunk BPTT vs full-sequence BPTT (scan, remat) — **RISKY**.
- Causal, no reverse pass vs bidirectional (Eq.7) — **JUSTIFIED** (we are AR).
- Self-rollout vs teacher-forced write (Sec4.3) — **RISKY** (same channel as LaCT).

**vs Scal3R**
- We train the write (BPTT); Scal3R never runs the outer loop (inference-only, training hard-disabled, ttt_utils:475-76) — **our contribution; Scal3R is a code template, NOT empirical support**.
- Direct ports that AGREE — **JUSTIFIED**: Muon=5, base_lr=1e-2, g_init=0.1, nh=1, inter_multi=4 (Scal3R cfg ≈ phase2). These match *inference* settings only.
- init gain: dropped sqrt(2), use 1/sqrt(fan_in) (LaCT-style); Scal3R keeps sqrt(2) — **minor, internally consistent**.
- `allreduce_fn=None` vs Scal3R GCS grad all-reduce before Muon (dist_utils:42-48) — **RISKY (verify under FSDP/context-parallel)** (vmem.py:380,229).
- Long-horizon via loop-closure+PGO — **NOT portable** (geometry).

**LIKELY-BUG (launch blocker), grounded:**
```
train_write=True => detach_state forced False for every VideoMemoryLayer (streaming_training.py:69-77)
  => step() never detaches new fast-weight state (vmem.py:362-363,386-388)
KV / cross-attn caches ARE detached per chunk (model/streaming_training.py:711-721)
ttt state is NOT detached except full sequence reset (clear_kv_cache; 407-408)
backward() is per chunk (distillation.py:1139)
```
With `streaming_max_length=24`, chunk≈21 / new=18, a sequence spans a 2nd grad-bearing chunk (~3 frames) whose `swiglu_apply` reads w0/w1/w2 still attached to chunk-1's already-freed graph → `RuntimeError: backward through the graph a second time`. If instead only one grad-chunk runs per sequence, cross-chunk write-training **never happens**. Either way → **must fix before launch.**

---

## 4. Does training the write on a 24-frame window fix 120-240s drift? — honest answer

**No, not robustly as configured.** Both reviewers concur; the hostile critique removes the one optimistic crutch.

The struck optimism: the synthesis argued renorm-to-init boundedness (vmem.py:167-188) is "the same mechanism Scal3R trains at 1-32 chunks." **False** — Scal3R never trains it (inference-only). Numerical boundedness (norm capped) is **not** identity-preservation (task correctness). Bounded ≠ remembers-the-subject.

Three structural reasons it likely does not extrapolate:

1. **Credit-assignment depth.** After the required cross-chunk detach (§3), BPTT-through-write spans one chunk = ≤7 writes ≈ 21 latent frames. We never differentiate "write at frame 5 caused drift at frame 100." LaCT does full BPTT over its 21-frame train seq (cfg:53); TTT-Video does full BPTT up to 63s (ttt_mlp scan). They optimize the write *for the horizon they evaluate*; we optimize for 5s.

2. **Horizon gap × self-rollout.** Train 5.8s, eval 120-240s = 20-40×. TTT-Video closes this with a curriculum (3→63s, Table2); LaCT has train≈eval. Neither shows 40× zero-shot extrapolation. And both teacher-force the write (LaCT vid_flow:221; TTTv 4.3) while we self-rollout (switch:243-263) — only ≤24 frames of that compounding is ever in the graph.

3. **The objective does not point at identity.** DMD-switch is per-frame distribution matching: `0.5*MSE(x0,(x0-grad).detach())[gradient_mask]`, `grad=real-fake` computed under `no_grad` then detached (dmd.py:185-227). Gradient to memory = "make this masked frame more probable under the teacher," a per-frame marginal. Identity drift is a long-range *pairwise* property the per-frame score barely sees. There is **no identity / feature-matching / recall term anywhere** (dimension 11 = NOT_FOUND). Risk: memory is trained to be a denoiser, not a memory.

The mechanisms the references use for the long horizon, and portability:
- TTT-Video curriculum on segment length — **PORTABLE, highest value** (scaffold present-but-commented at streaming_training.py:391-396).
- LaCT/TTTv untruncated BPTT over the train seq — **partially portable** (VRAM-bound; proven 24-latent ceiling on 2×48GiB).
- Scal3R variable-length 1-32-chunk training — **portable in principle, but Scal3R itself never trains** (so no empirical backing).
- Scal3R loop-closure+PGO; TTT-Video bidirectional — **NOT portable** (geometry / non-causal).

---

## 5. Ranked risks + mitigations

**CRITICAL**
- **C1 — Objective doesn't target identity.** DMD per-frame, detached score, no consistency term (dmd.py:185-227; mask streaming_switch_training.py:113-116; dim 11 NOT_FOUND). → Add/measure a consistency signal (DINO/face cosine across the switch, or contrastive recall on `<f_W(k),v>`); at minimum ablate that the DMD grad w.r.t. memory params correlates with an identity metric on held-out rollouts.
- **C2 — 10-40× horizon extrapolation unsupported by all 3 sources.** LaCT train≈eval; TTTv needs curriculum; Scal3R inference-only + PGO-carried (verified §1). → Re-enable curriculum (streaming_training.py:391-396); add intermediate evals (48/96 latent fr) to *measure* the curve; never cite Scal3R 990 as evidence.
- **C3 — Cross-chunk freed-graph crash / silent no-cross-chunk-training (LAUNCH BLOCKER).** (vmem.py:362-363,386-388; streaming_training.py:69-80; KV detached 711-721 but ttt not; backward per chunk distillation.py:1139). → Detach `TTTBlockCache.state` at each chunk boundary; ≥3-chunk smoke run asserting no backward error before scale-out.

**HIGH**
- **H1 — BPTT through bf16 Newton-Schulz Muon.** Degree-5 poly composed 5× after `1/‖·‖`, bf16, self-admitted 2-4% grad error (vmem.py:82-119,103,107); LaCT-video disables Muon. → Ship `muon_update_steps=0` (and/or fp32 NS) A/B; clip memory-param grad norm; Muon-on is the risky arm, not default.
- **H2 — Grad-norm watchdog blind.** `generator_grad_norm` hard-coded `0.0` (distillation.py:1145). 1-line fix; gates everything. → Log real (and per-module memory) grad norm before launch.
- **H3 — No metric + non-matched baseline ⇒ a "win" may be undemonstrable.** `gate_init=0.1` ⇒ ours≠baseline at step 0 (vmem.py:267-271; phase2_write.yaml:70); no identity metric in code. DMD loss ≤ baseline ≠ better identity (see C1). → Use `g_init=0.0` for a clean step-0==baseline arm OR train a matched mem-OFF checkpoint for identical steps; define+wire the identity metric now (KV-probe / DINO tooling in MEMORY); pre-register the Δ that counts as a win at 120s/240s.
- **H4 — Self-rollout write compounds drift, only ~5s in graph.** Writes own 4-step output (switch:243-263) vs teacher-forced refs; nothing cited shows convergence. → A/B teacher-forced-write vs self-rollout-write on short horizon; if self-rollout diverges faster, curriculum it in after a TF warm-up.

**MEDIUM**
- **M1 — Scal3R assumptions non-transferable** (bidir, update-then-apply, geometry loss, PGO-carried). Treat as code template, not empirical support.
- **M2 — `allreduce_fn=None` under FSDP** (vmem.py:380,229 vs Scal3R dist_utils:42-48). Confirm single-stream or wire all-reduce before Muon.
- **M3 — Anti-drift signals never trained over their target horizon.** `frames_since_switch` designed for t≫24 (video_adapted_gating.md:35-72) but `proj_extra` zero-init (vmem.py:287-289) and only ≤24 fr in graph. → Verify `proj_extra` receives non-trivial gradient; else it's decorative.
- **M4 — Drift assert false-trip / masks problems.** `assert max_drift>1.0` at g_init 0.1 (distillation.py:1545-1578); drift-from-init ≠ memory-used. → Guard at step 0; pair with `dbg_out_rms` + gate magnitude.
- **M5 — Stale doc.** `reference/write_path_research_report.md` says write is unconditional `no_grad`/zero grad — contradicted by `train_write`/`detach_state=False` (streaming_training.py:68-80; phase2_write.yaml:40). → Update before it misleads. (Scal3R "multi-head" was the only REJECTED external item; not load-bearing here.)

---

## 6. GO / NO-GO + pre-launch checklist

**Verdict: NO-GO as configured. Conditional GO for a SHORT diagnostic run only, after C3+H2 (and ideally H3). Do NOT commit GPU-weeks until (i) grad-norm log is real [H2], (ii) an identity metric exists + a matched mem-OFF baseline is run [H3/C1], (iii) teacher-forced-vs-self-rollout A/B [H4] shows convergence.**

Ordered checklist:
```
[1] BLOCKER  Detach TTTBlockCache.state at each chunk boundary (mirror KV detach).
             vmem.py:386-388; model/streaming_training.py:711-721
[2] BLOCKER  Log REAL generator_grad_norm (+ per-module mem grad). distillation.py:1145
[3] BLOCKER  Smoke run >=3 chunks, 1 GPU, train_write=true; assert no backward error. dist:1139
[4] HIGH     Define + wire identity-drift metric (DINO/face cosine across switch); pre-register
             win threshold @120s/240s. (dim 11 currently NOT_FOUND)
[5] HIGH     Add matched baseline: g_init=0.0 arm OR mem-OFF ckpt at equal steps. vmem.py:267-271
[6] HIGH     A/B Muon: muon_update_steps=0 (and/or fp32 NS) + mem-grad clip. vmem.py:103
[7] HIGH     A/B teacher-forced write vs self-rollout write, short horizon. switch:243-263
[8] MED      Re-enable horizon curriculum, gate bumps on drift metric. streaming_training.py:391-396
[9] MED      Confirm allreduce_fn need under FSDP; wire or document None. vmem.py:380
[10] MED     Guard drift assert at step 0. distillation.py:1545-1578
[11] MED     Update reference/write_path_research_report.md (no_grad claim is stale).
```

---

## 7. wandb watch-list (early signals)

```
SIGNAL                                  | HEALTHY                          | FAILURE / MEANING
----------------------------------------+----------------------------------+-----------------------------
generator_grad_norm (after fix #2)      | finite, stable                   | NaN/spike = Muon-bf16 / BPTT instab
memory/* fast-wt drift vs init          | rises >1.0 then PLATEAUS         | monotone climb = diverge; pinned = dead
dbg_out_rms                             | small@start(g0.1), grows w/ gate | ->0 gated off; explode = unstable
dbg_read_rms                            | roughly stable                   | drift up = qkv proj unstable
gate magnitude (proj.bias[2d:]+learned) | MOVES off 0.1                    | flat = not learning to use memory
dbg_agreement                           | high in-seg, DROPS@switch, recov | flat = proj_extra never trained
proj_extra grad (NEW, M3)               | non-trivial, nonzero             | ~0 = anti-drift signal decorative
identity metric @short eval (NEW, H3)   | < matched mem-OFF baseline       | >= baseline = memory not helping ID
generator_loss vs matched mem-OFF       | <= baseline early, lower post-sw | > baseline = memory hurting
VRAM                                    | fits (proven 24lf / 2x48GiB)     | OOM = reduce window
```
**Honest caveat:** there is **no** first-hundred-steps proxy for 240s identity consistency. The earliest *direct* evidence is the NEW identity metric (#4) on a short (e.g. 30-48 latent fr) eval-checkpoint below a matched mem-OFF baseline, **simultaneous with** gate rising AND drift plateauing AND agreement dropping at the switch. Full proof only at the 120-240s eval (`configs/benchmark_240s_single.yaml`). DMD `generator_loss` alone is **not** an identity signal (C1).

---

## 8. New ideas — each as an ASCII schema

**8.1 Cross-chunk detach (the blocker fix), with truncation made explicit**
```
SEQUENCE (max_len 24)
 chunk1[0..21] --write x7--> state1   loss1.backward()  [graph G1 freed]
                              | detach_state=False -> state1 KEEPS G1
 chunk2[21..24] reads state1 -> apply -> loss2.backward()
                              ^____ traverses FREED G1 => CRASH
 FIX at chunk boundary:  state1 = tuple(s.detach() for s in state1)
   => truncated-BPTT depth == within-chunk (the intended "window 24")
```

**8.2 Horizon curriculum (TTT-Video's proven mechanism; ours commented-out)**
```
stage:   [A]------[B]--------[C]----------> wall-clock
max_len:  24       48          96      (VRAM gate: stop at proven 24->? ceiling)
GATE TO NEXT BUMP: identity-metric green AND drift plateau AND grad-norm finite
ref: TTTv Table2 (3/9/18/30/63s); scaffold streaming_training.py:391-396
```

**8.3 Teacher-forced warm-up -> self-rollout schedule (de-risk H4)**
```
phase  write target            in-graph compounding
WARM   GT clean latents (TF)   none (matches LaCT vid_flow:221 / TTTv 4.3)
   |   prob p: TF -> self      anneal p:1->0
SELF   own 4-step output       full self-rollout (our novelty, switch:243-263)
A/B gate: SELF arm must not diverge faster than TF arm on short horizon
```

**8.4 Identity-targeted auxiliary signal (fills C1/H3 gap)**
```
DMD per-frame loss (existing) ----+
                                  +--> total
across-switch consistency  -------+
  L_id = 1 - cos( emb(frame_t<switch_anchor), emb(frame_t>switch) )
  emb = DINO / face embedding (KV-probe tooling per MEMORY)
  applied on gradient_mask region; measure corr(grad_mem, L_id) FIRST
[UNVERIFIED] exact embedding model + weight lambda not yet chosen
```

**8.5 Extrapolation curve instead of a single 240s leap (de-risk C2)**
```
eval horizons:  24 -- 48 -- 96 -- 240   (latent frames)
plot identity-metric vs horizon; report SLOPE not just endpoint
expectation set by refs: NO source shows 40x zero-shot -> treat 240 as out-of-dist
```

**8.6 Matched-baseline A/B (makes a "win" falsifiable, H3)**
```
arm OURS      : g_init=0.1 (or 0.0), memory ON, train_write
arm BASELINE  : g_init=0.0 / mem-OFF, SAME steps + data + seed
WIN := identity-metric(OURS) - identity-metric(BASELINE) >= pre-registered Δ @120s,240s
(DMD loss delta is necessary-not-sufficient)
```

---

**Bottom line:** C3 is the only hard launch blocker and its fix is agreed. But C1 (loss doesn't measure identity) + C2 (no source supports 40× extrapolation; Scal3R evidence was misread and is struck) mean a short, instrumented diagnostic is the *only* justified next step. Make the grad-norm log real, define an identity metric with a matched baseline, and run the teacher-forced-vs-self-rollout A/B before any long compute.