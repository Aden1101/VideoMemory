All critique facts verified: `forward_sequence` is never called from any pipeline (only defined + referenced in its own docstring); the live path is `step()` at causal_model.py:555; the write `step()` runs under unconditional `torch.no_grad()` at streaming_switch_training.py:244-258; `enable_grad` at 204 wraps the read-only (`ttt_update=False`) pass; `_detach()` keeps the graph in `.train()` but is moot under `no_grad`; Muon down-casts to bf16 at video_memory.py:102; the inner grads are token-summed rank-1 outer products (148-150). I have what I need to write the final report.

---

# Training the TTT Write-Path Inside a Causal Self-Rollout

A research report on putting principled gradient on the fast-weight *write* of a per-head SwiGLU memory, inside a causal, in-place-cached video self-rollout. Every code claim below is verified against the source; where the previous draft misread the code, the correction is stated explicitly.

---

## 1. The crux, stated correctly

We inject a per-head SwiGLU fast-weight memory at layer 5 of a 30-layer causal DiT (Wan2.1-1.3B distilled to a 4-step DMD generator). The READ `o = f_W(q) = (silu(q w0) ⊙ (q w2)) w1` is plain-differentiable; its params (q-proj) already receive gradient through the DMD loss. The WRITE is the problem: per clean chunk we take one gradient-**ascent** step on the inner score `s = ⟨f_W(k), v⟩` (video_memory.py:134 — "the fast weight ASCENDS this dot"; **not** a recall/regression objective — this matters, see Rank 4), with per-token learned rates `lr_i = softplus(W_lr x + b)` (video_memory.py:281-283), then Muon (Newton–Schulz, 5 iters, bf16 — video_memory.py:102) and a per-column weight-norm renorm that pins each column to its pre-update norm (video_memory.py:166-186). The write params are `θ_w = {k/v slices of qkv, k_norm, lrs Linear}`.

**Verified ground truth about the current state of training (this corrects the draft's central overstatement).**

1. The live rollout calls **only** `VideoMemoryLayer.step()` (causal_model.py:555). `forward_sequence` (video_memory.py:325) is **dead code in the training path** — never invoked by the pipeline. It is a parallel, teacher-forced, single-pass, no-cache, no-eviction routine, i.e. exactly the LaCT regime condemned in §4. "Recompute over a window" is therefore **not** turning on an existing path; it is writing a new one.
2. The write (`ttt_update=True`) is **unconditionally** wrapped in `with torch.no_grad()` (streaming_switch_training.py:244-258). The `enable_grad` toggle at line 204 covers only the **read-only** final denoising pass (`ttt_update=False`, line 218). **There is no flag to "lift" to make the write differentiable.**
3. Consequently `_detach()` (video_memory.py:303, returns `False` in `.train()`) is **moot for the write**: `no_grad` already prevents any graph from forming. So today `θ_w` receives **exactly zero** gradient, and `lr_mem` in `configs/ttt_phase1.yaml` currently trains only the read q-proj and the proj/AdaLN gate (which see grad through the read pass).

**The honest consequence:** *every* method in this report requires, as prerequisite work, removing the `no_grad` at line 244 **and** constructing a write graph that bit-matches `step()` + the `causal_model` attention/cache path. This is a real memory regression (the write graph over a window must be retained), not a flag flip.

### The exact write meta-gradient

Let `o_s` be the memory output at chunk `s`, `W_t` the fast-weight state after chunk `t`'s write, `ℓ_r` the inner score at chunk `r`. The exact gradient is

```
dL/dθ_w = Σ_t Σ_{s>t} (dL/do_s)(∂o_s/∂W_s) [ Π_{r=t+1}^{s-1} A_r ] (dW_{t+1}/dθ_w),
A_r = ∂WeightNorm · ( I + ∂Muon · Hess_W ℓ_r ).
```

**Correction to the "direct" factor (critique #5).** `dW_{t+1}/dθ_w` is **not** purely direct. `θ_w` enters chunk `t`'s update through `k_t, v_t` **and** through `lr_{0,1,2}` (themselves `softplus(W_lr · norm1(x_t))`), and `x_t` depends on `o_{<t}`, hence on **past** writes. So `θ_w` has a direct path (into chunk `t`'s update) and an indirect path (through `x_t`'s dependence on earlier outputs). The single-`Σ_t` form above absorbs the indirect path into the outer sum only if one is careful; any estimator claiming **exactness** (UORO) must carry it. For the lr-Linear and k_norm the indirect path is plausibly second-order-small; for the k/v projections feeding the SwiGLU it is not obviously negligible and must be either carried or explicitly argued away.

### Two obstacles that must never be conflated

- **Problem A — cost (generic).** `A_r` carries the inner-objective **Hessian** `Hess_W ℓ_r` (the score is bilinear-through-SwiGLU, so the Hessian is nonzero and dense), the **Jacobian of Muon** (5 Newton–Schulz iterations), and the product `Π A_r` over hundreds of chunks (deep BPTT keeping activations). Heavy, but on short clips payable, and it is *generic* — every TTT/fast-weight system pays some version of it.

- **Problem B — feasibility (ours alone).** The fixed-size rolling **attention** KV cache (3 sink + 9 local, FIFO, **overwritten in place** each step via `_apply_cache_updates`) destroys the BPTT tape for the **backbone**: tensors a later backward would read are evicted/overwritten before backward can run. Eviction is forced by long-video memory cost and is non-negotiable. **The decisive test for any method: does it avoid backprop through the overwritten/evicted rolling cache?** A method "solves B" only if yes.

### The structural split — and the precise limit of it (critique #3)

The fast-weight state `W_t` lives in `TTTBlockCache.state`, carried forward explicitly; it is **not** the attention KV cache. The `W_t` recurrence can therefore be attacked by forward-mode / windowed methods that never read an evicted attention tensor.

**But this does not make Rank 1 "B-free," and the draft's claim that it "dissolves B" was the single biggest overstatement.** The loss factor `dL/do_s` is produced by the **backbone**: `o_s` injects at layer 5 and feeds the remaining 25 layers, whose self-attention at every **future** chunk reads the in-place-rolled window. Across chunks, the previous call's `cache["k"]/["v"]` slots are overwritten in place and FIFO-evicted. The "clone before attention, destructive write deferred to `_apply_cache_updates`" structure only protects you **within a single forward call**; it does nothing across chunks. Therefore:

> **Honest statement.** Windowed-TBPTT solves B for the `W_t` recurrence and for **same-chunk** read. The **cross-chunk loss credit** `dL/do_s` (`s` more than one chunk back) still rides the backbone through the evicted cache, exactly as exact BPTT does. To get it you must **recompute the backbone over the window from retained chunk inputs** (re-deriving the cache states, not reading them) — expensive and bit-fragile — or **truncate `dL/do_s` to same-chunk**, which silently discards the long-horizon credit the memory exists for.

Only one family is genuinely B-immune for *both* factors: forward-only evolution strategies (Rank 2), whose immunity comes from the cache being **causal**, not from any cloning trick.

---

## 2. Ranked shortlist

Ranking criterion: **(true B-immunity) × (cost feasibility) × (bias/variance controllability)**, weighted toward B because it is the part nobody has solved and the part that fails silently.

### Rank 1 — Windowed recompute-TBPTT aligned to the KV cache

The most defensible way to put gradient on the **actual** write rule. Truncate BPTT to the live attention window and **recompute the backbone + write over that window from retained chunk inputs**, so a differentiable sub-graph exists that never reads an evicted tensor. It is **partially** B-immune (see below) — not the clean win the draft claimed.

**Math.** Replace `Π_{r=t+1}^{s-1} A_r` by `Π_{r=s-W}^{s-1} A_r` and detach `W_{s-W}`, with `W = sink + local = 3 + 9 = 12` chunks. The truncated estimator `g_W` obeys the Aicher–Foti–Fox geometric bound **iff the state recurrence contracts**: if `‖A_r‖ ≤ λ < 1`,

```
‖g − g_W‖ ≤ C · λ^W / (1 − λ).
```

**Correction to the bound's regime (critique #4).** `A_r` is the Jacobian of a **matrix-valued** map under weight-norm, which is a per-column projection onto a sphere: `∂WeightNorm = (I − ŵŵᵀ)/‖w‖ · (norm-restore)` removes the radial component and is non-expansive *along* the sphere but does **not** guarantee `λ < 1`; composed with `(I + ∂Muon·Hess)` it can be expansive tangentially. The relevant `λ = ρ(A_r)` (spectral radius / operator norm), measured, not assumed. And the bound's bad branch was mis-stated: for `λ > 1` the bound **diverges** (`λ^W/(1−λ)` blows up / flips sign); at `λ = 1` exactly it degrades to `C·W`. Since Muon orthogonalizes (update singular values `~ U(0.5, 1.5)`, video_memory.py:88-91) and weight-norm pins column norms, the realistic regime is `λ ≳ 1` — the bound's **divergent** branch. **Any "≈1.4% tail bias at λ≈0.7" number is fiction until `ρ(A_r)` is measured and is deleted from the cost accounting here.**

**Why it (partially) fits.** B is dissolved **for the `W_t` recurrence and same-chunk read** by recomputing from retained inputs `x_{s-W..s}` under `torch.utils.checkpoint(use_reentrant=False)`; everything older than `W` is detached on purpose (that is exactly where the exact graph does not exist anyway). What it does **not** dissolve is the cross-chunk `dL/do_s` backbone factor — that requires the in-window backbone recompute, which is the expensive part and must reproduce the forward **bit-for-bit** (rope offsets `rope_start_f`, sink/local split, rolling indices; an off-by-one silently corrupts the gradient).

**Costs and biases, re-costed honestly (critiques #1, #2, #8, #12).**
- **Build cost is real:** there is no existing grad-enabled write path to reuse; `forward_sequence` is the wrong (teacher-forced) regime. You write a new windowed `step`-matching forward and retain its graph (memory regression).
- **bf16 NS floor (critique #8).** `zeropower_via_newtonschulz5` hard-casts `X = G.bfloat16()` (video_memory.py:102), a `W`-independent 2–4% gradient-error floor on every differentiated write. The fp32 fix is **not cheap**: at `nh=1`, `head_dim = 1536`, so `w0_grad` is `(B, 1536, 6144)` and NS computes `X@Xᵀ ∈ (B,1536,1536)` and `A@A` (a 1536³ matmul), 5 iterations, **×3 grads × W=12 chunks** per backward. Re-cost as ~12× the per-step NS, not "a few small matmuls."
- **Muon is non-smooth, not merely bf16-noisy (critique #12).** NS approximates the orthogonal polar factor `G ↦ UV^T`; its Jacobian blows up like `1/(σ_i+σ_j)` near singular-value crossings / zero singular values. The inner grads are **token-summed rank-1 outer products** (`w0_grad = (k·lr0)ᵀ @ dgate_before_act`, video_memory.py:148-150): for chunk length `L < head_dim` these are **rank-deficient**, so most singular values are ≈0 and the Muon Jacobian is **ill-conditioned by construction**, even in fp32. Practical mitigation: a damped / `+εI`-regularized NS Jacobian, or stop-grad on Muon (which collapses Rank 1 to a biased first-order object — honest but lossy).

**First experiment.** *Measure `ρ(A_r)` first* (power-iterate the per-chunk write-Jacobian JVP `A_r v` on a 3–5-chunk clip, fp32 NS). If `λ ≲ 0.8`, `W = 12` is justified and the tail is small; build the windowed write graph with fp32 in-window NS and an `εI`-damped Muon Jacobian, train `θ_w`, and validate `g_W` cosine vs brute-force full-BPTT on a 2–3-chunk no-eviction toy **and** vs the eviction-on finite-difference check (§5, the decisive test). If `λ ≳ 1`, do **not** trust the window — go forward-only (Rank 2).

### Rank 2 — PES / NRES (forward-only evolution strategies) on the write-path

**The only family that is genuinely, unconditionally B-immune** — its math is indifferent to in-place mutation and eviction because it reads **only forward scalar losses**. Treat the whole rollout as a black-box `L(θ_w)`.

**Math.** Optimize the Gaussian-smoothed `L_σ(θ_w) = E_{ε∼N(0,σ²I)}[L(θ_w+ε)]`, antithetic estimator `g_ES = (1/(2Nσ²)) Σ_i ε^(i)[L(θ_w+ε^(i)) − L(θ_w−ε^(i))]`. Split the per-clip rollout into windows `s = 1..K`; each particle carries an accumulated perturbation `ξ_s^(i) = Σ_{u≤s} ε_u^(i)`:

```
g_PES = (1/(Nσ²)) Σ_i Σ_s ξ_s^(i) L_s^(i).
```

**The unbiasedness argument, stated correctly (critique #6).** PES is unbiased for `∇L_σ` from truncated windows **provided** each partial loss `L_s` depends only on perturbations at windows `≤ s`. The draft asserted this for free. The real justification is **cache causality**: the in-place FIFO cache is overwritten **forward in time**, so chunk-`s` outputs (and the loss computed from them) depend on perturbations applied at chunks `≤ s` only. *This is the one place the in-place cache actually helps* — it makes the causal decomposition hold. (Note the antithetic `±ε` rollouts genuinely diverge in cache state, because the memory injects at L5 → changes the generated frame → re-encoded into context at lines 244-258; PES remains valid because it is black-box over the **whole** rollout including the cache, and the per-window decomposition survives precisely *because* the cache coupling is causal. CRN fixes sampled timestep/noise/critic, but cannot and need not fix cache contents.)

**Variance, re-ranked (critique #6a).** PES variance grows with the number of windows `K`. The per-clip reset bounds the horizon to one clip (~15–20 chunks), but `K ≈ 15–20` with a window-per-chunk is **not** small, and ES variance `~ d_w·K/σ²` is exactly PES's hard regime. **Therefore NRES / ES-Single (variance independent of horizon; Li et al. NeurIPS 2023; Vicol ICML 2023) is the right default, not the fallback.** Promote it.

**Engineering that makes it work.** Variance scales with `d_w`: reparametrize the k/v projection as **LoRA rank-8** (~10⁵ params, not millions) and perturb `k_norm` + `lrs` directly; use antithetic pairs + **Common Random Numbers** (same timestep/noise/critic per `±ε`, or DMD stochasticity swamps the σ-scale signal); **freeze or EMA-slow the jointly-trained critic** during an accumulation window (it is non-stationary, critique #6/#13). Particles are embarrassingly parallel across DP/FSDP ranks.

**Loss-sparsity caveat (critique #13).** The realized DMD loss is **not** a clean `Σ_t L_t` over all chunks: gradient is enabled only on the final denoising step of chunks past `start_gradient_frame_index` (line 199) and effectively one chunk's output carries the grad per the `break` structure (line 224). For PES this is *fine in principle* (it reads scalar losses, no grad needed), but credit to writes in **non-loss** chunks flows only through cache/state coupling — precisely the weak, high-variance long-horizon signal. Quantify how many chunks actually contribute loss before assuming the decomposition is favorable.

**First experiment.** On a single clip, perturb **only** the rank-8 k/v reparam + `k_norm` + `lrs` (`d_w ~ 10⁵`), antithetic + CRN + frozen critic, **NRES variant**. Validate `g_PES` against finite-difference directional derivatives along 2–3 random directions on a 3-chunk clip (cosine alignment). Then the load-bearing question: does the PES-trained write **reduce long-video DMD / drift** vs the frozen-write E0 baseline?

### Rank 3 — Implicit differentiation of the regularized write (1-step IFT / iMAML-style)

**The clean, cheap way to get the *direct* `dW_{t+1}/dθ_w` factor without unrolling or differentiating Muon** (was missing from the draft, critique #11). It does **not** solve B (still needs `dL/do_s` cross-chunk), but it is strictly cleaner than UORO or BPTT-through-Muon for the direct factor, and it pairs naturally with Rank 4's regularizer.

**Math.** Add a strongly-convex retention regularizer to the inner objective so chunk-`t`'s write is the unique minimizer of `g_t(W; θ_w) = −⟨f_W(k_t), v_t⟩ + (1/2η)‖W − α W_{t-1}‖²_F` (the same `‖W − αW_prev‖²` as Rank 4). At the optimum `∂g/∂W = 0`, the implicit function theorem gives

```
dW_{t+1}/dθ_w = − (∂²g/∂W²)^{-1} (∂²g/∂W∂θ_w),
```

an O(1)-memory meta-gradient (one linear solve with the inner Hessian, via conjugate gradients on Hessian-vector products) with **no unrolling of Muon and no BPTT tape** — exactly the Deep-Equilibrium / iMAML construction.

**Why it (partially) fits.** It removes Problem A's worst pieces (no Muon Jacobian, no activation tape) for the direct factor, and the regularizer makes the inner problem well-posed (defeating critique #12's ill-conditioning by adding `(1/η)I` to the Hessian). It does **not** solve B: the cross-chunk `dL/do_s` still rides the backbone. **Caveat to be honest about:** the deployed rule is a **single** ascent step + Muon + weight-norm, *not* a converged minimizer of `g_t`. IFT gives the exact gradient of the *implicit fixed point*, which is a **different operator** from the one-step rule actually run. So this trains a clean surrogate dynamics, not the deployed dynamics — acceptable if you adopt the regularized fixed-point write as the *actual* deployed write (recommended), and a bias otherwise.

**First experiment.** Replace the one-step write with the regularized fixed-point write (2–3 CG inner iters) on a 3-chunk toy; check that IFT `dW/dθ_w` matches finite differences of the inner solve; confirm the do-no-harm init survives (`α→1, η→∞ ⇒ W ≈ W_prev`).

### Rank 4 — Robust + retention-regularized **local** write loss (MIRAS / Titans surrogate)

Not an estimator of the true gradient — a principled, cheap **local proxy** for the keep/forget dynamics pure recall ignores. The honest "abandon the exact meta-gradient" option.

**Math, sign-corrected (critique #9).** Per clean chunk, after the inner step `W_+`, minimize

```
L_loc = Σ_i [ −⟨f_{W_+}(k_i), v_i⟩                  (MATCH the deployed ASCENT score — see below)
            + (1/η_i)‖W_+ − α_i⊙W‖²_F              (LOCAL retention = Titans (1−α) gate)
            + (1/λ_i)‖W_+‖²_F ],                     (global retention / weight-norm proxy)
```

with data-dependent `α_i, η_i, λ_i = softplus/σ(W_g x_i + b_g)`.

**The sign correction is essential.** The draft's Huber **recall** term `ρ_δ(f_{W+}(k_i) − v_i)` trains toward a **different fixed point** than the deployed rule: the inner objective is gradient **ascent** on `⟨f_W(k),v⟩` (video_memory.py:134), and under weight-norm (column norms pinned) `⟨f_W(k),v⟩` grows by **direction alignment**, not by `f_W(k) → v`. Minimizing `‖f_W(k)−v‖` fits a target the real rule never converges to. Use the **negative score** `−⟨f_{W_+}(k_i),v_i⟩` to match deployed dynamics (or a robust Huberized version of the *score residual*, not the recall residual).

**Why it fits / what it buys.** `dL_loc/dθ_w` is single-chunk, single-inner-step: it never multiplies `Π A_r` and never reads an evicted tensor — the graph it needs **exists**, so it sidesteps B structurally. Cheap (~1.1–1.3× the memory-branch forward). The keep/forget gate is principled: the true cross-chunk Jacobian's scalar keep/forget image is Titans' `(1−α_t)` factor (MIRAS Prop. 3.2 links retention and FTRL via a Bregman premetric), so `α` is a learned, contractive parameterization of `A_r`'s per-step action.

**Honest bias.** It is a **surrogate, not an estimator**: it drops the distant readout coupling `dL/do_s` and the Muon/Hessian off-diagonal structure of `A_r`, keeping only a per-column scalar `α`. Bias is smallest for short horizons / slow forgetting, **largest precisely in the long-video regime the memory exists for**. It also does not train on the self-rollout distribution (Titans' teacher-forcing blind spot at the chunk level). Treat `α/η/λ` as a learned inductive bias.

**Warm-start schedule, made explicit (critique #9).** With `α→1, η,λ→∞`, the retention terms vanish and `L_loc` reduces to **pure score** on chunk 0, which immediately tugs the write away from the do-no-harm init. **Fix:** ramp `λ_aux` from 0 over the first few hundred steps **and** keep `α≈1` (strong retention) initially so the net update to `W` stays ≈0 while the gates learn; only then relax `α`.

**First experiment.** Add `λ_aux·L_loc` (warm-started inert) as an auxiliary on clean chunks. Decisive check: does `L_loc` **correlate with held-out DMD improvement**? If not, it is decorative. Then add k-step **detached** readback (read after `k>1` later writes) to expose some `Π A_r` and shrink the bias.

### Rank 5 — UORO (forward-mode RTRL): **dominated here, kept only for the record**

Carries a rank-1 estimate `G_t = dW_t/dθ_w ≈ s̃_t⊗θ̃_t` updated by JVPs of `F = WeightNorm∘(W+Muon_NS5(grad))`. Unbiased in exact arithmetic (Tallec–Ollivier). **Disqualified here on three independent grounds (critiques #7, #12):**
1. **Variance is unusable, not merely "noisy."** State dim `N ≈ 28M per head` (`w0,w1,w2 ≈ 9.4M` each). A single Rademacher probe of a 28M-dim state has relative std `~ √N ≈ 5000×` for the trace-like contractions needed. Taming it needs rank-`r` with `r ≈ N`.
2. **It pays Problem A every chunk forever:** JVPs through the 5 NS iterations **every step**, not just in a window — strictly worse cost than Rank 1.
3. **It hits the same non-smooth, ill-conditioned Muon Jacobian** (critique #12) on rank-deficient outer-product grads, every step.
The cheap fix (KF-RTRL) is **off the table**: its Kronecker factorization requires `F` to be a pointwise nonlinearity of a single matmul; our multiplicative SwiGLU grad + Muon + weight-norm violate this. **Verdict: dominated by Rank 1 on cost and by Rank 2 on B-immunity; do not pursue.**

---

## 3. Promising combinations

1. **Spectral diagnostic → method selection (do this first, unconditionally).** Power-iterate `ρ(A_r)`. It is the hinge for every truncation/forward method: Rank 1's bound, Rank 4's surrogate validity, the contraction assumption itself. One cheap measurement de-risks the program. `λ < 0.8` ⇒ Rank 1 viable; `λ ≳ 1` ⇒ Muon/weight-norm forbid clean truncation ⇒ go forward-only (Rank 2) or add explicit damping.

2. **Rank 4 (retention) as warm-start + contraction-inducer for Rank 1 / Rank 3.** Pre-train `θ_w` with `L_loc`; the learned gate `α<1` makes `B = α A_r + (1−α)I` contract toward `λ<1` — the surrogate **manufactures the regime that makes the principled methods valid**, and supplies the strongly-convex regularizer Rank 3's IFT needs. Mutually reinforcing.

3. **Rank 1 as a low-variance in-window control variate for Rank 2 (PES) — stated with the condition that makes it valid (critique #14).** A control variate `g = g_PES − β(c − E[c])` reduces variance **only** with positively-correlated `c` and the **optimal** `β = Cov(g_PES, c)/Var(c)` (β=1 with weak correlation can *increase* variance). Define `c` as a **paired-window PES estimate** restricted to the same `W=12` window as `g_TBPTT` (so `E[c]` is the windowed-PES expectation, which `g_TBPTT` approximates with low variance and known bias), and estimate `β` online. Use `g_TBPTT` to cancel the in-window component of `g_PES` and let PES carry only the cross-eviction-boundary credit. **Without the estimated `β` and the paired-window construction this combination is not well-defined — treat the β=1 form as a conjecture only.**

4. **Damped-phantom depth-1 fallback under Rank 1's hood.** If full windowed recompute is too fragile (rope/FSDP), the depth-1 stop-grad estimator with `W_carry = λ W_in + (1−λ)W_in.detach()` forces geometric decay `B = λ A_r + (1−λ)I` — a one-line, near-zero-risk first step that trades a smoothed-surrogate bias for guaranteed contraction.

---

## 4. What to avoid

- **Dense / structured RTRL and SnAp-1.** The cheap-RTRL premise (identity carry, linear key) is violated at three points — Muon (dense NS Jacobian), weight-norm (column coupling), SwiGLU self-recurrent gradient (`Hess_W ℓ ≠ 0`). The influence tensor is `~10¹³–10¹⁴` entries (fits in no memory). Its only affordable form is the diagonal approximation — the same first-order object as a 1-step detached write, buying nothing. Muon's NS fill-in destroys SnAp-1's support sparsity; detaching Muon to "fix" it collapses to that same biased object.

- **Linearize-and-scan / DeltaNet swap.** This does not linearize our write — it *replaces* the SwiGLU+Muon+weight-norm memory with a single linear matrix `S`, deleting the nonlinear associative capacity, throwing away Muon and the Scal3R-pretrained init, and **still** needing truncated BPTT to a non-evicted window. Pays B's tax while solving a different, weaker problem.

- **Pure teacher-forced LaCT training (i.e. wiring up `forward_sequence`) as the *only* method.** Cheap and feasible — it is literally the dead-code `forward_sequence` with per-chunk detach — but its surrogate bias points the **wrong way**: train the write on GT-clean k/v with full recomputed attention, deploy on the model's own drifted self-rollout k/v through a FIFO-evicted cache. It reintroduces, for the memory, the exposure-bias gap Self-Forcing/DMD exists to close for the backbone. Useful as a pretraining stage, wrong as the endpoint.

- **UORO** (see Rank 5): dominated on cost and variance; do not pursue.

- **Trusting any truncation/forward method before measuring `ρ(A_r)`.** Every "small bias" claim is conditional on `λ < 1`, and Muon + weight-norm conspicuously push `λ → 1`. The failure mode (a silently biased, possibly divergent gradient) is invisible without the spectral check.

---

## 5. The decisive cheap experiment the draft omitted (critique #15)

Every validation in a no-eviction toy is **blind to B**, because B is *defined* by eviction. The one cheap, B-aware adjudicator:

> **Eviction-on finite-difference probe.** Pick one write param component `θ_w[j]`. Compute the central difference `(L(θ + h e_j) − L(θ − h e_j)) / 2h` **through the full eviction-on rollout** — pure forward, no graph, hence B-immune and cheap. Compare to each method's `j`-th gradient component. This is the **only** check that measures the cross-eviction-boundary credit Rank 1 truncates, and the only one that distinguishes "solves B" from "silently truncated B." Run it for a handful of `j` spanning k-proj, v-proj, `k_norm`, and `lrs`.

---

## 6. Recommended sequence

1. **Measure `ρ(A_r)`** (power-iterate the per-chunk write-Jacobian JVP, fp32 NS, 3–5 chunks). Routes everything.
2. **Eviction-on finite-difference probe** (§5) for a few `θ_w` components — the B-aware ground truth.
3. **Validate machinery on toys:** brute-force full-BPTT `g` on a 2–3-chunk no-eviction toy; cosine-check windowed `g_W`, and PES `g_PES` vs finite differences. Cheap insurance against silent bugs.
4. **Ship the sign-corrected Rank 4 surrogate** (warm-started inert, `α≈1` first) and confirm `L_loc` correlates with held-out DMD. Opens the gate, learns a contractive keep/forget bias, and supplies the regularizer for Rank 3.
5. **Branch on `ρ(A_r)`:** if `λ ≲ 0.8`, build the windowed write graph (Rank 1) — remove the `no_grad` at line 244, retain the in-window graph, fp32 + `εI`-damped NS, `W=12`, validate against steps 2-3. If `λ ≳ 1`, go **NRES/ES-Single (Rank 2)** on a rank-8 k/v reparam, per-clip horizon, antithetic + CRN + frozen critic.
6. **Combine (Combination 3, with estimated β and paired-window PES):** Rank 1 in-window control variate + Rank 2 cross-boundary corrector — the cleanest route to a gradient that is simultaneously B-correct, low-bias, and low-variance.

Throughout: run Newton–Schulz in **fp32** wherever it sits in a gradient path (the 2–4% bf16 floor is `W`-independent and contaminates every differentiated method), regularize the **non-smooth, ill-conditioned** Muon Jacobian (`+εI` / damping) or stop-grad it, and keep `θ_w` **small** (variance and tractability of nearly every method scale with `dim(θ_w)`).

---

## The 2-3 highest-value directions to explore FIRST

**Direction 1 — Forward-only NRES/ES-Single on a low-dimensional write reparam (the only unconditional "yes on B").**
- *Math.* `∇L_σ(θ_w) = E_ε[ε L(θ_w+ε)]/σ²`, estimated antithetically; the per-clip causal decomposition is unbiased for `∇L_σ` **because the in-place cache is overwritten forward in time** (cache causality), so chunk-`s` loss depends only on perturbations at chunks `≤ s`.
- *Why it survives B.* It reads only forward scalar losses; it never forms `do_s/dW_s`, never differentiates Muon/NS/weight-norm, never reads an evicted tensor. The cache may be overwritten and `ttt_cache.state` reset freely. This is the lone family with an unqualified B-immunity, and its immunity is *because of*, not in spite of, the cache structure.
- *First experiment.* Single clip; perturb only a rank-8 k/v reparam + `k_norm` + `lrs` (`d_w ~ 10⁵`); antithetic + CRN + frozen critic; **ES-Single / NRES** for horizon-independent variance. Validate `g_PES` vs finite-difference directional derivatives on 3 chunks; then test whether the trained write reduces long-video drift vs frozen-write E0.

**Direction 2 — Sign-corrected retention surrogate (Rank 4) as warm-start + contraction-inducer, validated by correlation with DMD.**
- *Math.* Per-chunk `L_loc = Σ_i[ −⟨f_{W_+}(k_i),v_i⟩ + (1/η_i)‖W_+ − α_i⊙W‖² + (1/λ_i)‖W_+‖² ]` — the **score** term (not recall) matches the deployed ascent fixed point; `α` parameterizes the keep/forget image of `A_r` (Titans `(1−α)`, MIRAS Prop. 3.2).
- *Why it survives B.* Single-chunk, single-inner-step: the gradient graph it needs **exists** and never reads an evicted tensor. It is a deliberate, contraction-inducing surrogate (it *creates* the `λ<1` regime the principled methods require), not an estimator of the intractable `Π A_r`.
- *First experiment.* Add `λ_aux·L_loc` warm-started inert (`α≈1`, `λ_aux` ramped). Decisive check: does it **correlate with held-out DMD improvement**? If yes, keep as warm-start + regularizer; if no, discard as decorative.

**Direction 3 — The eviction-on finite-difference probe as the adjudicator, before committing to Rank 1.**
- *Math.* `(L(θ+h e_j) − L(θ−h e_j))/2h` through the full eviction-on rollout — the exact directional derivative of the *deployed* loss, B-immune by construction (pure forward).
- *Why it survives B.* No graph, no tape; eviction never matters because nothing is read backward. It is the only measurement that contains the cross-eviction-boundary credit that any TBPTT method truncates.
- *First experiment.* Probe 4 components (k-proj, v-proj, `k_norm`, `lrs`) at 2–3 chunks back. Compare against (a) Rank 1's `g_W`, (b) the Rank 4 surrogate gradient. The gap between the probe and `g_W` **is** the long-horizon credit Rank 1 discards — quantifying it tells you whether windowed-TBPTT is sufficient or whether the forward-only route (Direction 1) is mandatory.

These three first because: Direction 1 is the only genuinely B-immune trainer and its variance is the single open question (answerable on one clip); Direction 2 is cheap, de-risks the init, and manufactures the contraction every other method needs; Direction 3 is the cheapest possible measurement of whether B's truncation bias actually matters here, and it gates the entire Rank-1 investment.

**References.** Tallec & Ollivier, UORO (arXiv:1702.05043); Mujika et al., KF-RTRL (arXiv:1805.10842); Irie et al., RTRL limits (arXiv:2305.19044); Aicher et al., adaptive TBPTT bias bound (arXiv:1905.07473); Vicol et al., PES (arXiv:2112.13835); Vicol, ES-Single (PMLR v202, 2023); Li et al., NRES (arXiv:2304.12180); Salimans et al., ES (arXiv:1703.03864); Rajeswaran et al., iMAML (arXiv:1909.04630); Bai et al., Deep Equilibrium Models (arXiv:1909.01377); Wang et al., LaCT / TTT-Video (arXiv:2505.23884); Behrouz et al., Titans (arXiv:2501.00663), MIRAS (arXiv:2504.13173); Yang et al., DeltaNet (arXiv:2406.06484), Gated DeltaNet (arXiv:2412.06464). **Code grounding (verified):** `wan/modules/video_memory.py` — `zeropower_via_newtonschulz5` bf16 cast L102 and `US'V^T` note L88-91, `swiglu_fast_weight_gradient` rank-1 outer-product grads L148-150, `weight_norm_update` L157-187, `step` apply-then-write L307-322 with `_detach` L303-304, `forward_sequence` L325-337 (**dead code in training**); `wan/modules/causal_model.py` — memory at L5, `self.memory.step(...)` the live path L555, deferred cache write `_apply_cache_updates`; `pipeline/streaming_switch_training.py` — read-only `enable_grad` toggle L199/204 over `ttt_update=False` L218, **write unconditionally under `torch.no_grad()` L244-258** (`ttt_update=True` L257); `configs/ttt_phase1.yaml` — `lr_mem` (currently reaches only the read q-proj + AdaLN gate, **not** `θ_w`).