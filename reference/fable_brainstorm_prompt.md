# Fable brainstorm prompt — open, independent full-project review

You are a top-tier ML research advisor with deep knowledge of video diffusion, autoregressive/streaming generation,
test-time-training (TTT) / fast-weight memory, and linear-attention / SSM sequence models. Do a **complete, rigorous,
brutally honest, and INDEPENDENT** brainstorm of the project below. Understand what we are building and what we have
observed, then **go survey the entire relevant literature YOURSELF** — find and read every paper that matters, decide
for yourself what the real questions are, and reach your own conclusions. Do NOT just agree with us: we have spent
weeks on this and may be anchored. Be a skeptic. Prioritize truth over encouragement.

We are deliberately NOT giving you a reading list or a list of questions. Find the papers and frame the questions
yourself. If you have access to our codebase, study it to see the exact implementation; if not, work from the
description below.

## The project
Goal: fix long-horizon **identity / subject drift** in streaming, real-time, autoregressive video diffusion — keep a
subject visually consistent over long videos and across prompt switches / revisits, without quality degrading over
time.

Backbone / training: Wan2.1-T2V-1.3B student, distilled into a few-step causal generator via **Distribution Matching
Distillation (DMD)** with **autoregressive self-rollout** (the model trains on its own generated frames) and a rolling
KV cache; teacher Wan2.1-T2V-14B. This is the "Self-Forcing" formulation.

The stack our repo sits on: **Self-Forcing** (the base DMD causal-distillation framework, attention + KV cache, no
learnable memory) → **LongLive** (adds KV re-cache on prompt switch + a frame-level attention sink + streaming long
tuning; no learnable memory) → **Anchor-Forcing** (our repo; adds a *structural* KV-cache reorganization it calls
"Anchor Memory" — NOT a learnable memory — plus a "Tri-Region RoPE"). So the novelty we are chasing is a **learnable
recurrent / fast-weight memory** on top of a lineage that is entirely attention + KV-cache + sink tokens.

## Our idea (codename MAG)
Graft a learnable **fast-weight / test-time-training (TTT)** memory module into the backbone at a few blocks. Per
generator call it READS a fast-weight state via a SwiGLU apply, WRITES to the state by an inner-loop gradient step
(with a Muon / Newton-Schulz orthogonalization and a weight-norm or decay update), and INJECTS its read into the
residual stream via a gated AdaLN output whose gate is zero-initialized (so it starts as a no-op = do-no-harm). The
module was ported from a 3D-reconstruction TTT layer, not originally designed for video.

## What we have observed so far — treat these as OUR observations to verify and challenge, not settled truth
1. **Gate starvation.** With the output gate initialized to exactly 0, the memory contributes nothing to the video and
   the gradient to the read/write/seed parameters is exactly 0 — a self-reinforcing dead zone where nothing upstream of
   the gate ever learns and the gate has no incentive to open (opening injects an untrained, harmful read).
2. **The bottleneck seems to be the OBJECTIVE, not the mechanism.** The DMD loss has no identity/reconstruction term,
   and within the training horizon (which is ≤ the KV attention window) the KV cache already carries the recent frames,
   so the memory looks redundant and the loss is indifferent to it. Training the write end-to-end gave results *worse*
   than baseline, and worst at the training horizon (the gap shrank at longer lengths) — suggesting it is not
   horizon-limited, so more cross-chunk / longer BPTT did not look like the fix.
3. **Normalization vs decay.** The pure-TTT works we ported from use norm-conserving updates (or plain SGD), with no
   forget/decay gate; the forget gate lives in the gated-SSM family. When we removed our home-grown decay (back to
   norm-conserving) and cracked the gate open a hair, the gate then *opened on its own* and the memory's effective
   contribution grew — but whether that is *helpful* or *toxic* is not yet measured (an eval is pending).
4. **The field may have preempted us.** At least two very recent works appear to add a recurrent (gated-delta) memory to
   the *same* backbone + self-rollout DMD and beat LongLive / Self-Forcing, using a closed-form recurrence trained
   end-to-end with a co-trained backbone. If so, our multi-step Muon TTT on a mostly-frozen backbone with no identity
   objective may be both harder to train and less novel.
5. **Strategic doubt.** The general "long video without drift" problem looks largely solved without any learnable memory
   (recent methods reach minutes with just attention + KV + sink). The only niche where a learnable memory might have a
   unique job seems to be subject **re-identification / revisit** (a subject leaves or is evicted from the KV window and
   must return with the same identity), and the standard consistency metric may be largely blind to exactly that.

## Your mandate
Go do the research yourself. Survey the full landscape of relevant literature (streaming/causal video diffusion,
distillation, long-video generation, TTT / fast-weight layers, SSM / linear-attention / gated-delta memory, subject
consistency / re-identification, and anything else you judge relevant — including the works most similar to ours).
Decide what the real questions are. Then brainstorm the whole project end to end.

## Deliverable
Your own structured assessment, including: a brutally-honest verdict (is a learnable memory here viable AND novel, or
should we pivot?); the best memory mechanism and the right training objective given everything you find; the genuinely
defensible contribution left (if any); pivot options if the answer is no; and a ranked, concrete experiment plan
(assume ~3×48GB GPUs and a working Wan2.1-1.3B + DMD self-rollout pipeline). Cite the work you rely on, and explicitly
flag where you disagree with our observations above and why.
