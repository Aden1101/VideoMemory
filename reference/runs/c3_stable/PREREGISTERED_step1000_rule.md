# Pre-registered decision rule for c3_stable step-1000 (committed 2026-07-16, BEFORE observing ckpt-1000)

Committed after the Fable methodology audit, which showed: (a) the c3@500 raw identity delta (-0.059)
is inside the noise floor (t=-1.13, n=6, one prompt carries it) and is NOT by itself evidence of a ramp;
(b) the real "injection is active and differentially damping motion" evidence is the motion split
(DD memon 0.85 vs memzero 0.928, from 0.99/0.99 at step 250) with cos-dist 0.32 > 0.18 floor;
(c) EMA@500 is 60% still the step-250 snapshot (0.99^50=0.605) so it is DESCRIPTIVE, not decisive.
The deciding test is step 1000+ (EMA seed-weight decayed to 0.22 at 1000), which is where c2 flickered.

## What to run at ckpt-1000 (and 1250, 1500, ...)
3-way sink1 (sink_size 1), 240f AND 480f, on BOTH raw AND EMA weights (use_ema true).
Primary axis = memon - memzero tail-mean id_cos (clean, stock-independent). Report pos/6 count too.
Do NOT quote memon-vs-stock or memzero-vs-stock gaps under the sink1 recipe: the reused stock column
is sink_size 3 (NOT sink-starved), so those gaps mix a sink handicap with model differences.

## Reference: c2@1000 = the flicker we must NOT reproduce
id -0.253 (6/6 neg), SC dropped 0.95 -> 0.80, DD 0.91 but incoherent (flicker counted as motion).

## Decision (applied to BOTH raw and EMA at step 1000)
- PASS (pack HOLDS do-no-harm + dynamic):
    identity  memon-memzero tail id_cos >= -0.06  (within noise of do-no-harm)
    coherence SC memon >= 0.92                     (no flicker collapse)
    motion    DD memon >= 0.85 AND lag-2 shows real motion (not the flicker signature)
- FAIL (ramp/flicker like c2):
    id delta <= -0.15  OR  SC memon <= 0.85  OR  flicker signature (high lag-1 + low lag-2 + collapsed id_cos)
- PARTIAL (damped but drifting): anything in between = pack helped vs c2 but did not fully stabilize.

## Anti-cherry-pick commitments
1. Report raw AND EMA at every checkpoint from 750 on (no post-hoc column selection).
2. The identity number alone is NOT quoted as ramp evidence at any single checkpoint (it is within noise at n=6);
   the ramp claim requires the MOTION split + injection-dose-above-floor, per the audit.
3. This file is the pre-registered rule; the verdict at 1000 is read against THIS, not a rule written after.
