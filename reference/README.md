# `reference/`: the research record

This directory is the lab notebook of the project: what was tried, what was predicted before
each run, what came back, and which claims died along the way.

The full record is about 650 files accumulated between June and August 2026: probes, campaign
launchers, per-run configs, session notes. All of it is still on disk, and all of it is in git
history at commit `cf1d8f1`. The repo tracks the curated subset below, because 650 files makes
the interesting ones unfindable. `.gitignore` holds the explicit allowlist; adding a `!` line
puts any file back under version control.

Documents written between June and August are in a mix of French and English and are left as
written. They are a dated record with pre-registered thresholds, and a retro-translation would
weaken exactly what makes them evidence. Code added since is in English.

---

## Start here

| file | what it is |
|---|---|
| `RESEARCH_STATUS.md` | the living status doc, read this first |
| `runs/archive/RUNS_ARCHIVE.md` | index of every training run, with the finding each one closed |
| `REBUILD_2026-09-09.md` | what was rebuilt after the compute cluster was lost, and from what evidence |

## The method

| file | what it is |
|---|---|
| `memory2_probes_registry.md` | the sealed pre-registration log: thresholds and predictions written down before each run, with the refutations recorded. Includes the 10-seed robustness annex that turned a headline 15.33/19 into a median 13.5, and the device-variability control that measured 2 pairs of chain noise either way, which made smaller gaps uninterpretable. |
| `runs/c3_stable/PREREGISTERED_step1000_rule.md` | a single pre-registration, in isolation, as a worked example |

## The findings

| file | what it is |
|---|---|
| `composite_gates_report.md` | the big campaign report. The splice is read and copied for about 3 s (+0.827 paired excess, 23 up / 1 down), but the copy is orthogonal to re-identification (Pearson -0.075), and the decoded anchor is an image of the interlude, not of the subject (B closer than A on 21/24, p=3e-4). |
| `probe1_report.md` | the probing verdict: with a validated instrument, the memory state does not carry subject identity |
| `memory2_t0_report.md` | the objective change that followed, and its audit |
| `why_ours_fails_vs_papers.md` | line-by-line comparison of this implementation against TTT-Video, LaCT and Scal3R |
| `why_memory_degrades.md` | the shorter diagnosis |
| `ttt_design_review.md` | design review of the cell against the papers it derives from |
| `write_path_research_report.md` | the exact meta-gradient of the fast-weight write, and why the rolling KV cache makes it hard |
| `stock_longlive_fidelity_audit.md` | baseline audit |
| `vram_recap.md` | measured memory ceilings for read-only vs write training |
| `plan_entrainement_video_2026-08-07.md` | the last plan, with the fidelity-ladder decision gate |

## The instruments

| directory | what it does |
|---|---|
| `runs/eyeboards/` | blind judging boards: generator, scorer, selftests. Deterministic salted permutation, answer key never in the page, "cannot tell" counted separately, Wilson intervals. Standalone, 31 selftests. |
| `runs/probe1/` | the probing pipeline: subject masks, state replay, fixed probe battery, prober training, mechanical threshold application |
| `runs/frame_dec/` | the state-to-frame decoder: pair collection, trainer, and the three v2 fixes with their selftests |
| `runs/mem2/` | the offline writer trainer, via differentiable write replay |
| `runs/reid_scen/` | the re-identification scenario matrix (S1 to S4, same-scene variants) and its scorers |
| `runs/fidelity_ladder/` | the blind fidelity-ladder campaign: rung construction, anonymization, blur calibration |

## The numbers

`runs/*/eval*/metrics_c1_step*.txt` holds the measured 3-way tables (memory on / memory zeroed /
stock Anchor-Forcing) for the c1, c2, c3 and l1 campaigns. Identity, motion and VBench, per
checkpoint. These are the raw evidence behind the trajectory described in the report.

## The write-up

`rapport/` is the internship report, six sections plus a combined document. State as of
15 July 2026, so it predates the August work (the delta write rule, the channel campaign, the
decoder). Updating it is the main outstanding task.
