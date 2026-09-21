# The numbers behind the figure

Three-way evaluation tables, one per checkpoint: the memory active, the memory zeroed, and the
unmodified model, on the same prompts and the same seed. `scripts/make_figure.py` reads them
directly, so `assets/motion_identity.png` is regenerated from measurements and never from a
hand-copied table.

`index.json` holds the same numbers as data, one row per table, written by
`python scripts/results_index.py`; a test keeps it equal to the tables.

One directory per run, named like the config it was trained with in `configs/runs/`:

| directory | figure panel |
|---|---|
| `run1_write_current_chunk/` | 1. writes the frames it is generating |
| `run2_write_evicted_chunk/` | 2. writes the frames just forgotten |
| `run3_write_evicted_stable/` | 3. same, plus the stability fixes |
| `run4_from_longlive_lora/` | 4. same, from the previous fine-tune |

`evalN` is the training step. A `_sink1` suffix means the evaluation ran with one sink frame
instead of three, `_ema` reads the smoothed weights, `_480` doubles the video length. The figure
draws only the plain `evalN[_sink1]` series, one point per step; run 2 has step 500 at both sink
settings and the figure keeps the sink-1 one.

Three directories are controls on a single checkpoint, and the figure leaves them out:

| directory | what it is |
|---|---|
| `run1_…/eval_sink1` | step 1250 again, with one sink frame instead of three |
| `run1_…/eval_R1` | step 1250 again, with `MEM_REJECT_STREAM=1` on the memory-active arm: the part of the injection parallel to the residual stream is removed. The memory-zeroed arm is identical to `eval1250`, as it should be. It is a different arm, not a repeat measurement |
| `run2_…/eval1000_msc05` | step 1000 again, with `MEM_STREAM_SCALE=0.5` on the memory-active arm at inference |

## How a table is made

Four steps on one checkpoint, six held-out prompts, seed 42, 240 latent frames:

1. generate with the memory active;
2. generate again with `MEM_ZERO_ALL=1`, same config, which zeroes the injection and nothing else;
3. run VBench (subject consistency, background consistency, dynamic degree) on both, and on
   the unmodified model's videos, which are generated once and reused;
4. `python scripts/eval_metrics.py`, which adds the DINO first-to-last identity, the lag-1 and
   lag-2 motion and the paired distances, and writes `metrics.txt`.

The driver that chained these on the cluster is not included. `tests/test_eval_report.py` reads
every table back and checks that the script's report and decision code write it again.

## What is and is not comparable

The `stock` rows were not generated alongside each table. They come from one earlier
generation of the unmodified model (`memory_layers: []`, 3 sink frames, seed 42, 240 frames, the
same six prompts), reused in every table of runs 1 to 3; its VBench columns were filled in two
passes, subject consistency and dynamic degree first, background consistency later. Run 4 has
its own, generated the same way from the fine-tune it starts from.

So the two trained arms are always comparable to each other: same checkpoint, same seed, same
sink. An arm against `stock` is comparable only in run 1's plain tables. In every `_sink1` table
the arms run with one sink frame and `stock` with three, and that alone costs identity: on
run 1 at step 1250, `eval1250` against `eval_sink1` gives 0.667 to 0.606 with the memory active
and 0.773 to 0.685 with it zeroed.

The `stock` dynamic degree of runs 1 to 3 is 0.994, the metric's ceiling, so no arm can exceed
it. Dynamic degree is the fraction of clips VBench rates as dynamic, not an amount of motion.

One seed, six prompts, and no repeat of any measurement: there is no noise estimate here.

## The VERDICT lines

Each table ends with a verdict printed by `scripts/eval_metrics.py`. It answers one question,
whether the memory changes the output at all (`delta(memon, memzero)` against a 0.18 floor),
and then whether any VBench or drift number differs from `stock`. `ACTIVE-POSITIVE` does not
mean the memory helps, and the verdicts are not part of this project's conclusion. They are
left in because the tables are kept exactly as the script wrote them.

The tables call the thresholds pre-registered. Git cannot show that: they enter this
repository's history in the same commit as the first tables (`cf1d8f1`, 2026-08-13). They have
not changed since, and every table prints the ones it used. The French words in the verdict
line are a leftover of the script's first version, kept for the same reason as the verdicts.

The evaluation script wrote a hardcoded run name and step into every header it produced. Those
labels were wrong and have been rewritten from the directory they sit in; no measurement was
touched.
