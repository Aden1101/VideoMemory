# The numbers behind the figure

Three-way evaluation tables, one per checkpoint: the memory active, the memory zeroed, and the
unmodified model, on the same prompts and the same seed. `scripts/make_figure.py` reads them
directly, so `assets/motion_identity.png` is regenerated from measurements and never from a
hand-copied table.

One directory per run, named like the config it was trained with in `configs/runs/`:

| directory | figure panel |
|---|---|
| `run1_write_current_chunk/` | 1. writes the frames it is generating |
| `run2_write_evicted_chunk/` | 2. writes the frames just forgotten |
| `run3_write_evicted_stable/` | 3. same, plus the stability fixes |
| `run4_from_longlive_lora/` | 4. same, from the previous fine-tune |

`evalN` is the training step. A `_sink1` suffix means the evaluation ran with one sink frame
instead of three, `_ema` reads the smoothed weights, `_480` doubles the video length. The figure
draws only the plain `evalN[_sink1]` series, so every point on a line is one comparable setting.

The evaluation script wrote a hardcoded run name and step into every header it produced. Those
labels were wrong and have been rewritten from the directory they sit in; no measurement was
touched.
