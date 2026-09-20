# The numbers behind the figure

Three-way evaluation tables, one per checkpoint: the memory active, the memory zeroed, and the
unmodified model, on the same prompts and the same seed. `scripts/make_figure.py` reads them
directly, so `assets/motion_identity.png` is regenerated from measurements and never from a
hand-copied table.

One directory per run, named by its internal code so a panel of the figure can be traced back:

| directory | figure panel |
|---|---|
| `c1_cotrain/` | 1. writes the frames it is generating |
| `c2_eviction/` | 2. writes the frames just forgotten |
| `c3_stable/` | 3. same, plus the stability fixes |
| `l1_longlive/` | 4. same, from the previous fine-tune |

`evalN` is the training step. A `_sink1` suffix means the evaluation ran with one sink frame
instead of three; `_ema` reads the smoothed weights, `_480` doubles the video length. The figure
draws only the plain `evalN[_sink1]` series, so every point on a line is one comparable setting.

The header line inside each file always reads `c1_cotrain step-250`: it is hardcoded in the
evaluation script and never updated. The directory name is the reliable one.
