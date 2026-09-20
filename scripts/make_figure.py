#!/usr/bin/env python
"""README figure: the two metrics against the training step, one column per run.

The training step is on the x axis, so time reads left to right and the reader needs no
arrowheads or start/end labels to know which way a run went. An earlier version plotted
identity against motion, with the checkpoints joined in order; neither axis was time, so the
path folded back on itself and the lines crossed for no reason the data supports.

Top row is identity, bottom row is motion, sharing a y scale across the row so the four runs
are comparable. The dashed line in each panel is that run's own unmodified-model reference,
which is why panel 4's sits lower: it starts from a different fine-tune.

Each run was also evaluated with an EMA checkpoint and at 480 frames. Those are different
evaluation settings of the same step, not different steps, so mixing them into one line makes
it meaningless. Only the plain eval<step>[_sink1] series is drawn.

Reads results/*/eval*/metrics.txt directly, so the figure comes from the
measured numbers and never from a hand-copied table.

    python scripts/make_figure.py               # writes assets/motion_identity.png
"""
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# Panels are numbered and titled by what changed between runs; the directory on the left holds
# that run's measurements, and configs/runs/ holds the config it was trained with.
#
# All four share the same backbone (Wan2.1-1.3B, frozen), the same LoRA co-training (rank
# 256, unfrozen on every block) and the same objective: plain DMD distillation. Every extra
# loss term in the codebase (identity, anti-freeze, counterfactual advantage, motion
# matching, anti-flicker) is weighted 0 in all four configs. What differs is in the titles.
RUNS = [("run1_write_current_chunk", "1. writes the frames\nit is generating"),
        ("run2_write_evicted_chunk", "2. writes the frames\njust forgotten"),
        ("run3_write_evicted_stable", "3. same, plus the\nstability fixes"),
        ("run4_from_longlive_lora", "4. same, from the\nprevious fine-tune")]
KEEP = re.compile(r"^eval(\d+)(_sink1)?$")      # plain series only: no _ema, no _480, no _msc05
# One shape throughout: round dots on a line, told apart by colour. Mixing marker shapes
# made the legend something to decode before the chart could be read.
C = {"memon": "#2a78d6", "memzero": "#eb6834", "stock": "#1baf7a"}
INK, INK2, MUTED, LINE = "#0b0b0b", "#52514e", "#8a8880", "#dedad3"


def load(run):
    out = {}
    base = os.path.join("results", run)
    for d in sorted(os.listdir(base)):
        m = KEEP.match(d)
        if not m:
            continue
        path = os.path.join(base, d, "metrics.txt")
        if not os.path.exists(path):
            continue
        t = open(path).read()
        ident, dd = {}, {}
        for line in t.split("\n"):
            g = re.match(r"^(memon|memzero|stock)\s+\d+\s+[\d.]+\s+[\d.]+\s+\S+\s+([\d.]+)", line)
            if g:
                ident[g.group(1)] = float(g.group(2))
            g = re.match(r"^DD\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)", line)
            if g:
                dd = dict(zip(("memon", "memzero", "stock"), map(float, g.groups())))
        if ident and dd:
            out[int(m.group(1))] = (ident, dd)          # keyed by step -> deduplicated
    return [(s, *out[s]) for s in sorted(out)]


fig, axes = plt.subplots(2, 4, figsize=(11.6, 6.0), dpi=200, sharey="row")
fig.patch.set_facecolor("#ffffff")

ROWS = [(1, "identity kept\n(DINO, first vs last frame)"),      # index into (step, ident, dd)
        (2, "motion\n(VBench dynamic degree)")]

for col, (run, label) in enumerate(RUNS):
    pts = load(run)
    steps = [p[0] for p in pts]
    for row, (idx, ylab) in enumerate(ROWS):
        ax = axes[row][col]
        ax.set_facecolor("#ffffff")
        ax.axhline(pts[0][idx]["stock"], color=C["stock"], ls=(0, (5, 3)), lw=1.5, zorder=2)
        for arm in ("memzero", "memon"):
            ys = [p[idx][arm] for p in pts]
            ax.plot(steps, ys, "-o", color=C[arm], lw=1.8, ms=5.5,
                    markerfacecolor=C[arm], markeredgecolor="#ffffff", markeredgewidth=1.0,
                    zorder=3)
        ax.set_xticks(steps)
        ax.set_xticklabels([str(v) for v in steps], fontsize=8.5)
        ax.set_xlim(steps[0] - (steps[-1] - steps[0]) * .12,
                    steps[-1] + (steps[-1] - steps[0]) * .12)
        ax.grid(True, axis="y", color=LINE, lw=.7, alpha=.9)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(LINE)
        ax.tick_params(colors=MUTED, labelsize=8.5, length=0)
        if col == 0:
            ax.set_ylabel(ylab, fontsize=9.5, color=INK2, labelpad=8)
    axes[0][col].set_title(label, fontsize=10.5, color=INK, fontweight="bold",
                           loc="left", pad=10, linespacing=1.35)

axes[0][0].set_ylim(0.20, 0.85)
axes[1][0].set_ylim(-0.05, 1.10)
fig.suptitle("No run beats the unmodified model on both metrics at once",
             fontsize=14, color=INK, fontweight="bold", x=0.012, ha="left", y=0.975)
fig.text(0.5, 0.055, "training step", fontsize=10, color=INK2, ha="center")
fig.legend(handles=[
    Line2D([], [], marker="o", color=C["memon"], markerfacecolor=C["memon"],
           markeredgecolor="#ffffff", markersize=6, lw=1.8, label="memory active"),
    Line2D([], [], marker="o", color=C["memzero"], markerfacecolor=C["memzero"],
           markeredgecolor="#ffffff", markersize=6, lw=1.8, label="memory zeroed"),
    Line2D([], [], color=C["stock"], ls=(0, (5, 3)), lw=1.5, label="unmodified model")],
    loc="lower center", ncol=3, frameon=False, fontsize=10, labelcolor=INK2,
    handletextpad=.7, columnspacing=3.0, bbox_to_anchor=(0.5, -0.005))

fig.tight_layout(rect=[0.005, 0.085, 1, 0.925])
fig.savefig("assets/motion_identity.png", facecolor="#ffffff", bbox_inches="tight")

# The title is a claim about the data, so check it. "Beats on both metrics" means strictly
# better on identity AND on motion than this run's own unmodified reference. Single-metric
# wins do exist and are visible in the figure: run 1 ends above its identity line, having
# given up a third of its motion to get there.
beaten = []
for run, _ in RUNS:
    pts = load(run)
    sx, sy = pts[0][2]["stock"], pts[0][1]["stock"]
    for step, i, d in pts:
        for arm in ("memon", "memzero"):
            if d[arm] >= sx and i[arm] >= sy:
                beaten.append((run, step, arm))
    print(f"{run:14} {len(pts)} checkpoints : {[s for s, _, _ in pts]}  "
          f"reference ({sx:.3f}, {sy:.3f})")
assert not beaten, f"the title is false, these points beat their reference: {beaten}"
print("\ntitle verified: no arm is above AND right of its own reference")
