#!/usr/bin/env python
"""README figure: one panel per training run.

Each panel traces the memory-on and memory-zeroed arms across checkpoints, against that
run's own unmodified-model reference.

Small multiples rather than one scatter, because the four runs are four different models
and l1 does not even share the others' base or reference point. A single cloud implied a
homogeneity that does not exist, and hid what matters: how each run moves as training
proceeds.

Each run was also evaluated with an EMA checkpoint and at 480 frames. Those are different
evaluation settings of the same step, not different steps, so mixing them into one
trajectory makes the line meaningless. Only the plain eval<step>[_sink1] series is drawn.

Reads reference/runs/*/eval*/metrics_c1_step*.txt directly, so the figure comes from the
measured numbers and never from a hand-copied table.

    python reference/runs/make_figure.py        # writes assets/motion_identity.png
"""
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# Panels are numbered and titled by what changed between runs. The directory names on the
# left are internal run codes, kept only so each panel can be traced back to reference/runs/.
#
# All four share the same backbone (Wan2.1-1.3B, frozen), the same LoRA co-training (rank
# 256, unfrozen on every block) and the same objective: plain DMD distillation. Every extra
# loss term in the codebase (identity, anti-freeze, counterfactual advantage, motion
# matching, anti-flicker) is weighted 0 in all four configs. What differs is in the titles.
RUNS = [("c1_cotrain", "1. writes the frames it is generating"),
        ("c2_eviction", "2. writes the frames just forgotten"),
        ("c3_stable", "3. same, plus stability fixes"),
        ("l1_longlive", "4. same, from the previous fine-tune")]
KEEP = re.compile(r"^eval(\d+)(_sink1)?$")      # plain series only: no _ema, no _480, no _msc05
C = {"memon": "#2a78d6", "memzero": "#eb6834", "stock": "#1baf7a"}
M = {"memon": "o", "memzero": "^", "stock": "*"}
INK, INK2, MUTED, LINE = "#0b0b0b", "#52514e", "#8a8880", "#dedad3"


def load(run):
    out = {}
    base = os.path.join("reference/runs", run)
    for d in sorted(os.listdir(base)):
        m = KEEP.match(d)
        if not m:
            continue
        files = [f for f in os.listdir(os.path.join(base, d))
                 if f.startswith("metrics_c1_step") and f.endswith(".txt") and "stdout" not in f]
        if not files:
            continue
        t = open(os.path.join(base, d, sorted(files)[0])).read()
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


fig, axes = plt.subplots(2, 2, figsize=(9.4, 7.0), dpi=200, sharex=True, sharey=True)
fig.patch.set_facecolor("#ffffff")

for ax, (run, label) in zip(axes.ravel(), RUNS):
    ax.set_facecolor("#ffffff")
    pts = load(run)
    for arm in ("memzero", "memon"):
        xs = [d[arm] for _, _, d in pts]
        ys = [i[arm] for _, i, _ in pts]
        ax.plot(xs, ys, "-", color=C[arm], lw=1.4, alpha=.55, zorder=2)
        ax.scatter(xs, ys, s=46, marker=M[arm], c=C[arm], linewidths=1.1,
                   edgecolors="#ffffff", zorder=3)
    sx, sy = pts[0][2]["stock"], pts[0][1]["stock"]
    ax.scatter([sx], [sy], s=240, marker=M["stock"], c=C["stock"],
               linewidths=1.2, edgecolors="#ffffff", zorder=4)
    # arrowhead on the last leg, so the direction of training is unambiguous
    for arm in ("memzero", "memon"):
        (x0, y0), (x1, y1) = ((pts[-2][2][arm], pts[-2][1][arm]), (pts[-1][2][arm], pts[-1][1][arm]))
        ax.annotate("", xy=(x1, y1), xytext=(x0 + (x1 - x0) * .55, y0 + (y1 - y0) * .55),
                    arrowprops=dict(arrowstyle="-|>", color=C[arm], lw=1.4, alpha=.8,
                                    shrinkA=0, shrinkB=6), zorder=2)
    ax.annotate(f"step {pts[0][0]}", (pts[0][2]["memon"], pts[0][1]["memon"]),
                textcoords="offset points", xytext=(8, 6), fontsize=8, color=MUTED)
    ax.annotate(f"step {pts[-1][0]}", (pts[-1][2]["memon"], pts[-1][1]["memon"]),
                textcoords="offset points", xytext=(8, -13), fontsize=8, color=MUTED)
    ax.set_title(label, fontsize=10.5, color=INK, fontweight="bold", loc="left", pad=6)
    ax.grid(True, color=LINE, lw=.7, alpha=.9)
    ax.set_axisbelow(True)
    for s_ in ("top", "right"):
        ax.spines[s_].set_visible(False)
    for s_ in ("left", "bottom"):
        ax.spines[s_].set_color(LINE)
    ax.tick_params(colors=MUTED, labelsize=9, length=0)

axes[0][0].set_xlim(-0.05, 1.09)
axes[0][0].set_ylim(0.22, 0.83)
fig.supxlabel("motion   (VBench dynamic degree)", fontsize=10.5, color=INK2, y=0.055)
fig.supylabel("identity kept   (DINO, first vs last frame)", fontsize=10.5, color=INK2, x=0.035)
# Title only: the axes are labelled, the legend names the series, the step labels give the
# direction. Caveats (6 prompts per point, EMA and 480-frame evaluations excluded) are in the
# README paragraph next to the figure.
fig.suptitle("No run beats the unmodified model",
             fontsize=14, color=INK, fontweight="bold", x=0.035, ha="left", y=0.965)
fig.legend(handles=[
    Line2D([], [], marker=M["stock"], color="none", markerfacecolor=C["stock"],
           markeredgecolor="#ffffff", markersize=16, label="unmodified model, this run's own baseline"),
    Line2D([], [], marker=M["memzero"], color=C["memzero"], markerfacecolor=C["memzero"],
           markeredgecolor="#ffffff", markersize=8, lw=1.4, label="memory zeroed"),
    Line2D([], [], marker=M["memon"], color=C["memon"], markerfacecolor=C["memon"],
           markeredgecolor="#ffffff", markersize=8, lw=1.4, label="memory active")],
    loc="lower center", ncol=3, frameon=False, fontsize=9.5, labelcolor=INK2,
    handletextpad=.5, columnspacing=2.2, bbox_to_anchor=(0.5, -0.005))

fig.tight_layout(rect=[0.045, 0.075, 1, 0.925])
fig.savefig("assets/motion_identity.png", facecolor="#ffffff", bbox_inches="tight")

# The title is a claim about the data, so check it. "Beats" means strictly better on both
# axes than this run's own unmodified reference.
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
