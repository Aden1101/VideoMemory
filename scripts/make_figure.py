#!/usr/bin/env python
"""README figure: the two metrics against the training step, one column per run.

The training step is on the x axis, so time reads left to right and the reader needs no
arrowheads or start/end labels to know which way a run went.

Top row is identity, bottom row is motion, sharing a y scale across the row so the four runs
are comparable. The dashed line in each panel is that run's own unmodified-model reference,
which is why panel 4's sits lower: it starts from a different fine-tune.

Each run was also evaluated with an EMA checkpoint and at 480 frames. Those are different
evaluation settings of the same step, not different steps, so mixing them into one line makes
it meaningless. Only the plain eval<step>[_sink1] series is drawn.

The sink size is not the same everywhere, and the panels say so. Run 1 was evaluated with 3
sink frames, runs 2 to 4 with 1. The unmodified-model reference was generated once per
starting fine-tune, with 3 sink frames. So within a panel the two arms are comparable
everywhere, but in panels 2 to 4 they pay a sink handicap the dashed line does not: measured on
run 1 at step 1250, going from sink 3 to sink 1 costs 0.06 to 0.09 of identity.

Reads results/*/eval*/metrics.txt directly, so the figure comes from the
measured numbers and never from a hand-copied table.

    python scripts/make_figure.py               # writes assets/motion_identity.png
"""
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from results_index import ARMS, ROOT, read_table  # noqa: E402

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
# sink frames the two arms were evaluated with; the reference line is sink 3 in every panel
ARM_SINK = {"run1_write_current_chunk": 3, "run2_write_evicted_chunk": 1,
            "run3_write_evicted_stable": 1, "run4_from_longlive_lora": 1}
KEEP = re.compile(r"^eval(\d+)(_sink1)?$")      # plain series only: no _ema, no _480, no _msc05
# One shape throughout: round dots on a line, told apart by colour, so the legend has nothing
# to decode.
C = {"memon": "#2a78d6", "memzero": "#eb6834", "stock": "#1baf7a"}
INK, INK2, MUTED, LINE = "#0b0b0b", "#52514e", "#8a8880", "#dedad3"


def load(run):
    """[(step, identity per arm, dynamic degree per arm), ...] of one run, by step."""
    out = {}
    base = os.path.join(ROOT, "results", run)
    # One point per step. Run 2 has step 500 at both sink settings; directories are visited in
    # sorted order, so eval500_sink1 replaces eval500 and the line stays at sink 1 throughout.
    for d in sorted(os.listdir(base)):
        m = KEEP.match(d)
        path = os.path.join(base, d, "metrics.txt")
        if not m or not os.path.exists(path):
            continue
        table = read_table(path)
        ident = {arm: table["summaries"][arm]["id_cos"] for arm in ARMS}
        dd = {arm: table["vb"][arm]["DD"] for arm in ARMS}
        out[int(m.group(1))] = (ident, dd)
    return [(s, *out[s]) for s in sorted(out)]


ROWS = [(1, "identity kept\n(DINO, first vs last frame)"),      # index into (step, ident, dd)
        (2, "motion\n(VBench dynamic degree)")]


def identity_gaps(runs):
    """memory active minus memory zeroed, on identity, at every checkpoint of every run: the
    one comparison that holds in every panel, both arms sharing checkpoint, seed and sink."""
    return [ident["memon"] - ident["memzero"] for pts in runs.values() for _, ident, _ in pts]


def draw(runs, path=os.path.join(ROOT, "assets", "motion_identity.png")):
    """Two rows, identity then motion, one column per run; the title is computed from `runs`."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    gaps = identity_gaps(runs)
    n_lower = sum(g < 0 for g in gaps)
    fig, axes = plt.subplots(2, 4, figsize=(11.6, 6.0), dpi=200, sharey="row")
    fig.patch.set_facecolor("#ffffff")

    for col, (run, label) in enumerate(RUNS):
        pts = runs[run]
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
                               loc="left", pad=22, linespacing=1.35)
        note = (f"arms at sink {ARM_SINK[run]}" if ARM_SINK[run] == 3
                else f"arms at sink {ARM_SINK[run]}, reference at sink 3")
        axes[0][col].text(0.0, 1.035, note, transform=axes[0][col].transAxes,
                          fontsize=8.5, color=MUTED, ha="left", va="bottom")

    axes[0][0].set_ylim(0.20, 0.85)
    axes[1][0].set_ylim(-0.05, 1.10)
    fig.suptitle(f"Memory on scores lower identity than memory off at {n_lower} of {len(gaps)} "
                 f"checkpoints (mean gap {sum(gaps) / len(gaps):+.2f})",
                 fontsize=14, color=INK, fontweight="bold", x=0.012, ha="left", y=0.975)
    fig.text(0.5, 0.055, "training step", fontsize=10, color=INK2, ha="center")
    fig.legend(handles=[
        Line2D([], [], marker="o", color=C["memon"], markerfacecolor=C["memon"],
               markeredgecolor="#ffffff", markersize=6, lw=1.8, label="memory active"),
        Line2D([], [], marker="o", color=C["memzero"], markerfacecolor=C["memzero"],
               markeredgecolor="#ffffff", markersize=6, lw=1.8, label="memory zeroed"),
        Line2D([], [], color=C["stock"], ls=(0, (5, 3)), lw=1.5,
               label="unmodified model (sink 3)")],
        loc="lower center", ncol=3, frameon=False, fontsize=10, labelcolor=INK2,
        handletextpad=.7, columnspacing=3.0, bbox_to_anchor=(0.5, -0.005))

    fig.tight_layout(rect=[0.005, 0.085, 1, 0.925])
    fig.savefig(path, facecolor="#ffffff", bbox_inches="tight")


def check_claims(runs):
    """The README says no arm matches its reference on both metrics at once, so check it.
    Single-metric wins exist and are visible: run 1's memory-zeroed arm ends above its identity
    line, with a third fewer clips rated dynamic. For runs 2 to 4 the check is lenient on
    identity, since the arms are at sink 1 and the reference at sink 3, and nearly unfalsifiable
    on motion for runs 1 to 3, whose reference sits at the 0.99 ceiling."""
    gaps = identity_gaps(runs)
    n_lower = sum(g < 0 for g in gaps)
    beaten = []
    for run, _ in RUNS:
        pts = runs[run]
        sx, sy = pts[0][2]["stock"], pts[0][1]["stock"]
        for step, i, d in pts:
            for arm in ("memon", "memzero"):
                if d[arm] >= sx and i[arm] >= sy:
                    beaten.append((run, step, arm))
        print(f"{run:14} {len(pts)} checkpoints : {[s for s, _, _ in pts]}  "
              f"reference ({sx:.3f}, {sy:.3f})")
    assert not beaten, f"these points match or beat their reference on both metrics: {beaten}"
    print(f"\nno arm is at or above its own reference on both metrics; "
          f"memon < memzero on identity at {n_lower}/{len(gaps)} checkpoints, "
          f"mean gap {sum(gaps) / len(gaps):+.4f}")


def main():
    runs = {run: load(run) for run, _ in RUNS}
    draw(runs)
    check_claims(runs)      # after the drawing: a failed claim still leaves the figure to look at


if __name__ == "__main__":
    main()
