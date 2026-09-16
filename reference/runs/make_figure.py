#!/usr/bin/env python
# ============================================================================================
# The README figure: motion against identity, for the memory-on / memory-zeroed / unmodified
# arms, over every evaluation still on disk.
#
# Reads reference/runs/*/eval*/metrics_c1_step*.txt directly, so the figure is regenerated
# from the measured numbers and never from a hand-copied table.
#
#   python reference/runs/make_figure.py        # writes assets/motion_identity.png
# ============================================================================================
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

def load():
    import os, re
    rows = []
    for root, _, files in os.walk("reference/runs"):
        for f in files:
            if not (f.startswith("metrics_c1_step") and f.endswith(".txt")) or "stdout" in f:
                continue
            t = open(os.path.join(root, f)).read()
            ident, dd, sc = {}, {}, {}
            for line in t.split("\n"):
                m = re.match(r"^(memon|memzero|stock)\s+\d+\s+[\d.]+\s+[\d.]+\s+\S+\s+([\d.]+)", line)
                if m:
                    ident[m.group(1)] = float(m.group(2))
                m = re.match(r"^DD\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)", line)
                if m:
                    dd = dict(zip(("memon", "memzero", "stock"), map(float, m.groups())))
                m = re.match(r"^SC\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)", line)
                if m:
                    sc = dict(zip(("memon", "memzero", "stock"), map(float, m.groups())))
            if ident and dd and sc:
                rows.append({"ident": ident, "dd": dd, "sc": sc})
    assert rows, "no metrics file found - run from the repo root"
    return rows


rows = load()
C = {"memon": "#2a78d6", "memzero": "#eb6834", "stock": "#1baf7a"}
M = {"memon": "o", "memzero": "^", "stock": "*"}          # secondary encoding, not colour alone
INK, INK2, MUTED, LINE = "#0b0b0b", "#52514e", "#8a8880", "#dedad3"

fig, ax = plt.subplots(figsize=(7.8, 5.2), dpi=200)
fig.patch.set_facecolor("#ffffff"); ax.set_facecolor("#ffffff")

for arm, size in (("memzero", 54), ("memon", 54)):
    ax.scatter([r["dd"][arm] for r in rows], [r["ident"][arm] for r in rows],
               s=size, marker=M[arm], c=C[arm], alpha=.85,
               linewidths=1.1, edgecolors="#ffffff", zorder=3)
stock = sorted({(r["dd"]["stock"], r["ident"]["stock"]) for r in rows})
ax.scatter([p[0] for p in stock], [p[1] for p in stock], s=300, marker=M["stock"],
           c=C["stock"], linewidths=1.2, edgecolors="#ffffff", zorder=5)

ax.annotate("a frozen video still scores\n0.93 on subject consistency",
            xy=(0.012, 0.662), xytext=(0.085, 0.762), fontsize=9.5, color=INK2,
            arrowprops=dict(arrowstyle="-", color=MUTED, lw=1,
                            connectionstyle="angle3,angleA=0,angleB=60"), va="center")

ax.set_xlabel("motion   (VBench dynamic degree)", fontsize=10.5, color=INK2, labelpad=8)
ax.set_ylabel("identity kept   (DINO, first vs last frame)", fontsize=10.5, color=INK2, labelpad=8)
ax.set_title("Training the memory costs motion and does not return identity",
             fontsize=13.5, color=INK, fontweight="bold", loc="left", pad=30)
ax.text(0, 1.055, "one point per checkpoint, 6 prompts each · 30 evaluations over four training "
        "runs · better is up and to the right",
        transform=ax.transAxes, fontsize=9, color=MUTED)

ax.legend(handles=[
    Line2D([], [], marker=M["stock"], color="none", markerfacecolor=C["stock"],
           markeredgecolor="#ffffff", markersize=17, label="Anchor-Forcing, unmodified"),
    Line2D([], [], marker=M["memzero"], color="none", markerfacecolor=C["memzero"],
           markeredgecolor="#ffffff", markersize=9, label="our training, memory zeroed"),
    Line2D([], [], marker=M["memon"], color="none", markerfacecolor=C["memon"],
           markeredgecolor="#ffffff", markersize=9, label="our training, memory active")],
    loc="lower left", frameon=False, fontsize=9.5, labelcolor=INK2,
    handletextpad=.4, borderpad=0, labelspacing=.55)

ax.set_xlim(-0.05, 1.09); ax.set_ylim(0.22, 0.83)
ax.grid(True, color=LINE, lw=.8, alpha=.9); ax.set_axisbelow(True)
for s in ("top", "right"): ax.spines[s].set_visible(False)
for s in ("left", "bottom"): ax.spines[s].set_color(LINE)
ax.tick_params(colors=MUTED, labelsize=9.5, length=0)
fig.tight_layout()
fig.savefig("assets/motion_identity.png", facecolor="#ffffff", bbox_inches="tight")

mz = max(r["ident"]["memzero"] for r in rows); mo = max(r["ident"]["memon"] for r in rows)
print(f"controle du titre : identite max memzero {mz:.3f}, memon {mo:.3f}, stock {stock[-1][1]:.3f}")
print("aucun bras n'atteint le mouvement de stock en gardant son identite :",
      not any(r["dd"][a] >= stock[-1][0] and r["ident"][a] >= stock[-1][1]
              for r in rows for a in ("memon", "memzero")))
