#!/usr/bin/env python
"""README diagram: where the memory sits and what it is fed.

Top, the attention context of one transformer block while it generates a chunk: three sink
frames that are never evicted, a rolling window of nine, and the chunk that has just left the
window. Bottom left, the memory cell, which is written with that evicted chunk. Bottom right,
the block: attention reads the context, the memory branch reads the cell with the current
chunk's queries, and its output is gated and added to the residual stream.

    python scripts/make_architecture.py         # writes assets/architecture.png
"""
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# the colours of scripts/make_figure.py: blue is the memory, green what is kept, orange what is lost
BLUE, ORANGE, GREEN = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, MUTED, LINE = "#0b0b0b", "#52514e", "#8a8880", "#dedad3"
FRAME, GAP = 0.62, 0.10          # width of one latent frame and the space between two


def draw(path=os.path.join(ROOT, "assets", "architecture.png")):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

    fig, ax = plt.subplots(figsize=(11.5, 6.7))
    ax.set_xlim(0, 16.6)
    ax.set_ylim(0, 9.7)
    ax.axis("off")

    def box(x, y, w, h, face, edge, dashed=False, lw=1.2, radius=0.08):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={radius}",
                                    facecolor=face, edgecolor=edge, linewidth=lw,
                                    linestyle=(0, (4, 3)) if dashed else "solid"))

    def arrow(start, end, colour=INK2, bend=0.0, lw=1.6):
        ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=13, color=colour,
                                     linewidth=lw, connectionstyle=f"arc3,rad={bend}",
                                     shrinkA=2, shrinkB=2))

    def text(x, y, s, size=9.5, colour=INK2, weight="normal", ha="center", va="center"):
        ax.text(x, y, s, fontsize=size, color=colour, fontweight=weight, ha=ha, va=va,
                linespacing=1.35)

    def frames(x, y, n, face, edge, dashed=False):
        """n latent frames side by side from x; returns the x where the group ends."""
        for i in range(n):
            box(x + i * (FRAME + GAP), y, FRAME, 0.9, face, edge, dashed=dashed)
        return x + n * (FRAME + GAP) - GAP

    # ---- the attention context -----------------------------------------------------------
    y = 6.85
    text(0.3, 9.35, "The memory is written with the chunk that attention has just lost",
         size=11.5, colour=INK, weight="bold", ha="left")

    x0 = 0.3
    x1 = frames(x0, y, 3, "#fdece4", ORANGE, dashed=True)
    text((x0 + x1) / 2, y + 1.22, "evicted chunk", colour=ORANGE, weight="bold")
    text((x0 + x1) / 2, y - 0.45, "attention no\nlonger sees it", size=8.5, colour=MUTED)

    xs = x1 + 0.75
    xe = frames(xs, y, 3, "#e3f5ee", GREEN)
    text((xs + xe) / 2, y + 1.22, "sink", colour=GREEN, weight="bold")
    text((xs + xe) / 2, y - 0.45, "3 frames,\nnever evicted", size=8.5, colour=MUTED)

    xw = xe + 0.45
    xwe = xw
    for _ in range(3):                                  # nine frames, drawn as three chunks
        xwe = frames(xwe, y, 3, "#eeece7", MUTED) + 0.22
    xwe -= 0.22
    text((xw + xwe) / 2, y + 1.22, "rolling window", colour=INK2, weight="bold")
    text((xw + xwe) / 2, y - 0.45, "9 frames: when a new chunk enters,\nthe oldest one leaves",
         size=8.5, colour=MUTED)

    xc = xwe + 0.75
    xce = frames(xc, y, 3, "#e1edfa", BLUE)
    text((xc + xce) / 2, y + 1.22, "current chunk", colour=BLUE, weight="bold")
    text((xc + xce) / 2, y - 0.45, "3 frames\nbeing generated", size=8.5, colour=MUTED)

    # the context bracket: sink + window
    ax.plot([xs, xs, xwe, xwe], [y + 1.62, y + 1.74, y + 1.74, y + 1.62], color=LINE, lw=1.4)
    text((xs + xwe) / 2, y + 1.98, "KV cache: 12 frames of context", size=9, colour=MUTED)

    # ---- the memory cell -----------------------------------------------------------------
    box(0.3, 1.0, 5.6, 3.5, "#f4f8fd", BLUE, lw=1.6, radius=0.14)
    text(3.1, 4.05, "memory cell", size=11.5, colour=BLUE, weight="bold")
    text(3.1, 3.5, "fast weight W, a small MLP (w0, w1, w2)", size=9.5)
    text(0.62, 2.62, "write", size=9.5, colour=INK, weight="bold", ha="left")
    text(0.62, 1.78, "one gradient step on the evicted chunk,\n"
                     "orthogonalised (Muon), then each column\n"
                     "renormalised to its initial norm:\n"
                     "the state can rotate, it cannot grow", size=8.8, ha="left")
    arrow(((x0 + x1) / 2, y - 0.95), ((x0 + x1) / 2, 4.55), colour=ORANGE)
    text((x0 + x1) / 2 + 0.25, 5.1, "written into the memory\nonce it has left the window", size=8.5,
         colour=ORANGE, ha="left")

    # ---- the block ------------------------------------------------------------------------
    bx = 8.0
    box(bx, 0.55, 8.3, 4.6, "#ffffff", LINE, lw=1.4, radius=0.14)
    text(bx + 0.3, 4.75, "transformer block", size=9.5, colour=INK, weight="bold", ha="left")
    text(bx + 8.3, 0.25, "blocks 5, 16 and 27 of 30 carry the memory branch", size=8.5,
         colour=MUTED, ha="right")

    stream = 1.25                                            # the residual stream, left to right
    arrow((bx + 0.35, stream), (bx + 7.95, stream), colour=INK, lw=2.0)
    text(bx + 0.35, stream - 0.36, "residual stream x", size=8.8, colour=INK2, ha="left")

    box(bx + 0.9, 2.55, 2.5, 1.25, "#eeece7", MUTED)
    text(bx + 2.15, 3.18, "self-attention\nover sink + window", size=9)
    arrow((bx + 2.15, stream), (bx + 2.15, 2.5), colour=MUTED)
    arrow((bx + 2.9, 2.5), (bx + 2.9, stream), colour=MUTED)

    box(bx + 4.1, 2.55, 1.75, 1.25, "#e1edfa", BLUE)
    text(bx + 4.97, 3.18, "read\no = f_W(q)", size=9, colour=INK)
    box(bx + 6.1, 2.55, 1.7, 1.25, "#e1edfa", BLUE)
    text(bx + 6.95, 3.18, "bounded gate\n× AdaLN(o)", size=8.4, colour=INK)
    arrow((bx + 4.55, stream), (bx + 4.55, 2.5), colour=BLUE)
    text(bx + 4.4, 1.9, "q", size=9, colour=BLUE, ha="right")
    arrow((bx + 5.87, 3.18), (bx + 6.13, 3.18), colour=BLUE)
    arrow((bx + 6.92, 2.5), (bx + 6.92, stream + 0.17), colour=BLUE)
    ax.add_patch(plt.Circle((bx + 6.92, stream), 0.17, facecolor="#ffffff", edgecolor=INK, lw=1.4,
                            zorder=5))
    ax.text(bx + 6.92, stream - 0.01, "+", fontsize=11, color=INK, fontweight="bold",
            ha="center", va="center", zorder=6)
    text(bx + 7.1, 1.9, "injection", size=8.8, colour=BLUE, ha="left")

    # the cell feeds the read; the context feeds attention; the current chunk is what the block runs on
    # the state goes over the attention box and down into the read
    ax.plot([5.95, bx + 4.97], [4.2, 4.2], color=BLUE, lw=1.6, solid_capstyle="round")
    arrow((bx + 4.97, 4.2), (bx + 4.97, 3.83), colour=BLUE)
    text(6.95, 4.45, "state W", size=8.8, colour=BLUE)
    # the current chunk is what the block runs on
    arrow(((xc + xce) / 2, y - 0.95), ((xc + xce) / 2, 5.2), colour=BLUE)
    text((xc + xce) / 2 - 0.25, 5.45, "its tokens are\nthe stream x", size=8.5, colour=BLUE, ha="right")

    fig.savefig(path, dpi=200, facecolor="#ffffff", bbox_inches="tight")
    print(f"wrote {path}")


if __name__ == "__main__":
    draw()
