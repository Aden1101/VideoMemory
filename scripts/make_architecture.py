#!/usr/bin/env python
"""README diagrams, one question each. What the cell computes is given as equations in the
README, not drawn.

  assets/arch_context.png   what the memory is fed: three consecutive generation steps on one
                            time axis, the window sliding and the chunk it drops going to the memory
  assets/arch_block.png     where the memory sits in a transformer block

    python scripts/make_architecture.py
"""
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# the colours of scripts/make_figure.py: blue is the memory, green the sink, orange what attention loses
BLUE, ORANGE, GREEN = "#2a78d6", "#eb6834", "#1baf7a"
BLUE_F, ORANGE_F, GREEN_F, GREY_F = "#e4eefb", "#fdebe3", "#e2f5ee", "#efede8"
INK, INK2, MUTED, LINE = "#0b0b0b", "#52514e", "#8a8880", "#d9d5cd"


class Canvas:
    """A blank figure in data units, with the few shapes the diagrams are made of."""

    def __init__(self, width, height, scale=0.62):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.rcParams["font.family"] = ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"]
        plt.rcParams["mathtext.fontset"] = "stixsans"
        self.plt = plt
        self.fig, self.ax = plt.subplots(figsize=(width * scale, height * scale))
        self.ax.set_xlim(0, width)
        self.ax.set_ylim(0, height)
        self.ax.set_aspect("equal")
        self.ax.axis("off")
        self.fig.subplots_adjust(0, 0, 1, 1)

    def box(self, x, y, w, h, face="#ffffff", edge=INK2, lw=1.0, dashed=False, radius=0.12, z=2):
        from matplotlib.patches import FancyBboxPatch
        self.ax.add_patch(FancyBboxPatch(
            (x, y), w, h, boxstyle=f"round,pad=0,rounding_size={radius}", facecolor=face,
            edgecolor=edge, linewidth=lw, linestyle=(0, (3, 2.2)) if dashed else "solid", zorder=z))

    def module(self, x, y, w, h, label, face="#ffffff", edge=INK2, size=9.5, colour=INK, lw=1.0):
        """A labelled box, centred on (x, y)."""
        self.box(x - w / 2, y - h / 2, w, h, face, edge, lw=lw)
        self.text(x, y, label, size=size, colour=colour)

    def text(self, x, y, s, size=9.5, colour=INK2, weight="normal", ha="center", va="center",
             style="normal"):
        self.ax.text(x, y, s, fontsize=size, color=colour, fontweight=weight, ha=ha, va=va,
                     fontstyle=style, linespacing=1.3, zorder=6)

    def arrow(self, *points, colour=INK2, lw=1.1, head=True, dashed=False):
        """A polyline through `points`, with an arrowhead at the end."""
        from matplotlib.patches import FancyArrowPatch
        style = (0, (3, 2.2)) if dashed else "solid"
        xs, ys = zip(*points)
        if len(points) > 2:
            self.ax.plot(xs[:-1], ys[:-1], color=colour, lw=lw, linestyle=style,
                         solid_capstyle="round", solid_joinstyle="round", zorder=3)
        self.ax.add_patch(FancyArrowPatch(
            points[-2], points[-1], arrowstyle="-|>" if head else "-", mutation_scale=9,
            color=colour, linewidth=lw, linestyle=style, shrinkA=0, shrinkB=0, zorder=3))

    def plus(self, x, y, colour=INK2, r=0.17):
        self.ax.add_patch(self.plt.Circle((x, y), r, facecolor="#ffffff", edgecolor=colour,
                                          lw=1.0, zorder=4))
        self.ax.plot([x - r * 0.55, x + r * 0.55], [y, y], color=colour, lw=1.0, zorder=5)
        self.ax.plot([x, x], [y - r * 0.55, y + r * 0.55], color=colour, lw=1.0, zorder=5)

    def save(self, name):
        path = os.path.join(ROOT, "assets", name)
        self.fig.savefig(path, dpi=220, facecolor="#ffffff", bbox_inches="tight", pad_inches=0.08)
        self.plt.close(self.fig)
        print(f"wrote {path}")


# ---- 1. what the memory is fed ---------------------------------------------------------------
def draw_context():
    """Three consecutive steps on one time axis. At the step that generates chunk c, attention
    sees the sink (chunk 0) and the window (chunks c-3 to c-1); chunk c-4 has just left the
    window and is what the memory writes; the chunks before it are held by the memory alone."""
    chunks, cell, gap, chunk_gap = 8, 0.34, 0.06, 0.30
    chunk_w = 3 * cell + 2 * gap
    left, right = 2.35, 2.35 + chunks * chunk_w + (chunks - 1) * chunk_gap
    c = Canvas(right + 2.5, 6.35)

    def chunk_x(i):
        return left + i * (chunk_w + chunk_gap)

    def draw_chunk(i, y, face, edge, dashed=False, lw=1.0):
        for k in range(3):
            c.box(chunk_x(i) + k * (cell + gap), y, cell, 0.52, face, edge, lw=lw, dashed=dashed,
                  radius=0.05)

    # legend
    items = [("sink, never evicted", GREEN_F, GREEN, False), ("attention window", GREY_F, MUTED, False),
             ("being generated", BLUE_F, BLUE, False), ("written to the memory", ORANGE_F, ORANGE, False),
             ("held by the memory only", "#ffffff", LINE, True)]
    for offset, (label, face, edge, dashed) in zip((0, 2.75, 5.2, 7.55, 10.4), items):
        x = left + offset
        c.box(x, 5.72, 0.3, 0.3, face, edge, dashed=dashed, radius=0.05)
        c.text(x + 0.42, 5.87, label, size=8.6, ha="left")

    for row, current in enumerate((5, 6, 7)):
        y = 4.15 - row * 1.5
        c.text(left - 0.35, y + 0.26, f"generating\nchunk {current}", size=8.8, colour=INK2, ha="right")
        for i in range(chunks):
            if i == 0:
                draw_chunk(i, y, GREEN_F, GREEN)
            elif i == current:
                draw_chunk(i, y, BLUE_F, BLUE, lw=1.3)
            elif current - 3 <= i < current:
                draw_chunk(i, y, GREY_F, MUTED)
            elif i == current - 4:
                draw_chunk(i, y, ORANGE_F, ORANGE, lw=1.3)
            elif i < current - 4:
                draw_chunk(i, y, "#ffffff", LINE, dashed=True)
        # the bracket of what attention reads: window, plus the sink on its own
        wl, wr = chunk_x(current - 3), chunk_x(current - 1) + chunk_w
        c.ax.plot([wl, wl, wr, wr], [y + 0.62, y + 0.72, y + 0.72, y + 0.62], color=MUTED, lw=0.9)
        # the evicted chunk goes to the memory
        ex = chunk_x(current - 4) + chunk_w / 2
        c.arrow((ex, y - 0.08), (ex, y - 0.42), (right + 0.62, y - 0.42), colour=ORANGE)
        c.text(right + 0.5, y - 0.22, f"write chunk {current - 4}", size=8.4, colour=ORANGE, ha="right")

    c.text((chunk_x(2) + chunk_x(4) + chunk_w) / 2, 4.15 + 0.95, "rolling window, 9 frames", size=8.2,
           colour=MUTED)
    # one memory for the whole rollout: every step writes into the same state
    c.box(right + 0.65, 0.45, 1.5, 4.4, BLUE_F, BLUE, lw=1.2, radius=0.14)
    c.text(right + 1.4, 2.78, "memory\n$W$", size=10.5, colour=INK)
    # time axis
    ay = 0.3
    c.arrow((left, ay), (right + 0.1, ay), colour=MUTED, lw=0.9)
    for i in range(chunks):
        c.text(chunk_x(i) + chunk_w / 2, ay - 0.27, str(i), size=8.2, colour=MUTED)
    c.text(left - 0.35, ay - 0.27, "chunk", size=8.2, colour=MUTED, ha="right")
    c.text(right + 0.1, ay - 0.27, "time", size=8.2, colour=MUTED, ha="right")
    c.save("arch_context.png")


# ---- 2. where the memory sits ----------------------------------------------------------------
def draw_block():
    """The block, left to right along the residual stream: self-attention, the memory branch,
    cross-attention to the prompt, the feed-forward layer. The memory reads the normalised input
    of the block and adds a gated output; its state lives outside the block, across chunks."""
    c = Canvas(15.6, 5.6)
    y = 1.7                                    # the residual stream
    c.box(1.15, 1.05, 13.3, 3.3, face="#fbfaf8", edge=LINE, lw=1.0, radius=0.18, z=1)
    c.text(14.2, 4.08, "transformer block", size=9.2, colour=MUTED, ha="right", style="italic")
    c.arrow((0.2, y), (15.3, y), colour=INK, lw=1.4)
    c.text(0.2, y + 0.3, r"$x_{\mathrm{in}}$", size=11, colour=INK, ha="left")
    c.text(15.3, y + 0.3, r"$x_{\mathrm{out}}$", size=11, colour=INK, ha="right")

    def branch(x_in, x_out, label, face, edge, colour=INK, top=2.75, w=2.1, tap=True):
        mid = (x_in + x_out) / 2
        if tap:
            c.arrow((x_in, y), (x_in, top), (mid - w / 2, top), colour=edge)
        c.module(mid, top, w, 0.78, label, face=face, edge=edge, colour=colour)
        c.arrow((mid + w / 2, top), (x_out, top), (x_out, y + 0.17), colour=edge)
        c.plus(x_out, y, colour=edge if edge != MUTED else INK2)

    branch(1.9, 4.6, "self-attention", GREY_F, MUTED)
    c.text(3.25, 2.2, "sink + window", size=8.2, colour=MUTED)
    # the memory branch reads the same normalised input as self-attention, and adds after it
    branch(5.3, 8.4, "memory read + gate", BLUE_F, BLUE, w=2.5, tap=False)
    c.arrow((1.9, 2.75), (1.9, 3.6), (5.2, 3.6), (5.2, 2.75), (5.6, 2.75), colour=BLUE)
    c.text(3.55, 3.8, "same normalised input", size=8.2, colour=BLUE)
    branch(9.1, 11.6, "cross-attention", GREY_F, MUTED)
    c.text(10.35, 2.2, "prompt", size=8.2, colour=MUTED)
    branch(12.2, 14.0, "FFN", GREY_F, MUTED, w=1.2)

    # the state, outside the block
    c.module(6.85, 5.1, 1.5, 0.62, "state $W$", face=BLUE_F, edge=BLUE, lw=1.2)
    c.arrow((6.85, 4.79), (6.85, 3.16), colour=BLUE)
    c.arrow((3.0, 5.1), (6.08, 5.1), colour=ORANGE)
    c.text(2.9, 5.1, "evicted chunk", size=9, colour=ORANGE, ha="right")
    c.text(4.55, 5.33, "write", size=8.4, colour=ORANGE)
    c.arrow((7.62, 5.1), (9.4, 5.1), colour=BLUE, dashed=True)
    c.text(9.5, 5.1, "carried to the next chunk", size=8.6, colour=BLUE, ha="left")
    c.text(7.8, 0.72, "in blocks 5, 16 and 27 of 30", size=8.6, colour=MUTED)
    c.save("arch_block.png")


if __name__ == "__main__":
    draw_context()
    draw_block()
