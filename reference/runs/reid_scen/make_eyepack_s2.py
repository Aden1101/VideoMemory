#!/usr/bin/env python
# ============================================================================================
# EYEPACK S2 (audit 2026-08-02) -- for the eye, not for a metric.
#
# In S2 the subject A does NOT open the video and the attention sink pins the OPENER for the
# whole rollout, so the segment the campaign scores against ("A's first appearance") shows the
# opener, not A. This pack shows that directly:
#
#   * 3 hstacked mp4s [memzero | framebank | memdec20], native resolution, crf 23;
#   * one PNG board per cell-row: [contaminated reference | CLEAN A* | revisit of each arm].
#
# CLEAN A* = segment 0 of the S3 rollout of the SAME cell/seed, where A opens and nothing
# precedes it -- the "what A actually looks like here" picture.
# ============================================================================================
import os
import subprocess
import sys

REPO = "/shared/ssd_14T/home/adenelmi/Anchor-Forcing"
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [REPO, os.path.join(REPO, "reference/runs/reid_probe_l1"), HERE]

import torch                                                          # noqa: E402
from torchvision.io import read_video                                 # noqa: E402
import scenarios as S                                                 # noqa: E402

VID = "/shared/ssd_14T_project/adenelmi/reid_scen/videos"
OUT = os.path.join(HERE, "eyepack_s2")
CELLS = [("t2", "s0"), ("t4", "s0"), ("t11", "s0")]
ARMS = ["memzero", "framebank", "memdec20"]
FONT = ("/shared/ssd_14T/home/adenelmi/miniconda3/envs/longlive/lib/python3.10/"
        "site-packages/matplotlib/mpl-data/fonts/ttf/DejaVuSans.ttf")


def lat2pix(f):
    return 0 if f == 0 else 4 * f - 3


def hstack_mp4(cell, out):
    """[memzero | framebank | memdec20] side by side, labelled, native resolution, crf 23."""
    ti, si = cell
    ins, filt = [], []
    for i, a in enumerate(ARMS):
        ins += ["-i", f"{VID}/S2_{ti}_{si}_{a}.mp4"]
        filt.append(f"[{i}:v]drawtext=fontfile={FONT}:text='{a}':x=12:y=12:fontsize=34:"
                    f"fontcolor=white:box=1:boxcolor=black@0.6:boxborderw=8[v{i}]")
    filt.append("".join(f"[v{i}]" for i in range(len(ARMS))) + f"hstack=inputs={len(ARMS)}[out]")
    cmd = (["ffmpeg", "-y", "-hide_banner", "-loglevel", "error"] + ins
           + ["-filter_complex", ";".join(filt), "-map", "[out]",
              "-c:v", "libx264", "-crf", "23", "-pix_fmt", "yuv420p", out])
    subprocess.run(cmd, check=True)
    mb = os.path.getsize(out) / 1e6
    print(f"[eyepack] {os.path.basename(out)}  {mb:.1f} MB", flush=True)
    assert mb < 30, f"{out} is {mb:.1f} MB (> 30 MB budget)"


def board(path):
    """Rows = cells, columns = [contaminated reference | CLEAN A* | revisit per arm]."""
    from PIL import Image, ImageDraw, ImageFont
    w_fa = S.score_segments("S2", 240)[1]                              # (72, 117)
    w_rev = S.score_segments("S2", 240)[S.revisit_segments("S2")[0]]   # (192, 239)
    w_clean = S.score_segments("S3", 300)[0]                           # (20, 72)
    cols = ["reference du scoring\n(S2 seg1 ' 1re apparition de A ')",
            "CLEAN A*\n(S3 seg0, meme seed, A ouvre)"] + [f"retour S2\n{a}" for a in ARMS]

    def frame(path_mp4, win):
        v, _, _ = read_video(path_mp4, pts_unit="sec", output_format="THWC")
        p = min(lat2pix((win[0] + win[1]) // 2), v.shape[0] - 1)
        return Image.fromarray(v[p].numpy())

    tiles = []
    for ti, si in CELLS:
        row = [frame(f"{VID}/S2_{ti}_{si}_memzero.mp4", w_fa),
               frame(f"{VID}/S3_{ti}_{si}_memzero.mp4", w_clean)]
        row += [frame(f"{VID}/S2_{ti}_{si}_{a}.mp4", w_rev) for a in ARMS]
        tiles.append(row)
        print(f"[eyepack] board row {ti}_{si} built", flush=True)

    tw, th = tiles[0][0].size
    sc = 0.5
    tw, th = int(tw * sc), int(th * sc)
    head, lab, pad = 72, 150, 6
    W = lab + len(cols) * (tw + pad) + pad
    H = head + len(CELLS) * (th + pad) + pad
    img = Image.new("RGB", (W, H), (18, 18, 20))
    d = ImageDraw.Draw(img)
    f1 = ImageFont.truetype(FONT, 19)
    f2 = ImageFont.truetype(FONT, 17)
    for j, c in enumerate(cols):
        x = lab + j * (tw + pad)
        for k, line in enumerate(c.split("\n")):
            d.text((x + 6, 10 + k * 24), line, font=f1,
                   fill=(255, 210, 120) if j >= 2 else (170, 200, 255))
    for i, (ti, si) in enumerate(CELLS):
        y = head + i * (th + pad)
        A = S.TRIPLES[int(ti[1:])][0]
        d.text((6, y + 8), f"S2_{ti}_{si}", font=f1, fill=(240, 240, 240))
        words, line, ln = A.split(), "", 0
        for wd in words:
            if len(line) + len(wd) > 18:
                d.text((6, y + 36 + ln * 20), line, font=f2, fill=(150, 150, 150))
                line, ln = wd, ln + 1
            else:
                line = (line + " " + wd).strip()
        d.text((6, y + 36 + ln * 20), line, font=f2, fill=(150, 150, 150))
        for j, im in enumerate(tiles[i]):
            img.paste(im.resize((tw, th)), (lab + j * (tw + pad), y))
    img.save(path)
    print(f"[eyepack] {path}  ({os.path.getsize(path)/1e6:.1f} MB)", flush=True)


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    for cell in CELLS:
        hstack_mp4(cell, os.path.join(OUT, f"S2_{cell[0]}_{cell[1]}_memzero-framebank-memdec20.mp4"))
    board(os.path.join(OUT, "S2_board_reference_vs_cleanA_vs_arms.png"))
    print("EYEPACK DONE")
