#!/usr/bin/env python
# ============================================================================================
# EYEPACK LEGER du mini-round corrige (2026-08-03) -- connexion faible d'Aden :
#   * hstacks 4 bras, largeur totale 960 px, crf 30  -> cible < 2 Mo par fichier
#   * micro-clips "verdict" : [A* propre | retour config-Aden | retour wronganchor], UNIQUEMENT
#     le segment de retour (~3 s), 960 px, crf 30 -> quelques centaines de ko
#   * planche JPEG compressee (qualite 72) : 5 cellules x 8 colonnes
# Tout dans eyepack_textpoor2/.
# ============================================================================================
import os
import subprocess
import sys

REPO = "/shared/ssd_14T/home/adenelmi/Anchor-Forcing"
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [REPO, os.path.join(REPO, "reference/runs/reid_probe_l1"), HERE]

from torchvision.io import read_video                                   # noqa: E402
from PIL import Image                                                   # noqa: E402
import scenarios as S                                                   # noqa: E402

VID = "/shared/ssd_14T_project/adenelmi/reid_scen/videos"
OUT = os.path.join(HERE, "eyepack_textpoor2")
CELLS = [19, 18, 20, 21, 22]
HSTACK_ARMS = ["memzero", "memdec21un", "wronganchor", "framebank_gt"]
BOARD_ARMS = ["memzero", "memdec21un", "memdec20", "wronganchor", "noiseanchor", "framebank_gt"]
FONT = ("/shared/ssd_14T/home/adenelmi/miniconda3/envs/longlive/lib/python3.10/"
        "site-packages/matplotlib/mpl-data/fonts/ttf/DejaVuSans.ttf")


def lat2pix(f):
    return 0 if f == 0 else 4 * f - 3


def ff(args):
    subprocess.run(["ffmpeg", "-y", "-hide_banner", "-loglevel", "error"] + args, check=True)


def size_mb(p):
    return os.path.getsize(p) / 1e6


def hstack(cell, arms, out, w_total=960, crf=30, trim=None):
    """trim = (start_s, dur_s) pour ne garder que le segment de retour."""
    n = len(arms)
    w = w_total // n
    ins, filt = [], []
    for i, a in enumerate(arms):
        pre = ["-ss", str(trim[0]), "-t", str(trim[1])] if trim else []
        ins += pre + ["-i", f"{VID}/S2p_{cell}_{a}.mp4" if a != "A*" else
                      f"{VID}/SS1_{cell}_memzero.mp4"]
        lab = "A* propre" if a == "A*" else a
        filt.append(f"[{i}:v]scale={w}:-2,drawtext=fontfile={FONT}:text='{lab}':x=6:y=6:"
                    f"fontsize=16:fontcolor=white:box=1:boxcolor=black@0.6:boxborderw=4[v{i}]")
    filt.append("".join(f"[v{i}]" for i in range(n)) + f"hstack=inputs={n}[o]")
    ff(ins + ["-filter_complex", ";".join(filt), "-map", "[o]", "-c:v", "libx264",
              "-crf", str(crf), "-preset", "slow", "-pix_fmt", "yuv420p", "-an", out])
    print(f"[eye2] {os.path.basename(out)}  {size_mb(out):.2f} MB", flush=True)
    return size_mb(out)


def board(path):
    w_rev = S.score_segments("S2p", 240)[3]
    w_b = S.score_segments("S2p", 240)[2]
    w_clean = S.score_segments("SS1", 240)[0]
    cols = ["A* PROPRE", "B interlude (vu)"] + BOARD_ARMS
    cache = {}

    def frame(p, w):
        if (p, w) not in cache:
            v, _, _ = read_video(p, pts_unit="sec", output_format="THWC")
            i = min(lat2pix((w[0] + w[1]) // 2), v.shape[0] - 1)
            cache[(p, w)] = Image.fromarray(v[i].numpy()).resize((260, 150))
        return cache[(p, w)]

    from PIL import ImageDraw, ImageFont
    tw, th, lab, head, pad = 260, 150, 150, 54, 4
    img = Image.new("RGB", (lab + len(cols) * (tw + pad) + pad,
                            head + len(CELLS) * (th + pad) + pad), (18, 18, 20))
    d = ImageDraw.Draw(img)
    f1 = ImageFont.truetype(FONT, 15)
    f2 = ImageFont.truetype(FONT, 13)
    for j, c in enumerate(cols):
        d.text((lab + j * (tw + pad) + 4, 18), c, font=f1,
               fill=(255, 210, 120) if j >= 2 else (170, 200, 255))
    for i, ti in enumerate(CELLS):
        cell = f"t{ti}_s0"
        y = head + i * (th + pad)
        att = S.TEXTPOOR_ATTR.get(ti)
        d.text((4, y + 6), f"t{ti}", font=f1, fill=(240, 240, 240))
        d.text((4, y + 26), (f"{att[0]} vs {att[1]}" if att else "cheval corrige"), font=f2,
               fill=(150, 150, 150))
        d.text((4, y + 44), f"'{S.PARAPHRASES[ti][:22]}'", font=f2, fill=(120, 120, 120))
        row = [frame(f"{VID}/SS1_{cell}_memzero.mp4", w_clean),
               frame(f"{VID}/S2p_{cell}_memzero.mp4", w_b)]
        row += [frame(f"{VID}/S2p_{cell}_{a}.mp4", w_rev) for a in BOARD_ARMS]
        for j, im in enumerate(row):
            img.paste(im, (lab + j * (tw + pad), y))
    img.save(path, quality=72, optimize=True)
    print(f"[eye2] {os.path.basename(path)}  {size_mb(path):.2f} MB", flush=True)


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    # le retour = frames latentes 192-239 -> pixels ~765-956 -> 47.8 s a 16 fps, ~3 s
    trim = (lat2pix(192) / 16.0, (956 - 765) / 16.0)
    big = []
    for ti in CELLS:
        c = f"t{ti}_s0"
        big.append(hstack(c, HSTACK_ARMS, os.path.join(OUT, f"S2p_{c}_4bras.mp4")))
        hstack(c, ["A*", "memdec21un", "wronganchor"],
               os.path.join(OUT, f"S2p_{c}_verdict.mp4"), trim=None if False else trim)
    board(os.path.join(OUT, "S2p_board_textpoor2.jpg"))
    assert max(big) < 2.0, f"un hstack depasse 2 Mo ({max(big):.2f})"
    print("EYEPACK2 DONE")
