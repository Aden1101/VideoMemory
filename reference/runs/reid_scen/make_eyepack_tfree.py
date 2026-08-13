#!/usr/bin/env python
# ============================================================================================
# EYEPACK LEGER de la mini-batterie training-free (2026-08-04) -- connexion faible d'Aden :
#   * hstacks 4 bras, largeur 960 px, crf 30            -> < 2 Mo par fichier
#   * micro-clips "verdict" : [A* propre | ours | wrong], SEGMENT DE RETOUR seulement (~3 s)
#   * planche JPEG (qualite 72) : 6 cellules x 7 colonnes
# La planche des SIGMAS (l'ancre affutee elle-meme) est produite a part, AVANT les rollouts, par
# make_sdedit_board.py -- c'est elle qui a servi de gate au choix de sigma.
# Tout dans eyepack_tfree/.
# ============================================================================================
import argparse
import os
import subprocess
import sys

REPO = "/shared/ssd_14T/home/adenelmi/Anchor-Forcing"
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [REPO, os.path.join(REPO, "reference/runs/reid_probe_l1"), HERE]

from torchvision.io import read_video                                   # noqa: E402
from PIL import Image, ImageDraw, ImageFont                             # noqa: E402
import scenarios as S                                                   # noqa: E402

VID = "/shared/ssd_14T_project/adenelmi/reid_scen/videos"
OUT = os.path.join(HERE, "eyepack_tfree")
CELLS = [("SS2", 12, "cafe"), ("SS2", 13, "pelouse"), ("SS2", 14, "boulangerie"),
         ("S2", 2, "chat tabby"), ("S2", 4, "cheval blanc"), ("S2", 11, "theiere cuivre")]
CLEAN_REF = {"SS2": ("SS1", 240), "S2": ("S3", 300)}
FONT = ("/shared/ssd_14T/home/adenelmi/miniconda3/envs/longlive/lib/python3.10/"
        "site-packages/matplotlib/mpl-data/fonts/ttf/DejaVuSans.ttf")


def lat2pix(f):
    return 0 if f == 0 else 4 * f - 3


def ff(args):
    subprocess.run(["ffmpeg", "-y", "-hide_banner", "-loglevel", "error"] + args, check=True)


def size_mb(p):
    return os.path.getsize(p) / 1e6


def path_of(scen, ti, arm):
    if arm == "A*":
        rs, _ = CLEAN_REF[scen]
        return f"{VID}/{rs}_t{ti}_s0_memzero.mp4"
    return f"{VID}/{scen}_t{ti}_s0_{arm}.mp4"


def hstack(scen, ti, arms, out, w_total=960, crf=30, trim=None):
    n = len(arms)
    w = w_total // n
    ins, filt = [], []
    for i, a in enumerate(arms):
        pre = ["-ss", str(trim[0]), "-t", str(trim[1])] if trim else []
        ins += pre + ["-i", path_of(scen, ti, a)]
        lab = "A* propre" if a == "A*" else a
        filt.append(f"[{i}:v]scale={w}:-2,drawtext=fontfile={FONT}:text='{lab}':x=6:y=6:"
                    f"fontsize=15:fontcolor=white:box=1:boxcolor=black@0.6:boxborderw=4[v{i}]")
    filt.append("".join(f"[v{i}]" for i in range(n)) + f"hstack=inputs={n}[o]")
    ff(ins + ["-filter_complex", ";".join(filt), "-map", "[o]", "-c:v", "libx264",
              "-crf", str(crf), "-preset", "slow", "-pix_fmt", "yuv420p", "-an", out])
    print(f"[eye] {os.path.basename(out)}  {size_mb(out):.2f} MB", flush=True)
    return size_mb(out)


def board(path, arms):
    cols = ["A* PROPRE", "B interlude (vu)"] + arms
    cache = {}

    def frame(p, w):
        if (p, w) not in cache:
            v, _, _ = read_video(p, pts_unit="sec", output_format="THWC")
            i = min(lat2pix((w[0] + w[1]) // 2), v.shape[0] - 1)
            cache[(p, w)] = Image.fromarray(v[i].numpy()).resize((260, 150))
        return cache[(p, w)]

    tw, th, lab, head, pad = 260, 150, 150, 40, 4
    img = Image.new("RGB", (lab + len(cols) * (tw + pad) + pad,
                            head + len(CELLS) * (th + pad) + pad), (18, 18, 20))
    d = ImageDraw.Draw(img)
    f1 = ImageFont.truetype(FONT, 15)
    f2 = ImageFont.truetype(FONT, 13)
    for j, c in enumerate(cols):
        d.text((lab + j * (tw + pad) + 4, 14), c, font=f1,
               fill=(255, 210, 120) if j >= 2 else (170, 200, 255))
    for i, (scen, ti, name) in enumerate(CELLS):
        y = head + i * (th + pad)
        rs, rf = CLEAN_REF[scen]
        w_clean = S.score_segments(rs, rf)[0]
        rev = S.revisit_segments(scen)[0]
        w_rev = S.score_segments(scen, S.SCENARIOS[scen]["frames"])[rev]
        w_b = S.score_segments(scen, S.SCENARIOS[scen]["frames"])[rev - 1]
        d.text((4, y + 6), f"{scen} t{ti}", font=f1, fill=(240, 240, 240))
        d.text((4, y + 26), name, font=f2, fill=(150, 150, 150))
        d.text((4, y + 44), f"src t{S.wrong_anchor_triple(ti)}", font=f2, fill=(120, 120, 120))
        row = [(0, frame(path_of(scen, ti, "A*"), w_clean)),
               (1, frame(path_of(scen, ti, "memzero"), w_b))]
        # un bras dont le rollout n'a pas pu tourner (machine saturee) laisse sa case VIDE plutot
        # que de decaler la planche : la colonne reste lisible et le trou est visible.
        for j, a in enumerate(arms):
            p = path_of(scen, ti, a)
            if os.path.exists(p):
                row.append((2 + j, frame(p, w_rev)))
        for j, im in row:
            img.paste(im, (lab + j * (tw + pad), y))
    img.save(path, quality=72, optimize=True)
    print(f"[eye] {os.path.basename(path)}  {size_mb(path):.2f} MB", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ours", default="sdedit_ours06")
    ap.add_argument("--wrong", default="sdedit_wrong06")
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    HST = ["memzero", "sinkpurge", "memdec20", a.ours]
    BRD = ["memzero", "sinkpurge", "memdec20", "wronganchor", a.ours, a.wrong]
    big = []
    for scen, ti, _ in CELLS:
        rev = S.revisit_segments(scen)[0]
        lo, hi = S.score_segments(scen, S.SCENARIOS[scen]["frames"])[rev]
        trim = (lat2pix(lo) / 16.0, (lat2pix(hi) - lat2pix(lo)) / 16.0)
        have = [x for x in HST if os.path.exists(path_of(scen, ti, x))]
        if len(have) >= 2:
            big.append(hstack(scen, ti, have, os.path.join(OUT, f"{scen}_t{ti}_s0_{len(have)}bras.mp4")))
        vhave = [x for x in ["A*", a.ours, a.wrong] if os.path.exists(path_of(scen, ti, x))]
        if len(vhave) >= 2:
            hstack(scen, ti, vhave, os.path.join(OUT, f"{scen}_t{ti}_s0_verdict.mp4"), trim=trim)
    board(os.path.join(OUT, "board_tfree.jpg"), BRD)
    assert max(big) < 2.0, f"un hstack depasse 2 Mo ({max(big):.2f})"
    print("EYEPACK TFREE DONE")
