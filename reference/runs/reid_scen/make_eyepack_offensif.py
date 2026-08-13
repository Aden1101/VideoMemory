#!/usr/bin/env python
# ============================================================================================
# OEIL DU LOT OFFENSIF -- format LEGER impose par la memoire projet : micro-clips < 0.3 Mo et
# UNE planche JPEG. Aucun mp4 lourd, aucun base64.
#
#   * planche : une ligne par cellule, une colonne par bras, la frame du RETOUR (+8) ; la
#     premiere colonne est le A* PROPRE (ce qu'il FAUT retrouver), la derniere memzero (ce qu'on
#     obtient sans rien faire). C'est la seule facon de juger un contraste apparie a l'oeil.
#   * micro-clips : 2 s autour du raccord, crf 34, 480 px de large -> ~0.2 Mo piece.
#
#   python make_eyepack_offensif.py --lever 3
# ============================================================================================
import argparse
import os
import subprocess
import sys

REPO = "/shared/ssd_14T/home/adenelmi/Anchor-Forcing"
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [REPO, os.path.join(REPO, "reference/runs/reid_probe_l1"), HERE]

VID = "/shared/ssd_14T_project/adenelmi/reid_scen/videos"
OUT = os.path.join(HERE, "eyepack_offensif")
from score_offensif import LEVERS, CLEAN_REF                            # noqa: E402


def pix(f):
    return 0 if f == 0 else 4 * f - 3


def main():
    import torch
    from PIL import Image, ImageDraw
    from torchvision.io import read_video
    import scenarios as S

    ap = argparse.ArgumentParser()
    ap.add_argument("--lever", type=int, required=True, choices=[1, 2, 3])
    ap.add_argument("--clips", type=int, default=3, help="nombre de cellules avec micro-clips")
    args = ap.parse_args()
    spec = LEVERS[args.lever]
    os.makedirs(OUT, exist_ok=True)

    rows, labels = [], None
    for scen, ti, si in spec["cells"]:
        ref_scen, ref_frames = CLEAN_REF[scen]
        rev = S.revisit_segments(scen)[0]
        sw = S.SCENARIOS[scen]["schedule"][rev][0]
        cols, names = [], []
        p_ref = f"{VID}/{ref_scen}_t{ti}_s{si}_memzero.mp4"
        if os.path.exists(p_ref):
            v, _, _ = read_video(p_ref, pts_unit="sec", output_format="THWC")
            cols.append(v[min(pix(40), v.shape[0] - 1)])
            names.append("A* PROPRE")
        for a in spec["arms"]:
            p = f"{VID}/{scen}_t{ti}_s{si}_{a}.mp4"
            if not os.path.exists(p):
                continue
            v, _, _ = read_video(p, pts_unit="sec", output_format="THWC")
            cols.append(v[min(pix(sw + 8), v.shape[0] - 1)])
            names.append(a)
        if not cols:
            continue
        rows.append((f"{scen} t{ti}", cols))
        labels = names if labels is None or len(names) > len(labels) else labels

    if not rows:
        print("[eye] aucune video, rien a faire")
        return
    W, H = 300, 173
    pad, top, left = 4, 26, 86
    ncol = max(len(c) for _, c in rows)
    board = Image.new("RGB", (left + ncol * (W + pad), top + len(rows) * (H + pad)), (18, 18, 18))
    d = ImageDraw.Draw(board)
    for j, nm in enumerate(labels[:ncol]):
        d.text((left + j * (W + pad) + 4, 8), nm[:34], fill=(235, 235, 235))
    for i, (tag, cols) in enumerate(rows):
        d.text((4, top + i * (H + pad) + H // 2), tag, fill=(235, 235, 235))
        for j, fr in enumerate(cols):
            im = Image.fromarray(fr.numpy()).resize((W, H))
            board.paste(im, (left + j * (W + pad), top + i * (H + pad)))
    out = os.path.join(OUT, f"board_lever{args.lever}.jpg")
    board.save(out, quality=72, optimize=True)
    print(f"[eye] {out}  {os.path.getsize(out) / 1e6:.2f} Mo  "
          f"({len(rows)} cellules x {ncol} colonnes)")

    # micro-clips du raccord, sur les premieres cellules
    for scen, ti, si in spec["cells"][:args.clips]:
        rev = S.revisit_segments(scen)[0]
        sw = S.SCENARIOS[scen]["schedule"][rev][0]
        t0 = max(0, (pix(sw) - 16)) / 16.0
        for a in spec["arms"]:
            p = f"{VID}/{scen}_t{ti}_s{si}_{a}.mp4"
            if not os.path.exists(p):
                continue
            o = os.path.join(OUT, f"{scen}_t{ti}_s{si}_{a}_joint.mp4")
            subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-ss", f"{t0:.2f}", "-t", "2.5",
                            "-i", p, "-vf", "scale=480:-2", "-c:v", "libx264", "-crf", "34",
                            "-pix_fmt", "yuv420p", o], check=True)
            mb = os.path.getsize(o) / 1e6
            assert mb < 0.3, f"{o} fait {mb:.2f} Mo (budget 0.3 Mo)"
    tot = sum(os.path.getsize(os.path.join(OUT, f)) for f in os.listdir(OUT)) / 1e6
    print(f"[eye] eyepack_offensif/ total {tot:.2f} Mo")


if __name__ == "__main__":
    main()
