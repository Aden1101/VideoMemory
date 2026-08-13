#!/usr/bin/env python
# ============================================================================================
# EYEPACK des 2 CONTROLES REVIEWER S2 (2026-08-03) -- pour l'oeil, pas pour une metrique.
#
#   * hstack [memdec20 | noiseanchor | wronganchor] par cellule, resolution native, crf 23 ;
#   * un PNG : lignes = cellules, colonnes = [A* propre | memzero | memdec20 | noiseanchor |
#     wronganchor | A* propre de la cellule SOURCE de la mauvaise ancre].
#     La derniere colonne dit ce que la mauvaise ancre transportait : si wronganchor ressemble a
#     ELLE plutot qu'a A, la contamination est visible a l'oeil.
#
#   python make_eyepack_controls.py t5_s0 t9_s1
# ============================================================================================
import os
import subprocess
import sys

REPO = "/shared/ssd_14T/home/adenelmi/Anchor-Forcing"
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [REPO, os.path.join(REPO, "reference/runs/reid_probe_l1"), HERE]

from torchvision.io import read_video                                   # noqa: E402
import scenarios as S                                                   # noqa: E402

VID = "/shared/ssd_14T_project/adenelmi/reid_scen/videos"
OUT = os.path.join(HERE, "eyepack_textpoor")
ARMS = ["memzero", "memdec20", "wronganchor", "framebank_gt"]
FONT = ("/shared/ssd_14T/home/adenelmi/miniconda3/envs/longlive/lib/python3.10/"
        "site-packages/matplotlib/mpl-data/fonts/ttf/DejaVuSans.ttf")


def lat2pix(f):
    return 0 if f == 0 else 4 * f - 3


def hstack_mp4(cell, out):
    ins, filt = [], []
    for i, a in enumerate(ARMS):
        ins += ["-i", f"{VID}/S2p_{cell}_{a}.mp4"]
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


def board(cells, path):
    from PIL import Image, ImageDraw, ImageFont
    w_rev = S.score_segments("S2p", 240)[3]                              # (192, 239) le retour
    w_clean = S.score_segments("SS1", 240)[0]                            # (20, 78) A* propre
    w_a = S.score_segments("S2p", 240)[1]                                # (72, 117) A vu
    w_b = S.score_segments("S2p", 240)[2]                                # (132, 177) B vu
    cols = ["A* PROPRE\n(SS1 seg0, A ouvre)", "A tel que VU\n(S2p seg1, ce qui est ecrit)",
            "B interlude\n(S2p seg2)", "RETOUR memzero", "RETOUR memdec20",
            "RETOUR wronganchor", "RETOUR framebank_gt", "RETOUR framebank_oracle_gt"]

    def frame(path_mp4, win):
        v, _, _ = read_video(path_mp4, pts_unit="sec", output_format="THWC")
        p = min(lat2pix((win[0] + win[1]) // 2), v.shape[0] - 1)
        return Image.fromarray(v[p].numpy())

    tiles = []
    for cell in cells:
        ti = int(cell.split("_")[0][1:])
        si = int(cell.split("_")[1][1:])
        other = S.wrong_anchor_triple(ti)
        row = [frame(f"{VID}/SS1_{cell}_memzero.mp4", w_clean),
               frame(f"{VID}/S2p_{cell}_memzero.mp4", w_a),
               frame(f"{VID}/S2p_{cell}_memzero.mp4", w_b)]
        row += [frame(f"{VID}/S2p_{cell}_{a}.mp4", w_rev)
                for a in ARMS + ["framebank_oracle_gt"]]
        tiles.append(row)
        print(f"[eyepack] board row {cell} built (source = triple {other})", flush=True)

    tw, th = tiles[0][0].size
    tw, th = int(tw * 0.5), int(th * 0.5)
    head, lab, pad = 76, 155, 6
    W = lab + len(cols) * (tw + pad) + pad
    H = head + len(cells) * (th + pad) + pad
    img = Image.new("RGB", (W, H), (18, 18, 20))
    d = ImageDraw.Draw(img)
    f1 = ImageFont.truetype(FONT, 19)
    f2 = ImageFont.truetype(FONT, 17)
    for j, c in enumerate(cols):
        x = lab + j * (tw + pad)
        for k, line in enumerate(c.split("\n")):
            d.text((x + 6, 10 + k * 24), line, font=f1,
                   fill=(255, 210, 120) if j >= 3 else (170, 200, 255))
    for i, cell in enumerate(cells):
        y = head + i * (th + pad)
        A = S.TRIPLES[int(cell.split("_")[0][1:])][0]
        d.text((6, y + 8), f"S2_{cell}", font=f1, fill=(240, 240, 240))
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
    print(f"[eyepack] {path}  ({os.path.getsize(path) / 1e6:.1f} MB)", flush=True)


if __name__ == "__main__":
    cells = sys.argv[1:] or ["t17_s0", "t18_s0"]
    os.makedirs(OUT, exist_ok=True)
    for c in cells:
        hstack_mp4(c, os.path.join(OUT, f"S2p_{c}_4arms.mp4"))
    board(cells, os.path.join(OUT, "S2p_board_textpoor.png"))
    print("EYEPACK DONE")
