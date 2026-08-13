#!/usr/bin/env python
# ============================================================================================
# EYEPACK "FANTOME" (2026-08-03) -- verdict oeil d'Aden sur memdec21u : le sujet revient bien
# place, MAIS il reste un fantome flou a l'ANCIENNE position du sujet d'interlude, parce que la
# region effacee par le masque union recevait le contenu passe-bas de l'ancre. memdec21un y met
# du BRUIT a la place, pour que le prompt + le contexte la repeignent librement.
#
#   * 3 mp4 hstack [memdec21u | memdec21un | banque Echo], etiquetes, resolution native, < 30 MB
#   * 1 planche PNG ZOOMEE sur la region fantome : une ligne par cellule, colonnes
#     [avant le retour (ou est B) | memzero | memdec21u AVANT | memdec21un APRES | banque].
#     La boite du zoom est EXACTEMENT celle du scoreur (score_comp.ghost_box sur memzero), donc
#     ce que l'oeil regarde est ce que le chiffre ghost_hf mesure.
# ============================================================================================
import os
import subprocess
import sys

REPO = "/shared/ssd_14T/home/adenelmi/Anchor-Forcing"
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [REPO, os.path.join(REPO, "reference/runs/reid_probe_l1"), HERE]

from torchvision.io import read_video                                 # noqa: E402
import scenarios as S                                                 # noqa: E402
from score_comp import ghost_box, pix                                 # noqa: E402

SCEN = "SS2"
VID = "/shared/ssd_14T_project/adenelmi/reid_scen/videos"
OUT = os.path.join(HERE, "eyepack_fill")
CELLS = [("t12", "s0"), ("t13", "s0"), ("t14", "s0")]
ARMS = [("memdec21u", "union (fantome)"), ("memdec21up", "bruit PRIORITAIRE"),
        ("memdec21uf", "bruit + plancher 0.4")]
FONT = ("/shared/ssd_14T/home/adenelmi/miniconda3/envs/longlive/lib/python3.10/"
        "site-packages/matplotlib/mpl-data/fonts/ttf/DejaVuSans.ttf")


def hstack_mp4(cell, out):
    ti, si = cell
    ins, filt = [], []
    for i, (a, label) in enumerate(ARMS):
        ins += ["-i", f"{VID}/{SCEN}_{ti}_{si}_{a}.mp4"]
        filt.append(f"[{i}:v]drawtext=fontfile={FONT}:text='{label}':x=12:y=12:fontsize=34:"
                    f"fontcolor=white:box=1:boxcolor=black@0.6:boxborderw=8[v{i}]")
    filt.append("".join(f"[v{i}]" for i in range(len(ARMS))) + f"hstack=inputs={len(ARMS)}[out]")
    subprocess.run(["ffmpeg", "-y", "-hide_banner", "-loglevel", "error"] + ins
                   + ["-filter_complex", ";".join(filt), "-map", "[out]",
                      "-c:v", "libx264", "-crf", "23", "-pix_fmt", "yuv420p", out], check=True)
    mb = os.path.getsize(out) / 1e6
    print(f"[eyepack] {os.path.basename(out)}  {mb:.1f} MB", flush=True)
    assert mb < 30, f"{out} fait {mb:.1f} MB (budget 30 MB)"


def board(path, zoom=3.0):
    """Zoom sur la boite fantome, avant/apres."""
    from PIL import Image, ImageDraw, ImageFont
    rev = S.revisit_segments(SCEN)[0]
    sw = S.SCENARIOS[SCEN]["schedule"][rev][0]                 # 180
    f_before = pix(sw - 4)                                     # B encore en place
    f_after = pix(sw + 8)                                      # fenetre chaude du retour
    cols = ["AVANT le retour\n(la ou est B)", "sans memoire\n(le monde continue)",
            "memdec21u\nAVANT (fantome)", "memdec21up\nbruit PRIORITAIRE",
            "memdec21uf\nplancher 0.4 (FINAL?)"]

    def load(arm):
        v, _, _ = read_video(f"{VID}/{SCEN}_{ti}_{si}_{arm}.mp4", pts_unit="sec",
                             output_format="THWC")
        return v

    tiles, boxes = [], []
    for ti, si in CELLS:
        vz = load("memzero")
        box = ghost_box(vz[min(f_before, vz.shape[0] - 1)].permute(2, 0, 1).float() / 255.0)
        t, l, h, w = box
        boxes.append(box)

        def crop(v, f):
            return Image.fromarray(v[min(f, v.shape[0] - 1)][t:t + h, l:l + w].numpy())

        row = [crop(vz, f_before), crop(vz, f_after)]
        row += [crop(load(a), f_after) for a, _ in ARMS]
        tiles.append(row)
        print(f"[eyepack] zoom {ti}: boite top{t} left{l} {h}x{w}", flush=True)

    tw, th = int(tiles[0][0].size[0] * zoom / 3), int(tiles[0][0].size[1] * zoom / 3)
    head, lab, pad = 80, 170, 6
    img = Image.new("RGB", (lab + len(cols) * (tw + pad) + pad,
                            head + len(CELLS) * (th + pad) + pad), (18, 18, 20))
    d = ImageDraw.Draw(img)
    f1, f2 = ImageFont.truetype(FONT, 19), ImageFont.truetype(FONT, 16)
    for j, c in enumerate(cols):
        x = lab + j * (tw + pad)
        for k, line in enumerate(c.split("\n")):
            d.text((x + 6, 12 + k * 24), line, font=f1,
                   fill=(255, 210, 120) if j in (2, 3) else (170, 200, 255))
    for i, (ti, si) in enumerate(CELLS):
        y = head + i * (th + pad)
        d.text((6, y + 8), f"{SCEN}_{ti}", font=f1, fill=(240, 240, 240))
        d.text((6, y + 34), f"zoom {boxes[i][2]}x{boxes[i][3]}", font=f2, fill=(150, 150, 150))
        for j, im in enumerate(tiles[i]):
            img.paste(im.resize((tw, th)), (lab + j * (tw + pad), y))
    img.save(path)
    print(f"[eyepack] {path}  ({os.path.getsize(path)/1e6:.1f} MB)", flush=True)


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    for cell in CELLS:
        hstack_mp4(cell, os.path.join(OUT, f"{SCEN}_{cell[0]}_{cell[1]}_union-vs-prioritaire-vs-plancher.mp4"))
    board(os.path.join(OUT, f"{SCEN}_planche_zoom_fantome.png"))
    print("EYEPACK FILL DONE")
