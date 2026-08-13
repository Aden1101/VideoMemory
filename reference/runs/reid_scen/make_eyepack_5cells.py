#!/usr/bin/env python
# ============================================================================================
# EYEPACK SS2 A 5 CELLULES (2026-08-03) -- pour l'oeil, pas pour une metrique.
# Elargissement demande apres le verdict d'Aden (il prefere memdec21uf au fantome de
# memdec21u) : 2 triples meme-decor NEUFS (t15 garage, t16 serre) s'ajoutent aux 3 du
# pilote, pour voir si les defauts DURS du remplissage bruit (buisson halluciné sur t12)
# reapparaissent ailleurs.
#
# SS2 = X -> A -> B -> A sur 240 frames avec les QUATRE segments dans le MEME DECOR : seul le
# sujet change. Le decor n'achete plus le score, donc ce que l'oeil doit juger ici est
# exclusivement : "le sujet qui revient au segment 3, est-ce LE sujet A du segment 1 ?"
#
#   * 3 mp4 hstack a 4 panneaux [LongLive | sans memoire | banque Echo | notre memoire],
#     resolution native (832x480 par panneau), crf 23, < 30 MB chacun ;
#   * 1 planche PNG : une ligne par cellule, colonnes
#     [1re apparition de A (pipeline MAG) | 1re apparition de A (LongLive) | retour par bras].
#     Deux colonnes de reference parce que stockoff tourne sur SON pipeline (LongLive officiel) :
#     sa 1re apparition n'est pas celle des bras MAG, et chaque bras est score contre la sienne.
#     Les 3 bras MAG (memzero / framebank / memdec20) partagent le meme segment 1 : aucun hook
#     ne se declenche avant le revisit (selftest g), donc une seule vignette suffit pour eux.
# ============================================================================================
import os
import subprocess
import sys

REPO = "/shared/ssd_14T/home/adenelmi/Anchor-Forcing"
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [REPO, os.path.join(REPO, "reference/runs/reid_probe_l1"), HERE]

from torchvision.io import read_video                                 # noqa: E402
import scenarios as S                                                 # noqa: E402

SCEN = "SS2"
VID = "/shared/ssd_14T_project/adenelmi/reid_scen/videos"
OUT = os.path.join(HERE, "eyepack_5cells")
CELLS = [("t12", "s0"), ("t13", "s0"), ("t14", "s0"), ("t15", "s0"), ("t16", "s0")]
# ordre + etiquettes demandes
ARMS = [("stockoff", "LongLive"), ("memzero", "sans memoire"),
        ("framebank", "banque Echo"), ("memdec20", "NOTRE pure"),
        ("memdec21un", "NOTRE union+bruit")]
FONT = ("/shared/ssd_14T/home/adenelmi/miniconda3/envs/longlive/lib/python3.10/"
        "site-packages/matplotlib/mpl-data/fonts/ttf/DejaVuSans.ttf")


def lat2pix(f):
    return 0 if f == 0 else 4 * f - 3


def hstack_mp4(cell, out):
    """Les 4 bras cote a cote, etiquetes, resolution native, crf 23."""
    ti, si = cell
    ins, filt = [], []
    for i, (a, label) in enumerate(ARMS):
        ins += ["-i", f"{VID}/{SCEN}_{ti}_{si}_{a}.mp4"]
        filt.append(f"[{i}:v]drawtext=fontfile={FONT}:text='{label}':x=12:y=12:fontsize=34:"
                    f"fontcolor=white:box=1:boxcolor=black@0.6:boxborderw=8[v{i}]")
    filt.append("".join(f"[v{i}]" for i in range(len(ARMS))) + f"hstack=inputs={len(ARMS)}[out]")
    cmd = (["ffmpeg", "-y", "-hide_banner", "-loglevel", "error"] + ins
           + ["-filter_complex", ";".join(filt), "-map", "[out]",
              "-c:v", "libx264", "-crf", "26", "-pix_fmt", "yuv420p", out])
    subprocess.run(cmd, check=True)
    mb = os.path.getsize(out) / 1e6
    print(f"[eyepack] {os.path.basename(out)}  {mb:.1f} MB", flush=True)
    assert mb < 30, f"{out} is {mb:.1f} MB (> 30 MB budget)"


def board(path):
    """Lignes = cellules, colonnes = [A* PROPRE | 1re apparition A (MAG) |
    1re apparition A (LongLive) | retour de A par bras].

    La colonne A* PROPRE = segment 0 du rollout SS1 de la MEME cellule (A ouvre, rien ne le
    precede) : c'est A quand le modele n'a aucune interference. La comparer a la colonne
    suivante EST la trouvaille : la "1re apparition de A" contre laquelle la campagne score
    montre en realite X, le sujet de l'OUVERTURE, epingle dans le sink."""
    from PIL import Image, ImageDraw, ImageFont
    fa_seg = S.first_appearance(SCEN)[S.seg_roles(SCEN)[-1]]          # 1
    rev_seg = S.revisit_segments(SCEN)[0]                             # 3
    w_fa = S.score_segments(SCEN, 240)[fa_seg]                        # (72, 117)
    w_rev = S.score_segments(SCEN, 240)[rev_seg]                      # (192, 239)
    w_clean = S.score_segments("SS1", 240)[0]                         # (20, 78), A ouvre
    cols = ["A* PROPRE\n(SS1 seg0, A ouvre)",
            f"1re apparition de A\n(seg{fa_seg}, bras MAG)",
            f"1re apparition de A\n(seg{fa_seg}, LongLive)"] + [f"retour de A\n{lab}"
                                                                for _, lab in ARMS]

    def frame(path_mp4, win):
        v, _, _ = read_video(path_mp4, pts_unit="sec", output_format="THWC")
        p = min(lat2pix((win[0] + win[1]) // 2), v.shape[0] - 1)
        return Image.fromarray(v[p].numpy())

    tiles = []
    for ti, si in CELLS:
        row = [frame(f"{VID}/SS1_{ti}_{si}_memzero.mp4", w_clean),
               frame(f"{VID}/{SCEN}_{ti}_{si}_memzero.mp4", w_fa),
               frame(f"{VID}/{SCEN}_{ti}_{si}_stockoff.mp4", w_fa)]
        row += [frame(f"{VID}/{SCEN}_{ti}_{si}_{a}.mp4", w_rev) for a, _ in ARMS]
        tiles.append(row)
        print(f"[eyepack] board row {ti}_{si} built", flush=True)

    tw, th = tiles[0][0].size
    sc = 0.5
    tw, th = int(tw * sc), int(th * sc)
    head, lab, pad = 76, 160, 6
    W = lab + len(cols) * (tw + pad) + pad
    H = head + len(CELLS) * (th + pad) + pad
    img = Image.new("RGB", (W, H), (18, 18, 20))
    d = ImageDraw.Draw(img)
    f1 = ImageFont.truetype(FONT, 19)
    f2 = ImageFont.truetype(FONT, 16)
    for j, c in enumerate(cols):
        x = lab + j * (tw + pad)
        for k, line in enumerate(c.split("\n")):
            d.text((x + 6, 12 + k * 24), line, font=f1,
                   fill=(255, 210, 120) if j >= 3 else
                   ((130, 255, 150) if j == 0 else (170, 200, 255)))
    for i, (ti, si) in enumerate(CELLS):
        y = head + i * (th + pad)
        A = S.TRIPLES[int(ti[1:])][0]
        d.text((6, y + 8), f"{SCEN}_{ti}_{si}", font=f1, fill=(240, 240, 240))
        words, line, ln = A.split(), "", 0
        for wd in words:
            if len(line) + len(wd) > 20:
                d.text((6, y + 36 + ln * 19), line, font=f2, fill=(150, 150, 150))
                line, ln = wd, ln + 1
            else:
                line = (line + " " + wd).strip()
        d.text((6, y + 36 + ln * 19), line, font=f2, fill=(150, 150, 150))
        for j, im in enumerate(tiles[i]):
            img.paste(im.resize((tw, th)), (lab + j * (tw + pad), y))
    img.save(path)
    print(f"[eyepack] {path}  ({os.path.getsize(path)/1e6:.1f} MB)", flush=True)


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    for cell in CELLS:
        hstack_mp4(cell, os.path.join(
            OUT, f"{SCEN}_{cell[0]}_{cell[1]}_5bras.mp4"))
    board(os.path.join(OUT, f"{SCEN}_planche_5cellules_retours.png"))
    print("EYEPACK DONE")
