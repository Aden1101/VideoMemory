#!/usr/bin/env python
# ============================================================================================
# EYEPACK "ANCRE COMPOSITE" (2026-08-02) -- pour l'oeil, pas pour une metrique.
#
# LE CONSTAT D'ADEN. Sur le tier meme-decor, notre splice pleine-frame re-rend TOUTE la scene au
# retour (l'auvent de SS2_t14 passe bleu -> rose) alors que la banque ne ramene que le sujet dans
# une scene intacte. Le composite garde le SUJET de l'ancre et le FOND du contexte courant.
#
#   1) mp4 hstack a 3 panneaux [notre memoire | notre memoire COMPOSITE | banque Echo] sur les
#      cellules du verdict (t13 pelouse, t14 boulangerie), resolution native, < 30 MB ;
#   2) planche PNG "MECANISME" : par cellule, [fond = derniere vraie frame | ancre decodee |
#      masque sujet | frame composite construite | retour rendu]. Les 4 premieres colonnes sont
#      les LATENTS reellement splices (dumps MEM_COMP_DUMP) passes au VAE : on voit le composite
#      exactement tel que le modele l'a lu, et on peut chercher une couture au bord du masque.
#
#   python make_eyepack_comp.py [--arm memdec21c] [--cells 12,13,14]
# ============================================================================================
import argparse
import os
import subprocess
import sys

REPO = "/shared/ssd_14T/home/adenelmi/Anchor-Forcing"
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [REPO, os.path.join(REPO, "reference/runs/reid_probe_l1"), HERE]
os.chdir(REPO)                       # le VAE resout wan_models/... en relatif

VID = "/shared/ssd_14T_project/adenelmi/reid_scen/videos"
DUMPS = "/shared/ssd_14T_project/adenelmi/reid_scen/comp_dumps"
OUT = os.path.join(HERE, "eyepack_comp")
SCEN = "SS2"
FONT = ("/shared/ssd_14T/home/adenelmi/miniconda3/envs/longlive/lib/python3.10/"
        "site-packages/matplotlib/mpl-data/fonts/ttf/DejaVuSans.ttf")
MP4_ARMS = [("memdec20", "notre memoire (splice pleine frame)"),
            ("memdec21c", "COMPOSITE ellipse 0.60"),
            ("memdec21c75", "COMPOSITE ellipse 0.75"),
            ("framebank", "banque Echo")]


def lat2pix(f):
    return 0 if f == 0 else 4 * f - 3


def hstack_mp4(ti, si, out, arms):
    ins, filt = [], []
    for i, (a, label) in enumerate(arms):
        p = f"{VID}/{SCEN}_t{ti}_s{si}_{a}.mp4"
        assert os.path.exists(p), f"video manquante {p}"
        ins += ["-i", p]
        filt.append(f"[{i}:v]drawtext=fontfile={FONT}:text='{label}':x=12:y=12:fontsize=32:"
                    f"fontcolor=white:box=1:boxcolor=black@0.6:boxborderw=8[v{i}]")
    filt.append("".join(f"[v{i}]" for i in range(len(arms))) + f"hstack=inputs={len(arms)}[out]")
    subprocess.run(["ffmpeg", "-y", "-hide_banner", "-loglevel", "error"] + ins
                   + ["-filter_complex", ";".join(filt), "-map", "[out]",
                      "-c:v", "libx264", "-crf", "23", "-pix_fmt", "yuv420p", out], check=True)
    mb = os.path.getsize(out) / 1e6
    print(f"[eyepack] {os.path.basename(out)}  {mb:.1f} MB", flush=True)
    assert mb < 30, f"{out} fait {mb:.1f} MB (budget 30 MB)"


def board(path, cells, arm, seed):
    """Le MECANISME, colonne par colonne, sur les latents reellement splices."""
    import torch
    from PIL import Image, ImageDraw, ImageFont
    from torchvision.io import read_video
    from utils.wan_wrapper import WanVAEWrapper
    import scenarios as S

    device = torch.device("cuda")
    torch.set_grad_enabled(False)
    vae = WanVAEWrapper().to(device=device, dtype=torch.bfloat16).eval()

    def dec(lat):
        """(1,1,16,60,104) latent -> PIL image."""
        px = vae.decode_to_pixel(lat.to(device=device, dtype=torch.bfloat16), use_cache=False)
        vae.model.clear_cache()
        im = (px[0, 0].float() * 0.5 + 0.5).clamp(0, 1).permute(1, 2, 0).cpu().numpy()
        return Image.fromarray((im * 255).astype("uint8"))

    rev = S.revisit_segments(SCEN)[0]
    w_rev = S.score_segments(SCEN, S.SCENARIOS[SCEN]["frames"])[rev]
    sw = S.SCENARIOS[SCEN]["schedule"][rev][0]

    def frame(path_mp4, latent_f):
        v, _, _ = read_video(path_mp4, pts_unit="sec", output_format="THWC")
        return Image.fromarray(v[min(lat2pix(latent_f), v.shape[0] - 1)].numpy())

    cols = [f"FOND\nderniere vraie frame ({sw - 1})", "ANCRE\ndecodee de la memoire",
            "MASQUE sujet\n(blanc = ancre)", "COMPOSITE\nce qui est splice",
            f"RETOUR RENDU\n{arm} (+8)", f"RETOUR RENDU\n{arm} (milieu)"]
    tiles = []
    for ti in cells:
        d = torch.load(f"{DUMPS}/{SCEN}_t{ti}_s{seed}_{arm}.f{sw}.pt",
                       map_location="cpu", weights_only=False)
        m = d["mask"][0, 0, 0]
        mim = Image.fromarray((m.numpy() * 255).astype("uint8")).convert("RGB").resize(
            dec(d["bg"]).size, Image.NEAREST)
        mp4 = f"{VID}/{SCEN}_t{ti}_s{seed}_{arm}.mp4"
        tiles.append([dec(d["bg"]), dec(d["anchor"]), mim, dec(d["composite"]),
                      frame(mp4, sw + 8), frame(mp4, (w_rev[0] + w_rev[1]) // 2)])
        print(f"[eyepack] ligne t{ti} construite (crop={d['crop']}, feather={d['feather']})",
              flush=True)

    tw, th = tiles[0][0].size
    sc = 0.55
    tw, th = int(tw * sc), int(th * sc)
    head, lab, pad = 80, 170, 6
    W = lab + len(cols) * (tw + pad) + pad
    H = head + len(cells) * (th + pad) + pad
    img = Image.new("RGB", (W, H), (18, 18, 20))
    dr = ImageDraw.Draw(img)
    f1, f2 = ImageFont.truetype(FONT, 19), ImageFont.truetype(FONT, 16)
    for j, c in enumerate(cols):
        x = lab + j * (tw + pad)
        for k, line in enumerate(c.split("\n")):
            dr.text((x + 6, 14 + k * 24), line, font=f1,
                    fill=(255, 210, 120) if j >= 4 else (170, 200, 255))
    for i, ti in enumerate(cells):
        y = head + i * (th + pad)
        dr.text((6, y + 8), f"{SCEN}_t{ti}_s{seed}", font=f1, fill=(240, 240, 240))
        words, line, ln = S.TRIPLES[ti][0].split(), "", 0
        for wd in words:
            if len(line) + len(wd) > 22:
                dr.text((6, y + 36 + ln * 19), line, font=f2, fill=(150, 150, 150))
                line, ln = wd, ln + 1
            else:
                line = (line + " " + wd).strip()
        dr.text((6, y + 36 + ln * 19), line, font=f2, fill=(150, 150, 150))
        for j, im in enumerate(tiles[i]):
            img.paste(im.resize((tw, th)), (lab + j * (tw + pad), y))
    img.save(path)
    print(f"[eyepack] {path}  ({os.path.getsize(path)/1e6:.1f} MB)", flush=True)


def board_arms(path, cells, seed, arms):
    """Le RETOUR de A, une colonne par bras: c'est la comparaison que l'oeil doit trancher
    (sujet retrouve ET decor intact ?). Pas de VAE, juste des frames des mp4."""
    from PIL import Image, ImageDraw, ImageFont
    from torchvision.io import read_video
    import scenarios as S

    rev = S.revisit_segments(SCEN)[0]
    w_rev = S.score_segments(SCEN, S.SCENARIOS[SCEN]["frames"])[rev]
    sw = S.SCENARIOS[SCEN]["schedule"][rev][0]

    def frame(p, f):
        v, _, _ = read_video(p, pts_unit="sec", output_format="THWC")
        return Image.fromarray(v[min(lat2pix(f), v.shape[0] - 1)].numpy())

    cols = ["AVANT le retour\n(derniere frame de B)"] + [f"RETOUR\n{lab}" for _, lab in arms]
    tiles = []
    for ti in cells:
        row = [frame(f"{VID}/{SCEN}_t{ti}_s{seed}_memzero.mp4", sw - 1)]
        row += [frame(f"{VID}/{SCEN}_t{ti}_s{seed}_{a}.mp4", (w_rev[0] + w_rev[1]) // 2)
                for a, _ in arms]
        tiles.append(row)
        print(f"[eyepack] ligne bras t{ti} construite", flush=True)

    tw, th = tiles[0][0].size
    sc = 0.55
    tw, th = int(tw * sc), int(th * sc)
    head, lab, pad = 80, 170, 6
    img = Image.new("RGB", (lab + len(cols) * (tw + pad) + pad,
                            head + len(cells) * (th + pad) + pad), (18, 18, 20))
    dr = ImageDraw.Draw(img)
    f1, f2 = ImageFont.truetype(FONT, 19), ImageFont.truetype(FONT, 16)
    for j, c in enumerate(cols):
        x = lab + j * (tw + pad)
        for k, line in enumerate(c.split("\n")):
            dr.text((x + 6, 14 + k * 24), line, font=f1,
                    fill=(150, 150, 150) if j == 0 else (255, 210, 120))
    for i, ti in enumerate(cells):
        y = head + i * (th + pad)
        dr.text((6, y + 8), f"{SCEN}_t{ti}_s{seed}", font=f1, fill=(240, 240, 240))
        words, line, ln = S.TRIPLES[ti][0].split(), "", 0
        for wd in words:
            if len(line) + len(wd) > 22:
                dr.text((6, y + 36 + ln * 19), line, font=f2, fill=(150, 150, 150))
                line, ln = wd, ln + 1
            else:
                line = (line + " " + wd).strip()
        dr.text((6, y + 36 + ln * 19), line, font=f2, fill=(150, 150, 150))
        for j, im in enumerate(tiles[i]):
            img.paste(im.resize((tw, th)), (lab + j * (tw + pad), y))
    img.save(path)
    print(f"[eyepack] {path}  ({os.path.getsize(path)/1e6:.1f} MB)", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="memdec21c")
    ap.add_argument("--cells", default="12,13,14")
    ap.add_argument("--mp4-cells", default="13,14")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    cells = [int(x) for x in a.cells.split(",") if x]
    os.makedirs(OUT, exist_ok=True)
    arms = [(x, lab) for x, lab in MP4_ARMS]
    if a.arm != "memdec21c":
        arms = [(x if x != "memdec21c" else a.arm, lab) for x, lab in MP4_ARMS]
    for ti in [int(x) for x in a.mp4_cells.split(",") if x]:
        hstack_mp4(ti, a.seed, os.path.join(
            OUT, f"{SCEN}_t{ti}_s{a.seed}_notremem-composite-banque.mp4"), arms)
    board(os.path.join(OUT, f"{SCEN}_planche_MECANISME_{a.arm}.png"), cells, a.arm, a.seed)
    board_arms(os.path.join(OUT, f"{SCEN}_planche_RETOUR_par_bras.png"), cells, a.seed,
               [("memzero", "sans memoire"), ("framebank", "banque Echo"),
                ("memdec20", "notre mem (pleine frame)"),
                ("memdec21c", "COMPOSITE 0.60"), ("memdec21c75", "COMPOSITE 0.75")])
    print("EYEPACK COMP DONE")
