#!/usr/bin/env python
# ============================================================================================
# LA PLANCHE QUI MONTRE LE RESULTAT DE LA NUIT, a l'oeil et sans metrique.
#
# Une ligne par cellule, six colonnes :
#   A* PROPRE      le sujet qu'il FAUT retrouver (segment 0 du S3 : A ouvre, rien avant lui)
#   X ouvreur      le segment 0 du S2, epingle dans le sink pendant tout le rollout
#   B interlude    le segment juste avant le retour -- ce que le decodeur vient de voir EVINCER
#   ANCRE          l'image que notre memoire 2.0 decode et splice au retour
#   memdec20       le retour rendu avec NOTRE ancre
#   wronganchor    le retour rendu avec l'ancre d'une AUTRE cellule
# La question posee a l'oeil est simple : l'ANCRE ressemble-t-elle a A*, ou a B / X ?
#
#   python make_eyepack_anchor.py
# ============================================================================================
import os
import sys

REPO = "/shared/ssd_14T/home/adenelmi/Anchor-Forcing"
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [REPO, os.path.join(REPO, "reference/runs/reid_probe_l1"), HERE]

import torch                                                            # noqa: E402
from torchvision.io import read_video                                   # noqa: E402
import scenarios as S                                                   # noqa: E402
from prescreen_offensif import CACHE, VID                               # noqa: E402

OUT = os.path.join(HERE, "eyepack_offensif")
CELLS = [(0, 0), (2, 0), (3, 0), (4, 0), (6, 0), (11, 0)]
COLS = ["A* PROPRE\n(le sujet a retrouver)", "X ouvreur\n(epingle au sink)",
        "B interlude\n(ce qui vient d'etre EVINCE)", "ANCRE DECODEE\n(ce que la memoire rend)",
        "RETOUR memdec20\n(notre ancre)", "RETOUR wronganchor\n(ancre d'une autre cellule)"]


def pix(f):
    return 0 if f == 0 else 4 * f - 3


def main():
    from PIL import Image, ImageDraw
    torch.set_grad_enabled(False)
    os.makedirs(OUT, exist_ok=True)
    segs = S.score_segments("S2", 240)
    rows = []
    for ti, si in CELLS:
        cp = os.path.join(CACHE, f"S2_t{ti}_s{si}_anchor.pt")
        pA, pS = f"{VID}/S3_t{ti}_s{si}_memzero.mp4", f"{VID}/S2_t{ti}_s{si}_memzero.mp4"
        pD, pW = (f"{VID}/S2_t{ti}_s{si}_memdec20.mp4",
                  f"{VID}/S2_t{ti}_s{si}_wronganchor.mp4")
        if not all(map(os.path.exists, (cp, pA, pS, pD, pW))):
            continue

        def fr(path, latent):
            v, _, _ = read_video(path, pts_unit="sec", output_format="THWC")
            return Image.fromarray(v[min(pix(latent), v.shape[0] - 1)].numpy())

        anc = torch.load(cp, map_location="cpu")
        anc_im = Image.fromarray(
            (anc.clamp(0, 1).permute(1, 2, 0).numpy() * 255).astype("uint8"))
        rows.append((f"t{ti}", [fr(pA, 40), fr(pS, segs[0][0] + 15), fr(pS, segs[2][0] + 15),
                                anc_im, fr(pD, 188), fr(pW, 188)]))
        print(f"  t{ti} s{si} ok", flush=True)

    W, H, pad, top, left = 280, 162, 5, 46, 52
    board = Image.new("RGB", (left + 6 * (W + pad), top + len(rows) * (H + pad)), (16, 16, 16))
    d = ImageDraw.Draw(board)
    for j, nm in enumerate(COLS):
        for k, ln in enumerate(nm.split("\n")):
            d.text((left + j * (W + pad) + 4, 6 + 15 * k), ln[:40], fill=(240, 240, 240))
    for i, (tag, cols) in enumerate(rows):
        d.text((6, top + i * (H + pad) + H // 2), tag, fill=(240, 240, 240))
        for j, im in enumerate(cols):
            board.paste(im.resize((W, H)), (left + j * (W + pad), top + i * (H + pad)))
    out = os.path.join(OUT, "board_ancre_est_un_interlude.jpg")
    board.save(out, quality=70, optimize=True)
    print(f"[eye] {out}  {os.path.getsize(out) / 1e6:.2f} Mo  ({len(rows)} cellules x 6)")


if __name__ == "__main__":
    main()
