#!/usr/bin/env python
# ============================================================================================
# LEVIER 4 A L'OEIL : l'ancre decodee du SNAPSHOT de fin de segment A repare-t-elle le colis ?
# Colonnes : A* PROPRE | X ouvreur | B interlude | ANCRE COURANTE | ANCRE SNAPSHOT | retour rendu.
# Les deux ancres cote a cote, sur les 2 cellules du pilote : si le snapshot marchait, la 5e
# colonne devrait ressembler a la 1re. Le chiffre dit non (0/2) ; la planche le montre.
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
CELLS = [(2, 0), (3, 0)]
COLS = ["A* PROPRE\n(le sujet a rappeler)", "X ouvreur", "B interlude",
        "ANCRE COURANTE\n(etat au retour)", "ANCRE SNAPSHOT\n(etat fin de segment A)",
        "RETOUR memdec_snap"]


def pix(f):
    return 0 if f == 0 else 4 * f - 3


def main():
    from PIL import Image, ImageDraw
    torch.set_grad_enabled(False)
    os.makedirs(OUT, exist_ok=True)
    segs = S.score_segments("S2", 240)
    rows = []
    for ti, si in CELLS:
        pA, pS = f"{VID}/S3_t{ti}_s{si}_memzero.mp4", f"{VID}/S2_t{ti}_s{si}_memzero.mp4"
        pR = f"{VID}/S2_t{ti}_s{si}_memdec_snap.mp4"
        cur = os.path.join(CACHE, f"S2_t{ti}_s{si}_anchor.pt")
        snp = os.path.join(CACHE, f"S2_t{ti}_s{si}_snapanchor.pt")
        if not all(map(os.path.exists, (pA, pS, pR, cur, snp))):
            print(f"  t{ti}: incomplet, saute")
            continue

        def fr(path, latent):
            v, _, _ = read_video(path, pts_unit="sec", output_format="THWC")
            return Image.fromarray(v[min(pix(latent), v.shape[0] - 1)].numpy())

        def im(p):
            a = torch.load(p, map_location="cpu")
            return Image.fromarray((a.clamp(0, 1).permute(1, 2, 0).numpy() * 255).astype("uint8"))

        rows.append((f"t{ti}", [fr(pA, 40), fr(pS, segs[0][0] + 15), fr(pS, segs[2][0] + 15),
                                im(cur), im(snp), fr(pR, 188)]))
        print(f"  t{ti} ok", flush=True)
    if not rows:
        return
    W, H, pad, top, left = 300, 173, 5, 46, 52
    board = Image.new("RGB", (left + 6 * (W + pad), top + len(rows) * (H + pad)), (16, 16, 16))
    d = ImageDraw.Draw(board)
    for j, nm in enumerate(COLS):
        for k, ln in enumerate(nm.split("\n")):
            d.text((left + j * (W + pad) + 4, 6 + 15 * k), ln[:40], fill=(240, 240, 240))
    for i, (tag, cols) in enumerate(rows):
        d.text((6, top + i * (H + pad) + H // 2), tag, fill=(240, 240, 240))
        for j, x in enumerate(cols):
            board.paste(x.resize((W, H)), (left + j * (W + pad), top + i * (H + pad)))
    out = os.path.join(OUT, "board_levier4_snapshot.jpg")
    board.save(out, quality=72, optimize=True)
    print(f"[eye] {out}  {os.path.getsize(out) / 1e6:.2f} Mo")


if __name__ == "__main__":
    main()
