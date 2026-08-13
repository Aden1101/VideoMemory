#!/usr/bin/env python
# ============================================================================================
# MEMORY 2.0 -- PHASE B: the eye check. Same val ANCHORS, same frozen A2 states, decoded by
# TWO decoders side by side:  col 0 = the true evicted frame, col 1 = A2 (v1, per token),
# col 2 = the phase-B decoder (v2 + spatial mixer).  Plus, per sample, the two numbers the
# report argues about: norm_mse and cos_DINO(decoded, its own target).
#
# Usage: CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 \
#   python reference/runs/mem2/viz_phaseB.py --dec_b writer/mem2_writer_B1_final.pt --n 6
# ============================================================================================
import argparse
import os
import sys

REPO = "/shared/ssd_14T/home/adenelmi/Anchor-Forcing"
PROBE_DIR = os.path.join(REPO, "reference/runs/reid_probe_l1")
FRAME_DEC = os.path.join(REPO, "reference/runs/frame_dec")
HERE = os.path.dirname(os.path.abspath(__file__))
for p in (REPO, PROBE_DIR, FRAME_DEC, HERE):
    sys.path.insert(0, p)
os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")

import torch  # noqa: E402
import torch.nn.functional as Fn  # noqa: E402

import mem2_data as D  # noqa: E402
import train_writer as T  # noqa: E402
from wan.modules.video_memory import AnchorFrameDecoder  # noqa: E402
from build_bank_dino import latent_to_pixels, pixels_to_dino_input  # noqa: E402


def load_dec(path, device, mixer):
    dec = AnchorFrameDecoder(cond_dim=4096, cond_hidden=512, detach_states=False,
                             mixer=mixer).to(device)
    ck = torch.load(path, map_location="cpu", weights_only=False)
    sd = ck["decoder"] if "decoder" in ck else ck
    missing, unexpected = dec.load_state_dict(sd, strict=False)
    assert not unexpected and all(k.startswith("mixer.") for k in missing), (missing, unexpected)
    dec.eval()
    return dec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dec_a", default=T.WRITER_A2, help="reference decoder (A2, no mixer)")
    ap.add_argument("--dec_b", required=True, help="phase-B decoder (mixer)")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--n_png", type=int, default=6, help="rows kept in the PNG grid")
    ap.add_argument("--out", default=os.path.join(REPO, "reference/runs/mem2/eyepack_phaseB"))
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    argv = ["--phase", "B", "--load", "direct", "--device", a.device,
            "--writer_ckpt", T.WRITER_A2, "--freeze_writer", "--mixer",
            "--dec_ckpt", T.WRITER_A2, "--dino_log",
            "--ages", "5", "10", "20", "35", "--min_age", "5", "--anchor_frac", "1.0"]
    tr = T.WriterTrainer(T.parse_args(argv))
    dev = tr.device
    decA = load_dec(a.dec_a, dev, mixer=False)
    decB = load_dec(a.dec_b, dev, mixer=True)

    tgts, pa, pb, rowsel, names = [], [], [], [], []
    for meta in tr.val_vids[:a.n]:
        pos = tr.positions_for(meta, k=1, anchor_frac=1.0)
        sel = tr.dino_positions(meta, pos, k=1)
        if not sel:
            continue
        with torch.no_grad():
            _, target, st, cond = tr.forward_video(meta, pos, n_bptt=0, force_cond=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                ya, yb = decA(st, cond=cond), decB(st, cond=cond)
        i = sel[0]
        tgts.append(target[i].float())
        pa.append(ya[i].float())
        pb.append(yb[i].float())
        rowsel.append(tr.dino_key2row[(meta["video_id"], int(pos[i]["frame"]))])
        names.append(f"{meta['video_id']}/{meta['mode']}")
        del st
    tgt = torch.stack(tgts)
    A, B = torch.stack(pa), torch.stack(pb)

    print("\n[viz] per-sample: norm_mse (lower better) | cos_DINO(decoded, its target)",
          flush=True)
    from train_decoder import norm_mse
    rows = []
    with torch.no_grad():
        for i, nm in enumerate(names):
            ma = float(norm_mse(A[i:i + 1], tgt[i:i + 1]))
            mb = float(norm_mse(B[i:i + 1], tgt[i:i + 1]))
            za = tr.dino_z(A[i:i + 1])
            zb = tr.dino_z(B[i:i + 1])
            e = tr.dino_emb[rowsel[i]][None]
            ca, cb = float((za * e).sum()), float((zb * e).sum())
            rows.append((nm, ma, mb, ca, cb))
            print(f"[viz]   {nm:>20}: mse A2 {ma:.3f} -> B {mb:.3f} | "
                  f"cos_DINO A2 {ca:.3f} -> B {cb:.3f}", flush=True)
    n = len(rows)
    w_mse = sum(1 for r in rows if r[2] < r[1])
    w_cos = sum(1 for r in rows if r[4] > r[3])
    print(f"[viz] MEAN over {n} PAIRED val anchors (same states, same positions): "
          f"mse {sum(r[1] for r in rows) / n:.4f} -> {sum(r[2] for r in rows) / n:.4f} "
          f"({w_mse}/{n} better) | cos_DINO {sum(r[3] for r in rows) / n:.4f} -> "
          f"{sum(r[4] for r in rows) / n:.4f} ({w_cos}/{n} better)", flush=True)

    from torchvision.utils import save_image
    npng = min(a.n_png, len(names))
    with torch.no_grad():
        px = []
        for t in (tgt[:npng], A[:npng], B[:npng]):
            p = latent_to_pixels(tr.vae, t.to(torch.bfloat16))
            px.append(p.clamp(0, 1).cpu())
    tiles = []
    for i in range(npng):
        tiles += [px[0][i], px[1][i], px[2][i]]
    out = os.path.join(a.out, "target_vs_A2_vs_B.png")
    save_image(tiles, out, nrow=3)
    print(f"\n[viz] -> {out}  (col 0 = target, col 1 = A2 v1, col 2 = phase-B v2+mixer)",
          flush=True)


if __name__ == "__main__":
    main()
