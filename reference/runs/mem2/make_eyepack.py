#!/usr/bin/env python
# ============================================================================================
# EYE-PACK for Aden (memory 2.0, J3) -- pure visualization, no training, no metric.
# Everything decodes through the REAL VAE with train_decoder.py's idioms (WanVAEWrapper,
# decode_to_pixel, [-1,1] -> [0,1]) at native 480x832.
#
#   1) h_oracle_grid.png : 6 VAL videos (mix single/switch): true target frame | h-oracle
#      reconstruction (the trained readout of probe_h_oracle.py = the UPPER BOUND of 2.0).
#   2) toy_grid.png      : 4 of the 8 toy videos, targets >= 10 writes old:
#      true target | anchor decoded from the WARM-START state (l1 writer + v1.2 decoder)
#      | anchor decoded after the toy run (toy writer + toy decoder).
#   3) twin_pair_{1,2}.mp4 : 2 twin pairs (same prompt, seed +100000) hstacked, crf 23.
#
# Usage (GPU pre-check in the SAME command; NEVER GPU 2 while phase A runs, never idx6):
#   CUDA_DEVICE_ORDER=PCI_BUS_ID nvidia-smi --query-gpu=index,memory.free --format=csv && \
#   CUDA_VISIBLE_DEVICES=1 python reference/runs/mem2/make_eyepack.py
#   ... --only twins        # subset: grid_h | grid_toy | twins
# ============================================================================================
import argparse
import os
import random
import subprocess
import sys

REPO = "/shared/ssd_14T/home/adenelmi/Anchor-Forcing"
PROBE_DIR = os.path.join(REPO, "reference/runs/reid_probe_l1")
FRAME_DEC = os.path.join(REPO, "reference/runs/frame_dec")
HERE = os.path.dirname(os.path.abspath(__file__))
for p in (REPO, PROBE_DIR, FRAME_DEC, HERE):
    sys.path.insert(0, p)
os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")

import torch  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402

import mem2_data as D  # noqa: E402
import train_writer as TW  # noqa: E402
from probe_h_oracle import CACHE_DIR, HOracleDecoder  # noqa: E402
from train_decoder import build_vae  # noqa: E402

OUT = os.path.join(D.RUNS_DIR, "eyepack")
LABEL_H = 34


# ----------------------------- helpers ------------------------------------------------------
@torch.no_grad()
def vae_frames(vae, lat, device):
    """lat (N, 16, 60, 104) -> list of PIL images 480x832 (one latent frame each)."""
    px = vae.decode_to_pixel(lat.to(device=device, dtype=torch.bfloat16).unsqueeze(1),
                             use_cache=False, to_cpu=True).float()[:, 0]     # (N,3,480,832)
    px = ((px + 1.0) / 2.0).clamp(0, 1)
    return [Image.fromarray((f.permute(1, 2, 0).numpy() * 255).astype("uint8")) for f in px]


def _font(size=22):
    for p in ("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
              "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"):
        if os.path.exists(p):
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()


def grid(rows, col_titles, path, row_labels=None, pad=6, bg=(16, 16, 16)):
    """rows = list of lists of PIL images (same size). Adds a title bar per column and a
    per-row label strip on the left."""
    w, h = rows[0][0].size
    ncol = len(rows[0])
    lab_w = 250 if row_labels else 0
    W = lab_w + ncol * w + (ncol + 1) * pad
    H = LABEL_H + len(rows) * (h + pad) + pad
    canvas = Image.new("RGB", (W, H), bg)
    d = ImageDraw.Draw(canvas)
    f, fs = _font(24), _font(19)
    for c, t in enumerate(col_titles):
        d.text((lab_w + pad + c * (w + pad) + 8, 6), t, fill=(255, 255, 255), font=f)
    for r, row in enumerate(rows):
        y = LABEL_H + r * (h + pad)
        if row_labels:
            for i, line in enumerate(row_labels[r].split("\n")):
                d.text((8, y + 10 + 24 * i), line, fill=(200, 220, 255), font=fs)
        for c, im in enumerate(row):
            canvas.paste(im, (lab_w + pad + c * (w + pad), y))
    canvas.save(path)
    print(f"[eyepack] wrote {path} ({canvas.size[0]}x{canvas.size[1]})", flush=True)


# ----------------------------- 1) h-oracle grid ------------------------------------------------
def make_h_oracle_grid(args, device, vae):
    cache = os.path.join(CACHE_DIR, f"cache_p{args.n_pos}.pt")
    assert os.path.exists(cache), f"missing {cache} (run probe_h_oracle.py first)"
    data = torch.load(cache, map_location="cpu", weights_only=False)
    H, T, META = data["h"], data["target"], data["meta"]
    ck = torch.load(os.path.join(CACHE_DIR, "h_oracle.pt"), map_location="cpu",
                    weights_only=False)
    dec = HOracleDecoder().to(device)
    dec.load_state_dict(ck["decoder"])
    dec.eval()
    print(f"[eyepack] h-oracle readout loaded (val {ck['val']:.4f}, "
          f"anchor {ck['val_anchor']:.4f})", flush=True)

    val = [i for i, m in enumerate(META) if m["split"] == "val"]
    rng = random.Random(0)
    rng.shuffle(val)
    seen = set()

    def take(n, cond):
        out = []
        for i in val:
            if len(out) == n:
                break
            key = META[i]["video_id"].split(":")[1]      # corpus_idx: a twin and its
            if key in seen or not cond(META[i]):      # sibling share the prompt -> not "varied"
                continue                              # one row per DISTINCT prompt
            seen.add(key)
            out.append(i)
        return out

    pick = (take(2, lambda m: m["mode"] == "switch" and m["is_anchor"])
            + take(2, lambda m: m["mode"] == "switch" and not m["is_anchor"])
            + take(2, lambda m: m["mode"] == "single"))
    rows, labels = [], []
    for i in pick:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            pred = dec(H[i][None].to(device=device, dtype=torch.float32))
        tgt = T[i][None].float()
        imgs = vae_frames(vae, torch.cat([tgt, pred[:, 0].float().cpu()], 0), device)
        rows.append(imgs)
        m = META[i]
        labels.append(f"{m['video_id']}\n{m['mode']}"
                      + ("/ANCRE" if m["is_anchor"] else "")
                      + f"\nwrite {m['write_j']}, frame {m['frame']}")
    grid(rows, ["CIBLE VRAIE (frame évincée)", "h-ORACLE (lecture du h brut)"],
         os.path.join(OUT, "h_oracle_grid.png"), labels)


def m_mode(META, i):
    return META[i]["mode"]


# ----------------------------- 2) toy grid ------------------------------------------------------
def make_toy_grid(args, device, vae):
    toy_ck = os.path.join(TW.OUT_ROOT, "toy_main.pt")
    assert os.path.exists(toy_ck), \
        f"missing {toy_ck} (re-run: train_writer.py --toy --toy_variants main)"
    a = TW.parse_args(["--load", args.load, "--device", "cuda:0", "--K", "3",
                       "--ages", "10", "20", "--min_age", "10", "--anchor_frac", "0.5",
                       "--cond_dropout", "1.0", "--fixed_positions", "--seed", "0"])
    tr = TW.WriterTrainer(a)
    sw = [v for v in tr.train_vids if v["mode"] == "switch"][:8]      # the toy's own 8 videos
    pick = sw[:4]

    def decode(meta, positions):
        pred, target, _, _ = tr.forward_video(meta, positions, n_bptt=0)
        return pred.detach().float().cpu(), target.detach().float().cpu()

    before, targets, poss = {}, {}, {}
    with torch.no_grad():
        for meta in pick:
            pos = tr.positions_for(meta)
            poss[meta["video_id"]] = pos
            p, t = decode(meta, pos)
            before[meta["video_id"]], targets[meta["video_id"]] = p, t

    ck = torch.load(toy_ck, map_location="cpu", weights_only=False)
    for bi in tr.blocks:
        for name, w in ck["writer"][str(bi)].items():
            tr.wparams[(bi, name)].data.copy_(w.to(tr.device))
    tr.decoder.load_state_dict(ck["decoder"])
    print(f"[eyepack] toy ckpt loaded (step {ck['step']}), "
          f"writer/delta_l1 = {TW.delta_l1(tr.cells, tr.ref):.3e}", flush=True)
    after = {}
    with torch.no_grad():
        for meta in pick:
            p, _ = decode(meta, poss[meta["video_id"]])
            after[meta["video_id"]] = p

    rows, labels = [], []
    for meta in pick:
        vid = meta["video_id"]
        k = 0                                    # the first (anchor-first) position of the set
        lat = torch.cat([targets[vid][k, 0][None], before[vid][k, 0][None],
                         after[vid][k, 0][None]], 0)
        rows.append(vae_frames(vae, lat, device))
        p = poss[vid][k]
        labels.append(f"{vid}\n{meta['mode']}/{p['kind']}\nâge {p['age']} writes\n"
                      f"frame {p['frame']}")
    grid(rows, ["CIBLE VRAIE", "AVANT (écrivain l1 + déc. v1.2)", "APRÈS toy (200 steps)"],
         os.path.join(OUT, "toy_grid.png"), labels)


# ----------------------------- 3) twin pairs ----------------------------------------------------
@torch.no_grad()
def make_twins(args, device, vae):
    vids = D.load_index(verbose=False)
    by_id = {v["video_id"]: v for v in vids}
    tidx = D.twin_index(vids)
    pairs = [(e["main"], e["twin"]) for e in tidx.values() if e["main"] and e["twin"]]
    pairs.sort()
    cache = D.PartCache(size=2)
    from torchvision.io import write_video
    for n, (a_id, b_id) in enumerate(pairs[:args.n_twins], start=1):
        ma, mb = by_id[a_id], by_id[b_id]
        assert ma["prompt"] == mb["prompt"], "twin pairing broken"
        outs = []
        for meta in (ma, mb):
            lat = D.get_record(meta, cache)["latents"].float()          # (F,16,60,104)
            px = vae.decode_to_pixel(lat[None].to(device=device, dtype=torch.bfloat16),
                                     use_cache=False, to_cpu=True).float()[0]
            outs.append(((px + 1.0) / 2.0).clamp(0, 1))
        T = min(o.shape[0] for o in outs)
        vid = torch.cat([outs[0][:T], outs[1][:T]], dim=-1)            # hstack (T,3,H,2W)
        arr = (vid.permute(0, 2, 3, 1) * 255).clamp(0, 255).to(torch.uint8)
        path = os.path.join(OUT, f"twin_pair_{n}.mp4")
        write_video(path, arr, fps=16, video_codec="libx264",
                    options={"crf": "23", "pix_fmt": "yuv420p"})
        print(f"[eyepack] wrote {path} ({arr.shape[0]} frames, {arr.shape[2]}x{arr.shape[1]}, "
              f"{os.path.getsize(path) / 1e6:.1f} MB)\n"
              f"           seeds {ma['seed']} vs {mb['seed']} | {ma['prompt'][:90]}", flush=True)
        with open(os.path.join(OUT, f"twin_pair_{n}.txt"), "w") as f:
            f.write(f"left  = {a_id} seed {ma['seed']}\nright = {b_id} seed {mb['seed']}\n"
                    f"prompt: {ma['prompt']}\n")


def main():
    ap = argparse.ArgumentParser(description="mem2 eye-pack (viz only)")
    ap.add_argument("--only", default="", help="grid_h | grid_toy | twins")
    ap.add_argument("--n_pos", type=int, default=6)
    ap.add_argument("--n_twins", type=int, default=2)
    ap.add_argument("--load", default="direct")
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    os.chdir(REPO)                       # WanVAEWrapper reads repo-relative paths
    device = torch.device("cuda:0")
    vae = build_vae(device)
    want = args.only.split(",") if args.only else ["grid_h", "grid_toy", "twins"]
    if "grid_h" in want:
        make_h_oracle_grid(args, device, vae)
    if "twins" in want:
        make_twins(args, device, vae)
    if "grid_toy" in want:
        make_toy_grid(args, device, vae)
    print(f"[eyepack] done -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
