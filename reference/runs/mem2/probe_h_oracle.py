#!/usr/bin/env python
# ============================================================================================
# h-ORACLE PROBE -- the UPPER BOUND of the whole memory 2.0 chantier (gate F5.3a of
# reference/memory2_redteam.md; the design filed it as "risk n°3, only the 2.0 run will tell",
# the red-team showed it is MEASURABLE BEFORE the run, without any BPTT and without a video
# model).
#
# Argument (data-processing inequality): the writer's ONLY input is the eviction-feed hidden
# stream h. Whatever instance detail is not in h cannot be in the state, whatever the writer.
# So: train a decoder to predict the target frame DIRECTLY from the h of the chunk that
# CONTAINS it -- no state, no recurrence, no writer. That number bounds every possible writer.
#
#   h at write i  = (1, 4680, 1536) per block = the 3 frames of the evicted chunk, frame-major
#   target        = the LAST frame of that chunk = tokens [2*1560 : 3*1560]  <- ALIGNED
#   readout       = the AnchorFrameDecoder CONTENT path, verbatim capacity (per-token
#                   RMSNorm per block -> concat 3*1536 -> 4608->3072->GELU->1536 -> LN ->
#                   head 1536->64 -> the same unpatchify), ~19M params, UNCONDITIONED
#                   (no cond/FiLM: a text shortcut would inflate the bound, red-team F1.1).
#
# READING (pre-registered): the decoder v1.1 scored val 0.867 and v1.2 val 0.812 in the SAME
# normalized-MSE units (1.0 = predict zero). If the h-oracle lands near ~1.0 -> the instance is
# not linearly readable from h and memory 2.0 is bounded at birth. If it lands far below 0.8
# -> h carries the content and the bottleneck really is the STATE (the brief's thesis), which
# is what the writer training is for. (Caveat kept in the report: the val SETS differ -- v1.x
# val was 16 samples from 8 videos, red-team F4.1 -- so this is an order-of-magnitude read.)
#
# Also reported: the same probe on the ANCHOR targets only (last frame of A = the revisit
# target of the harness) and a SHUFFLED-h control (h of another sample -> must collapse to
# ~1.0, proving the probe is not leaking the target through anything else).
#
# Usage (GPU pre-check in the SAME command):
#   CUDA_DEVICE_ORDER=PCI_BUS_ID nvidia-smi --query-gpu=index,memory.free --format=csv && \
#   CUDA_VISIBLE_DEVICES=2 python reference/runs/mem2/probe_h_oracle.py --steps 3000
#   ... --extract_only          # build the compact (h_last_frame, target) cache and stop
# wandb MANDATORY (house rule).
# ============================================================================================
import argparse
import math
import os
import random
import sys
import time

REPO = "/shared/ssd_14T/home/adenelmi/Anchor-Forcing"
FRAME_DEC = os.path.join(REPO, "reference/runs/frame_dec")
HERE = os.path.dirname(os.path.abspath(__file__))
for p in (REPO, FRAME_DEC, HERE):
    sys.path.insert(0, p)
os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as Fn  # noqa: E402

import mem2_data as D  # noqa: E402
from wan.modules.video_memory import AnchorFrameDecoder, RMSNorm  # noqa: E402
from train_decoder import norm_mse, make_scheduler  # noqa: E402

CACHE_DIR = os.path.join(D.RUNS_DIR, "h_oracle")
V11_VAL, V12_VAL = 0.867, 0.812          # the numbers this probe is read against


# ----------------------------- the readout ------------------------------------------------------
class HOracleDecoder(nn.Module):
    """AnchorFrameDecoder's CONTENT path, fed by h tokens instead of state reads. Same widths,
    same output layout (the unpatchify method is reused verbatim from AnchorFrameDecoder, so a
    layout bug cannot make the two incomparable)."""

    _unpatchify = AnchorFrameDecoder._unpatchify

    def __init__(self, dim=1536, num_blocks=3, grid_h=30, grid_w=52, patch_h=2, patch_w=2,
                 z_dim=16, fuse_hidden=3072):
        super().__init__()
        self.dim, self.num_blocks = dim, num_blocks
        self.grid_h, self.grid_w = grid_h, grid_w
        self.patch_h, self.patch_w, self.z_dim = patch_h, patch_w, z_dim
        self.m = grid_h * grid_w
        self.read_norms = nn.ModuleList([RMSNorm(dim, eps=1e-5) for _ in range(num_blocks)])
        self.fuse1 = nn.Linear(num_blocks * dim, fuse_hidden)
        self.fuse2 = nn.Linear(fuse_hidden, dim)
        self.out_norm = nn.LayerNorm(dim, eps=1e-6)
        self.head = nn.Linear(dim, z_dim * patch_h * patch_w)
        nn.init.zeros_(self.head.weight)          # exactly-null frame at init -> loss == 1.0
        nn.init.zeros_(self.head.bias)

    def num_params(self):
        return sum(p.numel() for p in self.parameters())

    def forward(self, h):
        """h: (B, num_blocks, m, dim) -- the target frame's tokens at each memory block."""
        reads = [self.read_norms[b](h[:, b]) for b in range(self.num_blocks)]
        x = torch.cat(reads, dim=-1)
        x = self.fuse2(Fn.gelu(self.fuse1(x)))
        return self._unpatchify(self.head(self.out_norm(x)))


# ----------------------------- extraction --------------------------------------------------------
def extract(args):
    """(h of the target frame's tokens, target latent frame) for n_pos writes per video."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    vids = D.load_index(verbose=True)
    split = D.assign_split(vids, args.n_val_switch, args.n_val_single)
    cache = D.PartCache(size=1)
    order = sorted(range(len(vids)), key=lambda i: (vids[i]["part"], vids[i]["vi"]))
    H, T, META = [], [], []
    t0 = time.time()
    rng = random.Random(args.seed)
    for n, vi in enumerate(order):
        meta = vids[vi]
        nw = len(meta["write_chunks"])
        js = sorted(rng.sample(range(nw), min(args.n_pos, nw)))
        aj = D.anchor_write_index(meta)
        if aj is not None and aj not in js:
            js[0] = aj                                     # always keep the anchor target
            js = sorted(set(js))
        rec = D.get_record(meta, cache)
        F = meta["num_frame_per_block"]
        sl = D.frame_token_slice(F - 1, F=F)               # the chunk's LAST frame
        for j in js:
            hs = torch.stack([rec["h_seq"][str(b)][j][0, sl].clone()
                              for b in (5, 16, 27)])        # (3, 1560, 1536) fp16
            f = D.frame_of_write(meta, j)
            H.append(hs)
            T.append(rec["latents"][f].clone())
            META.append({"video_id": meta["video_id"], "split": split[meta["video_id"]],
                         "mode": meta["mode"], "write_j": int(j), "frame": int(f),
                         "is_anchor": bool(aj is not None and j == aj)})
        if n % 20 == 0:
            print(f"[h-oracle] extract {n + 1}/{len(vids)} videos, {len(H)} samples "
                  f"({time.time() - t0:.0f}s)", flush=True)
    cache.clear()
    H = torch.stack(H)                                     # (N, 3, 1560, 1536) fp16
    T = torch.stack(T)                                     # (N, 16, 60, 104) fp16
    path = os.path.join(CACHE_DIR, f"cache_p{args.n_pos}.pt")
    torch.save({"h": H, "target": T, "meta": META}, path + ".tmp")
    os.replace(path + ".tmp", path)
    print(f"[h-oracle] cache -> {path}: h {tuple(H.shape)} "
          f"({H.numel() * 2 / 1e9:.1f} GB), {len(META)} samples", flush=True)
    return {"h": H, "target": T, "meta": META}


# ----------------------------- train ---------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="h-oracle upper bound (gate F5.3a)")
    ap.add_argument("--n_pos", type=int, default=6, help="target writes sampled per video")
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--wd", type=float, default=0.01)
    ap.add_argument("--warmup", type=int, default=200)
    ap.add_argument("--min_lr_ratio", type=float, default=0.05)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--subject_weight", type=float, default=6.0, help="train loss (v1.2 recipe)")
    ap.add_argument("--n_val_switch", type=int, default=18)
    ap.add_argument("--n_val_single", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--log_every", type=int, default=50)
    ap.add_argument("--val_every", type=int, default=250)
    ap.add_argument("--extract_only", action="store_true")
    ap.add_argument("--run_name", default="mem2_h_oracle")
    ap.add_argument("--wandb_project", default="anchor-forcing-memory")
    args = ap.parse_args()
    assert os.environ.get("WANDB_MODE", "") != "disabled", \
        "house rule: WANDB_MODE=disabled is forbidden"

    path = os.path.join(CACHE_DIR, f"cache_p{args.n_pos}.pt")
    data = torch.load(path, map_location="cpu", weights_only=False) \
        if os.path.exists(path) else extract(args)
    if args.extract_only:
        return
    H, T, META = data["h"], data["target"], data["meta"]
    tr_idx = [i for i, m in enumerate(META) if m["split"] == "train"]
    va_idx = [i for i, m in enumerate(META) if m["split"] == "val"]
    anchor_va = [i for i in va_idx if META[i]["is_anchor"]]
    print(f"[h-oracle] {len(tr_idx)} train / {len(va_idx)} val samples "
          f"({len(anchor_va)} val anchors), h {tuple(H.shape)}", flush=True)

    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    dec = HOracleDecoder().to(device)
    print(f"[h-oracle] readout params: {dec.num_params():,} "
          f"(~{dec.num_params() / 1e6:.1f}M)", flush=True)
    opt = torch.optim.AdamW(dec.parameters(), lr=args.lr, weight_decay=args.wd,
                            betas=(0.9, 0.95))
    sched = make_scheduler(opt, args.steps, args.warmup, args.min_lr_ratio)

    import wandb
    os.makedirs(CACHE_DIR, exist_ok=True)
    wandb.init(project=args.wandb_project, name=args.run_name, dir=CACHE_DIR,
               config=dict(vars(args), n_params=dec.num_params(),
                           v11_val=V11_VAL, v12_val=V12_VAL))

    def batch(idx, n, shuffle_h=False):
        sel = random.sample(idx, min(n, len(idx)))
        h = H[sel].to(device=device, dtype=torch.float32)
        if shuffle_h:                                  # control: h of OTHER samples
            oth = random.sample(idx, len(sel))
            h = H[oth].to(device=device, dtype=torch.float32)
        t = T[sel].to(device=device, dtype=torch.float32).unsqueeze(1)
        return h, t

    @torch.no_grad()
    def evaluate(idx, shuffle_h=False, bs=8):
        dec.eval()
        tot, n = 0.0, 0
        for i in range(0, len(idx), bs):
            sel = idx[i:i + bs]
            h = H[sel].to(device=device, dtype=torch.float32)
            if shuffle_h:
                h = H[random.sample(idx, len(sel))].to(device=device, dtype=torch.float32)
            t = T[sel].to(device=device, dtype=torch.float32).unsqueeze(1)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                p = dec(h)
            tot += float(norm_mse(p, t)) * len(sel)          # UNWEIGHTED: comparable to v1.x
            n += len(sel)
        dec.train()
        return tot / max(1, n)

    t0 = time.time()
    best = 1e9
    for s in range(args.steps):
        h, t = batch(tr_idx, args.bs)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            pred = dec(h)
        loss = norm_mse(pred, t, subject_weight=args.subject_weight)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(dec.parameters(), args.clip)
        opt.step()
        sched.step()
        if (s + 1) % args.log_every == 0 or s == 0:
            wandb.log({"train/loss": float(loss), "train/grad_norm": float(gn),
                       "train/lr": sched.get_last_lr()[0],
                       "train/steps_per_s": (s + 1) / max(1e-6, time.time() - t0)}, step=s + 1)
            print(f"[h-oracle] step {s + 1}/{args.steps} train {float(loss):.4f} "
                  f"({(time.time() - t0) / (s + 1):.2f} s/step)", flush=True)
        if (s + 1) % args.val_every == 0 or s == args.steps - 1:
            v = evaluate(va_idx)
            va = evaluate(anchor_va) if anchor_va else float("nan")
            best = min(best, v)
            wandb.log({"val/loss": v, "val/loss_anchor": va,
                       "ref/v11_val": V11_VAL, "ref/v12_val": V12_VAL}, step=s + 1)
            print(f"[h-oracle] step {s + 1} VAL {v:.4f} (anchor {va:.4f}) "
                  f"| v1.1 {V11_VAL} v1.2 {V12_VAL}", flush=True)
    ctrl = evaluate(va_idx, shuffle_h=True)
    v = evaluate(va_idx)
    va = evaluate(anchor_va) if anchor_va else float("nan")
    wandb.log({"val/loss_final": v, "val/loss_anchor_final": va,
               "ctrl/shuffled_h": ctrl}, step=args.steps)
    torch.save({"decoder": dec.state_dict(), "val": v, "val_anchor": va, "ctrl": ctrl,
                "args": vars(args)}, os.path.join(CACHE_DIR, "h_oracle.pt"))
    print("\n[h-oracle] ================= GATE F5.3a =================")
    print(f"[h-oracle] val norm_mse (unweighted, held-out VIDEOS) : {v:.4f}")
    print(f"[h-oracle] val on ANCHOR targets (last frame of A)    : {va:.4f}")
    print(f"[h-oracle] control, shuffled h                        : {ctrl:.4f} (must be ~1.0)")
    print(f"[h-oracle] reference: decoder v1.1 {V11_VAL} / v1.2 {V12_VAL} "
          f"(state-based, other val set)")
    verdict = ("OPEN  -- h carries the frame content; the bottleneck is the STATE, "
               "not the h stream" if v < 0.8 * V12_VAL else
               "TIGHT -- h barely beats the state-based decoders: memory 2.0 is bounded "
               "close to its current plateau" if v < V12_VAL else
               "CLOSED -- even raw h does not reconstruct the target: no writer can")
    print(f"[h-oracle] VERDICT: {verdict}")
    print("[h-oracle] ===============================================\n", flush=True)
    wandb.finish()


if __name__ == "__main__":
    main()
