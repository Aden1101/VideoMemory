#!/usr/bin/env python
# ============================================================================================
# OFFLINE trainer for AnchorFrameDecoder (design v1 "frame de rappel decodee",
# plan_final_sink_virtuel.md DESIGN v1 FINAL + localplan_D_pivot.md).
#
# Trains the state->latent-frame decoder on (states, evicted frame) pairs written by the
# collector agent. Shard format (.pt = torch.save'd LIST OF DICTS, the exact contract):
#   {"states": [3 x (w0, w1, w2) fp16 cpu],       # memory_layers order, leading axis B*nh = 1
#    "target": fp16 (16, 60, 104),                # the evicted latent frame
#    "prompt": str, "chunk_idx": int}
# Collector format v2 adds "cond_emb" (pooled T5, fp16 (4096,)) per pair; v1 shards (no
# cond_emb) remain fully trainable WITHOUT --cond.
#
# v1.1 (recency-bias attack, plan_final CORRECTIF v1):
#   --cond            FiLM-condition the decoder on the pooled prompt embedding (v2 shards
#                     REQUIRED: clear error if a v1 shard is in the glob)
#   --warm_start CKPT initialize from a v1.0 checkpoint (weights only; the new cond/FiLM
#                     params keep their zero-init => step-0 function == the v1.0 model;
#                     optimizer/scheduler start fresh). Exclusive with --resume.
#
# Loss = per-sample MSE in latent space, normalized by the target's mean square (RMS^2)
# -> scale-free, init loss == 1.0 exactly (zero-init head). Optional --dino flag is a v2
# placeholder (NOT implemented in v1).
#
# wandb MANDATORY (house rule: never disabled), project anchor-forcing-memory, run frame_dec_v1.
# Every --viz_every steps: side-by-side PNG grid target-vs-reconstruction through the REAL VAE
# decode (VAE loaded like run_reid_probe.py::build_pipeline, but ONLY the VAE).
#
# Usage (single GPU, e.g. GPU 0):
#   python reference/runs/frame_dec/train_decoder.py \
#       --data_glob "reference/runs/frame_dec/data/*.pt" --steps 20000
# ============================================================================================
import argparse
import glob
import math
import os
import random
import sys
import time

REPO = "/shared/ssd_14T/home/adenelmi/Anchor-Forcing"
sys.path.insert(0, REPO)
HERE = os.path.dirname(os.path.abspath(__file__))
os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")  # house GPU rule

import torch

from wan.modules.video_memory import AnchorFrameDecoder


# ----------------------------- loss ---------------------------------------------------------
def norm_mse(pred, target, eps=1e-8, subject_weight=1.0):
    """MSE normalized per-sample by the target's mean square (RMS^2), then batch-averaged.
    Scale-free: a decoder stuck at zero scores exactly 1.0.
    subject_weight > 1 (v1.2): the CENTER crop60 region (where subjects live, what re-ID
    measures) is weighted W in the squared error; weights are mean-normalized so the loss
    scale (and the =1.0-at-zero property) is preserved. W=1.0 = exact v1.0/v1.1 loss."""
    pf = pred.float()
    tf = target.float()
    if subject_weight != 1.0:
        H, W_ = pf.shape[-2], pf.shape[-1]
        ch, cw = int(H * 0.6), int(W_ * 0.6)
        top, left = (H - ch) // 2, (W_ - cw) // 2
        w = torch.ones_like(tf)
        w[..., top:top + ch, left:left + cw] = subject_weight
        w = w / w.mean(dim=tuple(range(1, w.dim())), keepdim=True)
        se = (pf - tf).pow(2) * w
    else:
        se = (pf - tf).pow(2)
    se = se.flatten(1).mean(dim=1)
    denom = tf.flatten(1).pow(2).mean(dim=1).clamp_min(eps)
    return (se / denom).mean()


# ----------------------------- data ---------------------------------------------------------
def _load_pairs(path, require_cond=False):
    """Collector shards are DICTS {'format_version', ..., 'pairs': [...]} (collect_pairs.py
    format v1 or v2) -- unwrap to the pair list. Tolerates a bare list for forward-compat.
    require_cond=True (--cond): fail LOUDLY on a v1 shard (pairs without 'cond_emb')."""
    obj = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(obj, dict):
        assert "pairs" in obj, f"shard {path} has no 'pairs' key (keys: {list(obj)[:8]})"
        pairs = obj["pairs"]
    else:
        pairs = obj
    if require_cond and pairs and "cond_emb" not in pairs[0]:
        raise RuntimeError(
            f"--cond requires v2 shards with 'cond_emb', but {path} is format v1 "
            f"(collected by the v1.0 collector). Re-collect it with the v2 collector "
            f"(any mode adds cond_emb), point --data_glob at v2 shards only, or drop --cond.")
    return pairs


class ShardStream:
    """Streams collector shards without ever holding more than one in RAM (a single sample's
    states are ~170MB fp16: a shard is GBs). Per epoch: shuffle shard order, load one shard,
    shuffle its items, yield full batches (batches never straddle shards; remainder dropped).
    Optional holdout: the FIRST `holdout` items of the FIRST shard (sorted paths, deterministic)
    are reserved as a fixed val/viz set and excluded from training."""

    def __init__(self, paths, batch_size, holdout=0, seed=0, require_cond=False):
        self.paths = sorted(paths)
        assert self.paths, "no shards matched the data glob"
        self.bs = batch_size
        self.seed = seed
        self.require_cond = require_cond
        self.holdout_n = holdout
        self.holdout_items = []
        if holdout > 0:
            first = _load_pairs(self.paths[0], require_cond=require_cond)
            assert len(first) > 2 * holdout, \
                f"first shard has {len(first)} samples, too few for holdout={holdout}"
            self.holdout_items = first[:holdout]
            del first

    def batches(self):
        """Infinite generator of (items, epoch)."""
        epoch = 0
        while True:
            rng = random.Random(self.seed + 1000 * epoch)
            order = list(self.paths)
            rng.shuffle(order)
            for p in order:
                items = _load_pairs(p, require_cond=self.require_cond)
                if p == self.paths[0] and self.holdout_n > 0:
                    items = items[self.holdout_n:]
                idx = list(range(len(items)))
                rng.shuffle(idx)
                for i in range(0, len(idx) - self.bs + 1, self.bs):
                    yield [items[j] for j in idx[i:i + self.bs]], epoch
                del items
            epoch += 1


def collate(items, device, want_cond=False):
    """List of collector dicts -> (states fp32 on device, target (B,1,16,60,104) fp32,
    cond (B, cond_dim) fp32 or None). want_cond=False keeps the exact v1.0 tuple content
    (cond slot = None)."""
    n_blocks = len(items[0]["states"])
    states = []
    for b in range(n_blocks):
        ws = []
        for j in range(3):  # w0, w1, w2
            ts = [it["states"][b][j] for it in items]
            ts = [t if t.dim() == 3 else t[None] for t in ts]   # (1, di, dh) or (di, dh)
            ws.append(torch.cat(ts, dim=0).to(device=device, dtype=torch.float32))
        states.append(tuple(ws))
    target = torch.stack([it["target"] for it in items]).to(device=device, dtype=torch.float32)
    cond = None
    if want_cond:
        missing = [i for i, it in enumerate(items) if it.get("cond_emb") is None]
        if missing:
            raise RuntimeError(f"--cond but {len(missing)} pair(s) in the batch have no "
                               f"'cond_emb' (v1 shard leaked through the glob?)")
        cond = torch.stack([it["cond_emb"].reshape(-1) for it in items]) \
            .to(device=device, dtype=torch.float32)
    return states, target.unsqueeze(1), cond  # (B, 1, 16, 60, 104)


# ----------------------------- VAE (viz only) -----------------------------------------------
def build_vae(device):
    """ONLY the VAE, loaded exactly as run_reid_probe.py::build_pipeline does for the full
    pipeline: WanVAEWrapper (reads wan_models/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth relative to the
    repo root -> caller must chdir(REPO) first), cast bf16, moved to device. Frozen."""
    from utils.wan_wrapper import WanVAEWrapper
    vae = WanVAEWrapper().to(dtype=torch.bfloat16)
    vae.to(device=device)
    return vae


@torch.no_grad()
def save_viz_grid(vae, target, pred, path, max_n=4):
    """Side-by-side PNG for the human eye: one row per sample, col 0 = target (evicted) frame,
    col 1 = decoder reconstruction, both through the real VAE decode."""
    from torchvision.utils import save_image
    n = min(max_n, target.shape[0])
    lat = torch.cat([target[:n], pred[:n]], dim=0).to(device=target.device, dtype=torch.bfloat16)
    px = vae.decode_to_pixel(lat, use_cache=False, to_cpu=True).float()  # (2n, 1, 3, 480, 832) in [-1,1]
    px = ((px[:, 0] + 1.0) / 2.0).clamp(0, 1)
    tiles = []
    for i in range(n):
        tiles += [px[i], px[n + i]]
    save_image(tiles, path, nrow=2)


# ----------------------------- schedule / ckpt ----------------------------------------------
def make_scheduler(opt, total_steps, warmup, min_lr_ratio):
    def lr_lambda(s):
        if s < warmup:
            return (s + 1) / max(1, warmup)
        p = (s - warmup) / max(1, total_steps - warmup)
        p = min(p, 1.0)
        return min_lr_ratio + (1.0 - min_lr_ratio) * 0.5 * (1.0 + math.cos(math.pi * p))
    return torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)


def save_ckpt(path, decoder, opt, sched, step, args):
    torch.save({"decoder": decoder.state_dict(), "opt": opt.state_dict(),
                "sched": sched.state_dict(), "step": step, "args": vars(args)}, path)
    print(f"[frame_dec] ckpt saved: {path}", flush=True)


# ----------------------------- main ---------------------------------------------------------
def parse_args():
    ap = argparse.ArgumentParser(description="Offline AnchorFrameDecoder trainer (v1)")
    ap.add_argument("--data_glob", default="reference/runs/frame_dec/data/*.pt",
                    help="collector shards (.pt lists of dicts); relative to repo root")
    ap.add_argument("--run_name", default="frame_dec_v1")
    ap.add_argument("--wandb_project", default="anchor-forcing-memory")
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--bs", type=int, default=4)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--wd", type=float, default=0.01)
    ap.add_argument("--warmup", type=int, default=500)
    ap.add_argument("--min_lr_ratio", type=float, default=0.05)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--subject_weight", type=float, default=1.0,
                    help="v1.2: weight of the center crop60 region in the train loss (1.0 = off; val stays unweighted)")
    ap.add_argument("--holdout", type=int, default=8,
                    help="fixed val/viz samples reserved from the first shard (0 = off)")
    ap.add_argument("--log_every", type=int, default=20)
    ap.add_argument("--viz_every", type=int, default=500)
    ap.add_argument("--ckpt_every", type=int, default=1000)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--no_amp", action="store_true", help="disable bf16 autocast (fp32 all the way)")
    ap.add_argument("--resume", default="", help="checkpoint path to resume from")
    # ---- v1.1 conditioning (recency-bias attack) ----
    ap.add_argument("--cond", action="store_true",
                    help="FiLM-condition the decoder on the pooled prompt embedding "
                         "(reads 'cond_emb' from v2 shards; clear error on v1 shards)")
    ap.add_argument("--cond_dim", type=int, default=4096,
                    help="pooled T5 embedding dim (umt5-xxl = 4096)")
    ap.add_argument("--cond_hidden", type=int, default=512,
                    help="hidden width of the decoder's cond MLP")
    ap.add_argument("--warm_start", default="",
                    help="v1.0 decoder ckpt to warm-start from: loads all matching weights, "
                         "new cond/FiLM params keep their zero-init (exact v1.0 function at "
                         "step 0); optimizer/scheduler fresh. Exclusive with --resume.")
    ap.add_argument("--dino", action="store_true",
                    help="v2 placeholder: DINO perceptual loss on the decoded frame. "
                         "NOT implemented in v1 (arg exists so launch scripts are stable).")
    return ap.parse_args()


def main():
    args = parse_args()
    if args.dino:
        raise NotImplementedError("--dino is a v2 placeholder (perceptual loss); v1 = latent MSE only")
    # house rule: wandb is NEVER disabled
    assert os.environ.get("WANDB_MODE", "") != "disabled", \
        "house rule: WANDB_MODE=disabled is forbidden -- every run logs live to wandb"

    os.chdir(REPO)  # WanVAEWrapper + default data_glob use repo-relative paths
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    device = torch.device(args.device)

    run_dir = os.path.join(REPO, "reference/runs/frame_dec", args.run_name)
    ckpt_dir = os.path.join(run_dir, "ckpt")
    viz_dir = os.path.join(run_dir, "viz")
    for d in (run_dir, ckpt_dir, viz_dir):
        os.makedirs(d, exist_ok=True)

    # ---- data ----
    paths = glob.glob(args.data_glob if os.path.isabs(args.data_glob)
                      else os.path.join(REPO, args.data_glob))
    stream = ShardStream(paths, args.bs, holdout=args.holdout, seed=args.seed,
                         require_cond=args.cond)
    print(f"[frame_dec] {len(stream.paths)} shard(s), holdout={args.holdout}, "
          f"cond={'ON' if args.cond else 'off'}", flush=True)

    val_states, val_target, val_cond = (None, None, None)
    if stream.holdout_items:
        val_states, val_target, val_cond = collate(stream.holdout_items, device,
                                                   want_cond=args.cond)
        meta = os.path.join(viz_dir, "holdout_meta.txt")
        with open(meta, "w") as f:
            for i, it in enumerate(stream.holdout_items):
                f.write(f"[{i}] mode={it.get('mode', 'single')} "
                        f"chunk_idx={it.get('chunk_idx')} prompt={it.get('prompt')}\n")
        print(f"[frame_dec] holdout meta -> {meta}", flush=True)

    # ---- model / opt ----
    # fp32 params; bf16 autocast for the matmuls. cond_dim=0 (no --cond) is the STRICT v1.0
    # module (no cond params created, state_dict identical).
    decoder = AnchorFrameDecoder(cond_dim=(args.cond_dim if args.cond else 0),
                                 cond_hidden=args.cond_hidden).to(device)
    n_params = decoder.num_params()
    print(f"[frame_dec] AnchorFrameDecoder params: {n_params:,} (~{n_params/1e6:.1f}M)", flush=True)

    assert not (args.warm_start and args.resume), \
        "--warm_start (weights only, fresh opt) and --resume (full state) are exclusive"
    if args.warm_start:
        ck = torch.load(args.warm_start, map_location="cpu", weights_only=False)
        sd = ck["decoder"] if isinstance(ck, dict) and "decoder" in ck else ck
        missing, unexpected = decoder.load_state_dict(sd, strict=False)
        assert not unexpected, f"warm_start ckpt has unexpected keys: {list(unexpected)[:5]}"
        bad = [k for k in missing if not k.startswith(("cond_norm.", "cond_in.", "cond_out."))]
        assert not bad, f"warm_start missing NON-cond keys (wrong ckpt?): {bad[:5]}"
        print(f"[frame_dec] warm-started from {args.warm_start}: {len(sd)} keys loaded; "
              f"{len(missing)} new cond/FiLM params keep their zero-init"
              + (" (step-0 function == the v1.0 model exactly)" if missing else ""), flush=True)

    opt = torch.optim.AdamW(decoder.parameters(), lr=args.lr, weight_decay=args.wd,
                            betas=(0.9, 0.95))
    sched = make_scheduler(opt, args.steps, args.warmup, args.min_lr_ratio)

    start_step = 0
    if args.resume:
        ck = torch.load(args.resume, map_location="cpu", weights_only=False)
        decoder.load_state_dict(ck["decoder"])
        opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"])
        start_step = ck["step"]
        print(f"[frame_dec] resumed from {args.resume} @ step {start_step}", flush=True)

    vae = build_vae(device)

    import wandb  # mandatory, key in ~/.netrc (longlive env)
    wandb.init(project=args.wandb_project, name=args.run_name, dir=run_dir,
               config=dict(vars(args), n_params=n_params))

    amp = not args.no_amp
    step = start_step
    t0 = time.time()
    for items, epoch in stream.batches():
        if step >= args.steps:
            break
        states, target, cond = collate(items, device, want_cond=args.cond)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
            pred = decoder(states, cond=cond)   # cond=None == the exact v1.0 call
        loss = norm_mse(pred, target, subject_weight=args.subject_weight)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gnorm = torch.nn.utils.clip_grad_norm_(decoder.parameters(), args.clip)
        opt.step()
        sched.step()
        step += 1

        if step % args.log_every == 0 or step == 1:
            dt = time.time() - t0
            sps = args.log_every * args.bs / max(dt, 1e-6) if step > 1 else 0.0
            t0 = time.time()
            log = {"train/loss": loss.item(), "train/grad_norm": float(gnorm),
                   "train/lr": sched.get_last_lr()[0], "train/epoch": epoch,
                   "train/pred_rms": pred.detach().float().pow(2).mean().sqrt().item(),
                   "train/target_rms": target.float().pow(2).mean().sqrt().item(),
                   "train/samples_per_s": sps}
            if args.cond:
                # FiLM leaving its zero-init = the conditioning is actually being used
                log["train/cond_out_wrms"] = \
                    decoder.cond_out.weight.detach().float().pow(2).mean().sqrt().item()
            wandb.log(log, step=step)
            print(f"[frame_dec] step {step}/{args.steps} ep{epoch} "
                  f"loss {loss.item():.4f} lr {sched.get_last_lr()[0]:.2e} "
                  f"gnorm {float(gnorm):.2f} {sps:.2f} samp/s", flush=True)

        if step % args.viz_every == 0:
            decoder.eval()
            with torch.no_grad():
                if val_states is not None:
                    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
                        vpred = decoder(val_states, cond=val_cond)
                    vloss = norm_mse(vpred, val_target)
                    wandb.log({"val/loss": vloss.item()}, step=step)
                    print(f"[frame_dec] step {step} val/loss {vloss.item():.4f}", flush=True)
                    viz_pred, viz_tgt = vpred.float(), val_target
                else:
                    viz_pred, viz_tgt = pred.detach().float(), target
                png = os.path.join(viz_dir, f"step_{step:06d}.png")
                save_viz_grid(vae, viz_tgt, viz_pred, png)
                wandb.log({"viz/target_vs_recon": wandb.Image(png)}, step=step)
                print(f"[frame_dec] viz grid -> {png}", flush=True)
            decoder.train()

        if step % args.ckpt_every == 0:
            save_ckpt(os.path.join(ckpt_dir, f"dec_step{step}.pt"), decoder, opt, sched, step, args)
            save_ckpt(os.path.join(ckpt_dir, "latest.pt"), decoder, opt, sched, step, args)

    save_ckpt(os.path.join(ckpt_dir, "final.pt"), decoder, opt, sched, step, args)
    wandb.finish()
    print(f"[frame_dec] done at step {step}", flush=True)


if __name__ == "__main__":
    main()
