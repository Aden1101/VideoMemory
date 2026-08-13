#!/usr/bin/env python
# ============================================================================================
# SELFTESTS for the memory 2.0 writer trainer (J3 deliverable E).
#
# Everything that can be checked WITHOUT a GPU is checked without one (shapes, split, twin in
# the softmax, target ages, mask semantics, loss identities, warm-start delta == 0 on CPU, and
# the F3.3 end-to-end gradient path on a TINY cell). --gpu adds a short real smoke (2 steps,
# batch 1, --load direct so it starts in seconds) on the actual shards.
#
# The DIE-ON items each have a test here:
#   F2.3  S-E5  warm start == l1@2000-EMA bit for bit (delta_l1 == 0), on CPU.
#   F3.3  S-E6  AnchorFrameDecoder(detach_states=False) + a real backward that reaches
#               qkv/lrs/k_norm/decay_logit THROUGH the decoder (S1 test 2 only backprop'd a
#               dummy loss on the state, so this path was never covered).
#   F1.6  S-E4  sampled targets are AGED (never younger than --min_age) and the anchor target
#               is the last frame of A at a post-switch state.
#   F1.1  S-E2/S-E8  the same-prompt twin IS in the softmax; same-video negatives are masked
#               ONLY near the target (the other segment stays a negative).
#   F4.1  S-E3  per-video split, >= 20 val videos, twins never split.
#
# Usage:
#   python reference/runs/mem2/selftest_trainer.py            # CPU only
#   CUDA_DEVICE_ORDER=PCI_BUS_ID nvidia-smi --query-gpu=index,memory.free --format=csv && \
#   CUDA_VISIBLE_DEVICES=1 python reference/runs/mem2/selftest_trainer.py --gpu
# ============================================================================================
import argparse
import os
import random
import sys

REPO = "/shared/ssd_14T/home/adenelmi/Anchor-Forcing"
PROBE_DIR = os.path.join(REPO, "reference/runs/reid_probe_l1")
FRAME_DEC = os.path.join(REPO, "reference/runs/frame_dec")
HERE = os.path.dirname(os.path.abspath(__file__))
for p in (REPO, PROBE_DIR, FRAME_DEC, HERE):
    sys.path.insert(0, p)
os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")

import torch  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402

import mem2_data as D  # noqa: E402
import train_writer as TW  # noqa: E402
from wan.modules.video_memory import (AnchorFrameDecoder, VideoMemoryLayer,  # noqa: E402
                                      replay_writes)
from train_decoder import norm_mse  # noqa: E402

_FAILS = []


def check(name, ok, detail=""):
    print(f"[selftest][{'PASS' if ok else 'FAIL'}] {name}" + (f" -- {detail}" if detail else ""),
          flush=True)
    if not ok:
        _FAILS.append(name)
    return ok


# ----------------------------- S-E1 loss identities -------------------------------------------
def s_e1():
    torch.manual_seed(0)
    pred = torch.randn(5, 1, 16, 60, 104)
    tgt = torch.randn(5, 1, 16, 60, 104)
    for sw in (1.0, 6.0):
        a = float(norm_mse(pred, tgt, subject_weight=sw))
        b = float(TW.norm_mse_per(pred, tgt, subject_weight=sw).mean())
        check(f"S-E1 norm_mse_per(...).mean() == train_decoder.norm_mse (subject_weight={sw})",
              abs(a - b) < 1e-6, f"{a:.8f} vs {b:.8f}")
    z = torch.zeros_like(pred)
    check("S-E1b a zero prediction scores exactly 1.0 (scale-free loss preserved)",
          abs(float(norm_mse(z, tgt)) - 1.0) < 1e-5)


# ----------------------------- S-E2/S-E8 InfoNCE + bank semantics ------------------------------
def s_e2(bank_path):
    if not os.path.exists(bank_path):
        check("S-E2 bank present", False, f"{bank_path} missing -- run build_bank.py first")
        return None
    bank = TW.Bank(bank_path, torch.device("cpu"))
    vids = bank.videos
    by_id = {v["video_id"]: v for v in vids}
    # a switch video that HAS a twin is ideal; else any switch video
    metas = [v for v in vids if v["mode"] == "switch"]
    meta = metas[0]
    tgt_chunk = meta["write_chunks"][len(meta["write_chunks"]) // 2]
    neg, mask, pool = bank.negatives("train" if bank.split[meta["video_id"]] == "train"
                                     else "val", meta, [tgt_chunk], mask_chunks=2)
    ent = [bank.entries[i] for i in pool.tolist()]
    same = [k for k, e in enumerate(ent) if e["video_id"] == meta["video_id"]]
    near = [k for k in same if abs(ent[k]["chunk"] - tgt_chunk) <= 2]
    far = [k for k in same if abs(ent[k]["chunk"] - tgt_chunk) > 2]
    check("S-E8 F1.3 masking: same-video entries NEAR the target are masked",
          all(bool(mask[0, k]) for k in near), f"{len(near)} near entries")
    check("S-E8 F1.3 masking: same-video entries FAR from the target stay NEGATIVES "
          "(the anti-contamination pressure)",
          len(far) > 0 and not any(bool(mask[0, k]) for k in far), f"{len(far)} far entries")

    # the twin must be reachable in the softmax
    twin_v = [v for v in vids if v["twin"]]
    tv = None
    for v in twin_v:
        sib = D.twin_of(v["video_id"], bank.twin_index, by_id)
        if sib is not None:
            tv = by_id[sib]
            break
    ok_full = False
    if tv is not None:
        side = bank.split[tv["video_id"]]
        neg2, mask2, pool2 = bank.negatives(side, tv, [tv["write_chunks"][5]], n_neg=0)
        tw_entries = set(bank.twin_entries(tv).tolist())
        ok_full = bool(tw_entries & set(pool2.tolist()))
        # subsampled bank: the twin must be FORCE-added (mandate)
        neg3, mask3, pool3 = bank.negatives(side, tv, [tv["write_chunks"][5]], n_neg=64)
        ok_forced = bool(tw_entries & set(pool3.tolist()))
        check("S-E2 the same-prompt TWIN is in the softmax (full bank AND n_neg subsample)",
              ok_full and ok_forced,
              f"twin entries {len(tw_entries)}, full={ok_full}, forced={ok_forced}")
    else:
        check("S-E2 twin present in the index", False)

    # infonce mechanics: a masked negative can never win
    torch.manual_seed(0)
    z = torch.nn.functional.normalize(torch.randn(2, 16), dim=1)
    pos = torch.nn.functional.normalize(torch.randn(2, 16), dim=1)
    negs = torch.cat([z, torch.nn.functional.normalize(torch.randn(4, 16), dim=1)], 0)
    m = torch.zeros(2, negs.shape[0], dtype=torch.bool)
    m[0, 0] = True                                # the perfect (identical) negative, masked
    loss_m, acc_m = TW.infonce(z, pos, negs, 0.07, m)
    loss_u, acc_u = TW.infonce(z, pos, negs, 0.07, None)
    check("S-E2b infonce: masking an identical negative removes it from the softmax",
          float(loss_m) < float(loss_u) and torch.isfinite(loss_m),
          f"masked {float(loss_m):.3f} < unmasked {float(loss_u):.3f}")
    return bank


# ----------------------------- S-E3 split ------------------------------------------------------
def s_e3():
    vids = D.load_index(verbose=False)
    split = D.assign_split(vids, 18, 4)
    tidx = D.twin_index(vids)
    val = [v for v in vids if split[v["video_id"]] == "val"]
    check("S-E3 F4.1 per-video split with >= 20 val VIDEOS",
          len(val) >= 20, f"{len(val)} val / {len(vids)} total")
    bad = [c for c, e in tidx.items() if e["twin"] and e["main"]
           and split[e["twin"]] != split[e["main"]]]
    check("S-E3b twins never straddle the split", not bad, f"{len(bad)} offenders")
    # determinism
    check("S-E3c split is deterministic", D.assign_split(vids, 18, 4) == split)
    return vids, split


# ----------------------------- S-E4 target ages ------------------------------------------------
def s_e4(vids):
    rng = random.Random(0)
    ok_age, ok_frame, ok_anchor, n = True, True, True, 0
    for meta in vids[:60]:
        for min_age in (1, 10):
            pos = D.sample_positions(meta, rng, k=4, ages=(1, 2, 5, 10, 20),
                                     anchor_frac=0.5, min_age_writes=min_age)
            for p in pos:
                n += 1
                ok_age &= (p["age"] >= min_age and p["i"] - p["j"] + 1 == p["age"]
                           and 0 <= p["j"] <= p["i"] < len(meta["write_chunks"]))
                c = meta["write_chunks"][p["j"]]
                F, wc = meta["num_frame_per_block"], meta["window_chunks"]
                ok_frame &= (p["frame"] == (c - wc) * F + (F - 1))
                if p["kind"] == "anchor":
                    ok_anchor &= (p["frame"] == meta["switch_frame"] - 1
                                  and meta["write_chunks"][p["i"]]
                                  >= meta["switch_chunk"] + D.MIN_GAP_CHUNKS
                                  and p["cond"] == "a")
    check("S-E4 F1.6 sampled ages honour min_age and age == i - j + 1", ok_age, f"{n} positions")
    check("S-E4b target frame == last frame of the evicted chunk", ok_frame)
    check("S-E4c anchor positions: frame == switch_frame-1, state post-switch (>= +5 chunks), "
          "cond = A", ok_anchor)
    # aged targets really are older: age 20 must be 20 writes back
    meta = [v for v in vids if v["mode"] == "switch"][0]
    p20 = [p for p in D.sample_positions(meta, random.Random(1), k=40, ages=(20,),
                                         anchor_frac=0.0, min_age_writes=20)]
    check("S-E4d age-20 targets exist on 120f videos and sit 20 writes back",
          bool(p20) and all(p["age"] == 20 for p in p20), f"{len(p20)} sampled")


# ----------------------------- S-E5 warm start (CPU) --------------------------------------------
def s_e5(config_path):
    config = OmegaConf.load(config_path)
    config.distributed = False
    dev = torch.device("cpu")
    cells, blocks, _ = TW.build_cells(config, dev, mode="direct")
    ref, src = TW.ref_theta_l1(config, blocks, dev)
    d = TW.delta_l1(cells, ref)
    check("S-E5 F2.3 warm start: writer/delta_l1 == 0 vs the l1@2000-EMA checkpoint",
          d == 0.0, f"delta_l1={d:.3e}, source={src}, blocks={blocks}")
    # and it is NOT trivially zero: perturb one tensor and the metric must move
    with torch.no_grad():
        cells[blocks[0]].qkv.weight.add_(0.01)
    d2 = TW.delta_l1(cells, ref)
    check("S-E5b delta_l1 is a real metric (a perturbation moves it)", d2 > 0, f"{d2:.3e}")
    with torch.no_grad():
        cells[blocks[0]].qkv.weight.sub_(0.01)
    return cells, blocks, config


# ----------------------------- S-E6 the F3.3 gradient path (tiny, CPU) --------------------------
def s_e6():
    check("S-E6a AnchorFrameDecoder STILL defaults to detach_states=True (the silent bomb "
          "F3.3 warns about)", AnchorFrameDecoder().detach_states is True)
    torch.manual_seed(0)
    dim, gh, gw = 32, 3, 4
    cell = VideoMemoryLayer(dim=dim, num_heads=1, inter_multi=2, use_decay=True,
                            muon_update_steps=2, checkpoint_write=True, detach_state=False)
    dec = AnchorFrameDecoder(dim=dim, inter_multi=2, num_blocks=1, grid_h=gh, grid_w=gw,
                             fuse_hidden=16, detach_states=False)
    torch.nn.init.normal_(dec.head.weight, std=0.02)     # zero-init head would kill the grad
    h_seq = [torch.randn(1, 6, dim) for _ in range(6)]
    st = replay_writes(cell, h_seq, detach_state=False, use_checkpoint=True)
    pred = dec([st])
    loss = pred.pow(2).mean()
    loss.backward()
    grads = {n: (p.grad is not None and float(p.grad.abs().sum()) > 0)
             for n, p in cell.named_parameters()
             if n in TW.WRITE_PARAMS}
    check("S-E6 F3.3 the loss reaches theta_w THROUGH the decoder "
          "(detach_states=False, checkpointed writes)",
          all(grads.values()), str(grads))
    q = cell.q_norm.weight
    check("S-E6b negative control: q_norm (read-only) gets no gradient",
          q.grad is None or float(q.grad.abs().sum()) == 0.0)
    # and the bomb: with the DEFAULT decoder the writer gets nothing
    cell2 = VideoMemoryLayer(dim=dim, num_heads=1, inter_multi=2, use_decay=True,
                            muon_update_steps=2, detach_state=False)
    dec2 = AnchorFrameDecoder(dim=dim, inter_multi=2, num_blocks=1, grid_h=gh, grid_w=gw,
                              fuse_hidden=16)                      # detach_states default
    torch.nn.init.normal_(dec2.head.weight, std=0.02)
    st2 = replay_writes(cell2, h_seq, detach_state=False)
    dec2([st2]).pow(2).mean().backward()
    dead = all(p.grad is None or float(p.grad.abs().sum()) == 0.0
               for n, p in cell2.named_parameters() if n in TW.WRITE_PARAMS)
    check("S-E6c the default decoder (detach_states=True) gives the writer EXACTLY ZERO "
          "gradient -- the 2-day flat curve F3.3 predicted", dead)


# ----------------------------- S-E7 latent embedding --------------------------------------------
def s_e7():
    x = torch.randn(3, 1, 16, 60, 104, requires_grad=True)
    e = D.latent_embed(x)
    check("S-E7 latent_embed: L2-normalized, right dim, finite",
          abs(float(e.pow(2).sum(1).max()) - 1.0) < 1e-4
          and e.shape[1] == D.latent_embed_dim() and bool(torch.isfinite(e).all()),
          f"shape {tuple(e.shape)} (expected dim {D.latent_embed_dim()})")
    e.sum().backward()
    check("S-E7b latent_embed is differentiable (the anchor side carries the gradient)",
          x.grad is not None and float(x.grad.abs().sum()) > 0)
    # the crop really is the centre 60%
    y = torch.zeros(1, 1, 16, 60, 104)
    y[..., 0, 0] = 100.0                       # a corner spike must be cropped away
    check("S-E7c the crop is centred (corner content is dropped)",
          float(D.latent_embed(y).abs().sum()) == 0.0)


# ----------------------------- GPU smoke ---------------------------------------------------------
def gpu_smoke(args):
    if not torch.cuda.is_available():
        check("GPU smoke", False, "no CUDA device visible")
        return
    argv = ["--load", "direct", "--device", "cuda:0", "--bs", "1", "--K", "2",
            "--steps", "2", "--cache_videos", "2", "--n_neg", "512",
            "--prox", "1e-4", "--lam_inst", "0.3", "--lam_ramp", "0", "1"]
    a = TW.parse_args(argv)
    tr = TW.WriterTrainer(a)
    meta = [v for v in tr.train_vids if v["mode"] == "switch"][0]
    logs = tr.train_step([meta], lam=0.3)
    g = tr.masters[(tr.blocks[0], "qkv.weight")].grad
    check("GPU smoke: one full step (replay 35 writes -> decode -> recon+InfoNCE -> backward)",
          torch.isfinite(torch.tensor(logs["train/loss"])).item(),
          " ".join(f"{k}={v:.4f}" for k, v in logs.items() if isinstance(v, float)
                   and "by_" not in k))
    check("GPU smoke: F3.3 qkv.weight gradient non-zero and finite",
          g is not None and bool(torch.isfinite(g).all()) and float(g.abs().sum()) > 0,
          f"|grad|_1 = {float(g.abs().sum()):.3e}" if g is not None else "None")
    logs2 = tr.train_step([meta], lam=0.3)
    check("GPU smoke: second step runs and the causal margin is logged from step 0 (F1.1.4)",
          "ctrl/causal_margin" in logs2,
          f"margin = {logs2.get('ctrl/causal_margin', float('nan')):.4f}")
    d = TW.delta_l1(tr.cells, tr.ref)
    check("GPU smoke: theta_w actually MOVED after 2 optimizer steps", d > 0, f"delta_l1={d:.3e}")
    print(f"[selftest] peak VRAM (F5.4, K=2, bs=1, full BPTT): {tr.peak_vram:.1f} GB", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", action="store_true", help="also run the short GPU smoke")
    ap.add_argument("--config", default=TW.DEFAULT_CONFIG)
    ap.add_argument("--bank", default=TW.BANK_PT)
    args = ap.parse_args()

    s_e1()
    s_e2(args.bank)
    vids, split = s_e3()
    s_e4(vids)
    s_e6()
    s_e7()
    s_e5(args.config)
    if args.gpu:
        gpu_smoke(args)
    print(f"\n[selftest] {'ALL GREEN' if not _FAILS else f'{len(_FAILS)} FAILURE(S): {_FAILS}'}",
          flush=True)
    sys.exit(0 if not _FAILS else 1)


if __name__ == "__main__":
    main()
