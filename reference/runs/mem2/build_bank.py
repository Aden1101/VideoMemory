#!/usr/bin/env python
# ============================================================================================
# MEMORY 2.0 -- InfoNCE BANK builder (J2 deliverable A, design 3.4bis + red-team F1.1b/F1.3/F4.1).
#
# What a "bank" is here: the negatives of the instance term are TARGETS, i.e. DATA (latent
# frames of the corpus). They never move, so they are embedded ONCE, offline, and the trainer
# just does a matmul against them (MoCo without momentum). This script builds:
#
#   bank.pt          entries (metadata) + L2-normalized LATENT crop60 embeddings (N, Demb)
#                    + per-video pooled T5 cond (V, 4096) + cond k-NN + split + twin index
#   bank_latents.pt  the target latent frames themselves (N, 16, 60, 104) fp16 -- used by the
#                    h-oracle probe, the viz grids and any bank refresh (optional, --no_latents)
#
# EMBEDDING SPACE = LATENT, NOT DINO: the J3 mandate keeps the VAE out of the loss (it is only
# a viz tool, exactly as in train_decoder.py), so the instance term lives in latent crop60
# space: centre crop 60% (what the re-ID metric scores), 2x2 avg-pool, per-sample RMS norm,
# L2 norm (mem2_data.latent_embed). This is the design's documented `--inst_space latent`
# fallback, promoted to default by the mandate; it also removes red-team F1.5 (no differentiable
# VAE decode -> no pixel-adversarial anchor). --dino additionally stores a DINOv2 ViT-S/14
# embedding of the VAE-decoded target as an AUDIT space (F1.7: never the training space).
#
# TARGETS PER VIDEO: evicted frames sampled by AGE (spread over the whole write sequence, i.e.
# over all reachable ages, red-team F1.6) + FOR SWITCH VIDEOS the anchor frame (last frame of
# A, switch_frame - 1) which is the actual revisit target.
#
# SPLIT: per VIDEO (red-team F4.1), >= 20 val videos, stratified switch/single, and a TWIN
# ALWAYS lands on its sibling's side (key = corpus_idx). The 8 held-out probe pairs were
# excluded at collection time; this script re-greps every prompt/prompt_b of every shard that
# actually landed on disk and REFUSES to write a bank if anything hits.
#
# Usage:
#   python reference/runs/mem2/build_bank.py                       # build + selftests
#   python reference/runs/mem2/build_bank.py --selftest_only       # re-check an existing bank
#   python reference/runs/mem2/build_bank.py --dino                # + audit embeddings (GPU)
# ============================================================================================
import argparse
import os
import sys
import time

REPO = "/shared/ssd_14T/home/adenelmi/Anchor-Forcing"
HERE = os.path.dirname(os.path.abspath(__file__))
for p in (REPO, HERE):
    sys.path.insert(0, p)
os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")

import torch  # noqa: E402
import torch.nn.functional as Fn  # noqa: E402

import mem2_data as D  # noqa: E402

BANK_DIR = os.path.join(D.RUNS_DIR, "bank")
BANK_PT = os.path.join(BANK_DIR, "bank.pt")
BANK_LAT = os.path.join(BANK_DIR, "bank_latents.pt")

_FAILS = []


def check(name, ok, detail=""):
    print(f"[bank][{'PASS' if ok else 'FAIL'}] {name}" + (f" -- {detail}" if detail else ""),
          flush=True)
    if not ok:
        _FAILS.append(name)
    return ok


# ----------------------------- target sampling -----------------------------------------------
def target_writes(meta, n_per_video):
    """Write indices whose evicted frame becomes a bank target: n_per_video positions spread
    over the WHOLE write sequence (so every reachable age is represented) + the anchor write
    of switch videos (target = last frame of A). Sorted, deduplicated."""
    n = len(meta["write_chunks"])
    k = min(n_per_video, n)
    js = {int(round(i * (n - 1) / max(1, k - 1))) for i in range(k)} if k > 1 else {n - 1}
    aj = D.anchor_write_index(meta)
    if aj is not None:
        js.add(int(aj))
    return sorted(js)


# ----------------------------- DINO audit embedder (optional) ---------------------------------
def dino_audit_embed(latents, device, vae, batch=8):
    """VAE-decode -> centre crop60 in PIXEL space -> DINOv2 ViT-S/14 CLS, L2-normalized.
    AUDIT ONLY (the eval harness scores with ViT-B/14: different size, still the same family --
    red-team F1.7 -- so this is reported, never optimized)."""
    hub = os.path.expanduser("~/.cache/torch/hub")
    torch.hub.set_dir(hub)
    m = torch.hub.load(os.path.join(hub, "facebookresearch_dinov2_main"), "dinov2_vits14",
                       source="local", pretrained=True).eval().to(device)
    for p in m.parameters():
        p.requires_grad_(False)
    mean = torch.tensor((0.485, 0.456, 0.406), device=device).view(1, 3, 1, 1)
    std = torch.tensor((0.229, 0.224, 0.225), device=device).view(1, 3, 1, 1)
    out = []
    with torch.no_grad():
        for i in range(0, latents.shape[0], batch):
            lat = latents[i:i + batch].to(device=device, dtype=torch.bfloat16).unsqueeze(1)
            px = vae.decode_to_pixel(lat, use_cache=False, to_cpu=False).float()[:, 0]
            px = ((px + 1.0) / 2.0).clamp(0, 1)
            H, W = px.shape[-2:]
            ch, cw = int(H * 0.6), int(W * 0.6)
            px = px[..., (H - ch) // 2:(H - ch) // 2 + ch, (W - cw) // 2:(W - cw) // 2 + cw]
            x = Fn.interpolate(px, size=(224, 224), mode="bilinear", align_corners=False)
            out.append(Fn.normalize(m((x - mean) / std), dim=-1).cpu())
            if (i // batch) % 20 == 0:
                print(f"[bank]   dino {i}/{latents.shape[0]}", flush=True)
    del m
    torch.cuda.empty_cache()
    return torch.cat(out, 0)


# ----------------------------- build ----------------------------------------------------------
def build(args):
    os.makedirs(BANK_DIR, exist_ok=True)
    vids = D.load_index(rebuild=args.rebuild_index)
    vids_by_id = {v["video_id"]: v for v in vids}

    hits, n_held = D.held_out_hits(vids)
    check("held-out grep on the shards on disk (prompt AND prompt_b, both directions)",
          not hits, f"0 hit on {n_held} probe prompts, {len(vids)} videos"
          if not hits else f"{hits[:3]}")
    assert not hits, "HELD-OUT LEAK -- refusing to build a bank on a contaminated corpus"

    split = D.assign_split(vids, args.n_val_switch, args.n_val_single)
    tidx = D.twin_index(vids)

    # videos grouped by part = sequential mmap reads
    order = sorted(range(len(vids)), key=lambda i: (vids[i]["part"], vids[i]["vi"]))
    cache = D.PartCache(size=1)
    entries, lat_chunks = [], []
    conds = torch.zeros(len(vids), 4096, dtype=torch.float32)
    t0 = time.time()
    for n_done, vi in enumerate(order):
        meta = vids[vi]
        js = target_writes(meta, args.n_per_video)
        frames = [D.frame_of_write(meta, j) for j in js]
        lat = D.load_frames(meta, cache, frames)                     # (k, 16, 60, 104) fp32
        ca, cb = D.load_conds(meta, cache)
        conds[vi] = ca                                               # cond of the video's A
        aj = D.anchor_write_index(meta)
        for j, f, l in zip(js, frames, lat):
            entries.append({
                "video_id": meta["video_id"], "video_i": int(vi),
                "corpus_idx": meta["corpus_idx"], "twin": bool(meta["twin"]),
                "mode": meta["mode"], "split": split[meta["video_id"]],
                "write_j": int(j), "chunk": int(meta["write_chunks"][j]), "frame": int(f),
                "is_anchor": bool(aj is not None and j == aj),
                "cond_side": D.cond_for_frame(meta, f),
                "seed": meta["seed"],
            })
        lat_chunks.append(lat.to(torch.float16))
        if n_done % 20 == 0:
            print(f"[bank] {n_done + 1}/{len(vids)} videos ({time.time() - t0:.0f}s)",
                  flush=True)
    cache.clear()

    latents = torch.cat(lat_chunks, 0)
    del lat_chunks
    emb = D.latent_embed(latents.float(), frac=args.crop_frac, pool=args.pool).to(torch.float16)
    print(f"[bank] {len(entries)} entries, latent emb {tuple(emb.shape)} "
          f"({emb.numel() * 2 / 1e6:.0f} MB), latents {latents.numel() * 2 / 1e9:.2f} GB",
          flush=True)

    # ---- cond k-NN over VIDEOS (F1.1.3: negatives that cond cannot discriminate) ----
    id2i = {v["video_id"]: i for i, v in enumerate(vids)}
    cn = Fn.normalize(conds, dim=1)
    sim = cn @ cn.t()
    for i, v in enumerate(vids):                    # never self, never own twin
        sim[i, i] = -2.0
        tw = D.twin_of(v["video_id"], tidx, vids_by_id)
        if tw is not None:
            sim[i, id2i[tw]] = -2.0
    knn = sim.topk(args.knn, dim=1).indices                          # (V, k) video indices

    dino = None
    if args.dino:
        from train_decoder import build_vae                          # frame_dec helper
        os.chdir(REPO)
        device = torch.device("cuda")
        vae = build_vae(device)
        dino = dino_audit_embed(latents, device, vae).to(torch.float16)
        del vae
        torch.cuda.empty_cache()

    bank = {
        "format": "mem2_bank_v1",
        "entries": entries,
        "emb": emb,                                   # (N, Demb) fp16, L2-normalized
        "emb_space": {"kind": "latent_crop", "frac": args.crop_frac, "pool": args.pool,
                      "dim": int(emb.shape[1])},
        "dino_audit": dino,                           # (N, 384) fp16 or None
        "cond": conds.to(torch.float16),              # (V, 4096) pooled T5 of prompt A
        "cond_knn": knn,                              # (V, k) video indices, self+twin excluded
        "videos": vids,                               # the index, verbatim
        "split": split,
        "twin_index": tidx,
        "meta": {"n_per_video": args.n_per_video, "knn": args.knn,
                 "n_val_switch": args.n_val_switch, "n_val_single": args.n_val_single,
                 "created": time.strftime("%Y-%m-%d %H:%M:%S"),
                 "v3_dir": D.V3_DIR, "twin_dir": D.TWIN_DIR},
    }
    tmp = BANK_PT + ".tmp"
    torch.save(bank, tmp)
    os.replace(tmp, BANK_PT)
    print(f"[bank] wrote {BANK_PT} ({os.path.getsize(BANK_PT) / 1e6:.0f} MB)", flush=True)
    if not args.no_latents:
        tmp = BANK_LAT + ".tmp"
        torch.save({"latents": latents, "entries": entries}, tmp)
        os.replace(tmp, BANK_LAT)
        print(f"[bank] wrote {BANK_LAT} ({os.path.getsize(BANK_LAT) / 1e9:.2f} GB)", flush=True)
    return bank


# ----------------------------- selftests --------------------------------------------------------
def selftests(bank, args):
    vids = bank["videos"]
    vids_by_id = {v["video_id"]: v for v in vids}
    id2i = {v["video_id"]: i for i, v in enumerate(vids)}
    ent = bank["entries"]
    split, tidx = bank["split"], bank["twin_index"]
    emb = bank["emb"].float()

    # S-A1 held-out (re-run on the stored index)
    hits, n_held = D.held_out_hits(vids)
    check("S-A1 held-out grep: 0 probe prompt in the corpus", not hits,
          f"{n_held} held-out prompts checked against {len(vids)} videos")

    # S-A2 split per video, >= 20 val videos, both modes present
    val_v = [v for v in vids if split[v["video_id"]] == "val"]
    modes = {m: sum(1 for v in val_v if v["mode"] == m) for m in ("switch", "single")}
    check("S-A2 split: >= 20 VAL videos, both modes", len(val_v) >= 20 and all(modes.values()),
          f"{len(val_v)} val videos ({modes}), {len(vids) - len(val_v)} train")

    # S-A3 twins on the SAME side as their sibling; every twin has a sibling
    bad, orphan = [], []
    for cidx, e in tidx.items():
        if e["twin"] is None:
            continue
        if e["main"] is None:
            orphan.append(cidx)
            continue
        if split[e["twin"]] != split[e["main"]]:
            bad.append(cidx)
    check("S-A3 twin pairing: same split side, no orphan twin", not bad and not orphan,
          f"{sum(1 for e in tidx.values() if e['twin'])} twin pairs, "
          f"{len(bad)} split-crossing, {len(orphan)} orphan")

    # S-A3b the twin really is same prompt / different seed (the F1.1b contract)
    same_prompt, seed_ok = 0, 0
    for cidx, e in tidx.items():
        if e["twin"] and e["main"]:
            a, b = vids_by_id[e["main"]], vids_by_id[e["twin"]]
            same_prompt += int(a["prompt"] == b["prompt"])
            seed_ok += int(abs(b["seed"] - a["seed"]) == D.TWIN_SEED_OFFSET)
    n_tw = sum(1 for e in tidx.values() if e["twin"] and e["main"])
    check("S-A3b twins: identical prompt, seed shifted by exactly +100000",
          n_tw > 0 and same_prompt == n_tw and seed_ok == n_tw,
          f"{same_prompt}/{n_tw} same prompt, {seed_ok}/{n_tw} seed offset ok")

    # S-A4 embeddings: finite, unit norm, no NaN
    nrm = emb.pow(2).sum(1).sqrt()
    check("S-A4 embeddings finite and L2-normalized",
          bool(torch.isfinite(emb).all()) and float((nrm - 1).abs().max()) < 1e-2,
          f"N={emb.shape[0]}, dim={emb.shape[1]}, max|‖e‖-1|={float((nrm - 1).abs().max()):.2e}")

    # S-A5 anchor targets ARE the last frame of A
    anch = [e for e in ent if e["is_anchor"]]
    ok = all(e["frame"] == vids_by_id[e["video_id"]]["switch_frame"] - 1 for e in anch)
    check("S-A5 anchor entries: frame == switch_frame - 1 for every switch video",
          ok and len(anch) == sum(1 for v in vids if v["mode"] == "switch"),
          f"{len(anch)} anchor entries / {sum(1 for v in vids if v['mode'] == 'switch')} switch videos")

    # S-A6 ages covered: the sampled targets span the whole write range (not only recency)
    per_v = {}
    for e in ent:
        per_v.setdefault(e["video_id"], []).append(e["write_j"])
    spans = [max(v) - min(v) for v in per_v.values()]
    check("S-A6 targets spread over the write sequence (ages 1..T reachable)",
          min(spans) >= 20, f"min span {min(spans)} writes, median {sorted(spans)[len(spans) // 2]}")

    # S-A7 cond k-NN excludes self and twin
    knn = bank["cond_knn"]
    bad = 0
    for i, v in enumerate(vids):
        tw = D.twin_of(v["video_id"], tidx, vids_by_id)
        row = set(knn[i].tolist())
        if i in row or (tw is not None and id2i[tw] in row):
            bad += 1
    check("S-A7 cond k-NN: never self, never own twin", bad == 0, f"k={knn.shape[1]}")

    # S-A8 retrieval sanity + the F1.2 question: how hard are the same-prompt (twin) negatives?
    ei = torch.tensor([e["video_i"] for e in ent])
    self_top1 = 0
    twin_conf, n_twin_q = 0, 0
    sims = emb @ emb.t()
    for qi in range(0, emb.shape[0], max(1, emb.shape[0] // 400)):     # ~400 probes, cheap
        row = sims[qi].clone()
        self_top1 += int(row.argmax().item() == qi)
        vidx = ent[qi]["video_i"]
        tw = D.twin_of(ent[qi]["video_id"], tidx, vids_by_id)
        if tw is None:
            continue
        n_twin_q += 1
        tw_i = id2i[tw]
        mask_same = (ei == vidx)
        row[mask_same] = -2.0
        best_other = float(row.max())
        best_twin = float(row[(ei == tw_i)].max()) if int((ei == tw_i).sum()) else -2.0
        twin_conf += int(best_twin >= best_other - 1e-6)
    n_probe = len(range(0, emb.shape[0], max(1, emb.shape[0] // 400)))
    check("S-A8 retrieval sanity: an entry is its own nearest neighbour",
          self_top1 == n_probe, f"{self_top1}/{n_probe}")
    print(f"[bank][diag] same-prompt (twin) negatives are the hardest for "
          f"{twin_conf}/{n_twin_q} probes -> the twin IS the hard negative the InfoNCE must "
          f"be forced to see (red-team F1.1b)", flush=True)

    # S-A9 no train/val leakage of a corpus_idx
    per_cidx = {}
    for v in vids:
        per_cidx.setdefault(v["corpus_idx"], set()).add(split[v["video_id"]])
    check("S-A9 no corpus_idx spans both splits", all(len(s) == 1 for s in per_cidx.values()),
          f"{len(per_cidx)} distinct corpus_idx")


def main():
    ap = argparse.ArgumentParser(description="mem2 InfoNCE bank builder (J2 A)")
    ap.add_argument("--n_per_video", type=int, default=24,
                    help="target frames sampled per video (spread over all ages) + anchor")
    ap.add_argument("--knn", type=int, default=8, help="cond-space nearest videos kept (F1.1.3)")
    ap.add_argument("--n_val_switch", type=int, default=18)
    ap.add_argument("--n_val_single", type=int, default=4)
    ap.add_argument("--crop_frac", type=float, default=0.6)
    ap.add_argument("--pool", type=int, default=2)
    ap.add_argument("--no_latents", action="store_true")
    ap.add_argument("--dino", action="store_true", help="also store DINOv2 ViT-S AUDIT embeddings")
    ap.add_argument("--rebuild_index", action="store_true")
    ap.add_argument("--selftest_only", action="store_true")
    args = ap.parse_args()

    if args.selftest_only:
        bank = torch.load(BANK_PT, map_location="cpu", weights_only=False)
        print(f"[bank] loaded {BANK_PT}: {len(bank['entries'])} entries")
    else:
        bank = build(args)
    selftests(bank, args)
    print(f"\n[bank] {'ALL GREEN' if not _FAILS else f'{len(_FAILS)} FAILURE(S): {_FAILS}'}",
          flush=True)
    sys.exit(0 if not _FAILS else 1)


if __name__ == "__main__":
    main()
