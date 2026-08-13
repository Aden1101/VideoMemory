#!/usr/bin/env python
# ============================================================================================
# P1 -- CONTIGUOUS BLOCK ABLATION (registre memory2_probes_registry.md, probe P1). OFFLINE,
# write-replay only: no video model in the loop, no inference hook touched.
#
# QUESTION (pre-registered): how much of the state that the decoder reads at a REVISIT position
# actually comes from the writes of segment A? Protocol, per switch video of the v3 val split:
#
#   base : state = replay(h_0 .. h_i)                                   (the real trajectory)
#   abl  : state = replay(h'_0 .. h'_i) with h'_k = per-token MEAN of h_k for EVERY write k
#          of segment A (k <= anchor_write_index, i.e. every write whose evicted chunk lies
#          before switch_chunk) -- an "uninformative write of the same magnitude", exactly the
#          neutralisation of train_writer.influence_probe, applied to the whole A block instead
#          of one write.
#   ctrl : same indices replaced by the h of ANOTHER video (content swap, not content removal).
#
# MEASURES: ||state_abl - state_base|| / ||state_base|| per memory block, and the norm_mse of
# the DECODED anchor (decoder A2, val convention: unweighted, cond ON) base vs abl vs ctrl.
# Plus a VAE side-by-side PNG (target | base | abl | ctrl) for the eye.
#
# Two (writer, decoder) regimes are measured, both plateaus of the campaign:
#   A2  = mem2.0 phase-A writer + co-trained decoder  (arm memdec20)
#   l1  = stock l1@2000-EMA writer   + decoder v1.2   (arm memdec12)
#
# Usage (GPU pre-check in the SAME command, house rule):
#   CUDA_DEVICE_ORDER=PCI_BUS_ID nvidia-smi --query-gpu=index,memory.free --format=csv && \
#   CUDA_VISIBLE_DEVICES=0 python reference/runs/mem2/probe_p1_block_ablation.py
# ============================================================================================
import argparse
import json
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
from omegaconf import OmegaConf  # noqa: E402

import mem2_data as D  # noqa: E402
from wan.modules.video_memory import AnchorFrameDecoder, replay_writes  # noqa: E402
from train_writer import (DEC_V12, DEFAULT_CONFIG, WRITE_PARAMS, build_cells,  # noqa: E402
                          norm_mse_per)

MEM2_W = "/shared/ssd_14T_project/adenelmi/mem2_runs/writer"
WRITER_A2 = os.path.join(MEM2_W, "mem2_writer_A2_final.pt")
DEC_A2 = os.path.join(MEM2_W, "dec_A2_for_memdec.pt")
OUT_PNG = os.path.join(HERE, "eyepack_probes")
OUT_JSON = os.path.join(D.RUNS_DIR, "logs", "p1_block_ablation.json")


# ----------------------------- writer / decoder overlays ------------------------------------
def apply_writer(cells, blocks, ckpt_path):
    """Overlay a train_writer.py checkpoint's theta_w on the cells (same contract as the
    MEM_WRITER inference hook: ONLY the write parameters, nothing else). Returns (step, backup)
    so the stock weights can be put back for the next regime."""
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    assert "writer" in ck, f"{ckpt_path}: no 'writer' entry"
    backup = {}
    with torch.no_grad():
        for bi in blocks:
            sd = ck["writer"][str(bi)]
            unknown = set(sd) - set(WRITE_PARAMS)
            assert not unknown, f"writer ckpt block {bi} carries non-write params {sorted(unknown)}"
            for name, t in sd.items():
                p = cells[bi].get_parameter(name)
                backup[(bi, name)] = p.detach().clone()
                p.data.copy_(t.to(device=p.device, dtype=p.dtype))
    step = ck.get("step", "?")
    del ck
    return step, backup


def restore_writer(cells, backup):
    with torch.no_grad():
        for (bi, name), t in backup.items():
            cells[bi].get_parameter(name).data.copy_(t)


def build_decoder(ckpt_path, device):
    """Same construction as the MEM_DECODE inference hook (cond dims DETECTED from the
    state_dict, strict load, fp32 on device)."""
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = ck["decoder"] if isinstance(ck, dict) and "decoder" in ck else ck
    if "cond_in.weight" in sd:
        cond_hidden, cond_dim = (int(x) for x in sd["cond_in.weight"].shape)
    else:
        cond_dim, cond_hidden = 0, 512
    dec = AnchorFrameDecoder(cond_dim=cond_dim, cond_hidden=cond_hidden,
                             mixer=any(k.startswith("mixer.") for k in sd))
    dec.load_state_dict(sd, strict=True)
    dec.eval().to(device=device, dtype=torch.float32)
    for p in dec.parameters():
        p.requires_grad_(False)
    print(f"[p1] decoder {os.path.basename(ckpt_path)} (step "
          f"{ck.get('step', '?') if isinstance(ck, dict) else '?'}, cond_dim={cond_dim}, "
          f"{dec.num_params() / 1e6:.1f}M)", flush=True)
    return dec


# ----------------------------- positions ------------------------------------------------------
def anchor_positions(meta, k=3):
    """Up to k REVISIT positions (state index i, anchor target), deterministic: the eligible
    states at the 25/50/75 % quantiles of the post-switch range."""
    aj = D.anchor_write_index(meta)
    elig = [i for i in D.anchor_eligible_states(meta) if i >= aj] if aj is not None else []
    if not elig:
        return []
    if len(elig) <= k:
        sel = elig
    else:
        sel = [elig[int(round(q * (len(elig) - 1)))] for q in (0.25, 0.5, 0.75)][:k]
    frame = D.frame_of_write(meta, aj)
    return [{"i": int(i), "j": int(aj), "age": int(i - aj + 1), "frame": int(frame),
             "cond": "a", "kind": "anchor"} for i in sorted(set(sel))]


def rel_state_delta(sa, sb):
    """mean over the 3 fast-weight tensors of ||a - b|| / ||b||."""
    return sum(float((a.float() - b.float()).norm() / b.float().norm().clamp_min(1e-9))
               for a, b in zip(sa, sb)) / len(sb)


# ----------------------------- one video --------------------------------------------------------
@torch.no_grad()
def run_video(cells, blocks, decoder, meta, cache, positions, device, donor=None):
    """Returns dict of metrics + the decoded frames (base / abl / ctrl) for the PNG."""
    i_max = max(p["i"] for p in positions)
    aj = positions[0]["j"]
    h = D.load_h(meta, cache, blocks, upto=i_max, device=device)
    n_abl = aj + 1                                    # writes 0..aj = every segment-A write

    variants = {"base": None, "abl": "mean"}
    if donor is not None:
        variants["ctrl"] = "donor"
    h_don = None
    if donor is not None:
        h_don = D.load_h(donor, cache, blocks, upto=n_abl - 1, device=device)

    states, rel = {}, {}
    for vname, kind in variants.items():
        st = {}
        for bi in blocks:
            hs = list(h[bi][:i_max + 1])
            if kind == "mean":
                for kidx in range(n_abl):
                    hs[kidx] = hs[kidx].mean(dim=1, keepdim=True).expand_as(hs[kidx]).contiguous()
            elif kind == "donor":
                for kidx in range(n_abl):
                    hs[kidx] = h_don[bi][kidx]
            _, all_st = replay_writes(cells[bi], hs, detach_state=True, n_bptt_writes=0,
                                      use_checkpoint=False, return_all=True)
            st[bi] = all_st
        states[vname] = st
    del h, h_don

    idxs = [p["i"] for p in positions]
    ca, _ = D.load_conds(meta, cache)
    cond = ca[None].expand(len(idxs), -1).to(device).float()
    target = D.load_frames(meta, cache, [p["frame"] for p in positions]).to(device).unsqueeze(1)

    out = {"video_id": meta["video_id"], "mode": meta["mode"],
           "n_writes": len(meta["write_chunks"]), "n_ablated": int(n_abl),
           "switch_chunk": meta["switch_chunk"], "anchor_write": int(aj),
           "positions": [{"i": p["i"], "age": p["age"], "frame": p["frame"]} for p in positions]}
    frames = {}
    for vname in variants:
        dec_states = [tuple(torch.cat([states[vname][bi][i][kk] for i in idxs], 0)
                            for kk in range(3)) for bi in blocks]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            pred = decoder(dec_states, cond=cond)
        per = norm_mse_per(pred, target)              # val convention: unweighted
        out[f"loss_{vname}"] = float(per.mean())
        out[f"loss_{vname}_per"] = [float(x) for x in per]
        frames[vname] = pred.float().cpu()
        if vname != "base":
            # per-BLOCK relative state delta, at the LAST (oldest-anchor) decoded position
            for bi in blocks:
                out[f"rel_{vname}_b{bi}"] = rel_state_delta(states[vname][bi][idxs[-1]],
                                                            states["base"][bi][idxs[-1]])
            out[f"rel_{vname}"] = sum(out[f"rel_{vname}_b{bi}"] for bi in blocks) / len(blocks)
            out[f"dloss_{vname}"] = out[f"loss_{vname}"] - out["loss_base"]
            zb = D.latent_embed(frames["base"], frac=0.6, pool=2)
            zv = D.latent_embed(frames[vname], frac=0.6, pool=2)
            out[f"cos_dec_{vname}"] = float((zb * zv).sum(-1).mean())
    zt = D.latent_embed(target.cpu(), frac=0.6, pool=2)
    out["cos_target_base"] = float((zt * D.latent_embed(frames["base"], frac=0.6, pool=2)).sum(-1).mean())
    del states
    torch.cuda.empty_cache()
    return out, frames, target.cpu()


# ----------------------------- main ---------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_videos", type=int, default=12)
    ap.add_argument("--n_ctrl", type=int, default=4, help="videos that also get the donor control")
    ap.add_argument("--n_png", type=int, default=4)
    ap.add_argument("--k_pos", type=int, default=3)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--config", default=DEFAULT_CONFIG)
    ap.add_argument("--regimes", default="A2,l1")
    ap.add_argument("--no_wandb", action="store_true")
    args = ap.parse_args()

    os.chdir(REPO)
    os.makedirs(OUT_PNG, exist_ok=True)
    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    device = torch.device(args.device)
    torch.set_grad_enabled(False)
    config = OmegaConf.load(args.config)
    config.distributed = False

    cells, blocks, vae = build_cells(config, device, mode="pipeline")
    for c in cells.values():
        c.eval()
    vids = D.load_index(verbose=False)
    split = D.assign_split(vids, 18, 4)
    val_sw = [v for v in vids if split[v["video_id"]] == "val" and v["mode"] == "switch"
              and v["switch_chunk"] is not None and D.anchor_write_index(v) is not None]
    val_sw.sort(key=lambda v: v["video_id"])
    sel = val_sw[:args.n_videos]
    assert len(sel) >= min(10, args.n_videos), \
        f"only {len(sel)} val switch videos with an anchor write (registry asks for >= 10)"
    print(f"[p1] {len(sel)} val switch videos, blocks {blocks}", flush=True)

    cache = D.PartCache(size=3)
    results = {}
    png_payload = {}
    for regime in [r for r in args.regimes.split(",") if r]:
        backup = None
        if regime == "A2":
            step, backup = apply_writer(cells, blocks, WRITER_A2)
            dec = build_decoder(DEC_A2, device)
            print(f"[p1] regime A2: writer {os.path.basename(WRITER_A2)} (step {step}) "
                  f"+ decoder A2", flush=True)
        elif regime == "l1":
            dec = build_decoder(DEC_V12, device)
            print("[p1] regime l1: stock l1@2000-EMA writer + decoder v1.2", flush=True)
        else:
            raise SystemExit(f"unknown regime {regime}")

        rows = []
        for vi, meta in enumerate(sel):
            pos = anchor_positions(meta, k=args.k_pos)
            if not pos:
                print(f"[p1] {meta['video_id']}: no eligible anchor state, skipped", flush=True)
                continue
            donor = None
            if vi < args.n_ctrl:
                need = pos[0]["j"] + 1
                cands = [v for v in sel if v["video_id"] != meta["video_id"]
                         and len(v["write_chunks"]) >= need]
                donor = cands[(vi + 1) % len(cands)] if cands else None
            r, frames, target = run_video(cells, blocks, dec, meta, cache, pos, device, donor)
            r["regime"] = regime
            r["donor"] = donor["video_id"] if donor else None
            rows.append(r)
            print(f"[p1][{regime}] {meta['video_id']:>12} ({r['n_ablated']}/{r['n_writes']} writes "
                  f"ablated, ages {[p['age'] for p in pos]}): loss {r['loss_base']:.4f} -> "
                  f"{r['loss_abl']:.4f} (d {r['dloss_abl']:+.4f}) | rel_state {r['rel_abl']:.3f}"
                  + (f" | ctrl {r['loss_ctrl']:.4f} (d {r['dloss_ctrl']:+.4f}, rel "
                     f"{r['rel_ctrl']:.3f})" if donor else ""), flush=True)
            if regime == "A2" and len(png_payload) < args.n_png and donor is not None:
                png_payload[meta["video_id"]] = (target[0, 0], {k: v[0, 0] for k, v in frames.items()},
                                                 f"{meta['video_id']} {meta['mode']}\n"
                                                 f"age {pos[0]['age']}w, {r['n_ablated']} writes ablated")
        results[regime] = rows

        # ---- aggregate ----
        d = [r["dloss_abl"] for r in rows]
        rl = [r["rel_abl"] for r in rows]
        agg = {"n": len(rows),
               "loss_base": sum(r["loss_base"] for r in rows) / len(rows),
               "loss_abl": sum(r["loss_abl"] for r in rows) / len(rows),
               "dloss_abl_mean": sum(d) / len(d),
               "dloss_abl_median": sorted(d)[len(d) // 2],
               "dloss_abl_min": min(d), "dloss_abl_max": max(d),
               "n_pos": sum(1 for x in d if x > 0),
               "rel_state_mean": sum(rl) / len(rl), "rel_state_min": min(rl),
               "rel_state_max": max(rl)}
        for bi in blocks:
            agg[f"rel_state_b{bi}"] = sum(r[f"rel_abl_b{bi}"] for r in rows) / len(rows)
        ctrl = [r for r in rows if r.get("dloss_ctrl") is not None]
        if ctrl:
            agg["n_ctrl"] = len(ctrl)
            agg["dloss_ctrl_mean"] = sum(r["dloss_ctrl"] for r in ctrl) / len(ctrl)
            agg["rel_ctrl_mean"] = sum(r["rel_ctrl"] for r in ctrl) / len(ctrl)
        results[f"{regime}_agg"] = agg
        print(f"\n[p1][{regime}] ===== AGGREGATE ({agg['n']} videos) =====")
        print(f"[p1][{regime}] ||d state|| / ||state||  : mean {agg['rel_state_mean']:.3f} "
              f"[{agg['rel_state_min']:.3f}, {agg['rel_state_max']:.3f}]  "
              + " ".join(f"b{bi} {agg[f'rel_state_b{bi}']:.3f}" for bi in blocks))
        print(f"[p1][{regime}] norm_mse anchor          : {agg['loss_base']:.4f} -> "
              f"{agg['loss_abl']:.4f}   DELTA {agg['dloss_abl_mean']:+.4f} "
              f"(median {agg['dloss_abl_median']:+.4f}, range [{agg['dloss_abl_min']:+.4f}, "
              f"{agg['dloss_abl_max']:+.4f}], {agg['n_pos']}/{agg['n']} worse)")
        if ctrl:
            print(f"[p1][{regime}] donor control ({agg['n_ctrl']} vids): dloss "
                  f"{agg['dloss_ctrl_mean']:+.4f}, rel_state {agg['rel_ctrl_mean']:.3f}")
        print(f"[p1][{regime}] REGISTRY: red line = dloss < 0.02 (pure-TTT advocate), "
              f"devil's advocate 0.10-0.20 with rel_state 0.35-0.60\n", flush=True)
        if backup is not None:
            restore_writer(cells, backup)
        del dec
        torch.cuda.empty_cache()

    with open(OUT_JSON, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[p1] wrote {OUT_JSON}", flush=True)

    # ---- eye PNG (VAE, native resolution) ----
    if png_payload:
        from make_eyepack import grid, vae_frames
        rows, labels = [], []
        for vid, (tgt, fr, lab) in png_payload.items():
            imgs = vae_frames(vae, torch.stack([tgt, fr["base"], fr["abl"], fr["ctrl"]]), device)
            rows.append(imgs)
            labels.append(lab)
        p = os.path.join(OUT_PNG, "p1_target_base_abl_ctrl.png")
        grid(rows, ["target (true anchor)", "base replay", "A-block ABLATED",
                    "A-block <- other video"], p, row_labels=labels)
        print(f"[p1] eye-pack -> {p}", flush=True)

    if not args.no_wandb:
        import wandb
        wandb.init(project="anchor-forcing-memory", name="mem2_probe_p1",
                   dir=D.RUNS_DIR, config=vars(args))
        wandb.log({f"p1/{r}/{k}": v for r in results if r.endswith("_agg")
                   for k, v in results[r].items() if isinstance(v, (int, float))}, step=0)
        wandb.finish()


if __name__ == "__main__":
    main()
