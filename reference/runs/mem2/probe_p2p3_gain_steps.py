#!/usr/bin/env python
# ============================================================================================
# P2/P3 -- WRITE DOSAGE GRID (registre memory2_probes_registry.md): if the write+read channel
# is the bottleneck, does simply writing HARDER (gain alpha on the post-Muon update) or MORE
# OFTEN (n inner steps per evicted chunk) move the decoded anchor?
#
# OFFLINE re-simulation by write-replay, exactly the P1 machinery. The modified write rule is a
# LOCAL MONKEYPATCH of VideoMemoryLayer.write_only inside this script -- wan/modules/
# video_memory.py is NOT touched (mandate). Parity is asserted, not assumed: the (alpha=1, n=1)
# point must reproduce the stock write BITWISE (test S0 below), otherwise the whole grid is void.
#
# The gain is applied AFTER the Muon orthogonalization: Newton-Schulz normalizes its input, so
# scaling the RAW gradient is a strict no-op at muon_update_steps=5. Row-norm capping
# (gated-delta) still applies afterwards, so a large alpha saturates rather than explodes --
# `rowcap_frac` is reported for every grid point.
#
# Usage (GPU pre-check in the SAME command, house rule):
#   CUDA_DEVICE_ORDER=PCI_BUS_ID nvidia-smi --query-gpu=index,memory.free --format=csv && \
#   CUDA_VISIBLE_DEVICES=1 python reference/runs/mem2/probe_p2p3_gain_steps.py
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
from wan.modules import video_memory as VM  # noqa: E402
from wan.modules.video_memory import replay_writes  # noqa: E402
from train_writer import DEFAULT_CONFIG, build_cells, norm_mse_per  # noqa: E402
from probe_p1_block_ablation import (DEC_A2, WRITER_A2, anchor_positions,  # noqa: E402
                                     apply_writer, build_decoder)

OUT_JSON = os.path.join(D.RUNS_DIR, "logs", "p2p3_gain_steps.json")


# ----------------------------- the patched write rule -----------------------------------------
def make_write_only(alpha, n_steps):
    """VideoMemoryLayer.write_only with (i) a GAIN alpha on the post-Muon update and (ii)
    n_steps inner gradient steps per evicted chunk (each step re-linearizes at the current
    fast weight). alpha=1, n_steps=1 == the stock rule, op for op."""
    def write_only(self, x_write, state, allreduce_fn=None, use_checkpoint=None):
        _, k, v, lr0, lr1, lr2, B = self._qkv_lr(x_write)
        decay, init_norms = self._decay_and_norms(B)
        w0, w1, w2 = state
        for _ in range(n_steps):
            g0, g1, g2 = VM.swiglu_fast_weight_gradient(k, v, w0, w1, w2, lr0, lr1, lr2)
            if allreduce_fn is not None:
                g0, g1, g2 = allreduce_fn(g0, g1, g2)
            g0 = VM.zeropower_via_newtonschulz5(g0, self.muon_update_steps)
            g1 = VM.zeropower_via_newtonschulz5(g1, self.muon_update_steps)
            g2 = VM.zeropower_via_newtonschulz5(g2, self.muon_update_steps)
            if alpha != 1.0:
                g0, g1, g2 = alpha * g0, alpha * g1, alpha * g2
            if decay is None:                      # renorm-to-init branch (not our config)
                n0 = w0.detach().norm(dim=1, keepdim=True)
                n1 = w1.detach().norm(dim=1, keepdim=True)
                n2 = w2.detach().norm(dim=1, keepdim=True)
                w0, w1, w2 = w0 + g0, w1 + g1, w2 + g2
                w0 = w0 / (w0.norm(dim=1, keepdim=True) + 1e-5) * n0
                w1 = w1 / (w1.norm(dim=1, keepdim=True) + 1e-5) * n1
                w2 = w2 / (w2.norm(dim=1, keepdim=True) + 1e-5) * n2
            else:
                d0, d1, d2 = decay
                n0, n1, n2 = init_norms
                w0, w1, w2 = d0 * w0 + g0, d1 * w1 + g1, d2 * w2 + g2
                w0 = w0 / (w0.norm(dim=1, keepdim=True) / (n0 + 1e-5)).clamp(min=1.0)
                w1 = w1 / (w1.norm(dim=1, keepdim=True) / (n1 + 1e-5)).clamp(min=1.0)
                w2 = w2 / (w2.norm(dim=1, keepdim=True) / (n2 + 1e-5)).clamp(min=1.0)
        return (w0, w1, w2)
    return write_only


class Patched:
    """Context manager: swap VideoMemoryLayer.write_only for the grid rule, put it back after."""

    def __init__(self, alpha, n_steps):
        self.fn = make_write_only(alpha, n_steps)

    def __enter__(self):
        self.orig = VM.VideoMemoryLayer.write_only
        VM.VideoMemoryLayer.write_only = self.fn
        return self

    def __exit__(self, *a):
        VM.VideoMemoryLayer.write_only = self.orig


def rowcap_frac(cells, blocks, state_by_block):
    """Fraction of fast-weight ROWS pinned at their init norm (gated-delta cap) = saturation."""
    tot, n = 0.0, 0
    for bi in blocks:
        cell = cells[bi]
        for t, w in zip(state_by_block[bi], (cell.w0, cell.w1, cell.w2)):
            r = t.float().norm(dim=1) / w.detach().float().norm(dim=1).clamp_min(1e-9)
            tot += float((r >= 1.0 - 1e-3).float().mean())
            n += 1
    return tot / n


# ----------------------------- main -------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_videos", type=int, default=10)
    ap.add_argument("--k_pos", type=int, default=3)
    ap.add_argument("--alphas", type=float, nargs="+", default=[1, 2, 4, 8])
    ap.add_argument("--steps", type=int, nargs="+", default=[1, 2, 4])
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--config", default=DEFAULT_CONFIG)
    ap.add_argument("--no_wandb", action="store_true")
    args = ap.parse_args()

    os.chdir(REPO)
    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    device = torch.device(args.device)
    torch.set_grad_enabled(False)
    config = OmegaConf.load(args.config)
    config.distributed = False
    cells, blocks, _ = build_cells(config, device, mode="direct")
    for c in cells.values():
        c.eval()
    apply_writer(cells, blocks, WRITER_A2)
    dec = build_decoder(DEC_A2, device)

    vids = D.load_index(verbose=False)
    split = D.assign_split(vids, 18, 4)
    sel = sorted([v for v in vids if split[v["video_id"]] == "val" and v["mode"] == "switch"
                  and D.anchor_write_index(v) is not None], key=lambda v: v["video_id"])
    sel = sel[:args.n_videos]
    print(f"[p2p3] {len(sel)} val switch videos, blocks {blocks}, writer A2 + decoder A2",
          flush=True)
    cache = D.PartCache(size=3)

    # ---- S0: the (alpha=1, n=1) patch must be BITWISE the stock write ------------------------
    meta0 = sel[0]
    pos0 = anchor_positions(meta0, k=1)
    h0 = D.load_h(meta0, cache, blocks, upto=pos0[0]["i"], device=device)
    ref = {bi: replay_writes(cells[bi], h0[bi], detach_state=True, n_bptt_writes=0,
                             use_checkpoint=False) for bi in blocks}
    with Patched(1.0, 1):
        got = {bi: replay_writes(cells[bi], h0[bi], detach_state=True, n_bptt_writes=0,
                                 use_checkpoint=False) for bi in blocks}
    dmax = max(float((a - b).abs().max()) for bi in blocks for a, b in zip(ref[bi], got[bi]))
    print(f"[p2p3][S0] patched (alpha=1, n=1) vs stock write_only: max|diff| {dmax:.3e} -> "
          f"{'PASS (bitwise)' if dmax == 0.0 else 'FAIL -- the grid would be void'}", flush=True)
    assert dmax == 0.0, "P2/P3 parity broken: the monkeypatch is not the stock rule at (1, 1)"
    del h0, ref, got

    # ---- the grid ----------------------------------------------------------------------------
    grid = {}
    per_video = {}
    for meta in sel:
        pos = anchor_positions(meta, k=args.k_pos)
        if not pos:
            continue
        i_max = max(p["i"] for p in pos)
        idxs = [p["i"] for p in pos]
        h = D.load_h(meta, cache, blocks, upto=i_max, device=device)
        ca, _ = D.load_conds(meta, cache)
        cond = ca[None].expand(len(idxs), -1).to(device).float()
        target = D.load_frames(meta, cache, [p["frame"] for p in pos]).to(device).unsqueeze(1)
        base_state = None
        for a in args.alphas:
            for n in args.steps:
                # LOW-MEMORY replay: keep only the K sampled states (segment-wise replay from
                # the running state) instead of all ~35 -- ~2.5 GB instead of ~8, so this probe
                # can co-run with the inference campaign on the same GPU.
                with Patched(float(a), int(n)):
                    st = {}
                    for bi in blocks:
                        state, prev, keep = None, 0, []
                        for i in idxs:
                            state = replay_writes(cells[bi], h[bi][prev:i + 1], state=state,
                                                  detach_state=True, n_bptt_writes=0,
                                                  use_checkpoint=False)
                            keep.append(state)
                            prev = i + 1
                        st[bi] = keep
                dec_states = [tuple(torch.cat([st[bi][t][kk] for t in range(len(idxs))], 0)
                                    for kk in range(3)) for bi in blocks]
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    pred = dec(dec_states, cond=cond)
                loss = float(norm_mse_per(pred, target).mean())
                last = {bi: st[bi][-1] for bi in blocks}
                cap = rowcap_frac(cells, blocks, last)
                if (a, n) == (args.alphas[0], args.steps[0]):
                    base_state = last
                    rel = 0.0
                else:
                    rel = sum(float((last[bi][t].float() - base_state[bi][t].float()).norm()
                                    / base_state[bi][t].float().norm().clamp_min(1e-9))
                              for bi in blocks for t in range(3)) / (3 * len(blocks))
                key = f"a{a:g}_n{n}"
                grid.setdefault(key, {"alpha": float(a), "n": int(n), "loss": [], "cap": [],
                                      "rel": []})
                grid[key]["loss"].append(loss)
                grid[key]["cap"].append(cap)
                grid[key]["rel"].append(rel)
                per_video.setdefault(meta["video_id"], {})[key] = loss
                del st, dec_states, pred
        del h, target
        torch.cuda.empty_cache()
        print(f"[p2p3] {meta['video_id']} done", flush=True)

    print("\n[p2p3] ===== GRID (mean over videos): norm_mse of the decoded anchor =====")
    header = "alpha \\ n |" + "".join(f"   n={n}      " for n in args.steps)
    print("[p2p3] " + header)
    ref_loss = sum(grid[f"a{args.alphas[0]:g}_n{args.steps[0]}"]["loss"]) \
        / len(grid[f"a{args.alphas[0]:g}_n{args.steps[0]}"]["loss"])
    out = {"ref_loss": ref_loss, "grid": {}, "per_video": per_video, "n_videos": len(per_video)}
    for a in args.alphas:
        row = f"[p2p3] a={a:<6g} |"
        for n in args.steps:
            g = grid[f"a{a:g}_n{n}"]
            m = sum(g["loss"]) / len(g["loss"])
            cap = sum(g["cap"]) / len(g["cap"])
            rel = sum(g["rel"]) / len(g["rel"])
            out["grid"][f"a{a:g}_n{n}"] = {"alpha": float(a), "n": int(n), "loss": m,
                                           "dloss": m - ref_loss, "rowcap": cap, "rel_state": rel}
            row += f" {m:.4f}({m - ref_loss:+.3f}) "
        print(row)
    print("[p2p3] rowcap_frac / rel_state vs (1,1):")
    for a in args.alphas:
        print("[p2p3] a=%-6g | " % a + "  ".join(
            f"n={n}: cap {out['grid'][f'a{a:g}_n{n}']['rowcap']:.3f} "
            f"rel {out['grid'][f'a{a:g}_n{n}']['rel_state']:.3f}" for n in args.steps))
    best = min(out["grid"].values(), key=lambda g: g["loss"])
    print(f"[p2p3] best grid point: alpha={best['alpha']:g} n={best['n']} -> {best['loss']:.4f} "
          f"({best['dloss']:+.4f} vs the deployed rule)", flush=True)

    with open(OUT_JSON, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[p2p3] wrote {OUT_JSON}", flush=True)

    if not args.no_wandb:
        import wandb
        wandb.init(project="anchor-forcing-memory", name="mem2_probe_p2p3", dir=D.RUNS_DIR,
                   config=vars(args))
        wandb.log({f"p2p3/{k}/{m}": v for k, g in out["grid"].items()
                   for m, v in g.items()}, step=0)
        wandb.finish()


if __name__ == "__main__":
    main()
