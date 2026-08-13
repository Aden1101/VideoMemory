#!/usr/bin/env python
# ============================================================================================
# MEMORY 2.0 -- PHASE B selftests (deliverable 4). Run BEFORE the B1/B2 launches.
#
#  1  mixer init == IDENTITY, bitwise: AnchorFrameDecoder(mixer=True) warm-started from a
#     mixer-less state_dict returns EXACTLY the v1 frame (torch.equal, not allclose).
#  2  A2 warm start: the ONLY missing keys are `mixer.*`; every other tensor is bitwise the
#     checkpoint's (a silently re-initialized fuse/head would look like "the mixer helps").
#  3  mixer structure: last pointwise zero-init, parameter budget under 40M.
#  4  DINO bank: shapes / L2 / row alignment / twins on the same split side / anchors covered.
#  5  states are DATA: replay with n_bptt=0 is BITWISE the full-BPTT replay (this is what lets
#     phase B cut the graph at the writer -- it replaces the disk state cache, which is not an
#     option at 170 MB per position).
#  6  the phase-B loss is DIFFERENTIABLE through ONE VAE frame: real states -> decoder v2 ->
#     VAE decode -> crop60 -> DINO -> InfoNCE, and the gradient comes back non-zero on the
#     decoder (head, mixer, queries) -- and on the WRITER when the graph is kept.
#
# Usage:  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 \
#         python reference/runs/mem2/selftest_phaseB.py            (--cpu_only to skip 5/6)
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
from omegaconf import OmegaConf  # noqa: E402

import mem2_data as D  # noqa: E402
from wan.modules.video_memory import AnchorFrameDecoder, replay_writes  # noqa: E402

FAILS = []


def check(name, ok, detail=""):
    print(f"[phaseB][{'PASS' if ok else 'FAIL'}] {name}" + (f" -- {detail}" if detail else ""),
          flush=True)
    if not ok:
        FAILS.append(name)
    return ok


def rand_states(n_blocks=3, B=2, dim=1536, dh=6144, device="cpu", dtype=torch.float32, seed=0):
    g = torch.Generator(device="cpu").manual_seed(seed)
    st = []
    for _ in range(n_blocks):
        w0 = (torch.randn(B, dim, dh, generator=g) * 0.02).to(device=device, dtype=dtype)
        w1 = (torch.randn(B, dh, dim, generator=g) * 0.02).to(device=device, dtype=dtype)
        w2 = (torch.randn(B, dim, dh, generator=g) * 0.02).to(device=device, dtype=dtype)
        st.append((w0, w1, w2))
    return st


# ---------------------------------------------------------------------------------------------
def t1_mixer_identity(device="cpu"):
    torch.manual_seed(0)
    d1 = AnchorFrameDecoder(cond_dim=4096, cond_hidden=512, detach_states=False).to(device)
    torch.manual_seed(0)
    d2 = AnchorFrameDecoder(cond_dim=4096, cond_hidden=512, detach_states=False,
                            mixer=True).to(device)
    missing, unexpected = d2.load_state_dict(d1.state_dict(), strict=False)
    check("v1 state_dict loads into the v2 decoder, missing keys are ONLY the mixer",
          not unexpected and missing and all(k.startswith("mixer.") for k in missing),
          f"{len(missing)} missing (mixer), {len(unexpected)} unexpected")
    # a zero-init head returns a null frame whatever the mixer does -> perturb the head first,
    # otherwise the test would pass on a broken mixer
    with torch.no_grad():
        for d in (d1, d2):
            torch.manual_seed(7)
            d.head.weight.normal_(0, 0.02)
            d.head.bias.normal_(0, 0.02)
    st = rand_states(device=device)
    cond = torch.randn(2, 4096, device=device)
    d1.eval(), d2.eval()
    with torch.no_grad():
        y1, y2 = d1(st, cond=cond), d2(st, cond=cond)
    check("mixer at init == IDENTITY (decoder v2 output BITWISE equal to v1)",
          torch.equal(y1, y2),
          f"max |diff| {float((y1 - y2).abs().max()):.3e}, out {tuple(y2.shape)}")
    # and it is NOT trivially identity once trained: perturbing pw2 must change the frame
    with torch.no_grad():
        d2.mixer[0].pw2.weight.normal_(0, 0.02)
    with torch.no_grad():
        y3 = d2(st, cond=cond)
    check("the mixer is live (a non-zero pw2 changes the frame)", not torch.equal(y1, y3),
          f"max |diff| {float((y1 - y3).abs().max()):.3e}")
    return d1, d2


def t2_a2_warm_start(a2_path, device="cpu"):
    if not os.path.exists(a2_path):
        return check("A2 checkpoint present", False, a2_path)
    ck = torch.load(a2_path, map_location="cpu", weights_only=False)
    sd = ck["decoder"]
    dec = AnchorFrameDecoder(cond_dim=4096, cond_hidden=512, detach_states=False,
                             mixer=True).to(device)
    missing, unexpected = dec.load_state_dict(sd, strict=False)
    check("A2 decoder warm start: no unexpected key, missing keys are ONLY the mixer",
          not unexpected and all(k.startswith("mixer.") for k in missing),
          f"missing {len(missing)} / unexpected {len(unexpected)}")
    got = dict(dec.named_parameters())
    got.update(dict(dec.named_buffers()))
    diffs = {k: float((got[k].detach().cpu().float() - v.float()).abs().max())
             for k, v in sd.items() if k in got}
    worst = max(diffs.values()) if diffs else -1.0
    check("every NON-mixer tensor is BITWISE the A2 checkpoint",
          len(diffs) == len(sd) and worst == 0.0,
          f"{len(diffs)}/{len(sd)} tensors, max |delta| {worst:.3e}")
    # the writer half of the same ckpt (what phase B freezes)
    check("A2 ckpt carries the 3 writer blocks",
          set(ck["writer"].keys()) == {"5", "16", "27"}, str(sorted(ck["writer"].keys())))
    return dec


def t3_mixer_structure(d2):
    zeros = all(float(b.pw2.weight.abs().sum()) == 0.0 and float(b.pw2.bias.abs().sum()) == 0.0
                for b in d2.mixer) if hasattr(d2, "mixer") else False
    # (t1 perturbs mixer[0].pw2 at the end, so re-instantiate a clean one for this check)
    fresh = AnchorFrameDecoder(cond_dim=4096, cond_hidden=512, mixer=True)
    v1 = AnchorFrameDecoder(cond_dim=4096, cond_hidden=512)
    zeros = all(float(b.pw2.weight.abs().sum()) == 0.0 and float(b.pw2.bias.abs().sum()) == 0.0
                for b in fresh.mixer)
    check("mixer last pointwise is ZERO-INIT (weight and bias)", zeros)
    check("decoder v2 under the 40M budget",
          fresh.num_params() < 40e6,
          f"{fresh.num_params() / 1e6:.2f}M (v1 {v1.num_params() / 1e6:.2f}M, "
          f"mixer +{(fresh.num_params() - v1.num_params()) / 1e6:.2f}M)")
    check("the mixer mixes SPATIALLY (depthwise 3x3 on the 30x52 grid)",
          all(b.dw.kernel_size == (3, 3) and b.dw.groups == b.dw.in_channels
              for b in fresh.mixer))


def t4_dino_bank():
    from build_bank import BANK_PT
    from build_bank_dino import BANK_DINO
    if not os.path.exists(BANK_DINO):
        return check("DINO bank present", False, BANK_DINO)
    bank = torch.load(BANK_PT, map_location="cpu", weights_only=False)
    d = torch.load(BANK_DINO, map_location="cpu", weights_only=False)
    emb, entries, split = d["emb"].float(), bank["entries"], bank["split"]
    check("DINO bank shape (N entries, 768) and L2-normalized",
          emb.shape == (len(entries), 768)
          and bool(((emb.norm(dim=1) - 1).abs() < 2e-3).all()),
          f"{tuple(emb.shape)}, {len(entries)} entries")
    tidx = D.twin_index(bank["videos"])
    pairs = [(e["main"], e["twin"]) for e in tidx.values() if e["main"] and e["twin"]]
    check("twins paired by corpus_idx AND on the same side of the split",
          bool(pairs) and all(split[a] == split[b] for a, b in pairs),
          f"{len(pairs)} pairs")
    # the trainer's (video_id, frame) -> row map must cover every switch anchor
    key2row = {(e["video_id"], int(e["frame"])): r for r, e in enumerate(entries)}
    anchors = [e for e in entries if e["is_anchor"]]
    check("every anchor entry is addressable by (video_id, frame)",
          all((e["video_id"], int(e["frame"])) in key2row for e in anchors),
          f"{len(anchors)} anchors, {len(key2row)} keys")
    val_rows = [r for r, e in enumerate(entries) if e["split"] == "val"]
    check("val side of the DINO bank is non-trivial (negatives exist)",
          len(val_rows) > 300, f"{len(val_rows)} val rows / {len(entries)}")
    return d


# ---------------------------------------------------------------------------------------------
def t5_t6_gpu(args):
    device = torch.device(args.device)
    os.chdir(REPO)
    from train_writer import build_cells, WRITE_PARAMS
    config = OmegaConf.load(args.config)
    config.distributed = False
    cells, blocks, _ = build_cells(config, device, "direct")

    # phase-B writer = A2
    ck = torch.load(args.writer_ckpt, map_location="cpu", weights_only=False)
    with torch.no_grad():
        for bi in blocks:
            for name in WRITE_PARAMS:
                p = cells[bi].get_parameter(name)
                p.copy_(ck["writer"][str(bi)][name].to(device=p.device, dtype=p.dtype))
    for bi in blocks:
        for p in cells[bi].parameters():
            p.requires_grad_(False)
        for name in WRITE_PARAMS:
            cells[bi].get_parameter(name).requires_grad_(True)

    vids = D.load_index(verbose=False)
    cache = D.PartCache(size=1)
    meta = next(v for v in vids if v["mode"] == "switch")
    upto = min(9, len(meta["write_chunks"]) - 1)
    h = D.load_h(meta, cache, blocks, upto=upto, device=device)

    # ---- 5: n_bptt=0 is BITWISE the full-BPTT replay (states are data) ----------------------
    same, mx = True, 0.0
    for bi in blocks:
        st_full = replay_writes(cells[bi], h[bi], detach_state=False, n_bptt_writes=None,
                                use_checkpoint=True)
        st_cut = replay_writes(cells[bi], h[bi], detach_state=False, n_bptt_writes=0,
                               use_checkpoint=True)
        for a, b in zip(st_full, st_cut):
            same &= torch.equal(a.detach(), b.detach())
            mx = max(mx, float((a.detach().float() - b.detach().float()).abs().max()))
        del st_full
    check("replay n_bptt=0 == full BPTT, BITWISE (phase B may cut the writer graph for free)",
          same, f"max |delta| {mx:.3e} over {len(blocks)} blocks")

    # ---- 6: differentiable VAE(1 frame) -> DINO, gradient back to the decoder ---------------
    states_g = []
    for bi in blocks:
        st = replay_writes(cells[bi], h[bi], detach_state=False, n_bptt_writes=2,
                           use_checkpoint=True)
        states_g.append(tuple(t for t in st))
    dec = AnchorFrameDecoder(cond_dim=4096, cond_hidden=512, detach_states=False,
                             mixer=True).to(device)
    missing, unexpected = dec.load_state_dict(ck["decoder"], strict=False)
    assert not unexpected and all(k.startswith("mixer.") for k in missing)
    with torch.no_grad():                       # break the mixer's identity so its grad is real
        for b in dec.mixer:
            b.pw2.weight.normal_(0, 1e-3)
    ca, _ = D.load_conds(meta, cache)
    cond = ca[None].to(device)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        pred = dec(states_g, cond=cond)
    check("decoded frame shape", tuple(pred.shape[1:]) == (1, 16, 60, 104), str(tuple(pred.shape)))

    from build_bank_dino import latent_to_pixels, pixels_to_dino_input, BANK_DINO
    from train_decoder import build_vae
    from run_reid_probe import get_dino
    vae = build_vae(device)
    dino = get_dino(device)
    torch.cuda.reset_peak_memory_stats()
    px = latent_to_pixels(vae, pred.to(torch.bfloat16), use_checkpoint=bool(args.vae_ckpt))
    check("VAE decodes ONE latent frame to ONE 480x832 pixel frame in [0,1]",
          tuple(px.shape[1:]) == (3, 480, 832) and float(px.min()) >= 0.0
          and float(px.max()) <= 1.0, f"{tuple(px.shape)} range "
          f"[{float(px.min()):.3f}, {float(px.max()):.3f}]")
    x = pixels_to_dino_input(px, crop_frac=0.6, size=224)
    z = Fn.normalize(dino(x), dim=-1)
    dbank = torch.load(BANK_DINO, map_location="cpu", weights_only=False)["emb"]
    tgt = dbank[:64].to(device=device, dtype=torch.float32)
    loss = -(z @ tgt.t()).mean()
    loss.backward()
    peak = torch.cuda.max_memory_allocated() / 1e9

    def gnorm(pred_fn):
        return sum(float(p.grad.abs().sum()) for n, p in dec.named_parameters()
                   if pred_fn(n) and p.grad is not None)
    g_head = gnorm(lambda n: n.startswith("head"))
    g_mix = gnorm(lambda n: n.startswith("mixer"))
    g_q = gnorm(lambda n: n.startswith("queries"))
    g_w = sum(float(cells[bi].get_parameter("qkv.weight").grad.abs().sum())
              for bi in blocks if cells[bi].get_parameter("qkv.weight").grad is not None)
    check("gradient flows through the VAE+DINO back to the decoder HEAD", g_head > 0,
          f"|grad|_1 = {g_head:.3e}")
    check("gradient flows back to the MIXER", g_mix > 0, f"|grad|_1 = {g_mix:.3e}")
    check("gradient flows back to the decoder QUERIES (the state readout)", g_q > 0,
          f"|grad|_1 = {g_q:.3e}")
    check("gradient even reaches the WRITER when the graph is kept (n_bptt=2)", g_w > 0,
          f"|grad|_1 = {g_w:.3e}")
    check("all decoder gradients finite",
          all(torch.isfinite(p.grad).all() for p in dec.parameters() if p.grad is not None))
    print(f"[phaseB] peak VRAM of the DINO backward alone (K=1, vae_ckpt={args.vae_ckpt}): "
          f"{peak:.1f} GB", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(PROBE_DIR, "probe_l1.yaml"))
    ap.add_argument("--writer_ckpt",
                    default="/shared/ssd_14T_project/adenelmi/mem2_runs/writer/"
                            "mem2_writer_A2_final.pt")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--vae_ckpt", type=int, default=1)
    ap.add_argument("--cpu_only", action="store_true")
    args = ap.parse_args()

    print("=== PHASE B SELFTESTS ===", flush=True)
    _, d2 = t1_mixer_identity()
    t2_a2_warm_start(args.writer_ckpt)
    t3_mixer_structure(d2)
    t4_dino_bank()
    if not args.cpu_only:
        t5_t6_gpu(args)
    print(f"\n[phaseB] {'ALL PASS' if not FAILS else 'FAILURES: ' + ', '.join(FAILS)}",
          flush=True)
    sys.exit(0 if not FAILS else 1)


if __name__ == "__main__":
    main()
