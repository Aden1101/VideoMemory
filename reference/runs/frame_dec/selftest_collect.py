#!/usr/bin/env python
"""frame_dec collection SELFTEST: one short rollout (30 frames = 10 chunks) on GPU.

Checks:
  0. corpus selection runs, held-out exclusion = 0 hit on the 16 probe prompts (CPU, fails
     fast before any GPU work), regex filter yields a sane fraction of the corpus.
  1. saved states are non-None on the 3 memory blocks, fp16 cpu, expected shapes, finite,
     and evolve between chunk positions (the write is live, not a frozen snapshot).
  2. The target frame is the just-evicted latent frame: bitwise equal to
     latents[0, evicted_frame] (fp16 cast), and the model's own eviction-feed debug trace
     (MEM_EVICT_DEBUG) confirms popped == chunk_idx - window_chunks with wrote=True.
  3. a shard .pt written with the collected pairs (single + switch mixed) reloads
     bitwise-identical, format_version == 2.
  4. ttt state written exactly on config.model_kwargs.memory_layers (no more, no less).
  5. v2 single-prompt pairs carry cond_emb (fp16 cpu (4096,), finite, nonzero), mode,
     target_frame == evicted_frame.
  6. SWITCH mode (v1.1): a short A->B rollout; per pair: A totally evicted from the window
     (chunk bookkeeping and the in-rollout fifo assert), target == the last latent frame
     of A (bitwise, fp16), cond_emb == pooled prompt-A embedding (same as single-mode pairs
     of the same prompt), switch bookkeeping coherent, states evolve post-switch.
  7. mode/switch pickers: --mix assignment ~50/50 and deterministic; pick_switch
     deterministic, switch_frame in the configured set, prompt B != prompt A.

Run (Aden, longlive env, GPU visible):
  CUDA_VISIBLE_DEVICES=0 python reference/runs/frame_dec/selftest_collect.py
"""
import os
import sys

REPO = "/shared/ssd_14T/home/adenelmi/Anchor-Forcing"
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(REPO, "reference/runs/reid_probe_l1"))

import torch
from omegaconf import OmegaConf

from collect_pairs import (COND_DIM, DEFAULT_CONFIG, DEFAULT_SWITCH_FRAMES, choose_mode,
                           flush_part, held_out_prompts, pair_nbytes, pick_switch,
                           rollout_collect, select_prompts, shard_header)
from run_reid_probe import build_pipeline, release_vram


def check_cond_emb(pair, tag):
    ce = pair["cond_emb"]
    assert ce.dtype == torch.float16 and ce.device.type == "cpu", \
        f"{tag}: cond_emb dtype/device {ce.dtype}/{ce.device}"
    assert tuple(ce.shape) == (COND_DIM,), f"{tag}: cond_emb shape {tuple(ce.shape)}"
    assert torch.isfinite(ce).all(), f"{tag}: non-finite cond_emb"
    assert ce.abs().max().item() > 0, f"{tag}: cond_emb is all-zeros"


def main():
    frames = 30                      # 10 chunks; eligible collection positions = 5..9
    torch.set_grad_enabled(False)

    # ---- 0. corpus + held-out (CPU, before any GPU work) -----------------------------------
    selected = select_prompts(n=400, corpus_seed=1234)   # asserts 0 held-out hit internally
    held = held_out_prompts()
    assert len(held) == 16, f"expected 16 held-out probe prompts, got {len(held)}"
    print("[selftest] corpus selection OK; first selected prompt:")
    print(f"    [{selected[0][0]}] {selected[0][1][:100]}...")

    # ---- rollout ---------------------------------------------------------------------------
    device = torch.device("cuda")
    config = OmegaConf.load(DEFAULT_CONFIG)
    config.distributed = False
    mem_layers = sorted(int(x) for x in config.model_kwargs.memory_layers)

    pipeline = build_pipeline(config, device)
    test_cidx, test_prompt = selected[0]
    res = rollout_collect(pipeline, config, test_prompt, test_cidx, frames,
                          n_positions=99, device=device,
                          forced_positions=[5, 6, 7, 8, 9], debug_evict=True)
    pairs, latents = res["pairs"], res["latents"]

    # ---- 4. ttt written exactly on the memory blocks ---------------------------------------
    assert res["written"] == mem_layers, \
        f"ttt state written on {res['written']}, expected exactly {mem_layers}"
    print(f"[selftest] ttt state written exactly on {res['written']}")

    # ---- 1. states: non-None (asserted during collection), fp16 cpu, shapes, finite --------
    assert len(pairs) == 5, f"expected 5 pairs (chunks 5..9), got {len(pairs)}"
    exp_shapes = [(1, 1536, 6144), (1, 6144, 1536), (1, 1536, 6144)]
    for pi, pair in enumerate(pairs):
        assert len(pair["states"]) == len(mem_layers), \
            f"pair {pi}: {len(pair['states'])} state tuples, expected {len(mem_layers)}"
        for bi, st in zip(mem_layers, pair["states"]):
            assert isinstance(st, tuple) and len(st) == 3, f"pair {pi} block {bi}: bad tuple"
            for t, es in zip(st, exp_shapes):
                assert t is not None, f"pair {pi} block {bi}: None state tensor"
                assert t.dtype == torch.float16 and t.device.type == "cpu", \
                    f"pair {pi} block {bi}: dtype/device {t.dtype}/{t.device}"
                assert tuple(t.shape) == es, f"pair {pi} block {bi}: shape {tuple(t.shape)} != {es}"
                assert torch.isfinite(t).all(), f"pair {pi} block {bi}: non-finite state values"
    # states must evolve between positions (write live): compare w0 of block 5, pair0 vs pair4
    d_state = (pairs[0]["states"][0][0].float() - pairs[4]["states"][0][0].float()).abs().max().item()
    assert d_state > 0, "state identical across chunk positions, write pass not live"
    print(f"[selftest] states OK on blocks {mem_layers} "
          f"(max |w0(c5) - w0(c9)| = {d_state:.5f} > 0)")

    # ---- 2. target == the just-evicted latent frame ----------------------------------------
    wc = res["window_chunks"]
    for pair in pairs:
        assert pair["evicted_chunk"] == pair["chunk_idx"] - wc
        assert pair["evicted_frame"] == pair["evicted_chunk"] * pair["num_frame_per_block"] \
            + (pair["num_frame_per_block"] - 1)
        ref = latents[0, pair["evicted_frame"]].to(torch.float16)
        assert torch.equal(pair["target"], ref), \
            f"chunk {pair['chunk_idx']}: target != latent frame {pair['evicted_frame']}"
        assert pair["target"].abs().max().item() > 0, "target frame is all-zeros"
    print(f"[selftest] targets == evicted latent frames "
          f"{[p['evicted_frame'] for p in pairs]} (bitwise, fp16)")

    # ---- 5. v2 fields on single-prompt pairs -------------------------------------------------
    for pair in pairs:
        assert pair["mode"] == "single", f"single rollout produced mode {pair['mode']!r}"
        assert pair["target_frame"] == pair["evicted_frame"], \
            "single pair: target_frame must alias evicted_frame"
        check_cond_emb(pair, f"single chunk {pair['chunk_idx']}")
    assert torch.equal(pairs[0]["cond_emb"], pairs[-1]["cond_emb"]), \
        "cond_emb differs within one rollout (must be the SAME pooled prompt embedding)"
    print(f"[selftest] v2 single pairs OK: cond_emb fp16 cpu ({COND_DIM},) finite/nonzero, "
          f"mode='single', target_frame == evicted_frame")

    # model's own eviction trace: popped chunk == chunk_idx - wc, wrote=True past the sink
    dbg = res["evict_dbg"]
    assert dbg, "MEM_EVICT_DEBUG trace empty, eviction_feed not live?"
    by_chunk = {}
    for e in dbg:
        by_chunk.setdefault(e["chunk"], []).append(e)
    for pair in pairs:
        evs = by_chunk.get(pair["chunk_idx"], [])
        assert evs, f"no evict-feed trace for chunk {pair['chunk_idx']}"
        for e in evs:
            assert e["window_chunks"] == wc and e["sink_chunks"] == res["sink_chunks"]
            assert e["popped"] == pair["evicted_chunk"], \
                f"model popped chunk {e['popped']}, collector assumed {pair['evicted_chunk']}"
            assert e["wrote"], f"chunk {pair['chunk_idx']}: popped but NOT written (sink?)"
    print("[selftest] model evict-feed trace agrees: popped == chunk_idx - "
          f"{wc}, wrote=True on all collected positions")

    # ---- 6. SWITCH mode (v1.1 recency-bias attack) -------------------------------------------
    frames_sw = 45                    # 15 chunks; switch at frame 15 -> switch_chunk 5
    switch_frame = 15
    cidx_b, prompt_b = selected[1]
    assert prompt_b != test_prompt
    res_sw = rollout_collect(pipeline, config, test_prompt, test_cidx, frames_sw,
                             n_positions=99, device=device,
                             forced_positions=[10, 12, 14],
                             switch_frame=switch_frame, prompt_b=prompt_b,
                             corpus_idx_b=cidx_b)
    sw_pairs, sw_lat = res_sw["pairs"], res_sw["latents"]
    assert res_sw["mode"] == "switch" and res_sw["switch_chunk"] == 5
    assert res_sw["written"] == mem_layers
    assert len(sw_pairs) == 3, f"expected 3 switch pairs, got {len(sw_pairs)}"
    ref_A = sw_lat[0, switch_frame - 1].to(torch.float16)   # the last latent frame of A
    assert ref_A.abs().max().item() > 0, "last-A reference frame is all-zeros"
    for pair in sw_pairs:
        assert pair["mode"] == "switch"
        assert pair["switch_frame"] == switch_frame and pair["switch_chunk"] == 5
        assert pair["prompt"] == test_prompt and pair["prompt_b"] == prompt_b
        assert pair["corpus_idx_b"] == cidx_b
        assert pair["target_frame"] == switch_frame - 1
        # A totally evicted: the window of chunk c covers chunks [c-wc+1, c], all past A
        # (the collector also asserted this on the model's real fifo during the rollout)
        w_lo = pair["chunk_idx"] - pair["window_chunks"] + 1
        assert w_lo > pair["switch_chunk"], \
            f"chunk {pair['chunk_idx']}: window low chunk {w_lo} <= switch_chunk 5"
        assert torch.equal(pair["target"], ref_A), \
            f"chunk {pair['chunk_idx']}: target != last A frame {switch_frame - 1} (bitwise)"
        check_cond_emb(pair, f"switch chunk {pair['chunk_idx']}")
        # cond is prompt A's pooled embedding: same prompt + same encoder as the single
        # rollout above -> must match its cond_emb (fp16 tolerance for device nondeterminism)
        assert torch.allclose(pair["cond_emb"].float(), pairs[0]["cond_emb"].float(),
                              rtol=0.0, atol=1e-2), \
            "switch cond_emb != pooled prompt-A embedding of the single rollout"
    d_sw = (sw_pairs[0]["states"][0][0].float()
            - sw_pairs[-1]["states"][0][0].float()).abs().max().item()
    assert d_sw > 0, "switch states identical across positions, B writes not live"
    print(f"[selftest] SWITCH OK: A evicted (window > chunk 5) on chunks "
          f"{[p['chunk_idx'] for p in sw_pairs]}, target == last-A frame {switch_frame - 1} "
          f"(bitwise), cond_emb == pooled A, states evolve (max d = {d_sw:.5f})")

    # ---- 7. mode / switch pickers (CPU, deterministic) ---------------------------------------
    modes = [choose_mode(i, False, True, 31337) for i, _ in selected]
    frac = modes.count("switch") / len(modes)
    assert 0.35 <= frac <= 0.65, f"--mix split off 50/50: {frac:.2f} switch"
    assert modes == [choose_mode(i, False, True, 31337) for i, _ in selected], \
        "choose_mode not deterministic"
    assert all(choose_mode(i, True, False, 0) == "switch" for i, _ in selected[:10])
    assert all(choose_mode(i, False, False, 0) == "single" for i, _ in selected[:10])
    pk1 = pick_switch(test_cidx, selected, DEFAULT_SWITCH_FRAMES, 4242)
    pk2 = pick_switch(test_cidx, selected, DEFAULT_SWITCH_FRAMES, 4242)
    assert pk1 == pk2, "pick_switch not deterministic"
    assert pk1[0] in DEFAULT_SWITCH_FRAMES and pk1[1] != test_cidx
    print(f"[selftest] pickers OK: mix split {100 * frac:.1f}% switch (deterministic), "
          f"pick_switch({test_cidx}) = frame {pk1[0]}, B corpus_idx {pk1[1]}")

    # ---- 3. shard save / reload roundtrip (mixed single + switch, format v2) -----------------
    header = shard_header(config, DEFAULT_CONFIG, mem_layers, frames, 5, collect_mode="mix")
    out_dir = os.path.join(HERE, "selftest_out")
    os.makedirs(out_dir, exist_ok=True)
    mixed = pairs + sw_pairs
    part = flush_part(out_dir, "selftest", 0, mixed, header)
    path = os.path.join(out_dir, part["file"])
    back = torch.load(path, map_location="cpu", weights_only=False)
    assert back["format_version"] == 2 and back["memory_layers"] == mem_layers
    assert back["cond_dim"] == COND_DIM and back["cond_key"] == "cond_emb"
    assert len(back["pairs"]) == len(mixed)
    for p0, p1 in zip(mixed, back["pairs"]):
        assert p0["prompt"] == p1["prompt"] and p0["chunk_idx"] == p1["chunk_idx"]
        assert p0["mode"] == p1["mode"] and p0["target_frame"] == p1["target_frame"]
        assert p0.get("prompt_b") == p1.get("prompt_b")
        assert torch.equal(p0["target"], p1["target"])
        assert torch.equal(p0["cond_emb"], p1["cond_emb"])
        for st0, st1 in zip(p0["states"], p1["states"]):
            for t0, t1 in zip(st0, st1):
                assert torch.equal(t0, t1)
    per_pair = sum(pair_nbytes(p) for p in pairs) / len(pairs)
    print(f"[selftest] MIXED shard roundtrip OK, format v2 ({part['bytes'] / 1e9:.2f} GB, "
          f"{per_pair / 1e6:.1f} MB/pair measured -> 400 videos x 8 pos = "
          f"{per_pair * 8 * 400 / 1e9:.0f} GB projected)")
    os.remove(path)

    del pipeline
    release_vram()
    print("[selftest] PASS")


if __name__ == "__main__":
    main()
