#!/usr/bin/env python
"""SELFTEST S1, replay-bitwise (memory 2.0, J1). The founding gate of the write-replay plan
(reference/memory2_design.md 3.7 S1: "ECHEC = stop chantier").

Claim under test (design 1.1): under MEM_ZERO_ALL=1 the eviction-feed h are data (pure
functions of the frozen backbone), so replaying the write chain offline with
video_memory.replay_writes() from the init state reproduces the in-situ fast-weight states
exactly, bitwise, not approximately.

What it runs (one GPU, a few minutes, l1@2000-EMA via probe_l1.yaml):
  rollout : one short single-prompt rollout (default 45 frames = 15 chunks, MEM_ZERO_ALL=1)
            through collect_pairs.rollout_collect(v3_capture=True): h kept bf16 in RAM
            (lossless), post-write states snapshotted after every write (3 blocks).
  test 1  : replay_writes (no grad) == in-situ states at every write, torch.equal on
            w0/w1/w2, blocks 5/16/27. The core S1 assert.
  test 2  : replay with grad (checkpoint_write path, l1 has ckpt_write=true) -> final state
            still bitwise-equal (checkpoint forward parity) and a dummy loss on the final
            state backprops non-ZERO grads to qkv/{weight,bias}, lrs/{weight,bias},
            k_norm.weight, decay_logit; q_norm.weight gets no grad (negative control: q is
            computed and discarded by the write path). Also re-runs the forward with
            use_checkpoint=False -> same bits (checkpoint vs direct parity).
  test 3  : truncated BPTT (n_bptt_writes=3): final state bitwise-identical (the forward
            is unchanged, only the graph is cut) + grads still non-zero + grads differ
            from the full-BPTT grads (proof the truncation actually cut the graph).
  test 4  : (unless --skip-roundtrip) the campaign storage path: second identical rollout
            with the fp16 round-trip poke (v3_fp16_roundtrip=True), replay from the stored
            fp16 h cast back to bf16 == the poked in-situ states, bitwise; and the two
            rollouts' latents are byte-identical (the poke provably touches nothing
            observable under MEM_ZERO_ALL=1). Reports the S4 counters (rt_changed,
            subnormals, absmax).

Usage (Aden launches; needs a visible GPU, `import wan` touches CUDA):
  cd reference/runs/frame_dec && python selftest_replay.py            # full run
  python selftest_replay.py --frames 30 --skip-roundtrip              # quicker smoke
Exit code 0 = ALL green; 1 = at least one check failed (chantier gate closed).
"""
import argparse
import os
import sys

REPO = "/shared/ssd_14T/home/adenelmi/Anchor-Forcing"
PROBE_DIR = os.path.join(REPO, "reference/runs/reid_probe_l1")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO)
sys.path.insert(0, PROBE_DIR)
sys.path.insert(0, HERE)

import torch
from omegaconf import OmegaConf

from collect_pairs import rollout_collect, _find_causal_model, DEFAULT_CONFIG  # noqa: E402
from run_reid_probe import build_pipeline  # noqa: E402
from wan.modules.video_memory import replay_writes  # noqa: E402

# any re-identifiable-style prompt works: the selftest checks bitwise mechanics, not quality.
# (Not one of the 16 held-out probe prompts, so no leak whatever this file is reused for.)
PROMPT = "a brown bear with one white ear fishing in a shallow mountain river"

_FAILS = []


def check(name, ok, detail=""):
    tag = "PASS" if ok else "FAIL"
    print(f"[selftest][{tag}] {name}" + (f": {detail}" if detail else ""), flush=True)
    if not ok:
        _FAILS.append(name)
    return ok


def states_equal(st_a, st_b):
    """(w0,w1,w2) tuples -> (bitwise_equal, max_abs_diff). Compared on CPU (GPU->CPU move of
    bf16 is bit-identical)."""
    max_d = 0.0
    eq = True
    for a, b in zip(st_a, st_b):
        a = a.detach().to("cpu")
        b = b.detach().to("cpu")
        if not torch.equal(a, b):
            eq = False
            max_d = max(max_d, float((a.float() - b.float()).abs().max()))
    return eq, max_d


def write_params(layer):
    """The theta_w of the design (3.2): every parameter the write path consumes."""
    ps = {"qkv.weight": layer.qkv.weight, "qkv.bias": layer.qkv.bias,
          "lrs.weight": layer.lrs.weight, "lrs.bias": layer.lrs.bias}
    if hasattr(layer.k_norm, "weight") and layer.k_norm.weight is not None:
        ps["k_norm.weight"] = layer.k_norm.weight
    if hasattr(layer, "decay_logit"):
        ps["decay_logit"] = layer.decay_logit
    return ps


def dummy_loss(state):
    return sum(t.float().pow(2).mean() for t in state)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default=DEFAULT_CONFIG)
    ap.add_argument("--frames", type=int, default=45, help="rollout length (30-45)")
    ap.add_argument("--n-bptt", type=int, default=3, help="truncation depth for test 3")
    ap.add_argument("--skip-roundtrip", action="store_true",
                    help="skip test 4 (second rollout with the fp16 storage poke)")
    args = ap.parse_args()

    # bitwise-parity preconditions: the in-situ rollout and the replay must see the same
    # backend numerics. Inference processes never touch these (only trainer/distillation.py
    # sets TF32, and it is not imported here); assert so a future import cannot silently
    # change the fp32 lrs matmul.
    assert not torch.backends.cuda.matmul.allow_tf32, \
        "TF32 is ON: the fp32 lrs matmul would diverge from the deployment function"
    assert torch.cuda.is_available(), "selftest needs a GPU (wan/pipeline are CUDA-bound)"
    device = torch.device("cuda")

    config = OmegaConf.load(args.config)
    config.distributed = False
    mem_layers = sorted(int(x) for x in config.model_kwargs.memory_layers)
    F = int(config.num_frame_per_block)
    assert args.frames % F == 0, f"--frames {args.frames} not a multiple of F={F}"

    print(f"[selftest] building pipeline from {args.config} (l1@2000-EMA)", flush=True)
    pipeline = build_pipeline(config, device)
    _cm = _find_causal_model(pipeline.generator.model)
    assert _cm is not None, "CausalWanModel not found under the peft wrappers"
    layers = {bi: _cm.blocks[bi].memory for bi in mem_layers}
    for bi, layer in layers.items():
        assert layer is not None, f"block {bi} has no memory cell"
        assert layer.qkv.weight.dtype == torch.bfloat16, \
            f"block {bi} memory not in bf16 (deployment dtype)"

    # ---- rollout 1: MEM_ZERO_ALL=1, h kept lossless bf16, state snapshot at every write ----
    print(f"[selftest] rollout 1/{1 if args.skip_roundtrip else 2}: {args.frames} frames, "
          f"h bf16 in RAM, snap every write", flush=True)
    res = rollout_collect(pipeline, config, PROMPT, 0, args.frames, 0, device,
                          v3_capture=True, v3_fp16_roundtrip=False, v3_state_snap_every=1)
    assert os.environ.get("MEM_ZERO_ALL") == "1", "rollout did not run under MEM_ZERO_ALL=1"
    wc, sc = res["window_chunks"], res["sink_chunks"]
    expected = [c for c in range(args.frames // F) if c >= wc + sc]
    check("structure: write_chunks", res["write_chunks"] == expected,
          f"{res['write_chunks']} (wc={wc}, sc={sc})")
    n_writes = len(expected)
    for bi in mem_layers:
        check(f"structure: block {bi} h/snap counts",
              len(res["h_seq"][bi]) == n_writes
              and len(res["check_states"][str(bi)]) == n_writes,
              f"{len(res['h_seq'][bi])} h, {len(res['check_states'][str(bi)])} snaps, "
              f"{n_writes} writes")

    # ================= test 1: S1 replay-bitwise (no grad), every write, 3 blocks ==========
    for bi in mem_layers:
        layer = layers[bi]
        h_dev = [h.to(device) for h in res["h_seq"][bi]]          # cpu bf16 -> cuda, bitwise
        with torch.no_grad():
            _, all_states = replay_writes(layer, h_dev, detach_state=False, return_all=True)
        ok, first_bad, max_d = True, -1, 0.0
        for i, ((chk, st_ref), st_rep) in enumerate(zip(res["check_states"][str(bi)],
                                                        all_states)):
            eq, d = states_equal(st_ref, st_rep)
            if not eq:
                ok, first_bad, max_d = False, i, d
                break
        check(f"test1 S1 replay-bitwise block {bi}", ok,
              f"{n_writes} writes all torch.equal" if ok
              else f"FIRST divergence at write {first_bad} (chunk "
                   f"{res['check_states'][str(bi)][first_bad][0]}), max|d|={max_d:.3e}")

    # ================= test 2: gradients + checkpoint forward parity ========================
    ref_final = {bi: res["check_states"][str(bi)][-1][1] for bi in mem_layers}
    full_grads = {}
    for bi in mem_layers:
        layer = layers[bi]
        h_dev = [h.to(device) for h in res["h_seq"][bi]]
        wp = write_params(layer)
        for p in layer.parameters():
            p.requires_grad_(False)
        for p in wp.values():
            p.requires_grad_(True)
            p.grad = None
        # negative control: q_norm participates only in the (discarded) read query
        if hasattr(layer.q_norm, "weight") and layer.q_norm.weight is not None:
            layer.q_norm.weight.requires_grad_(True)
            layer.q_norm.weight.grad = None

        final = replay_writes(layer, h_dev, detach_state=False)   # grad ON, ckpt_write path
        eq, d = states_equal(ref_final[bi], final)
        check(f"test2 grad-mode forward bitwise block {bi}", eq,
              "" if eq else f"max|d|={d:.3e} (checkpointed grad forward != in-situ)")
        with torch.enable_grad():                                  # explicit direct path
            final_nc = replay_writes(layer, h_dev, detach_state=False, use_checkpoint=False)
        eq_nc, d_nc = states_equal(ref_final[bi], final_nc)
        check(f"test2 no-checkpoint forward bitwise block {bi}", eq_nc,
              "" if eq_nc else f"max|d|={d_nc:.3e} (direct grad forward != in-situ)")
        del final_nc

        dummy_loss(final).backward()
        grads_ok, detail = True, []
        for name, p in wp.items():
            g = p.grad
            bad = g is None or not torch.isfinite(g).all() or float(g.abs().sum()) == 0.0
            if bad:
                grads_ok = False
            detail.append(f"{name}={'NULL' if g is None else f'{float(g.abs().sum()):.3e}'}")
        check(f"test2 non-zero write-path grads block {bi}", grads_ok, ", ".join(detail))
        if hasattr(layer.q_norm, "weight") and layer.q_norm.weight is not None:
            qg = layer.q_norm.weight.grad
            check(f"test2 q_norm gets NO grad block {bi}",
                  qg is None or float(qg.abs().sum()) == 0.0,
                  "q is discarded by the write path")
        full_grads[bi] = {n: p.grad.detach().clone() for n, p in wp.items()
                          if p.grad is not None}
        for p in wp.values():
            p.grad = None
        del final
        torch.cuda.empty_cache()

    # ================= test 3: truncated BPTT (n_bptt_writes) ==============================
    for bi in mem_layers:
        layer = layers[bi]
        h_dev = [h.to(device) for h in res["h_seq"][bi]]
        wp = write_params(layer)
        for p in wp.values():
            p.grad = None
        final_t = replay_writes(layer, h_dev, detach_state=False,
                                n_bptt_writes=args.n_bptt)
        eq, d = states_equal(ref_final[bi], final_t)
        check(f"test3 truncated(n={args.n_bptt}) forward bitwise block {bi}", eq,
              "forward unchanged, only the graph is cut" if eq else f"max|d|={d:.3e}")
        dummy_loss(final_t).backward()
        grads_ok = all(p.grad is not None and torch.isfinite(p.grad).all()
                       and float(p.grad.abs().sum()) > 0.0 for p in wp.values())
        check(f"test3 truncated grads non-zero block {bi}", grads_ok)
        differs = any(n in full_grads[bi]
                      and not torch.equal(p.grad, full_grads[bi][n])
                      for n, p in wp.items() if p.grad is not None)
        check(f"test3 truncated grads != full-BPTT grads block {bi}", differs,
              "the truncation actually cut the graph")
        for p in wp.values():
            p.grad = None
        del final_t
        torch.cuda.empty_cache()

    # ================= test 4: campaign fp16 storage path (round-trip poke) ================
    if not args.skip_roundtrip:
        print("[selftest] rollout 2/2: identical seed, fp16 round-trip poke ON", flush=True)
        res2 = rollout_collect(pipeline, config, PROMPT, 0, args.frames, 0, device,
                               v3_capture=True, v3_fp16_roundtrip=True,
                               v3_state_snap_every=1)
        check("test4 latents byte-identical (poke invisible under MEM_ZERO_ALL=1)",
              torch.equal(res["latents"], res2["latents"]))
        print(f"[selftest] S4 counters: rt_changed={res2['h_rt_changed']} / "
              f"{res2['h_elems']} elems, subnormals={res2['h_subnormals']}, "
              f"absmax={res2['h_absmax']:.2f}", flush=True)
        for bi in mem_layers:
            layer = layers[bi]
            # the trainer's exact dequant path: stored fp16 -> bf16, then replay
            h_dev = [h.to(device).to(torch.bfloat16) for h in res2["h_seq"][bi]]
            with torch.no_grad():
                _, all_states = replay_writes(layer, h_dev, detach_state=False,
                                              return_all=True)
            ok, first_bad, max_d = True, -1, 0.0
            for i, ((chk, st_ref), st_rep) in enumerate(zip(res2["check_states"][str(bi)],
                                                            all_states)):
                eq, d = states_equal(st_ref, st_rep)
                if not eq:
                    ok, first_bad, max_d = False, i, d
                    break
            check(f"test4 replay-from-fp16-storage bitwise block {bi}", ok,
                  "" if ok else f"FIRST divergence at write {first_bad}, max|d|={max_d:.3e}")

    print("\n[selftest] " + ("ALL GREEN, the write-replay gate S1 is OPEN"
                             if not _FAILS else
                             f"{len(_FAILS)} FAILURE(S): {_FAILS}, STOP CHANTIER (design "
                             f"3.7 S1), debug before anything else"), flush=True)
    sys.exit(0 if not _FAILS else 1)


if __name__ == "__main__":
    main()
