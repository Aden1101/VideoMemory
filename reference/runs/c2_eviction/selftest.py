"""c2 eviction-feed + cache-dropout selftest.

Phases:
  --make-ref : (run on the PRE-c2 code) run a short deterministic rollout and save the
               reference outputs/states to selftest_ref.pt. Run it twice (two processes)
               to also validate cross-process bitwise determinism before trusting phase A.
  --full     : (run on the c2 code)
      A. defaults byte-identical: eviction_feed=False + cache_dropout_prob=0 rollout must
         be BITWISE equal to the pre-c2 reference (assert max|diff| == 0).
      B. eviction-feed timing: with eviction_feed=True, the memory's x_write chunk index
         must be exactly window_chunks behind the current chunk, sink chunks never
         written, no write at all during fifo warm-up (state frozen at init).
      C. cache-dropout: prob=1.0 -> runtime sink==dropped value on every self-attn,
         a rollout still runs; restore -> config sink again.
      D. grad/checkpoint smoke: eviction_feed=True + gradient checkpointing + train_write,
         one grad chunk backward()s without error and memory params receive grads.

Config mirrors c1_cotrain model_kwargs (local_attn_size=12, sink_size=3, mem layers
5/16/27, num_frame_per_block=3 => window_chunks=4, sink_chunks=1).

Usage (GPU visible required; `import wan` touches CUDA at import):
  CUDA_VISIBLE_DEVICES=2 <longlive python> reference/runs/c2_eviction/selftest.py --make-ref
  CUDA_VISIBLE_DEVICES=2 <longlive python> reference/runs/c2_eviction/selftest.py --full
"""
import argparse
import os
import sys

REPO = "/shared/ssd_14T/home/adenelmi/Anchor-Forcing"
sys.path.insert(0, REPO)
os.chdir(REPO)

import torch
import torch.distributed as dist

REF_PATH = os.path.join(REPO, "reference/runs/c2_eviction/selftest_ref.pt")

# c1_cotrain model_kwargs (the run c2 inherits from)
MODEL_KW = dict(
    model_name="wan_models/Wan2.1-T2V-1.3B", timestep_shift=5.0, is_causal=True,
    local_attn_size=12, sink_size=3, memory_layers=[5, 16, 27], gate_init=0.1,
    bounded_gate=True, single_norm=False, use_decay=False, muon_update_steps=5,
    ckpt_write=True, gate_extra=[], bounded_mod=True, rel_norm=True, dc_remove=True,
    mem_stream_scale=1.0, ema_alpha=0.0,
)
NUM_FRAME_PER_BLOCK = 3
WINDOW_CHUNKS = 4   # ceil(12 / 3)
SINK_CHUNKS = 1     # ceil(3 / 3)


def build_gen(**extra):
    torch.manual_seed(1234)
    torch.cuda.manual_seed_all(1234)
    from utils.wan_wrapper import WanDiffusionWrapper
    gen = WanDiffusionWrapper(**MODEL_KW, **extra)
    gen = gen.to(device="cuda", dtype=torch.bfloat16)
    gen.eval()
    return gen


def build_pipe(gen, **extra):
    from pipeline.streaming_training import StreamingTrainingPipeline
    steps = torch.tensor([1000, 750, 500, 250], dtype=torch.long)
    pipe = StreamingTrainingPipeline(
        denoising_step_list=steps, scheduler=gen.get_scheduler(), generator=gen,
        num_frame_per_block=NUM_FRAME_PER_BLOCK, same_step_across_blocks=True,
        last_step_only=False, context_noise=0,
        local_attn_size=12, slice_last_frames=27, rope_enhance=True, **extra)
    pipe.scheduler.timesteps = pipe.scheduler.timesteps.to("cuda")  # mirrors model/base.py
    return pipe


def rollout(pipe, n_chunks=8, capture_states=False, grad_last=False):
    """n_chunks sub-blocks of 3 frames each, chunk-by-chunk so per-chunk state is visible."""
    torch.manual_seed(777)
    torch.cuda.manual_seed_all(777)
    cond = {"prompt_embeds": torch.randn(1, 512, 4096, device="cuda", dtype=torch.bfloat16)}
    pipe.kv_cache1 = None
    pipe.ttt_cache = None
    pipe.crossattn_cache = None
    pipe._initialize_kv_cache(1, torch.bfloat16, "cuda")
    pipe._initialize_crossattn_cache(1, torch.bfloat16, "cuda")
    outs, states = [], []
    loss = None
    for c in range(n_chunks):
        noise = torch.randn(1, NUM_FRAME_PER_BLOCK, 16, 60, 104,
                            device="cuda", dtype=torch.bfloat16)
        want_grad = grad_last and c == n_chunks - 1
        if want_grad:
            out, _, _ = pipe.generate_chunk_with_cache(
                noise, cond, current_start_frame=c * NUM_FRAME_PER_BLOCK,
                requires_grad=True, keep_last_write_grad=True)
            loss = out.float().sum()
        else:
            with torch.no_grad():
                out, _, _ = pipe.generate_chunk_with_cache(
                    noise, cond, current_start_frame=c * NUM_FRAME_PER_BLOCK,
                    requires_grad=False)
        outs.append(out.detach())
        if capture_states:
            states.append([
                None if tc.state is None else tuple(t.detach().clone() for t in tc.state)
                for tc in pipe.ttt_cache])
    final_state = {
        i: tuple(t.detach().to(torch.float32).cpu() for t in tc.state)
        for i, tc in enumerate(pipe.ttt_cache) if tc.state is not None}
    return torch.cat(outs, dim=1).detach().cpu(), final_state, states, loss


def run_reference_rollout():
    gen = build_gen()
    pipe = build_pipe(gen)
    outs, final_state, _, _ = rollout(pipe, n_chunks=8)
    del pipe, gen
    torch.cuda.empty_cache()
    return outs, final_state


def phase_a():
    print("\n=== phase A: defaults byte-identical to pre-c2 reference ===")
    assert os.path.exists(REF_PATH), f"missing reference {REF_PATH}; run --make-ref on the pre-c2 code first"
    ref = torch.load(REF_PATH, weights_only=True)
    outs, final_state = run_reference_rollout()
    diff = (outs.float() - ref["outs"].float()).abs().max().item()
    print(f"output latents: shape={tuple(outs.shape)} max|diff| = {diff}")
    assert diff == 0.0, f"BYTE-IDENTITY FAILED on outputs: max|diff|={diff}"
    for i, st in ref["final_state"].items():
        i = int(i)
        assert i in final_state, f"memory state missing for block {i}"
        for a, b in zip(final_state[i], st):
            d = (a - b).abs().max().item()
            assert d == 0.0, f"BYTE-IDENTITY FAILED on block {i} state: max|diff|={d}"
    print(f"memory fast-weight states (blocks {sorted(final_state)}): max|diff| = 0.0")
    print("phase A PASS: eviction_feed=False + cache_dropout_prob=0 is byte-identical")


def phase_b():
    print("\n=== phase B: eviction-feed timing ===")
    os.environ["MEM_EVICT_DEBUG"] = "1"
    gen = build_gen(eviction_feed=True)
    pipe = build_pipe(gen)
    gen.model._evict_dbg = []
    n_chunks = 8
    _, _, states, _ = rollout(pipe, n_chunks=n_chunks, capture_states=True)
    dbg = gen.model._evict_dbg
    os.environ.pop("MEM_EVICT_DEBUG", None)
    mem_blocks = sorted({e["block"] for e in dbg})
    print(f"debug entries: {len(dbg)} over memory blocks {mem_blocks}")
    assert len(dbg) == n_chunks * len(MODEL_KW["memory_layers"]), \
        f"expected one entry per write pass per memory block, got {len(dbg)}"
    for e in dbg:
        assert e["window_chunks"] == WINDOW_CHUNKS and e["sink_chunks"] == SINK_CHUNKS, e
        if e["chunk"] < WINDOW_CHUNKS:                      # fifo warm-up: nothing evicted yet
            assert e["popped"] is None and not e["wrote"], f"early write! {e}"
        else:
            assert e["popped"] == e["chunk"] - WINDOW_CHUNKS, \
                f"eviction lag wrong (expected {WINDOW_CHUNKS} chunks): {e}"
            assert e["wrote"] == (e["popped"] >= SINK_CHUNKS), f"sink gating wrong: {e}"
    lags = sorted({e["chunk"] - e["popped"] for e in dbg if e["popped"] is not None})
    first_write = min(e["chunk"] for e in dbg if e["wrote"])
    n_writes = sum(1 for e in dbg if e["wrote"]) // len(mem_blocks)
    print(f"x_write lag: {lags} chunks (= window_chunks {WINDOW_CHUNKS}); "
          f"first real write at chunk {first_write} (chunk {first_write - WINDOW_CHUNKS} evicted, "
          f"sink chunks 0..{SINK_CHUNKS-1} skipped); {n_writes} writes over {n_chunks} chunks")
    # memory state must stay at init until the first real write, then move
    bi = MODEL_KW["memory_layers"][0]
    for c in range(1, n_chunks):
        same = all(torch.equal(a, b) for a, b in zip(states[c][bi], states[0][bi]))
        expect_same = c < first_write
        assert same == expect_same, \
            f"block {bi} state {'changed' if not same else 'frozen'} at chunk {c}, expected first write at {first_write}"
    print(f"state timeline OK: block {bi} fast-weights frozen at init through chunk "
          f"{first_write-1}, first updated by the chunk-{first_write} write")
    # fifo stays bounded
    for i, tc in enumerate(pipe.ttt_cache):
        assert len(getattr(tc, "fifo", [])) <= WINDOW_CHUNKS, f"fifo unbounded on block {i}"
    print("fifo bounded (<= window_chunks entries per memory block)")
    del pipe, gen
    torch.cuda.empty_cache()
    print("phase B PASS: memory writes exactly the window_chunks-delayed (evicted) chunk")


def sink_values(gen):
    vals = {}
    for name, m in gen.model.named_modules():
        if hasattr(m, "_sink_runtime") and m.__class__.__name__ == "CausalWanSelfAttention":
            vals[name] = int(m._sink_runtime)
    return vals


def phase_c():
    print("\n=== phase C: cache-dropout runtime sink ===")
    gen = build_gen(eviction_feed=True)
    pipe = build_pipe(gen, cache_dropout_prob=1.0, cache_dropout_sink=0)
    v0 = set(sink_values(gen).values())
    assert v0 == {3}, f"initial runtime sink should be config value 3, got {v0}"
    pipe.clear_kv_cache()                      # prob=1.0 -> always drops
    v1 = set(sink_values(gen).values())
    assert pipe._cache_dropout_active and v1 == {0}, \
        f"dropout should set runtime sink 0 on all blocks, got active={pipe._cache_dropout_active} {v1}"
    print(f"dropout rollout: runtime sink {v0} -> {v1} on all "
          f"{len(sink_values(gen))} self-attn modules")
    _ = rollout(pipe, n_chunks=6)              # sink-starved rollout must run cleanly
    print("6-chunk rollout with runtime sink=0 + memory ON: no crash")
    pipe._set_runtime_sink(None)
    v2 = set(sink_values(gen).values())
    assert v2 == {3}, f"restore failed: {v2}"
    print(f"restore: runtime sink back to config value {v2}")
    pipe.cache_dropout_prob = 0.0
    pipe.clear_kv_cache()                      # prob 0 -> must not touch the sink
    assert set(sink_values(gen).values()) == {3}
    print("cache_dropout_prob=0: clear_kv_cache leaves the sink untouched")
    del pipe, gen
    torch.cuda.empty_cache()
    print("phase C PASS")


def phase_d():
    print("\n=== phase D: grad + gradient-checkpointing smoke (eviction_feed=True) ===")
    gen = build_gen(eviction_feed=True)
    gen.model.gradient_checkpointing = True
    for p in gen.model.parameters():
        p.requires_grad_(True)
    pipe = build_pipe(gen, train_write=True)
    _, _, _, loss = rollout(pipe, n_chunks=6, grad_last=True)
    assert loss is not None and loss.requires_grad, "grad chunk built no graph"
    loss.backward()
    bi = MODEL_KW["memory_layers"][0]
    mem = gen.model.blocks[bi].memory
    grads = [p.grad for p in mem.parameters() if p.grad is not None]
    total = sum(g.abs().sum().item() for g in grads)
    print(f"backward OK through checkpointed blocks with mem_x_write kwarg; "
          f"block {bi} memory: {len(grads)} params with grad, sum|grad|={total:.3e}")
    assert len(grads) > 0 and total > 0, "no gradient reached the memory"
    del pipe, gen
    torch.cuda.empty_cache()
    print("phase D PASS")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--make-ref", action="store_true")
    ap.add_argument("--full", action="store_true")
    args = ap.parse_args()

    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29517")
    dist.init_process_group("nccl", rank=0, world_size=1)
    torch.cuda.set_device(0)

    if args.make_ref:
        outs, final_state = run_reference_rollout()
        if os.path.exists(REF_PATH):           # second run = determinism check
            ref = torch.load(REF_PATH, weights_only=True)
            d = (outs.float() - ref["outs"].float()).abs().max().item()
            print(f"determinism check vs existing ref: max|diff| = {d}")
            assert d == 0.0, "cross-process nondeterminism: byte-identity test would be void"
            print("cross-process determinism CONFIRMED (bitwise)")
        else:
            torch.save({"outs": outs, "final_state": final_state}, REF_PATH)
            print(f"reference saved: {REF_PATH} outs={tuple(outs.shape)} "
                  f"state blocks={sorted(final_state)}")
    if args.full:
        phase_a()
        phase_b()
        phase_c()
        phase_d()
        print("\nALL PHASES PASS")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
