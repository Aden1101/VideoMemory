"""c3 stability-pack selftest.

Verifies (1) with EVERY c3 mechanism at its default-off value the model FORWARD is byte-identical
to the pre-c3 reference, and (2) each new mechanism actually FIRES when enabled.

c3 mechanisms:
  A. mem_stream_scale (model_kwargs) : hard-cap injection to a fraction of the local stream RMS.
  B. use_decay        (model_kwargs) : gated-delta forget on the fast weight (adds decay_logit param).
  C. LoRA-EMA (utils/distributed.py EMA_FSDP, trainable-only) : damps the weight trajectory.
  D. cosine lr decay  (trainer _lr_sched_step)   : anneal both gen groups after lr_decay_start.
  E. mem-grad clip    (trainer _clip_mem_grads)  : clip the FSDP-ignored memory grads.
  F. cache_dropout anneal (pipeline _effective_cache_dropout_prob) : prob start -> end over N steps.

Phases:
  P0  forward byte-identity : all c3 flags OFF (mem_stream_scale=1.0, use_decay=False,
      eviction_feed=False, cache_dropout=0) -> forward bitwise-equal to the c2/pre-c3 ref.
  P1  cache_dropout anneal  : default (end=None) constant == start; anneal on -> prob(step) interpolates.
  P2  lr cosine schedule    : default (decay_start>max_iters) leaves lr; on -> lr(step) follows cosine,
      mem group uses its own floor.
  P3  mem-grad clip         : default (max_grad_norm_mem~1e9) leaves grad; on -> grad norm capped.
  P4  LoRA-EMA trainable-only: shadow excludes frozen params; ema.update moves the shadow (diverges).
  P5  use_decay fresh-init  : use_decay=True builds a decay_logit ~ sigmoid^-1(0.95) per mem layer;
      use_decay=False builds none.
  P6  mem_stream_scale fires: identical nonzero injection, mem_stream_scale 0.5 vs 1.0 -> outputs differ.

Usage (GPU visible required; `import wan` touches CUDA at import):
  CUDA_VISIBLE_DEVICES=<free-gpu> <longlive python> reference/runs/c3_stable/selftest.py --full
The c2 reference (reference/runs/c2_eviction/selftest_ref.pt) is reused for P0.
"""
import argparse
import importlib.util
import math
import os
import sys
from types import SimpleNamespace

REPO = "/shared/ssd_14T/home/adenelmi/Anchor-Forcing"
sys.path.insert(0, REPO)
os.chdir(REPO)

import torch
import torch.nn as nn
import torch.distributed as dist

C2_REF = os.path.join(REPO, "reference/runs/c2_eviction/selftest_ref.pt")


def build_gen_override(c2, **over):
    """Like c2.build_gen but OVERRIDES keys already in MODEL_KW (build_gen passes them as **extra,
    which collides). Same deterministic seed + dtype/device."""
    from utils.wan_wrapper import WanDiffusionWrapper
    kw = {**c2.MODEL_KW, **over}
    torch.manual_seed(1234)
    torch.cuda.manual_seed_all(1234)
    gen = WanDiffusionWrapper(**kw)
    gen = gen.to(device="cuda", dtype=torch.bfloat16)
    gen.eval()
    return gen


def _load_c2_selftest():
    """Import the c2 selftest module to reuse build_gen / build_pipe / rollout / run_reference_rollout."""
    spec = importlib.util.spec_from_file_location(
        "c2_selftest", os.path.join(REPO, "reference/runs/c2_eviction/selftest.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------- P0
def p0_forward_byte_identical(c2):
    print("\n=== P0: forward byte-identical (all c3 flags OFF == pre-c3) ===")
    assert os.path.exists(C2_REF), f"missing c2 ref {C2_REF}"
    ref = torch.load(C2_REF, weights_only=True)
    # c2.build_gen uses MODEL_KW (mem_stream_scale=1.0, use_decay=False); build_pipe adds no
    # cache_dropout; run_reference_rollout runs eviction_feed=False -> exactly the pre-c3 path.
    outs, final_state = c2.run_reference_rollout()
    diff = (outs.float() - ref["outs"].float()).abs().max().item()
    print(f"output latents: shape={tuple(outs.shape)} max|diff| = {diff}")
    assert diff == 0.0, f"BYTE-IDENTITY FAILED: max|diff|={diff}"
    for i, st in ref["final_state"].items():
        i = int(i)
        for a, b in zip(final_state[i], st):
            d = (a - b).abs().max().item()
            assert d == 0.0, f"state block {i} diff {d}"
    print("P0 PASS: c3 diff does not perturb the forward at defaults (max|diff|=0 on outputs + states)")


# ---------------------------------------------------------------------------- P1
def p1_cache_dropout_anneal():
    print("\n=== P1: cache_dropout anneal ===")
    from pipeline.streaming_training import StreamingTrainingPipeline
    # bare object (no model load): exercise the pure prob logic.
    def make(prob, end, steps):
        o = StreamingTrainingPipeline.__new__(StreamingTrainingPipeline)
        o.cache_dropout_prob = float(prob)
        o.cache_dropout_prob_end = float(end) if end is not None else float(prob)
        o.cache_dropout_anneal_steps = int(steps)
        o._train_step = 0
        return o
    # default-off: end=None -> constant == start for every step (byte-identical to c2)
    off = make(0.3, None, 0)
    for s in (0, 400, 800, 5000):
        off.set_train_step(s)
        p = off._effective_cache_dropout_prob()
        assert p == 0.3, f"anneal-off should stay 0.3, got {p} at step {s}"
    print("default-off (end=None): prob constant 0.3 at steps 0/400/800/5000  [byte-identical]")
    # anneal on: 0.3 -> 0.1 over 800
    on = make(0.3, 0.1, 800)
    exp = {0: 0.3, 200: 0.25, 400: 0.2, 800: 0.1, 1600: 0.1}
    shown = {}
    for s, e in exp.items():
        on.set_train_step(s)
        p = on._effective_cache_dropout_prob()
        shown[s] = p
        assert abs(p - e) < 1e-9, f"step {s}: expected {e}, got {p}"
    print("anneal-on 0.3->0.1/800: prob(step) = " +
          "{" + ", ".join(f"{s}:{p:.3f}" for s, p in shown.items()) + "}")
    print("P1 PASS: anneal fires (interpolates start->end, clamps) and is inert at defaults")


# ---------------------------------------------------------------------------- P2/P3 trainer stubs
def _make_trainer_stub():
    from trainer.distillation import Trainer
    t = Trainer.__new__(Trainer)
    return t


def p2_lr_schedule():
    print("\n=== P2: cosine lr schedule ===")
    base = [1.0e-5, 5.0e-5]   # [LoRA, mem]
    def fresh(decay_start, fl=0.1, flm=0.05):
        t = _make_trainer_stub()
        t.step = 0
        t.config = SimpleNamespace(max_iters=2000)
        t._base_lrs = list(base)
        t._lr_decay_start = decay_start
        t._lr_floor_frac = fl
        t._lr_mem_floor_frac = flm
        pg = [{"lr": base[0]}, {"lr": base[1]}]
        t.generator_optimizer = SimpleNamespace(param_groups=pg)
        return t, pg
    # default-off: decay_start beyond max_iters -> lr untouched at every step
    t, pg = fresh(10**9)
    for s in (0, 500, 1999):
        t.step = s; t._lr_sched_step()
        assert pg[0]["lr"] == base[0] and pg[1]["lr"] == base[1], f"lr moved at step {s} with sched off"
    print(f"default-off (decay_start=1e9): lr stays [{base[0]:.1e}, {base[1]:.1e}]  [byte-identical]")
    # on: decay_start=400, cosine to floor at max_iters=2000
    t, pg = fresh(400)
    def lr_at(s):
        t.step = s; t._lr_sched_step(); return pg[0]["lr"], pg[1]["lr"]
    l0 = lr_at(400)                       # start of decay -> cos=1 -> base
    assert abs(l0[0] - base[0]) < 1e-12 and abs(l0[1] - base[1]) < 1e-12, l0
    lend = lr_at(2000)                    # end -> cos=0 -> base*floor
    assert abs(lend[0] - base[0]*0.1) < 1e-12, lend
    assert abs(lend[1] - base[1]*0.05) < 1e-12, lend
    lmid = lr_at(1200)                    # prog=0.5 -> cos=0.5 -> base*(floor+(1-floor)*0.5)
    exp0 = base[0]*(0.1 + 0.9*0.5); exp1 = base[1]*(0.05 + 0.95*0.5)
    assert abs(lmid[0]-exp0) < 1e-12 and abs(lmid[1]-exp1) < 1e-12, (lmid, exp0, exp1)
    print(f"on: lr(400)=[{l0[0]:.3e},{l0[1]:.3e}]  lr(1200)=[{lmid[0]:.3e},{lmid[1]:.3e}]  "
          f"lr(2000)=[{lend[0]:.3e},{lend[1]:.3e}] (LoRA floor 0.1x, mem floor 0.05x)")
    print("P2 PASS: cosine fires per-group with distinct floors; inert at defaults")


def p3_mem_grad_clip():
    print("\n=== P3: mem-grad clip ===")
    def one(max_norm):
        t = _make_trainer_stub()
        p = nn.Parameter(torch.zeros(4, 4))
        p.grad = torch.full((4, 4), 10.0)          # norm = 10*4 = 40
        t.mem_params = [p]
        t.max_grad_norm_mem = float(max_norm)
        t._clip_mem_grads()
        return p.grad.norm().item()
    n_off = one(1e9)
    assert abs(n_off - 40.0) < 1e-4, n_off
    print(f"default-off (max_grad_norm_mem=1e9): grad norm unchanged = {n_off:.3f}  [byte-identical]")
    n_on = one(1.0)
    assert abs(n_on - 1.0) < 1e-4, n_on
    print(f"on (max_grad_norm_mem=1.0): grad norm 40.0 -> {n_on:.3f}")
    print("P3 PASS: mem-grad clip fires; inert at huge default")


# ---------------------------------------------------------------------------- P4
def p4_lora_ema_trainable_only():
    print("\n=== P4: LoRA-EMA trainable-only shadow diverges ===")
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    from utils.distributed import EMA_FSDP
    torch.manual_seed(0)
    m = nn.Sequential(nn.Linear(8, 8, bias=False), nn.Linear(8, 8, bias=False)).cuda()
    for p in m[0].parameters():
        p.requires_grad_(False)                    # freeze the first linear
    fm = FSDP(m, use_orig_params=True)
    ema = EMA_FSDP(fm, decay=0.9)
    keys = sorted(ema.shadow.keys())
    print(f"shadow keys (trainable-only): {keys}")
    assert any("1." in k for k in keys), "trainable linear missing from shadow"
    assert not any("0." in k for k in keys), f"FROZEN linear leaked into shadow: {keys}"
    # snapshot, mutate the trainable param, update EMA -> shadow must move toward the new value
    k1 = [k for k in keys if "1." in k][0]
    before = ema.shadow[k1].clone()
    with FSDP.summon_full_params(fm):
        for n, p in fm.module.named_parameters():
            if "1." in n:
                p.data.add_(1.0)                   # shift trainable weights by +1
    ema.update(fm)
    after = ema.shadow[k1]
    moved = (after - before).abs().max().item()
    exp_move = 0.1 * 1.0                            # (1-decay)*delta
    print(f"shadow[{k1}] moved max|Δ| = {moved:.4f} (expected ~{exp_move:.2f} = (1-0.9)*1.0)")
    assert moved > 1e-3, "EMA shadow did NOT move on update (EMA not firing)"
    assert abs(moved - exp_move) < 5e-2, f"EMA move {moved} far from expected {exp_move}"
    del fm, m
    torch.cuda.empty_cache()
    print("P4 PASS: shadow is trainable-only (frozen excluded) and diverges from init on update")


# ---------------------------------------------------------------------------- P5
def p5_use_decay_fresh_init(c2):
    print("\n=== P5: use_decay fresh-init (decay_logit) ===")
    from wan.modules.video_memory import VideoMemoryLayer
    gen_on = build_gen_override(c2, use_decay=True)
    n_dl, vals = 0, []
    for _, mod in gen_on.model.named_modules():
        if isinstance(mod, VideoMemoryLayer):
            assert hasattr(mod, "decay_logit"), "use_decay=True but no decay_logit param"
            n_dl += 1
            vals.append(torch.sigmoid(mod.decay_logit).item())
    assert n_dl == 3, f"expected 3 memory layers with decay_logit, got {n_dl}"
    assert all(abs(v - 0.95) < 1e-3 for v in vals), f"decay init != 0.95: {vals}"
    print(f"use_decay=True: {n_dl} memory layers each with decay_logit, sigmoid ~ {vals[0]:.4f} (init 0.95)")
    del gen_on
    torch.cuda.empty_cache()
    gen_off = build_gen_override(c2, use_decay=False)
    has = any(hasattr(mod, "decay_logit") for _, mod in gen_off.model.named_modules()
              if isinstance(mod, VideoMemoryLayer))
    assert not has, "use_decay=False should NOT create decay_logit"
    print("use_decay=False: no decay_logit created  [byte-identical parameter set]")
    del gen_off
    torch.cuda.empty_cache()
    print("P5 PASS: decay_logit built fresh iff use_decay=True")


# ---------------------------------------------------------------------------- P6
def p6_mem_stream_scale_fires(c2):
    print("\n=== P6: mem_stream_scale changes the injection ===")
    # fresh proj is zero-init (no injection), so mem_stream_scale would be invisible. Force an
    # identical NONZERO proj on both gens, then the two scales must produce different outputs.
    from wan.modules.video_memory import VideoMemoryLayer
    def build_with_proj(scale):
        gen = build_gen_override(c2, mem_stream_scale=scale)
        g = torch.Generator(device="cuda").manual_seed(4242)
        for _, mod in gen.model.named_modules():
            if isinstance(mod, VideoMemoryLayer):
                w = torch.randn(mod.proj.weight.shape, generator=g,
                                device="cuda", dtype=mod.proj.weight.dtype) * 0.02
                with torch.no_grad():
                    mod.proj.weight.copy_(w)
        return gen
    gen1 = build_with_proj(1.0)
    pipe1 = c2.build_pipe(gen1)
    o1, _, _, _ = c2.rollout(pipe1, n_chunks=4)
    del pipe1, gen1; torch.cuda.empty_cache()
    gen2 = build_with_proj(0.5)
    pipe2 = c2.build_pipe(gen2)
    o2, _, _, _ = c2.rollout(pipe2, n_chunks=4)
    del pipe2, gen2; torch.cuda.empty_cache()
    d = (o1.float() - o2.float()).abs().max().item()
    print(f"mem_stream_scale 1.0 vs 0.5 (identical nonzero proj): max|diff| = {d:.4e}")
    assert d > 0.0, "mem_stream_scale had NO effect (injection not scaled)"
    print("P6 PASS: mem_stream_scale scales the injection (nonzero output difference)")


def p7_ema_save_reload_roundtrip():
    print("\n=== P7: EMA shadow save->reload round-trip (LoRA mode) ===")
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    from utils.distributed import EMA_FSDP
    torch.manual_seed(3)
    m = nn.Sequential(nn.Linear(8, 8, bias=False), nn.Linear(8, 8, bias=False)).cuda()
    for p in m[0].parameters():
        p.requires_grad_(False)                        # frozen "base"
    fm = FSDP(m, use_orig_params=True)
    ema = EMA_FSDP(fm, decay=0.9)
    # move the shadow off its init snapshot so we test real (non-trivial) values
    with FSDP.summon_full_params(fm):
        for n, p in fm.module.named_parameters():
            if "1." in n:
                p.data.add_(0.5)
    ema.update(fm)
    orig = {k: v.clone() for k, v in ema.shadow.items()}
    assert not any("0." in k for k in orig), f"frozen key leaked pre-save: {list(orig)}"

    # ---- SAVE: exactly the trainer LoRA branch dict shape (generator_ema key) ----
    ckpt = {
        "generator_lora": {"dummy.lora_A.weight": torch.zeros(2, 2)},
        "critic_lora": {"dummy.lora_A.weight": torch.zeros(2, 2)},
        "generator_memory": {"blocks.5.memory.w0": torch.zeros(1, 4, 4)},
        "step": 500,
    }
    if ema is not None:                                 # mirrors: if self.generator_ema is not None
        ckpt["generator_ema"] = ema.state_dict()
    path = os.path.join(REPO, "reference/runs/c3_stable/_selftest_ckpt.pt")
    torch.save(ckpt, path)

    # ---- RELOAD: the resume path stashes ckpt["generator_ema"], a FRESH EMA restores it ----
    ld = torch.load(path, weights_only=False)
    assert "generator_ema" in ld, "generator_ema MISSING from the saved LoRA checkpoint (mechanism lost!)"
    stashed = ld["generator_ema"]
    torch.manual_seed(999)                              # different init so restore is a real overwrite
    m2 = nn.Sequential(nn.Linear(8, 8, bias=False), nn.Linear(8, 8, bias=False)).cuda()
    for p in m2[0].parameters():
        p.requires_grad_(False)
    fm2 = FSDP(m2, use_orig_params=True)
    ema2 = EMA_FSDP(fm2, decay=0.9)
    ema2.load_state_dict(stashed)                       # mirrors trainer resume: load stashed shadow
    # keys present + trainable-only + tensors bitwise-match the pre-save shadow
    assert set(ema2.shadow.keys()) == set(orig.keys()), \
        f"key mismatch after reload: {set(ema2.shadow)} vs {set(orig)}"
    assert not any("0." in k for k in ema2.shadow), "frozen key appeared after reload"
    for k in orig:
        d = (ema2.shadow[k].float() - orig[k].float()).abs().max().item()
        assert d == 0.0, f"shadow[{k}] changed across save/reload: max|diff|={d}"
    print(f"saved keys under 'generator_ema': {sorted(stashed.keys())}")
    print(f"reload match: {len(orig)} trainable shadow keys, max|diff| = 0.0 (bitwise)")
    os.remove(path)
    del fm, fm2, m, m2
    torch.cuda.empty_cache()
    print("P7 PASS: generator_ema is saved in the LoRA ckpt and restores bitwise on resume")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", action="store_true")
    args = ap.parse_args()

    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29519")
    dist.init_process_group("nccl", rank=0, world_size=1)
    torch.cuda.set_device(0)

    c2 = _load_c2_selftest()

    if args.full:
        # cheap, model-free logic tests first (fail fast), then the model-load ones.
        p1_cache_dropout_anneal()
        p2_lr_schedule()
        p3_mem_grad_clip()
        p4_lora_ema_trainable_only()
        p7_ema_save_reload_roundtrip()
        p0_forward_byte_identical(c2)
        p5_use_decay_fresh_init(c2)
        p6_mem_stream_scale_fires(c2)
        print("\nALL PHASES PASS")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
