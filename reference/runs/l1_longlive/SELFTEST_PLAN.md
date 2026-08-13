# l1_longlive selftest plan (run BEFORE launching the real 2000-step run)

Verdict recap (2026-07-17 investigation, all verified by reading real files, no GPU used):

- **Case A: drop-in.** LongLive's released `models/lora.pt` (HF Efficient-Large-Model/LongLive-1.3B,
  downloaded to `/shared/ssd_14T_project/adenelmi/longlive_ckpt/models/`) has top keys
  `['generator_lora', 'critic_lora', 'step']` with 600 peft-format keys each
  (`base_model.model.blocks.N.<module>.lora_{A,B}.weight`, rank 256) — the EXACT format
  `trainer/distillation.py` L227 requires. Keyset AND shapes are identical to our `ckpt/final.pt`.
- **`ckpt/base.pt` is byte-identical to `models/longlive_base.pt`** (cmp: no diff, both 5676334208
  bytes). Our substrate already IS LongLive's base; only the LoRA delta changes.
- Weights genuinely differ (`final.pt` vs `lora.pt` diverge at byte 196545; mean |delta| ~1.7e-2 on
  sampled tensors; all lora_B nonzero in both) — the swap is real, not a no-op.
- `lora.pt` carries `step: 3000`; config `resume_step: false` (already in c3/l1) keeps step 0.
  WITHOUT it the run would exit instantly (3000 > max_iters 2000).
- Arch: shapes (256,1536), blocks 0-29, ffn 8960 = Wan2.1-T2V-1.3B, 30 blocks, dim 1536.
  Memory graft at blocks [5,16,27] < 30: compatible. NO blocker.
- LongLive step-2 trained with local_attn_size 12 / sink_size 3 / num_frame_per_block 3 /
  timestep_shift 5.0 / denoising [1000,750,500,250] / rank=alpha=256 — same as our config, so the
  LoRA is in-distribution for our attention-window settings
  (source: /shared/ssd_14T/home/adenelmi/LongLive/longlive/configs/longlive_train_long.yaml).

All commands below assume `cd /shared/ssd_14T/home/adenelmi/Anchor-Forcing` and
`PY=/shared/ssd_14T/home/adenelmi/miniconda3/envs/longlive/bin/python`. Per ops memory: GPU probes on
GPU 0 (scratch), `CUDA_DEVICE_ORDER=PCI_BUS_ID`, smi idx6 excluded.

## Check 0 — CPU format validation (no GPU, ~1 min) [already ran clean 2026-07-17; re-run to be sure]

```bash
$PY - <<'EOF'
import torch
ck = torch.load("/shared/ssd_14T_project/adenelmi/longlive_ckpt/models/lora.pt", map_location="cpu", weights_only=False)
assert "generator_lora" in ck and "critic_lora" in ck, f"loader L227 would ValueError: {list(ck.keys())}"
assert len(ck["generator_lora"]) == 600 and len(ck["critic_lora"]) == 600
ours = torch.load("ckpt/final.pt", map_location="cpu", weights_only=False)
assert set(ck["generator_lora"]) == set(ours["generator_lora"])   # exact peft key-name match
assert set(ck["critic_lora"])   == set(ours["critic_lora"])
assert all(ck["generator_lora"][k].shape == ours["generator_lora"][k].shape for k in ck["generator_lora"])
print("OK: lora.pt passes the distillation.py L227 check; keyset+shapes == final.pt; step =", ck["step"])
EOF
cmp /shared/ssd_14T/home/adenelmi/Anchor-Forcing/ckpt/base.pt \
    /shared/ssd_14T_project/adenelmi/longlive_ckpt/models/longlive_base.pt \
  && echo "base BYTE-IDENTICAL (expected)"
```

Expected: both `OK` lines. If cmp ever fails, STOP — the download changed / re-verify.

## Check 1 — peft load-path probe (1 GPU, ~3 min)

Exercises the actual `peft.set_peft_model_state_dict` alignment (distillation.py L254) outside FSDP.
`import wan` touches CUDA, hence GPU needed.

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 $PY - <<'EOF'
import sys, torch; sys.path.insert(0, "/shared/ssd_14T/home/adenelmi/Anchor-Forcing")
from omegaconf import OmegaConf
import peft
from utils.wan_wrapper import WanDiffusionWrapper
from utils.lora_utils import configure_lora_for_model
cfg = OmegaConf.load("reference/runs/l1_longlive/l1_longlive.yaml")
gen = WanDiffusionWrapper(**cfg.model_kwargs)          # memory graft at [5,16,27] included
base = torch.load(cfg.generator_ckpt, map_location="cpu")
r = gen.load_state_dict(base["generator"], strict=False)
assert not [k for k in r.missing_keys if "memory" not in k] and not r.unexpected_keys
print(f"base loaded, {len(r.missing_keys)} memory keys kept at init (expect 39)")
gen.model = configure_lora_for_model(gen.model, "generator", lora_config=cfg.adapter)
ck = torch.load(cfg.lora_ckpt, map_location="cpu", weights_only=False)
out = peft.set_peft_model_state_dict(gen.model, ck["generator_lora"])
unexpected = getattr(out, "unexpected_keys", [])
assert not unexpected, f"unexpected: {list(unexpected)[:5]}"
sd = peft.get_peft_model_state_dict(gen.model)
k = "base_model.model.blocks.0.self_attn.q.lora_B.weight"
assert torch.equal(sd[k].float().cpu(), ck["generator_lora"][k]), "LoRA tensor did not land"
print("OK: LongLive generator_lora loads + round-trips through peft on the memory-grafted model")
EOF
```

Expected: `39 memory keys kept at init` + final `OK`. (If `configure_lora_for_model`'s signature
differs, mirror the call from inference.py; the assertion logic stays the same.)

## Check 2 — 6-step dry train run (4 GPUs, ~15-30 min)

Uses `reference/runs/l1_longlive/l1_dry.yaml` = l1_longlive.yaml with only max_iters 6, log_iters 6,
use_wandb false, logdir `/shared/ssd_14T_project/adenelmi/l1_dry`.

```bash
cd /shared/ssd_14T/home/adenelmi/Anchor-Forcing
mkdir -p /shared/ssd_14T_project/adenelmi/l1_dry
export CUDA_VISIBLE_DEVICES=0,1,2,3 CUDA_DEVICE_ORDER=PCI_BUS_ID NCCL_P2P_DISABLE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,garbage_collection_threshold:0.8
$PY -m torch.distributed.run --nproc_per_node=4 --max-restarts=0 --master_port=30115 \
  train.py --config_path reference/runs/l1_longlive/l1_dry.yaml --no-one-logger \
  2>&1 | tee reference/runs/l1_longlive/l1_dry.log
```

Then grep the log — ALL of these must hold:

```bash
L=reference/runs/l1_longlive/l1_dry.log
grep -c "Generator weights loaded successfully (39 memory keys kept at init)" $L   # >=1
grep -c "Using explicit LoRA checkpoint: /shared/ssd_14T_project/adenelmi/longlive_ckpt/models/lora.pt" $L  # >=1
grep -c "Loading LoRA generator weights: 600 keys in checkpoint" $L                # >=1
grep -c "Loading LoRA critic weights: 600 keys in checkpoint" $L                   # >=1
grep -c "Training steps (start=0)" $L      # >=1  <- resume_step:false held; NOT start=3000
grep -ci "ValueError\|Traceback\|NaN\|shape mismatch" $L                           # 0
grep -c "Resuming LoRA training from step" $L                                      # 0
```

Loss finite (reads the tfevents the SummaryWriter wrote even with wandb off):

```bash
$PY - <<'EOF'
import math
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
ea = EventAccumulator("/shared/ssd_14T_project/adenelmi/l1_dry"); ea.Reload()
tags = [t for t in ea.Tags()["scalars"] if "loss" in t.lower()]
assert tags, f"no loss scalars; tags={ea.Tags()['scalars'][:20]}"
for t in tags:
    vals = [e.value for e in ea.Scalars(t)]
    assert vals and all(math.isfinite(v) for v in vals), (t, vals)
    print(f"OK {t}: {vals}")
EOF
```

Saved-ckpt sanity (save fires at step 6 since log_iters=6):

```bash
$PY - <<'EOF'
import torch, glob
p = sorted(glob.glob("/shared/ssd_14T_project/adenelmi/l1_dry/checkpoint_model_*/model.pt"))[-1]
ck = torch.load(p, map_location="cpu", weights_only=False)
print(p, "->", list(ck.keys()))
assert "generator_lora" in ck and "critic_lora" in ck
mem = ck.get("generator_memory", {})
print(f"generator_lora {len(ck['generator_lora'])}, critic_lora {len(ck['critic_lora'])}, memory {len(mem)} keys")
EOF
```

Expected: generator_lora 600 / critic_lora 600 / memory 39 keys (the c3 save path also stashes
generator_ema once past ema_start_step 250 — absent at step 6, that is fine).

## Check 3 — LongLive-stock render column (1 GPU, GPU 0)

`reference/runs/l1_longlive/stock_longlive.yaml` = the stockaf.yaml pattern with lora.pt: memory off
(`memory_layers: []`), LongLive LoRA, same held-out prompts/seed 42/240f. Safe because lora.pt has
no memory keys (the []-crash footgun only bites our memory-carrying ckpts).

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 $PY inference/inference.py \
  --config_path reference/runs/l1_longlive/stock_longlive.yaml
```

(Use the same inference entrypoint the c3 evals used — see
`reference/runs/c3_stable/eval250_sink1/run_eval.sh` for the exact invocation if it differs.)

Purpose: (a) proves the LoRA produces coherent video through OUR inference stack before burning a
training run; (b) becomes the standing 3rd column (3-way rule: mem-ON | mem-OFF | stock) for every
l1 eval. NOTE: do NOT expect byte-identity vs the LongLive repo's own inference.py — our fork's
inference code has diverged (global_sink/rope_enhance handling); the check is qualitative health +
later VBench, not bit-exactness. Eyeball 2-3 clips: if the stock column is broken (grain/freeze/
dissolve), STOP — that is a code-path incompatibility, not a memory problem.

## Check 4 — memory graft init + drift (covered inside Check 2, verify explicitly)

- `39 memory keys kept at init` line present (base.pt and lora.pt carry no memory keys; the graft
  inits fresh, exactly like c3).
- `use_decay: true` adds fresh `decay_logit` params — they are memory params, so they are inside the
  39-key init path; no strict-load caveat (same as c3, which also started memory fresh).
- `assert_memory_drift: true` is in the config: the trainer's own assert fires if memory params do
  not receive grads/updates. A clean 6-step dry run with no assertion error = memory is training.
- Optional direct probe on the step-6 dry ckpt: `generator_memory` values should differ from a fresh
  init if any write-train step ran (mem params update every step; LoRA-gen updates only every 5).

## Launch (after all checks pass)

```bash
bash reference/runs/l1_longlive/launch_l1.sh "0,1,2,3" 30114   # mirrors launch_c3.sh; wandb name = l1_longlive (basename of logdir)
```

Reminder: c3_stable may still be holding GPUs — decide stop/hot-swap first (do-not-disrupt rule
applies only to HEALTHY runs). 4 GPUs max without asking Aden.

## Rollback

Everything is config-only. Reverting = launch c3_stable.yaml again (`lora_ckpt: ckpt/final.pt`).
No training code was modified for the re-base.
