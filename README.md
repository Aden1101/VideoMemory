# Fast-weight memory for long-form video generation

A research fork of [Anchor-Forcing](https://github.com/vivoCameraResearch/Anchor-Forcing)
(Wan2.1-T2V-1.3B, causal streaming, DMD self-rollout) that grafts a **fast-weight / test-time
training memory** into the backbone, to attack identity drift over minutes of generated video.

The upstream project's own README is preserved at [`reference/UPSTREAM_README.md`](reference/UPSTREAM_README.md).

---

## The question

Autoregressive video models keep a fixed attention window. Anything older than that window is
literally evicted from the context. Identity is precisely long-range information: if the subject
established at second 0 is out of context at second 60, nothing constrains the model to preserve
it except the prompt — and the prompt only specifies a *category* ("a woman in a red coat"),
never the *instance*. Those under-specified attributes drift freely.

Can a learnable fast-weight memory carry the instance across the eviction boundary?

## The answer, honestly

**No, not under this training objective — and the reason generalises.** The headline is not the
memory module; it is a methodology finding, demonstrated three times over:

> The training objective (DMD, chunk-local) and the standard evaluation metrics (VBench subject
> consistency, naive DINO drift) reward **the same wrong thing: temporal self-similarity**. A
> learnable memory therefore converges to a freeze or a DC-injection degeneracy that the
> dashboard scores as *healthy*. VBench rates a dissolving clip at 0.94. "Consistent because
> frozen."

Three positive results were killed by their own authors along the way: a "memory is toxic"
finding that turned out to be a DC-injection artifact, a "do-no-harm" improvement that turned
out to be a freeze artifact, and a channel-content claim falsified by the observation that an
anchor decoded from **someone else's** video works just as well as our own.

What survived is a measured negative with a validated instrument, plus one live lead: a
training-free **delta write rule** that reads the subject where the deployed SwiGLU write does
not (median 13.5/19 vs 7.0 over 10 seeds, non-overlapping distributions, with a background
control at chance to rule out scene memorisation).

## If you read three files

1. **[`wan/modules/video_memory.py`](wan/modules/video_memory.py)** — the mechanism. The
   fast-weight cell (SwiGLU state, Muon write, renorm-to-init), the training-free `linear_delta`
   write rule bit-locked to its offline oracle, and the state→frame decoder.
2. **[`model/streaming_training.py`](model/streaming_training.py)** — the experimental design,
   from `_twin_snapshot` down. A counterfactual mem-OFF twin of the *same chunk*, sharing
   history, KV cache and noise, so the only differential lever is the injection. Then four
   objective terms whose anti-degeneracy structure is blocked *in construction* rather than by
   tuning — each guard documented against the failure that actually produced it.
3. **[`reference/memory2_probes_registry.md`](reference/memory2_probes_registry.md)** — the
   method. Thresholds sealed before each run, predictions recorded with their refutations, a
   10-seed annex that downgrades the project's own best number, and a device-variability control
   that makes small differences explicitly uninterpretable.

[`reference/README.md`](reference/README.md) indexes the rest of the research record.

## What is mine

The base import (commit `392b3f8`) is byte-identical to upstream
`vivoCameraResearch/Anchor-Forcing@25bdfba`, verified over the full tree, and the checkpoints
sha256-match the published release. Everything after that commit is this work: the memory cell
and its write rules, the identity and anti-degeneracy objectives, the evaluation harness, the
probing pipeline, the blind judging boards, and the device-agnostic pass described below.

The video backbone, the DMD distillation and the streaming pipeline are upstream's.

## Running it

Weights (~26 GB) and a two-pass launch, because the text encoder and the generator do not fit in
memory together on a consumer machine:

```bash
python download_weights.py                    # Wan2.1-T2V-1.3B + Anchor-Forcing checkpoints
python encode_prompts.py --file ckpt/prompts/bench/vbench_all_dimension_extended.txt --limit 1
AF_LAZY_TEXT_ENCODER=1 PYTHONPATH=. python inference/inference.py --config_path configs/inference.yaml
```

Pass 1 encodes the prompts and **exits** — only ending the process actually returns the 11.4 GB
to the OS on unified memory. Pass 2 never constructs the encoder.

Training needs CUDA and two 48 GB cards; see [`reference/vram_recap.md`](reference/vram_recap.md)
for the measured ceilings.

### Devices

The code runs on CUDA, on Apple Metal and on CPU. On CUDA every device call delegates to the
original `torch.cuda` one, so no previously measured number moves — `tests/test_device.py`
asserts that by reimplementing the pre-patch RoPE and demanding bitwise equality. Off CUDA,
float64 falls back to float32 where Metal has no float64 at all; that trade-off is declared, and
its effect over a long rollout is not yet measured.

```bash
python tests/test_device.py            # device layer, incl. the bitwise check
python tests/test_inference_smoke.py   # real model, streaming, no weights needed
python tests/test_prompt_cache.py      # prompt cache, incl. bitwise read-back
```

## Status

The memory work is closed on plain drift and documented as a negative result. The open lead is
the delta write rule and the state→frame decoder, whose three known defects are diagnosed and
fixed but not yet retrained. See [`reference/RESEARCH_STATUS.md`](reference/RESEARCH_STATUS.md).

## License

Upstream is CC BY-NC-SA 4.0; this fork inherits it. Non-commercial, share-alike.
