# Fast-weight memory for long-form video generation

A research fork of [Anchor-Forcing](https://github.com/vivoCameraResearch/Anchor-Forcing)
(Wan2.1-T2V-1.3B, causal streaming, DMD distillation) that adds a **fast-weight memory** to the
backbone, to fight identity drift over minutes of generated video.

Upstream's README is kept at [`reference/UPSTREAM_README.md`](reference/UPSTREAM_README.md).

## The problem

The model generates 3 latent frames at a time and attends to a rolling window of 12 frames plus
3 pinned sink frames. Everything older is evicted. Identity is long-range information, so once
the subject established at second 0 leaves the window, only the prompt constrains it — and the
prompt gives a category ("a woman in a red coat"), never the instance. Those attributes drift.

## How it works

A memory branch is inserted in some transformer blocks (layers 5, 16, 27). Each one keeps a
**fast weight** `W` that is rewritten during generation, not by the optimizer.

```
read    o = f_W(q)          f_W = SwiGLU, q from the current chunk
write   W ← Muon-orthogonalized ascent step on ⟨f_W(k), v⟩,
          then each column renormalized to its initial norm
inject  x ← x + gate(x) · AdaLN(o)      into the residual stream
```

The write is fed the chunk that has **just been evicted** from the attention window, so the
memory holds what attention can no longer see rather than duplicating it.

| file | what is in it |
|---|---|
| [`wan/modules/video_memory.py`](wan/modules/video_memory.py) | the memory cell: fast-weight state, the write rules, the gated injection, the state→frame decoder |
| [`wan/modules/causal_model.py`](wan/modules/causal_model.py) | the backbone, the rolling KV cache, and the eviction feed |
| [`model/streaming_training.py`](model/streaming_training.py) | training objectives, built around a counterfactual memory-off twin of the same chunk |
| [`utils/device.py`](utils/device.py) | one source of truth for the device (CUDA / Metal / CPU) |
| [`reference/`](reference/README.md) | the research record: reports, instruments, measured numbers |

## Result

The memory does not beat attention plus sink on plain identity drift. That is the honest
outcome, and the reason generalises: the training objective (chunk-local DMD) and the usual
metrics (VBench subject consistency, DINO drift) reward the **same wrong thing — temporal
self-similarity** — so a learnable memory converges to a freeze or a constant-bias degeneracy
that the dashboard scores as healthy. VBench rates a dissolving clip at 0.94.

One lead is still open: a training-free **delta write rule** reads the subject out of the state
where the deployed rule does not (median 13.5/19 vs 7.0 over 10 seeds).

## Running it

```bash
python download_weights.py                    # Wan2.1-T2V-1.3B + Anchor-Forcing checkpoints, ~26 GB
python encode_prompts.py --file ckpt/prompts/bench/vbench_all_dimension_extended.txt --limit 1
AF_LAZY_TEXT_ENCODER=1 PYTHONPATH=. python inference/inference.py --config_path configs/inference.yaml
```

Two passes because the text encoder (11.4 GB) and the generator do not fit in memory together on
a consumer machine: pass 1 encodes and exits, pass 2 never loads the encoder. Training needs
CUDA and two 48 GB cards.

Runs on CUDA, Apple Metal and CPU. On CUDA every device call delegates to the original
`torch.cuda` one, so no previously measured number moves — `tests/test_device.py` checks that by
reimplementing the pre-patch rotary embedding and demanding bitwise equality.

```bash
python tests/test_device.py            # device layer
python tests/test_inference_smoke.py   # real model, streaming, no weights needed
```

## Credit

The base import (`392b3f8`) is byte-identical to upstream `vivoCameraResearch/Anchor-Forcing`.
The backbone, the DMD distillation and the streaming pipeline are theirs. Everything after that
commit — the memory cell, its write rules, the training objectives and the evaluation code — is
this work. License CC BY-NC-SA 4.0, inherited from upstream.
