# Does memory read/write training fit in VRAM?

Hardware: 48 GiB GPUs (47.4 GiB usable). Training window = 42 latent frames (~10.3 s) unless a shorter window is stated. Memory measured at the single heaviest training step.

## One GPU

| Memory layers | Read only | Read + write training |
|---|---|---|
| 1  (layer 5) | **FITS**  (44.7 GiB) | OOM |
| 2  (layers 5, 10) | **FITS**  (45.4 GiB) | OOM |
| 3  (layers 5, 10, 15) | OOM | OOM |

On a single GPU the read path fits (up to 2 memory layers), but write training never fits, not even with one layer. This is why we move to two GPUs.

**What the terms mean:**
- **Memory layers**: how many memory modules we insert in the network, and at which depth (layer 5, then 10, then 15). More layers = more capacity but more memory used.
- **Read only**: the memory is used to recall, but we do not train how it writes. Cheaper.
- **Read + write training**: we also train how the memory writes, which requires extra backprop through the write step. Much heavier.
- **FITS / OOM**: the run survived the heaviest step / ran out of memory and crashed.

## Two GPUs

*All rows below use a single memory layer (at layer 5).*

| What we ran | Result | Memory missing to fit |
|---|---|---|
| Write training on (window 42) | OOM | 560 MiB |
| + checkpoint the write step | OOM | 166 MiB |
| + shorter window (33 frames) | OOM | 80 MiB |
| + shorter window (27 frames) | OOM | 42 MiB  (fragmentation only) |
| + anti-fragmentation allocator, window 27 | OOM | still short |
| + anti-fragmentation allocator, window 24 | **FITS** | 0  (45.2 / 47.4 GiB used) |

Using two GPUs makes write training fit: the gap drops from impossible (one GPU) to 560 MiB, then a series of cheap changes brings it down until it fits. The largest training window that fits is **24 frames (~5.8 s)**, measured. Window 27 still just misses, even with the anti-fragmentation allocator.

**What the terms mean:**
- **Two GPUs**: the model is split so each card only stores half of it, which frees memory on each card.
- **Window**: the number of frames we train the write through. A shorter window needs less memory (it does not shorten the generated video, only the training span).
- **Checkpoint the write step**: recompute some values instead of storing them, trading a little compute for memory.
- **Anti-fragmentation allocator**: a PyTorch setting that lets it reuse memory that is otherwise reserved but stranded.
- **Memory missing to fit**: how many MiB short the run was at the heaviest step. Lower is closer; 0 means it fits.
- **Fragmentation**: memory is free in total but split into pieces too small to use, so the run cannot grab one contiguous block even though the total looks available.

## In one line

The memory read path fits on a single GPU. Write training fits on two GPUs, with a training window of up to 24 frames (~5.8 s), measured.
