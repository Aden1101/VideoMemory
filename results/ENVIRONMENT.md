# Where the numbers were measured

What is known from the repository, and what is not. Nothing here was reconstructed from memory.

| | |
|---|---|
| hardware | CUDA, 4 GPUs for the training runs (`train.sh`); GPU model: not recorded |
| seed, prompts, length | seed 42, six held-out prompts, 240 latent frames (480 for the `_480` tables) |
| identity metric | DINOv2 ViT-B/14, `facebookresearch/dinov2` through `torch.hub`, first frame against last |
| VBench | subject consistency, background consistency, dynamic degree; version: not recorded |
| torch, CUDA, driver | not recorded; `requirements.txt` gives the ranges the code was written against |
| Python | 3.10, the environment upstream's README sets up |

One seed and no repeated measurement, so none of these tables carries a noise estimate.
