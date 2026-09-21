"""Noise for the control arms, drawn apart from the rollout's random stream."""
import torch


def private_noise(like, seed):
    """Unit Gaussian noise shaped like `like`, on its device, in float32, from a generator of
    its own. The main random stream is left alone, so two arms that differ by a control still
    generate from identical noise; the seed includes the switch frame, so a rollout with two
    revisits draws twice."""
    generator = torch.Generator(device=like.device)
    generator.manual_seed(int(seed))
    return torch.randn(like.shape, generator=generator, device=like.device, dtype=torch.float32)
