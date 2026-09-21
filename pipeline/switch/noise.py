"""Noise for the control arms, drawn apart from the rollout's random stream."""
import torch

# Default seeds of the private draws, each offset by the switch frame at its call site.
ANCHOR_NOISE_SEED = 192837465
REPLAY_NOISE_SEED = 987654321


def private_noise(like, seed):
    """Unit Gaussian noise shaped like `like`, on its device, in float32, from a generator of
    its own. The main random stream is left alone, so two arms that differ by a control still
    generate from identical noise; the seed includes the switch frame, so a rollout with two
    revisits draws twice."""
    generator = torch.Generator(device=like.device)
    generator.manual_seed(int(seed))
    return torch.randn(like.shape, generator=generator, device=like.device, dtype=torch.float32)


def mix_with_noise(x, noise, sigma):
    """sqrt(1 - sigma^2) * x + sigma * noise: for unit-variance inputs the result keeps the RMS
    of x, so the dose changes the content and not the scale. sigma = 1 leaves only the noise."""
    return ((1.0 - sigma ** 2) ** 0.5) * x + sigma * noise
