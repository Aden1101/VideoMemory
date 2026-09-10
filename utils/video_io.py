# ============================================================================================
# VIDEO WRITING -- one entry point.
#
# 2026-09-09. `torchvision.io.write_video` was deprecated and then REMOVED from recent
# torchvision releases (gone as of 0.24). The repo imports it at the top of four files, so on
# any recent install the inference path breaks at import time, with a message that says
# nothing about the real problem.
#
# This module keeps torchvision when it is there (behaviour and encoding strictly unchanged on
# a README-conformant install, torchvision 0.21) and otherwise falls back to imageio, which is
# already in requirements.txt. Nothing to pin, nothing to reinstall.
#
# Contract shared by both paths: `frames` is a (T, H, W, C) uint8 tensor, as torchvision
# requires.
# ============================================================================================
import numpy as np
import torch

try:                                     # torchvision < 0.24
    from torchvision.io import write_video as _tv_write_video
    _BACKEND = "torchvision"
except ImportError:                      # torchvision >= 0.24, or torchvision absent
    _tv_write_video = None
    _BACKEND = "imageio"


def backend():
    return _BACKEND


def write_video(filename, video_array, fps, **kwargs):
    """Write an mp4. Signature deliberately identical to torchvision.io.write_video."""
    if _tv_write_video is not None:
        return _tv_write_video(filename, video_array, fps=fps, **kwargs)

    import imageio.v2 as imageio
    if isinstance(video_array, torch.Tensor):
        video_array = video_array.detach().cpu()
        if video_array.dtype != torch.uint8:
            video_array = video_array.clamp(0, 255).to(torch.uint8)
        video_array = video_array.numpy()
    video_array = np.asarray(video_array, dtype=np.uint8)
    assert video_array.ndim == 4 and video_array.shape[-1] in (1, 3), \
        f"expected (T, H, W, C) uint8, got {video_array.shape}"
    # macro_block_size=1: without it imageio silently resizes to a multiple of 16, which would
    # change the geometry of videos that get judged by eye.
    with imageio.get_writer(filename, fps=fps, macro_block_size=1,
                            codec=kwargs.pop("video_codec", "libx264")) as w:
        for frame in video_array:
            w.append_data(frame)
