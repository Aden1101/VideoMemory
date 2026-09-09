# ============================================================================================
# ECRITURE DE VIDEO -- une seule porte d'entree.
#
# 2026-09-09. `torchvision.io.write_video` a ete deprecie puis RETIRE des versions recentes de
# torchvision (il n'existe plus a partir de la 0.24). Le depot l'importe en haut de quatre
# fichiers, donc sur une installation recente le chemin d'inference casse a l'import, avec un
# message qui ne dit rien du probleme reel.
#
# Ce module garde torchvision quand il est la (comportement et encodage strictement inchanges
# sur une installation conforme au README, torchvision 0.21) et retombe sinon sur imageio, qui
# est deja dans requirements.txt. Rien a epingler, rien a reinstaller.
#
# Contrat commun aux deux voies : `frames` est un tenseur (T, H, W, C) en uint8, comme
# torchvision l'exige.
# ============================================================================================
import numpy as np
import torch

try:                                     # torchvision < 0.24
    from torchvision.io import write_video as _tv_write_video
    _BACKEND = "torchvision"
except ImportError:                      # torchvision >= 0.24, ou torchvision absent
    _tv_write_video = None
    _BACKEND = "imageio"


def backend():
    return _BACKEND


def write_video(filename, video_array, fps, **kwargs):
    """Ecrit un mp4. Signature volontairement identique a torchvision.io.write_video."""
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
        f"attendu (T, H, W, C) uint8, recu {video_array.shape}"
    # macro_block_size=1 : sans ca imageio redimensionne en silence vers un multiple de 16,
    # ce qui changerait la geometrie des videos jugees a l'oeil.
    with imageio.get_writer(filename, fps=fps, macro_block_size=1,
                            codec=kwargs.pop("video_codec", "libx264")) as w:
        for frame in video_array:
            w.append_data(frame)
