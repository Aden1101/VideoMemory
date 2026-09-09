# ============================================================================================
# UNE SEULE SOURCE DE VERITE POUR L'ACCELERATEUR.
#
# Ajoute le 2026-09-09. Le depot supposait partout que l'accelerateur etait une carte NVIDIA :
# le peripherique etait construit en interrogeant le pilote CUDA a l'import (utils/memory.py),
# les chronometres etaient des evenements CUDA, l'AMP venait de torch.cuda.amp, et le RoPE
# passait par du float64. Aucune de ces hypotheses n'est necessaire ; elles etaient juste
# dispersees dans six fichiers.
#
# CONTRAT, non negociable : sur CUDA, chaque fonction ci-dessous delegue a l'appel torch.cuda
# d'origine, a l'identique. Aucun chiffre deja mesure ne peut bouger. Ce module ne rend pas le
# code "compatible Mac", il retire une hypothese materielle -- Metal et le processeur en
# beneficient, une carte NVIDIA ne voit strictement aucune difference.
#
# Ce que ce module n'est PAS : une couche d'abstraction generale sur les accelerateurs. Il
# expose exactement les cinq choses que ce depot demandait au pilote, et rien de plus.
#
# Surcharge manuelle : AF_DEVICE=cpu|mps|cuda|cuda:1 dans l'environnement.
# ============================================================================================
import os
import time

import torch

cpu = torch.device("cpu")


def _detect():
    forced = os.environ.get("AF_DEVICE", "").strip()
    if forced:
        return torch.device(forced)
    if torch.cuda.is_available():
        return torch.device(f"cuda:{torch.cuda.current_device()}")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return cpu


# Resolu une fois, a l'import, comme l'ancien `gpu` -- mais sans jamais interroger un pilote
# absent. C'est la ligne qui faisait planter tout le chemin d'inference hors CUDA.
device = _detect()
gpu = device                      # ancien nom, garde pour ne rien casser en amont


def kind(dev=None):
    return (dev or device).type


def is_cuda(dev=None):
    return kind(dev) == "cuda"


def autocast_device_type(dev=None):
    """Le premier argument de torch.autocast. 'cpu' n'a pas de bf16 utile partout, mais c'est
    a l'appelant de decider d'activer ou non ; ici on ne fait que nommer le peripherique."""
    return kind(dev)


def supports_float64(dev=None):
    """Metal ne supporte PAS le float64, a aucun degre : la conversion leve une TypeError.
    Le RoPE de Wan calcule ses phases en float64 ; sur CUDA on garde ce dtype exactement (donc
    les memes chiffres qu'avant), ailleurs on retombe en float32."""
    return kind(dev) != "mps"


def phase_dtype(dev=None):
    """Le dtype des calculs de phase du RoPE. float64 partout ou il existe."""
    return torch.float64 if supports_float64(dev) else torch.float32


def high_precision_dtype(x=None):
    """Le dtype "haute precision" du depot : float64 partout ou il existe, float32 sur Metal.

    Accepte un tensor, un torch.device, une chaine, ou rien (le peripherique courant). Le code
    fait un `.double()` a plusieurs endroits ou la precision compte reellement -- conversion
    flux/x0, ajout de bruit du scheduler, pertes DMD -- parce qu'une erreur d'arrondi y est
    multipliee par 1/sigma quand sigma est petit. Sur CUDA et CPU on garde donc float64 et le
    calcul est celui d'origine ; Metal ne supportant pas du tout le float64, on y retombe en
    float32 et ce compromis est declare, pas subi. (2026-09-09)
    """
    if isinstance(x, torch.Tensor):
        dev = x.device
    elif x is None:
        dev = device
    else:
        dev = torch.device(x)
    return torch.float64 if dev.type != "mps" else torch.float32


def hp(x, dev=None):
    """`x` deplace sur `dev` (defaut : son propre peripherique) et converti en haute precision.

    L'ORDRE compte : convertir puis deplacer echouerait, parce qu'un tensor float64 ne peut pas
    etre transfere vers Metal. On deplace d'abord, on convertit ensuite.
    """
    target = torch.device(dev) if dev is not None else x.device
    return x.to(device=target, dtype=high_precision_dtype(target))


def synchronize(dev=None):
    k = kind(dev)
    if k == "cuda":
        torch.cuda.synchronize(dev)
    elif k == "mps":
        torch.mps.synchronize()


def empty_cache(dev=None):
    k = kind(dev)
    if k == "cuda":
        torch.cuda.empty_cache()
    elif k == "mps":
        torch.mps.empty_cache()


def total_memory_gb(dev=None):
    k = kind(dev)
    if k == "cuda":
        return torch.cuda.get_device_properties(dev).total_memory / (1024 ** 3)
    if k == "mps":
        # le plafond de travail conseille par Metal, pas la RAM totale : c'est bien la
        # quantite qu'un processus peut esperer utiliser sur une memoire unifiee
        return torch.mps.recommended_max_memory() / (1024 ** 3)
    return float(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")) / (1024 ** 3)


def free_memory_gb(dev=None):
    """Memoire disponible en Go.

    CUDA : le calcul d'origine, mot pour mot (libre selon le pilote + reserve inactive de
    l'allocateur de PyTorch, qui est reutilisable sans passer par le pilote).
    Metal : `mem_get_info` n'existe pas ; on prend le plafond conseille moins ce que
    l'allocateur detient reellement. C'est une approximation, plus prudente que la verite,
    et elle sert au meme usage : decider s'il faut deplacer des modules.
    """
    k = kind(dev)
    if k == "cuda":
        stats = torch.cuda.memory_stats(dev)
        inactive_reserved = stats["reserved_bytes.all.current"] - stats["active_bytes.all.current"]
        free_driver, _ = torch.cuda.mem_get_info(dev)
        return (free_driver + inactive_reserved) / (1024 ** 3)
    if k == "mps":
        used = torch.mps.driver_allocated_memory() if hasattr(torch.mps, "driver_allocated_memory") \
            else torch.mps.current_allocated_memory()
        return max(0.0, torch.mps.recommended_max_memory() - used) / (1024 ** 3)
    return total_memory_gb(dev)


class Timer:
    """Chronometre a l'interface des evenements CUDA (record / elapsed_time en ms).

    CUDA et Metal ont tous deux de vrais evenements ; ailleurs on retombe sur l'horloge murale
    apres synchronisation. Le point important est que `elapsed_time` rende TOUJOURS des
    millisecondes, quelle que soit la voie : un chronometre qui change d'unite selon la machine
    est pire que pas de chronometre.
    """

    __slots__ = ("_dev", "_impl", "_t")

    def __init__(self, enable_timing=True, dev=None):
        self._dev = dev or device
        k = self._dev.type
        if k == "cuda":
            self._impl = torch.cuda.Event(enable_timing=enable_timing)
        elif k == "mps" and hasattr(torch.mps, "Event"):
            self._impl = torch.mps.Event(enable_timing=enable_timing)
        else:
            self._impl = None
        self._t = None

    def record(self):
        if self._impl is not None:
            self._impl.record()
        else:
            synchronize(self._dev)
            self._t = time.perf_counter()

    def elapsed_time(self, end):
        """Millisecondes entre ce chronometre et `end` (meme sens que torch.cuda.Event)."""
        if self._impl is not None:
            return self._impl.elapsed_time(end._impl)
        return (end._t - self._t) * 1000.0


def describe():
    """Une ligne pour les journaux : ce qu'on a trouve et ce qu'on va utiliser."""
    k = kind()
    if k == "cuda":
        name = torch.cuda.get_device_name(device)
    elif k == "mps":
        name = "Apple GPU (Metal)"
    else:
        name = "CPU"
    return (f"peripherique={device} ({name}) | memoire {free_memory_gb():.1f}/"
            f"{total_memory_gb():.1f} Go | float64 "
            f"{'oui' if supports_float64() else 'non (RoPE en float32)'}")
