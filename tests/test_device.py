#!/usr/bin/env python
# ============================================================================================
# La passe "agnostique au peripherique" du 2026-09-09, verifiee.
#
# LA PROMESSE A TENIR, et c'est la seule qui compte : sur CUDA (et sur CPU) le code doit se
# comporter EXACTEMENT comme avant. Le depot a deja perdu des campagnes sur des comparaisons
# dont un bras n'etait pas ce qu'on croyait, et le controle de device du 07/08 a mesure que
# changer de machine deplace le resultat de deux paires sur dix-neuf. Un patch de portabilite
# qui bougerait la moindre valeur sur le materiel d'origine serait pire que pas de patch.
#
# D3 est donc le test central : il reimplemente le RoPE d'AVANT le patch et exige une egalite
# BIT A BIT avec la version actuelle sur un peripherique qui supporte le float64.
#
#   python tests/test_device.py
# ============================================================================================
import os
import sys

import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils.device import (Timer, describe, device, free_memory_gb,  # noqa: E402
                          phase_dtype, supports_float64, synchronize, total_memory_gb)
from wan.modules.causal_model import causal_rope_apply  # noqa: E402
from wan.modules.model import rope_freqs_to, rope_params, rope_phase_dtype  # noqa: E402

OK = []


def check(name, cond, detail=""):
    assert cond, f"FAIL {name} {detail}"
    OK.append(name)
    print(f"  ok  {name}")


print(describe())

# ----------------------------- la reference : le code d'AVANT le patch -----------------------
def causal_rope_apply_original(x, grid_sizes, freqs, start_frame=0):
    """Copie verbatim de la version pre-2026-09-09 (float64 en dur)."""
    n, c = x.size(2), x.size(3) // 2
    freqs = freqs.split([c - 2 * (c // 3), c // 3, c // 3], dim=1)
    output = []
    for i, (f, h, w) in enumerate(grid_sizes.tolist()):
        seq_len = f * h * w
        x_i = torch.view_as_complex(x[i, :seq_len].to(torch.float64).reshape(seq_len, n, -1, 2))
        freqs_i = torch.cat([
            freqs[0][start_frame:start_frame + f].view(f, 1, 1, -1).expand(f, h, w, -1),
            freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
            freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1)], dim=-1).reshape(seq_len, 1, -1)
        x_i = torch.view_as_real(x_i * freqs_i).flatten(2)
        x_i = torch.cat([x_i, x[i, seq_len:]])
        output.append(x_i)
    return torch.stack(output).type_as(x)


HEADS, HEAD_DIM = 2, 16
F, H, W = 3, 4, 5
SEQ = F * H * W
torch.manual_seed(0)
d = HEAD_DIM
freqs_cpu = torch.cat([rope_params(1024, d - 4 * (d // 6)),
                       rope_params(1024, 2 * (d // 6)),
                       rope_params(1024, 2 * (d // 6))], dim=1)
x_cpu = torch.randn(1, SEQ, HEADS, HEAD_DIM)
grid = torch.tensor([[F, H, W]])

# --- D1: le peripherique se resout sans jamais interroger un pilote absent -------------------
check("D1 peripherique resolu", device.type in ("cuda", "mps", "cpu"), device.type)
check("D1b memoire lisible", total_memory_gb() > 0 and free_memory_gb() >= 0,
      f"{free_memory_gb():.1f}/{total_memory_gb():.1f}")

# --- D2: le float64 est conserve partout ou il existe ----------------------------------------
check("D2 CPU garde le float64", phase_dtype(torch.device("cpu")) is torch.float64)
check("D2b CUDA garderait le float64", supports_float64(torch.device("cuda")))
check("D2c Metal seul retombe en float32", phase_dtype(torch.device("mps")) is torch.float32)
check("D2d la table CPU reste en complex128",
      rope_freqs_to(freqs_cpu, torch.device("cpu")).dtype is torch.complex128)
check("D2e rope_phase_dtype suit la table",
      rope_phase_dtype(freqs_cpu) is torch.float64
      and rope_phase_dtype(freqs_cpu.to(torch.complex64)) is torch.float32)

# --- D3: EGALITE BIT A BIT avec le code d'avant, sur un peripherique float64 -----------------
ref = causal_rope_apply_original(x_cpu, grid, freqs_cpu, start_frame=2)
new = causal_rope_apply(x_cpu, grid, freqs_cpu, start_frame=2)
check("D3 RoPE identique AU BIT PRES au code d'origine (CPU)", torch.equal(ref, new),
      f"max|diff| = {float((ref - new).abs().max())}")
ref0 = causal_rope_apply_original(x_cpu, grid, freqs_cpu)
check("D3b idem sans decalage de frame", torch.equal(ref0, causal_rope_apply(x_cpu, grid, freqs_cpu)))

# --- D4: sur Metal, ca tourne, et ca tombe au bon endroit ------------------------------------
if device.type == "mps":
    fq = rope_freqs_to(freqs_cpu, device)
    check("D4 la table est abaissee en complex64 pour Metal", fq.dtype is torch.complex64)
    got = causal_rope_apply(x_cpu.to(device), grid, fq, start_frame=2).cpu()
    err = float((got - ref).abs().max())
    check("D4b le RoPE s'execute sur Metal", got.shape == ref.shape)
    # tolerance : c'est un ecart float32-vs-float64 sur une rotation de norme 1, donc de
    # l'ordre de l'epsilon float32 relatif a l'amplitude de x, pas un ecart de methode.
    check("D4c ecart conforme a une simple perte de precision", err < 1e-4, f"max|diff| = {err:.2e}")
    check("D4d la norme est preservee (c'est une rotation)",
          abs(float(got.norm()) - float(ref.norm())) < 1e-3)
    check("D4e float64 refuse sur Metal, comme prevu",
          not supports_float64(device))
else:
    print("  --  D4 ignore (pas de peripherique Metal ici)")

# --- D5: le chronometre rend des millisecondes, quelle que soit la voie -----------------------
a, b = Timer(), Timer()
a.record()
_ = torch.randn(600, 600, device=device) @ torch.randn(600, 600, device=device)
b.record()
synchronize()
ms = a.elapsed_time(b)
check("D5 chronometre en millisecondes", 0.0 <= ms < 60000.0, f"{ms} ms")

# --- D6: l'ecriture video a une voie utilisable ------------------------------------------------
from utils.video_io import backend, write_video  # noqa: E402
import tempfile  # noqa: E402
with tempfile.TemporaryDirectory() as tmp:
    p = os.path.join(tmp, "t.mp4")
    write_video(p, torch.randint(0, 255, (6, 32, 32, 3), dtype=torch.uint8), fps=8)
    check(f"D6 ecriture video via {backend()}", os.path.getsize(p) > 0)

print(f"\n{len(OK)}/{len(OK)} tests peripherique PASS")
