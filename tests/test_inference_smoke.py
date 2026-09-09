#!/usr/bin/env python
# ============================================================================================
# FUMEE DU CHEMIN D'INFERENCE, sur le peripherique local, sans poids ni donnees.
#
# Un VRAI CausalWanModel miniature (poids aleatoires, geometrie de tetes reelle 12 x 128 parce
# que le cache KV du pipeline l'impose), deroule en STREAMING sur trois chunks. On ne juge
# aucune image : on verifie que la chaine s'execute et que le cache avance -- RoPE, attention,
# cache KV roulant, sink, cellule memoire TTT.
#
# Ce test existe parce que la passe "agnostique au peripherique" du 2026-09-09 a fait tomber
# quatre murs successifs qu'aucun test d'import n'aurait vus : le float64 du RoPE, celui du
# plongement temporel, cinq appels d'attention qui contournaient le repartiteur, et l'ecriture
# video. Chacun ne se revele qu'a l'execution, un cran plus loin que le precedent.
#
#   python tests/test_inference_smoke.py            # peripherique detecte
#   AF_DEVICE=cpu python tests/test_inference_smoke.py
# ============================================================================================
import os
import sys
import time

import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils.device import describe, device, synchronize  # noqa: E402
from wan.modules.causal_model import CausalWanModel  # noqa: E402

HEADS, HEAD_DIM = 12, 128          # impose par _initialize_kv_cache du pipeline
DIM = HEADS * HEAD_DIM
LAYERS, F, H, W = 2, 3, 16, 16
FRAME_SEQ = (H // 2) * (W // 2)
LOCAL, CHUNKS = 12, 3
DT = torch.bfloat16                # le pipeline reel tourne en bf16 ; la memoire TTT y descend

OK = []


def check(name, cond, detail=""):
    assert cond, f"FAIL {name} {detail}"
    OK.append(name)
    print(f"  ok  {name}")


def main():
    print(describe())
    torch.manual_seed(0)
    m = CausalWanModel(dim=DIM, ffn_dim=2048, num_heads=HEADS, num_layers=LAYERS, text_dim=64,
                       text_len=16, local_attn_size=LOCAL, sink_size=3, memory_layers=(1,),
                       in_dim=16, out_dim=16).to(device=device, dtype=DT).eval()
    print(f"  modele miniature : {sum(p.numel() for p in m.parameters())/1e6:.1f} M params, "
          f"{LAYERS} couches, {FRAME_SEQ} tokens/frame")

    kv = [{"k": torch.zeros(1, LOCAL * FRAME_SEQ, HEADS, HEAD_DIM, dtype=DT, device=device),
           "v": torch.zeros(1, LOCAL * FRAME_SEQ, HEADS, HEAD_DIM, dtype=DT, device=device),
           "global_end_index": torch.tensor([0], dtype=torch.long, device=device),
           "local_end_index": torch.tensor([0], dtype=torch.long, device=device)}
          for _ in range(LAYERS)]
    xa = [{"k": torch.zeros(1, 512, HEADS, HEAD_DIM, dtype=DT, device=device),
           "v": torch.zeros(1, 512, HEADS, HEAD_DIM, dtype=DT, device=device),
           "is_init": False} for _ in range(LAYERS)]

    # La tete de sortie est zero-init (causal_model.py:1780, la regle "jamais de bruit dans le
    # cache"), donc un modele neuf rend EXACTEMENT zero. On la perturbe, sinon S4 et S5
    # testeraient 0 == 0 et ne verraient plus jamais une regression du chemin.
    with torch.no_grad():
        m.head.head.weight.normal_(0, 0.02)

    ctx = [torch.randn(16, 64, device=device, dtype=DT)]
    t = torch.ones(1, F, device=device, dtype=DT) * 1000.0     # un pas de temps PAR frame
    ends, outs = [], []

    with torch.no_grad():
        for c in range(CHUNKS):
            x = [torch.randn(16, F, H, W, device=device, dtype=DT)]
            t0 = time.time()
            out = m(x, t=t, context=ctx, seq_len=F * FRAME_SEQ, kv_cache=kv, crossattn_cache=xa,
                    current_start=c * F * FRAME_SEQ)
            synchronize()
            ends.append(int(kv[0]["global_end_index"]))
            outs.append(out)
            print(f"  .. chunk {c} : {tuple(out[0].shape)} en {(time.time()-t0)*1000:5.0f} ms, "
                  f"cache a {ends[-1]} tokens")

    check("S1 la forme de sortie est celle de l'entree",
          tuple(outs[0][0].shape) == (16, F, H, W), str(tuple(outs[0][0].shape)))
    check("S2 aucune valeur non finie", all(bool(torch.isfinite(o).all()) for o in outs))
    check("S3 le cache KV avance d'un chunk a l'autre",
          ends == [(c + 1) * F * FRAME_SEQ for c in range(CHUNKS)], str(ends))
    check("S4 la sortie n'est pas identiquement nulle (tete perturbee)",
          float(outs[0].abs().max()) > 0)
    check("S5 deux chunks differents donnent des sorties differentes",
          not torch.equal(outs[0], outs[1]))
    check("S6 le cache d'attention croisee a bien ete initialise", xa[0]["is_init"])

    print(f"\n{len(OK)}/{len(OK)} fumee inference PASS sur {device}")


if __name__ == "__main__":
    main()
