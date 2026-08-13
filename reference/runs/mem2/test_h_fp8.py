"""h-fp8 tolerance test (memoire 2.0, gate collecte vague 2).

Question: si on stockait les h en float8_e4m3fn au lieu de fp16 (disque /2), les etats
finaux apres replay et l'ancre decodee changent-ils de facon mesurable ?
Protocole: N videos des shards v3 existants, replay_writes sur (a) h fp16 tel quel,
(b) h roundtrip fp16->fp8e4m3->fp16, cellules l1@2000-EMA (build_cells 'direct'), decodeur
v1.2. Verdict PASS si rel-L2 etat < 1e-2 et rel-L1 decode < 5e-2 partout (la maison a
mesure une tolerance dose sigma=0.6 -> 89% du gain cote ancres, ces seuils sont 10x plus
serres que ce que l'eval tolere).
"""
import glob
import os
import sys

import torch
from omegaconf import OmegaConf

sys.path.insert(0, "/shared/ssd_14T/home/adenelmi/Anchor-Forcing")
sys.path.insert(0, "/shared/ssd_14T/home/adenelmi/Anchor-Forcing/reference/runs/mem2")
sys.path.insert(0, "/shared/ssd_14T/home/adenelmi/Anchor-Forcing/reference/runs/reid_probe_l1")
sys.path.insert(0, "/shared/ssd_14T/home/adenelmi/Anchor-Forcing/reference/runs/frame_dec")

from wan.modules.video_memory import AnchorFrameDecoder, replay_writes  # noqa: E402
from train_writer import DEC_V12, DEFAULT_CONFIG, build_cells  # noqa: E402

V3_DIR = "/shared/ssd_14T_project/adenelmi/mem2_h_v3"
N_VIDEOS = 6


def main():
    device = torch.device("cuda")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    config = OmegaConf.load(DEFAULT_CONFIG)
    cells, blocks, _ = build_cells(config, device, mode="direct")

    decoder = AnchorFrameDecoder(cond_dim=4096, cond_hidden=512,
                                 detach_states=False).to(device)
    ck = torch.load(DEC_V12, map_location="cpu", weights_only=False)
    decoder.load_state_dict(ck["decoder"] if "decoder" in ck else ck, strict=True)
    decoder.eval()

    vids = []
    for p in sorted(glob.glob(os.path.join(V3_DIR, "hreplay_*_part000.pt"))):
        d = torch.load(p, map_location="cpu", weights_only=False)
        vids.extend(d["videos"] if isinstance(d, dict) and "videos" in d else d)
        if len(vids) >= N_VIDEOS:
            break
    vids = vids[:N_VIDEOS]
    print(f"[fp8] {len(vids)} videos, blocks {blocks}, decoder v1.2, cells l1@2000-EMA")

    worst_state, worst_dec = 0.0, 0.0
    for vi, v in enumerate(vids):
        rels, coss = [], []
        states_a, states_b = [], []
        for bi in blocks:
            hs = [t.to(device).to(torch.bfloat16) for t in v["h_seq"][str(bi)]]
            hs8 = [t.to(device).to(torch.float8_e4m3fn).to(torch.bfloat16)
                   for t in v["h_seq"][str(bi)]]
            with torch.no_grad():
                sa = replay_writes(cells[bi], hs)
                sb = replay_writes(cells[bi], hs8)
            states_a.append(sa)
            states_b.append(sb)
            for a, b in zip(sa, sb):
                a32, b32 = a.float(), b.float()
                rels.append(((a32 - b32).norm() / (a32.norm() + 1e-8)).item())
                coss.append(torch.nn.functional.cosine_similarity(
                    a32.flatten(), b32.flatten(), dim=0).item())
        with torch.no_grad():
            fa = decoder(states_a, cond=None)
            fb = decoder(states_b, cond=None)
        dec_rel = ((fa - fb).abs().mean() / (fa.abs().mean() + 1e-8)).item()
        worst_state = max(worst_state, max(rels))
        worst_dec = max(worst_dec, dec_rel)
        print(f"[fp8] vid{vi} ({v['mode']}, {len(v['h_seq'][str(blocks[0])])} writes): "
              f"state rel-L2 max {max(rels):.2e} cos min {min(coss):.6f} "
              f"decode rel-L1 {dec_rel:.2e}")
    ok = worst_state < 1e-2 and worst_dec < 5e-2
    print(f"[fp8] worst state rel-L2 {worst_state:.2e}, worst decode rel-L1 {worst_dec:.2e}"
          f" -> {'PASS: collecte vague 2 en fp8 (disque /2)' if ok else 'FAIL: rester fp16'}")


if __name__ == "__main__":
    main()
