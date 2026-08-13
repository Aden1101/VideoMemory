"""Is the S2 revisit segment a picture of A?

The S2 re-ID reference (segment 1 = A's 'first appearance') is polluted by the opener that the
attention sink pins. So score the S2 revisit of every arm against an UNPOLLUTED rendering of the
same subject with the same seed: segment 0 of the S3 rollout of the SAME (triple, seed) cell,
where A opens the video and nothing precedes it. cell_seed is scenario-independent, so S3 seg0 is
the canonical 'what A looks like for this cell'.
"""
import sys, os, json, glob, statistics as st, torch
REPO = "/shared/ssd_14T/home/adenelmi/Anchor-Forcing"
sys.path[:0] = [REPO, REPO + "/reference/runs/reid_probe_l1", REPO + "/reference/runs/reid_scen"]
from torchvision.io import read_video
from run_reid_probe import dino_embed, sample_seg, seg_cos
import scenarios as S

V = "/shared/ssd_14T_project/adenelmi/reid_scen/videos"
dev = torch.device("cuda")
w2 = S.score_segments("S2", 240)
w3 = S.score_segments("S3", 300)
REV = S.revisit_segments("S2")[0]                        # 3
FA = S.first_appearance("S2")[S.seg_roles("S2")[REV]]    # 1
FPS = 16.0


def emb_full(path, lo, hi):
    """DINO embedding of the scoring window [lo, hi] in LATENT frames, verbatim
    run_reid_scen.score_video_scen conventions (sample_seg + stride 8, full frame)."""
    v, _, _ = read_video(path, pts_unit="sec", output_format="THWC")
    return dino_embed(sample_seg(v.permute(0, 3, 1, 2).float() / 255., lo, hi, 8), dev)


cells = sorted({os.path.basename(p).split("_memzero")[0]
                for p in glob.glob(f"{V}/S2_t*_s*_memzero.mp4")})
E_clean, E_fa = {}, {}
for cell in cells:
    _, ti, si = cell.split("_")
    E_clean[cell] = emb_full(f"{V}/S3_{ti}_{si}_memzero.mp4", *w3[0])   # A opens, unpolluted
    E_fa[cell] = emb_full(f"{V}/{cell}_memzero.mp4", *w2[FA])           # arm-independent
print(f"cached clean-A and first-appearance embeddings for {len(cells)} cells", flush=True)

out = {"fa_vs_clean": [seg_cos(E_fa[c], E_clean[c]) for c in cells]}
print(f"S2 'first appearance of A' vs CLEAN A : {st.mean(out['fa_vs_clean']):.3f}", flush=True)
for arm in ("memzero", "memdec20", "framebank", "oracle1"):
    rows = []
    for cell in cells:
        p = f"{V}/{cell}_{arm}.mp4"
        if not os.path.exists(p):
            continue
        rows.append(seg_cos(emb_full(p, *w2[REV]), E_clean[cell]))
    out[arm] = rows
    print(f"S2 {arm:10s} n={len(rows)}  cos(revisit, CLEAN A) = {st.mean(rows):.3f}", flush=True)
json.dump(out, open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                 "clean_A_reference.json"), "w"), indent=1)
print("DONE")
