"""Objective test of the S2 metric pathology: is the 'first appearance of A' segment
actually a picture of A, or a picture of the OPENER (sink) scene?"""
import sys, os, json, glob, statistics as st, torch
REPO = "/shared/ssd_14T/home/adenelmi/Anchor-Forcing"
sys.path[:0] = [REPO, REPO + "/reference/runs/reid_probe_l1", REPO + "/reference/runs/reid_scen"]
from torchvision.io import read_video
from run_reid_probe import dino_embed, sample_seg, seg_cos
import scenarios as S

V = "/shared/ssd_14T_project/adenelmi/reid_scen/videos"
dev = torch.device("cuda")
out = {}
for scen in ("S2", "S3"):
    wins = S.score_segments(scen, S.SCENARIOS[scen]["frames"])
    fa = S.first_appearance(scen); roles = S.seg_roles(scen); rev = S.revisit_segments(scen)[0]
    src = fa[roles[rev]]
    op = 0
    for arm in ("memzero", "memdec20", "framebank"):
        rows = []
        for p in sorted(glob.glob(f"{V}/{scen}_t*_s*_{arm}.mp4")):
            v, _, _ = read_video(p, pts_unit="sec", output_format="THWC")
            vid = v.permute(0, 3, 1, 2).float() / 255.
            E = {i: dino_embed(sample_seg(vid, *wins[i], 8), dev) for i in wins}
            rows.append([seg_cos(E[src], E[op]), seg_cos(E[rev], E[op]), seg_cos(E[rev], E[src])])
            del vid, v, E
        out[f"{scen}_{arm}"] = rows
        print(f"{scen} {arm:10s} n={len(rows)}  cos(firstA,opener)={st.mean(r[0] for r in rows):.3f}"
              f"  cos(revisit,opener)={st.mean(r[1] for r in rows):.3f}"
              f"  cos(revisit,firstA)={st.mean(r[2] for r in rows):.3f}", flush=True)
json.dump(out, open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                 "scene_contamination.json"), "w"), indent=1)
print("DONE")
