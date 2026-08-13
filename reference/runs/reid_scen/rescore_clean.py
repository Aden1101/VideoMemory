#!/usr/bin/env python
# ============================================================================================
# S2 / S2p RE-SCORING AGAINST A CLEAN REFERENCE  +  INSTANCE TEST   (audit 2026-08-02)
#
# WHY. In S2/S2p the subject A does NOT open the video, and with local_attn_size 12 +
# sink_size 1 + global_sink the opening scene is pinned in the sink for the whole rollout. The
# segment the campaign scores against ("A's first appearance") therefore shows the OPENER, not A:
# measured at cos 0.165 from a clean rendering of A. Every arm that actually resurrects A is
# punished, every arm that re-imports the opener is rewarded. See section 8 of
# reference/scenarios_design_report.md.
#
# THE CLEAN REFERENCE. A* = segment 0 of the S3 rollout of the SAME cell (same triple, same
# seed -- cell_seed does not depend on the scenario), where A OPENS the video and nothing
# precedes it. Segment 0 is produced before any switch, so it is identical across the arms of a
# given pipeline; the MAG arms are scored against the MAG A*, `stockoff` against its OWN A*
# (different checkpoint => a MAG reference would confound pipeline with arm).
#
# THE INSTANCE TEST. cos(revisit, A* of THIS seed) vs cos(revisit, A* of the OTHER seed of the
# same triple = same prompt, different instance). Equal => the "return of A" is a re-render from
# the prompt (FiLM conditioning), not a recall. Same-seed > other-seed => real instance recall.
#
#   python rescore_clean.py --triples 0,1,2,3           # shard
#   python rescore_clean.py --mode aggregate            # tables from the shard JSONs
# ============================================================================================
import argparse
import glob
import json
import os
import sys

REPO = "/shared/ssd_14T/home/adenelmi/Anchor-Forcing"
HERE = os.path.dirname(os.path.abspath(__file__))
PROBE_DIR = os.path.join(REPO, "reference/runs/reid_probe_l1")
sys.path[:0] = [REPO, PROBE_DIR, HERE]

VID = "/shared/ssd_14T_project/adenelmi/reid_scen/videos"
OUT = os.path.join(HERE, "results", "clean_rescore")
SCENS = ("S2", "S2p")
# segment 0 of S3 is produced before any switch => identical for every MAG arm; memzero is the
# one that exists for all 24 cells. stockoff carries its own pipeline, hence its own reference.
REF_ARM = {"stockoff": "stockoff"}     # arm -> arm whose S3 segment 0 is its clean A*


def _shard_path(tag):
    return os.path.join(OUT, f"shard_{tag}.json")


# ----------------------------------------------------------------------------------------
def measure(triples, seeds, device):
    import torch
    from torchvision.io import read_video
    from run_reid_probe import dino_embed, sample_seg, center_crop, seg_cos
    import scenarios as S

    w_rev = S.score_segments("S2", 240)[S.revisit_segments("S2")[0]]   # (192, 239)
    w_fa = S.score_segments("S2", 240)[1]                              # contaminated reference
    w_clean = S.score_segments("S3", 300)[0]                           # (20, 72), A opens

    cache = {}

    def emb(path, lo, hi):
        """(full, crop60) DINO embeddings of the latent window [lo, hi], verbatim
        run_reid_scen.score_video_scen conventions (sample_seg, stride 8)."""
        key = (path, lo, hi)
        if key in cache:
            return cache[key]
        assert os.path.exists(path), f"missing video {path}"
        v, _, _ = read_video(path, pts_unit="sec", output_format="THWC")
        vid = v.permute(0, 3, 1, 2).float() / 255.
        seg = sample_seg(vid, lo, hi, 8)
        out = (dino_embed(seg, device), dino_embed(center_crop(seg), device))
        cache[key] = out
        return out

    res = {"cells": {}}
    for ti in triples:
        for si in seeds:
            cell = f"t{ti}_s{si}"
            rec = {"triple": ti, "seed_idx": si}
            # clean A*, this seed and the other seed of the same triple, per pipeline
            other = 1 - si if si in (0, 1) else None
            clean = {}
            for pipe, arm in (("mag", "memzero"), ("stock", "stockoff")):
                clean[pipe] = emb(f"{VID}/S3_t{ti}_s{si}_{arm}.mp4", *w_clean)
                if other is not None:
                    clean[pipe + "_other"] = emb(f"{VID}/S3_t{ti}_s{other}_{arm}.mp4", *w_clean)
            for scen in SCENS:
                arms = [a for a in S.DEFAULT_ARMS[scen]]
                sc = {}
                # the contaminated reference itself, vs clean A* (identical across MAG arms)
                fa = emb(f"{VID}/{scen}_t{ti}_s{si}_memzero.mp4", *w_fa)
                sc["_contaminated_ref_vs_cleanA"] = {
                    "full": seg_cos(fa[0], clean["mag"][0]),
                    "crop60": seg_cos(fa[1], clean["mag"][1])}
                for a in arms:
                    p = f"{VID}/{scen}_t{ti}_s{si}_{a}.mp4"
                    if not os.path.exists(p):
                        continue
                    e = emb(p, *w_rev)
                    pipe = "stock" if REF_ARM.get(a) else "mag"
                    row = {"full": seg_cos(e[0], clean[pipe][0]),
                           "crop60": seg_cos(e[1], clean[pipe][1]),
                           "ref_pipeline": pipe}
                    if other is not None:
                        row["full_other_seed"] = seg_cos(e[0], clean[pipe + "_other"][0])
                        row["crop60_other_seed"] = seg_cos(e[1], clean[pipe + "_other"][1])
                    sc[a] = row
                rec[scen] = sc
            res["cells"][cell] = rec
            # the cache holds DINO EMBEDDINGS (a few frames x 768), not videos: keeping it
            # across cells is what makes the other-seed reference of one cell free for the next.
            print(f"[rescore] {cell} done ({len(cache)} embeddings cached)", flush=True)
    return res


# ----------------------------------------------------------------------------------------
def aggregate():
    import statistics as st
    from run_reid_probe import sign_test_p
    import scenarios as S

    cells = {}
    for p in sorted(glob.glob(_shard_path("*"))):
        with open(p) as f:
            cells.update(json.load(f)["cells"])
    assert cells, f"no shard JSON in {OUT}"
    print(f"\n{len(cells)} cells aggregated from {len(glob.glob(_shard_path('*')))} shards")

    # ---- the old (contaminated) numbers, straight from the campaign score JSONs ------------
    old = {}
    for scen in SCENS:
        for p in glob.glob(os.path.join(HERE, "results", f"{scen}_t*_s*_scores.json")):
            with open(p) as f:
                c = json.load(f)
            key = os.path.basename(p).replace(f"{scen}_", "").replace("_scores.json", "")
            for a in S.DEFAULT_ARMS[scen]:
                if a in c:
                    old.setdefault((scen, a), {})[key] = c[a]["full"]["reid_resA_origA"]

    summary = {}
    for scen in SCENS:
        arms = [a for a in S.DEFAULT_ARMS[scen]]
        print(f"\n================ {scen}: re-ID vs CLEAN A* (segment 0 of S3, same cell/seed) "
              f"================")
        ref = [c[scen]["_contaminated_ref_vs_cleanA"]["full"] for c in cells.values()]
        refc = [c[scen]["_contaminated_ref_vs_cleanA"]["crop60"] for c in cells.values()]
        print(f"  [reference sanity] the OLD reference ('A's first appearance') vs CLEAN A*: "
              f"full {st.mean(ref):.4f}  crop60 {st.mean(refc):.4f}   <-- this is what the "
              f"campaign scored against")
        summary[scen] = {"n_cells": len(cells),
                         "contaminated_ref_vs_cleanA": {"full": st.mean(ref),
                                                        "crop60": st.mean(refc)},
                         "variants": {}}
        for variant in ("full", "crop60"):
            base = {k: c[scen]["memzero"][variant] for k, c in cells.items()
                    if "memzero" in c[scen]}
            vs = {"arm_means": {}, "paired_vs_memzero": {}, "old_arm_means": {}}
            print(f"\n  [{variant}]  arm            CLEAN score      delta vs memzero"
                  f"        (OLD contaminated score / delta)")
            for a in arms:
                vals = {k: c[scen][a][variant] for k, c in cells.items() if a in c[scen]}
                if not vals:
                    continue
                m = st.mean(vals.values())
                vs["arm_means"][a] = m
                o = old.get((scen, a), {})
                om = st.mean(o.values()) if o else float("nan")
                vs["old_arm_means"][a] = om
                if a == "memzero":
                    print(f"        {a:17s} {m:.4f}          --                    "
                          f"({om:.4f} / --)")
                    continue
                d = [vals[k] - base[k] for k in vals if k in base]
                od = ([o[k] - old[(scen, 'memzero')][k] for k in o
                       if k in old.get((scen, 'memzero'), {})] if o else [])
                mm, npos = st.mean(d), sum(1 for x in d if x > 0)
                sd = st.stdev(d) if len(d) > 1 else 0.0
                p = sign_test_p(npos, len(d) - npos)
                vs["paired_vs_memzero"][a] = {
                    "mean_delta": mm, "sd": sd, "sem": sd / max(1, len(d)) ** 0.5,
                    "n": len(d), "n_pos": npos, "n_neg": len(d) - npos, "sign_p": p,
                    "old_mean_delta": st.mean(od) if od else float("nan"),
                    "deltas": {k: vals[k] - base[k] for k in vals if k in base}}
                print(f"        {a:17s} {m:.4f}          {mm:+.4f}  {npos:2d}+/{len(d)-npos:2d}- "
                      f"p={p:.4f}   ({om:.4f} / {st.mean(od) if od else float('nan'):+.4f})")
            summary[scen]["variants"][variant] = vs

    # ---- INSTANCE TEST -------------------------------------------------------------------
    print(f"\n================ INSTANCE TEST (S2): is the returned A THIS cell's A? "
          f"================")
    print("  same-seed A* vs other-seed A* (same prompt, different instance), paired per cell")
    inst = {}
    for scen in SCENS:
        for a in S.DEFAULT_ARMS[scen]:
            for variant in ("full", "crop60"):
                rows = [(c[scen][a][variant], c[scen][a].get(variant + "_other_seed"))
                        for c in cells.values()
                        if a in c[scen] and c[scen][a].get(variant + "_other_seed") is not None]
                if not rows:
                    continue
                d = [x - y for x, y in rows]
                npos = sum(1 for x in d if x > 0)
                p = sign_test_p(npos, len(d) - npos)
                sd = st.stdev(d) if len(d) > 1 else 0.0
                inst[f"{scen}_{a}_{variant}"] = {
                    "same_seed": st.mean(x for x, _ in rows),
                    "other_seed": st.mean(y for _, y in rows),
                    "mean_delta": st.mean(d), "sd": sd,
                    "sem": sd / max(1, len(d)) ** 0.5, "n": len(d),
                    "n_pos": npos, "n_neg": len(d) - npos, "sign_p": p}
                if variant == "full":
                    r = inst[f"{scen}_{a}_{variant}"]
                    print(f"    {scen:4s} {a:17s} same-seed {r['same_seed']:.4f}   "
                          f"other-seed {r['other_seed']:.4f}   delta {r['mean_delta']:+.4f} "
                          f"+-{r['sem']:.4f}   {r['n_pos']}+/{r['n_neg']}-  p={r['sign_p']:.4f}")
    summary["instance_test"] = inst

    out = os.path.join(OUT, "summary_clean_rescore.json")
    with open(out, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[rescore] wrote {out}")
    return summary


def power(device):
    """POWER CHECK for the instance test: are two SEEDS of the same prompt even distinguishable
    by this metric? For each triple, compare
        between = cos(A* seed0, A* seed1)        two different instances of the same prompt
        within  = split-half cos inside each A*  the same instance, two halves of one segment
    If between ~ within, DINO cannot tell the two instances apart and the instance test is void
    by construction; a clear gap means the test had headroom and a null result is informative."""
    import statistics as st
    import torch
    from torchvision.io import read_video
    from run_reid_probe import dino_embed, sample_seg, seg_cos
    import scenarios as S
    w = S.score_segments("S3", 300)[0]

    def e(ti, si):
        v, _, _ = read_video(f"{VID}/S3_t{ti}_s{si}_memzero.mp4", pts_unit="sec",
                             output_format="THWC")
        return dino_embed(sample_seg(v.permute(0, 3, 1, 2).float() / 255., *w, 8), device)

    rows = []
    for ti in range(len(S.TRIPLES)):
        a, b = e(ti, 0), e(ti, 1)
        h = max(1, a.shape[0] // 2)
        rows.append({"triple": ti, "between": seg_cos(a, b),
                     "within": 0.5 * (seg_cos(a[:h], a[h:]) + seg_cos(b[:h], b[h:]))})
        print(f"  triple {ti:2d}: between-seed {rows[-1]['between']:.4f}   "
              f"within-seed (split-half) {rows[-1]['within']:.4f}   "
              f"headroom {rows[-1]['within'] - rows[-1]['between']:+.4f}", flush=True)
    bt = st.mean(r["between"] for r in rows)
    wt = st.mean(r["within"] for r in rows)
    print(f"\n  MEAN between-seed A* {bt:.4f}   within-seed A* {wt:.4f}   "
          f"HEADROOM {wt - bt:+.4f}  ({sum(1 for r in rows if r['within'] > r['between'])}"
          f"/{len(rows)} triples with within > between)")
    out = os.path.join(OUT, "instance_power.json")
    with open(out, "w") as f:
        json.dump({"per_triple": rows, "mean_between": bt, "mean_within": wt}, f, indent=1)
    print(f"[rescore] wrote {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["measure", "aggregate", "power"], default="measure")
    ap.add_argument("--triples", type=str, default="0,1,2,3,4,5,6,7,8,9,10,11")
    ap.add_argument("--seeds", type=str, default="0,1")
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    if args.mode == "aggregate":
        aggregate()
        return
    import torch
    torch.set_grad_enabled(False)
    if args.mode == "power":
        power(torch.device("cuda"))
        return
    triples = [int(x) for x in args.triples.split(",") if x != ""]
    seeds = [int(x) for x in args.seeds.split(",") if x != ""]
    res = measure(triples, seeds, torch.device("cuda"))
    tag = "-".join(str(t) for t in triples)
    with open(_shard_path(tag), "w") as f:
        json.dump(res, f, indent=1)
    print(f"[rescore] wrote {_shard_path(tag)}")


if __name__ == "__main__":
    main()
