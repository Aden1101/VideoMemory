#!/usr/bin/env python
# ============================================================================================
# MANDATORY selftests for the scenario matrix + the Echo-Forcing-style frame-bank baseline.
#
#   (a) LEGACY BITWISE   : pipeline.inference_schedule() with the legacy [(0,A),(sw1,B),(sw2,A)]
#                          schedule == pipeline.inference() bitwise, on memzero / oracle /
#                          oracle1 / oracle3b / memdec20. This is what makes the whole thing
#                          "additive": the existing campaigns stay reproducible.
#   (b) FRAMEBANK == ORACLE1 : on A->B->A the pool holds a single A scene, so the bank ceiling
#                          (framebank_oracle) AND the real router (framebank) must splice the
#                          exact frame oracle1 splices -> bitwise equality, i.e. the baseline
#                          differs from the upper bound ONLY by its selection.
#   (c) NO ENV LEAK      : memzero replayed AFTER every other arm == the first memzero bitwise.
#   (d) FIRST-APPEARANCE SCORING : synthetic S2-shaped videos where the textures are known.
#                          Video 1 (honest) must score ~1; video 2 (segment 0 repainted with the
#                          revisit texture, first appearance repainted differently) must score
#                          LOW -- a "always compare to segment 0" bug scores ~1 there.
#   (g) MEM_DECODE SPLICE GATING : the decoded anchor must fire ONLY when a subject that has
#                          already been seen re-enters -- 0 splice on the opening switch and on
#                          every fresh scene, 1 in S2/S3, 2 in S4. Counted from the
#                          "[MEM_DECODE] decoded anchor spliced" lines of a real rollout. Also
#                          covers MEM_DECODE_ROUTED (arm memdec20r): same QUAND as framebank
#                          (its router's own decision), and BITWISE legacy when the variable is
#                          absent from the environment.
#   (e) py_compile        : every file touched.
#   (f) HELD-OUT GREP     : none of the 36 scenario prompts appears in the writer/decoder
#                          training corpus (nor in the v3 shards actually on disk).
#
# CPU-only subset:  python selftest_scenarios.py --cpu-only        (e, f, and the index math)
# Full (needs a GPU with >25 GB free):  python selftest_scenarios.py
# ============================================================================================
import argparse
import json
import os
import subprocess
import sys

REPO = "/shared/ssd_14T/home/adenelmi/Anchor-Forcing"
HERE = os.path.dirname(os.path.abspath(__file__))
PROBE_DIR = os.path.join(REPO, "reference/runs/reid_probe_l1")
FRAME_DEC = os.path.join(REPO, "reference/runs/frame_dec")
sys.path.insert(0, REPO)
sys.path.insert(0, PROBE_DIR)
sys.path.insert(0, FRAME_DEC)
sys.path.insert(0, HERE)

OUT = os.path.join(HERE, "selftest_out")
FRAMES, SW1, SW2 = 36, 12, 27          # same tiny rollout as reference/runs/mem2/selftest_probes

_RESULTS = []


def check(name, ok, detail=""):
    _RESULTS.append((name, bool(ok), detail))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f"  --  {detail}" if detail else ""),
          flush=True)
    assert ok, f"SELFTEST FAILED: {name} ({detail})"


# ============================================================================================
# (e) py_compile
# ============================================================================================
def test_e_compile():
    files = [os.path.join(REPO, "pipeline/switch_causal_inference.py"),
             os.path.join(HERE, "scenarios.py"),
             os.path.join(HERE, "run_reid_scen.py"),
             os.path.join(HERE, "selftest_scenarios.py"),
             os.path.join(PROBE_DIR, "run_reid_probe.py")]
    for f in files:
        r = subprocess.run([sys.executable, "-m", "py_compile", f],
                           capture_output=True, text=True)
        check(f"(e) py_compile {os.path.relpath(f, REPO)}", r.returncode == 0,
              r.stderr.strip()[:200])


# ============================================================================================
# (f) held-out grep: the scenario prompts must be absent from the training corpus
# ============================================================================================
def test_f_heldout():
    import scenarios as S
    from collect_pairs import find_l1_corpus_path, _norm_text
    prompts = [_norm_text(p) for p in S.scen_prompts()]
    check("(f) scenario prompts are unique", len(set(prompts)) == len(prompts),
          f"{len(prompts)} prompts, {len(set(prompts))} distinct")

    corpus = find_l1_corpus_path()
    hits = []
    with open(corpus, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            c = _norm_text(line)
            if not c:
                continue
            for p in prompts:
                if p in c or c in p:
                    hits.append((i, p[:60], c[:60]))
    check("(f) 0 scenario prompt hits the l1/v3 training corpus (both directions)",
          not hits, f"{len(prompts)} prompts vs {os.path.basename(corpus)}; hits={hits[:3]}")

    idx = "/shared/ssd_14T_project/adenelmi/mem2_runs/video_index.json"
    if os.path.exists(idx):
        with open(idx) as f:
            vids = json.load(f)
        vh = []
        for v in vids:
            for key in ("prompt", "prompt_b"):
                q = _norm_text(v.get(key) or "")
                if q and any(p in q or q in p for p in prompts):
                    vh.append((v.get("video_id"), key, q[:60]))
        check("(f) 0 scenario prompt hits the v3 shards actually on disk", not vh,
              f"{len(vids)} collected videos; hits={vh[:3]}")
    else:
        print(f"[skip] v3 index not found ({idx}), corpus grep only")

    # the 8 legacy pairs must survive verbatim inside TRIPLES 0-7 (cross-campaign comparability)
    from run_reid_probe import PAIRS
    ok = all(tuple(S.TRIPLES[i][:2]) == tuple(PAIRS[i]) for i in range(len(PAIRS)))
    check("(f) TRIPLES[0:8][:2] == the 8 legacy PAIRS verbatim", ok)


# ============================================================================================
# index math / schedule bookkeeping (pure python, no GPU)
# ============================================================================================
def test_schedule_math():
    import scenarios as S
    exp = {
        "S1": ([0, 1, 0], {0: 0, 1: 1}, [2]),
        "S2": ([2, 0, 1, 0], {2: 0, 0: 1, 1: 2}, [3]),
        "S3": ([0, 1, 2, 0], {0: 0, 1: 1, 2: 2}, [3]),
        "S4": ([0, 1, 0, 2, 0], {0: 0, 1: 1, 2: 3}, [2, 4]),
        # same-scene tier: same shapes as S2 / S1, run on SAMESCENE_TRIPLE_IDS
        "SS2": ([2, 0, 1, 0], {2: 0, 0: 1, 1: 2}, [3]),
        "SS1": ([0, 1, 0], {0: 0, 1: 1}, [2]),
    }
    for scen, (roles, fa, rev) in exp.items():
        check(f"(d0) {scen} roles/first-appearance/revisits",
              S.seg_roles(scen) == roles and S.first_appearance(scen) == fa
              and S.revisit_segments(scen) == rev,
              f"{S.seg_roles(scen)} {S.first_appearance(scen)} {S.revisit_segments(scen)}")
        nfpb = 3
        starts = [f for f, _ in S.SCENARIOS[scen]["schedule"]]
        check(f"(d0) {scen} start frames are multiples of {nfpb} and increasing",
              all(f % nfpb == 0 for f in starts) and starts == sorted(set(starts)))
        wins = S.score_segments(scen, S.SCENARIOS[scen]["frames"])
        bounds = S.seg_bounds(scen)
        ok = all(bounds[i][0] < wins[i][0] and wins[i][1] < bounds[i][1] for i in wins)
        check(f"(d0) {scen} scoring windows stay strictly inside their segment", ok,
              f"{wins} in {bounds}")
    # S2's re-ID reference is segment 1, NOT the opening -- the whole point of the matrix
    check("(d0) S2 re-ID reference is the NON-OPENING segment 1",
          S.first_appearance("S2")[S.seg_roles("S2")[3]] == 1)
    # the paraphrase variants must keep EXACTLY the same subject bookkeeping while changing the
    # prompt index of every revisit (that is the whole point: same experiment, harder routing)
    for base in ("S2", "S3", "S4"):
        p = base + "p"
        rev = S.revisit_segments(base)
        check(f"(d0) {p} keeps the subject/revisit structure of {base}",
              S.seg_roles(p) == S.seg_roles(base)
              and S.revisit_segments(p) == rev
              and S.first_appearance(p) == S.first_appearance(base),
              f"roles={S.seg_roles(p)} revisits={S.revisit_segments(p)}")
        pi = S.seg_prompt_idx(p)
        check(f"(d0) {p} conditions every revisit on the PARAPHRASE (prompt index 3)",
              all(pi[i] == 3 for i in rev)
              and all(pi[i] != 3 for i in range(len(pi)) if i not in rev),
              f"prompt indices {pi}, revisits {rev}")


# ============================================================================================
# (ss) SAME-SCENE tier: the prompts of a same-scene triple must describe ONE decor and differ
#      only by the subject, and the SS scenarios must be the S2 / S1 shapes on that tier.
# ============================================================================================
def test_ss_samescene():
    import scenarios as S

    check("(ss) SAMESCENE_TRIPLE_IDS are the APPENDED indices (0-11 untouched)",
          S.SAMESCENE_TRIPLE_IDS == list(range(12, 12 + len(S.SAMESCENE_TRIPLES)))
          and all(S.TRIPLES[i] == S.SAMESCENE_TRIPLES[i - 12] for i in S.SAMESCENE_TRIPLE_IDS),
          f"ids={S.SAMESCENE_TRIPLE_IDS}, {len(S.TRIPLES)} triples")
    for i in S.SAMESCENE_TRIPLE_IDS:
        suf = S.samescene_suffix(i)
        A, B, C = S.TRIPLES[i]
        check(f"(ss) triple {i}: the 3 prompts share a verbatim scene clause (>= 8 words)",
              len(suf.split()) >= 8 and all(p.endswith(suf) for p in (A, B, C)),
              f"'{suf}'")
        # A and B differ ONLY by their leading subject clause: identical from the scene clause on
        check(f"(ss) triple {i}: A and B differ only by the subject",
              A[: -len(suf)] != B[: -len(suf)] and A != B,
              f"A subject='{A[: -len(suf)].strip()}' | B subject='{B[: -len(suf)].strip()}'")
        check(f"(ss) triple {i}: the OPENER C is a THIRD subject of the same place",
              C not in (A, B) and C[: -len(suf)].strip() not in
              (A[: -len(suf)].strip(), B[: -len(suf)].strip()),
              f"C subject='{C[: -len(suf)].strip()}'")
    # the SS scenarios are the S2 / S1 shapes verbatim (so every downstream convention holds)
    for ss, base in (("SS2", "S2"), ("SS1", "S1")):
        check(f"(ss) {ss} is the {base} schedule verbatim on the same-scene tier",
              S.SCENARIOS[ss]["schedule"] == S.SCENARIOS[base]["schedule"]
              and S.SCENARIOS[ss]["frames"] == S.SCENARIOS[base]["frames"]
              and S.SCENARIOS[ss]["triples"] == S.SAMESCENE_TRIPLE_IDS,
              f"{S.SCENARIOS[ss]['schedule']}")
        check(f"(ss) {ss} scoring windows == {base} (same conventions, comparable numbers)",
              S.score_segments(ss, 240) == S.score_segments(base, 240)
              and S.revisit_segments(ss) == S.revisit_segments(base),
              f"{S.score_segments(ss, 240)}")
    check("(ss) SS2 scores the revisit against the NON-OPENING first appearance (segment 1)",
          S.first_appearance("SS2")[S.seg_roles("SS2")[3]] == 1
          and S.SCENARIOS["SS2"]["schedule"][0][1] == 2)
    check("(ss) SS1 segment 0 is A with nothing before it (clean A* for the SS2 cell)",
          S.SCENARIOS["SS1"]["schedule"][0][1] == 0)
    # verbatim revisit prompt on both -> the default routing threshold applies
    check("(ss) SS1/SS2 keep the verbatim routing threshold",
          S.fb_tau("SS2") == 0.90 and S.fb_tau("SS1") == 0.90)


# ============================================================================================
# (d) first-appearance scoring on synthetic videos
# ============================================================================================
def test_d_scoring(device):
    import torch
    import scenarios as S
    from torchvision.io import write_video
    import run_reid_scen as R

    # tiny S2-shaped test scenario (same segment structure, 60 latent frames)
    S.SCENARIOS["TSTS2"] = {"frames": 60, "schedule": [(0, 2), (15, 0), (30, 1), (45, 0)],
                            "desc": "synthetic scoring test"}
    H, W, NPIX = 96, 160, 4 * 60
    # 4 patterns DINO actually separates (measured cos: diag 1.0, off-diag 0.41-0.71). Random
    # noise textures do NOT work: DINO maps them all to "colored noise" (cos 0.996).
    import math
    yy = torch.linspace(0, 1, H).view(H, 1).expand(H, W)
    xx = torch.linspace(0, 1, W).view(1, W).expand(H, W)
    _base = torch.tensor([[1., 0., 0.], [0., .3, 1.], [0., 1., .2], [1., 1., 0.]])
    _alt = torch.tensor([[0., 0., .2], [1., 1., .9], [.4, 0., .6], [.1, .1, .1]])

    def pat(k):
        if k == 0:
            m = (torch.sin(2 * math.pi * 3 * yy) > 0).float()               # h-stripes
        elif k == 1:
            m = (((xx - .5) ** 2 + (yy - .5) ** 2) < .08).float()           # disc
        elif k == 2:
            m = (torch.sin(2 * math.pi * 5 * xx) > 0).float()               # v-stripes
        else:
            m = ((torch.sin(2 * math.pi * 3 * xx)
                  * torch.sin(2 * math.pi * 3 * yy)) > 0).float()           # checker
        return _base[k].view(3, 1, 1) * m + _alt[k].view(3, 1, 1) * (1 - m)

    tex = {k: pat(k) for k in range(4)}

    def make(path, seg_tex):
        """seg_tex[i] = texture id painted on segment i (latent frames -> pixel frames)."""
        vid = torch.zeros(NPIX, 3, H, W)
        for i, (lo, hi) in enumerate(S.seg_bounds("TSTS2")):
            plo = 0 if lo == 0 else 4 * lo - 3
            phi = min(4 * (hi - 1), NPIX - 1)
            vid[plo:phi + 1] = tex[seg_tex[i]]
        arr = (vid.permute(0, 2, 3, 1) * 255).clamp(0, 255).to(torch.uint8)
        write_video(path, arr, fps=16)

    os.makedirs(OUT, exist_ok=True)
    p1 = os.path.join(OUT, "synth_honest.mp4")
    p2 = os.path.join(OUT, "synth_trap.mp4")
    make(p1, {0: 0, 1: 1, 2: 2, 3: 1})    # honest: revisit (seg3) == first appearance (seg1)
    make(p2, {0: 1, 1: 3, 2: 2, 3: 1})    # trap: seg3 == SEGMENT 0, first appearance differs
    s1 = R.score_video_scen(p1, "TSTS2", device, settle=2, resettle=2)
    s2 = R.score_video_scen(p2, "TSTS2", device, settle=2, resettle=2)
    check("(d) honest synthetic: revisit matches its FIRST APPEARANCE (segment 1)",
          s1["full"]["reid_resA_origA"] > 0.99,
          f"reid={s1['full']['reid_resA_origA']:.4f} ctrl_resA_B={s1['full']['ctrl_resA_B']:.4f}"
          f" ctrl_B_origA={s1['full']['ctrl_B_origA']:.4f}")
    check("(d) honest synthetic: the interlude control is clearly lower",
          s1["full"]["reid_resA_origA"] - s1["full"]["ctrl_resA_B"] > 0.05,
          f"delta={s1['full']['reid_resA_origA'] - s1['full']['ctrl_resA_B']:.4f}")
    check("(d) TRAP synthetic: scoring does NOT fall back on segment 0",
          s2["full"]["reid_resA_origA"] < 0.90,
          f"reid={s2['full']['reid_resA_origA']:.4f} (a 'compare to segment 0' bug -> 1.0000, "
          f"since segment 3 is pixel-identical to segment 0 in this video)")
    check("(d) crop60 variant follows the same first-appearance bookkeeping",
          s1["crop60"]["reid_resA_origA"] > 0.99 and s2["crop60"]["reid_resA_origA"] < 0.98,
          f"honest={s1['crop60']['reid_resA_origA']:.4f} trap={s2['crop60']['reid_resA_origA']:.4f}")
    check("(d) revisit bookkeeping in the score JSON",
          s1["revisits"] == [3] and s1["first_appearance"]["0"] == 1,
          f"revisits={s1['revisits']} first_appearance={s1['first_appearance']}")
    del S.SCENARIOS["TSTS2"]


# ============================================================================================
# (a) (b) (c) -- the GPU tests, one pipeline build for all of them
# ============================================================================================
def test_abc(device):
    import torch
    from omegaconf import OmegaConf
    from utils.misc import set_seed
    import scenarios as S
    from run_reid_probe import build_pipeline, ARM_ENV, release_vram, MEM2_WRITER_DEFAULT, \
        MEM2_DEC_DEFAULT

    cfg = OmegaConf.load(os.path.join(PROBE_DIR, "probe_l1.yaml"))
    cfg.distributed = False
    torch.set_grad_enabled(False)
    A, B = S.TRIPLES[0][0], S.TRIPLES[0][1]
    seed = int(cfg.seed)

    def apply_env(arm, table):
        for k in S._OFF_SCEN:                 # union of every hook key -> nothing can leak
            os.environ[k] = S._OFF_SCEN[k]
        for k, v in table[arm].items():
            os.environ[k] = v
        if arm in ("memdec20", "memdec20r"):
            assert os.path.exists(MEM2_DEC_DEFAULT), MEM2_DEC_DEFAULT
            assert os.path.exists(MEM2_WRITER_DEFAULT), MEM2_WRITER_DEFAULT
            os.environ["MEM_DECODE"] = MEM2_DEC_DEFAULT
            os.environ["MEM_WRITER"] = MEM2_WRITER_DEFAULT

    pipeline = build_pipeline(cfg, device)

    def noise():
        set_seed(seed)
        return torch.randn([1, FRAMES, 16, 60, 104], device=device, dtype=torch.bfloat16)

    def run_legacy(arm):
        apply_env(arm, ARM_ENV)
        _, lat = pipeline.inference(
            noise=noise(), text_prompts_first=[A], text_prompts_second=[B],
            switch_frame_index=SW1, text_prompts_third=None, second_switch_frame_index=SW2,
            return_latents=True, low_memory=True)
        return lat.float().cpu()

    def run_sched(arm, table, scen_schedule=((0, 0), (SW1, 1), (SW2, 0)), prompts=None,
                  subjects=None, tau=None):
        apply_env(arm, table)
        if tau is not None:
            os.environ["FRAMEBANK_TAU"] = str(tau)
        _, lat = pipeline.inference_schedule(
            noise=noise(), prompts=list(prompts or [A, B]),
            schedule=[tuple(x) for x in scen_schedule], subjects=subjects,
            return_latents=True, low_memory=True)
        return lat.float().cpu(), list(getattr(pipeline, "_fb_events", []) or [])

    lat = {}
    # ---- (a) legacy vs schedule, bitwise --------------------------------------------------
    for arm in ("memzero", "oracle", "oracle1", "oracle3b", "memdec20"):
        a = run_legacy(arm)
        b, _ = run_sched(arm, ARM_ENV)
        d = (a - b).abs().max().item()
        lat[arm] = b
        check(f"(a) inference_schedule == inference BITWISE on the legacy schedule [{arm}]",
              d == 0.0, f"max|diff| = {d:.3e}")
    # sanity: the arms are not all the same rollout (otherwise (a) would be vacuous)
    d_probe = (lat["memzero"] - lat["oracle1"]).abs().max().item()
    check("(a) sanity: the hooks DO change the rollout (oracle1 != memzero)", d_probe > 0,
          f"max|diff| = {d_probe:.3e}")

    # ---- (b) framebank on A->B->A == oracle1 ----------------------------------------------
    for arm in ("framebank_oracle", "framebank"):
        b, ev = run_sched(arm, S.SCEN_ARM_ENV)
        d = (b - lat["oracle1"]).abs().max().item()
        rev = [e for e in ev if e["seg"] == 2]
        check(f"(b) {arm} on A->B->A == oracle1 BITWISE (bank of 1 A scene)", d == 0.0,
              f"max|diff| = {d:.3e}")
        check(f"(b) {arm} routed the revisit to the right scene", len(rev) == 1
              and rev[0]["recalled"] and rev[0]["correct"] and rev[0]["chosen_seg"] == 0,
              json.dumps(rev[0] if rev else {})[:220])
        first = [e for e in ev if e["seg"] == 1]
        check(f"(b) {arm} did NOT falsely recall at the fresh B switch",
              len(first) == 1 and not first[0]["recalled"],
              json.dumps(first[0] if first else {})[:220])
        lat[arm] = b

    # ---- (b2) a REAL multi-candidate route: S4-shaped A->B->A->C->A on 36 frames -----------
    C = S.TRIPLES[0][2]
    sched4 = ((0, 0), (9, 1), (15, 0), (24, 2), (30, 0))
    _, ev4 = run_sched("framebank", S.SCEN_ARM_ENV, sched4, prompts=[A, B, C])
    at_rev = [e for e in ev4 if e["seg"] in (2, 4)]
    check("(b2) N-switch router fires at BOTH returns of A and picks A each time",
          len(at_rev) == 2 and all(e["recalled"] and e["correct"] for e in at_rev),
          " | ".join(f"seg{e['seg']}->seg{e['chosen_seg']} smax={e.get('s_max', float('nan')):.4f}"
                     for e in at_rev))
    check("(b2) the pool GROWS (4 scenes banked by the last switch)",
          max(e["bank_size"] for e in ev4) == 4,
          f"bank sizes {[e['bank_size'] for e in ev4]}")
    off = [e for e in ev4 if e["seg"] not in (2, 4)]
    check("(b2) no false recall at the fresh B and C switches",
          not any(e["recalled"] for e in off),
          " | ".join(f"seg{e['seg']} smax={e.get('s_max', float('nan')):.4f}" for e in off))

    # ---- (b3) PARAPHRASED revisit: the revisit is conditioned on A', subject stays 0 --------
    # Triple 6 is used here (measured margin cos(A',A) 0.7803 vs worst distractor 0.5805): the
    # point of this check is the SUBJECT bookkeeping and the non-trivial match, not the value of
    # tau at its edge -- triples 0 and 10 sit within 0.01 of the default threshold on purpose.
    A6, B6, C6, Ap6 = S.TRIPLES[6][0], S.TRIPLES[6][1], S.TRIPLES[6][2], S.PARAPHRASES[6]
    sched4p = ((0, 0), (9, 1), (15, 3), (24, 2), (30, 3))
    _, ev5 = run_sched("framebank", S.SCEN_ARM_ENV, sched4p, prompts=[A6, B6, C6, Ap6],
                       subjects=[0, 1, 0, 2, 0], tau=S.fb_tau("S4p"))
    rev5 = [e for e in ev5 if e["seg"] in (2, 4)]
    check("(b3) paraphrased revisit: the SUBJECT bookkeeping still marks segments 2 and 4",
          len(rev5) == 2, f"routing events at {[e['seg'] for e in ev5]}")
    # seg2 is the only NON-TRIVIAL match (A' vs the verbatim A of segment 0). By seg4 the bank
    # already holds segment 2, itself conditioned on A', so a cosine of exactly 1.0 there is the
    # correct behaviour, not a leak.
    check("(b3) paraphrased revisit: the router matches A' to the verbatim A scene",
          all(e["recalled"] and e["correct"] for e in rev5)
          and rev5[0]["s_max"] < 0.999 and rev5[0]["chosen_seg"] == 0,
          " | ".join(f"seg{e['seg']}->seg{e['chosen_seg']} smax={e['s_max']:.4f} "
                     f"correct={e['correct']}" for e in rev5))

    # ---- (g) MEM_DECODE splice GATING ------------------------------------------------------
    # The decoded anchor is an intervention that only makes sense at a RETURN. Firing it on a
    # first appearance would contaminate the very segment the re-ID is scored against, so the
    # count and the POSITION of the splices are pinned here on the three schedule shapes.
    import contextlib
    import io as _io

    def run_capture(arm, table, sched, prompts=None, subjects=None, tau=None):
        buf = _io.StringIO()
        with contextlib.redirect_stdout(buf):
            lat, ev = run_sched(arm, table, sched, prompts=prompts, subjects=subjects, tau=tau)
        return lat, ev, buf.getvalue()

    def splice_segments(txt):
        """Segment indices at which a decoded anchor was spliced. `_switch_recache` prints the
        splice line BEFORE inference_schedule announces the segment it just switched to, so a
        pending splice is attributed to the NEXT 'switch to segment' line."""
        pending, out = 0, []
        for line in txt.splitlines():
            if "[MEM_DECODE] decoded anchor spliced" in line:
                pending += 1
            elif line.startswith("[schedule] switch to segment "):
                out += [int(line.split("segment ")[1].split()[0])] * pending
                pending = 0
        assert pending == 0, "a MEM_DECODE splice was printed outside any switch"
        return out

    C0 = S.TRIPLES[0][2]
    shapes = {
        # tiny replicas of the campaign schedules (36 frames, num_frame_per_block 3)
        "S2-shaped C->A->B->A": (((0, 2), (9, 0), (18, 1), (27, 0)), None, [3]),
        "S3-shaped A->B->C->A": (((0, 0), (9, 1), (18, 2), (27, 0)), None, [3]),
        "S4-shaped A->B->A->C->A": (((0, 0), (9, 1), (15, 0), (24, 2), (30, 0)), None, [2, 4]),
        # negative control: no subject ever comes back -> the hook must never fire
        "no-revisit A->B->C": (((0, 0), (9, 1), (18, 2)), None, []),
    }
    for name, (sched, subj, expect) in shapes.items():
        _, _, txt = run_capture("memdec20", S.SCEN_ARM_ENV, sched, prompts=[A, B, C0],
                                subjects=subj)
        got = splice_segments(txt)
        check(f"(g) memdec20 splices ONLY at the revisits [{name}]", got == expect,
              f"splices at segments {got}, expected {expect}")

    # the paraphrase case: the revisit is asked for with A', the SUBJECT bookkeeping must still
    # be what fires the hook (a "same prompt string" rule would splice nothing here).
    A6, B6, C6, Ap6 = S.TRIPLES[6][0], S.TRIPLES[6][1], S.TRIPLES[6][2], S.PARAPHRASES[6]
    _, _, txt = run_capture("memdec20", S.SCEN_ARM_ENV, ((0, 2), (9, 0), (18, 1), (27, 3)),
                            prompts=[A6, B6, C6, Ap6], subjects=[2, 0, 1, 0])
    check("(g) memdec20 splices at a PARAPHRASED revisit [S2p-shaped]",
          splice_segments(txt) == [3], f"splices at {splice_segments(txt)}")

    # ---- (g2) MEM_DECODE_ROUTED: same WHEN as framebank, different WHAT --------------------
    for name, (sched, subj, expect) in shapes.items():
        _, ev_fb, _ = run_capture("framebank", S.SCEN_ARM_ENV, sched, prompts=[A, B, C0],
                                  subjects=subj)
        _, ev_rt, txt = run_capture("memdec20r", S.SCEN_ARM_ENV, sched, prompts=[A, B, C0],
                                    subjects=subj)
        dec_fb = [(e["seg"], e["recalled"]) for e in ev_fb]
        dec_rt = [(e["seg"], e["recalled"]) for e in ev_rt]
        check(f"(g2) memdec20r routes EXACTLY like framebank [{name}]", dec_fb == dec_rt,
              f"framebank {dec_fb} vs memdec20r {dec_rt}")
        got = splice_segments(txt)
        check(f"(g2) memdec20r splices where its router recalled [{name}]",
              got == [s for s, r in dec_rt if r], f"splices {got}, router {dec_rt}")

    # ---- (g3) the variable ABSENT from the environment changes nothing ---------------------
    apply_env("memdec20", S.SCEN_ARM_ENV)
    os.environ.pop("MEM_DECODE_ROUTED", None)
    _, l_abs = pipeline.inference(
        noise=noise(), text_prompts_first=[A], text_prompts_second=[B],
        switch_frame_index=SW1, text_prompts_third=None, second_switch_frame_index=SW2,
        return_latents=True, low_memory=True)
    d = (l_abs.float().cpu() - lat["memdec20"]).abs().max().item()
    check("(g3) legacy inference() BITWISE with MEM_DECODE_ROUTED ABSENT from the env",
          d == 0.0, f"max|diff| = {d:.3e}")
    apply_env("memdec20", S.SCEN_ARM_ENV)
    os.environ.pop("MEM_DECODE_ROUTED", None)
    _, b_abs = pipeline.inference_schedule(
        noise=noise(), prompts=[A, B], schedule=[(0, 0), (SW1, 1), (SW2, 0)],
        return_latents=True, low_memory=True)
    d = (b_abs.float().cpu() - lat["memdec20"]).abs().max().item()
    check("(g3) inference_schedule BITWISE with MEM_DECODE_ROUTED absent vs '0'", d == 0.0,
          f"max|diff| = {d:.3e}")

    # ---- (c) no env leak between arms -----------------------------------------------------
    again, _ = run_sched("memzero", S.SCEN_ARM_ENV)
    d = (again - lat["memzero"]).abs().max().item()
    check("(c) memzero replayed AFTER every other arm == the first memzero BITWISE", d == 0.0,
          f"max|diff| = {d:.3e}")
    from pipeline.switch_causal_inference import _find_causal_model
    cm = _find_causal_model(pipeline.generator.model)
    check("(c) MEM_WRITER overlay restored (no stale theta_w)",
          getattr(cm, "_memwriter_applied", None) is None,
          str(getattr(cm, "_memwriter_applied", None)))

    del pipeline
    release_vram()


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--cpu-only", action="store_true")
    args = ap.parse_args(argv)
    os.makedirs(OUT, exist_ok=True)
    test_e_compile()
    test_f_heldout()
    test_schedule_math()
    test_ss_samescene()
    if not args.cpu_only:
        import torch
        device = torch.device("cuda")
        test_d_scoring(device)
        test_abc(device)
    print("\n================ SELFTEST SUMMARY ================")
    for n, ok, d in _RESULTS:
        print(f"  {'PASS' if ok else 'FAIL':4}  {n}")
    print(f"{sum(1 for _, ok, _ in _RESULTS if ok)}/{len(_RESULTS)} checks passed"
          + ("  (CPU-ONLY subset)" if args.cpu_only else ""))


if __name__ == "__main__":
    main()
