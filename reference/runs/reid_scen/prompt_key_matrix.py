#!/usr/bin/env python
# ============================================================================================
# tau calibration for the FRAMEBANK router: the pooled-UMT5 cosine between every pair of
# scenario prompts. A revisit uses the VERBATIM prompt of the first appearance, so the correct
# match always scores 1.0; what decides FRAMEBANK_TAU is how high the highest WRONG pair goes
# (a false recall = the baseline splicing the wrong scene into a fresh segment).
#   python prompt_key_matrix.py            # writes results/prompt_key_matrix.json
# ============================================================================================
import json
import os
import sys

REPO = "/shared/ssd_14T/home/adenelmi/Anchor-Forcing"
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "reference/runs/reid_probe_l1"))
sys.path.insert(0, HERE)

import torch                                                       # noqa: E402
from pipeline.switch_causal_inference import _memdec_pooled_cond   # noqa: E402
import scenarios as S                                              # noqa: E402


def main():
    from utils.wan_wrapper import WanTextEncoder
    device = torch.device("cuda")
    torch.set_grad_enabled(False)
    enc = WanTextEncoder().to(device=device, dtype=torch.bfloat16).eval()
    prompts = S.scen_prompts()
    keys = []
    for p in prompts:
        k = _memdec_pooled_cond(enc(text_prompts=[p])).float()
        keys.append(k / k.norm().clamp_min(1e-12))
    K = torch.stack(keys)
    M = (K @ K.T).numpy()

    n_t = len(S.TRIPLES)
    # index layout of scen_prompts(): [A0,B0,C0, A1,B1,C1, ...] then [A'0, A'1, ...]
    def tri(i):
        return i // 3 if i < 3 * n_t else i - 3 * n_t

    within, across = [], []
    for i in range(len(prompts)):
        for j in range(i + 1, len(prompts)):
            (within if tri(i) == tri(j) else across).append((float(M[i][j]), i, j))
    within.sort(reverse=True)
    across.sort(reverse=True)
    # the two quantities that decide FRAMEBANK_TAU, per triple:
    #   correct  = cos(paraphrase A', A)          (the p-variant match the router must find)
    #   distract = max cos(A', B) , cos(A', C)    (the wrong scenes in the same rollout)
    rows = []
    for t in range(n_t):
        ai, bi, ci, pi = 3 * t, 3 * t + 1, 3 * t + 2, 3 * n_t + t
        rows.append((t, float(M[pi][ai]), max(float(M[pi][bi]), float(M[pi][ci]))))
    print(f"[tau] {len(prompts)} prompts ({n_t} triples + {n_t} paraphrases).")
    print("[tau] VERBATIM revisit (S1-S4): correct match = 1.0000 by construction")
    hard_within = [(s, i, j) for s, i, j in within if not (i >= 3 * n_t or j >= 3 * n_t)]
    print(f"[tau]   worst in-rollout distractor over all triples: {hard_within[0][0]:.4f} "
          f"({prompts[hard_within[0][1]][:40]!r} | {prompts[hard_within[0][2]][:40]!r})")
    print("[tau] PARAPHRASED revisit (S2p/S3p/S4p), per triple: cos(A',A) vs worst distractor")
    worst_margin = 1e9
    for t, c, d in rows:
        worst_margin = min(worst_margin, c - d)
        flag = "" if c - d > 0.05 else "   <-- THIN MARGIN"
        print(f"       t{t:<2} correct {c:.4f}   distractor {d:.4f}   margin {c - d:+.4f}{flag}")
    lo = min(c for _, c, _ in rows)
    hi = max(d for _, _, d in rows)
    print(f"[tau] paraphrase: min correct {lo:.4f}, max distractor {hi:.4f}, "
          f"worst per-triple margin {worst_margin:+.4f}")
    print(f"[tau] ACROSS-triple max cosine (never co-occurs in a rollout): {across[0][0]:.4f}")
    mx = hard_within[0][0]
    print(f"\n[tau] FRAMEBANK_TAU: verbatim scenarios need tau in ({mx:.4f}, 1.0) -> default 0.90 "
          f"has {0.90 - mx:+.4f} of margin and CANNOT mis-route.")
    print(f"[tau]                paraphrase scenarios need tau in ({hi:.4f}, {lo:.4f}) -> "
          + (f"use tau = {0.5 * (hi + lo):.2f}" if hi < lo else
             "NO single threshold separates them: mis-routing is possible, which is exactly "
             "what framebank_oracle measures"))
    out = os.path.join(HERE, "results", "prompt_key_matrix.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump({"prompts": prompts, "cos": M.tolist(),
                   "verbatim_worst_distractor": mx, "across_triple_max": across[0][0],
                   "paraphrase_per_triple": [{"triple": t, "correct": c, "worst_distractor": d}
                                             for t, c, d in rows],
                   "top_within": [{"cos": s, "a": prompts[i], "b": prompts[j]}
                                  for s, i, j in within[:20]]}, f, indent=2)
    print(f"[tau] wrote {out}")


if __name__ == "__main__":
    main()
