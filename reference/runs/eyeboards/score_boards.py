#!/usr/bin/env python
"""Blind judging boards, scorer. Takes the code returned by the judge + the sealed key, and
produces the table. No torch, no GPU, no media: it only reads text.

  python score_boards.py --key key_duel2.json --code "duel2~1~gdg...~a1b2c3"
  python score_boards.py --key key_pair2.json --code_file answers.txt

What it refuses to do, and why (every guard comes from a real mistake in this project)
 - It never dilutes "cannot tell" into chance: it counts them separately and computes the
   score over decided cells only, printing both counts. 60% over 10 decided cells out of 30 is
   not the same result as 60% over 30.
 - It never reports a p-value without the count and the interval: the registry's 10-seed annex
   showed that a single point always presents as more solid than it is.
 - It checks the checksum: a code pasted from the wrong board is a loud error, not a plausible
   score.
"""
import argparse
import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from boards import CODE_VERSION, code_checksum  # noqa: E402

UNSURE = "?"
BLANK = "."


# ----------------------------- statistiques ---------------------------------------------------
def binom_p_two_sided(k, n, p0=0.5):
    """Exact two-sided p. Pure python (no scipy on the audit box)."""
    if n == 0:
        return 1.0
    def pmf(i):
        return math.comb(n, i) * (p0 ** i) * ((1.0 - p0) ** (n - i))
    obs = pmf(k)
    return min(1.0, sum(pmf(i) for i in range(n + 1) if pmf(i) <= obs * (1.0 + 1e-12)))


def wilson(k, n, z=1.96):
    """95% Wilson interval: honest at small n, unlike the normal interval which runs outside
    [0,1] as soon as the rate approaches the edges, and our n is always small."""
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    r = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    # hard clamp: at the extremes (k=0 or k=n) floating-point arithmetic leaves [0,1] by about
    # 1e-17, and an interval printed as -0.00 casts doubt on the rest of the table.
    return (min(1.0, max(0.0, (c - r) / d)), min(1.0, max(0.0, (c + r) / d)))


# ----------------------------- lecture du code ------------------------------------------------
def parse_code(code, key):
    """`<board_id>~<version>~<payload>~<checksum>` -> list of answers, one per slot."""
    code = code.strip().replace("\n", "").replace(" ", "")
    parts = code.split("~")
    assert len(parts) == 4, (
        f"malformed code: expected 4 fields separated by ~, got {len(parts)}. "
        f"Paste the complete line produced by the board.")
    bid, ver, payload, chk = parts
    assert bid == key["board_id"], (
        f"this code comes from board '{bid}', the key is for '{key['board_id']}'")
    assert ver == CODE_VERSION, f"code version {ver}, scorer version {CODE_VERSION}"
    assert chk == key["checksum"] == code_checksum(bid, key["kind"]), \
        "invalid checksum: truncated code, or a key that does not match the board"
    assert len(payload) == key["n_slots"], (
        f"{len(payload)} answers for {key['n_slots']} slots: incomplete code")
    allowed = {s["slot"]: set(s["answers"]) | {BLANK} for s in key["slots"]}
    for i, ch in enumerate(payload):
        assert ch in allowed[i], f"slot {i}: answer '{ch}' not in {sorted(allowed[i])}"
    return list(payload)


# ----------------------------- les trois lectures ----------------------------------------------
def score_pair2(ans, key):
    """An answer key exists: the true subject is on a known side. We report the hit rate over
    decided cells, plus a breakdown by source (decoder, light blur, strong blur...) when the meta
    carries it, that breakdown is what places the decoder output on the blur ladder, which is
    the very question of the 22/08 board."""
    per_src, rows = {}, []
    hit = seen = unsure = 0
    for it in key["items"]:
        a = ans[it["slot"]]
        src = it.get("meta", {}).get("source", "(no source)")
        d = per_src.setdefault(src, {"hit": 0, "n": 0, "unsure": 0})
        if a in (UNSURE, BLANK):
            unsure += 1
            d["unsure"] += 1
            rows.append((it["id"], src, "cannot tell"))
            continue
        ok = (a == it["true_side"])
        hit += ok; seen += 1
        d["hit"] += ok; d["n"] += 1
        rows.append((it["id"], src, "hit" if ok else "miss"))
    out = {"kind": "pair2", "n_total": len(key["items"]), "n_decided": seen,
           "n_unsure": unsure, "hits": hit,
           "acc": hit / seen if seen else None,
           "p": binom_p_two_sided(hit, seen),
           "ci": wilson(hit, seen), "rows": rows,
           "by_source": {k: {**v, "acc": (v["hit"] / v["n"]) if v["n"] else None,
                             "p": binom_p_two_sided(v["hit"], v["n"])}
                         for k, v in sorted(per_src.items())}}
    return out


def score_duel2(ans, key):
    """No answer key: this is a preference between two arms. We count wins per arm and test them
    against a coin flip. The winning arm is not 'the right one', it is the one the eye picks more
    often, and without both counts the sentence means nothing."""
    wins, unsure, rows = {}, 0, []
    for it in key["items"]:
        a = ans[it["slot"]]
        if a in (UNSURE, BLANK):
            unsure += 1
            rows.append((it["id"], "cannot tell"))
            continue
        arm = it["sides"][a]
        wins[arm] = wins.get(arm, 0) + 1
        rows.append((it["id"], arm))
    for it in key["items"]:                       # an arm never picked must still show as 0
        for arm in it["sides"].values():
            wins.setdefault(arm, 0)
    seen = sum(wins.values())
    arms = sorted(wins)
    k = wins[arms[0]] if arms else 0
    return {"kind": "duel2", "n_total": len(key["items"]), "n_decided": seen,
            "n_unsure": unsure, "wins": wins, "arms": arms,
            "rate": {a: (wins[a] / seen if seen else None) for a in arms},
            "p": binom_p_two_sided(k, seen),
            "ci_first_arm": wilson(k, seen), "rows": rows}


def score_rank3(ans, key):
    """Two readings, as the board asked for them: the mean rank per variant (relative comparison,
    which survives even if no variant convinces) and the absolute judgment."""
    ranks, judge, duels = {}, {}, {}
    for it in key["items"]:
        seen_here = {}
        for letter, meta in it["letters"].items():
            v = meta["variant"]
            r, j = ans[meta["rank_slot"]], ans[meta["judge_slot"]]
            if r != BLANK:
                ranks.setdefault(v, []).append(int(r))
                seen_here[v] = int(r)
            if j != BLANK:
                d = judge.setdefault(v, {"o": 0, "a": 0, "n": 0})
                d[j] += 1
        names = sorted(seen_here)
        for i in range(len(names)):
            for j2 in range(i + 1, len(names)):
                a_, b_ = names[i], names[j2]
                if seen_here[a_] == seen_here[b_]:
                    continue                       # tie: counts for neither
                w = a_ if seen_here[a_] < seen_here[b_] else b_   # rank 1 = best
                pair = tuple(sorted((a_, b_)))
                d = duels.setdefault(pair, {pair[0]: 0, pair[1]: 0})
                d[w] += 1
    return {"kind": "rank3", "n_items": len(key["items"]),
            "mean_rank": {v: sum(r) / len(r) for v, r in sorted(ranks.items())},
            "n_ranked": {v: len(r) for v, r in sorted(ranks.items())},
            "judge": {v: d for v, d in sorted(judge.items())},
            "duels": {f"{a} vs {b}": d for (a, b), d in sorted(duels.items())}}


SCORERS = {"pair2": score_pair2, "duel2": score_duel2, "rank3": score_rank3}


def score(key, code):
    ans = parse_code(code, key)
    return SCORERS[key["kind"]](ans, key)


# ----------------------------- rendu -----------------------------------------------------------
def render(res):
    L = [f"board {res['kind']}"]
    if res["kind"] == "pair2":
        L += [f"  decided {res['n_decided']}/{res['n_total']}  "
              f"(cannot tell: {res['n_unsure']})"]
        if res["acc"] is None:
            L += ["  NO decided cell: no score."]
        else:
            lo, hi = res["ci"]
            L += [f"  hits {res['hits']}/{res['n_decided']} = {res['acc']:.3f}  "
                  f"CI95 [{lo:.2f}, {hi:.2f}]  p={res['p']:.3g}  (chance 0.5)"]
        for src, d in res["by_source"].items():
            a = f"{d['acc']:.3f}" if d["acc"] is not None else "n/a"
            L += [f"    {src:<16} {d['hit']}/{d['n']} = {a}  p={d['p']:.3g}  "
                  f"(cannot tell {d['unsure']})"]
    elif res["kind"] == "duel2":
        L += [f"  decided {res['n_decided']}/{res['n_total']}  "
              f"(cannot tell: {res['n_unsure']})"]
        for a in res["arms"]:
            r = res["rate"][a]
            L += [f"    {a:<16} {res['wins'][a]} wins" +
                  (f" = {r:.3f}" if r is not None else "")]
        if res["n_decided"]:
            lo, hi = res["ci_first_arm"]
            L += [f"  p={res['p']:.3g} against a coin flip  "
                  f"(CI95 of {res['arms'][0]} [{lo:.2f}, {hi:.2f}])"]
    else:
        L += [f"  {res['n_items']} rows"]
        L += ["  mean rank (1 = best):"]
        for v, r in sorted(res["mean_rank"].items(), key=lambda kv: kv[1]):
            L += [f"    {v:<20} {r:.2f}   (n={res['n_ranked'][v]})"]
        L += ["  absolute judgment:"]
        for v, d in res["judge"].items():
            L += [f"    {v:<20} yes {d['o']}  roughly {d['a']}  no {d['n']}"]
        if res["duels"]:
            L += ["  rank duels:"]
            for name, d in res["duels"].items():
                L += [f"    {name:<34} " + "  ".join(f"{k} {v}" for k, v in d.items())]
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser(description="blind judging board scorer")
    ap.add_argument("--key", required=True)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--code")
    g.add_argument("--code_file")
    ap.add_argument("--json", action="store_true", help="machine-readable output")
    a = ap.parse_args()
    with open(a.key, encoding="utf8") as f:
        key = json.load(f)
    code = a.code if a.code else open(a.code_file, encoding="utf8").read()
    res = score(key, code)
    print(json.dumps(res, indent=1, ensure_ascii=False) if a.json else render(res))


if __name__ == "__main__":
    main()
