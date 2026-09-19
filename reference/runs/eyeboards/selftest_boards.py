#!/usr/bin/env python
"""Board SELFTESTS, pure CPU, no real media, ~1 s. Run before generating a campaign: a wrong
board costs an evening of generation and, worse, an eye judgment that cannot be asked for
again identically.
  python reference/runs/eyeboards/selftest_boards.py
"""
import json
import os
import shutil
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import boards  # noqa: E402
from boards import CODE_VERSION, build_duel2, build_pair2, build_rank3, perm  # noqa: E402
from score_boards import parse_code, score, wilson  # noqa: E402

OK = []


def check(name, cond, detail=""):
    assert cond, f"FAIL {name} {detail}"
    OK.append(name)
    print(f"  ok  {name}")


TMP = tempfile.mkdtemp(prefix="eyeboards_selftest_")
try:
    r_pair, r_rank, r_duel = boards._demo(os.path.join(TMP, "demo"))

    def load(p):
        with open(p, encoding="utf8") as f:
            return json.load(f)

    def page(p):
        with open(p, encoding="utf8") as f:
            return f.read()

    k_pair, k_rank, k_duel = (load(r["key"]) for r in (r_pair, r_rank, r_duel))

    # --- B1: the permutation is deterministic and depends on the question ----------------------
    check("B1 perm reproducible", perm("b", "s01", 3) == perm("b", "s01", 3))
    check("B1b perm changes with the question", perm("b", "s01", 3) != perm("b", "s02", 3)
          or perm("b", "s01", 3) != perm("b", "s03", 3))
    check("B1c perm is a real permutation",
          all(sorted(perm("b", f"s{i}", 3)) == [0, 1, 2] for i in range(50)))
    counts = {}
    for i in range(400):                       # it must also be balanced: a shuffle that
        counts[perm("b", f"s{i}", 2)[0]] = counts.get(perm("b", f"s{i}", 2)[0], 0) + 1
    check("B1d perm balanced over 2 slots", 150 < counts.get(0, 0) < 250, f"{counts}")

    # --- B2: the key never leaks into the page --------------------------------------------------
    # the test that matters. A board containing the arm name or the file path is not a blind
    # judgment, it is a lost measurement.
    for r, forbidden in ((r_duel, ["memon", "memoff"]),
                         (r_rank, ["v1_ancien", "v2_mixer", "v2_mixer_sharp"])):
        h = page(r["html"])
        leaked = [w for w in forbidden if w in h]
        check(f"B2 no arm name in {os.path.basename(r['html'])}", not leaked, f"{leaked}")
    for r in (r_pair, r_rank, r_duel):
        h = page(r["html"])
        check(f"B2b no file path in {os.path.basename(r['html'])}",
              "/media/" not in h and ".png\"" not in h and TMP not in h)
        # 'true' on its own legitimately appears in aria-pressed="true": we look for the key's
        # field names, which must never cross over.
        check(f"B2c no key field in {os.path.basename(r['html'])}",
              "true_side" not in h and "twin" not in h and "variant" not in h
              and "sides" not in h)

    # --- B3: the answer key really is spread over both sides -----------------------------------
    sides = [it["true_side"] for it in k_pair["items"]]
    check("B3 the answer key is not constant", len(set(sides)) == 2 or len(sides) < 4, f"{sides}")

    # --- B4: round trip code -> score, with a perfect judge ------------------------------------
    perfect = "".join(it["true_side"] for it in sorted(k_pair["items"], key=lambda x: x["slot"]))
    code = f"{k_pair['board_id']}~{CODE_VERSION}~{perfect}~{k_pair['checksum']}"
    res = score(k_pair, code)
    check("B4 perfect judge -> 1.0", res["acc"] == 1.0 and res["n_unsure"] == 0)
    inverse = "".join({"g": "d", "d": "g"}[c] for c in perfect)
    res0 = score(k_pair, f"{k_pair['board_id']}~{CODE_VERSION}~{inverse}~{k_pair['checksum']}")
    check("B4b always-wrong judge -> 0.0", res0["acc"] == 0.0)

    # --- B5: "cannot tell" is counted separately, not diluted ----------------------------------
    mixed = "?" + perfect[1:]
    res1 = score(k_pair, f"{k_pair['board_id']}~{CODE_VERSION}~{mixed}~{k_pair['checksum']}")
    check("B5 doubts leave the denominator",
          res1["n_unsure"] == 1 and res1["n_decided"] == len(perfect) - 1 and res1["acc"] == 1.0)
    allq = "?" * len(perfect)
    res2 = score(k_pair, f"{k_pair['board_id']}~{CODE_VERSION}~{allq}~{k_pair['checksum']}")
    check("B5b all doubts -> no score, not a zero", res2["acc"] is None)

    # --- B6: invalid codes are loudly refused ---------------------------------------------------
    for bad, why in ((f"{k_pair['board_id']}~{CODE_VERSION}~{perfect}~deadbe", "bad checksum"),
                     (f"other~{CODE_VERSION}~{perfect}~{k_pair['checksum']}", "wrong board"),
                     (f"{k_pair['board_id']}~{CODE_VERSION}~{perfect[:-1]}~{k_pair['checksum']}",
                      "truncated code"),
                     (f"{k_pair['board_id']}~{CODE_VERSION}~{'x'*len(perfect)}~"
                      f"{k_pair['checksum']}", "answer outside the options"),
                     ("garbage", "format")):
        try:
            parse_code(bad, k_pair)
            raise SystemExit(f"FAIL B6: '{why}' was accepted")
        except AssertionError:
            pass
    check("B6 the 5 invalid codes are refused", True)

    # --- B7: duel2 counts wins per arm, not per side --------------------------------------------
    # Catches: the bug that would make the permutation pointless (counting L/R instead of arm).
    always_g = "g" * k_duel["n_slots"]
    rd = score(k_duel, f"{k_duel['board_id']}~{CODE_VERSION}~{always_g}~{k_duel['checksum']}")
    left_arms = [it["sides"]["g"] for it in k_duel["items"]]
    expect = {a: left_arms.count(a) for a in set(left_arms)}
    check("B7 wins attributed to the arm shown on the left",
          all(rd["wins"][a] == n for a, n in expect.items()), f"{rd['wins']} vs {expect}")
    check("B7b both arms appear, even at zero", len(rd["arms"]) == 2, f"{rd['arms']}")
    check("B7c total wins == decided cells",
          sum(rd["wins"].values()) == rd["n_decided"])

    # --- B8: rank3, rank 1 = best, duels consistent -------------------------------------------
    # We build a judge who always puts the 'v2_mixer_sharp' variant first.
    payload = ["."] * k_rank["n_slots"]
    for it in k_rank["items"]:
        order = sorted(it["letters"].items(),
                       key=lambda kv: (kv[1]["variant"] != "v2_mixer_sharp", kv[1]["variant"]))
        for rank, (_letter, meta) in enumerate(order, start=1):
            payload[meta["rank_slot"]] = str(rank)
            payload[meta["judge_slot"]] = "o" if rank == 1 else "n"
    rr = score(k_rank, f"{k_rank['board_id']}~{CODE_VERSION}~{''.join(payload)}~"
                       f"{k_rank['checksum']}")
    check("B8 the preferred variant has the best mean rank",
          min(rr["mean_rank"], key=rr["mean_rank"].get) == "v2_mixer_sharp", f"{rr['mean_rank']}")
    check("B8b it is exactly 1.0", abs(rr["mean_rank"]["v2_mixer_sharp"] - 1.0) < 1e-9)
    check("B8c it wins all its duels",
          all(d["v2_mixer_sharp"] == len(k_rank["items"])
              for name, d in rr["duels"].items() if "v2_mixer_sharp" in d), f"{rr['duels']}")
    check("B8d the absolute judgment is broken down",
          rr["judge"]["v2_mixer_sharp"]["o"] == len(k_rank["items"]))

    # --- B9: Wilson stays inside [0,1] at the edges ---------------------------------------------
    lo, hi = wilson(0, 5)
    check("B9 Wilson clamped at the bottom", lo >= 0.0 and hi <= 1.0, f"{lo},{hi}")
    lo, hi = wilson(5, 5)
    check("B9b Wilson clamped at the top", lo >= 0.0 and hi <= 1.0, f"{lo},{hi}")

    # --- B10: the Artifact ceiling is enforced at build time -----------------------------------
    old = boards.ARTIFACT_MAX_BYTES
    boards.ARTIFACT_MAX_BYTES = 100
    try:
        build_duel2([{"id": "s0", "arms": {"a": r_duel["key"], "b": r_duel["key"]}}],
                    os.path.join(TMP, "toobig"))
        raise SystemExit("FAIL B10: a board above the ceiling was accepted")
    except AssertionError as e:
        check("B10 the 16 MB ceiling is enforced", "ceiling" in str(e) or "MB" in str(e))
    finally:
        boards.ARTIFACT_MAX_BYTES = old

    # --- B11: the page is self-contained --------------------------------------------------------
    for r in (r_pair, r_rank, r_duel):
        h = page(r["html"])
        check(f"B11 no external resource in {os.path.basename(r['html'])}",
              "http://" not in h and "https://" not in h)

    print(f"\n{len(OK)}/{len(OK)} board selftests PASS")
finally:
    shutil.rmtree(TMP, ignore_errors=True)
