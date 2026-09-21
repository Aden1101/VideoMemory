#!/usr/bin/env python
"""The published tables as data. `read_table` turns one results/*/*/metrics.txt back into
numbers; it is the only parser of those files, used by the figure and by the tests.

    python scripts/results_index.py         # the 29 tables as JSON on stdout, one row per table

Plain Python, no dependency.
"""
import ast
import glob
import json
import os
import re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ARMS = ("memon", "memzero", "stock")
VBENCH = ("SC", "BC", "DD")


def table_paths():
    return sorted(glob.glob(os.path.join(ROOT, "results", "*", "*", "metrics.txt")))


def _number(text):
    return None if text in ("n/a", "None") else float(text)


def read_table(path):
    """One table as a dict: per-arm summaries, VBench scores and their gap to stock, the two
    paired distances, the identity gap to stock with its parity flag, the verdict, the noise
    floor used, and the raw lines."""
    lines = open(path).read().rstrip("\n").split("\n")
    text = "\n".join(lines)
    summaries, vb, vb_diffs = {}, {arm: {} for arm in ARMS}, {}
    for line in lines:
        cells = line.split()
        if cells and cells[0] in ARMS and len(cells) == 6:
            summaries[cells[0]] = {"n": int(cells[1]), "lag1": float(cells[2]), "lag2": float(cells[3]),
                                   "flow": _number(cells[4]), "id_cos": float(cells[5])}
        elif cells and cells[0] in VBENCH and len(cells) == 5:
            for arm, cell in zip(ARMS, cells[1:4]):
                vb[arm][cells[0]] = _number(cell)
            vb_diffs[cells[0]] = _number(cells[4])

    def paired(label):
        m = re.search(re.escape(label) + r"\s+DINO cos-dist = (\S+)\s+pixel-L2 = (\S+)\s+\(n=(\d+) paired\)",
                      text)
        return float(m.group(1)), float(m.group(2)), int(m.group(3))

    per_video = ast.literal_eval(re.search(r"per-prompt DINO cos-dist: (\{.*\})", text).group(1))
    drift = re.search(r"identity-drift\(memon\) - drift\(stock\) = (\S+)\s+\(parity=(\w+)\)", text)
    return {
        "lines": lines,
        "summaries": summaries, "vb": vb, "vb_diffs": vb_diffs,
        "on_zero": paired("delta(memon, memzero)") + (per_video,),
        "zero_stock": paired("delta(memzero, AF-stock)") + ({},),
        "drift_diff": _number(drift.group(1)),
        "drift_par": {"True": True, "False": False, "None": None}[drift.group(2)],
        "verdict": text.split("VERDICT: ", 1)[1].rsplit("\n", 1)[0],
        "noise_floor": float(re.search(r"NOISE_FLOOR\(DINO cos-dist\)=(\S+)", text).group(1)),
    }


def index_rows():
    """One row per table: where it is, how it was evaluated, and its numbers per arm."""
    rows = []
    for path in table_paths():
        run, evaluation = os.path.relpath(path, os.path.join(ROOT, "results")).split(os.sep)[:2]
        step = re.match(r"eval(\d+)", evaluation)
        t = read_table(path)
        rows.append({
            "run": run, "eval": evaluation, "step": int(step.group(1)) if step else None,
            "arms": {arm: {**t["summaries"][arm], **t["vb"][arm]} for arm in ARMS},
            "memon_vs_memzero_dino": t["on_zero"][0], "memzero_vs_stock_dino": t["zero_stock"][0],
            "verdict": t["verdict"].split(" ", 1)[0],
        })
    return rows


if __name__ == "__main__":
    print(json.dumps(index_rows(), indent=1))
