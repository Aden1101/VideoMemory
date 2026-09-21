"""scripts/eval_metrics.py against the tables it produced: every results/*/*/metrics.txt is read
back into numbers, and the script's own decision and report code must give the file again.
No model, no video; torch is replaced by a stand-in when it is not installed.

Only the title line differs by design: the published tables were relabelled with their run.
The numbers in a table are rounded, so a verdict that sat within rounding of a threshold could
legitimately differ; none of the published ones does.

    python tests/test_eval_report.py        # or: python -m pytest tests/test_eval_report.py
"""
import ast
import glob
import importlib.util
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TABLES = sorted(glob.glob(os.path.join(ROOT, "results", "*", "*", "metrics.txt")))


def _load_script():
    finder = None
    try:
        import torch  # noqa: F401
    except ImportError:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from test_imports import _StubFinder
        finder = _StubFinder()
        sys.meta_path.insert(0, finder)
    try:
        spec = importlib.util.spec_from_file_location(
            "_eval_metrics_under_test", os.path.join(ROOT, "scripts", "eval_metrics.py"))
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        if finder is not None:
            sys.meta_path.remove(finder)


metrics = _load_script()


def _number(text):
    return None if text in ("n/a", "None") else float(text)


def read_table(path):
    """The inputs of render_report, recovered from a published table."""
    lines = open(path).read().rstrip("\n").split("\n")
    text = "\n".join(lines)
    summaries, vb, vb_diffs = {}, {arm: {} for arm in metrics.ARMS}, {}
    for line in lines:
        cells = line.split()
        if cells and cells[0] in metrics.ARMS and len(cells) == 6:
            summaries[cells[0]] = {"n": int(cells[1]), "lag1": float(cells[2]), "lag2": float(cells[3]),
                                   "flow": _number(cells[4]), "id_cos": float(cells[5])}
        elif cells and cells[0] in metrics.VB_DIMS.values() and len(cells) == 5:
            for arm, cell in zip(metrics.ARMS, cells[1:4]):
                vb[arm][cells[0]] = _number(cell)
            vb_diffs[cells[0]] = _number(cells[4])

    def paired(label):
        m = re.search(re.escape(label) + r"\s+DINO cos-dist = (\S+)\s+pixel-L2 = (\S+)\s+\(n=(\d+) paired\)", text)
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


def test_there_are_tables_to_check():
    assert len(TABLES) >= 29, len(TABLES)


def test_the_report_code_writes_every_published_table_again():
    for path in TABLES:
        t = read_table(path)
        report = metrics.render_report(t["summaries"], t["vb"], t["vb_diffs"], t["on_zero"],
                                       t["zero_stock"], t["drift_diff"], t["drift_par"],
                                       t["verdict"], t["noise_floor"]).split("\n")
        assert len(report) == len(t["lines"]), path
        for i, (ours, published) in enumerate(zip(report, t["lines"])):
            if i == 1:                                  # the relabelled title
                assert published.endswith(ours), (path, published)
            else:
                assert ours == published, (path, i, ours, published)


def test_the_decision_code_reaches_every_published_verdict():
    for path in TABLES:
        t = read_table(path)
        vb_diffs, vb_par, drift_diff, drift_par = metrics.compare_to_stock(
            t["summaries"]["memon"], t["summaries"]["stock"], t["vb"])
        assert vb_diffs == t["vb_diffs"], (path, vb_diffs, t["vb_diffs"])
        # the identity gap is printed from unrounded means: use the published one for the verdict
        verdict = metrics.verdict_of(t["summaries"]["memon"], t["on_zero"][0], t["noise_floor"],
                                     vb_diffs, vb_par, t["drift_diff"], t["drift_par"])
        assert verdict == t["verdict"], (path, verdict, t["verdict"])


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"  ok  {t.__name__}")
    print(f"\n{len(tests)}/{len(tests)} PASS on {len(TABLES)} tables")
