"""scripts/eval_metrics.py against the tables it produced: every results/*/*/metrics.txt is read
back into numbers, and the script's own decision and report code must give the file again.
No model, no video; torch is replaced by a stand-in when it is not installed.

Only the title line differs by design: the published tables were relabelled with their run.
The numbers in a table are rounded, so a verdict that sat within rounding of a threshold could
legitimately differ; none of the published ones does.

    python tests/test_eval_report.py        # or: python -m pytest tests/test_eval_report.py
"""
import importlib.util
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from scripts.results_index import INDEX, index_rows, read_table, table_paths  # noqa: E402

TABLES = table_paths()


def _load_script():
    finder = None
    try:
        import torch  # noqa: F401
    except ImportError:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from test_imports import _StubFinder
        finder = _StubFinder()
        sys.meta_path.insert(0, finder)
    before = set(sys.modules)
    try:
        spec = importlib.util.spec_from_file_location(
            "_eval_metrics_under_test", os.path.join(ROOT, "scripts", "eval_metrics.py"))
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        if finder is not None:
            sys.meta_path.remove(finder)
            for name in set(sys.modules) - before:      # leave no stand-in behind for other tests
                del sys.modules[name]


metrics = _load_script()


def test_there_are_tables_to_check():
    assert len(TABLES) >= 29, len(TABLES)


def test_the_index_says_what_the_tables_say():
    assert json.load(open(INDEX)) == index_rows(), "run python scripts/results_index.py"


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


def test_verdicts_on_the_cases_no_published_table_shows():
    """Every branch of the decision, checked against the rule written out flat: the published
    tables are all ACTIVE-NEGATIVE or close to it, and none has a missing number."""
    nan = float("nan")
    lag, floor = metrics.FREEZE_LAG, 0.18

    def flat(s_on, d, vb_par, drift_par):
        frozen = s_on["lag2"] < lag and s_on["lag1"] < lag
        flicker = s_on["lag2"] < lag and s_on["lag1"] >= lag
        if frozen:
            return "FREEZE"
        if flicker:
            return "FLICKER"
        if not (d >= floor):
            return "INERT"
        if vb_par and drift_par:
            return "ACTIVE-NEUTRAL"
        return "ACTIVE (" if vb_par is None or drift_par is None else "ACTIVE-POSITIVE"   # gap +0.05

    for lag1 in (0.0, 0.5, nan):
        for lag2 in (0.0, 0.5, nan):
            for d in (0.0, 0.5, nan):
                for vb_par in (True, False, None):
                    for drift_par in (True, False, None):
                        s_on = {"lag1": lag1, "lag2": lag2}
                        verdict = metrics.verdict_of(s_on, d, floor, {"SC": 0.1, "BC": None, "DD": -0.1},
                                                     vb_par, 0.05, drift_par)
                        assert verdict.startswith(flat(s_on, d, vb_par, drift_par)), (s_on, d, verdict)


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"  ok  {t.__name__}")
    print(f"\n{len(tests)}/{len(tests)} PASS on {len(TABLES)} tables")
