"""Sentences of the README that the repository can check by itself. No torch, no plotting.

    python tests/test_readme_claims.py      # or: python -m pytest tests/test_readme_claims.py
"""
import glob
import importlib.util
import os
import re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUN_CONFIGS = sorted(glob.glob(os.path.join(ROOT, "configs", "runs", "*.yaml")))


def _make_figure():
    spec = importlib.util.spec_from_file_location(
        "_make_figure_under_test", os.path.join(ROOT, "scripts", "make_figure.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)             # matplotlib is only imported when drawing
    return module


def test_identity_is_lower_with_the_memory_at_10_of_15_checkpoints_by_007():
    figure = _make_figure()
    gaps = figure.identity_gaps({run: figure.load(run) for run, _ in figure.RUNS})
    assert len(gaps) == 15 and sum(g < 0 for g in gaps) == 10
    assert abs(sum(gaps) / len(gaps) - (-0.07)) < 0.005


def test_no_arm_matches_its_reference_on_both_metrics():
    figure = _make_figure()
    figure.check_claims({run: figure.load(run) for run, _ in figure.RUNS})      # asserts inside


def test_the_four_runs_are_plain_dmd():
    """Every extra loss term is weighted zero: no run config sets a weight of the twin
    objective, nor the identity and flicker weights, to anything but 0."""
    source = open(os.path.join(ROOT, "model", "twin_objective.py")).read()
    weights = set(re.findall(r"^    (\w+_weight): float = 0\.0", source, re.M))
    assert {"adv_loss_weight", "mm_rel_weight", "dnh_weight", "act_weight"} <= weights, weights
    weights |= {"id_loss_weight", "tv_loss_weight"}
    assert len(RUN_CONFIGS) == 4
    for path in RUN_CONFIGS:
        for name, value in re.findall(r"^\s*(\w+_weight):\s*([^\s#]+)", open(path).read(), re.M):
            if name in weights:
                assert float(value) == 0.0, (os.path.basename(path), name, value)


def test_the_memory_sits_in_blocks_5_16_and_27():
    for path in RUN_CONFIGS:
        block = re.search(r"memory_layers:\s*\n((?:\s*-\s*\d+\s*\n)+)", open(path).read())
        assert block, os.path.basename(path)
        assert [int(n) for n in re.findall(r"\d+", block.group(1))] == [5, 16, 27], path


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"  ok  {t.__name__}")
    print(f"\n{len(tests)}/{len(tests)} PASS")
