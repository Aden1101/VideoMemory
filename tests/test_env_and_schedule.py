"""The arm switches and the prompt schedule. No torch, no model: runs anywhere in a second.

    python -m pytest tests/test_env_and_schedule.py      # or: python tests/test_env_and_schedule.py
"""
import importlib.util
import os
import re
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from utils import env  # noqa: E402


def _load(path, name):
    """Import one file without its package, whose __init__ pulls torch in."""
    spec = importlib.util.spec_from_file_location(name, os.path.join(ROOT, path))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module          # dataclasses looks its module up there
    spec.loader.exec_module(module)
    return module


schedule = _load("pipeline/switch/schedule.py", "_schedule_under_test")


# ---- reading the environment ---------------------------------------------------------------
def _with(value):
    os.environ.pop("MEM_TEST", None)
    if value is not None:
        os.environ["MEM_TEST"] = value


def test_unset_and_empty_mean_the_default():
    for value in (None, ""):
        _with(value)
        assert env.env_int("MEM_TEST") == 0 and env.env_int("MEM_TEST", -1) == -1
        assert env.env_float("MEM_TEST", 0.6) == 0.6
        assert env.env_str("MEM_TEST") == "" and not env.env_on("MEM_TEST")
    os.environ.pop("MEM_TEST", None)


def test_set_values():
    _with(" 2 ")
    assert env.env_int("MEM_TEST") == 2 and env.env_float("MEM_TEST") == 2.0
    assert env.env_str("MEM_TEST") == "2" and env.env_on("MEM_TEST")
    for off in ("0", "0.0", "-1"):
        _with(off)
        assert not env.env_on("MEM_TEST")
    _with("ckpt/decoder.pt")                     # a path switches an arm on too
    assert env.env_on("MEM_TEST")
    os.environ.pop("MEM_TEST", None)


def test_the_registry_lists_exactly_what_the_source_reads():
    files = subprocess.check_output(["git", "ls-files", "*.py"], cwd=ROOT).decode().split()
    read = set()
    readers = r"(?:env_int|env_float|env_str|env_on|environ\.get|environ\.pop|environ)"
    pattern = re.compile(readers + r"""[(\[]\s*["']([A-Z][A-Z_0-9]*)["']""")
    for f in files:
        if f.startswith("tests/"):
            continue
        read |= {n for n in pattern.findall(open(os.path.join(ROOT, f)).read())
                 if n.startswith(env._ARM_PREFIXES)}
    listed = set(env.ARM_VARIABLES)
    assert read == listed, (sorted(read - listed), sorted(listed - read))
    assert all(what.strip() for what in env.ARM_VARIABLES.values())


def test_a_misspelt_arm_is_refused():
    os.environ["MEM_DECOD"] = "1"
    try:
        env.check_arm_variables()
    except ValueError as e:
        assert "MEM_DECOD" in str(e)
    else:
        raise AssertionError("a misspelt arm variable was accepted")
    finally:
        del os.environ["MEM_DECOD"]
    env.check_arm_variables()                    # and a clean environment passes


# ---- the schedule --------------------------------------------------------------------------
def test_a_b_a_is_one_revisit():
    plan = schedule.parse_schedule([(0, 0), (21, 1), (42, 0)], 2, None, 63, 3)
    assert plan.is_revisit == [False, False, True]
    assert plan.end == [21, 42, 63]
    assert plan.first_appearance == {0: 0, 1: 1}
    # the anchor of the revisit is the last frame of the segment that first showed its subject
    assert plan.end[plan.first_appearance[plan.subject[2]]] - 1 == 20


def test_a_paraphrase_is_a_revisit_only_when_subjects_say_so():
    sched = [(0, 0), (21, 1), (42, 2)]           # three different prompt strings
    assert parse(sched).is_revisit == [False, False, False]
    assert parse(sched, subjects=[7, 8, 7]).is_revisit == [False, False, True]


def parse(sched, subjects=None):
    return schedule.parse_schedule(sched, 3, subjects, 63, 3)


def test_anchor_and_return_of_each_segment():
    # A, B, A again, C: only A comes back, and its anchor closes its first segment
    plan = schedule.parse_schedule([(0, 0), (21, 1), (42, 0), (51, 2)], 3, None, 63, 3)
    assert [plan.returns_later(i) for i in range(4)] == [True, False, False, False]
    assert plan.anchor_end(2) == 21 and plan.anchor_end(1) == 42 and plan.anchor_end(3) == 63
    lines = plan.describe(["a red fox", "a blue car", "a green boat"])
    assert len(lines) == 5 and "revisits at segments [2]" in lines[0]
    assert "REVISIT of seg0" in lines[3] and "REVISIT" not in lines[2]


def test_invalid_schedules_are_refused():
    for bad, subjects in (([(3, 0)], None),                    # does not start at 0
                          ([(0, 0), (20, 1)], None),           # not on a block boundary
                          ([(0, 0), (21, 1), (21, 0)], None),  # not strictly increasing
                          ([(0, 0), (63, 1)], None),           # empty last segment
                          ([(0, 5)], None),                    # prompt index out of range
                          ([(0, 0), (21, 1)], [0])):           # subjects of the wrong length
        try:
            parse(bad, subjects)
        except AssertionError:
            continue
        raise AssertionError(f"accepted {bad} with subjects {subjects}")


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"  ok  {t.__name__}")
    print(f"\n{len(tests)}/{len(tests)} PASS")
