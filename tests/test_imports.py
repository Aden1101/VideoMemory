"""Import every module this work wrote or touched, for real, with the heavy third-party
packages replaced by stubs. No torch needed, a second to run, Python 3.10 or later (upstream
writes `X | None` in signatures).

It catches what compiling a file cannot: an import cycle, a name missing at import time, a
class whose bases do not combine, a decorator applied to the wrong thing. It runs no function
body, so it says nothing about behaviour.

    python tests/test_imports.py        # or: python -m pytest tests/test_imports.py
"""
import importlib
import importlib.abc
import importlib.machinery
import os
import sys
import types

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODULES = [
    "wan.modules.video_memory", "wan.modules.memory_inject", "wan.modules.additive_attention",
    "wan.modules.anchor_decoder", "wan.modules.causal_model",
    "utils.env", "utils.device", "utils.prompt_cache", "utils.video_io", "utils.wan_wrapper",
    "pipeline.switch", "pipeline.switch.noise", "pipeline.switch.schedule",
    "pipeline.switch.driver", "pipeline.switch_causal_inference", "pipeline.memory_rollout",
    "pipeline.streaming_training", "pipeline.streaming_switch_training",
    "model.twin_objective", "model.streaming_training",
    "trainer.memory_training", "trainer.distillation",
]
# top-level packages to stub: everything heavy the repository imports
STUBBED = {
    "torch", "einops", "diffusers", "omegaconf", "peft", "wandb", "tqdm", "numpy", "cv2",
    "imageio", "av", "lmdb", "PIL", "torchvision", "transformers", "tokenizers", "ftfy", "regex",
    "easydict", "datasets", "safetensors", "accelerate", "open_clip", "dashscope", "matplotlib",
    "huggingface_hub", "flash_attn", "sageattention", "xfuser", "yunchang", "timm", "scipy",
    "pycocotools", "decord", "sentencepiece",
}


class _StubType(type):
    """Metaclass of a stand-in: any attribute is another stand-in, it can be subclassed,
    subscripted, iterated (empty), used in `X | None`, and called, in which case it hands a
    function back unchanged so that it also works as a decorator."""

    def __getattr__(cls, name):
        if name.startswith("__"):
            raise AttributeError(name)
        return _stub(name)

    def __call__(cls, *args, **kwargs):
        if len(args) == 1 and callable(args[0]) and not kwargs and not isinstance(args[0], type):
            return args[0]
        return object.__new__(Stub)

    def __getitem__(cls, item):
        return cls

    def __iter__(cls):
        return iter(())

    def __or__(cls, other):
        return cls

    __ror__ = __or__


class Stub(metaclass=_StubType):
    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        return Stub()

    def __call__(self, *args, **kwargs):
        if len(args) == 1 and callable(args[0]) and not kwargs and not isinstance(args[0], type):
            return args[0]
        return Stub()

    def __getitem__(self, item):
        return Stub()

    def __iter__(self):
        return iter(())

    def __bool__(self):
        return False

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def _arith(self, *other):
        return Stub()

    __add__ = __radd__ = __sub__ = __rsub__ = __mul__ = __rmul__ = _arith
    __truediv__ = __rtruediv__ = __pow__ = __rpow__ = __floordiv__ = __mod__ = __neg__ = _arith


def _stub(name):
    return _StubType(name, (Stub,), {})


class _StubFinder(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    def find_spec(self, name, path, target=None):
        if name.split(".")[0] in STUBBED:
            return importlib.machinery.ModuleSpec(name, self, is_package=True)
        return None

    def create_module(self, spec):
        module = types.ModuleType(spec.name)
        module.__path__ = []

        def attribute(name):
            if name.startswith("__"):
                raise AttributeError(name)
            return _stub(name)

        module.__getattr__ = attribute
        return module

    def exec_module(self, module):
        pass


def import_all():
    """{module: error message} for every module that failed to import."""
    finder = _StubFinder()
    sys.meta_path.insert(0, finder)
    sys.path.insert(0, ROOT)
    before = set(sys.modules)
    failures = {}
    try:
        for name in MODULES:
            try:
                importlib.import_module(name)
            except Exception as error:                  # noqa: BLE001, reported below
                failures[name] = f"{type(error).__name__}: {error}"
    finally:
        sys.meta_path.remove(finder)
        for name in set(sys.modules) - before:          # leave no stub behind for other tests
            del sys.modules[name]
    return failures


def test_every_module_imports():
    if sys.version_info < (3, 10):
        import pytest
        pytest.skip("upstream's signatures need Python 3.10")
    failures = import_all()
    assert not failures, failures


if __name__ == "__main__":
    if sys.version_info < (3, 10):
        sys.exit("needs Python 3.10 or later: upstream writes `X | None` in signatures")
    failed = import_all()
    for name, error in failed.items():
        print(f"  FAIL {name}: {error}")
    print(f"{len(MODULES) - len(failed)}/{len(MODULES)} modules import")
    sys.exit(1 if failed else 0)
