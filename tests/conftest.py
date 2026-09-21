"""Without torch, pytest collects only the tests that do not need it."""
import importlib.util

NEEDS_TORCH = ["test_device.py", "test_grad.py", "test_memory_cell.py", "test_prompt_cache.py",
               "test_inference_smoke.py", "test_switch_window.py"]
collect_ignore = [] if importlib.util.find_spec("torch") else NEEDS_TORCH
