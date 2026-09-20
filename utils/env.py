"""Reading the environment variables that switch the measurement arms on.

Every arm under pipeline/switch/ is off unless its variable is set, and a variable set to the
empty string counts as unset, so `MEM_X= python ...` turns an arm off the same way as not
exporting it.
"""
import os


def env_int(name, default=0):
    return int(os.environ.get(name, "") or default)


def env_float(name, default=0.0):
    return float(os.environ.get(name, "") or default)


def env_str(name):
    return os.environ.get(name, "").strip()
