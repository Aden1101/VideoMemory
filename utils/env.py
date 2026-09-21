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


def env_on(name):
    """Is this arm switched on: set to something other than empty, zero or a negative number."""
    value = env_str(name)
    try:
        return float(value) > 0
    except ValueError:
        return bool(value)


# Every variable the code reads to switch an arm or a control. Each is documented where it is
# read (pipeline/switch/, wan/modules/memory_inject.py, model/twin_objective.py).
# tests/test_env_registry.py keeps this list equal to what the source actually reads.
ARM_VARIABLES = frozenset("""
ANCHOR_BLUR_SIGMA ANCHOR_DUMP ANCHOR_FROM_FILE
FRAMEBANK FRAMEBANK_GT FRAMEBANK_TAU FRAMEBANK_TIE
MEM_ADDITIVE_KV MEM_ADD_GAIN MEM_ADD_MASS_DUMP MEM_ADD_MASS_SRC MEM_ADD_POS
MEM_ANCHOR_T MEM_ANCHOR_T_SEED
MEM_COMP_ALPHA MEM_COMP_ANCHOR_NOISE MEM_COMP_CROP MEM_COMP_DUMP MEM_COMP_FEATHER MEM_COMP_FILL
MEM_COMP_FILL_FLOOR MEM_COMP_FILL_SIGMA MEM_COMP_GATE_HI MEM_COMP_GATE_LO MEM_COMP_INVERT
MEM_COMP_PROMPT_GATE MEM_COMP_SHIFT MEM_COMP_UNIFORM MEM_COMP_UNION
MEM_DC_REMOVE MEM_DECODE MEM_DECODE_COMPOSITE MEM_DECODE_DUMP MEM_DECODE_NOISE
MEM_DECODE_NOISE_SEED MEM_DECODE_ROUTED MEM_DECODE_SLOTS MEM_DECODE_SLOT_OFFSET MEM_DECODE_SRC
MEM_DECODE_THRU MEM_EPISODIC MEM_EVICT_DEBUG MEM_REJECT_STREAM
MEM_SDEDIT MEM_SDEDIT_FRAMES MEM_SDEDIT_SEED MEM_SINKPURGE MEM_STREAM_SCALE MEM_WRITER
MEM_ZERO_ALL
REPLAY_PAST REPLAY_PAST_BLOCKS REPLAY_PAST_FRAMES REPLAY_PAST_NOISE REPLAY_PAST_POS
REPLAY_PAST_SLOT
TWIN_SELFTEST
""".split())
_ARM_PREFIXES = ("MEM_", "REPLAY_", "ANCHOR_", "FRAMEBANK", "TWIN_")


def check_arm_variables():
    """Refuse to run with an arm variable the code does not read. A misspelt name would
    otherwise run the default arm and label its output as the one that was asked for."""
    unknown = sorted(k for k in os.environ if k.startswith(_ARM_PREFIXES) and k not in ARM_VARIABLES)
    if unknown:
        raise ValueError(f"unknown arm variable(s) {unknown}: not read anywhere in this code. "
                         f"See utils/env.py for the ones that are.")

