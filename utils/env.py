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


def env_choice(name, allowed, default=""):
    """A variable that takes one of a few words, lower-cased; anything else is refused rather
    than silently run as the default arm."""
    value = env_str(name).lower() or default
    if value != default and value not in allowed:
        raise ValueError(f"{name}={value!r}: expected one of {sorted(allowed)}"
                         + (f", or unset for {default!r}" if default else ", or unset"))
    return value


def env_on(name):
    """Is this arm switched on: set to something other than empty, zero or a negative number."""
    value = env_str(name)
    try:
        return float(value) > 0
    except ValueError:
        return bool(value)


# Every variable the code reads to switch an arm or a control, with one line on what it does.
# The full story of each is where it is read (pipeline/switch/, wan/modules/memory_inject.py,
# model/twin_objective.py). All are off when unset. `python -m utils.env` prints this table, and
# tests/test_env_and_schedule.py keeps it equal to what the source actually reads.
ARM_VARIABLES = {
    # the memory branch itself
    "MEM_ZERO_ALL": "1 zeroes the memory injection and nothing else: the memory-off arm",
    "MEM_STREAM_SCALE": "overrides the config's injection scale, a fraction of the stream's RMS",
    "MEM_REJECT_STREAM": "1 removes the part of the injection parallel to the residual stream",
    "MEM_DC_REMOVE": "1 subtracts the token mean of the injection, as the dc_remove option does",
    "MEM_EVICT_DEBUG": "1 prints which chunk each memory block is fed by the eviction queue",
    "TWIN_SELFTEST": "1 keeps the memory on in the training twin, to check the twin is exact",
    # replaying real past frames at a prompt switch
    "REPLAY_PAST": "1 fills the attention window with the frames of the subject's first appearance",
    "REPLAY_PAST_FRAMES": "only the last k window slots come from the past; 0 is the whole window",
    "REPLAY_PAST_POS": "head: the replayed slots go to the oldest end of the window",
    "REPLAY_PAST_SLOT": "a single past frame, j slots before the most recent one",
    "REPLAY_PAST_NOISE": "standard deviation of the noise that degrades the replayed frames",
    "REPLAY_PAST_BLOCKS": "comma-separated transformer blocks that keep the replayed window",
    # an anchor decoded from the memory state
    "MEM_DECODE": "checkpoint of the decoder that turns the memory state into an anchor frame",
    "MEM_DECODE_SLOTS": "repeat the anchor over the last k window slots",
    "MEM_DECODE_SLOT_OFFSET": "j >= 0: the anchor replaces the true frame j slots before the latest one",
    "MEM_DECODE_DUMP": "path prefix where the decoded anchor latent is saved",
    "MEM_DECODE_SRC": "control: the anchor decoded from another video's memory, same decoder",
    "MEM_DECODE_NOISE": "control: mix the anchor with noise at the window's statistics, 1 is all noise",
    "MEM_DECODE_NOISE_SEED": "seed of that noise, so paired runs do not share one draw",
    # controls on a real anchor
    "ANCHOR_DUMP": "path where the real anchor latent of the first appearance is saved",
    "ANCHOR_FROM_FILE": "control: replace the anchor with a latent frame loaded from this file",
    "ANCHOR_BLUR_SIGMA": "control: blur the anchor in pixel space with this radius",
}
_ARM_PREFIXES = ("MEM_", "REPLAY_", "ANCHOR_", "FRAMEBANK", "TWIN_")


def check_arm_variables():
    """Refuse to run with an arm variable the code does not read. A misspelt name would
    otherwise run the default arm and label its output as the one that was asked for."""
    unknown = sorted(k for k in os.environ if k.startswith(_ARM_PREFIXES) and k not in ARM_VARIABLES)
    if unknown:
        raise ValueError(f"unknown arm variable(s) {unknown}: not read anywhere in this code. "
                         f"See utils/env.py for the ones that are.")


if __name__ == "__main__":
    width = max(map(len, ARM_VARIABLES))
    for name, what in ARM_VARIABLES.items():
        print(f"{name:<{width}}  {what}")
