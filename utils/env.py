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
    "MEM_WRITER": "checkpoint of write parameters trained offline, laid over the cells",
    "MEM_EPISODIC": "1 snapshots the state when a subject is left and restores it at its return",
    "TWIN_SELFTEST": "1 keeps the memory on in the training twin, to check the twin is exact",
    # replaying real past frames at a prompt switch
    "REPLAY_PAST": "1 fills the attention window with the frames of the subject's first appearance",
    "REPLAY_PAST_FRAMES": "only the last k window slots come from the past; 0 is the whole window",
    "REPLAY_PAST_POS": "head: the oldest end of the window; sink0: the sink slot instead",
    "REPLAY_PAST_SLOT": "a single past frame, j slots before the most recent one",
    "REPLAY_PAST_NOISE": "standard deviation of the noise that degrades the replayed frames",
    "REPLAY_PAST_BLOCKS": "comma-separated transformer blocks that keep the replayed window",
    # the frame bank baseline (Echo-Forcing)
    "FRAMEBANK": "key: route past frames back by prompt cosine; truth: by ground-truth subject",
    "FRAMEBANK_TAU": "cosine threshold of the router, 0.90 by default",
    "FRAMEBANK_TIE": "recent (default) or first: which banked frame wins a tie",
    "FRAMEBANK_GT": "1 takes the moment of the recall from the schedule, not from the router",
    # an anchor decoded from the memory state
    "MEM_DECODE": "checkpoint of the decoder that turns the memory state into an anchor frame",
    "MEM_DECODE_THRU": "1 decodes the state when the anchor chunk is evicted, not at the return",
    "MEM_DECODE_SLOTS": "repeat the anchor over the last k window slots",
    "MEM_DECODE_SLOT_OFFSET": "j >= 0: the anchor replaces the true frame j slots before the latest (schedules only)",
    "MEM_DECODE_ROUTED": "1 recalls the decoded anchor only when the frame bank's router says so",
    "MEM_DECODE_DUMP": "path prefix where the decoded anchor latent is saved",
    "MEM_DECODE_SRC": "control: the anchor decoded from another video's memory, same decoder",
    "MEM_DECODE_NOISE": "control: mix the anchor with noise at the window's statistics, 1 is all noise",
    "MEM_DECODE_NOISE_SEED": "seed of that noise, so paired runs do not share one draw",
    # controls on a real anchor
    "ANCHOR_DUMP": "path where the real anchor latent of the first appearance is saved",
    "ANCHOR_FROM_FILE": "control: replace the anchor with a latent frame loaded from this file",
    "ANCHOR_BLUR_SIGMA": "control: blur the anchor in pixel space with this radius",
    "MEM_ANCHOR_T": "noise the anchor to level t in (0, 1000] and declare that t for its slots",
    "MEM_ANCHOR_T_SEED": "seed of that noise",
    # sharpening the anchor
    "MEM_SDEDIT": "noise level of the noise-then-denoise pass that sharpens the anchor",
    "MEM_SDEDIT_FRAMES": "frames in the block that pass runs on; 1 is its control",
    "MEM_SDEDIT_SEED": "seed of that pass",
    # compositing the anchor onto the current background
    "MEM_DECODE_COMPOSITE": "1 blends the anchor's centre onto the last true frame, not the whole frame",
    "MEM_COMP_CROP": "fraction of height and width inside the elliptical mask, 0.6 by default",
    "MEM_COMP_FEATHER": "relative half-width of the mask's soft edge, 0.3 by default",
    "MEM_COMP_ALPHA": "dose of the composite in [0, 1]; 0 gives the anchor back untouched",
    "MEM_COMP_PROMPT_GATE": "1 scales that dose by how close the entering and leaving prompts are",
    "MEM_COMP_GATE_LO": "prompt cosine under which the gated dose is 0, 0.50 by default",
    "MEM_COMP_GATE_HI": "prompt cosine above which the gated dose is 1, 0.80 by default",
    "MEM_COMP_UNION": "1 masks the union of the anchor's subject and the current subject",
    "MEM_COMP_FILL": "noise, priority or floor: what fills the union where the two subjects differ",
    "MEM_COMP_FILL_FLOOR": "share of the anchor kept everywhere under fill=floor, 0.4 by default",
    "MEM_COMP_FILL_SIGMA": "how much of that filler is noise, 1 is pure noise",
    "MEM_COMP_SHIFT": "control: mask centre moved by this fraction of the width",
    "MEM_COMP_INVERT": "control: 1 inverts the mask, anchor at the border",
    "MEM_COMP_UNIFORM": "control: 1 flattens the mask to its mean, same dose without the shape",
    "MEM_COMP_ANCHOR_NOISE": "control: noise mixed into the anchor, mask kept",
    "MEM_COMP_DUMP": "path prefix where background, anchor, composite and mask are saved",
    # the anchor as extra attention tokens, and the sink
    "MEM_ADDITIVE_KV": "1 adds the anchor as extra key/value tokens instead of evicting a frame",
    "MEM_ADD_GAIN": "attention-mass gain of those tokens; 1 is plain concatenation",
    "MEM_ADD_POS": "RoPE frame of those tokens, or auto: one frame before the window",
    "MEM_ADD_MASS_SRC": "1 reports the attention mass per source: sink, window, anchor",
    "MEM_ADD_MASS_DUMP": "file that receives that report, one entry per chunk",
    "MEM_SINKPURGE": "1 overwrites the sink with the current scene at a revisit, 2 at every switch",
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
