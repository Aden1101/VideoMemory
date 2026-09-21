"""Ways of bringing a past subject back at a prompt switch, one module per idea.

Each module holds a mixin that SwitchCausalInferencePipeline inherits from, plus the controls
that tell whether an idea helps because of what it carries or just by being there. Nothing here
runs unless its environment variable is set.
"""
from pipeline.switch.additive_kv import AdditiveKVMixin
from pipeline.switch.anchor_controls import AnchorControlsMixin
from pipeline.switch.decoded_anchor import AnchorDecodeMixin
from pipeline.switch.composite import CompositeMixin
from pipeline.switch.frame_bank import FrameBankMixin
from pipeline.switch.memory_state import MemoryStateMixin
from pipeline.switch.replay import ReplayPastMixin
from pipeline.switch.sdedit import SdeditMixin
from pipeline.switch.sink import SinkMixin

__all__ = ["AdditiveKVMixin", "AnchorControlsMixin", "AnchorDecodeMixin", "CompositeMixin",
           "FrameBankMixin", "MemoryStateMixin", "ReplayPastMixin", "SdeditMixin", "SinkMixin"]
