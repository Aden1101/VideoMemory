"""Ways of bringing a past subject back at a prompt switch, one module per idea.

Each module holds a mixin that SwitchCausalInferencePipeline inherits from, plus the controls
that tell whether an idea helps because of what it carries or just by being there. Nothing here
runs unless its environment variable is set.

Two ideas are kept, the ones the README reports: real past frames (replay.py) and an anchor
decoded from the memory (decoded_anchor.py). The others that were tried (a subject/background
composite, the anchor as extra attention tokens, a frame bank routed by prompt similarity, a
sharpening pass, a sink refresh) are at the git tag `with-all-arms`.
"""
from pipeline.switch.anchor_controls import AnchorControlsMixin
from pipeline.switch.decoded_anchor import AnchorDecodeMixin
from pipeline.switch.replay import ReplayPastMixin

__all__ = ["AnchorControlsMixin", "AnchorDecodeMixin", "ReplayPastMixin"]
