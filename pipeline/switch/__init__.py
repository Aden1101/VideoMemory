# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES
#
# Licensed under the Apache License, Version 2.0 (the "License").
# You may not use this file except in compliance with the License.
# To view a copy of this license, visit http://www.apache.org/licenses/LICENSE-2.0
#
# No warranties are given. The work is provided "AS IS", without warranty of any kind, express or implied.
#
# SPDX-License-Identifier: Apache-2.0
"""Ways of bringing a past subject back at a prompt switch, one module per idea.

Each module holds a mixin that SwitchCausalInferencePipeline inherits from, plus the controls
that tell whether an idea helps because of what it carries or just by being there. Nothing here
runs unless its environment variable is set.
"""
from pipeline.switch.additive_kv import AdditiveKVMixin
from pipeline.switch.anchor_controls import AnchorControlsMixin
from pipeline.switch.anchor_decode import AnchorDecodeMixin
from pipeline.switch.composite import CompositeMixin
from pipeline.switch.frame_bank import FrameBankMixin
from pipeline.switch.memory_state import MemoryStateMixin
from pipeline.switch.sdedit import SdeditMixin
from pipeline.switch.sink import SinkMixin

__all__ = ["AdditiveKVMixin", "AnchorControlsMixin", "AnchorDecodeMixin", "CompositeMixin",
           "FrameBankMixin", "MemoryStateMixin", "SdeditMixin", "SinkMixin"]
