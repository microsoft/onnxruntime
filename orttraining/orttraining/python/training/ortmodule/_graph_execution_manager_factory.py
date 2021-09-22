# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

from ._training_manager import TrainingManager
from ._inference_manager import InferenceManager
from .debug_options import DebugOptions
from ._fallback import _FallbackManager
from typing import Union


class GraphExecutionManagerFactory(object):
    def __init__(self, module, debug_options: DebugOptions, fallback_manager: _FallbackManager, custom_op_set: Union[bool, set] = False):
        self._training_manager = TrainingManager(module, debug_options, fallback_manager, custom_op_set)
        self._inference_manager = InferenceManager(module, debug_options, fallback_manager, custom_op_set)

    def __call__(self, is_training):
        if is_training:
            return self._training_manager
        else:
            return self._inference_manager
