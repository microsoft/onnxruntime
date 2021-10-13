# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

from . import ONNX_OPSET_VERSION
from ._training_manager import TrainingManager
from ._inference_manager import InferenceManager
from .debug_options import DebugOptions
from ._fallback import _FallbackManager


class GraphExecutionManagerFactory(object):
    def __init__(self, module, debug_options: DebugOptions, fallback_manager: _FallbackManager,
                 opset_version=ONNX_OPSET_VERSION):
        self._training_manager = TrainingManager(module, debug_options, fallback_manager,
                                                 opset_version=opset_version)
        self._inference_manager = InferenceManager(module, debug_options, fallback_manager,
                                                   opset_version=opset_version)

    def __call__(self, is_training):
        if is_training:
            return self._training_manager
        else:
            return self._inference_manager
