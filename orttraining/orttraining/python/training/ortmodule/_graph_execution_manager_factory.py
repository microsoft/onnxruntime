# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

from ._training_manager import TrainingManager
from ._inference_manager import InferenceManager


class GraphExecutionManagerFactory(object):
    def __init__(self, module, onnx_model_parameters=None, device=None):
        self._training_manager = TrainingManager(module, onnx_model_parameters, device)
        self._inference_manager = InferenceManager(module, device)

    def __call__(self, is_training):
        if is_training:
            return self._training_manager
        else:
            return self._inference_manager
