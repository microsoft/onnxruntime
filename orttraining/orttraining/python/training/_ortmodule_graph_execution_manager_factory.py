# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

from ._ortmodule_training_manager import TrainingManager
from ._ortmodule_inference_manager import InferenceManager


class GraphExecutionManagerFactory(object):
    def __init__(self, module):
        self._training_manager = TrainingManager(module)
        self._inference_manager = InferenceManager(module)

    def __call__(self, is_training):
        if is_training:
            return self._training_manager
        else:
            return self._inference_manager
