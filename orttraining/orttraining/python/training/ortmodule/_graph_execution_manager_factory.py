# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

from ._training_manager import TrainingManager
from ._inference_manager import InferenceManager


class GraphExecutionManagerFactory(object):
    def __init__(self, module, **kwargs):
        self._training_manager = TrainingManager(module, **kwargs)
        self._inference_manager = InferenceManager(module, **kwargs)

    def __call__(self, is_training):
        if is_training:
            return self._training_manager
        else:
            return self._inference_manager
