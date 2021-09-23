# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

from .ortmodule import ORTModule

from onnxruntime_maia import supported_modules

class ORTModuleMAIA(ORTModule):
    """Extends ORTModule with MAIA specific scenario."""

    def __init__(self, module, debug_options=None):
        super().__init__(module, debug_options)

        # Modify GraphExecutionManager internally to support MAIA's custom ops
        for training_mode in [False, True]:
            self._execution_manager(training_mode)._export_extra_kwargs = {supported_modules()}
