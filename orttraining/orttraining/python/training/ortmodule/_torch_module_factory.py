# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# _torch_module_factory.py

from logging import Logger

from ._fallback import _FallbackManager
from ._torch_module_ort import TorchModuleORT
from .options import DebugOptions


class TorchModuleFactory:
    def __call__(self, module, debug_options: DebugOptions, fallback_manager: _FallbackManager, logger: Logger):
        """Creates a TorchModule instance based on the input module."""

        return TorchModuleORT(module, debug_options, fallback_manager, logger)
