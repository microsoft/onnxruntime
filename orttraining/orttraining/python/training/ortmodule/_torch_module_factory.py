# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# _torch_module_factory.py

from . import ONNX_OPSET_VERSION
from ._torch_module_ort import TorchModuleORT
from .debug_options import DebugOptions
from ._fallback import _FallbackManager


class TorchModuleFactory:
    def __call__(self, module, debug_options: DebugOptions, fallback_manager: _FallbackManager,
                 opset_version=ONNX_OPSET_VERSION):
        """Creates a TorchModule instance based on the input module."""

        return TorchModuleORT(module, debug_options, fallback_manager, opset_version=opset_version)
