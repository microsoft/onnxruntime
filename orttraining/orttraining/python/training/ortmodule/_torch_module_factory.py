# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# _torch_module_factory.py

from ._io import _FlattenedModule
from ._torch_module_ort import TorchModuleORT
from ._graph_execution_manager import GraphExecutionManager

import torch


class TorchModuleFactory:
    def __call__(self,
                 module: torch.nn.Module,
                 flattened_module: _FlattenedModule,
                 execution_manager: GraphExecutionManager):
        """Creates a TorchModule instance based on the input module."""

        return TorchModuleORT(module, flattened_module, execution_manager)
