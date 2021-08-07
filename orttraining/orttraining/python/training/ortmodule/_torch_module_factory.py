# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# _torch_module_factory.py

from ._torch_module import TorchModule


class TorchModuleFactory:
    def __call__(self, module, debug_options):
        """Creates a TorchModule instance based on the input module."""

        return TorchModule(module, debug_options)
