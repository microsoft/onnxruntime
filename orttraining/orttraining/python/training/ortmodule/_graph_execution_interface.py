# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# _graph_execution_interface.py

from abc import ABC


class GraphExecutionInterface(ABC):
    def __init__(self, module):
        self._original_module = module
        self._validate_module_type(module)

    def forward(self):
        """Executes the forward method for ORTModule

        This is an abstract method and must be overridden by a concrete implementation.
        This is the only method that the user should call on a concrete instance of the GraphExecutionInterface
        """

        raise NotImplementedError(f"forward is not implemented for {type(self)}")

    def _validate_module_type(self, module):
        """Validates the type of the input module

        This is an abstract method and must be overridden by a concrete implementation.
        This is the only method that the user should call on a concrete instance of the GraphExecutionInterface
        """

        raise NotImplementedError(f"_validate_module_type is not implemented for {type(self)}")
