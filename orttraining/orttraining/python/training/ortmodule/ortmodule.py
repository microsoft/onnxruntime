# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

from . import _io
from ._graph_execution_manager_factory import GraphExecutionManagerFactory
from ._utils import _PytorchModuleMetadata

from onnxruntime.training import register_custom_ops_pytorch_exporter

import functools
import torch
from typing import Iterator, Optional, Tuple, TypeVar, Set, Callable

# Needed to override PyTorch methods
T = TypeVar('T', bound='Module')

class ORTModule(torch.nn.Module):
    """Extends user's :class:`torch.nn.Module` model to leverage ONNX Runtime super fast training engine.

    ORTModule specializes the user's :class:`torch.nn.Module` model, providing :meth:`~torch.nn.Module.forward`,
    :meth:`~torch.nn.Module.backward` along with all others :class:`torch.nn.Module`'s APIs.
    """

    def __init__(self, module):
        assert isinstance(
            module, torch.nn.Module), "'module' must be a torch.nn.Module"

        # Create forward dynamically, so each ORTModule instance will have its own copy.
        # This is needed to be able to copy the forward signatures from the original PyTorch models
        # and possibly have different signatures for different instances.
        def _forward(self, *inputs, **kwargs):
            '''Forward pass starts here and continues at `_ORTModuleFunction.forward`

            ONNX model is exported the first time this method is executed.
            Next, we build a full training graph with module_gradient_graph_builder.
            Finally, we instantiate the ONNX Runtime InferenceSession.
            '''

            return self._execution_manager(self._is_training()).forward(*inputs, **kwargs)

        # Bind the forward method.
        self.forward = _forward.__get__(self)
        # Copy the forward signature from the PyTorch module.
        functools.update_wrapper(
            self.forward.__func__, module.forward.__func__)

        super(ORTModule, self).__init__()

        # Support contrib OPs
        register_custom_ops_pytorch_exporter.register_custom_op(is_ortmodule=True)

        # User module is wrapped to use its initializers and save computed gradients
        # along with the module that flattens both input and output of the user module
        # inside _PytorchModuleMetadata
        self._module_metadata = _PytorchModuleMetadata(module, _io._FlattenedModule(module))

        self._execution_manager = GraphExecutionManagerFactory(self._module_metadata.flattened_module)

    # IMPORTANT: DO NOT add code here
    # This declaration is for automatic document generation purposes only
    # The actual forward implementation is bound during ORTModule initialization
    def forward(self, *inputs, **kwargs):
        '''Dummy documentation for forward method'''
        ...

    def _apply(self, fn):
        """Override original method to delegate execution to the flattened PyTorch user module"""

        # Delegation must happen to _flattened_module since methods depend on
        # _apply to recursively apply the internal setting changes
        self._module_metadata.flattened_module._apply(fn)
        return self

    def apply(self: T, fn: Callable[['Module'], None]) -> T:
        """Override original method to delegate execution to the flattened PyTorch user module"""

        # Delegation must happen to _flattened_module since methods depend on
        # apply to recursively apply the internal setting changes
        self._module_metadata.flattened_module.apply(fn)
        return self

    def _is_training(self):
        return self.training and torch.is_grad_enabled()

    def train(self: T, mode: bool = True) -> T:
        """Override original method to delegate execution to the flattened PyTorch user module"""

        # Since _modules is empty, the task needs to be delegated to _module.flattened_module.train
        # which will recursively update the original_module
        self.training = mode
        self._module_metadata.flattened_module.train(mode)
        return self

    def state_dict(self, destination=None, prefix='', keep_vars=False):
        """Override original method to delegate execution to the original PyTorch user module"""

        # Override the state_dict() method so that the state dict key names
        # do not contain the flattened_module._original_module prefix
        return self._module_metadata.original_module.state_dict(
            destination=destination, prefix=prefix, keep_vars=keep_vars)

    def load_state_dict(self, state_dict: 'OrderedDict[str, Tensor]',
                        strict: bool = True):
        """Override original method to delegate execution to the original PyTorch user module"""

        # Override the load_state_dict() method so that the loaded state dict
        # key names does not need to contain the _module.flattened_module._original_module prefix
        return self._module_metadata.original_module.load_state_dict(
            state_dict, strict=strict)

    def register_buffer(self, name: str, tensor: Optional[torch.Tensor], persistent: bool = True) -> None:
        """Override original method to delegate execution to the original PyTorch user module"""
        self._module_metadata.original_module.register_buffer(name, tensor, persistent=persistent)

    def register_parameter(self, name: str, param: Optional[torch.nn.Parameter]) -> None:
        """Override original method to delegate execution to the original PyTorch user module"""
        self._module_metadata.original_module.register_parameter(name, param)

    def get_parameter(self, target: str) -> torch.nn.Parameter:
        """Override original method to delegate execution to the original PyTorch user module"""
        return self._module_metadata.original_module.get_parameter(target)

    def get_buffer(self, target: str) -> torch.Tensor:
        """Override original method to delegate execution to the original PyTorch user module"""
        return self._module_metadata.original_module.get_buffer(target)

    def parameters(self, recurse: bool = True) -> Iterator[torch.nn.Parameter]:
        """Override original method to delegate execution to the original PyTorch user module"""
        yield from self._module_metadata.original_module.parameters(recurse=recurse)

    def named_parameters(self, prefix: str = '', recurse: bool = True) -> Iterator[Tuple[str, torch.nn.Parameter]]:
        """Override original method to delegate execution to the original PyTorch user module"""
        yield from self._module_metadata.original_module.named_parameters(prefix=prefix, recurse=recurse)

    def buffers(self, recurse: bool = True) -> Iterator[torch.Tensor]:
        """Override original method to delegate execution to the original PyTorch user module"""
        yield from self._module_metadata.original_module.buffers(recurse=recurse)

    def named_buffers(self, prefix: str = '', recurse: bool = True) -> Iterator[Tuple[str, torch.Tensor]]:
        """Override original method to delegate execution to the original PyTorch user module"""
        yield from self._module_metadata.original_module.named_buffers(prefix=prefix, recurse=recurse)

    def _replicate_for_data_parallel(self):
        """Raises a NotImplementedError exception since ORTModule is not compatible with torch.nn.DataParallel

        torch.nn.DataParallel requires the model to be replicated across multiple devices, and
        in this process, ORTModule tries to export the model to onnx on multiple devices with the same
        sample input. Because of this multiple device export with the same sample input, torch throws an
        exception that reads: "RuntimeError: Input, output and indices must be on the current device"
        which can be vague to the user since they might not be aware of what happens behind the scene.

        We therefore try to preemptively catch use of ORTModule with torch.nn.DataParallel and throw a
        more meaningful exception.

        Users must use torch.nn.parallel.DistributedDataParallel instead of torch.nn.DataParallel
        which does not need model replication and is also recommended by torch to use instead.
        """

        raise NotImplementedError("ORTModule is not compatible with torch.nn.DataParallel. "
                                  "Please use torch.nn.parallel.DistributedDataParallel instead.")

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                                missing_keys, unexpected_keys, error_msgs):
        """Override original method to delegate execution to the original PyTorch user module"""

        # PyTorch load_state_dict implementation does not recursively call load_state_dict on its sub-modules. 
        # Instead, it creates a recursive function and invokes _load_from_state_dict on all child modules.
        # For the scenario where an ORTModule is a sub-module of another module, loading of the state
        # dictionary requires the _load_from_state_dict to be overridden to prevent an error.
        self._module_metadata.original_module._load_from_state_dict(state_dict, prefix, local_metadata, strict,
                                missing_keys, unexpected_keys, error_msgs)

    def named_children(self) -> Iterator[Tuple[str, 'Module']]:
        """Override original method to delegate execution to the original PyTorch user module"""

        yield from self._module_metadata.original_module.named_children()

    def modules(self) -> Iterator['Module']:
        """Override original method to delegate execution to the original PyTorch user module"""

        yield from self._module_metadata.original_module.modules()

    def named_modules(self, memo: Optional[Set['Module']] = None, prefix: str = '', **kwargs):
        """Override original method to delegate execution to the original PyTorch user module"""

        # PyTorch >1.8.1 has an extra arg remove_duplicate that is not present in 1.8.1
        # To support both, use kwargs
        yield from self._module_metadata.original_module.named_modules(memo, prefix, **kwargs)

    def add_module(self, name: str, module: Optional['Module']) -> None:
        """Raises a NotImplementedError exception since ORTModule does not support adding modules to it"""

        raise NotImplementedError("ORTModule does not support adding modules to it.")
