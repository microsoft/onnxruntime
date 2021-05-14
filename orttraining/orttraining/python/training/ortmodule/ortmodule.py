# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

from . import _io
from ._graph_execution_manager_factory import GraphExecutionManagerFactory

from onnxruntime.training import register_custom_ops_pytorch_exporter

import functools
import torch
from typing import Iterator, Optional, Tuple, TypeVar

# Needed to override PyTorch methods
T = TypeVar('T', bound='Module')

class ORTModule(torch.nn.Module):
    """Extends user's :class:`torch.nn.Module` model to leverage ONNX Runtime super fast training engine.

    ORTModule specializes the user's :class:`torch.nn.Module` model, providing :meth:`~torch.nn.Module.forward`,
    :meth:`~torch.nn.Module.backward` along with all others :class:`torch.nn.Module` APIs leveraging ONNX Runtime.

    ORTModule interacts with

    * | :class:`GraphExecutionManagerFactory` which returns a :class:`GraphExecutionManager` based on
      | whether or not the user's torch module is in training mode or eval mode

    * :class:`GraphExecutionManager` builds and execute the forward/backward graphs

        * | :class:`InferenceManager(GraphExecutionManager)` builds, optimizes and execute the inference ONNX graph

        * | :class:`TrainingManager(GraphExecutionManager)` builds, optimizes and executes the
          | training onnx graph. The :class:`GraphExecutionManager` first exports the user model into an ONNX model.
          | Next, :class:`GraphExecutionManager` interacts with :class:`OrtModuleGraphBuilder` to optimize it.
          | Finally, :class:`ExecutionAgent` is instantiated using the optimized ONNX graph to
          | facilitate execution of the forward/backward subgraphs of the ONNX model

    * py:module:`_io` provides utilities to transform the user inputs and outputs for the model

        * | It facilitates in flattening the output from the user's PyTorch model (since exporting
          | of nested structures is not supported at the moment)

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
        self._original_module = module

        # Get the module that flattens both input and output
        self._flattened_module = _io._FlattenedModule(self._original_module)

        self._execution_manager = GraphExecutionManagerFactory(self._flattened_module)

    # IMPORTANT: DO NOT add code here
    # This declaration is for automatic document generation purposes only
    # The actual forward implementation is bound during ORTModule initialization
    def forward(self, *inputs, **kwargs):
        '''Dummy documentation for forward method'''
        ...

    def _is_training(self):
        return self._flattened_module.training and torch.is_grad_enabled()

    def state_dict(self, destination=None, prefix='', keep_vars=False):
        """Override original method to delegate execution to the base module"""

        # Override the state_dict() method so that the state dict key names
        # do not contain the _flattened_module._original_module prefix
        return self._original_module.state_dict(
            destination=destination, prefix=prefix, keep_vars=keep_vars)

    def load_state_dict(self, state_dict: 'OrderedDict[str, Tensor]',
                        strict: bool = True):
        """Override original method to delegate execution to the base module"""

        # Override the load_state_dict() method so that the loaded state dict
        # key names does not need to contain the _flattened_module._original_module prefix
        return self._original_module.load_state_dict(
            state_dict, strict=strict)

    def register_buffer(self, name: str, tensor: Optional[torch.Tensor], persistent: bool = True) -> None:
        """Override original method to delegate execution to the base module"""
        self._original_module.register_buffer(name, tensor, persistent=persistent)

    def register_parameter(self, name: str, param: Optional[torch.nn.Parameter]) -> None:
        """Override original method to delegate execution to the base module"""
        self._original_module.register_parameter(name, param)

    def get_parameter(self, target: str) -> torch.nn.Parameter:
        """Override original method to delegate execution to the base module"""
        return self._original_module.get_parameter(target)

    def get_buffer(self, target: str) -> torch.Tensor:
        """Override original method to delegate execution to the base module"""
        return self._original_module.get_buffer(target)

    def parameters(self, recurse: bool = True) -> Iterator[torch.nn.Parameter]:
        """Override original method to delegate execution to the base module"""
        yield from self._original_module.parameters(recurse=recurse)

    def named_parameters(self, prefix: str = '', recurse: bool = True) -> Iterator[Tuple[str, torch.nn.Parameter]]:
        """Override original method to delegate execution to the base module"""
        yield from self._original_module.named_parameters(prefix=prefix, recurse=recurse)

    def buffers(self, recurse: bool = True) -> Iterator[torch.Tensor]:
        """Override original method to delegate execution to the base module"""
        yield from self._original_module.buffers(recurse=recurse)

    def named_buffers(self, prefix: str = '', recurse: bool = True) -> Iterator[Tuple[str, torch.Tensor]]:
        """Override original method to delegate execution to the base module"""
        yield from self._original_module.named_buffers(prefix=prefix, recurse=recurse)

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
