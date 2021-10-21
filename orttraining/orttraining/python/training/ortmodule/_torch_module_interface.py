# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# _torch_module_interface.py

from collections import OrderedDict
import torch
from typing import Iterator, Optional, Tuple, TypeVar, Callable


T = TypeVar('T', bound='torch.nn.Module')


class TorchModuleInterface:
    """Abstract class that provides the function signatures for the torch.nn.Module

    Concrete implementations should inherit from this class and provide necessary executions.
    """

    def __init__(self, module):
        self._original_module = module

    @property
    def module(self):
        """The original user provided module that this class manages.

        This property provides access to methods and properties on the original module.
        """

        raise NotImplementedError(f"module is not implemented for {type(self)}.")

    ###################################################
    # The methods below are part of torch.nn.Module API
    ###################################################

    def forward(self):
        """Executes the forward method for ORTModule

        This is an abstract method and must be overridden by a concrete implementation.
        """

        raise NotImplementedError(f"forward is not implemented for {type(self)}.")

    def _apply(self, fn):
        raise NotImplementedError(f"_apply is not implemented for {type(self)}.")

    def apply(self: T, fn: Callable[[T], None]) -> T:
        raise NotImplementedError(f"apply is not implemented for {type(self)}.")

    def is_training(self):
        raise NotImplementedError(f"is_training is not implemented for {type(self)}.")

    def train(self: T, mode: bool = True) -> T:
        raise NotImplementedError(f"train is not implemented for {type(self)}.")

    def state_dict(self, destination=None, prefix='', keep_vars=False):
        raise NotImplementedError(f"state_dict is not implemented for {type(self)}.")

    def load_state_dict(self, state_dict: 'OrderedDict[str, torch.Tensor]',
                        strict: bool = True):
        raise NotImplementedError(f"load_state_dict is not implemented for {type(self)}.")

    def register_buffer(self, name: str, tensor: Optional[torch.Tensor], persistent: bool = True) -> None:
        raise NotImplementedError(f"register_buffer is not implemented for {type(self)}.")

    def register_parameter(self, name: str, param: Optional[torch.nn.Parameter]) -> None:
        raise NotImplementedError(f"register_parameter is not implemented for {type(self)}.")

    def get_parameter(self, target: str) -> torch.nn.Parameter:
        raise NotImplementedError(f"get_parameter is not implemented for {type(self)}.")

    def get_buffer(self, target: str) -> torch.Tensor:
        raise NotImplementedError(f"get_buffer is not implemented for {type(self)}.")

    def parameters(self, recurse: bool = True) -> Iterator[torch.nn.Parameter]:
        raise NotImplementedError(f"parameters is not implemented for {type(self)}.")

    def named_parameters(self, prefix: str = '', recurse: bool = True) -> Iterator[Tuple[str, torch.nn.Parameter]]:
        raise NotImplementedError(f"named_parameters is not implemented for {type(self)}.")

    def buffers(self, recurse: bool = True) -> Iterator[torch.Tensor]:
        raise NotImplementedError(f"buffers is not implemented for {type(self)}.")

    def named_buffers(self, prefix: str = '', recurse: bool = True) -> Iterator[Tuple[str, torch.Tensor]]:
        raise NotImplementedError(f"named_buffers is not implemented for {type(self)}.")

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        raise NotImplementedError(f"_load_from_state_dict is not implemented for {type(self)}.")

    def named_children(self) -> Iterator[Tuple[str, T]]:
        raise NotImplementedError(f"named_children is not implemented for {type(self)}.")

    def modules(self) -> Iterator[T]:
        raise NotImplementedError(f"modules is not implemented for {type(self)}.")

    def named_modules(self, *args, **kwargs):
        raise NotImplementedError(f"named_modules is not implemented for {type(self)}.")

    def _replicate_for_data_parallel(self):
        raise NotImplementedError(f"_replicate_for_data_parallel is not implemented for {type(self)}.")

    def add_module(self, name: str, module: Optional['Module']) -> None:
        raise NotImplementedError(f"add_module is not implemented for {type(self)}.")
