# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# _torch_module_pytorch.py

from . import _io
from .debug_options import DebugOptions
from ._graph_execution_manager_factory import GraphExecutionManagerFactory
from ._torch_module_interface import TorchModuleInterface
from ._fallback import _FallbackManager

from collections import OrderedDict
import functools
import torch
from typing import Iterator, Optional, Tuple, TypeVar, Callable


T = TypeVar('T', bound='torch.nn.Module')


class TorchModulePytorch(TorchModuleInterface):

    def __init__(self, module: torch.nn.Module):
        super().__init__(module)
        self._original_module = module

    def _apply(self, fn):
        self._original_module._apply(fn)
        return self

    def apply(self: T, fn: Callable[[T], None]) -> T:
        self._original_module.apply(fn)
        return self

    def is_training(self):
        return self._original_module.training and torch.is_grad_enabled()

    def train(self: T, mode: bool = True) -> T:
        self._original_module.train(mode)
        return self

    def state_dict(self, destination=None, prefix='', keep_vars=False):
        return self._original_module.state_dict(destination=destination, prefix=prefix, keep_vars=keep_vars)

    def load_state_dict(self, state_dict: 'OrderedDict[str, torch.Tensor]', strict: bool = True):
        return self._original_module.load_state_dict(state_dict, strict=strict)

    def register_buffer(self, name: str, tensor: Optional[torch.Tensor], persistent: bool = True) -> None:
        self._original_module.register_buffer(name, tensor, persistent=persistent)

    def register_parameter(self, name: str, param: Optional[torch.nn.Parameter]) -> None:
        self._original_module.register_parameter(name, param)

    def get_parameter(self, target: str) -> torch.nn.Parameter:
        return self._original_module.get_parameter(target)

    def get_buffer(self, target: str) -> torch.Tensor:
        return self._original_module.get_buffer(target)

    def parameters(self, recurse: bool = True) -> Iterator[torch.nn.Parameter]:
        yield from self._original_module.parameters(recurse=recurse)

    def named_parameters(self, prefix: str = '', recurse: bool = True) -> Iterator[Tuple[str, torch.nn.Parameter]]:
        yield from self._original_module.named_parameters(prefix=prefix, recurse=recurse)

    def buffers(self, recurse: bool = True) -> Iterator[torch.Tensor]:
        yield from self._original_module.buffers(recurse=recurse)

    def named_buffers(self, prefix: str = '', recurse: bool = True) -> Iterator[Tuple[str, torch.Tensor]]:
        yield from self._original_module.named_buffers(prefix=prefix, recurse=recurse)

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        self._original_module._load_from_state_dict(state_dict, prefix, local_metadata, strict,
                                                    missing_keys, unexpected_keys, error_msgs)

    def named_children(self) -> Iterator[Tuple[str, T]]:
        yield from self._original_module.named_children()

    def modules(self) -> Iterator[T]:
        yield from self._original_module.modules()

    def named_modules(self, *args, **kwargs):
        # PyTorch >1.8.1 has an extra arg remove_duplicate that is not present in 1.8.1
        # To support both, use args and kwargs (since user can call the method with only positional args or kwargs)
        yield from self._original_module.named_modules(*args, **kwargs)
