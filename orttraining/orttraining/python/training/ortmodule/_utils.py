# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
from dataclasses import dataclass
from typing import Dict, Sequence, Tuple, Union

import onnxruntime
from onnxruntime.capi.onnxruntime_inference_collection import OrtValue
from onnxruntime.capi import _pybind_state as C

import torch
from torch.utils.dlpack import from_dlpack, to_dlpack

from torch.utils.cpp_extension import ROCM_HOME

def _ortvalue_to_torch_tensor(ortvalue):
    # PyTorch's to_dlpack() uses same config for both torch.bool and torch.uint8,
    # and convert the config to torch.uint8 tensor duing from_dlpack().
    # So we need to convert the torch tensor to torch.bool type if OrtValue is bool tensor.
    torch_tensor = from_dlpack(ortvalue.to_dlpack())
    return torch_tensor.to(torch.bool) if ortvalue.data_type() == 'tensor(bool)' else torch_tensor


def _ortvalue_from_torch_tensor(torch_tensor):
    return C.OrtValue.from_dlpack(to_dlpack(torch_tensor), torch_tensor.dtype == torch.bool)


def _torch_tensor_from_dl_pack(dlpack, ortvalue):
    torch_tensor = from_dlpack(dlpack)
    return torch_tensor.to(torch.bool) if ortvalue.data_type() == 'tensor(bool)' else torch_tensor


def _check_same_device(device, argument_str, *args):
    '''Check that all tensor arguments in *args reside on the same device as the input device'''

    assert isinstance(device, torch.device), '`device` must be a valid `torch.device` object'
    for arg in args:
        if arg is not None and isinstance(arg, torch.Tensor):
            arg_device = torch.device(arg.device)
            if arg_device != device:
                raise RuntimeError(
                    f"{argument_str} found on device {arg_device}, but expected it to be on module device {device}.")


def get_device_index(device):
    if isinstance(device, str):
        # could be 'cuda:0', 'cuda:1', or 'cpu'. with cpu, set index=0
        device = torch.device(device)
    elif isinstance(device, int):
        return device
    return 0 if device.index is None else device.index


def get_device_str(device):
    if isinstance(device, str):
        # could be 'cuda:0', 'cuda:1', or 'cpu'. with cpu, set index=0
        if device.find(':') == -1:
            device += ':' + str(torch.cuda.current_device())
    elif isinstance(device, int):
        device = 'cuda:' + str(device)
    elif isinstance(device, torch.device):
        if device.index is None:
            device = device.type + ':' + str(torch.cuda.current_device())
        else:
            device = device.type + ':' + str(device.index)
    else:
        raise RuntimeError('Unsupported device type')
    return device


def get_device_from_module(module):
    '''Returns the first device found in the `module`'s parameters or None'''
    device = None
    try:
        device = next(module.parameters()).device
        for param in module.parameters():
            if param.device != device:
                raise RuntimeError('ORTModule supports a single device per model for now')
    except StopIteration:
        # Model doesn't have a device set to any of the model parameters
        pass
    return device


def get_device_from_inputs(args, kwargs):
    '''Returns device from first PyTorch Tensor within args or kwargs'''

    device = None
    if args:
        device = torch.device(args[0].device)
    elif kwargs:
        device = torch.device(next(iter(kwargs.values())).device)
    return device


def _create_iobinding(io_binding, inputs, model, device):
    '''Creates IO binding for a `model` inputs and output'''
    for idx, value_info in enumerate(model.graph.input):
        io_binding.bind_ortvalue_input(value_info.name, OrtValue(_ortvalue_from_torch_tensor(inputs[idx])))

    for value_info in model.graph.output:
        io_binding.bind_output(value_info.name, device.type, device_id=get_device_index(device))


@dataclass(frozen=True)
class SessionConfig:
    """Dataclass containing session_options, providers and provider_options.

    Attributes:
        session_options: session options (log level, optimization options, etc see session_options.h)
        providers: Sequence of providers in order of decreasing precedence. For example
            ['CUDAExecutionProvider', 'CPUExecutionProvider'] means execute a node using
            CUDAExecutionProvider if capable, otherwise execute using CPUExecutionProvider. Values can
            either be provider names or tuples of (provider name, options dict). When any options are
            given in providers, provider_options should not be used.
        provider_options: Sequence of options dicts corresponding to providers in 'providers' attribute.
    """
    session_options: onnxruntime.SessionOptions
    providers: Sequence[Union[str, Tuple[str, Dict]]]
    provider_options: Sequence[Dict]


class SessionConfigFactory:
    def __init__(self, use_external_gpu_allocator: bool = True, loglevel: int = 2):
        """Initializes a SessionConfigFactory

        Args:
            use_external_gpu_allocator: Use GPU memory allocator from torch (requires torch_gpu_allocator c++ extension)
            loglevel: onnxruntime log level (0:Verbose, 1:Info, 2:Warning. 3:Error, 4:Fatal. Default is 2)
        """
        self._use_external_gpu_allocator = use_external_gpu_allocator
        self._is_rocm_pytorch = (torch.version.hip is not None) and (ROCM_HOME is not None)
        self._loglevel = loglevel

    def session_config_from_device(self, device: torch.device) -> SessionConfig:
        """Creates and returns the session configuration to be used for the ExecutionAgent.

        Args:
            device: torch device indicating where the computation should happen.

        onnxruntime inference session needs to be re-initialized when the priority of the
        device to execute the computation changes. This is why this method takes a device.
        """
        providers = None
        provider_options = None
        if device.type == 'cuda':
            # Configure the InferenceSessions to use the specific GPU on which the model is placed.
            providers = (["ROCMExecutionProvider"] if self._is_rocm_pytorch else ["CUDAExecutionProvider"])
            providers.append("CPUExecutionProvider")
            if self._use_external_gpu_allocator and torch.cuda.is_available():
                # CPP extension to get torch GPU allocator's alloc and free function addresses
                from onnxruntime.training.ortmodule.torch_cpp_extensions import torch_gpu_allocator
                torch_alloc = torch_gpu_allocator.gpu_caching_allocator_raw_alloc_address()
                torch_free = torch_gpu_allocator.gpu_caching_allocator_raw_delete_address()
                provider_options = [{"device_id": str(device.index),
                                    "gpu_external_alloc": str(torch_alloc),
                                    "gpu_external_free": str(torch_free)}, {}]
            else:
                provider_options = [{"device_id": str(device.index)}, {}]
        elif device.type == 'cpu':
            providers = ["CPUExecutionProvider"]
            provider_options = [{}]

        session_options = onnxruntime.SessionOptions()
        session_options.enable_mem_pattern = False
        session_options.enable_mem_reuse = False
        session_options.use_deterministic_compute = False
        # default to PRIORITY_BASED execution order
        session_options.execution_order = onnxruntime.ExecutionOrder.PRIORITY_BASED
        session_options.log_severity_level = self._loglevel

        return SessionConfig(session_options, providers, provider_options)
