# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

from onnxruntime.capi.onnxruntime_inference_collection import OrtValue
from onnxruntime.capi import _pybind_state as C

import torch
from torch.utils.dlpack import from_dlpack, to_dlpack
from torch.utils.cpp_extension import load_inline


def _ortvalue_to_torch_tensor(ortvalue):
    # PyTorch's to_dlpack() uses same config for both torch.bool and torch.uint8,
    # and convert the config to torch.uint8 tensor duing from_dlpack().
    # So we need to convert the torch tensor to torch.bool type if OrtValue is bool tensor.
    torch_tensor = from_dlpack(ortvalue.to_dlpack())
    return torch_tensor.to(torch.bool) if ortvalue.data_type() == 'tensor(bool)' else torch_tensor

def _ortvalue_from_torch_tensor(torch_tensor):
    return C.OrtValue.from_dlpack(to_dlpack(torch_tensor), torch_tensor.dtype == torch.bool)

def _load_torch_gpu_allocator_cpp_extension(verbosity, is_rocm_pytorch):
    gpu_identifier = "hip" if is_rocm_pytorch else "cuda"
    gpu_allocator_header = "HIPCachingAllocator" if is_rocm_pytorch else "CUDACachingAllocator"
    torch_gpu_allocator_addresses_cpp_source = f'''
        #include <torch/extension.h>
        #include <c10/{gpu_identifier}/{gpu_allocator_header}.h>

        size_t gpu_caching_allocator_raw_alloc_address() {{
            return reinterpret_cast<size_t>(&c10::{gpu_identifier}::{gpu_allocator_header}::raw_alloc);
        }}

        size_t gpu_caching_allocator_raw_delete_address() {{
            return reinterpret_cast<size_t>(&c10::{gpu_identifier}::{gpu_allocator_header}::raw_delete);
        }}
    '''

    return load_inline(name='inline_extension',
                       cpp_sources=[torch_gpu_allocator_addresses_cpp_source],
                       extra_cflags=['-D__HIP_PLATFORM_HCC__=1' if is_rocm_pytorch else ''],
                       functions=['gpu_caching_allocator_raw_alloc_address',
                                  'gpu_caching_allocator_raw_delete_address'],
                       verbose=verbosity,
                       with_cuda=True)


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

def _create_iobinding(io_binding, inputs, model, device):
    '''Creates IO binding for a `model` inputs and output'''
    for idx, value_info in enumerate(model.graph.input):
        io_binding.bind_ortvalue_input(value_info.name, OrtValue(_ortvalue_from_torch_tensor(inputs[idx])))

    for value_info in model.graph.output:
        io_binding.bind_output(value_info.name, device.type, device_id=get_device_index(device))
