# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

from onnxruntime.capi.onnxruntime_inference_collection import OrtValue
from onnxruntime.capi import _pybind_state as C
from ._fallback import _FallbackManager, ORTModuleFallbackException, ORTModuleDeviceException, wrap_exception

import inspect
import torch
from torch.utils.dlpack import from_dlpack, to_dlpack
from typing import List
import warnings


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
                raise wrap_exception(ORTModuleDeviceException,
                                     RuntimeError(
                                         f"{argument_str} found on device {arg_device}, but expected it to be on module device {device}."))


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
        raise wrap_exception(ORTModuleDeviceException, RuntimeError('Unsupported device type'))
    return device


def get_device_from_module(module):
    '''Returns the first device found in the `module`'s parameters or None

    Args:
        module (torch.nn.Module): PyTorch model to extract device from

    Raises:
        ORTModuleFallbackException: When more than one device is found at `module`
    '''
    device = None
    try:
        device = next(module.parameters()).device
        for param in module.parameters():
            if param.device != device:
                raise wrap_exception(ORTModuleDeviceException,
                                     RuntimeError('ORTModule supports a single device per model'))
    except StopIteration:
        # Model doesn't have a device set to any of the model parameters
        pass
    return device


def get_device_from_inputs(args, kwargs):
    '''Returns device from first PyTorch Tensor within args or kwargs

    Args:
        args: List with inputs
        kwargs: Dictionary with inputs
    '''

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

def check_for_name_collisions(ortmodule: torch.nn.Module, user_module: torch.nn.Module,
                              ortmodule_non_method_attribute_names: List[str]):
    """Raises if there are any common attributes between the user's model and ORTModule.

    Args:
        ortmodule: the ORTModule instance
        user_module: the user's torch.nn.Module
        ortmodule_non_method_attribute_names: any non method attribute names that ORTModule sets on itself

    Raises:
        UserWarning: If there are any overlapping methods between the ortmodule and user_module (except forward)
    """

    ortmodule_methods = dict(inspect.getmembers(ortmodule, predicate=inspect.ismethod))
    torch_module_methods = dict(inspect.getmembers(torch.nn.Module, predicate=inspect.isfunction))
    user_module_methods = inspect.getmembers(user_module,
        predicate=lambda x : inspect.ismethod(x) and not x.__name__.startswith('__'))

    # Check if any element of ortmodule_non_method_attribute_names is an attribute of user's model
    for name in ortmodule_non_method_attribute_names:
        if hasattr(user_module, name):
            warnings.warn(f"User Module's attribute name {name} collides with ORTModule's attribute name. "
                    "User Module's attribute may not be returned when trying to retrieve the attribute through ORTModule.")

    # Check if any user defined method collides with ORTModule's method
    for method_name, method in user_module_methods:
        if method_name not in torch_module_methods or method.__func__ != torch_module_methods[method_name]:
            if method_name == 'forward':
                continue

            # This is a user defined/overriden method. Check for collisions.
            if method_name in ortmodule_methods:
                warnings.warn(f"User Module's attribute name {method_name} collides with ORTModule's attribute name. "
                    "User Module's method may not be called upon invocation through ORTModule.")
