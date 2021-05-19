# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

import onnxruntime
import sys
import torch

from torch.utils.dlpack import from_dlpack
from . import _utils

def call_python_forward_function(forward_function, requires_grad_flags, tensor_type_flags, is_training_mode, *args):
    try:
        wrapped_args = []
        for grad_flag, tensor_flag, arg in zip(requires_grad_flags, tensor_type_flags, args):
            if tensor_flag:
                # Got a tensor. Assume it's a DLPack tensor
                # and convert it to Pytorch tensor.
                wrapped_arg = from_dlpack(arg).detach().clone().contiguous()
                if is_training_mode and grad_flag:
                    wrapped_arg.requires_grad = True
                else:
                    wrapped_arg.requires_grad = False

                wrapped_args.append(wrapped_arg)
            else:
                # Use non-tensor as is. It's a PyObject*.
                wrapped_args.append(arg)

        unwrapped_values = []
        ctx = None
        with torch.enable_grad():
            new_wrapped_args = []
            for grad_flag, tensor_flag, arg in zip(requires_grad_flags, tensor_type_flags, wrapped_args):
                if tensor_flag and grad_flag:
                    # "multiply one" helps change the torch tensor's is_leaf to be False.
                    # This is required when the torch tensor is updated in-place during forward pass.
                    # we cannot use view here, because PyTorch handels grad_fn for view differently.
                    non_leaf_arg = arg * arg.new_ones((1,))
                    new_wrapped_args.append(non_leaf_arg)
                else:
                    new_wrapped_args.append(arg)
            result = forward_function(*new_wrapped_args)

            if isinstance(result, torch.Tensor):
                ort_value = _utils._ortvalue_from_torch_tensor(result)
                unwrapped_values = [ort_value]
                ctx = result.grad_fn
            elif isinstance(result, tuple) or isinstance(result, list):
                for value in result:
                    unwrapped_value = _utils._ortvalue_from_torch_tensor(value)
                    unwrapped_values.append(unwrapped_value)
                    if ctx is None and value is not None and hasattr(value, 'grad_fn'):
                        ctx = value.grad_fn
            else:
                raise Exception('Unsupported returned type: ', type(
                    result), ' by calling ', forward_function)

        if is_training_mode:
            # Must extract one valid context from result tensors.
            assert ctx is not None

        for value in unwrapped_values:
            # Mitigate the timing between "returning pure address back to C++" and "assign the ortvalue to PythonOP's output".
            # Todo: refine by returning Torch tensor back to C++, then we construct OrtValue from it.
            onnxruntime.register_python_object(value)

        unwrapped_ptrs = [int(id(ctx))]
        for v in unwrapped_values:
            unwrapped_ptrs.append(int(v.ortvalue_ptr()))

        return tuple(unwrapped_ptrs)
    except:
        # Flush buffers. Otherwise, calling this from C++ may lose them.
        sys.stdout.flush()
        sys.stderr.flush()
        raise


def call_python_backward_function(backward_function, requires_grad_flags, tensor_type_flags, is_training_mode, *args):
    try:
        wrapped_args = []
        for requires_grad, tensor_flag, arg in zip(requires_grad_flags, tensor_type_flags, args):
            if tensor_flag:
                # Got a tensor. Assume it's a DLPack tensor
                # and convert it to Pytorch tensor.
                wrapped_arg = from_dlpack(arg).clone().contiguous()
                if requires_grad:
                    wrapped_arg.requires_grad = True
                else:
                    wrapped_arg.requires_grad = False
                wrapped_args.append(wrapped_arg)
            else:
                # Use non-tensor as is. It's a PyObject*.
                wrapped_args.append(arg)

        unwrapped_values = []
        result = backward_function(*wrapped_args)
        if isinstance(result, torch.Tensor):
            # TODO: We need to confirm
            #   1. The ownership of result is transferred to DLPack tensor from Pytorch.
            #   2. The ownership of result is transferred to ORTValue from DLPack.
            # If they are all true, we can remove the object register code below.
            ort_value = _utils._ortvalue_from_torch_tensor(result)
            unwrapped_values = [ort_value]
        elif isinstance(result, tuple) or isinstance(result, list):
            for value in result:
                if value is None:
                    continue
                if not isinstance(value, torch.Tensor):
                    raise Exception('Unsupported returned element type: ', type(
                        value), ' by calling ', backward_function)
                unwrapped_value = _utils._ortvalue_from_torch_tensor(value)
                unwrapped_values.append(unwrapped_value)
        else:
            raise Exception('Unsupported returned type: ', type(
                result), ' by calling ', backward_function)

        for value in unwrapped_values:
            # Mitigate the timing between "returning pure address back to C++" and "assign the ortvalue to PythonOPGrad's output".
            # Todo: refine by returning Torch tensor back to C++, then we construct OrtValue from it.
            onnxruntime.register_python_object(value)

        unwrapped_ptrs = []
        for value in unwrapped_values:
            unwrapped_ptrs.append(int(value.ortvalue_ptr()))

        return tuple(unwrapped_ptrs)
    except:
        # Flush buffers. Otherwise, calling this from C++ may lose them.
        sys.stdout.flush()
        sys.stderr.flush()
        raise
