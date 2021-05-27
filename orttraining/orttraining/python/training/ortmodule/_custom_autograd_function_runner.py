# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

import onnxruntime
import sys
import torch

from torch.utils.dlpack import from_dlpack, to_dlpack
from . import _utils


def call_python_forward_function(
        forward_function,
        requires_grad_flags,
        tensor_type_flags,
        is_training_mode,
        inplace,
        *args):
    try:
        wrapped_args = []
        for grad_flag, tensor_flag, arg in zip(requires_grad_flags, tensor_type_flags, args):
            if tensor_flag:
                # Got a tensor. Assume it's a DLPack tensor
                # and convert it to Pytorch tensor.
                if not inplace:
                    wrapped_arg = from_dlpack(arg)
                else:
                    wrapped_arg = from_dlpack(arg).detach().contiguous()

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
                    # We cannot use view here, because PyTorch handels grad_fn for view differently.
                    non_leaf_arg = arg * arg.new_ones((1,))
                    new_wrapped_args.append(non_leaf_arg)
                else:
                    new_wrapped_args.append(arg)
            result = forward_function(*new_wrapped_args)

            if isinstance(result, torch.Tensor):
                ctx = result.grad_fn
                unwrapped_values = [ctx, to_dlpack(result)]
            elif isinstance(result, tuple) or isinstance(result, list):
                ctx = result[0].grad_fn
                unwrapped_values.append(ctx)
                for value in result:
                    if not isinstance(value, torch.Tensor):
                        raise Exception('Unsupported returned element type: ', type(
                            value), ' by calling ', forward_function)
                    unwrapped_values.append(to_dlpack(value))
            else:
                raise Exception('Unsupported returned type: ', type(
                    result), ' by calling ', forward_function)

        if is_training_mode:
            # Must extract one valid context from result tensors.
            assert ctx is not None
        else:
            assert ctx is None

        return tuple(unwrapped_values)
    except:
        # Flush buffers. Otherwise, calling this from C++ may lose them.
        sys.stdout.flush()
        sys.stderr.flush()
        raise


def call_python_backward_function(
        backward_function,
        requires_grad_flags,
        tensor_type_flags,
        is_training_mode,
        inplace,
        *args):
    try:
        wrapped_args = []
        for grad_flag, tensor_flag, arg in zip(requires_grad_flags, tensor_type_flags, args):
            if tensor_flag:
                # Got a tensor. Assume it's a DLPack tensor
                # and convert it to Pytorch tensor.
                #wrapped_arg = from_dlpack(arg).clone().contiguous()
                if not inplace:
                    wrapped_arg = from_dlpack(arg).contiguous()
                else:
                    wrapped_arg = from_dlpack(arg).detach().contiguous()


                if is_training_mode and grad_flag:
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
            unwrapped_values = [to_dlpack(result)]
        elif isinstance(result, tuple) or isinstance(result, list):
            for value in result:
                if value is None:
                    continue
                if not isinstance(value, torch.Tensor):
                    raise Exception('Unsupported returned element type: ', type(
                        value), ' by calling ', backward_function)
                unwrapped_values.append(to_dlpack(value))
        else:
            raise Exception('Unsupported returned type: ', type(
                result), ' by calling ', backward_function)

        return tuple(unwrapped_values)
    except:
        # Flush buffers. Otherwise, calling this from C++ may lose them.
        sys.stdout.flush()
        sys.stderr.flush()
        raise
