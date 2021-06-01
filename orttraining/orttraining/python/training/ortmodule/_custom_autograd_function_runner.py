# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

import sys
import torch

from torch.utils.dlpack import from_dlpack, to_dlpack


def wrap_as_dlpack_or_not(grad_flag, tensor_flag, inplace_flag, training_mode_flag, arg):
    '''
    If the input is a DLPack tensor, we wrap it as a torch.Tensor and
    set up its attributes according to other input flags. Otherwise,
    we return the input as is.

    grad_flag: indicate if "arg" requires gradient. This is only valid if
            "arg" is a DLPack tensor.
    tensor_flag: indicate if "arg" is a DLPack tensor.
    inplace_flag: indicate if "arg" may be modified in custom function. 
    training_mode_flag: indicate if the top-level model is running
                        under training (or inference) mode.
    arg: a DLPack tensor or a normal Python object (e.g, a tuple of ints).
    '''
    if tensor_flag:
        # Got a tensor. Assume it's a DLPack tensor
        # and convert it to Pytorch tensor.
        if not inplace_flag:
            wrapped_arg = from_dlpack(arg)
        else:
            wrapped_arg = from_dlpack(arg).detach().contiguous()

        # Only requires gradient when running under training mode
        # and the associated tensor has grad_flag=True (i.e.,
        # "requires_grad=True" in the original Pytorch script).
        wrapped_arg.requires_grad = training_mode_flag and grad_flag

        return wrapped_arg
    else:
        # Use non-tensor as is. It's a PyObject*.
        return arg

def call_python_forward_function(
        forward_function,
        requires_grad_flags,
        tensor_type_flags,
        is_training_mode,
        inplace,
        *args):
    '''
    This function bridges the gap between ORT variables and autograd.Function.apply.
    It conducts basic casting from ORT to Pytorch (before calling "forward_function") and from Pytorch to ORT
    (after calling "forward_function"). It also enable autograd in Pytorch. It formats returned outputs,
    for example, dropping None's from forward_function's output list.

    Args:
        forward_function: pointer to autograd.Function.apply (e.g., MyReLU.apply).
        requires_grad_flags: requires_grad_flags[i] indicates if the i-th arg needs gradient.
        tensor_type_flags: tensor_type_flagsi] indicates the type of the i-th arg.
        is_training_mode: indicates if this model is running under training mode.
        inplace: indicates if args can be modified inside the custom function.
        args: inputs to "backward_function".
    '''
    def generate_non_leaf_or_not(grad_flag, tensor_flag, arg):
        if tensor_flag and grad_flag:
            # "multiply one" helps change the torch tensor's is_leaf to be False.
            # This is required when the torch tensor is updated in-place during forward pass.
            # We cannot use view here, because PyTorch handels grad_fn for view differently.
            non_leaf_arg = arg * arg.new_ones((1,))
            return non_leaf_arg
        else:
            return arg

    def wrap_all_outputs(result, training_mode_flag):
        def extract_context(result):
            # Search for context among all outputs.
            ctx = None
            for arg in result:
                if not isinstance(arg, torch.Tensor) or not hasattr(arg, 'grad_fn'):
                    continue
                # Use the first context we see because all of arg's
                # share the same one.
                ctx = arg.grad_fn
                break
            if training_mode_flag:
                # Must extract one valid context from result tensors.
                assert ctx is not None
            else:
                # Context must not present under non-training mode.
                assert ctx is None

            return ctx

        if isinstance(result, torch.Tensor):
            ctx = extract_context([result])
            return [ctx, to_dlpack(result)]
        elif isinstance(result, tuple) or isinstance(result, list):
            ctx = extract_context(result)
            wrapped = [ctx]
            wrapped.extend(list(to_dlpack(value) for value in result))
            # Inside the returned list, first element is context and the rest
            # are DLPack tensors.
            return wrapped
        else:
            raise TypeError('Unsupported returned type: ', type(result))

    try:
        wrapped_args = list(wrap_as_dlpack_or_not(grad_flag, tensor_flag, inplace, is_training_mode, arg)
                            for grad_flag, tensor_flag, arg in zip(requires_grad_flags, tensor_type_flags, args))

        with torch.enable_grad():
            # Another level of wrap to avoid requires_grad=True for leaf variables.
            new_wrapped_args = list(generate_non_leaf_or_not(grad_flag, tensor_flag, arg)
                                    for grad_flag, tensor_flag, arg in zip(requires_grad_flags, tensor_type_flags, wrapped_args))

            # Run autograd.Function.apply(...).
            result = forward_function(*new_wrapped_args)

            # Extract results as DLPack tensors plus autograd context. Also skips all None values.
            unwrapped_values = wrap_all_outputs(result, is_training_mode)

        return tuple(unwrapped_values)
    except:
        # Flush buffers. Otherwise, calling this from C++ may lose them.
        print('Exception happens when running ', forward_function)
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
    '''
    This function bridges the gap between ORT variables and autograd.Function.backward.
    It conducts basic casting from ORT to Pytorch (before calling "backward_function")
    and from Pytorch to ORT (after calling "backward_function").  It formats returned
    outputs, example, dropping None's from backward_function's output list.

    Args:
        backward_function: pointer to autograd.Function.backward (e.g., MyReLU.backward).
        requires_grad_flags: requires_grad_flags[i] indicates if the i-th arg needs gradient.
        tensor_type_flags: tensor_type_flagsi] indicates the type of the i-th arg.
        is_training_mode: indicates if this model is running under training mode.
        inplace: indicates if args can be modified inside the custom function.
        args: inputs to "backward_function".
    '''
    def wrap_all_outputs(result):
        if isinstance(result, torch.Tensor):
            return [to_dlpack(result)]
        elif isinstance(result, tuple) or isinstance(result, list):
            return [to_dlpack(value) for value in result if value is not None]
        else:
            raise Exception('Unsupported returned type: ', type(result))

    try:
        # Prepare inputs for calling Python function.
        wrapped_args = list(wrap_as_dlpack_or_not(grad_flag, tensor_flag, inplace, is_training_mode, arg)
                            for grad_flag, tensor_flag, arg in zip(requires_grad_flags, tensor_type_flags, args))

        # Call Python function.
        result = backward_function(*wrapped_args)

        # Extract results as DLPack tensor list.
        wrapped_returned_args = wrap_all_outputs(result)

        return tuple(wrapped_returned_args)
    except:
        # Flush buffers. Otherwise, calling this from C++ may lose them.
        print('Exception happens when running ', backward_function)
        sys.stdout.flush()
        sys.stderr.flush()
        raise
