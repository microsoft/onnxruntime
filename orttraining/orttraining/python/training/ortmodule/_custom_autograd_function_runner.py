# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

import sys

import torch
from torch.utils.dlpack import from_dlpack, to_dlpack

from onnxruntime.training.ortmodule.torch_cpp_extensions import torch_interop_utils

from ._fallback import ORTModuleFallbackException, ORTModuleIOError, _FallbackManager, wrap_exception  # noqa: F401


def wrap_as_dlpack_or_not(grad_flag, tensor_flag, inplace_flag, training_mode_flag, arg):
    """
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
    """
    if tensor_flag:
        # Got a tensor. Assume it's a DLPack tensor
        # and convert it to Pytorch tensor.
        if training_mode_flag:
            wrapped_arg = from_dlpack(arg).detach().clone()
            # TODO: This clone is just a workround to fix the bug that
            # input saved for backward may be "released" by ORT.
            # we need a follow up fix to avoid the copy overhead.
        else:
            wrapped_arg = from_dlpack(arg)

        # Only requires gradient when running under training mode
        # and the associated tensor has grad_flag=True (i.e.,
        # "requires_grad=True" in the original Pytorch script).
        wrapped_arg.requires_grad = training_mode_flag and grad_flag

        return wrapped_arg

    # Use non-tensor as is. It's a PyObject*.
    return arg


def call_python_forward_function(
    forward_function, requires_grad_flags, tensor_type_flags, is_training_mode, inplace, *args
):
    """
    This function bridges the gap between ORT variables and autograd.Function.apply.
    It conducts basic casting from ORT to Pytorch (before calling "forward_function") and from Pytorch to ORT
    (after calling "forward_function"). It also enable autograd in Pytorch. It formats returned outputs,
    for example, dropping None's from forward_function's output list.

    The major difference between call_python_forward_function and call_python_backward_function is that
    in the forward one, we have extra code to process autograd context from Pytorch.

    Args:
        forward_function: pointer to autograd.Function.apply (e.g., MyReLU.apply).
        requires_grad_flags: requires_grad_flags[i] indicates if the i-th arg needs gradient.
        tensor_type_flags: tensor_type_flagsi] indicates the type of the i-th arg.
        is_training_mode: indicates if this model is running under training mode.
        inplace: indicates if args can be modified inside the custom function.
        args: inputs to "backward_function".
    """

    def generate_non_leaf_or_not(grad_flag, tensor_flag, arg, is_training_mode, is_inplace):
        if is_training_mode and tensor_flag and grad_flag and is_inplace:
            # "multiply one" helps change the torch tensor's is_leaf to be False.
            # This is required when the torch tensor is updated in-place during forward pass.
            # We cannot use view here, because PyTorch handles grad_fn for view differently.
            non_leaf_arg = arg * 1
            return non_leaf_arg
        else:
            return arg

    def wrap_all_outputs(result, training_mode_flag):
        # This is mainly to hold grad_fn references by registering it into our PyNodeSharedPointerPool.
        def register_context(result):
            # Search for context among all outputs.
            ctx = None
            # All forward outputs of torch.autograd.Function shared a same gradient function pointer,
            # so here we just get the first tensor having grad_fn attribute.
            # (https://github.com/pytorch/pytorch/blob/15532595209d2daf34d35e10f8d3d3b64966aea2/torch/csrc/autograd/custom_function.cpp#L267)
            first_tensor_output = None
            for arg in result:
                if not isinstance(arg, torch.Tensor) or not hasattr(arg, "grad_fn"):
                    continue
                # Use the first context we see because all of arg's
                # share the same one.
                ctx = arg.grad_fn
                first_tensor_output = arg
                break

            # Context can be None because not all autograd.Function's are differentiable. The function
            # https://github.com/pytorch/pytorch/blob/d701357d921ef167d42c125e65b6f7da6be3ad0f/torch/csrc/autograd/custom_function.cpp#L209?
            # means if all output of forward function are not differentiable, then grad_fn will be None (not be set).
            # For example,
            #  class Bar(torch.autograd.Function):
            #      # A non-differentiable autograd Function whose forard output
            #      # doesn't have grad_fn attribute.
            #      @staticmethod
            #      def forward(ctx, x):
            #          y = torch.ones_like(x)
            #          return y

            #      @staticmethod
            #      def backward(ctx, dy):
            #          dx = torch.zeros_like(dy)
            #          return dx

            if training_mode_flag and ctx:
                #         FORWARD                                                    BACKWARD FUNCTION CONNECTIONS
                # input_1 (leaf, constructed by from_dlpack)   <----reference----  AccumulateGrad gradient function
                #             ↓                                                                 ↑
                # autograd.Function apply()                        ------------>    autograd.Function backward()
                #             ↓                                    |                            ↑
                #    output_1, output_2   --- shared_ptr<PyNode> ---                            ↑
                #             ↓                                                       previous gradient function

                # We remove the edges starting between current autograd.Function's gradient function and
                # it's input's gradient function (e.g. AccumulateGrad gradient function), then
                # AccumulateGrad gradient function will be destroyed, releasing the reference to input_1
                # (https://github.com/pytorch/pytorch/blob/15532595209d2daf34d35e10f8d3d3b64966aea2/torch/csrc/autograd/functions/accumulate_grad.cpp#L21).
                # The next edges are stored in Node, with which we can get next gradient function.
                # https://github.com/pytorch/pytorch/blob/15532595209d2daf34d35e10f8d3d3b64966aea2/torch/csrc/autograd/function.h#L527
                # filter out the None in the saved_tensors.
                saved_tensors = [t for t in ctx.saved_tensors if t is not None]
                torch_interop_utils.clear_grad_fns_for_next_edges(first_tensor_output, saved_tensors)
                torch_interop_utils.register_grad_fn(id(ctx), first_tensor_output)
            return ctx

        if isinstance(result, torch.Tensor):
            ctx = register_context([result])
            return [ctx, to_dlpack(result)]
        elif isinstance(result, (tuple, list)):
            ctx = register_context(result)
            wrapped = [ctx]
            wrapped.extend(list(to_dlpack(value) if value is not None else None for value in result))
            # Inside the returned list, first element is context and the rest
            # are DLPack tensors.
            return wrapped
        else:
            raise wrap_exception(
                ORTModuleIOError,
                TypeError(f"ORTModule does not support the following model output type {type(result)}."),
            )

    try:
        wrapped_args = list(
            wrap_as_dlpack_or_not(grad_flag, tensor_flag, inplace, is_training_mode, arg)
            for grad_flag, tensor_flag, arg in zip(requires_grad_flags, tensor_type_flags, args)
        )

        with torch.set_grad_enabled(is_training_mode):
            # Another level of wrap to avoid requires_grad=True for leaf variables.
            new_wrapped_args = list(
                generate_non_leaf_or_not(grad_flag, tensor_flag, arg, is_training_mode, inplace)
                for grad_flag, tensor_flag, arg in zip(requires_grad_flags, tensor_type_flags, wrapped_args)
            )

            # Run autograd.Function.apply(...).
            result = forward_function(*new_wrapped_args)

            # Extract results as DLPack tensors plus autograd context. Also skips all None values.
            unwrapped_values = wrap_all_outputs(result, is_training_mode)

        return tuple(unwrapped_values)
    except Exception as e:
        # Flush buffers. Otherwise, calling this from C++ may lose them.
        print("Exception happens when running ", forward_function)
        sys.stdout.flush()
        sys.stderr.flush()
        raise wrap_exception(ORTModuleFallbackException, e)  # noqa: B904


def call_python_backward_function(
    backward_function, requires_grad_flags, tensor_type_flags, is_training_mode, inplace, *args
):
    """
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
    """
    with torch.no_grad():

        def wrap_all_outputs(result):
            if isinstance(result, torch.Tensor):
                return [to_dlpack(result)]
            elif isinstance(result, (tuple, list)):
                return [to_dlpack(value) if value is not None else None for value in result]
            else:
                raise wrap_exception(
                    ORTModuleIOError,
                    TypeError(f"ORTModule does not support the following model output type {type(result)}."),
                )

        try:
            # Backward inputs should not require gradients.
            assert all(grad_flag == 0 for grad_flag in requires_grad_flags)

            # Prepare inputs for calling Python function.
            wrapped_args = list(
                wrap_as_dlpack_or_not(grad_flag, tensor_flag, inplace, is_training_mode, arg)
                for grad_flag, tensor_flag, arg in zip(requires_grad_flags, tensor_type_flags, args)
            )

            # Call Python function.
            result = backward_function(*wrapped_args)

            # Extract results as DLPack tensor list.
            wrapped_returned_args = wrap_all_outputs(result)

            ctx = wrapped_args[0]
            torch_interop_utils.unregister_grad_fn(id(ctx))

            return tuple(wrapped_returned_args)
        except Exception as e:
            # Flush buffers. Otherwise, calling this from C++ may lose them.
            print("Exception happens when running ", backward_function)
            sys.stdout.flush()
            sys.stderr.flush()
            raise wrap_exception(ORTModuleFallbackException, e)  # noqa: B904
