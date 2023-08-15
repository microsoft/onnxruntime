# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

import sys
import warnings
from collections import OrderedDict
from typing import Callable, Dict, List, Optional, Tuple, Union

import torch
from torch.utils.dlpack import from_dlpack, to_dlpack

from onnxruntime.training.ortmodule.torch_cpp_extensions import torch_interop_utils

from ._fallback import ORTModuleFallbackException, ORTModuleIOError, _FallbackManager, wrap_exception  # noqa: F401


class CustomFuncOpKernelInfo:
    """Store the kernel specific information retrieved with the first-time run."""

    def __init__(self, kernel_invoke_id: str):
        # kernel_invoke_id is a string contains session thread id, op kernel creation time stamp in ms, a random int,
        # and address of op_kernel pointer. This can guarantee the uniqueness of the key in case of multiple
        # instances of a same named PythonOp/PythonOpGrad in one session, or multiple sessions.
        self.kernel_invoke_id = kernel_invoke_id

        # For the tensors generated from ORT backend, there is special handling here:
        # 1. For the first time run for the kernel (the uniqueness of the kernel is defined by kernel_invoke_id),
        # all such tensors will be cloned in case they are saved in context (but ORT backend is not aware of the
        # reference, may release the content of the tensor before it is needed in backward). Once
        # `autograd.Function.apply` completes, by checking the existence of the tensor in the saved_tensors,
        # `_GlobalOpKernelInfoMap` is updated to save the input indices that are saved in context.
        # 2. For the subsequent runs, if the input index is in `input_indices_to_save_in_ctx`, the tensor
        # will be cloned before fed into `autograd.Function.apply` as input.
        self.input_indices_to_save_in_ctx: List[int] = []

        # To align with PyTorch `ctx.set_materialize_grads(False|True)``
        # materialize_grads_config is a map from output index to (device, dtype, shape) of the output tensor, used
        # for materializing the gradient of the output tensor in backward.
        self.materialize_grads: bool = False
        self.materialize_grads_config: Optional[Dict[int, Tuple[torch.device, torch.dtype, torch.shape]]] = None


# Store the kernel specific information that cannot be retrieved and saved by PyTorch exporter.
# For those infos that can only be retrieved with real run, we try to collect them in the first time run.
# key: kernel_invoke_id, value: CustomFuncOpKernelInfo.
_GlobalOpKernelInfoMap: Dict[str, CustomFuncOpKernelInfo] = {}


def _get_context(forward_tensor_outputs: List[torch.Tensor]) -> Tuple[any, Optional[torch.Tensor]]:
    """Search for context among all outputs.

    Note1: All forward outputs of torch.autograd.Function shared the same gradient function pointer,
        so here we just get the first tensor having grad_fn attribute.
        (https://github.com/PyTorch/PyTorch/blob/15532595209d2daf34d35e10f8d3d3b64966aea2/torch/csrc/autograd/custom_function.cpp#L267)

    Note2: Context can be None because NOT all torch.autograd.Function's are differentiable. The function
        https://github.com/PyTorch/PyTorch/blob/d701357d921ef167d42c125e65b6f7da6be3ad0f/torch/csrc/autograd/custom_function.cpp#L209?
        means if all output of forward function is not differentiable, then grad_fn will be None (not be set).

        For example,
            class Bar(torch.autograd.Function):
                # A non-differentiable autograd Function whose forard output
                # doesn't have grad_fn attribute.
                @staticmethod
                def forward(ctx, x):
                    y = torch.ones_like(x)
                    return y

                @staticmethod
                def backward(ctx, dy):
                    dx = torch.zeros_like(dy)
                    return dx

    Returns:
        ctx: context of the autograd.Function.
        tensor: a tensor that owns the context.

    """
    ctx = None
    first_tensor_output = None
    for arg in forward_tensor_outputs:
        if not isinstance(arg, torch.Tensor) or not hasattr(arg, "grad_fn"):
            continue

        if arg.grad_fn is None:
            # For following case, it is possible grad_fn exist, but its value is None,
            # so we need to continue to search for the first tensor having a non-None grad_fn.
            #
            # >>> w = torch.randn(5, 6)
            # >>> hasattr(w, "grad_fn")
            # True
            # >>> w.grad_fn is None
            # True
            # >>> w, ... = CustomFunc.apply(w) # where CustomFunc forward just return w and other tensors.
            #
            # Then hasattr(w, "grad_fn") is True, but w.grad_fn is None.
            continue
        # Use the first context we see because all of arg's share the same one.
        ctx = arg.grad_fn
        first_tensor_output = arg
        break
    if first_tensor_output is not None:
        assert ctx is not None, "ctx should not be None if first_tensor_output is not None."
    return (ctx, first_tensor_output)


def _finalize_traing_mode_forward(
    kernel_invoke_id: str,
    input_tensors_from_ort: Dict[int, torch.Tensor],
    forward_output_tensors: List[Union[torch.Tensor, None]],
):
    """Complete the epilogue of forward runner for training mode.

    Args:
        kernel_invoke_id: kernel_invoke_id of the PythonOp kernel unique id.
        input_tensors_from_ort: input tensors generated from ORT backend.
        forward_output_tensors: output tensors of the autograd.Function.

    Things to do:
    1. Try to get context from forward output tensors.
    2. Remove the gradient functions between current autograd.Function and its input's gradient function, because
       in ORT we don't depend on PyTorch's autograd engine.
    3. Register the current autograd.Function's gradient function into our PyNodeSharedPointerPool.
    4. Save kernel specific information into _GlobalOpKernelInfoMap in the first-time kernel run.
    """

    ctx, tensor_owning_ctx = _get_context(forward_output_tensors)

    # ctx being None in training mode means the forward function is not differentiable, so backward is not needed.
    if ctx is None:
        return None

    # Filter out the None in the saved_tensors.
    saved_tensors = [t for t in ctx.saved_tensors if t is not None]

    ctx.fw_kernel_invoke_id = kernel_invoke_id

    # If this is the first time run, collect kernel specific information.
    if kernel_invoke_id not in _GlobalOpKernelInfoMap:
        kernel_info = CustomFuncOpKernelInfo(kernel_invoke_id)
        _GlobalOpKernelInfoMap[kernel_invoke_id] = kernel_info
        if len(saved_tensors):
            # Check tensors generated by ORT is in the saved_tensors or not.
            # If yes, save the input index of the tensor in the _GlobalOpKernelInfoMap.
            kernel_info.input_indices_to_save_in_ctx = [
                arg_index
                for arg_index, tensor in input_tensors_from_ort.items()
                if any(tensor is saved_tensor for saved_tensor in saved_tensors)
            ]
            warnings.warn("Add input index to _GlobalOpKernelInfoMap, to avoid extra copy in every iteration.")
        kernel_info.materialize_grads = torch_interop_utils.get_materialize_grads(tensor_owning_ctx)
        kernel_info.materialize_grads_config = OrderedDict()
        if kernel_info.materialize_grads:
            for output_index, tensor in enumerate(forward_output_tensors):
                if isinstance(tensor, torch.Tensor):
                    kernel_info.materialize_grads_config[output_index] = (
                        tensor.device,
                        tensor.dtype,
                        tensor.shape,
                    )

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
    # (https://github.com/PyTorch/PyTorch/blob/15532595209d2daf34d35e10f8d3d3b64966aea2/torch/csrc/autograd/functions/accumulate_grad.cpp#L21).
    # The next edges are stored in Node, with which we can get next gradient function.
    # https://github.com/PyTorch/PyTorch/blob/15532595209d2daf34d35e10f8d3d3b64966aea2/torch/csrc/autograd/function.h#L527
    torch_interop_utils.clear_grad_fns_for_next_edges(tensor_owning_ctx, saved_tensors)

    # This is mainly to hold grad_fn references by registering it into our PyNodeSharedPointerPool.
    torch_interop_utils.register_grad_fn_and_remove_from_autograd(id(ctx), tensor_owning_ctx)

    return ctx


def call_python_forward_function(
    forward_function: Callable,
    requires_grad_flags: List[bool],
    tensor_type_flags: List[int],
    is_training_mode: bool,
    inplace: bool,
    kernel_invoke_id: str,
    *args,
):
    """
    This function bridges the gap between ORT variables and autograd.Function.apply.
    It conducts basic casting from ORT to PyTorch (before calling "forward_function") and from PyTorch to ORT
    (after calling "forward_function"). It also enable autograd in PyTorch. It formats returned outputs,
    for example, dropping None's from forward_function's output list.

    The major difference between call_python_forward_function and call_python_backward_function is that
    in the forward one, we have extra code to process autograd context from PyTorch.

    Args:
        forward_function: pointer to autograd.Function.apply (e.g., MyReLU.apply).
        requires_grad_flags: requires_grad_flags[i] indicates if the i-th arg needs gradient.
        tensor_type_flags: tensor_type_flags[i] indicates the type of the i-th arg, 0 - non-tensor, 1 - tensor.
        is_training_mode: indicates if this model is running under training mode.
        inplace: indicates if args can be modified inside the custom function.
        args: inputs to "backward_function".
    """

    def generate_non_leaf_or_not(grad_flag, tensor_flag, arg, is_training_mode, is_inplace):
        if is_training_mode and tensor_flag and grad_flag and is_inplace:
            # "multiply one" helps change the torch tensor's is_leaf to False.
            # This is required when the torch tensor is updated in-place during forward pass.
            # We cannot use view here, because PyTorch handles grad_fn for view differently.
            non_leaf_arg = arg * 1
            return non_leaf_arg
        else:
            return arg

    try:
        wrapped_args = []
        tensor_input_args_map = OrderedDict()

        # Be noted: in inference mode, we won't insert any information into _GlobalOpKernelInfoMap, because ctx
        # will always be None in the first time run.
        input_indices_to_save_in_ctx = None  # Uninitialized
        if kernel_invoke_id in _GlobalOpKernelInfoMap:
            input_indices_to_save_in_ctx = _GlobalOpKernelInfoMap[kernel_invoke_id].input_indices_to_save_in_ctx

        for arg_index, (grad_flag, tensor_flag, arg) in enumerate(zip(requires_grad_flags, tensor_type_flags, args)):
            if tensor_flag:
                # Assume it's a DLPack tensor# and convert it to PyTorch tensor.
                # Note1:
                #   If it's first-time kernel invocation, input_indices_to_save_in_ctx is None, we do the
                #   copy for all tensor. Otherwise, we only copy the tensors whose indices are in
                #   input_indices_to_save_in_ctx.
                #
                # Note2:
                #   For inference mode, we don't need do the copy because ctx will be None,
                #   so nothing will be saved for ctx.
                if is_training_mode and (
                    input_indices_to_save_in_ctx is None or arg_index in input_indices_to_save_in_ctx
                ):
                    wrapped_arg = from_dlpack(arg).detach().clone()
                else:
                    wrapped_arg = from_dlpack(arg)

                # Only requires gradient when running under training mode
                # and the associated tensor has grad_flag=True (i.e.,
                # "requires_grad=True" in the original PyTorch script).
                wrapped_arg.requires_grad = is_training_mode and grad_flag
                wrapped_args.append(wrapped_arg)
                tensor_input_args_map[arg_index] = wrapped_arg

            else:
                # Use non-tensor as is. It's a PyObject*.
                wrapped_args.append(arg)

        with torch.set_grad_enabled(is_training_mode):
            # Another level of wrap to avoid requires_grad=True for leaf variables.
            new_wrapped_args = list(
                generate_non_leaf_or_not(grad_flag, tensor_flag, arg, is_training_mode, inplace)
                for grad_flag, tensor_flag, arg in zip(requires_grad_flags, tensor_type_flags, wrapped_args)
            )

            # Run autograd.Function.apply(...).
            # TODO(pengwa): looks we are assuming all outputs will be either Tensor or None.
            # We should revisit if it is possible to support other types of output, for example int, or, etc.
            # But that might also requires some work in backend.
            result = forward_function(*new_wrapped_args)

            # Extract results as DLPack tensors plus autograd context. Also skips all None values.
            if isinstance(result, torch.Tensor):
                ctx = None
                if is_training_mode:
                    ctx = _finalize_traing_mode_forward(kernel_invoke_id, tensor_input_args_map, [result])
                unwrapped_values = [ctx, to_dlpack(result)]
            elif isinstance(result, (tuple, list)):
                ctx = None
                if is_training_mode:
                    ctx = _finalize_traing_mode_forward(kernel_invoke_id, tensor_input_args_map, result)
                wrapped = [ctx]
                wrapped.extend(list(to_dlpack(value) if value is not None else None for value in result))
                # Inside the returned list, first element is context and the rest
                # are DLPack tensors.
                unwrapped_values = wrapped
            else:
                raise wrap_exception(
                    ORTModuleIOError,
                    TypeError(f"ORTModule does not support the following model output type {type(result)}."),
                )
        return tuple(unwrapped_values)
    except Exception as e:
        # Flush buffers. Otherwise, calling this from C++ may lose them.
        print("Exception happens when running ", forward_function)
        sys.stdout.flush()
        sys.stderr.flush()
        raise wrap_exception(ORTModuleFallbackException, e)  # noqa: B904


def call_python_backward_function(
    backward_function: Callable,
    requires_grad_flags: List[bool],
    tensor_type_flags: List[int],
    is_training_mode: bool,
    inplace: bool,
    kernel_invoke_id: str,
    *args,
):
    """
    This function bridges the gap between ORT variables and autograd.Function.backward.
    It conducts basic casting from ORT to PyTorch (before calling "backward_function")
    and from PyTorch to ORT (after calling "backward_function").  It formats returned
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
            ctx = args[0]
            fw_kernel_invoke_id = ctx.fw_kernel_invoke_id
            wrapped_args = []
            for grad_input_index, (grad_flag, tensor_flag, arg) in enumerate(
                zip(requires_grad_flags, tensor_type_flags, args)
            ):
                # If an input is a tensor, it is possible we get a None also when it is optional as grad input.
                if tensor_flag:
                    if arg is None:
                        if _GlobalOpKernelInfoMap[fw_kernel_invoke_id].materialize_grads:
                            config = _GlobalOpKernelInfoMap[fw_kernel_invoke_id].materialize_grads_config
                            # ignore the first input, which is the ctx.
                            device, dtype, shape = config[grad_input_index - 1]
                            wrapped_arg = torch.zeros(shape, device=device, dtype=dtype)
                        else:
                            wrapped_arg = arg
                    else:
                        # Assume it's a DLPack tensor# and convert it to PyTorch tensor.
                        wrapped_arg = from_dlpack(arg)

                    if wrapped_arg is not None:
                        # Only requires gradient when running under training mode
                        # and the associated tensor has grad_flag=True (i.e.,
                        # "requires_grad=True" in the original PyTorch script).
                        wrapped_arg.requires_grad = is_training_mode and grad_flag
                    wrapped_args.append(wrapped_arg)
                else:
                    # Use non-tensor as is. It's a PyObject*.
                    wrapped_args.append(arg)

            # Call Python function.
            result = backward_function(*wrapped_args)

            # Extract results as DLPack tensor list.
            wrapped_returned_args = wrap_all_outputs(result)

            torch_interop_utils.unregister_grad_fn(id(ctx))

            return tuple(wrapped_returned_args)
        except Exception as e:
            # Flush buffers. Otherwise, calling this from C++ may lose them.
            print("Exception happens when running ", backward_function)
            sys.stdout.flush()
            sys.stderr.flush()
            raise wrap_exception(ORTModuleFallbackException, e)  # noqa: B904
