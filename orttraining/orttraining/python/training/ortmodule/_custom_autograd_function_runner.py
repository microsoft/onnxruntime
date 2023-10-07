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
from ._utils import get_rank


def _log_warning(message: str):
    """Configure the logger for PythonOp runner according to following rules.
    1. If multiple processes are used, the rank will be appended
       to the logger name.
    2. The logger will be disabled for non-zero ranks.
    """
    if get_rank() == 0:
        warnings.warn(f"[rank-{get_rank()}] {message}")


class CustomFuncOpKernelInfo:
    """Store the kernel-specific information retrieved with the first-time run."""

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
        # 2. For the subsequent runs, if the input index is in `tensor_input_indices_to_save_in_ctx`, the tensor
        # will be cloned before fed into `autograd.Function.apply` as input.
        self.tensor_input_indices_to_save_in_ctx: Optional[List[int]] = None

        # To align with PyTorch `ctx.set_materialize_grads(False|True)``
        # materialize_grads_config is a map from output index to (device, dtype, shape) of the output tensor, used
        # for materializing the gradient of the output tensor in backward.
        self.materialize_grads: bool = False
        self.materialize_grads_config: Optional[Dict[int, Tuple[torch.device, torch.dtype, torch.shape]]] = None

        # For the tensors generated from ORT backend, there is special handling here:
        # 1. For the first time run for the kernel (the uniqueness of the kernel is defined by kernel_invoke_id),
        # all such tensors will be cloned (with gradient) in case they are marked as dirty (if not cloned, but marked
        # as dirty, PyTorch will complain the tensor is a leaf, should not be used for inplace update). Once
        # `autograd.Function.apply` completes, by checking the existence of the tensor in the dirty_tensors,
        # `_GlobalOpKernelInfoMap` is updated to save the input indices that are marked as dirty.
        # 2. For the subsequent runs, if the input index is in `tensor_input_indices_for_mark_dirty`, the tensor
        # will be cloned (with gradient) before fed into `autograd.Function.apply` as input.
        self.tensor_input_indices_for_mark_dirty: Optional[List[int]] = None

        # A list of output indices that needs to be clone before returned, due to inplace update analysis.
        self.output_indices_for_clone: Optional[List[int]] = None


# Store the kernel-specific information that cannot be retrieved and saved by PyTorch exporter.
# For the infos that can only be retrieved with real run, we try to collect them in the first time run.
# key: kernel_invoke_id, value: CustomFuncOpKernelInfo.
_GlobalOpKernelInfoMap: Dict[str, CustomFuncOpKernelInfo] = {}


def _process_inplace_outputs(
    kernel_info: CustomFuncOpKernelInfo,
    func_name: str,
    input_tensors_of_kernel_run: Dict[int, Union[torch.Tensor, None]],
    all_outputs_of_kernel_run: List[Union[torch.Tensor, any]],
    all_outputs_to_tensor_inputs_reuse_map: List[int],
    raw_input_tensors_used_inplace: Dict[int, Union[torch.Tensor, None]],
    is_backward=False,
):
    """Special handling for in-place reusing in forward or backward.

    Args:
        kernel_info: kernel-specific information.
        func_name: name of the autograd.Function.
        input_tensors_of_kernel_run: all tensor input tensors used to run the autograd.Function forward/backward.
        all_outputs_of_kernel_run: all outputs of the autograd.Function forward/backward.
        all_outputs_to_tensor_inputs_reuse_map: a list of the same length of kernel outputs, each element representing
            which input index it is reusing. If there is no reuse, the value is -1.
        raw_input_tensors_used_inplace: a dict of raw input tensors marked as inplace in
            `all_outputs_to_tensor_inputs_reuse_map`, the key is the tensor input index, value is the raw input tensor.
        is_backward: indicates if this is backward or forward.

    Procedures:
    1. Detect all outputs to tensor inputs reuse mapping.
    2. Validate the detected inplace_map with the registered inplace_map in ORT. For the output tensor,
        2.0 If the reuse mapping value is the same in both inplace_map and detected inplace_map:
            2.0.1 Most likely, we don't need to do anything, except 2.0.2.
            2.0.2 Conditions:
                > During forward run,
                > The output tensor is reusing one of input tensors,
                > The raw input tensor to be reused given from ORT is copied to run the forward kernels
                    (for two possible reasons:
                    a. the first time forward run, all inputs will be copied to detect
                    `tensor_input_indices_to_save_in_ctx`;
                    b. for every iteration, the input needs to be cloned because it is in
                    `tensor_input_indices_to_save_in_ctx`).

                In this case, need to copy the output tensor back to the raw input tensor, to make it compatible with
                ORT statistically planned buffer reuse.
        2.1 If the reuse mapping value is NOT equal in both inplace_map and detected inplace_map:
            2.1.1 If the detected reuse input index is -1 (e.g. there is NO buffer reuse for this output),
                while user specified reuse input index is NOT -1 (ORT planned the reuse), we raise an error.
            2.1.2 If the detected reuse input index is NOT -1 (e.g. there is buffer reuse for this output),
                while user specified reuse input index is -1 (ORT did not plan the reuse). We will try to clone the
                output tensor before returning to ORT, to align with ORT's NO Buffer reuse plan; otherwise, once the
                input buffer is released by ORT memory planner, the output tensor read/write will be corrupted.
                Raise a warning to notify users to update inplace_map explicitly for performance consideration.
            2.1.3 Other cases (for example user gives a wrong mapping index compared with detected ones), raise an
                error.
    3. Do copies for 2.1.2 cases.
    4. Do copies for 2.0.2 cases.
    """

    log_prefix = f"{func_name}->{'Backward' if is_backward else 'Forward'}: "
    input_tensor_address_list = [
        t.data_ptr() if isinstance(t, torch.Tensor) else -1 for t in input_tensors_of_kernel_run.values()
    ]
    if is_backward:
        input_tensor_address_list = [-1, *input_tensor_address_list]  # skip the context input

    is_first_time_init = kernel_info.output_indices_for_clone is None
    # If this is the first time run, collect runtime tensor reuse mapping.
    if is_first_time_init:
        # Procedure 1: Detect all outputs to tensor inputs reuse mapping, according to `all_outputs_of_kernel_run` and
        # `input_tensors_of_kernel_run`.
        assert len(all_outputs_to_tensor_inputs_reuse_map) == len(all_outputs_of_kernel_run), (
            f"{log_prefix}all_outputs_to_tensor_inputs_reuse_map and kernel run outputs should have the same length."
            f"all_outputs_to_tensor_inputs_reuse_map: {all_outputs_to_tensor_inputs_reuse_map}, "
            f"kernel run outputs: {all_outputs_of_kernel_run}"
        )

        # Detect all outputs to tensor inputs reuse mapping.
        detected_reuse_map = [-1] * (len(all_outputs_of_kernel_run))
        for output_index, arg in enumerate(all_outputs_of_kernel_run):
            if not isinstance(arg, torch.Tensor):
                continue
            if arg.data_ptr() in input_tensor_address_list:
                input_index = input_tensor_address_list.index(arg.data_ptr())
                detected_reuse_map[output_index] = input_index

        # Procedure 2: Validate the detected inplace_map with the registered inplace_map in ORT.
        output_indices_for_clone = (
            []
        )  # collect the output indices that need to be cloned before returned in case 2.1.2.
        for output_index, (detected_inplace_index, inplace_index) in enumerate(
            zip(detected_reuse_map, all_outputs_to_tensor_inputs_reuse_map)
        ):
            if inplace_index == detected_inplace_index:
                continue

            if (
                inplace_index in raw_input_tensors_used_inplace
                and raw_input_tensors_used_inplace[inplace_index] is None
            ):
                # Use specified inplace input index, but the input tensor is None, which means the input is not
                # a tensor, so we don't do further checks.
                continue

            # If users register inplace_map (alloc planner will do buffer reuse),
            # but detected inplace_map indicates it is NO inplace reusing, we raise an error.
            if inplace_index != -1 and detected_inplace_index == -1:
                raise RuntimeError(
                    f"{log_prefix}Fatal: "
                    f"ONNX Op attribute 'tensor_reuse_map' indicates {output_index}-th output is reusing input "
                    f"{inplace_index}, but detected inplace_map indicates it is NOT reusing any input. "
                    "Please update inplace_map explicitly to make it consistent "
                    f"to avoid undefined behavior due to ORT's memory reuse plan. "
                    f"inplace_map: {all_outputs_to_tensor_inputs_reuse_map}, "
                    f"detected inplace_map: {detected_reuse_map}"
                )

            if inplace_index == -1 and detected_inplace_index != -1:
                output_indices_for_clone.append(output_index)
                continue

            raise RuntimeError(
                f"{log_prefix}Fatal: "
                f"ONNX Op attribute 'inplace_map' indicates {inplace_index}-th output is reusing "
                f"input index {detected_inplace_index}, but detected inplace_map indicates it is reusing "
                f"input index {inplace_index}. Please update inplace_map explicitly to avoid undefined behavior "
                f"due to memory reuse. inplace_map: {all_outputs_to_tensor_inputs_reuse_map}, "
                f"detected inplace_map: {detected_reuse_map}"
            )

        kernel_info.output_indices_for_clone = output_indices_for_clone

    assert kernel_info.output_indices_for_clone is not None

    # Procedure 3: Do copies for 2.1.2 cases.
    for output_index in kernel_info.output_indices_for_clone:
        _log_warning(
            f"{log_prefix}ONNX Op attribute "
            f"'tensor_reuse_map' doesn't indicate {output_index}-th output is reusing any input, "
            f"but detected inplace_map indicates it is reusing some input index. "
            "A clone will be done before returning to ORT, to align with ORT's NO Buffer reuse plan. "
            "Please update inplace_map explicitly to avoid such a copy."
        )
        all_outputs_of_kernel_run[output_index] = all_outputs_of_kernel_run[output_index].detach().clone()

    # Procedure 4: Do copies for 2.0.2 cases.
    if is_backward is False and (
        is_first_time_init
        or kernel_info.tensor_input_indices_to_save_in_ctx
        or kernel_info.tensor_input_indices_for_mark_dirty
    ):
        for raw_tensor_input_index, raw_input_tensor in raw_input_tensors_used_inplace.items():
            # raw_input_tensor can be None for backward run, but backward won't go here.
            if not isinstance(raw_input_tensor, torch.Tensor):
                continue

            # We did not do the check with tensor_input_indices_to_save_in_ctx/tensor_input_indices_for_mark_dirty
            # because even for those tensor indices not in
            # tensor_input_indices_to_save_in_ctx/tensor_input_indices_for_mark_dirty, we still need to do the
            # copy for the first-time run.
            if raw_input_tensor.data_ptr() == input_tensor_address_list[raw_tensor_input_index]:
                # If the raw input tensor is not copied, we don't need this handling.
                continue

            copied = False  # for each tensor, we don't do the copy once.
            output_indices_reusing_current_raw_input = [
                output_index
                for output_index, input_index in enumerate(all_outputs_to_tensor_inputs_reuse_map)
                if input_index == raw_tensor_input_index
            ]
            output_tensor_address = all_outputs_of_kernel_run[output_indices_reusing_current_raw_input[0]].data_ptr()
            for output_index in output_indices_reusing_current_raw_input:
                assert (
                    output_tensor_address == all_outputs_of_kernel_run[output_index].data_ptr()
                ), "Outputs reusing the same input tensor should have the same address."

                if not copied:
                    # Only need a copy once.
                    raw_input_tensor.copy_(all_outputs_of_kernel_run[output_index])
                    _log_warning(
                        f"{log_prefix}Copy output tensor {output_index} to raw input tensor {raw_tensor_input_index}. "
                        f"{'Provide output to input reuse mapping to avoid the copy overhead.' if not is_first_time_init else ''}"
                    )
                    copied = True

                all_outputs_of_kernel_run[output_index] = raw_input_tensor


def _get_context(forward_tensor_outputs: List[torch.Tensor]) -> Tuple[any, Optional[torch.Tensor]]:
    """Search for context among all outputs.

    Note 1: All forward outputs of torch.autograd.Function shared the same gradient function pointer,
        so here we just get the first tensor having grad_fn attribute.
        (https://github.com/PyTorch/PyTorch/blob/15532595209d2daf34d35e10f8d3d3b64966aea2/torch/csrc/autograd/custom_function.cpp#L267)

    Note 2: Context can be None because NOT all torch.autograd.Function's are differentiable. The function
        https://github.com/PyTorch/PyTorch/blob/d701357d921ef167d42c125e65b6f7da6be3ad0f/torch/csrc/autograd/custom_function.cpp#L209?
        means if all output of the forward function is not differentiable, then grad_fn will be None (not be set).

        For example,
            class Bar(torch.autograd.Function):
                # A non-differentiable autograd Function whose forward output
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
            # For the following case, it is possible grad_fn exists, but its value is None,
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


def _finalize_training_mode_forward(
    kernel_invoke_id: str,
    func_name: str,
    input_tensors_used_for_fw_run: Dict[int, torch.Tensor],
    forward_output_tensors: List[Union[torch.Tensor, None]],
):
    """Complete the epilogue of forward runner for training mode.

    Args:
        kernel_invoke_id: kernel_invoke_id of the PythonOp kernel unique id.
        input_tensors_from_ort: input tensors generated from ORT backend.
        forward_output_tensors: output tensors of the autograd.Function.

    Things to do:
    1. Try to get context from forward output tensors.
    2. Remove the gradient functions between the current autograd.Function and its input's gradient function, because
       in ORT we don't depend on PyTorch's autograd engine.
    3. Register the current autograd.Function's gradient function into our PyNodeSharedPointerPool.
    4. Save kernel-specific information into _GlobalOpKernelInfoMap in the first-time kernel run.
    """

    ctx, tensor_owning_ctx = _get_context(forward_output_tensors)

    kernel_info = _GlobalOpKernelInfoMap[kernel_invoke_id]

    # ctx being None in training mode means the forward function is not differentiable, so backward is not needed.
    if ctx is None:
        # If this is the first time run, collect kernel-specific information.
        if kernel_info.tensor_input_indices_to_save_in_ctx is None:
            kernel_info.tensor_input_indices_to_save_in_ctx = []

        if kernel_info.tensor_input_indices_for_mark_dirty is None:
            kernel_info.tensor_input_indices_for_mark_dirty = []

        return None

    # Filter out the None in the saved_tensors.
    saved_tensors = [t for t in ctx.saved_tensors if t is not None]

    ctx.fw_kernel_invoke_id = kernel_invoke_id

    # If this is the first time run, collect kernel-specific information.
    if kernel_info.tensor_input_indices_to_save_in_ctx is None:
        kernel_info.tensor_input_indices_to_save_in_ctx = []
        if len(saved_tensors):
            # Check tensors generated by ORT are in the saved_tensors or not.
            # If yes, save the input index of the tensor in the _GlobalOpKernelInfoMap.
            kernel_info.tensor_input_indices_to_save_in_ctx = [
                tensor_input_index
                for tensor_input_index, tensor in input_tensors_used_for_fw_run.items()
                if any(tensor is saved_tensor for saved_tensor in saved_tensors)
            ]
            _log_warning(
                f"{func_name}: Add input index to _GlobalOpKernelInfoMap, to avoid extra copy in every iteration."
            )
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

    if kernel_info.tensor_input_indices_for_mark_dirty is None:
        kernel_info.tensor_input_indices_for_mark_dirty = []
        # Check tensors generated by ORT are marked as dirty(for inplace update) or not.
        # If yes, save the input index of the tensor in the _GlobalOpKernelInfoMap.
        are_tensors_marked_as_dirty = torch_interop_utils.are_tensors_marked_as_dirty(
            tensor_owning_ctx, [t for t in input_tensors_used_for_fw_run.values()]
        )
        kernel_info.tensor_input_indices_for_mark_dirty = [
            tensor_input_index
            for is_dirty, (tensor_input_index, tensor) in zip(
                are_tensors_marked_as_dirty, input_tensors_used_for_fw_run.items()
            )
            if is_dirty is True
        ]
        _log_warning(f"{func_name}: Add input index to _GlobalOpKernelInfoMap, to support leaf node do inplace update.")

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
    inplace_map: List[int],
    kernel_invoke_id: str,
    func_name: Union[bytes, str],
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
        inplace_map: a list of the same length of kernel outputs, each element represents which input index
          it is reusing. If there is no reuse, the value is -1.
        args: inputs to "backward_function".
    """

    try:
        func_name = func_name.decode("utf-8") if isinstance(func_name, bytes) else func_name
        # If this is the first time run, collect runtime tensor reuse mapping.
        if kernel_invoke_id not in _GlobalOpKernelInfoMap:
            kernel_info = CustomFuncOpKernelInfo(kernel_invoke_id)
            _GlobalOpKernelInfoMap[kernel_invoke_id] = kernel_info

        kernel_info = _GlobalOpKernelInfoMap[kernel_invoke_id]

        tensor_input_indices_to_save_in_ctx = kernel_info.tensor_input_indices_to_save_in_ctx
        tensor_input_indices_for_mark_dirty = kernel_info.tensor_input_indices_for_mark_dirty

        # Collect the tensor address for all inputs used for run forward, used for reuse detection.
        tensor_input_index = 0
        # If the input is reused, we need to save the raw input tensor for special handling.
        raw_input_tensors_used_inplace = OrderedDict()  # Orders matter here.
        input_tensors_used_for_fw_run = OrderedDict()  # Orders matter here.

        wrapped_args = []
        for _, (grad_flag, tensor_flag, arg) in enumerate(zip(requires_grad_flags, tensor_type_flags, args)):
            if tensor_flag:
                # Assume it's a DLPack tensor and convert it to PyTorch tensor.
                wrapped_arg = from_dlpack(arg)

                if tensor_input_index in inplace_map:
                    raw_input_tensors_used_inplace[tensor_input_index] = wrapped_arg

                # Note1:
                #   If it's first-time kernel invocation, tensor_input_indices_to_save_in_ctx is None, we do the
                #   copy for all tensors. Otherwise, we only copy the tensors whose indices are in
                #   tensor_input_indices_to_save_in_ctx.
                # Note2:
                #   For inference mode, we don't need to do the copy because ctx will be None,
                #   so nothing will be saved for ctx.
                if is_training_mode and (
                    tensor_input_indices_to_save_in_ctx is None
                    or tensor_input_index in tensor_input_indices_to_save_in_ctx
                ):
                    wrapped_arg = wrapped_arg.detach().clone()

                # Only requires gradient when running under training mode
                # and the associated tensor has grad_flag=True (i.e.,
                # "requires_grad=True" in the original PyTorch script).
                wrapped_arg.requires_grad = is_training_mode and grad_flag

                # Note3:
                #   If it's not first-time kernel invocation, tensor_input_indices_for_mark_dirty is None, we do the
                #   mul for all tensors. Otherwise, we only mul by one for the tensors whose indices are in
                #   tensor_input_indices_for_mark_dirty.
                if is_training_mode and (
                    tensor_input_indices_for_mark_dirty is None
                    or tensor_input_index in tensor_input_indices_for_mark_dirty
                ):
                    # To fix this issue:
                    # "a leaf Variable that requires grad has been used in an in-place operation."
                    with torch.set_grad_enabled(True):
                        wrapped_arg = wrapped_arg.clone()

                wrapped_args.append(wrapped_arg)
                input_tensors_used_for_fw_run[tensor_input_index] = wrapped_arg

                tensor_input_index += 1
            else:
                # Use non-tensor as is. It's a PyObject*.
                wrapped_args.append(arg)

        with torch.set_grad_enabled(is_training_mode):
            # Run autograd.Function.apply(...).
            # TODO(pengwa): looks like we are assuming all outputs will be either Tensor or None.
            # We should revisit if it is possible to support other types of output, for example int, or, etc.
            # But that might also require some work in backend.
            result = forward_function(*wrapped_args)

            results = []
            if isinstance(result, torch.Tensor):
                results = [result]
            elif isinstance(result, (tuple, list)):
                results = [r for r in result]
            else:
                raise wrap_exception(
                    ORTModuleIOError,
                    TypeError(f"ORTModule does not support the following model output type {type(result)}."),
                )

            ctx = None
            if is_training_mode:
                ctx = _finalize_training_mode_forward(
                    kernel_invoke_id, func_name, input_tensors_used_for_fw_run, results
                )

            final_rets = [ctx]
            final_rets.extend(results)

            _process_inplace_outputs(
                kernel_info,
                func_name,
                input_tensors_used_for_fw_run,
                final_rets,
                inplace_map,
                raw_input_tensors_used_inplace,
            )

            dlpacks = [final_rets[0]]
            dlpacks.extend(list(to_dlpack(value) if value is not None else None for value in final_rets[1:]))

            # Inside the returned list, the first element is context and the rest
            # are DLPack tensors.
        return tuple(dlpacks)
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
    inplace_map: List[int],
    kernel_invoke_id: str,
    func_name: Union[bytes, str],
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
        tensor_type_flags: tensor_type_flags[i] indicates the type of the i-th arg.
        is_training_mode: indicates if this model is running under training mode.
        inplace_map: a list of the same length of kernel outputs, each element represents which input index
          it is reusing. If there is no reuse, the value is -1.
        args: inputs to "backward_function".
    """
    func_name = func_name.decode("utf-8") if isinstance(func_name, bytes) else func_name
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
            # If this is the first time run, collect runtime tensor reuse mapping.
            if kernel_invoke_id not in _GlobalOpKernelInfoMap:
                kernel_info = CustomFuncOpKernelInfo(kernel_invoke_id)
                _GlobalOpKernelInfoMap[kernel_invoke_id] = kernel_info

            kernel_info = _GlobalOpKernelInfoMap[kernel_invoke_id]

            # Backward inputs should not require gradients.
            assert all(grad_flag == 0 for grad_flag in requires_grad_flags)

            # Prepare inputs for calling Python function.
            ctx = args[0]
            fw_kernel_invoke_id = ctx.fw_kernel_invoke_id
            wrapped_args = []

            # Collect the tensor address for all inputs used for run backward, used for reuse detection.
            tensor_input_index = 1  # skip the context input
            # If input is reused, we need to save the raw input tensor for special handling.
            raw_input_tensors_used_inplace = OrderedDict()  # Orders matter here.
            input_tensors_used_for_bw_run = OrderedDict()  # Orders matter here.
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

                        if grad_input_index in inplace_map:
                            raw_input_tensors_used_inplace[tensor_input_index] = arg

                    else:
                        # Assume it's a DLPack tensor# and convert it to PyTorch tensor.
                        wrapped_arg = from_dlpack(arg)

                        if grad_input_index in inplace_map:
                            raw_input_tensors_used_inplace[tensor_input_index] = wrapped_arg

                    # This may include None values.
                    input_tensors_used_for_bw_run[tensor_input_index] = wrapped_arg

                    if wrapped_arg is not None:
                        # Only requires gradient when running under training mode
                        # and the associated tensor has grad_flag=True (i.e.,
                        # "requires_grad=True" in the original PyTorch script).
                        wrapped_arg.requires_grad = is_training_mode and grad_flag

                    wrapped_args.append(wrapped_arg)
                    tensor_input_index += 1
                else:
                    # Use non-tensor as is. It's a PyObject*.
                    wrapped_args.append(arg)

            # Call Python function.
            result = backward_function(*wrapped_args)

            # Extract results as DLPack tensor list.
            if isinstance(result, torch.Tensor):
                result = [result]
            elif isinstance(result, (tuple, list)):
                result = list(result)
            else:
                raise wrap_exception(
                    ORTModuleIOError,
                    TypeError(f"ORTModule does not support the following model output type {type(result)}."),
                )

            _process_inplace_outputs(
                kernel_info,
                func_name,
                input_tensors_used_for_bw_run,
                result,
                inplace_map,
                raw_input_tensors_used_inplace,
                is_backward=True,
            )

            wrapped_returned_args = wrap_all_outputs(result)

            torch_interop_utils.unregister_grad_fn(id(ctx))

            return tuple(wrapped_returned_args)
        except Exception as e:
            # Flush buffers. Otherwise, calling this from C++ may lose them.
            print("Exception happens when running ", backward_function)
            sys.stdout.flush()
            sys.stderr.flush()
            raise wrap_exception(ORTModuleFallbackException, e)  # noqa: B904
