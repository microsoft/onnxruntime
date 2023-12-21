# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------


from typing import List, Optional, Tuple, Union

import onnx
import torch

from onnxruntime.training.utils import log_memory_usage, extract_data_and_schema, unflatten_data_using_schema, ORTModelInputOutputType


from ._subscriber_base import RuntimeStates, SubscriberBase


_PRE_FW_PASS_PHASE = "pre-fw-pass"
_POST_FW_PASS_PHASE = "post-fw-pass"
_PRE_BW_PASS_PHASE = "pre-bw-pass"
_POST_BW_PASS_PHASE = "post-bw-pass"

class _InspectMemoryUsage(torch.autograd.Function):
    """This class is used to print the memory statistics in the forward and backward passes."""

    @staticmethod
    def forward(ctx, phase: str, run_ctx: RuntimeStates, module: torch.nn.Module,
                *input_tensor_list: Tuple[torch.Tensor, ...]) -> Tuple[torch.Tensor, ...]:
        """Make sure there is the same number of `tensor` inputs and outputs.
        """
        ctx.current_step = run_ctx.global_states.execution_step
        ctx.phase = phase
        ctx.module = module

        assert ctx.phase in [_PRE_FW_PASS_PHASE, _POST_FW_PASS_PHASE], f"Invalid phase {ctx.phase}"

        # The step is not always consistent with the step in users' training loops.
        # It is a counter of how many times the forward+backward pass is called.
        log_memory_usage(f"{ctx.phase}", rank_0_only=True, step_info=f"step {ctx.current_step}", module=ctx.module)

        return tuple(t.detach().requires_grad_(t.requires_grad) for t in input_tensor_list)

    @staticmethod
    def backward(ctx, *grad_output: Tuple[Optional[torch.Tensor], ...]) -> Tuple[Optional[torch.Tensor], ...]:
        phase = ctx.phase
        if ctx.phase == _PRE_FW_PASS_PHASE:
            phase = _POST_BW_PASS_PHASE
        elif ctx.phase == _POST_FW_PASS_PHASE:
            phase = _PRE_BW_PASS_PHASE
        log_memory_usage(f"{phase}", rank_0_only=True, step_info=f"step {ctx.current_step}", module=ctx.module)
        return (None, None, None, *tuple(g for g in grad_output))

    @staticmethod
    def infer_shape(
        node: onnx.NodeProto,
        tensor_input_shapes: List[Optional[List[Union[int, str]]]],
        tensor_input_dtypes: List[torch.onnx.TensorProtoDataType],
    ) -> Tuple[List[Optional[List[Union[int, str]]]], List[torch.onnx.TensorProtoDataType]]:
        return tensor_input_shapes, tensor_input_dtypes

    @staticmethod
    def alias_input(node_proto_str: str):
        node = onnx.NodeProto()
        node.ParseFromString(node_proto_str)
        non_tensor_fw_input_count = 3
        fw_output_count = len(node.output) - 1  # exclude the first output appended in ONNX
        fw_alias_map = [-1] * fw_output_count
        bw_alias_map = [-1] * (non_tensor_fw_input_count + len(node.input))

        for i in range(fw_output_count):
            fw_alias_map[i] = i + non_tensor_fw_input_count

        tensor_input_index = 0
        for i in range(len(bw_alias_map)):
            if i < non_tensor_fw_input_count:
                continue
            bw_alias_map[i] = tensor_input_index
            tensor_input_index += 1
        return fw_alias_map, bw_alias_map



class MemoryStatisticsSubscriber(SubscriberBase):
    """
    This subscriber is used to print the memory statistics in the forward and backward passes.
    """

    def __init__(
        self,
        start_step: Union[None, int] = None,
        end_step: Union[None, int] = None,
    ):
        """
        Steps in [start_step, end_step) will run subscriber actions.

        Args:
            start_step: the first step that runs subscriber actions.
            end_step: the end step (exclusively) that runs subscriber actions.
        """
        super().__init__(start_step=start_step, end_step=end_step)

    def pre_forward_outmost_module_apply_impl(
        self,
        run_ctx: RuntimeStates,
        module: torch.nn.Module,
        args: ORTModelInputOutputType,
        kwargs: ORTModelInputOutputType,
    ) -> Tuple[ORTModelInputOutputType, ORTModelInputOutputType]:

        flatten_args_tensor_list, args_schema = extract_data_and_schema(args)
        flatten_kwargs_tensor_list, kwargs_schema = extract_data_and_schema(kwargs)
        flatten_out = _InspectMemoryUsage.apply(_PRE_FW_PASS_PHASE, run_ctx, module,
                                                 *(flatten_args_tensor_list + flatten_kwargs_tensor_list))
        args_tensors = flatten_out[:len(flatten_args_tensor_list)]
        kwargs_tensors = flatten_out[len(flatten_args_tensor_list):]
        restored_args = unflatten_data_using_schema(args_tensors, args_schema)
        restored_kwargs = unflatten_data_using_schema(kwargs_tensors, kwargs_schema)

        return restored_args, restored_kwargs


    def post_forward_outmost_module_apply_impl(
        self,
        run_ctx: RuntimeStates,
        module: torch.nn.Module,
        args: ORTModelInputOutputType,
        outputs: ORTModelInputOutputType,
    ) -> Tuple[ORTModelInputOutputType, ORTModelInputOutputType]:

        flatten_output_tensor_list, output_schema = extract_data_and_schema(outputs)
        output_tensors = _InspectMemoryUsage.apply(_POST_FW_PASS_PHASE, run_ctx, module, *flatten_output_tensor_list)
        restored_outputs = unflatten_data_using_schema(output_tensors, output_schema)

        return args, restored_outputs
