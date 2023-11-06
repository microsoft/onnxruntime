# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------


from typing import Dict, List, Optional, Tuple, Union

import torch
from onnx import ModelProto, NodeProto, TensorProto, ValueInfoProto, helper

from onnxruntime.capi._pybind_state import (
    register_input_alias_function,
    register_shape_inference_function,
    register_torch_autograd_function,
)
from onnxruntime.training.utils import pytorch_type_to_onnx_dtype

from ._custom_autograd_function_exporter import register_custom_function_schema_supplementary
from ._utils import get_fully_qualified_class_name

STAGE3_PULL_WEIGHT_TRIGGER_NAME = "pull_weight_trigger"
STAGE3_PULL_WEIGHT_TRIGGER_OUTPUT_DTYPE = TensorProto.FLOAT
STAGE3_PULL_WEIGHT_TRIGGER_OUTPUT_SHAPE = [1]


def post_processing_enable_zero_stage3_compat(
    exported_model: ModelProto,
    zero_stage3_named_params: Dict[str, torch.nn.parameter.Parameter],
    all_param_names: List[str],
) -> ModelProto:
    """This function is used to enable zero stage3 compatibility.

    Args:
        exported_model (ModelProto): The exported model.
        zero_stage3_named_params (Optional[Dict[str, torch.nn.parameter.Parameter]]): The offload named parameters.
        all_param_names (List[str]): All parameter names.
    """

    # Register symbolic shape inference functions for PythonOp used in DeepSpeed ZeRO stage3.
    _register_symbolic_shape_infer_functions()

    _register_alias_input_functions()

    # Create weight retrieving function using zero_stage3_named_params.
    func_full_qual_name = _create_weight_retrieval_function(zero_stage3_named_params)

    consumer_map = {}
    for node in exported_model.graph.node:
        for inp in node.input:
            if inp not in consumer_map:
                consumer_map[inp] = []

            if node not in consumer_map[inp]:
                consumer_map[inp].append(node)

    def _get_param_pull_trigger_name(param_name: str) -> str:
        return f"pull_{param_name}"

    def _get_func_name(node: NodeProto) -> Optional[str]:
        for attr in node.attribute:
            if attr.name == "func_name":
                return attr.s.decode("utf-8") if isinstance(attr.s, bytes) else attr.s
        return None

    # Create weight retrieving PythonOp.
    new_input, weight_pull_node = _create_weight_retrieval_pythonop(
        zero_stage3_named_params,
        func_full_qual_name,
        STAGE3_PULL_WEIGHT_TRIGGER_NAME,
        [_get_param_pull_trigger_name(pname) for pname in zero_stage3_named_params],
        STAGE3_PULL_WEIGHT_TRIGGER_OUTPUT_DTYPE,
        STAGE3_PULL_WEIGHT_TRIGGER_OUTPUT_SHAPE,
    )

    from onnxruntime.training.utils.hooks._zero_offload_subscriber import ORTZeROOffloadPreForwardFunction

    pre_forward_function_name = get_fully_qualified_class_name(ORTZeROOffloadPreForwardFunction)

    # Connect weight consumers to use the full-sized parameter output of ORTZeROOffloadPreForwardFunction.
    for graph_input in exported_model.graph.input:
        if graph_input.name not in zero_stage3_named_params:
            continue

        if graph_input.name not in consumer_map:
            continue

        consumers = consumer_map[graph_input.name]
        pre_forward_pythonop_node = None

        for c in consumers:
            if c.op_type != "PythonOp":
                continue

            func_name = _get_func_name(c)
            if func_name == pre_forward_function_name:
                assert (
                    pre_forward_pythonop_node is None
                ), "Multiple ORTZeROOffloadPreForwardFunction nodes found, it should not happen"
                pre_forward_pythonop_node = c

        if pre_forward_pythonop_node is None:
            raise RuntimeError(
                "Fail to find ORTZeROOffloadPreForwardFunction for partitioned param: " + graph_input.name
            )

        pull_weight_trigger_input_name = _get_param_pull_trigger_name(graph_input.name)
        index_offset_on_python_op_input = []
        for i, input_name in enumerate(pre_forward_pythonop_node.input):
            if input_name == graph_input.name:
                index_offset_on_python_op_input.append(i)

        assert (
            len(index_offset_on_python_op_input) == 1
        ), f"index_offset_on_python_op_input length is not 1: {index_offset_on_python_op_input} for node {pre_forward_pythonop_node.name}, input {graph_input.name}, {pre_forward_pythonop_node.input}"

        reverse_index_among_inputs = index_offset_on_python_op_input[0] - len(pre_forward_pythonop_node.input)

        pre_forward_pythonop_node.input[index_offset_on_python_op_input[0]] = pull_weight_trigger_input_name

        _update_python_op_input_related_attributes(
            pre_forward_pythonop_node,
            pull_weight_trigger_input_name,
            len(STAGE3_PULL_WEIGHT_TRIGGER_OUTPUT_SHAPE),  # new rank
            STAGE3_PULL_WEIGHT_TRIGGER_OUTPUT_DTYPE,  # new data type
        )

        output_index = reverse_index_among_inputs + len(pre_forward_pythonop_node.output)

        ready_weight_name = f"ready_{graph_input.name}"
        pre_forward_pythonop_node.output[output_index] = ready_weight_name

        # Update consumer's input to use the full-sized parameter output of ORTZeROOffloadPreForwardFunction.
        for c in consumers:
            new_inputs = [c_input for c_input in c.input]
            for c_input_index in range(len(c.input)):
                if c.input[c_input_index] == graph_input.name:
                    new_inputs[c_input_index] = ready_weight_name
            del c.input[:]
            c.input.extend(new_inputs)

        # If the consumer of original `graph_input.name` is PythonOp, we need also update its attributes because now
        # `graph_input.name` as output of pre_forward_pythonop_node, is full-sized parameter, the rank might differ
        # from the original one.
        for c in consumers:
            if c == pre_forward_pythonop_node or c.op_type != "PythonOp":
                continue
            _update_python_op_input_related_attributes(
                c,
                graph_input.name,
                len(zero_stage3_named_params[graph_input.name].ds_shape),  # new rank
                pytorch_type_to_onnx_dtype(zero_stage3_named_params[graph_input.name].dtype),  # new data type
            )

    # Delete exported_model.graph.input
    graph_inputs_to_remove = [
        graph_input for graph_input in exported_model.graph.input if graph_input.name in zero_stage3_named_params
    ]
    for input_to_remove in graph_inputs_to_remove:
        exported_model.graph.input.remove(input_to_remove)

    # Re-order graph input to make sure the weight pull trigger is before all parameter inputs.
    offset = 0
    for graph_input in exported_model.graph.input:
        if graph_input.name in all_param_names:
            break
        offset += 1

    exported_model.graph.input.insert(offset, new_input)
    exported_model.graph.node.insert(0, weight_pull_node)

    patch_export_functions(zero_stage3_named_params)

    return exported_model


def _create_weight_retrieval_function(
    zero_stage3_named_params: Optional[Dict[str, torch.nn.parameter.Parameter]]
) -> str:
    """This function is used to create a weight retrieving function using zero_stage3_named_params."""

    class WeightRetrievalFunction(torch.autograd.Function):
        @staticmethod
        def forward(ctx, weight_in_trigger):
            params = list(zero_stage3_named_params.values())
            ctx.params = params
            ctx.dtype = weight_in_trigger.dtype
            ctx.device = weight_in_trigger.device
            ctx.shape = weight_in_trigger.shape
            return (torch.zeros(ctx.shape, device=ctx.device, dtype=ctx.dtype),) * len(params)

        @staticmethod
        def backward(ctx, *grad_outputs):
            return torch.zeros(ctx.shape, device=ctx.device, dtype=ctx.dtype)

        @staticmethod
        def infer_shape(
            node: NodeProto,
            tensor_input_shapes: List[Optional[List[Union[int, str]]]],
            tensor_input_dtypes: List[torch.onnx.TensorProtoDataType],
        ) -> Tuple[List[Optional[List[Union[int, str]]]], List[torch.onnx.TensorProtoDataType]]:
            param_count = len(zero_stage3_named_params.values())
            tensor_output_shapes = [
                tensor_input_shapes[0],
            ] * param_count
            tensor_output_dtypes = [
                tensor_input_dtypes[0],
            ] * param_count

            return tensor_output_shapes, tensor_output_dtypes

    func_full_qual_name = get_fully_qualified_class_name(WeightRetrievalFunction)
    register_torch_autograd_function(func_full_qual_name, WeightRetrievalFunction)

    register_custom_function_schema_supplementary(WeightRetrievalFunction)

    return func_full_qual_name


def _register_symbolic_shape_infer_functions():
    """This function is used to register symbolic shape inference functions for PythonOp used in
    DeepSpeed ZeRO stage3."""

    def _simple_pass_through_infer_shape(
        node: NodeProto,
        tensor_input_shapes: List[Optional[List[Union[int, str]]]],
        tensor_input_dtypes: List[torch.onnx.TensorProtoDataType],
    ) -> Tuple[List[Optional[List[Union[int, str]]]], List[torch.onnx.TensorProtoDataType]]:
        return tensor_input_shapes, tensor_input_dtypes

    register_shape_inference_function(
        "deepspeed.runtime.zero.parameter_offload.PreBackwardFunction", _simple_pass_through_infer_shape
    )
    register_shape_inference_function(
        "deepspeed.runtime.zero.parameter_offload.PostBackwardFunction", _simple_pass_through_infer_shape
    )

    def _linear_infer_shape(
        node: NodeProto,
        tensor_input_shapes: List[Optional[List[Union[int, str]]]],
        tensor_input_dtypes: List[torch.onnx.TensorProtoDataType],
    ) -> Tuple[List[Optional[List[Union[int, str]]]], List[torch.onnx.TensorProtoDataType]]:
        # output = input.matmul(weight.t())
        tensor_input_shapes[0]  # input
        shape2 = tensor_input_shapes[1]  # weight
        output_shape = tensor_input_shapes[0]
        output_shape[-1] = shape2[-2]
        return [output_shape], [tensor_input_dtypes[0]]

    register_shape_inference_function("deepspeed.runtime.zero.linear.LinearFunctionForZeroStage3", _linear_infer_shape)


def _register_alias_input_functions():
    """This function is used to register symbolic shape inference functions for PythonOp used in
    DeepSpeed ZeRO stage3."""

    def _alias_input(node_proto_str: str):
        node: NodeProto = NodeProto()
        node.ParseFromString(node_proto_str)
        non_tensor_fw_input_count = 2

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

    register_input_alias_function("deepspeed.runtime.zero.parameter_offload.PreBackwardFunction", _alias_input)
    register_input_alias_function("deepspeed.runtime.zero.parameter_offload.PostBackwardFunction", _alias_input)


def _create_weight_retrieval_pythonop(
    zero_stage3_named_params: Optional[Dict[str, torch.nn.parameter.Parameter]],
    func_full_qual_name: str,
    input_name: str,
    output_names: List[str],
    pull_weight_trigger_output_dtype: int,
    pull_weight_trigger_output_shape: List[int],
) -> Tuple[ValueInfoProto, NodeProto]:
    """This function is used to create a weight retrieving PythonOp."""
    offload_param_count = 0 if zero_stage3_named_params is None else len(zero_stage3_named_params)
    new_input = helper.make_tensor_value_info(
        input_name, pull_weight_trigger_output_dtype, pull_weight_trigger_output_shape
    )
    output_rank_for_pull_weight_trigger = len(pull_weight_trigger_output_shape)
    output_dtype_for_pull_weight_trigger = pull_weight_trigger_output_dtype
    output_tensor_ranks = [
        output_rank_for_pull_weight_trigger,
    ] * offload_param_count
    output_tensor_types = [
        output_dtype_for_pull_weight_trigger,
    ] * offload_param_count

    node_attributes = {
        "comment": "",
        "input_convention": "d",
        "input_tensor_ranks": [len(pull_weight_trigger_output_shape)],
        "input_tensor_types": [pull_weight_trigger_output_dtype],
        "output_tensor_ranks": output_tensor_ranks,
        "output_tensor_types": output_tensor_types,
        "training_mode": 1,
        "func_name": func_full_qual_name,
    }

    weight_pull_node = helper.make_node(
        "PythonOp",
        [input_name],
        ["pull_weight_trigger_ctx", *output_names],
        "pull_weight_trigger",  # node name
        "PythonOp for weight retrieving.",
        "com.microsoft",
        **node_attributes,
    )

    return new_input, weight_pull_node


def _update_python_op_input_related_attributes(
    node: NodeProto, input_name: str, new_rank: int, new_dtype: torch.onnx.TensorProtoDataType
):
    """This function is used to update PythonOp's input related attributes, e.g.
        input_tensor_ranks and input_tensor_types.

    Args:
        node (NodeProto): The PythonOp node.
        input_name (str): The input name to be updated.
        new_rank (int): The new rank of the input, to be used in input_tensor_ranks.
        new_dtype (torch.onnx.TensorProtoDataType): The new data type of the input, to be used in input_tensor_types.
    """
    input_tensor_ranks = None
    input_tensor_dtypes = None
    rank_attr = None
    dtype_attr = None
    for attr in node.attribute:
        if attr.name == "input_tensor_ranks":
            input_tensor_ranks = attr.ints
            rank_attr = attr
        if attr.name == "input_tensor_types":
            input_tensor_dtypes = attr.ints
            dtype_attr = attr

    assert input_tensor_ranks is not None, "input_tensor_ranks is None"
    assert input_tensor_dtypes is not None, "input_tensor_dtypes is None"

    for index, node_input_name in enumerate(node.input):
        if node_input_name == input_name:
            input_tensor_ranks[index] = new_rank
            input_tensor_dtypes[index] = int(new_dtype)

    node.attribute.remove(rank_attr)
    node.attribute.remove(dtype_attr)
    node.attribute.append(helper.make_attribute("input_tensor_ranks", input_tensor_ranks))
    node.attribute.append(helper.make_attribute("input_tensor_types", input_tensor_dtypes))


def patch_export_functions(zero_stage3_named_params: Dict[str, torch.nn.parameter.Parameter]):
    import torch.onnx.symbolic_helper as sym_help
    from torch.onnx._internal import _beartype

    @_beartype.beartype
    def _get_tensor_rank(x) -> Optional[int]:
        ### Adapted from https://github.com/pytorch/pytorch/blob/185515368bcd7d94ac06ab1634f22b747b03c6d9/torch/onnx/symbolic_helper.py#L561
        # Retrieve the real rank for the stage3 weights, because stage3 weights are all (0).
        import typing

        from torch import _C

        input_name = x.debugName()
        if input_name in zero_stage3_named_params:
            rank = len(zero_stage3_named_params[input_name].ds_shape)
            return rank

        if not sym_help._is_tensor(x) or x.type() is None:
            return None
        x_type = x.type()
        x_type = typing.cast(_C.TensorType, x_type)
        return x_type.dim()

    torch.onnx.symbolic_helper._get_tensor_rank = _get_tensor_rank
