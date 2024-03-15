# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

from __future__ import annotations

import ctypes

import torch
from onnx import ModelProto, NodeProto, TensorProto, helper

from onnxruntime.training.utils import pytorch_type_to_onnx_dtype

from ._pythonop_helper import make_pythonop_node

MEM_EFFICIENT_PARAM_TRIGGER_INPUT_NAME = "mem_efficient_pull_weight_trigger"
MEM_EFFICIENT_PARAM_TRIGGER_OUTPUT_DTYPE = TensorProto.FLOAT
MEM_EFFICIENT_PARAM_TRIGGER_OUTPUT_SHAPE = [1]


def get_params_connected_to_pull_param_trigger(
    named_params: dict[str, torch.nn.parameter.Parameter], exported_model: ModelProto
):
    # Be noted, some parameters might not in graph input because they are not used in forward, so we filtered them also.
    onnx_initializer_names = {p.name for p in exported_model.graph.input}
    return {k: v for k, v in named_params if v.requires_grad and k in onnx_initializer_names}


def get_params_not_connected_to_pull_param_trigger(
    named_params: dict[str, torch.nn.parameter.Parameter], exported_model: ModelProto
):
    # Be noted, some parameters might not in graph input because they are not used in forward, so we filtered them also.
    onnx_initializer_names = {p.name for p in exported_model.graph.input}
    return [v for k, v in named_params if not v.requires_grad and k in onnx_initializer_names]


def post_processing_enable_mem_efficient_training(
    exported_model: ModelProto,
    named_params: dict[str, torch.nn.parameter.Parameter],
) -> tuple[bool, ModelProto]:
    """This function is used to enable zero stage3 compatibility.

    Args:
        exported_model (ModelProto): The exported model.
        named_params (Optional[Dict[str, torch.nn.parameter.Parameter]]): The full parameter map.

    Returns:
        tuple[bool, ModelProto]: A tuple of bool and ModelProto. The bool indicates whether the model is modified.

    """
    trainable_named_params = get_params_connected_to_pull_param_trigger(named_params, exported_model)
    if len(trainable_named_params) == 0:
        return False, exported_model

    # Create weight retrieving function using trainable_named_params.
    param_pull_trigger_func_class = _create_param_trigger_function(trainable_named_params)
    param_retrieve_func_class = _create_param_retrieval_function(trainable_named_params)

    def _get_param_pull_trigger_name(param_name: str) -> str:
        return f"pull_{param_name}"

    # Create weight retrieving PythonOp.
    inputs = [
        helper.make_tensor_value_info(
            MEM_EFFICIENT_PARAM_TRIGGER_INPUT_NAME,
            MEM_EFFICIENT_PARAM_TRIGGER_OUTPUT_DTYPE,  # Use the same data type with output for the input
            MEM_EFFICIENT_PARAM_TRIGGER_OUTPUT_SHAPE,
        )
    ]

    outputs = [
        helper.make_tensor_value_info(
            _get_param_pull_trigger_name(pname),
            MEM_EFFICIENT_PARAM_TRIGGER_OUTPUT_DTYPE,
            MEM_EFFICIENT_PARAM_TRIGGER_OUTPUT_SHAPE,
        )
        for pname in trainable_named_params
    ]

    weight_pull_node = make_pythonop_node(
        "weight_pull_trigger",
        inputs,
        outputs,
        param_pull_trigger_func_class,
        training_mode=1,
        safe_run_mode=0,
    )

    graph_inputs_to_remove = []
    input_offset = 0
    for graph_input in exported_model.graph.input:
        if graph_input.name not in trainable_named_params:
            continue

        graph_inputs_to_remove.append(graph_input)

        # Create the param retrieval function for this parameter.
        node_inputs = [
            helper.make_tensor_value_info(
                _get_param_pull_trigger_name(graph_input.name),
                MEM_EFFICIENT_PARAM_TRIGGER_OUTPUT_DTYPE,
                MEM_EFFICIENT_PARAM_TRIGGER_OUTPUT_SHAPE,
            ),
            graph_input.name,  # Second param is a string, which represents the param_name
        ]

        node_outputs = [
            helper.make_tensor_value_info(
                graph_input.name,  # output use the same name as weight
                int(pytorch_type_to_onnx_dtype(trainable_named_params[graph_input.name].dtype)),
                list(trainable_named_params[graph_input.name].shape),
            ),
        ]

        new_node = make_pythonop_node(
            f"weight_retrieval_{graph_input.name}",
            node_inputs,
            node_outputs,
            param_retrieve_func_class,
            training_mode=1,
            safe_run_mode=0,
        )
        exported_model.graph.node.insert(input_offset, new_node)
        input_offset += 1

    # Delete exported_model.graph.input
    names_to_remove = [input.name for input in graph_inputs_to_remove]
    value_infos_to_remove = [
        value_info for value_info in exported_model.graph.value_info if value_info.name in names_to_remove
    ]
    for value_info in value_infos_to_remove:
        exported_model.graph.value_info.remove(value_info)

    for input_to_remove in graph_inputs_to_remove:
        exported_model.graph.input.remove(input_to_remove)

    # Re-order graph input to make sure the weight pull trigger is the first user input.
    offset = 0  # Find the first trainable param, and insert the new input before it, as part of user inputs.
    for input in exported_model.graph.input:
        if input.name in named_params:
            break
        offset += 1
    exported_model.graph.input.insert(offset, inputs[0])
    exported_model.graph.node.insert(0, weight_pull_node)

    return True, exported_model


_PARAM_FUNCTION_INDEX = [0]


def _create_param_trigger_function(trainable_named_params: dict[str, torch.nn.parameter.Parameter]):
    """This function is used to create a weight retrieving function using trainable_named_params."""

    @staticmethod
    def forward(ctx, weight_in_trigger):
        params = list(trainable_named_params.values())
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
        tensor_input_shapes: list[list[int | str] | None],
        tensor_input_dtypes: list[torch.onnx.TensorProtoDataType],
    ) -> tuple[list[list[int | str] | None], list[torch.onnx.TensorProtoDataType]]:
        param_count = len(trainable_named_params.values())
        tensor_output_shapes = [
            tensor_input_shapes[0],
        ] * param_count
        tensor_output_dtypes = [
            tensor_input_dtypes[0],
        ] * param_count

        return tensor_output_shapes, tensor_output_dtypes

    _PARAM_FUNCTION_INDEX[0] += 1

    return type(
        f"ParamTriggerFunction_{_PARAM_FUNCTION_INDEX[0]}",
        (torch.autograd.Function,),
        {
            "forward": forward,
            "backward": backward,
            "infer_shape": infer_shape,
        },
    )


def _create_param_retrieval_function(trainable_named_params: dict[str, torch.nn.parameter.Parameter]):
    """This function is used to create a weight retrieving function using trainable_named_params."""

    @staticmethod
    def forward(ctx, param_trigger, param_name):
        ctx.param_name = param_name
        ctx.dtype = param_trigger.dtype
        ctx.device = param_trigger.device
        ctx.shape = param_trigger.shape
        return trainable_named_params[param_name]

    @staticmethod
    def backward(ctx, *grad_outputs):
        trainable_named_params[ctx.param_name].backward(grad_outputs[0])
        return torch.zeros(ctx.shape, device=ctx.device, dtype=ctx.dtype), None

    @staticmethod
    def infer_shape(
        node: NodeProto,
        tensor_input_shapes: list[list[int | str] | None],
        tensor_input_dtypes: list[torch.onnx.TensorProtoDataType],
    ) -> tuple[list[list[int | str] | None], list[torch.onnx.TensorProtoDataType]]:
        input_pointer_scalars_attr_name = "input_pointer_scalars"
        found = [attr for attr in node.attribute if attr.name == input_pointer_scalars_attr_name]

        assert len(found) == 1
        input_pointer_scalars = found[0].ints

        # Restore the nn.Module from the pointer.
        param_name = ctypes.cast(input_pointer_scalars[0], ctypes.py_object).value

        tensor_output_shapes = [
            list(trainable_named_params[param_name].shape),
        ]
        tensor_output_dtypes = [
            int(pytorch_type_to_onnx_dtype(trainable_named_params[param_name].dtype)),
        ]

        return tensor_output_shapes, tensor_output_dtypes

    _PARAM_FUNCTION_INDEX[0] += 1
    return type(
        f"ParamRetrievalFunction_{_PARAM_FUNCTION_INDEX[0]}",
        (torch.autograd.Function,),
        {
            "forward": forward,
            "backward": backward,
            "infer_shape": infer_shape,
        },
    )
