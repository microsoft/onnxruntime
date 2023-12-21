# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------


from contextlib import contextmanager
from typing import Dict, List, Optional, Tuple, Union
import ctypes
import torch
from onnx import ModelProto, NodeProto, TensorProto, ValueInfoProto, helper

from onnxruntime.capi._pybind_state import (
    register_torch_autograd_function, register_miscellaneous_const_input
)
from onnxruntime.training.utils import pytorch_type_to_onnx_dtype

from ._custom_autograd_function_exporter import register_custom_function_schema_supplementary
from ._utils import get_fully_qualified_class_name

MEM_EFFICIENT_GRAD_TRIGGER_NAME = "mem_efficient_pull_weight_trigger"
MEM_EFFICIENT_GRAD_TRIGGER_OUTPUT_DTYPE = TensorProto.FLOAT
MEM_EFFICIENT_GRAD_TRIGGER_OUTPUT_SHAPE = [1]


def post_processing_enable_mem_efficient_training(
    exported_model: ModelProto,
    named_params: Dict[str, torch.nn.parameter.Parameter],
) -> ModelProto:
    """This function is used to enable zero stage3 compatibility.

    Args:
        exported_model (ModelProto): The exported model.
        trainable_named_params (Optional[Dict[str, torch.nn.parameter.Parameter]]): The offload named parameters.
        all_param_names (List[str]): All parameter names.
    """
    trainable_named_params = {k: v for k, v in named_params if v.requires_grad}

    # Create weight retrieving function using trainable_named_params.
    func_full_qual_name = _create_param_trigger_function(trainable_named_params)

    param_retrieve_func_full_qual_name = _create_param_retrieval_function(trainable_named_params)

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
        trainable_named_params,
        func_full_qual_name,
        MEM_EFFICIENT_GRAD_TRIGGER_NAME,
        [_get_param_pull_trigger_name(pname) for pname in trainable_named_params],
        MEM_EFFICIENT_GRAD_TRIGGER_OUTPUT_DTYPE,
        MEM_EFFICIENT_GRAD_TRIGGER_OUTPUT_SHAPE,
    )



    # Connect weight consumers to use the full-sized parameter output of ORTZeROOffloadPreForwardFunction.
    graph_inputs_to_remove = []
    for graph_input in reversed(exported_model.graph.input):
        if graph_input.name not in trainable_named_params:
            continue


        graph_inputs_to_remove.append(graph_input)

        if graph_input.name not in consumer_map:
            continue

        # Create the param retrieval function for this parameter.
        param_retrieve_func_input_arg_name = _get_param_pull_trigger_name(graph_input.name)
        # def _create_param_retrieval_pythonop(
        #     param_name:str,
        #     func_full_qual_name: str,
        #     pull_weight_trigger_output_name: str,
        #     pull_weight_trigger_output_dtype: int,
        #     pull_weight_trigger_output_shape: List[int],
        #     output_names: List[str],
        #     param_output_dtype: int,
        #     param_output_shape: List[int],


        new_node = _create_param_retrieval_pythonop(graph_input.name, # param_name
                                                    param_retrieve_func_full_qual_name,
                                                    param_retrieve_func_input_arg_name,
                                                    MEM_EFFICIENT_GRAD_TRIGGER_OUTPUT_DTYPE,
                                                    MEM_EFFICIENT_GRAD_TRIGGER_OUTPUT_SHAPE,
                                                    [graph_input.name], # output use the same name as weight
                                                    int(pytorch_type_to_onnx_dtype(trainable_named_params[graph_input.name].dtype)),
                                                    list(trainable_named_params[graph_input.name].shape))

        exported_model.graph.node.insert(0, new_node)



    # Delete exported_model.graph.input
    for input_to_remove in graph_inputs_to_remove:
        exported_model.graph.input.remove(input_to_remove)

    # Re-order graph input to make sure the weight pull trigger is before all parameter inputs.
    # We assume the parameters are always at the end of the graph input list.
    offset = 0
    for graph_input in exported_model.graph.input:
        if graph_input.name in named_params:
            break
        offset += 1

    exported_model.graph.input.insert(offset, new_input)
    exported_model.graph.node.insert(0, weight_pull_node)

    # Update safe_run_mode attribute for PythonOp.
    _allowed_unsafe_run_python_op_names = [
        func_full_qual_name,
        param_retrieve_func_full_qual_name,
    ]

    for node in exported_model.graph.node:
        if node.op_type == "PythonOp":
            func_name = None
            safe_run_mode_attr = None
            for attr in node.attribute:
                if attr.name == "func_name":
                    func_name = attr.s.decode("utf-8") if isinstance(attr.s, bytes) else attr.s
                if attr.name == "safe_run_mode":
                    safe_run_mode_attr = attr

            if func_name in _allowed_unsafe_run_python_op_names:
                if safe_run_mode_attr:
                    node.attribute.remove(safe_run_mode_attr)
                node.attribute.append(helper.make_attribute("safe_run_mode", 0))

    return exported_model


def _create_param_trigger_function(
    trainable_named_params: Optional[Dict[str, torch.nn.parameter.Parameter]]
) -> str:
    """This function is used to create a weight retrieving function using trainable_named_params."""

    class ParamTriggerFunction(torch.autograd.Function):
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
            tensor_input_shapes: List[Optional[List[Union[int, str]]]],
            tensor_input_dtypes: List[torch.onnx.TensorProtoDataType],
        ) -> Tuple[List[Optional[List[Union[int, str]]]], List[torch.onnx.TensorProtoDataType]]:
            param_count = len(trainable_named_params.values())
            tensor_output_shapes = [
                tensor_input_shapes[0],
            ] * param_count
            tensor_output_dtypes = [
                tensor_input_dtypes[0],
            ] * param_count

            return tensor_output_shapes, tensor_output_dtypes

    func_full_qual_name = get_fully_qualified_class_name(ParamTriggerFunction)
    register_torch_autograd_function(func_full_qual_name, ParamTriggerFunction)
    register_custom_function_schema_supplementary(ParamTriggerFunction)

    return func_full_qual_name


def _create_param_retrieval_function(
    trainable_named_params: Dict[str, torch.nn.parameter.Parameter]
) -> str:
    """This function is used to create a weight retrieving function using trainable_named_params."""

    class ParamRetrievalFunction(torch.autograd.Function):
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
            tensor_input_shapes: List[Optional[List[Union[int, str]]]],
            tensor_input_dtypes: List[torch.onnx.TensorProtoDataType],
        ) -> Tuple[List[Optional[List[Union[int, str]]]], List[torch.onnx.TensorProtoDataType]]:
            input_pointer_scalars_attr_name = "input_pointer_scalars"
            found = [attr for attr in node.attribute if attr.name == input_pointer_scalars_attr_name]

            assert len(found) == 1
            input_pointer_scalars = found[0].ints

            # Restore the nn.Module from the pointer.
            param_name = ctypes.cast(input_pointer_scalars[0], ctypes.py_object).value

            tensor_output_shapes = [list(trainable_named_params[param_name].shape),]
            tensor_output_dtypes = [int(pytorch_type_to_onnx_dtype(trainable_named_params[param_name].dtype)),]

            return tensor_output_shapes, tensor_output_dtypes

    func_full_qual_name = get_fully_qualified_class_name(ParamRetrievalFunction)
    register_torch_autograd_function(func_full_qual_name, ParamRetrievalFunction)
    register_custom_function_schema_supplementary(ParamRetrievalFunction)

    return func_full_qual_name




def _create_weight_retrieval_pythonop(
    trainable_named_params: Optional[Dict[str, torch.nn.parameter.Parameter]],
    func_full_qual_name: str,
    input_name: str,
    output_names: List[str],
    pull_weight_trigger_output_dtype: int,
    pull_weight_trigger_output_shape: List[int],
) -> Tuple[ValueInfoProto, NodeProto]:
    """This function is used to create a weight retrieving PythonOp."""
    offload_param_count = 0 if trainable_named_params is None else len(trainable_named_params)
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



def _create_param_retrieval_pythonop(
    param_name:str,
    func_full_qual_name: str,
    pull_weight_trigger_output_name: str,
    pull_weight_trigger_output_dtype: int,
    pull_weight_trigger_output_shape: List[int],
    output_names: List[str],
    param_output_dtype: int,
    param_output_shape: List[int],
) -> NodeProto:
    """This function is used to create a weight retrieving PythonOp."""

    output_tensor_ranks = [
        len(param_output_shape),
    ]
    output_tensor_types = [
        param_output_dtype,
    ]



    node_attributes = {
        "comment": "",
        "input_convention": "dc",
        "input_tensor_ranks": [len(pull_weight_trigger_output_shape)],
        "input_tensor_types": [pull_weight_trigger_output_dtype],
        "input_pointer_scalars": [id(param_name)],
        "input_pointer_scalar_positions": [1],
        "output_tensor_ranks": output_tensor_ranks,
        "output_tensor_types": output_tensor_types,
        "training_mode": 1,
        "func_name": func_full_qual_name,
    }

    register_miscellaneous_const_input(param_name)



    weight_pull_node = helper.make_node(
        "PythonOp",
        [pull_weight_trigger_output_name],
        [f"param_retrieval_ctx_{param_name}", *output_names],
        f"param_retrieval_{param_name}",  # node name
        "PythonOp for weight retrieving.",
        "com.microsoft",
        **node_attributes,
    )

    return weight_pull_node
