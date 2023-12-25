# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

from __future__ import annotations
import inspect

import torch

from onnxruntime.capi._pybind_state import (
    register_miscellaneous_const_input,
    register_torch_autograd_function,
)

from ._utils import get_fully_qualified_class_name
from ._custom_autograd_function_exporter import register_custom_function_schema_supplementary


import onnx

PYTHON_OP_DOMAIN = "com.microsoft"
PYTHON_OP_TYPE = "PythonOp"

PYTHON_OP_ATTRIBUTE_FUNC_NAME = "func_name"
PYTHON_OP_ATTRIBUTE_SAFE_RUN_MODE = "safe_run_mode"
PYTHON_OP_ATTRIBUTE_TRAINING_MODE = "training_mode"


def set_safe_run_mode(model: onnx.ModelProto, allowed_unsafe_run_python_op_names: list[str]) -> onnx.ModelProto:
    # Update safe_run_mode attribute for PythonOp.
    for node in model.graph.node:
        if node.domain == PYTHON_OP_DOMAIN and node.op_type == PYTHON_OP_TYPE:
            func_name = None
            safe_run_mode_attr = None
            for attr in node.attribute:
                if attr.name == PYTHON_OP_ATTRIBUTE_FUNC_NAME:
                    func_name = attr.s.decode("utf-8") if isinstance(attr.s, bytes) else attr.s
                if attr.name == PYTHON_OP_ATTRIBUTE_SAFE_RUN_MODE:
                    safe_run_mode_attr = attr

            if func_name in allowed_unsafe_run_python_op_names:
                if safe_run_mode_attr:
                    node.attribute.remove(safe_run_mode_attr)
                node.attribute.append(onnx.helper.make_attribute(PYTHON_OP_ATTRIBUTE_SAFE_RUN_MODE, 0))

    return model

_PYTHON_OP_INCRE_INDEX = [0]

def make_pythonop_node(
    name_prefix: str,
    inputs: list[onnx.ValueInfoProto | int | bool | float | tuple[int, ...] | tuple[bool, ...] | tuple[float, ...] | object ],
    outputs: list[onnx.ValueInfoProto],
    func_class: torch.autograd.Function,
    training_mode:int,
    safe_run_mode:int,
    ) -> onnx.NodeProto:

    assert issubclass(func_class, torch.autograd.Function), "func_class must be a subclass of torch.autograd.Function."

    assert len(inputs) > 0, f"inputs must not be empty for function {func_class}."
    assert len(outputs) > 0, f"outputs must not be empty for function {func_class}."

    all_input_parameters: list[inspect.Parameter] = list(inspect.signature(func_class.forward).parameters.values())

    # Remove the first parameter (ctx) from inspected parameter list.
    assert len(inputs) == len(all_input_parameters) - 1, f"The number of inputs ({len(inputs)}) must match the number of parameters ({len(all_input_parameters) - 1}) of the forward function."

    func_full_qual_name = get_fully_qualified_class_name(func_class)

    input_tensor_types = []
    input_tensor_ranks = []

    input_bool_scalars = []
    input_bool_scalar_positions = []

    input_int_scalars = []
    input_int_scalar_positions = []

    input_float_scalars = []
    input_float_scalar_positions = []

    input_bool_tuples = []
    input_bool_tuple_positions = []
    input_bool_tuple_begins = []

    input_int_tuples = []
    input_int_tuple_positions = []
    input_int_tuple_begins = []

    input_float_tuples = []
    input_float_tuple_positions = []
    input_float_tuple_begins = []

    input_pointer_scalars = []
    input_pointer_scalar_positions = []

    tensor_args = []
    debug_comment = ""
    cconv = ""
    # Encode inputs to torch.autograd.Function.
    for i, arg in enumerate(inputs):
        if isinstance(arg, onnx.ValueInfoProto):
            # Got a tensor variable.
            tensor_args.append(arg.name)
            input_tensor_types.append(arg.type.tensor_type.elem_type)
            input_tensor_ranks.append(len(arg.type.tensor_type.shape.dim))
            cconv += 'd'
            continue

        cconv += 'c'

        # Got a non-tensor variable.
        if isinstance(arg, float):
            # A float.
            input_float_scalar_positions.append(i)
            input_float_scalars.append(arg)
            continue
        # bool check MUST be before int check since bool is a subclass of int
        elif isinstance(arg, bool):
            # A bool.
            input_bool_scalar_positions.append(i)
            input_bool_scalars.append(int(arg))
            continue
        elif isinstance(arg, int):
            # A int.
            input_int_scalar_positions.append(i)
            input_int_scalars.append(arg)
            continue

        is_bool_tuple = False
        is_int_tuple = False
        is_float_tuple = False
        if isinstance(arg, tuple) and len(arg) > 0:
            # bool check MUST be before int check since bool is a subclass of int.
            is_bool_tuple = all(isinstance(ele, bool) for ele in arg)
            is_int_tuple = not is_bool_tuple and all(isinstance(ele, int) for ele in arg)
            is_float_tuple = not is_bool_tuple and not is_int_tuple and all(isinstance(ele, float) for ele in arg)

        # Only support tuple of bool, int or float, for other types, handle it as a pointer.
        if is_bool_tuple:
            # A tuple of bool.
            input_bool_tuple_positions.append(i)
            input_bool_tuple_begins.append(len(input_bool_tuples))
            input_bool_tuples.extend([int(ele) for ele in arg])
            continue
        elif is_int_tuple:
            # A tuple of ints.
            input_int_tuple_positions.append(i)
            input_int_tuple_begins.append(len(input_int_tuples))
            input_int_tuples.extend(list(arg))
            continue
        elif is_float_tuple:
            # A tuple of floats.
            input_float_tuple_positions.append(i)
            input_float_tuple_begins.append(len(input_float_tuples))
            input_float_tuples.extend(list(arg))
            continue

        from onnxruntime.training.utils.hooks._statistics_subscriber import _InspectActivation

        is_inspect_activation = func_full_qual_name == get_fully_qualified_class_name(_InspectActivation)
        if is_inspect_activation and isinstance(arg, str):
            # _InspectActivation is a special case where the first argument is a string
            # that is used to determine the activation name to be inspected.
            debug_comment += arg

        # All other inputs are accessed via "pointers".
        input_pointer_scalar_positions.append(i)
        input_pointer_scalars.append(id(arg))

        # For pointer (for example, ProcessGroup passed to PythonOp) needed for PythonOp execution,
        # we append it into a global store to hold a reference (in case it is released after module exported).
        register_miscellaneous_const_input(arg)

    output_tensor_types = []
    output_tensor_ranks = []
    for arg in outputs:
        output_tensor_types.append(arg.type.tensor_type.elem_type)
        output_tensor_ranks.append(len(arg.type.tensor_type.shape.dim))


    attrs = {
        "func_name": func_full_qual_name,
        "input_convention": cconv,
        # "outputs": len(outputs),
        "input_tensor_types": input_tensor_types,
        "input_tensor_ranks": input_tensor_ranks,
        "output_tensor_types": output_tensor_types,
        "output_tensor_ranks": output_tensor_ranks,
        "training_mode": training_mode,
        "safe_run_mode": safe_run_mode,
        "comment": debug_comment,
    }

    if len(input_bool_scalars) > 0:
        attrs["input_bool_scalars"] = input_bool_scalars
        attrs["input_bool_scalar_positions"] = input_bool_scalar_positions
    if len(input_int_scalars) > 0:
        attrs["input_int_scalars"] = input_int_scalars
        attrs["input_int_scalar_positions"] = input_int_scalar_positions
    if len(input_float_scalars) > 0:
        attrs["input_float_scalars"] = input_float_scalars
        attrs["input_float_scalar_positions"] = input_float_scalar_positions
    if len(input_bool_tuples) > 0:
        attrs["input_bool_tuples"] = input_bool_tuples
        attrs["input_bool_tuple_positions"] = input_bool_tuple_positions
        attrs["input_bool_tuple_begins"] = input_bool_tuple_begins
    if len(input_int_tuples) > 0:
        attrs["input_int_tuples"] = input_int_tuples
        attrs["input_int_tuple_positions"] = input_int_tuple_positions
        attrs["input_int_tuple_begins"] = input_int_tuple_begins
    if len(input_float_tuples) > 0:
        attrs["input_float_tuples"] = input_float_tuples
        attrs["input_float_tuple_positions"] = input_float_tuple_positions
        attrs["input_float_tuple_begins"] = input_float_tuple_begins
    if len(input_pointer_scalars) > 0:
        attrs["input_pointer_scalars"] = input_pointer_scalars
        attrs["input_pointer_scalar_positions"] = input_pointer_scalar_positions


    # Register function with class names.
    register_torch_autograd_function(func_full_qual_name, func_class)

    register_custom_function_schema_supplementary(func_class)

    _PYTHON_OP_INCRE_INDEX[0] += 1
    node_name = f"{name_prefix}_{_PYTHON_OP_INCRE_INDEX[0]}"

    node = onnx.helper.make_node(
        PYTHON_OP_TYPE,
        tensor_args,
        [f"{node_name}_ctx", *[output.name for output in outputs]],
        node_name,  # node name
        "",
        PYTHON_OP_DOMAIN,
        **attrs,
    )

    return node
