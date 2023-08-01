# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import unittest
from copy import deepcopy
from typing import Optional, Sequence, Tuple

import numpy as np
from onnx import ModelProto, NodeProto, TensorProto, ValueInfoProto, checker, helper

import onnxruntime as ort


def make_vi_like(vi: ValueInfoProto, name: str) -> ValueInfoProto:
    """Makes a copy of `vi` with a new name."""
    new_vi = deepcopy(vi)
    new_vi.name = name
    return new_vi


def make_optional_tensor_value_info(name: str, elem_type: int, shape: Sequence[int]) -> ValueInfoProto:
    """Makes a `ValueInfoProto` with optional type."""
    tensor_type_proto = helper.make_tensor_type_proto(
        elem_type=elem_type,
        shape=shape,
    )
    opt_type_proto = helper.make_optional_type_proto(tensor_type_proto)

    vi = helper.make_tensor_value_info(name, elem_type, shape)
    vi.type.CopyFrom(opt_type_proto)
    return vi


def make_optional_vi(vi: ValueInfoProto, name: Optional[str] = None) -> ValueInfoProto:
    """Makes a copy of `vi` with optional type."""
    name = name or vi.name + ".opt"
    vi_type = vi.type.tensor_type
    vi_shape = [d.dim_param if d.dim_param else d.dim_value for d in vi_type.shape.dim]
    opt_vi = make_optional_tensor_value_info(name, vi_type.elem_type, vi_shape)
    return opt_vi


def make_const(vi: ValueInfoProto, name: str, value: int = 0) -> Tuple[ValueInfoProto, NodeProto, TensorProto]:
    """Creates a constant 1D tensor from `vi`."""
    const_vi = make_vi_like(vi, name)
    const_shape = [d.dim_value for d in vi.type.tensor_type.shape.dim]
    const_shape_tensor = helper.make_tensor(f"{name}.shape", TensorProto.INT64, [len(const_shape)], const_shape)
    const_fill = helper.make_tensor(f"{name}.const.value", const_vi.type.tensor_type.elem_type, [1], [value])
    const_node = helper.make_node(
        "ConstantOfShape",
        inputs=[const_shape_tensor.name],
        outputs=[const_vi.name],
        name=f"ConstantOfShape.{name}",
        value=const_fill,
    )
    return const_vi, const_node, const_shape_tensor


# This is a three-layer nested control flow ops model.
# The innermost subgraphs have the outer scope values that are the inputs, x2 and x3, of the top-level graph.
def make_opt_nested_greater_or_equal() -> ModelProto:
    """
    Creates a nested graph with (`optional(x1)`, `x2`, x3`) tensor inputs.

    `x3` is similar to an optional input with default value of -1.
    """
    # Inputs/outputs
    x1_vi = helper.make_tensor_value_info("x1", TensorProto.FLOAT, [1, 2])
    x2_vi = helper.make_tensor_value_info("x2", TensorProto.FLOAT, [1, 2])
    x3_vi = helper.make_tensor_value_info("x3", TensorProto.FLOAT, [1])
    y_vi = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 2])
    opt_x1_vi = make_optional_vi(x1_vi, name="x1.opt")

    # Add `x1` and `x2` subgraph
    y1_vi = make_vi_like(y_vi, "x1.add.x2")
    input_get_elem_node = helper.make_node(
        "OptionalGetElement",
        inputs=[opt_x1_vi.name],
        outputs=[x1_vi.name],
        name="OptionalGetElement.Input",
    )
    add_node = helper.make_node("Add", inputs=[x1_vi.name, x2_vi.name], outputs=[y1_vi.name], name="Add_Op")
    add_x1_x2_subgraph = helper.make_graph(
        [input_get_elem_node, add_node],
        name="add-x1-x2-subgraph",
        inputs=[],
        outputs=[y1_vi],
        value_info=[opt_x1_vi, x1_vi, x2_vi],
    )

    # Add `x2` and const subgraph
    y2_vi = make_vi_like(y_vi, "x2.add.const")
    const_vi, const_node, const_shape_tensor = make_const(x1_vi, "x1.const", value=1)
    add_const_node = helper.make_node(
        "Add",
        inputs=[const_vi.name, x2_vi.name],
        outputs=[y2_vi.name],
        name="Add_Const",
    )
    add_x2_const_subgraph = helper.make_graph(
        [const_node, add_const_node],
        name="add-x2-const-subgraph",
        inputs=[],
        outputs=[y2_vi],
        value_info=[const_vi, x2_vi],
        initializer=[const_shape_tensor],
    )

    # Add `x3` and const subgraph
    add_const_out_vi = make_vi_like(x3_vi, "out.1")
    y3_vi = make_vi_like(y_vi, "x3.add.const")
    const_vi, const_node, const_shape_tensor = make_const(x3_vi, "x3.const", value=2)
    add_const_node = helper.make_node(
        "Add",
        inputs=[const_vi.name, x3_vi.name],
        outputs=[add_const_out_vi.name],
        name="Add_Const.1",
    )
    expand_shape = helper.make_tensor(f"{add_const_out_vi}.shape", TensorProto.INT64, [2], [1, 2])
    expand_node = helper.make_node(
        "Expand",
        inputs=[add_const_out_vi.name, expand_shape.name],
        outputs=[y3_vi.name],
        name="Expand.out",
    )
    add_x3_const_subgraph = helper.make_graph(
        [const_node, add_const_node, expand_node],
        name="add-x3-const-subgraph",
        inputs=[],
        outputs=[y3_vi],
        value_info=[x3_vi, const_vi, add_const_out_vi],
        initializer=[const_shape_tensor, expand_shape],
    )

    # Subgraph flow based on `x3` value
    y3_if_vi = make_vi_like(y_vi, "x3.if.out")
    x3_eq_vi, x3_const_node, x3_const_shape_tensor = make_const(x3_vi, "x3.equal", value=0)
    x3_ge_vi = helper.make_tensor_value_info(
        "x3_ge",
        TensorProto.BOOL,
        shape=[1],
    )
    x3_ge_node = helper.make_node(
        "GreaterOrEqual",
        inputs=[x3_vi.name, x3_eq_vi.name],
        outputs=[x3_ge_vi.name],
        name="GreaterOrEqual.Target",
    )
    x3_has_elem_vi = helper.make_tensor_value_info(
        "x3_has_elem",
        TensorProto.BOOL,
        shape=[],  # scalar
    )
    x3_has_elem_node = helper.make_node(
        "Squeeze",
        inputs=[x3_ge_vi.name],
        outputs=[x3_has_elem_vi.name],
        name="Squeeze.x3",
    )
    if_input_node = helper.make_node(
        "If",
        inputs=[x3_has_elem_vi.name],  # condition
        outputs=[y3_if_vi.name],
        name="If.OptionalHasElement.x3",
        then_branch=add_x3_const_subgraph,
        else_branch=add_x2_const_subgraph,
    )
    x3_subgraph = helper.make_graph(
        [x3_const_node, x3_ge_node, x3_has_elem_node, if_input_node],
        name="x3-subgraph",
        inputs=[],
        outputs=[y3_if_vi],
        value_info=[x3_vi, x3_eq_vi, x3_ge_vi, x3_has_elem_vi],
        initializer=[x3_const_shape_tensor],
    )

    # Construct main graph
    x1_has_elem_vi = helper.make_tensor_value_info(
        "x1_has_elem",
        TensorProto.BOOL,
        shape=[],  # scalar
    )
    x1_has_elem_node = helper.make_node(
        "OptionalHasElement",
        inputs=[opt_x1_vi.name],
        outputs=[x1_has_elem_vi.name],
        name="OptionalHasElement.x1",
    )
    if_input_node = helper.make_node(
        "If",
        inputs=[x1_has_elem_vi.name],  # condition
        outputs=[y_vi.name],
        name="If.OptionalHasElement.x1",
        then_branch=add_x1_x2_subgraph,
        else_branch=x3_subgraph,
    )
    graph = helper.make_graph(
        [x1_has_elem_node, if_input_node],
        "opt-graph",
        [opt_x1_vi, x2_vi, x3_vi],
        [y_vi],
        value_info=[x1_has_elem_vi],
    )

    m = helper.make_model(
        graph,
        opset_imports=[
            helper.make_opsetid("", 15),
        ],
    )

    checker.check_model(m, full_check=True)

    return m


def test_nested_optional_greater_or_equal(use_trt: bool = False) -> None:
    m = make_opt_nested_greater_or_equal()

    providers = ["CUDAExecutionProvider"]
    if use_trt:
        providers.insert(0, "TensorrtExecutionProvider")
    session = ort.InferenceSession(
        m.SerializeToString(),
        providers=providers,
    )

    x1_name, x2_name, x3_name = (i.name for i in m.graph.input)
    session.run(
        [m.graph.output[0].name],
        {
            x1_name: None,
            x2_name: np.ones((1, 2), dtype=np.float32),
            x3_name: np.array([-1], dtype=np.float32),
        },
    )

    return


# ORT has a similar unit test Test3LayerNestedSubgraph where this 3-layer nested graph consumes the same initializer in different subgraphs.
# However, this unit test is slightly different. This is also a 3-layer nested graph but consumes the outer scope values (which are the inputs
# of the top-level graph) in different subgraphs.
class TestNestedControlFlowOpsGraph(unittest.TestCase):
    # We currently only test CUDA/TRT EP due to users only raise this issue when using CUDA/TRT EP.
    @unittest.skipIf(
        "TensorrtExecutionProvider" not in ort.get_available_providers()
        and "CUDAExecutionProvider" not in ort.get_available_providers(),
        reason="Test CUDA/TRT EP only",
    )
    def test_3_level_control_flow_ops_graph(self):
        if "CUDAExecutionProvider" in ort.get_available_providers():
            test_nested_optional_greater_or_equal(use_trt=False)
        if "TensorrtExecutionProvider" in ort.get_available_providers():
            test_nested_optional_greater_or_equal(use_trt=True)


if __name__ == "__main__":
    unittest.main(module=__name__, buffer=True)
