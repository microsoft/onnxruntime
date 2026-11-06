# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import numpy as np
import onnx
import pytest
from onnx import TensorProto, helper, numpy_helper
from test_layernorm_reduction_axes import (
    FusionLayerNormalization,
    OnnxModel,
    ReferenceEvaluator,
    make_model,
    run_native,
)


def rename_graph(graph, prefix, shared=()):
    def rename(name):
        return name if name in shared else prefix + name

    for node in graph.node:
        for i, name in enumerate(node.input):
            node.input[i] = rename(name)
        for i, name in enumerate(node.output):
            node.output[i] = rename(name)
    for tensor in graph.initializer:
        tensor.name = rename(tensor.name)
    for value in graph.output:
        value.name = rename(value.name)
    del graph.input[:]
    graph.node.insert(0, helper.make_node("Identity", ["X"], [prefix + "X"]))
    graph.name = prefix + "scope"
    return graph


def set_value(graph, name, value, storage="initializer"):
    tensor = next(t for t in graph.initializer if t.name == name)
    replacement = numpy_helper.from_array(np.array(value, np.float32), name)
    tensor.CopyFrom(replacement)
    if storage == "constant":
        graph.initializer.remove(tensor)
        graph.node.insert(0, helper.make_node("Constant", [], [name], value=replacement))


def oracle(x, epsilon=1e-5, power=2):
    centered = x - np.mean(x, axis=-1, keepdims=True)
    variance = np.mean(np.power(centered, np.float32(power)), axis=-1, keepdims=True)
    return centered / np.sqrt(variance + np.float32(epsilon)) * np.linspace(
        0.5, 1.5, 4, dtype=np.float32
    ) + np.linspace(-0.25, 0.25, 4, dtype=np.float32)


def check(model, feeds, expected, *, expected_fused=0, force=False):
    onnx.checker.check_model(model)
    np.testing.assert_allclose(ReferenceEvaluator(model).run(None, feeds)[0], expected, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(run_native(model, feeds), expected, rtol=1e-5, atol=1e-5)
    wrapped = OnnxModel(model)
    FusionLayerNormalization(wrapped, force=force).apply()
    onnx.checker.check_model(wrapped.model)
    assert len(wrapped.get_nodes_by_op_type("LayerNormalization")) == expected_fused
    np.testing.assert_allclose(ReferenceEvaluator(wrapped.model).run(None, feeds)[0], expected, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(run_native(wrapped.model, feeds), expected, rtol=1e-5, atol=1e-5)


X = np.arange(24, dtype=np.float32).reshape(2, 3, 4) / 100


@pytest.mark.parametrize("parameter,value", [("epsilon", 9e-5), ("power", 4)])
@pytest.mark.parametrize("force", [True, False])
def test_overridable_scalar_is_not_frozen(parameter, value, force):
    model = make_model([-1], [-1])
    model.graph.input.append(helper.make_tensor_value_info(parameter, TensorProto.FLOAT, []))
    check(model, {"X": X, parameter: np.array(value, np.float32)}, oracle(X, **{parameter: value}), force=force)


@pytest.mark.parametrize("parameter,value", [("epsilon", 9e-5), ("power", 4)])
@pytest.mark.parametrize("storage", ["initializer", "constant"])
@pytest.mark.parametrize("condition", [True, False])
def test_sibling_constants_use_their_own_binding(parameter, value, storage, condition):
    then = rename_graph(make_model([-1], [-1]).graph, "then_", shared=[parameter])
    otherwise = rename_graph(make_model([-1], [-1]).graph, "else_", shared=[parameter])
    set_value(then, parameter, value, storage)
    default = 1e-5 if parameter == "epsilon" else 2
    set_value(otherwise, parameter, default, storage)
    graph = helper.make_graph(
        [helper.make_node("If", ["condition"], ["Y"], then_branch=then, else_branch=otherwise)],
        "constant_binding_scopes",
        [
            helper.make_tensor_value_info("X", TensorProto.FLOAT, [2, 3, 4]),
            helper.make_tensor_value_info("condition", TensorProto.BOOL, []),
        ],
        [helper.make_tensor_value_info("Y", TensorProto.FLOAT, [2, 3, 4])],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)], ir_version=8)
    check(model, {"X": X, "condition": np.array(condition)}, oracle(X, **{parameter: value if condition else default}))


@pytest.mark.parametrize("parameter", ["epsilon", "power"])
@pytest.mark.parametrize("storage", ["initializer", "constant"])
@pytest.mark.parametrize("tensor_shape", [(), (1,)])
def test_immutable_scalar_constants_still_fuse(parameter, storage, tensor_shape):
    model = make_model([-1], [-1])
    value = 1e-5 if parameter == "epsilon" else 2
    tensor = next(t for t in model.graph.initializer if t.name == parameter)
    replacement = numpy_helper.from_array(np.full(tensor_shape, value, np.float32), parameter)
    tensor.CopyFrom(replacement)
    if storage == "constant":
        model.graph.initializer.remove(tensor)
        model.graph.node.insert(0, helper.make_node("Constant", [], [parameter], value=replacement))
    check(model, {"X": X}, oracle(X), expected_fused=1)


@pytest.mark.parametrize("parameter", ["epsilon", "power"])
def test_single_element_constants_must_not_expand_rank(parameter):
    model = make_model([-1], [-1])
    value = 1e-5 if parameter == "epsilon" else 2
    tensor = next(t for t in model.graph.initializer if t.name == parameter)
    tensor.CopyFrom(numpy_helper.from_array(np.full((1, 1, 1, 1), value, np.float32), parameter))
    model.graph.output[0].CopyFrom(helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 2, 3, 4]))
    check(model, {"X": X}, oracle(X)[None, ...])


def test_vector_epsilon_is_preserved():
    model = make_model([-1], [-1])
    epsilon = np.array([1e-5, 2e-5, 5e-5, 9e-5], np.float32)
    tensor = next(t for t in model.graph.initializer if t.name == "epsilon")
    tensor.CopyFrom(numpy_helper.from_array(epsilon, "epsilon"))
    check(model, {"X": X}, oracle(X, epsilon=epsilon))


@pytest.mark.parametrize("parameter", ["epsilon", "power"])
@pytest.mark.parametrize("condition", [True, False])
def test_unambiguous_outer_scope_constant_still_fuses(parameter, condition):
    branches = [rename_graph(make_model([-1], [-1]).graph, prefix, shared=[parameter]) for prefix in ["then_", "else_"]]
    for branch in branches:
        branch.initializer.remove(next(t for t in branch.initializer if t.name == parameter))
    value = 1e-5 if parameter == "epsilon" else 2
    graph = helper.make_graph(
        [helper.make_node("If", ["condition"], ["Y"], then_branch=branches[0], else_branch=branches[1])],
        "outer_binding",
        [
            helper.make_tensor_value_info("X", TensorProto.FLOAT, [2, 3, 4]),
            helper.make_tensor_value_info("condition", TensorProto.BOOL, []),
        ],
        [helper.make_tensor_value_info("Y", TensorProto.FLOAT, [2, 3, 4])],
        [numpy_helper.from_array(np.array(value, np.float32), parameter)],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)], ir_version=8)
    check(model, {"X": X, "condition": np.array(condition)}, oracle(X), expected_fused=2)


def test_exponent_must_be_exactly_two():
    model = make_model([-1], [-1])
    power = np.nextafter(np.float32(2), np.float32(3))
    set_value(model.graph, "power", power)
    # Negative centered values raised to a noninteger power have a different domain.
    with np.errstate(invalid="ignore"):
        check(model, {"X": X}, oracle(X, power=power))


@pytest.mark.parametrize("parameter,value", [("epsilon", 9e-5), ("power", 4)])
def test_loop_scalar_input_is_not_a_sibling_constant(parameter, value):
    body = rename_graph(make_model([-1], [-1]).graph, "loop_", shared=[parameter])
    body.initializer.remove(next(t for t in body.initializer if t.name == parameter))
    body.input.extend(
        [
            helper.make_tensor_value_info("iteration", TensorProto.INT64, []),
            helper.make_tensor_value_info("body_condition", TensorProto.BOOL, []),
            helper.make_tensor_value_info(parameter, TensorProto.FLOAT, []),
            helper.make_tensor_value_info("carry_in", TensorProto.FLOAT, [2, 3, 4]),
        ]
    )
    body.node.extend(
        [
            helper.make_node("Identity", ["body_condition"], ["next_condition"]),
            helper.make_node("Identity", [parameter], ["next_scalar"]),
        ]
    )
    result = helper.make_tensor_value_info("loop_Y", TensorProto.FLOAT, [2, 3, 4])
    del body.output[:]
    body.output.extend(
        [
            helper.make_tensor_value_info("next_condition", TensorProto.BOOL, []),
            helper.make_tensor_value_info("next_scalar", TensorProto.FLOAT, []),
            result,
        ]
    )
    sibling = rename_graph(make_model([-1], [-1]).graph, "sibling_", shared=[parameter])
    other = rename_graph(make_model([-1], [-1]).graph, "other_", shared=[parameter])
    graph = helper.make_graph(
        [
            helper.make_node("If", ["condition"], ["unused"], then_branch=sibling, else_branch=other),
            helper.make_node(
                "Loop", ["trip_count", "condition", "initial_scalar", "X"], ["final_scalar", "Y"], body=body
            ),
        ],
        "loop_constant_binding",
        [helper.make_tensor_value_info("X", TensorProto.FLOAT, [2, 3, 4])],
        [helper.make_tensor_value_info("Y", TensorProto.FLOAT, [2, 3, 4])],
        [
            numpy_helper.from_array(np.array(1, np.int64), "trip_count"),
            numpy_helper.from_array(np.array(True), "condition"),
            numpy_helper.from_array(np.array(value, np.float32), "initial_scalar"),
        ],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)], ir_version=8)
    check(model, {"X": X}, oracle(X, **{parameter: value}))
