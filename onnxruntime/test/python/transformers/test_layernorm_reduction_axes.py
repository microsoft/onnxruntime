# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import importlib.util
import sys
from pathlib import Path

import numpy as np
import onnx
import pytest
from onnx import TensorProto, helper, numpy_helper
from onnx.reference import ReferenceEvaluator

import onnxruntime as ort

# Prefer the checkout under test, without requiring the unrelated PyTorch export utilities.
source = Path(__file__).resolve().parents[3] / "python/tools/transformers"
if source.is_dir():
    sys.path.insert(0, str(source))

if importlib.util.find_spec("fusion_layernorm") is not None:
    from fusion_layernorm import FusionLayerNormalization
    from onnx_model import OnnxModel
else:
    from onnxruntime.transformers.fusion_layernorm import FusionLayerNormalization
    from onnxruntime.transformers.onnx_model import OnnxModel


def make_model(
    mean_axes,
    variance_axes,
    *,
    opset=17,
    shape=(2, 3, 4),
    keep_mean=1,
    keep_variance=1,
    axes_input="initializer",
    noop=0,
    symbolic=False,
    internal_input=False,
):
    scale = np.linspace(0.5, 1.5, shape[-1], dtype=np.float32)
    bias = np.linspace(-0.25, 0.25, shape[-1], dtype=np.float32)
    initializers = [
        numpy_helper.from_array(np.array(2, np.float32), "power"),
        numpy_helper.from_array(np.array(1e-5, np.float32), "epsilon"),
        numpy_helper.from_array(scale, "scale"),
        numpy_helper.from_array(bias, "bias"),
    ]
    model_shape = [f"dim_{i}" for i in range(len(shape))] if symbolic else shape
    inputs = [helper.make_tensor_value_info("X", TensorProto.FLOAT, model_shape)]
    nodes = []

    def reduce_mean(input_name, output_name, axes, keepdims):
        attributes = {} if keepdims is None else {"keepdims": keepdims}
        node_inputs = [input_name]
        if opset < 18:
            if axes:
                attributes["axes"] = axes
        else:
            attributes["noop_with_empty_axes"] = noop
            if axes is not None:
                axes_name = output_name + "_axes"
                node_inputs.append(axes_name)
                tensor = numpy_helper.from_array(np.asarray(axes, dtype=np.int64), axes_name)
                if axes_input == "constant":
                    nodes.append(helper.make_node("Constant", [], [axes_name], value=tensor))
                elif axes_input in ("initializer", "overridable"):
                    initializers.append(tensor)
                if axes_input in ("dynamic", "overridable"):
                    inputs.append(helper.make_tensor_value_info(axes_name, TensorProto.INT64, [len(axes)]))
        result = helper.make_node("ReduceMean", node_inputs, [output_name], **attributes)
        if opset < 18 and axes == []:
            result.attribute.append(helper.make_attribute("axes", [], attr_type=onnx.AttributeProto.INTS))
        return result

    nodes.append(reduce_mean("X", "mean", mean_axes, keep_mean))
    nodes.extend(
        [
            helper.make_node("Sub", ["X", "mean"], ["centered"]),
            helper.make_node("Pow", ["centered", "power"], ["squared"]),
        ]
    )
    nodes.append(reduce_mean("squared", "variance", variance_axes, keep_variance))
    nodes.extend(
        [
            helper.make_node("Add", ["variance", "epsilon"], ["stabilized"]),
            helper.make_node("Sqrt", ["stabilized"], ["stddev"]),
            helper.make_node("Div", ["centered", "stddev"], ["normalized"]),
            helper.make_node("Mul", ["normalized", "scale"], ["scaled"]),
            helper.make_node("Add", ["scaled", "bias"], ["Y"]),
        ]
    )
    if internal_input:
        for node in nodes:
            for index, name in enumerate(node.input):
                if name == "X":
                    node.input[index] = "internal"
        nodes.insert(0, helper.make_node("Identity", ["X"], ["internal"]))
    graph = helper.make_graph(
        nodes,
        "normalization_axes",
        inputs,
        [helper.make_tensor_value_info("Y", TensorProto.FLOAT, model_shape)],
        initializers,
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", opset)], ir_version=8)
    onnx.checker.check_model(model)
    return model


def run_native(model, feeds):
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    options.log_severity_level = 3
    session = ort.InferenceSession(model.SerializeToString(), options, providers=["CPUExecutionProvider"])
    return session.run(None, feeds)[0]


def check_fusion(mean_axes, variance_axes, *, expected_fused, feed_axes=None, force=False, **kwargs):
    model = make_model(mean_axes, variance_axes, **kwargs)
    shape = kwargs.get("shape", (2, 3, 4))
    x = (np.arange(np.prod(shape), dtype=np.float32).reshape(shape) ** 2 + 3) / 11
    feeds = {"X": x}
    if kwargs.get("axes_input") in ("dynamic", "overridable"):
        if feed_axes is not None:
            mean_axes, variance_axes = feed_axes
        feeds.update(
            mean_axes=np.asarray(mean_axes, dtype=np.int64), variance_axes=np.asarray(variance_axes, dtype=np.int64)
        )

    def mean(value, axes, keepdims):
        if axes is not None and len(axes) == 0 and kwargs.get("noop", 0):
            return value
        axis = tuple(axes) if axes else None
        return np.mean(value, axis=axis, keepdims=keepdims is None or bool(keepdims))

    centered = x - mean(x, mean_axes, kwargs.get("keep_mean", 1))
    variance = mean(centered**2, variance_axes, kwargs.get("keep_variance", 1))
    expected = centered / np.sqrt(variance + np.float32(1e-5))
    expected = expected * np.linspace(0.5, 1.5, shape[-1], dtype=np.float32)
    expected += np.linspace(-0.25, 0.25, shape[-1], dtype=np.float32)
    np.testing.assert_allclose(ReferenceEvaluator(model).run(None, feeds)[0], expected, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(run_native(model, feeds), expected, rtol=1e-5, atol=1e-5)

    wrapper = OnnxModel(model)
    FusionLayerNormalization(wrapper, force=force).apply()
    fused = len(wrapper.get_nodes_by_op_type("LayerNormalization"))
    actual = ReferenceEvaluator(wrapper.model).run(None, feeds)[0]
    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(run_native(wrapper.model, feeds), expected, rtol=1e-5, atol=1e-5)
    assert fused == expected_fused
    onnx.checker.check_model(wrapper.model)


@pytest.mark.parametrize("opset", [17, 18])
@pytest.mark.parametrize("mean_axes,variance_axes", [([-1], [-1]), ([2], [2]), ([-1], [2]), ([2], [-1])])
def test_last_dimension_fuses_and_preserves_values(opset, mean_axes, variance_axes):
    check_fusion(mean_axes, variance_axes, expected_fused=1, opset=opset)


@pytest.mark.parametrize("opset", [17, 18])
@pytest.mark.parametrize(
    "mean_axes,variance_axes", [([1], [1]), ([-1], [1]), ([1], [-1]), ([-2, -1], [-2, -1]), (None, None), ([], [])]
)
def test_other_reduction_axes_are_preserved(opset, mean_axes, variance_axes):
    check_fusion(mean_axes, variance_axes, expected_fused=0, opset=opset)


@pytest.mark.parametrize("opset", [17, 18])
@pytest.mark.parametrize("keep_mean,keep_variance", [(0, 1), (1, 0), (0, 0)])
def test_dimension_dropping_reductions_are_preserved(opset, keep_mean, keep_variance):
    check_fusion(
        [-1], [-1], expected_fused=0, opset=opset, shape=(3, 3), keep_mean=keep_mean, keep_variance=keep_variance
    )


@pytest.mark.parametrize("opset", [17, 18])
def test_default_keepdims_still_fuses(opset):
    check_fusion([-1], [-1], expected_fused=1, opset=opset, keep_mean=None, keep_variance=None)


@pytest.mark.parametrize("axes_input", ["initializer", "constant", "dynamic", "overridable"])
def test_axes_inputs_respect_mutability(axes_input):
    is_mutable = axes_input in ("dynamic", "overridable")
    check_fusion(
        [-1],
        [-1],
        expected_fused=int(not is_mutable),
        opset=18,
        axes_input=axes_input,
        feed_axes=([1], [1]) if is_mutable else None,
    )


def test_empty_axes_noop_is_preserved():
    check_fusion([], [], expected_fused=0, opset=18, noop=1)


def test_symbolic_dimensions_with_known_rank_still_fuse():
    check_fusion([2], [-1], expected_fused=1, opset=18, symbolic=True)


@pytest.mark.parametrize("axes,expected_fused", [([-1], 1), ([2], 0)])
def test_unknown_intermediate_rank_is_conservative(axes, expected_fused):
    check_fusion(axes, axes, expected_fused=expected_fused, opset=18, internal_input=True)


def test_force_does_not_bypass_reduction_semantics():
    check_fusion([1], [1], expected_fused=0, force=True)
