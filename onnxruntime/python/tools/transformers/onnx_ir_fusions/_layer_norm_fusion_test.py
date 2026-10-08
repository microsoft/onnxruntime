# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Unit tests for the LayerNormalization onnxscript rewrite rule."""

from __future__ import annotations

import unittest

import numpy as np
import onnx
import onnx.helper as helper
from onnx_ir_fusions import layer_norm_fusion_rules
from onnx_ir_fusions._testing import op_counts, to_ir
from onnxscript.rewriter import rewrite


def _make_constant(name: str, values: list[float]) -> onnx.TensorProto:
    dimensions = [] if len(values) == 1 else [len(values)]
    return helper.make_tensor(name, onnx.TensorProto.FLOAT, dimensions, values)


def _build_layer_norm_model(
    with_bias: bool = True,
    epsilon: float = 1e-5,
    keepdims: int = 1,
    exponent_values: list[float] | None = None,
    epsilon_values: list[float] | None = None,
) -> onnx.ModelProto:
    hidden = 4
    exponent_values = [2.0] if exponent_values is None else exponent_values
    epsilon_values = [epsilon] if epsilon_values is None else epsilon_values
    inits = [
        helper.make_tensor("axes1", onnx.TensorProto.INT64, [1], [-1]),
        helper.make_tensor("axes2", onnx.TensorProto.INT64, [1], [-1]),
        _make_constant("exponent", exponent_values),
        _make_constant("epsilon", epsilon_values),
        helper.make_tensor("weight", onnx.TensorProto.FLOAT, [hidden], np.ones(hidden, np.float32)),
    ]
    nodes = [
        helper.make_node("ReduceMean", ["x", "axes1"], ["mean"], keepdims=keepdims),
        helper.make_node("Sub", ["x", "mean"], ["diff"]),
        helper.make_node("Pow", ["diff", "exponent"], ["sq"]),
        helper.make_node("ReduceMean", ["sq", "axes2"], ["var"], keepdims=keepdims),
        helper.make_node("Add", ["var", "epsilon"], ["var_eps"]),
        helper.make_node("Sqrt", ["var_eps"], ["std"]),
        helper.make_node("Div", ["diff", "std"], ["normalized"]),
        helper.make_node("Mul", ["normalized", "weight"], ["scaled"]),
    ]
    if with_bias:
        inits.append(helper.make_tensor("bias", onnx.TensorProto.FLOAT, [hidden], np.zeros(hidden, np.float32)))
        nodes.append(helper.make_node("Add", ["scaled", "bias"], ["y"]))
    else:
        nodes.append(helper.make_node("Identity", ["scaled"], ["y"]))

    graph = helper.make_graph(
        nodes,
        "layernorm",
        [helper.make_tensor_value_info("x", onnx.TensorProto.FLOAT, [2, 3, hidden])],
        [helper.make_tensor_value_info("y", onnx.TensorProto.FLOAT, [2, 3, hidden])],
        initializer=inits,
    )
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 20)])


class TestLayerNormFusion(unittest.TestCase):
    def test_fuses_layer_norm_with_bias(self):
        model = to_ir(_build_layer_norm_model(with_bias=True))
        rewrite(model, pattern_rewrite_rules=layer_norm_fusion_rules())

        counts = op_counts(model)
        self.assertEqual(counts.get("LayerNormalization", 0), 1)
        self.assertEqual(counts.get("ReduceMean", 0), 0)
        self.assertEqual(counts.get("Sqrt", 0), 0)
        node = next(n for n in model.graph if n.op_type == "LayerNormalization")
        self.assertTrue(np.isclose(node.attributes.get_float("epsilon"), 1e-5))

    def test_fuses_layer_norm_without_bias(self):
        model = to_ir(_build_layer_norm_model(with_bias=False))
        rewrite(model, pattern_rewrite_rules=layer_norm_fusion_rules())

        counts = op_counts(model)
        self.assertEqual(counts.get("LayerNormalization", 0), 1)
        self.assertEqual(counts.get("ReduceMean", 0), 0)

    def test_does_not_fuse_without_keepdims(self):
        model = to_ir(_build_layer_norm_model(keepdims=0))
        rewrite(model, pattern_rewrite_rules=layer_norm_fusion_rules())

        counts = op_counts(model)
        self.assertEqual(counts.get("LayerNormalization", 0), 0)
        self.assertEqual(counts.get("ReduceMean", 0), 2)

    def test_does_not_fuse_vector_exponent(self):
        model = to_ir(_build_layer_norm_model(exponent_values=[2.0] * 4))
        rewrite(model, pattern_rewrite_rules=layer_norm_fusion_rules())

        self.assertEqual(op_counts(model).get("LayerNormalization", 0), 0)

    def test_does_not_fuse_vector_epsilon(self):
        model = to_ir(_build_layer_norm_model(epsilon_values=[1e-5] * 4))
        rewrite(model, pattern_rewrite_rules=layer_norm_fusion_rules())

        self.assertEqual(op_counts(model).get("LayerNormalization", 0), 0)


if __name__ == "__main__":
    unittest.main()
