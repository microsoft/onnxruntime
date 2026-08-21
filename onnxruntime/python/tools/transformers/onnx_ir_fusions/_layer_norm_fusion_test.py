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
from onnx import TensorProto
from onnx_ir_fusions import layer_norm_fusion_rules
from onnx_ir_fusions._testing import op_counts, to_ir
from onnxscript.rewriter import rewrite


def _build_layer_norm_model(with_bias: bool = True, epsilon: float = 1e-5) -> onnx.ModelProto:
    hidden = 4
    inits = [
        helper.make_tensor("axes1", TensorProto.INT64, [1], [-1]),
        helper.make_tensor("axes2", TensorProto.INT64, [1], [-1]),
        helper.make_tensor("exponent", TensorProto.FLOAT, [], [2.0]),
        helper.make_tensor("epsilon", TensorProto.FLOAT, [], [epsilon]),
        helper.make_tensor("weight", TensorProto.FLOAT, [hidden], np.ones(hidden, np.float32)),
    ]
    nodes = [
        helper.make_node("ReduceMean", ["x", "axes1"], ["mean"]),
        helper.make_node("Sub", ["x", "mean"], ["diff"]),
        helper.make_node("Pow", ["diff", "exponent"], ["sq"]),
        helper.make_node("ReduceMean", ["sq", "axes2"], ["var"]),
        helper.make_node("Add", ["var", "epsilon"], ["var_eps"]),
        helper.make_node("Sqrt", ["var_eps"], ["std"]),
        helper.make_node("Div", ["diff", "std"], ["normalized"]),
        helper.make_node("Mul", ["normalized", "weight"], ["scaled"]),
    ]
    if with_bias:
        inits.append(helper.make_tensor("bias", TensorProto.FLOAT, [hidden], np.zeros(hidden, np.float32)))
        nodes.append(helper.make_node("Add", ["scaled", "bias"], ["y"]))
    else:
        nodes.append(helper.make_node("Identity", ["scaled"], ["y"]))

    graph = helper.make_graph(
        nodes,
        "layernorm",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [2, 3, hidden])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [2, 3, hidden])],
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


if __name__ == "__main__":
    unittest.main()
