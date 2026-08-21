# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Unit tests for the SkipLayerNormalization onnxscript rewrite rule."""

from __future__ import annotations

import unittest

import numpy as np
import onnx
import onnx.helper as helper
from onnx import TensorProto
from onnx_ir_fusions import skip_layer_norm_rules
from onnx_ir_fusions._testing import op_counts, to_ir
from onnxscript.rewriter import rewrite


def _build_skip_layer_norm_model(with_bias: bool = True) -> onnx.ModelProto:
    hidden = 4
    inits = [helper.make_tensor("weight", TensorProto.FLOAT, [hidden], np.ones(hidden, np.float32))]
    ln_inputs = ["add_out", "weight"]
    if with_bias:
        inits.append(helper.make_tensor("bias", TensorProto.FLOAT, [hidden], np.zeros(hidden, np.float32)))
        ln_inputs.append("bias")
    nodes = [
        helper.make_node("Add", ["x", "skip"], ["add_out"]),
        helper.make_node("LayerNormalization", ln_inputs, ["y"], axis=-1, epsilon=1e-5),
    ]
    graph = helper.make_graph(
        nodes,
        "skiplayernorm",
        [
            helper.make_tensor_value_info("x", TensorProto.FLOAT, [2, 3, hidden]),
            helper.make_tensor_value_info("skip", TensorProto.FLOAT, [2, 3, hidden]),
        ],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [2, 3, hidden])],
        initializer=inits,
    )
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 20)])


class TestSkipLayerNormFusion(unittest.TestCase):
    def test_fuses_skip_layer_norm_with_bias(self):
        model = to_ir(_build_skip_layer_norm_model(with_bias=True))
        rewrite(model, pattern_rewrite_rules=skip_layer_norm_rules())

        counts = op_counts(model)
        self.assertEqual(counts.get("SkipLayerNormalization", 0), 1)
        self.assertEqual(counts.get("LayerNormalization", 0), 0)
        self.assertEqual(counts.get("Add", 0), 0)
        node = next(n for n in model.graph if n.op_type == "SkipLayerNormalization")
        self.assertEqual(node.domain, "com.microsoft")

    def test_fuses_skip_layer_norm_without_bias(self):
        model = to_ir(_build_skip_layer_norm_model(with_bias=False))
        rewrite(model, pattern_rewrite_rules=skip_layer_norm_rules())

        counts = op_counts(model)
        self.assertEqual(counts.get("SkipLayerNormalization", 0), 1)
        self.assertEqual(counts.get("LayerNormalization", 0), 0)


if __name__ == "__main__":
    unittest.main()
