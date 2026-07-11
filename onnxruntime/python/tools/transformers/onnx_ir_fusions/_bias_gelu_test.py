# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Unit tests for the BiasGelu onnxscript rewrite rule."""

from __future__ import annotations

import unittest

import numpy as np
import onnx
import onnx.helper as helper
from onnx import TensorProto
from onnx_ir_fusions import bias_gelu_rules
from onnx_ir_fusions._testing import op_counts, to_ir
from onnxscript.rewriter import rewrite


def _build_bias_gelu_model(approximate: str = "tanh") -> onnx.ModelProto:
    bias = helper.make_tensor("bias", TensorProto.FLOAT, [4], np.arange(4, dtype=np.float32))
    nodes = [
        helper.make_node("Add", ["x", "bias"], ["add_out"]),
        helper.make_node("Gelu", ["add_out"], ["y"], approximate=approximate),
    ]
    graph = helper.make_graph(
        nodes,
        "biasgelu",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [2, 4])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [2, 4])],
        initializer=[bias],
    )
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 20)])


class TestBiasGeluFusion(unittest.TestCase):
    def test_fuses_bias_gelu(self):
        model = to_ir(_build_bias_gelu_model(approximate="tanh"))
        rewrite(model, pattern_rewrite_rules=bias_gelu_rules())

        counts = op_counts(model)
        self.assertEqual(counts.get("BiasGelu", 0), 1)
        self.assertEqual(counts.get("Gelu", 0), 0)
        self.assertEqual(counts.get("Add", 0), 0)
        node = next(n for n in model.graph if n.op_type == "BiasGelu")
        self.assertEqual(node.domain, "com.microsoft")

    def test_does_not_fuse_exact_gelu(self):
        # BiasGelu uses the tanh approximation; an exact Gelu must be left alone.
        model = to_ir(_build_bias_gelu_model(approximate="none"))
        rewrite(model, pattern_rewrite_rules=bias_gelu_rules())

        counts = op_counts(model)
        self.assertEqual(counts.get("BiasGelu", 0), 0)
        self.assertEqual(counts.get("Gelu", 0), 1)


if __name__ == "__main__":
    unittest.main()
