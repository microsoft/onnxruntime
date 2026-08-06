# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Unit tests for the QuickGelu onnxscript rewrite rule."""

from __future__ import annotations

import unittest

import numpy as np
import onnx
import onnx.helper as helper
from onnx import TensorProto
from onnx_ir_fusions import quick_gelu_rules
from onnx_ir_fusions._testing import op_counts, to_ir
from onnxscript.rewriter import rewrite


def _build_quickgelu_model(alpha: float = 1.7021484375) -> onnx.ModelProto:
    alpha_init = helper.make_tensor("alpha", TensorProto.FLOAT, [], [alpha])
    nodes = [
        helper.make_node("Mul", ["x", "alpha"], ["scaled"]),
        helper.make_node("Sigmoid", ["scaled"], ["sig"]),
        helper.make_node("Mul", ["x", "sig"], ["y"]),
    ]
    graph = helper.make_graph(
        nodes,
        "quickgelu",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [2, 4])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [2, 4])],
        initializer=[alpha_init],
    )
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 20)])


class TestQuickGeluFusion(unittest.TestCase):
    def test_fuses_quickgelu(self):
        model = to_ir(_build_quickgelu_model())
        rewrite(model, pattern_rewrite_rules=quick_gelu_rules())

        counts = op_counts(model)
        self.assertEqual(counts.get("QuickGelu", 0), 1)
        self.assertEqual(counts.get("Sigmoid", 0), 0)
        self.assertEqual(counts.get("Mul", 0), 0)

        node = next(n for n in model.graph if n.op_type == "QuickGelu")
        self.assertEqual(node.domain, "com.microsoft")
        self.assertTrue(np.isclose(node.attributes.get_float("alpha"), 1.7021484375))

    def test_does_not_fuse_wrong_alpha(self):
        model = to_ir(_build_quickgelu_model(alpha=2.0))
        rewrite(model, pattern_rewrite_rules=quick_gelu_rules())

        counts = op_counts(model)
        self.assertEqual(counts.get("QuickGelu", 0), 0)
        self.assertEqual(counts.get("Sigmoid", 0), 1)


if __name__ == "__main__":
    unittest.main()
