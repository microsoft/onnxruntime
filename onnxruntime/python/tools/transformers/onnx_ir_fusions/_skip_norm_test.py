# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Unit tests for the SkipSimplifiedLayerNormalization onnxscript rewrite rule."""

from __future__ import annotations

import unittest

import numpy as np
import onnx
import onnx.helper as helper
from onnx import TensorProto
from onnx_ir_fusions import skip_norm_rules
from onnx_ir_fusions._testing import op_counts, to_ir
from onnxscript.rewriter import rewrite


def _build_skip_norm_model() -> onnx.ModelProto:
    hidden = 4
    weight = helper.make_tensor("weight", TensorProto.FLOAT, [hidden], np.ones(hidden, np.float32))
    nodes = [
        helper.make_node("Add", ["x", "skip"], ["add_out"]),
        helper.make_node("RMSNormalization", ["add_out", "weight"], ["y"], axis=-1, epsilon=1e-6),
    ]
    graph = helper.make_graph(
        nodes,
        "skipnorm",
        [
            helper.make_tensor_value_info("x", TensorProto.FLOAT, [2, 3, hidden]),
            helper.make_tensor_value_info("skip", TensorProto.FLOAT, [2, 3, hidden]),
        ],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [2, 3, hidden])],
        initializer=[weight],
    )
    # RMSNormalization is an ONNX op since opset 23.
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 23)])


class TestSkipNormFusion(unittest.TestCase):
    def test_fuses_skip_norm(self):
        model = to_ir(_build_skip_norm_model())
        rewrite(model, pattern_rewrite_rules=skip_norm_rules())

        counts = op_counts(model)
        self.assertEqual(counts.get("SkipSimplifiedLayerNormalization", 0), 1)
        self.assertEqual(counts.get("RMSNormalization", 0), 0)
        self.assertEqual(counts.get("Add", 0), 0)
        node = next(n for n in model.graph if n.op_type == "SkipSimplifiedLayerNormalization")
        self.assertEqual(node.domain, "com.microsoft")


if __name__ == "__main__":
    unittest.main()
