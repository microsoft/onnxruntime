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
from onnx_ir_fusions import skip_norm_rules
from onnx_ir_fusions._testing import op_counts, to_ir
from onnxscript.rewriter import rewrite


def _build_skip_norm_model(
    input_shape: list[int] | None = None,
    skip_shape: list[int] | None = None,
    axis: int = -1,
    gamma_shape: list[int] | None = None,
    data_type: int = onnx.TensorProto.FLOAT,
) -> onnx.ModelProto:
    input_shape = [2, 3, 4] if input_shape is None else input_shape
    skip_shape = input_shape if skip_shape is None else skip_shape
    hidden = input_shape[-1]
    gamma_shape = [hidden] if gamma_shape is None else gamma_shape
    weight = helper.make_tensor("weight", data_type, gamma_shape, np.ones(gamma_shape, np.float32))
    nodes = [
        helper.make_node("Add", ["x", "skip"], ["add_out"]),
        helper.make_node("RMSNormalization", ["add_out", "weight"], ["y"], axis=axis, epsilon=1e-6),
    ]
    graph = helper.make_graph(
        nodes,
        "skipnorm",
        [
            helper.make_tensor_value_info("x", data_type, input_shape),
            helper.make_tensor_value_info("skip", data_type, skip_shape),
        ],
        [helper.make_tensor_value_info("y", data_type, input_shape)],
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

    def test_does_not_fuse_non_last_axis(self):
        model = to_ir(_build_skip_norm_model(axis=1))
        rewrite(model, pattern_rewrite_rules=skip_norm_rules())

        self.assertEqual(op_counts(model).get("SkipSimplifiedLayerNormalization", 0), 0)

    def test_does_not_fuse_rank_4_inputs(self):
        model = to_ir(_build_skip_norm_model(input_shape=[2, 3, 4, 5]))
        rewrite(model, pattern_rewrite_rules=skip_norm_rules())

        self.assertEqual(op_counts(model).get("SkipSimplifiedLayerNormalization", 0), 0)

    def test_does_not_fuse_broadcast_skip(self):
        model = to_ir(_build_skip_norm_model(skip_shape=[1, 1, 4]))
        rewrite(model, pattern_rewrite_rules=skip_norm_rules())

        self.assertEqual(op_counts(model).get("SkipSimplifiedLayerNormalization", 0), 0)

    def test_does_not_fuse_invalid_gamma_shape(self):
        model = to_ir(_build_skip_norm_model(gamma_shape=[1, 4]))
        rewrite(model, pattern_rewrite_rules=skip_norm_rules())

        self.assertEqual(op_counts(model).get("SkipSimplifiedLayerNormalization", 0), 0)

    def test_does_not_fuse_double_input(self):
        model = to_ir(_build_skip_norm_model(data_type=onnx.TensorProto.DOUBLE))
        rewrite(model, pattern_rewrite_rules=skip_norm_rules())

        self.assertEqual(op_counts(model).get("SkipSimplifiedLayerNormalization", 0), 0)


if __name__ == "__main__":
    unittest.main()
