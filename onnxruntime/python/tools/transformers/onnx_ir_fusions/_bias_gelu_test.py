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
from onnx_ir_fusions import bias_gelu_rules
from onnx_ir_fusions._testing import op_counts, to_ir
from onnxscript.rewriter import rewrite


def _build_bias_gelu_model(
    approximate: str = "tanh",
    input_shape: list[int] | None = None,
    bias_shape: list[int] | None = None,
    bias_first: bool = False,
) -> onnx.ModelProto:
    input_shape = [2, 4] if input_shape is None else input_shape
    bias_shape = [4] if bias_shape is None else bias_shape
    bias = helper.make_tensor(
        "bias", onnx.TensorProto.FLOAT, bias_shape, np.arange(int(np.prod(bias_shape)), dtype=np.float32)
    )
    add_inputs = ["bias", "x"] if bias_first else ["x", "bias"]
    nodes = [
        helper.make_node("Add", add_inputs, ["add_out"]),
        helper.make_node("Gelu", ["add_out"], ["y"], approximate=approximate),
    ]
    output_shape = input_shape or bias_shape
    graph = helper.make_graph(
        nodes,
        "biasgelu",
        [helper.make_tensor_value_info("x", onnx.TensorProto.FLOAT, input_shape)],
        [helper.make_tensor_value_info("y", onnx.TensorProto.FLOAT, output_shape)],
        initializer=[bias],
    )
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 20)])


class TestBiasGeluFusion(unittest.TestCase):
    def test_fuses_bias_gelu(self):
        model = to_ir(_build_bias_gelu_model(approximate="tanh"))
        rewrite(model, pattern_rewrite_rules=bias_gelu_rules())

        counts = op_counts(model)
        self.assertEqual(counts.get("FastGelu", 0), 1)
        self.assertEqual(counts.get("Gelu", 0), 0)
        self.assertEqual(counts.get("Add", 0), 0)
        node = next(n for n in model.graph if n.op_type == "FastGelu")
        self.assertEqual(node.domain, "com.microsoft")

    def test_fuses_bias_first(self):
        model = to_ir(_build_bias_gelu_model(bias_first=True))
        rewrite(model, pattern_rewrite_rules=bias_gelu_rules())

        self.assertEqual(op_counts(model).get("FastGelu", 0), 1)

    def test_fuses_exact_gelu_to_bias_gelu(self):
        # Exact Gelu uses the erf-based BiasGelu target, not FastGelu.
        model = to_ir(_build_bias_gelu_model(approximate="none"))
        rewrite(model, pattern_rewrite_rules=bias_gelu_rules())

        counts = op_counts(model)
        self.assertEqual(counts.get("FastGelu", 0), 0)
        self.assertEqual(counts.get("BiasGelu", 0), 1)
        self.assertEqual(counts.get("Gelu", 0), 0)

    def test_does_not_fuse_non_vector_bias(self):
        model = to_ir(_build_bias_gelu_model(bias_shape=[1, 4]))
        rewrite(model, pattern_rewrite_rules=bias_gelu_rules())

        self.assertEqual(op_counts(model).get("FastGelu", 0), 0)

    def test_does_not_fuse_mismatched_bias_length(self):
        model = to_ir(_build_bias_gelu_model(input_shape=[2, 3, 4], bias_shape=[1]))
        rewrite(model, pattern_rewrite_rules=bias_gelu_rules())

        self.assertEqual(op_counts(model).get("FastGelu", 0), 0)

    def test_does_not_fuse_scalar_input(self):
        model = to_ir(_build_bias_gelu_model(input_shape=[], bias_shape=[4]))
        rewrite(model, pattern_rewrite_rules=bias_gelu_rules())

        self.assertEqual(op_counts(model).get("FastGelu", 0), 0)


if __name__ == "__main__":
    unittest.main()
