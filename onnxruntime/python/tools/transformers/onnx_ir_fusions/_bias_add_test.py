# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Unit tests for the BiasAdd onnxscript rewrite rule."""

from __future__ import annotations

import unittest

import numpy as np
import onnx
import onnx.helper as helper
from onnx import TensorProto
from onnx_ir_fusions import bias_add_rules
from onnx_ir_fusions._testing import op_counts, to_ir
from onnxscript.rewriter import rewrite


def _build_bias_add_model(bias_first: bool = False, bias_ndim: int = 1) -> onnx.ModelProto:
    bias_dims = [4] if bias_ndim == 1 else [2, 4]
    bias_init = helper.make_tensor(
        "bias", TensorProto.FLOAT, bias_dims, np.arange(int(np.prod(bias_dims)), dtype=np.float32)
    )
    inner_inputs = ["bias", "x"] if bias_first else ["x", "bias"]
    nodes = [
        helper.make_node("Add", inner_inputs, ["inner"]),
        helper.make_node("Add", ["inner", "skip"], ["y"]),
    ]
    graph = helper.make_graph(
        nodes,
        "biasadd",
        [
            helper.make_tensor_value_info("x", TensorProto.FLOAT, [2, 4]),
            helper.make_tensor_value_info("skip", TensorProto.FLOAT, [2, 4]),
        ],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [2, 4])],
        initializer=[bias_init],
    )
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 20)])


class TestBiasAddFusion(unittest.TestCase):
    def test_fuses_bias_second(self):
        model = to_ir(_build_bias_add_model(bias_first=False))
        rewrite(model, pattern_rewrite_rules=bias_add_rules())

        counts = op_counts(model)
        self.assertEqual(counts.get("BiasAdd", 0), 1)
        self.assertEqual(counts.get("Add", 0), 0)
        node = next(n for n in model.graph if n.op_type == "BiasAdd")
        self.assertEqual(node.domain, "com.microsoft")

    def test_fuses_bias_first(self):
        model = to_ir(_build_bias_add_model(bias_first=True))
        rewrite(model, pattern_rewrite_rules=bias_add_rules())

        counts = op_counts(model)
        self.assertEqual(counts.get("BiasAdd", 0), 1)
        self.assertEqual(counts.get("Add", 0), 0)

    def test_does_not_fuse_2d_bias(self):
        model = to_ir(_build_bias_add_model(bias_ndim=2))
        rewrite(model, pattern_rewrite_rules=bias_add_rules())

        counts = op_counts(model)
        self.assertEqual(counts.get("BiasAdd", 0), 0)
        self.assertEqual(counts.get("Add", 0), 2)


if __name__ == "__main__":
    unittest.main()
