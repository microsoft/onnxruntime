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
from onnx_ir_fusions import skip_layer_norm_rules
from onnx_ir_fusions._testing import op_counts, to_ir
from onnxscript.rewriter import rewrite


def _build_skip_layer_norm_model(
    with_bias: bool = True,
    input_shape: list[int] | None = None,
    skip_shape: list[int] | None = None,
    axis: int = -1,
    gamma_shape: list[int] | None = None,
    beta_shape: list[int] | None = None,
    data_type: int = onnx.TensorProto.FLOAT,
) -> onnx.ModelProto:
    input_shape = [2, 3, 4] if input_shape is None else input_shape
    skip_shape = input_shape if skip_shape is None else skip_shape
    hidden = input_shape[-1]
    gamma_shape = [hidden] if gamma_shape is None else gamma_shape
    beta_shape = [hidden] if beta_shape is None else beta_shape
    inits = [helper.make_tensor("weight", data_type, gamma_shape, np.ones(gamma_shape, dtype=np.float32))]
    ln_inputs = ["add_out", "weight"]
    if with_bias:
        inits.append(helper.make_tensor("bias", data_type, beta_shape, np.zeros(beta_shape, dtype=np.float32)))
        ln_inputs.append("bias")
    nodes = [
        helper.make_node("Add", ["x", "skip"], ["add_out"]),
        helper.make_node("LayerNormalization", ln_inputs, ["y"], axis=axis, epsilon=1e-5),
    ]
    graph = helper.make_graph(
        nodes,
        "skiplayernorm",
        [
            helper.make_tensor_value_info("x", data_type, input_shape),
            helper.make_tensor_value_info("skip", data_type, skip_shape),
        ],
        [helper.make_tensor_value_info("y", data_type, input_shape)],
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

    def test_does_not_fuse_rank_4_inputs(self):
        model = to_ir(_build_skip_layer_norm_model(input_shape=[2, 3, 4, 5]))
        rewrite(model, pattern_rewrite_rules=skip_layer_norm_rules())

        self.assertEqual(op_counts(model).get("SkipLayerNormalization", 0), 0)

    def test_does_not_fuse_unsupported_broadcast_skip(self):
        model = to_ir(_build_skip_layer_norm_model(skip_shape=[1, 1, 4]))
        rewrite(model, pattern_rewrite_rules=skip_layer_norm_rules())

        self.assertEqual(op_counts(model).get("SkipLayerNormalization", 0), 0)

    def test_does_not_fuse_invalid_affine_shapes(self):
        model = to_ir(_build_skip_layer_norm_model(gamma_shape=[1, 4], beta_shape=[1, 4]))
        rewrite(model, pattern_rewrite_rules=skip_layer_norm_rules())

        self.assertEqual(op_counts(model).get("SkipLayerNormalization", 0), 0)

    def test_does_not_fuse_double_input(self):
        model = to_ir(_build_skip_layer_norm_model(data_type=onnx.TensorProto.DOUBLE))
        rewrite(model, pattern_rewrite_rules=skip_layer_norm_rules())

        self.assertEqual(op_counts(model).get("SkipLayerNormalization", 0), 0)


if __name__ == "__main__":
    unittest.main()
