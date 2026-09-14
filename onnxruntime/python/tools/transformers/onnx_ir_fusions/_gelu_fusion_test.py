# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Unit tests for the Gelu (Erf / tanh) onnxscript rewrite rules."""

from __future__ import annotations

import math
import unittest

import numpy as np
import onnx
import onnx.helper as helper
import onnx_ir as ir
from onnx_ir_fusions import gelu_fusion_rules
from onnx_ir_fusions._testing import op_counts, to_ir
from onnxscript.rewriter import rewrite

_SQRT2 = math.sqrt(2.0)
_SQRT_2_OVER_PI = math.sqrt(2.0 / math.pi)
_GELU_COEFF = 0.044715


def _scalar(name: str, value: float) -> onnx.TensorProto:
    return helper.make_tensor(name, onnx.TensorProto.FLOAT, [], [value])


def _formula_constant(name: str, value: float, vector_constant: str | None) -> onnx.TensorProto:
    if name == vector_constant:
        return helper.make_tensor(name, onnx.TensorProto.FLOAT, [4], np.full(4, value, dtype=np.float32))
    return _scalar(name, value)


def _build_exact_gelu_model(
    opset: int = 20,
    vector_constant: str | None = None,
) -> onnx.ModelProto:
    inits = [
        _formula_constant("sqrt2", _SQRT2, vector_constant),
        _formula_constant("one", 1.0, vector_constant),
        _formula_constant("half", 0.5, vector_constant),
    ]
    nodes = [
        helper.make_node("Div", ["x", "sqrt2"], ["xdiv"]),
        helper.make_node("Erf", ["xdiv"], ["erf"]),
        helper.make_node("Add", ["erf", "one"], ["addone"]),
        helper.make_node("Mul", ["x", "addone"], ["mulx"]),
        helper.make_node("Mul", ["mulx", "half"], ["y"]),
    ]
    graph = helper.make_graph(
        nodes,
        "exactgelu",
        [helper.make_tensor_value_info("x", onnx.TensorProto.FLOAT, [2, 4])],
        [helper.make_tensor_value_info("y", onnx.TensorProto.FLOAT, [2, 4])],
        initializer=inits,
    )
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", opset)])


def _build_approx_gelu_model() -> onnx.ModelProto:
    inits = [
        _scalar("three", 3.0),
        _scalar("coeff", _GELU_COEFF),
        _scalar("sqrt_2pi", _SQRT_2_OVER_PI),
        _scalar("one", 1.0),
        _scalar("half", 0.5),
    ]
    nodes = [
        helper.make_node("Pow", ["x", "three"], ["x3"]),
        helper.make_node("Mul", ["coeff", "x3"], ["scaled_cube"]),
        helper.make_node("Add", ["x", "scaled_cube"], ["inner"]),
        helper.make_node("Mul", ["sqrt_2pi", "inner"], ["scaled"]),
        helper.make_node("Tanh", ["scaled"], ["tanh"]),
        helper.make_node("Add", ["tanh", "one"], ["addone"]),
        helper.make_node("Mul", ["x", "addone"], ["mulx"]),
        helper.make_node("Mul", ["mulx", "half"], ["y"]),
    ]
    graph = helper.make_graph(
        nodes,
        "approxgelu",
        [helper.make_tensor_value_info("x", onnx.TensorProto.FLOAT, [2, 4])],
        [helper.make_tensor_value_info("y", onnx.TensorProto.FLOAT, [2, 4])],
        initializer=inits,
    )
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 20)])


class TestGeluFusion(unittest.TestCase):
    def test_fuses_exact_gelu(self):
        model = to_ir(_build_exact_gelu_model())
        rewrite(model, pattern_rewrite_rules=gelu_fusion_rules())

        counts = op_counts(model)
        self.assertEqual(counts.get("Gelu", 0), 1)
        self.assertEqual(counts.get("Erf", 0), 0)
        node = next(n for n in model.graph if n.op_type == "Gelu")
        self.assertEqual(node.attributes.get_string("approximate"), "none")

    def test_fuses_approx_gelu(self):
        model = to_ir(_build_approx_gelu_model())
        rewrite(model, pattern_rewrite_rules=gelu_fusion_rules())

        counts = op_counts(model)
        self.assertEqual(counts.get("Gelu", 0), 1)
        self.assertEqual(counts.get("Tanh", 0), 0)
        node = next(n for n in model.graph if n.op_type == "Gelu")
        self.assertEqual(node.attributes.get_string("approximate"), "tanh")

    def test_does_not_fuse_vector_formula_constant(self):
        model = to_ir(_build_exact_gelu_model(vector_constant="sqrt2"))
        rewrite(model, pattern_rewrite_rules=gelu_fusion_rules())

        counts = op_counts(model)
        self.assertEqual(counts.get("Gelu", 0), 0)
        self.assertEqual(counts.get("Erf", 0), 1)

    def test_does_not_emit_gelu_below_opset_20(self):
        model = to_ir(_build_exact_gelu_model(opset=13))
        rewrite(model, pattern_rewrite_rules=gelu_fusion_rules())

        counts = op_counts(model)
        self.assertEqual(counts.get("Gelu", 0), 0)
        self.assertEqual(counts.get("Erf", 0), 1)
        onnx.checker.check_model(ir.to_proto(model))


if __name__ == "__main__":
    unittest.main()
