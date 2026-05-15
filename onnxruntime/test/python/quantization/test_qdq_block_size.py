#!/usr/bin/env python
# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------
"""Tests for opset-21 block_size attribute in the QDQ static-quantization pipeline."""

from __future__ import annotations

import os
import tempfile
import unittest

import numpy as np
import onnx
import onnx.helper
import onnx.numpy_helper
import onnx.shape_inference
from op_test_utils import TestDataFeeds

from onnxruntime.quantization import QuantFormat, QuantType, quantize_static


def _make_matmul_model(opset: int) -> onnx.ModelProto:
    """Build a minimal MatMul model (A x B) where B is a constant initializer."""
    b_data = np.random.default_rng(42).uniform(-1.0, 1.0, (64, 32)).astype(np.float32)
    a_info = onnx.helper.make_tensor_value_info("A", onnx.TensorProto.FLOAT, [1, 64])
    y_info = onnx.helper.make_tensor_value_info("Y", onnx.TensorProto.FLOAT, None)
    b_init = onnx.numpy_helper.from_array(b_data, "B")
    node = onnx.helper.make_node("MatMul", ["A", "B"], ["Y"], name="MatMul0")
    graph = onnx.helper.make_graph([node], "test", [a_info], [y_info], initializer=[b_init])
    model = onnx.helper.make_model(
        graph,
        opset_imports=[onnx.helper.make_opsetid("", opset)],
    )
    model.ir_version = 10 if opset >= 21 else 8
    return onnx.shape_inference.infer_shapes(model)


def _make_data_reader(n: int = 3):
    data = [{"A": np.random.default_rng(i).uniform(-1.0, 1.0, (1, 64)).astype(np.float32)} for i in range(n)]
    return TestDataFeeds(data)


class TestQDQBlockSize(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory(prefix="ort.qdq.blocksize_")

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def _path(self, name: str) -> str:
        return os.path.join(self._tmp.name, name)

    def test_block_size_emits_attribute(self):
        """quantize_static with BlockSize=32 must emit block_size=32 on Q and DQ nodes."""
        float_path = self._path("matmul_f32_emit.onnx")
        qdq_path = self._path("matmul_qdq_emit.onnx")
        onnx.save_model(_make_matmul_model(21), float_path)

        quantize_static(
            float_path,
            qdq_path,
            _make_data_reader(),
            quant_format=QuantFormat.QDQ,
            activation_type=QuantType.QInt8,
            weight_type=QuantType.QInt8,
            op_types_to_quantize=["MatMul"],
            extra_options={"WeightSymmetric": True, "ActivationSymmetric": True, "BlockSize": 32},
        )

        model = onnx.load_model(qdq_path)
        nodes_with_bs = [n for n in model.graph.node if any(a.name == "block_size" for a in n.attribute)]
        self.assertGreater(len(nodes_with_bs), 0, "Expected at least one Q/DQ node with block_size attribute")
        for node in nodes_with_bs:
            bs_attr = next(a for a in node.attribute if a.name == "block_size")
            self.assertEqual(bs_attr.i, 32, f"Expected block_size=32, got {bs_attr.i}")

    def test_block_size_scale_shape(self):
        """Scale and zero_point initializers must be 2-D with n_blocks as first dimension."""
        float_path = self._path("matmul_f32_shape.onnx")
        qdq_path = self._path("matmul_qdq_shape.onnx")
        onnx.save_model(_make_matmul_model(21), float_path)

        block_size = 32
        quantize_static(
            float_path,
            qdq_path,
            _make_data_reader(),
            quant_format=QuantFormat.QDQ,
            activation_type=QuantType.QInt8,
            weight_type=QuantType.QInt8,
            op_types_to_quantize=["MatMul"],
            extra_options={"WeightSymmetric": True, "ActivationSymmetric": True, "BlockSize": block_size},
        )

        model = onnx.load_model(qdq_path)
        inits = {i.name: i for i in model.graph.initializer}

        # B has shape [64, 32]; quantization axis=0; n_blocks = ceil(64/32) = 2
        weight_scale = inits.get("B_scale")
        self.assertIsNotNone(weight_scale, "B_scale initializer not found")
        scale_shape = list(weight_scale.dims)
        self.assertEqual(len(scale_shape), 2, f"Expected 2-D scale, got shape {scale_shape}")
        expected_n_blocks = (64 + block_size - 1) // block_size
        self.assertEqual(scale_shape[0], expected_n_blocks, f"Expected {expected_n_blocks} blocks, got {scale_shape}")

    def test_block_size_below_opset_raises(self):
        """BlockSize > 0 on an opset-13 model must raise or auto-upgrade to opset >= 21."""
        float_path = self._path("matmul_f32_opset13.onnx")
        qdq_path = self._path("matmul_qdq_opset13.onnx")
        onnx.save_model(_make_matmul_model(13), float_path)

        try:
            quantize_static(
                float_path,
                qdq_path,
                _make_data_reader(),
                quant_format=QuantFormat.QDQ,
                activation_type=QuantType.QInt8,
                weight_type=QuantType.QInt8,
                op_types_to_quantize=["MatMul"],
                extra_options={"WeightSymmetric": True, "ActivationSymmetric": True, "BlockSize": 32},
            )
            out_model = onnx.load_model(qdq_path)
            onnx_opset = next(
                (e.version for e in out_model.opset_import if e.domain in ("", "ai.onnx")),
                0,
            )
            self.assertGreaterEqual(onnx_opset, 21, "Auto-upgraded model must be at opset >= 21")
        except ValueError:
            pass  # explicit rejection is also valid

    def test_block_size_zero_unchanged(self):
        """With no BlockSize option, no block_size attribute should appear on any node."""
        float_path = self._path("matmul_f32_zero.onnx")
        qdq_path = self._path("matmul_qdq_zero.onnx")
        onnx.save_model(_make_matmul_model(21), float_path)

        quantize_static(
            float_path,
            qdq_path,
            _make_data_reader(),
            quant_format=QuantFormat.QDQ,
            activation_type=QuantType.QInt8,
            weight_type=QuantType.QInt8,
            op_types_to_quantize=["MatMul"],
            extra_options={"WeightSymmetric": True, "ActivationSymmetric": True},
        )

        model = onnx.load_model(qdq_path)
        nodes_with_bs = [n for n in model.graph.node if any(a.name == "block_size" for a in n.attribute)]
        self.assertEqual(len(nodes_with_bs), 0, "No block_size attribute expected when BlockSize=0")


if __name__ == "__main__":
    unittest.main()
