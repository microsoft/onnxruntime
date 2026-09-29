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
from onnxruntime.quantization.quant_utils import compute_scale_zp_blocked, quantize_onnx_initializer


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


def _make_conv_model(opset: int) -> onnx.ModelProto:
    """Build a minimal Conv model with a 4-D weight initializer (out_ch, in_ch, kH, kW)."""
    rng = np.random.default_rng(7)
    # Weight shape: [8, 1, 3, 3] — rank 4
    w_data = rng.uniform(-1.0, 1.0, (8, 1, 3, 3)).astype(np.float32)
    x_info = onnx.helper.make_tensor_value_info("X", onnx.TensorProto.FLOAT, [1, 1, 8, 8])
    y_info = onnx.helper.make_tensor_value_info("Y", onnx.TensorProto.FLOAT, None)
    w_init = onnx.numpy_helper.from_array(w_data, "W")
    node = onnx.helper.make_node("Conv", ["X", "W"], ["Y"], name="Conv0")
    graph = onnx.helper.make_graph([node], "test_conv", [x_info], [y_info], initializer=[w_init])
    model = onnx.helper.make_model(
        graph,
        opset_imports=[onnx.helper.make_opsetid("", opset)],
    )
    model.ir_version = 10 if opset >= 21 else 8
    return onnx.shape_inference.infer_shapes(model)


def _make_data_reader(n: int = 3):
    data = [{"A": np.random.default_rng(i).uniform(-1.0, 1.0, (1, 64)).astype(np.float32)} for i in range(n)]
    return TestDataFeeds(data)


def _make_conv_data_reader(n: int = 3):
    data = [{"X": np.random.default_rng(i).uniform(-1.0, 1.0, (1, 1, 8, 8)).astype(np.float32)} for i in range(n)]
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

    def test_block_size_below_opset_auto_upgrades_to_opset21(self):
        """BlockSize > 0 on an opset-13 model must auto-upgrade the model to opset >= 21."""
        float_path = self._path("matmul_f32_opset13.onnx")
        qdq_path = self._path("matmul_qdq_opset13.onnx")
        onnx.save_model(_make_matmul_model(13), float_path)

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

    def test_block_size_rank4_weight_raises(self):
        """BlockSize > 0 on a Conv with a rank-4 weight must raise NotImplementedError.

        The ONNX opset-21 spec requires scale/zero_point to have the same rank as the
        input tensor. Our implementation only supports rank-2 weight tensors for per-block
        quantization; rank-4 (Conv) is explicitly rejected.
        """
        float_path = self._path("conv_f32_rank4.onnx")
        qdq_path = self._path("conv_qdq_rank4.onnx")
        onnx.save_model(_make_conv_model(21), float_path)

        with self.assertRaises(NotImplementedError):
            quantize_static(
                float_path,
                qdq_path,
                _make_conv_data_reader(),
                quant_format=QuantFormat.QDQ,
                activation_type=QuantType.QInt8,
                weight_type=QuantType.QInt8,
                op_types_to_quantize=["Conv"],
                extra_options={"WeightSymmetric": True, "ActivationSymmetric": True, "BlockSize": 4},
            )

    def test_block_size_axis1_scale_shape(self):
        """scale/zero_point for axis=1 must have shape (M, n_blocks), not (n_blocks, M).

        Regression test for the moveaxis bug: compute_scale_zp_blocked was not moving
        the block axis back to its original position before returning, so axis=1 results
        were transposed relative to the ONNX opset-21 spec requirement that
        scale.shape[axis] == ceil(input.shape[axis] / block_size).
        """
        rng = np.random.default_rng(0)
        # weight (M=4, N=8), quantize along axis=1 with block_size=4
        # => n_blocks = ceil(8/4) = 2; expected scale shape: (4, 2)
        weight = rng.uniform(-1.0, 1.0, (4, 8)).astype(np.float32)
        quant_type = onnx.TensorProto.INT8
        axis = 1
        block_size = 4

        zero_points, scales = compute_scale_zp_blocked(
            weight, quant_type, axis=axis, block_size=block_size, symmetric=True
        )

        expected_shape = (4, 2)
        self.assertEqual(
            scales.shape,
            expected_shape,
            f"Expected scale shape {expected_shape}, got {scales.shape}",
        )
        self.assertEqual(
            zero_points.shape,
            expected_shape,
            f"Expected zero_point shape {expected_shape}, got {zero_points.shape}",
        )

        # Verify the quantize_onnx_initializer round-trip produces the correct output shape.
        weight_proto = onnx.numpy_helper.from_array(weight, "W")
        q_proto = quantize_onnx_initializer(
            weight_proto,
            quant_type,
            zero_point=zero_points,
            scale=scales,
            axis=axis,
            block_size=block_size,
        )
        q_data = onnx.numpy_helper.to_array(q_proto)
        self.assertEqual(
            q_data.shape,
            weight.shape,
            f"Quantized weight shape {q_data.shape} must match original {weight.shape}",
        )

        # Dequantize and check round-trip error is within one quantization step.
        # Expand scale/zp from (4, 2) to (4, 8) by repeating each block entry block_size
        # times along axis=1, then trim to the actual weight width.
        N = weight.shape[axis]  # noqa: N806
        scale_expanded = np.repeat(scales, block_size, axis=axis)[:, :N].astype(np.float32)
        zp_expanded = np.repeat(zero_points, block_size, axis=axis)[:, :N].astype(np.float32)
        dequant = (q_data.astype(np.float32) - zp_expanded) * scale_expanded
        max_err = float(np.abs(dequant - weight).max())
        # For INT8, max quantization error must be <= 0.5 * max(scale).
        self.assertLessEqual(
            max_err,
            0.5 * float(scales.max()) + 1e-6,
            f"Dequantization round-trip error {max_err:.6f} exceeds allowed bound",
        )

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
