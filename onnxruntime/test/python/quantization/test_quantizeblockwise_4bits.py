#!/usr/bin/env python
# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

import unittest
from importlib.util import find_spec

import numpy as np
import numpy.typing as npt


def dequantize_blockwise_4bits(quant_values, scale, zero_point, rows, cols):
    expand_quant_value = (np.repeat(quant_values, 2, -1).reshape(*quant_values.shape, 2) >> [0, 4]) & 0x0F
    expand_quant_value = expand_quant_value.reshape(*quant_values.shape[:-1], -1)
    aligned_scale = scale.reshape(*quant_values.shape[:-1], 1)
    expand_zero_point = (np.repeat(zero_point, 2, -1).reshape(-1, 2) >> [0, 4]) & 0xF
    expand_zero_point = expand_zero_point.reshape(-1)
    if (quant_values.size // quant_values.shape[-1]) & 1:
        expand_zero_point = expand_zero_point[:-1]
    expand_zero_point = expand_zero_point.reshape(*quant_values.shape[:-1], -1)
    float_values = ((expand_quant_value - expand_zero_point) * aligned_scale).astype(scale.dtype)
    float_values = float_values.reshape(cols, -1)
    float_values = float_values[:, :rows]
    return float_values


def quantize_blockwise_4bits_ref(matrix_float: npt.ArrayLike, block_size: int, is_symmetric: bool):
    if len(matrix_float.shape) != 2:
        raise ValueError("Current int4 block quantization only supports 2D tensors!")
    rows, cols = matrix_float.shape

    blob_size = block_size // 2
    k_blocks = (rows + block_size - 1) // block_size
    padded_rows = k_blocks * block_size
    pad_len = padded_rows - rows
    matrix_float_padded = matrix_float
    if pad_len > 0:
        matrix_float_padded = np.pad(matrix_float, ((0, pad_len), (0, 0)), "constant")

    matrix_float_padded = np.transpose(matrix_float_padded)

    zero_point_unpacked = np.zeros((cols, k_blocks), dtype=np.uint8)
    scales = np.zeros((cols, k_blocks), dtype=matrix_float_padded.dtype)
    for k_id in range(0, rows, block_size):
        if is_symmetric:
            bmax = (np.abs(matrix_float_padded[:, k_id : k_id + block_size])).max(-1).astype(np.float32)
            scale = bmax / (-8.0)
            zp = 8
            zero_point_unpacked[:, k_id // block_size] = zp
            scales[:, k_id // block_size] = scale
        else:
            vmax = matrix_float_padded[:, k_id : k_id + block_size].astype(np.float32).max(-1)
            vmin = matrix_float_padded[:, k_id : k_id + block_size].astype(np.float32).min(-1)
            vmin = np.minimum(vmin, 0.0).astype(np.float32)
            vmax = np.maximum(vmax, 0.0).astype(np.float32)
            scale = (vmax - vmin) / ((1 << 4) - 1)
            zero_point_fp = vmin
            zero_point_fp[scale != 0.0] = 0.0 - vmin / scale
            zp = np.minimum(15, np.maximum(0, np.round(zero_point_fp)))
            zero_point_unpacked[:, k_id // block_size] = zp
            scales[:, k_id // block_size] = scale

    reciprocal_scale = np.where(scales != 0.0, 1.0 / scales, 0.0)
    int8_values = matrix_float_padded.reshape(cols, k_blocks, block_size) * reciprocal_scale.reshape(
        cols, k_blocks, 1
    ) + zero_point_unpacked.reshape(cols, k_blocks, 1)
    int8_values = np.clip(np.round(int8_values.reshape(cols, -1).astype(np.float32)), 0, 15).astype("uint8")

    zero_point = zero_point_unpacked.reshape(-1)
    zero_point = np.concatenate((zero_point, np.array([8], dtype=np.uint8))) if zero_point.shape[0] & 1 else zero_point
    zero_point = zero_point[0::2] | (zero_point[1::2] << 4)
    zero_point = zero_point.reshape(-1)

    packed = (int8_values[:, 0::2]) | (int8_values[:, 1::2] << 4)
    packed = packed.reshape(cols, k_blocks, blob_size)
    scales = scales.reshape(-1)

    return (packed, scales, zero_point)


def quantize_blockwise_4bits_target(matrix_float: npt.ArrayLike, block_size: int, is_symmetric: bool):
    if len(matrix_float.shape) != 2:
        raise ValueError("Current int4 block quantization only supports 2D tensors!")
    rows, cols = matrix_float.shape

    k_blocks = (rows + block_size - 1) // block_size
    packed = np.zeros((cols, k_blocks, block_size // 2), dtype="uint8")
    scales = np.zeros((cols * k_blocks), dtype=matrix_float.dtype)
    zero_point = np.full((cols * k_blocks + 1) // 2, 136, dtype="uint8")
    from onnxruntime.capi._pybind_state import quantize_matmul_4bits

    quantize_matmul_4bits(packed, matrix_float, scales, zero_point, block_size, cols, rows, is_symmetric)
    return (packed, scales, zero_point)


class TestQuantizeBlockwise4Bits(unittest.TestCase):
    @unittest.skipIf(
        find_spec("onnxruntime.training"), "Skip because training package doesn't has quantize_matmul_4bits"
    )
    def test_quantize_blockwise_4bits(self):
        for rows, cols in [(128, 128), (32, 128), (128, 32), (52, 128), (128, 52), (73, 123)]:
            for block_size in [16, 32, 64, 128]:
                for type in [np.float32, np.float16]:
                    for is_symmetric in [True, False]:
                        matrix_float = np.random.rand(rows, cols).astype(type)
                        quant_value_ref, scales_ref, zero_point_ref = quantize_blockwise_4bits_ref(
                            matrix_float, block_size, is_symmetric
                        )
                        quant_value, scales, zero_point = quantize_blockwise_4bits_target(
                            matrix_float, block_size, is_symmetric
                        )
                        assert np.allclose(scales_ref, scales)
                        assert np.allclose(zero_point_ref, zero_point)
                        assert np.allclose(
                            dequantize_blockwise_4bits(quant_value_ref, scales, zero_point, rows, cols),
                            dequantize_blockwise_4bits(quant_value, scales, zero_point, rows, cols),
                            atol=1.2 * abs(scales).max(),
                        )


if __name__ == "__main__":
    unittest.main()
