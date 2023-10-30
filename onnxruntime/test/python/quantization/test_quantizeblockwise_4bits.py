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


def dequantize_blockwise_4bits(quant_values, scale, zero_point, valid_len):
    blob_size = quant_values.shape[0]
    block_size = blob_size * 2

    quant_float = np.zeros((block_size), dtype=scale.dtype)
    for b in range(blob_size):
        v = quant_values[b]
        quant_float[2 * b] = ((v & 0xF) - zero_point) * scale if 2 * b < valid_len else 0.0
        quant_float[2 * b + 1] = ((v >> 4) - zero_point) * scale if 2 * b + 1 < valid_len else 0.0
    return quant_float


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

    packed = np.zeros((cols, k_blocks, blob_size), dtype="uint8")
    scales = np.zeros((cols, k_blocks), dtype=matrix_float_padded.dtype)
    zero_point = np.full((cols, (k_blocks + 1) // 2), 136, dtype="uint8")

    matrix_float_padded = np.transpose(matrix_float_padded)
    for n in range(cols):
        for k_id in range(0, rows, block_size):
            if is_symmetric:
                amax_idx = np.argmax(np.abs(matrix_float_padded[n, k_id : k_id + block_size]))
                bmax = np.float32(matrix_float_padded[n, k_id + amax_idx])
                scale = bmax / (-8.0)
                zp = 8
            else:
                vmin = np.min(np.float32(matrix_float_padded[n, k_id : k_id + block_size]))
                vmax = np.max(np.float32(matrix_float_padded[n, k_id : k_id + block_size]))
                vmin = min(vmin, 0.0)
                vmax = max(vmax, 0.0)
                scale = (vmax - vmin) / ((1 << 4) - 1)
                zero_point_fp = vmin
                if scale != 0.0:
                    zero_point_fp = 0.0 - vmin / scale
                zp = min(15, max(0, round(zero_point_fp)))

            reciprocal_scale = 1.0 / scale if scale != 0 else 0.0
            block_idx = k_id // block_size
            scales[n, block_idx] = scale
            zp_pair = zero_point[n, block_idx // 2]
            zero_point[n, block_idx // 2] = ((zp_pair & 0x0F) | (zp << 4)) if (block_idx & 1) else ((zp_pair & 0xF0) | zp)

            blk_int0 = np.clip(
                np.round(np.float32(matrix_float_padded[n, k_id : k_id + block_size : 2] * reciprocal_scale + zp)),
                0,
                15,
            ).astype("uint8")
            blk_int1 = np.clip(
                np.round(np.float32(matrix_float_padded[n, k_id + 1 : k_id + block_size : 2] * reciprocal_scale + zp)),
                0,
                15,
            ).astype("uint8")
            packed[n, block_idx] = np.bitwise_or(blk_int0, np.left_shift(blk_int1, 4))

    return (packed, scales, zero_point)


def quantize_blockwise_4bits_target(matrix_float: npt.ArrayLike, block_size: int, is_symmetric: bool):
    if len(matrix_float.shape) != 2:
        raise ValueError("Current int4 block quantization only supports 2D tensors!")
    rows, cols = matrix_float.shape

    k_blocks = (rows + block_size - 1) // block_size
    packed = np.zeros((cols, k_blocks, block_size // 2), dtype="uint8")
    scales = np.zeros((cols, k_blocks), dtype=matrix_float.dtype)
    zero_point = np.full((cols, (k_blocks + 1) // 2), 136, dtype="uint8")
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
                        for c in range(quant_value_ref.shape[0]):
                            for k in range(quant_value_ref.shape[1]):
                                assert np.allclose(
                                    dequantize_blockwise_4bits(
                                        quant_value_ref[c, k],
                                        scales_ref[c, k],
                                        (zero_point_ref[c, k//2] >> 4)
                                        if (k & 1)
                                        else (zero_point_ref[c, k//2] & 0x0F),
                                        min(block_size, rows - k * block_size),
                                    ),
                                    dequantize_blockwise_4bits(
                                        quant_value[c, k],
                                        scales[c, k],
                                        (zero_point[c, k//2] >> 4) if (k & 1) else (zero_point[c, k//2] & 0x0F),
                                        min(block_size, rows - k * block_size),
                                    ),
                                    atol=1.2 * abs(scales[c, k]),
                                )


if __name__ == "__main__":
    unittest.main()
