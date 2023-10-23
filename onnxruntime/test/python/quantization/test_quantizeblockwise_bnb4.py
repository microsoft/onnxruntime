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

quant_enums = {"FP4": 0, "NF4": 1}


def quantize_block_fp4(block: npt.ArrayLike):
    # quantize a block of float32 values to uint8 by simulating a binary search using pivots
    # could have used (block[:,None] - quant_map).argmin(axis=1) but there are some mismatches due to
    # floating point precision
    # block: 1-D array of normalized [-1,1] float32 values, len(block) % 2 == 0

    # pivots to find the quantization index
    # only half of the pivots are needed since the other half is symmetric
    pivots = np.array(
        [0.00260417, 0.0859375, 0.20833333, 0.29166667, 0.4166667, 0.583333, 0.8333333, 1], dtype=np.float32
    )
    # indices are not 0,1,2,3,4,5,6,7 because it is a floating point data type
    pivot_indices = np.array([0, 1, 6, 7, 4, 5, 2, 3], dtype=np.uint8)

    # signs of the block
    signs = (block < 0).astype(np.uint8) * 8

    # find the uint8 quantization index
    # argmax finds the first occurance of True
    quant_indices = pivot_indices[(np.abs(block)[:, None] <= pivots).argmax(axis=1)] + signs

    return np.bitwise_or(np.left_shift(quant_indices[::2], 4), quant_indices[1::2])


def quantize_block_nf4(block: npt.ArrayLike):
    pivots = np.array(
        [
            -0.8480964004993439,
            -0.6106329262256622,
            -0.4599952697753906,
            -0.33967943489551544,
            -0.23460740596055984,
            -0.13791173323988914,
            -0.045525018125772476,
            0.03979014977812767,
            0.1202552504837513,
            0.2035212516784668,
            0.2920137718319893,
            0.3893125355243683,
            0.5016634166240692,
            0.6427869200706482,
            0.8614784181118011,
            1.0,
        ],
        dtype=np.float32,
    )

    quant_indices = (block[:, None] <= pivots).argmax(axis=1).astype(np.uint8)

    return np.bitwise_or(np.left_shift(quant_indices[::2], 4), quant_indices[1::2])


def quantize_blockwise_bnb4_ref(matrix_float: npt.ArrayLike, block_size: int, quant_type: str, target=None):
    if len(matrix_float.shape) != 2:
        raise ValueError("Current bnb4 block quantization only supports 2D tensors!")

    numel = matrix_float.size
    num_blocks = (numel + block_size - 1) // block_size
    quantized_numel = (numel + 1) // 2

    packed = np.zeros(quantized_numel, dtype=np.uint8)
    absmax = np.zeros(num_blocks, dtype=matrix_float.dtype)

    flattened_matrix_float = matrix_float.flatten()
    for block_idx in range(num_blocks):
        block_len = min(block_size, numel - block_idx * block_size)
        block = np.float32(flattened_matrix_float[block_idx * block_size : block_idx * block_size + block_len])

        block_absmax = np.max(np.abs(block))
        reciprocal_absmax = 1.0 / block_absmax if block_absmax != 0 else 0.0
        absmax[block_idx] = block_absmax

        if block_len % 2 != 0:
            block = np.append(block, 0.0)
            block_len += 1

        block *= reciprocal_absmax
        start = block_idx * block_size // 2
        end = start + block_len // 2
        if quant_type == "FP4":
            packed[start:end] = quantize_block_fp4(block)
        else:
            packed[start:end] = quantize_block_nf4(block)

    return (packed, absmax)


def quantize_blockwise_bnb4_target(matrix_float: npt.ArrayLike, block_size: int, quant_type: str):
    if len(matrix_float.shape) != 2:
        raise ValueError("Current int4 block quantization only supports 2D tensors!")
    quant_type_enum = quant_enums[quant_type]

    n, k = matrix_float.shape  # already transposed
    numel = n * k
    num_blocks = (numel + block_size - 1) // block_size
    quantized_numel = (numel + 1) // 2

    packed = np.zeros(quantized_numel, dtype="uint8")
    absmax = np.zeros(num_blocks, dtype=matrix_float.dtype)
    from onnxruntime.capi._pybind_state import quantize_matmul_bnb4

    quantize_matmul_bnb4(packed, matrix_float, absmax, block_size, quant_type_enum, n, k)
    return (packed, absmax)


class TestQuantizeBlockwiseBnb4(unittest.TestCase):
    @unittest.skipIf(
        find_spec("onnxruntime.training"), "Skip because training package doesn't has quantize_matmul_bnb4"
    )
    def test_quantize_blockwise_bnb4(self):
        for quant_type in ["FP4", "NF4"]:
            for k, n in [(128, 128), (32, 128), (128, 32), (52, 128), (128, 52), (73, 123)]:
                for block_size in [16, 32, 64, 128]:
                    for type in [np.float32, np.float16]:
                        matrix_float = np.random.uniform(-1, 1, (k, n)).astype(type)
                        quant_value_ref, absmax_ref = quantize_blockwise_bnb4_ref(matrix_float, block_size, quant_type)
                        quant_value, absmax = quantize_blockwise_bnb4_target(matrix_float, block_size, quant_type)
                        assert np.allclose(quant_value_ref, quant_value)
                        assert np.allclose(absmax_ref, absmax)


if __name__ == "__main__":
    unittest.main()
