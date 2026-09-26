#!/usr/bin/env python
# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

import unittest
from importlib.util import find_spec

import numpy as np

from onnxruntime.quantization.neural_compressor.weight_only import (
    quant_tensor,
    quant_tensor_k_quant_cpu,
    quant_tensor_k_quant_cuda,
)

GROUP_SIZE = 32


def dequantize(q_weight, scale, zero_point):
    return (q_weight - zero_point.astype(np.float64)) * scale


def squared_error(quantized, data):
    return np.sum((dequantize(*quantized) - data.reshape(-1, GROUP_SIZE)) ** 2)


class TestWeightOnlyKQuant(unittest.TestCase):
    @staticmethod
    def weights():
        # Heavy-tailed, like the weights of a trained transformer.
        return (np.random.default_rng(0).standard_t(4, size=(256, 512)) * 0.02).astype(np.float32)

    def test_quant_tensor_k_quant_cpu_returns_valid_codes(self):
        data = self.weights()
        groups = data.size // GROUP_SIZE
        for num_bits in (4, 8):
            with self.subTest(num_bits=num_bits):
                q_weight, scale, zero_point = quant_tensor_k_quant_cpu(data, num_bits, GROUP_SIZE)
                self.assertEqual(q_weight.shape, (groups, GROUP_SIZE))
                self.assertEqual(scale.shape, (groups, 1))
                self.assertEqual(zero_point.shape, (groups, 1))
                self.assertEqual(zero_point.dtype, np.uint8)
                self.assertTrue(np.all(q_weight == np.round(q_weight)))
                self.assertTrue(np.all((q_weight >= 0) & (q_weight <= 2**num_bits - 1)))
                self.assertTrue(np.all(zero_point <= 2**num_bits - 1))
                self.assertTrue(np.all(scale > 0))

    def test_quant_tensor_k_quant_cpu_beats_min_max_rounding(self):
        data = self.weights()
        for num_bits in (4, 8):
            with self.subTest(num_bits=num_bits):
                k_quant_error = squared_error(quant_tensor_k_quant_cpu(data, num_bits, GROUP_SIZE), data)
                rtn_error = squared_error(quant_tensor(data, num_bits, GROUP_SIZE, "asym", "uint"), data)
                self.assertLess(k_quant_error, 0.97 * rtn_error)

    def test_quant_tensor_k_quant_cpu_does_not_clip_when_group_is_one_signed(self):
        data = np.stack([np.linspace(0.5, 1.0, GROUP_SIZE), -np.linspace(0.5, 1.0, GROUP_SIZE), np.zeros(GROUP_SIZE)])
        data = data.astype(np.float32)
        q_weight, scale, zero_point = quant_tensor_k_quant_cpu(data, 4, GROUP_SIZE)
        error = np.abs(dequantize(q_weight, scale, zero_point) - data)
        self.assertTrue(np.all(error <= scale / 2 + 1e-6))

    @unittest.skipUnless(find_spec("cupy") and find_spec("torch"), "requires cupy and torch")
    def test_quant_tensor_k_quant_cuda_matches_cpu(self):
        import torch  # noqa: PLC0415

        if not torch.cuda.is_available():
            self.skipTest("requires CUDA")
        data = self.weights()
        for num_bits in (4, 8):
            with self.subTest(num_bits=num_bits):
                cpu = quant_tensor_k_quant_cpu(data, num_bits, GROUP_SIZE)
                cuda = quant_tensor_k_quant_cuda(data, num_bits, GROUP_SIZE)
                # Float32 sums may round differently on the GPU and flip a near-tie between two candidates.
                same_groups = np.mean(np.all(cuda[0] == cpu[0], axis=1) & (cuda[2] == cpu[2])[:, 0])
                self.assertGreater(same_groups, 0.99)
                self.assertAlmostEqual(squared_error(cuda, data) / squared_error(cpu, data), 1.0, delta=1e-3)


if __name__ == "__main__":
    unittest.main()
