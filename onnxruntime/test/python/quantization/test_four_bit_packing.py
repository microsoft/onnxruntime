# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import unittest

import numpy as np
from ml_dtypes import int4, uint4
from onnx import TensorProto, checker, helper, numpy_helper

import onnxruntime as ort
from onnxruntime.quantization.quant_utils import pack_bytes_to_4bit, quantize_onnx_initializer


def pack_reference(values):
    result = bytearray()
    for start in range(0, len(values), 2):
        low = int(values[start]) % 16
        high = int(values[start + 1]) % 16 if start + 1 < len(values) else 0
        result.append(low + 16 * high)
    return result


class TestFourBitPacking(unittest.TestCase):
    def test_all_signed_and_unsigned_pairs(self):
        for signed in (False, True):
            with self.subTest(signed=signed):
                numbers = range(-8, 8) if signed else range(16)
                values = np.array(
                    [number for first in numbers for second in numbers for number in (first, second)],
                    dtype=np.int8 if signed else np.uint8,
                )
                actual = pack_bytes_to_4bit(values.tobytes())
                self.assertIsInstance(actual, bytearray)
                self.assertEqual(actual, pack_reference(values))

    def test_lengths_and_odd_padding(self):
        for length in (0, 1, 2, 3, 15, 16, 17, 255, 256, 257, 65535, 65536, 65537):
            with self.subTest(length=length):
                values = ((np.arange(length, dtype=np.int64) * 7) % 16).astype(np.uint8)
                actual = pack_bytes_to_4bit(values.tobytes())
                self.assertEqual(actual, pack_reference(values))
                self.assertEqual(len(actual), (length + 1) // 2)
                if length % 2:
                    self.assertEqual(actual[-1] >> 4, 0)

    def test_source_ownership_and_buffer_inputs(self):
        expected = bytearray([0x10, 0xF2, 0x08])
        for factory in (bytes, bytearray, memoryview):
            with self.subTest(factory=factory.__name__):
                storage = bytearray([0, 1, 2, 15, 8])
                source = factory(storage)
                result = pack_bytes_to_4bit(source)
                self.assertEqual(result, expected)
                self.assertEqual(storage, bytearray([0, 1, 2, 15, 8]))
                result[0] = 255
                self.assertEqual(storage[0], 0)
                storage[-1] = 0
                self.assertEqual(result[-1], 8)

    def test_noncontiguous_and_nonbyte_sequences_keep_element_order(self):
        sources = {
            "strided_view": memoryview(bytearray([0, 9, 1, 9, 2]))[::2],
            "reversed_view": memoryview(bytearray([8, 15, 2, 1, 0]))[::-1],
            "signed_view": memoryview(np.array([-8, -1, 0, 7, -2], dtype=np.int8)),
            "wider_view": memoryview(np.array([-8, -1, 0, 7, -2], dtype=np.int16)),
            "list": [0, 1, 2, 15, 8],
            "strided_array": np.array([0, 9, 1, 9, 2], dtype=np.uint8)[::2],
        }
        for name, source in sources.items():
            with self.subTest(source=name):
                before = list(source)
                result = pack_bytes_to_4bit(source)
                self.assertEqual(result, pack_reference(source))
                self.assertIsInstance(result, bytearray)
                self.assertEqual(list(source), before)
                result[0] ^= 0xFF
                self.assertEqual(list(source), before)

    def test_small_signed_pattern_has_expected_nibble_order(self):
        self.assertEqual(
            pack_bytes_to_4bit(np.array([-8, -1, 0, 7, -2], dtype=np.int8).tobytes()),
            bytearray([0xF8, 0x70, 0x0E]),
        )

    def test_initializer_raw_bytes_round_trip(self):
        for quant_type, quant_dtype in ((TensorProto.INT4, int4), (TensorProto.UINT4, uint4)):
            for shape in ((3, 5), (4, 4)):
                with self.subTest(quant_type=quant_type, shape=shape):
                    numbers = np.arange(np.prod(shape), dtype=np.int16) % 16
                    if quant_type == TensorProto.INT4:
                        numbers -= 8
                    expected = numbers.reshape(shape)
                    weights = expected.astype(np.float32) * 0.25
                    quantized = quantize_onnx_initializer(
                        numpy_helper.from_array(weights, "weight"),
                        quant_type,
                        np.array(0, dtype=quant_dtype),
                        np.array(0.25, dtype=np.float32),
                    )
                    self.assertEqual(quantized.raw_data, bytes(pack_reference(numbers)))
                    restored = TensorProto()
                    restored.ParseFromString(quantized.SerializeToString())
                    np.testing.assert_array_equal(numpy_helper.to_array(restored).astype(np.int16), expected)

    def test_native_cpu_dequantization_and_matmul(self):
        for quant_type, quant_dtype in ((TensorProto.INT4, int4), (TensorProto.UINT4, uint4)):
            for dtype, tensor_type in ((np.float16, TensorProto.FLOAT16), (np.float32, TensorProto.FLOAT)):
                with self.subTest(quant_type=quant_type, dtype=dtype):
                    numbers = np.arange(15, dtype=np.int16)
                    if quant_type == TensorProto.INT4:
                        numbers -= 8
                    weights = numbers.reshape(5, 3).astype(dtype) * dtype(0.25)
                    scale = np.array(0.25, dtype=dtype)
                    zero_point = np.array(0, dtype=quant_dtype)
                    quantized = quantize_onnx_initializer(
                        numpy_helper.from_array(weights, "weight"), quant_type, zero_point, scale
                    )
                    graph = helper.make_graph(
                        [
                            helper.make_node("DequantizeLinear", [quantized.name, "scale", "zero_point"], ["decoded"]),
                            helper.make_node("MatMul", ["input", "decoded"], ["output"]),
                        ],
                        "packed_weight_matmul",
                        [helper.make_tensor_value_info("input", tensor_type, [2, 5])],
                        [
                            helper.make_tensor_value_info("decoded", tensor_type, [5, 3]),
                            helper.make_tensor_value_info("output", tensor_type, [2, 3]),
                        ],
                        [
                            quantized,
                            numpy_helper.from_array(scale, "scale"),
                            numpy_helper.from_array(zero_point, "zero_point"),
                        ],
                    )
                    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 21)], ir_version=10)
                    checker.check_model(model)
                    options = ort.SessionOptions()
                    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
                    session = ort.InferenceSession(
                        model.SerializeToString(), sess_options=options, providers=["CPUExecutionProvider"]
                    )
                    data = np.arange(10, dtype=dtype).reshape(2, 5) / dtype(4)
                    decoded, actual = session.run(None, {"input": data})
                    np.testing.assert_array_equal(decoded, weights)
                    np.testing.assert_array_equal(
                        actual, (data.astype(np.float64) @ weights.astype(np.float64)).astype(dtype)
                    )


if __name__ == "__main__":
    unittest.main()
