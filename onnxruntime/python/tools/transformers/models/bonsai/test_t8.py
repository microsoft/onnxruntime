# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import onnx
from benchmark_t8 import compare
from benchmark_t8_layouts import compare_variants
from convert_to_t8 import convert_model
from onnx import helper, numpy_helper
from t8_codec import decode_q2, encode_q2

import onnxruntime as ort


def pack_codes(codes):
    rows, width = codes.shape
    return (
        (codes.reshape(rows, -1, 4) << np.array([0, 2, 4, 6], dtype=np.uint8))
        .sum(axis=-1)
        .astype(np.uint8)
        .reshape(rows, width // 128, 32)
    )


class T8CodecTest(unittest.TestCase):
    def test_all_bytes_and_lanes(self):
        values = np.arange(256, dtype=np.uint16) % 243
        trits = ((values[None, :] // np.array([1, 3, 9, 27, 81], dtype=np.uint16)[:, None]) % 3).astype(np.uint8)
        packed = pack_codes((trits + 1) % 3)
        scales = np.ones((5, 2), dtype=np.float32)
        planes, s = encode_q2(packed, scales)
        self.assertEqual(planes.shape, (1, 8, 8))
        self.assertEqual(s.dtype, np.float16)
        np.testing.assert_array_equal(decode_q2(planes, 5), packed)
        for bit in range(8):
            for group in range(8):
                expected = sum(int((values[group * 32 + lane] >> bit) & 1) << lane for lane in range(32))
                self.assertEqual(int(planes[0, bit, group]), expected)

    def test_padded_rows_multiple_chunks_and_scales(self):
        rng = np.random.default_rng(7)
        rows, width = 647, 256
        packed = pack_codes(rng.integers(0, 3, (rows, width), dtype=np.uint8))
        scales = rng.integers(1, 32, (rows, width // 128)).astype(np.float32) / 128
        planes, s = encode_q2(packed, scales)
        np.testing.assert_array_equal(decode_q2(planes, rows), packed)
        restored_scales = s.transpose(0, 2, 1).reshape(-1, width // 128)
        np.testing.assert_array_equal(restored_scales[:rows], scales)
        np.testing.assert_array_equal(restored_scales[rows:], 0)
        self.assertLess(planes.nbytes + s.nbytes, packed.nbytes + scales.nbytes)

    def test_rejects_nonternary_and_inexact_scales(self):
        packed = np.full((5, 1, 32), 0x55, dtype=np.uint8)
        scales = np.ones((5, 1), dtype=np.float32)
        with self.assertRaisesRegex(ValueError, "code 3"):
            encode_q2(packed | 0x03, scales)
        with self.assertRaisesRegex(ValueError, "exactly representable"):
            encode_q2(packed, scales * np.float32(0.1))
        with self.assertRaisesRegex(ValueError, "finite"):
            encode_q2(packed, scales * np.nan)


class T8BenchmarkTest(unittest.TestCase):
    def test_layout_benchmark_checks_all_variants(self):
        sessions = {name: Mock() for name in ("current", "tiled", "masks")}
        for session in sessions.values():
            session.run.return_value = [np.array([[1.0, -0.5]], dtype=np.float32)]
        result = compare_variants(sessions, {}, warmup=0, iterations=3)
        self.assertTrue(result["bit_identical"])
        self.assertEqual(set(result["median_ms"]), set(sessions))
        for samples in result["samples_seconds"].values():
            self.assertEqual(len(samples), 3)
        sessions["masks"].run.return_value = [np.array([[1.0000001, -0.5]], dtype=np.float32)]
        with self.assertRaises(AssertionError):
            compare_variants(sessions, {}, warmup=0, iterations=1)

    def test_layout_benchmark_rejects_nonfinite_outputs(self):
        sessions = {name: Mock() for name in ("current", "masks")}
        sessions["current"].run.return_value = [np.array([[1.0]], dtype=np.float32)]
        sessions["masks"].run.return_value = [np.array([[np.nan]], dtype=np.float32)]
        with self.assertRaisesRegex(ValueError, "non-finite output"):
            compare_variants(sessions, {}, warmup=0, iterations=1)

    def test_rejects_matching_nonfinite_outputs(self):
        for value in (np.nan, np.inf, -np.inf):
            with self.subTest(value=value):
                session = Mock()
                session.run.return_value = [np.array([[value]], dtype=np.float32)]
                with self.assertRaisesRegex(ValueError, "non-finite benchmark output"):
                    compare(session, session, {}, warmup=0, iterations=1)

    def test_finite_outputs_and_timing_samples(self):
        session = Mock()
        session.run.return_value = [np.array([[0.0, 1.5, -0.75]], dtype=np.float32)]
        result = compare(session, session, {}, warmup=0, iterations=1)
        self.assertEqual(result["max_abs_error"], 0.0)
        self.assertEqual(len(result["samples_seconds"]["q2"]), 1)
        self.assertEqual(len(result["samples_seconds"]["t8"]), 1)


class T8ConversionTest(unittest.TestCase):
    def test_external_data_and_shared_initializers(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            packed = np.full((7, 1, 32), 0x24, dtype=np.uint8)
            scales = np.arange(1, 8, dtype=np.float32)[:, None] / 16
            tensors = [
                numpy_helper.from_array(packed, "B"),
                numpy_helper.from_array(scales, "S"),
                numpy_helper.from_array(np.full((7, 1), 0x55, dtype=np.uint8), "Z"),
            ]
            nodes = [
                helper.make_node("Identity", ["B"], ["B_alias"]),
                helper.make_node(
                    "MatMulNBits",
                    ["A", "B_alias", "S", "Z"],
                    ["Y1"],
                    domain="com.microsoft",
                    K=128,
                    N=7,
                    bits=2,
                    block_size=128,
                ),
                helper.make_node(
                    "MatMulNBits",
                    ["A", "B", "S", "Z"],
                    ["Y2"],
                    domain="com.microsoft",
                    K=128,
                    N=7,
                    bits=2,
                    block_size=128,
                ),
                helper.make_node("Add", ["Y1", "Y2"], ["Y"]),
            ]
            graph = helper.make_graph(
                nodes,
                "test",
                [helper.make_tensor_value_info("A", onnx.TensorProto.FLOAT, [1, 128])],
                [helper.make_tensor_value_info("Y", onnx.TensorProto.FLOAT, [1, 7])],
                tensors,
            )
            model = helper.make_model(
                graph,
                opset_imports=[helper.make_opsetid("", 21), helper.make_opsetid("com.microsoft", 1)],
                ir_version=10,
            )
            source, output = root / "q2.onnx", root / "t8.onnx"
            onnx.save_model(
                model,
                source,
                save_as_external_data=True,
                all_tensors_to_one_file=True,
                location="q2.data",
                size_threshold=0,
            )
            summary = convert_model(source, output)
            self.assertEqual(summary["unique_matrices"], 1)
            self.assertEqual(summary["nodes_converted"], 2)
            converted = onnx.load(output)
            self.assertEqual([node.op_type for node in converted.graph.node], ["MatMulTernary", "MatMulTernary", "Add"])
            self.assertEqual(len(converted.graph.initializer), 2)
            values = {tensor.name: numpy_helper.to_array(tensor) for tensor in converted.graph.initializer}
            np.testing.assert_array_equal(decode_q2(values["B.t8"], 7), packed)
            onnx.checker.check_model(str(output))
            with self.assertRaisesRegex(ValueError, "never overwrites"):
                convert_model(source, output)

    def test_copies_retained_tensors_when_data_file_names_match(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bias = np.arange(7, dtype=np.float32) - 3
            tensors = [
                numpy_helper.from_array(np.full((7, 1, 32), 0x24, dtype=np.uint8), "B"),
                numpy_helper.from_array(np.arange(1, 8, dtype=np.float32)[:, None] / 16, "S"),
                numpy_helper.from_array(np.full((7, 1), 0x55, dtype=np.uint8), "Z"),
                numpy_helper.from_array(bias, "bias"),
            ]
            nodes = [
                helper.make_node(
                    "MatMulNBits",
                    ["A", "B", "S", "Z"],
                    ["Y0"],
                    domain="com.microsoft",
                    K=128,
                    N=7,
                    bits=2,
                    block_size=128,
                ),
                helper.make_node("Add", ["Y0", "bias"], ["Y"]),
            ]
            graph = helper.make_graph(
                nodes,
                "test",
                [helper.make_tensor_value_info("A", onnx.TensorProto.FLOAT, [1, 128])],
                [helper.make_tensor_value_info("Y", onnx.TensorProto.FLOAT, [1, 7])],
                tensors,
            )
            model = helper.make_model(
                graph,
                opset_imports=[helper.make_opsetid("", 21), helper.make_opsetid("com.microsoft", 1)],
                ir_version=10,
            )
            (root / "q2").mkdir()
            source, output = root / "q2" / "model.onnx", root / "t8" / "model.onnx"
            onnx.save_model(
                model,
                source,
                save_as_external_data=True,
                all_tensors_to_one_file=True,
                location="model.onnx.data",
                size_threshold=0,
            )
            convert_model(source, output)
            values = {tensor.name: numpy_helper.to_array(tensor) for tensor in onnx.load(output).graph.initializer}
            np.testing.assert_array_equal(values["bias"], bias)

    def test_matrices_sharing_one_component_get_unique_names(self):
        rng = np.random.default_rng(0)
        tensors = [
            numpy_helper.from_array(pack_codes(rng.integers(0, 3, (7, 128), dtype=np.uint8)), "B1"),
            numpy_helper.from_array(pack_codes(rng.integers(0, 3, (7, 128), dtype=np.uint8)), "B2"),
            numpy_helper.from_array(np.arange(1, 8, dtype=np.float32)[:, None] / 16, "S1"),
            numpy_helper.from_array(np.arange(8, 15, dtype=np.float32)[:, None] / 16, "S2"),
            numpy_helper.from_array(np.full((7, 1), 0x55, dtype=np.uint8), "Z"),
        ]
        attrs = {"K": 128, "N": 7, "bits": 2, "block_size": 128}
        cases = {"shared scales": [("B1", "S1"), ("B2", "S1")], "shared B": [("B1", "S1"), ("B1", "S2")]}
        for case, pairs in cases.items():
            with self.subTest(case), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                nodes = [
                    helper.make_node("MatMulNBits", ["A", b, s, "Z"], [f"Y{i}"], domain="com.microsoft", **attrs)
                    for i, (b, s) in enumerate(pairs)
                ]
                graph = helper.make_graph(
                    nodes,
                    "test",
                    [helper.make_tensor_value_info("A", onnx.TensorProto.FLOAT, [1, 128])],
                    [helper.make_tensor_value_info(f"Y{i}", onnx.TensorProto.FLOAT, [1, 7]) for i in range(2)],
                    tensors,
                )
                model = helper.make_model(
                    graph,
                    opset_imports=[helper.make_opsetid("", 21), helper.make_opsetid("com.microsoft", 1)],
                    ir_version=10,
                )
                source, output = root / "q2.onnx", root / "t8.onnx"
                onnx.save_model(model, source)
                summary = convert_model(source, output)
                self.assertEqual(summary["unique_matrices"], 2)
                converted = onnx.load(output)
                self.assertEqual(len(converted.graph.initializer), 4)
                self.assertEqual(len({name for node in converted.graph.node for name in node.input[1:]}), 4)
                a = {"A": rng.standard_normal((1, 128)).astype(np.float32)}
                expected = ort.InferenceSession(str(source), providers=["CPUExecutionProvider"]).run(None, a)
                actual = ort.InferenceSession(str(output), providers=["CPUExecutionProvider"]).run(None, a)
                for want, got in zip(expected, actual, strict=True):
                    np.testing.assert_allclose(got, want, rtol=1e-5, atol=1e-5)

    def test_validates_zero_points_of_cached_matrices(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            tensors = [
                numpy_helper.from_array(np.full((7, 1, 32), 0x24, dtype=np.uint8), "B"),
                numpy_helper.from_array(np.arange(1, 8, dtype=np.float32)[:, None] / 16, "S"),
                numpy_helper.from_array(np.full((7, 1), 0x55, dtype=np.uint8), "Z1"),
                numpy_helper.from_array(np.zeros((7, 1), dtype=np.uint8), "Z0"),
            ]
            attrs = {"K": 128, "N": 7, "bits": 2, "block_size": 128}
            nodes = [
                helper.make_node("MatMulNBits", ["A", "B", "S", "Z1"], ["Y1"], domain="com.microsoft", **attrs),
                helper.make_node("MatMulNBits", ["A", "B", "S", "Z0"], ["Y2"], domain="com.microsoft", **attrs),
                helper.make_node("Add", ["Y1", "Y2"], ["Y"]),
            ]
            graph = helper.make_graph(
                nodes,
                "test",
                [helper.make_tensor_value_info("A", onnx.TensorProto.FLOAT, [1, 128])],
                [helper.make_tensor_value_info("Y", onnx.TensorProto.FLOAT, [1, 7])],
                tensors,
            )
            model = helper.make_model(
                graph,
                opset_imports=[helper.make_opsetid("", 21), helper.make_opsetid("com.microsoft", 1)],
                ir_version=10,
            )
            source = root / "q2.onnx"
            onnx.save_model(model, source)
            with self.assertRaisesRegex(ValueError, "every zero point must equal 1"):
                convert_model(source, root / "t8.onnx")


if __name__ == "__main__":
    unittest.main()
