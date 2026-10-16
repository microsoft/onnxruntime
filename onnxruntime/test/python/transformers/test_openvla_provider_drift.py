# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root.
# --------------------------------------------------------------------------

import hashlib
import io
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch
from zipfile import ZipFile

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "python/tools/transformers/models/openvla"))
from compare_provider_outputs import compare_captures, compare_outputs, exit_code, load_inputs, run_model


class TestProviderDrift(unittest.TestCase):
    def test_input_snapshot_supports_file_name_and_hashes_loaded_bytes(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "inputs.npz"
            array = np.ones(2, dtype=np.float32)
            stream = io.BytesIO()
            np.save(stream, array, allow_pickle=False)
            with ZipFile(path, "w") as archive:
                archive.writestr("file.npy", stream.getvalue())
            feeds, digest = load_inputs(path)
            np.testing.assert_array_equal(feeds["file"], array)
            self.assertEqual(digest, hashlib.sha256(path.read_bytes()).hexdigest())

    def test_duplicate_or_pickle_inputs_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            stream = io.BytesIO()
            np.save(stream, np.ones(2, dtype=np.float32), allow_pickle=False)
            duplicate = root / "duplicate.npz"
            with ZipFile(duplicate, "w") as archive:
                archive.writestr("x.npy", stream.getvalue())
                archive.writestr("x", stream.getvalue())
            with self.assertRaisesRegex(ValueError, "Duplicate"):
                load_inputs(duplicate)
            objects = root / "objects.npz"
            np.savez(objects, x=np.array(["not numeric"], dtype=object))
            with self.assertRaises(ValueError):
                load_inputs(objects)

    def test_combined_tolerance_and_coordinates(self):
        reference = np.array([[0, 10]], dtype=np.float32)
        actual = np.array([[0.0002, 10.0005]], dtype=np.float32)
        result = compare_outputs(reference, actual)
        self.assertEqual(result["failing_elements"], 1)
        self.assertEqual(result["first_failing_coordinates"], [[0, 0]])

    def test_signed_zero_is_not_bitwise_preserved(self):
        result = compare_outputs(np.array([0], np.float32), np.array([-0.0], np.float32))
        self.assertEqual(result["failing_elements"], 0)
        self.assertFalse(result["bitwise_equal"])

    def test_large_integers_are_not_rounded_before_comparison(self):
        result = compare_outputs(np.array([2**63 - 1], np.int64), np.array([2**63 - 2], np.int64))
        self.assertEqual(result["failing_elements"], 1)

    def test_empty_output_is_recorded(self):
        result = compare_outputs(np.empty((1, 0), np.float32), np.empty((1, 0), np.float32))
        self.assertEqual(result["elements"], 0)
        self.assertTrue(result["bitwise_equal"])

    def test_invalid_outputs_or_tolerances_raise(self):
        x = np.ones((2,), dtype=np.float32)
        for other in (x.astype(np.float64), x.reshape(1, 2), x * np.nan):
            with self.subTest(other=other), self.assertRaises(ValueError):
                compare_outputs(x, other)
        for tolerance in (-1, float("nan"), float("inf")):
            with self.subTest(tolerance=tolerance), self.assertRaises(ValueError):
                compare_outputs(x, x, atol=tolerance)

    def test_behavior_changing_capture_cannot_attribute_a_kernel(self):
        x = np.ones((2,), dtype=np.float32)
        changed = x.copy()
        changed[0] = np.nextafter(changed[0], np.float32(2))
        result = compare_captures({"file": x}, {"file": x}, {"file": changed, "tap": x}, {"file": x, "tap": x})
        self.assertFalse(result["instrumentation_preserved"])
        self.assertIsNone(result["captures"])
        self.assertEqual(exit_code(result), 3)

    def test_preserved_capture_keeps_additional_comparisons(self):
        x = np.ones((2,), dtype=np.float32)
        result = compare_captures(
            {"file": x},
            {"file": x},
            {"file": x, "tap": x},
            {"file": x, "tap": x + np.float32(0.01)},
        )
        self.assertTrue(result["instrumentation_preserved"])
        self.assertEqual(result["captures"]["tap"]["failing_elements"], 2)
        self.assertEqual(exit_code(result), 2)

    def test_missing_output_or_provider_run_is_rejected(self):
        x = np.ones((2,), dtype=np.float32)
        for arguments in (
            ({}, {}),
            ({"x": x}, {"y": x}),
            ({"x": x}, {"x": x}, {"x": x}, None),
            ({"x": x}, {"x": x}, {"tap": x}, {"tap": x}),
        ):
            with self.subTest(arguments=arguments), self.assertRaises(ValueError):
                compare_captures(*arguments)

    def test_unavailable_cuda_never_becomes_a_cpu_run(self):
        with (
            tempfile.TemporaryDirectory() as temporary,
            patch("compare_provider_outputs.ort.get_available_providers", return_value=["CPUExecutionProvider"]),
            self.assertRaisesRegex(RuntimeError, "unavailable"),
        ):
            run_model(Path("unused.onnx"), {}, "CUDAExecutionProvider", Path(temporary) / "run")

    def test_initialization_fallback_and_tf32_are_rejected(self):
        for actual_provider, options in (
            ("CPUExecutionProvider", {"CPUExecutionProvider": {}}),
            ("CUDAExecutionProvider", {"CUDAExecutionProvider": {"use_tf32": "1"}}),
        ):
            with self.subTest(provider=actual_provider), tempfile.TemporaryDirectory() as temporary:
                session = Mock()
                session.get_providers.return_value = [actual_provider]
                session.get_provider_options.return_value = options
                with (
                    patch(
                        "compare_provider_outputs.ort.get_available_providers", return_value=["CUDAExecutionProvider"]
                    ),
                    patch("compare_provider_outputs.ort.preload_dlls"),
                    patch("compare_provider_outputs.ort.InferenceSession", return_value=session),
                    self.assertRaises(RuntimeError),
                ):
                    run_model(Path("unused.onnx"), {}, "CUDAExecutionProvider", Path(temporary) / "run")
                session.run.assert_not_called()

    def test_real_cpu_session_preserves_external_initializers_and_named_output(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            model = helper.make_model(
                helper.make_graph(
                    [helper.make_node("MatMul", ["x", "weight"], ["file"])],
                    "test",
                    [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 2])],
                    [helper.make_tensor_value_info("file", TensorProto.FLOAT, [1, 2])],
                    [numpy_helper.from_array(np.eye(2, dtype=np.float32), name="weight")],
                ),
                opset_imports=[helper.make_opsetid("", 18)],
                ir_version=10,
            )
            path = root / "model.onnx"
            onnx.save_model(
                model,
                path,
                save_as_external_data=True,
                all_tensors_to_one_file=True,
                location="weights.bin",
                size_threshold=0,
            )
            before = (root / "weights.bin").read_bytes()
            x = np.ones((1, 2), dtype=np.float32)
            outputs, record = run_model(path, {"x": x}, "CPUExecutionProvider", root / "run")
            np.testing.assert_array_equal(outputs["file"], x)
            artifact = Path(record["outputs"]["file"]["path"])
            np.testing.assert_array_equal(np.load(artifact, allow_pickle=False), x)
            self.assertEqual((root / "weights.bin").read_bytes(), before)
            self.assertGreater(record["kernel_counts"]["CPUExecutionProvider"], 0)
            with self.assertRaises(FileExistsError):
                run_model(path, {"x": x}, "CPUExecutionProvider", root / "run")


if __name__ == "__main__":
    unittest.main()
