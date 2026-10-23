# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

"""CPU-only tests for the external-weight loading benchmark; no ORT import needed."""

import argparse
import json
import shutil
import subprocess
import sys
import unittest
import uuid
from pathlib import Path

import benchmark_cuda_model_loading as benchmark
import numpy as np
import onnx


class TestBenchmarkHelpers(unittest.TestCase):
    def test_parse_logs_aggregates_only_successful_loads(self):
        stderr = """
[I:onnxruntime] CUDA external data loader: path=pinned bytes=4096
[I:onnxruntime] CUDA external data loader: path=pinned bytes=8192
[W:onnxruntime] CUDA external data loader: GDS failed; falling back to pinned
[I:onnxruntime] CUDA external data loader: path=gds bytes=4096
unrelated bytes=999 path=pageable
"""
        observed, warnings = benchmark.parse_loader_logs(stderr)
        self.assertEqual(observed, {"pinned": 12288, "gds": 4096})
        self.assertEqual(len(warnings), 1)

    def test_parse_mixed_python_utf8_and_native_utf16le_logs(self):
        native = (
            "[I:onnxruntime] CUDA external data loader: path=directstorage bytes=4096\r\n"
            "[W:onnxruntime] CUDA external data loader: DirectStorage failed; falling back to pinned\r\n"
            "[I:onnxruntime] CUDA external data loader: path=pinned bytes=8192\r\n"
        )
        python_warning = "UserWarning: failed to collect package version info\n"
        stderr = (
            python_warning.encode("utf-8")
            + native.encode("utf-16le")
            + b"[I:onnxruntime] CUDA external data loader: path=pinned bytes=4096\n"
        ).decode("utf-8")
        original = stderr
        observed, warnings = benchmark.parse_loader_logs(stderr)
        self.assertEqual(stderr, original)
        self.assertIn("\x00", stderr)
        self.assertEqual(observed, {"directstorage": 4096, "pinned": 12288})
        self.assertEqual(warnings, [native.splitlines()[1]])
        self.assertEqual(benchmark.classify_path("directstorage", observed, 16384), "mixed")
        self.assertEqual(benchmark.classify_path("directstorage", observed, 32768), "incomplete")

    def test_parse_native_routes_and_fallback_without_claiming_requested_path(self):
        for path in ("pageable", "pinned", "gds", "directstorage"):
            with self.subTest(path=path):
                text = f"[I:onnxruntime] CUDA external data loader: path={path} bytes=4096\n"
                observed, warnings = benchmark.parse_loader_logs(text.encode("utf-16le").decode("utf-8"))
                self.assertEqual(observed, {path: 4096})
                self.assertEqual(warnings, [])
                self.assertEqual(benchmark.classify_path(path, observed, 4096), "verified")
                if path != "directstorage":
                    self.assertEqual(benchmark.classify_path("directstorage", observed, 4096), "fallback")

    def test_parse_does_not_strip_arbitrary_nuls_to_manufacture_marker(self):
        text = "CUDA external data lo\x00ader: path=directstorage bytes=4096"
        observed, warnings = benchmark.parse_loader_logs(text)
        self.assertEqual(observed, {})
        self.assertEqual(warnings, [])
        self.assertEqual(benchmark.classify_path("directstorage", observed, 4096), "unverified")

    def test_classify_never_claims_verified_without_full_evidence(self):
        cases = [
            ("gds", {}, 4096, "unverified"),
            ("gds", {"gds": 4096}, 8192, "incomplete"),
            ("gds", {"gds": 8192}, 4096, "incomplete"),
            ("gds", {"pinned": 4096}, 4096, "fallback"),
            ("gds", {"gds": 4096, "pinned": 4096}, 8192, "mixed"),
            ("gds", {"gds": 4096}, 4096, "verified"),
            ("directstorage", {"directstorage": 4096}, 4096, "verified"),
            ("directstorage", {"pageable": 4096}, 4096, "fallback"),
            ("pageable", {"pageable": 4096}, 4096, "verified"),
            ("pinned", {"pinned": 4096}, 4096, "verified"),
            ("gds", {"gds": 0}, 4096, "unverified"),
        ]
        for expected, observed, size, status in cases:
            with self.subTest(expected=expected, observed=observed, size=size):
                self.assertEqual(benchmark.classify_path(expected, observed, size), status)

    def test_provider_options_use_same_device_and_explicit_routes(self):
        for platform_name, api in (("win32", "directstorage"), ("linux", "gds")):
            for path in benchmark.PATHS:
                with self.subTest(platform=platform_name, path=path):
                    options = benchmark.provider_options(path, 2, 8, platform_name)
                    self.assertEqual(options["device_id"], 2)
                    self.assertEqual(options["external_data_loader_reading_threads"], 0 if path == "pageable" else 8)
                    self.assertEqual(options[f"external_data_loader_use_{api}"], int(path == "direct"))
                    other = "gds" if api == "directstorage" else "directstorage"
                    self.assertEqual(options[f"external_data_loader_use_{other}"], 0)
        self.assertIsNone(benchmark.direct_path("darwin"))
        with self.assertRaises(ValueError):
            benchmark.provider_options("direct", 0, 4, "darwin")

    def test_summary_separates_fallback_from_direct(self):
        samples = [
            {"requested_path": "direct", "status": "verified", "observed_bytes": {"gds": 2**30}, "seconds": seconds}
            for seconds in (1.0, 2.0, 3.0)
        ]
        samples.append(
            {"requested_path": "direct", "status": "fallback", "observed_bytes": {"pinned": 2**30}, "seconds": 10.0}
        )
        verified, fallback = benchmark.summarize_samples(samples, 2**30)
        self.assertEqual(verified["count"], 3)
        self.assertEqual(verified["seconds"]["median"], 2)
        self.assertEqual(verified["seconds"]["mean"], 2)
        self.assertEqual(verified["seconds"]["stdev"], 1)
        self.assertAlmostEqual(verified["seconds"]["p95"], 2.9)
        self.assertEqual(verified["effective_gib_per_second"]["median"], 0.5)
        self.assertIn("end-to-end InferenceSession initialization time", verified["throughput_definition"])
        self.assertIn("not storage bandwidth", verified["throughput_definition"])
        self.assertEqual(fallback["count"], 1)
        self.assertEqual(fallback["seconds"]["stdev"], 0)
        self.assertEqual(fallback["effective_gib_per_second"]["median"], 0.1)
        json.dumps([verified, fallback], allow_nan=False)

    def test_worker_command_creates_fresh_process_with_equivalent_inputs(self):
        args = benchmark.create_parser().parse_args(
            ["--model", "a model.onnx", "--inputs", "inputs.npz", "--expected-outputs", "outputs.npz"]
        )
        args.external_data = ["a weight.bin"]
        args.evict_file_cache = True
        command = benchmark.worker_command(args, "pinned")
        self.assertEqual(command[0], sys.executable)
        self.assertEqual(Path(command[1]), Path(benchmark.__file__).resolve())
        parsed = benchmark.create_parser().parse_args(command[2:])
        self.assertTrue(parsed.worker)
        self.assertEqual(parsed.path, "pinned")
        self.assertEqual(parsed.model, args.model)
        self.assertEqual(parsed.inputs, args.inputs)
        self.assertEqual(parsed.expected_outputs, args.expected_outputs)
        self.assertEqual(parsed.device_id, args.device_id)
        self.assertEqual(parsed.external_data, args.external_data)
        self.assertTrue(parsed.evict_file_cache)

    def test_directstorage_summary_does_not_claim_zero_host_copy(self):
        sample = {
            "requested_path": "direct",
            "status": "verified",
            "observed_bytes": {"directstorage": 4096},
            "seconds": 1.0,
        }
        summary = benchmark.summarize_samples([sample], 4096)[0]
        description = summary["observed_path_descriptions"]["directstorage"]
        self.assertIn("host/upload staging", description)
        self.assertIn("not zero-host-copy", description)
        self.assertIn("shared D3D12/CUDA destination", description)

    def test_cli_help_does_not_import_ort(self):
        result = subprocess.run(
            [sys.executable, "-S", str(Path(benchmark.__file__).resolve()), "--help"],
            capture_output=True,
            text=True,
            check=True,
        )
        self.assertIn("--generate-model", result.stdout)
        self.assertIn("OS-managed", result.stdout)

    def test_cli_rejects_invalid_comparison_options(self):
        for extra in (["--reading-threads", "0"], ["--rtol", "nan"], ["--atol", "-1"]):
            result = subprocess.run(
                [sys.executable, "-S", str(Path(benchmark.__file__).resolve()), "--model", "unused.onnx", *extra],
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(result.returncode, 2, result.stderr)
            self.assertNotIn("ModuleNotFoundError", result.stderr)

    def test_numeric_bounds(self):
        self.assertEqual(benchmark.reading_thread_count("64"), 64)
        self.assertEqual(benchmark.reading_thread_count("0"), 0)
        for value in ("-1", "65"):
            with self.assertRaises(argparse.ArgumentTypeError):
                benchmark.reading_thread_count(value)
        with self.assertRaises(argparse.ArgumentTypeError):
            benchmark.positive_int("0")
        with self.assertRaises(argparse.ArgumentTypeError):
            benchmark.nonnegative_int("-1")

    def test_verification_rejects_wrong_values_names_shapes_types_and_nonfinite(self):
        reference = {"output": np.ones((1, 4), dtype=np.float32)}
        benchmark.verify_outputs(reference, reference, 0, 0)
        cases = [
            {"output": np.zeros((1, 4), dtype=np.float32)},
            {"wrong_name": reference["output"]},
            {"output": np.ones(4, dtype=np.float32)},
            {"output": np.ones((1, 4), dtype=np.float64)},
            {"output": np.full((1, 4), np.nan, dtype=np.float32)},
        ]
        for actual in cases:
            with self.subTest(actual=actual), self.assertRaises((ValueError, AssertionError)):
                benchmark.verify_outputs(actual, reference, 0, 0)


class TestGeneratedFixture(unittest.TestCase):
    def setUp(self):
        self.directory = Path.cwd() / f".benchmark_cuda_loading_test_{uuid.uuid4().hex}"
        self.directory.mkdir()
        self.addCleanup(shutil.rmtree, self.directory)

    def test_generated_weights_are_deterministic_aligned_and_correct(self):
        generated = [
            benchmark.generate_model(self.directory / name, weight_count=3, dimension=32, seed=123)
            for name in ("first", "second")
        ]
        for name in ("model.onnx", "weights.bin"):
            self.assertEqual(
                (self.directory / "first" / name).read_bytes(), (self.directory / "second" / name).read_bytes()
            )
        model_path, inputs_path, expected_path = generated[0]
        metadata = benchmark.model_metadata(model_path)
        self.assertEqual(metadata["external_weight_bytes"], 3 * 4096)
        self.assertEqual(metadata["external_tensor_count"], 3)
        self.assertTrue(metadata["offsets_and_lengths_4096_aligned"])
        metadata_only = onnx.load(model_path, load_external_data=False)
        for index, weight in enumerate(metadata_only.graph.initializer):
            external = {entry.key: entry.value for entry in weight.external_data}
            self.assertEqual(int(external["offset"]), index * 4096)
            self.assertEqual(int(external["length"]), 4096)
            self.assertEqual(weight.raw_data, b"")
        model = onnx.load(model_path)
        onnx.checker.check_model(model)
        with np.load(inputs_path, allow_pickle=False) as inputs, np.load(expected_path, allow_pickle=False) as expected:
            actual = {
                f"output_{index}": inputs["input"] @ onnx.numpy_helper.to_array(tensor)
                for index, tensor in enumerate(model.graph.initializer)
            }
            benchmark.verify_outputs(actual, dict(expected), 0, 0)
            actual["output_0"][0, 0] += 1
            with self.assertRaises(AssertionError):
                benchmark.verify_outputs(actual, dict(expected), 0, 0)

    def test_generation_refuses_overwrites_and_unaligned_dimensions(self):
        benchmark.generate_model(self.directory, 1, 32, 0)
        with self.assertRaises(FileExistsError):
            benchmark.generate_model(self.directory, 1, 32, 0)
        with self.assertRaises(ValueError):
            benchmark.generate_model(self.directory / "invalid", 1, 33, 0)

    def test_metadata_rejects_truncated_external_data(self):
        model_path, _, _ = benchmark.generate_model(self.directory, 1, 32, 0)
        (self.directory / "weights.bin").write_bytes(b"\0")
        with self.assertRaisesRegex(ValueError, "outside"):
            benchmark.model_metadata(model_path)


if __name__ == "__main__":
    unittest.main()
