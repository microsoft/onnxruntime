# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
"""Offline tests of native qualification tooling; no real models or downloads."""

import dataclasses
import importlib.util
import io
import json
import os
import runpy
import sys
import tempfile
import unittest
from contextlib import contextmanager, redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import onnx
from onnxruntime_test_python_phi3_demo import FakeAdapter, toy_assets
from onnxruntime_test_python_phi3_demo import demo as phi3
from onnxruntime_test_python_phi3_demo_assets import qualification_package
from onnxruntime_test_python_sd15_demo import FakeTokenizer
from onnxruntime_test_python_sd15_demo import demo as sd15
from onnxruntime_test_python_sd15_demo import toy_assets as sd_assets

from onnxruntime.capi import onnxruntime_pybind11_state as native

_PATH = Path(__file__).resolve().parents[3] / "samples/python/stateful_qualification.py"
_SPEC = importlib.util.spec_from_file_location("stateful_qualification", _PATH)
assert _SPEC is not None and _SPEC.loader is not None
qual = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = qual
with mock.patch.object(sys, "path", [str(_PATH.parent), *sys.path]):
    _SPEC.loader.exec_module(qual)


@contextmanager
def phi_case(budget=1):
    with toy_assets() as (root, _, _):
        package = phi3.build_package(root, "cpu", max_new_tokens=budget)
        with mock.patch.object(phi3, "LocalTokenizer", return_value=FakeAdapter()):
            yield qual.load_case("phi3", package), package


def fake_result(tokens=None, *, invocations=1, generated=1, completed=True):
    value = np.array([2], dtype=np.int64) if tokens is None else tokens
    info = SimpleNamespace(
        completed=completed, invocations=invocations, iterations=1, generated_tokens=generated, exit_reasons=()
    )
    return SimpleNamespace(
        get_info=lambda: info,
        get_outputs=lambda: {
            "tokens": SimpleNamespace(numpy=lambda: value),
            "image": SimpleNamespace(numpy=lambda: np.zeros((1, 512, 512, 3), dtype=np.float32)),
            "unsafe": SimpleNamespace(numpy=lambda: np.array([True])),
        },
    )


@unittest.skipUnless(sys.platform == "linux", "Qualification measurement tooling targets Linux")
@unittest.skipUnless(native._stateful.available, "Stateful execution is unavailable in this build")
class TestStatefulQualification(unittest.TestCase):
    def test_statistics_counts_percentiles_and_invalid_samples(self):
        actual = qual.statistics([5, 1, 3, 2, 4])
        self.assertEqual(actual["samples_seconds"], [5, 1, 3, 2, 4])
        self.assertEqual(actual["median_seconds"], 3)
        self.assertEqual(actual["p95_seconds"], 5)
        self.assertEqual(actual["count"], 5)
        self.assertEqual(qual.statistics([2, 1])["median_seconds"], 1.5)
        self.assertEqual(qual.statistics([1e308, 1e308])["median_seconds"], 1e308)
        for values in ([], [[1]], [True], ["1"], [1j], [0], [-1], [float("nan")], [float("inf")]):
            with self.subTest(values=values), self.assertRaises(ValueError):
                qual.statistics(values)

    def test_cuda_completion_is_checked_not_enqueue_only(self):
        qual.synchronizer(False)()
        function = mock.Mock(return_value=0)
        with mock.patch.object(qual.ctypes, "CDLL", return_value=SimpleNamespace(cudaDeviceSynchronize=function)):
            synchronize = qual.synchronizer(True)
        synchronize()
        self.assertEqual(function.argtypes, [])
        self.assertIs(function.restype, qual.ctypes.c_int)
        function.return_value = 17
        with self.assertRaisesRegex(RuntimeError, "17"):
            synchronize()
        with mock.patch.object(qual.ctypes, "CDLL", side_effect=OSError("missing")), self.assertRaises(OSError):
            qual.synchronizer(True)

    def test_native_cpu_latency_steps_resources_and_zero_budget(self):
        for budget in (0, 1, 16):
            with self.subTest(budget=budget), phi_case(budget) as (case, _), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                latency = qual.measure(case, "latency", root)
                self.assertEqual(len(latency["warmups"]), 2)
                self.assertEqual(len(latency["measured"]), 5)
                self.assertEqual(latency["steady"]["count"], 5)
                self.assertEqual(latency["cold"]["units"], min(budget, 2))
                if budget == 0:
                    self.assertIsNone(latency["throughput"])
                    self.assertEqual(latency["throughput_applicability"], "no generated tokens")
                else:
                    self.assertGreater(latency["throughput"], 0)
                phases = qual.measure(case, "steps", root)
                self.assertTrue(phases["steps"])
                first = phases["step_observed_time_to_first_token_seconds"]
                self.assertTrue(first is None if budget == 0 else first > 0)
                resources = qual.measure(case, "resources", root)
                self.assertEqual(len(resources["cycles"]), 2)
                self.assertGreater(resources["before_setup"]["rss_bytes"], 0)
                self.assertGreater(resources["after_close"]["process_peak_rss_bytes"], 0)

    def test_case_metadata_validation_and_cuda_adapter_arguments(self):
        for model in ("phi3", "sd15"):
            config = (
                {"target": "cuda", "device_id": 0, "max_new_tokens": 16}
                if model == "phi3"
                else {"device_id": 0, "ddim": {"num_inference_steps": 20, "guidance_scale": 7.5}}
            )
            module = mock.Mock()
            module.package_configuration.return_value = config
            module.configuration.return_value = config
            module.LocalTokenizer.return_value = FakeAdapter() if model == "phi3" else FakeTokenizer()
            execution = mock.Mock()
            module.create_execution.return_value = (execution, FakeAdapter(), 2) if model == "phi3" else execution
            with (
                mock.patch.object(qual.importlib, "import_module", return_value=module),
                mock.patch.object(qual.ort, "get_build_info", return_value="cuda-plugin-ep=1"),
                mock.patch.object(qual, "synchronizer", return_value=mock.Mock()),
            ):
                case = qual.load_case(model, Path("/package"))
                actual, decoder, _ = case.create()
                self.assertIs(actual, execution)
                self.assertTrue(case.cuda)
                if model == "phi3":
                    self.assertEqual(decoder([2, 3]), "2")
                else:
                    self.assertIsNone(decoder)
                    self.assertEqual(module.create_execution.call_args.kwargs["seed"], 42)
                config["device_id"] = 1
                with self.assertRaisesRegex(ValueError, "ordinal"):
                    qual.load_case(model, Path("/package"))
                config["device_id"] = 0
                if model == "phi3":
                    config["max_new_tokens"] = 2
                else:
                    config["ddim"]["num_inference_steps"] = 2
                with self.assertRaises(ValueError):
                    qual.load_case(model, Path("/package"))
                if model == "phi3":
                    config["max_new_tokens"] = 16
                else:
                    config["ddim"]["num_inference_steps"] = 20
                    config["ddim"]["guidance_scale"] = 1.0
                    with self.assertRaises(ValueError):
                        qual.load_case(model, Path("/package"))
                    config["ddim"]["guidance_scale"] = 7.5
                with (
                    mock.patch.object(qual.ort, "get_build_info", return_value="cpu"),
                    self.assertRaisesRegex(ValueError, "CUDA-plugin"),
                ):
                    qual.load_case(model, Path("/package"))
        with self.assertRaises(ValueError):
            qual.load_case("other", Path("/package"))

    def test_gpu_memory_scope_missing_rows_and_explicit_failures(self):
        with (
            mock.patch.object(qual.os, "getpid", return_value=100),
            mock.patch.object(qual.subprocess, "check_output", return_value="100, 4\n101, N/A\n100, 2\n"),
        ):
            result = qual.gpu_memory()
            self.assertTrue(result["observed"])
            self.assertEqual(result["used_bytes"], 6 * 1024 * 1024)
        with mock.patch.object(qual.subprocess, "check_output", return_value=""):
            self.assertIsNone(qual.gpu_memory()["used_bytes"])
        with (
            mock.patch.object(qual.os, "getpid", return_value=100),
            mock.patch.object(qual.subprocess, "check_output", return_value="100, N/A\n"),
            self.assertRaises(ValueError),
        ):
            qual.gpu_memory()
        with (
            mock.patch.object(qual.subprocess, "check_output", side_effect=OSError("missing")),
            self.assertRaises(OSError),
        ):
            qual.gpu_memory()

    def test_glibc_and_process_memory_scope_and_platform_failure(self):
        values = qual._MallInfo()
        values.uordblks, values.hblkhd, values.hblks = 123, 456, 2
        function = mock.Mock(return_value=values)
        with (
            mock.patch.object(qual.ctypes, "CDLL", return_value=SimpleNamespace(mallinfo2=function)),
            mock.patch.object(qual, "gpu_memory", return_value={"observed": True, "used_bytes": 100}),
        ):
            value = qual.memory_snapshot(True)
            self.assertEqual(value["glibc_in_use_bytes"], 123)
            self.assertEqual(value["glibc_mmapped_bytes"], 456)
            self.assertEqual(value["glibc_mmap_regions"], 2)
            self.assertEqual(value["gpu"]["used_bytes"], 100)
        with mock.patch.object(qual.sys, "platform", "win32"), self.assertRaisesRegex(RuntimeError, "Linux"):
            qual.memory_snapshot(False)

    def test_output_accounting_native_dispatch_and_failure_are_not_defaults(self):
        with phi_case() as (case, _), tempfile.TemporaryDirectory() as temporary:
            stage = Path(temporary)
            execution = mock.Mock()
            execution.get_dispatch_info.return_value = SimpleNamespace(
                origin=SimpleNamespace(name="NATIVE_EP"), provider_calls=1, cpu_calls=0
            )
            record = qual.describe(dataclasses.replace(case, cuda=True), execution, str, fake_result(), stage, "x")
            self.assertEqual(record["units"], 1)
            execution.get_dispatch_info.return_value.cpu_calls = 1
            with self.assertRaisesRegex(RuntimeError, "native sampler"):
                qual.describe(dataclasses.replace(case, cuda=True), execution, str, fake_result(), stage, "x")
            for tokens in (
                np.array([-1], dtype=np.int64),
                np.array([2], dtype=np.float32),
                np.array([[2]], dtype=np.int64),
            ):
                with self.subTest(tokens=tokens), self.assertRaisesRegex(ValueError, "INT64"):
                    qual.describe(case, execution, str, fake_result(tokens), stage, "x")
            for result in (
                fake_result(generated=0),
                fake_result(invocations=2),
                fake_result(tokens=np.array([2, 3], dtype=np.int64), generated=2),
            ):
                with self.subTest(info=result.get_info()), self.assertRaisesRegex(ValueError, "accounting"):
                    qual.describe(case, execution, str, result, stage, "x")
            with self.assertRaisesRegex(ValueError, "decoder"):
                qual.describe(case, execution, None, fake_result(), stage, "x")
            with self.assertRaisesRegex(RuntimeError, "complete"):
                qual.describe(case, execution, str, fake_result(completed=False), stage, "x")
            with self.assertRaisesRegex(ValueError, "phase"):
                qual.measure(case, "other", stage)
            sd_case = dataclasses.replace(
                case, model="sd15", configuration={"ddim": {"num_inference_steps": 20, "guidance_mode": "batched"}}
            )
            with self.assertRaisesRegex(ValueError, "accounting"):
                qual.describe(sd_case, execution, None, fake_result(), stage, "x")
            with self.assertRaisesRegex(ValueError, "blocked"):
                qual.describe(sd_case, execution, None, fake_result(invocations=25), stage, "x")
            self.assertEqual(list(stage.iterdir()), [])

    def test_resource_visibility_and_step_bound_fail_explicitly_and_close(self):
        with tempfile.TemporaryDirectory() as temporary:
            execution = mock.MagicMock()
            case = qual.Case("phi3", {"max_new_tokens": 0}, True, mock.Mock(), lambda: (execution, str, 0.0))
            with (
                mock.patch.object(qual, "memory_snapshot", return_value={"gpu": {"observed": False}}),
                mock.patch.object(qual, "completed_run", return_value={}) as completed,
                self.assertRaisesRegex(RuntimeError, "observable"),
            ):
                qual.measure(case, "resources", Path(temporary))
            self.assertEqual(completed.call_count, 2)
            execution.__exit__.assert_called_once()
            execution.__exit__.reset_mock()
            execution.step.return_value.__enter__.return_value.get_info.return_value = SimpleNamespace(
                component_name="prefill", executed=True, generated_token=None, completed=False
            )
            with self.assertRaisesRegex(RuntimeError, "bound"):
                qual.measure(case, "steps", Path(temporary))
            self.assertEqual(execution.step.call_count, 2)
            execution.__exit__.assert_called_once()
            execution.__exit__.reset_mock()
            execution.run.side_effect = RuntimeError("native failure requiring Reset")
            with self.assertRaisesRegex(RuntimeError, "requiring Reset"):
                qual.measure(case, "latency", Path(temporary))
            execution.__exit__.assert_called_once()

    @unittest.skipUnless(importlib.util.find_spec("PIL") is not None, "SD output fixtures require optional Pillow")
    def test_real_cpu_sd_fixtures_measure_both_guidance_modes_and_safety(self):
        for sequential in (False, True):
            with self.subTest(sequential=sequential), sd_assets() as root:
                package = sd15.build_package(root, steps=20, sequential=sequential)
                case = qual.Case(
                    "sd15",
                    sd15.configuration(package),
                    False,
                    lambda: None,
                    lambda package=package: (
                        sd15.create_execution(package, qual.SD15_PROMPT, adapter=FakeTokenizer()),
                        None,
                        0.0,
                    ),
                )
                stage = root / "measurements"
                stage.mkdir()
                latency = qual.measure(case, "latency", stage)
                self.assertEqual(latency["cold"]["invocations"], 45 if sequential else 25)
                self.assertEqual(latency["throughput_units"], "images/second")
                self.assertTrue(all(not row["unsafe"] for row in latency["measured"]))
                self.assertEqual(len(list(stage.glob("*.png"))), 8)
                steps = root / "step-measurements"
                steps.mkdir()
                self.assertEqual(qual.measure(case, "steps", steps)["result"]["units"], 1)

    def test_report_publication_is_exclusive_and_rolls_back_partial_links(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            stage = root / "stage"
            stage.mkdir()
            (stage / "sample.png").write_bytes(b"fixture")
            output = root / "result.json"
            qual.publish_report(output, {"ok": True}, stage)
            self.assertEqual(json.loads(output.read_text()), {"ok": True})
            self.assertEqual((root / "sample.png").read_bytes(), b"fixture")
            with self.assertRaises(FileExistsError):
                qual.publish_report(output, {}, stage)
            (root / "sample.png").unlink()
            with self.assertRaises(FileExistsError):
                qual.publish_report(output, {}, stage)
            self.assertFalse((root / "sample.png").exists())
            self.assertEqual(json.loads(output.read_text()), {"ok": True})
            with self.assertRaises(ValueError):
                qual.publish_report(root / "bad.json", {"bad": float("nan")}, stage)
            self.assertFalse((root / "bad.json").exists())

    def test_main_entry_cleanup_existing_outputs_and_native_errors(self):
        with phi_case() as (case, package), tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / "run.json"
            args = ["--model", "phi3", "--package", str(package), "--phase", "latency", "--output", str(output)]
            with mock.patch.object(qual, "load_case", return_value=case), redirect_stdout(io.StringIO()):
                qual.main(args)
            self.assertEqual(json.loads(output.read_text())["steady"]["count"], 5)
            self.assertFalse(list(root.glob(".qualification-*")))
            with self.assertRaises(FileExistsError):
                qual.main(args)
            with (
                mock.patch.object(qual, "measure", side_effect=RuntimeError("native error")),
                mock.patch.object(qual.sys, "argv", [str(_PATH), *args[:-1], str(root / "failed.json")]),
                mock.patch.object(qual.sys, "path", [str(_PATH.parent), *sys.path]),
                mock.patch("stateful_qualification.load_case", return_value=case),
                self.assertRaisesRegex(RuntimeError, "native error"),
            ):
                qual.main()
            self.assertFalse(list(root.glob(".qualification-*")))
            self.assertFalse((root / "failed.json").exists())
            with (
                mock.patch.object(sys, "argv", [str(_PATH), *args[:-1], str(root / "entry.json")]),
                mock.patch.object(sys, "path", [str(_PATH.parent), *sys.path]),
                mock.patch.object(phi3, "LocalTokenizer", return_value=FakeAdapter()),
                redirect_stdout(io.StringIO()),
            ):
                runpy.run_path(str(_PATH), run_name="__main__")
            self.assertTrue((root / "entry.json").is_file())

    def test_existing_tiny_native_examples_have_unit_coverage(self):
        example = _PATH.with_name("stateful_execution.py")
        with mock.patch.object(sys, "argv", [str(example)]), redirect_stdout(io.StringIO()) as output:
            runpy.run_path(str(example), run_name="__main__")
        self.assertIn("Feedback: 5.0", output.getvalue())
        spec = importlib.util.spec_from_file_location("tiny_example", example)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        graph = onnx.helper.make_graph([], "empty", [], [])
        with mock.patch.object(module.ort, "InferenceSession") as create:
            module.session(graph, "CUDAExecutionProvider")
        options = create.call_args.args[1]
        self.assertEqual(options.get_session_config_entry("session.disable_cpu_ep_fallback"), "1")
        self.assertEqual(create.call_args.kwargs["providers"], ["CUDAExecutionProvider"])
        self.assertFalse(create.call_args.kwargs["enable_fallback"])

    @unittest.skipUnless(importlib.util.find_spec("PIL") is not None, "SD package fixtures require optional Pillow")
    def test_isolated_real_fixture_packages_preserve_existing_manifests(self):
        sd_assets_module = importlib.import_module("onnxruntime_test_python_sd15_demo_assets")

        with toy_assets() as (root, _, _), tempfile.TemporaryDirectory() as temporary:
            source = phi3.build_package(root, "cpu")
            before = (source / "manifest.json").read_bytes()
            with mock.patch.dict(os.environ, {"ORT_STATEFUL_QUALIFICATION_OUTPUT": temporary}):
                variant = qualification_package(root, "cpu", 0)
            self.assertEqual(variant, Path(temporary) / "cpu-0")
            self.assertEqual(phi3.package_configuration(variant)["max_new_tokens"], 0)
            self.assertEqual((source / "manifest.json").read_bytes(), before)
            with (
                mock.patch.dict(os.environ, {"ORT_STATEFUL_QUALIFICATION_OUTPUT": ""}),
                self.assertRaisesRegex(ValueError, "empty"),
            ):
                qualification_package(root, "cpu", 0)
        with sd_assets() as root, tempfile.TemporaryDirectory() as temporary:
            source = sd15.build_package(root)
            before = (source / "manifest.json").read_bytes()
            with mock.patch.dict(os.environ, {"ORT_STATEFUL_QUALIFICATION_OUTPUT": temporary}):
                variant = sd_assets_module.sequential_package(root)
            self.assertEqual(variant.parent, Path(temporary))
            self.assertEqual(sd15.configuration(variant)["ddim"]["guidance_mode"], "sequential")
            self.assertEqual((source / "manifest.json").read_bytes(), before)
            with (
                mock.patch.dict(os.environ, {"ORT_STATEFUL_QUALIFICATION_OUTPUT": " "}),
                self.assertRaisesRegex(ValueError, "empty"),
            ):
                sd_assets_module.qualification_root(root)


if __name__ == "__main__":
    unittest.main()
