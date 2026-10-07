# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import contextlib
import gc
import io
import os
import subprocess
import sys
import tempfile
import time
import types
import unittest
from pathlib import Path
from typing import ClassVar
from unittest.mock import MagicMock, patch

import numpy as np
from onnx import TensorProto, helper, numpy_helper, save

import onnxruntime as ort
from onnxruntime.tools import fpa_intb_tune as tune


class TestFpAIntBTune(unittest.TestCase):
    signature: ClassVar[dict[str, str]] = {
        "device_name": "Test GPU",
        "sm": "80",
        "cuda_runtime": "13040",
        "ort_version": "1.28.0",
    }

    def write_cache(self, prefix, **overrides):
        header = {
            **self.signature,
            "ort_cuda_gemm_tactic_cache": "v1",
            "table": "matmulnbits_fpa_intb",
            "tactic_selection_version": "2",
            **overrides,
        }
        path = Path(str(prefix) + tune._CACHE_TABLE_SUFFIX)
        path.write_text(
            "".join(f"# {k}\t{v}\n" for k, v in header.items()) + "n_16b\tm_bucket\tvalid_config\n16\t1\t1\n",
            encoding="utf-8",
        )
        return path

    def test_cpu_fallback_rejects_stale_output(self):
        with tempfile.TemporaryDirectory() as directory:
            prefix = str(Path(directory) / "cache")
            path = self.write_cache(prefix, device_name="Stale GPU")
            original = path.read_bytes()
            session = MagicMock()
            session.get_providers.return_value = ["CPUExecutionProvider"]
            with (
                patch.object(ort, "get_available_providers", return_value=["CUDAExecutionProvider"]),
                patch.object(ort, "InferenceSession", return_value=session) as create,
                self.assertRaisesRegex(RuntimeError, "did not activate"),
            ):
                tune.tune("model.onnx", prefix, [1], False)
            self.assertFalse(create.call_args.kwargs["enable_fallback"])
            self.assertEqual(original, path.read_bytes())

    def test_valid_existing_hit_succeeds_without_rewrite(self):
        with tempfile.TemporaryDirectory() as directory:
            prefix = str(Path(directory) / "cache")
            path = self.write_cache(prefix)
            timestamp = path.stat().st_mtime_ns
            session = MagicMock()
            session.get_providers.return_value = ["CUDAExecutionProvider"]
            session.get_provider_options.return_value = {"CUDAExecutionProvider": {"device_id": "0"}}

            def create_session(_model, options, **_kwargs):
                self.write_cache(options.get_session_config_entry("ep.cuda.gemm_tactic_cache_tuning_results_prefix"))
                return session

            with (
                patch.object(ort, "get_available_providers", return_value=["CUDAExecutionProvider"]),
                patch.object(ort, "InferenceSession", side_effect=create_session),
                patch.object(tune, "_current_signature", return_value=self.signature),
            ):
                self.assertEqual(tune.tune("model.onnx", prefix, [1], False), str(path))
            session.disable_fallback.assert_called_once()
            self.assertEqual(timestamp, path.stat().st_mtime_ns)

    def test_unrelated_existing_cache_does_not_report_success(self):
        with tempfile.TemporaryDirectory() as directory:
            prefix = str(Path(directory) / "cache")
            path = self.write_cache(prefix)
            original = path.read_bytes()
            session = MagicMock()
            session.get_providers.return_value = ["CUDAExecutionProvider"]
            session.get_provider_options.return_value = {"CUDAExecutionProvider": {"device_id": "0"}}
            with (
                patch.object(ort, "get_available_providers", return_value=["CUDAExecutionProvider"]),
                patch.object(ort, "InferenceSession", return_value=session),
                patch.object(tune, "_current_signature", return_value=self.signature),
                self.assertRaisesRegex(RuntimeError, "did not select any successful"),
            ):
                tune.tune("unrelated.onnx", prefix, [1], False)
            self.assertEqual(original, path.read_bytes())

    def test_cpu_package_with_cuda_plugin_reports_metadata_limitation(self):
        with (
            patch.dict(sys.modules, {"onnxruntime.capi.build_and_package_info": types.SimpleNamespace()}),
            self.assertRaisesRegex(RuntimeError, "CPU packages with a registered CUDA plugin are not supported"),
        ):
            tune._current_signature(0)

    def test_selected_model_entries_must_reach_output_cache(self):
        with tempfile.TemporaryDirectory() as directory:
            prefix = str(Path(directory) / "cache")
            path = self.write_cache(prefix)
            session = MagicMock()
            session.get_providers.return_value = ["CUDAExecutionProvider"]
            session.get_provider_options.return_value = {"CUDAExecutionProvider": {"device_id": "0"}}

            def create_session(_model, options, **_kwargs):
                report = self.write_cache(
                    options.get_session_config_entry("ep.cuda.gemm_tactic_cache_tuning_results_prefix")
                )
                report.write_text(report.read_text().replace("16\t1\t1", "32\t1\t1"), encoding="utf-8")
                return session

            with (
                patch.object(ort, "get_available_providers", return_value=["CUDAExecutionProvider"]),
                patch.object(ort, "InferenceSession", side_effect=create_session),
                patch.object(tune, "_current_signature", return_value=self.signature),
                self.assertRaisesRegex(RuntimeError, "not persisted"),
            ):
                tune.tune("model.onnx", prefix, [1], False)
            self.assertTrue(path.exists())

    def test_incompatible_or_missing_cache_is_an_error(self):
        with tempfile.TemporaryDirectory() as directory:
            prefix = str(Path(directory) / "cache")
            for field in ("device_name", "sm", "cuda_runtime", "ort_version", "tactic_selection_version"):
                with self.subTest(field=field):
                    path = self.write_cache(prefix, **{field: "stale"})
                    with self.assertRaisesRegex(RuntimeError, "Inapplicable"):
                        tune._summarize_cache(str(path), self.signature)
            path.unlink()
            with self.assertRaisesRegex(RuntimeError, "No tactic cache"):
                tune._summarize_cache(str(path), self.signature)

    def test_signature_uses_native_ort_version(self):
        class FakeDriver:
            def __getattr__(self, _):
                return lambda *args: 0

        fake_build_info = types.SimpleNamespace(cuda_version="13.0")
        with (
            patch.dict(sys.modules, {"onnxruntime.capi.build_and_package_info": fake_build_info}),
            patch.object(tune.ctypes, "CDLL", return_value=FakeDriver(), create=True),
            patch.object(tune.ctypes, "WinDLL", return_value=FakeDriver(), create=True),
            patch.object(tune.ort, "get_version_string", return_value="1.28.0"),
            patch.object(tune.ort, "__version__", "1.28.0.dev20260101"),
        ):
            signature = tune._current_signature(0)

        self.assertEqual(signature["ort_version"], "1.28.0")

    def test_bf16_dummy_feed_uses_typed_ortvalue(self):
        model = helper.make_model(
            helper.make_graph(
                [helper.make_node("Identity", ["x"], ["y"])],
                "bf16",
                [helper.make_tensor_value_info("x", TensorProto.BFLOAT16, ["m", 4])],
                [helper.make_tensor_value_info("y", TensorProto.BFLOAT16, ["m", 4])],
            ),
            opset_imports=[helper.make_opsetid("", 21)],
            ir_version=10,
        )
        session = ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
        feeds = tune._make_dummy_inputs(session, 3)
        self.assertEqual(feeds["x"].element_type(), TensorProto.BFLOAT16)
        outputs = session.run_with_ort_values(None, feeds)
        self.assertEqual(outputs[0].shape(), [3, 4])
        self.assertEqual(outputs[0].element_type(), TensorProto.BFLOAT16)


def _lock_file(file):
    if sys.platform == "win32":
        import msvcrt  # noqa: PLC0415

        file.seek(0)
        msvcrt.locking(file.fileno(), msvcrt.LK_LOCK, 1)
    else:
        import fcntl  # noqa: PLC0415

        fcntl.flock(file, fcntl.LOCK_EX)


def _unlock_file(file):
    if sys.platform == "win32":
        import msvcrt  # noqa: PLC0415

        file.seek(0)
        msvcrt.locking(file.fileno(), msvcrt.LK_UNLCK, 1)
    else:
        import fcntl  # noqa: PLC0415

        fcntl.flock(file, fcntl.LOCK_UN)


def _gpu_worker(model, prefix, output, chunk="0"):
    plugin = os.environ.get("ORT_CUDA_PLUGIN_PATH")
    if plugin:
        ort.register_execution_provider_library("CUDAExecutionProvider", plugin)
    ort.set_default_logger_severity(1)
    options = tune._make_session_options(prefix, [1, 64])
    if chunk != "0":
        os.environ["ORT_MATMULNBITS_FORCE_CHUNKED"] = "1"
        options.add_session_config_entry("ep.cuda.matmul_nbits_m_chunk_size", chunk)
    options.log_severity_level = 1
    session = ort.InferenceSession(model, options, providers=["CUDAExecutionProvider"], enable_fallback=False)
    if "CUDAExecutionProvider" not in session.get_providers():
        raise RuntimeError("CUDA is not active")
    session.disable_fallback()
    outputs = []
    for m in (1, 64, 65):  # 65 exercises lazy bucket 128 and its teardown flush.
        inputs = np.arange(m * 128, dtype=np.float32).reshape(m, 128) % 17 / 17
        outputs.append(session.run(None, {"A": inputs.astype(np.float16)})[0])
    np.savez(output, *outputs)
    Path(output + ".ready").touch()
    del session
    gc.collect()


@unittest.skipUnless(os.environ.get("ORT_RUN_CUDA_TACTIC_CACHE_TESTS") == "1", "opt-in CUDA cache integration")
class TestFpAIntBCacheCuda(unittest.TestCase):
    def test_fresh_process_reuse_repair_and_lock_contention(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rng = np.random.default_rng(42)
            weights = rng.integers(0, 256, size=(64, 4, 16), dtype=np.uint8)
            scales = np.full((64, 4), 0.125, dtype=np.float16)
            model = helper.make_model(
                helper.make_graph(
                    [
                        helper.make_node(
                            "MatMulNBits",
                            ["A", "B", "scales"],
                            ["Y"],
                            domain="com.microsoft",
                            K=128,
                            N=64,
                            bits=4,
                            block_size=32,
                        )
                    ],
                    "cache",
                    [helper.make_tensor_value_info("A", TensorProto.FLOAT16, ["m", 128])],
                    [helper.make_tensor_value_info("Y", TensorProto.FLOAT16, ["m", 64])],
                    [numpy_helper.from_array(weights, "B"), numpy_helper.from_array(scales, "scales")],
                ),
                opset_imports=[helper.make_opsetid("", 21), helper.make_opsetid("com.microsoft", 1)],
                ir_version=10,
            )
            model_path = root / "model.onnx"
            save(model, model_path)
            prefix = str(root / "cache")

            def command(output, chunk=0):
                return [sys.executable, __file__, "--gpu-worker", str(model_path), prefix, str(output), str(chunk)]

            def run(name, chunk=0):
                output = root / f"{name}.npz"
                result = subprocess.run(
                    command(output, chunk), check=False, capture_output=True, text=True, timeout=180
                )
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                with np.load(output) as arrays:
                    return {key: arrays[key] for key in arrays.files}, result.stderr

            cold, _ = run("cold")
            cache_path = Path(prefix + tune._CACHE_TABLE_SUFFIX)
            self.assertTrue(cache_path.exists())
            original = cache_path.read_bytes()
            warm, log = run("warm")
            self.assertIn("validated fpA_intB tactics from", log)
            self.assertEqual(original, cache_path.read_bytes())
            for key in cold:
                np.testing.assert_allclose(warm[key], cold[key], rtol=1e-3, atol=1e-3)

            # Exercise the actual tool's session-specific evidence, not just its file parser.
            if not os.environ.get("ORT_CUDA_PLUGIN_PATH"):
                args = [
                    sys.executable,
                    "-m",
                    "onnxruntime.tools.fpa_intb_tune",
                    "--model",
                    str(model_path),
                    "--output-prefix",
                    prefix,
                    "--m-values",
                    "1,64",
                    "--no-inference",
                ]
                result = subprocess.run(args, check=False, capture_output=True, text=True, timeout=180)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                self.assertIn("selected this run:", result.stdout)
                self.assertEqual(original, cache_path.read_bytes())

                unrelated = helper.make_model(
                    helper.make_graph(
                        [helper.make_node("Identity", ["A"], ["Y"])],
                        "unrelated",
                        [helper.make_tensor_value_info("A", TensorProto.FLOAT, [1])],
                        [helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1])],
                    ),
                    opset_imports=[helper.make_opsetid("", 21)],
                    ir_version=10,
                )
                unrelated_path = root / "unrelated.onnx"
                save(unrelated, unrelated_path)
                args[args.index("--model") + 1] = str(unrelated_path)
                result = subprocess.run(args, check=False, capture_output=True, text=True, timeout=180)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("did not select any successful", result.stderr)
                self.assertNotIn("Cache ready", result.stdout)
                self.assertEqual(original, cache_path.read_bytes())

            lines = cache_path.read_text(encoding="utf-8").splitlines()
            header_index = next(i for i, line in enumerate(lines) if not line.startswith("#"))
            columns = lines[header_index].split("\t")
            large = next(
                line.split("\t")
                for line in lines[header_index + 1 :]
                if line.split("\t")[columns.index("m_bucket")] == "64"
            )
            large[columns.index("m_bucket")] = "8192"
            cache_path.write_text("\n".join([*lines, "\t".join(large)]) + "\n", encoding="utf-8")
            chunked, log = run("chunked", chunk=64)
            self.assertIn("Loaded 2 validated fpA_intB tactics", log)
            unpacked = np.empty((64, 128), dtype=np.float32)
            packed = weights.reshape(64, 64)
            unpacked[:, ::2] = packed & 15
            unpacked[:, 1::2] = packed >> 4
            dense = (unpacked - 8) * 0.125
            for key, m in zip(cold, (1, 64, 65), strict=True):
                inputs = (np.arange(m * 128, dtype=np.float32).reshape(m, 128) % 17 / 17).astype(np.float16)
                reference = inputs.astype(np.float32) @ dense.T
                # A final one-row chunk uses GEMV instead of GEMM, with different FP16 rounding.
                np.testing.assert_allclose(chunked[key], reference, rtol=1e-3, atol=1e-2)
                np.testing.assert_allclose(cold[key], reference, rtol=1e-3, atol=1e-2)
            cache_path.write_bytes(original)

            for i in range(header_index + 1, len(lines)):
                row = lines[i].split("\t")
                if row[columns.index("m_bucket")] == "64":
                    row[columns.index("split_k")] = "3"  # K=128 cannot be split into three aligned K tiles.
                    lines[i] = "\t".join(row)
            cache_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
            repaired, log = run("repaired")
            self.assertTrue("Rejecting cached GEMM tactic" in log or "Dropping incompatible cached" in log, log)
            for key in cold:
                np.testing.assert_allclose(repaired[key], cold[key], rtol=1e-3, atol=1e-3)

            # Holding the cross-process file lock must not prevent construction or cached inference.
            # Remove an initial bucket to force construction to produce dirty rows; a pure cache hit
            # would not exercise the old construction-time Flush() path.
            lines = cache_path.read_text(encoding="utf-8").splitlines()
            lines = lines[: header_index + 1] + [
                line for line in lines[header_index + 1 :] if line.split("\t")[columns.index("m_bucket")] != "64"
            ]
            cache_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
            with open(str(cache_path) + ".lock", "a+b") as lock, tempfile.TemporaryFile(mode="w+t") as log_file:
                _lock_file(lock)
                output = root / "locked.npz"
                process = subprocess.Popen(command(output), stdout=log_file, stderr=subprocess.STDOUT, text=True)
                try:
                    deadline = time.monotonic() + 60
                    while not Path(str(output) + ".ready").exists() and process.poll() is None:
                        if time.monotonic() >= deadline:
                            self.fail("File lock blocked session construction or inference")
                        time.sleep(0.1)
                    self.assertTrue(Path(str(output) + ".ready").exists())
                finally:
                    _unlock_file(lock)
                    try:
                        process.wait(timeout=60)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()
                        raise
                log_file.seek(0)
                self.assertEqual(process.returncode, 0, log_file.read())


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--gpu-worker":
        _gpu_worker(*sys.argv[2:])
    else:
        with contextlib.redirect_stdout(io.StringIO()):
            unittest.main()
