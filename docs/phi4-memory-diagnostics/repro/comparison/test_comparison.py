# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
"""Bounded, standard-library-only tests. No GPU, network, models, or packages needed."""

from __future__ import annotations

import argparse
import contextlib
import csv
import hashlib
import io
import json
import multiprocessing
import os
import shutil
import signal
import struct
import subprocess
import sys
import threading
import time
import types
import unittest
import uuid
from pathlib import Path
from unittest.mock import patch

import prepare_benchmark_artifacts as prepare
import run_ort_llama_benchmark as benchmark

HERE = Path(__file__).resolve().parent
OPTIONAL_MODULES = {"psutil", "pynvml", "onnxruntime_genai", "llama_cpp", "huggingface_hub"}


class WorkerWatchdogExpired(BaseException):
    """Bypass production Exception handlers if their deadline regresses."""


@contextlib.contextmanager
def worker_test_watchdog(seconds):
    def expired(signum, frame):
        raise WorkerWatchdogExpired(f"independent worker-test watchdog expired after {seconds:g}s")

    previous_handler = signal.getsignal(signal.SIGALRM)
    previous_timer = signal.getitimer(signal.ITIMER_REAL)
    started = time.monotonic()
    signal.signal(signal.SIGALRM, expired)
    signal.setitimer(signal.ITIMER_REAL, seconds, 1.0)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous_handler)
        delay, interval = previous_timer
        if delay:
            delay = max(0.000001, delay - (time.monotonic() - started))
        signal.setitimer(signal.ITIMER_REAL, delay, interval)


def cpu_worker(sender, mode):
    """Finite synthetic child workloads; never initialize a runtime or GPU."""
    if mode == "large":
        sender.send({"rows": [{"request_index": index, "payload": str(index) * 40} for index in range(10000)]})
    elif mode == "exception":
        raise RuntimeError("synthetic child exception")
    elif mode == "empty":
        return
    elif mode in ("stall", "partial"):
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        sender.send({"pid": os.getpid()})
        if mode == "partial":
            # A complete length header but incomplete body makes recv() block,
            # even though a preceding poll() would say the pipe is readable.
            os.write(sender.fileno(), struct.pack("!i", 1000000) + b"x")
        time.sleep(15)
    else:
        sender.send({"status": "complete", "value": 42})


def cpu_benchmark_worker(sender, fail):
    with patch.object(benchmark, "run_ort") as run:
        if fail:
            run.side_effect = ValueError("synthetic inference failure")
        else:
            run.return_value = {"runtime": "ort", "value": 42}
        benchmark.benchmark_worker(sender, "ort", Path("."), "prompt", 100, 2, 0.01, False, False, Path("unused"))


def cpu_sequential_worker(sender, fail):
    def run(*args, on_row, **kwargs):
        row = {"runtime": "ort", "request_index": 1, "context": 100, "peak_vram_mib": 42, "resident_after_mib": 40}
        on_row(row)
        if fail:
            raise ValueError("synthetic second request failure")
        return [row]

    with patch.object(benchmark, "run_ort_sequential", side_effect=run):
        benchmark.sequential_worker(sender, "ort", Path("."), "prompt", 100, 2, 3, 0.01, False, Path("unused"))


@unittest.skipUnless(hasattr(signal, "setitimer"), "POSIX independent worker-test watchdog")
class WorkerTests(unittest.TestCase):
    def run_child(self, target, args, timeout=3, watchdog_seconds=None):
        children_before = {child.pid for child in multiprocessing.active_children()}
        threads_before = set(threading.enumerate())
        started = time.monotonic()
        try:
            with worker_test_watchdog(watchdog_seconds or timeout + 3 * benchmark.WORKER_CLEANUP_SECONDS + 2):
                result = benchmark.run_worker(target, args, timeout)
        finally:
            # Tests are serial: only children started during this invocation are
            # ours. Never signal a pre-existing child when rescuing a regression.
            leaked_children = [child for child in multiprocessing.active_children() if child.pid not in children_before]
            for child in leaked_children:
                child.terminate()
                child.join(timeout=1)
                if child.is_alive():
                    child.kill()
                    child.join(timeout=1)
                if not child.is_alive():
                    child.close()
        self.assertFalse(leaked_children, "production worker cleanup leaked a child")
        self.assertLess(time.monotonic() - started, timeout + 3 * benchmark.WORKER_CLEANUP_SECONDS + 1)
        self.assertEqual({child.pid for child in multiprocessing.active_children()}, children_before)
        self.assertEqual(set(threading.enumerate()), threads_before)
        return result

    def test_independent_watchdog_bounds_join_before_drain_regression(self):
        def broken_run_worker(target, args, timeout):
            context = multiprocessing.get_context("spawn")
            receiver, sender = context.Pipe(duplex=False)
            child = context.Process(target=target, args=(sender, *args))
            try:
                child.start()
                sender.close()
                child.join()  # Deliberate reproduction of the reviewed deadlock.
            finally:
                receiver.close()
                sender.close()

        children_before = {child.pid for child in multiprocessing.active_children()}
        handler_before = signal.getsignal(signal.SIGALRM)
        timer_before = signal.getitimer(signal.ITIMER_REAL)
        started = time.monotonic()
        with (
            patch.object(benchmark, "run_worker", side_effect=broken_run_worker),
            self.assertRaises(WorkerWatchdogExpired),
        ):
            self.run_child(cpu_worker, ("large",), watchdog_seconds=0.5)
        self.assertLess(time.monotonic() - started, 3.5)
        self.assertEqual({child.pid for child in multiprocessing.active_children()}, children_before)
        self.assertEqual(signal.getsignal(signal.SIGALRM), handler_before)
        self.assertEqual(signal.getitimer(signal.ITIMER_REAL)[1], timer_before[1])
        if timer_before[0] == 0:
            self.assertEqual(signal.getitimer(signal.ITIMER_REAL)[0], 0)

    def test_normal_and_oversized_results(self):
        self.assertEqual(self.run_child(cpu_worker, ("normal",))["value"], 42)
        result = self.run_child(cpu_worker, ("large",))
        self.assertNotIn("worker_error", result)
        self.assertEqual(len(result["rows"]), 10000)
        self.assertEqual(result["rows"][-1]["request_index"], 9999)

    @unittest.skipUnless(sys.platform == "linux", "Linux process signals and pipe framing")
    def test_stall_and_partial_transfer_deadlines_kill_and_reap(self):
        for mode in ("stall", "partial"):
            with self.subTest(mode=mode):
                result = self.run_child(cpu_worker, (mode,), timeout=1)
                self.assertIn("wall deadline", result["worker_error"])
                with self.assertRaises(ProcessLookupError):
                    os.kill(result["pid"], 0)

    def test_exception_exit_and_no_result(self):
        for mode, message in (("exception", "exited with code"), ("empty", "returned no result")):
            with self.subTest(mode=mode):
                self.assertIn(message, self.run_child(cpu_worker, (mode,))["worker_error"])

    def test_benchmark_worker_normal_and_exception_result(self):
        self.assertEqual(self.run_child(cpu_benchmark_worker, (False,))["value"], 42)
        result = self.run_child(cpu_benchmark_worker, (True,))
        self.assertEqual(result["status"], "error")
        self.assertIn("synthetic inference failure", result["error"])

    def test_sequential_worker_keeps_completed_row_on_exception(self):
        result = self.run_child(cpu_sequential_worker, (True,))
        self.assertIn("second request failure", result["error"])
        self.assertEqual(result["partial_rows"][0]["peak_vram_mib"], 42)
        with patch.object(benchmark, "run_worker", return_value=result):
            rows = benchmark.run_sequential_isolated("ort", Path("."), "prompt", 100, 2, 3, 0.01, False, Path("."))
        self.assertEqual([row["status"] for row in rows], ["complete", "error", "error"])
        self.assertEqual([row["request_index"] for row in rows], [1, 2, 3])
        self.assertEqual(rows[0]["peak_vram_mib"], 42)

    def test_worker_deadline_validation(self):
        for timeout in (0, -1, float("inf"), float("nan")):
            with self.subTest(timeout=timeout), self.assertRaises(ValueError):
                benchmark.run_worker(cpu_worker, ("normal",), timeout)

    def test_isolation_wrappers_preserve_measurements_and_diagnostics(self):
        with patch.object(benchmark, "run_worker", side_effect=OSError("cannot start worker")):
            rows = benchmark.run_sequential_isolated("ort", Path("."), "prompt", 100, 2, 3, 0.01, False, Path("."))
        self.assertEqual([row["status"] for row in rows], ["error"] * 3)
        self.assertIn("cannot start worker", rows[0]["error"])
        with patch.object(
            benchmark,
            "run_worker",
            return_value={"value": 42, "error": "original failure", "worker_error": "exit timeout"},
        ):
            row = benchmark.run_isolated("ort", Path("."), "prompt", 100, 2, 0.01, False, False, Path("."))
        self.assertEqual(row["value"], 42)
        self.assertEqual(row["status"], "error")
        self.assertIn("original failure", row["error"])
        self.assertIn("exit timeout", row["error"])
        completed = {"runtime": "ort", "request_index": 1, "context": 100, "peak_vram_mib": 42}
        for result, count, statuses in (
            ({"partial_rows": [completed], "worker_error": "wall deadline"}, 3, ["complete", "error", "error"]),
            ({"rows": [completed], "worker_error": "exit timeout"}, 1, ["error"]),
            ({"rows": [completed]}, 1, ["complete"]),
            ({"rows": [completed]}, 2, ["complete", "error"]),
        ):
            with self.subTest(result=result), patch.object(benchmark, "run_worker", return_value=result):
                rows = benchmark.run_sequential_isolated(
                    "ort", Path("."), "prompt", 100, 2, count, 0.01, False, Path(".")
                )
            self.assertEqual([row["status"] for row in rows], statuses)
            self.assertEqual(rows[0]["peak_vram_mib"], 42)


class RejectOptionalImports:
    def __init__(self):
        self.original_import = __import__
        self.attempts = []

    def __call__(self, name, *args, **kwargs):
        if name.split(".")[0] in OPTIONAL_MODULES:
            self.attempts.append(name)
            raise AssertionError(f"Unexpected optional import: {name}")
        return self.original_import(name, *args, **kwargs)


class HostTests(unittest.TestCase):
    def setUp(self):
        self.root = HERE / f"host-test-work-{uuid.uuid4().hex}"
        self.root.mkdir()
        self.addCleanup(shutil.rmtree, self.root)
        self.ort = self.root / "ort model"
        self.ort.mkdir()
        for name in ("model.onnx", "model.onnx.data", "tokenizer.json", "tokenizer_config.json"):
            (self.ort / name).write_text("{}")
        (self.ort / "genai_config.json").write_text(json.dumps({"model": {"decoder": {"filename": "model.onnx"}}}))
        self.gguf = self.root / "model with spaces.gguf"
        self.gguf.write_bytes(b"not a real model")
        self.output = self.root / "output with spaces"
        self.args = [
            "--ort-model",
            str(self.ort),
            "--gguf-model",
            str(self.gguf),
            "--gpu-index",
            "2",
            "--output-dir",
            str(self.output),
        ]

    def assert_bad(self, args):
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
            benchmark.parse_args(args)
        self.assertEqual(error.exception.code, 2)
        self.assertFalse(self.output.exists())

    def filesystem_snapshot(self):
        return {
            str(path.relative_to(self.root)): (
                path.stat().st_mtime_ns,
                path.read_bytes() if path.is_file() else None,
            )
            for path in (self.root, *self.root.rglob("*"))
        }

    def test_defaults_preserve_historical_workload(self):
        args = benchmark.parse_args(self.args)
        self.assertEqual(args.contexts, [2048, 4096, 8192, 16384, 32768])
        self.assertEqual((args.output_tokens, args.repetitions, args.sampling_interval_ms), (64, 3, 10))
        self.assertEqual((args.sequential_requests, args.sequential_context), (1, 8192))
        self.assertEqual(args.runtime, "both")
        self.assertEqual(args.worker_timeout_seconds, 3600)

    def test_spaces_and_relative_paths(self):
        with patch("os.getcwd", return_value=str(self.root)):
            args = benchmark.parse_args(
                [
                    "--ort-model",
                    "ort model",
                    "--runtime",
                    "ort",
                    "--gpu-index",
                    "0",
                    "--output-dir",
                    "new output",
                ]
            )
        self.assertEqual(args.ort_model, self.ort)
        self.assertEqual(args.output_dir, self.root / "new output")

    def test_single_runtime_does_not_require_other_model(self):
        for runtime, option, path in (("ort", "--ort-model", self.ort), ("llama_cpp", "--gguf-model", self.gguf)):
            with self.subTest(runtime=runtime):
                benchmark.parse_args(
                    [
                        "--runtime",
                        runtime,
                        option,
                        str(path),
                        "--gpu-index",
                        "0",
                        "--output-dir",
                        str(self.output),
                    ]
                )

    def test_required_models_gpu_and_output(self):
        for option in ("--ort-model", "--gguf-model", "--gpu-index", "--output-dir"):
            args = self.args.copy()
            index = args.index(option)
            del args[index : index + 2]
            with self.subTest(option=option):
                self.assert_bad(args)

    def test_invalid_numbers_and_contexts(self):
        for option, value in (
            ("--gpu-index", "-1"),
            ("--output-tokens", "0"),
            ("--repetitions", "-1"),
            ("--sampling-interval-ms", "nan"),
            ("--sampling-interval-ms", "inf"),
            ("--sampling-interval-ms", "0"),
            ("--worker-timeout-seconds", "0"),
            ("--worker-timeout-seconds", "-1"),
            ("--worker-timeout-seconds", "nan"),
            ("--worker-timeout-seconds", "inf"),
            ("--sequential-requests", "0"),
            ("--sequential-context", "0"),
            ("--contexts", "64"),
            ("--contexts", "2048,,4096"),
            ("--contexts", "bad"),
            ("--contexts", "-1"),
        ):
            with self.subTest(option=option, value=value):
                self.assert_bad([*self.args, option, value])

    def test_inactive_matrix_context_does_not_change_sequential(self):
        benchmark.parse_args([*self.args, "--sequential-requests", "2", "--contexts", "1"])
        self.assert_bad([*self.args, "--sequential-requests", "2", "--sequential-context", "64"])

    def test_missing_and_empty_files(self):
        for path in (
            self.gguf,
            *(
                self.ort / name
                for name in (
                    "genai_config.json",
                    "model.onnx",
                    "model.onnx.data",
                    "tokenizer.json",
                    "tokenizer_config.json",
                )
            ),
        ):
            original = path.read_bytes()
            for empty in (True, False):
                with self.subTest(file=path.name, empty=empty):
                    if empty:
                        path.write_bytes(b"")
                    else:
                        path.unlink()
                    self.assert_bad(self.args)
                    path.write_bytes(original)

    def test_config_structure_and_decoder_name(self):
        for content in ("not json", "null", "{}", '{"model":{"decoder":{"filename":"other.onnx"}}}'):
            with self.subTest(content=content):
                (self.ort / "genai_config.json").write_text(content)
                self.assert_bad(self.args)

    def test_output_collisions_and_explicit_reuse(self):
        self.output.mkdir()
        (self.output / "report.md").write_text("old")
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            benchmark.parse_args(self.args)
        benchmark.parse_args([*self.args, "--allow-existing-output"])
        shutil.rmtree(self.output)
        self.assert_bad([*self.args, "--output-dir", str(self.ort / "results")])
        self.assert_bad([*self.args, "--output-dir", str(self.gguf / "results")])

    def test_invalid_mode_combinations(self):
        self.assert_bad([*self.args, "--runtime", "llama_cpp", "--ort-device-allocator"])
        self.assert_bad([*self.args, "--runtime", "llama_cpp", "--log-arena"])
        self.assert_bad([*self.args, "--sequential-requests", "2", "--ort-device-allocator"])

    def test_validation_has_no_optional_imports_gpu_writes_or_downloads(self):
        guard = RejectOptionalImports()
        before = self.filesystem_snapshot()
        with (
            contextlib.redirect_stdout(io.StringIO()),
            patch("builtins.__import__", side_effect=guard),
            patch.object(benchmark, "configure_gpu", side_effect=AssertionError("GPU initialization attempted")),
        ):
            self.assertEqual(benchmark.main([*self.args, "--validate-only"]), 0)
            self.assertEqual(
                prepare.main(
                    [
                        "--plan",
                        "--download-root",
                        str(self.root / "downloads"),
                        "--output-dir",
                        str(self.output),
                    ]
                ),
                0,
            )
        self.assertEqual(guard.attempts, [])
        self.assertEqual(self.filesystem_snapshot(), before)

    def test_python_help_without_site_packages(self):
        code = f"""
import builtins
import runpy
import sys
optional = {sorted(OPTIONAL_MODULES)!r}
attempts = []
original_import = builtins.__import__
def guarded_import(name, *args, **kwargs):
    if name.split(".")[0] in optional:
        attempts.append(name)
        raise AssertionError("Optional import attempted: " + name)
    return original_import(name, *args, **kwargs)
builtins.__import__ = guarded_import
sys.argv = sys.argv[1:]
try:
    runpy.run_path(sys.argv[0], run_name="__main__")
finally:
    if attempts:
        raise AssertionError(attempts)
"""
        before = self.filesystem_snapshot()
        for name in ("run_ort_llama_benchmark.py", "prepare_benchmark_artifacts.py"):
            with self.subTest(script=name):
                completed = subprocess.run(
                    [sys.executable, "-B", "-S", "-c", code, str(HERE / name), "--help"],
                    cwd=self.root,
                    capture_output=True,
                    text=True,
                    timeout=10,
                    check=False,
                )
                self.assertEqual(completed.returncode, 0, completed.stderr)
                self.assertIn("usage:", completed.stdout)
        self.assertEqual(self.filesystem_snapshot(), before)

    def test_shell_help_does_not_invoke_python(self):
        before = self.filesystem_snapshot()
        for arguments in ([], ["--help"], ["--python", "/does/not/exist", "--help"]):
            completed = subprocess.run(
                ["bash", str(HERE / "prepare_ort_llama_benchmark.sh"), *arguments],
                cwd=self.root,
                capture_output=True,
                text=True,
                timeout=10,
                check=False,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertIn("setup ONLY", completed.stdout)
        self.assertEqual(self.filesystem_snapshot(), before)

    def test_shell_requires_explicit_action_and_handles_missing_values(self):
        for arguments in (
            ["--python"],
            [
                "--python",
                sys.executable,
                "--download-root",
                str(self.root / "downloads"),
                "--output-dir",
                str(self.output),
            ],
            ["--python", sys.executable, "--run-benchmark"],
            ["--python", sys.executable, "--install-dependencies"],
        ):
            completed = subprocess.run(
                ["bash", str(HERE / "prepare_ort_llama_benchmark.sh"), *arguments],
                cwd=self.root,
                capture_output=True,
                text=True,
                timeout=10,
                check=False,
            )
            self.assertEqual(completed.returncode, 2, completed.stderr)
        self.assertFalse(self.output.exists())
        self.assertFalse((self.root / "downloads").exists())

    def test_shell_plan_from_foreign_working_directory(self):
        before = self.filesystem_snapshot()
        completed = subprocess.run(
            [
                "bash",
                str(HERE / "prepare_ort_llama_benchmark.sh"),
                "--python",
                sys.executable,
                "--plan",
                "--download-root",
                "download space",
                "--output-dir",
                "manifest space",
            ],
            cwd=self.root,
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        plan = json.loads(completed.stdout)
        self.assertEqual(plan[0]["local_dir"], str(self.root / "download space" / "ort"))
        self.assertEqual(plan[0]["allow_patterns"], ["gpu/gpu-int4-rtn-block-32/*"])
        self.assertEqual(plan[1]["allow_patterns"], ["Phi-4-mini-instruct-Q4_K_M.gguf"])
        self.assertFalse((self.root / "download space").exists())
        self.assertFalse((self.root / "manifest space").exists())
        self.assertEqual(self.filesystem_snapshot(), before)

    def test_setup_requires_immutable_revisions(self):
        for value in ("main", "fc04c8", "z" * 40):
            with self.assertRaises(argparse.ArgumentTypeError):
                prepare.revision(value)
        self.assertEqual(prepare.revision("A" * 40), "a" * 40)
        self.assertEqual(prepare.revision(prepare.ORT_REVISION), prepare.ORT_REVISION)
        self.assertEqual(prepare.revision(prepare.GGUF_REVISION), prepare.GGUF_REVISION)

    def test_file_record_calculates_hash_and_size(self):
        self.assertEqual(
            prepare.file_record(self.gguf, self.root),
            {
                "name": self.gguf.name,
                "bytes": len(b"not a real model"),
                "sha256": hashlib.sha256(b"not a real model").hexdigest(),
            },
        )

    def test_explicit_download_calls_only_pinned_selections(self):
        hub = types.SimpleNamespace(snapshot_download=unittest.mock.Mock())
        target = self.root / "download"
        ort = target / "ort" / prepare.ORT_VARIANT
        gguf = target / "gguf" / prepare.GGUF_FILENAME
        ort.parent.mkdir(parents=True)
        shutil.copytree(self.ort, ort)
        gguf.parent.mkdir()
        shutil.copyfile(self.gguf, gguf)
        with (
            patch.dict(sys.modules, {"huggingface_hub": hub}),
            patch.object(prepare, "version", return_value="test-only"),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            prepare.main(["--download", "--download-root", str(target), "--output-dir", str(self.output)])
        self.assertEqual(hub.snapshot_download.call_count, 2)
        calls = hub.snapshot_download.call_args_list
        self.assertEqual(calls[0].kwargs["revision"], prepare.ORT_REVISION)
        self.assertEqual(calls[1].kwargs["revision"], prepare.GGUF_REVISION)
        manifest = json.loads((self.output / "artifact-manifest.json").read_text())
        self.assertEqual(manifest["environment"]["historical_huggingface_hub_version"], "unknown")
        self.assertEqual(manifest["model"]["gguf"]["bytes"], self.gguf.stat().st_size)

    def test_gpu_index_maps_to_same_uuid_for_cuda(self):
        nvml = unittest.mock.Mock()
        nvml.nvmlDeviceGetUUID.return_value = b"GPU-example"
        with patch.dict(sys.modules, {"pynvml": nvml}), patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "9"}):
            self.assertEqual(benchmark.configure_gpu(2), "GPU-example")
            self.assertEqual(os.environ["CUDA_VISIBLE_DEVICES"], "GPU-example")
            self.assertEqual(os.environ["BENCHMARK_GPU_INDEX"], "2")
        nvml.nvmlDeviceGetHandleByIndex.assert_called_once_with(2)
        nvml.nvmlShutdown.assert_called_once_with()

    def test_prompt_exact_historical_construction(self):
        for context in (2048, 4096, 8192, 16384, 32768):
            seed = "Benchmark context sentence for Phi-4 runtime comparison. "
            self.assertEqual(
                benchmark.build_prompt(context),
                (seed * max(1, context // 9))[: max(100, context * 5)] + "\nAnswer briefly:",
            )

    def test_whole_device_sample_and_window_calculations(self):
        sampler = benchmark.ResourceSampler.__new__(benchmark.ResourceSampler)
        mib = 1024 * 1024
        sampler.samples = [{"vram_bytes": value * mib, "ram_bytes": (value + 1) * mib} for value in (10, 8, 20, 12, 9)]
        sampler._inference_start_index = 1
        sampler._inference_end_index = 4
        metrics = benchmark.sampled_metrics(sampler, 10 * mib, 8 * mib)
        self.assertEqual(
            metrics,
            {
                "load_vram_mib": 10,
                "pre_init_vram_mib": 10,
                "post_init_vram_mib": 8,
                "init_vram_delta_mib": 0,
                "peak_inference_vram_mib": 20,
                "peak_inference_delta_mib": 10,
                "peak_vram_mib": 20,
                "idle_vram_mib": 9,
                "peak_ram_mib": 21,
            },
        )

    def test_matrix_order_and_results(self):
        calls = []

        def run(runtime, model, prompt, context, output_tokens, interval, device, log, log_path, timeout):
            calls.append((context, runtime, device))
            self.assertEqual((output_tokens, interval), (64, 0.01))
            self.assertEqual(timeout, 3600)
            return {"runtime": runtime, "ttft_ms": 1, "decode_tokens_per_second": 2}

        with (
            patch.object(benchmark, "configure_gpu", return_value="GPU-test"),
            patch.object(benchmark, "run_isolated", side_effect=run),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            status = benchmark.main(
                [*self.args, "--contexts", "2048,4096", "--repetitions", "2", "--ort-device-allocator"]
            )
        self.assertEqual(status, 0)
        self.assertEqual(
            calls,
            [
                (context, runtime, device)
                for context in (2048, 4096)
                for _ in range(2)
                for runtime, device in (("ort", False), ("ort", True), ("llama_cpp", False))
            ],
        )
        with (self.output / "results.csv").open() as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual(len(rows), 12)
        self.assertEqual([row["repetition"] for row in rows[:6]], ["1"] * 3 + ["2"] * 3)
        self.assertTrue((self.output / "report.md").is_file())
        self.assertEqual(json.loads((self.output / "run-config.json").read_text())["gpu_uuid"], "GPU-test")

    def test_sequential_order_and_outputs(self):
        calls = []

        def run(runtime, model, prompt, context, output_tokens, count, interval, log, log_path, timeout):
            calls.append(runtime)
            return [
                {
                    "runtime": runtime,
                    "request_index": index,
                    "context": context,
                    "peak_vram_mib": 100,
                    "resident_after_mib": 90,
                    "status": "complete",
                }
                for index in range(1, count + 1)
            ]

        with (
            patch.object(benchmark, "configure_gpu", return_value="GPU-test"),
            patch.object(benchmark, "run_sequential_isolated", side_effect=run),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            self.assertEqual(benchmark.main([*self.args, "--sequential-requests", "3"]), 0)
        self.assertEqual(calls, ["ort", "llama_cpp"])
        with (self.output / "sequential.csv").open() as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual(len(rows), 6)
        self.assertFalse((self.output / "results.csv").exists())
        self.assertIn("Sequential requests", (self.output / "report.md").read_text())

    def test_matrix_all_success_mixed_and_all_failure_exit_status(self):
        for statuses in (("complete", "complete"), ("error", "complete"), ("error", "oom")):
            with self.subTest(statuses=statuses):
                results = [
                    {
                        "runtime": runtime,
                        "status": status,
                        "ttft_ms": 1,
                        "decode_tokens_per_second": 2,
                        **({"error": "synthetic failure"} if status != "complete" else {}),
                    }
                    for runtime, status in zip(("ort", "llama_cpp"), statuses, strict=True)
                ]
                with (
                    patch.object(benchmark, "configure_gpu", return_value="GPU-test"),
                    patch.object(benchmark, "run_isolated", side_effect=results) as run,
                    contextlib.redirect_stdout(io.StringIO()),
                ):
                    code = benchmark.main(
                        [*self.args, "--contexts", "2048", "--repetitions", "1", "--allow-existing-output"]
                    )
                self.assertEqual(code, int(any(status != "complete" for status in statuses)))
                self.assertEqual(run.call_count, 2)
                with (self.output / "results.csv").open() as stream:
                    self.assertEqual([row["status"] for row in csv.DictReader(stream)], list(statuses))
                report = (self.output / "report.md").read_text()
                self.assertIn("Requested condition outcomes", report)
                for runtime, status in zip(("ort", "llama_cpp"), statuses, strict=True):
                    allocator = "baseline" if runtime == "ort" else "n/a"
                    complete = int(status == "complete")
                    self.assertIn(
                        f"| matrix | {runtime} | {allocator} | 2048 | {complete} | {1 - complete} | 1 |", report
                    )
                if code:
                    self.assertIn("synthetic failure", report)
                    self.assertIn("| matrix | ort | baseline | 2048 | 0 | 1 | 1 |", report)

    def test_sequential_all_success_mixed_and_all_failure_exit_status(self):
        for statuses in (("complete", "complete"), ("error", "complete"), ("error", "oom")):
            with self.subTest(statuses=statuses):
                results = [
                    [
                        {
                            "runtime": runtime,
                            "status": status,
                            "request_index": index,
                            "context": 8192,
                            **(
                                {"peak_vram_mib": 42, "resident_after_mib": 40}
                                if status == "complete"
                                else {"error": "synthetic sequential failure"}
                            ),
                        }
                        for index in range(1, 3)
                    ]
                    for runtime, status in zip(("ort", "llama_cpp"), statuses, strict=True)
                ]
                with (
                    patch.object(benchmark, "configure_gpu", return_value="GPU-test"),
                    patch.object(benchmark, "run_sequential_isolated", side_effect=results) as run,
                    contextlib.redirect_stdout(io.StringIO()),
                ):
                    code = benchmark.main([*self.args, "--sequential-requests", "2", "--allow-existing-output"])
                self.assertEqual(code, int(any(status != "complete" for status in statuses)))
                self.assertEqual(run.call_count, 2)
                with (self.output / "sequential.csv").open() as stream:
                    self.assertEqual(
                        [row["status"] for row in csv.DictReader(stream)], [s for s in statuses for _ in (0, 1)]
                    )
                report = (self.output / "report.md").read_text()
                for runtime, status in zip(("ort", "llama_cpp"), statuses, strict=True):
                    complete = 2 if status == "complete" else 0
                    self.assertIn(f"| sequential | {runtime} | n/a | 8192 | {complete} | {2 - complete} | 2 |", report)
                    if status != "complete":
                        for index in (1, 2):
                            self.assertIn(
                                f'"request_index": {index}, "runtime": "{runtime}", "status": "{status}"', report
                            )
                if code:
                    self.assertIn("synthetic sequential failure", report)
                    self.assertIn("| sequential | ort | n/a | 8192 | 0 | 2 | 2 |", report)

    def test_report_medians_signed_deltas_and_error_exclusion(self):
        rows = []
        for runtime, allocator, init, peak in (
            ("ort", "baseline", 100, 200),
            ("ort", "baseline", 120, 220),
            ("ort", "device", 80, 150),
            ("llama_cpp", "n/a", 50, 100),
        ):
            rows.append(
                {
                    "runtime": runtime,
                    "ort_allocator_mode": allocator,
                    "status": "complete",
                    "context_tokens": 2048,
                    "post_init_vram_mib": init,
                    "peak_inference_vram_mib": peak,
                    "ttft_ms": 2,
                    "decode_tokens_per_second": 4,
                }
            )
        rows.append(
            {
                "runtime": "ort",
                "ort_allocator_mode": "baseline",
                "status": "oom",
                "context_tokens": 2048,
                "post_init_vram_mib": 99999,
            }
        )
        benchmark.write_report(rows, self.output)
        text = (self.output / "report.md").read_text()
        self.assertIn("| 2048 | 110.00 | 80.00 | 50.00 | 210.00 | 150.00 | 100.00 | -30.00 | -60.00 |", text)
        self.assertIn("**110.00 MiB**", text)


if __name__ == "__main__":
    unittest.main()
