#!/usr/bin/env python3
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# Optional runtime imports must stay lazy for host-only help and validation.
# ruff: noqa: PLC0415
"""Run an ORT GenAI versus llama.cpp benchmark on Linux."""

from __future__ import annotations

import argparse
import csv
import gc
import io
import json
import math
import multiprocessing
import os
import queue
import re
import statistics
import threading
import time
from collections.abc import Callable
from contextlib import contextmanager, redirect_stderr
from pathlib import Path
from typing import Any

DEFAULT_WORKER_TIMEOUT_SECONDS = 3600.0
WORKER_CLEANUP_SECONDS = 1.0


class ResourceSampler:
    def __init__(self, interval_seconds: float = 0.01) -> None:
        import psutil
        import pynvml

        self._nvml = pynvml
        self.interval_seconds = interval_seconds
        self.samples: list[dict[str, int]] = []
        self._inference_start_index: int | None = None
        self._inference_end_index: int | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._process = psutil.Process()

    def __enter__(self) -> ResourceSampler:  # noqa: PYI034 - retain Python 3.10 compatibility
        self._nvml.nvmlInit()
        gpu_index = int(os.environ["BENCHMARK_GPU_INDEX"])
        self._handle = self._nvml.nvmlDeviceGetHandleByIndex(gpu_index)
        self._sample_once()
        self._thread = threading.Thread(target=self._sample, daemon=True)
        self._thread.start()
        return self

    def _sample_once(self) -> None:
        memory = self._nvml.nvmlDeviceGetMemoryInfo(self._handle)
        self.samples.append(
            {
                "vram_bytes": int(memory.used),
                "ram_bytes": self._process.memory_info().rss,
            }
        )

    def snapshot_vram_bytes(self) -> int:
        self._sample_once()
        return self.samples[-1]["vram_bytes"]

    def begin_inference(self) -> None:
        self._sample_once()
        self._inference_start_index = len(self.samples) - 1

    def end_inference(self) -> None:
        self._sample_once()
        self._inference_end_index = len(self.samples)

    def _sample(self) -> None:
        while not self._stop.wait(self.interval_seconds):
            self._sample_once()

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        self._sample_once()
        self._nvml.nvmlShutdown()

    @property
    def initial_vram_bytes(self) -> int:
        return self.samples[0]["vram_bytes"]

    @property
    def peak_vram_bytes(self) -> int:
        return max(sample["vram_bytes"] for sample in self.samples)

    @property
    def peak_inference_vram_bytes(self) -> int:
        if self._inference_start_index is None or self._inference_end_index is None:
            raise RuntimeError("Inference sampling window was not completed")
        inference_samples = self.samples[self._inference_start_index : self._inference_end_index]
        return max(sample["vram_bytes"] for sample in inference_samples)

    @property
    def idle_vram_bytes(self) -> int:
        return self.samples[-1]["vram_bytes"]

    @property
    def peak_ram_bytes(self) -> int:
        return max(sample["ram_bytes"] for sample in self.samples)


def build_prompt(target_tokens: int) -> str:
    seed = "Benchmark context sentence for Phi-4 runtime comparison. "
    repetitions = max(1, target_tokens // 9)
    return (seed * repetitions)[: max(100, target_tokens * 5)] + "\nAnswer briefly:"


def as_mib(value: int) -> float:
    return round(value / (1024 * 1024), 2)


@contextmanager
def redirect_native_stderr(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    saved_stderr = os.dup(2)
    try:
        with path.open("ab", buffering=0) as stream:
            os.dup2(stream.fileno(), 2)
            yield
    finally:
        os.dup2(saved_stderr, 2)
        os.close(saved_stderr)


def parse_arena_log(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    content = path.read_text(encoding="utf-8", errors="replace")
    extension_count = 0
    cuda_totals: list[int] = []
    awaiting_cuda_total = False
    for line in content.splitlines():
        extension = re.search(r"Extending BFCArena for (.+?)\.", line)
        if extension:
            awaiting_cuda_total = extension.group(1).strip().lower() == "cuda"
            if awaiting_cuda_total:
                extension_count += 1
            continue
        total = re.search(r"Total allocated bytes:\s+(\d+)", line)
        if total and awaiting_cuda_total:
            cuda_totals.append(int(total.group(1)))
            awaiting_cuda_total = False
    return {
        "final_arena_size_mib": (as_mib(cuda_totals[-1]) if cuda_totals else None),
        "extension_count": extension_count,
        "allocation_count": None,
        "allocation_count_note": (
            "Not emitted by stock ONNX Runtime at verbose level; "
            "DumpMemoryLog requires a custom ORT build or allocator instrumentation."
        ),
        "free_chunk_note": (
            "Not emitted by stock ONNX Runtime at verbose level; free-chunk details "
            "are part of the same unavailable DumpMemoryLog output."
        ),
    }


def sampled_metrics(
    sampler: ResourceSampler,
    pre_init_vram_bytes: int,
    post_init_vram_bytes: int,
) -> dict[str, float]:
    peak_inference_vram_bytes = sampler.peak_inference_vram_bytes
    return {
        "load_vram_mib": as_mib(sampler.initial_vram_bytes),
        "pre_init_vram_mib": as_mib(pre_init_vram_bytes),
        "post_init_vram_mib": as_mib(post_init_vram_bytes),
        "init_vram_delta_mib": as_mib(max(post_init_vram_bytes - pre_init_vram_bytes, 0)),
        "peak_inference_vram_mib": as_mib(peak_inference_vram_bytes),
        "peak_inference_delta_mib": as_mib(max(peak_inference_vram_bytes - pre_init_vram_bytes, 0)),
        "peak_vram_mib": as_mib(sampler.peak_vram_bytes),
        "idle_vram_mib": as_mib(sampler.idle_vram_bytes),
        "peak_ram_mib": as_mib(sampler.peak_ram_bytes),
    }


def run_ort(
    model_path: Path,
    prompt: str,
    context_tokens: int,
    output_tokens: int,
    sampling_interval_seconds: float,
    use_device_allocator: bool,
    log_arena: bool = False,
) -> dict[str, Any]:
    import onnxruntime_genai as og

    if not og.is_cuda_available():
        raise RuntimeError("ONNX Runtime GenAI was built without CUDA support")

    model_artifact_bytes = sum(path.stat().st_size for path in model_path.glob("model.onnx*") if path.is_file())
    if model_artifact_bytes == 0:
        raise RuntimeError(f"No ONNX model artifacts found in {model_path}")

    with ResourceSampler(sampling_interval_seconds) as sampler:
        load_started = time.perf_counter()
        config = og.Config(str(model_path))
        config.clear_providers()
        config.append_provider("CUDA")
        config.set_provider_option("CUDA", "device_id", "0")
        session_options: dict[str, Any] = {
            "session.use_device_allocator_for_initializers": ("1" if use_device_allocator else "0")
        }
        if log_arena:
            session_options["log_severity_level"] = 1
            session_options["log_verbosity_level"] = 0
        config.overlay(
            json.dumps(
                {
                    "model": {
                        "decoder": {
                            "session_options": session_options,
                        }
                    }
                }
            )
        )
        pre_init_vram_bytes = sampler.snapshot_vram_bytes()
        model = og.Model(config)
        post_init_vram_bytes = sampler.snapshot_vram_bytes()
        if model.device_type != "CUDA":
            raise RuntimeError(f"ORT model is not fully configured for CUDA: {model.device_type!r}")
        init_device_bytes = post_init_vram_bytes - pre_init_vram_bytes
        residency_ratio = init_device_bytes / model_artifact_bytes
        if residency_ratio < 0.9:
            raise RuntimeError(
                "ORT model initializer residency check failed: "
                f"{as_mib(init_device_bytes):.2f} MiB allocated on GPU for "
                f"{as_mib(model_artifact_bytes):.2f} MiB of ONNX artifacts "
                f"({residency_ratio:.1%})"
            )
        tokenizer = og.Tokenizer(model)
        load_ms = (time.perf_counter() - load_started) * 1000
        input_ids = tokenizer.encode(prompt)
        if len(input_ids) + output_tokens > context_tokens:
            input_ids = input_ids[: context_tokens - output_tokens]

        params = og.GeneratorParams(model)
        params.set_search_options(max_length=len(input_ids) + output_tokens, batch_size=1)
        generator = og.Generator(model, params)
        sampler.begin_inference()
        started = time.perf_counter()
        generator.append_tokens(input_ids)
        first_token_ms: float | None = None
        generated = 0
        while not generator.is_done() and generated < output_tokens:
            generator.generate_next_token()
            generated += 1
            if first_token_ms is None:
                first_token_ms = (time.perf_counter() - started) * 1000
        total_ms = (time.perf_counter() - started) * 1000
        sampler.end_inference()
        del generator, params, tokenizer, model, config
        gc.collect()
        time.sleep(3.0)

    decode_seconds = max((total_ms - (first_token_ms or 0)) / 1000, 1e-9)
    return {
        "runtime": "ort",
        "ort_allocator_mode": "device" if use_device_allocator else "baseline",
        "gpu_resident": True,
        "gpu_residency_check": (f"CUDA device type; init allocation is {residency_ratio:.1%} of ONNX artifact bytes"),
        "model_artifact_mib": as_mib(model_artifact_bytes),
        "gpu_init_residency_ratio": round(residency_ratio, 4),
        "prompt_tokens": len(input_ids),
        "completion_tokens": generated,
        "load_ms": round(load_ms, 3),
        "ttft_ms": round(first_token_ms, 3) if first_token_ms is not None else None,
        "total_latency_ms": round(total_ms, 3),
        "decode_tokens_per_second": round(max(generated - 1, 0) / decode_seconds, 3),
        **sampled_metrics(sampler, pre_init_vram_bytes, post_init_vram_bytes),
    }


def run_llama(
    model_path: Path,
    prompt: str,
    context_tokens: int,
    output_tokens: int,
    sampling_interval_seconds: float,
) -> dict[str, Any]:
    import llama_cpp

    if not llama_cpp.llama_supports_gpu_offload():
        raise RuntimeError("llama.cpp was built without GPU offload support")

    with ResourceSampler(sampling_interval_seconds) as sampler:
        load_started = time.perf_counter()
        pre_init_vram_bytes = sampler.snapshot_vram_bytes()
        load_log = io.StringIO()
        with redirect_stderr(load_log):
            model = llama_cpp.Llama(
                model_path=str(model_path),
                n_gpu_layers=-1,
                split_mode=llama_cpp.LLAMA_SPLIT_MODE_NONE,
                main_gpu=0,
                n_ctx=context_tokens,
                verbose=True,
            )
        post_init_vram_bytes = sampler.snapshot_vram_bytes()
        if model.model_params.n_gpu_layers != 0x7FFFFFFF:
            raise RuntimeError("llama.cpp did not preserve the all-layer offload request")
        if model.model_params.split_mode != llama_cpp.LLAMA_SPLIT_MODE_NONE:
            raise RuntimeError("llama.cpp unexpectedly enabled multi-GPU splitting")
        offload_matches = [
            (int(done), int(total))
            for done, total in re.findall(
                r"offloaded\s+(\d+)/(\d+)\s+layers to GPU",
                load_log.getvalue(),
            )
        ]
        if not offload_matches:
            raise RuntimeError("llama.cpp did not emit a GPU layer-offload summary")
        if not all(done == total and total > 0 for done, total in offload_matches):
            raise RuntimeError(f"llama.cpp did not offload every offloadable layer: {offload_matches}")
        load_ms = (time.perf_counter() - load_started) * 1000
        prompt_ids = model.tokenize(prompt.encode("utf-8"))
        if len(prompt_ids) + output_tokens > context_tokens:
            prompt_ids = prompt_ids[: context_tokens - output_tokens]
            prompt = model.detokenize(prompt_ids).decode("utf-8", errors="ignore")

        sampler.begin_inference()
        started = time.perf_counter()
        first_token_ms: float | None = None
        generated_text: list[str] = []
        for chunk in model.create_completion(
            prompt,
            max_tokens=output_tokens,
            temperature=0.0,
            stream=True,
        ):
            text = chunk["choices"][0]["text"]
            if text and first_token_ms is None:
                first_token_ms = (time.perf_counter() - started) * 1000
            generated_text.append(text)
        total_ms = (time.perf_counter() - started) * 1000
        sampler.end_inference()
        completion_ids = model.tokenize(
            "".join(generated_text).encode("utf-8"),
            add_bos=False,
        )
        del model
        gc.collect()
        time.sleep(3.0)

    generated = len(completion_ids)
    decode_seconds = max((total_ms - (first_token_ms or 0)) / 1000, 1e-9)
    return {
        "runtime": "llama_cpp",
        "ort_allocator_mode": "n/a",
        "gpu_resident": True,
        "gpu_residency_check": (
            f"llama.cpp offloaded {offload_matches[-1][0]}/{offload_matches[-1][1]} offloadable layers"
        ),
        "gpu_offload_layers": offload_matches[-1][0],
        "gpu_total_layers": offload_matches[-1][1],
        "prompt_tokens": len(prompt_ids),
        "completion_tokens": generated,
        "load_ms": round(load_ms, 3),
        "ttft_ms": round(first_token_ms, 3) if first_token_ms is not None else None,
        "total_latency_ms": round(total_ms, 3),
        "decode_tokens_per_second": round(max(generated - 1, 0) / decode_seconds, 3),
        **sampled_metrics(sampler, pre_init_vram_bytes, post_init_vram_bytes),
    }


def run_ort_sequential(
    model_path: Path,
    prompt: str,
    context_tokens: int,
    output_tokens: int,
    request_count: int,
    sampling_interval_seconds: float,
    log_arena: bool,
    on_row: Callable[[dict[str, Any]], None] | None = None,
) -> list[dict[str, Any]]:
    import onnxruntime_genai as og

    if not og.is_cuda_available():
        raise RuntimeError("ONNX Runtime GenAI was built without CUDA support")

    model_artifact_bytes = sum(path.stat().st_size for path in model_path.glob("model.onnx*") if path.is_file())
    if model_artifact_bytes == 0:
        raise RuntimeError(f"No ONNX model artifacts found in {model_path}")

    with ResourceSampler(sampling_interval_seconds) as sampler:
        config = og.Config(str(model_path))
        config.clear_providers()
        config.append_provider("CUDA")
        config.set_provider_option("CUDA", "device_id", "0")
        session_options: dict[str, Any] = {"session.use_device_allocator_for_initializers": "0"}
        if log_arena:
            session_options["log_severity_level"] = 1
            session_options["log_verbosity_level"] = 0
        config.overlay(
            json.dumps(
                {
                    "model": {
                        "decoder": {
                            "session_options": session_options,
                        }
                    }
                }
            )
        )
        pre_init_vram_bytes = sampler.snapshot_vram_bytes()
        model = og.Model(config)
        post_init_vram_bytes = sampler.snapshot_vram_bytes()
        if model.device_type != "CUDA":
            raise RuntimeError(f"ORT model is not fully configured for CUDA: {model.device_type!r}")
        residency_ratio = (post_init_vram_bytes - pre_init_vram_bytes) / model_artifact_bytes
        if residency_ratio < 0.9:
            raise RuntimeError(
                "ORT model initializer residency check failed: "
                f"{residency_ratio:.1%} of ONNX artifact bytes allocated on GPU"
            )
        tokenizer = og.Tokenizer(model)
        input_ids = tokenizer.encode(prompt)
        if len(input_ids) + output_tokens > context_tokens:
            input_ids = input_ids[: context_tokens - output_tokens]

        rows: list[dict[str, Any]] = []
        for request_index in range(1, request_count + 1):
            params = og.GeneratorParams(model)
            params.set_search_options(
                max_length=len(input_ids) + output_tokens,
                batch_size=1,
            )
            generator = og.Generator(model, params)
            sampler.begin_inference()
            generator.append_tokens(input_ids)
            generated = 0
            while not generator.is_done() and generated < output_tokens:
                generator.generate_next_token()
                generated += 1
            sampler.end_inference()
            peak_vram_mib = as_mib(sampler.peak_inference_vram_bytes)
            del generator, params
            gc.collect()
            resident_after_mib = as_mib(sampler.snapshot_vram_bytes())
            rows.append(
                {
                    "runtime": "ort",
                    "request_index": request_index,
                    "context": context_tokens,
                    "peak_vram_mib": peak_vram_mib,
                    "resident_after_mib": resident_after_mib,
                }
            )
            if on_row is not None:
                on_row(rows[-1])
        del tokenizer, model, config
    return rows


def run_llama_sequential(
    model_path: Path,
    prompt: str,
    context_tokens: int,
    output_tokens: int,
    request_count: int,
    sampling_interval_seconds: float,
    on_row: Callable[[dict[str, Any]], None] | None = None,
) -> list[dict[str, Any]]:
    import llama_cpp

    if not llama_cpp.llama_supports_gpu_offload():
        raise RuntimeError("llama.cpp was built without GPU offload support")

    with ResourceSampler(sampling_interval_seconds) as sampler:
        load_log = io.StringIO()
        with redirect_stderr(load_log):
            model = llama_cpp.Llama(
                model_path=str(model_path),
                n_gpu_layers=-1,
                split_mode=llama_cpp.LLAMA_SPLIT_MODE_NONE,
                main_gpu=0,
                n_ctx=context_tokens,
                verbose=True,
            )
        offload_matches = [
            (int(done), int(total))
            for done, total in re.findall(
                r"offloaded\s+(\d+)/(\d+)\s+layers to GPU",
                load_log.getvalue(),
            )
        ]
        if not offload_matches or not all(done == total and total > 0 for done, total in offload_matches):
            raise RuntimeError(f"llama.cpp full GPU layer offload was not verified: {offload_matches}")
        prompt_ids = model.tokenize(prompt.encode("utf-8"))
        if len(prompt_ids) + output_tokens > context_tokens:
            prompt_ids = prompt_ids[: context_tokens - output_tokens]
            prompt = model.detokenize(prompt_ids).decode("utf-8", errors="ignore")

        rows: list[dict[str, Any]] = []
        for request_index in range(1, request_count + 1):
            sampler.begin_inference()
            for _chunk in model.create_completion(
                prompt,
                max_tokens=output_tokens,
                temperature=0.0,
                stream=True,
            ):
                pass
            sampler.end_inference()
            peak_vram_mib = as_mib(sampler.peak_inference_vram_bytes)
            gc.collect()
            resident_after_mib = as_mib(sampler.snapshot_vram_bytes())
            rows.append(
                {
                    "runtime": "llama_cpp",
                    "request_index": request_index,
                    "context": context_tokens,
                    "peak_vram_mib": peak_vram_mib,
                    "resident_after_mib": resident_after_mib,
                }
            )
            if on_row is not None:
                on_row(rows[-1])
        del model
    return rows


def error_result(runtime: str, error: BaseException) -> dict[str, Any]:
    message = f"{type(error).__name__}: {error}"
    status = "oom" if "out of memory" in message.lower() else "error"
    return {"runtime": runtime, "status": status, "error": message}


def run_worker(
    target: Callable[..., None],
    args: tuple[Any, ...],
    timeout_seconds: float,
) -> dict[str, Any]:
    """Drain while running; only this invocation's child may be signalled."""
    if not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
        raise ValueError("worker timeout must be finite and positive")
    context = multiprocessing.get_context("spawn")
    receiver, sender = context.Pipe(duplex=False)
    process = context.Process(target=target, args=(sender, *args))
    messages: queue.Queue[tuple[str, Any]] = queue.Queue()
    partial_rows: list[dict[str, Any]] = []
    result: dict[str, Any] = {}

    def receive() -> None:
        # Connection.poll()/Queue.get(timeout=...) only bound the initial read:
        # a partial frame can still block recv(). Keep it off the deadline thread.
        try:
            while True:
                messages.put(("message", receiver.recv()))
        except EOFError:
            messages.put(("closed", None))
        except Exception as error:
            messages.put(("error", f"{type(error).__name__}: {error}"))
        finally:
            receiver.close()

    reader = threading.Thread(target=receive, daemon=True)
    deadline = time.monotonic() + timeout_seconds
    try:
        process.start()
        sender.close()
        reader.start()
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(f"worker exceeded wall deadline of {timeout_seconds:g} seconds")
            try:
                kind, message = messages.get(timeout=remaining)
            except queue.Empty as error:
                raise TimeoutError(f"worker exceeded wall deadline of {timeout_seconds:g} seconds") from error
            if kind == "closed":
                break
            if kind == "error":
                raise RuntimeError(f"worker result delivery failed: {message}")
            if "row" in message:
                partial_rows.append(message["row"])
            else:
                result = message
        process.join(timeout=max(0.0, deadline - time.monotonic()))
        if process.is_alive():
            raise TimeoutError(f"worker exceeded wall deadline of {timeout_seconds:g} seconds during exit")
        if process.exitcode != 0:
            raise RuntimeError(f"worker exited with code {process.exitcode}")
        if not result:
            raise RuntimeError("worker returned no result")
    except Exception as error:
        result["worker_error"] = f"{type(error).__name__}: {error}"
    finally:
        sender.close()
        if process.pid is not None:
            if process.is_alive():
                process.terminate()
                process.join(timeout=WORKER_CLEANUP_SECONDS)
            if process.is_alive():
                process.kill()
                process.join(timeout=WORKER_CLEANUP_SECONDS)
            if process.is_alive():
                result["worker_error"] = (
                    result.get("worker_error", "") + "; owned worker could not be reaped after terminate/kill"
                )
            else:
                process.join(timeout=0)
                process.close()
        if reader.ident is not None:
            reader.join(timeout=WORKER_CLEANUP_SECONDS)
            if reader.is_alive():
                result["worker_error"] = result.get("worker_error", "") + "; result reader did not stop"
        else:
            receiver.close()
    result["partial_rows"] = partial_rows
    return result


def benchmark_worker(
    queue: Any,
    runtime: str,
    model_path: Path,
    prompt: str,
    context_tokens: int,
    output_tokens: int,
    sampling_interval_seconds: float,
    use_device_allocator: bool,
    log_arena: bool,
    arena_log_path: Path,
) -> None:
    try:
        if runtime == "ort":
            if log_arena and not use_device_allocator:
                os.environ["ORTGENAI_ORT_VERBOSE_LOGGING"] = "1"
                with redirect_native_stderr(arena_log_path):
                    result = run_ort(
                        model_path,
                        prompt,
                        context_tokens,
                        output_tokens,
                        sampling_interval_seconds,
                        use_device_allocator,
                        log_arena=True,
                    )
            else:
                result = run_ort(
                    model_path,
                    prompt,
                    context_tokens,
                    output_tokens,
                    sampling_interval_seconds,
                    use_device_allocator,
                )
        else:
            result = run_llama(
                model_path,
                prompt,
                context_tokens,
                output_tokens,
                sampling_interval_seconds,
            )
        queue.send(result)
    except BaseException as error:
        queue.send(error_result(runtime, error))


def run_isolated(
    runtime: str,
    model_path: Path,
    prompt: str,
    context_tokens: int,
    output_tokens: int,
    sampling_interval_seconds: float,
    use_device_allocator: bool,
    log_arena: bool,
    arena_log_path: Path,
    worker_timeout_seconds: float = DEFAULT_WORKER_TIMEOUT_SECONDS,
) -> dict[str, Any]:
    result = run_worker(
        benchmark_worker,
        (
            runtime,
            model_path,
            prompt,
            context_tokens,
            output_tokens,
            sampling_interval_seconds,
            use_device_allocator,
            log_arena,
            arena_log_path,
        ),
        worker_timeout_seconds,
    )
    result.pop("partial_rows", None)
    if "worker_error" in result:
        diagnostic = "; ".join(str(result[key]) for key in ("error", "worker_error") if key in result)
        result.pop("worker_error")
        result.update(error_result(runtime, RuntimeError(diagnostic)))
    return result


def sequential_worker(
    queue: Any,
    runtime: str,
    model_path: Path,
    prompt: str,
    context_tokens: int,
    output_tokens: int,
    request_count: int,
    sampling_interval_seconds: float,
    log_arena: bool,
    arena_log_path: Path,
) -> None:
    def on_row(row: dict[str, Any]) -> None:
        queue.send({"row": row})

    try:
        if runtime == "ort":
            if log_arena:
                os.environ["ORTGENAI_ORT_VERBOSE_LOGGING"] = "1"
                with redirect_native_stderr(arena_log_path):
                    rows = run_ort_sequential(
                        model_path,
                        prompt,
                        context_tokens,
                        output_tokens,
                        request_count,
                        sampling_interval_seconds,
                        log_arena=True,
                        on_row=on_row,
                    )
            else:
                rows = run_ort_sequential(
                    model_path,
                    prompt,
                    context_tokens,
                    output_tokens,
                    request_count,
                    sampling_interval_seconds,
                    log_arena=False,
                    on_row=on_row,
                )
        else:
            rows = run_llama_sequential(
                model_path,
                prompt,
                context_tokens,
                output_tokens,
                request_count,
                sampling_interval_seconds,
                on_row=on_row,
            )
        queue.send({"rows": rows})
    except BaseException as error:
        queue.send({"error": f"{type(error).__name__}: {error}"})


def run_sequential_isolated(
    runtime: str,
    model_path: Path,
    prompt: str,
    context_tokens: int,
    output_tokens: int,
    request_count: int,
    sampling_interval_seconds: float,
    log_arena: bool,
    arena_log_path: Path,
    worker_timeout_seconds: float = DEFAULT_WORKER_TIMEOUT_SECONDS,
) -> list[dict[str, Any]]:
    try:
        result = run_worker(
            sequential_worker,
            (
                runtime,
                model_path,
                prompt,
                context_tokens,
                output_tokens,
                request_count,
                sampling_interval_seconds,
                log_arena,
                arena_log_path,
            ),
            worker_timeout_seconds,
        )
    except Exception as error:
        result = {"error": f"{type(error).__name__}: {error}"}
    rows = [{"status": "complete", **row} for row in result.get("rows", result.get("partial_rows", []))]
    error = "; ".join(result[key] for key in ("error", "worker_error") if key in result)
    if not error and len(rows) != request_count:
        error = f"worker returned {len(rows)} of {request_count} requested rows"
    if error:
        failed = error_result(runtime, RuntimeError(error))
        for request_index in range(len(rows) + 1, request_count + 1):
            rows.append({**failed, "request_index": request_index, "context": context_tokens})
        if rows and all(row["status"] == "complete" for row in rows):
            # Preserve measurements even if model cleanup/worker exit failed.
            rows[-1].update(failed)
    return rows


def write_sequential_csv(rows: list[dict[str, Any]], output_dir: Path) -> None:
    fields = [
        "runtime",
        "request_index",
        "context",
        "peak_vram_mib",
        "resident_after_mib",
        "status",
        "error",
    ]
    with (output_dir / "sequential.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_report(
    rows: list[dict[str, Any]],
    output_dir: Path,
    sequential_rows: list[dict[str, Any]] | None = None,
    arena_summary: dict[str, Any] | None = None,
    sampling_interval_ms_override: float | None = None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    if rows:
        fields = sorted({key for row in rows for key in row})
        with (output_dir / "results.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)

    completed = [row for row in rows if row.get("status") == "complete"]
    sampling_interval_ms = next(
        (row["sampling_interval_ms"] for row in rows if "sampling_interval_ms" in row),
        sampling_interval_ms_override if sampling_interval_ms_override is not None else "unknown",
    )
    lines = [
        "# Phi-4-mini ORT versus llama.cpp GPU-memory diagnostic",
        "",
        f"NVML sampling interval: **{sampling_interval_ms} ms** on the selected physical GPU.",
        "Each row runs in an isolated process and is persisted after completion.",
        "ORT requires CUDA device type and post-session GPU allocation of at least 90% of",
        "the ONNX artifact bytes. The published graph may retain shape/control nodes on CPU;",
        "this check targets model initializer residency rather than requiring every graph",
        "operator to use CUDA. llama.cpp requires all offloadable layers to report GPU",
        "placement and intentionally retains a small non-offloadable CPU input buffer.",
        "The ORT `device` variant sets",
        "`session.use_device_allocator_for_initializers=1`; allocator deltas are device",
        "minus baseline, so negative values mean the device variant used less VRAM.",
        "",
        "## Per-run measurements",
        "",
        "| Runtime | Allocator | Context | Rep | Init VRAM MiB | Init delta MiB | Peak inference MiB | TTFT ms | Decode tok/s | Status |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    outcomes: dict[tuple[str, str, str, Any], list[dict[str, Any]]] = {}
    for mode, mode_rows in (("matrix", rows), ("sequential", sequential_rows or [])):
        for row in mode_rows:
            key = (
                mode,
                row["runtime"],
                row.get("ort_allocator_mode", "n/a"),
                row.get("context_tokens", row.get("context", "")),
            )
            outcomes.setdefault(key, []).append(row)
    outcome_lines = [
        "",
        "## Requested condition outcomes",
        "",
        "Any failed requested row makes this invocation unsuccessful; partial results are retained.",
        "Medians below include successful rows only, not failed rows. Counts show incomplete conditions.",
        "Previously loaded matrix rows (when output reuse is explicit) are reported but not rerun.",
        "",
        "| Mode | Runtime | Allocator | Context | Complete | Failed | Total |",
        "|---|---|---|---:|---:|---:|---:|",
    ]
    for (mode, runtime, allocator, context), condition_rows in outcomes.items():
        success_count = sum(row.get("status", "complete") == "complete" for row in condition_rows)
        outcome_lines.append(
            f"| {mode} | {runtime} | {allocator} | {context} | {success_count} | "
            f"{len(condition_rows) - success_count} | {len(condition_rows)} |"
        )
    failures = [
        {"mode": mode, **row}
        for mode, mode_rows in (("matrix", rows), ("sequential", sequential_rows or []))
        for row in mode_rows
        if row.get("status", "complete") != "complete"
    ]
    if failures:
        outcome_lines += ["", "### Failure diagnostics", ""]
        for row in failures:
            outcome_lines.append(
                "- "
                + json.dumps(
                    {
                        key: row[key]
                        for key in (
                            "mode",
                            "runtime",
                            "ort_allocator_mode",
                            "context_tokens",
                            "context",
                            "repetition",
                            "request_index",
                            "status",
                            "error",
                        )
                        if key in row
                    },
                    sort_keys=True,
                )
            )
    else:
        outcome_lines += ["", "No failed rows recorded."]
    # Insert before the measurement tables, keeping every requested condition visible.
    measurement_index = lines.index("## Per-run measurements")
    lines[measurement_index:measurement_index] = [*outcome_lines, ""]
    for row in rows:
        lines.append(
            f"| {row.get('runtime', '')} | {row.get('ort_allocator_mode', '')} | "
            f"{row.get('context_tokens', '')} | {row.get('repetition', '')} | "
            f"{row.get('post_init_vram_mib', '')} | "
            f"{row.get('init_vram_delta_mib', '')} | "
            f"{row.get('peak_inference_vram_mib', '')} | "
            f"{row.get('ttft_ms', '')} | {row.get('decode_tokens_per_second', '')} | "
            f"{row.get('status', '')} |"
        )
    if completed:
        groups: dict[tuple[str, str, int], list[dict[str, Any]]] = {}
        for row in completed:
            key = (
                row["runtime"],
                row.get("ort_allocator_mode", "n/a"),
                int(row["context_tokens"]),
            )
            groups.setdefault(key, []).append(row)

        def median_value(
            runtime: str,
            allocator: str,
            context_tokens: int,
            field: str,
        ) -> float | None:
            group = groups.get((runtime, allocator, context_tokens), [])
            values = [float(row[field]) for row in group if row.get(field) is not None]
            return statistics.median(values) if values else None

        contexts = sorted({int(row["context_tokens"]) for row in rows})
        lines += [
            "",
            "## Median staged-memory comparison",
            "",
            "| Context | ORT baseline init | ORT device init | llama.cpp init | ORT baseline peak inference | ORT device peak inference | llama.cpp peak inference | Allocator init delta | Allocator peak delta |",
            "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
        gap_candidates: list[tuple[float, int, str, str]] = []
        for context_tokens in contexts:
            ort_init = median_value("ort", "baseline", context_tokens, "post_init_vram_mib")
            ort_device_init = median_value("ort", "device", context_tokens, "post_init_vram_mib")
            llama_init = median_value("llama_cpp", "n/a", context_tokens, "post_init_vram_mib")
            ort_peak = median_value("ort", "baseline", context_tokens, "peak_inference_vram_mib")
            ort_device_peak = median_value("ort", "device", context_tokens, "peak_inference_vram_mib")
            llama_peak = median_value("llama_cpp", "n/a", context_tokens, "peak_inference_vram_mib")
            allocator_init_delta = (
                ort_device_init - ort_init if ort_device_init is not None and ort_init is not None else None
            )
            allocator_peak_delta = (
                ort_device_peak - ort_peak if ort_device_peak is not None and ort_peak is not None else None
            )
            lines.append(
                f"| {context_tokens} | {format_metric(ort_init)} | "
                f"{format_metric(ort_device_init)} | {format_metric(llama_init)} | "
                f"{format_metric(ort_peak)} | {format_metric(ort_device_peak)} | "
                f"{format_metric(llama_peak)} | "
                f"{format_metric(allocator_init_delta, signed=True)} | "
                f"{format_metric(allocator_peak_delta, signed=True)} |"
            )
            if ort_init is not None and llama_init is not None:
                gap_candidates.append(
                    (
                        abs(ort_init - llama_init),
                        context_tokens,
                        "init/load",
                        f"ORT baseline {ort_init:.2f} MiB vs llama.cpp {llama_init:.2f} MiB",
                    )
                )
            if ort_peak is not None and llama_peak is not None:
                gap_candidates.append(
                    (
                        abs(ort_peak - llama_peak),
                        context_tokens,
                        "inference compute",
                        f"ORT baseline {ort_peak:.2f} MiB vs llama.cpp {llama_peak:.2f} MiB",
                    )
                )

        if gap_candidates:
            gap_mib, context_tokens, phase, comparison = max(gap_candidates)
            lines += [
                "",
                "### Largest ORT-versus-llama.cpp memory gap",
                "",
                f"The largest median absolute gap is **{gap_mib:.2f} MiB** at context "
                f"**{context_tokens}**, during **{phase}** ({comparison}).",
            ]

        lines += [
            "",
            "## Median speed summary",
            "",
            "| Runtime | Allocator | Context | TTFT ms | Decode tok/s |",
            "|---|---|---:|---:|---:|",
        ]
        for (runtime, allocator, context_tokens), group in sorted(groups.items()):
            speed = [float(row["decode_tokens_per_second"]) for row in group]
            ttft = [float(row["ttft_ms"]) for row in group if row.get("ttft_ms") is not None]
            lines.append(
                f"| {runtime} | {allocator} | {context_tokens} | "
                f"{format_metric(statistics.median(ttft) if ttft else None)} | "
                f"{format_metric(statistics.median(speed))} |"
            )
    if sequential_rows:
        lines += [
            "",
            "## Sequential requests",
            "",
            "Requests ran back-to-back in one process with one loaded model. Values below",
            "are direct NVML measurements; no cause is inferred.",
            "",
            "| Runtime | Request | Context | Peak VRAM MiB | Resident after MiB | Status |",
            "|---|---:|---:|---:|---:|---|",
        ]
        for row in sequential_rows:
            lines.append(
                f"| {row['runtime']} | {row['request_index']} | {row['context']} | "
                f"{row.get('peak_vram_mib', '')} | {row.get('resident_after_mib', '')} | "
                f"{row.get('status', 'complete')} |"
            )
    if arena_summary:
        lines += [
            "",
            "## ORT CUDA BFC arena log",
            "",
            f"- Raw log: `{output_dir / 'ort-arena.log'}`",
            f"- Final reported CUDA arena backing size: `{format_metric(arena_summary['final_arena_size_mib'])} MiB`",
            f"- CUDA arena extensions: `{arena_summary['extension_count']}`",
            "- Allocation count: `not available`",
            f"- Note: {arena_summary['allocation_count_note']}",
            "- Free-chunk information: `not available`",
            f"- Note: {arena_summary['free_chunk_note']}",
            "- Logging control: `ORTGENAI_ORT_VERBOSE_LOGGING=1` (set before "
            "`onnxruntime_genai.Config`). `ORT_LOGGING_LEVEL` is not consumed by "
            "ONNX Runtime GenAI 0.15.2.",
        ]
    (output_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def read_existing_results(output_dir: Path) -> list[dict[str, Any]]:
    path = output_dir / "results.csv"
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def format_metric(value: float | None, signed: bool = False) -> str:
    if value is None:
        return "n/a"
    return f"{value:+.2f}" if signed else f"{value:.2f}"


def parse_contexts(value: str) -> list[int]:
    try:
        contexts = [int(item.strip()) for item in value.split(",")]
    except ValueError as error:
        raise argparse.ArgumentTypeError("contexts must be positive comma-separated integers") from error
    if not contexts or any(context <= 0 for context in contexts):
        raise argparse.ArgumentTypeError("contexts must be positive comma-separated integers")
    return contexts


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ort-model", type=Path, help="Exact directory containing genai_config.json")
    parser.add_argument("--gguf-model", type=Path, help="Exact GGUF filename")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--gpu-index", type=int, required=True, help="Physical NVML GPU index (not a CUDA ordinal)")
    parser.add_argument(
        "--validate-only", action="store_true", help="Validate paths/options without GPU or optional imports"
    )
    parser.add_argument(
        "--allow-existing-output",
        action="store_true",
        help="Explicitly allow replacing outputs and merging existing matrix results into a sequential report",
    )
    parser.add_argument("--contexts", type=parse_contexts, default=parse_contexts("2048,4096,8192,16384,32768"))
    parser.add_argument("--output-tokens", type=int, default=64)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--runtime", choices=("ort", "llama_cpp", "both"), default="both")
    parser.add_argument(
        "--worker-timeout-seconds",
        type=float,
        default=DEFAULT_WORKER_TIMEOUT_SECONDS,
        help="Finite positive wall deadline per worker, including load/inference/result transfer/exit (default: 3600)",
    )
    parser.add_argument(
        "--sampling-interval-ms",
        type=float,
        default=10.0,
        help="NVML sampling interval in milliseconds (default: 10)",
    )
    parser.add_argument(
        "--ort-device-allocator",
        action="store_true",
        help="Also run ORT with session.use_device_allocator_for_initializers=1",
    )
    parser.add_argument(
        "--sequential-requests",
        type=int,
        default=1,
        help="Run N requests per runtime in one process; 1 keeps matrix behavior",
    )
    parser.add_argument(
        "--sequential-context",
        type=int,
        default=8192,
        help="Context length for sequential-request mode (default: 8192)",
    )
    parser.add_argument(
        "--log-arena",
        action="store_true",
        help="Capture verbose ORT baseline CUDA BFC arena logs",
    )
    args = parser.parse_args(argv)

    if (
        args.output_tokens <= 0
        or args.repetitions <= 0
        or args.sampling_interval_ms <= 0
        or not math.isfinite(args.sampling_interval_ms)
        or args.worker_timeout_seconds <= 0
        or not math.isfinite(args.worker_timeout_seconds)
        or args.sequential_requests <= 0
        or args.sequential_context <= 0
    ):
        parser.error(
            "--output-tokens, --repetitions, --sampling-interval-ms, --worker-timeout-seconds, "
            "--sequential-requests, and --sequential-context must be positive"
        )
    if args.gpu_index < 0:
        parser.error("--gpu-index must be nonnegative")
    active_contexts = args.contexts if args.sequential_requests == 1 else [args.sequential_context]
    if any(context <= args.output_tokens for context in active_contexts):
        parser.error("every active context must exceed --output-tokens")
    if args.ort_device_allocator and (args.runtime == "llama_cpp" or args.sequential_requests > 1):
        parser.error("--ort-device-allocator requires matrix mode with ORT selected")
    if args.log_arena and args.runtime == "llama_cpp":
        parser.error("--log-arena requires ORT")
    try:
        if args.runtime in ("ort", "both"):
            if args.ort_model is None:
                parser.error("--ort-model is required for the selected runtime")
            args.ort_model = args.ort_model.expanduser().resolve()
            validate_ort_model(args.ort_model)
        if args.runtime in ("llama_cpp", "both"):
            if args.gguf_model is None:
                parser.error("--gguf-model is required for the selected runtime")
            args.gguf_model = args.gguf_model.expanduser().resolve()
            require_file(args.gguf_model)
            if args.gguf_model.suffix.lower() != ".gguf":
                parser.error("--gguf-model must name a .gguf file")
        args.output_dir = args.output_dir.expanduser().resolve()
        for model in (args.ort_model, args.gguf_model):
            if model is None:
                continue
            model_path = model.expanduser().resolve()
            if args.output_dir == model_path or (model_path.is_dir() and model_path in args.output_dir.parents):
                parser.error("--output-dir must not be inside the model artifacts")
        if args.output_dir.exists():
            if not args.output_dir.is_dir():
                parser.error("--output-dir is not a directory")
            if any(args.output_dir.iterdir()) and not args.allow_existing_output:
                parser.error("--output-dir is nonempty; choose a fresh directory or --allow-existing-output")
        ancestor = args.output_dir
        while not ancestor.exists():
            ancestor = ancestor.parent
        if not ancestor.is_dir():
            parser.error("--output-dir has a non-directory ancestor")
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.error(str(error))
    return args


def require_file(path: Path) -> None:
    if not path.is_file() or path.stat().st_size == 0:
        raise ValueError(f"Required nonempty file not found: {path}")


def validate_ort_model(path: Path) -> None:
    # The historical residency denominator is model.onnx*, not arbitrary decoder files.
    for name in ("genai_config.json", "model.onnx", "model.onnx.data", "tokenizer.json", "tokenizer_config.json"):
        require_file(path / name)
    config = json.loads((path / "genai_config.json").read_text(encoding="utf-8"))
    if config["model"]["decoder"]["filename"] != "model.onnx":
        raise ValueError("This reproduction requires decoder filename model.onnx")


def configure_gpu(index: int) -> str:
    import pynvml

    pynvml.nvmlInit()
    try:
        handle = pynvml.nvmlDeviceGetHandleByIndex(index)
        uuid = pynvml.nvmlDeviceGetUUID(handle)
        if isinstance(uuid, bytes):
            uuid = uuid.decode("ascii")
    finally:
        pynvml.nvmlShutdown()
    # UUID selection avoids assuming CUDA enumeration matches physical NVML indices.
    os.environ["CUDA_VISIBLE_DEVICES"] = uuid
    os.environ["BENCHMARK_GPU_INDEX"] = str(index)
    return uuid


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.validate_only:
        print(json.dumps(vars(args), default=str, indent=2))
        return 0
    gpu_uuid = configure_gpu(args.gpu_index)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "run-config.json").write_text(
        json.dumps({**vars(args), "gpu_uuid": gpu_uuid}, default=str, indent=2) + "\n",
        encoding="utf-8",
    )
    arena_log_path = args.output_dir / "ort-arena.log"
    if args.log_arena:
        arena_log_path.write_text("", encoding="utf-8")

    sampling_interval_seconds = args.sampling_interval_ms / 1000
    if args.sequential_requests > 1:
        prompt = build_prompt(args.sequential_context)
        sequential_rows: list[dict[str, Any]] = []
        if args.runtime in ("ort", "both"):
            sequential_rows.extend(
                run_sequential_isolated(
                    "ort",
                    args.ort_model,
                    prompt,
                    args.sequential_context,
                    args.output_tokens,
                    args.sequential_requests,
                    sampling_interval_seconds,
                    args.log_arena,
                    arena_log_path,
                    args.worker_timeout_seconds,
                )
            )
        if args.runtime in ("llama_cpp", "both"):
            sequential_rows.extend(
                run_sequential_isolated(
                    "llama_cpp",
                    args.gguf_model,
                    prompt,
                    args.sequential_context,
                    args.output_tokens,
                    args.sequential_requests,
                    sampling_interval_seconds,
                    False,
                    arena_log_path,
                    args.worker_timeout_seconds,
                )
            )
        write_sequential_csv(sequential_rows, args.output_dir)
        arena_summary = parse_arena_log(arena_log_path) if args.log_arena else None
        write_report(
            read_existing_results(args.output_dir),
            args.output_dir,
            sequential_rows=sequential_rows,
            arena_summary=arena_summary,
            sampling_interval_ms_override=args.sampling_interval_ms,
        )
        for row in sequential_rows:
            print(json.dumps(row, sort_keys=True), flush=True)
        return 0 if sequential_rows and all(row.get("status") == "complete" for row in sequential_rows) else 1

    runners: list[tuple[str, Path, bool]] = []
    if args.runtime in ("ort", "both"):
        runners.append(("ort", args.ort_model, False))
        if args.ort_device_allocator:
            runners.append(("ort", args.ort_model, True))
    if args.runtime in ("llama_cpp", "both"):
        runners.append(("llama_cpp", args.gguf_model, False))

    rows: list[dict[str, Any]] = []
    for context in args.contexts:
        prompt = build_prompt(context)
        for repetition in range(1, args.repetitions + 1):
            for runtime, model_path, use_device_allocator in runners:
                base = {
                    "context_tokens": context,
                    "repetition": repetition,
                    "status": "complete",
                    "sampling_interval_ms": args.sampling_interval_ms,
                    "physical_gpu_index": int(os.environ.get("BENCHMARK_GPU_INDEX", "0")),
                    "ort_allocator_mode": (
                        "device"
                        if runtime == "ort" and use_device_allocator
                        else "baseline"
                        if runtime == "ort"
                        else "n/a"
                    ),
                }
                try:
                    result = run_isolated(
                        runtime,
                        model_path,
                        prompt,
                        context,
                        args.output_tokens,
                        sampling_interval_seconds,
                        use_device_allocator,
                        args.log_arena,
                        arena_log_path,
                        args.worker_timeout_seconds,
                    )
                    result = {**base, **result}
                except Exception as error:
                    result = {**base, **error_result(runtime, error)}
                rows.append(result)
                write_report(
                    rows,
                    args.output_dir,
                    arena_summary=(parse_arena_log(arena_log_path) if args.log_arena else None),
                )
                print(json.dumps(result, sort_keys=True), flush=True)
    return 0 if rows and all(row.get("status") == "complete" for row in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
