# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Benchmark ORT GenAI latency and GPU memory with workspace preallocation.

The controller launches one fresh worker process per mode. This keeps model
initialization, provider allocator state, and memory-pattern state
isolated between scratch and planned-workspace measurements.

The script supports CUDA and WebGPU. It requires an onnxruntime-genai Python
package built against an ONNX Runtime containing the workspace-estimation
changes under test.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib
import itertools
import json
import math
import os
import shutil
import statistics
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np

CUDA_MODES = ("scratch", "matmul", "combined")
WEBGPU_MODES = ("scratch", "planned")
ALL_MODES = tuple(dict.fromkeys((*CUDA_MODES, *WEBGPU_MODES)))


def modes_for_provider(execution_provider: str) -> tuple[str, ...]:
    if execution_provider == "cuda":
        return CUDA_MODES
    if execution_provider == "webgpu":
        return WEBGPU_MODES
    raise ValueError(f"Unsupported execution provider: {execution_provider}")


def percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = fraction * (len(ordered) - 1)
    lower = math.floor(index)
    upper = math.ceil(index)
    if lower == upper:
        return ordered[lower]
    weight = index - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def trimmed_mean(values: list[float], trim_fraction: float = 0.1) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    trim = int(len(ordered) * trim_fraction)
    selected = ordered[trim : len(ordered) - trim] if trim > 0 else ordered
    return statistics.fmean(selected)


def summarize(values: list[float]) -> dict[str, float | int]:
    if not values:
        return {
            "count": 0,
            "mean": 0.0,
            "trimmed_mean": 0.0,
            "median": 0.0,
            "p90": 0.0,
            "p99": 0.0,
            "min": 0.0,
            "max": 0.0,
            "stall_count": 0,
            "stall_rate_pct": 0.0,
        }

    median = statistics.median(values)
    stall_count = sum(value > 3.0 * median for value in values)
    return {
        "count": len(values),
        "mean": statistics.fmean(values),
        "trimmed_mean": trimmed_mean(values),
        "median": median,
        "p90": percentile(values, 0.90),
        "p99": percentile(values, 0.99),
        "min": min(values),
        "max": max(values),
        "stall_count": stall_count,
        "stall_rate_pct": 100.0 * stall_count / len(values),
    }


def summarize_paired_changes(
    process_results: list[dict[str, Any]],
    metric: str,
    candidate_mode: str,
    reference_mode: str,
) -> dict[str, Any]:
    changes: list[float] = []
    for block_index in sorted({int(result["block_index"]) for result in process_results}):
        block = {str(result["mode"]): result for result in process_results if int(result["block_index"]) == block_index}
        candidate_summary = block[candidate_mode][metric]
        reference_summary = block[reference_mode][metric]
        if int(candidate_summary["count"]) == 0 or int(reference_summary["count"]) == 0:
            continue
        candidate = float(candidate_summary["trimmed_mean"])
        reference = float(reference_summary["trimmed_mean"])
        changes.append(100.0 * (candidate / reference - 1.0))

    change_summary = summarize(changes)
    del change_summary["stall_count"]
    del change_summary["stall_rate_pct"]
    return {
        "candidate_mode": candidate_mode,
        "reference_mode": reference_mode,
        "change_pct": change_summary,
        "changes_by_block_pct": changes,
    }


def summarize_paired_memory_changes(
    process_results: list[dict[str, Any]],
    candidate_mode: str,
    reference_mode: str,
) -> dict[str, Any]:
    differences_mib: list[float] = []
    changes_pct: list[float] = []
    for block_index in sorted({int(result["block_index"]) for result in process_results}):
        block = {str(result["mode"]): result for result in process_results if int(result["block_index"]) == block_index}
        candidate = float(block[candidate_mode]["memory"]["device_peak_mib"])
        reference = float(block[reference_mode]["memory"]["device_peak_mib"])
        differences_mib.append(candidate - reference)
        changes_pct.append(100.0 * (candidate / reference - 1.0))

    difference_summary = summarize(differences_mib)
    change_summary = summarize(changes_pct)
    for summary in (difference_summary, change_summary):
        del summary["stall_count"]
        del summary["stall_rate_pct"]
    return {
        "candidate_mode": candidate_mode,
        "reference_mode": reference_mode,
        "difference_mib": difference_summary,
        "differences_by_block_mib": differences_mib,
        "change_pct": change_summary,
        "changes_by_block_pct": changes_pct,
    }


def make_prompt_tokens(config: dict[str, Any], length: int, seed: int) -> np.ndarray:
    model_config = config["model"]
    vocab_size = int(model_config["vocab_size"])
    bos_token_id = int(model_config["bos_token_id"])
    excluded = {bos_token_id, int(model_config.get("pad_token_id", bos_token_id))}
    eos = model_config.get("eos_token_id", [])
    excluded.update(eos if isinstance(eos, list) else [eos])

    rng = np.random.default_rng(seed)
    tokens = rng.integers(0, vocab_size, size=length, dtype=np.int32)
    for index, token in enumerate(tokens):
        replacement = token
        while int(replacement) in excluded:
            replacement = rng.integers(0, vocab_size, dtype=np.int32)
        tokens[index] = replacement
    tokens[0] = bos_token_id
    return tokens


def build_max_shape_override(config: dict[str, Any], prompt_tokens: int, capacity: int) -> str:
    decoder = config["model"]["decoder"]
    inputs = decoder["inputs"]
    if "attention_mask" not in inputs:
        return f"{inputs['input_ids']}:[{prompt_tokens}]"

    shapes = [f"{inputs['input_ids']}:[1,{prompt_tokens}]", f"{inputs['attention_mask']}:[1,{capacity}]"]
    num_layers = int(decoder["num_hidden_layers"])
    num_kv_heads = int(decoder["num_key_value_heads"])
    head_size = int(decoder["head_size"])
    past_key_pattern = inputs["past_key_names"]
    past_value_pattern = inputs["past_value_names"]
    cache_shape = f":[1,{num_kv_heads},{capacity},{head_size}]"
    for layer in range(num_layers):
        shapes.append((past_key_pattern % layer) + cache_shape)
        shapes.append((past_value_pattern % layer) + cache_shape)
    return ";".join(shapes)


def create_benchmark_config(
    model_path: Path,
    execution_provider: str,
    mode: str,
    prompt_tokens: int,
    generated_tokens: int,
    fpa_intb: bool,
    device_id: int,
    output_directory: Path,
    webgpu_cache_mode: str | None = None,
    webgpu_controlled_comparison: bool = False,
) -> Path:
    if mode not in modes_for_provider(execution_provider):
        raise ValueError(f"Mode {mode!r} is invalid for {execution_provider}")
    if execution_provider == "webgpu" and fpa_intb:
        raise ValueError("fpA-intB is only supported for CUDA")
    if execution_provider != "webgpu" and (webgpu_cache_mode is not None or webgpu_controlled_comparison):
        raise ValueError("WebGPU provider controls require the WebGPU execution provider")
    if webgpu_cache_mode not in (None, "disabled", "bucket"):
        raise ValueError("WebGPU storage buffer cache mode must be disabled or bucket")
    source_config_path = model_path / "genai_config.json"
    config = json.loads(source_config_path.read_text(encoding="utf-8"))
    decoder = config["model"]["decoder"]

    capacity = prompt_tokens + generated_tokens
    if capacity > int(config["model"]["context_length"]):
        raise ValueError(f"Prompt plus generation ({capacity}) exceeds the model context length")
    session_options = decoder.setdefault("session_options", {})
    if execution_provider == "cuda":
        provider_options = session_options.setdefault("provider_options", [])
        cuda_options = next((provider["cuda"] for provider in provider_options if "cuda" in provider), None)
        if cuda_options is None:
            cuda_options = {}
            provider_options.append({"cuda": cuda_options})
        cuda_options["device_id"] = str(device_id)
    else:
        webgpu_options = next(
            (
                options
                for provider in session_options.get("provider_options", [])
                for name, options in provider.items()
                if name.lower() == "webgpu"
            ),
            {},
        )
        if webgpu_cache_mode is not None:
            webgpu_options["storageBufferCacheMode"] = webgpu_cache_mode
        if webgpu_controlled_comparison:
            webgpu_options.update(enableInt64="1", enableGraphCapture="0", preferredLayout="NHWC")
        session_options["provider_options"] = [{"webgpu": webgpu_options}]

    session_options["session.enable_static_workspace_preallocation"] = "0" if mode == "scratch" else "1"
    session_options["session.max_shape_override"] = build_max_shape_override(config, prompt_tokens, capacity)
    if execution_provider == "cuda":
        session_options["ep.cuda.fpa_intb_gemm"] = "1" if fpa_intb else "0"
        if mode == "combined":
            session_options["ep.cuda.gqa_workspace_max_total_sequence_length"] = str(capacity)
        else:
            session_options.pop("ep.cuda.gqa_workspace_max_total_sequence_length", None)
    else:
        session_options.pop("ep.cuda.fpa_intb_gemm", None)
        session_options.pop("ep.cuda.gqa_workspace_max_total_sequence_length", None)

    search = config.setdefault("search", {})
    use_engine = "engine" in config
    search.update(
        {
            "batch_size": 1,
            "do_sample": False,
            "early_stopping": False,
            "max_length": capacity,
            "min_length": 0 if use_engine else capacity,
            "num_beams": 1,
            "num_return_sequences": 1,
            "past_present_share_buffer": True,
            "random_seed": 0,
        }
    )

    output_directory.mkdir(parents=True, exist_ok=True)
    for source in model_path.iterdir():
        if source.name == source_config_path.name:
            continue
        destination = output_directory / source.name
        if source.is_dir():
            shutil.copytree(source, destination, copy_function=os.link)
        else:
            os.link(source, destination)

    config_path = output_directory / "genai_config.json"
    config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")
    return config_path


class NvidiaSmiMemorySampler:
    """Samples device-wide GPU memory in a memory-only worker process."""

    def __init__(self, device_id: int, interval_ms: int) -> None:
        self._device_id = device_id
        self._interval_ms = interval_ms
        self._process: subprocess.Popen[str] | None = None
        self._thread: threading.Thread | None = None
        self.samples: list[tuple[int, int]] = []
        self._error: str | None = None
        self._peak_start_ns: int | None = None

    def start(self) -> None:
        command = [
            "nvidia-smi",
            f"--id={self._device_id}",
            "--query-gpu=memory.used",
            "--format=csv,noheader,nounits",
            f"--loop-ms={self._interval_ms}",
        ]
        self._process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
        )

        def read_samples() -> None:
            assert self._process is not None
            assert self._process.stdout is not None
            for line in self._process.stdout:
                if not line.strip():
                    continue
                try:
                    value = int(line.strip())
                except ValueError:
                    self._error = f"Invalid nvidia-smi memory sample: {line.strip()}"
                    return
                self.samples.append((time.time_ns(), value))

        self._thread = threading.Thread(target=read_samples, daemon=True)
        self._thread.start()
        deadline = time.monotonic() + 5.0
        while not self.samples and self._error is None and time.monotonic() < deadline:
            if self._process.poll() is not None:
                break
            time.sleep(0.01)
        if not self.samples:
            self.stop()
            raise RuntimeError("nvidia-smi did not produce a memory sample")

    def reset_peak(self) -> int:
        self._peak_start_ns, baseline = self.samples[-1]
        return baseline

    def summarize_window(self, start_ns: int | None = None, end_ns: int | None = None) -> dict[str, float | int]:
        if not self.samples:
            raise RuntimeError("nvidia-smi did not produce a memory sample")
        start_ns = self.samples[0][0] if start_ns is None else start_ns
        end_ns = self.samples[-1][0] if end_ns is None else end_ns
        selected = [(timestamp, value) for timestamp, value in self.samples if start_ns <= timestamp <= end_ns]
        if not selected:
            raise RuntimeError("No nvidia-smi samples fell inside the measurement window")
        baseline = selected[0][1]
        peak = max(value for _, value in selected)
        gaps = [right[0] - left[0] for left, right in itertools.pairwise(selected)]
        return {
            "device_baseline_mib": baseline,
            "device_peak_mib": peak,
            "device_peak_delta_mib": peak - baseline,
            "sample_count": len(selected),
            "max_sample_gap_ms": max(gaps, default=0) / 1_000_000,
        }

    def stop(self) -> int:
        unexpected_exit = self._process is not None and self._process.poll() is not None
        if self._process is not None and not unexpected_exit:
            self._process.terminate()
            try:
                self._process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._process.kill()
                self._process.wait(timeout=5)
        if self._thread is not None:
            self._thread.join(timeout=5)
            if self._thread.is_alive():
                raise RuntimeError("nvidia-smi sample reader did not terminate")
        stderr = ""
        if self._process is not None:
            if self._process.stdout is not None:
                self._process.stdout.close()
            if self._process.stderr is not None:
                stderr = self._process.stderr.read().strip()
                self._process.stderr.close()
        if self._error is not None:
            raise RuntimeError(self._error)
        if unexpected_exit:
            raise RuntimeError(f"nvidia-smi exited unexpectedly: {stderr}")
        return int(self.summarize_window(self._peak_start_ns)["device_peak_mib"])


def run_generation(
    og: Any,
    model: Any,
    prompt: np.ndarray,
    generated_tokens: int,
) -> dict[str, float | list[float] | int]:
    request_start = time.perf_counter()
    params = og.GeneratorParams(model)
    total_length = len(prompt) + generated_tokens
    params.set_search_options(
        batch_size=1,
        do_sample=False,
        early_stopping=False,
        max_length=total_length,
        min_length=total_length,
        num_beams=1,
        num_return_sequences=1,
        random_seed=0,
    )
    generator = og.Generator(model, params)
    append_start = time.perf_counter()
    generator.append_tokens(prompt)
    append_end = time.perf_counter()
    generator.generate_next_token()
    generator.get_next_tokens()
    first_token_end = time.perf_counter()

    decode_ms: list[float] = []
    while not generator.is_done():
        token_start = time.perf_counter()
        generator.generate_next_token()
        generator.get_next_tokens()
        decode_ms.append((time.perf_counter() - token_start) * 1000.0)

    scenario_end = time.perf_counter()
    sequence = np.asarray(generator.get_sequence(0), dtype=np.int32)
    if len(sequence) != total_length or len(decode_ms) != generated_tokens - 1:
        raise RuntimeError(
            f"Expected {generated_tokens} generated tokens and {generated_tokens - 1} decode evaluations, "
            f"got {len(sequence) - len(prompt)} tokens and {len(decode_ms)} evaluations"
        )
    result = {
        "generator_setup_ms": (append_start - request_start) * 1000.0,
        "append_tokens_ms": (append_end - append_start) * 1000.0,
        "sampling_ms": (first_token_end - append_end) * 1000.0,
        "request_ttft_ms": (first_token_end - request_start) * 1000.0,
        "model_ttft_ms": (first_token_end - append_start) * 1000.0,
        "request_scenario_ms": (scenario_end - request_start) * 1000.0,
        "model_scenario_ms": (scenario_end - append_start) * 1000.0,
        "decode_total_ms": (scenario_end - first_token_end) * 1000.0,
        "decode_ms": decode_ms,
        "output_hash": hashlib.sha256(sequence.tobytes()).hexdigest(),
        "output_length": len(sequence),
    }
    del generator
    gc.collect()
    return result


def run_engine_generation(
    og: Any,
    engine: Any,
    prompt: np.ndarray,
    generated_tokens: int,
) -> dict[str, float | list[float] | int]:
    request_start = time.perf_counter()
    request_options = og.RequestOptions()
    request_options.set_max_session_tokens(len(prompt) + generated_tokens)
    request = engine.create_request(options=request_options)
    turn_options = og.TurnOptions(request)
    turn_options.set_do_sample(False)
    turn_options.set_min_generated_tokens(generated_tokens)
    turn_options.set_max_generated_tokens(generated_tokens)
    turn_options.set_seed(0)

    append_start = time.perf_counter()
    request.begin_turn(prompt, turn_options)
    append_end = time.perf_counter()
    first_token_end: float | None = None
    decode_ms: list[float] = []
    output_tokens: list[int] = []
    event_buffer = engine.create_event_buffer(8)

    try:
        while engine.has_pending_requests():
            token_start = time.perf_counter()
            events = engine.run(event_buffer)
            token_end = time.perf_counter()
            for event in events:
                if event.flags & og.EngineEventFlags.FAILED:
                    raise RuntimeError(f"Engine generation failed; error_code={event.error_code}")
                if event.request is not request:
                    raise RuntimeError("Engine returned an event for an unexpected request")
                if event.flags & og.EngineEventFlags.TOKEN:
                    output_tokens.append(int(event.token))
                    if first_token_end is None:
                        first_token_end = token_end
                    else:
                        decode_ms.append((token_end - token_start) * 1000.0)
    finally:
        request.close()

    if first_token_end is None:
        raise RuntimeError("Engine request completed without producing a token")

    scenario_end = time.perf_counter()
    if len(output_tokens) != generated_tokens or len(decode_ms) != generated_tokens - 1:
        raise RuntimeError(f"Expected {generated_tokens} generated tokens, got {len(output_tokens)}")
    sequence = np.concatenate((prompt, np.asarray(output_tokens, dtype=np.int32)))
    result = {
        "generator_setup_ms": (append_start - request_start) * 1000.0,
        "append_tokens_ms": (append_end - append_start) * 1000.0,
        "sampling_ms": (first_token_end - append_end) * 1000.0,
        "request_ttft_ms": (first_token_end - request_start) * 1000.0,
        "model_ttft_ms": (first_token_end - append_start) * 1000.0,
        "request_scenario_ms": (scenario_end - request_start) * 1000.0,
        "model_scenario_ms": (scenario_end - append_start) * 1000.0,
        "decode_total_ms": (scenario_end - first_token_end) * 1000.0,
        "decode_ms": decode_ms,
        "output_hash": hashlib.sha256(sequence.tobytes()).hexdigest(),
        "output_length": len(sequence),
    }
    gc.collect()
    return result


def run_worker(args: argparse.Namespace) -> dict[str, Any]:
    try:
        og = importlib.import_module("onnxruntime_genai")
    except ImportError as error:
        raise RuntimeError(
            "onnxruntime-genai is not installed. Install or build a package "
            "that uses the ONNX Runtime changes under test."
        ) from error

    model_path = Path(args.model_path).resolve()
    source_config = json.loads((model_path / "genai_config.json").read_text(encoding="utf-8"))
    prompt = make_prompt_tokens(source_config, args.prompt_tokens, args.seed)

    with tempfile.TemporaryDirectory(
        prefix="ort-genai-workspace-",
        dir=model_path.parent,
    ) as temporary_directory:
        config_path = create_benchmark_config(
            model_path,
            args.execution_provider,
            args.mode,
            args.prompt_tokens,
            args.generated_tokens,
            args.fpa_intb,
            args.device_id,
            Path(temporary_directory),
            args.webgpu_storage_buffer_cache_mode,
            args.webgpu_controlled_comparison,
        )
        effective_session_options = json.loads(config_path.read_text(encoding="utf-8"))["model"]["decoder"][
            "session_options"
        ]
        model_load_start = time.perf_counter()
        model = og.Model(str(config_path.parent))
        use_engine = "engine" in source_config
        engine = og.Engine(model) if use_engine else None
        model_load_ms = (time.perf_counter() - model_load_start) * 1000.0
        generate = (
            (lambda: run_engine_generation(og, engine, prompt, args.generated_tokens))
            if engine is not None
            else (lambda: run_generation(og, model, prompt, args.generated_tokens))
        )

        for _ in range(args.warmups):
            generate()

        samples: list[dict[str, Any]] = []
        memory: dict[str, int] | None = None
        memory_window: dict[str, int] | None = None
        if args.phase == "memory":
            sampler = (
                None
                if args.external_memory_sampling
                else NvidiaSmiMemorySampler(args.device_id, args.memory_sample_interval_ms)
            )
            if sampler is not None:
                sampler.start()
                baseline_mib = sampler.reset_peak()
            memory_start_ns = time.time_ns()
            try:
                for _ in range(args.memory_iterations):
                    samples.append(generate())
            finally:
                memory_end_ns = time.time_ns()
                if sampler is not None:
                    peak_mib = sampler.stop()
            memory_window = {"start": memory_start_ns, "end": memory_end_ns}
            if sampler is not None:
                memory = {
                    "device_baseline_mib": baseline_mib,
                    "device_peak_mib": peak_mib,
                    "device_peak_delta_mib": peak_mib - baseline_mib,
                }
        else:
            for _ in range(args.iterations):
                samples.append(generate())

        generator_setup = [float(sample["generator_setup_ms"]) for sample in samples]
        append_tokens = [float(sample["append_tokens_ms"]) for sample in samples]
        sampling = [float(sample["sampling_ms"]) for sample in samples]
        request_ttft = [float(sample["request_ttft_ms"]) for sample in samples]
        model_ttft = [float(sample["model_ttft_ms"]) for sample in samples]
        request_scenario = [float(sample["request_scenario_ms"]) for sample in samples]
        model_scenario = [float(sample["model_scenario_ms"]) for sample in samples]
        decode_total = [float(sample["decode_total_ms"]) for sample in samples]
        decode = [float(value) for sample in samples for value in sample["decode_ms"]]
        output_hashes = {str(sample["output_hash"]) for sample in samples}
        lengths = {int(sample["output_length"]) for sample in samples}
        if len(output_hashes) != 1 or len(lengths) != 1:
            raise RuntimeError("Generated output changed between identical measured iterations")

        return {
            "execution_provider": args.execution_provider,
            "mode": args.mode,
            "phase": args.phase,
            "model_path": str(model_path),
            "fpa_intb": args.fpa_intb,
            "effective_session_options": effective_session_options,
            "prompt_tokens": args.prompt_tokens,
            "generated_tokens": args.generated_tokens,
            "generation_api": "engine" if use_engine else "generator",
            "warmups": args.warmups,
            "iterations": len(samples),
            "model_load_ms": model_load_ms,
            "generator_setup_ms": summarize(generator_setup),
            "append_tokens_ms": summarize(append_tokens),
            "sampling_ms": summarize(sampling),
            "request_ttft_ms": summarize(request_ttft),
            "model_ttft_ms": summarize(model_ttft),
            "request_scenario_ms": summarize(request_scenario),
            "model_scenario_ms": summarize(model_scenario),
            "decode_total_ms": summarize(decode_total),
            "decode_token_ms": summarize(decode),
            "output_hashes": sorted(output_hashes),
            "output_lengths": sorted(lengths),
            "memory": memory,
            "memory_window_unix_ns": memory_window,
        }


def worker_command(args: argparse.Namespace, mode: str, output_path: Path) -> list[str]:
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker",
        "--model-path",
        args.model_path,
        "--execution-provider",
        args.execution_provider,
        "--mode",
        mode,
        "--phase",
        args.phase,
        "--prompt-tokens",
        str(args.prompt_tokens),
        "--generated-tokens",
        str(args.generated_tokens),
        "--warmups",
        str(args.warmups),
        "--iterations",
        str(args.iterations),
        "--seed",
        str(args.seed),
        "--device-id",
        str(args.device_id),
        "--memory-sample-interval-ms",
        str(args.memory_sample_interval_ms),
        "--memory-iterations",
        str(args.memory_iterations),
        "--worker-output",
        str(output_path),
    ]
    command.append("--fpa-intb" if args.fpa_intb else "--no-fpa-intb")
    if args.webgpu_storage_buffer_cache_mode is not None:
        command.extend(("--webgpu-storage-buffer-cache-mode", args.webgpu_storage_buffer_cache_mode))
    if args.webgpu_controlled_comparison:
        command.append("--webgpu-controlled-comparison")
    return command


def run_controller(args: argparse.Namespace) -> dict[str, Any]:
    output_path = Path(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    process_results: list[dict[str, Any]] = []

    modes = modes_for_provider(args.execution_provider)
    mode_orders = tuple(itertools.permutations(modes))
    orders = [mode_orders[index % len(mode_orders)] for index in range(args.repetitions)]

    with tempfile.TemporaryDirectory(prefix="ort-genai-workspace-controller-") as temporary_directory:
        temporary_path = Path(temporary_directory)
        process_index = 0
        for block_index, order in enumerate(orders):
            for mode in order:
                worker_output = temporary_path / f"worker-{process_index}.json"
                subprocess.run(worker_command(args, mode, worker_output), check=True)
                result = json.loads(worker_output.read_text(encoding="utf-8"))
                result["block_index"] = block_index
                result["process_index"] = process_index
                process_results.append(result)
                process_index += 1
                print(
                    f"{mode:8s} block={block_index} "
                    f"ttft={result['request_ttft_ms']['trimmed_mean']:.3f} ms "
                    f"scenario={result['request_scenario_ms']['trimmed_mean']:.3f} ms "
                    f"tpot={result['decode_token_ms']['trimmed_mean']:.3f} ms",
                    flush=True,
                )

    output_hashes = {output_hash for result in process_results for output_hash in result["output_hashes"]}
    output_lengths = {output_length for result in process_results for output_length in result["output_lengths"]}
    output_hashes_by_mode = {
        mode: sorted(
            {
                output_hash
                for result in process_results
                if result["mode"] == mode
                for output_hash in result["output_hashes"]
            }
        )
        for mode in modes
    }
    mode_hash_sets_match = len({tuple(hashes) for hashes in output_hashes_by_mode.values()}) == 1
    output_lengths_match = len(output_lengths) == 1
    outputs_match = mode_hash_sets_match and output_lengths_match

    aggregate: dict[str, Any] = {}
    metrics = (
        "generator_setup_ms",
        "append_tokens_ms",
        "sampling_ms",
        "request_ttft_ms",
        "model_ttft_ms",
        "request_scenario_ms",
        "model_scenario_ms",
        "decode_total_ms",
        "decode_token_ms",
    )
    for mode in modes:
        mode_results = [result for result in process_results if result["mode"] == mode]
        aggregate[mode] = {
            metric: summarize([float(result[metric]["trimmed_mean"]) for result in mode_results]) for metric in metrics
        }
        if args.phase == "memory":
            aggregate[mode]["memory"] = {
                metric: summarize([float(result["memory"][metric]) for result in mode_results])
                for metric in (
                    "device_baseline_mib",
                    "device_peak_mib",
                    "device_peak_delta_mib",
                )
            }
        else:
            aggregate[mode]["memory"] = None

    comparisons = (
        (("matmul", "scratch"), ("combined", "scratch"), ("combined", "matmul"))
        if args.execution_provider == "cuda"
        else (("planned", "scratch"),)
    )
    paired_changes = None
    paired_memory = None
    if args.phase != "memory":
        paired_changes = {
            metric: {
                f"{candidate}_vs_{reference}": summarize_paired_changes(process_results, metric, candidate, reference)
                for candidate, reference in comparisons
            }
            for metric in metrics
        }
    else:
        paired_memory = {
            f"{candidate}_vs_{reference}": summarize_paired_memory_changes(process_results, candidate, reference)
            for candidate, reference in comparisons
        }

    report = {
        "execution_provider": args.execution_provider,
        "phase": args.phase,
        "model_path": str(Path(args.model_path).resolve()),
        "fpa_intb": args.fpa_intb,
        "prompt_tokens": args.prompt_tokens,
        "generated_tokens": args.generated_tokens,
        "repetitions": args.repetitions,
        "process_results": process_results,
        "aggregate": aggregate,
        "paired_changes": paired_changes,
        "paired_memory": paired_memory,
        "output_validation": {
            "outputs_match": outputs_match,
            "mode_hash_sets_match": mode_hash_sets_match,
            "single_output_hash": len(output_hashes) == 1,
            "hashes_by_mode": output_hashes_by_mode,
            "output_lengths": sorted(output_lengths),
        },
        "memory_scope": (
            "nvidia-smi device-wide memory.used; run on an otherwise idle GPU" if args.phase == "memory" else None
        ),
    }
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    if not outputs_match:
        raise RuntimeError(f"Generated output hash sets differ by mode: {output_hashes_by_mode}")
    return report


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True, help="Directory containing genai_config.json and model.onnx")
    parser.add_argument("--execution-provider", choices=("cuda", "webgpu"), default="cuda")
    parser.add_argument("--phase", choices=("ttft", "scenario", "memory"), default="ttft")
    parser.add_argument("--mode", choices=ALL_MODES, default="scratch", help=argparse.SUPPRESS)
    parser.add_argument("--prompt-tokens", type=int, default=1024)
    parser.add_argument("--generated-tokens", type=int, default=128)
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--repetitions", type=int, default=6)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument(
        "--device-id",
        type=int,
        default=0,
        help="CUDA device / nvidia-smi index; for WebGPU only index 0 is supported on an otherwise idle single-GPU system",
    )
    parser.add_argument("--memory-sample-interval-ms", type=int, default=5)
    parser.add_argument("--memory-iterations", type=int, default=1)
    parser.add_argument("--webgpu-storage-buffer-cache-mode", choices=("disabled", "bucket"))
    parser.add_argument(
        "--webgpu-controlled-comparison",
        action="store_true",
        help="Match the C++ WebGPU benchmark: enableInt64=1, enableGraphCapture=0, preferredLayout=NHWC",
    )
    parser.add_argument(
        "--external-memory-sampling",
        action="store_true",
        help="Memory workers only: emit the measurement window; the caller must collect device memory externally",
    )
    parser.add_argument("--fpa-intb", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--output", default="ort_genai_workspace_benchmark.json")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--worker-output", help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.prompt_tokens <= 0:
        parser.error("--prompt-tokens must be positive")
    if args.generated_tokens <= 0:
        parser.error("--generated-tokens must be positive")
    if args.warmups < 0 or args.iterations <= 0 or args.repetitions <= 0:
        parser.error("warmups must be non-negative; iterations and repetitions must be positive")
    if args.device_id < 0 or args.memory_sample_interval_ms <= 0 or args.memory_iterations <= 0:
        parser.error("device-id must be non-negative; memory-sample-interval-ms and memory-iterations must be positive")
    if args.execution_provider != "webgpu" and (
        args.webgpu_storage_buffer_cache_mode is not None or args.webgpu_controlled_comparison
    ):
        parser.error("WebGPU provider controls require --execution-provider webgpu")
    if args.external_memory_sampling and (not args.worker or args.phase != "memory"):
        parser.error("--external-memory-sampling requires --worker --phase memory")
    if args.phase == "ttft":
        args.generated_tokens = 1
    valid_modes = modes_for_provider(args.execution_provider)
    if args.mode not in valid_modes:
        parser.error(f"--mode {args.mode!r} is invalid for {args.execution_provider}; choose from {valid_modes}")
    if args.execution_provider == "webgpu" and args.fpa_intb:
        parser.error("--fpa-intb is only valid with --execution-provider cuda")
    if args.execution_provider == "webgpu" and args.device_id != 0:
        parser.error("WebGPU adapter selection by --device-id is not supported; use index 0 on a single-GPU system")
    if args.worker and not args.worker_output:
        parser.error("--worker-output is required with --worker")
    return args


def main() -> None:
    args = parse_arguments()
    if args.worker:
        result = run_worker(args)
        Path(args.worker_output).write_text(json.dumps(result, indent=2), encoding="utf-8")
        return
    run_controller(args)


if __name__ == "__main__":
    main()
