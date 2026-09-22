# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Run matched WebGPU cache/workspace controls in C++ and ORT GenAI.

The two harnesses retain their different workloads. Only cache mode and
workspace planning vary within a harness. Memory processes are separate from
latency processes and use one external sampler for both measurement windows.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

BENCHMARK_PATH = Path(__file__).resolve().parents[1] / "cuda" / "ort_genai_workspace_benchmark.py"
SPEC = importlib.util.spec_from_file_location("genai_benchmark", BENCHMARK_PATH)
assert SPEC is not None and SPEC.loader is not None
benchmark = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(benchmark)

CONFIGURATIONS = (("disabled", "scratch"), ("disabled", "planned"), ("bucket", "planned"), ("bucket", "scratch"))


def parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cpp-executable", type=Path, required=True)
    parser.add_argument("--genai-python", type=Path, default=Path(sys.executable))
    parser.add_argument("--qwen15-model-path", type=Path, required=True)
    parser.add_argument("--qwen7-model-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--case", action="append", help="Repeatable MODEL:PROMPT, e.g. qwen15:8192; defaults to the three focused cases"
    )
    parser.add_argument("--warmups", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--memory-iterations", type=int, default=3)
    parser.add_argument("--repetitions", type=int, default=2)
    parser.add_argument("--memory-sample-interval-ms", type=int, default=5)
    args = parser.parse_args()
    if (
        min(args.warmups, args.iterations, args.memory_iterations, args.repetitions, args.memory_sample_interval_ms)
        <= 0
    ):
        parser.error("All run counts and the sampling interval must be positive")
    args.cases = []
    for case in args.case or ("qwen15:8192", "qwen15:12288", "qwen7:4096"):
        model, separator, length = case.partition(":")
        if not separator or model not in ("qwen15", "qwen7") or not length.isdecimal() or int(length) <= 0:
            parser.error(f"Invalid case: {case}")
        args.cases.append((model, int(length)))
    for name in ("cpp_executable", "genai_python", "qwen15_model_path", "qwen7_model_path", "output_dir"):
        setattr(args, name, getattr(args, name).resolve())
    for path in (args.cpp_executable, args.genai_python):
        if not path.is_file():
            parser.error(f"Executable not found: {path}")
    for model_path in (args.qwen15_model_path, args.qwen7_model_path):
        if not (model_path / "model.onnx").is_file() or not (model_path / "genai_config.json").is_file():
            parser.error(f"Model package not found: {model_path}")
    return args


def wait_for_idle_gpu():
    deadline = time.monotonic() + 60
    while True:
        values = subprocess.check_output(
            ["nvidia-smi", "--id=0", "--query-gpu=memory.used,utilization.gpu", "--format=csv,noheader,nounits"],
            text=True,
        ).strip()
        memory, utilization = (int(value.strip()) for value in values.split(","))
        if memory == 0 and utilization == 0:
            return values
        if time.monotonic() >= deadline:
            raise RuntimeError(f"GPU 0 is not idle: memory MiB, utilization % = {values}")
        time.sleep(2)


def run_worker(args, snapshot, model, prompt, harness, phase, block, cache, mode):
    model_path = args.qwen15_model_path if model == "qwen15" else args.qwen7_model_path
    stem = f"{model}-{prompt}-{harness}-{phase}-{block}-{cache}-{mode}"
    log_path = args.output_dir / f"{stem}.log"
    worker_output = args.output_dir / f"{stem}.json"
    env = os.environ.copy()
    env.pop("ORT_MATMULNBITS_TRACE_LEGACY_WORKSPACE", None)
    env.pop("ORTGENAI_ORT_VERBOSE_LOGGING", None)
    if harness == "cpp":
        env.update(
            {
                "ORT_WEBGPU_WORKSPACE_BENCHMARK_MODEL": "qwen2.5-1.5b" if model == "qwen15" else "qwen2.5-7b",
                "ORT_WEBGPU_WORKSPACE_BENCHMARK_MODEL_PATH": str(model_path / "model.onnx"),
                "ORT_WEBGPU_WORKSPACE_BENCHMARK_SEQUENCE_LENGTH": str(prompt),
                "ORT_WEBGPU_WORKSPACE_BENCHMARK_PREALLOCATION": "1" if mode == "planned" else "0",
                "ORT_WEBGPU_WORKSPACE_BENCHMARK_STORAGE_BUFFER_CACHE_MODE": cache,
                "ORT_WEBGPU_WORKSPACE_BENCHMARK_PHASE": phase,
                "ORT_WEBGPU_WORKSPACE_BENCHMARK_WARMUPS": str(args.warmups),
                "ORT_WEBGPU_WORKSPACE_BENCHMARK_ITERATIONS": str(args.iterations),
                "ORT_WEBGPU_WORKSPACE_BENCHMARK_MEMORY_RUNS": str(args.memory_iterations),
            }
        )
        command = [
            str(args.cpp_executable),
            "--gtest_filter=MatMulNBitsWorkspace.WebGpuQwen25WorkspacePreallocationBenchmark",
            "--gtest_color=no",
        ]
        cwd = args.cpp_executable.parent
    else:
        command = [
            str(args.genai_python),
            str(snapshot),
            "--worker",
            "--execution-provider",
            "webgpu",
            "--model-path",
            str(model_path),
            "--mode",
            mode,
            "--phase",
            "scenario" if phase == "latency" else "memory",
            "--prompt-tokens",
            str(prompt),
            "--generated-tokens",
            "128",
            "--warmups",
            str(args.warmups),
            "--iterations",
            str(args.iterations),
            "--memory-iterations",
            str(args.memory_iterations),
            "--webgpu-storage-buffer-cache-mode",
            cache,
            "--webgpu-controlled-comparison",
            "--worker-output",
            str(worker_output),
        ]
        if phase == "memory":
            command.append("--external-memory-sampling")
        cwd = args.genai_python.parent
    record = {
        "model": model,
        "prompt_tokens": prompt,
        "harness": harness,
        "phase": phase,
        "block": block,
        "cache": cache,
        "mode": mode,
        "command": command,
        "log": str(log_path),
        "idle_before": wait_for_idle_gpu(),
    }
    (args.output_dir / "running.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    print(f"START {stem}", flush=True)
    sampler = benchmark.NvidiaSmiMemorySampler(0, args.memory_sample_interval_ms) if phase == "memory" else None
    if sampler is not None:
        sampler.start()
    started = time.monotonic()
    try:
        with log_path.open("wb") as stream:
            completed = subprocess.run(command, check=False, cwd=cwd, env=env, stdout=stream, stderr=subprocess.STDOUT)
    finally:
        if sampler is not None:
            sampler.stop()
            with (args.output_dir / f"{stem}-samples.csv").open("w", newline="", encoding="utf-8") as stream:
                writer = csv.writer(stream)
                writer.writerow(("unix_ns", "device_memory_mib"))
                writer.writerows(sampler.samples)
    record.update(returncode=completed.returncode, elapsed_seconds=time.monotonic() - started)
    if sampler is not None:
        record["whole_process_memory"] = sampler.summarize_window()
    text = log_path.read_bytes().replace(b"\x00", b"").decode("utf-8", errors="replace")
    if completed.returncode != 0:
        record["status"] = "oom" if "VK_ERROR_OUT_OF_DEVICE_MEMORY" in text else "failed"
        print(f"{record['status'].upper()} {stem}", flush=True)
        return record
    if harness == "cpp":
        lines = [line for line in text.splitlines() if "[ WEBGPU WORKSPACE BENCHMARK ]" in line]
        if len(lines) != 1 or "[  PASSED  ] 1 test." not in text:
            raise RuntimeError(f"C++ benchmark did not execute exactly one successful test: {log_path}")
        result = dict(re.findall(r"(\w+)=([^\s]+)", lines[0]))
        expected = {
            "storage_buffer_cache_mode": cache,
            "enable_int64": "1",
            "enable_graph_capture": "0",
            "preferred_layout": "NHWC",
            "warmup_runs": str(args.warmups),
            "sequence_length": str(prompt),
            "workspace_preallocation": "1" if mode == "planned" else "0",
        }
        if any(result[key] != value for key, value in expected.items()):
            raise RuntimeError(f"Unexpected C++ configuration: {result}")
        if mode == "planned" and int(result["planned_workspace_nodes"]) == 0:
            raise RuntimeError("C++ planned mode did not declare any workspace")
        if phase == "latency":
            if int(result["measured_runs"]) != args.iterations or int(result["memory_runs"]) != 0:
                raise RuntimeError("Unexpected C++ latency run counts")
            record["prefill_ms"] = float(result["average_ms"])
        else:
            if int(result["memory_runs"]) != args.memory_iterations or int(result["measured_runs"]) != 0:
                raise RuntimeError("Unexpected C++ memory run counts")
        window = {"start": int(result["memory_start_unix_ns"]), "end": int(result["memory_end_unix_ns"])}
        worker_output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    else:
        result = json.loads(worker_output.read_text(encoding="utf-8"))
        expected_options = [
            {
                "webgpu": {
                    "storageBufferCacheMode": cache,
                    "enableInt64": "1",
                    "enableGraphCapture": "0",
                    "preferredLayout": "NHWC",
                }
            }
        ]
        if result["effective_session_options"]["provider_options"] != expected_options:
            raise RuntimeError(f"Unexpected GenAI provider configuration: {result['effective_session_options']}")
        expected_iterations = args.iterations if phase == "latency" else args.memory_iterations
        if (
            result["output_lengths"] != [prompt + 128]
            or result["iterations"] != expected_iterations
            or result["decode_token_ms"]["count"] != expected_iterations * 127
        ):
            raise RuntimeError(f"GenAI request/token count mismatch: {worker_output}")
        record["output_hashes"] = result["output_hashes"]
        if phase == "latency":
            for target, source in (
                ("prefill_ms", "append_tokens_ms"),
                ("ttft_ms", "request_ttft_ms"),
                ("scenario_ms", "request_scenario_ms"),
                ("decode_total_ms", "decode_total_ms"),
                ("tpot_ms", "decode_token_ms"),
            ):
                record[target] = result[source]["trimmed_mean"]
        window = result["memory_window_unix_ns"]
    record["worker_output"] = str(worker_output)
    if sampler is not None:
        record["post_warmup_memory"] = sampler.summarize_window(window["start"], window["end"])
        record["memory_window_unix_ns"] = window
    record["status"] = "passed"
    print(
        f"PASS {stem} prefill_ms={record.get('prefill_ms')} whole={record.get('whole_process_memory')} "
        f"post={record.get('post_warmup_memory')}",
        flush=True,
    )
    return record


def main():
    args = parse_arguments()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = args.output_dir / "results.json"
    if manifest.exists():
        raise FileExistsError(f"Use a new output directory to preserve existing results: {manifest}")
    snapshot = args.output_dir / "ort_genai_workspace_benchmark.py"
    shutil.copy2(BENCHMARK_PATH, snapshot)
    provenance = {
        "arguments": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "sha256": {
            str(path): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (snapshot, Path(__file__), args.cpp_executable)
        },
        "gpu": subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,name,driver_version,memory.total", "--format=csv,noheader"], text=True
        ).strip(),
    }
    (args.output_dir / "provenance.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    records = []
    for model, prompt in args.cases:
        for phase in ("latency", "memory"):
            for block in range(args.repetitions):
                configurations = CONFIGURATIONS if block % 2 == 0 else tuple(reversed(CONFIGURATIONS))
                harnesses = ("cpp", "genai") if block % 2 == 0 else ("genai", "cpp")
                for harness in harnesses:
                    for cache, mode in configurations:
                        record = run_worker(args, snapshot, model, prompt, harness, phase, block, cache, mode)
                        records.append(record)
                        manifest.write_text(json.dumps(records, indent=2), encoding="utf-8")
                        if record["status"] == "failed":
                            raise RuntimeError(f"Non-OOM worker failure; inspect {record['log']}")
                        hashes = {
                            value
                            for item in records
                            if item["model"] == model and item["prompt_tokens"] == prompt
                            for value in item.get("output_hashes", [])
                        }
                        if len(hashes) > 1:
                            raise RuntimeError(
                                f"GenAI output hashes changed between controlled configurations: {hashes}"
                            )
                        time.sleep(2)
    (args.output_dir / "running.json").unlink()
    print(f"COMPLETE: {len(records)} workers; {sum(r['status'] == 'oom' for r in records)} OOM results", flush=True)


if __name__ == "__main__":
    main()
