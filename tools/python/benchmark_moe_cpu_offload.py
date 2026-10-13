#!/usr/bin/env python3

"""Measure warmed GenAI decode with optional CUDA MoE CPU offload and adaptive swaps."""

import argparse
import ctypes
import importlib
import json
import math
import os
import resource
import statistics
import sys
import threading
import time
from pathlib import Path

from qmoe_prompt_runner import configure_provider, format_prompt, redirect_native_stderr


class NvmlMemory:
    class Info(ctypes.Structure):
        _fields_ = [
            ("total", ctypes.c_ulonglong),
            ("free", ctypes.c_ulonglong),
            ("used", ctypes.c_ulonglong),
        ]

    def __init__(self, physical_gpu_index):
        self.library = ctypes.CDLL("libnvidia-ml.so.1")
        self.library.nvmlInit_v2.restype = ctypes.c_int
        self.library.nvmlDeviceGetHandleByIndex_v2.argtypes = [
            ctypes.c_uint,
            ctypes.POINTER(ctypes.c_void_p),
        ]
        self.library.nvmlDeviceGetHandleByIndex_v2.restype = ctypes.c_int
        self.library.nvmlDeviceGetMemoryInfo.argtypes = [ctypes.c_void_p, ctypes.POINTER(self.Info)]
        self.library.nvmlDeviceGetMemoryInfo.restype = ctypes.c_int
        self.library.nvmlShutdown.restype = ctypes.c_int
        self._check(self.library.nvmlInit_v2())
        self.handle = ctypes.c_void_p()
        self._check(self.library.nvmlDeviceGetHandleByIndex_v2(physical_gpu_index, ctypes.byref(self.handle)))

    @staticmethod
    def _check(result):
        if result != 0:
            raise RuntimeError(f"NVML call failed with error {result}.")

    def used_mib(self):
        info = self.Info()
        self._check(self.library.nvmlDeviceGetMemoryInfo(self.handle, ctypes.byref(info)))
        return info.used / (1024 * 1024)

    def close(self):
        self._check(self.library.nvmlShutdown())


def current_rss_mib():
    fields = Path("/proc/self/statm").read_text(encoding="utf-8").split()
    return int(fields[1]) * os.sysconf("SC_PAGE_SIZE") / (1024 * 1024)


class MemorySampler:
    def __init__(self, physical_gpu_index, interval_seconds=0.05):
        self.nvml = NvmlMemory(physical_gpu_index)
        self.interval_seconds = interval_seconds
        self.baseline_cpu_rss_mib = current_rss_mib()
        self.baseline_gpu_used_mib = self.nvml.used_mib()
        self.peak_cpu_rss_mib = self.baseline_cpu_rss_mib
        self.peak_gpu_used_mib = self.baseline_gpu_used_mib
        self.phase = "model_load"
        self.phase_peaks = {}
        self.error = None
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _sample(self):
        cpu_rss_mib = current_rss_mib()
        gpu_used_mib = self.nvml.used_mib()
        self.peak_cpu_rss_mib = max(self.peak_cpu_rss_mib, cpu_rss_mib)
        self.peak_gpu_used_mib = max(self.peak_gpu_used_mib, gpu_used_mib)
        peaks = self.phase_peaks.setdefault(self.phase, {"cpu_rss_mib": cpu_rss_mib, "gpu_used_mib": gpu_used_mib})
        peaks["cpu_rss_mib"] = max(peaks["cpu_rss_mib"], cpu_rss_mib)
        peaks["gpu_used_mib"] = max(peaks["gpu_used_mib"], gpu_used_mib)

    def _run(self):
        try:
            while not self.stop_event.wait(self.interval_seconds):
                self._sample()
        except Exception as error:
            # Transport background sampling failures to the caller instead of emitting valid-looking results.
            self.error = error

    def set_phase(self, phase):
        self.phase = phase
        self._sample()

    def snapshot(self):
        return {"cpu_rss_mib": current_rss_mib(), "gpu_used_mib": self.nvml.used_mib()}

    def stop(self):
        self.stop_event.set()
        self.thread.join()
        if self.error is not None:
            self.nvml.close()
            raise self.error
        self._sample()
        result = {
            "baseline_cpu_rss_mib": self.baseline_cpu_rss_mib,
            "baseline_gpu_used_mib": self.baseline_gpu_used_mib,
            "peak_cpu_rss_mib": max(
                self.peak_cpu_rss_mib,
                resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
            ),
            "peak_gpu_used_mib": self.peak_gpu_used_mib,
            "phase_peaks": self.phase_peaks,
        }
        self.nvml.close()
        return result


def model_session_options(args):
    session_options = {
        "log_severity_level": 1 if args.counter_log else 3,
        "session.enable_moe_expert_statistics": "1" if args.counter_log else "0",
    }
    if args.counter_log or args.cpu_offload_experts > 0:
        session_options["session.moe_expert_counter_alpha"] = str(args.counter_alpha)
    if args.state is not None:
        session_options["session.moe_expert_counter_state_file"] = str(args.state)
    if args.cpu_offload_experts > 0:
        session_options["session.moe_cpu_offload_experts"] = str(args.cpu_offload_experts)
        session_options["session.moe_expert_swap_epsilon"] = str(args.swap_epsilon)
        session_options["session.moe_max_in_flight_swaps_per_device"] = "4" if args.swap else "0"
    return session_options


def create_model(genai, args):
    config = genai.Config(str(args.model))
    configure_provider(config, args.provider, 1)
    if args.provider == "cuda":
        config.set_provider_option("cuda", "enable_cuda_graph", "1" if args.cuda_graph else "0")
    elif args.cuda_graph or args.cpu_offload_experts > 0:
        raise ValueError("CUDA Graph and MoE CPU offload require the CUDA provider.")
    config.overlay(json.dumps({"model": {"decoder": {"session_options": model_session_options(args)}}}))
    return genai.Model(config)


def load_prompts(path):
    data = json.loads(path.read_text(encoding="utf-8"))
    prompts = [item["prompt"] if isinstance(item, dict) else item for item in data]
    if not prompts or any(not isinstance(prompt, str) or not prompt for prompt in prompts):
        raise ValueError("The benchmark requires non-empty prompts.")
    return prompts


def summarize(values):
    if not values:
        raise ValueError("Cannot summarize an empty timing sample.")
    return {
        "mean_ms": statistics.mean(values) * 1000,
        "median_ms": statistics.median(values) * 1000,
        "min_ms": min(values) * 1000,
        "max_ms": max(values) * 1000,
    }


def decode_summary(iteration_seconds):
    # The first generation call uses logits already computed by append_tokens(), not another inference.
    return summarize(iteration_seconds[1:])


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument("--state", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--prompt-limit", type=int)
    parser.add_argument("--provider", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--cpu-offload-experts", type=int, required=True)
    parser.add_argument("--counter-alpha", type=float, default=0.9)
    parser.add_argument("--swap-epsilon", type=float, default=0.1)
    parser.add_argument("--swap", action=argparse.BooleanOptionalAction, required=True)
    parser.add_argument("--counter-log", action=argparse.BooleanOptionalAction, required=True)
    parser.add_argument("--cuda-graph", action=argparse.BooleanOptionalAction, required=True)
    args = parser.parse_args()
    if args.max_new_tokens < 2:
        parser.error("--max-new-tokens must be at least 2 to measure actual decode inference.")
    if args.cpu_offload_experts < 0:
        parser.error("--cpu-offload-experts must be non-negative.")
    if not math.isfinite(args.counter_alpha) or not 0 <= args.counter_alpha <= 1:
        parser.error("--counter-alpha must be finite and in [0, 1].")
    if not math.isfinite(args.swap_epsilon) or args.swap_epsilon < 0:
        parser.error("--swap-epsilon must be finite and non-negative.")
    return args


def main():
    args = parse_args()
    prompts = load_prompts(args.prompts)
    if args.prompt_limit is not None:
        if args.prompt_limit <= 0 or args.prompt_limit > len(prompts):
            raise ValueError("--prompt-limit must be between 1 and the number of available prompts.")
        prompts = prompts[: args.prompt_limit]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    visible_gpu = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if not visible_gpu.isdigit():
        raise ValueError("CUDA_VISIBLE_DEVICES must contain one physical GPU index.")

    genai = importlib.import_module("onnxruntime_genai")

    memory_sampler = MemorySampler(int(visible_gpu))
    with redirect_native_stderr(args.log):
        model_start = time.perf_counter()
        model = create_model(genai, args)
        model_load_seconds = time.perf_counter() - model_start
        memory_after_model_load = memory_sampler.snapshot()
        tokenizer = genai.Tokenizer(model)
        results = []
        for prompt_index, prompt in enumerate(prompts, start=1):
            memory_sampler.set_phase(f"prompt_{prompt_index}")
            prompt_tokens = tokenizer.encode(format_prompt(tokenizer, prompt, False))
            params = genai.GeneratorParams(model)
            params.set_search_options(
                max_length=len(prompt_tokens) + args.max_new_tokens,
                min_length=len(prompt_tokens) + args.max_new_tokens,
                do_sample=False,
            )
            generator = genai.Generator(model, params)
            print(f"[qmoe_prompt_suite] prompt_start {prompt_index}", file=sys.stderr, flush=True)
            print(f"[qmoe_prompt_suite] prefill_start {prompt_index}", file=sys.stderr, flush=True)
            prefill_start = time.perf_counter()
            generator.append_tokens(prompt_tokens)
            prefill_seconds = time.perf_counter() - prefill_start
            print(f"[qmoe_prompt_suite] prefill_end {prompt_index}", file=sys.stderr, flush=True)

            iteration_seconds = []
            generated_tokens = []
            while not generator.is_done() and len(iteration_seconds) < args.max_new_tokens:
                iteration = len(iteration_seconds)
                print(f"[qmoe_prompt_suite] iteration_start {prompt_index} {iteration}", file=sys.stderr, flush=True)
                start = time.perf_counter()
                generator.generate_next_token()
                iteration_seconds.append(time.perf_counter() - start)
                generated_tokens.extend(generator.get_next_tokens().tolist())
                print(f"[qmoe_prompt_suite] iteration_end {prompt_index} {iteration}", file=sys.stderr, flush=True)
            print(f"[qmoe_prompt_suite] prompt_end {prompt_index}", file=sys.stderr, flush=True)
            if len(iteration_seconds) != args.max_new_tokens or len(generated_tokens) != args.max_new_tokens:
                raise RuntimeError(f"Prompt {prompt_index} did not produce {args.max_new_tokens} tokens.")
            results.append(
                {
                    "prompt_index": prompt_index,
                    "prompt": prompt,
                    "prompt_tokens": len(prompt_tokens),
                    "generated_tokens": len(generated_tokens),
                    "prefill_ms": prefill_seconds * 1000,
                    "first_decode_run_ms": iteration_seconds[1] * 1000,
                    "decode": decode_summary(iteration_seconds),
                    "steady_decode": summarize(iteration_seconds[2:]) if len(iteration_seconds) > 2 else None,
                    "total_generation_seconds": prefill_seconds + sum(iteration_seconds),
                }
            )

    output = {
        "name": args.name,
        "provider": args.provider,
        "model_load_seconds": model_load_seconds,
        "cpu_offload_experts": args.cpu_offload_experts,
        "counter_alpha": args.counter_alpha,
        "swap_epsilon": args.swap_epsilon,
        "swap": args.swap,
        "swaps_applicable": args.swap and args.cpu_offload_experts > 0,
        "counter_log": args.counter_log,
        "cuda_graph": args.cuda_graph,
        "max_new_tokens": args.max_new_tokens,
        "memory_after_model_load": memory_after_model_load,
        "memory": memory_sampler.stop(),
        "prompts": results,
    }
    args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
