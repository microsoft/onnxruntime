# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
"""Linux native-only qualification measurements; explicit local packages, no downloads."""

import argparse
import ctypes
import gc
import importlib
import json
import math
import os
import platform
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

import onnxruntime as ort
from onnxruntime import stateful as sf

if sys.platform == "linux":
    import resource

PHI3_PROMPT = "Answer briefly: what is 2 + 2?"
SD15_PROMPT = "A peaceful mountain lake at sunrise, soft golden light, landscape painting"
WARMUPS = 2
SAMPLES = 5


@dataclass(frozen=True)
class Case:
    model: str
    configuration: dict
    cuda: bool
    synchronize: Callable[[], None]
    create: Callable[[], tuple[sf.StatefulExecution, Callable[[list[int]], str] | None, float]]


class TimedTokenizer:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.encode_seconds = 0.0

    def encode(self, text, **kwargs):
        started = time.perf_counter()
        value = self.tokenizer.encode(text, **kwargs)
        self.encode_seconds += time.perf_counter() - started
        return value

    def decode(self, tokens):
        return self.tokenizer.decode(tokens)


def statistics(samples) -> dict:
    values = np.asarray(samples)
    if values.ndim != 1 or not values.size or values.dtype.kind not in "iuf":
        raise ValueError("Timing samples must be a nonempty numeric vector")
    if not np.isfinite(values).all() or np.any(values <= 0):
        raise ValueError("Completed-work timing samples must be finite and positive")
    ordered = np.sort(values)
    middle = len(ordered) // 2
    median = float(ordered[middle])
    if len(ordered) % 2 == 0:
        lower = float(ordered[middle - 1])
        median = lower + (median - lower) / 2
    return {
        "samples_seconds": values.tolist(),
        "count": int(values.size),
        "median_seconds": median,
        "p95_seconds": float(ordered[math.ceil(0.95 * len(ordered)) - 1]),
        "percentile_method": "nearest_rank",
    }


def synchronizer(cuda: bool) -> Callable[[], None]:
    if not cuda:
        return lambda: None
    runtime = ctypes.CDLL(os.environ.get("ORT_STATEFUL_CUDART", "libcudart.so.13"))
    function = runtime.cudaDeviceSynchronize
    function.argtypes = []
    function.restype = ctypes.c_int

    def synchronize():
        status = function()
        if status != 0:
            raise RuntimeError(f"CUDA completion synchronization failed: {status}")

    return synchronize


def load_case(model: str, package: Path) -> Case:
    if model not in ("phi3", "sd15"):
        raise ValueError("Unknown qualification model")
    module = importlib.import_module(f"{model}_demo")
    config = module.package_configuration(package) if model == "phi3" else module.configuration(package)
    cuda = model == "sd15" or config["target"] == "cuda"
    if config["device_id"] != 0:
        raise ValueError("M24 qualification uses device ordinal 0")
    if model == "phi3":
        if config["max_new_tokens"] not in (0, 1, 16):
            raise ValueError("M24 Phi-3 budgets are 0, 1 and 16")
    elif config["ddim"]["num_inference_steps"] != 20 or config["ddim"]["guidance_scale"] != 7.5:
        raise ValueError("M24 SD1.5 requires 20 updates and guidance 7.5")
    if cuda and "cuda-plugin-ep=1" not in ort.get_build_info():
        raise ValueError("Real CUDA qualification requires the locally built CUDA-plugin runtime")

    def create():
        if model == "phi3":
            tokenizer = TimedTokenizer(module.LocalTokenizer(package / "decoder" / config["target"]))
            execution, _, _ = module.create_execution(package, PHI3_PROMPT, tokenizer)
            return execution, tokenizer.decode, tokenizer.encode_seconds
        tokenizer = TimedTokenizer(module.LocalTokenizer(package / "tokenizer"))
        execution = module.create_execution(package, SD15_PROMPT, seed=42, adapter=tokenizer)
        return execution, None, tokenizer.encode_seconds

    return Case(model, config, cuda, synchronizer(cuda), create)


def gpu_memory() -> dict:
    output = subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=pid,used_gpu_memory", "--format=csv,noheader,nounits"], text=True
    )
    amounts = []
    for line in output.splitlines():
        pid, amount = line.split(",")
        if int(pid) == os.getpid():
            amounts.append(int(amount) * 1024 * 1024)
    return {
        "observed": bool(amounts),
        "used_bytes": sum(amounts) if amounts else None,
        "scope": "current process across CUDA devices; NVML observation, not an allocation-call count",
    }


class _MallInfo(ctypes.Structure):
    _fields_ = tuple(
        (name, ctypes.c_size_t)
        for name in (
            "arena",
            "ordblks",
            "smblks",
            "hblks",
            "hblkhd",
            "usmblks",
            "fsmblks",
            "uordblks",
            "fordblks",
            "keepcost",
        )
    )


def memory_snapshot(cuda: bool) -> dict:
    if sys.platform != "linux":
        raise RuntimeError("Resource qualification requires Linux")
    function = ctypes.CDLL(None).mallinfo2
    function.argtypes = []
    function.restype = _MallInfo
    values = function()
    return {
        "rss_bytes": int(Path("/proc/self/statm").read_text().split()[1]) * os.sysconf("SC_PAGE_SIZE"),
        "process_peak_rss_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024,
        "glibc_in_use_bytes": int(values.uordblks),
        "glibc_mmapped_bytes": int(values.hblkhd),
        "glibc_mmap_regions": int(values.hblks),
        "glibc_scope": "process malloc statistics; not ORT-only; excludes non-malloc mappings",
        "gpu": gpu_memory() if cuda else {"observed": False, "reason": "CPU EP workload"},
    }


def describe(case: Case, execution, decoder, result, stage: Path, label: str) -> dict:
    info = result.get_info()
    if not info.completed:
        raise RuntimeError("Run result did not complete the native chain")
    outputs = result.get_outputs()
    record = {
        "invocations": info.invocations,
        "iterations": info.iterations,
        "generated_tokens": info.generated_tokens,
        "exit_reasons": [value.name for value in info.exit_reasons],
    }
    if case.model == "phi3":
        tokens = outputs["tokens"].numpy()
        if tokens.dtype != np.int64 or tokens.ndim != 1 or np.any(tokens < 0):
            raise ValueError("Native tokens must be a nonnegative INT64 vector")
        if (
            len(tokens) != info.generated_tokens
            or len(tokens) > case.configuration["max_new_tokens"]
            or info.invocations != max(1, len(tokens))
        ):
            raise ValueError("Native Phi-3 invocation/token accounting differs from its policy")
        if decoder is None:
            raise ValueError("Phi-3 output needs its caller decoder")
        dispatch = execution.get_dispatch_info(sf.ProviderFeature.SAMPLER, sf.ProviderOperation.SAMPLE_GREEDY, 1)
        if case.cuda and len(tokens) and (not dispatch.provider_calls or dispatch.cpu_calls):
            raise RuntimeError("CUDA generation did not use the required native sampler")
        record.update(
            units=len(tokens),
            token_ids=tokens.tolist(),
            text=decoder(tokens.tolist()),
            sampler_origin=dispatch.origin.name,
            sampler_provider_calls=dispatch.provider_calls,
            sampler_cpu_calls=dispatch.cpu_calls,
        )
        return record
    policy = case.configuration["ddim"]
    expected = 5 + policy["num_inference_steps"] * (2 if policy["guidance_mode"] == "sequential" else 1)
    if info.invocations != expected:
        raise ValueError("Native SD1.5 model invocation accounting differs from its policy")
    module = importlib.import_module("sd15_demo")
    image = stage / f"{label}.png"
    module.save_image(outputs["image"].numpy(), outputs["unsafe"].numpy(), image)
    record.update(units=1, unsafe=False, image=image.name, image_sha256=module.digest(image))
    return record


def completed_run(case: Case, execution, decoder, stage: Path, label: str) -> dict:
    case.synchronize()
    started = time.perf_counter()
    with execution.run() as result:
        case.synchronize()
        completed = time.perf_counter()
        record = describe(case, execution, decoder, result, stage, label)
        record["output_seconds"] = time.perf_counter() - completed
    record["seconds"] = completed - started
    statistics([record["seconds"]])
    return record


def measure(case: Case, phase: str, stage: Path, prefix: str = "") -> dict:
    if phase not in ("latency", "resources", "steps"):
        raise ValueError("Unknown measurement phase")
    record = {"phase": phase, "model": case.model, "configuration": case.configuration}
    if phase == "resources":
        record["before_setup"] = memory_snapshot(case.cuda)
    started = time.perf_counter()
    execution, decoder, encoding = case.create()
    record["setup_seconds"] = time.perf_counter() - started
    record["tokenization_seconds_within_setup"] = encoding
    with execution:
        if phase == "latency":
            record["cold"] = completed_run(case, execution, decoder, stage, prefix + "cold")
            warmups, samples = [], []
            for name, count, collection in (("warmup", WARMUPS, warmups), ("sample", SAMPLES, samples)):
                for index in range(count):
                    reset = time.perf_counter()
                    execution.reset()
                    reset_seconds = time.perf_counter() - reset
                    value = completed_run(case, execution, decoder, stage, f"{prefix}{name}-{index}")
                    value["reset_seconds"] = reset_seconds
                    collection.append(value)
            record["warmups"] = warmups
            record["measured"] = samples
            record["steady"] = statistics([value["seconds"] for value in samples])
            units = sum(value["units"] for value in samples)
            record["throughput_units"] = "tokens/second" if case.model == "phi3" else "images/second"
            record["throughput"] = units / math.fsum(value["seconds"] for value in samples) if units else None
            record["throughput_applicability"] = "measured" if units else "no generated tokens"
        elif phase == "resources":
            record["after_setup"] = memory_snapshot(case.cuda)
            for index in range(WARMUPS):
                completed_run(case, execution, decoder, stage, f"{prefix}warmup-{index}")
                execution.reset()
            record["after_warmup"] = memory_snapshot(case.cuda)
            if case.cuda and not record["after_warmup"]["gpu"]["observed"]:
                raise RuntimeError("GPU process memory was not observable after completed native warmup")
            cycles = []
            for index in range(2):
                outcome = completed_run(case, execution, decoder, stage, f"{prefix}cycle-{index}")
                after_run = memory_snapshot(case.cuda)
                execution.reset()
                cycles.append({"result": outcome, "after_run": after_run, "after_reset": memory_snapshot(case.cuda)})
            record["cycles"] = cycles
        else:
            bound = (
                case.configuration["max_new_tokens"] + 2
                if case.model == "phi3"
                else case.configuration["ddim"]["num_inference_steps"] + 6
            )
            steps = []
            case.synchronize()
            began = time.perf_counter()
            first_token = None
            for _ in range(bound):
                before = time.perf_counter()
                with execution.step() as result:
                    case.synchronize()
                    now = time.perf_counter()
                    info = result.get_info()
                    steps.append({"component": info.component_name, "seconds": now - before, "executed": info.executed})
                    if first_token is None and info.generated_token is not None:
                        first_token = now - began
                    if info.completed:
                        record["result"] = describe(case, execution, decoder, result, stage, prefix + "steps")
                        break
            else:
                raise RuntimeError("Native stepping exceeded the configured completion bound")
            record["steps"] = steps
            record["step_observed_time_to_first_token_seconds"] = first_token
            record["timing_scope"] = "separate Step pass, including Python boundary and completion synchronization"
    del execution, decoder
    if phase == "resources":
        gc.collect()
        record["after_close"] = memory_snapshot(case.cuda)
        record["peak_scope"] = "process RSS high-water; GPU phase-boundary samples; traced GPU peaks are separate"
    return record


def publish_report(output: Path, record: dict, stage: Path) -> None:
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", prefix=".report-", dir=stage) as stream:
        json.dump(record, stream, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        pending = Path(stream.name)
        created = []
        committed = False
        try:
            for source in [*sorted(stage.glob("*.png")), pending]:
                destination = output if source == pending else output.parent / source.name
                destination.hardlink_to(source)
                created.append(destination)
            committed = True
        finally:
            if not committed:
                for destination in reversed(created):
                    destination.unlink()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("phi3", "sd15"), required=True)
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--phase", choices=("latency", "resources", "steps"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    case = load_case(args.model, args.package)
    with tempfile.TemporaryDirectory(prefix=".qualification-", dir=args.output.parent) as directory:
        stage = Path(directory)
        record = measure(case, args.phase, stage, args.output.stem + "-")
        record["environment"] = {
            "python": sys.version,
            "platform": platform.platform(),
            "libc": platform.libc_ver(),
            "runtime": ort.__file__,
            "build": ort.get_build_info(),
            "pid": os.getpid(),
            "measurement": "native execution through Python; output handling excluded from native timing",
        }
        publish_report(args.output, record, stage)
    print(args.output)


if __name__ == "__main__":
    main()
