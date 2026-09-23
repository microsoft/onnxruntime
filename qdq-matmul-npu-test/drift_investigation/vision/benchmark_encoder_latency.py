"""Benchmark matched vision encoders on ORT CPU and a precompiled NPU model."""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import time
from typing import Any

import numpy as np
import run_acc
import run_winml_ep


def timed_runs(session: Any, inputs: dict[str, np.ndarray], warmups: int, iterations: int):
    for _ in range(warmups):
        session.run(None, inputs)
    durations = []
    outputs = []
    for _ in range(iterations):
        start = time.perf_counter_ns()
        outputs = session.run(None, inputs)
        durations.append((time.perf_counter_ns() - start) / 1_000_000)
    return outputs, durations


def summarize(durations: list[float]) -> dict[str, Any]:
    return {
        "samples_ms": durations,
        "median_ms": float(np.median(durations)),
        "mean_ms": float(np.mean(durations)),
        "p10_ms": float(np.percentile(durations, 10)),
        "p90_ms": float(np.percentile(durations, 90)),
    }


def _parse_named_paths(entries: list[list[str]], kind: str) -> dict[str, tuple[Path, Path]]:
    result: dict[str, tuple[Path, Path]] = {}
    for name, first, second in entries:
        if name in result:
            raise ValueError(f"repeated {kind} name: {name}")
        result[name] = (Path(first).resolve(), Path(second).resolve())
    if not result:
        raise ValueError(f"at least one --{kind} is required")
    return result


def _parse_masks(entries: list[str], input_names: set[str]) -> dict[str, str]:
    masks: dict[str, str] = {}
    for entry in entries:
        name, separator, mask = entry.partition("=")
        if not separator or name not in input_names or not mask or name in masks:
            raise ValueError(f"invalid or repeated --valid-mask value {entry!r}")
        masks[name] = mask
    return masks


def benchmark_case(
    ort: Any,
    device: Any,
    model: Path,
    compiled: Path,
    input_path: Path,
    valid_mask_name: str | None,
    warmups: int,
    iterations: int,
    provider_options: dict[str, str] | None = None,
) -> dict[str, Any]:
    cpu_options = ort.SessionOptions()
    cpu_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    cpu = ort.InferenceSession(str(model), sess_options=cpu_options, providers=["CPUExecutionProvider"])
    inputs, valid = run_acc.load_inputs(cpu, input_path, valid_mask_name)

    ep_options = ort.SessionOptions()
    ep_options.add_provider_for_devices([device], provider_options or {})
    ep_options.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
    ep = ort.InferenceSession(str(compiled), sess_options=ep_options)
    if [item.name for item in cpu.get_inputs()] != [item.name for item in ep.get_inputs()]:
        raise RuntimeError(f"{model.name}: CPU and EP input names differ")
    if [item.name for item in cpu.get_outputs()] != [item.name for item in ep.get_outputs()]:
        raise RuntimeError(f"{model.name}: CPU and EP output names differ")

    cpu_outputs, cpu_durations = timed_runs(cpu, inputs, warmups, iterations)
    ep_outputs, ep_durations = timed_runs(ep, inputs, warmups, iterations)
    if len(cpu_outputs) != 1 or len(ep_outputs) != 1:
        raise RuntimeError(f"{model.name}: expected one encoder output")
    cpu_out, ep_out = cpu_outputs[0], ep_outputs[0]
    if cpu_out.shape != ep_out.shape or cpu_out.dtype != ep_out.dtype:
        raise RuntimeError(f"{model.name}: CPU/EP output shape or dtype differs")
    if valid is not None:
        if valid.shape != cpu_out.shape[: valid.ndim]:
            raise RuntimeError(f"{model.name}: valid mask shape differs from encoder output")
        cpu_out, ep_out = cpu_out[valid], ep_out[valid]
    metrics = run_acc.measure_float_output(cpu_out, ep_out, 0, 0)
    cpu_stats, ep_stats = summarize(cpu_durations), summarize(ep_durations)
    result = {
        "model": str(model),
        "model_sha256": run_acc.file_sha256(model),
        "compiled_model": str(compiled),
        "ep_context_nodes": run_acc.context_node_count(compiled),
        "cpu": cpu_stats,
        "npu": ep_stats,
        "median_speedup_cpu_over_npu": cpu_stats["median_ms"] / ep_stats["median_ms"],
        "cpu_npu_mean_abs_error": metrics["mean_abs_error"],
        "cpu_npu_cosine": metrics["cosine_similarity"],
    }
    del cpu, ep
    gc.collect()
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--variant",
        action="append",
        nargs=3,
        metavar=("NAME", "CPU_MODEL", "COMPILED_MODEL"),
        required=True,
        help="Named source/precompiled model pair (repeatable).",
    )
    parser.add_argument(
        "--input-set",
        action="append",
        nargs=2,
        metavar=("NAME", "NPZ"),
        required=True,
        help="Named input NPZ archive (repeatable).",
    )
    parser.add_argument(
        "--valid-mask",
        action="append",
        default=[],
        metavar="INPUT_SET=ARRAY_NAME",
        help="NPZ boolean mask used only for accuracy metrics (repeatable).",
    )
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--iterations", type=int, default=7)
    parser.add_argument("--provider", choices=run_winml_ep.NPU_PROVIDERS, default="openvino")
    parser.add_argument("--provider-option", action="append", default=[], metavar="KEY=VALUE")
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.warmups < 1 or args.iterations < 3:
        raise ValueError("require at least one warmup and three measured runs")

    variants = _parse_named_paths(args.variant, "variant")
    input_pairs = _parse_named_paths([[name, path, path] for name, path in args.input_set], "input-set")
    inputs = {name: paths[0] for name, paths in input_pairs.items()}
    masks = _parse_masks(args.valid_mask, set(inputs))
    provider_options = run_winml_ep.parse_provider_options(args.provider_option)

    for model, compiled in variants.values():
        if not model.is_file() or not compiled.is_file() or run_acc.context_node_count(compiled) != 1:
            raise RuntimeError(f"model or single-EPContext artifact missing: {model}, {compiled}")
    for input_path in inputs.values():
        if not input_path.is_file():
            raise RuntimeError(f"input archive missing: {input_path}")
    config = {
        "warmups": args.warmups,
        "iterations": args.iterations,
        "provider": run_winml_ep.PROVIDER_NAMES[args.provider],
        "provider_options": provider_options,
        "cpu_fallback": False,
        "cpu_optimization": "ORT_ENABLE_ALL",
        "models": {
            name: {"source": run_acc.file_sha256(paths[0]), "compiled": run_acc.file_sha256(paths[1])}
            for name, paths in variants.items()
        },
        "inputs": {name: run_acc.file_sha256(path) for name, path in inputs.items()},
        "valid_masks": masks,
    }
    report_path = args.report.resolve()
    if report_path.exists():
        if not args.resume:
            raise ValueError(f"report exists; use --resume or another output path: {report_path}")
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if report["config"] != config:
            raise ValueError("existing report configuration differs")
    else:
        if args.resume:
            raise ValueError(f"cannot resume a nonexistent report: {report_path}")
        report = {"config": config, "results": {}}
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    provider_name = run_winml_ep.PROVIDER_NAMES[args.provider]
    ort = run_winml_ep.register_provider(provider_name)
    device = run_winml_ep.find_npu_device(ort, provider_name)
    for input_name, input_path in inputs.items():
        report["results"].setdefault(input_name, {})
        for variant_name, (model, compiled) in variants.items():
            if variant_name in report["results"][input_name]:
                print(f"Skipping completed {input_name}/{variant_name}", flush=True)
                continue
            print(f"Benchmarking {input_name}/{variant_name}", flush=True)
            result = benchmark_case(
                ort,
                device,
                model,
                compiled,
                input_path,
                masks.get(input_name),
                args.warmups,
                args.iterations,
                provider_options,
            )
            report["results"][input_name][variant_name] = result
            report_path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
            print(
                f"  CPU {result['cpu']['median_ms']:.1f} ms, NPU {result['npu']['median_ms']:.1f} ms, "
                f"speedup {result['median_speedup_cpu_over_npu']:.2f}x, "
                f"cosine {result['cpu_npu_cosine']:.6f}",
                flush=True,
            )
    print(f"Saved encoder latency comparison: {report_path}")


if __name__ == "__main__":
    main()
