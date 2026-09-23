#!/usr/bin/env python3
"""Compare CPU and Windows ML NPU outputs for the same ONNX inputs."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any, TypedDict

import numpy as np
import onnx

from compile_winml_ep_model import COMPILE_PROVIDERS, compile_model
from gemma_synthetic_data import (
    DEFAULT_PADDING_FRACTION,
    HIDDEN_STATE_CLIP,
    HIDDEN_STATE_STD,
)
from run_winml_ep import (
    NPU_PROVIDERS,
    PROVIDER_NAMES,
    find_npu_device,
    registered_provider_library,
    make_inputs,
    numpy_dtype,
    parse_provider_options,
    register_provider,
)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path, help="Path to a fixed-shape ONNX model.")
    parser.add_argument(
        "--provider",
        choices=NPU_PROVIDERS,
        default="vitisai",
        help="Windows ML NPU provider; default: vitisai.",
    )
    parser.add_argument("--seed", type=int, default=0, help="Input generation seed.")
    parser.add_argument(
        "--cpu-optimization",
        choices=("disable", "basic", "all"),
        default="all",
        help="CPU graph optimizations; use basic to avoid QLinearMatMul fusion in QDQ studies.",
    )
    parser.add_argument("--inputs", type=Path, help="Use model inputs from an .npz archive instead of random inputs.")
    parser.add_argument(
        "--valid-mask", metavar="NAME",
        help="Boolean array in --inputs, matching the leading dimensions of each output.",
    )
    parser.add_argument("--report-json", type=Path, help="Write numeric comparison metrics for repeatable experiments.")
    parser.add_argument("--output-arrays", type=Path, help="Save CPU/EP outputs for paired ablation-delta comparisons.")
    compiled = parser.add_mutually_exclusive_group()
    compiled.add_argument("--compile-output", type=Path, help="Compile to a new EPContext model and run it.")
    compiled.add_argument("--compiled-model", type=Path, help="Run a previously compiled EPContext model.")
    parser.add_argument(
        "--hidden-state-std",
        type=float,
        default=HIDDEN_STATE_STD,
        help="Standard deviation for bounded synthetic floating-point inputs.",
    )
    parser.add_argument(
        "--hidden-state-clip",
        type=float,
        default=HIDDEN_STATE_CLIP,
        help="Absolute bound for synthetic floating-point inputs.",
    )
    parser.add_argument(
        "--padding-fraction",
        type=float,
        default=DEFAULT_PADDING_FRACTION,
        help="Trailing fraction masked as padded for an attention_mask input.",
    )
    parser.add_argument(
        "--warmup-iterations",
        type=int,
        default=1,
        help="Warmup runs for each provider before the compared run.",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=1,
        help="Measured runs per provider; reports mean and median latency.",
    )
    parser.add_argument(
        "--atol",
        type=float,
        default=1e-3,
        help="Absolute tolerance used for the elementwise accuracy decision.",
    )
    parser.add_argument(
        "--rtol",
        type=float,
        default=1e-2,
        help="Relative tolerance used for the elementwise accuracy decision.",
    )
    parser.add_argument(
        "--no-cpu-fallback",
        action="store_true",
        help="Require every graph node to run without CPU fallback.",
    )
    parser.add_argument(
        "--provider-option",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Provider-specific option; may be supplied multiple times.",
    )
    parser.add_argument(
        "--log-severity-level",
        type=int,
        choices=range(5),
        default=2,
        help="ORT logging level: 0 verbose through 4 fatal.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if not args.model.is_file():
        raise ValueError(f"model does not exist: {args.model}")
    if args.warmup_iterations < 0:
        raise ValueError("--warmup-iterations cannot be negative")
    if args.iterations <= 0:
        raise ValueError("--iterations must be greater than zero")
    if not np.isfinite(args.atol) or args.atol < 0:
        raise ValueError("--atol must be a finite non-negative number")
    if not np.isfinite(args.rtol) or args.rtol < 0:
        raise ValueError("--rtol must be a finite non-negative number")
    if not np.isfinite(args.hidden_state_std) or args.hidden_state_std <= 0:
        raise ValueError("--hidden-state-std must be positive and finite")
    if not np.isfinite(args.hidden_state_clip) or args.hidden_state_clip <= 0:
        raise ValueError("--hidden-state-clip must be positive and finite")
    if not 0.0 <= args.padding_fraction < 1.0:
        raise ValueError("--padding-fraction must be in the range [0, 1)")
    if args.valid_mask and not args.inputs:
        raise ValueError("--valid-mask requires --inputs")
    if (args.compiled_model or args.compile_output) and not args.no_cpu_fallback:
        raise ValueError("--no-cpu-fallback is required when using an EPContext model")
    if args.report_json is not None and args.report_json.exists():
        raise ValueError(f"report already exists: {args.report_json}")
    if args.output_arrays is not None and args.output_arrays.exists():
        raise ValueError(f"output arrays already exist: {args.output_arrays}")
    if args.output_arrays is not None and args.report_json is None:
        raise ValueError("--output-arrays requires --report-json")
    if args.output_arrays is not None and args.output_arrays.resolve() == args.report_json.resolve():
        raise ValueError("--output-arrays and --report-json must be different paths")
    if args.compile_output is not None and args.provider not in COMPILE_PROVIDERS:
        raise ValueError(f"--compile-output supports only {COMPILE_PROVIDERS}")
    if args.compiled_model is not None and not args.compiled_model.is_file():
        raise ValueError(f"compiled model does not exist: {args.compiled_model}")
    if args.compile_output is not None and args.compile_output.exists():
        raise ValueError(f"compiled output already exists: {args.compile_output}")


def create_cpu_session(ort: Any, model_path: Path, log_level: int, optimization: str = "all") -> Any:
    options = ort.SessionOptions()
    options.log_severity_level = log_level
    levels = {
        "disable": ort.GraphOptimizationLevel.ORT_DISABLE_ALL,
        "basic": ort.GraphOptimizationLevel.ORT_ENABLE_BASIC,
        "all": ort.GraphOptimizationLevel.ORT_ENABLE_ALL,
    }
    options.graph_optimization_level = levels[optimization]
    return ort.InferenceSession(
        str(model_path),
        sess_options=options,
        providers=["CPUExecutionProvider"],
    )


def create_npu_session(
    ort: Any,
    model_path: Path,
    log_level: int,
    provider_options: dict[str, str],
    allow_cpu_fallback: bool,
    provider_name: str,
) -> Any:
    options = ort.SessionOptions()
    options.log_severity_level = log_level
    options.add_provider_for_devices(
        [find_npu_device(ort, provider_name)],
        provider_options,
    )
    if not allow_cpu_fallback:
        options.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
    return ort.InferenceSession(str(model_path), sess_options=options)


def run_session(
    session: Any,
    inputs: dict[str, np.ndarray],
    warmup_iterations: int,
    iterations: int = 1,
) -> tuple[list[np.ndarray], list[float]]:
    for _ in range(warmup_iterations):
        session.run(None, inputs)

    latencies = []
    outputs = []
    for _ in range(iterations):
        start = time.perf_counter_ns()
        outputs = session.run(None, inputs)
        latencies.append((time.perf_counter_ns() - start) / 1_000_000.0)
    return outputs, latencies


def load_inputs(session: Any, path: Path, mask_name: str | None) -> tuple[dict[str, np.ndarray], np.ndarray | None]:
    if not path.is_file():
        raise ValueError(f"inputs archive does not exist: {path}")
    with np.load(path, allow_pickle=False) as archive:
        expected = {value.name for value in session.get_inputs()}
        available = set(archive.files)
        if available != expected | ({mask_name} if mask_name else set()):
            required = sorted(expected | ({mask_name} if mask_name else set()))
            raise ValueError(f"input archive keys must be {required}; got {sorted(available)}")
        inputs = {name: archive[name] for name in expected}
        mask = archive[mask_name] if mask_name else None

    for model_input in session.get_inputs():
        value = inputs[model_input.name]
        expected_dtype = numpy_dtype(model_input.type)
        if value.dtype != expected_dtype or len(value.shape) != len(model_input.shape) or any(
            isinstance(dim, int) and dim != actual for dim, actual in zip(model_input.shape, value.shape)
        ):
            raise ValueError(
                f"{model_input.name}: expected {model_input.shape} {expected_dtype}, got {value.shape} {value.dtype}"
            )
    if mask is not None and mask.dtype != np.dtype(np.bool_):
        raise ValueError(f"{mask_name}: valid mask must be bool, got {mask.dtype}")
    return inputs, mask


def context_node_count(path: Path) -> int:
    model = onnx.load(path, load_external_data=False)
    count = sum(node.op_type == "EPContext" for node in model.graph.node)
    if count == 0:
        raise RuntimeError(f"compiled model has no EPContext nodes: {path}")
    other_ops = sorted({node.op_type for node in model.graph.node if node.op_type != "EPContext"})
    if other_ops:
        raise RuntimeError(f"compiled model still has non-EPContext nodes {other_ops}: {path}")
    return count


def validate_compiled_source(source: Path, compiled: Path) -> None:
    source_proto = onnx.load(source, load_external_data=False)
    compiled_proto = onnx.load(compiled, load_external_data=False)
    source_metadata = {item.key: item.value for item in source_proto.metadata_props}
    compiled_metadata = {item.key: item.value for item in compiled_proto.metadata_props}
    for key in (
        "qdq_ablation_pairs_removed",
        "qdq_ablation_selection_sha256",
        "gemma_vision_padding_mask",
    ):
        if key in source_metadata or key in compiled_metadata:
            if source_metadata.get(key) != compiled_metadata.get(key):
                raise ValueError(f"compiled model metadata {key!r} does not match the source model")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def model_external_data_sha256(path: Path) -> dict[str, str]:
    proto = onnx.load(path, load_external_data=False)
    locations = {
        entry.value
        for initializer in proto.graph.initializer
        for entry in initializer.external_data
        if entry.key == "location"
    }
    return {location: file_sha256(path.parent / location) for location in sorted(locations)}


def device_identity(device: Any) -> dict[str, str | int]:
    hardware = device.device
    return {
        "ep_vendor": str(device.ep_vendor),
        "hardware_vendor": str(hardware.vendor),
        "vendor_id": int(hardware.vendor_id),
        "device_id": int(hardware.device_id),
        "device_type": str(hardware.type),
    }


def format_number(value: float | None) -> str:
    return "undefined" if value is None else f"{value:.6g}"


def cosine_similarity(reference: np.ndarray, actual: np.ndarray) -> float:
    reference_flat = reference.ravel()
    actual_flat = actual.ravel()
    denominator = np.linalg.norm(reference_flat) * np.linalg.norm(actual_flat)
    if denominator == 0:
        return 1.0 if np.array_equal(reference_flat, actual_flat) else 0.0
    return float(np.dot(reference_flat, actual_flat) / denominator)


class FloatMetrics(TypedDict):
    max_abs_error: float
    mean_abs_error: float
    rmse: float
    relative_l2_error: float | None
    cosine_similarity: float
    within_tolerance_percent: float


def measure_float_output(
    reference: np.ndarray,
    actual: np.ndarray,
    atol: float,
    rtol: float,
) -> FloatMetrics:
    reference_float = reference.astype(np.float64, copy=False)
    actual_float = actual.astype(np.float64, copy=False)
    if not np.isfinite(reference_float).all() or not np.isfinite(actual_float).all():
        raise RuntimeError("CPU or EP output contains NaN or infinity")

    if reference_float.size == 0:
        return {
            "max_abs_error": 0.0,
            "mean_abs_error": 0.0,
            "rmse": 0.0,
            "relative_l2_error": 0.0,
            "cosine_similarity": 1.0,
            "within_tolerance_percent": 100.0,
        }

    difference = actual_float - reference_float
    absolute_difference = np.abs(difference)
    max_absolute_error = float(np.max(absolute_difference))
    mean_absolute_error = float(np.mean(absolute_difference))
    rmse = float(np.sqrt(np.mean(np.square(difference))))
    reference_norm = float(np.linalg.norm(reference_float.ravel()))
    difference_norm = float(np.linalg.norm(difference.ravel()))
    relative_l2_error = difference_norm / reference_norm if reference_norm != 0 else (
        0.0 if difference_norm == 0 else None
    )
    cosine = cosine_similarity(reference_float, actual_float)
    close = np.isclose(
        reference_float,
        actual_float,
        rtol=rtol,
        atol=atol,
        equal_nan=False,
    )
    within_tolerance = float(np.count_nonzero(close)) / close.size * 100.0

    return {
        "max_abs_error": max_absolute_error,
        "mean_abs_error": mean_absolute_error,
        "rmse": rmse,
        "relative_l2_error": relative_l2_error,
        "cosine_similarity": cosine,
        "within_tolerance_percent": within_tolerance,
    }


def compare_float_output(
    reference: np.ndarray,
    actual: np.ndarray,
    atol: float,
    rtol: float,
) -> list[str]:
    metrics = measure_float_output(reference, actual, atol, rtol)
    return [
        format_number(metrics["max_abs_error"]),
        format_number(metrics["mean_abs_error"]),
        format_number(metrics["rmse"]),
        format_number(metrics["relative_l2_error"]),
        format_number(metrics["cosine_similarity"]),
        f"{metrics['within_tolerance_percent']:.3f}%",
    ]


def compare_exact_output(
    reference: np.ndarray,
    actual: np.ndarray,
) -> list[str]:
    equal = reference == actual
    matching = (
        float(np.count_nonzero(equal)) / equal.size * 100.0
        if equal.size
        else 100.0
    )
    return [
        "-",
        "-",
        "-",
        "-",
        "-",
        f"{matching:.3f}%",
    ]


def compare_outputs(
    output_names: list[str],
    cpu_outputs: list[np.ndarray],
    npu_outputs: list[np.ndarray],
    atol: float,
    rtol: float,
    valid_mask: np.ndarray | None = None,
) -> dict[str, dict[str, float | None]]:
    if len(cpu_outputs) != len(npu_outputs):
        raise RuntimeError(
            f"CPU returned {len(cpu_outputs)} outputs, but NPU returned "
            f"{len(npu_outputs)}"
        )

    report: dict[str, dict[str, float | None]] = {}
    for name, reference, actual in zip(output_names, cpu_outputs, npu_outputs):
        if reference.shape != actual.shape:
            raise RuntimeError(
                f"output {name!r} shape differs: CPU {reference.shape}, "
                f"NPU {actual.shape}"
            )
        if reference.dtype != actual.dtype:
            raise RuntimeError(
                f"output {name!r} dtype differs: CPU {reference.dtype}, "
                f"NPU {actual.dtype}"
            )

        floating = np.issubdtype(reference.dtype, np.inexact)
        values: dict[str, float | None]
        if floating:
            measured = measure_float_output(reference, actual, atol, rtol)
            values = dict(measured)
            metrics = [
                format_number(measured["max_abs_error"]),
                format_number(measured["mean_abs_error"]),
                format_number(measured["rmse"]),
                format_number(measured["relative_l2_error"]),
                format_number(measured["cosine_similarity"]),
                f"{measured['within_tolerance_percent']:.3f}%",
            ]
        else:
            metrics = compare_exact_output(reference, actual)
            values = {"matching_percent": float(np.mean(reference == actual) * 100) if reference.size else 100.0}

        valid_values = None
        if valid_mask is not None:
            if reference.shape[: valid_mask.ndim] != valid_mask.shape:
                raise ValueError(
                    f"valid mask shape {valid_mask.shape} does not match output {name!r}: {reference.shape}"
                )
            if not np.any(valid_mask):
                raise ValueError("valid mask selects no output elements")
            if floating:
                valid_values = measure_float_output(reference[valid_mask], actual[valid_mask], atol, rtol)
                values.update({
                    "valid_mean_abs_error": valid_values["mean_abs_error"],
                    "valid_cosine_similarity": valid_values["cosine_similarity"],
                    "valid_relative_l2_error": valid_values["relative_l2_error"],
                })
        report[name] = values

        shape = "x".join(str(dimension) for dimension in reference.shape)
        print()
        print(f"Output: {name}")
        print(f"- Shape: {shape}")
        print(f"- Dtype: {reference.dtype}")
        print(f"- Max abs error: {metrics[0]}")
        print(f"- Mean abs error: {metrics[1]}")
        print(f"- RMSE: {metrics[2]}")
        print(f"- Relative L2: {metrics[3]}")
        print(f"- Cosine similarity: {metrics[4]}")
        print(f"- Within tolerance: {metrics[5]}")
        if valid_values is not None:
            print(f"- Valid-only MAE: {format_number(valid_values['mean_abs_error'])}")
            print(f"- Valid-only cosine: {format_number(valid_values['cosine_similarity'])}")
    return report


def main() -> None:
    args = parse_args()
    validate_args(args)
    provider_name = PROVIDER_NAMES[args.provider]
    provider_options = parse_provider_options(args.provider_option)
    model_path = args.model.resolve()

    ort = register_provider(provider_name)
    cpu_session = create_cpu_session(ort, model_path, args.log_severity_level, args.cpu_optimization)
    if args.inputs is not None:
        inputs, valid_mask = load_inputs(cpu_session, args.inputs, args.valid_mask)
    else:
        inputs = make_inputs(cpu_session, args.seed, args.hidden_state_std, args.hidden_state_clip,
                             args.padding_fraction)
        valid_mask = None
    npu_model = model_path
    ep_context_nodes = None
    if args.compile_output is not None:
        npu_model = args.compile_output.resolve()
        npu_model.parent.mkdir(parents=True, exist_ok=True)
        compile_model(ort, model_path, npu_model, provider_name, provider_options, False, True)
        ep_context_nodes = context_node_count(npu_model)
    elif args.compiled_model is not None:
        npu_model = args.compiled_model.resolve()
        ep_context_nodes = context_node_count(npu_model)
        validate_compiled_source(model_path, npu_model)
    npu_session = create_npu_session(
        ort,
        npu_model,
        args.log_severity_level,
        provider_options,
        not args.no_cpu_fallback,
        provider_name,
    )

    cpu_outputs, cpu_latencies = run_session(
        cpu_session,
        inputs,
        args.warmup_iterations,
        args.iterations,
    )
    npu_outputs, npu_latencies = run_session(
        npu_session,
        inputs,
        args.warmup_iterations,
        args.iterations,
    )
    cpu_latency_ms = float(np.mean(cpu_latencies))
    npu_latency_ms = float(np.mean(npu_latencies))

    print(f"Model:              {model_path}")
    if args.inputs is not None:
        print(f"Input archive:      {args.inputs.resolve()}")
    else:
        print(f"Input seed:         {args.seed}")
    print(f"CPU provider:       CPUExecutionProvider")
    print(f"CPU optimization:   {args.cpu_optimization}")
    print(f"NPU provider:       {provider_name}")
    if ep_context_nodes is not None:
        print(f"Compiled EPContext nodes: {ep_context_nodes} ({npu_model})")
    print(f"NPU options:        {provider_options or '{}'}")
    print(f"CPU fallback:       {not args.no_cpu_fallback}")
    if args.inputs is None:
        print(
            "Synthetic inputs:   "
            f"std={args.hidden_state_std:g}, clip={args.hidden_state_clip:g}, "
            f"padding={args.padding_fraction:.1%}"
        )
    print(f"Tolerance:          atol={args.atol:g}, rtol={args.rtol:g}")
    print(f"CPU latency:        {cpu_latency_ms:.3f} ms mean, {np.median(cpu_latencies):.3f} ms median")
    print(f"NPU latency:        {npu_latency_ms:.3f} ms mean, {np.median(npu_latencies):.3f} ms median")
    print(f"Measured runs:      {args.iterations} (after {args.warmup_iterations} warmups per provider)")

    output_names = [output.name for output in cpu_session.get_outputs()]
    npu_output_names = [output.name for output in npu_session.get_outputs()]
    if output_names != npu_output_names:
        raise RuntimeError(f"CPU output names {output_names} do not match EP output names {npu_output_names}")
    results = compare_outputs(
        output_names,
        cpu_outputs,
        npu_outputs,
        args.atol,
        args.rtol,
        valid_mask,
    )
    if args.report_json is not None:
        provider_library = registered_provider_library(provider_name)
        device = device_identity(find_npu_device(ort, provider_name))
        source_proto = onnx.load(model_path, load_external_data=False)
        source_metadata = {entry.key: entry.value for entry in source_proto.metadata_props}
        model_digest = file_sha256(model_path)
        external_digests = model_external_data_sha256(model_path)
        input_digest = file_sha256(args.inputs) if args.inputs else None
        if args.output_arrays is not None:
            args.output_arrays.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                args.output_arrays,
                **{f"cpu_{index}": array for index, array in enumerate(cpu_outputs)},
                **{f"ep_{index}": array for index, array in enumerate(npu_outputs)},
            )
        args.report_json.parent.mkdir(parents=True, exist_ok=True)
        args.report_json.write_text(
            json.dumps(
                {
                    "model": str(model_path),
                    "model_sha256": model_digest,
                    "external_data_sha256": external_digests,
                    "ablation_pairs_removed": int(source_metadata.get("qdq_ablation_pairs_removed", "0")),
                    "inputs": str(args.inputs.resolve()) if args.inputs else None,
                    "inputs_sha256": input_digest,
                    "valid_mask": args.valid_mask,
                    "output_arrays": str(args.output_arrays.resolve()) if args.output_arrays else None,
                    "output_names": output_names,
                    "provider": provider_name,
                    "ort_version": ort.__version__,
                    "provider_library": str(provider_library.resolve()),
                    "provider_library_sha256": file_sha256(provider_library),
                    "device": device,
                    "cpu_optimization": args.cpu_optimization,
                    "provider_options": provider_options,
                    "cpu_fallback": not args.no_cpu_fallback,
                    "compiled_model": str(npu_model) if ep_context_nodes is not None else None,
                    "ep_context_nodes": ep_context_nodes,
                    "cpu_latency_ms": cpu_latency_ms,
                    "npu_latency_ms": npu_latency_ms,
                    "cpu_latency_median_ms": float(np.median(cpu_latencies)),
                    "npu_latency_median_ms": float(np.median(npu_latencies)),
                    "cpu_latency_p90_ms": float(np.percentile(cpu_latencies, 90)),
                    "npu_latency_p90_ms": float(np.percentile(npu_latencies, 90)),
                    "iterations": args.iterations,
                    "outputs": results,
                },
                indent=2,
                allow_nan=False,
            ) + "\n",
            encoding="utf-8",
        )
        print(f"Report:             {args.report_json.resolve()}")


if __name__ == "__main__":
    main()
