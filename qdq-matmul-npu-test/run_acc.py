#!/usr/bin/env python3
"""Compare CPU and Windows ML NPU outputs for the same ONNX inputs."""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

import numpy as np

from gemma_synthetic_data import (
    DEFAULT_PADDING_FRACTION,
    HIDDEN_STATE_CLIP,
    HIDDEN_STATE_STD,
)
from run_winml_ep import (
    NPU_PROVIDERS,
    PROVIDER_NAMES,
    find_npu_device,
    make_inputs,
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


def create_cpu_session(ort: Any, model_path: Path, log_level: int) -> Any:
    options = ort.SessionOptions()
    options.log_severity_level = log_level
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
) -> tuple[list[np.ndarray], float]:
    for _ in range(warmup_iterations):
        session.run(None, inputs)

    start = time.perf_counter_ns()
    outputs = session.run(None, inputs)
    latency_ms = (time.perf_counter_ns() - start) / 1_000_000.0
    return outputs, latency_ms


def format_number(value: float) -> str:
    return f"{value:.6g}"


def cosine_similarity(reference: np.ndarray, actual: np.ndarray) -> float:
    reference_flat = reference.ravel()
    actual_flat = actual.ravel()
    denominator = np.linalg.norm(reference_flat) * np.linalg.norm(actual_flat)
    if denominator == 0:
        return 1.0 if np.array_equal(reference_flat, actual_flat) else 0.0
    return float(np.dot(reference_flat, actual_flat) / denominator)


def compare_float_output(
    reference: np.ndarray,
    actual: np.ndarray,
    atol: float,
    rtol: float,
) -> list[str]:
    reference_float = reference.astype(np.float64, copy=False)
    actual_float = actual.astype(np.float64, copy=False)

    if reference_float.size == 0:
        return ["0", "0", "0", "0", "1", "100.000%"]

    difference = actual_float - reference_float
    absolute_difference = np.abs(difference)
    max_absolute_error = float(np.max(absolute_difference))
    mean_absolute_error = float(np.mean(absolute_difference))
    rmse = float(np.sqrt(np.mean(np.square(difference))))
    reference_norm = float(np.linalg.norm(reference_float.ravel()))
    difference_norm = float(np.linalg.norm(difference.ravel()))
    relative_l2_error = (
        difference_norm / reference_norm
        if reference_norm != 0
        else (0.0 if difference_norm == 0 else float("inf"))
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

    return [
        format_number(max_absolute_error),
        format_number(mean_absolute_error),
        format_number(rmse),
        format_number(relative_l2_error),
        format_number(cosine),
        f"{within_tolerance:.3f}%",
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
) -> None:
    if len(cpu_outputs) != len(npu_outputs):
        raise RuntimeError(
            f"CPU returned {len(cpu_outputs)} outputs, but NPU returned "
            f"{len(npu_outputs)}"
        )

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

        if np.issubdtype(reference.dtype, np.inexact):
            metrics = compare_float_output(reference, actual, atol, rtol)
        else:
            metrics = compare_exact_output(reference, actual)

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


def main() -> None:
    args = parse_args()
    validate_args(args)
    provider_name = PROVIDER_NAMES[args.provider]
    provider_options = parse_provider_options(args.provider_option)
    model_path = args.model.resolve()

    ort = register_provider(provider_name)
    cpu_session = create_cpu_session(ort, model_path, args.log_severity_level)
    inputs = make_inputs(
        cpu_session,
        args.seed,
        args.hidden_state_std,
        args.hidden_state_clip,
        args.padding_fraction,
    )
    npu_session = create_npu_session(
        ort,
        model_path,
        args.log_severity_level,
        provider_options,
        not args.no_cpu_fallback,
        provider_name,
    )

    cpu_outputs, cpu_latency_ms = run_session(
        cpu_session,
        inputs,
        args.warmup_iterations,
    )
    npu_outputs, npu_latency_ms = run_session(
        npu_session,
        inputs,
        args.warmup_iterations,
    )

    print(f"Model:              {model_path}")
    print(f"Input seed:         {args.seed}")
    print(f"CPU provider:       CPUExecutionProvider")
    print(f"NPU provider:       {provider_name}")
    print(f"NPU options:        {provider_options or '{}'}")
    print(f"CPU fallback:       {not args.no_cpu_fallback}")
    print(
        "Synthetic inputs:   "
        f"std={args.hidden_state_std:g}, clip={args.hidden_state_clip:g}, "
        f"padding={args.padding_fraction:.1%}"
    )
    print(f"Tolerance:          atol={args.atol:g}, rtol={args.rtol:g}")
    print(f"CPU latency:        {cpu_latency_ms:.3f} ms")
    print(f"NPU latency:        {npu_latency_ms:.3f} ms")

    output_names = [output.name for output in cpu_session.get_outputs()]
    compare_outputs(
        output_names,
        cpu_outputs,
        npu_outputs,
        args.atol,
        args.rtol,
    )


if __name__ == "__main__":
    main()
