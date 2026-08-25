# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

"""Benchmark a static ONNX model with WebGPU graph capture and fixed device I/O."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import numpy as np

import onnxruntime as ort

ORT_TYPE_TO_NUMPY = {
    "tensor(bool)": np.bool_,
    "tensor(double)": np.float64,
    "tensor(float)": np.float32,
    "tensor(float16)": np.float16,
    "tensor(int8)": np.int8,
    "tensor(int16)": np.int16,
    "tensor(int32)": np.int32,
    "tensor(int64)": np.int64,
    "tensor(uint8)": np.uint8,
    "tensor(uint16)": np.uint16,
    "tensor(uint32)": np.uint32,
    "tensor(uint64)": np.uint64,
}


def parse_input_spec(value: str) -> tuple[str, Path]:
    name, separator, path = value.partition("=")
    if not separator or not name or not path:
        raise argparse.ArgumentTypeError("input must use NAME=FILE.npy")
    return name, Path(path)


def static_shape(name: str, shape: list[int | str | None]) -> list[int]:
    if not all(isinstance(dimension, int) for dimension in shape):
        raise ValueError(f"Graph capture requires a static shape for {name}: {shape}")
    return [int(dimension) for dimension in shape]


def create_session(arguments: argparse.Namespace) -> ort.InferenceSession:
    options = ort.SessionOptions()
    options.enable_mem_pattern = False
    options.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
    options.add_session_config_entry(
        "ep.webgpuexecutionprovider.enableGraphCapture",
        "1" if arguments.graph_capture else "0",
    )
    options.add_session_config_entry(
        "ep.webgpuexecutionprovider.maxNumPendingDispatches",
        str(arguments.max_pending_dispatches),
    )
    if arguments.backend:
        options.add_session_config_entry("ep.webgpuexecutionprovider.dawnBackendType", arguments.backend)
    options.add_session_config_entry("ep.webgpuexecutionprovider.powerPreference", "high-performance")
    return ort.InferenceSession(
        arguments.model,
        sess_options=options,
        providers=["WebGpuExecutionProvider"],
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument(
        "--input",
        action="append",
        required=True,
        type=parse_input_spec,
        metavar="NAME=FILE.npy",
        help="Model input and NumPy file. Repeat for each model input.",
    )
    parser.add_argument("--backend", help="Optional Dawn backend, such as D3D12, Vulkan, or Metal.")
    parser.add_argument("--max-pending-dispatches", type=int, default=16)
    parser.add_argument("--warmups", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument(
        "--include-input-copies",
        action="store_true",
        help="Include host-to-device input updates in every timed iteration.",
    )
    capture_group = parser.add_mutually_exclusive_group()
    capture_group.add_argument(
        "--graph-capture",
        dest="graph_capture",
        action="store_true",
        help="Enable WebGPU graph capture (default).",
    )
    capture_group.add_argument(
        "--no-graph-capture",
        dest="graph_capture",
        action="store_false",
        help="Run the identical fixed-I/O workload without graph capture, for a matched baseline.",
    )
    parser.set_defaults(graph_capture=True)
    parser.add_argument("--output-json", type=Path)
    arguments = parser.parse_args()
    if arguments.warmups < 0 or arguments.iterations < 1:
        parser.error("--warmups must be non-negative and --iterations must be positive.")

    input_paths = dict(arguments.input)
    if len(input_paths) != len(arguments.input):
        raise ValueError("Each input name must be specified exactly once.")

    create_start = time.perf_counter()
    session = create_session(arguments)
    session_creation_ms = (time.perf_counter() - create_start) * 1000.0

    io_binding = session.io_binding()
    host_inputs: dict[str, np.ndarray] = {}
    device_inputs: dict[str, ort.OrtValue] = {}
    for metadata in session.get_inputs():
        if metadata.name not in input_paths:
            raise ValueError(f"Missing --input for {metadata.name}")
        host_input = np.load(input_paths[metadata.name])
        expected_shape = static_shape(metadata.name, metadata.shape)
        if list(host_input.shape) != expected_shape:
            raise ValueError(f"Input {metadata.name} has shape {host_input.shape}; expected {expected_shape}")
        device_input = session.create_ortvalue_from_shape_and_type(expected_shape, host_input.dtype, "webgpu")
        ort.copy_tensors([ort.OrtValue.ortvalue_from_numpy(host_input)], [device_input])
        io_binding.bind_ortvalue_input(metadata.name, device_input)
        host_inputs[metadata.name] = host_input
        device_inputs[metadata.name] = device_input

    unexpected_inputs = set(input_paths) - set(host_inputs)
    if unexpected_inputs:
        raise ValueError(f"Unknown model inputs: {sorted(unexpected_inputs)}")

    device_outputs: dict[str, ort.OrtValue] = {}
    for metadata in session.get_outputs():
        element_type = ORT_TYPE_TO_NUMPY.get(metadata.type)
        if element_type is None:
            raise ValueError(f"Unsupported output type for {metadata.name}: {metadata.type}")
        device_output = session.create_ortvalue_from_shape_and_type(
            static_shape(metadata.name, metadata.shape),
            element_type,
            "webgpu",
        )
        io_binding.bind_ortvalue_output(metadata.name, device_output)
        device_outputs[metadata.name] = device_output

    def run_once(include_input_copies: bool) -> tuple[float, list[np.ndarray]]:
        start = time.perf_counter()
        if include_input_copies:
            for name, device_input in device_inputs.items():
                ort.copy_tensors([ort.OrtValue.ortvalue_from_numpy(host_inputs[name])], [device_input])
        session.run_with_iobinding(io_binding)
        outputs = io_binding.copy_outputs_to_cpu()
        return (time.perf_counter() - start) * 1000.0, outputs

    try:
        first_inference_ms, outputs = run_once(False)
        for _ in range(arguments.warmups):
            _, outputs = run_once(arguments.include_input_copies)

        samples_ms = [run_once(arguments.include_input_copies)[0] for _ in range(arguments.iterations)]
        result = {
            "model": str(arguments.model.resolve()),
            "provider": session.get_providers()[0],
            "backend": arguments.backend,
            "max_pending_dispatches": arguments.max_pending_dispatches,
            "graph_capture": arguments.graph_capture,
            "warmups": arguments.warmups,
            "iterations": arguments.iterations,
            "input_boundary": ("host-to-device-per-run" if arguments.include_input_copies else "gpu-resident"),
            "output_boundary": "device-to-host-per-run",
            "session_creation_ms": session_creation_ms,
            "first_inference_ms": first_inference_ms,
            "mean_ms": statistics.mean(samples_ms),
            "stddev_ms": statistics.stdev(samples_ms) if len(samples_ms) > 1 else 0.0,
            "p50_ms": float(np.percentile(samples_ms, 50)),
            "p90_ms": float(np.percentile(samples_ms, 90)),
            "inferences_per_second": 1000.0 / statistics.mean(samples_ms),
            "samples_ms": samples_ms,
            "output_names": list(device_outputs),
            "output_shapes": [list(output.shape) for output in outputs],
        }
        serialized_result = json.dumps(result, indent=2)
        print(serialized_result)
        if arguments.output_json:
            arguments.output_json.write_text(serialized_result + "\n", encoding="utf-8")
    finally:
        if arguments.graph_capture:
            session.release_captured_graph()
        io_binding.clear_binding_inputs()
        io_binding.clear_binding_outputs()


if __name__ == "__main__":
    main()
