# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
"""Manual Conv reproductions for https://github.com/microsoft/onnxruntime/issues/32777.

Requires numpy, onnx and a WebGPU-enabled onnxruntime build, or the WebGPU plugin
package with --plugin. The accuracy scenario is a single Conv with 64 outputs;
the chain scenario is two Conv+Relu blocks with dynamic spatial dimensions.
The large chain can hang/reset affected GPUs. It is not an automatic CI test.

Examples:
    python webgpu_conv_repro.py --channels 99
    python webgpu_conv_repro.py --scenario chain --size 280 --plugin
    python webgpu_conv_repro.py --scenario chain --size 560 --max-pending-dispatches 1

Profiling is opt-in because it can change timeout behavior. Compare separate
processes when changing context-level options such as the pending-dispatch limit.
Record the adapter and driver alongside results; WebGPU can also select a
software adapter, so successful execution alone does not establish GPU coverage.
"""

import argparse
import json
import platform
import time
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

import onnxruntime as ort


def positive_int(value):
    value = int(value)
    if value <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return value


def make_model(scenario, channels, size):
    rng = np.random.default_rng(0)
    initializers, nodes = [], []
    chain = scenario == "chain"
    output_channels = channels if chain else 64
    previous = "input"
    for i in range(2 if chain else 1):
        weights = (rng.standard_normal((output_channels, channels, 3, 3)) * (0.01 if chain else 0.05)).astype(
            np.float32
        )
        bias = rng.standard_normal(output_channels).astype(np.float32) if chain else np.zeros(64, np.float32)
        initializers.extend([numpy_helper.from_array(weights, f"w{i}"), numpy_helper.from_array(bias, f"b{i}")])
        nodes.append(
            helper.make_node(
                "Conv", [previous, f"w{i}", f"b{i}"], [f"c{i}"], name=f"conv{i}", kernel_shape=[3, 3], pads=[1, 1, 1, 1]
            )
        )
        previous = f"c{i}"
        if chain:
            nodes.append(helper.make_node("Relu", [previous], [f"r{i}"], name=f"relu{i}"))
            previous = f"r{i}"
    nodes[-1].output[0] = "output"
    spatial = ["H", "W"] if chain else [size, size]
    graph = helper.make_graph(
        nodes,
        "webgpu_conv_repro",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, channels, *spatial])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, output_channels, *spatial])],
        initializers,
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18 if chain else 11)])
    model.ir_version = 9
    onnx.checker.check_model(model)
    shape = (1, channels, size, size)
    data = (
        rng.random(shape, dtype=np.float32)
        if chain
        else np.random.default_rng(1).standard_normal(shape).astype(np.float32)
    )
    return model.SerializeToString(), data


def make_gpu_session(model, args):
    options = ort.SessionOptions()
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    options.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
    options.enable_profiling = args.profile
    if args.profile:
        prefix = args.output.with_suffix("") if args.output else Path("webgpu_conv_repro")
        options.profile_file_prefix = str(prefix) + "-profile"
    provider_options = {}
    if args.backend:
        provider_options["dawnBackendType"] = args.backend
    if args.max_pending_dispatches is not None:
        provider_options["maxNumPendingDispatches"] = str(args.max_pending_dispatches)
    if args.plugin:
        import onnxruntime_ep_webgpu as webgpu_plugin  # noqa: PLC0415

        ort.register_execution_provider_library("webgpu_conv_repro", webgpu_plugin.get_library_path())
        devices = [d for d in ort.get_ep_devices() if d.ep_name == webgpu_plugin.get_ep_name()]
        if not devices:
            raise RuntimeError("The WebGPU plugin did not expose a device")
        options.add_provider_for_devices([devices[0]], provider_options)
        return ort.InferenceSession(model, sess_options=options)
    if "WebGpuExecutionProvider" not in ort.get_available_providers():
        raise RuntimeError("This ORT build has no built-in WebGPU EP; use a WebGPU-enabled build or --plugin")
    if args.adapter_index is not None:
        provider_options["adapterIndex"] = str(args.adapter_index)
    for key, value in provider_options.items():
        options.add_session_config_entry(f"ep.webgpuexecutionprovider.{key}", value)
    return ort.InferenceSession(model, sess_options=options, providers=["WebGpuExecutionProvider"])


def compare_outputs(actual, expected):
    if actual.shape != expected.shape:
        raise RuntimeError(f"Output shape mismatch: {actual.shape} versus {expected.shape}")
    actual, expected = actual.ravel(), expected.ravel()
    max_error = max_reference = error_squared = reference_squared = 0.0
    mismatches = 0
    finite = True
    # Bound temporary CPU memory for the roughly 380 MB outputs in the chain case.
    for start in range(0, actual.size, 1 << 20):
        a = actual[start : start + (1 << 20)].astype(np.float64)
        b = expected[start : start + (1 << 20)].astype(np.float64)
        if not np.isfinite(b).all():
            raise RuntimeError("CPU reference contains non-finite values")
        valid = np.isfinite(a)
        finite = finite and bool(valid.all())
        error = np.abs(a - b)
        mismatches += int(np.count_nonzero(~valid | (error > 3e-5 + 3e-4 * np.abs(b))))
        if valid.all():
            max_error = max(max_error, float(error.max()))
            max_reference = max(max_reference, float(np.abs(b).max()))
            error_squared += float(np.dot(error, error))
            reference_squared += float(np.dot(b, b))
    return {
        "finite": finite,
        "mismatched_elements": mismatches,
        "output_elements": int(actual.size),
        "max_abs_error": max_error if finite else None,
        "normalized_max_error": max_error / max_reference if finite and max_reference else None,
        "relative_l2_error": (error_squared / reference_squared) ** 0.5 if finite and reference_squared else None,
        "absolute_tolerance": 3e-5,
        "relative_tolerance": 3e-4,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scenario", choices=("accuracy", "chain"), default="accuracy")
    parser.add_argument("--channels", type=positive_int, help="input channels (default: accuracy=99, chain=303)")
    parser.add_argument("--size", type=positive_int, help="height and width (default: accuracy=128, chain=560)")
    parser.add_argument("--plugin", action="store_true", help="use the installed onnxruntime-ep-webgpu package")
    parser.add_argument("--backend", choices=("D3D12", "Vulkan"), help="otherwise use the platform default")
    parser.add_argument("--adapter-index", type=int, help="built-in EP only; otherwise use default adapter selection")
    parser.add_argument("--max-pending-dispatches", type=positive_int, help="otherwise keep the provider default")
    parser.add_argument("--profile", action="store_true", help="enable ORT profiling; may change timeout behavior")
    parser.add_argument("--output", type=Path, help="also write the JSON result to this file")
    args = parser.parse_args()
    if args.adapter_index is not None and (args.adapter_index < 0 or args.plugin):
        parser.error("--adapter-index must be non-negative and is only supported with the built-in EP")
    if args.max_pending_dispatches is not None and args.max_pending_dispatches > 4096:
        parser.error("--max-pending-dispatches must be in [1, 4096]")
    channels = args.channels or (303 if args.scenario == "chain" else 99)
    size = args.size or (560 if args.scenario == "chain" else 128)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "scenario": args.scenario,
        "channels": channels,
        "size": size,
        "ort_version": ort.__version__,
        "platform": platform.platform(),
        "plugin": args.plugin,
        "profiling": args.profile,
        "backend": args.backend,
        "adapter_index": args.adapter_index,
        "max_pending_dispatches": args.max_pending_dispatches,
    }
    try:
        model, data = make_model(args.scenario, channels, size)
        session = make_gpu_session(model, args)
        report["providers"] = session.get_providers()
        start = time.perf_counter()
        actual = session.run(None, {"input": data})[0]
        report["webgpu_seconds"] = time.perf_counter() - start
        if args.profile:
            report["profile_path"] = session.end_profiling()
        cpu = ort.InferenceSession(model, providers=["CPUExecutionProvider"])
        expected = cpu.run(None, {"input": data})[0]
        report.update(compare_outputs(actual, expected))
        report["status"] = "passed" if report["mismatched_elements"] == 0 else "numerical_mismatch"
    except Exception as error:
        report["status"] = "execution_failed"
        report["error"] = str(error)
    text = json.dumps(report, indent=2, allow_nan=False) + "\n"
    print(text, end="")
    if args.output:
        args.output.write_text(text, encoding="utf-8")
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
