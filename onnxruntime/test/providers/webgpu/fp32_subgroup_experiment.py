# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
"""Opt-in resident-buffer MatMulNBits benchmark; not a production dispatch test.

Use the integrated ORT Python package and experimental WebGPU plugin. On macOS,
--backend Metal queries the real adapter instead of assuming matrix support.
Native GPU timestamps require the separate profiling plugin used by this study.
"""

import argparse
import ctypes
import gc
import hashlib
import importlib
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path


def errors(np, actual, reference):
    actual = actual.astype(np.float64)
    finite = np.isfinite(actual) & np.isfinite(reference)
    difference = actual[finite] - reference[finite]
    return {
        "elements": actual.size,
        "nonfinite": int((~np.isfinite(actual)).sum()),
        "max_abs": float(np.max(np.abs(difference))) if difference.size else None,
        "relative_l2": float(np.linalg.norm(difference) / max(np.linalg.norm(reference[finite]), 1e-30))
        if difference.size
        else None,
        "rmse": float(np.sqrt(np.mean(difference**2))) if difference.size else None,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ort-package", type=Path)
    parser.add_argument("--plugin", required=True, type=Path)
    parser.add_argument(
        "--model", type=Path, help="Pruned Qwen directory; otherwise use deterministic synthetic weights."
    )
    parser.add_argument("--backend", choices=["Vulkan", "Metal"], default="Vulkan")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--routes", nargs="+", default=["dp4a", "mixed-64x64x32", "mixed-64x128x32", "mixed-64x128x64", "f32", "fp16"]
    )
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=4)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--stress-only", action="store_true")
    parser.add_argument("--scenario", help="Run only the named scenario, retaining its original random seed.")
    parser.add_argument("--gpu", action="store_true", help="Requires TtftFlushTrace profiling plugin.")
    parser.add_argument("--activations", type=Path, help="Directory of untimed real-model input captures.")
    args = parser.parse_args()
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    if args.ort_package:
        sys.path.insert(0, str(args.ort_package.resolve()))
    np = importlib.import_module("numpy")
    onnx = importlib.import_module("onnx")
    ort = importlib.import_module("onnxruntime")

    ort.register_execution_provider_library("fp32-subgroup-experiment", str(args.plugin.resolve()))
    library = ctypes.CDLL(str(args.plugin.resolve()))
    if args.gpu:
        library.TtftFlushTrace.argtypes = []
        library.TtftFlushTrace.restype = None
    devices = [d for d in ort.get_ep_devices() if d.ep_name == "WebGpuExecutionProvider"]
    if args.backend == "Vulkan":
        pci = (
            subprocess.check_output(["nvidia-smi", "--query-gpu=pci.bus_id", "--format=csv,noheader"], text=True)
            .strip()
            .lower()[-12:]
        )
        devices = [d for d in devices if d.device.metadata.get("pci_bus_id") == pci]
    assert len(devices) == 1, devices
    os.environ["QWEN_CAPS_PATH"] = str(args.output / "device-caps.json")
    os.environ["ORT_WEBGPU_EXPERIMENTAL_MATMUL_NBITS_CAPS"] = str(args.output / "experimental-device-caps.json")
    os.environ["ORT_WEBGPU_EP_SHADER_DUMP_FILE"] = str(args.output / "shaders.txt")

    def clients():
        if args.backend == "Metal":
            return []
        return (
            subprocess.check_output(
                ["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader"], text=True
            )
            .strip()
            .splitlines()
        )

    def activate(route):
        os.environ.pop("ORT_WEBGPU_EXPERIMENTAL_MATMUL_NBITS_FP32", None)
        os.environ.pop("ORT_WEBGPU_EXPERIMENTAL_MATMUL_NBITS_TILE", None)
        if route.startswith("mixed"):
            os.environ["ORT_WEBGPU_EXPERIMENTAL_MATMUL_NBITS_FP32"] = "mixed"
            os.environ["ORT_WEBGPU_EXPERIMENTAL_MATMUL_NBITS_TILE"] = route.split("-", 1)[1]
        elif route in ("f32", "auto"):
            os.environ["ORT_WEBGPU_EXPERIMENTAL_MATMUL_NBITS_FP32"] = route

    def session_for(M, N, K, weights, scales, zero, bias, route):
        inputs = ["A", "B", "scales"]
        initializers = [onnx.numpy_helper.from_array(weights, "B"), onnx.numpy_helper.from_array(scales, "scales")]
        if zero is not None or bias is not None:
            inputs += ["zero" if zero is not None else "", "", "bias" if bias is not None else ""]
        if zero is not None:
            initializers.append(onnx.numpy_helper.from_array(zero, "zero"))
        if bias is not None:
            initializers.append(onnx.numpy_helper.from_array(bias, "bias"))
        node = onnx.helper.make_node(
            "MatMulNBits",
            inputs,
            ["Y"],
            name="fp32-experiment",
            domain="com.microsoft",
            N=N,
            K=K,
            bits=4,
            block_size=32,
            accuracy_level=4,
        )
        graph = onnx.helper.make_graph(
            [node],
            "fp32-subgroup-experiment",
            [onnx.helper.make_tensor_value_info("A", onnx.TensorProto.FLOAT16, [M, K])],
            [onnx.helper.make_tensor_value_info("Y", onnx.TensorProto.FLOAT16, [M, N])],
            initializer=initializers,
        )
        model = onnx.helper.make_model(
            graph,
            opset_imports=[onnx.helper.make_opsetid("", 21), onnx.helper.make_opsetid("com.microsoft", 1)],
            ir_version=10,
        )
        options = ort.SessionOptions()
        options.intra_op_num_threads = options.inter_op_num_threads = 1
        options.log_severity_level = 3
        options.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
        options.add_provider_for_devices(
            devices,
            {
                "dawnBackendType": args.backend,
                "validationMode": "basic",
                "enableGraphCapture": "0",
                "deviceId": "0",
                "enableMatmulFp32Accumulation": str(int(route != "fp16")),
            },
        )
        return ort.InferenceSession(model.SerializeToString(), sess_options=options)

    nodes, initializers = {}, {}
    if args.model:
        graph = onnx.load(args.model / "model.onnx", load_external_data=False).graph
        initializers = {tensor.name: tensor for tensor in graph.initializer}
        for node in graph.node:
            if node.op_type == "MatMulNBits":
                attrs = {a.name: onnx.helper.get_attribute_value(a) for a in node.attribute}
                nodes.setdefault((attrs["N"], attrs["K"]), node)
    families = [(17408, 5120), (5120, 17408), (10240, 5120), (5120, 6144), (6144, 5120), (14336, 5120)]
    shapes = [(M, N, K, "qwen" if args.model else "synthetic") for N, K in families for M in (512, 291)]
    shapes += [(86, N, K, "qwen" if args.model else "synthetic") for N, K in families[:2]]
    stress = [
        (291, 128, 5120, "overflow-cancel"),
        (257, 192, 5120, "zero-scale-bias"),
        (256, 128, 5120, "rounding"),
        (256, 128, 5120, "small-cancel"),
        (256, 128, 5120, "product-overflow"),
        (256, 128, 5120, "dequant-overflow"),
        (256, 128, 5120, "dequant-impulse"),
    ]
    shapes = stress if args.stress_only else shapes + stress
    if args.smoke:
        shapes = shapes[:1]
    result = {
        "arguments": vars(args)
        | {
            "plugin": str(args.plugin),
            "output": str(args.output),
            "model": str(args.model),
            "ort_package": str(args.ort_package),
            "activations": str(args.activations),
        },
        "plugin_sha256": hashlib.sha256(args.plugin.read_bytes()).hexdigest(),
        "gpu_clients_before": clients(),
        "shapes": [],
        "trials": [],
    }
    for shape_index, (M, N, K, scenario) in enumerate(shapes):  # noqa: N806
        if args.scenario and scenario != args.scenario:
            continue
        rng = np.random.default_rng(20261028 + shape_index)
        tag = f"M{M}-N{N}-K{K}-{scenario}"
        zero = bias = None
        zp = np.full((N, K // 32), 8, np.int16)
        provenance = {}
        os.environ.pop("ORT_WEBGPU_EXPERIMENTAL_MATMUL_NBITS_VALIDATE", None)
        if scenario not in ("qwen", "synthetic"):
            os.environ["ORT_WEBGPU_EXPERIMENTAL_MATMUL_NBITS_VALIDATE"] = "1"
        if scenario == "qwen":
            node = nodes[(N, K)]
            if args.activations and M == 512:
                metadata = json.loads((args.activations / f"M{M}-N{N}-K{K}.json").read_text())
                if metadata["node"] == "MatMulNBitsMlp":
                    assert node.name == "/model/layers.0/mlp/gate_proj/MatMul_Q4"
                    provenance["fused_capture"] = "First unfused gate in MatMulNBitsMlp, before the up projection."
                else:
                    node = next(n for n in graph.node if n.name == metadata["node"])
            weights = onnx.numpy_helper.to_array(initializers[node.input[1]], base_dir=str(args.model)).copy()
            scales = onnx.numpy_helper.to_array(initializers[node.input[2]], base_dir=str(args.model)).copy()
            provenance.update({"node": node.name, "weight": node.input[1], "scales": node.input[2]})
        else:
            weights = rng.integers(0, 256, (N, K // 32, 16), dtype=np.uint8)
            scales = rng.uniform(0.001, 0.02, (N, K // 32)).astype(np.float16)
        activations = rng.normal(0.0, 0.5, (M, K)).astype(np.float16)
        if args.activations and scenario == "qwen" and M == 512:
            activations = np.fromfile(args.activations / f"M{M}-N{N}-K{K}.f16", dtype=np.float16).reshape(M, K)
            provenance["activations"] = str(args.activations / f"M{M}-N{N}-K{K}.f16")
        if scenario == "overflow-cancel":
            weights.fill(0x99)
            scales.fill(0.5)
            activations.fill(64)
            activations[:, K // 2 :] = -64
        elif scenario == "small-cancel":
            weights.fill(0x99)
            scales.fill(0.5)
            activations.fill(1)
            activations[:, K // 2 :] = -1
            activations[:, -1] = -0.999
        elif scenario == "product-overflow":
            weights.fill(0x99)
            scales.fill(64)
            activations.fill(2048)
            activations[:, K // 2 :] = -2048
        elif scenario == "dequant-overflow":
            weights.fill(0xFF)
            scales.fill(16384)
            activations.fill(2**-10)
            activations[:, K // 2 :] = -(2**-10)
        elif scenario == "dequant-impulse":
            scales.fill(np.float16(0.1234))
            activations.fill(0)
            activations[np.arange(M), np.arange(M) * 19 % K] = 1
        elif scenario == "rounding":
            scales.fill(np.float16(0.1234))
        elif scenario == "zero-scale-bias":
            zp = rng.integers(0, 16, (N, K // 32), dtype=np.int16)
            zero = (zp[:, 0::2] | (zp[:, 1::2] << 4)).astype(np.uint8)
            scales[:, 0::4] = 0
            scales[:, 1::4] *= -1
            bias = rng.normal(0, 0.2, N).astype(np.float16)
        columns = np.unique(np.linspace(0, N - 1, min(N, 64), dtype=np.int64))
        packed = weights[columns].reshape(len(columns), K // 2)
        integer = np.empty((len(columns), K), np.int16)
        integer[:, 0::2] = packed & 15
        integer[:, 1::2] = packed >> 4
        integer -= np.repeat(zp[columns], 32, axis=1)
        dequantized = integer.astype(np.float64) * np.repeat(scales[columns], 32, axis=1).astype(np.float64)
        reference = activations.astype(np.float64) @ dequantized.T
        rounded_reference = activations.astype(np.float64) @ dequantized.astype(np.float16).astype(np.float64).T
        if bias is not None:
            reference += bias[columns].astype(np.float64)
            rounded_reference += bias[columns].astype(np.float64)
        np.savez(
            args.output / f"{tag}-reference.npz",
            columns=columns,
            reference=reference,
            rounded_reference=rounded_reference,
        )
        entry = {
            "tag": tag,
            "shape": [M, N, K],
            "provenance": provenance,
            "errors": {},
            "weight_sha256": hashlib.sha256(weights.tobytes()).hexdigest(),
            "scale_sha256": hashlib.sha256(scales.tobytes()).hexdigest(),
            "activation_sha256": hashlib.sha256(activations.tobytes()).hexdigest(),
        }
        sessions = {}
        for route in args.routes:
            activate(route)
            session = session_for(M, N, K, weights, scales, zero, bias, route)
            a = session.create_ortvalue_from_shape_and_type([M, K], np.float16, "gpu", 0, vendor_id=0)
            a.update_inplace(activations)
            y = session.create_ortvalue_from_shape_and_type([M, N], np.float16, "gpu", 0, vendor_id=0)
            binding = session.io_binding()
            binding.bind_ortvalue_input("A", a)
            binding.bind_ortvalue_output("Y", y)
            sessions[route] = session, binding, a, y
        for phase, count in (("warmup", args.warmups), ("gpu" if args.gpu else "clean", args.trials)):
            for repetition in range(count):
                order = args.routes[repetition % len(args.routes) :] + args.routes[: repetition % len(args.routes)]
                if repetition % 2:
                    order.reverse()
                for route in order:
                    activate(route)
                    session, binding, a, y = sessions[route]
                    stem = f"{tag}-{route}-{phase}-{repetition}"
                    if phase == "gpu":
                        os.environ["QWEN_GPU_TRACE"] = str(args.output / f"{stem}.jsonl")
                        os.environ["QWEN_GPU_STAGE"] = "micro"
                    start = time.perf_counter_ns()
                    for _ in range(args.repeats):
                        session.run_with_iobinding(binding)
                        binding.synchronize_outputs()
                    elapsed = (time.perf_counter_ns() - start) / 1e6 / args.repeats
                    output = binding.copy_outputs_to_cpu()[0]
                    if phase == "gpu":
                        library.TtftFlushTrace()
                    os.environ.pop("QWEN_GPU_TRACE", None)
                    os.environ.pop("QWEN_GPU_STAGE", None)
                    if phase != "warmup":
                        sample = {
                            "shape": tag,
                            "route": route,
                            "repetition": repetition,
                            "wall_ms": elapsed,
                            "gpu_clients": clients(),
                        }
                        trace = args.output / f"{stem}.jsonl"
                        if trace.exists():
                            traces = [json.loads(line) for line in trace.read_text().splitlines()]
                            kernels = [k for event in traces for k in event["kernels"]]
                            sample["gpu_ms"] = sum(k["end_ns"] - k["start_ns"] for k in kernels) / 1e6 / args.repeats
                            sample["programs"] = sorted({k["name"] for k in kernels})
                        result["trials"].append(sample)
                        if repetition == 0:
                            selected = output[:, columns]
                            np.save(args.output / f"{tag}-{route}.npy", selected)
                            entry["errors"][route] = {
                                "exact_dequant": errors(np, selected, reference),
                                "fp16_dequant": errors(np, selected, rounded_reference),
                                "all_output_nonfinite": int((~np.isfinite(output)).sum()),
                            }
        for route in args.routes:
            samples = [v for v in result["trials"] if v["shape"] == tag and v["route"] == route]
            print(
                tag,
                route,
                statistics.median(v["wall_ms"] for v in samples),
                entry["errors"][route]["exact_dequant"],
                flush=True,
            )
        result["shapes"].append(entry)
        (args.output / "result.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
        sessions.clear()
        del session, binding, a, y, output
        gc.collect()
    if not result["shapes"]:
        parser.error("--scenario did not match the selected model/stress/smoke cases")
    result["status"] = "completed"
    (args.output / "result.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
