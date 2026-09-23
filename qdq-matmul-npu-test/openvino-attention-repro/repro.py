"""Standalone weight-free mask-QDQ/Add/Softmax reproducer for the WinML OpenVINO NPU."""

from __future__ import annotations

import argparse
import ctypes
import importlib.util
import json
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, checker, helper, numpy_helper


PROVIDER = "OpenVINOExecutionProvider"
WIDTH = 8
CASES = {"float_mask": None, "mask_scale_0p5": 0.5, "mask_scale_1": 1.0, "mask_scale_2": 2.0}


def build_model(scale: float | None) -> onnx.ModelProto:
    mask_name = "mask"
    initializers = []
    nodes = []
    if scale is not None:
        zero_point = int(100 / scale)
        initializers = [
            numpy_helper.from_array(np.array(scale, dtype=np.float32), "mask_scale"),
            numpy_helper.from_array(np.array(zero_point, dtype=np.uint16), "mask_zero_point"),
        ]
        nodes.extend([
            helper.make_node("QuantizeLinear", ["mask", "mask_scale", "mask_zero_point"], ["mask_q"]),
            helper.make_node("DequantizeLinear", ["mask_q", "mask_scale", "mask_zero_point"], ["mask_dq"]),
        ])
        mask_name = "mask_dq"
    nodes.extend([
        helper.make_node("Add", ["scores", mask_name], ["masked_scores"]),
        helper.make_node("Softmax", ["masked_scores"], ["output"], axis=-1),
    ])
    shape = [1, 1, WIDTH]
    graph = helper.make_graph(
        nodes, "mask_qdq_add_softmax",
        [helper.make_tensor_value_info(name, TensorProto.FLOAT, shape) for name in ("scores", "mask")],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, shape)],
        initializer=initializers,
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 21)])
    checker.check_model(model)
    return model


def find_library() -> Path:
    try:
        import winui3.microsoft.windows.ai.machinelearning as winml
        from winui3.microsoft.windows.applicationmodel.dynamicdependency.bootstrap import InitializeOptions, initialize
    except ImportError as error:
        raise RuntimeError("Install the Windows ML Python projections or pass --provider-library") from error
    with initialize(options=InitializeOptions.ON_NO_MATCH_SHOW_UI):
        provider = next(
            (candidate for candidate in winml.ExecutionProviderCatalog.get_default().find_all_providers()
             if candidate.name == PROVIDER),
            None,
        )
        if provider is None:
            raise RuntimeError(f"{PROVIDER} is not installed in the Windows ML catalog")
        ready = provider.ensure_ready_async().get()
        if ready.status != winml.ExecutionProviderReadyResultState.SUCCESS:
            raise RuntimeError(f"{PROVIDER} unavailable: {ready.status}: {ready.diagnostic_text}")
        return Path(provider.library_path)


def register_openvino(ort, library: Path | None):
    if library is None:
        library = find_library()
    library = library.resolve()
    if not library.is_file():
        raise FileNotFoundError(f"OpenVINO provider library does not exist: {library}")
    if not hasattr(ort, "register_execution_provider_library") or not hasattr(ort, "ModelCompiler"):
        raise RuntimeError("Use a Windows ML ONNX Runtime build with plugin registration and ModelCompiler")
    ort.register_execution_provider_library(PROVIDER, str(library))
    devices = [
        device for device in ort.get_ep_devices()
        if device.ep_name == PROVIDER and device.device.type == ort.OrtHardwareDeviceType.NPU
    ]
    if len(devices) != 1:
        raise RuntimeError(f"expected one {PROVIDER} NPU device, found {len(devices)}")
    return library, devices[0]


def ep_session_options(ort, device):
    options = ort.SessionOptions()
    options.add_provider_for_devices([device], {})
    options.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
    return options


def compile_and_run(
    ort, model: Path, output_dir: Path, device, inputs: dict[str, np.ndarray]
) -> tuple[np.ndarray, str | None]:
    context = output_dir / f"{model.stem}_ctx.onnx"
    ort.ModelCompiler(
        ep_session_options(ort, device), model, embed_compiled_data_into_model=False,
        flags=ort.OrtCompileApiFlags.ERROR_IF_NO_NODES_COMPILED,
        graph_optimization_level=ort.GraphOptimizationLevel.ORT_ENABLE_BASIC,
    ).compile_to_file(context)
    compiled = onnx.load(context, load_external_data=False)
    if len(compiled.graph.node) != 1 or compiled.graph.node[0].op_type != "EPContext":
        raise RuntimeError(f"{model.name}: expected one EPContext without CPU fallback")
    versions = [
        attribute.s.decode("utf-8")
        for attribute in compiled.graph.node[0].attribute if attribute.name == "ep_sdk_version"
    ]
    if len(versions) > 1:
        raise RuntimeError(f"{model.name}: ambiguous EP SDK version in context")
    session = ort.InferenceSession(str(context), sess_options=ep_session_options(ort, device))
    return session.run(None, inputs)[0], versions[0] if versions else None


def run(output_dir: Path, provider_library: Path | None = None, cpu_only: bool = False) -> dict:
    if output_dir.exists():
        raise FileExistsError(f"output directory already exists: {output_dir}")
    output_dir.mkdir(parents=True)
    import onnxruntime as ort

    inputs = {
        "scores": np.linspace(-1, 1, WIDTH, dtype=np.float32).reshape(1, 1, WIDTH),
        "mask": np.zeros((1, 1, WIDTH), dtype=np.float32),
    }
    np.savez_compressed(output_dir / "inputs.npz", **inputs)
    cpu_options = ort.SessionOptions()
    cpu_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    cpu_outputs = {}
    for name, scale in CASES.items():
        model = output_dir / f"{name}.onnx"
        onnx.save(build_model(scale), model)
        cpu_outputs[name] = ort.InferenceSession(
            str(model), sess_options=cpu_options, providers=["CPUExecutionProvider"]
        ).run(None, inputs)[0]
    reference = cpu_outputs["float_mask"]
    if any(not np.array_equal(reference, output) for output in cpu_outputs.values()):
        raise RuntimeError("CPU outputs differ: this input is not a semantics-preserving QDQ control")

    result = {
        "ort_version": ort.__version__,
        "inputs": "all-zero mask; float32 scores linspace(-1, 1, 8); shape [1, 1, 8]",
        "cpu_outputs_bitwise_identical": True,
        "models": {name: [node.op_type for node in build_model(scale).graph.node] for name, scale in CASES.items()},
        "ep_tested": not cpu_only,
    }
    if not cpu_only:
        library, device = register_openvino(ort, provider_library)
        ep_outputs = {}
        result["provider_library_name"] = library.name
        result["device"] = {
            "ep_vendor": str(device.ep_vendor),
            "vendor_id": int(device.device.vendor_id),
            "device_id": int(device.device.device_id),
        }
        result["cases"] = {}
        sdk_versions = set()
        for name in CASES:
            ep_output, sdk_version = compile_and_run(ort, output_dir / f"{name}.onnx", output_dir, device, inputs)
            sdk_versions.add(sdk_version)
            ep_outputs[name] = ep_output
            result["cases"][name] = {
                "cpu_ep_mae": float(np.mean(np.abs(reference.astype(np.float64) - ep_output))),
                "ep_context_nodes": 1,
                "ep_output": ep_output.ravel().tolist(),
            }
        if len(sdk_versions) != 1:
            raise RuntimeError(f"inconsistent EP SDK versions in compiled contexts: {sdk_versions}")
        result["ep_sdk_version"] = sdk_versions.pop()
        control_mae = result["cases"]["float_mask"]["cpu_ep_mae"]
        buggy_mae = result["cases"]["mask_scale_1"]["cpu_ep_mae"]
        result["reproduced"] = control_mae < 1e-3 and buggy_mae > max(1e-3, 10 * control_mae)
        result["scale_half_matches_control"] = bool(
            np.array_equal(ep_outputs["mask_scale_0p5"], ep_outputs["float_mask"])
        )
    (output_dir / "results.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"Saved models, inputs, and results: {output_dir}")
    if cpu_only:
        print("CPU-only control passed; OpenVINO NPU was NOT tested.")
    else:
        print(f"Reproduced: {result['reproduced']}; CPU/EP MAE by case: "
              f"{ {name: case['cpu_ep_mae'] for name, case in result['cases'].items()} }")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--provider-library", type=Path, help="OpenVINO WinML plugin DLL; default: Windows ML catalog.")
    parser.add_argument(
        "--cpu-only", action="store_true", help="Generate models and validate CPU controls without an NPU."
    )
    args = parser.parse_args()
    if args.provider_library is not None and args.cpu_only:
        parser.error("--provider-library cannot be used with --cpu-only")
    if not args.cpu_only:
        spec = importlib.util.find_spec("onnxruntime")
        if spec is None or spec.origin is None:
            raise RuntimeError("Install the Windows ML ONNX Runtime Python package")
        dll = Path(spec.origin).parent / "capi" / "onnxruntime.dll"
        if not dll.is_file():
            raise RuntimeError(f"Windows ML ONNX Runtime DLL is missing: {dll}")
        ctypes.WinDLL(str(dll))
    result = run(args.output_dir.resolve(), args.provider_library, args.cpu_only)
    if not args.cpu_only and not result["reproduced"]:
        raise RuntimeError("The OpenVINO NPU discrepancy was not reproduced; inspect results.json")


if __name__ == "__main__":
    main()
