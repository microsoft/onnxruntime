#!/usr/bin/env python3
"""Smoke test for the onnxruntime-ep-webgpu Python package.

Tests:
1. Package import and library path resolution
2. EP registration with ONNX Runtime
3. Device discovery
4. Inference with a simple Mul model (requires WebGPU-capable hardware)

The inference test is skipped gracefully if no WebGPU device is available
(e.g., on CPU-only build agents).
"""

import os
import platform
import sys
import tempfile
import traceback
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper

import onnxruntime as ort

VERBOSE = os.environ.get("ORT_TEST_VERBOSE", "").strip().lower() in ("1", "true", "yes")


def debug_print(*args, **kwargs):
    """Print only when ORT_TEST_VERBOSE is set to a truthy value."""
    if VERBOSE:
        print(*args, **kwargs)


def create_mul_model(output_dir: Path) -> Path:
    """Create a simple Mul model in `output_dir` and return the path to the saved .onnx file."""
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [2, 3])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [2, 3])
    z = helper.make_tensor_value_info("z", TensorProto.FLOAT, [2, 3])

    mul_node = helper.make_node("Mul", inputs=["x", "y"], outputs=["z"])

    graph = helper.make_graph([mul_node], "mul_graph", [x, y], [z])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 7

    model_path = output_dir / "mul.onnx"
    onnx.save(model, str(model_path))
    return model_path


def print_environment_info():
    """Print diagnostic information about the runtime environment."""
    print(f"  Python: {sys.version}")
    print(f"  Platform: {platform.platform()}")
    print(f"  Architecture: {platform.machine()}")
    print(f"  ONNX Runtime version: {ort.__version__}")
    print(f"  ONNX Runtime location: {ort.__file__}")
    print(f"  Available providers (built-in): {ort.get_available_providers()}")
    # Print relevant environment variables
    for var in sorted(os.environ):
        lower = var.lower()
        if any(kw in lower for kw in ["onnx", "ort", "gpu", "cuda", "vulkan", "webgpu", "dawn", "path", "ld_library"]):
            print(f"  ENV {var}={os.environ[var]}")


def test_import_and_library_path():
    """Test that the package imports and the library path is valid."""
    import onnxruntime_ep_webgpu as webgpu_ep  # noqa: PLC0415  # `import` should be at the top-level of a file.

    debug_print(f"  Package location: {webgpu_ep.__file__}")
    pkg_dir = Path(webgpu_ep.__file__).parent
    debug_print(f"  Package directory contents: {sorted(p.name for p in pkg_dir.iterdir())}")

    lib_path = webgpu_ep.get_library_path()
    assert Path(lib_path).is_file(), f"Library path does not exist: {lib_path}"
    print(f"OK: Library path: {lib_path}")

    ep_name = webgpu_ep.get_ep_name()
    assert ep_name == "WebGpuExecutionProvider", f"Unexpected EP name: {ep_name}"
    print(f"OK: EP name: {ep_name}")

    ep_names = webgpu_ep.get_ep_names()
    assert ep_names == ["WebGpuExecutionProvider"], f"Unexpected EP names: {ep_names}"
    print(f"OK: EP names: {ep_names}")


def test_registration_and_inference():
    """Test EP registration, device discovery, and inference."""
    import onnxruntime_ep_webgpu as webgpu_ep  # noqa: PLC0415  # `import` should be at the top-level of a file.

    lib_path = webgpu_ep.get_library_path()
    ep_name = webgpu_ep.get_ep_name()
    registration_name = "webgpu_plugin_test"

    # Register the plugin EP
    debug_print(f"  Registering library: {lib_path}")
    debug_print(f"  Library file size: {Path(lib_path).stat().st_size} bytes")
    ort.register_execution_provider_library(registration_name, lib_path)
    print(f"OK: Registered EP library as '{registration_name}'")

    try:
        # Discover devices
        all_devices = ort.get_ep_devices()
        debug_print(f"  All devices: {[(d.ep_name, getattr(d, 'device_id', 'N/A')) for d in all_devices]}")
        webgpu_devices = [d for d in all_devices if d.ep_name == ep_name]
        print(f"Found {len(webgpu_devices)} WebGPU device(s)")

        if not webgpu_devices:
            print("SKIP: No WebGPU devices available — skipping inference test")
            return

        # Create session with WebGPU EP
        sess_options = ort.SessionOptions()
        sess_options.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
        sess_options.add_provider_for_devices(webgpu_devices, {})
        assert sess_options.has_providers(), "SessionOptions should have providers after add_provider_for_devices"
        print("OK: Session options configured with WebGPU EP")

        with tempfile.TemporaryDirectory() as model_dir:
            model_path = create_mul_model(Path(model_dir))
            debug_print(f"  Model path: {model_path}")
            sess = ort.InferenceSession(model_path, sess_options=sess_options)
            debug_print(f"  Session providers: {sess.get_providers()}")
            print("OK: InferenceSession created")

            # Run inference
            x = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
            y = np.array([[2.0, 3.0, 4.0], [5.0, 6.0, 7.0]], dtype=np.float32)
            expected = x * y

            outputs = sess.run(None, {"x": x, "y": y})
            result = outputs[0]

            np.testing.assert_allclose(result, expected, rtol=1e-5, atol=1e-5)
            print("OK: Inference result matches expected output")

            del sess
            print("OK: Session released")

    finally:
        ort.unregister_execution_provider_library(registration_name)
        print(f"OK: Unregistered EP library '{registration_name}'")


def main():
    print("=== WebGPU Plugin EP Python Package Test ===")

    if VERBOSE:
        # Set verbose ORT logging so ORT internals are visible in CI logs
        ort.set_default_logger_severity(0)

        print("\n--- Environment ---")
        print_environment_info()

    print("\n--- Test 1: Import and library path ---")
    test_import_and_library_path()

    print("\n--- Test 2: Registration and inference ---")
    test_registration_and_inference()

    print("\n=== All tests passed ===")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\nFAILED: {e}", file=sys.stderr)
        traceback.print_exc()
        sys.exit(1)
