# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import os
import sys

import numpy as np
import onnx
import torch
import torch.nn.functional as F
from onnx import TensorProto, helper, save

import onnxruntime as onnxrt

try:
    import faulthandler

    faulthandler.enable()
except ImportError:
    pass


def create_add_model(model_path):
    # Create a simple Add model: Y = A + B
    node_def = helper.make_node("Add", ["A", "B"], ["Y"])
    graph_def = helper.make_graph(
        [node_def],
        "test-model-add",
        [
            helper.make_tensor_value_info("A", TensorProto.FLOAT, [3, 2]),
            helper.make_tensor_value_info("B", TensorProto.FLOAT, [3, 2]),
        ],
        [helper.make_tensor_value_info("Y", TensorProto.FLOAT, [3, 2])],
    )
    model_def = helper.make_model(graph_def, producer_name="onnx-example")
    save(model_def, model_path)


def create_matmul_model(model_path):
    # Create a simple MatMul model: Y = A @ B
    node_def = helper.make_node("MatMul", ["A", "B"], ["Y"])
    graph_def = helper.make_graph(
        [node_def],
        "test-model-matmul",
        [
            helper.make_tensor_value_info("A", TensorProto.FLOAT, [3, 4]),
            helper.make_tensor_value_info("B", TensorProto.FLOAT, [4, 5]),
        ],
        [helper.make_tensor_value_info("Y", TensorProto.FLOAT, [3, 5])],
    )
    model_def = helper.make_model(graph_def, producer_name="onnx-example")
    save(model_def, model_path)


def create_gemm_model(model_path, alpha=1.0, beta=1.0, transA=0, transB=0):
    # Create a simple Gemm model: Y = alpha*A*B + beta*C
    node_def = helper.make_node("Gemm", ["A", "B", "C"], ["Y"], alpha=alpha, beta=beta, transA=transA, transB=transB)

    m = 3
    k = 4
    n = 5
    shape_a = [m, k] if transA == 0 else [k, m]
    shape_b = [k, n] if transB == 0 else [n, k]
    shape_c = [n]  # Test broadcast

    graph_def = helper.make_graph(
        [node_def],
        "test-model-gemm",
        [
            helper.make_tensor_value_info("A", TensorProto.FLOAT, shape_a),
            helper.make_tensor_value_info("B", TensorProto.FLOAT, shape_b),
            helper.make_tensor_value_info("C", TensorProto.FLOAT, shape_c),
        ],
        [helper.make_tensor_value_info("Y", TensorProto.FLOAT, [m, n])],
    )
    model_def = helper.make_model(graph_def, producer_name="onnx-example")
    save(model_def, model_path)


def create_conv_model(model_path):
    # Create a simple Conv model: Y = Conv(X, W)
    node_def = helper.make_node("Conv", ["X", "W"], ["Y"], pads=[1, 1, 1, 1], strides=[1, 1], dilations=[1, 1], group=1)
    graph_def = helper.make_graph(
        [node_def],
        "test-model-conv",
        [
            helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 2, 4, 4]),
            helper.make_tensor_value_info("W", TensorProto.FLOAT, [3, 2, 3, 3]),
        ],
        [helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 3, 4, 4])],
    )
    opset = onnx.OperatorSetIdProto()
    opset.version = 11
    model_def = helper.make_model(graph_def, producer_name="onnx-example", opset_imports=[opset])
    save(model_def, model_path)


def test_operator(target_device, model_creator, inputs, expected_fn, ep_name="CudaPluginExecutionProvider"):
    model_path = "temp.onnx"
    try:
        model_creator(model_path)
        sess_options = onnxrt.SessionOptions()
        sess_options.add_provider_for_devices([target_device], {})
        sess = onnxrt.InferenceSession(model_path, sess_options=sess_options)

        active_providers = sess.get_providers()
        if ep_name not in active_providers:
            print(f"FAILURE: {ep_name} is NOT active for this operator. Providers: {active_providers}")
            return False

        print(f"(Session created with {active_providers})", end=" ", flush=True)
        print("Running...", end=" ", flush=True)
        res = sess.run(None, inputs)
        print("Done.", end=" ", flush=True)
        expected = expected_fn(inputs)
        np.testing.assert_allclose(res[0], expected, rtol=1e-3, atol=1e-3)
        return True
    finally:
        if os.path.exists(model_path):
            os.remove(model_path)


def test_cuda_plugin_registration():
    ep_lib_path = os.environ.get("ORT_CUDA_PLUGIN_PATH")
    if not ep_lib_path:
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
        ep_lib_path = os.path.join(base_dir, "build", "cuda", "Release", "libonnxruntime_providers_cuda_plugin.so")

    if not os.path.exists(ep_lib_path):
        print(f"Error: Plugin library not found at: {ep_lib_path}")
        sys.exit(1)

    ep_name = "CudaPluginExecutionProvider"
    print(f"Attempting to register plugin from: {ep_lib_path}", flush=True)

    try:
        onnxrt.register_execution_provider_library(ep_name, ep_lib_path)
        print("Registration successful", flush=True)
    except Exception as e:
        print(f"Registration failed: {e}", flush=True)
        return

    devices = onnxrt.get_ep_devices()
    plugin_devices = [d for d in devices if d.ep_name == ep_name]
    if not plugin_devices:
        print("Error: No plugin devices found!", flush=True)
        sys.exit(1)

    target_device = plugin_devices[0]
    print(f"Using device: {target_device.ep_name}", flush=True)

    # Test Add
    print("Testing Add...", end=" ", flush=True)
    a = np.random.rand(3, 2).astype(np.float32)
    b = np.random.rand(3, 2).astype(np.float32)
    if test_operator(target_device, create_add_model, {"A": a, "B": b}, lambda x: x["A"] + x["B"]):
        print("PASS")
    else:
        print("FAIL")
        sys.exit(1)

    # Test MatMul
    print("Testing MatMul...", end=" ", flush=True)
    a = np.random.rand(3, 4).astype(np.float32)
    b = np.random.rand(4, 5).astype(np.float32)
    if test_operator(target_device, create_matmul_model, {"A": a, "B": b}, lambda x: x["A"] @ x["B"]):
        print("PASS")
    else:
        print("FAIL")
        sys.exit(1)

    # Test Gemm
    print("Testing Gemm...", end=" ", flush=True)
    alpha, beta = 2.0, 0.5
    a = np.random.rand(3, 4).astype(np.float32)
    b = np.random.rand(4, 5).astype(np.float32)
    c = np.random.rand(5).astype(np.float32)
    if test_operator(
        target_device,
        lambda p: create_gemm_model(p, alpha=alpha, beta=beta),
        {"A": a, "B": b, "C": c},
        lambda x: alpha * (x["A"] @ x["B"]) + beta * x["C"],
    ):
        print("PASS")
    else:
        print("FAIL")
        sys.exit(1)

    # Test Conv
    print("Testing Conv...", end=" ", flush=True)

    x = np.random.rand(1, 2, 4, 4).astype(np.float32)
    w = np.random.rand(3, 2, 3, 3).astype(np.float32)

    def expected_conv(inputs):
        return F.conv2d(torch.from_numpy(inputs["X"]), torch.from_numpy(inputs["W"]), padding=1).numpy()

    if test_operator(target_device, create_conv_model, {"X": x, "W": w}, expected_conv):
        print("PASS")
    else:
        print("FAIL")
        sys.exit(1)

    print("\nAll Stage 2 tests finished successfully.", flush=True)


if __name__ == "__main__":
    test_cuda_plugin_registration()
