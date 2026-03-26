# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import os
import tempfile
import unittest

import numpy as np
import onnx
import torch
import torch.nn.functional as F
from cuda_plugin_ep_helper import CUDA_PLUGIN_EP_NAME, _get_default_cuda_plugin_ep_path, should_test_with_cuda_plugin_ep
from onnx import TensorProto, helper, save

import onnxruntime as onnxrt

try:
    import faulthandler

    faulthandler.enable()
except ImportError:
    pass


class _PluginRegistrationState:
    attempted = False
    succeeded = False


TEST_PASS = "PASS"
TEST_SKIP = "SKIP"
TEST_FAIL = "FAIL"


def require_cuda_plugin_ep():
    if not should_test_with_cuda_plugin_ep():
        raise unittest.SkipTest("CUDA plugin EP is not enabled for testing")

    if _PluginRegistrationState.attempted:
        if not _PluginRegistrationState.succeeded:
            raise unittest.SkipTest("CUDA plugin EP is not built or could not be registered")
        return

    _PluginRegistrationState.attempted = True

    ep_lib_path = os.environ.get("ORT_CUDA_PLUGIN_PATH", "")
    if not ep_lib_path:
        detected_path = _get_default_cuda_plugin_ep_path()
        ep_lib_path = detected_path if detected_path else ""

    if not ep_lib_path or not os.path.exists(ep_lib_path):
        raise unittest.SkipTest("CUDA plugin EP is not built or could not be registered")

    try:
        onnxrt.register_execution_provider_library(CUDA_PLUGIN_EP_NAME, ep_lib_path)
        _PluginRegistrationState.succeeded = True
    except Exception:
        providers = {device.ep_name for device in onnxrt.get_ep_devices()}
        if CUDA_PLUGIN_EP_NAME in providers:
            _PluginRegistrationState.succeeded = True

    if not _PluginRegistrationState.succeeded:
        raise unittest.SkipTest("CUDA plugin EP is not built or could not be registered")


def get_cuda_plugin_device():
    require_cuda_plugin_ep()

    devices = onnxrt.get_ep_devices()
    plugin_devices = [device for device in devices if device.ep_name == CUDA_PLUGIN_EP_NAME]
    if not plugin_devices:
        raise unittest.SkipTest("CUDA plugin EP registered, but no plugin devices were enumerated")

    return plugin_devices[0]


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


def create_batch_norm_model(model_path):
    """Create a BatchNormalization model for NHWC testing."""
    num_channels = 3
    node_def = helper.make_node(
        "BatchNormalization",
        ["X", "scale", "B", "input_mean", "input_var"],
        ["Y"],
        epsilon=1e-5,
    )
    # scale, B, mean, var are 1D tensors of shape [num_channels]
    scale_init = helper.make_tensor(
        "scale", TensorProto.FLOAT, [num_channels], np.ones(num_channels, dtype=np.float32).tolist()
    )
    bias_init = helper.make_tensor(
        "B", TensorProto.FLOAT, [num_channels], np.zeros(num_channels, dtype=np.float32).tolist()
    )
    mean_init = helper.make_tensor(
        "input_mean", TensorProto.FLOAT, [num_channels], np.zeros(num_channels, dtype=np.float32).tolist()
    )
    var_init = helper.make_tensor(
        "input_var", TensorProto.FLOAT, [num_channels], np.ones(num_channels, dtype=np.float32).tolist()
    )

    graph_def = helper.make_graph(
        [node_def],
        "test-model-batchnorm",
        [helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, num_channels, 4, 4])],
        [helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, num_channels, 4, 4])],
        initializer=[scale_init, bias_init, mean_init, var_init],
    )
    opset = onnx.OperatorSetIdProto()
    opset.version = 15
    model_def = helper.make_model(graph_def, producer_name="onnx-example", opset_imports=[opset])
    save(model_def, model_path)


def create_maxpool_model(model_path):
    """Create a MaxPool model for NHWC testing."""
    node_def = helper.make_node(
        "MaxPool",
        ["X"],
        ["Y"],
        kernel_shape=[2, 2],
        strides=[2, 2],
    )
    graph_def = helper.make_graph(
        [node_def],
        "test-model-maxpool",
        [helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 3, 4, 4])],
        [helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 3, 2, 2])],
    )
    opset = onnx.OperatorSetIdProto()
    opset.version = 12
    model_def = helper.make_model(graph_def, producer_name="onnx-example", opset_imports=[opset])
    save(model_def, model_path)


def create_avgpool_model(model_path):
    """Create an AveragePool model for NHWC testing."""
    node_def = helper.make_node(
        "AveragePool",
        ["X"],
        ["Y"],
        kernel_shape=[2, 2],
        strides=[2, 2],
    )
    graph_def = helper.make_graph(
        [node_def],
        "test-model-avgpool",
        [helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 3, 4, 4])],
        [helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 3, 2, 2])],
    )
    opset = onnx.OperatorSetIdProto()
    opset.version = 12
    model_def = helper.make_model(graph_def, producer_name="onnx-example", opset_imports=[opset])
    save(model_def, model_path)


def make_bias_dropout_model():
    """Create a deterministic BiasDropout model by forcing inference mode."""
    node = helper.make_node(
        "BiasDropout",
        ["X", "bias", "residual", "ratio", "training_mode"],
        ["Y", "mask"],
        domain="com.microsoft",
    )
    graph = helper.make_graph(
        [node],
        "test-BiasDropout",
        [
            helper.make_tensor_value_info("X", TensorProto.FLOAT, [2, 4]),
            helper.make_tensor_value_info("bias", TensorProto.FLOAT, [4]),
            helper.make_tensor_value_info("residual", TensorProto.FLOAT, [2, 4]),
            helper.make_tensor_value_info("ratio", TensorProto.FLOAT, []),
            helper.make_tensor_value_info("training_mode", TensorProto.BOOL, []),
        ],
        [
            helper.make_tensor_value_info("Y", TensorProto.FLOAT, [2, 4]),
            helper.make_tensor_value_info("mask", TensorProto.BOOL, [2, 4]),
        ],
    )
    opset_onnx = onnx.OperatorSetIdProto()
    opset_onnx.version = 13
    opset_ms = onnx.OperatorSetIdProto()
    opset_ms.domain = "com.microsoft"
    opset_ms.version = 1
    return helper.make_model(graph, opset_imports=[opset_onnx, opset_ms])


def run_operator_test(
    target_device, model_creator, inputs, expected_fn, ep_name=CUDA_PLUGIN_EP_NAME, session_config=None
):
    with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as tmp:
        model_path = tmp.name
    try:
        model_creator(model_path)
        sess_options = onnxrt.SessionOptions()
        if session_config:
            for key, value in session_config.items():
                sess_options.add_session_config_entry(key, value)
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


def run_provider_options_test(provider_options, expect_plugin_provider=True):
    require_cuda_plugin_ep()
    with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as tmp:
        model_path = tmp.name
    try:
        create_add_model(model_path)
        providers = [(CUDA_PLUGIN_EP_NAME, provider_options), "CPUExecutionProvider"]
        sess = onnxrt.InferenceSession(model_path, providers=providers)
        active_providers = sess.get_providers()

        if expect_plugin_provider and CUDA_PLUGIN_EP_NAME not in active_providers:
            print(f"FAILURE: {CUDA_PLUGIN_EP_NAME} is NOT active. Providers: {active_providers}")
            return False
        if not expect_plugin_provider and CUDA_PLUGIN_EP_NAME in active_providers:
            print(f"FAILURE: {CUDA_PLUGIN_EP_NAME} unexpectedly active. Providers: {active_providers}")
            return False

        a = np.random.rand(3, 2).astype(np.float32)
        b = np.random.rand(3, 2).astype(np.float32)
        res = sess.run(None, {"A": a, "B": b})
        np.testing.assert_allclose(res[0], a + b, rtol=1e-3, atol=1e-3)
        return True
    except Exception as e:
        if expect_plugin_provider:
            print(f"FAIL ({e})")
            return False

        print(f"Expected failure for provider options {provider_options}: {e}")
        return True
    finally:
        if os.path.exists(model_path):
            os.remove(model_path)


def _run_registration_checks(test_case: unittest.TestCase):
    target_device = get_cuda_plugin_device()
    print(f"Using registered plugin: {CUDA_PLUGIN_EP_NAME}", flush=True)
    print(f"Using device: {target_device.ep_name}", flush=True)

    x = np.random.rand(1, 2, 4, 4).astype(np.float32)
    w = np.random.rand(3, 2, 3, 3).astype(np.float32)

    def expected_conv(inputs):
        return F.conv2d(torch.from_numpy(inputs["X"]), torch.from_numpy(inputs["W"]), padding=1).numpy()

    stage2_cases = [
        (
            "Add",
            create_add_model,
            {"A": np.random.rand(3, 2).astype(np.float32), "B": np.random.rand(3, 2).astype(np.float32)},
            lambda feed: feed["A"] + feed["B"],
            None,
        ),
        (
            "MatMul",
            create_matmul_model,
            {"A": np.random.rand(3, 4).astype(np.float32), "B": np.random.rand(4, 5).astype(np.float32)},
            lambda feed: feed["A"] @ feed["B"],
            None,
        ),
        (
            "Gemm",
            lambda model_path: create_gemm_model(model_path, alpha=2.0, beta=0.5),
            {
                "A": np.random.rand(3, 4).astype(np.float32),
                "B": np.random.rand(4, 5).astype(np.float32),
                "C": np.random.rand(5).astype(np.float32),
            },
            lambda feed: 2.0 * (feed["A"] @ feed["B"]) + 0.5 * feed["C"],
            None,
        ),
        ("Conv", create_conv_model, {"X": x, "W": w}, expected_conv, None),
    ]

    for name, model_creator, inputs, expected_fn, session_config in stage2_cases:
        print(f"Testing {name}...", end=" ", flush=True)
        result = run_operator_test(target_device, model_creator, inputs, expected_fn, session_config=session_config)
        with test_case.subTest(op=name):
            test_case.assertTrue(
                result,
                f"{name} plugin registration test failed",
            )
        print(TEST_PASS if result else TEST_FAIL, flush=True)

    print("\nAll Stage 2 tests finished successfully.", flush=True)

    nhwc_config = {"ep.cuda.prefer_nhwc_layout": "1"}

    def expected_batchnorm(inputs):
        return inputs["X"] / np.sqrt(1.0 + 1e-5)

    stage3_cases = [
        (
            "Conv (NHWC)",
            create_conv_model,
            {
                "X": np.random.rand(1, 2, 4, 4).astype(np.float32),
                "W": np.random.rand(3, 2, 3, 3).astype(np.float32),
            },
            expected_conv,
        ),
        (
            "BatchNormalization (NHWC)",
            create_batch_norm_model,
            {"X": np.random.rand(1, 3, 4, 4).astype(np.float32)},
            expected_batchnorm,
        ),
        (
            "MaxPool (NHWC)",
            create_maxpool_model,
            {"X": np.random.rand(1, 3, 4, 4).astype(np.float32)},
            lambda feed: F.max_pool2d(torch.from_numpy(feed["X"]), kernel_size=2, stride=2).numpy(),
        ),
        (
            "AveragePool (NHWC)",
            create_avgpool_model,
            {"X": np.random.rand(1, 3, 4, 4).astype(np.float32)},
            lambda feed: F.avg_pool2d(torch.from_numpy(feed["X"]), kernel_size=2, stride=2).numpy(),
        ),
    ]

    for name, model_creator, inputs, expected_fn in stage3_cases:
        print(f"Testing {name}...", end=" ", flush=True)
        result = run_operator_test(target_device, model_creator, inputs, expected_fn, session_config=nhwc_config)
        with test_case.subTest(op=name):
            test_case.assertTrue(
                result,
                f"{name} plugin NHWC test failed",
            )
        print(TEST_PASS if result else TEST_FAIL, flush=True)

    print("\nAll Stage 3 NHWC tests finished successfully.", flush=True)

    provider_option_cases = [
        ("provider options with valid device_id/use_tf32", {"device_id": "0", "use_tf32": "0"}, True),
        ("provider options with invalid device_id", {"device_id": "999"}, False),
    ]

    print("\nTesting provider options path...", flush=True)
    for name, provider_options, expect_plugin_provider in provider_option_cases:
        print(f"Testing {name}...", end=" ", flush=True)
        result = run_provider_options_test(provider_options, expect_plugin_provider=expect_plugin_provider)
        with test_case.subTest(op=name):
            test_case.assertTrue(
                result,
                f"{name} failed",
            )
        print(TEST_PASS if result else TEST_FAIL, flush=True)


def _make_simple_model(op_type, inputs_info, outputs_info, attrs=None, opset=13, domain=""):
    """Helper to create a simple single-node ONNX model.

    Args:
        op_type: ONNX op type string
        inputs_info: list of (name, elem_type, shape) tuples
        outputs_info: list of (name, elem_type, shape) tuples
        attrs: dict of node attributes
        opset: opset version
        domain: op domain (empty string for default ONNX domain)
    """
    input_names = [info[0] for info in inputs_info]
    output_names = [info[0] for info in outputs_info]
    node = helper.make_node(op_type, input_names, output_names, domain=domain, **(attrs or {}))
    graph = helper.make_graph(
        [node],
        f"test-{op_type}",
        [helper.make_tensor_value_info(n, t, s) for n, t, s in inputs_info],
        [helper.make_tensor_value_info(n, t, s) for n, t, s in outputs_info],
    )
    opset_import = [onnx.OperatorSetIdProto()]
    opset_import[0].version = opset
    if domain:
        ms_opset = onnx.OperatorSetIdProto()
        ms_opset.domain = domain
        ms_opset.version = 1
        opset_import.append(ms_opset)
    model = helper.make_model(graph, opset_imports=opset_import)
    return model


def _run_model_test(
    target_device, op_name, model, feed_dict, expected_fn, ep_name=CUDA_PLUGIN_EP_NAME, rtol=1e-3, atol=1e-3
):
    """Run a single op test: save model, create session, run, compare."""
    with tempfile.NamedTemporaryFile(suffix=f"_{op_name}.onnx", delete=False) as tmp:
        model_path = tmp.name
    try:
        save(model, model_path)
        sess_options = onnxrt.SessionOptions()
        sess_options.add_provider_for_devices([target_device], {})
        sess = onnxrt.InferenceSession(model_path, sess_options=sess_options)
        active_providers = sess.get_providers()
        if ep_name not in active_providers:
            print(f"{TEST_SKIP} ({ep_name} not active)")
            return TEST_SKIP
        res = sess.run(None, feed_dict)
        expected = expected_fn(feed_dict)
        if isinstance(expected, (list, tuple)):
            for r, e in zip(res, expected, strict=False):
                np.testing.assert_allclose(r, e, rtol=rtol, atol=atol)
        else:
            np.testing.assert_allclose(res[0], expected, rtol=rtol, atol=atol)
        return TEST_PASS
    except Exception as e:
        print(f"{TEST_FAIL} ({e})")
        return TEST_FAIL
    finally:
        if os.path.exists(model_path):
            os.remove(model_path)


def _run_stage5_checks(test_case: unittest.TestCase):
    """Stage 5: Test all ops enabled during Stage 5 (5A through 5D)."""
    target_device = get_cuda_plugin_device()
    passed = 0
    failed = 0
    skipped = 0

    def run_test(name, model, feed, expected_fn, rtol=1e-3, atol=1e-3):
        nonlocal passed, failed, skipped
        print(f"  {name}...", end=" ", flush=True)
        result = _run_model_test(target_device, name, model, feed, expected_fn, rtol=rtol, atol=atol)
        with test_case.subTest(op=name):
            if result == TEST_PASS:
                passed += 1
                print(TEST_PASS, flush=True)
                return

            if result == TEST_SKIP:
                skipped += 1
                print(TEST_SKIP, flush=True)
                return

            failed += 1
            print(TEST_FAIL, flush=True)
            test_case.fail(f"{name} Stage 5 plugin op test failed")

    print("\n==================== Stage 5: Expanded Op Tests ====================", flush=True)
    f_dtype = TensorProto.FLOAT

    # ---- 5A/5B: Standard ops ----
    print("\n--- Standard Ops (5A/5B) ---", flush=True)

    # Reshape
    model = _make_simple_model(
        "Reshape", [("X", f_dtype, [2, 3, 4]), ("shape", TensorProto.INT64, [2])], [("Y", f_dtype, [6, 4])]
    )
    # Need shape as initializer; build manually
    shape_init = helper.make_tensor("shape", TensorProto.INT64, [2], [6, 4])
    model.graph.initializer.append(shape_init)
    x = np.random.rand(2, 3, 4).astype(np.float32)
    run_test("Reshape", model, {"X": x}, lambda f: f["X"].reshape(6, 4))

    # Split (opset 18 supports num_outputs; use split input for opset 13)
    node = helper.make_node("Split", ["X", "split"], ["Y1", "Y2"], axis=0)
    graph = helper.make_graph(
        [node],
        "test-Split",
        [helper.make_tensor_value_info("X", f_dtype, [6, 4])],
        [helper.make_tensor_value_info("Y1", f_dtype, [3, 4]), helper.make_tensor_value_info("Y2", f_dtype, [3, 4])],
    )
    opset = onnx.OperatorSetIdProto()
    opset.version = 13
    model = helper.make_model(graph, opset_imports=[opset])
    model.graph.initializer.append(helper.make_tensor("split", TensorProto.INT64, [2], [3, 3]))
    x = np.random.rand(6, 4).astype(np.float32)
    run_test("Split", model, {"X": x}, lambda f: [f["X"][:3], f["X"][3:]])

    # Concat
    model = _make_simple_model(
        "Concat", [("A", f_dtype, [2, 3]), ("B", f_dtype, [2, 3])], [("Y", f_dtype, [4, 3])], attrs={"axis": 0}
    )
    a = np.random.rand(2, 3).astype(np.float32)
    b = np.random.rand(2, 3).astype(np.float32)
    run_test("Concat", model, {"A": a, "B": b}, lambda f: np.concatenate([f["A"], f["B"]], axis=0))

    # Gather
    gather_model = _make_simple_model(
        "Gather",
        [("X", f_dtype, [5, 4]), ("indices", TensorProto.INT64, [3])],
        [("Y", f_dtype, [3, 4])],
        attrs={"axis": 0},
        opset=13,
    )
    x = np.random.rand(5, 4).astype(np.float32)
    idx = np.array([0, 2, 4], dtype=np.int64)
    run_test("Gather", gather_model, {"X": x, "indices": idx}, lambda f: f["X"][f["indices"]])

    # Unsqueeze (opset 13 uses axes as input)
    node = helper.make_node("Unsqueeze", ["X", "axes"], ["Y"])
    graph = helper.make_graph(
        [node],
        "test-Unsqueeze",
        [helper.make_tensor_value_info("X", f_dtype, [3, 4])],
        [helper.make_tensor_value_info("Y", f_dtype, [1, 3, 4])],
    )
    opset = onnx.OperatorSetIdProto()
    opset.version = 13
    model = helper.make_model(graph, opset_imports=[opset])
    axes_init = helper.make_tensor("axes", TensorProto.INT64, [1], [0])
    model.graph.initializer.append(axes_init)
    x = np.random.rand(3, 4).astype(np.float32)
    run_test("Unsqueeze", model, {"X": x}, lambda f: np.expand_dims(f["X"], 0))

    # Tile
    node = helper.make_node("Tile", ["X", "repeats"], ["Y"])
    graph = helper.make_graph(
        [node],
        "test-Tile",
        [helper.make_tensor_value_info("X", f_dtype, [2, 3])],
        [helper.make_tensor_value_info("Y", f_dtype, [4, 9])],
    )
    opset = onnx.OperatorSetIdProto()
    opset.version = 13
    model = helper.make_model(graph, opset_imports=[opset])
    repeats_init = helper.make_tensor("repeats", TensorProto.INT64, [2], [2, 3])
    model.graph.initializer.append(repeats_init)
    x = np.random.rand(2, 3).astype(np.float32)
    run_test("Tile", model, {"X": x}, lambda f: np.tile(f["X"], (2, 3)))

    # CumSum
    node = helper.make_node("CumSum", ["X", "axis"], ["Y"])
    graph = helper.make_graph(
        [node],
        "test-CumSum",
        [helper.make_tensor_value_info("X", f_dtype, [3, 4])],
        [helper.make_tensor_value_info("Y", f_dtype, [3, 4])],
    )
    opset = onnx.OperatorSetIdProto()
    opset.version = 14
    model = helper.make_model(graph, opset_imports=[opset])
    axis_init = helper.make_tensor("axis", TensorProto.INT64, [], [1])
    model.graph.initializer.append(axis_init)
    x = np.random.rand(3, 4).astype(np.float32)
    run_test("CumSum", model, {"X": x}, lambda f: np.cumsum(f["X"], axis=1))

    # ConstantOfShape
    node = helper.make_node(
        "ConstantOfShape", ["shape"], ["Y"], value=helper.make_tensor("value", TensorProto.FLOAT, [1], [3.14])
    )
    graph = helper.make_graph(
        [node],
        "test-ConstantOfShape",
        [helper.make_tensor_value_info("shape", TensorProto.INT64, [2])],
        [helper.make_tensor_value_info("Y", f_dtype, None)],
    )
    opset = onnx.OperatorSetIdProto()
    opset.version = 9
    model = helper.make_model(graph, opset_imports=[opset])
    run_test(
        "ConstantOfShape",
        model,
        {"shape": np.array([2, 3], dtype=np.int64)},
        lambda f: np.full((2, 3), 3.14, dtype=np.float32),
    )

    # SpaceToDepth
    model = _make_simple_model(
        "SpaceToDepth", [("X", f_dtype, [1, 2, 4, 4])], [("Y", f_dtype, [1, 8, 2, 2])], attrs={"blocksize": 2}, opset=13
    )
    x = np.random.rand(1, 2, 4, 4).astype(np.float32)

    def space_to_depth(f):
        inp = f["X"]
        b, c, h, w = inp.shape
        bs = 2
        # ONNX SpaceToDepth: rearrange blocks of spatial data into depth
        # (b, c, h, w) -> (b, c, h/bs, bs, w/bs, bs) -> (b, c*bs*bs, h/bs, w/bs)
        tmp = inp.reshape(b, c, h // bs, bs, w // bs, bs)
        tmp = tmp.transpose(0, 3, 5, 1, 2, 4)
        return tmp.reshape(b, c * bs * bs, h // bs, w // bs)

    run_test("SpaceToDepth", model, {"X": x}, space_to_depth)

    # Pad
    node = helper.make_node("Pad", ["X", "pads", "constant_value"], ["Y"])
    graph = helper.make_graph(
        [node],
        "test-Pad",
        [helper.make_tensor_value_info("X", f_dtype, [2, 3])],
        [helper.make_tensor_value_info("Y", f_dtype, [4, 5])],
    )
    opset = onnx.OperatorSetIdProto()
    opset.version = 13
    model = helper.make_model(graph, opset_imports=[opset])
    model.graph.initializer.append(helper.make_tensor("pads", TensorProto.INT64, [4], [1, 1, 1, 1]))
    model.graph.initializer.append(helper.make_tensor("constant_value", TensorProto.FLOAT, [], [0.0]))
    x = np.random.rand(2, 3).astype(np.float32)
    run_test("Pad", model, {"X": x}, lambda f: np.pad(f["X"], ((1, 1), (1, 1)), constant_values=0))

    # Slice
    node = helper.make_node("Slice", ["X", "starts", "ends", "axes"], ["Y"])
    graph = helper.make_graph(
        [node],
        "test-Slice",
        [helper.make_tensor_value_info("X", f_dtype, [4, 6])],
        [helper.make_tensor_value_info("Y", f_dtype, [2, 4])],
    )
    opset = onnx.OperatorSetIdProto()
    opset.version = 13
    model = helper.make_model(graph, opset_imports=[opset])
    model.graph.initializer.append(helper.make_tensor("starts", TensorProto.INT64, [2], [1, 1]))
    model.graph.initializer.append(helper.make_tensor("ends", TensorProto.INT64, [2], [3, 5]))
    model.graph.initializer.append(helper.make_tensor("axes", TensorProto.INT64, [2], [0, 1]))
    x = np.random.rand(4, 6).astype(np.float32)
    run_test("Slice", model, {"X": x}, lambda f: f["X"][1:3, 1:5])

    # Resize (nearest)
    node = helper.make_node("Resize", ["X", "", "scales"], ["Y"], mode="nearest")
    graph = helper.make_graph(
        [node],
        "test-Resize",
        [helper.make_tensor_value_info("X", f_dtype, [1, 1, 2, 2])],
        [helper.make_tensor_value_info("Y", f_dtype, [1, 1, 4, 4])],
    )
    opset = onnx.OperatorSetIdProto()
    opset.version = 13
    model = helper.make_model(graph, opset_imports=[opset])
    model.graph.initializer.append(helper.make_tensor("scales", TensorProto.FLOAT, [4], [1.0, 1.0, 2.0, 2.0]))
    x = np.random.rand(1, 1, 2, 2).astype(np.float32)
    run_test("Resize", model, {"X": x}, lambda f: np.repeat(np.repeat(f["X"], 2, axis=2), 2, axis=3))

    # Sum (variadic)
    model = _make_simple_model(
        "Sum",
        [("A", f_dtype, [3, 4]), ("B", f_dtype, [3, 4]), ("C", f_dtype, [3, 4])],
        [("Y", f_dtype, [3, 4])],
        opset=13,
    )
    a = np.random.rand(3, 4).astype(np.float32)
    b = np.random.rand(3, 4).astype(np.float32)
    c = np.random.rand(3, 4).astype(np.float32)
    run_test("Sum_variadic", model, {"A": a, "B": b, "C": c}, lambda f: f["A"] + f["B"] + f["C"])

    # ---- 5C: CPU base class ops ----
    print("\n--- CPU Base Class Ops (5C) ---", flush=True)

    # Upsample (deprecated but still present)
    node = helper.make_node("Upsample", ["X", "scales"], ["Y"], mode="nearest")
    graph = helper.make_graph(
        [node],
        "test-Upsample",
        [helper.make_tensor_value_info("X", f_dtype, [1, 1, 2, 2])],
        [helper.make_tensor_value_info("Y", f_dtype, [1, 1, 4, 4])],
    )
    opset = onnx.OperatorSetIdProto()
    opset.version = 9
    model = helper.make_model(graph, opset_imports=[opset])
    model.graph.initializer.append(helper.make_tensor("scales", TensorProto.FLOAT, [4], [1.0, 1.0, 2.0, 2.0]))
    x = np.random.rand(1, 1, 2, 2).astype(np.float32)
    run_test("Upsample", model, {"X": x}, lambda f: np.repeat(np.repeat(f["X"], 2, axis=2), 2, axis=3))

    # DepthToSpace
    model = _make_simple_model(
        "DepthToSpace",
        [("X", f_dtype, [1, 8, 2, 2])],
        [("Y", f_dtype, [1, 2, 4, 4])],
        attrs={"blocksize": 2, "mode": "DCR"},
        opset=13,
    )
    x = np.random.rand(1, 8, 2, 2).astype(np.float32)

    def depth_to_space_dcr(f):
        inp = f["X"]
        b, c, h, w = inp.shape
        bs = 2
        return (
            inp.reshape(b, bs, bs, c // (bs * bs), h, w)
            .transpose(0, 3, 4, 1, 5, 2)
            .reshape(b, c // (bs * bs), h * bs, w * bs)
        )

    run_test("DepthToSpace", model, {"X": x}, depth_to_space_dcr)

    # ---- 5D: Contrib Ops ----
    print("\n--- Contrib Ops (5D) ---", flush=True)

    # FastGelu (com.microsoft domain)
    node = helper.make_node("FastGelu", ["X"], ["Y"], domain="com.microsoft")
    graph = helper.make_graph(
        [node],
        "test-FastGelu",
        [helper.make_tensor_value_info("X", f_dtype, [2, 4])],
        [helper.make_tensor_value_info("Y", f_dtype, [2, 4])],
    )
    opset_onnx = onnx.OperatorSetIdProto()
    opset_onnx.version = 13
    opset_ms = onnx.OperatorSetIdProto()
    opset_ms.domain = "com.microsoft"
    opset_ms.version = 1
    model = helper.make_model(graph, opset_imports=[opset_onnx, opset_ms])
    x = np.random.rand(2, 4).astype(np.float32)

    def fast_gelu_ref(f):
        x = f["X"]
        # FastGelu approximation: x * sigmoid(1.702 * x)
        return x * (1.0 / (1.0 + np.exp(-1.702 * x)))

    run_test("FastGelu", model, {"X": x}, fast_gelu_ref, rtol=1e-2, atol=1e-2)

    # BiasDropout (com.microsoft). We force inference mode so the op is deterministic.
    model = make_bias_dropout_model()
    x = np.random.rand(2, 4).astype(np.float32)
    bias = np.random.rand(4).astype(np.float32)
    residual = np.random.rand(2, 4).astype(np.float32)
    ratio = np.array(0.5, dtype=np.float32)
    training_mode = np.array(False, dtype=np.bool_)
    run_test(
        "BiasDropout",
        model,
        {
            "X": x,
            "bias": bias,
            "residual": residual,
            "ratio": ratio,
            "training_mode": training_mode,
        },
        lambda feed: feed["X"] + feed["bias"] + feed["residual"],
    )

    # SkipLayerNormalization (com.microsoft)
    hidden_size = 8
    node = helper.make_node(
        "SkipLayerNormalization",
        ["X", "skip", "gamma", "beta"],
        ["Y", "mean", "inv_std_var", "input_skip_bias_sum"],
        domain="com.microsoft",
        epsilon=1e-5,
    )
    graph = helper.make_graph(
        [node],
        "test-SkipLayerNorm",
        [
            helper.make_tensor_value_info("X", f_dtype, [2, hidden_size]),
            helper.make_tensor_value_info("skip", f_dtype, [2, hidden_size]),
            helper.make_tensor_value_info("gamma", f_dtype, [hidden_size]),
            helper.make_tensor_value_info("beta", f_dtype, [hidden_size]),
        ],
        [
            helper.make_tensor_value_info("Y", f_dtype, [2, hidden_size]),
            helper.make_tensor_value_info("mean", f_dtype, None),
            helper.make_tensor_value_info("inv_std_var", f_dtype, None),
            helper.make_tensor_value_info("input_skip_bias_sum", f_dtype, None),
        ],
    )
    opset_onnx = onnx.OperatorSetIdProto()
    opset_onnx.version = 13
    opset_ms = onnx.OperatorSetIdProto()
    opset_ms.domain = "com.microsoft"
    opset_ms.version = 1
    model = helper.make_model(graph, opset_imports=[opset_onnx, opset_ms])
    x = np.random.rand(2, hidden_size).astype(np.float32)
    skip = np.random.rand(2, hidden_size).astype(np.float32)
    gamma = np.ones(hidden_size, dtype=np.float32)
    beta = np.zeros(hidden_size, dtype=np.float32)

    def skip_layer_norm_ref(f):
        added = f["X"] + f["skip"]
        mean = added.mean(axis=-1, keepdims=True)
        var = added.var(axis=-1, keepdims=True)
        normed = (added - mean) / np.sqrt(var + 1e-5)
        return normed * f["gamma"] + f["beta"]

    run_test(
        "SkipLayerNorm",
        model,
        {"X": x, "skip": skip, "gamma": gamma, "beta": beta},
        skip_layer_norm_ref,
        rtol=1e-2,
        atol=1e-2,
    )

    # ---- Summary ----
    total = passed + failed + skipped
    print(f"\n--- Stage 5 Results: {passed} passed, {failed} failed, {skipped} skipped ({total} total) ---", flush=True)
    test_case.assertEqual(failed, 0, f"Stage 5 had {failed} failing plugin op checks")
    print("All Stage 5 tests finished successfully.", flush=True)


class TestCudaPluginEP(unittest.TestCase):
    def test_cuda_plugin_registration(self):
        _run_registration_checks(self)

    def test_cuda_plugin_stage5_ops(self):
        _run_stage5_checks(self)


if __name__ == "__main__":
    unittest.main()
