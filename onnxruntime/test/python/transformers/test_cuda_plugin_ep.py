# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import os
import tempfile
import unittest

import numpy as np
import torch
import torch.nn.functional as F
from cuda_plugin_ep_helper import CUDA_PLUGIN_EP_NAME, ensure_cuda_plugin_ep_registered, should_test_with_cuda_plugin_ep
from onnx import OperatorSetIdProto, TensorProto, helper, save

import onnxruntime as onnxrt

try:
    import faulthandler

    faulthandler.enable()
except ImportError:
    # faulthandler is optional in some Python runtimes used by CI.
    pass


TEST_PASS = "PASS"
TEST_SKIP = "SKIP"
TEST_FAIL = "FAIL"
EP_GRAPH_ASSIGNMENT_CONFIG_KEY = "session.record_ep_graph_assignment_info"


def require_cuda_plugin_ep():
    if not should_test_with_cuda_plugin_ep():
        raise unittest.SkipTest("CUDA plugin EP is not enabled for testing")

    if not ensure_cuda_plugin_ep_registered():
        raise unittest.SkipTest("CUDA plugin EP is not built or could not be registered")


def get_cuda_plugin_device():
    return get_cuda_plugin_devices()[0]


def get_cuda_plugin_devices():
    require_cuda_plugin_ep()

    try:
        devices = onnxrt.get_ep_devices()
    except Exception as exc:
        raise unittest.SkipTest(f"Failed to enumerate CUDA plugin EP devices: {exc}") from exc

    plugin_devices = [device for device in devices if device.ep_name == CUDA_PLUGIN_EP_NAME]
    if not plugin_devices:
        raise unittest.SkipTest("CUDA plugin EP registered, but no plugin devices were enumerated")

    return plugin_devices


def get_cuda_plugin_device_by_id(device_id: int):
    expected_device_id = str(device_id)
    for device in get_cuda_plugin_devices():
        if device.ep_options.get("device_id") == expected_device_id:
            return device
        if device.ep_metadata.get("cuda_device_id") == expected_device_id:
            return device

    raise unittest.SkipTest(f"CUDA plugin EP device_id={device_id} is not available in this environment")


def _create_session_options(session_config=None):
    sess_options = onnxrt.SessionOptions()
    if session_config:
        for key, value in session_config.items():
            sess_options.add_session_config_entry(key, value)

    # Require graph-assignment data so the tests validate that nodes actually run on the plugin.
    sess_options.add_session_config_entry(EP_GRAPH_ASSIGNMENT_CONFIG_KEY, "1")
    return sess_options


def _format_assigned_node(node):
    domain = node.domain or "ai.onnx"
    if node.name:
        return f"{domain}::{node.op_type}:{node.name}"
    return f"{domain}::{node.op_type}"


def _get_assigned_nodes(session, ep_name):
    assignment_info = list(session.get_provider_graph_assignment_info())
    assigned_nodes = []
    for subgraph in assignment_info:
        if subgraph.ep_name == ep_name:
            assigned_nodes.extend(subgraph.get_nodes())

    return assigned_nodes, assignment_info


def _format_assignment_summary(assignment_info):
    if not assignment_info:
        return "<none>"

    summaries = []
    for subgraph in assignment_info:
        node_summary = ", ".join(_format_assigned_node(node) for node in subgraph.get_nodes()) or "<no nodes>"
        summaries.append(f"{subgraph.ep_name}[{node_summary}]")

    return "; ".join(summaries)


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
    opset = OperatorSetIdProto()
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
    opset = OperatorSetIdProto()
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
    opset = OperatorSetIdProto()
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
    opset = OperatorSetIdProto()
    opset.version = 12
    model_def = helper.make_model(graph_def, producer_name="onnx-example", opset_imports=[opset])
    save(model_def, model_path)


def make_bias_dropout_model():
    """Create a deterministic BiasDropout model by forcing inference mode."""
    node = helper.make_node(
        "BiasDropout",
        ["X", "bias", "residual", "ratio", "training_mode"],
        ["Y", ""],
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
        [helper.make_tensor_value_info("Y", TensorProto.FLOAT, [2, 4])],
    )
    opset_onnx = OperatorSetIdProto()
    opset_onnx.version = 13
    opset_ms = OperatorSetIdProto()
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
        sess_options = _create_session_options(session_config)
        sess_options.add_provider_for_devices([target_device], {})
        sess = onnxrt.InferenceSession(model_path, sess_options=sess_options)

        active_providers = sess.get_providers()
        assigned_nodes, assignment_info = _get_assigned_nodes(sess, ep_name)
        if not assigned_nodes:
            print(
                f"FAILURE: {ep_name} was assigned no nodes. Providers: {active_providers}. "
                f"Assignments: {_format_assignment_summary(assignment_info)}"
            )
            return False

        print(
            f"(Session created with {active_providers}; assigned nodes: "
            f"{', '.join(_format_assigned_node(node) for node in assigned_nodes)})",
            flush=True,
        )
        res = sess.run(None, inputs)
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
        sess = onnxrt.InferenceSession(model_path, sess_options=_create_session_options(), providers=providers)
        active_providers = sess.get_providers()
        assigned_nodes, assignment_info = _get_assigned_nodes(sess, CUDA_PLUGIN_EP_NAME)

        if expect_plugin_provider and not assigned_nodes:
            print(
                f"FAILURE: {CUDA_PLUGIN_EP_NAME} was assigned no nodes. Providers: {active_providers}. "
                f"Assignments: {_format_assignment_summary(assignment_info)}"
            )
            return False
        if not expect_plugin_provider and assigned_nodes:
            print(
                f"FAILURE: {CUDA_PLUGIN_EP_NAME} unexpectedly owned nodes. "
                f"Assignments: {_format_assignment_summary(assignment_info)}"
            )
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


def _expected_conv(inputs):
    return F.conv2d(torch.from_numpy(inputs["X"]), torch.from_numpy(inputs["W"]), padding=1).numpy()


_NHWC_CONFIG = {"ep.cuda.prefer_nhwc_layout": "1"}


def _expected_batchnorm(inputs):
    return inputs["X"] / np.sqrt(1.0 + 1e-5)


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
    opset_import = [OperatorSetIdProto()]
    opset_import[0].version = opset
    if domain:
        ms_opset = OperatorSetIdProto()
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
        sess_options = _create_session_options()
        sess_options.add_provider_for_devices([target_device], {})
        sess = onnxrt.InferenceSession(model_path, sess_options=sess_options)
        active_providers = sess.get_providers()
        assigned_nodes, assignment_info = _get_assigned_nodes(sess, ep_name)
        if not assigned_nodes:
            print(
                f"{TEST_FAIL} ({ep_name} was assigned no nodes; providers={active_providers}; "
                f"assignments={_format_assignment_summary(assignment_info)})"
            )
            return TEST_FAIL
        res = sess.run(None, feed_dict)
        expected = expected_fn(feed_dict)
        if isinstance(expected, (list, tuple)):
            if len(res) != len(expected):
                raise AssertionError(f"{op_name} produced {len(res)} outputs, expected {len(expected)}")

            for r, e in zip(res, expected, strict=True):
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


class TestCudaPluginEP(unittest.TestCase):
    # ---- Registration tests (verify nodes run on the plugin EP) ----

    def test_registration_add(self):
        target_device = get_cuda_plugin_device()
        inputs = {"A": np.random.rand(3, 2).astype(np.float32), "B": np.random.rand(3, 2).astype(np.float32)}
        result = run_operator_test(target_device, create_add_model, inputs, lambda feed: feed["A"] + feed["B"])
        self.assertTrue(result, "Add plugin registration test failed")

    def test_registration_matmul(self):
        target_device = get_cuda_plugin_device()
        inputs = {"A": np.random.rand(3, 4).astype(np.float32), "B": np.random.rand(4, 5).astype(np.float32)}
        result = run_operator_test(target_device, create_matmul_model, inputs, lambda feed: feed["A"] @ feed["B"])
        self.assertTrue(result, "MatMul plugin registration test failed")

    def test_registration_gemm(self):
        target_device = get_cuda_plugin_device()
        inputs = {
            "A": np.random.rand(3, 4).astype(np.float32),
            "B": np.random.rand(4, 5).astype(np.float32),
            "C": np.random.rand(5).astype(np.float32),
        }
        result = run_operator_test(
            target_device,
            lambda model_path: create_gemm_model(model_path, alpha=2.0, beta=0.5),
            inputs,
            lambda feed: 2.0 * (feed["A"] @ feed["B"]) + 0.5 * feed["C"],
        )
        self.assertTrue(result, "Gemm plugin registration test failed")

    def test_registration_conv(self):
        target_device = get_cuda_plugin_device()
        inputs = {
            "X": np.random.rand(1, 2, 4, 4).astype(np.float32),
            "W": np.random.rand(3, 2, 3, 3).astype(np.float32),
        }
        result = run_operator_test(target_device, create_conv_model, inputs, _expected_conv)
        self.assertTrue(result, "Conv plugin registration test failed")

    # ---- Provider options tests ----

    def test_provider_options_valid(self):
        result = run_provider_options_test({"device_id": "0", "use_tf32": "0"}, expect_plugin_provider=True)
        self.assertTrue(result, "Provider options with valid device_id/use_tf32 failed")

    def test_provider_options_invalid_device(self):
        result = run_provider_options_test({"device_id": "999"}, expect_plugin_provider=False)
        self.assertTrue(result, "Provider options with invalid device_id failed")

    def test_provider_options_second_device(self):
        plugin_devices = get_cuda_plugin_devices()
        if len(plugin_devices) < 2:
            self.skipTest("Multi-GPU CUDA plugin EP test requires at least two plugin devices")

        target_device = get_cuda_plugin_device_by_id(1)

        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as tmp:
            model_path = tmp.name
        try:
            create_add_model(model_path)
            providers = [(CUDA_PLUGIN_EP_NAME, {"device_id": "1"}), "CPUExecutionProvider"]
            sess = onnxrt.InferenceSession(model_path, sess_options=_create_session_options(), providers=providers)

            active_providers = sess.get_providers()
            assigned_nodes, assignment_info = _get_assigned_nodes(sess, CUDA_PLUGIN_EP_NAME)
            self.assertTrue(
                assigned_nodes,
                f"{CUDA_PLUGIN_EP_NAME} was assigned no nodes. Providers: {active_providers}. "
                f"Assignments: {_format_assignment_summary(assignment_info)}",
            )

            provider_options = sess.get_provider_options()
            self.assertEqual(
                provider_options[CUDA_PLUGIN_EP_NAME].get("device_id"),
                "1",
                f"Expected provider option device_id=1, got {provider_options[CUDA_PLUGIN_EP_NAME]}",
            )
            self.assertEqual(target_device.ep_options.get("device_id"), "1")

            a = np.random.rand(3, 2).astype(np.float32)
            b = np.random.rand(3, 2).astype(np.float32)
            res = sess.run(None, {"A": a, "B": b})
            np.testing.assert_allclose(res[0], a + b, rtol=1e-3, atol=1e-3)
        finally:
            if os.path.exists(model_path):
                os.remove(model_path)

    # ---- NHWC layout tests ----

    def test_nhwc_conv(self):
        target_device = get_cuda_plugin_device()
        inputs = {
            "X": np.random.rand(1, 2, 4, 4).astype(np.float32),
            "W": np.random.rand(3, 2, 3, 3).astype(np.float32),
        }
        result = run_operator_test(
            target_device, create_conv_model, inputs, _expected_conv, session_config=_NHWC_CONFIG
        )
        self.assertTrue(result, "Conv (NHWC) plugin test failed")

    def test_nhwc_batch_normalization(self):
        target_device = get_cuda_plugin_device()
        inputs = {"X": np.random.rand(1, 3, 4, 4).astype(np.float32)}
        result = run_operator_test(
            target_device, create_batch_norm_model, inputs, _expected_batchnorm, session_config=_NHWC_CONFIG
        )
        self.assertTrue(result, "BatchNormalization (NHWC) plugin test failed")

    def test_nhwc_maxpool(self):
        target_device = get_cuda_plugin_device()
        inputs = {"X": np.random.rand(1, 3, 4, 4).astype(np.float32)}
        result = run_operator_test(
            target_device,
            create_maxpool_model,
            inputs,
            lambda feed: F.max_pool2d(torch.from_numpy(feed["X"]), kernel_size=2, stride=2).numpy(),
            session_config=_NHWC_CONFIG,
        )
        self.assertTrue(result, "MaxPool (NHWC) plugin test failed")

    def test_nhwc_avgpool(self):
        target_device = get_cuda_plugin_device()
        inputs = {"X": np.random.rand(1, 3, 4, 4).astype(np.float32)}
        result = run_operator_test(
            target_device,
            create_avgpool_model,
            inputs,
            lambda feed: F.avg_pool2d(torch.from_numpy(feed["X"]), kernel_size=2, stride=2).numpy(),
            session_config=_NHWC_CONFIG,
        )
        self.assertTrue(result, "AveragePool (NHWC) plugin test failed")

    # ---- Standard op tests ----

    def test_op_reshape(self):
        target_device = get_cuda_plugin_device()
        f_dtype = TensorProto.FLOAT
        model = _make_simple_model(
            "Reshape", [("X", f_dtype, [2, 3, 4]), ("shape", TensorProto.INT64, [2])], [("Y", f_dtype, [6, 4])]
        )
        model.graph.initializer.append(helper.make_tensor("shape", TensorProto.INT64, [2], [6, 4]))
        x = np.random.rand(2, 3, 4).astype(np.float32)
        result = _run_model_test(target_device, "Reshape", model, {"X": x}, lambda f: f["X"].reshape(6, 4))
        self.assertEqual(result, TEST_PASS, "Reshape plugin op test failed")

    def test_op_split(self):
        target_device = get_cuda_plugin_device()
        f_dtype = TensorProto.FLOAT
        node = helper.make_node("Split", ["X", "split"], ["Y1", "Y2"], axis=0)
        graph = helper.make_graph(
            [node],
            "test-Split",
            [helper.make_tensor_value_info("X", f_dtype, [6, 4])],
            [
                helper.make_tensor_value_info("Y1", f_dtype, [3, 4]),
                helper.make_tensor_value_info("Y2", f_dtype, [3, 4]),
            ],
        )
        opset = OperatorSetIdProto()
        opset.version = 13
        model = helper.make_model(graph, opset_imports=[opset])
        model.graph.initializer.append(helper.make_tensor("split", TensorProto.INT64, [2], [3, 3]))
        x = np.random.rand(6, 4).astype(np.float32)
        result = _run_model_test(target_device, "Split", model, {"X": x}, lambda f: [f["X"][:3], f["X"][3:]])
        self.assertEqual(result, TEST_PASS, "Split plugin op test failed")

    def test_op_concat(self):
        target_device = get_cuda_plugin_device()
        f_dtype = TensorProto.FLOAT
        model = _make_simple_model(
            "Concat", [("A", f_dtype, [2, 3]), ("B", f_dtype, [2, 3])], [("Y", f_dtype, [4, 3])], attrs={"axis": 0}
        )
        a = np.random.rand(2, 3).astype(np.float32)
        b = np.random.rand(2, 3).astype(np.float32)
        result = _run_model_test(
            target_device, "Concat", model, {"A": a, "B": b}, lambda f: np.concatenate([f["A"], f["B"]], axis=0)
        )
        self.assertEqual(result, TEST_PASS, "Concat plugin op test failed")

    def test_op_gather(self):
        target_device = get_cuda_plugin_device()
        f_dtype = TensorProto.FLOAT
        model = _make_simple_model(
            "Gather",
            [("X", f_dtype, [5, 4]), ("indices", TensorProto.INT64, [3])],
            [("Y", f_dtype, [3, 4])],
            attrs={"axis": 0},
            opset=13,
        )
        x = np.random.rand(5, 4).astype(np.float32)
        idx = np.array([0, 2, 4], dtype=np.int64)
        result = _run_model_test(
            target_device, "Gather", model, {"X": x, "indices": idx}, lambda f: f["X"][f["indices"]]
        )
        self.assertEqual(result, TEST_PASS, "Gather plugin op test failed")

    def test_op_unsqueeze(self):
        target_device = get_cuda_plugin_device()
        f_dtype = TensorProto.FLOAT
        node = helper.make_node("Unsqueeze", ["X", "axes"], ["Y"])
        graph = helper.make_graph(
            [node],
            "test-Unsqueeze",
            [helper.make_tensor_value_info("X", f_dtype, [3, 4])],
            [helper.make_tensor_value_info("Y", f_dtype, [1, 3, 4])],
        )
        opset = OperatorSetIdProto()
        opset.version = 13
        model = helper.make_model(graph, opset_imports=[opset])
        model.graph.initializer.append(helper.make_tensor("axes", TensorProto.INT64, [1], [0]))
        x = np.random.rand(3, 4).astype(np.float32)
        result = _run_model_test(target_device, "Unsqueeze", model, {"X": x}, lambda f: np.expand_dims(f["X"], 0))
        self.assertEqual(result, TEST_PASS, "Unsqueeze plugin op test failed")

    def test_op_tile(self):
        target_device = get_cuda_plugin_device()
        f_dtype = TensorProto.FLOAT
        node = helper.make_node("Tile", ["X", "repeats"], ["Y"])
        graph = helper.make_graph(
            [node],
            "test-Tile",
            [helper.make_tensor_value_info("X", f_dtype, [2, 3])],
            [helper.make_tensor_value_info("Y", f_dtype, [4, 9])],
        )
        opset = OperatorSetIdProto()
        opset.version = 13
        model = helper.make_model(graph, opset_imports=[opset])
        model.graph.initializer.append(helper.make_tensor("repeats", TensorProto.INT64, [2], [2, 3]))
        x = np.random.rand(2, 3).astype(np.float32)
        result = _run_model_test(target_device, "Tile", model, {"X": x}, lambda f: np.tile(f["X"], (2, 3)))
        self.assertEqual(result, TEST_PASS, "Tile plugin op test failed")

    def test_op_cumsum(self):
        target_device = get_cuda_plugin_device()
        f_dtype = TensorProto.FLOAT
        node = helper.make_node("CumSum", ["X", "axis"], ["Y"])
        graph = helper.make_graph(
            [node],
            "test-CumSum",
            [helper.make_tensor_value_info("X", f_dtype, [3, 4])],
            [helper.make_tensor_value_info("Y", f_dtype, [3, 4])],
        )
        opset = OperatorSetIdProto()
        opset.version = 14
        model = helper.make_model(graph, opset_imports=[opset])
        model.graph.initializer.append(helper.make_tensor("axis", TensorProto.INT64, [], [1]))
        x = np.random.rand(3, 4).astype(np.float32)
        result = _run_model_test(target_device, "CumSum", model, {"X": x}, lambda f: np.cumsum(f["X"], axis=1))
        self.assertEqual(result, TEST_PASS, "CumSum plugin op test failed")

    def test_op_constant_of_shape(self):
        target_device = get_cuda_plugin_device()
        node = helper.make_node(
            "ConstantOfShape", ["shape"], ["Y"], value=helper.make_tensor("value", TensorProto.FLOAT, [1], [3.14])
        )
        graph = helper.make_graph(
            [node],
            "test-ConstantOfShape",
            [helper.make_tensor_value_info("shape", TensorProto.INT64, [2])],
            [helper.make_tensor_value_info("Y", TensorProto.FLOAT, None)],
        )
        opset = OperatorSetIdProto()
        opset.version = 9
        model = helper.make_model(graph, opset_imports=[opset])
        result = _run_model_test(
            target_device,
            "ConstantOfShape",
            model,
            {"shape": np.array([2, 3], dtype=np.int64)},
            lambda f: np.full((2, 3), 3.14, dtype=np.float32),
        )
        self.assertEqual(result, TEST_PASS, "ConstantOfShape plugin op test failed")

    def test_op_space_to_depth(self):
        target_device = get_cuda_plugin_device()
        f_dtype = TensorProto.FLOAT
        model = _make_simple_model(
            "SpaceToDepth",
            [("X", f_dtype, [1, 2, 4, 4])],
            [("Y", f_dtype, [1, 8, 2, 2])],
            attrs={"blocksize": 2},
            opset=13,
        )
        x = np.random.rand(1, 2, 4, 4).astype(np.float32)

        def expected(f):
            inp = f["X"]
            b, c, h, w = inp.shape
            bs = 2
            tmp = inp.reshape(b, c, h // bs, bs, w // bs, bs)
            tmp = tmp.transpose(0, 3, 5, 1, 2, 4)
            return tmp.reshape(b, c * bs * bs, h // bs, w // bs)

        result = _run_model_test(target_device, "SpaceToDepth", model, {"X": x}, expected)
        self.assertEqual(result, TEST_PASS, "SpaceToDepth plugin op test failed")

    def test_op_pad(self):
        target_device = get_cuda_plugin_device()
        f_dtype = TensorProto.FLOAT
        node = helper.make_node("Pad", ["X", "pads", "constant_value"], ["Y"])
        graph = helper.make_graph(
            [node],
            "test-Pad",
            [helper.make_tensor_value_info("X", f_dtype, [2, 3])],
            [helper.make_tensor_value_info("Y", f_dtype, [4, 5])],
        )
        opset = OperatorSetIdProto()
        opset.version = 13
        model = helper.make_model(graph, opset_imports=[opset])
        model.graph.initializer.append(helper.make_tensor("pads", TensorProto.INT64, [4], [1, 1, 1, 1]))
        model.graph.initializer.append(helper.make_tensor("constant_value", TensorProto.FLOAT, [], [0.0]))
        x = np.random.rand(2, 3).astype(np.float32)
        result = _run_model_test(
            target_device, "Pad", model, {"X": x}, lambda f: np.pad(f["X"], ((1, 1), (1, 1)), constant_values=0)
        )
        self.assertEqual(result, TEST_PASS, "Pad plugin op test failed")

    def test_op_slice(self):
        target_device = get_cuda_plugin_device()
        f_dtype = TensorProto.FLOAT
        node = helper.make_node("Slice", ["X", "starts", "ends", "axes"], ["Y"])
        graph = helper.make_graph(
            [node],
            "test-Slice",
            [helper.make_tensor_value_info("X", f_dtype, [4, 6])],
            [helper.make_tensor_value_info("Y", f_dtype, [2, 4])],
        )
        opset = OperatorSetIdProto()
        opset.version = 13
        model = helper.make_model(graph, opset_imports=[opset])
        model.graph.initializer.append(helper.make_tensor("starts", TensorProto.INT64, [2], [1, 1]))
        model.graph.initializer.append(helper.make_tensor("ends", TensorProto.INT64, [2], [3, 5]))
        model.graph.initializer.append(helper.make_tensor("axes", TensorProto.INT64, [2], [0, 1]))
        x = np.random.rand(4, 6).astype(np.float32)
        result = _run_model_test(target_device, "Slice", model, {"X": x}, lambda f: f["X"][1:3, 1:5])
        self.assertEqual(result, TEST_PASS, "Slice plugin op test failed")

    def test_op_resize(self):
        target_device = get_cuda_plugin_device()
        f_dtype = TensorProto.FLOAT
        node = helper.make_node("Resize", ["X", "", "scales"], ["Y"], mode="nearest")
        graph = helper.make_graph(
            [node],
            "test-Resize",
            [helper.make_tensor_value_info("X", f_dtype, [1, 1, 2, 2])],
            [helper.make_tensor_value_info("Y", f_dtype, [1, 1, 4, 4])],
        )
        opset = OperatorSetIdProto()
        opset.version = 13
        model = helper.make_model(graph, opset_imports=[opset])
        model.graph.initializer.append(helper.make_tensor("scales", TensorProto.FLOAT, [4], [1.0, 1.0, 2.0, 2.0]))
        x = np.random.rand(1, 1, 2, 2).astype(np.float32)
        result = _run_model_test(
            target_device, "Resize", model, {"X": x}, lambda f: np.repeat(np.repeat(f["X"], 2, axis=2), 2, axis=3)
        )
        self.assertEqual(result, TEST_PASS, "Resize plugin op test failed")

    def test_op_sum_variadic(self):
        target_device = get_cuda_plugin_device()
        f_dtype = TensorProto.FLOAT
        model = _make_simple_model(
            "Sum",
            [("A", f_dtype, [3, 4]), ("B", f_dtype, [3, 4]), ("C", f_dtype, [3, 4])],
            [("Y", f_dtype, [3, 4])],
            opset=13,
        )
        a = np.random.rand(3, 4).astype(np.float32)
        b = np.random.rand(3, 4).astype(np.float32)
        c = np.random.rand(3, 4).astype(np.float32)
        result = _run_model_test(
            target_device, "Sum_variadic", model, {"A": a, "B": b, "C": c}, lambda f: f["A"] + f["B"] + f["C"]
        )
        self.assertEqual(result, TEST_PASS, "Sum_variadic plugin op test failed")

    # ---- CPU base class op tests ----

    def test_op_upsample(self):
        target_device = get_cuda_plugin_device()
        f_dtype = TensorProto.FLOAT
        node = helper.make_node("Upsample", ["X", "scales"], ["Y"], mode="nearest")
        graph = helper.make_graph(
            [node],
            "test-Upsample",
            [helper.make_tensor_value_info("X", f_dtype, [1, 1, 2, 2])],
            [helper.make_tensor_value_info("Y", f_dtype, [1, 1, 4, 4])],
        )
        opset = OperatorSetIdProto()
        opset.version = 9
        model = helper.make_model(graph, opset_imports=[opset])
        model.graph.initializer.append(helper.make_tensor("scales", TensorProto.FLOAT, [4], [1.0, 1.0, 2.0, 2.0]))
        x = np.random.rand(1, 1, 2, 2).astype(np.float32)
        result = _run_model_test(
            target_device, "Upsample", model, {"X": x}, lambda f: np.repeat(np.repeat(f["X"], 2, axis=2), 2, axis=3)
        )
        self.assertEqual(result, TEST_PASS, "Upsample plugin op test failed")

    def test_op_depth_to_space(self):
        target_device = get_cuda_plugin_device()
        f_dtype = TensorProto.FLOAT
        model = _make_simple_model(
            "DepthToSpace",
            [("X", f_dtype, [1, 8, 2, 2])],
            [("Y", f_dtype, [1, 2, 4, 4])],
            attrs={"blocksize": 2, "mode": "DCR"},
            opset=13,
        )
        x = np.random.rand(1, 8, 2, 2).astype(np.float32)

        def expected(f):
            inp = f["X"]
            b, c, h, w = inp.shape
            bs = 2
            return (
                inp.reshape(b, bs, bs, c // (bs * bs), h, w)
                .transpose(0, 3, 4, 1, 5, 2)
                .reshape(b, c // (bs * bs), h * bs, w * bs)
            )

        result = _run_model_test(target_device, "DepthToSpace", model, {"X": x}, expected)
        self.assertEqual(result, TEST_PASS, "DepthToSpace plugin op test failed")

    # ---- Contrib op tests ----

    def test_op_fast_gelu(self):
        target_device = get_cuda_plugin_device()
        f_dtype = TensorProto.FLOAT
        node = helper.make_node("FastGelu", ["X"], ["Y"], domain="com.microsoft")
        graph = helper.make_graph(
            [node],
            "test-FastGelu",
            [helper.make_tensor_value_info("X", f_dtype, [2, 4])],
            [helper.make_tensor_value_info("Y", f_dtype, [2, 4])],
        )
        opset_onnx = OperatorSetIdProto()
        opset_onnx.version = 13
        opset_ms = OperatorSetIdProto()
        opset_ms.domain = "com.microsoft"
        opset_ms.version = 1
        model = helper.make_model(graph, opset_imports=[opset_onnx, opset_ms])
        x = np.random.rand(2, 4).astype(np.float32)

        def expected(f):
            v = f["X"]
            return v * (1.0 / (1.0 + np.exp(-1.702 * v)))

        result = _run_model_test(target_device, "FastGelu", model, {"X": x}, expected, rtol=1e-2, atol=1e-2)
        self.assertEqual(result, TEST_PASS, "FastGelu plugin op test failed")

    def test_op_bias_dropout(self):
        target_device = get_cuda_plugin_device()
        model = make_bias_dropout_model()
        x = np.random.rand(2, 4).astype(np.float32)
        bias = np.random.rand(4).astype(np.float32)
        residual = np.random.rand(2, 4).astype(np.float32)
        ratio = np.array(0.5, dtype=np.float32)
        training_mode = np.array(False, dtype=np.bool_)
        feed = {"X": x, "bias": bias, "residual": residual, "ratio": ratio, "training_mode": training_mode}
        result = _run_model_test(
            target_device, "BiasDropout", model, feed, lambda f: f["X"] + f["bias"] + f["residual"]
        )
        self.assertEqual(result, TEST_PASS, "BiasDropout plugin op test failed")

    def test_op_dropout_opset7(self):
        """Dropout opset 7-9: simple in/out, no mask. Verifies old-version registration in dropout.cc."""
        target_device = get_cuda_plugin_device()
        f_dtype = TensorProto.FLOAT
        node = helper.make_node("Dropout", ["X"], ["Y"], ratio=0.0)
        graph = helper.make_graph(
            [node],
            "test-Dropout-opset7",
            [helper.make_tensor_value_info("X", f_dtype, [2, 4])],
            [helper.make_tensor_value_info("Y", f_dtype, [2, 4])],
        )
        opset = OperatorSetIdProto()
        opset.version = 7
        model = helper.make_model(graph, opset_imports=[opset])
        x = np.random.rand(2, 4).astype(np.float32)
        result = _run_model_test(target_device, "Dropout_opset7", model, {"X": x}, lambda f: f["X"])
        self.assertEqual(result, TEST_PASS, "Dropout opset 7 plugin op test failed")

    def test_op_dropout_opset10(self):
        """Dropout opset 10-11: data + optional mask output. Verifies old-version registration in dropout.cc."""
        target_device = get_cuda_plugin_device()
        f_dtype = TensorProto.FLOAT
        node = helper.make_node("Dropout", ["X"], ["Y", "mask"])
        graph = helper.make_graph(
            [node],
            "test-Dropout-opset10",
            [helper.make_tensor_value_info("X", f_dtype, [2, 4])],
            [
                helper.make_tensor_value_info("Y", f_dtype, [2, 4]),
                helper.make_tensor_value_info("mask", TensorProto.BOOL, [2, 4]),
            ],
        )
        opset = OperatorSetIdProto()
        opset.version = 10
        model = helper.make_model(graph, opset_imports=[opset])
        x = np.random.rand(2, 4).astype(np.float32)
        result = _run_model_test(
            target_device,
            "Dropout_opset10",
            model,
            {"X": x},
            lambda f: [f["X"], np.zeros((2, 4), dtype=bool)],
        )
        self.assertEqual(result, TEST_PASS, "Dropout opset 10 plugin op test failed")

    def test_op_dequantize_linear_opset21(self):
        """DequantizeLinear opset 21 uses TWO_TYPED_KERNEL_EX — verifies the new adapter macro."""
        target_device = get_cuda_plugin_device()
        node = helper.make_node("DequantizeLinear", ["x", "x_scale", "x_zero_point"], ["y"])
        x_data = np.array([0, 1, 2, 3, 4, 5], dtype=np.uint8)
        scale_data = np.array(0.5, dtype=np.float32)
        zp_data = np.array(2, dtype=np.uint8)
        graph = helper.make_graph(
            [node],
            "test-DequantizeLinear-opset21",
            [
                helper.make_tensor_value_info("x", TensorProto.UINT8, [6]),
                helper.make_tensor_value_info("x_scale", TensorProto.FLOAT, []),
                helper.make_tensor_value_info("x_zero_point", TensorProto.UINT8, []),
            ],
            [helper.make_tensor_value_info("y", TensorProto.FLOAT, [6])],
        )
        opset = OperatorSetIdProto()
        opset.version = 21
        model = helper.make_model(graph, opset_imports=[opset])
        feed = {"x": x_data, "x_scale": scale_data, "x_zero_point": zp_data}
        result = _run_model_test(
            target_device,
            "DequantizeLinear_opset21",
            model,
            feed,
            lambda f: (f["x"].astype(np.float32) - f["x_zero_point"].astype(np.float32)) * f["x_scale"],
        )
        self.assertEqual(result, TEST_PASS, "DequantizeLinear opset 21 plugin op test failed")

    def test_op_quantize_linear_opset21(self):
        """QuantizeLinear opset 21 uses TWO_TYPED_KERNEL_EX — verifies the new adapter macro."""
        target_device = get_cuda_plugin_device()
        node = helper.make_node("QuantizeLinear", ["x", "y_scale", "y_zero_point"], ["y"])
        x_data = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 2.5], dtype=np.float32)
        scale_data = np.array(0.5, dtype=np.float32)
        zp_data = np.array(0, dtype=np.uint8)
        graph = helper.make_graph(
            [node],
            "test-QuantizeLinear-opset21",
            [
                helper.make_tensor_value_info("x", TensorProto.FLOAT, [6]),
                helper.make_tensor_value_info("y_scale", TensorProto.FLOAT, []),
                helper.make_tensor_value_info("y_zero_point", TensorProto.UINT8, []),
            ],
            [helper.make_tensor_value_info("y", TensorProto.UINT8, [6])],
        )
        opset = OperatorSetIdProto()
        opset.version = 21
        model = helper.make_model(graph, opset_imports=[opset])
        feed = {"x": x_data, "y_scale": scale_data, "y_zero_point": zp_data}
        result = _run_model_test(
            target_device,
            "QuantizeLinear_opset21",
            model,
            feed,
            lambda f: np.clip(
                np.round(f["x"] / f["y_scale"]).astype(np.float32) + f["y_zero_point"].astype(np.float32),
                0,
                255,
            ).astype(np.uint8),
            atol=1,
        )
        self.assertEqual(result, TEST_PASS, "QuantizeLinear opset 21 plugin op test failed")

    def test_op_gather_block_quantized(self):
        """GatherBlockQuantized uses THREE_TYPED_KERNEL_EX — verifies the new adapter macro."""
        target_device = get_cuda_plugin_device()
        # GatherBlockQuantized: gathers rows from a block-quantized weight matrix.
        # data shape [4, 16] (uint8), scales shape [4, 1] (float), indices [2] (int64)
        # bits=8, block_size=16 (must be >= 16 and power of 2), quantize_axis=last
        node = helper.make_node(
            "GatherBlockQuantized",
            ["data", "indices", "scales"],
            ["output"],
            domain="com.microsoft",
            gather_axis=0,
            quantize_axis=1,
            block_size=16,
            bits=8,
        )
        data = np.random.randint(0, 255, size=(4, 16), dtype=np.uint8)
        scales = np.random.rand(4, 1).astype(np.float32) * 0.1 + 0.01
        indices = np.array([0, 2], dtype=np.int64)
        graph = helper.make_graph(
            [node],
            "test-GatherBlockQuantized",
            [
                helper.make_tensor_value_info("data", TensorProto.UINT8, [4, 16]),
                helper.make_tensor_value_info("indices", TensorProto.INT64, [2]),
                helper.make_tensor_value_info("scales", TensorProto.FLOAT, [4, 1]),
            ],
            [helper.make_tensor_value_info("output", TensorProto.FLOAT, [2, 16])],
        )
        opset_onnx = OperatorSetIdProto()
        opset_onnx.version = 21
        opset_ms = OperatorSetIdProto()
        opset_ms.domain = "com.microsoft"
        opset_ms.version = 1
        model = helper.make_model(graph, opset_imports=[opset_onnx, opset_ms])
        feed = {"data": data, "indices": indices, "scales": scales}

        def expected(f):
            # Gather rows [0, 2], then dequantize: float_val = uint8_val * scale
            gathered_data = f["data"][f["indices"]]  # [2, 16]
            gathered_scales = f["scales"][f["indices"]]  # [2, 1]
            return gathered_data.astype(np.float32) * gathered_scales

        result = _run_model_test(target_device, "GatherBlockQuantized", model, feed, expected, rtol=1e-2, atol=1e-2)
        self.assertEqual(result, TEST_PASS, "GatherBlockQuantized plugin op test failed")

    def test_op_skip_layer_norm(self):
        target_device = get_cuda_plugin_device()
        f_dtype = TensorProto.FLOAT
        hidden_size = 8
        node = helper.make_node(
            "SkipLayerNormalization",
            ["X", "skip", "gamma", "beta"],
            ["Y", "", "", "input_skip_bias_sum"],
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
                helper.make_tensor_value_info("input_skip_bias_sum", f_dtype, None),
            ],
        )
        opset_onnx = OperatorSetIdProto()
        opset_onnx.version = 13
        opset_ms = OperatorSetIdProto()
        opset_ms.domain = "com.microsoft"
        opset_ms.version = 1
        model = helper.make_model(graph, opset_imports=[opset_onnx, opset_ms])
        x = np.random.rand(2, hidden_size).astype(np.float32)
        skip = np.random.rand(2, hidden_size).astype(np.float32)
        gamma = np.ones(hidden_size, dtype=np.float32)
        beta = np.zeros(hidden_size, dtype=np.float32)

        def expected(f):
            added = f["X"] + f["skip"]
            mean = added.mean(axis=-1, keepdims=True)
            var = added.var(axis=-1, keepdims=True)
            normed = (added - mean) / np.sqrt(var + 1e-5)
            return [normed * f["gamma"] + f["beta"], added]

        result = _run_model_test(
            target_device,
            "SkipLayerNorm",
            model,
            {"X": x, "skip": skip, "gamma": gamma, "beta": beta},
            expected,
            rtol=1e-2,
            atol=1e-2,
        )
        self.assertEqual(result, TEST_PASS, "SkipLayerNorm plugin op test failed")

    # ---- Tests for previously-excluded ops (identity, crop, dynamicslice) ----

    def test_op_identity(self):
        """Identity op: previously excluded from plugin due to TensorSeq; now Tensor-only."""
        target_device = get_cuda_plugin_device()
        f_dtype = TensorProto.FLOAT
        model = _make_simple_model("Identity", [("X", f_dtype, [3, 4])], [("Y", f_dtype, [3, 4])])
        x = np.random.rand(3, 4).astype(np.float32)
        result = _run_model_test(target_device, "Identity", model, {"X": x}, lambda f: f["X"])
        self.assertEqual(result, TEST_PASS, "Identity plugin op test failed")

    def test_op_identity_opset25(self):
        """Identity opset 25: highest opset, uses V type constraint (Tensor subset in plugin)."""
        target_device = get_cuda_plugin_device()
        f_dtype = TensorProto.FLOAT
        model = _make_simple_model("Identity", [("X", f_dtype, [2, 5])], [("Y", f_dtype, [2, 5])], opset=25)
        x = np.random.rand(2, 5).astype(np.float32)
        result = _run_model_test(target_device, "Identity_opset25", model, {"X": x}, lambda f: f["X"])
        self.assertEqual(result, TEST_PASS, "Identity opset 25 plugin op test failed")

    def test_op_crop(self):
        """Crop (opset 1, contrib): previously excluded from plugin."""
        target_device = get_cuda_plugin_device()
        f_dtype = TensorProto.FLOAT
        node = helper.make_node("Crop", ["input"], ["output"], border=[1, 1, 1, 1])
        graph = helper.make_graph(
            [node],
            "test-Crop",
            [helper.make_tensor_value_info("input", f_dtype, [1, 1, 4, 4])],
            [helper.make_tensor_value_info("output", f_dtype, [1, 1, 2, 2])],
        )
        opset = OperatorSetIdProto()
        opset.version = 1
        model = helper.make_model(graph, opset_imports=[opset])
        x = np.arange(16, dtype=np.float32).reshape(1, 1, 4, 4)
        result = _run_model_test(
            target_device,
            "Crop",
            model,
            {"input": x},
            lambda f: f["input"][:, :, 1:3, 1:3],
        )
        self.assertEqual(result, TEST_PASS, "Crop plugin op test failed")

    def test_plugin_ep_claims_key_ops(self):
        """Session-based probing: verify the plugin EP claims key ops via graph assignment."""
        target_device = get_cuda_plugin_device()

        # Representative ops the plugin EP must claim (op_type, domain, opset, inputs, outputs, attrs).
        # One representative per major op family; ops already covered by dedicated test_registration_*
        # or test_op_* tests (Add, MatMul, Gemm, Conv, …) are intentionally excluded here.
        probe_specs = [
            # binary elementwise (Sub — Add is tested by test_registration_add)
            (
                "Sub",
                "",
                13,
                [("A", TensorProto.FLOAT, [2, 4]), ("B", TensorProto.FLOAT, [2, 4])],
                [("Y", TensorProto.FLOAT, [2, 4])],
                None,
            ),
            # unary activation
            ("Relu", "", 13, [("X", TensorProto.FLOAT, [2, 4])], [("Y", TensorProto.FLOAT, [2, 4])], None),
            # reduction-style
            ("Softmax", "", 13, [("X", TensorProto.FLOAT, [2, 4])], [("Y", TensorProto.FLOAT, [2, 4])], {"axis": -1}),
            # data-movement
            (
                "Transpose",
                "",
                13,
                [("X", TensorProto.FLOAT, [2, 4])],
                [("Y", TensorProto.FLOAT, [4, 2])],
                {"perm": [1, 0]},
            ),
            # type-dispatch
            (
                "Cast",
                "",
                13,
                [("X", TensorProto.FLOAT, [2, 4])],
                [("Y", TensorProto.FLOAT16, [2, 4])],
                {"to": int(TensorProto.FLOAT16)},
            ),
            # second unary
            ("Sigmoid", "", 13, [("X", TensorProto.FLOAT, [2, 4])], [("Y", TensorProto.FLOAT, [2, 4])], None),
            # cuDNN: ConvTranspose (Conv already tested by test_registration_conv)
            (
                "ConvTranspose",
                "",
                13,
                [("X", TensorProto.FLOAT, [1, 2, 3, 3]), ("W", TensorProto.FLOAT, [2, 3, 3, 3])],
                [("Y", TensorProto.FLOAT, [1, 3, 5, 5])],
                None,
            ),
            # cuDNN: LRN (local response normalization)
            (
                "LRN",
                "",
                13,
                [("X", TensorProto.FLOAT, [1, 2, 4, 4])],
                [("Y", TensorProto.FLOAT, [1, 2, 4, 4])],
                {"size": 3},
            ),
        ]

        claimed = []
        not_claimed = []
        errors = []

        for op_type, domain, opset, inputs_info, outputs_info, attrs in probe_specs:
            model = _make_simple_model(op_type, inputs_info, outputs_info, attrs=attrs, opset=opset, domain=domain)
            with tempfile.NamedTemporaryFile(suffix=f"_probe_{op_type}.onnx", delete=False) as tmp:
                model_path = tmp.name
            try:
                save(model, model_path)
                sess_options = _create_session_options()
                sess_options.graph_optimization_level = onnxrt.GraphOptimizationLevel.ORT_DISABLE_ALL
                sess_options.add_provider_for_devices([target_device], {})
                sess = onnxrt.InferenceSession(model_path, sess_options=sess_options)
                assigned_nodes, _ = _get_assigned_nodes(sess, CUDA_PLUGIN_EP_NAME)
                if assigned_nodes:
                    claimed.append(op_type)
                else:
                    not_claimed.append(op_type)
            except Exception as e:
                errors.append((op_type, str(e)[:120]))
            finally:
                if os.path.exists(model_path):
                    os.remove(model_path)

        # All probed ops should be claimed by the plugin EP
        self.assertFalse(
            not_claimed,
            f"Plugin EP did not claim these key ops: {not_claimed}",
        )
        self.assertFalse(
            errors,
            f"Errors probing ops: {errors}",
        )
        self.assertGreater(len(claimed), 0, "No ops were claimed at all")

    # ---- Newly-included ops that previously lacked tests ----

    def test_op_einsum(self):
        """Test Einsum op (recently un-excluded from plugin build)."""
        target_device = get_cuda_plugin_device()
        model = _make_simple_model(
            "Einsum",
            [("A", TensorProto.FLOAT, [2, 3]), ("B", TensorProto.FLOAT, [3, 4])],
            [("Y", TensorProto.FLOAT, [2, 4])],
            attrs={"equation": "ij,jk->ik"},
            opset=12,
        )
        feed = {"A": np.random.rand(2, 3).astype(np.float32), "B": np.random.rand(3, 4).astype(np.float32)}
        result = _run_model_test(target_device, "Einsum", model, feed, lambda f: f["A"] @ f["B"])
        self.assertEqual(result, TEST_PASS, "Einsum test failed")

    def test_op_einsum_batch(self):
        """Test Einsum op with batch matrix multiply."""
        target_device = get_cuda_plugin_device()
        model = _make_simple_model(
            "Einsum",
            [("A", TensorProto.FLOAT, [2, 3, 4]), ("B", TensorProto.FLOAT, [2, 4, 5])],
            [("Y", TensorProto.FLOAT, [2, 3, 5])],
            attrs={"equation": "bij,bjk->bik"},
            opset=12,
        )
        feed = {"A": np.random.rand(2, 3, 4).astype(np.float32), "B": np.random.rand(2, 4, 5).astype(np.float32)}
        result = _run_model_test(target_device, "Einsum_batch", model, feed, lambda f: np.matmul(f["A"], f["B"]))
        self.assertEqual(result, TEST_PASS, "Einsum batch test failed")

    def test_op_softmax(self):
        target_device = get_cuda_plugin_device()
        model = _make_simple_model(
            "Softmax",
            [("X", TensorProto.FLOAT, [2, 5])],
            [("Y", TensorProto.FLOAT, [2, 5])],
            attrs={"axis": 1},
            opset=13,
        )
        feed = {"X": np.random.rand(2, 5).astype(np.float32)}

        def expected(f):
            x = f["X"]
            e = np.exp(x - np.max(x, axis=1, keepdims=True))
            return e / np.sum(e, axis=1, keepdims=True)

        result = _run_model_test(target_device, "Softmax", model, feed, expected)
        self.assertEqual(result, TEST_PASS, "Softmax test failed")

    def test_op_relu(self):
        target_device = get_cuda_plugin_device()
        model = _make_simple_model(
            "Relu",
            [("X", TensorProto.FLOAT, [3, 4])],
            [("Y", TensorProto.FLOAT, [3, 4])],
            opset=14,
        )
        feed = {"X": np.random.randn(3, 4).astype(np.float32)}
        result = _run_model_test(target_device, "Relu", model, feed, lambda f: np.maximum(f["X"], 0))
        self.assertEqual(result, TEST_PASS, "Relu test failed")

    def test_op_sigmoid(self):
        target_device = get_cuda_plugin_device()
        model = _make_simple_model(
            "Sigmoid",
            [("X", TensorProto.FLOAT, [3, 4])],
            [("Y", TensorProto.FLOAT, [3, 4])],
            opset=13,
        )
        feed = {"X": np.random.randn(3, 4).astype(np.float32)}
        result = _run_model_test(target_device, "Sigmoid", model, feed, lambda f: 1.0 / (1.0 + np.exp(-f["X"])))
        self.assertEqual(result, TEST_PASS, "Sigmoid test failed")

    def test_op_tanh(self):
        target_device = get_cuda_plugin_device()
        model = _make_simple_model(
            "Tanh",
            [("X", TensorProto.FLOAT, [3, 4])],
            [("Y", TensorProto.FLOAT, [3, 4])],
            opset=13,
        )
        feed = {"X": np.random.randn(3, 4).astype(np.float32)}
        result = _run_model_test(target_device, "Tanh", model, feed, lambda f: np.tanh(f["X"]))
        self.assertEqual(result, TEST_PASS, "Tanh test failed")

    def test_op_transpose(self):
        target_device = get_cuda_plugin_device()
        model = _make_simple_model(
            "Transpose",
            [("X", TensorProto.FLOAT, [2, 3, 4])],
            [("Y", TensorProto.FLOAT, [4, 2, 3])],
            attrs={"perm": [2, 0, 1]},
            opset=13,
        )
        feed = {"X": np.random.rand(2, 3, 4).astype(np.float32)}
        result = _run_model_test(target_device, "Transpose", model, feed, lambda f: np.transpose(f["X"], (2, 0, 1)))
        self.assertEqual(result, TEST_PASS, "Transpose test failed")

    def test_op_cast(self):
        target_device = get_cuda_plugin_device()
        model = _make_simple_model(
            "Cast",
            [("X", TensorProto.FLOAT, [3, 4])],
            [("Y", TensorProto.FLOAT16, [3, 4])],
            attrs={"to": int(TensorProto.FLOAT16)},
            opset=13,
        )
        feed = {"X": np.random.rand(3, 4).astype(np.float32)}
        result = _run_model_test(
            target_device, "Cast", model, feed, lambda f: f["X"].astype(np.float16), rtol=1e-2, atol=1e-2
        )
        self.assertEqual(result, TEST_PASS, "Cast test failed")

    def test_op_where(self):
        target_device = get_cuda_plugin_device()
        model = _make_simple_model(
            "Where",
            [
                ("cond", TensorProto.BOOL, [3, 4]),
                ("X", TensorProto.FLOAT, [3, 4]),
                ("Y", TensorProto.FLOAT, [3, 4]),
            ],
            [("out", TensorProto.FLOAT, [3, 4])],
            opset=16,
        )
        cond = np.random.randint(0, 2, size=(3, 4)).astype(bool)
        x = np.random.rand(3, 4).astype(np.float32)
        y = np.random.rand(3, 4).astype(np.float32)
        feed = {"cond": cond, "X": x, "Y": y}
        result = _run_model_test(target_device, "Where", model, feed, lambda f: np.where(f["cond"], f["X"], f["Y"]))
        self.assertEqual(result, TEST_PASS, "Where test failed")

    def test_op_flatten(self):
        target_device = get_cuda_plugin_device()
        model = _make_simple_model(
            "Flatten",
            [("X", TensorProto.FLOAT, [2, 3, 4])],
            [("Y", TensorProto.FLOAT, [2, 12])],
            attrs={"axis": 1},
            opset=13,
        )
        feed = {"X": np.random.rand(2, 3, 4).astype(np.float32)}
        result = _run_model_test(target_device, "Flatten", model, feed, lambda f: f["X"].reshape(2, 12))
        self.assertEqual(result, TEST_PASS, "Flatten test failed")

    def test_op_argmax(self):
        target_device = get_cuda_plugin_device()
        model = _make_simple_model(
            "ArgMax",
            [("X", TensorProto.FLOAT, [3, 5])],
            [("Y", TensorProto.INT64, [3, 1])],
            attrs={"axis": 1, "keepdims": 1},
            opset=13,
        )
        feed = {"X": np.random.rand(3, 5).astype(np.float32)}
        result = _run_model_test(
            target_device, "ArgMax", model, feed, lambda f: np.argmax(f["X"], axis=1).reshape(3, 1)
        )
        self.assertEqual(result, TEST_PASS, "ArgMax test failed")

    def test_op_topk(self):
        target_device = get_cuda_plugin_device()
        model = _make_simple_model(
            "TopK",
            [("X", TensorProto.FLOAT, [3, 6]), ("K", TensorProto.INT64, [1])],
            [("values", TensorProto.FLOAT, [3, 3]), ("indices", TensorProto.INT64, [3, 3])],
            attrs={"axis": 1},
            opset=11,
        )
        x = np.random.rand(3, 6).astype(np.float32)
        k = np.array([3], dtype=np.int64)
        feed = {"X": x, "K": k}

        def expected(f):
            idx = np.argsort(-f["X"], axis=1)[:, :3]
            vals = np.take_along_axis(f["X"], idx, axis=1)
            return [vals, idx]

        result = _run_model_test(target_device, "TopK", model, feed, expected)
        self.assertEqual(result, TEST_PASS, "TopK test failed")

    def test_op_layer_normalization(self):
        """Test LayerNormalization — critical for transformer models."""
        target_device = get_cuda_plugin_device()
        normalized_shape = 8
        model = _make_simple_model(
            "LayerNormalization",
            [
                ("X", TensorProto.FLOAT, [2, 3, normalized_shape]),
                ("scale", TensorProto.FLOAT, [normalized_shape]),
                ("bias", TensorProto.FLOAT, [normalized_shape]),
            ],
            [("Y", TensorProto.FLOAT, [2, 3, normalized_shape])],
            attrs={"axis": -1, "epsilon": 1e-5},
            opset=17,
        )
        scale = np.ones(normalized_shape, dtype=np.float32)
        bias = np.zeros(normalized_shape, dtype=np.float32)
        scale_init = helper.make_tensor("scale", TensorProto.FLOAT, [normalized_shape], scale.tolist())
        bias_init = helper.make_tensor("bias", TensorProto.FLOAT, [normalized_shape], bias.tolist())
        model.graph.initializer.append(scale_init)
        model.graph.initializer.append(bias_init)

        x = np.random.rand(2, 3, normalized_shape).astype(np.float32)
        feed = {"X": x}

        def expected(f):
            x = f["X"]
            mean = np.mean(x, axis=-1, keepdims=True)
            var = np.var(x, axis=-1, keepdims=True)
            return (x - mean) / np.sqrt(var + 1e-5) * scale + bias

        result = _run_model_test(target_device, "LayerNormalization", model, feed, expected)
        self.assertEqual(result, TEST_PASS, "LayerNormalization test failed")

    def test_op_instance_normalization(self):
        target_device = get_cuda_plugin_device()
        n_channels = 3
        model = _make_simple_model(
            "InstanceNormalization",
            [
                ("X", TensorProto.FLOAT, [1, n_channels, 4, 4]),
                ("scale", TensorProto.FLOAT, [n_channels]),
                ("B", TensorProto.FLOAT, [n_channels]),
            ],
            [("Y", TensorProto.FLOAT, [1, n_channels, 4, 4])],
            attrs={"epsilon": 1e-5},
            opset=6,
        )
        scale = np.ones(n_channels, dtype=np.float32)
        bias = np.zeros(n_channels, dtype=np.float32)
        model.graph.initializer.append(helper.make_tensor("scale", TensorProto.FLOAT, [n_channels], scale.tolist()))
        model.graph.initializer.append(helper.make_tensor("B", TensorProto.FLOAT, [n_channels], bias.tolist()))

        x = np.random.rand(1, n_channels, 4, 4).astype(np.float32)
        feed = {"X": x}

        def expected(f):
            x = f["X"]
            result = np.empty_like(x)
            for c in range(n_channels):
                ch = x[0, c]
                mean = ch.mean()
                var = ch.var()
                result[0, c] = (ch - mean) / np.sqrt(var + 1e-5) * scale[c] + bias[c]
            return result

        result = _run_model_test(target_device, "InstanceNormalization", model, feed, expected)
        self.assertEqual(result, TEST_PASS, "InstanceNormalization test failed")

    def test_op_conv_transpose(self):
        target_device = get_cuda_plugin_device()
        model = _make_simple_model(
            "ConvTranspose",
            [
                ("X", TensorProto.FLOAT, [1, 3, 4, 4]),
                ("W", TensorProto.FLOAT, [3, 2, 3, 3]),
            ],
            [("Y", TensorProto.FLOAT, [1, 2, 6, 6])],
            attrs={"kernel_shape": [3, 3], "strides": [1, 1], "pads": [0, 0, 0, 0]},
            opset=11,
        )
        x = np.random.rand(1, 3, 4, 4).astype(np.float32)
        w = np.random.rand(3, 2, 3, 3).astype(np.float32)
        feed = {"X": x, "W": w}

        def expected(f):
            return F.conv_transpose2d(torch.from_numpy(f["X"]), torch.from_numpy(f["W"])).numpy()

        result = _run_model_test(target_device, "ConvTranspose", model, feed, expected)
        self.assertEqual(result, TEST_PASS, "ConvTranspose test failed")

    def test_op_reduce_mean(self):
        target_device = get_cuda_plugin_device()
        model = _make_simple_model(
            "ReduceMean",
            [("X", TensorProto.FLOAT, [3, 4, 5])],
            [("Y", TensorProto.FLOAT, [3, 1, 5])],
            attrs={"axes": [1], "keepdims": 1},
            opset=13,
        )
        feed = {"X": np.random.rand(3, 4, 5).astype(np.float32)}
        result = _run_model_test(
            target_device, "ReduceMean", model, feed, lambda f: np.mean(f["X"], axis=1, keepdims=True)
        )
        self.assertEqual(result, TEST_PASS, "ReduceMean test failed")

    def test_op_reduce_sum(self):
        target_device = get_cuda_plugin_device()
        model = _make_simple_model(
            "ReduceSum",
            [("X", TensorProto.FLOAT, [3, 4, 5]), ("axes", TensorProto.INT64, [1])],
            [("Y", TensorProto.FLOAT, [3, 1, 5])],
            attrs={"keepdims": 1},
            opset=13,
        )
        axes_init = helper.make_tensor("axes", TensorProto.INT64, [1], [1])
        model.graph.initializer.append(axes_init)
        feed = {"X": np.random.rand(3, 4, 5).astype(np.float32)}
        result = _run_model_test(
            target_device, "ReduceSum", model, feed, lambda f: np.sum(f["X"], axis=1, keepdims=True)
        )
        self.assertEqual(result, TEST_PASS, "ReduceSum test failed")

    def test_op_gather_nd(self):
        target_device = get_cuda_plugin_device()
        model = _make_simple_model(
            "GatherND",
            [
                ("data", TensorProto.FLOAT, [2, 3, 4]),
                ("indices", TensorProto.INT64, [2, 1]),
            ],
            [("Y", TensorProto.FLOAT, [2, 4])],
            attrs={"batch_dims": 1},
            opset=12,
        )
        data = np.random.rand(2, 3, 4).astype(np.float32)
        indices = np.array([[1], [2]], dtype=np.int64)
        feed = {"data": data, "indices": indices}

        def expected(f):
            return np.array([f["data"][0, 1], f["data"][1, 2]])

        result = _run_model_test(target_device, "GatherND", model, feed, expected)
        self.assertEqual(result, TEST_PASS, "GatherND test failed")

    def test_op_scatter_elements(self):
        target_device = get_cuda_plugin_device()
        model = _make_simple_model(
            "ScatterElements",
            [
                ("data", TensorProto.FLOAT, [3, 3]),
                ("indices", TensorProto.INT64, [2, 3]),
                ("updates", TensorProto.FLOAT, [2, 3]),
            ],
            [("Y", TensorProto.FLOAT, [3, 3])],
            attrs={"axis": 0},
            opset=16,
        )
        data = np.zeros((3, 3), dtype=np.float32)
        indices = np.array([[1, 0, 2], [0, 2, 1]], dtype=np.int64)
        updates = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
        feed = {"data": data, "indices": indices, "updates": updates}

        def expected(f):
            result = f["data"].copy()
            for i in range(2):
                for j in range(3):
                    result[f["indices"][i, j], j] = f["updates"][i, j]
            return result

        result = _run_model_test(target_device, "ScatterElements", model, feed, expected)
        self.assertEqual(result, TEST_PASS, "ScatterElements test failed")

    def test_op_onehot(self):
        target_device = get_cuda_plugin_device()
        model = _make_simple_model(
            "OneHot",
            [
                ("indices", TensorProto.INT64, [4]),
                ("depth", TensorProto.INT64, [1]),
                ("values", TensorProto.FLOAT, [2]),
            ],
            [("Y", TensorProto.FLOAT, [4, 6])],
            attrs={"axis": 1},
            opset=11,
        )
        indices = np.array([0, 2, 4, 5], dtype=np.int64)
        depth = np.array([6], dtype=np.int64)
        values = np.array([0.0, 1.0], dtype=np.float32)
        feed = {"indices": indices, "depth": depth, "values": values}

        def expected(f):
            result = np.zeros((4, 6), dtype=np.float32)
            for i, idx in enumerate(f["indices"]):
                result[i, idx] = 1.0
            return result

        result = _run_model_test(target_device, "OneHot", model, feed, expected)
        self.assertEqual(result, TEST_PASS, "OneHot test failed")

    # NOTE: Range is excluded — it runs on CPU (shape computation op, not claimed by CUDA EP).

    def test_op_non_zero(self):
        target_device = get_cuda_plugin_device()
        model = _make_simple_model(
            "NonZero",
            [("X", TensorProto.FLOAT, [3, 4])],
            [("Y", TensorProto.INT64, None)],
            opset=13,
        )
        x = np.array([[1, 0, 3, 0], [0, 5, 0, 7], [0, 0, 0, 10]], dtype=np.float32)
        feed = {"X": x}
        result = _run_model_test(target_device, "NonZero", model, feed, lambda f: np.array(np.nonzero(f["X"])))
        self.assertEqual(result, TEST_PASS, "NonZero test failed")

    def test_op_grid_sample(self):
        target_device = get_cuda_plugin_device()
        n, c, h, w = 1, 1, 4, 4
        model = _make_simple_model(
            "GridSample",
            [
                ("X", TensorProto.FLOAT, [n, c, h, w]),
                ("grid", TensorProto.FLOAT, [n, 2, 2, 2]),
            ],
            [("Y", TensorProto.FLOAT, [n, c, 2, 2])],
            attrs={"mode": "bilinear", "padding_mode": "zeros", "align_corners": 0},
            opset=16,
        )
        x = np.random.rand(n, c, h, w).astype(np.float32)
        grid = np.random.rand(n, 2, 2, 2).astype(np.float32) * 2 - 1  # in [-1, 1]
        feed = {"X": x, "grid": grid}

        def expected(f):
            return F.grid_sample(
                torch.from_numpy(f["X"]),
                torch.from_numpy(f["grid"]),
                mode="bilinear",
                padding_mode="zeros",
                align_corners=False,
            ).numpy()

        result = _run_model_test(target_device, "GridSample", model, feed, expected, rtol=1e-3, atol=1e-3)
        self.assertEqual(result, TEST_PASS, "GridSample test failed")

    def test_op_gelu(self):
        """Test Gelu contrib op — important for transformer models."""
        target_device = get_cuda_plugin_device()
        model = _make_simple_model(
            "Gelu",
            [("X", TensorProto.FLOAT, [2, 8])],
            [("Y", TensorProto.FLOAT, [2, 8])],
            domain="com.microsoft",
            opset=13,
        )
        feed = {"X": np.random.randn(2, 8).astype(np.float32)}

        def expected(f):
            return torch.nn.functional.gelu(torch.from_numpy(f["X"])).numpy()

        result = _run_model_test(target_device, "Gelu", model, feed, expected)
        self.assertEqual(result, TEST_PASS, "Gelu test failed")

    def test_op_bias_gelu(self):
        """Test BiasGelu contrib op."""
        target_device = get_cuda_plugin_device()
        model = _make_simple_model(
            "BiasGelu",
            [("X", TensorProto.FLOAT, [2, 8]), ("bias", TensorProto.FLOAT, [8])],
            [("Y", TensorProto.FLOAT, [2, 8])],
            domain="com.microsoft",
            opset=13,
        )
        bias = np.random.randn(8).astype(np.float32)
        model.graph.initializer.append(helper.make_tensor("bias", TensorProto.FLOAT, [8], bias.tolist()))
        feed = {"X": np.random.randn(2, 8).astype(np.float32)}

        def expected(f):
            x = torch.from_numpy(f["X"]) + torch.from_numpy(bias)
            return torch.nn.functional.gelu(x).numpy()

        result = _run_model_test(target_device, "BiasGelu", model, feed, expected)
        self.assertEqual(result, TEST_PASS, "BiasGelu test failed")

    def test_op_fused_matmul(self):
        """Test FusedMatMul contrib op (MatMul with alpha)."""
        target_device = get_cuda_plugin_device()
        model = _make_simple_model(
            "FusedMatMul",
            [("A", TensorProto.FLOAT, [3, 4]), ("B", TensorProto.FLOAT, [4, 5])],
            [("Y", TensorProto.FLOAT, [3, 5])],
            attrs={"alpha": 2.0},
            domain="com.microsoft",
            opset=13,
        )
        feed = {"A": np.random.rand(3, 4).astype(np.float32), "B": np.random.rand(4, 5).astype(np.float32)}
        result = _run_model_test(target_device, "FusedMatMul", model, feed, lambda f: 2.0 * (f["A"] @ f["B"]))
        self.assertEqual(result, TEST_PASS, "FusedMatMul test failed")

    def test_op_trilu(self):
        target_device = get_cuda_plugin_device()
        model = _make_simple_model(
            "Trilu",
            [("X", TensorProto.FLOAT, [4, 4])],
            [("Y", TensorProto.FLOAT, [4, 4])],
            attrs={"upper": 1},
            opset=14,
        )
        feed = {"X": np.random.rand(4, 4).astype(np.float32)}
        result = _run_model_test(target_device, "Trilu", model, feed, lambda f: np.triu(f["X"]))
        self.assertEqual(result, TEST_PASS, "Trilu test failed")

    def test_op_matmul_integer(self):
        """Test MatMulInteger — used in INT8 quantization pipelines."""
        target_device = get_cuda_plugin_device()
        model = _make_simple_model(
            "MatMulInteger",
            [
                ("A", TensorProto.INT8, [3, 4]),
                ("B", TensorProto.INT8, [4, 5]),
            ],
            [("Y", TensorProto.INT32, [3, 5])],
            opset=10,
        )
        a = np.random.randint(-128, 127, size=(3, 4)).astype(np.int8)
        b = np.random.randint(-128, 127, size=(4, 5)).astype(np.int8)
        feed = {"A": a, "B": b}
        result = _run_model_test(
            target_device,
            "MatMulInteger",
            model,
            feed,
            lambda f: f["A"].astype(np.int32) @ f["B"].astype(np.int32),
        )
        self.assertEqual(result, TEST_PASS, "MatMulInteger test failed")

    # ---- MemcpyFromHost / MemcpyToHost tests ----
    # These tests explicitly place MemcpyFromHost/MemcpyToHost nodes in the graph
    # to directly exercise the plugin-side copy kernels.

    def test_memcpy_from_host_explicit(self):
        """Explicit MemcpyFromHost node: CPU input → GPU copy → Relu on GPU."""
        target_device = get_cuda_plugin_device()
        # X (CPU) → MemcpyFromHost → X_gpu → Relu → Y
        copy_node = helper.make_node("MemcpyFromHost", ["X"], ["X_gpu"])
        relu_node = helper.make_node("Relu", ["X_gpu"], ["Y"])
        graph = helper.make_graph(
            [copy_node, relu_node],
            "test-explicit-memcpy-from-host",
            [helper.make_tensor_value_info("X", TensorProto.FLOAT, [3, 4])],
            [helper.make_tensor_value_info("Y", TensorProto.FLOAT, [3, 4])],
        )
        opset = OperatorSetIdProto()
        opset.version = 13
        model = helper.make_model(graph, opset_imports=[opset])
        feed = {"X": np.random.randn(3, 4).astype(np.float32)}
        result = _run_model_test(
            target_device,
            "MemcpyFromHost_explicit",
            model,
            feed,
            lambda f: np.maximum(f["X"], 0),
        )
        self.assertEqual(result, TEST_PASS, "Explicit MemcpyFromHost test failed")

    def test_memcpy_to_host_explicit(self):
        """Explicit MemcpyToHost node: GPU Add → GPU-to-CPU copy → output."""
        target_device = get_cuda_plugin_device()
        # A, B → Add (GPU) → sum_gpu → MemcpyToHost → Y (CPU)
        add_node = helper.make_node("Add", ["A", "B"], ["sum_gpu"])
        copy_node = helper.make_node("MemcpyToHost", ["sum_gpu"], ["Y"])
        graph = helper.make_graph(
            [add_node, copy_node],
            "test-explicit-memcpy-to-host",
            [
                helper.make_tensor_value_info("A", TensorProto.FLOAT, [2, 3]),
                helper.make_tensor_value_info("B", TensorProto.FLOAT, [2, 3]),
            ],
            [helper.make_tensor_value_info("Y", TensorProto.FLOAT, [2, 3])],
        )
        opset = OperatorSetIdProto()
        opset.version = 13
        model = helper.make_model(graph, opset_imports=[opset])
        feed = {
            "A": np.random.rand(2, 3).astype(np.float32),
            "B": np.random.rand(2, 3).astype(np.float32),
        }
        result = _run_model_test(
            target_device,
            "MemcpyToHost_explicit",
            model,
            feed,
            lambda f: f["A"] + f["B"],
        )
        self.assertEqual(result, TEST_PASS, "Explicit MemcpyToHost test failed")

    def test_memcpy_roundtrip_explicit(self):
        """Explicit both directions: CPU → MemcpyFromHost → Relu (GPU) → MemcpyToHost → CPU."""
        target_device = get_cuda_plugin_device()
        copy_in = helper.make_node("MemcpyFromHost", ["X"], ["X_gpu"])
        relu_node = helper.make_node("Relu", ["X_gpu"], ["relu_out"])
        copy_out = helper.make_node("MemcpyToHost", ["relu_out"], ["Y"])
        graph = helper.make_graph(
            [copy_in, relu_node, copy_out],
            "test-explicit-memcpy-roundtrip",
            [helper.make_tensor_value_info("X", TensorProto.FLOAT, [4, 5])],
            [helper.make_tensor_value_info("Y", TensorProto.FLOAT, [4, 5])],
        )
        opset = OperatorSetIdProto()
        opset.version = 13
        model = helper.make_model(graph, opset_imports=[opset])
        feed = {"X": np.random.randn(4, 5).astype(np.float32)}
        result = _run_model_test(
            target_device,
            "MemcpyRoundtrip_explicit",
            model,
            feed,
            lambda f: np.maximum(f["X"], 0),
        )
        self.assertEqual(result, TEST_PASS, "Explicit MemcpyFromHost→Relu→MemcpyToHost roundtrip test failed")


if __name__ == "__main__":
    unittest.main()
