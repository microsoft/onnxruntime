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
    require_cuda_plugin_ep()

    try:
        devices = onnxrt.get_ep_devices()
    except Exception as exc:
        raise unittest.SkipTest(f"Failed to enumerate CUDA plugin EP devices: {exc}") from exc

    plugin_devices = [device for device in devices if device.ep_name == CUDA_PLUGIN_EP_NAME]
    if not plugin_devices:
        raise unittest.SkipTest("CUDA plugin EP registered, but no plugin devices were enumerated")

    return plugin_devices[0]


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


if __name__ == "__main__":
    unittest.main()
