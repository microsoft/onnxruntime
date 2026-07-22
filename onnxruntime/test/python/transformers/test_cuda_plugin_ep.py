# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import json
import os
import tempfile
import unittest
from contextlib import contextmanager

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
NO_CUDNN_PLUGIN_TEST = os.getenv("ORT_TEST_CUDA_PLUGIN_NO_CUDNN", "").upper() in {"1", "ON", "TRUE", "YES"}
requires_cudnn = unittest.skipIf(NO_CUDNN_PLUGIN_TEST, "test requires cuDNN-backed CUDA plugin kernels")
# Use the latest released ai.onnx opset so the model builders stay current as ONNX releases new opsets.
DEFAULT_ONNX_OPSET = max(v for (d, v) in helper.OP_SET_ID_VERSION_MAP if d == "ai.onnx")


def _make_released_opset_model(graph, producer_name="onnx-example"):
    opset = OperatorSetIdProto()
    opset.version = DEFAULT_ONNX_OPSET
    return helper.make_model(graph, producer_name=producer_name, opset_imports=[opset])


def _plugin_provider_options(extra_options=None):
    options = {"enable_cudnn": "0"} if NO_CUDNN_PLUGIN_TEST else {}
    if extra_options:
        options.update(extra_options)
    return options


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


def get_cuda_plugin_device_id(device):
    device_id = device.ep_options.get("device_id")
    if device_id is None:
        device_id = device.ep_metadata.get("cuda_device_id")

    if device_id is None:
        raise unittest.SkipTest("CUDA plugin EP device metadata did not include a CUDA device_id")

    try:
        return int(device_id)
    except (TypeError, ValueError) as exc:
        raise unittest.SkipTest(f"CUDA plugin EP device metadata had non-integer device_id={device_id!r}") from exc


def is_cuda_mempool_unsupported_error(exc: Exception) -> bool:
    message = str(exc)
    return "cudaMemPoolCreate failed" in message and (
        "cudaErrorNotSupported" in message or "operation not supported" in message
    )


# The fpA_intB MatMulNBits pre-pack path (which triggers the EP-provided allocator use this test
# guards) requires a compute capability >= 7.5 (Turing) device; see MatMulNBits::MatMulNBits.
FPA_INTB_MIN_COMPUTE_CAPABILITY = (7, 5)


def get_cuda_compute_capability(device):
    """Return the device (major, minor) CUDA compute capability from plugin EP metadata, or None."""
    value = device.ep_metadata.get("cuda_compute_capability")
    if not value:
        return None
    try:
        major_str, _, minor_str = value.partition(".")
        return (int(major_str), int(minor_str))
    except (TypeError, ValueError):
        return None


def is_fpa_intb_unsupported_error(exc: Exception) -> bool:
    """True if the error indicates the fpA_intB MatMulNBits path is unsupported for this device/config."""
    message = str(exc)
    return "fpA_intB" in message


@contextmanager
def scoped_env(name: str, value: str):
    old_value = os.environ.get(name)
    os.environ[name] = value
    try:
        yield
    finally:
        if old_value is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = old_value


def _create_session_options(session_config=None):
    sess_options = onnxrt.SessionOptions()
    if session_config:
        for key, value in session_config.items():
            sess_options.add_session_config_entry(key, value)

    # Require graph-assignment data so the tests validate that nodes actually run on the plugin.
    sess_options.add_session_config_entry(EP_GRAPH_ASSIGNMENT_CONFIG_KEY, "1")
    return sess_options


class _CudaOrtValueBinding:
    def __init__(self, shape, dtype, device_id):
        if dtype != np.float32:
            raise TypeError(f"Unsupported CUDA graph binding dtype: {dtype}")

        self._dtype = np.float32
        # Allocate a GPU-backed OrtValue with a stable device address so CUDA graph
        # capture/replay can reuse the same memory across runs.
        self.ort_value = onnxrt.OrtValue.ortvalue_from_shape_and_type(list(shape), self._dtype, "cuda", device_id)

    def update_inplace(self, data):
        # Copy host data into the existing GPU buffer without changing its address.
        self.ort_value.update_inplace(np.ascontiguousarray(data, dtype=self._dtype))

    def numpy(self):
        return self.ort_value.numpy()


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
    model_def = _make_released_opset_model(graph_def)
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
    model_def = _make_released_opset_model(graph_def)
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
    model_def = _make_released_opset_model(graph_def)
    save(model_def, model_path)


def create_matmul_nbits_model(model_path, k=64, n=64, bits=4, block_size=32):
    # Create a MatMulNBits (com.microsoft) model with a runtime-prepacked (weight_prepacked=0)
    # fp16 quantized weight. During session initialization the CUDA kernel pre-packs the constant
    # weight via MatMulNBits::PrePack_B, which allocates scratch through the EP-provided allocator
    # (IAllocator::MakeUniquePtr(alloc, ...)). This is a regression guard: the CUDA-EP-as-plugin
    # adapter previously forwarded a null allocator into PrePack, so this model crashed at session
    # creation with "IAllocator::ValidateAllocator ... allocator != nullptr was false".
    k_blocks = (k + block_size - 1) // block_size
    blob_size = block_size * bits // 8

    # Deterministic quantized weight and positive scales. Exact numerics are not asserted here;
    # the test only requires that pre-packing runs (and no longer crashes) end to end.
    b = np.full((n, k_blocks, blob_size), 0x88, dtype=np.uint8)
    scales = np.full((n, k_blocks), 0.02, dtype=np.float16)

    node_def = helper.make_node(
        "MatMulNBits",
        ["A", "B", "scales"],
        ["Y"],
        domain="com.microsoft",
        K=k,
        N=n,
        bits=bits,
        block_size=block_size,
        weight_prepacked=0,
    )
    graph_def = helper.make_graph(
        [node_def],
        "test-model-matmulnbits",
        [helper.make_tensor_value_info("A", TensorProto.FLOAT16, [2, k])],
        [helper.make_tensor_value_info("Y", TensorProto.FLOAT16, [2, n])],
        initializer=[
            helper.make_tensor("B", TensorProto.UINT8, b.shape, b.tobytes(), raw=True),
            helper.make_tensor("scales", TensorProto.FLOAT16, scales.shape, scales.tobytes(), raw=True),
        ],
    )
    opset_onnx = OperatorSetIdProto()
    opset_onnx.version = 21
    opset_ms = OperatorSetIdProto()
    opset_ms.domain = "com.microsoft"
    opset_ms.version = 1
    model_def = helper.make_model(graph_def, producer_name="onnx-example", opset_imports=[opset_onnx, opset_ms])
    model_def.ir_version = 10
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
    target_device,
    model_creator,
    inputs,
    expected_fn,
    ep_name=CUDA_PLUGIN_EP_NAME,
    session_config=None,
    nhwc_ops=None,
):
    with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as tmp:
        model_path = tmp.name
    try:
        model_creator(model_path)
        sess_options = _create_session_options(session_config)
        sess_options.add_provider_for_devices([target_device], _plugin_provider_options())
        sess = onnxrt.InferenceSession(model_path, sess_options=sess_options)

        active_providers = sess.get_providers()
        assigned_nodes, assignment_info = _get_assigned_nodes(sess, ep_name)
        if not assigned_nodes:
            print(
                f"FAILURE: {ep_name} was assigned no nodes. Providers: {active_providers}. "
                f"Assignments: {_format_assignment_summary(assignment_info)}"
            )
            return False

        # Structural assertion: verify NHWC domain assignment when requested
        if nhwc_ops:
            _assert_nhwc_domain_assigned(sess, ep_name, nhwc_ops)

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

    # When we expect the plugin provider to work, verify that at least one plugin device is available.
    # Device enumeration can fail in some CI environments even when the plugin library loads successfully.
    if expect_plugin_provider:
        try:
            devices = onnxrt.get_ep_devices()
            plugin_devices = [d for d in devices if d.ep_name == CUDA_PLUGIN_EP_NAME]
            if not plugin_devices:
                raise unittest.SkipTest(
                    f"{CUDA_PLUGIN_EP_NAME} registered but no plugin devices enumerated in this environment"
                )
        except unittest.SkipTest:
            raise
        except Exception as exc:
            raise unittest.SkipTest(f"Failed to enumerate plugin EP devices: {exc}") from exc

    with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as tmp:
        model_path = tmp.name
    try:
        create_add_model(model_path)
        sess_options = _create_session_options()
        if expect_plugin_provider:
            target_device = get_cuda_plugin_device_by_id(int(provider_options.get("device_id", "0")))
            sess_options.add_provider_for_devices([target_device], _plugin_provider_options(provider_options))
            sess = onnxrt.InferenceSession(model_path, sess_options=sess_options)
        else:
            try:
                target_device = get_cuda_plugin_device_by_id(int(provider_options.get("device_id", "0")))
            except unittest.SkipTest:
                sess = onnxrt.InferenceSession(
                    model_path, sess_options=sess_options, providers=["CPUExecutionProvider"]
                )
            else:
                sess_options.add_provider_for_devices([target_device], _plugin_provider_options(provider_options))
                sess = onnxrt.InferenceSession(model_path, sess_options=sess_options)
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


def run_auto_registered_provider_options_test(provider_options):
    require_cuda_plugin_ep()

    with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as tmp:
        model_path = tmp.name
    try:
        create_add_model(model_path)
        sess_options = _create_session_options()
        providers = [(CUDA_PLUGIN_EP_NAME, _plugin_provider_options(provider_options)), "CPUExecutionProvider"]
        sess = onnxrt.InferenceSession(model_path, sess_options=sess_options, providers=providers)

        active_providers = sess.get_providers()
        assigned_nodes, assignment_info = _get_assigned_nodes(sess, CUDA_PLUGIN_EP_NAME)
        if not assigned_nodes:
            print(
                f"FAILURE: {CUDA_PLUGIN_EP_NAME} was assigned no nodes. Providers: {active_providers}. "
                f"Assignments: {_format_assignment_summary(assignment_info)}"
            )
            return False

        a = np.random.rand(3, 2).astype(np.float32)
        b = np.random.rand(3, 2).astype(np.float32)
        res = sess.run(None, {"A": a, "B": b})
        np.testing.assert_allclose(res[0], a + b, rtol=1e-3, atol=1e-3)
        return True
    except Exception as e:
        print(f"FAIL ({e})")
        return False
    finally:
        if os.path.exists(model_path):
            os.remove(model_path)


def _expected_conv(inputs):
    return F.conv2d(torch.from_numpy(inputs["X"]), torch.from_numpy(inputs["W"]), padding=1).numpy()


_NHWC_CONFIG = {"ep.cuda.prefer_nhwc_layout": "1"}


def _assert_nhwc_domain_assigned(session, ep_name, expected_ops):
    """Assert that NHWC layout transformation occurred for the expected ops.

    The framework's NHWC layout transformer rewrites eligible ops to the internal NHWC domain
    and wraps them with Transpose nodes. We verify NHWC transformation by checking:
    1. If the assignment API surfaces NHWC-domain nodes, verify expected ops are present.
    2. Otherwise, fall back to checking that Transpose nodes were assigned (their presence
       indicates the layout transformer ran and the NHWC kernel was found).

    Args:
        session: An InferenceSession with graph assignment info enabled.
        ep_name: Name of the EP to check (e.g., CUDA_PLUGIN_EP_NAME).
        expected_ops: Set or list of op_type strings expected to have NHWC transformation.

    Returns:
        True if evidence of NHWC transformation is found. Raises AssertionError otherwise.
    """
    assigned_nodes, _ = _get_assigned_nodes(session, ep_name)

    # Check for NHWC-domain nodes directly (preferred when the API surfaces them).
    nhwc_domain = "com.ms.internal.nhwc"
    nhwc_ops_found = {n.op_type for n in assigned_nodes if n.domain == nhwc_domain}
    if nhwc_ops_found:
        missing = set(expected_ops) - nhwc_ops_found
        if missing:
            raise AssertionError(
                f"Expected NHWC-domain nodes for {sorted(missing)} but only found "
                f"{sorted(nhwc_ops_found)} in {ep_name} NHWC assignments."
            )
        return True

    # Fallback: the NHWC transformation inserts Transpose nodes around the target op.
    transpose_count = sum(1 for n in assigned_nodes if n.op_type == "Transpose")
    if transpose_count == 0:
        all_ops = [f"{n.domain or 'ai.onnx'}::{n.op_type}" for n in assigned_nodes]
        raise AssertionError(
            f"Expected NHWC layout transformation for {sorted(expected_ops)} but no Transpose "
            f"nodes were found in {ep_name} assignments. Assigned ops: {all_ops}. "
            f"This indicates the NHWC kernel was not found for the target op(s)."
        )
    return True


def _run_nhwc_model_test(target_device, op_name, model, feed_dict, expected_fn, nhwc_ops=None, rtol=1e-3, atol=1e-3):
    """Run an NHWC test: verify domain assignment and numerical correctness.

    Args:
        target_device: EP device to test on.
        op_name: Op type name (for display and default NHWC assertion).
        model: ONNX model proto.
        feed_dict: Input feed dictionary.
        expected_fn: Function(feed_dict) -> expected output(s).
        nhwc_ops: Set of op_types expected in NHWC domain (defaults to {op_name}).
        rtol: Relative tolerance for output comparison.
        atol: Absolute tolerance for output comparison.

    Returns:
        TEST_PASS or TEST_FAIL string.
    """
    if nhwc_ops is None:
        nhwc_ops = {op_name}
    with tempfile.NamedTemporaryFile(suffix=f"_{op_name}_nhwc.onnx", delete=False) as tmp:
        model_path = tmp.name
    try:
        save(model, model_path)
        sess_options = _create_session_options(_NHWC_CONFIG)
        sess_options.add_provider_for_devices([target_device], _plugin_provider_options())
        sess = onnxrt.InferenceSession(model_path, sess_options=sess_options)
        assigned_nodes, assignment_info = _get_assigned_nodes(sess, CUDA_PLUGIN_EP_NAME)
        if not assigned_nodes:
            print(
                f"{TEST_FAIL} ({CUDA_PLUGIN_EP_NAME} was assigned no nodes; "
                f"assignments={_format_assignment_summary(assignment_info)})"
            )
            return TEST_FAIL

        # Structural assertion: verify NHWC domain assignment
        _assert_nhwc_domain_assigned(sess, CUDA_PLUGIN_EP_NAME, nhwc_ops)

        res = sess.run(None, feed_dict)
        expected = expected_fn(feed_dict)
        if isinstance(expected, (list, tuple)):
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
        sess_options.add_provider_for_devices([target_device], _plugin_provider_options())
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


def _run_cpu_reference_model(model, feed_dict):
    """Run a model on CPU EP and return all outputs for reference comparisons."""
    with tempfile.NamedTemporaryFile(suffix="_cpu_reference.onnx", delete=False) as tmp:
        model_path = tmp.name
    try:
        save(model, model_path)
        sess = onnxrt.InferenceSession(model_path, providers=["CPUExecutionProvider"])
        return sess.run(None, feed_dict)
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

    @requires_cudnn
    def test_registration_conv(self):
        target_device = get_cuda_plugin_device()
        inputs = {
            "X": np.random.rand(1, 2, 4, 4).astype(np.float32),
            "W": np.random.rand(3, 2, 3, 3).astype(np.float32),
        }
        result = run_operator_test(target_device, create_conv_model, inputs, _expected_conv)
        self.assertTrue(result, "Conv plugin registration test failed")

    def test_registration_matmul_nbits_prepack(self):
        # Regression guard for the CUDA-EP-as-plugin pre-pack allocator bug: a fp16 MatMulNBits
        # weight is pre-packed during session initialization, which allocates scratch through the
        # EP-provided allocator. The plugin op-kernel adapter previously forwarded a null allocator
        # into PrePack, crashing session creation with an IAllocator::ValidateAllocator failure.
        target_device = get_cuda_plugin_device()

        # The fpA_intB pre-pack path only runs on SM >= 7.5. Skip deterministically on older GPUs
        # so the test does not silently pass without exercising the allocator-forwarding path.
        compute_capability = get_cuda_compute_capability(target_device)
        if compute_capability is None:
            self.skipTest("CUDA plugin EP device metadata did not include cuda_compute_capability")
        if compute_capability < FPA_INTB_MIN_COMPUTE_CAPABILITY:
            self.skipTest(
                "MatMulNBits fpA_intB pre-pack requires compute capability >= 7.5, but device reports "
                f"{compute_capability[0]}.{compute_capability[1]}"
            )

        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as tmp:
            model_path = tmp.name
        try:
            create_matmul_nbits_model(model_path)
            sess_options = _create_session_options()
            sess_options.add_provider_for_devices([target_device], _plugin_provider_options())

            # Force the fpA_intB path so weight pre-packing (MatMulNBits::PrePack_B) runs during
            # session creation, which is where the null-allocator crash used to happen. Given the
            # deterministic capability check above, session creation is expected to succeed, so any
            # exception here (including the "allocator != nullptr" assertion) is a genuine regression
            # and is deliberately propagated instead of skipped.
            with scoped_env("ORT_FPA_INTB_GEMM", "1"):
                sess = onnxrt.InferenceSession(model_path, sess_options=sess_options)

                assigned_nodes, assignment_info = _get_assigned_nodes(sess, CUDA_PLUGIN_EP_NAME)
                self.assertTrue(
                    assigned_nodes,
                    f"{CUDA_PLUGIN_EP_NAME} was assigned no nodes. "
                    f"Assignments: {_format_assignment_summary(assignment_info)}",
                )

                a = np.random.rand(2, 64).astype(np.float16)
                # Some devices/configs may still lack a valid fpA_intB tactic at runtime; skip only
                # for that known-unsupported case and re-raise anything else so real regressions
                # (e.g. incorrect kernels or the allocator assertion) are not hidden.
                try:
                    res = sess.run(None, {"A": a})
                except Exception as exc:
                    if is_fpa_intb_unsupported_error(exc):
                        self.skipTest(f"fpA_intB MatMulNBits not supported on this device: {exc}")
                    raise
                self.assertEqual(res[0].shape, (2, 64))
                self.assertTrue(np.isfinite(res[0].astype(np.float32)).all(), "MatMulNBits produced non-finite output")
        finally:
            if os.path.exists(model_path):
                os.remove(model_path)

    # ---- Provider options tests ----

    def test_provider_options_valid(self):
        result = run_provider_options_test({"device_id": "0", "use_tf32": "0"}, expect_plugin_provider=True)
        self.assertTrue(result, "Provider options with valid device_id/use_tf32 failed")

    @requires_cudnn
    def test_auto_registered_provider_options_valid(self):
        result = run_auto_registered_provider_options_test(
            {"device_id": "0", "ep.cuda.use_tf32": "0", "ep.cuda.prefer_nhwc_layout": "0"}
        )
        self.assertTrue(result, "Auto-registered provider options with prefixed CUDA options failed")

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
            sess_options = _create_session_options()
            sess_options.add_provider_for_devices([target_device], _plugin_provider_options({"device_id": "1"}))
            sess = onnxrt.InferenceSession(model_path, sess_options=sess_options)

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

    @requires_cudnn
    def test_nhwc_conv(self):
        target_device = get_cuda_plugin_device()
        inputs = {
            "X": np.random.rand(1, 2, 4, 4).astype(np.float32),
            "W": np.random.rand(3, 2, 3, 3).astype(np.float32),
        }
        result = run_operator_test(
            target_device,
            create_conv_model,
            inputs,
            _expected_conv,
            session_config=_NHWC_CONFIG,
            nhwc_ops={"Conv"},
        )
        self.assertTrue(result, "Conv (NHWC) plugin test failed")

    @requires_cudnn
    def test_nhwc_batch_normalization(self):
        target_device = get_cuda_plugin_device()
        inputs = {"X": np.random.rand(1, 3, 4, 4).astype(np.float32)}
        result = run_operator_test(
            target_device,
            create_batch_norm_model,
            inputs,
            _expected_batchnorm,
            session_config=_NHWC_CONFIG,
            nhwc_ops={"BatchNormalization"},
        )
        self.assertTrue(result, "BatchNormalization (NHWC) plugin test failed")

    @requires_cudnn
    def test_nhwc_maxpool(self):
        target_device = get_cuda_plugin_device()
        inputs = {"X": np.random.rand(1, 3, 4, 4).astype(np.float32)}
        result = run_operator_test(
            target_device,
            create_maxpool_model,
            inputs,
            lambda feed: F.max_pool2d(torch.from_numpy(feed["X"]), kernel_size=2, stride=2).numpy(),
            session_config=_NHWC_CONFIG,
            nhwc_ops={"MaxPool"},
        )
        self.assertTrue(result, "MaxPool (NHWC) plugin test failed")

    @requires_cudnn
    def test_nhwc_avgpool(self):
        target_device = get_cuda_plugin_device()
        inputs = {"X": np.random.rand(1, 3, 4, 4).astype(np.float32)}
        result = run_operator_test(
            target_device,
            create_avgpool_model,
            inputs,
            lambda feed: F.avg_pool2d(torch.from_numpy(feed["X"]), kernel_size=2, stride=2).numpy(),
            session_config=_NHWC_CONFIG,
            nhwc_ops={"AveragePool"},
        )
        self.assertTrue(result, "AveragePool (NHWC) plugin test failed")

    @requires_cudnn
    def test_nhwc_conv_transpose(self):
        target_device = get_cuda_plugin_device()
        # ConvTranspose: input [1,2,4,4], weight [2,3,3,3] -> output [1,3,6,6] with stride=2, padding=1, output_padding=1
        f_dtype = TensorProto.FLOAT
        node = helper.make_node(
            "ConvTranspose",
            ["X", "W"],
            ["Y"],
            strides=[2, 2],
            pads=[1, 1, 1, 1],
            output_padding=[1, 1],
            group=1,
        )
        graph = helper.make_graph(
            [node],
            "test-ConvTranspose",
            [
                helper.make_tensor_value_info("X", f_dtype, [1, 2, 4, 4]),
                helper.make_tensor_value_info("W", f_dtype, [2, 3, 3, 3]),
            ],
            [helper.make_tensor_value_info("Y", f_dtype, [1, 3, 6, 6])],
        )
        opset = OperatorSetIdProto()
        opset.version = 11
        model = helper.make_model(graph, opset_imports=[opset])
        x = np.random.rand(1, 2, 4, 4).astype(np.float32)
        w = np.random.rand(2, 3, 3, 3).astype(np.float32)

        def expected_fn(feed):
            return F.conv_transpose2d(
                torch.from_numpy(feed["X"]),
                torch.from_numpy(feed["W"]),
                stride=2,
                padding=1,
                output_padding=1,
            ).numpy()

        result = _run_nhwc_model_test(target_device, "ConvTranspose", model, {"X": x, "W": w}, expected_fn)
        self.assertEqual(result, TEST_PASS, "ConvTranspose (NHWC) plugin test failed")

    @requires_cudnn
    def test_nhwc_global_max_pool(self):
        target_device = get_cuda_plugin_device()
        f_dtype = TensorProto.FLOAT
        model = _make_simple_model(
            "GlobalMaxPool",
            [("X", f_dtype, [1, 3, 4, 4])],
            [("Y", f_dtype, [1, 3, 1, 1])],
            opset=12,
        )
        x = np.random.rand(1, 3, 4, 4).astype(np.float32)

        def expected_fn(feed):
            t = torch.from_numpy(feed["X"])
            return F.adaptive_max_pool2d(t, output_size=1).numpy()

        result = _run_nhwc_model_test(target_device, "GlobalMaxPool", model, {"X": x}, expected_fn)
        self.assertEqual(result, TEST_PASS, "GlobalMaxPool (NHWC) plugin test failed")

    @requires_cudnn
    def test_nhwc_global_average_pool(self):
        target_device = get_cuda_plugin_device()
        f_dtype = TensorProto.FLOAT
        model = _make_simple_model(
            "GlobalAveragePool",
            [("X", f_dtype, [1, 3, 4, 4])],
            [("Y", f_dtype, [1, 3, 1, 1])],
            opset=12,
        )
        x = np.random.rand(1, 3, 4, 4).astype(np.float32)

        def expected_fn(feed):
            t = torch.from_numpy(feed["X"])
            return F.adaptive_avg_pool2d(t, output_size=1).numpy()

        result = _run_nhwc_model_test(target_device, "GlobalAveragePool", model, {"X": x}, expected_fn)
        self.assertEqual(result, TEST_PASS, "GlobalAveragePool (NHWC) plugin test failed")

    def test_nhwc_depth_to_space(self):
        target_device = get_cuda_plugin_device()
        f_dtype = TensorProto.FLOAT
        # DepthToSpace: [1,4,2,2] -> [1,1,4,4] with blocksize=2
        model = _make_simple_model(
            "DepthToSpace",
            [("X", f_dtype, [1, 4, 2, 2])],
            [("Y", f_dtype, [1, 1, 4, 4])],
            attrs={"blocksize": 2, "mode": "DCR"},
            opset=13,
        )
        x = np.random.rand(1, 4, 2, 2).astype(np.float32)

        def expected_fn(feed):
            # DCR mode: depth, column, row
            t = feed["X"]  # [1, 4, 2, 2]
            b = 2
            n, c, h, w = t.shape
            t = t.reshape(n, b, b, c // (b * b), h, w)
            t = t.transpose(0, 3, 4, 1, 5, 2)  # [n, c/b^2, h, b, w, b]
            return t.reshape(n, c // (b * b), h * b, w * b)

        result = _run_nhwc_model_test(target_device, "DepthToSpace", model, {"X": x}, expected_fn)
        self.assertEqual(result, TEST_PASS, "DepthToSpace (NHWC) plugin test failed")

    def test_nhwc_space_to_depth(self):
        target_device = get_cuda_plugin_device()
        f_dtype = TensorProto.FLOAT
        # SpaceToDepth: [1,1,4,4] -> [1,4,2,2] with blocksize=2
        model = _make_simple_model(
            "SpaceToDepth",
            [("X", f_dtype, [1, 1, 4, 4])],
            [("Y", f_dtype, [1, 4, 2, 2])],
            attrs={"blocksize": 2},
            opset=13,
        )
        x = np.random.rand(1, 1, 4, 4).astype(np.float32)

        def expected_fn(feed):
            t = feed["X"]  # [1, 1, 4, 4]
            b = 2
            n, c, h, w = t.shape
            t = t.reshape(n, c, h // b, b, w // b, b)
            t = t.transpose(0, 3, 5, 1, 2, 4)  # [n, b, b, c, h/b, w/b]
            return t.reshape(n, c * b * b, h // b, w // b)

        result = _run_nhwc_model_test(target_device, "SpaceToDepth", model, {"X": x}, expected_fn)
        self.assertEqual(result, TEST_PASS, "SpaceToDepth (NHWC) plugin test failed")

    @requires_cudnn
    def test_nhwc_lrn(self):
        target_device = get_cuda_plugin_device()
        f_dtype = TensorProto.FLOAT
        # LRN: [1,3,4,4] with size=3, alpha=0.0001, beta=0.75, bias=1.0
        model = _make_simple_model(
            "LRN",
            [("X", f_dtype, [1, 3, 4, 4])],
            [("Y", f_dtype, [1, 3, 4, 4])],
            attrs={"size": 3, "alpha": 0.0001, "beta": 0.75, "bias": 1.0},
            opset=13,
        )
        x = np.random.rand(1, 3, 4, 4).astype(np.float32)

        def expected_fn(feed):
            t = torch.from_numpy(feed["X"])
            return F.local_response_norm(t, size=3, alpha=0.0001, beta=0.75, k=1.0).numpy()

        result = _run_nhwc_model_test(target_device, "LRN", model, {"X": x}, expected_fn)
        self.assertEqual(result, TEST_PASS, "LRN (NHWC) plugin test failed")

    def test_nhwc_grid_sample(self):
        target_device = get_cuda_plugin_device()
        f_dtype = TensorProto.FLOAT
        # GridSample: X [1,1,4,4], grid [1,3,3,2] -> Y [1,1,3,3]
        model = _make_simple_model(
            "GridSample",
            [("X", f_dtype, [1, 1, 4, 4]), ("grid", f_dtype, [1, 3, 3, 2])],
            [("Y", f_dtype, [1, 1, 3, 3])],
            attrs={"mode": "linear", "padding_mode": "zeros", "align_corners": 0},
            opset=20,
        )
        x = np.random.rand(1, 1, 4, 4).astype(np.float32)
        # Grid values in [-1, 1]
        grid = np.random.rand(1, 3, 3, 2).astype(np.float32) * 2 - 1

        def expected_fn(feed):
            t = torch.from_numpy(feed["X"])
            g = torch.from_numpy(feed["grid"])
            return F.grid_sample(t, g, mode="bilinear", padding_mode="zeros", align_corners=False).numpy()

        result = _run_nhwc_model_test(target_device, "GridSample", model, {"X": x, "grid": grid}, expected_fn)
        self.assertEqual(result, TEST_PASS, "GridSample (NHWC) plugin test failed")

    @requires_cudnn
    def test_nhwc_conv_with_resource_accounting(self):
        # Smoke test for the NHWC two-pass partitioning flow combined with the resource
        # accountant (session.resource_cuda_partitioning_settings). The NHWC layout
        # transform makes the CUDA EP claim Conv nodes tentatively on the first pass; the
        # budget for surviving nodes is committed only after the second pass. This guards
        # against regressions where dropped first-pass tags would leak phantom budget. With
        # a large limit, the Conv must still be claimed by the plugin and run correctly.
        target_device = get_cuda_plugin_device()
        inputs = {
            "X": np.random.rand(1, 2, 4, 4).astype(np.float32),
            "W": np.random.rand(3, 2, 3, 3).astype(np.float32),
        }
        # Large ad-hoc memory limit (1 GB in KB) with no stats file (trailing comma) so the
        # Conv comfortably fits and remains assigned to the plugin EP.
        session_config = {
            **_NHWC_CONFIG,
            "session.resource_cuda_partitioning_settings": "1048576,",
        }
        result = run_operator_test(
            target_device,
            create_conv_model,
            inputs,
            _expected_conv,
            session_config=session_config,
            nhwc_ops={"Conv"},
        )
        self.assertTrue(result, "Conv (NHWC + resource accounting) plugin test failed")

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

    def test_op_resize_antialias(self):
        target_device = get_cuda_plugin_device()
        f_dtype = TensorProto.FLOAT
        node = helper.make_node(
            "Resize",
            ["X", "", "scales"],
            ["Y"],
            mode="linear",
            antialias=1,
            coordinate_transformation_mode="half_pixel",
        )
        graph = helper.make_graph(
            [node],
            "test-Resize-antialias",
            [helper.make_tensor_value_info("X", f_dtype, [1, 1, 4, 4])],
            [helper.make_tensor_value_info("Y", f_dtype, [1, 1, 2, 2])],
        )
        opset = OperatorSetIdProto()
        opset.version = 18
        model = helper.make_model(graph, opset_imports=[opset])
        model.graph.initializer.append(helper.make_tensor("scales", TensorProto.FLOAT, [4], [1.0, 1.0, 0.5, 0.5]))
        x = np.random.rand(1, 1, 4, 4).astype(np.float32)

        def expected(feed):
            return _run_cpu_reference_model(model, feed)[0]

        result = _run_model_test(target_device, "Resize_antialias", model, {"X": x}, expected, atol=1e-4)
        self.assertEqual(result, TEST_PASS, "Resize antialias plugin op test failed")

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
        ]

        if not NO_CUDNN_PLUGIN_TEST:
            probe_specs.extend(
                [
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
            )

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
                sess_options.add_provider_for_devices([target_device], _plugin_provider_options())
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

    @requires_cudnn
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

    @requires_cudnn
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

    def test_op_log_softmax(self):
        target_device = get_cuda_plugin_device()
        model = _make_simple_model(
            "LogSoftmax",
            [("X", TensorProto.FLOAT, [2, 5])],
            [("Y", TensorProto.FLOAT, [2, 5])],
            attrs={"axis": 1},
            opset=13,
        )
        feed = {"X": np.random.rand(2, 5).astype(np.float32)}

        def expected(f):
            x = f["X"]
            shifted = x - np.max(x, axis=1, keepdims=True)
            return shifted - np.log(np.sum(np.exp(shifted), axis=1, keepdims=True))

        result = _run_model_test(target_device, "LogSoftmax", model, feed, expected)
        self.assertEqual(result, TEST_PASS, "LogSoftmax test failed")

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

    def test_op_argmin(self):
        target_device = get_cuda_plugin_device()
        model = _make_simple_model(
            "ArgMin",
            [("X", TensorProto.FLOAT, [3, 5])],
            [("Y", TensorProto.INT64, [3, 1])],
            attrs={"axis": 1, "keepdims": 1},
            opset=13,
        )
        feed = {"X": np.random.rand(3, 5).astype(np.float32)}
        result = _run_model_test(
            target_device, "ArgMin", model, feed, lambda f: np.argmin(f["X"], axis=1).reshape(3, 1)
        )
        self.assertEqual(result, TEST_PASS, "ArgMin test failed")

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

    @requires_cudnn
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

    @requires_cudnn
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
            [("Y", TensorProto.FLOAT, [3, 4, 1])],
            attrs={"axes": [2], "keepdims": 1},
            opset=13,
        )
        feed = {"X": np.random.rand(3, 4, 5).astype(np.float32)}
        result = _run_model_test(
            target_device, "ReduceMean", model, feed, lambda f: np.mean(f["X"], axis=2, keepdims=True)
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

    def _run_reduce_sum_integer_last_axis(self, onnx_dtype, np_dtype):
        # Mirrors the qwen attention_mask usage: a rank-2 integer ReduceSum over the last axis.
        # Integer ReduceSum takes a specialized path that does not use the float matrix fast path,
        # so without cuDNN it must fall back to the general native kernel. Covering it here keeps the
        # no-cuDNN CI honest for the exact layout that broke real models.
        target_device = get_cuda_plugin_device()
        model = _make_simple_model(
            "ReduceSum",
            [("X", onnx_dtype, [2, 8]), ("axes", TensorProto.INT64, [1])],
            [("Y", onnx_dtype, [2, 1])],
            attrs={"keepdims": 1},
            opset=13,
        )
        axes_init = helper.make_tensor("axes", TensorProto.INT64, [1], [1])
        model.graph.initializer.append(axes_init)
        feed = {"X": np.arange(16, dtype=np_dtype).reshape(2, 8)}
        return _run_model_test(target_device, "ReduceSum", model, feed, lambda f: np.sum(f["X"], axis=1, keepdims=True))

    def test_op_reduce_sum_int64_last_axis(self):
        result = self._run_reduce_sum_integer_last_axis(TensorProto.INT64, np.int64)
        self.assertEqual(result, TEST_PASS, "ReduceSum int64 test failed")

    def test_op_reduce_sum_int32_last_axis(self):
        result = self._run_reduce_sum_integer_last_axis(TensorProto.INT32, np.int32)
        self.assertEqual(result, TEST_PASS, "ReduceSum int32 test failed")

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

    def test_op_scatter_nd(self):
        target_device = get_cuda_plugin_device()
        model = _make_simple_model(
            "ScatterND",
            [
                ("data", TensorProto.FLOAT, [4, 4]),
                ("indices", TensorProto.INT64, [2, 1]),
                ("updates", TensorProto.FLOAT, [2, 4]),
            ],
            [("Y", TensorProto.FLOAT, [4, 4])],
            opset=16,
        )
        data = np.arange(16, dtype=np.float32).reshape(4, 4)
        indices = np.array([[0], [2]], dtype=np.int64)
        updates = np.array([[100, 101, 102, 103], [200, 201, 202, 203]], dtype=np.float32)
        feed = {"data": data, "indices": indices, "updates": updates}

        def expected(f):
            result = f["data"].copy()
            result[0] = f["updates"][0]
            result[2] = f["updates"][1]
            return result

        result = _run_model_test(target_device, "ScatterND", model, feed, expected)
        self.assertEqual(result, TEST_PASS, "ScatterND test failed")

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

    # ---- CUDA Graph capture/replay tests ----

    def _create_cuda_graph_session(self, model_path, extra_session_config=None, provider_options=None):
        """Create a session with CUDA graph capture enabled for the plugin EP."""
        sess_options = _create_session_options()
        sess_options.add_session_config_entry("ep.cuda.enable_cuda_graph", "1")
        if extra_session_config:
            for key, value in extra_session_config.items():
                sess_options.add_session_config_entry(key, value)
        provider_options = _plugin_provider_options({"enable_cuda_graph": "1", **(provider_options or {})})
        target_device = get_cuda_plugin_device_by_id(int(provider_options.get("device_id", "0")))
        sess_options.add_provider_for_devices([target_device], provider_options)
        return onnxrt.InferenceSession(model_path, sess_options=sess_options)

    def _setup_cuda_graph_io(self, session, input_shapes, output_shapes, device_id=0):
        """Pre-allocate GPU OrtValues and set up IOBinding for graph capture."""
        io_binding = session.io_binding()
        input_ort_values = {}
        output_ort_values = {}

        for inp in session.get_inputs():
            shape = input_shapes[inp.name]
            binding = _CudaOrtValueBinding(shape, np.float32, device_id)
            input_ort_values[inp.name] = binding
            io_binding.bind_ortvalue_input(inp.name, binding.ort_value)

        for out in session.get_outputs():
            shape = output_shapes[out.name]
            binding = _CudaOrtValueBinding(shape, np.float32, device_id)
            output_ort_values[out.name] = binding
            io_binding.bind_ortvalue_output(out.name, binding.ort_value)

        return io_binding, input_ort_values, output_ort_values

    def test_cuda_graph_capture_and_replay(self):
        """Test CUDA graph warmup → capture → replay with default arena allocator."""
        target_device = get_cuda_plugin_device()
        cuda_device_id = get_cuda_plugin_device_id(target_device)

        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as tmp:
            model_path = tmp.name
        try:
            create_matmul_model(model_path)
            session = self._create_cuda_graph_session(model_path)

            assigned_nodes, assignment_info = _get_assigned_nodes(session, CUDA_PLUGIN_EP_NAME)
            self.assertTrue(
                assigned_nodes,
                f"{CUDA_PLUGIN_EP_NAME} was assigned no nodes. "
                f"Assignments: {_format_assignment_summary(assignment_info)}",
            )

            input_shapes = {"A": [3, 4], "B": [4, 5]}
            output_shapes = {"Y": [3, 5]}
            io_binding, input_vals, output_vals = self._setup_cuda_graph_io(
                session, input_shapes, output_shapes, cuda_device_id
            )

            # Set deterministic input data.
            rng = np.random.default_rng(0)
            a = rng.random((3, 4), dtype=np.float32)
            b = rng.random((4, 5), dtype=np.float32)
            input_vals["A"].update_inplace(a)
            input_vals["B"].update_inplace(b)

            # First run: ORT handles warmup + capture + first replay automatically.
            session.run_with_iobinding(io_binding)
            result = output_vals["Y"].numpy()
            np.testing.assert_allclose(result, a @ b, rtol=1e-3, atol=1e-3)

            # Second run: should take the graph replay fast path.
            session.run_with_iobinding(io_binding)
            result = output_vals["Y"].numpy()
            np.testing.assert_allclose(result, a @ b, rtol=1e-3, atol=1e-3)
        finally:
            if os.path.exists(model_path):
                os.remove(model_path)

    def test_cuda_graph_replay_with_updated_input(self):
        """Test that CUDA graph replay produces correct results after in-place input update."""
        target_device = get_cuda_plugin_device()
        cuda_device_id = get_cuda_plugin_device_id(target_device)

        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as tmp:
            model_path = tmp.name
        try:
            create_matmul_model(model_path)
            session = self._create_cuda_graph_session(model_path)

            input_shapes = {"A": [3, 4], "B": [4, 5]}
            output_shapes = {"Y": [3, 5]}
            io_binding, input_vals, output_vals = self._setup_cuda_graph_io(
                session, input_shapes, output_shapes, cuda_device_id
            )

            # Initial data.
            a1 = np.random.rand(3, 4).astype(np.float32)
            b1 = np.random.rand(4, 5).astype(np.float32)
            input_vals["A"].update_inplace(a1)
            input_vals["B"].update_inplace(b1)

            # First run: warmup + capture + replay.
            session.run_with_iobinding(io_binding)
            np.testing.assert_allclose(output_vals["Y"].numpy(), a1 @ b1, rtol=1e-3, atol=1e-3)

            # Update inputs in-place (same shape, different data) and replay.
            a2 = np.random.rand(3, 4).astype(np.float32) * 10
            b2 = np.random.rand(4, 5).astype(np.float32) * 10
            input_vals["A"].update_inplace(a2)
            input_vals["B"].update_inplace(b2)
            session.run_with_iobinding(io_binding)
            np.testing.assert_allclose(output_vals["Y"].numpy(), a2 @ b2, rtol=1e-3, atol=1e-3)
        finally:
            if os.path.exists(model_path):
                os.remove(model_path)

    def test_cuda_graph_with_mempool(self):
        """Test CUDA graph capture/replay with CUDA native mempool allocator."""
        target_device = get_cuda_plugin_device()
        cuda_device_id = get_cuda_plugin_device_id(target_device)

        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as tmp:
            model_path = tmp.name
        try:
            create_matmul_model(model_path)
            try:
                session = self._create_cuda_graph_session(
                    model_path,
                    extra_session_config={"ep.cuda.arena.use_cuda_mempool": "1"},
                )
            except Exception as exc:
                if is_cuda_mempool_unsupported_error(exc):
                    self.skipTest("CUDA mempool is not supported on this device/driver configuration")
                raise

            assigned_nodes, assignment_info = _get_assigned_nodes(session, CUDA_PLUGIN_EP_NAME)
            self.assertTrue(
                assigned_nodes,
                f"{CUDA_PLUGIN_EP_NAME} was assigned no nodes. "
                f"Assignments: {_format_assignment_summary(assignment_info)}",
            )

            input_shapes = {"A": [3, 4], "B": [4, 5]}
            output_shapes = {"Y": [3, 5]}
            io_binding, input_vals, output_vals = self._setup_cuda_graph_io(
                session, input_shapes, output_shapes, cuda_device_id
            )

            a = np.random.rand(3, 4).astype(np.float32)
            b = np.random.rand(4, 5).astype(np.float32)
            input_vals["A"].update_inplace(a)
            input_vals["B"].update_inplace(b)

            # First run: warmup + capture + replay via mempool path.
            session.run_with_iobinding(io_binding)
            np.testing.assert_allclose(output_vals["Y"].numpy(), a @ b, rtol=1e-3, atol=1e-3)

            # Replay fast path.
            session.run_with_iobinding(io_binding)
            np.testing.assert_allclose(output_vals["Y"].numpy(), a @ b, rtol=1e-3, atol=1e-3)

            # Update and replay.
            a2 = np.random.rand(3, 4).astype(np.float32) * 5
            b2 = np.random.rand(4, 5).astype(np.float32) * 5
            input_vals["A"].update_inplace(a2)
            input_vals["B"].update_inplace(b2)
            session.run_with_iobinding(io_binding)
            np.testing.assert_allclose(output_vals["Y"].numpy(), a2 @ b2, rtol=1e-3, atol=1e-3)
        finally:
            if os.path.exists(model_path):
                os.remove(model_path)

    def test_cuda_graph_annotation_id(self):
        """Test multiple CUDA graphs captured with different annotation IDs and input shapes.

        This simulates the common use case where an application captures separate
        graphs for different input shapes (e.g., different batch sizes or sequence
        lengths) and replays the appropriate graph based on a runtime annotation ID.
        """
        target_device = get_cuda_plugin_device()
        cuda_device_id = get_cuda_plugin_device_id(target_device)

        # Build a MatMul model with dynamic dimensions so different shapes can be used.
        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as tmp:
            model_path = tmp.name
        try:
            node_def = helper.make_node("MatMul", ["A", "B"], ["Y"])
            graph_def = helper.make_graph(
                [node_def],
                "test-matmul-dynamic",
                [
                    helper.make_tensor_value_info("A", TensorProto.FLOAT, ["M", "K"]),
                    helper.make_tensor_value_info("B", TensorProto.FLOAT, ["K", "N"]),
                ],
                [helper.make_tensor_value_info("Y", TensorProto.FLOAT, ["M", "N"])],
            )
            model_def = _make_released_opset_model(graph_def)
            save(model_def, model_path)

            session = self._create_cuda_graph_session(model_path)

            # --- Graph 1: shape [2, 3] @ [3, 4] ---
            shapes1_in = {"A": [2, 3], "B": [3, 4]}
            shapes1_out = {"Y": [2, 4]}
            io1, iv1, ov1 = self._setup_cuda_graph_io(session, shapes1_in, shapes1_out, cuda_device_id)

            a1 = np.random.rand(2, 3).astype(np.float32)
            b1 = np.random.rand(3, 4).astype(np.float32)
            iv1["A"].update_inplace(a1)
            iv1["B"].update_inplace(b1)

            ro1 = onnxrt.RunOptions()
            ro1.add_run_config_entry("gpu_graph_id", "1")
            session.run_with_iobinding(io1, ro1)
            np.testing.assert_allclose(ov1["Y"].numpy(), a1 @ b1, rtol=1e-3, atol=1e-3)

            # --- Graph 2: different shape [4, 5] @ [5, 6] ---
            shapes2_in = {"A": [4, 5], "B": [5, 6]}
            shapes2_out = {"Y": [4, 6]}
            io2, iv2, ov2 = self._setup_cuda_graph_io(session, shapes2_in, shapes2_out, cuda_device_id)

            a2 = np.random.rand(4, 5).astype(np.float32)
            b2 = np.random.rand(5, 6).astype(np.float32)
            iv2["A"].update_inplace(a2)
            iv2["B"].update_inplace(b2)

            ro2 = onnxrt.RunOptions()
            ro2.add_run_config_entry("gpu_graph_id", "2")
            session.run_with_iobinding(io2, ro2)
            np.testing.assert_allclose(ov2["Y"].numpy(), a2 @ b2, rtol=1e-3, atol=1e-3)

            # --- Replay graph 1 with updated data (same shape) ---
            a3 = np.random.rand(2, 3).astype(np.float32) * 7
            b3 = np.random.rand(3, 4).astype(np.float32) * 7
            iv1["A"].update_inplace(a3)
            iv1["B"].update_inplace(b3)
            session.run_with_iobinding(io1, ro1)
            np.testing.assert_allclose(ov1["Y"].numpy(), a3 @ b3, rtol=1e-3, atol=1e-3)

            # --- Replay graph 2 with updated data (same shape) ---
            a4 = np.random.rand(4, 5).astype(np.float32) * 3
            b4 = np.random.rand(5, 6).astype(np.float32) * 3
            iv2["A"].update_inplace(a4)
            iv2["B"].update_inplace(b4)
            session.run_with_iobinding(io2, ro2)
            np.testing.assert_allclose(ov2["Y"].numpy(), a4 @ b4, rtol=1e-3, atol=1e-3)
        finally:
            if os.path.exists(model_path):
                os.remove(model_path)

    def test_cuda_graph_second_device(self):
        """Test CUDA graph capture/replay on a non-default plugin device."""
        plugin_devices = get_cuda_plugin_devices()
        if len(plugin_devices) < 2:
            self.skipTest("Multi-GPU CUDA graph test requires at least two plugin devices")

        target_device = get_cuda_plugin_device_by_id(1)
        cuda_device_id = get_cuda_plugin_device_id(target_device)

        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as tmp:
            model_path = tmp.name
        try:
            create_matmul_model(model_path)
            session = self._create_cuda_graph_session(model_path, provider_options={"device_id": "1"})

            provider_options = session.get_provider_options()
            self.assertEqual(
                provider_options[CUDA_PLUGIN_EP_NAME].get("device_id"),
                "1",
                f"Expected provider option device_id=1, got {provider_options[CUDA_PLUGIN_EP_NAME]}",
            )

            assigned_nodes, assignment_info = _get_assigned_nodes(session, CUDA_PLUGIN_EP_NAME)
            self.assertTrue(
                assigned_nodes,
                f"{CUDA_PLUGIN_EP_NAME} was assigned no nodes. "
                f"Assignments: {_format_assignment_summary(assignment_info)}",
            )

            input_shapes = {"A": [3, 4], "B": [4, 5]}
            output_shapes = {"Y": [3, 5]}
            io_binding, input_vals, output_vals = self._setup_cuda_graph_io(
                session, input_shapes, output_shapes, cuda_device_id
            )

            a = np.random.rand(3, 4).astype(np.float32)
            b = np.random.rand(4, 5).astype(np.float32)
            input_vals["A"].update_inplace(a)
            input_vals["B"].update_inplace(b)

            session.run_with_iobinding(io_binding)
            np.testing.assert_allclose(output_vals["Y"].numpy(), a @ b, rtol=1e-3, atol=1e-3)

            a2 = np.random.rand(3, 4).astype(np.float32) * 11
            b2 = np.random.rand(4, 5).astype(np.float32) * 11
            input_vals["A"].update_inplace(a2)
            input_vals["B"].update_inplace(b2)
            session.run_with_iobinding(io_binding)
            np.testing.assert_allclose(output_vals["Y"].numpy(), a2 @ b2, rtol=1e-3, atol=1e-3)
        finally:
            if os.path.exists(model_path):
                os.remove(model_path)

    def test_cuda_graph_add_model(self):
        """Test CUDA graph capture/replay with a simple Add model (arena-backed)."""
        target_device = get_cuda_plugin_device()
        cuda_device_id = get_cuda_plugin_device_id(target_device)

        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as tmp:
            model_path = tmp.name
        try:
            create_add_model(model_path)
            session = self._create_cuda_graph_session(model_path)

            assigned_nodes, assignment_info = _get_assigned_nodes(session, CUDA_PLUGIN_EP_NAME)
            self.assertTrue(
                assigned_nodes,
                f"{CUDA_PLUGIN_EP_NAME} was assigned no nodes. "
                f"Assignments: {_format_assignment_summary(assignment_info)}",
            )

            input_shapes = {"A": [3, 2], "B": [3, 2]}
            output_shapes = {"Y": [3, 2]}
            io_binding, input_vals, output_vals = self._setup_cuda_graph_io(
                session, input_shapes, output_shapes, cuda_device_id
            )

            a = np.random.rand(3, 2).astype(np.float32)
            b = np.random.rand(3, 2).astype(np.float32)
            input_vals["A"].update_inplace(a)
            input_vals["B"].update_inplace(b)

            # Warmup + capture + replay.
            session.run_with_iobinding(io_binding)
            np.testing.assert_allclose(output_vals["Y"].numpy(), a + b, rtol=1e-3, atol=1e-3)

            # Replay with updated data.
            a2 = np.random.rand(3, 2).astype(np.float32) * 100
            b2 = np.random.rand(3, 2).astype(np.float32) * 100
            input_vals["A"].update_inplace(a2)
            input_vals["B"].update_inplace(b2)
            session.run_with_iobinding(io_binding)
            np.testing.assert_allclose(output_vals["Y"].numpy(), a2 + b2, rtol=1e-3, atol=1e-3)
        finally:
            if os.path.exists(model_path):
                os.remove(model_path)

    # ---- IOBinding / Sync tests ----

    def test_iobinding_add(self):
        """Run a simple Add model using IOBinding to exercise the EP Sync path.

        Binding CPU inputs forces ORT to stage host-to-device copies on the
        plugin sync stream before kernel execution begins.
        """
        target_device = get_cuda_plugin_device()
        cuda_device_id = get_cuda_plugin_device_id(target_device)

        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as tmp:
            model_path = tmp.name
        try:
            create_add_model(model_path)
            sess_options = _create_session_options()
            sess_options.add_provider_for_devices([target_device], _plugin_provider_options())
            sess = onnxrt.InferenceSession(model_path, sess_options=sess_options)

            assigned_nodes, assignment_info = _get_assigned_nodes(sess, CUDA_PLUGIN_EP_NAME)
            self.assertTrue(
                assigned_nodes,
                f"{CUDA_PLUGIN_EP_NAME} was assigned no nodes. "
                f"Assignments: {_format_assignment_summary(assignment_info)}",
            )

            a = np.random.rand(3, 2).astype(np.float32)
            b = np.random.rand(3, 2).astype(np.float32)

            io_binding = sess.io_binding()
            io_binding.bind_cpu_input("A", a)
            io_binding.bind_cpu_input("B", b)
            io_binding.bind_output("Y", "cuda", cuda_device_id)

            # Exercise the EP Sync callback explicitly. run_with_iobinding()
            # alone does not call SynchronizeInputs().
            io_binding.synchronize_inputs()
            sess.run_with_iobinding(io_binding)

            # No explicit synchronize_outputs() is needed here because
            # copy_outputs_to_cpu() performs the blocking device-to-host copy.
            result = io_binding.copy_outputs_to_cpu()[0]
            np.testing.assert_allclose(result, a + b, rtol=1e-3, atol=1e-3)
        finally:
            if os.path.exists(model_path):
                os.remove(model_path)

    def test_iobinding_matmul(self):
        """Run a MatMul model using IOBinding to exercise the EP Sync path."""
        target_device = get_cuda_plugin_device()
        cuda_device_id = get_cuda_plugin_device_id(target_device)

        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as tmp:
            model_path = tmp.name
        try:
            create_matmul_model(model_path)
            sess_options = _create_session_options()
            sess_options.add_provider_for_devices([target_device], _plugin_provider_options())
            sess = onnxrt.InferenceSession(model_path, sess_options=sess_options)

            assigned_nodes, assignment_info = _get_assigned_nodes(sess, CUDA_PLUGIN_EP_NAME)
            self.assertTrue(
                assigned_nodes,
                f"{CUDA_PLUGIN_EP_NAME} was assigned no nodes. "
                f"Assignments: {_format_assignment_summary(assignment_info)}",
            )

            a = np.random.rand(3, 4).astype(np.float32)
            b = np.random.rand(4, 5).astype(np.float32)

            io_binding = sess.io_binding()
            io_binding.bind_cpu_input("A", a)
            io_binding.bind_cpu_input("B", b)
            io_binding.bind_output("Y", "cuda", cuda_device_id)

            # Exercise the EP Sync callback explicitly. run_with_iobinding()
            # alone does not call SynchronizeInputs().
            io_binding.synchronize_inputs()
            sess.run_with_iobinding(io_binding)

            # No explicit synchronize_outputs() is needed here because
            # copy_outputs_to_cpu() performs the blocking device-to-host copy.
            result = io_binding.copy_outputs_to_cpu()[0]
            np.testing.assert_allclose(result, a @ b, rtol=1e-3, atol=1e-3)
        finally:
            if os.path.exists(model_path):
                os.remove(model_path)

    # ---- Profiling tests ----

    def _run_profiling_test(self):
        """Run a model with session-level profiling enabled and verify the JSON output.

        Validates that profiling produces a valid JSON file with standard event
        fields. GPU Kernel event validation (CUPTI) is handled by the C++ test
        (cuda_plugin_profiling_test.cc) which can directly probe CUPTI availability.
        """
        target_device = get_cuda_plugin_device()

        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as tmp:
            model_path = tmp.name
        profile_file = None
        try:
            create_matmul_model(model_path)
            sess_options = _create_session_options()
            sess_options.add_provider_for_devices([target_device], _plugin_provider_options())

            profile_prefix = os.path.join(tempfile.gettempdir(), "cuda_plugin_ep_profiling_test")
            sess_options.enable_profiling = True
            sess_options.profile_file_prefix = profile_prefix

            sess = onnxrt.InferenceSession(model_path, sess_options=sess_options)

            assigned_nodes, assignment_info = _get_assigned_nodes(sess, CUDA_PLUGIN_EP_NAME)
            self.assertTrue(
                assigned_nodes,
                f"{CUDA_PLUGIN_EP_NAME} was assigned no nodes. "
                f"Assignments: {_format_assignment_summary(assignment_info)}",
            )

            a = np.random.rand(3, 4).astype(np.float32)
            b = np.random.rand(4, 5).astype(np.float32)
            sess.run(None, {"A": a, "B": b})

            profile_file = sess.end_profiling()
            self.assertTrue(profile_file, "No profile file returned")
            self.assertTrue(os.path.exists(profile_file), f"Profile file not found: {profile_file}")

            with open(profile_file) as f:
                profile_data = json.load(f)

            self.assertIsInstance(profile_data, list)
            self.assertGreater(len(profile_data), 0, "Profile JSON is empty")

            # Every event entry must have standard tracing fields.
            required_keys = {"pid", "dur", "ts", "ph", "name", "args"}
            for entry in profile_data:
                if not isinstance(entry, dict):
                    continue
                if "name" not in entry:
                    continue
                for key in required_keys:
                    self.assertIn(key, entry, f"Missing '{key}' in profile entry: {entry}")

            # If GPU kernel events are present, validate their metadata.
            kernel_events = [e for e in profile_data if isinstance(e, dict) and e.get("cat") == "Kernel"]
            if kernel_events:
                for event in kernel_events:
                    self.assertIn("ts", event)
                    self.assertIn("dur", event)
                    self.assertGreaterEqual(event["dur"], 0)
                    args = event.get("args", {})
                    self.assertIn("stream", args, f"GPU kernel event missing 'stream': {event}")
                    self.assertIn("block_x", args, f"GPU kernel event missing 'block_x': {event}")
            else:
                print("Note: No GPU Kernel events in profile (CUPTI may not be available).")

        finally:
            if os.path.exists(model_path):
                os.remove(model_path)
            if profile_file and os.path.exists(profile_file):
                os.remove(profile_file)

    def test_session_profiling(self):
        """Verify session-level profiling produces valid output with the CUDA Plugin EP."""
        self._run_profiling_test()


if __name__ == "__main__":
    unittest.main()
