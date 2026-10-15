# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import numpy as np
import pytest
import torch
from onnx import TensorProto, helper, numpy_helper

import onnxruntime as ort

pytestmark = pytest.mark.skipif(
    "CUDAExecutionProvider" not in ort.get_available_providers()
    or not torch.cuda.is_available()
    or torch.cuda.get_device_capability()[0] < 8,
    reason="CUDA SM80+ required",
)


@pytest.mark.skipif("CUDAExecutionProvider" not in ort.get_available_providers(), reason="CUDA required")
@pytest.mark.parametrize("block_size", [64, 128])
@pytest.mark.parametrize("rows", [1, 3, 32])
def test_int2_large_experts_packed_graph_parity(monkeypatch, block_size, rows):
    monkeypatch.setenv("ORT_QMOE_INT2_FAKE_INT4_ACTIVATIONS", "0")
    experts, hidden, inter = 512, 128, 128
    rng = np.random.default_rng(42)
    arrays = {
        "w1": rng.integers(0, 256, (experts, inter * 2, hidden // 4), dtype=np.uint8),
        "w2": rng.integers(0, 256, (experts, hidden, inter // 4), dtype=np.uint8),
        "s1": rng.uniform(-0.02, 0.02, (experts, inter * 2, hidden // block_size)).astype(np.float16),
        "s2": rng.uniform(-0.02, 0.02, (experts, hidden, inter // block_size)).astype(np.float16),
        "b1": rng.normal(0, 0.01, (experts, inter * 2)).astype(np.float16),
        "b2": rng.normal(0, 0.01, (experts, hidden)).astype(np.float16),
    }
    node = helper.make_node(
        "QMoE",
        ["x", "router", "w1", "s1", "b1", "w2", "s2", "b2"],
        ["y"],
        domain="com.microsoft",
        expert_weight_bits=2,
        block_size=block_size,
        weights_prepacked=0,
        quant_type="int",
        k=10,
        activation_type="swiglu",
        swiglu_fusion=1,
        normalize_routing_weights=1,
    )
    graph = helper.make_graph(
        [node],
        "int2_large_experts",
        [
            helper.make_tensor_value_info("x", TensorProto.FLOAT16, [rows, hidden]),
            helper.make_tensor_value_info("router", TensorProto.FLOAT16, [rows, experts]),
        ],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT16, [rows, hidden])],
        [numpy_helper.from_array(value, name) for name, value in arrays.items()],
    )
    model = helper.make_model(
        graph, opset_imports=[helper.make_opsetid("", 21), helper.make_opsetid("com.microsoft", 1)]
    )
    model.ir_version = 10
    feeds = {
        "x": rng.normal(size=(rows, hidden)).astype(np.float16),
        "router": rng.normal(size=(rows, experts)).astype(np.float16),
    }
    monkeypatch.setenv("ORT_ENABLE_QMOE_INT2_PREFILL", "1")
    monkeypatch.setenv("ORT_QMOE_INT2_LARGE_EXPERTS", "0")
    reference_session = ort.InferenceSession(model.SerializeToString(), providers=["CUDAExecutionProvider"])
    reference = reference_session.run(None, feeds)[0]
    del reference_session
    monkeypatch.setenv("ORT_QMOE_INT2_LARGE_EXPERTS", "1")
    options = ort.SessionOptions()
    options.add_session_config_entry("ep.cuda.qmoe_int_dequant_max_scratch_bytes", "1")
    session = ort.InferenceSession(
        model.SerializeToString(),
        sess_options=options,
        providers=[("CUDAExecutionProvider", {"enable_cuda_graph": "1"})],
    )
    values = {name: ort.OrtValue.ortvalue_from_numpy(data, "cuda", 0) for name, data in feeds.items()}
    binding = session.io_binding()
    for name, value in values.items():
        binding.bind_ortvalue_input(name, value)
    binding.bind_output("y", "cuda", 0)
    session.run_with_iobinding(binding)
    captured = binding.copy_outputs_to_cpu()[0]
    session.run_with_iobinding(binding)
    replayed = binding.copy_outputs_to_cpu()[0]
    assert captured.shape == (rows, hidden)
    assert np.isfinite(captured).all()
    np.testing.assert_array_equal(captured, replayed)
    np.testing.assert_allclose(captured, reference, atol=0.005, rtol=0.02)
