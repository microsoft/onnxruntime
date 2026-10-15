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


def _fake_int4(values):
    groups = values.float().reshape(*values.shape[:-1], -1, 64)
    scales = groups.abs().amax(-1, keepdim=True) / 7
    safe_scales = torch.where(scales == 0, torch.ones_like(scales), scales)
    return (torch.round(groups / safe_scales).clamp(-7, 7) * scales).reshape_as(values).to(values.dtype)


def _initializer(name, values):
    if values.dtype == torch.uint8:
        return numpy_helper.from_array(values.numpy(), name)
    tensor = numpy_helper.from_array(values.view(torch.uint16).numpy(), name)
    tensor.data_type = TensorProto.BFLOAT16 if values.dtype == torch.bfloat16 else TensorProto.FLOAT16
    return tensor


def _case(rows, dtype, fake_activations=True):
    experts, hidden, inter = 512, 128, 128
    generator = torch.Generator().manual_seed(42)
    codes = [
        torch.randint(0, 4, (experts, inter * 2, hidden), generator=generator, dtype=torch.uint8),
        torch.randint(0, 4, (experts, hidden, inter), generator=generator, dtype=torch.uint8),
    ]
    codes[0][0, :128] = 2
    arrays = {}
    matrices = []
    for index, code in enumerate(codes, 1):
        packed = code[..., 0::4] | (code[..., 1::4] << 2) | (code[..., 2::4] << 4) | (code[..., 3::4] << 6)
        scales = (torch.rand((*code.shape[:-1], code.shape[-1] // 64), generator=generator) * 0.1 - 0.05).to(dtype)
        arrays[f"w{index}"] = packed
        arrays[f"s{index}"] = scales
        arrays[f"b{index}"] = (torch.randn(code.shape[:-1], generator=generator) * 0.01).to(dtype)
        matrices.append(
            ((code[:10].float() - 2).reshape(10, code.shape[1], -1, 64) * scales[:10].float().unsqueeze(-1)).reshape(
                10, code.shape[1], code.shape[2]
            )
        )
    arrays["b1"][0, :128] = 0
    element_type = TensorProto.BFLOAT16 if dtype == torch.bfloat16 else TensorProto.FLOAT16
    node = helper.make_node(
        "QMoE",
        ["x", "router", "w1", "s1", "b1", "w2", "s2", "b2"],
        ["y"],
        domain="com.microsoft",
        expert_weight_bits=2,
        block_size=64,
        weights_prepacked=0,
        quant_type="int",
        k=10,
        activation_type="swiglu",
        swiglu_fusion=1,
        normalize_routing_weights=1,
    )
    graph = helper.make_graph(
        [node],
        "int2_fake_int4",
        [
            helper.make_tensor_value_info("x", element_type, [rows, hidden]),
            helper.make_tensor_value_info("router", element_type, [rows, experts]),
        ],
        [helper.make_tensor_value_info("y", element_type, [rows, hidden])],
        [_initializer(name, values) for name, values in arrays.items()],
    )
    model = helper.make_model(
        graph, opset_imports=[helper.make_opsetid("", 21), helper.make_opsetid("com.microsoft", 1)]
    )
    model.ir_version = 10
    x = (torch.randn((rows, hidden), generator=generator) * 0.2).to(dtype)
    x[0, :64] = 0
    x[0, 64:68] = torch.tensor([7, 0.5, 1.5, -0.5], dtype=dtype)
    router = torch.full((rows, experts), -100, dtype=dtype)
    router[:, :10] = torch.linspace(-1, 1, 10).to(dtype)
    routing = torch.softmax(router[:, :10].float(), -1)
    expected = torch.zeros((rows, hidden))
    quantized_x = _fake_int4(x) if fake_activations else x
    for expert in range(10):
        gate_up = (quantized_x.float() @ matrices[0][expert].T).to(dtype).float()
        gate_up += arrays["b1"][expert].float()
        activated = (torch.nn.functional.silu(gate_up[:, 0::2]) * gate_up[:, 1::2]).to(dtype)
        down_input = _fake_int4(activated) if fake_activations else activated
        down = (down_input.float() @ matrices[1][expert].T).to(dtype).float()
        expected += (down + arrays["b2"][expert].float()) * routing[:, expert : expert + 1]
    return model, {"x": x, "router": router}, expected.to(dtype).float().numpy(), element_type


@pytest.mark.skipif("CUDAExecutionProvider" not in ort.get_available_providers(), reason="CUDA required")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("rows", [1, 3, 32])
def test_int2_fake_int4_reference_and_graph(monkeypatch, dtype, rows):
    model, feeds, reference, element_type = _case(rows, dtype)
    monkeypatch.setenv("ORT_QMOE_INT2_LARGE_EXPERTS", "1")
    monkeypatch.setenv("ORT_ENABLE_QMOE_INT2_PREFILL", "1")
    monkeypatch.setenv("ORT_QMOE_INT2_FAKE_INT4_ACTIVATIONS", "1")
    options = ort.SessionOptions()
    options.add_session_config_entry("ep.cuda.qmoe_int_dequant_max_scratch_bytes", "1")
    session = ort.InferenceSession(
        model.SerializeToString(),
        sess_options=options,
        providers=[("CUDAExecutionProvider", {"enable_cuda_graph": "1"})],
    )
    device_feeds = {name: value.cuda() for name, value in feeds.items()}
    output = torch.empty((rows, 128), dtype=dtype, device="cuda")
    binding = session.io_binding()
    for name, value in device_feeds.items():
        binding.bind_input(name, "cuda", 0, element_type, tuple(value.shape), value.data_ptr())
    binding.bind_output("y", "cuda", 0, element_type, tuple(output.shape), output.data_ptr())
    torch.cuda.synchronize()
    session.run_with_iobinding(binding)
    torch.cuda.synchronize()
    captured = output.float().cpu().numpy().copy()
    session.run_with_iobinding(binding)
    torch.cuda.synchronize()
    np.testing.assert_array_equal(captured, output.float().cpu().numpy())
    assert np.isfinite(captured).all()
    np.testing.assert_allclose(captured, reference, atol=0.0005 if dtype == torch.float16 else 0.003, rtol=0.03)
    unquantized_reference = _case(rows, dtype, fake_activations=False)[2]
    assert np.max(np.abs(captured - unquantized_reference)) > 0.004
    np.testing.assert_array_equal(device_feeds["x"].cpu().float().numpy(), feeds["x"].float().numpy())


@pytest.mark.skipif("CUDAExecutionProvider" not in ort.get_available_providers(), reason="CUDA required")
def test_int2_fake_int4_rejects_unquantized_fallback(monkeypatch):
    model, feeds, _, _ = _case(1, torch.float16)
    monkeypatch.setenv("ORT_QMOE_INT2_FAKE_INT4_ACTIVATIONS", "1")
    monkeypatch.setenv("ORT_ENABLE_QMOE_INT2_PREFILL", "1")
    monkeypatch.setenv("ORT_QMOE_INT2_LARGE_EXPERTS", "0")
    session = ort.InferenceSession(model.SerializeToString(), providers=["CUDAExecutionProvider"])
    with pytest.raises(ort.capi.onnxruntime_pybind11_state.Fail, match="refusing an unquantized fallback"):
        session.run(None, {name: value.numpy() for name, value in feeds.items()})


@pytest.mark.skipif("CUDAExecutionProvider" not in ort.get_available_providers(), reason="CUDA required")
def test_int2_fake_int4_rejects_disabled_grouped_path(monkeypatch):
    model, _, _, _ = _case(1, torch.float16)
    monkeypatch.setenv("ORT_QMOE_INT2_FAKE_INT4_ACTIVATIONS", "1")
    monkeypatch.setenv("ORT_ENABLE_QMOE_INT2_PREFILL", "0")
    with pytest.raises(
        (ort.capi.onnxruntime_pybind11_state.RuntimeException, ort.capi.onnxruntime_pybind11_state.Fail),
        match="INT4 activation simulation requires ORT_ENABLE_QMOE_INT2_PREFILL",
    ):
        ort.InferenceSession(model.SerializeToString(), providers=["CUDAExecutionProvider"])
