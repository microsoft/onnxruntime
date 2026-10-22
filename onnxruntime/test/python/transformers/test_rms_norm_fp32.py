# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Exact FP32 block RMSNorm arithmetic with FP16 input/output storage."""

import numpy as np
import pytest
from onnx import TensorProto, helper

import onnxruntime as ort


def make_model(rows, fused, epsilon=1e-6, width=3072):
    inputs = [
        helper.make_tensor_value_info("x", TensorProto.FLOAT16, [rows, width]),
        helper.make_tensor_value_info("scale", TensorProto.FLOAT, [width]),
    ]
    output = helper.make_tensor_value_info("y", TensorProto.FLOAT16, [rows, width])
    if fused:
        nodes = [helper.make_node("RMSNormFp32", ["x", "scale"], ["y"], domain="com.microsoft", epsilon=epsilon)]
    else:
        nodes = [
            helper.make_node("Cast", ["x"], ["xf"], to=TensorProto.FLOAT),
            helper.make_node("RMSNormalization", ["xf", "scale"], ["norm"], axis=-1, epsilon=epsilon),
            helper.make_node("Cast", ["norm"], ["y"], to=TensorProto.FLOAT16),
        ]
    return helper.make_model(
        helper.make_graph(nodes, "block_norm", inputs, [output]),
        opset_imports=[helper.make_opsetid("", 23), helper.make_opsetid("com.microsoft", 1)],
        ir_version=10,
    )


@pytest.mark.skipif("CUDAExecutionProvider" not in ort.get_available_providers(), reason="CUDA required")
@pytest.mark.parametrize("rows", [0, 1, 2, 4, 8, 33, 128, 512, 1024, 4096])
@pytest.mark.parametrize("epsilon", [0.0, 1e-6, 1e-3])
def test_bitwise_fp32_oracle(rows, epsilon):
    rng = np.random.default_rng(143 + rows)
    x = (rng.standard_normal((rows, 3072)) * 5).astype(np.float16)
    if rows:
        x[0] = np.resize(np.array([65504, -65504, 0, -0.0, 1e-7, -1e-7, 0.25], np.float16), 3072)
    if rows > 1:
        x[1] = 0
    if rows > 2:
        x[2] *= np.float16(1e-5)
    feeds = {"x": x, "scale": rng.uniform(-3, 3, 3072).astype(np.float32)}
    outputs = []
    for fused in (False, True):
        options = ort.SessionOptions()
        options.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
        session = ort.InferenceSession(
            make_model(rows, fused, epsilon).SerializeToString(),
            options,
            providers=["CUDAExecutionProvider"],
        )
        outputs.append(session.run(None, feeds)[0])
    np.testing.assert_array_equal(outputs[0].view(np.uint16), outputs[1].view(np.uint16))


@pytest.mark.skipif("CUDAExecutionProvider" not in ort.get_available_providers(), reason="CUDA required")
@pytest.mark.parametrize("rows", [1, 2, 8, 512, 1024])
def test_capture_replay(rows):
    rng = np.random.default_rng(774 + rows)
    values = {
        "x": ort.OrtValue.ortvalue_from_numpy(rng.standard_normal((rows, 3072)).astype(np.float16), "cuda", 0),
        "scale": ort.OrtValue.ortvalue_from_numpy(rng.uniform(-3, 3, 3072).astype(np.float32), "cuda", 0),
    }
    sessions, bindings, outputs = [], [], []
    for fused in (False, True):
        session = ort.InferenceSession(
            make_model(rows, fused).SerializeToString(),
            providers=[("CUDAExecutionProvider", {"enable_cuda_graph": "1"})],
        )
        binding = session.io_binding()
        for name, value in values.items():
            binding.bind_ortvalue_input(name, value)
        output = ort.OrtValue.ortvalue_from_shape_and_type([rows, 3072], np.float16, "cuda", 0)
        binding.bind_ortvalue_output("y", output)
        sessions.append(session)
        bindings.append(binding)
        outputs.append(output)
    for _ in range(3):
        values["x"].update_inplace(rng.standard_normal((rows, 3072)).astype(np.float16))
        for session, binding in zip(sessions, bindings, strict=True):
            session.run_with_iobinding(binding)
        np.testing.assert_array_equal(outputs[0].numpy().view(np.uint16), outputs[1].numpy().view(np.uint16))


@pytest.mark.skipif("CUDAExecutionProvider" not in ort.get_available_providers(), reason="CUDA required")
@pytest.mark.parametrize("kind", ["width", "scale", "epsilon"])
def test_invalid_contract(kind):
    model = make_model(2, True, width=128 if kind == "width" else 3072)
    if kind == "scale":
        model.graph.input[1].type.tensor_type.shape.dim[0].dim_value = 64
    elif kind == "epsilon":
        model.graph.node[0].attribute[0].f = -1.0
    with pytest.raises(Exception, match=r"RMSNormFp32|epsilon"):
        session = ort.InferenceSession(model.SerializeToString(), providers=["CUDAExecutionProvider"])
        session.run(
            None,
            {
                v.name: np.ones(
                    [d.dim_value for d in v.type.tensor_type.shape.dim],
                    dtype=np.float32 if v.name == "scale" else np.float16,
                )
                for v in model.graph.input
            },
        )
