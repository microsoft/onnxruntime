# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Bitwise oracle for the width-128 FP32 RMSNorm/QuickGelu fusion."""

import numpy as np
import pytest
from onnx import TensorProto, helper

import onnxruntime as ort


def make_model(rows, fused, epsilon=1e-6):
    inputs = [helper.make_tensor_value_info(name, TensorProto.FLOAT16, [rows, 128]) for name in ("x", "gate")]
    scale = helper.make_tensor_value_info("scale", TensorProto.FLOAT, [128])
    output = helper.make_tensor_value_info("y", TensorProto.FLOAT16, [rows, 128])
    if fused:
        nodes = [
            helper.make_node(
                "GatedRMSNormExact",
                ["x", "gate", "scale"],
                ["y"],
                domain="com.microsoft",
                epsilon=epsilon,
            )
        ]
    else:
        nodes = [
            helper.make_node("Cast", ["x"], ["xf"], to=TensorProto.FLOAT),
            helper.make_node("Cast", ["gate"], ["gf"], to=TensorProto.FLOAT),
            helper.make_node(
                "RMSNormalization",
                ["xf", "scale"],
                ["norm"],
                axis=-1,
                stash_type=1,
                epsilon=epsilon,
            ),
            helper.make_node("QuickGelu", ["gf"], ["activated"], domain="com.microsoft", alpha=1.0),
            helper.make_node("Mul", ["norm", "activated"], ["product"]),
            helper.make_node("Cast", ["product"], ["y"], to=TensorProto.FLOAT16),
        ]
    return helper.make_model(
        helper.make_graph(nodes, "gated_norm", [*inputs, scale], [output]),
        opset_imports=[helper.make_opsetid("", 23), helper.make_opsetid("com.microsoft", 1)],
        ir_version=10,
    )


@pytest.mark.skipif("CUDAExecutionProvider" not in ort.get_available_providers(), reason="CUDA required")
@pytest.mark.parametrize("rows", [0, 1, 2, 64, 128, 256, 512, 65536])
@pytest.mark.parametrize("epsilon", [0.0, 1e-6, 1e-3])
def test_bitwise_unfused_oracle(rows, epsilon):
    rng = np.random.default_rng(891 + rows)
    x = (rng.standard_normal((rows, 128)) * 3).astype(np.float16)
    gate = (rng.standard_normal((rows, 128)) * 12).astype(np.float16)
    if rows:
        x[0] = np.resize(np.array([0, -0.0, 65504, -65504, 1e-7, -1e-7, 0.03125, -0.25], np.float16), 128)
        gate[0] = np.resize(np.array([-100, -90, -87, -20, -1, -0.0, 0, 1, 20, 100], np.float16), 128)
    if rows > 1:
        x[1] = 0
    if rows > 2:
        x[2] *= np.float16(1e-5)
    feeds = {"x": x, "gate": gate, "scale": rng.uniform(-2, 2, 128).astype(np.float32)}
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
@pytest.mark.parametrize("rows", [64, 128, 512, 65536])
def test_cuda_graph_replay(rows):
    rng = np.random.default_rng(912 + rows)
    values = {
        name: ort.OrtValue.ortvalue_from_numpy(rng.standard_normal((rows, 128)).astype(np.float16), "cuda", 0)
        for name in ("x", "gate")
    }
    values["scale"] = ort.OrtValue.ortvalue_from_numpy(rng.uniform(-3, 3, 128).astype(np.float32), "cuda", 0)
    sessions, bindings, outputs = [], [], []
    for fused in (False, True):
        session = ort.InferenceSession(
            make_model(rows, fused).SerializeToString(),
            providers=[("CUDAExecutionProvider", {"enable_cuda_graph": "1"})],
        )
        binding = session.io_binding()
        for name, value in values.items():
            binding.bind_ortvalue_input(name, value)
        output = ort.OrtValue.ortvalue_from_shape_and_type([rows, 128], np.float16, "cuda", 0)
        binding.bind_ortvalue_output("y", output)
        sessions.append(session)
        bindings.append(binding)
        outputs.append(output)
    for _ in range(3):
        for name in ("x", "gate"):
            values[name].update_inplace(rng.standard_normal((rows, 128)).astype(np.float16))
        for session, binding in zip(sessions, bindings, strict=True):
            session.run_with_iobinding(binding)
        np.testing.assert_array_equal(outputs[0].numpy().view(np.uint16), outputs[1].numpy().view(np.uint16))


@pytest.mark.skipif("CUDAExecutionProvider" not in ort.get_available_providers(), reason="CUDA required")
@pytest.mark.parametrize("kind", ["gate", "scale", "epsilon"])
def test_invalid_contract(kind):
    model = make_model(2, True)
    if kind == "gate":
        model.graph.input[1].type.tensor_type.shape.dim[0].dim_value = 1
    elif kind == "scale":
        model.graph.input[2].type.tensor_type.shape.dim[0].dim_value = 64
    else:
        model.graph.node[0].attribute[0].f = -1.0
    with pytest.raises(Exception, match=r"GatedRMSNorm|epsilon"):
        session = ort.InferenceSession(model.SerializeToString(), providers=["CUDAExecutionProvider"])
        feeds = {
            value.name: np.ones(
                [d.dim_value for d in value.type.tensor_type.shape.dim],
                dtype=np.float32 if value.name == "scale" else np.float16,
            )
            for value in model.graph.input
        }
        session.run(None, feeds)
