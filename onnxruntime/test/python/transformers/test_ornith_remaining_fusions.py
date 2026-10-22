# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Bitwise component oracles for exact Ornith graph fusions."""

import os
from unittest.mock import patch

import numpy as np
import pytest
from onnx import TensorProto as TP  # noqa: N817
from onnx import helper as h

import onnxruntime as ort

pytestmark = pytest.mark.skipif("CUDAExecutionProvider" not in ort.get_available_providers(), reason="CUDA required")


def session(nodes, inputs, outputs, graph=False):
    model = h.make_model(
        h.make_graph(nodes, "fusions", inputs, outputs),
        opset_imports=[h.make_opsetid("", 23), h.make_opsetid("com.microsoft", 1)],
        ir_version=10,
    )
    options = ort.SessionOptions()
    options.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
    return ort.InferenceSession(
        model.SerializeToString(),
        options,
        providers=[("CUDAExecutionProvider", {"enable_cuda_graph": str(int(graph))})],
    )


def value(name, dtype, shape):
    return h.make_tensor_value_info(name, dtype, shape)


def compare(sessions, feeds, graph):
    if not graph:
        return [s.run(None, f) for s, f in zip(sessions, feeds, strict=True)]
    bindings, outputs, keepalive = [], [], []
    for s, feed in zip(sessions, feeds, strict=True):
        binding = s.io_binding()
        values = {k: ort.OrtValue.ortvalue_from_numpy(v, "cuda", 0) for k, v in feed.items()}
        for k, v in values.items():
            binding.bind_ortvalue_input(k, v)
        for o in s.get_outputs():
            binding.bind_output(o.name, "cuda", 0)
        bindings.append(binding)
        keepalive.append(values)
    for _ in range(3):
        for s, binding in zip(sessions, bindings, strict=True):
            s.run_with_iobinding(binding)
    for binding in bindings:
        outputs.append(binding.copy_outputs_to_cpu())
    return outputs


@pytest.mark.parametrize("rows", [0, 1, 2, 8, 1024])
@pytest.mark.parametrize("graph", [False, True])
def test_attention_sigmoid_multiply(rows, graph):
    if graph and rows == 0:
        pytest.skip("empty graph has no GPU work")
    rng = np.random.default_rng(419)
    inputs = [value(n, TP.FLOAT16, [rows, 8192]) for n in ("x", "gate")]
    outputs = [value("y", TP.FLOAT16, [rows, 8192])]
    reference = [
        h.make_node("Sigmoid", ["gate"], ["sigmoid"]),
        h.make_node("Mul", ["x", "sigmoid"], ["y"]),
    ]
    fused = [h.make_node("SigmoidMul", ["x", "gate"], ["y"], domain="com.microsoft")]
    feeds = {n: rng.normal(0, 8, (rows, 8192)).astype(np.float16) for n in ("x", "gate")}
    if rows:
        feeds["gate"][0, :12] = [-100, -90, -20, -10, -1, -0.0, 0, 1, 10, 20, 90, 100]
    a, b = compare([session(n, inputs, outputs, graph) for n in (reference, fused)], [feeds, feeds], graph)
    np.testing.assert_array_equal(a[0].view(np.uint16), b[0].view(np.uint16))


@pytest.mark.parametrize("heads", [2, 32])
@pytest.mark.parametrize("rows", [1, 2, 8, 1024])
@pytest.mark.parametrize("graph", [False, True])
def test_qk_norm_rotary(heads, rows, graph):
    rng = np.random.default_rng(713)
    inputs = [
        value("x", TP.FLOAT16, [rows, heads, 256]),
        value("scale", TP.FLOAT, [256]),
        value("cos", TP.FLOAT16, [1, rows, 32]),
        value("sin", TP.FLOAT16, [1, rows, 32]),
        value("shape", TP.INT64, [3]),
    ]
    outputs = [value("y", TP.FLOAT16, [1, rows, heads * 256])]
    reference = [
        h.make_node("Cast", ["x"], ["xf"], to=TP.FLOAT),
        h.make_node("RMSNormalization", ["xf", "scale"], ["norm"], axis=-1, epsilon=1e-6),
        h.make_node("Cast", ["norm"], ["half"], to=TP.FLOAT16),
        h.make_node("Reshape", ["half", "shape"], ["packed"], allowzero=1),
        h.make_node(
            "RotaryEmbedding",
            ["packed", "cos", "sin"],
            ["y"],
            num_heads=heads,
            rotary_embedding_dim=64,
            interleaved=0,
        ),
    ]
    fused = [
        h.make_node(
            "RMSNormRotaryFp32",
            ["x", "scale", "cos", "sin"],
            ["y"],
            domain="com.microsoft",
            epsilon=1e-6,
        )
    ]
    x = rng.normal(0, 5, (rows, heads, 256)).astype(np.float16)
    if rows:
        x[0] = np.resize(np.array([65504, -65504, 0, -0.0, 1e-7, -1e-7, 0.25], np.float16), (heads, 256))
    feeds = {
        "x": x,
        "scale": rng.uniform(-2, 2, 256).astype(np.float32),
        "cos": rng.uniform(-1, 1, (1, rows, 32)).astype(np.float16),
        "sin": rng.uniform(-1, 1, (1, rows, 32)).astype(np.float16),
        "shape": np.array([1, rows, heads * 256], np.int64),
    }
    # Shape is CPU metadata in the reference, not part of the fused CUDA op.
    a = session(reference, inputs, outputs).run(None, feeds)[0]
    del feeds["shape"]
    s = session(fused, inputs[:-1], outputs, graph)
    b = compare([s], [feeds], graph)[0][0]
    np.testing.assert_array_equal(a.view(np.uint16), b.view(np.uint16))


@pytest.mark.parametrize(
    "rows,capture,engine",
    [
        (1, 8, "recurrent"),
        (2, 8, "recurrent"),
        (8, 8, "recurrent"),
        (256, 0, "chunked"),
        (256, 0, "chunked_split"),
        (1024, 8, ""),
    ],
)
@pytest.mark.parametrize("graph", [False, True])
def test_packed_gdn_half_gates(rows, capture, engine, graph):
    rng = np.random.default_rng(314)
    hk, hv, dk, dv = 16, 64, 128, 128
    gate_shape = [rows, hv]
    state_shape = [1, hv, dv, dk]
    common = [
        value("cu", TP.INT32, [2]),
        value("decay", TP.FLOAT, gate_shape),
        value("beta", TP.FLOAT, gate_shape),
        value("state", TP.FLOAT16, state_shape),
        value("a_log", TP.FLOAT, [hv]),
        value("dt_bias", TP.FLOAT, [hv]),
    ]
    if capture:
        common.append(value("capture", TP.INT32, [1]))
    outputs = [value("y", TP.FLOAT16, [rows, hv, dv]), value("final", TP.FLOAT16, state_shape)]
    if capture:
        outputs.append(value("capsule", TP.FLOAT, [1, capture * (hv + hk * dk + hv * dv)]))
    qkv = {
        n: rng.normal(0, 0.05, (rows, heads, width)).astype(np.float16)
        for n, heads, width in (("q", hk, dk), ("k", hk, dk), ("v", hv, dv))
    }
    gates = {n: rng.normal(0, 0.5, gate_shape).astype(np.float16) for n in ("decay", "beta")}
    feeds = {
        **qkv,
        **{n: v.astype(np.float32) for n, v in gates.items()},
        "cu": np.array([0, rows], np.int32),
        "state": rng.normal(0, 0.05, state_shape).astype(np.float16),
        "a_log": rng.uniform(-2, 1, hv).astype(np.float32),
        "dt_bias": rng.uniform(-1, 1, hv).astype(np.float32),
    }
    if capture:
        feeds["capture"] = np.array([min(rows, capture)], np.int32)
    sessions, all_feeds = [], []
    with patch.dict(os.environ, {"ORT_GDN_PLAN": engine}):
        for packed in (False, True):
            if packed:
                inputs = [value("packed", TP.FLOAT16, [rows, (2 * hk * dk + hv * dv)])]
                common_copy = [
                    value(
                        v.name,
                        TP.FLOAT16 if v.name in gates else v.type.tensor_type.elem_type,
                        [d.dim_value for d in v.type.tensor_type.shape.dim],
                    )
                    for v in common
                ]
                inputs += common_copy
                names = ["packed", "", "", *[v.name for v in common]]
                f = {n: v for n, v in feeds.items() if n not in qkv}
                f.update(gates)
                f["packed"] = np.concatenate([v.reshape(rows, -1) for v in qkv.values()], axis=1)
            else:
                inputs = [value(n, TP.FLOAT16, list(v.shape)) for n, v in qkv.items()] + common
                names = [v.name for v in inputs]
                f = feeds
            if engine == "chunked_split":
                inputs = [v for v in inputs if v.name != "cu"]
                names[3] = ""
                f = {n: v for n, v in f.items() if n != "cu"}
            node = h.make_node(
                "GatedDeltaNet",
                names,
                [v.name for v in outputs],
                domain="com.microsoft",
                gate_activation="qwen",
                beta_activation="sigmoid",
                qk_l2_norm=1,
                state_update_capacity=capture,
                scale=0.0,
            )
            sessions.append(session([node], inputs, outputs, graph))
            all_feeds.append(f)
    a, b = compare(sessions, all_feeds, graph)
    for i in (0, 1):
        np.testing.assert_array_equal(a[i].view(np.uint16), b[i].view(np.uint16))
    if capture:
        count = min(rows, capture)
        start = 0
        for width in (hv, hk * dk, hv * dv):
            np.testing.assert_array_equal(
                a[2][:, start : start + count * width].view(np.uint32),
                b[2][:, start : start + count * width].view(np.uint32),
            )
            start += capture * width
