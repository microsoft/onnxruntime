# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
"""Numeric check for com.microsoft.HyperConnectionLowRankMix against a torch reference.

The reference is the unfused subgraph the operator replaces,

    [HyperConnectionPostMix] -> BranchwiseRMSNorm -> MatMul(down) -> ScaledSiLU -> MatMul(up) -> Sigmoid
        -> HyperConnectionPreMix, and MatMul(inject) -> Div -> Sigmoid -> Mul,

evaluated with float32 math and rounded to the activation type at the points the standard operators round
(see the schema documentation), with a correctly rounded sigmoid. The fused path (few rows, 16-bit types) and
the cuBLAS based path (many rows, float32, shapes the fused kernels do not support) are both covered.
"""

# Tensor dimensions keep the upper-case names the schema doc gives them.
# ruff: noqa: N806

import unittest

import numpy as np
import torch
from onnx import TensorProto as TP  # noqa: N817
from onnx import helper

import onnxruntime as ort

DTYPES = {
    TP.FLOAT: torch.float32,
    TP.FLOAT16: torch.float16,
    TP.BFLOAT16: torch.bfloat16,
}

# (num_branches, hidden, rank): the real Qwen3.8-Flash-Next block first, then small and irregular shapes.
REAL_SHAPE = (4, 2560, 320)
SMALL_SHAPES = [
    (4, 64, 16),
    (2, 32, 8),
    (8, 64, 32),
    (1, 48, 8),
    (4, 1288, 40),  # two K slices per branch in the fused path
    (4, 4096, 512),  # four K slices per branch and two weight chunks in the up kernel
    (3, 40, 12),  # not supported by the fused kernels (odd branch count, rank % 8 != 0)
    (4, 100, 16),  # hidden % 8 != 0
]
ATTRS = dict(epsilon=1e-6, reduction_scale=0.25, silu_alpha=0.25, gate_div=4.0, gate_mul=2.0)


def has_cuda():
    return "CUDAExecutionProvider" in ort.get_available_providers()


def build_model(io_elem, C, H, R, prologue, attrs):
    """Graph with float32 inputs/outputs; Casts move the activations to the type under test."""
    F = TP.FLOAT
    K = C * H
    node_inputs = ["streams", "norm_scale", "down_weight", "up_weight", "inject_weight"]
    graph_inputs = [
        ("streams", ["rows", K]),
        ("norm_scale", [K]),
        ("down_weight", [K, R]),
        ("up_weight", [R, K]),
        ("inject_weight", [K, C]),
    ]
    if prologue:
        node_inputs += ["prev_block_output", "prev_post_gate"]
        graph_inputs += [("prev_block_output", ["rows", H]), ("prev_post_gate", ["rows", C])]
    outputs = ["pre_output", "post_gate", "streams_out"]

    nodes = [helper.make_node("Cast", [n], [n + "_t"], to=io_elem, name=n + "/cast") for n, _ in graph_inputs]
    nodes.append(
        helper.make_node(
            "HyperConnectionLowRankMix",
            [n + "_t" for n in node_inputs],
            [n + "_t" for n in outputs],
            domain="com.microsoft",
            num_branches=C,
            **attrs,
        )
    )
    nodes += [helper.make_node("Cast", [n + "_t"], [n], to=F, name=n + "/castback") for n in outputs]
    graph = helper.make_graph(
        nodes,
        "hc_low_rank_mix",
        [helper.make_tensor_value_info(n, F, shape) for n, shape in graph_inputs],
        [
            helper.make_tensor_value_info("pre_output", F, ["rows", H]),
            helper.make_tensor_value_info("post_gate", F, ["rows", C]),
            helper.make_tensor_value_info("streams_out", F, ["rows", K]),
        ],
    )
    model = helper.make_model(
        graph, opset_imports=[helper.make_opsetid("", 17), helper.make_opsetid("com.microsoft", 1)]
    )
    model.ir_version = 10
    return model


def make_inputs(dtype, rows, C, H, R, prologue, seed):
    g = torch.Generator().manual_seed(seed)
    K = C * H

    def rnd(*shape, scale=1.0, shift=0.0):
        return (torch.randn(*shape, generator=g) * scale + shift).to(dtype).float()

    inputs = {
        "streams": rnd(rows, K),
        "norm_scale": rnd(K, scale=0.1, shift=1.0),
        "down_weight": rnd(K, R, scale=K**-0.5),
        "up_weight": rnd(R, K, scale=2.0 * R**-0.5),
        "inject_weight": rnd(K, C, scale=K**-0.5),
    }
    if prologue:
        inputs["prev_block_output"] = rnd(rows, H)
        inputs["prev_post_gate"] = (torch.rand(rows, C, generator=g) * 2.0).to(dtype).float()
    return inputs


def reference(inputs, dtype, C, H, attrs):
    def rd(t):
        return t.to(dtype).float()

    def sigmoid(t):
        return torch.sigmoid(t.double()).float()

    x = inputs["streams"]
    rows = x.shape[0]
    s = x.view(rows, C, H)
    if "prev_block_output" in inputs:
        s = rd(s + inputs["prev_post_gate"][:, :, None] * inputs["prev_block_output"][:, None, :])
    inv_rms = torch.rsqrt(s.pow(2).mean(-1, keepdim=True) + attrs["epsilon"])
    hn = rd(s * inv_rms * inputs["norm_scale"].view(C, H))
    hn_flat = hn.view(rows, C * H)

    down = rd(hn_flat @ inputs["down_weight"])
    z = rd(down * attrs["silu_alpha"])
    hidden = rd(z * rd(sigmoid(z)))
    up = rd(hidden @ inputs["up_weight"])
    g = rd(sigmoid(up)).view(rows, C, H)
    pre = rd(attrs["reduction_scale"] * (hn * g).sum(1))
    inject = rd(hn_flat @ inputs["inject_weight"])
    gate = rd(rd(sigmoid(rd(inject / attrs["gate_div"]))) * attrs["gate_mul"])
    return pre, gate, s.reshape(rows, C * H)


class SessionCache:
    def __init__(self):
        self.sessions = {}

    def get(self, io_elem, C, H, R, prologue, attrs):
        key = (io_elem, C, H, R, prologue, tuple(sorted(attrs.items())))
        if key not in self.sessions:
            model = build_model(io_elem, C, H, R, prologue, attrs)
            so = ort.SessionOptions()
            so.log_severity_level = 3
            self.sessions[key] = ort.InferenceSession(
                model.SerializeToString(), so, providers=["CUDAExecutionProvider"]
            )
        return self.sessions[key]


def tolerances(io_elem):
    # One ulp of the output type plus slack for the different fp32 accumulation order flipping a rounding.
    return {TP.FLOAT: (1e-3, 1e-5), TP.FLOAT16: (4e-3, 2e-3), TP.BFLOAT16: (3e-2, 8e-3)}[io_elem]


@unittest.skipUnless(has_cuda(), "HyperConnectionLowRankMix is a CUDA-only kernel")
class TestHyperConnectionLowRankMix(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cache = SessionCache()

    def _check(self, io_elem, rows, shape, prologue, attrs=ATTRS, seed=0):
        C, H, R = shape
        dtype = DTYPES[io_elem]
        inputs = make_inputs(dtype, rows, C, H, R, prologue, seed)
        session = self.cache.get(io_elem, C, H, R, prologue, attrs)
        pre, gate, streams_out = session.run(None, {k: v.numpy() for k, v in inputs.items()})
        exp_pre, exp_gate, exp_streams = reference(inputs, dtype, C, H, attrs)
        rtol, atol = tolerances(io_elem)
        msg = f"dtype={dtype} rows={rows} shape={shape} prologue={prologue}"
        np.testing.assert_allclose(pre, exp_pre.numpy(), rtol=rtol, atol=atol, err_msg="pre_output " + msg)
        np.testing.assert_allclose(gate, exp_gate.numpy(), rtol=rtol, atol=atol, err_msg="post_gate " + msg)
        # The prologue is a single fp32 operation with one rounding: it matches exactly (up to fma contraction).
        np.testing.assert_allclose(streams_out, exp_streams.numpy(), rtol=rtol, atol=atol, err_msg="streams_out " + msg)
        return pre, gate, streams_out

    def test_real_shape(self):
        for io_elem in DTYPES:
            for rows in (1, 2, 4, 16, 128):
                for prologue in (False, True):
                    with self.subTest(io_elem=io_elem, rows=rows, prologue=prologue):
                        self._check(io_elem, rows, REAL_SHAPE, prologue)

    def test_small_shapes(self):
        for io_elem in DTYPES:
            for shape in SMALL_SHAPES:
                for rows in (1, 2, 16):
                    for prologue in (False, True):
                        with self.subTest(io_elem=io_elem, shape=shape, rows=rows, prologue=prologue):
                            self._check(io_elem, rows, shape, prologue, seed=1)

    def test_default_attributes(self):
        attrs = dict(epsilon=1e-5, reduction_scale=1.0, silu_alpha=1.0, gate_div=1.0, gate_mul=1.0)
        for io_elem in DTYPES:
            for rows in (1, 3):
                with self.subTest(io_elem=io_elem, rows=rows):
                    self._check(io_elem, rows, (4, 64, 16), True, attrs=attrs, seed=2)

    def test_other_gate_constants(self):
        attrs = dict(epsilon=1e-6, reduction_scale=0.5, silu_alpha=0.75, gate_div=3.0, gate_mul=1.5)
        for io_elem in (TP.FLOAT16, TP.BFLOAT16):
            for rows in (1, 2):
                with self.subTest(io_elem=io_elem, rows=rows):
                    self._check(io_elem, rows, REAL_SHAPE, True, attrs=attrs, seed=3)

    def test_chain_of_blocks(self):
        """Two consecutive calls through the prologue equal the explicit post-mix between them."""
        C, H, R = REAL_SHAPE
        for io_elem in (TP.FLOAT16, TP.BFLOAT16):
            dtype = DTYPES[io_elem]
            with self.subTest(io_elem=io_elem):
                inputs = make_inputs(dtype, 2, C, H, R, False, seed=4)
                first = self.cache.get(io_elem, C, H, R, False, ATTRS)
                second = self.cache.get(io_elem, C, H, R, True, ATTRS)
                _, gate, streams = first.run(None, {k: v.numpy() for k, v in inputs.items()})
                # Use the block "output" as an independent tensor (a stand-in for the attention/MLP result).
                block_out = torch.randn(2, H, generator=torch.Generator().manual_seed(8)).to(dtype).float().numpy()
                chained = dict(inputs)
                chained["streams"] = streams
                chained["prev_block_output"] = block_out
                chained["prev_post_gate"] = gate
                out = second.run(None, {k: np.asarray(v) for k, v in chained.items()})
                ref_inputs = {k: torch.from_numpy(np.asarray(v)) for k, v in chained.items()}
                exp = reference(ref_inputs, dtype, C, H, ATTRS)
                rtol, atol = tolerances(io_elem)
                np.testing.assert_allclose(out[0], exp[0].numpy(), rtol=rtol, atol=atol)
                np.testing.assert_allclose(out[1], exp[1].numpy(), rtol=rtol, atol=atol)

    def test_fused_rows_are_independent(self):
        """The fused kernels use a fixed reduction order: a row gives the same bits at any batch size."""
        C, H, R = REAL_SHAPE
        for io_elem in (TP.FLOAT16, TP.BFLOAT16):
            dtype = DTYPES[io_elem]
            with self.subTest(io_elem=io_elem):
                inputs = make_inputs(dtype, 4, C, H, R, True, seed=5)
                batch = self.cache.get(io_elem, C, H, R, True, ATTRS).run(
                    None, {k: v.numpy() for k, v in inputs.items()}
                )
                for r in range(4):
                    row_inputs = {
                        k: (v[r : r + 1] if k in ("streams", "prev_block_output", "prev_post_gate") else v).numpy()
                        for k, v in inputs.items()
                    }
                    single = self.cache.get(io_elem, C, H, R, True, ATTRS).run(None, row_inputs)
                    for b, s in zip(batch, single, strict=True):
                        np.testing.assert_array_equal(b[r : r + 1], s)

    def test_no_prologue_streams_out_is_passthrough(self):
        C, H, R = 4, 64, 16
        inputs = make_inputs(torch.bfloat16, 2, C, H, R, False, seed=6)
        session = self.cache.get(TP.BFLOAT16, C, H, R, False, ATTRS)
        _, _, streams_out = session.run(None, {k: v.numpy() for k, v in inputs.items()})
        np.testing.assert_array_equal(streams_out, inputs["streams"].numpy())

    def test_cuda_graph_capture_and_replay(self):
        """No host synchronization or unmanaged allocation: the op can be captured and replayed."""
        C, H, R = REAL_SHAPE
        for io_elem in (TP.FLOAT16, TP.BFLOAT16):
            for rows in (1, 2):
                with self.subTest(io_elem=io_elem, rows=rows):
                    dtype = DTYPES[io_elem]
                    inputs = make_inputs(dtype, rows, C, H, R, True, seed=9)
                    model = build_model(io_elem, C, H, R, True, ATTRS)
                    so = ort.SessionOptions()
                    so.log_severity_level = 3
                    graph_session = ort.InferenceSession(
                        model.SerializeToString(),
                        so,
                        providers=[("CUDAExecutionProvider", {"enable_cuda_graph": "1"})],
                    )
                    binding = graph_session.io_binding()
                    keep = []
                    for name, value in inputs.items():
                        ortvalue = ort.OrtValue.ortvalue_from_numpy(value.numpy(), "cuda", 0)
                        keep.append(ortvalue)
                        binding.bind_ortvalue_input(name, ortvalue)
                    shapes = {"pre_output": (rows, H), "post_gate": (rows, C), "streams_out": (rows, C * H)}
                    outputs = {}
                    for name, shape in shapes.items():
                        outputs[name] = ort.OrtValue.ortvalue_from_shape_and_type(shape, np.float32, "cuda", 0)
                        binding.bind_ortvalue_output(name, outputs[name])
                    graph_session.run_with_iobinding(binding)  # captures
                    graph_session.run_with_iobinding(binding)  # replays
                    expected = self.cache.get(io_elem, C, H, R, True, ATTRS).run(
                        None, {k: v.numpy() for k, v in inputs.items()}
                    )
                    for name, exp in zip(shapes, expected, strict=True):
                        np.testing.assert_array_equal(outputs[name].numpy(), exp, err_msg=name)

    def test_zero_rows(self):
        C, H, R = 4, 64, 16
        inputs = make_inputs(torch.float16, 0, C, H, R, False, seed=7)
        session = self.cache.get(TP.FLOAT16, C, H, R, False, ATTRS)
        pre, gate, streams_out = session.run(None, {k: v.numpy() for k, v in inputs.items()})
        self.assertEqual(pre.shape, (0, H))
        self.assertEqual(gate.shape, (0, C))
        self.assertEqual(streams_out.shape, (0, C * H))


if __name__ == "__main__":
    unittest.main()
