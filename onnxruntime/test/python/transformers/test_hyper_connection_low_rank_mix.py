# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
"""Numeric check for com.microsoft.HyperConnectionLowRankMix against a torch reference.

The reference is the unfused subgraph the operator replaces,

    [HyperConnectionPostMix] -> BranchwiseRMSNorm -> MatMul(down) -> ScaledSiLU -> MatMul(up) -> Sigmoid
        -> HyperConnectionPreMix, and MatMul(inject) -> Div -> Sigmoid -> Mul,

evaluated with float32 math and rounded to the activation type at the points the standard operators round
(see the schema documentation), with a correctly rounded sigmoid. The fused path (few rows, 16-bit types) and
the cuBLAS based path (many rows, float32, shapes the fused kernels do not support) are both covered.
BF16 cases require a confirmed compute capability of 8.0+; FP32/FP16 cases remain enabled on older GPUs.
"""

# Tensor dimensions keep the upper-case names the schema doc gives them.
# ruff: noqa: N806

import unittest
from unittest import mock

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


def has_bf16_support():
    # The BFloat16 kernels need compute capability 8.0+ (cuBLAS BFloat16 GEMM, cp.async in the fused kernels).
    return torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8


HALF_TYPES = (TP.FLOAT16, TP.BFLOAT16) if has_bf16_support() else (TP.FLOAT16,)
IO_TYPES = (TP.FLOAT, *HALF_TYPES)


def make_initializer(name, io_elem, tensor):
    dtype = DTYPES[io_elem]
    raw = tensor.to(dtype).contiguous().view(torch.uint8).numpy().tobytes()
    return helper.make_tensor(name, io_elem, list(tensor.shape), raw, raw=True)


def build_model(io_elem, C, H, R, prologue, attrs, weights=None):
    """Graph with float32 inputs/outputs; Casts move the activations to the type under test."""
    F = TP.FLOAT
    K = C * H
    weight_names = ("norm_scale", "down_weight", "up_weight", "inject_weight")
    node_inputs = ["streams"] + [n if weights is not None else n + "_t" for n in weight_names]
    graph_inputs = [("streams", ["rows", K])]
    initializers = []
    if weights is not None:
        initializers = [make_initializer(n, io_elem, weights[n]) for n in weight_names]
    else:
        graph_inputs += [
            ("norm_scale", [K]),
            ("down_weight", [K, R]),
            ("up_weight", [R, K]),
            ("inject_weight", [K, C]),
        ]
    if prologue:
        node_inputs += ["prev_block_output_t", "prev_post_gate_t"]
        graph_inputs += [("prev_block_output", ["rows", H]), ("prev_post_gate", ["rows", C])]
    outputs = ["pre_output", "post_gate", "streams_out"]

    nodes = [helper.make_node("Cast", [n], [n + "_t"], to=io_elem, name=n + "/cast") for n, _ in graph_inputs]
    node_inputs[0] = "streams_t"
    nodes.append(
        helper.make_node(
            "HyperConnectionLowRankMix",
            node_inputs,
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
        initializer=initializers,
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


class TestCudaCapabilities(unittest.TestCase):
    def test_bf16_requires_sm80(self):
        for capability, expected in (((7, 5), False), ((8, 0), True), ((9, 0), True)):
            with (
                self.subTest(capability=capability),
                mock.patch.object(torch.cuda, "is_available", return_value=True),
                mock.patch.object(torch.cuda, "get_device_capability", return_value=capability),
            ):
                self.assertEqual(has_bf16_support(), expected)

    def test_bf16_without_cuda(self):
        with (
            mock.patch.object(torch.cuda, "is_available", return_value=False),
            mock.patch.object(torch.cuda, "get_device_capability") as get_capability,
        ):
            self.assertFalse(has_bf16_support())
            get_capability.assert_not_called()


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
        for io_elem in IO_TYPES:
            for rows in (1, 2, 4, 16, 128):
                for prologue in (False, True):
                    with self.subTest(io_elem=io_elem, rows=rows, prologue=prologue):
                        self._check(io_elem, rows, REAL_SHAPE, prologue)

    def test_small_shapes(self):
        for io_elem in IO_TYPES:
            for shape in SMALL_SHAPES:
                for rows in (1, 2, 16):
                    for prologue in (False, True):
                        with self.subTest(io_elem=io_elem, shape=shape, rows=rows, prologue=prologue):
                            self._check(io_elem, rows, shape, prologue, seed=1)

    def test_default_attributes(self):
        attrs = dict(epsilon=1e-5, reduction_scale=1.0, silu_alpha=1.0, gate_div=1.0, gate_mul=1.0)
        for io_elem in IO_TYPES:
            for rows in (1, 3):
                with self.subTest(io_elem=io_elem, rows=rows):
                    self._check(io_elem, rows, (4, 64, 16), True, attrs=attrs, seed=2)

    def test_other_gate_constants(self):
        attrs = dict(epsilon=1e-6, reduction_scale=0.5, silu_alpha=0.75, gate_div=3.0, gate_mul=1.5)
        for io_elem in HALF_TYPES:
            for rows in (1, 2):
                with self.subTest(io_elem=io_elem, rows=rows):
                    self._check(io_elem, rows, REAL_SHAPE, True, attrs=attrs, seed=3)

    def test_chain_of_blocks(self):
        """Two consecutive blocks in the same graph equal the explicit post-mix between them."""
        C, H, R = REAL_SHAPE
        weight_names = ("norm_scale", "down_weight", "up_weight", "inject_weight")
        for io_elem in HALF_TYPES:
            dtype = DTYPES[io_elem]
            with self.subTest(io_elem=io_elem):
                inputs0 = make_inputs(dtype, 2, C, H, R, False, seed=4)
                inputs1 = make_inputs(dtype, 2, C, H, R, True, seed=8)
                block_out = inputs1["prev_block_output"]
                initializers = [make_initializer(f"w0_{k}", io_elem, inputs0[k]) for k in weight_names] + [
                    make_initializer(f"w1_{k}", io_elem, inputs1[k]) for k in weight_names
                ]
                nodes = [
                    helper.make_node("Cast", ["streams"], ["streams_t"], to=io_elem),
                    helper.make_node("Cast", ["block_out"], ["block_out_t"], to=io_elem),
                    helper.make_node(
                        "HyperConnectionLowRankMix",
                        ["streams_t", *[f"w0_{k}" for k in weight_names]],
                        ["pre0_t", "gate0_t", "streams0_t"],
                        domain="com.microsoft",
                        num_branches=C,
                        **ATTRS,
                    ),
                    helper.make_node(
                        "HyperConnectionLowRankMix",
                        ["streams0_t", *[f"w1_{k}" for k in weight_names], "block_out_t", "gate0_t"],
                        ["pre_output_t", "post_gate_t", "streams_out_t"],
                        domain="com.microsoft",
                        num_branches=C,
                        **ATTRS,
                    ),
                    helper.make_node("Cast", ["pre_output_t"], ["pre_output"], to=TP.FLOAT),
                    helper.make_node("Cast", ["post_gate_t"], ["post_gate"], to=TP.FLOAT),
                ]
                graph = helper.make_graph(
                    nodes,
                    "hc_chain",
                    [
                        helper.make_tensor_value_info("streams", TP.FLOAT, [2, C * H]),
                        helper.make_tensor_value_info("block_out", TP.FLOAT, [2, H]),
                    ],
                    [
                        helper.make_tensor_value_info("pre_output", TP.FLOAT, [2, H]),
                        helper.make_tensor_value_info("post_gate", TP.FLOAT, [2, C]),
                    ],
                    initializer=initializers,
                )
                model = helper.make_model(
                    graph, opset_imports=[helper.make_opsetid("", 17), helper.make_opsetid("com.microsoft", 1)]
                )
                model.ir_version = 10
                so = ort.SessionOptions()
                so.log_severity_level = 3
                session = ort.InferenceSession(model.SerializeToString(), so, providers=["CUDAExecutionProvider"])
                out = session.run(None, {"streams": inputs0["streams"].numpy(), "block_out": block_out.numpy()})

                _, gate0, streams0 = reference(inputs0, dtype, C, H, ATTRS)
                chained = dict(inputs1)
                chained["streams"] = streams0
                chained["prev_block_output"] = block_out
                chained["prev_post_gate"] = gate0
                exp = reference(chained, dtype, C, H, ATTRS)
                rtol, atol = tolerances(io_elem)
                np.testing.assert_allclose(out[0], exp[0].numpy(), rtol=rtol, atol=atol)
                np.testing.assert_allclose(out[1], exp[1].numpy(), rtol=rtol, atol=atol)

    def test_same_graph_computed_weight(self):
        """A fused producer's [8, 8] pre_output used as a weight by a second fused node must not race under PDL."""
        rows = 8
        prod_C, H, prod_R = 8, 8, 512
        cons_C, cons_R = 1, 8
        weight_names = ("norm_scale", "down_weight", "up_weight", "inject_weight")
        for io_elem in HALF_TYPES:
            dtype = DTYPES[io_elem]
            for dynamic_weight in ("down_weight", "up_weight"):
                with self.subTest(io_elem=io_elem, dynamic_weight=dynamic_weight):
                    prod_in = make_inputs(dtype, rows, prod_C, H, prod_R, False, seed=10)
                    cons_in = make_inputs(dtype, rows, cons_C, H, cons_R, False, seed=11)
                    initializers = [make_initializer(f"p_{k}", io_elem, prod_in[k]) for k in weight_names] + [
                        make_initializer(f"c_{k}", io_elem, cons_in[k]) for k in weight_names if k != dynamic_weight
                    ]
                    cons_weights = ["p_pre_t" if k == dynamic_weight else f"c_{k}" for k in weight_names]
                    nodes = [
                        helper.make_node("Cast", ["p_streams"], ["p_streams_t"], to=io_elem),
                        helper.make_node("Cast", ["c_streams"], ["c_streams_t"], to=io_elem),
                        helper.make_node(
                            "HyperConnectionLowRankMix",
                            ["p_streams_t", *[f"p_{k}" for k in weight_names]],
                            ["p_pre_t", "p_gate_t"],
                            domain="com.microsoft",
                            num_branches=prod_C,
                            **ATTRS,
                        ),
                        helper.make_node(
                            "HyperConnectionLowRankMix",
                            ["c_streams_t", *cons_weights],
                            ["pre_output_t", "post_gate_t"],
                            domain="com.microsoft",
                            num_branches=cons_C,
                            **ATTRS,
                        ),
                        helper.make_node("Cast", ["pre_output_t"], ["pre_output"], to=TP.FLOAT),
                        helper.make_node("Cast", ["post_gate_t"], ["post_gate"], to=TP.FLOAT),
                    ]
                    graph = helper.make_graph(
                        nodes,
                        "hc_computed_weight",
                        [
                            helper.make_tensor_value_info("p_streams", TP.FLOAT, [rows, prod_C * H]),
                            helper.make_tensor_value_info("c_streams", TP.FLOAT, [rows, cons_C * H]),
                        ],
                        [
                            helper.make_tensor_value_info("pre_output", TP.FLOAT, [rows, H]),
                            helper.make_tensor_value_info("post_gate", TP.FLOAT, [rows, cons_C]),
                        ],
                        initializer=initializers,
                    )
                    model = helper.make_model(
                        graph, opset_imports=[helper.make_opsetid("", 17), helper.make_opsetid("com.microsoft", 1)]
                    )
                    model.ir_version = 10
                    so = ort.SessionOptions()
                    so.log_severity_level = 3
                    session = ort.InferenceSession(model.SerializeToString(), so, providers=["CUDAExecutionProvider"])
                    pre, gate = session.run(
                        None,
                        {
                            "p_streams": prod_in["streams"].numpy(),
                            "c_streams": cons_in["streams"].numpy(),
                        },
                    )
                    prod_pre, _, _ = reference(prod_in, dtype, prod_C, H, ATTRS)
                    ref_cons = dict(cons_in)
                    ref_cons[dynamic_weight] = prod_pre
                    exp_pre, exp_gate, _ = reference(ref_cons, dtype, cons_C, H, ATTRS)
                    rtol, atol = tolerances(io_elem)
                    np.testing.assert_allclose(pre, exp_pre.numpy(), rtol=rtol, atol=atol)
                    np.testing.assert_allclose(gate, exp_gate.numpy(), rtol=rtol, atol=atol)

    def test_fused_rows_are_independent(self):
        """The fused kernels use a fixed reduction order: a row gives the same bits at any batch size."""
        C, H, R = REAL_SHAPE
        for io_elem in HALF_TYPES:
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
        io_elem = HALF_TYPES[-1]
        inputs = make_inputs(DTYPES[io_elem], 2, C, H, R, False, seed=6)
        session = self.cache.get(io_elem, C, H, R, False, ATTRS)
        _, _, streams_out = session.run(None, {k: v.numpy() for k, v in inputs.items()})
        np.testing.assert_array_equal(streams_out, inputs["streams"].numpy())

    def test_cuda_graph_capture_and_replay(self):
        """No host synchronization or unmanaged allocation: the op can be captured and replayed."""
        C, H, R = REAL_SHAPE
        for io_elem in HALF_TYPES:
            for rows in (1, 2):
                with self.subTest(io_elem=io_elem, rows=rows):
                    dtype = DTYPES[io_elem]
                    inputs = make_inputs(dtype, rows, C, H, R, True, seed=9)
                    model = build_model(io_elem, C, H, R, True, ATTRS, weights=inputs)
                    so = ort.SessionOptions()
                    so.log_severity_level = 3
                    graph_session = ort.InferenceSession(
                        model.SerializeToString(),
                        so,
                        providers=[("CUDAExecutionProvider", {"enable_cuda_graph": "1"})],
                    )
                    binding = graph_session.io_binding()
                    keep = []
                    shapes = {"pre_output": (rows, H), "post_gate": (rows, C), "streams_out": (rows, C * H)}
                    outputs = {}
                    use_dlpack = torch.cuda.is_available()
                    if use_dlpack:
                        for name in ("streams", "prev_block_output", "prev_post_gate"):
                            dev_in = inputs[name].cuda().contiguous()
                            ortvalue = ort.OrtValue.from_dlpack(dev_in.__dlpack__())
                            keep.extend((dev_in, ortvalue))
                            binding.bind_ortvalue_input(name, ortvalue)
                        for name, shape in shapes.items():
                            dev_out = torch.empty(shape, dtype=torch.float32, device="cuda")
                            ort_out = ort.OrtValue.from_dlpack(dev_out.__dlpack__())
                            keep.extend((dev_out, ort_out))
                            outputs[name] = dev_out
                            binding.bind_ortvalue_output(name, ort_out)
                    else:
                        for name in ("streams", "prev_block_output", "prev_post_gate"):
                            ortvalue = ort.OrtValue.ortvalue_from_numpy(inputs[name].numpy(), "cuda", 0)
                            keep.append(ortvalue)
                            binding.bind_ortvalue_input(name, ortvalue)
                        for name, shape in shapes.items():
                            outputs[name] = ort.OrtValue.ortvalue_from_shape_and_type(shape, np.float32, "cuda", 0)
                            binding.bind_ortvalue_output(name, outputs[name])
                    graph_session.run_with_iobinding(binding)  # captures
                    graph_session.run_with_iobinding(binding)  # replays
                    expected = self.cache.get(io_elem, C, H, R, True, ATTRS).run(
                        None, {k: v.numpy() for k, v in inputs.items()}
                    )
                    for name, exp in zip(shapes, expected, strict=True):
                        actual = outputs[name].cpu().numpy() if use_dlpack else outputs[name].numpy()
                        np.testing.assert_array_equal(actual, exp, err_msg=name)

    def test_zero_rows(self):
        C, H, R = 4, 64, 16
        inputs = make_inputs(torch.float16, 0, C, H, R, False, seed=7)
        session = self.cache.get(TP.FLOAT16, C, H, R, False, ATTRS)
        pre, gate, streams_out = session.run(None, {k: v.numpy() for k, v in inputs.items()})
        self.assertEqual(pre.shape, (0, H))
        self.assertEqual(gate.shape, (0, C))
        self.assertEqual(streams_out.shape, (0, C * H))

    def test_unfused_prologue_without_streams_out(self):
        """When streams_out is omitted on the unfused path with a prologue, pre_output and post_gate stay exact."""
        C, H, R = 4, 64, 16
        rows = 16
        K = C * H
        weight_names = ("norm_scale", "down_weight", "up_weight", "inject_weight")
        for io_elem in HALF_TYPES:
            dtype = DTYPES[io_elem]
            with self.subTest(io_elem=io_elem):
                inputs = make_inputs(dtype, rows, C, H, R, True, seed=12)
                initializers = [make_initializer(k, io_elem, inputs[k]) for k in weight_names]
                nodes = [
                    helper.make_node("Cast", ["streams"], ["streams_t"], to=io_elem),
                    helper.make_node("Cast", ["prev_block_output"], ["prev_block_output_t"], to=io_elem),
                    helper.make_node("Cast", ["prev_post_gate"], ["prev_post_gate_t"], to=io_elem),
                    helper.make_node(
                        "HyperConnectionLowRankMix",
                        ["streams_t", *weight_names, "prev_block_output_t", "prev_post_gate_t"],
                        ["pre_output_t", "post_gate_t"],
                        domain="com.microsoft",
                        num_branches=C,
                        **ATTRS,
                    ),
                    helper.make_node("Cast", ["pre_output_t"], ["pre_output"], to=TP.FLOAT),
                    helper.make_node("Cast", ["post_gate_t"], ["post_gate"], to=TP.FLOAT),
                ]
                graph = helper.make_graph(
                    nodes,
                    "hc_unfused_no_streams_out",
                    [
                        helper.make_tensor_value_info("streams", TP.FLOAT, [rows, K]),
                        helper.make_tensor_value_info("prev_block_output", TP.FLOAT, [rows, H]),
                        helper.make_tensor_value_info("prev_post_gate", TP.FLOAT, [rows, C]),
                    ],
                    [
                        helper.make_tensor_value_info("pre_output", TP.FLOAT, [rows, H]),
                        helper.make_tensor_value_info("post_gate", TP.FLOAT, [rows, C]),
                    ],
                    initializer=initializers,
                )
                model = helper.make_model(
                    graph, opset_imports=[helper.make_opsetid("", 17), helper.make_opsetid("com.microsoft", 1)]
                )
                model.ir_version = 10
                so = ort.SessionOptions()
                so.log_severity_level = 3
                session = ort.InferenceSession(model.SerializeToString(), so, providers=["CUDAExecutionProvider"])
                pre, gate = session.run(
                    None,
                    {
                        "streams": inputs["streams"].numpy(),
                        "prev_block_output": inputs["prev_block_output"].numpy(),
                        "prev_post_gate": inputs["prev_post_gate"].numpy(),
                    },
                )
                exp_pre, exp_gate, _ = reference(inputs, dtype, C, H, ATTRS)
                rtol, atol = tolerances(io_elem)
                np.testing.assert_allclose(pre, exp_pre.numpy(), rtol=rtol, atol=atol)
                np.testing.assert_allclose(gate, exp_gate.numpy(), rtol=rtol, atol=atol)

    def test_unfused_fp16_cancelling_dot_products(self):
        """FP16 general-path GEMMs must disallow reduced-precision split-K reductions that would overflow FP16."""
        C, H = 4, 5120
        K = C * H
        half_k = K // 2
        attrs = dict(ATTRS, epsilon=0.0)
        for rows, R in ((9, 6), (9, 8)):
            with self.subTest(rows=rows, R=R):
                down = torch.zeros(K, R, dtype=torch.float32)
                down[: half_k - 1, :] = 32.0
                down[half_k : K - 1, :] = -32.0
                for j in range(R):
                    down[half_k - 1, j] = 0.0625 * ((j % 4) - 1)

                inject = torch.zeros(K, C, dtype=torch.float32)
                inject[: half_k - 1, :] = 32.0
                inject[half_k : K - 1, :] = -32.0
                for c in range(C):
                    inject[half_k - 1, c] = 0.0625 * (c - 1)

                inputs = {
                    "streams": torch.ones(rows, K, dtype=torch.float32),
                    "norm_scale": torch.full((K,), 16.0, dtype=torch.float32),
                    "down_weight": down,
                    "up_weight": torch.full((R, K), 0.125, dtype=torch.float32),
                    "inject_weight": inject,
                }
                session = self.cache.get(TP.FLOAT16, C, H, R, False, attrs)
                pre, gate, streams_out = session.run(None, {k: v.numpy() for k, v in inputs.items()})
                exp_pre, exp_gate, exp_streams = reference(inputs, torch.float16, C, H, attrs)
                rtol, atol = tolerances(TP.FLOAT16)
                np.testing.assert_allclose(pre, exp_pre.numpy(), rtol=rtol, atol=atol)
                np.testing.assert_allclose(gate, exp_gate.numpy(), rtol=rtol, atol=atol)
                np.testing.assert_allclose(streams_out, exp_streams.numpy(), rtol=rtol, atol=atol)


if __name__ == "__main__":
    unittest.main()
