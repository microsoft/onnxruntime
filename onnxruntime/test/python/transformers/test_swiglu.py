# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
"""Numeric parity for the com.microsoft SwiGLU CUDA kernel against a PyTorch reference.

The T-typed inputs are fed as float and cast inside the graph, so the harness never has to hold
a bfloat16 numpy array.
"""

import os
import tempfile
import unittest

import numpy as np
import torch
from onnx import TensorProto as TP  # noqa: N817
from onnx import helper, load_model

import onnxruntime as ort


def has_cuda():
    return "CUDAExecutionProvider" in ort.get_available_providers()


TORCH_OF = {TP.FLOAT: torch.float32, TP.FLOAT16: torch.float16, TP.BFLOAT16: torch.bfloat16}
NAME_OF = {TP.FLOAT: "float32", TP.FLOAT16: "float16", TP.BFLOAT16: "bfloat16"}
ELEM_TYPES = [TP.FLOAT, TP.FLOAT16, TP.BFLOAT16]

# Both sides round to the same activation dtype, so the only gap is the fp32 intermediate, and a
# couple of ULPs of the output dtype covers it.
TOL = {TP.FLOAT: 2e-5, TP.FLOAT16: 6e-3, TP.BFLOAT16: 4e-2}


def rt(x, elem):
    """The round trip a Cast pair to the activation dtype performs."""
    return x.to(TORCH_OF[elem]).to(torch.float32)


def cast_in(nodes, name, elem):
    if elem == TP.FLOAT:
        return name
    nodes.append(helper.make_node("Cast", [name], [name + "_t"], to=elem, name="cast_in_" + name))
    return name + "_t"


def cast_out(nodes, name, elem):
    if elem == TP.FLOAT:
        return name
    nodes.append(helper.make_node("Cast", [name], [name + "_f"], to=TP.FLOAT, name="cast_out_" + name))
    return name + "_f"


def run(model, feeds):
    sess = ort.InferenceSession(model.SerializeToString(), providers=["CUDAExecutionProvider"])
    return sess.run(None, feeds)


def swiglu_reference(gate, up, limit, alpha=1.0, beta=0.0, round_activation_to=None):
    g, u = gate.clone(), up.clone()
    if limit > 0.0:
        g = g.clamp(max=limit)
        u = u.clamp(min=-limit, max=limit)
    activated = g * torch.sigmoid(alpha * g)
    if round_activation_to is not None:
        activated = rt(activated, round_activation_to)
    return activated * (u + beta)


def build_swiglu(elem, limit, fused, shape, alpha=1.0, beta=0.0, round_activation=0):
    nodes = []
    inputs = [helper.make_tensor_value_info("gate", TP.FLOAT, shape)]
    node_inputs = [cast_in(nodes, "gate", elem)]
    if not fused:
        inputs.append(helper.make_tensor_value_info("up", TP.FLOAT, shape))
        node_inputs.append(cast_in(nodes, "up", elem))
    nodes.append(
        helper.make_node(
            "SwiGLU",
            node_inputs,
            ["y"],
            name="swiglu",
            domain="com.microsoft",
            limit=limit,
            activation_alpha=alpha,
            activation_beta=beta,
            round_activation=round_activation,
        )
    )
    out_shape = list(shape)
    if fused:
        out_shape[-1] //= 2
    out = cast_out(nodes, "y", elem)
    graph = helper.make_graph(nodes, "swiglu", inputs, [helper.make_tensor_value_info(out, TP.FLOAT, out_shape)])
    model = helper.make_model(
        graph, opset_imports=[helper.make_opsetid("", 17), helper.make_opsetid("com.microsoft", 1)]
    )
    model.ir_version = 10
    return model


def build_quick_gelu_mul(elem, shape, alpha):
    """QuickGelu(gate) * up in the activation dtype, the pattern SwiGluFusion rewrites."""
    nodes = []
    gate = cast_in(nodes, "gate", elem)
    up = cast_in(nodes, "up", elem)
    nodes.append(helper.make_node("QuickGelu", [gate], ["gated"], domain="com.microsoft", alpha=alpha))
    nodes.append(helper.make_node("Mul", ["gated", up], ["y"]))
    out = cast_out(nodes, "y", elem)
    graph = helper.make_graph(
        nodes,
        "quick_gelu_mul",
        [helper.make_tensor_value_info(n, TP.FLOAT, shape) for n in ("gate", "up")],
        [helper.make_tensor_value_info(out, TP.FLOAT, shape)],
    )
    model = helper.make_model(
        graph, opset_imports=[helper.make_opsetid("", 17), helper.make_opsetid("com.microsoft", 1)]
    )
    model.ir_version = 10
    return model


def run_optimized(model, feeds, level):
    """Runs `model` at an optimization level and returns the output with the op types left in the graph."""
    with tempfile.TemporaryDirectory() as tmp:
        options = ort.SessionOptions()
        options.graph_optimization_level = level
        options.optimized_model_filepath = os.path.join(tmp, "optimized.onnx")
        sess = ort.InferenceSession(model.SerializeToString(), options, providers=["CUDAExecutionProvider"])
        ops = {n.op_type for n in load_model(options.optimized_model_filepath).graph.node}
        return sess.run(None, feeds)[0], ops


# (alpha, gate values, up): gate values whose QuickGelu output falls into the subnormal range of the
# activation dtype or flushes to zero. A large `up` scales the error from rounding that output back up.
# The second float case is the one where alpha * gate is far below float exp's range.
NEGATIVE_TAILS = {
    TP.FLOAT: [
        (1.0, [-88.0, -90.0, -95.0, -100.0, -103.0, -104.0], 1e30),
        (2.0**-93, [-(2.0**100), -(2.0**100) * 1.1], 1e20),
    ],
    TP.FLOAT16: [(1.0, [-12.0, -16.0, -17.0, -20.0, -24.0, -30.0], 65504.0)],
    TP.BFLOAT16: [(1.0, [-60.0, -90.0, -95.0, -96.0, -97.0, -100.0], 1e38)],
}


def tail_cases(elem):
    cases = []
    for alpha, tail, big in NEGATIVE_TAILS[elem]:
        gate = np.array([tail], dtype=np.float32)
        cases.append((alpha, {"gate": gate, "up": np.full_like(gate, big)}))
    return cases


def bulk_feeds(rows=4, cols=128):
    """Random values over the normal range, where a 1-ulp rounding difference is the only possible gap."""
    rng = np.random.default_rng(7)
    return 1.702, {
        "gate": rng.normal(0.0, 4.0, (rows, cols)).astype(np.float32),
        "up": rng.normal(0.0, 4.0, (rows, cols)).astype(np.float32),
    }


def parity_cases(elem):
    cases = {f"negative tail {i}": case for i, case in enumerate(tail_cases(elem))}
    cases["bulk"] = bulk_feeds()
    return cases


@unittest.skipUnless(has_cuda(), "SwiGLU is a CUDA-only kernel")
class TestSwiGLU(unittest.TestCase):
    def assert_close(self, got, want, tol, tag):
        got = np.asarray(got, dtype=np.float32)
        want = np.asarray(want, dtype=np.float32)
        self.assertEqual(got.shape, want.shape, f"{tag}: shape {got.shape} != {want.shape}")
        max_diff = float(np.abs(got - want).max())
        self.assertLessEqual(max_diff, tol, f"{tag}: max |d| = {max_diff:.3e}")

    def _run(self, elem, limit, fused, alpha=1.0, beta=0.0, round_activation=0):
        torch.manual_seed(0)
        rows, cols = 5, 128
        # Wide enough that a limit of 2 actually saturates both halves.
        raw = torch.randn(rows, 2 * cols) * 4.0
        if fused:
            shape = [rows, 2 * cols]
            feeds = {"gate": raw.numpy()}
        else:
            shape = [rows, cols]
            feeds = {"gate": raw[:, :cols].numpy().copy(), "up": raw[:, cols:].numpy().copy()}

        got = run(build_swiglu(elem, limit, fused, shape, alpha, beta, round_activation), feeds)[0]

        gate = rt(raw[:, :cols], elem)
        up = rt(raw[:, cols:], elem)
        if limit > 0.0:
            self.assertTrue(bool((gate > limit).any()), "the gate half never reaches the limit")
            self.assertTrue(bool((up.abs() > limit).any()), "the up half never reaches the limit")
        want = rt(swiglu_reference(gate, up, limit, alpha, beta, elem if round_activation else None), elem)
        self.assert_close(
            got,
            want.numpy(),
            TOL[elem],
            f"swiglu limit={limit} alpha={alpha} beta={beta} round={round_activation} fused={fused} {NAME_OF[elem]}",
        )

    def test_two_input(self):
        for elem in ELEM_TYPES:
            for limit in (0.0, 2.0):
                with self.subTest(dtype=NAME_OF[elem], limit=limit):
                    self._run(elem, limit, fused=False)

    def test_fused_single_input(self):
        for elem in ELEM_TYPES:
            for limit in (0.0, 2.0):
                with self.subTest(dtype=NAME_OF[elem], limit=limit):
                    self._run(elem, limit, fused=True)

    def test_activation_alpha_beta(self):
        """The GPT-OSS-style (alpha=1.702, beta=1.0) contract MoE/QMoE also implement."""
        for elem in ELEM_TYPES:
            for alpha, beta in ((1.702, 1.0), (0.5, -0.25)):
                for round_activation in (0, 1):
                    with self.subTest(dtype=NAME_OF[elem], alpha=alpha, beta=beta, round_activation=round_activation):
                        self._run(elem, 7.0, False, alpha, beta, round_activation)
                        self._run(elem, 7.0, True, alpha, beta, round_activation)

    def test_fused_matches_two_input(self):
        """The internal split must land on the same halves an explicit Split would."""
        torch.manual_seed(1)
        rows, cols = 3, 64
        raw = torch.randn(rows, 2 * cols) * 3.0
        fused = run(build_swiglu(TP.FLOAT, 1.5, True, [rows, 2 * cols]), {"gate": raw.numpy()})[0]
        split = run(
            build_swiglu(TP.FLOAT, 1.5, False, [rows, cols]),
            {"gate": raw[:, :cols].numpy().copy(), "up": raw[:, cols:].numpy().copy()},
        )[0]
        np.testing.assert_array_equal(fused, split)

    def test_float32_negative_gate_large_up(self):
        gate = torch.tensor([[-17.0, -20.0, -30.0]], dtype=torch.float32)
        up = torch.full_like(gate, 1e8)
        want = swiglu_reference(gate, up, 0.0).numpy()
        self.assertTrue(bool((want < 0.0).all()))
        for fused in (False, True):
            with self.subTest(fused=fused):
                if fused:
                    feeds = {"gate": torch.cat((gate, up), dim=-1).numpy()}
                    shape = [1, 6]
                else:
                    feeds = {"gate": gate.numpy(), "up": up.numpy()}
                    shape = [1, 3]
                got = run(build_swiglu(TP.FLOAT, 0.0, fused, shape), feeds)[0]
                np.testing.assert_allclose(got, want, rtol=2e-6, atol=0.0)

    def test_float32_underflowing_sigmoid_tail(self):
        """alpha * gate is below float exp's range, but the multiply by up brings the tail back."""
        gate = np.array([[-(2.0**100), -(2.0**100) * 1.1]], dtype=np.float32)
        up = np.full_like(gate, 1e20)
        alpha = 2.0**-93
        g = gate.astype(np.float64)
        e = np.exp(g * alpha)
        want = (g * e / (1.0 + e) * up.astype(np.float64)).astype(np.float32)
        self.assertTrue(bool((want < 0.0).all()))
        for fused in (False, True):
            with self.subTest(fused=fused):
                if fused:
                    feeds = {"gate": np.concatenate((gate, up), axis=-1)}
                    shape = [1, 4]
                else:
                    feeds = {"gate": gate, "up": up}
                    shape = [1, 2]
                got = run(build_swiglu(TP.FLOAT, 0.0, fused, shape, alpha=alpha), feeds)[0]
                np.testing.assert_allclose(got, want, rtol=1e-5, atol=0.0)

    def test_quick_gelu_mul_fusion_parity(self):
        """The automatic QuickGelu + Mul fusion must not change the unfused result, bit for bit."""
        for elem in ELEM_TYPES:
            for case, (alpha, feeds) in parity_cases(elem).items():
                with self.subTest(dtype=NAME_OF[elem], case=case):
                    model = build_quick_gelu_mul(elem, list(feeds["gate"].shape), alpha)
                    unfused, unfused_ops = run_optimized(model, feeds, ort.GraphOptimizationLevel.ORT_DISABLE_ALL)
                    fused, fused_ops = run_optimized(model, feeds, ort.GraphOptimizationLevel.ORT_ENABLE_ALL)
                    self.assertIn("QuickGelu", unfused_ops)
                    self.assertIn("SwiGLU", fused_ops)
                    self.assertNotIn("QuickGelu", fused_ops)
                    np.testing.assert_array_equal(fused, unfused)

    def test_round_activation_matches_quick_gelu_mul(self):
        """round_activation=1 reproduces QuickGelu followed by Mul, including where QuickGelu's output is lost."""
        for elem in ELEM_TYPES:
            for case, (alpha, feeds) in parity_cases(elem).items():
                with self.subTest(dtype=NAME_OF[elem], case=case):
                    shape = list(feeds["gate"].shape)
                    got = run(build_swiglu(elem, 0.0, False, shape, alpha=alpha, round_activation=1), feeds)[0]
                    want, _ = run_optimized(
                        build_quick_gelu_mul(elem, shape, alpha), feeds, ort.GraphOptimizationLevel.ORT_DISABLE_ALL
                    )
                    np.testing.assert_array_equal(got, want)

    def test_default_rounds_the_result_once(self):
        """Without round_activation the activation stays in float, so the tail keeps what QuickGelu + Mul loses."""
        rtol = {TP.FLOAT: 1e-5, TP.FLOAT16: 2e-3, TP.BFLOAT16: 1.6e-2}
        for elem in ELEM_TYPES:
            with self.subTest(dtype=NAME_OF[elem]):
                differs_from_rounded = False
                for alpha, feeds in tail_cases(elem):
                    shape = list(feeds["gate"].shape)
                    got = run(build_swiglu(elem, 0.0, False, shape, alpha=alpha), feeds)[0]
                    rounded = run(build_swiglu(elem, 0.0, False, shape, alpha=alpha, round_activation=1), feeds)[0]
                    differs_from_rounded |= bool((got != rounded).any())

                    gate = rt(torch.from_numpy(feeds["gate"]), elem).double()
                    up = rt(torch.from_numpy(feeds["up"]), elem).double()
                    want = rt((gate * torch.sigmoid(alpha * gate) * up).float(), elem).numpy()
                    np.testing.assert_allclose(got, want, rtol=rtol[elem], atol=0.0)
                self.assertTrue(differs_from_rounded, "round_activation changed nothing on the tail")

    def test_round_activation_flushes_the_tail(self):
        """The exact cases behind the fusion review: QuickGelu rounds to a subnormal or zero before the Mul."""
        cases = {
            # QuickGelu(-20) is -2^-24, the smallest fp16 subnormal, instead of -4.12e-8.
            TP.FLOAT16: (-20.0, 65504.0, -(2.0**-24) * 65504.0, -0.0027, 2e-3),
            # QuickGelu(-100) is below the smallest bf16 subnormal, so it is zero instead of -3.7e-42.
            TP.BFLOAT16: (-100.0, 1e38, 0.0, -3.7e-4, 2e-2),
        }
        for elem, (gate, up, rounded, unrounded, rtol) in cases.items():
            with self.subTest(dtype=NAME_OF[elem]):
                feeds = {"gate": np.array([[gate]], np.float32), "up": np.array([[up]], np.float32)}
                for round_activation, expected in ((1, rounded), (0, unrounded)):
                    got = run(build_swiglu(elem, 0.0, False, [1, 1], round_activation=round_activation), feeds)[0]
                    np.testing.assert_allclose(got, [[expected]], rtol=rtol, atol=0.0)

    def test_rejects_mismatched_shapes(self):
        model = build_swiglu(TP.FLOAT, 0.0, False, [2, 8])
        model.graph.input[1].type.tensor_type.shape.dim[0].dim_value = 1
        with self.assertRaisesRegex(Exception, "must match"):
            run(model, {"gate": np.zeros((2, 8), np.float32), "up": np.zeros((1, 8), np.float32)})

    def test_rejects_odd_fused_width(self):
        model = build_swiglu(TP.FLOAT, 0.0, True, [2, 7])
        with self.assertRaisesRegex(Exception, "must be even"):
            run(model, {"gate": np.zeros((2, 7), np.float32)})


if __name__ == "__main__":
    unittest.main()
