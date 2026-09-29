# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
"""Numeric parity for the com.microsoft SwiGLU CUDA kernel against a PyTorch reference.

The T-typed inputs are fed as float and cast inside the graph, so the harness never has to hold
a bfloat16 numpy array.
"""

import unittest

import numpy as np
import torch
from onnx import TensorProto as TP  # noqa: N817
from onnx import helper

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


def swiglu_reference(gate, up, limit, alpha=1.0, beta=0.0):
    g, u = gate.clone(), up.clone()
    if limit > 0.0:
        g = g.clamp(max=limit)
        u = u.clamp(min=-limit, max=limit)
    return g * torch.sigmoid(alpha * g) * (u + beta)


def build_swiglu(elem, limit, fused, shape, alpha=1.0, beta=0.0):
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


@unittest.skipUnless(has_cuda(), "SwiGLU is a CUDA-only kernel")
class TestSwiGLU(unittest.TestCase):
    def assert_close(self, got, want, tol, tag):
        got = np.asarray(got, dtype=np.float32)
        want = np.asarray(want, dtype=np.float32)
        self.assertEqual(got.shape, want.shape, f"{tag}: shape {got.shape} != {want.shape}")
        max_diff = float(np.abs(got - want).max())
        self.assertLessEqual(max_diff, tol, f"{tag}: max |d| = {max_diff:.3e}")

    def _run(self, elem, limit, fused, alpha=1.0, beta=0.0):
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

        got = run(build_swiglu(elem, limit, fused, shape, alpha, beta), feeds)[0]

        gate = rt(raw[:, :cols], elem)
        up = rt(raw[:, cols:], elem)
        if limit > 0.0:
            self.assertTrue(bool((gate > limit).any()), "the gate half never reaches the limit")
            self.assertTrue(bool((up.abs() > limit).any()), "the up half never reaches the limit")
        want = rt(swiglu_reference(gate, up, limit, alpha, beta), elem)
        self.assert_close(
            got,
            want.numpy(),
            TOL[elem],
            f"swiglu limit={limit} alpha={alpha} beta={beta} fused={fused} {NAME_OF[elem]}",
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
                with self.subTest(dtype=NAME_OF[elem], alpha=alpha, beta=beta):
                    self._run(elem, 7.0, fused=False, alpha=alpha, beta=beta)
                    self._run(elem, 7.0, fused=True, alpha=alpha, beta=beta)

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
