# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
"""Numeric parity for the fused LLM building blocks in com.microsoft:

  SwiGLU, SinkhornNormalize, QKNormRotaryEmbedding, InverseRotaryRegroup, MoERouter

Every reference here is a transcription of the schema doc in contrib_defs.cc, including the
places the fused kernel rounds to the activation dtype. The T-typed inputs are fed as float
and cast inside the graph, so the harness never has to hold a bfloat16 numpy array.
"""

# Tensor dimensions keep the upper-case names the schema doc gives them.

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


def rt(x, elem):
    """The round trip a Cast pair to the activation dtype performs."""
    return x.to(TORCH_OF[elem]).to(torch.float32)


def cast_in(nodes, name, elem):
    """Narrow a float graph input to the activation dtype the kernel is registered for."""
    if elem == TP.FLOAT:
        return name
    nodes.append(helper.make_node("Cast", [name], [name + "_t"], to=elem, name="cast_in_" + name))
    return name + "_t"


def cast_out(nodes, name, elem):
    if elem == TP.FLOAT:
        return name
    nodes.append(helper.make_node("Cast", [name], [name + "_f"], to=TP.FLOAT, name="cast_out_" + name))
    return name + "_f"


def make_model(nodes, inputs, outputs, name):
    graph = helper.make_graph(nodes, name, inputs, outputs)
    model = helper.make_model(
        graph, opset_imports=[helper.make_opsetid("", 17), helper.make_opsetid("com.microsoft", 1)]
    )
    model.ir_version = 10
    return model


def run(model, feeds, out_names=None):
    sess = ort.InferenceSession(model.SerializeToString(), providers=["CUDAExecutionProvider"])
    return sess.run(out_names, feeds)


def block_scale(x, block, limit, floor):
    """The graph's Log/Div/Ceil/Pow chain: a power-of-two scale per block of `block`."""
    a = x.reshape(-1, block).abs().amax(-1, keepdim=True)
    r = (a / limit).clamp(min=floor)
    s = torch.pow(torch.tensor(2.0), torch.ceil(torch.log(r) / float(np.log(2.0))))
    return torch.where(a > 0, s, torch.ones_like(s))


def fp8_round_trip(x, block=64):
    s = block_scale(x, block, 448.0, 1e-30)
    q = (x.reshape(-1, block) / s).clamp(-448.0, 448.0)
    q = q.to(torch.float8_e4m3fn).to(torch.float32)
    return (q * s).reshape(x.shape)


# Tolerances: both sides round to the same activation dtype, so the only gap is the fp32
# intermediate, and a couple of ULPs of the output dtype covers it.
TOL = {TP.FLOAT: 2e-5, TP.FLOAT16: 6e-3, TP.BFLOAT16: 4e-2}


class FusedOpTestCase(unittest.TestCase):
    def assert_close(self, got, want, tol, tag, max_frac=0.0):
        got = np.asarray(got, dtype=np.float32)
        want = np.asarray(want, dtype=np.float32)
        self.assertEqual(got.shape, want.shape, f"{tag}: shape {got.shape} != {want.shape}")
        ad = np.abs(got - want)
        if max_frac <= 0.0:
            self.assertLessEqual(float(ad.max()), tol, f"{tag}: max |d| = {ad.max():.3e}")
            return
        # A simulated FP8 grid is a step function, so a 1-ULP difference in the reduction that
        # feeds it moves a value a whole step. Rare flips are expected; a systematic error is
        # not, which is what the mean guards.
        frac = float((ad > tol).mean())
        self.assertLessEqual(frac, max_frac, f"{tag}: {frac * 100:.3f}% over {tol} (max {ad.max():.3e})")
        self.assertLessEqual(float(ad.mean()), tol, f"{tag}: mean |d| = {ad.mean():.3e}")


# --------------------------------------------------------------------------------------
# SwiGLU
# --------------------------------------------------------------------------------------


def swiglu_reference(gate, up, limit, alpha=1.0, beta=0.0):
    g, u = gate.clone(), up.clone()
    if limit > 0.0:
        g = g.clamp(max=limit)
        u = u.clamp(min=-limit, max=limit)
    return g * torch.sigmoid(alpha * g) * (u + beta)


def build_swiglu(elem, limit, fused, shape, alpha=1.0, beta=0.0):
    nodes = []
    inputs = [helper.make_tensor_value_info("gate", TP.FLOAT, shape)]
    gate = cast_in(nodes, "gate", elem)
    node_inputs = [gate]
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
    return make_model(nodes, inputs, [helper.make_tensor_value_info(out, TP.FLOAT, out_shape)], "swiglu")


@unittest.skipUnless(has_cuda(), "SwiGLU is a CUDA-only kernel")
class TestSwiGLU(FusedOpTestCase):
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


# --------------------------------------------------------------------------------------
# MoERouter
# --------------------------------------------------------------------------------------


def softplus(x):
    # ORT's Softplus keeps the exponent non-positive on both branches.
    return np.where(x > 0, x + np.log(np.exp(-np.abs(x)) + 1.0), np.log(np.exp(-np.abs(x)) + 1.0))


# The graph fills the unselected experts with a large negative value rather than -inf; fp16
# cannot hold -1e30, so that build uses -1e4, which masks identically under a softmax.
MASKED = {TP.FLOAT: -1e30, TP.BFLOAT16: -1e30, TP.FLOAT16: -1e4}


def moe_router_reference(scores, bias, expert_ids, cfg, elem):
    tokens, num_experts = scores.shape
    topk = cfg["topk"]
    start, count = cfg["start"], cfg["count"]
    affinity = np.sqrt(softplus(scores.astype(np.float32)))

    probs = np.full((tokens, count), MASKED[elem], dtype=np.float32)
    scale = np.zeros((tokens, 1), dtype=np.float32)
    for t in range(tokens):
        if expert_ids is not None:
            chosen = [int(e) for e in expert_ids[t]]
        else:
            sel = affinity[t].copy()
            if bias is not None:
                sel = sel + bias
            chosen = []
            for _ in range(topk):
                # Largest wins, lowest expert index breaks a tie.
                best = int(np.argmax(sel))
                chosen.append(best)
                sel[best] = -np.inf
        weights = affinity[t, chosen]
        weights = weights / weights.sum()
        local = 0.0
        for j, e in enumerate(chosen):
            if start <= e < start + count:
                probs[t, e - start] = np.log(weights[j])
                local += float(weights[j])
        scale[t, 0] = local * cfg["route_scale"]
    return rt(torch.from_numpy(probs), elem).numpy(), scale


def build_moe_router(cfg, elem, tokens, num_experts, with_bias, with_ids):
    nodes = []
    inputs = [helper.make_tensor_value_info("scores", TP.FLOAT, [tokens, num_experts])]
    node_inputs = ["scores"]
    if with_bias:
        inputs.append(helper.make_tensor_value_info("bias", TP.FLOAT, [num_experts]))
        node_inputs.append("bias")
    elif with_ids:
        node_inputs.append("")
    if with_ids:
        inputs.append(helper.make_tensor_value_info("expert_ids", TP.INT64, [tokens, cfg["topk"]]))
        node_inputs.append("expert_ids")

    nodes.append(
        helper.make_node(
            "MoERouter",
            node_inputs,
            ["router_probs", "weight_scale"],
            name="router",
            domain="com.microsoft",
            topk=cfg["topk"],
            scoring=cfg.get("scoring", "sqrt_softplus"),
            selection=cfg.get("selection", "noaux_tc"),
            local_expert_start=cfg["start"],
            local_expert_count=cfg["count"],
            route_scale=cfg["route_scale"],
            dtype=int(elem),
        )
    )
    probs = cast_out(nodes, "router_probs", elem)
    outputs = [
        helper.make_tensor_value_info(probs, TP.FLOAT, [tokens, cfg["count"]]),
        helper.make_tensor_value_info("weight_scale", TP.FLOAT, [tokens, 1]),
    ]
    return make_model(nodes, inputs, outputs, "moe_router"), [probs, "weight_scale"]


@unittest.skipUnless(has_cuda(), "MoERouter is a CUDA-only kernel")
class TestMoERouter(FusedOpTestCase):
    def _run(self, cfg, elem, tokens=7, num_experts=16, with_bias=True, with_ids=False, seed=0):
        rng = np.random.default_rng(seed)
        scores = rng.standard_normal((tokens, num_experts), dtype=np.float32)
        feeds = {"scores": scores}
        bias = None
        if with_bias:
            bias = rng.standard_normal(num_experts, dtype=np.float32) * 0.3
            feeds["bias"] = bias
        expert_ids = None
        if with_ids:
            expert_ids = np.stack([rng.permutation(num_experts)[: cfg["topk"]] for _ in range(tokens)]).astype(np.int64)
            feeds["expert_ids"] = expert_ids

        model, names = build_moe_router(cfg, elem, tokens, num_experts, with_bias, with_ids)
        got_probs, got_scale = run(model, feeds, names)
        want_probs, want_scale = moe_router_reference(scores, bias, expert_ids, cfg, elem)

        tag = f"router {NAME_OF[elem]} start={cfg['start']} count={cfg['count']} ids={with_ids}"
        self.assert_close(got_probs, want_probs, TOL[elem] if elem != TP.FLOAT else 1e-6, tag + " probs")
        self.assert_close(got_scale, want_scale, 1e-6, tag + " scale")
        return got_probs, got_scale, want_probs

    def test_sqrt_softplus_noaux_tc(self):
        for elem in ELEM_TYPES:
            with self.subTest(dtype=NAME_OF[elem]):
                cfg = {"topk": 4, "start": 0, "count": 16, "route_scale": 2.5}
                _, scale, _ = self._run(cfg, elem)
                # Every expert is local, so the whole weight comes back.
                np.testing.assert_allclose(scale, cfg["route_scale"], rtol=1e-6)

    def test_selection_topk_without_bias(self):
        for elem in ELEM_TYPES:
            with self.subTest(dtype=NAME_OF[elem]):
                cfg = {"topk": 3, "start": 0, "count": 16, "route_scale": 1.0, "selection": "topk"}
                self._run(cfg, elem, with_bias=False)

    def test_expert_parallel_slicing(self):
        num_experts, topk = 32, 6
        for elem in ELEM_TYPES:
            for start in (0, 8, 24):
                with self.subTest(dtype=NAME_OF[elem], start=start):
                    cfg = {"topk": topk, "start": start, "count": 8, "route_scale": 1.5}
                    probs, scale, want = self._run(cfg, elem, tokens=9, num_experts=num_experts, seed=5)
                    masked = probs <= MASKED[elem] * 0.5
                    self.assertTrue(masked.any(), "no expert was masked out on this rank")
                    # A token with no local expert must get a zero scale, which annihilates the
                    # degenerate uniform softmax of an all-negative row.
                    for t in range(probs.shape[0]):
                        if masked[t].all():
                            self.assertEqual(float(scale[t, 0]), 0.0)
                        else:
                            self.assertGreater(float(scale[t, 0]), 0.0)

    def test_slices_partition_the_weight(self):
        """The per-rank scales must add back up to route_scale * 1."""
        num_experts, topk, count = 32, 6, 8
        cfg = {"topk": topk, "start": 0, "count": count, "route_scale": 1.0}
        total = None
        for start in range(0, num_experts, count):
            _, scale, _ = self._run(dict(cfg, start=start), TP.FLOAT, tokens=9, num_experts=num_experts, seed=5)
            total = scale if total is None else total + scale
        np.testing.assert_allclose(total, 1.0, atol=1e-6)

    def test_hash_routing(self):
        for elem in ELEM_TYPES:
            with self.subTest(dtype=NAME_OF[elem]):
                cfg = {"topk": 4, "start": 4, "count": 8, "route_scale": 1.0}
                self._run(cfg, elem, num_experts=16, with_bias=False, with_ids=True, seed=7)

    def test_hash_routing_overrides_selection(self):
        """The ids fix the choice, so the affinities must not be able to move it."""
        cfg = {"topk": 2, "start": 0, "count": 4, "route_scale": 1.0}
        scores = np.array([[3.0, -2.0, -2.0, 1.0]], dtype=np.float32)
        ids = np.array([[1, 2]], dtype=np.int64)
        model, names = build_moe_router(cfg, TP.FLOAT, 1, 4, with_bias=False, with_ids=True)
        probs, scale = run(model, {"scores": scores, "expert_ids": ids}, names)
        self.assertLess(probs[0, 0], MASKED[TP.FLOAT] * 0.5)
        self.assertLess(probs[0, 3], MASKED[TP.FLOAT] * 0.5)
        aff = np.sqrt(softplus(scores[0, [1, 2]]))
        np.testing.assert_allclose(np.exp(probs[0, [1, 2]]), aff / aff.sum(), rtol=1e-6)
        np.testing.assert_allclose(scale, 1.0, rtol=1e-6)

    def test_selection_topk_rejects_bias(self):
        cfg = {"topk": 2, "start": 0, "count": 4, "route_scale": 1.0, "selection": "topk"}
        model, names = build_moe_router(cfg, TP.FLOAT, 3, 4, with_bias=True, with_ids=False)
        rng = np.random.default_rng(0)
        feeds = {
            "scores": rng.standard_normal((3, 4), dtype=np.float32),
            "bias": rng.standard_normal(4, dtype=np.float32),
        }
        with self.assertRaises(Exception) as cm:
            run(model, feeds, names)
        self.assertIn("bias is only used by selection='noaux_tc'", str(cm.exception))

    def test_unsupported_scoring_is_rejected(self):
        cfg = {"topk": 2, "start": 0, "count": 4, "route_scale": 1.0, "scoring": "softmax"}
        model, _ = build_moe_router(cfg, TP.FLOAT, 3, 4, with_bias=True, with_ids=False)
        with self.assertRaises(Exception) as cm:
            ort.InferenceSession(model.SerializeToString(), providers=["CUDAExecutionProvider"])
        self.assertIn("sqrt_softplus", str(cm.exception))


if __name__ == "__main__":
    unittest.main()
