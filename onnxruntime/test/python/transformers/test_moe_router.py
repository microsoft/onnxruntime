# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
"""Numeric parity for the com.microsoft MoERouter CUDA kernel against a NumPy reference."""

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
SCORINGS = ["sqrt_softplus", "softmax", "sigmoid"]
TOL = {TP.FLOAT: 1e-6, TP.FLOAT16: 6e-3, TP.BFLOAT16: 4e-2}

# The kernel fills the unselected experts with a large negative value rather than -inf; fp16
# cannot hold -1e30, so that type uses -1e4, which masks identically under a softmax.
MASKED = {TP.FLOAT: -1e30, TP.BFLOAT16: -1e30, TP.FLOAT16: -1e4}


def rt(x, elem):
    """The round trip a Cast pair to the activation dtype performs."""
    return x.to(TORCH_OF[elem]).to(torch.float32)


def softplus(x):
    # ORT's Softplus keeps the exponent non-positive on both branches.
    return np.where(x > 0, x + np.log(np.exp(-np.abs(x)) + 1.0), np.log(np.exp(-np.abs(x)) + 1.0))


def affinity_of(scores, scoring):
    scores = scores.astype(np.float32)
    if scoring == "sqrt_softplus":
        return np.sqrt(softplus(scores))
    if scoring == "sigmoid":
        return 1.0 / (1.0 + np.exp(-scores))
    e = np.exp(scores - scores.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


def moe_router_reference(scores, bias, expert_ids, cfg, elem):
    tokens = scores.shape[0]
    topk = cfg["topk"]
    start, count = cfg["start"], cfg["count"]
    affinity = affinity_of(scores, cfg.get("scoring", "sqrt_softplus"))

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
    probs = "router_probs"
    if elem != TP.FLOAT:
        nodes.append(helper.make_node("Cast", [probs], [probs + "_f"], to=TP.FLOAT, name="cast_out"))
        probs += "_f"
    outputs = [
        helper.make_tensor_value_info(probs, TP.FLOAT, [tokens, cfg["count"]]),
        helper.make_tensor_value_info("weight_scale", TP.FLOAT, [tokens, 1]),
    ]
    graph = helper.make_graph(nodes, "moe_router", inputs, outputs)
    model = helper.make_model(
        graph, opset_imports=[helper.make_opsetid("", 17), helper.make_opsetid("com.microsoft", 1)]
    )
    model.ir_version = 10
    return model, [probs, "weight_scale"]


def run(model, feeds, out_names):
    sess = ort.InferenceSession(model.SerializeToString(), providers=["CUDAExecutionProvider"])
    return sess.run(out_names, feeds)


@unittest.skipUnless(has_cuda(), "MoERouter is a CUDA-only kernel")
class TestMoERouter(unittest.TestCase):
    def assert_close(self, got, want, tol, tag):
        got = np.asarray(got, dtype=np.float32)
        want = np.asarray(want, dtype=np.float32)
        self.assertEqual(got.shape, want.shape, f"{tag}: shape {got.shape} != {want.shape}")
        max_diff = float(np.abs(got - want).max())
        self.assertLessEqual(max_diff, tol, f"{tag}: max |d| = {max_diff:.3e}")

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

        tag = (
            f"router {NAME_OF[elem]} scoring={cfg.get('scoring', 'sqrt_softplus')} "
            f"start={cfg['start']} count={cfg['count']} ids={with_ids}"
        )
        self.assert_close(got_probs, want_probs, TOL[elem], tag + " probs")
        self.assert_close(got_scale, want_scale, 1e-6, tag + " scale")
        return got_probs, got_scale

    def test_scoring_with_noaux_tc(self):
        for scoring in SCORINGS:
            for elem in ELEM_TYPES:
                with self.subTest(scoring=scoring, dtype=NAME_OF[elem]):
                    cfg = {"topk": 4, "start": 0, "count": 16, "route_scale": 2.5, "scoring": scoring}
                    _, scale = self._run(cfg, elem)
                    # Every expert is local, so the whole weight comes back.
                    np.testing.assert_allclose(scale, cfg["route_scale"], rtol=1e-6)

    def test_scoring_with_topk(self):
        for scoring in SCORINGS:
            for elem in ELEM_TYPES:
                with self.subTest(scoring=scoring, dtype=NAME_OF[elem]):
                    cfg = {"topk": 3, "start": 0, "count": 16, "route_scale": 1.0, "selection": "topk"}
                    cfg["scoring"] = scoring
                    self._run(cfg, elem, with_bias=False)

    def test_many_experts(self):
        """More experts than threads in the block, so every reduction strides."""
        for scoring in SCORINGS:
            with self.subTest(scoring=scoring):
                cfg = {"topk": 8, "start": 128, "count": 256, "route_scale": 1.0, "scoring": scoring}
                self._run(cfg, TP.FLOAT, tokens=5, num_experts=384, seed=3)

    def test_expert_parallel_slicing(self):
        num_experts, topk = 32, 6
        for elem in ELEM_TYPES:
            for start in (0, 8, 24):
                with self.subTest(dtype=NAME_OF[elem], start=start):
                    cfg = {"topk": topk, "start": start, "count": 8, "route_scale": 1.5}
                    probs, scale = self._run(cfg, elem, tokens=9, num_experts=num_experts, seed=5)
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
            _, scale = self._run(dict(cfg, start=start), TP.FLOAT, tokens=9, num_experts=num_experts, seed=5)
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

    def test_hash_routing_ignores_out_of_range_ids(self):
        cfg = {"topk": 2, "start": 0, "count": 4, "route_scale": 1.0}
        scores = np.zeros((2, 4), dtype=np.float32)
        ids = np.array([[1, 1000], [-1, 4]], dtype=np.int64)
        model, names = build_moe_router(cfg, TP.FLOAT, 2, 4, with_bias=False, with_ids=True)
        probs, scale = run(model, {"scores": scores, "expert_ids": ids}, names)
        # The valid id keeps the whole weight; a row with none routes nowhere.
        self.assertAlmostEqual(float(probs[0, 1]), 0.0, places=6)
        self.assertAlmostEqual(float(scale[0, 0]), 1.0, places=6)
        self.assertTrue(bool((probs[1] <= MASKED[TP.FLOAT] * 0.5).all()))
        self.assertEqual(float(scale[1, 0]), 0.0)

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

    def test_unknown_scoring_is_rejected(self):
        cfg = {"topk": 2, "start": 0, "count": 4, "route_scale": 1.0, "scoring": "relu"}
        model, _ = build_moe_router(cfg, TP.FLOAT, 3, 4, with_bias=True, with_ids=False)
        with self.assertRaises(Exception) as cm:
            ort.InferenceSession(model.SerializeToString(), providers=["CUDAExecutionProvider"])
        self.assertIn("scoring must be", str(cm.exception))


if __name__ == "__main__":
    unittest.main()
