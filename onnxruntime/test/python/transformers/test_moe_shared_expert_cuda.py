# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Parity tests for shared-expert fusion (``num_shared_experts``) in the
CUDA ``MoE`` and ``QMoE`` contrib ops.

The fused op treats the last ``num_shared_experts`` expert slots as always-on
shared experts: their (raw, pre-sigmoid) gate logits live in the last columns of
``router_probs`` and they are selected for every token with weight
``sigmoid(gate)``, excluded from the routed softmax / top-k / normalization::

    out = sum_topk( normalized_routed_w_i * RoutedExpert_i(x) )
        + sum_j   sigmoid(gate_j)        * SharedExpert_j(x)

The reference is built by running the *same* op for the routed-only part
(``num_shared_experts=0`` over the routed experts) and for each shared expert in
isolation (a single-expert op), then combining them in numpy with the sigmoid
gate. This avoids reconstructing the SwiGLU / weight layout by hand, so the test
isolates exactly the routing-fusion logic that was added.
"""

import unittest

import numpy as np
from onnx import TensorProto, helper

import onnxruntime

try:
    import torch
except ImportError:
    torch = None  # type: ignore[assignment]

try:
    from test_qmoe_cuda import quant_dequant_blockwise
except ImportError:
    quant_dequant_blockwise = None  # type: ignore[assignment]

HAS_CUDA = "CUDAExecutionProvider" in onnxruntime.get_available_providers()


def _make_moe_graph(num_experts, hidden_size, inter_size, top_k, num_shared_experts, normalize):
    """Build a single fp32 fused-SwiGLU MoE op graph.

    fc1 weights are ``[num_experts, 2*inter_size, hidden_size]`` (interleaved
    gate/up, ``swiglu_fusion=1``) and fc2 weights are
    ``[num_experts, hidden_size, inter_size]``. ``router_probs`` has
    ``num_experts`` columns.
    """
    nodes = [
        helper.make_node(
            "MoE",
            ["input", "router_probs", "fc1_experts_weights", "", "fc2_experts_weights"],
            ["output"],
            "MoE_0",
            k=top_k,
            normalize_routing_weights=normalize,
            use_sparse_mixer=0,
            activation_type="swiglu",
            swiglu_fusion=1,
            num_shared_experts=num_shared_experts,
            domain="com.microsoft",
        )
    ]

    fc1_shape = [num_experts, 2 * inter_size, hidden_size]
    fc2_shape = [num_experts, hidden_size, inter_size]
    initializers = [
        helper.make_tensor("fc1_experts_weights", TensorProto.FLOAT, fc1_shape, _W1.reshape(-1).tolist()),
        helper.make_tensor("fc2_experts_weights", TensorProto.FLOAT, fc2_shape, _W2.reshape(-1).tolist()),
    ]

    graph_inputs = [
        helper.make_tensor_value_info("input", TensorProto.FLOAT, ["num_rows", hidden_size]),
        helper.make_tensor_value_info("router_probs", TensorProto.FLOAT, ["num_rows", num_experts]),
    ]
    graph_outputs = [helper.make_tensor_value_info("output", TensorProto.FLOAT, ["num_rows", hidden_size])]

    graph = helper.make_graph(nodes, "moe_shared_expert_test", graph_inputs, graph_outputs, initializers)
    model = helper.make_model(
        graph, opset_imports=[helper.make_opsetid("com.microsoft", 1), helper.make_opsetid("", 17)]
    )
    return model.SerializeToString()


# Module-level weight tensors are referenced by _make_moe_graph (which slices
# them per-sub-test). Populated by each test via _set_weights.
_W1 = None
_W2 = None


def _set_weights(w1, w2):
    global _W1, _W2
    _W1, _W2 = w1, w2


def _run(model_bytes, inp, router):
    sess = onnxruntime.InferenceSession(model_bytes, providers=["CUDAExecutionProvider"])
    return sess.run(["output"], {"input": inp, "router_probs": router})[0]


class TestSharedExpertFusion(unittest.TestCase):
    @unittest.skipUnless(HAS_CUDA, "CUDA EP not available")
    def test_fused_shared_expert_parity(self):
        rng = np.random.default_rng(0)
        hidden_size = 64
        inter_size = 32
        num_routed = 8
        top_k = 2
        num_shared = 1
        num_rows = 6
        total = num_routed + num_shared

        w1 = (rng.standard_normal((total, 2 * inter_size, hidden_size)) * 0.05).astype(np.float32)
        w2 = (rng.standard_normal((total, hidden_size, inter_size)) * 0.05).astype(np.float32)
        inp = (rng.standard_normal((num_rows, hidden_size)) * 0.5).astype(np.float32)
        routed_logits = (rng.standard_normal((num_rows, num_routed)) * 1.0).astype(np.float32)
        shared_gate = (rng.standard_normal((num_rows, num_shared)) * 1.0).astype(np.float32)

        # --- Fused op: total experts, last one shared ---
        _set_weights(w1, w2)
        fused_model = _make_moe_graph(total, hidden_size, inter_size, top_k, num_shared, normalize=1)
        router_full = np.concatenate([routed_logits, shared_gate], axis=1).astype(np.float32)
        fused_out = _run(fused_model, inp, router_full)

        # --- Reference part 1: routed-only (normalized top-k over routed experts) ---
        _set_weights(w1[:num_routed], w2[:num_routed])
        routed_model = _make_moe_graph(num_routed, hidden_size, inter_size, top_k, 0, normalize=1)
        routed_out = _run(routed_model, inp, routed_logits)

        # --- Reference part 2: shared expert in isolation (single-expert op, weight 1.0) ---
        _set_weights(w1[num_routed:], w2[num_routed:])
        shared_model = _make_moe_graph(num_shared, hidden_size, inter_size, 1, 0, normalize=0)
        shared_router = shared_gate[:, :1].astype(np.float32)  # single column -> softmax weight 1.0
        shared_out = _run(shared_model, inp, shared_router)

        sigmoid_gate = 1.0 / (1.0 + np.exp(-shared_gate[:, :1]))
        reference = routed_out + sigmoid_gate * shared_out

        np.testing.assert_allclose(fused_out, reference, rtol=2e-3, atol=2e-3)


def _make_qmoe_graph(
    num_experts, hidden_size, inter_size, top_k, num_shared_experts, normalize, block_size, qweights, scales
):
    """Build a single fp16 INT4 fused-SwiGLU QMoE op graph from prepacked weights.

    ``qweights`` = (fc1_qweight, fc2_qweight); ``scales`` = (fc1_scales, fc2_scales),
    each already stacked over ``num_experts`` and laid out as the CUDA QMoE op
    expects (prepacked weights, scales ``[E, N, blocks]``).
    """
    fc1_qw, fc2_qw = qweights
    fc1_sc, fc2_sc = scales

    node = helper.make_node(
        "QMoE",
        ["input", "router_probs", "fc1_experts_weights", "fc1_scales", "", "fc2_experts_weights", "fc2_scales", ""],
        ["output"],
        "QMoE_0",
        k=top_k,
        normalize_routing_weights=normalize,
        use_sparse_mixer=0,
        activation_type="swiglu",
        swiglu_fusion=1,
        expert_weight_bits=4,
        block_size=block_size,
        num_shared_experts=num_shared_experts,
        domain="com.microsoft",
    )

    initializers = [
        helper.make_tensor(
            "fc1_experts_weights",
            TensorProto.UINT8,
            list(fc1_qw.shape),
            np.ascontiguousarray(fc1_qw.cpu().numpy().astype(np.uint8)).tobytes(),
            raw=True,
        ),
        helper.make_tensor(
            "fc2_experts_weights",
            TensorProto.UINT8,
            list(fc2_qw.shape),
            np.ascontiguousarray(fc2_qw.cpu().numpy().astype(np.uint8)).tobytes(),
            raw=True,
        ),
        helper.make_tensor(
            "fc1_scales", TensorProto.FLOAT16, list(fc1_sc.shape), fc1_sc.to(torch.float16).flatten().cpu().tolist()
        ),
        helper.make_tensor(
            "fc2_scales", TensorProto.FLOAT16, list(fc2_sc.shape), fc2_sc.to(torch.float16).flatten().cpu().tolist()
        ),
    ]

    graph_inputs = [
        helper.make_tensor_value_info("input", TensorProto.FLOAT16, ["num_rows", hidden_size]),
        helper.make_tensor_value_info("router_probs", TensorProto.FLOAT16, ["num_rows", num_experts]),
    ]
    graph_outputs = [helper.make_tensor_value_info("output", TensorProto.FLOAT16, ["num_rows", hidden_size])]
    graph = helper.make_graph([node], "qmoe_shared_expert_test", graph_inputs, graph_outputs, initializers)
    model = helper.make_model(
        graph, opset_imports=[helper.make_opsetid("com.microsoft", 1), helper.make_opsetid("", 17)]
    )
    return model.SerializeToString()


class TestSharedExpertFusionQMoE(unittest.TestCase):
    @unittest.skipUnless(HAS_CUDA, "CUDA EP not available")
    def test_fused_shared_expert_qmoe_parity(self):
        torch.manual_seed(0)
        hidden_size = 128
        inter_size = 64
        num_routed = 8
        top_k = 2
        num_shared = 1
        num_rows = 6
        block_size = 64
        total = num_routed + num_shared
        device = "cuda"

        # Per-expert fc1 = [2*inter, hidden] (interleaved gate/up), fc2 = [hidden, inter].
        fc1_w = torch.randn(total, 2 * inter_size, hidden_size, device=device) * 0.05
        fc2_w = torch.randn(total, hidden_size, inter_size, device=device) * 0.05

        def quantize_all(idx_lo, idx_hi):
            fc1_qw, fc1_sc, fc2_qw, fc2_sc = [], [], [], []
            for e in range(idx_lo, idx_hi):
                s1, q1, _, _ = quant_dequant_blockwise(fc1_w[e], block_size, True, False)
                s2, q2, _, _ = quant_dequant_blockwise(fc2_w[e], block_size, True, False)
                fc1_qw.append(q1)
                fc1_sc.append(s1)
                fc2_qw.append(q2)
                fc2_sc.append(s2)
            return (torch.stack(fc1_qw), torch.stack(fc2_qw)), (torch.stack(fc1_sc), torch.stack(fc2_sc))

        inp = (torch.randn(num_rows, hidden_size, device=device) * 0.5).to(torch.float16)
        routed_logits = torch.randn(num_rows, num_routed, device=device).to(torch.float16)
        shared_gate = torch.randn(num_rows, num_shared, device=device).to(torch.float16)

        def run(model_bytes, router):
            sess = onnxruntime.InferenceSession(model_bytes, providers=["CUDAExecutionProvider"])
            out = sess.run(
                ["output"],
                {
                    "input": inp.cpu().numpy(),
                    "router_probs": router.cpu().numpy(),
                },
            )[0]
            return out.astype(np.float32)

        # Fused: total experts, last shared.
        qw_all, sc_all = quantize_all(0, total)
        fused_model = _make_qmoe_graph(total, hidden_size, inter_size, top_k, num_shared, 1, block_size, qw_all, sc_all)
        router_full = torch.cat([routed_logits, shared_gate], dim=1)
        fused_out = run(fused_model, router_full)

        # Routed-only.
        qw_r, sc_r = quantize_all(0, num_routed)
        routed_model = _make_qmoe_graph(num_routed, hidden_size, inter_size, top_k, 0, 1, block_size, qw_r, sc_r)
        routed_out = run(routed_model, routed_logits)

        # Shared-only (single expert, weight 1.0).
        qw_s, sc_s = quantize_all(num_routed, total)
        shared_model = _make_qmoe_graph(num_shared, hidden_size, inter_size, 1, 0, 0, block_size, qw_s, sc_s)
        shared_out = run(shared_model, shared_gate[:, :1])

        sigmoid_gate = 1.0 / (1.0 + np.exp(-shared_gate[:, :1].cpu().numpy().astype(np.float32)))
        reference = routed_out + sigmoid_gate * shared_out

        # INT4 + fp16 GEMM: allow a small absolute tolerance on the combined output.
        np.testing.assert_allclose(fused_out, reference, rtol=5e-2, atol=5e-2)


if __name__ == "__main__":
    unittest.main()
