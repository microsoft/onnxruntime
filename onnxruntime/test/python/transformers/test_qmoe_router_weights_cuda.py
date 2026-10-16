# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# -------------------------------------------------------------------------
"""CUDA QMoE: separate router_weights (noaux_tc routing) and single-token top-k 9 decode with relu2 experts.

router_probs only selects the experts; router_weights supplies the mixing weights. The reference is a plain torch
evaluation of the dequantized int4 block-wise experts. T = 1 with k = 9 exercises the MoE decode GEMV with 9 expanded rows.
"""

import os
import unittest

import onnx
import torch
from onnx import TensorProto, helper

import onnxruntime as ort

BLOCK = 32
HIDDEN = 256
INTER = 256


_plugin_state = {"registered": False}


def _session(model_path):
    options = ort.SessionOptions()
    plugin = os.environ.get("ORT_TEST_CUDA_PLUGIN_EP_LIBRARY")
    if plugin:  # run against a plugin-EP build
        if not _plugin_state["registered"]:
            ort.register_execution_provider_library("CUDAExecutionProvider", plugin)
            _plugin_state["registered"] = True
        devices = [d for d in ort.get_ep_devices() if d.ep_name == "CUDAExecutionProvider"]
        options.add_provider_for_devices(devices[:1], {})
        return ort.InferenceSession(model_path, sess_options=options)
    return ort.InferenceSession(model_path, sess_options=options, providers=["CUDAExecutionProvider"])


def _quantize_int4_blockwise(weight):
    """[E, N, K] float -> (packed uint8 [E, N, K/2], scales [E, N, K/BLOCK], dequantized [E, N, K])."""
    e, n, k = weight.shape
    blocks = weight.reshape(e, n, k // BLOCK, BLOCK)
    scale = blocks.abs().amax(-1, keepdim=True) / 7 + 1e-8
    q = torch.clamp(torch.round(blocks / scale), -8, 7) + 8
    dequant = ((q - 8) * scale).reshape(e, n, k)
    q = q.reshape(e, n, k).to(torch.uint8)
    packed = q[..., 0::2] | (q[..., 1::2] << 4)
    return packed, scale.squeeze(-1), dequant


def _tensor(name, t):
    kind = {torch.uint8: TensorProto.UINT8, torch.float16: TensorProto.FLOAT16, torch.bfloat16: TensorProto.BFLOAT16}[
        t.dtype
    ]
    raw = t.view(torch.int16).numpy().tobytes() if t.dtype in (torch.float16, torch.bfloat16) else t.numpy().tobytes()
    return helper.make_tensor(name, kind, list(t.shape), raw, raw=True)


@unittest.skipUnless(
    torch.cuda.is_available()
    and ("CUDAExecutionProvider" in ort.get_available_providers() or os.environ.get("ORT_TEST_CUDA_PLUGIN_EP_LIBRARY")),
    "CUDA EP not available",
)
class TestQMoERouterWeightsCuda(unittest.TestCase):
    def _run(self, dtype, activation, tokens, num_experts, top_k, use_router_weights):
        torch.manual_seed(0)
        ort_type = TensorProto.BFLOAT16 if dtype == torch.bfloat16 else TensorProto.FLOAT16
        w1, w2 = torch.randn(num_experts, INTER, HIDDEN) * 0.05, torch.randn(num_experts, HIDDEN, INTER) * 0.05
        p1, s1, d1 = _quantize_int4_blockwise(w1)
        p2, s2, d2 = _quantize_int4_blockwise(w2)
        x = (torch.randn(tokens, HIDDEN) * 0.5).to(dtype)

        logits = torch.randn(tokens, num_experts)
        selected = torch.topk(logits, top_k)[1]
        probs = torch.zeros(tokens, num_experts).scatter(1, selected, 1.0).to(dtype)  # selection only
        mix = torch.rand(tokens, num_experts).to(dtype)  # aggregation weights of every expert

        inputs = ["x", "probs", "w1", "s1", "", "w2", "s2", "", "", "", "", "", "", ""]
        graph_inputs = [
            helper.make_tensor_value_info("x", ort_type, [tokens, HIDDEN]),
            helper.make_tensor_value_info("probs", ort_type, [tokens, num_experts]),
        ]
        if use_router_weights:
            inputs.append("mix")
            graph_inputs.append(helper.make_tensor_value_info("mix", ort_type, [tokens, num_experts]))
        node = helper.make_node(
            "QMoE",
            inputs,
            ["y"],
            domain="com.microsoft",
            k=top_k,
            activation_type=activation,
            normalize_routing_weights=0,
            expert_weight_bits=4,
            block_size=BLOCK,
            weights_prepacked=0,
        )
        graph = helper.make_graph(
            [node],
            "qmoe_router_weights",
            graph_inputs,
            [helper.make_tensor_value_info("y", ort_type, [tokens, HIDDEN])],
            initializer=[
                _tensor("w1", p1),
                _tensor("s1", s1.to(dtype)),
                _tensor("w2", p2),
                _tensor("s2", s2.to(dtype)),
            ],
        )
        model = helper.make_model(
            graph, opset_imports=[helper.make_opsetid("", 21), helper.make_opsetid("com.microsoft", 1)]
        )
        model.ir_version = 10
        path = os.path.join(os.environ.get("TMPDIR", "/tmp"), f"qmoe_rw_{os.getpid()}.onnx")
        onnx.save(model, path)
        try:
            session = _session(path)
        finally:
            os.remove(path)
        binding = session.io_binding()
        for name, t in (("x", x), ("probs", probs)) + ((("mix", mix),) if use_router_weights else ()):
            binding.bind_ortvalue_input(name, ort.OrtValue.from_dlpack(t.cuda().contiguous().__dlpack__()))
        binding.bind_output("y", "cuda", 0)
        session.run_with_iobinding(binding)
        y = torch.from_dlpack(binding.get_outputs()[0]).float().cpu()

        act = {"relu": torch.relu, "relu2": lambda h: torch.relu(h) ** 2, "silu": torch.nn.functional.silu}[activation]
        # without router_weights the op softmaxes router_probs and uses the result for both selection and aggregation
        weights = mix.float() if use_router_weights else torch.softmax(probs.float(), dim=-1)
        ref = torch.zeros(tokens, HIDDEN)
        for t in range(tokens):
            for e in selected[t].tolist():
                h = act(x[t].float() @ d1[e].to(dtype).float().T).to(dtype).float()
                ref[t] += (h @ d2[e].to(dtype).float().T) * weights[t, e].float()
        return ((y - ref).norm() / ref.norm()).item()

    def test_router_weights_aggregation(self):
        for dtype in (torch.float16, torch.bfloat16):
            for tokens in (1, 5):
                with self.subTest(dtype=dtype, tokens=tokens):
                    self.assertLess(self._run(dtype, "relu", tokens, 16, 4, True), 0.03)

    def test_relu2_top9_decode(self):
        for tokens in (1, 3):
            with self.subTest(tokens=tokens):
                self.assertLess(self._run(torch.bfloat16, "relu2", tokens, 32, 9, True), 0.03)
                self.assertLess(self._run(torch.bfloat16, "relu2", tokens, 32, 9, False), 0.03)


if __name__ == "__main__":
    unittest.main()
