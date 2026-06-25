# --------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.  See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

import unittest

import numpy
import torch
import torch.nn.functional as F
from cuda_plugin_ep_helper import resolve_cuda_plugin_ep
from onnx import helper

import onnxruntime

try:
    from onnx import TensorProto

    has_onnx = True
except ImportError:
    has_onnx = False

onnxruntime.preload_dlls()

build_info = onnxruntime.get_build_info()
has_fp8_qmoe = ", fp8-qmoe=" in build_info

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

torch.manual_seed(42)
numpy.random.seed(42)


def swiglu_ref(x, alpha=1.702, limit=7.0):
    dim = x.shape[-1]
    x = x.view(-1, dim // 2, 2)
    gate, linear = x[..., 0], x[..., 1]
    if limit is not None:
        gate = gate.clamp(max=limit)
        linear = linear.clamp(min=-limit, max=limit)
    return gate * torch.sigmoid(alpha * gate) * (linear + 1)


def quantize_weight_to_fp8(weight):
    if not hasattr(torch, "float8_e4m3fn"):
        raise unittest.SkipTest("PyTorch build does not expose torch.float8_e4m3fn")

    global_scale = torch.tensor(1.0, dtype=torch.float32, device=weight.device)
    fp8_weight = weight.float().to(torch.float8_e4m3fn)
    raw_weight = fp8_weight.view(torch.uint8).contiguous()
    dequantized = fp8_weight.float() * global_scale
    return raw_weight, global_scale, dequantized


def create_fp8_moe_onnx_graph(
    num_tokens,
    hidden_size,
    inter_size,
    num_experts,
    top_k,
    onnx_dtype,
    fc1_weights,
    fc1_global_scale,
    fc2_weights,
    fc2_global_scale,
    use_swiglu=False,
):
    if not hasattr(TensorProto, "FLOAT8E4M3FN"):
        raise unittest.SkipTest("ONNX TensorProto.FLOAT8E4M3FN is not available")

    inputs = [
        "input",  # 0
        "router_probs",  # 1
        "fc1_weights",  # 2: float8e4m3fn weights
        "",  # 3: fc1_scales, unused for fp8
        "",  # 4: fc1_bias
        "fc2_weights",  # 5: float8e4m3fn weights
        "",  # 6: fc2_scales, unused for fp8
        "",  # 7: fc2_bias
        "",  # 8: fc3_weights
        "",  # 9: fc3_scales
        "",  # 10: fc3_bias
        "",  # 11: fc1_zero_points
        "",  # 12: fc2_zero_points
        "",  # 13: fc3_zero_points
        "",  # 14: router_weights
        "fc1_global_scale",  # 15
        "fc2_global_scale",  # 16
    ]

    activation = "swiglu" if use_swiglu else "silu"
    nodes = [
        helper.make_node(
            "QMoE",
            inputs,
            ["output"],
            "QMoE_FP8",
            k=top_k,
            normalize_routing_weights=1,
            activation_type=activation,
            expert_weight_bits=8,
            quant_type="fp8",
            swiglu_fusion=1 if use_swiglu else 0,
            swiglu_limit=7.0,
            activation_alpha=1.702,
            activation_beta=1.0,
            domain="com.microsoft",
        )
    ]

    initializers = []
    for name, tensor in [("fc1_weights", fc1_weights), ("fc2_weights", fc2_weights)]:
        arr = numpy.ascontiguousarray(tensor.cpu().numpy().astype(numpy.uint8))
        initializers.append(
            helper.make_tensor(name, TensorProto.FLOAT8E4M3FN, list(tensor.shape), arr.tobytes(), raw=True)
        )

    for name, tensor in [("fc1_global_scale", fc1_global_scale), ("fc2_global_scale", fc2_global_scale)]:
        vals = tensor.cpu().float().flatten().tolist()
        initializers.append(helper.make_tensor(name, TensorProto.FLOAT, [num_experts], vals, raw=False))

    graph_inputs = [
        helper.make_tensor_value_info("input", onnx_dtype, [num_tokens, hidden_size]),
        helper.make_tensor_value_info("router_probs", onnx_dtype, [num_tokens, num_experts]),
    ]
    graph_outputs = [helper.make_tensor_value_info("output", onnx_dtype, [num_tokens, hidden_size])]

    graph = helper.make_graph(nodes, "QMoE_FP8_Test", graph_inputs, graph_outputs, initializers)
    model = helper.make_model(graph)
    return model.SerializeToString()


@unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
@unittest.skipIf(not has_onnx, "ONNX not available")
@unittest.skipIf(not has_fp8_qmoe, "CUDA QMoE FP8 kernels not enabled in this build")
class TestQMoEFP8(unittest.TestCase):
    def _run_fp8_moe_test(self, hidden_size, inter_size, num_experts, top_k, num_tokens, onnx_dtype, use_swiglu=False):
        torch.manual_seed(42)
        numpy.random.seed(42)

        torch_dtype = torch.float16 if onnx_dtype == TensorProto.FLOAT16 else torch.bfloat16
        fc1_n = 2 * inter_size if use_swiglu else inter_size
        fc2_n = hidden_size

        fc1_weights, fc1_scales, fc1_deq = [], [], []
        fc2_weights, fc2_scales, fc2_deq = [], [], []
        for _ in range(num_experts):
            w1 = torch.randn(fc1_n, hidden_size, device=device) * 0.1
            q1, s1, d1 = quantize_weight_to_fp8(w1)
            fc1_weights.append(q1)
            fc1_scales.append(s1)
            fc1_deq.append(d1)

            w2 = torch.randn(fc2_n, inter_size, device=device) * 0.1
            q2, s2, d2 = quantize_weight_to_fp8(w2)
            fc2_weights.append(q2)
            fc2_scales.append(s2)
            fc2_deq.append(d2)

        fc1_weights = torch.stack(fc1_weights, dim=0)
        fc2_weights = torch.stack(fc2_weights, dim=0)
        fc1_global_scale = torch.stack(fc1_scales)
        fc2_global_scale = torch.stack(fc2_scales)
        fc1_deq = torch.stack(fc1_deq, dim=0)
        fc2_deq = torch.stack(fc2_deq, dim=0)

        onnx_model = create_fp8_moe_onnx_graph(
            num_tokens=num_tokens,
            hidden_size=hidden_size,
            inter_size=inter_size,
            num_experts=num_experts,
            top_k=top_k,
            onnx_dtype=onnx_dtype,
            fc1_weights=fc1_weights,
            fc1_global_scale=fc1_global_scale,
            fc2_weights=fc2_weights,
            fc2_global_scale=fc2_global_scale,
            use_swiglu=use_swiglu,
        )

        opts = onnxruntime.SessionOptions()
        opts.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_DISABLE_ALL
        session = onnxruntime.InferenceSession(
            onnx_model, opts, providers=[resolve_cuda_plugin_ep("CUDAExecutionProvider")]
        )

        input_tensor = torch.randn(num_tokens, hidden_size, device=device, dtype=torch_dtype)
        router_logits = torch.randn(num_tokens, num_experts, device=device, dtype=torch_dtype)
        output_tensor = torch.zeros(num_tokens, hidden_size, device=device, dtype=torch_dtype)

        iobinding = session.io_binding()
        iobinding.bind_input("input", "cuda", 0, onnx_dtype, input_tensor.shape, input_tensor.data_ptr())
        iobinding.bind_input("router_probs", "cuda", 0, onnx_dtype, router_logits.shape, router_logits.data_ptr())
        iobinding.bind_output("output", "cuda", 0, onnx_dtype, output_tensor.shape, output_tensor.data_ptr())

        iobinding.synchronize_inputs()
        session.run_with_iobinding(iobinding)
        iobinding.synchronize_outputs()

        ref_output = self._compute_reference(
            input_tensor, router_logits, fc1_deq, fc2_deq, num_experts, top_k, use_swiglu, torch_dtype
        )
        max_diff = (output_tensor.float() - ref_output.float()).abs().max().item()
        dtype_tag = "FP16" if torch_dtype == torch.float16 else "BF16"
        act_tag = "SwiGLU" if use_swiglu else "SiLU"
        print(
            f"FP8 MoE test: {dtype_tag} {act_tag} tokens={num_tokens} experts={num_experts} "
            f"hidden={hidden_size} inter={inter_size} max_diff={max_diff:.6f}"
        )

        atol = 0.08 if torch_dtype == torch.bfloat16 else 0.05
        self.assertLess(max_diff, atol, f"FP8 MoE parity check failed: max_diff={max_diff:.6f} > atol={atol}")

    @staticmethod
    def _compute_reference(input_tensor, router_logits, fc1_deq, fc2_deq, num_experts, top_k, use_swiglu, torch_dtype):
        num_tokens = input_tensor.shape[0]
        hidden_size = input_tensor.shape[1]
        topk_vals, topk_idx = torch.topk(router_logits.float(), top_k, dim=-1)
        routing_weights = F.softmax(topk_vals, dim=1)

        output = torch.zeros(num_tokens, hidden_size, device=input_tensor.device, dtype=torch.float32)
        expert_mask = F.one_hot(topk_idx, num_classes=num_experts).permute(2, 1, 0)
        for expert in range(num_experts):
            idx, top_x = torch.where(expert_mask[expert])
            if top_x.shape[0] == 0:
                continue

            hidden = input_tensor.float()[top_x] @ fc1_deq[expert].float().T
            hidden = swiglu_ref(hidden) if use_swiglu else F.silu(hidden)
            hidden = hidden @ fc2_deq[expert].float().T
            hidden = hidden * routing_weights[top_x, idx, None]
            output.index_add_(0, top_x, hidden)

        return output.to(torch_dtype)

    def test_fp8_fp16_silu_basic(self):
        self._run_fp8_moe_test(256, 256, 4, 2, 32, TensorProto.FLOAT16)

    def test_fp8_bf16_silu_basic(self):
        self._run_fp8_moe_test(256, 256, 4, 2, 32, TensorProto.BFLOAT16)

    def test_fp8_fp16_swiglu(self):
        self._run_fp8_moe_test(256, 256, 4, 2, 32, TensorProto.FLOAT16, use_swiglu=True)

    def test_fp8_fp16_top4(self):
        self._run_fp8_moe_test(256, 256, 8, 4, 32, TensorProto.FLOAT16)


if __name__ == "__main__":
    unittest.main()
