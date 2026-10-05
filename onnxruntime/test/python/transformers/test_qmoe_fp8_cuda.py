# --------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.  See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

import os
import unittest

import numpy
import pytest
import torch
import torch.nn.functional as F
from cuda_plugin_ep_helper import resolve_cuda_plugin_ep
from onnx import helper, load_model_from_string

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


def create_block_fp8_moe_graph(tensors, top_k, onnx_dtype, block_size, fusion=0, normalize=1, initializers=False):
    """Use runtime weights so the official 512-expert shape needs no >2GB protobuf."""
    names = [""] * 17
    indices = {
        "input": 0,
        "router_probs": 1,
        "fc1_weights": 2,
        "fc1_scales": 3,
        "fc1_bias": 4,
        "fc2_weights": 5,
        "fc2_scales": 6,
        "fc2_bias": 7,
        "fc3_weights": 8,
        "fc3_scales": 9,
        "fc1_zero_points": 11,
        "fc2_zero_points": 12,
        "fc3_zero_points": 13,
        "fc1_global_scale": 15,
        "fc2_global_scale": 16,
    }
    graph_inputs = []
    graph_initializers = []
    input_types = {}
    for name, tensor in tensors.items():
        names[indices[name]] = name
        if name.endswith("_zero_points"):
            dtype = TensorProto.UINT8
        elif "weights" in name:
            dtype = TensorProto.FLOAT8E4M3FN
        elif name in ("input", "router_probs"):
            dtype = onnx_dtype
        else:
            dtype = {
                torch.float32: TensorProto.FLOAT,
                torch.float16: TensorProto.FLOAT16,
                torch.bfloat16: TensorProto.BFLOAT16,
                torch.int32: TensorProto.INT32,
            }[tensor.dtype]
        input_types[name] = dtype
        if initializers and name not in ("input", "router_probs"):
            raw = tensor.contiguous().view(torch.uint8).cpu().numpy().tobytes()
            graph_initializers.append(helper.make_tensor(name, dtype, list(tensor.shape), raw, raw=True))
        else:
            graph_inputs.append(helper.make_tensor_value_info(name, dtype, list(tensor.shape)))
    node = helper.make_node(
        "QMoE",
        names,
        ["output"],
        domain="com.microsoft",
        k=top_k,
        normalize_routing_weights=normalize,
        activation_type="swiglu" if fusion else "silu",
        swiglu_fusion=fusion,
        expert_weight_bits=8,
        quant_type="fp8",
        block_size=block_size,
    )
    graph = helper.make_graph(
        [node],
        "BlockFP8",
        graph_inputs,
        [helper.make_tensor_value_info("output", onnx_dtype, list(tensors["input"].shape))],
        graph_initializers,
    )
    model = helper.make_model(
        graph, opset_imports=[helper.make_opsetid("", 21), helper.make_opsetid("com.microsoft", 1)]
    )
    return model.SerializeToString(), input_types


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


@unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
@unittest.skipIf(not has_onnx, "ONNX not available")
@unittest.skipIf(not has_fp8_qmoe, "CUDA QMoE FP8 kernels not enabled in this build")
class TestQMoEBlockFP8(unittest.TestCase):
    @staticmethod
    def _inputs(
        hidden=160,
        inter=80,
        experts=16,
        tokens=3,
        block=128,
        fusion=0,
        dtype=torch.bfloat16,
        scale_dtype=torch.bfloat16,
    ):
        if not hasattr(torch, "float8_e4m3fn"):
            raise unittest.SkipTest("PyTorch build does not expose torch.float8_e4m3fn")

        torch.manual_seed(2026)
        tensors = {
            "input": torch.randn(tokens, hidden, device=device, dtype=dtype) * 0.2,
            "router_probs": torch.randn(tokens, experts, device=device, dtype=dtype),
        }
        shapes = {"fc1": (2 * inter if fusion else inter, hidden), "fc2": (hidden, inter)}
        if not fusion:
            shapes["fc3"] = (inter, hidden)
        for name, (n, k) in shapes.items():
            tensors[f"{name}_weights"] = (
                (torch.randn(experts, n, k, device=device) * 0.1).to(torch.float8_e4m3fn).view(torch.uint8)
            )
            # Deliberately vary every expert and both block coordinates, not just a global scalar.
            tensors[f"{name}_scales"] = (
                torch.rand(experts, (n + block - 1) // block, (k + block - 1) // block, device=device) + 0.25
            ).to(scale_dtype)
        return tensors

    @staticmethod
    def _session(tensors, block=128, fusion=0, top_k=10, normalize=1, initializers=False, row_tile_size=0):
        dtype = TensorProto.BFLOAT16 if tensors["input"].dtype == torch.bfloat16 else TensorProto.FLOAT16
        model, input_types = create_block_fp8_moe_graph(tensors, top_k, dtype, block, fusion, normalize, initializers)
        options = onnxruntime.SessionOptions()
        options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_DISABLE_ALL
        options.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
        options.add_session_config_entry("ep.cuda.qmoe_row_tile_size", str(row_tile_size))
        return onnxruntime.InferenceSession(
            model, options, providers=[resolve_cuda_plugin_ep("CUDAExecutionProvider")]
        ), input_types

    @classmethod
    def _execute(cls, tensors, block=128, fusion=0, top_k=10, normalize=1, session=None, initializers=False):
        if session is None:
            session = cls._session(tensors, block, fusion, top_k, normalize, initializers)
        session, input_types = session
        output = torch.empty_like(tensors["input"])
        binding = session.io_binding()
        for graph_input in session.get_inputs():
            name = graph_input.name
            tensor = tensors[name]
            binding.bind_input(name, "cuda", 0, input_types[name], tensor.shape, tensor.data_ptr())
        binding.bind_output("output", "cuda", 0, input_types["input"], output.shape, output.data_ptr())
        torch.cuda.synchronize()
        binding.synchronize_inputs()
        session.run_with_iobinding(binding)
        binding.synchronize_outputs()
        return output

    @staticmethod
    def _reference(tensors, block, fusion, top_k=10, normalize=1):
        x = tensors["input"]
        probs = tensors["router_probs"].float().softmax(-1)
        values, experts = probs.topk(top_k, dim=-1)
        if normalize:
            values /= values.sum(-1, keepdim=True)
        output = torch.zeros_like(x, dtype=torch.float32)

        def weight(name, expert):
            raw = tensors[f"{name}_weights"][expert].view(torch.float8_e4m3fn).float()
            scale = tensors[f"{name}_scales"][expert].repeat_interleave(block, 0).repeat_interleave(block, 1)
            return (raw * scale[: raw.shape[0], : raw.shape[1]]).to(x.dtype).float()

        for expert in experts.unique().tolist():
            rows, slots = torch.where(experts == expert)
            projection = (x[rows].float() @ weight("fc1", expert).T).to(x.dtype).float()
            if "fc1_bias" in tensors:
                projection += tensors["fc1_bias"][expert].float()
            if fusion:
                if fusion == 1:
                    gate, up = projection[:, 0::2], projection[:, 1::2]
                else:
                    gate, up = projection.chunk(2, -1)
            else:
                gate = projection
                up = (
                    (x[rows].float() @ weight("fc3", expert).T).to(x.dtype).float() if "fc3_weights" in tensors else 1.0
                )
            activation = (F.silu(gate) * up).to(x.dtype).float()
            projected = (activation @ weight("fc2", expert).T).to(x.dtype).float()
            if "fc2_bias" in tensors:
                projected += tensors["fc2_bias"][expert].float()
            output.index_add_(0, rows, projected * values[rows, slots, None])
        return output.to(x.dtype)

    def _parity(self, block=128, fusion=0, normalize=1, **kwargs):
        tensors = self._inputs(block=block, fusion=fusion, **kwargs)
        actual = self._execute(tensors, block, fusion, normalize=normalize)
        expected = self._reference(tensors, block, fusion, normalize=normalize)
        torch.testing.assert_close(actual.float(), expected.float(), atol=0.003, rtol=0.04)

    def test_block_fp8_bf16_separate_fc3_top10(self):
        self._parity()

    def test_block_fp8_fp16_separate_fc3_top10(self):
        self._parity(dtype=torch.float16)

    def test_block_fp8_scale_types(self):
        for dtype in (torch.float16, torch.bfloat16):
            for scale_dtype in (torch.float32, torch.float16, torch.bfloat16):
                with self.subTest(dtype=dtype, scale_dtype=scale_dtype):
                    self._parity(dtype=dtype, scale_dtype=scale_dtype)

    def test_block_fp8_initializers(self):
        for dtype in (torch.float16, torch.bfloat16):
            for scale_dtype in (torch.float32, torch.float16, torch.bfloat16):
                with self.subTest(dtype=dtype, scale_dtype=scale_dtype):
                    tensors = self._inputs(dtype=dtype, scale_dtype=scale_dtype)
                    actual = self._execute(tensors, initializers=True)
                    expected = self._reference(tensors, 128, 0)
                    torch.testing.assert_close(actual.float(), expected.float(), atol=0.003, rtol=0.04)

    def test_integer_scales_still_require_activation_type(self):
        for dtype in (torch.float16, torch.bfloat16):
            for scale_dtype in (torch.float32, torch.float16, torch.bfloat16):
                if scale_dtype == dtype:
                    continue
                for initializers in (False, True):
                    with self.subTest(dtype=dtype, scale_dtype=scale_dtype, initializers=initializers):
                        tensors = self._inputs(hidden=128, inter=128, dtype=dtype, scale_dtype=scale_dtype)
                        del tensors["fc3_weights"], tensors["fc3_scales"]
                        for fc in ("fc1", "fc2"):
                            tensors[f"{fc}_scales"] = torch.ones(16, 128, 1, device=device, dtype=scale_dtype)
                        onnx_dtype = TensorProto.FLOAT16 if dtype == torch.float16 else TensorProto.BFLOAT16
                        model_bytes, input_types = create_block_fp8_moe_graph(
                            tensors, 10, onnx_dtype, 128, initializers=initializers
                        )
                        model = load_model_from_string(model_bytes)
                        for attr in model.graph.node[0].attribute:
                            if attr.name == "quant_type":
                                attr.s = b"int"
                        for graph_input in model.graph.input:
                            if graph_input.name.endswith("_weights"):
                                graph_input.type.tensor_type.elem_type = TensorProto.UINT8
                        for initializer in model.graph.initializer:
                            if initializer.name.endswith("_weights"):
                                initializer.data_type = TensorProto.UINT8
                        for name in input_types:
                            if name.endswith("_weights"):
                                input_types[name] = TensorProto.UINT8
                        options = onnxruntime.SessionOptions()
                        options.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
                        with self.assertRaisesRegex(Exception, "scales must match the activation type"):
                            session = onnxruntime.InferenceSession(
                                model.SerializeToString(),
                                options,
                                providers=[resolve_cuda_plugin_ep("CUDAExecutionProvider")],
                            )
                            self._execute(tensors, session=(session, input_types))

    def test_block_fp8_swiglu_layouts(self):
        for fusion in (1, 2):
            with self.subTest(fusion=fusion):
                self._parity(fusion=fusion)

    def test_block_fp8_qwen_projection_shapes(self):
        # The gate/up boundary is five 128-row scale blocks, not a power of two.
        for fusion in (0, 2):
            with self.subTest(fusion=fusion):
                self._parity(hidden=2560, inter=640, experts=16, tokens=2, fusion=fusion)

    def test_block_fp8_partial_blocks(self):
        self._parity(block=64, normalize=0)

    def test_block_fp8_without_fc3(self):
        tensors = self._inputs()
        del tensors["fc3_weights"], tensors["fc3_scales"]
        actual = self._execute(tensors)
        expected = self._reference(tensors, 128, 0)
        torch.testing.assert_close(actual.float(), expected.float(), atol=0.003, rtol=0.04)

    def test_block_fp8_dynamic_scales(self):
        tensors = self._inputs()
        session = self._session(tensors)
        first = self._execute(tensors, session=session)
        tensors["fc2_scales"] *= 2
        second = self._execute(tensors, session=session)
        torch.testing.assert_close(second, first * 2)

    def test_block_fp8_dynamic_weights(self):
        tensors = self._inputs()
        session = self._session(tensors)
        self._execute(tensors, session=session)
        tensors["fc3_weights"].zero_()
        actual = self._execute(tensors, session=session)
        torch.testing.assert_close(actual, torch.zeros_like(actual), atol=0, rtol=0)

    def test_block_fp8_sparse_routing_with_bias_and_tiles(self):
        for fusion in (0, 1, 2):
            for row_tile_size in (0, 2):
                with self.subTest(fusion=fusion, row_tile_size=row_tile_size):
                    tensors = self._inputs(experts=32, tokens=5, fusion=fusion)
                    tensors["fc2_bias"] = torch.randn(32, 160, device=device, dtype=torch.bfloat16) * 0.1
                    if fusion == 1:
                        tensors["fc1_bias"] = torch.randn(32, 160, device=device, dtype=torch.bfloat16) * 0.1
                    session = self._session(tensors, fusion=fusion, top_k=2, row_tile_size=row_tile_size)
                    # Repeated experts share slots; subsequent calls and tiles change the active set.
                    for selected in (
                        [[31, 7], [7, 31], [31, 7], [7, 31], [2, 19]],
                        [[0, 25], [25, 0], [25, 0], [0, 25], [30, 1]],
                    ):
                        tensors["router_probs"].fill_(-10)
                        selected_ids = torch.tensor(selected, device=device)
                        tensors["router_probs"].scatter_(
                            1,
                            selected_ids,
                            torch.tensor([2.0, 1.0], device=device, dtype=torch.bfloat16).expand(5, -1),
                        )
                        actual = self._execute(tensors, session=session)
                        expected = self._reference(tensors, 128, fusion, top_k=2)
                        torch.testing.assert_close(actual.float(), expected.float(), atol=0.003, rtol=0.04)

    def test_block_fp8_all_experts_selected(self):
        tensors = self._inputs(experts=10, tokens=1)
        actual = self._execute(tensors)
        expected = self._reference(tensors, 128, 0)
        torch.testing.assert_close(actual.float(), expected.float(), atol=0.003, rtol=0.04)

    def test_block_fp8_rejects_invalid_scales(self):
        for name in ("fc1_scales", "fc2_scales", "fc3_scales"):
            for bad in ("missing", "shape", "dtype"):
                with self.subTest(name=name, bad=bad):
                    tensors = self._inputs()
                    if bad == "missing":
                        del tensors[name]
                    elif bad == "shape":
                        tensors[name] = tensors[name].flatten()
                    else:
                        tensors[name] = tensors[name].int()
                    with self.assertRaisesRegex(Exception, name):
                        self._execute(tensors)

    def test_block_fp8_rejects_zero_points(self):
        for fc in (1, 2, 3):
            with self.subTest(fc=fc):
                tensors = self._inputs()
                tensors[f"fc{fc}_zero_points"] = torch.zeros(1, device=device, dtype=torch.uint8)
                with self.assertRaisesRegex(Exception, r"zero_points|Type parameter \(T1\).*bound to different types"):
                    self._execute(tensors)

    def test_block_fp8_rejects_global_scales(self):
        tensors = self._inputs()
        tensors["fc1_global_scale"] = torch.ones(16, device=device)
        with self.assertRaisesRegex(Exception, "global scales"):
            self._execute(tensors)

    def test_block_fp8_rejects_transposed_weights(self):
        tensors = self._inputs()
        for name in ("fc1_weights", "fc2_weights", "fc3_weights"):
            tensors[name] = tensors[name].transpose(1, 2).contiguous()
        with self.assertRaisesRegex(Exception, "row-major"):
            self._execute(tensors)

    @unittest.skipUnless(os.getenv("ORT_RUN_LARGE_FP8_QMOE_TEST") == "1", "Opt-in official 512-expert memory test")
    def test_block_fp8_qwen38_flash_next_official_shape(self):
        for tokens in (1, 17):
            with self.subTest(tokens=tokens):
                tensors = self._inputs(hidden=2560, inter=640, experts=512, tokens=tokens)
                # Avoid BF16 top-k ties so the reference selects exactly the same experts.
                tensors["router_probs"] = torch.stack(
                    [(torch.randperm(512, device=device).float() / 128 - 2).bfloat16() for _ in range(tokens)]
                )
                actual = self._execute(tensors)
                expected = self._reference(tensors, 128, 0)
                torch.testing.assert_close(actual.float(), expected.float(), atol=0.003, rtol=0.04)


@pytest.mark.skipif(not torch.cuda.is_available() or not has_fp8_qmoe, reason="CUDA FP8 QMoE required")
def test_block_fp8_compact_decode_scratch(capfd, monkeypatch):
    monkeypatch.setenv("ORT_ENABLE_QMOE_KERNEL_DEBUG_INFO", "1")
    monkeypatch.setenv("ORT_ENABLE_FP8_FUSED", "0")
    tensors = TestQMoEBlockFP8._inputs(experts=512, tokens=1)
    tensors["router_probs"] = (torch.randperm(512, device=device).float() / 128 - 2).bfloat16().unsqueeze(0)
    actual = TestQMoEBlockFP8._execute(tensors)
    expected = TestQMoEBlockFP8._reference(tensors, 128, 0)
    torch.testing.assert_close(actual.float(), expected.float(), atol=0.003, rtol=0.04)
    assert "QMoE FP8 ExpertCapacity=10 DequantWeightBytes=768000" in capfd.readouterr().out


@pytest.mark.skipif(not torch.cuda.is_available() or not has_fp8_qmoe, reason="CUDA FP8 QMoE required")
@pytest.mark.parametrize("tokens,path", [(1, "fp8_fused_gemv"), (32, "fp8_fused_gemm")])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("scale_dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_block_fp8_fused_without_weight_scratch(capfd, monkeypatch, tokens, path, dtype, scale_dtype):
    tensors = TestQMoEBlockFP8._inputs(
        experts=512 if tokens == 1 else 16, tokens=tokens, dtype=dtype, scale_dtype=scale_dtype
    )
    monkeypatch.setenv("ORT_ENABLE_QMOE_KERNEL_DEBUG_INFO", "1")
    monkeypatch.setenv("ORT_ENABLE_FP8_FUSED", "0")
    reference = TestQMoEBlockFP8._execute(tensors)
    capfd.readouterr()
    monkeypatch.setenv("ORT_ENABLE_FP8_FUSED", "1")
    actual = TestQMoEBlockFP8._execute(tensors)
    torch.testing.assert_close(actual.float(), reference.float(), atol=0.003, rtol=0.04)
    log = capfd.readouterr().out
    assert path in log
    assert "DequantWeightBytes=0" in log


@pytest.mark.skipif(not torch.cuda.is_available() or not has_fp8_qmoe, reason="CUDA FP8 QMoE required")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("fusion", [0, 1, 2])
@pytest.mark.parametrize("row_tile_size", [0, 17])
def test_block_fp8_fused_gemm_partial_tiles(monkeypatch, dtype, fusion, row_tile_size):
    monkeypatch.setenv("ORT_ENABLE_FP8_FUSED", "1")
    tensors = TestQMoEBlockFP8._inputs(
        hidden=72, inter=40, experts=32, tokens=35, block=32, fusion=fusion, dtype=dtype
    )
    tensors["router_probs"].fill_(-10)
    selected = [31, 7, 2, 19, 25, 0, 17, 11, 9, 23]
    tensors["router_probs"][:, selected] = torch.linspace(2, 1, 10, device=device, dtype=dtype)
    tensors["fc2_bias"] = torch.randn(32, 72, device=device, dtype=dtype) * 0.01
    if fusion == 1:
        tensors["fc1_bias"] = torch.randn(32, 80, device=device, dtype=dtype) * 0.01
    session = TestQMoEBlockFP8._session(tensors, block=32, fusion=fusion, row_tile_size=row_tile_size)
    actual = TestQMoEBlockFP8._execute(tensors, block=32, fusion=fusion, session=session)
    expected = TestQMoEBlockFP8._reference(tensors, 32, fusion)
    torch.testing.assert_close(actual.float(), expected.float(), atol=0.003, rtol=0.04)


if __name__ == "__main__":
    unittest.main()
