# --------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.  See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------
#
# Tests for QMoE WFP4AFP8 (W4A8) quantization on CUDA.
#
# WFP4AFP8 mode pairs MXFP4 weights with FP8 e4m3 activations. The QMoE
# operator selects the path based on SM:
#
#   - SM100+ (Blackwell): native CUTLASS block-scaled tensor op path. The
#     runner accepts BF16/FP16 input and quantizes it to MXFP8 (FP8 + per-block
#     ue8m0 scales) inside expandInputRowsKernel before the FP8 x MXFP4 GEMM.
#   - SM<100: dequantize-then-A16 fallback. MXFP4 weights are decoded to
#     BF16/FP16 and fed into the dense A16 MoE runner.
#
# These tests are skipped when the GPU does not support FP4 (SM<90 or
# `ENABLE_FP4` not defined in the build). The TestQMoEWFP4AFP8Native class
# additionally requires SM100+ at runtime.
#
# Per-expert FP8 activation global scales (inputs 18/19) are accepted by the
# schema and validated by the operator. They are reserved for the future
# Variant A (global-scaled FP8) native path; the current native path uses the
# Variant B (MXFP8 block-scaled) plumbing where activation block scales are
# computed by the runner at runtime.
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

# Reuse the MXFP4 quantization utilities from the FP4 test module.
from test_qmoe_fp4_cuda import quantize_weight_to_mxfp4, swiglu_ref

onnxruntime.preload_dlls()

build_info = onnxruntime.get_build_info()
has_fp4_qmoe = ", fp4-qmoe=" in build_info
has_fp8_qmoe = ", fp8-qmoe=" in build_info

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

torch.manual_seed(42)
numpy.random.seed(42)


def _cuda_sm():
    if not torch.cuda.is_available():
        return 0
    cc = torch.cuda.get_device_capability()
    return cc[0] * 10 + cc[1]


def create_wfp4afp8_moe_onnx_graph(
    num_tokens,
    hidden_size,
    inter_size,
    num_experts,
    top_k,
    onnx_dtype,
    fc1_weights,
    fc2_weights,
    fc1_block_scales,
    fc1_global_scale,
    fc2_block_scales,
    fc2_global_scale,
    use_swiglu=False,
    fc1_act_scale=None,
    fc2_act_scale=None,
):
    """Build an ONNX model exercising QMoE with quant_type='wfp4afp8' (W4A8)."""
    inputs = [
        "input",  # 0
        "router_probs",  # 1
        "fc1_weights",  # 2
        "fc1_scales",  # 3 (float8e8m0 MXFP4 block scales)
        "",  # 4 fc1_bias
        "fc2_weights",  # 5
        "fc2_scales",  # 6 (float8e8m0 MXFP4 block scales)
        "",  # 7 fc2_bias
        "",  # 8 fc3_weights
        "",  # 9 fc3_scales
        "",  # 10 fc3_bias
        "",  # 11 fc1_zero_points
        "",  # 12 fc2_zero_points
        "",  # 13 fc3_zero_points
        "",  # 14 router_weights
        "fc1_global_scale",  # 15
        "fc2_global_scale",  # 16
        "fc1_act_scale" if fc1_act_scale is not None else "",  # 17
        "fc2_act_scale" if fc2_act_scale is not None else "",  # 18
    ]

    activation = "swiglu" if use_swiglu else "silu"

    nodes = [
        helper.make_node(
            "QMoE",
            inputs,
            ["output"],
            "QMoE_WFP4AFP8",
            k=top_k,
            normalize_routing_weights=1,
            activation_type=activation,
            expert_weight_bits=4,
            quant_type="wfp4afp8",
            swiglu_fusion=1 if use_swiglu else 0,
            swiglu_limit=7.0,
            activation_alpha=1.702,
            activation_beta=1.0,
            domain="com.microsoft",
        ),
    ]

    initializers = []

    for name, tensor in [("fc1_weights", fc1_weights), ("fc2_weights", fc2_weights)]:
        arr = numpy.ascontiguousarray(tensor.cpu().numpy().astype(numpy.uint8))
        initializers.append(helper.make_tensor(name, TensorProto.UINT8, list(tensor.shape), arr.tobytes(), raw=True))

    for name, tensor in [
        ("fc1_scales", fc1_block_scales),
        ("fc2_scales", fc2_block_scales),
    ]:
        arr = numpy.ascontiguousarray(tensor.cpu().numpy().astype(numpy.uint8))
        initializers.append(
            helper.make_tensor(name, TensorProto.FLOAT8E8M0, list(tensor.shape), arr.tobytes(), raw=True)
        )

    for name, tensor in [
        ("fc1_global_scale", fc1_global_scale),
        ("fc2_global_scale", fc2_global_scale),
    ]:
        vals = tensor.cpu().float().flatten().tolist()
        initializers.append(helper.make_tensor(name, TensorProto.FLOAT, [num_experts], vals, raw=False))

    if fc1_act_scale is not None:
        vals = fc1_act_scale.cpu().float().flatten().tolist()
        initializers.append(
            helper.make_tensor("fc1_act_scale", TensorProto.FLOAT, list(fc1_act_scale.shape), vals, raw=False)
        )
    if fc2_act_scale is not None:
        vals = fc2_act_scale.cpu().float().flatten().tolist()
        initializers.append(
            helper.make_tensor("fc2_act_scale", TensorProto.FLOAT, list(fc2_act_scale.shape), vals, raw=False)
        )

    graph_inputs = [
        helper.make_tensor_value_info("input", onnx_dtype, [num_tokens, hidden_size]),
        helper.make_tensor_value_info("router_probs", onnx_dtype, [num_tokens, num_experts]),
    ]
    graph_outputs = [
        helper.make_tensor_value_info("output", onnx_dtype, [num_tokens, hidden_size]),
    ]
    graph = helper.make_graph(nodes, "QMoE_WFP4AFP8_Test", graph_inputs, graph_outputs, initializers)
    model = helper.make_model(graph)
    return model.SerializeToString()


@unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
@unittest.skipIf(not has_onnx, "ONNX not available")
@unittest.skipIf(not (has_fp4_qmoe and has_fp8_qmoe), "CUDA QMoE WFP4AFP8 kernels not enabled in this build")
class TestQMoEWFP4AFP8(unittest.TestCase):
    """Tests for W4A8 (MXFP4 weight + FP8 activation) MoE quantization.

    Exercises whichever path the operator selects on the current SM. On SM<100 the
    operator uses the dequantize-then-A16 fallback, which matches the dequant
    reference exactly; on SM100+ it uses the native FP8 x MXFP4 path, which adds
    FP8 activation quantization noise. The tolerance is therefore widened on
    SM100+.
    """

    # Looser tolerance on SM100+ to account for FP8 activation quantization noise
    # introduced by the native path. The dequant-fallback path matches the
    # reference within ordinary FP16/BF16 noise.
    NATIVE_PATH_SM = 100

    def _skip_if_no_fp4(self):
        sm = _cuda_sm()
        if sm < 90:
            self.skipTest(f"WFP4AFP8 requires SM90+ for the fallback path, got SM{sm}")

    def _atol(self, torch_dtype):
        sm = _cuda_sm()
        if sm >= self.NATIVE_PATH_SM:
            # FP8 activation quantization adds error proportional to the per-block
            # max abs activation. We pick a generous tolerance that still catches
            # systematic dispatch / scale-handling regressions.
            return 0.50 if torch_dtype == torch.bfloat16 else 0.45
        return 0.15 if torch_dtype == torch.bfloat16 else 0.12

    def _run(
        self,
        hidden_size,
        inter_size,
        num_experts,
        top_k,
        num_tokens,
        onnx_dtype,
        use_swiglu=False,
        with_act_scale=False,
        per_expert_act_scale=False,
    ):
        self._skip_if_no_fp4()

        torch.manual_seed(42)
        numpy.random.seed(42)

        torch_dtype = torch.float16 if onnx_dtype == TensorProto.FLOAT16 else torch.bfloat16

        fc1_n = 2 * inter_size if use_swiglu else inter_size
        fc1_k = hidden_size
        fc2_n = hidden_size
        fc2_k = inter_size

        fc1_packed, fc1_bs, fc1_gs, fc1_deq = [], [], [], []
        fc2_packed, fc2_bs, fc2_gs, fc2_deq = [], [], [], []
        for _ in range(num_experts):
            w1 = torch.randn(fc1_n, fc1_k, device=device) * 0.1
            p1, b1, g1, d1 = quantize_weight_to_mxfp4(w1, 32)
            fc1_packed.append(p1)
            fc1_bs.append(b1)
            fc1_gs.append(torch.tensor(g1, dtype=torch.float32))
            fc1_deq.append(d1)
            w2 = torch.randn(fc2_n, fc2_k, device=device) * 0.1
            p2, b2, g2, d2 = quantize_weight_to_mxfp4(w2, 32)
            fc2_packed.append(p2)
            fc2_bs.append(b2)
            fc2_gs.append(torch.tensor(g2, dtype=torch.float32))
            fc2_deq.append(d2)

        fc1_weights = torch.stack(fc1_packed, dim=0)
        fc2_weights = torch.stack(fc2_packed, dim=0)
        fc1_block_scales = torch.stack(fc1_bs, dim=0)
        fc2_block_scales = torch.stack(fc2_bs, dim=0)
        fc1_global_scale = torch.stack(fc1_gs)
        fc2_global_scale = torch.stack(fc2_gs)
        fc1_deq_all = torch.stack(fc1_deq, dim=0)
        fc2_deq_all = torch.stack(fc2_deq, dim=0)

        fc1_act_scale = None
        fc2_act_scale = None
        if with_act_scale:
            shape = [num_experts] if per_expert_act_scale else [1]
            fc1_act_scale = torch.full(shape, 1.0, dtype=torch.float32)
            fc2_act_scale = torch.full(shape, 1.0, dtype=torch.float32)

        onnx_model = create_wfp4afp8_moe_onnx_graph(
            num_tokens=num_tokens,
            hidden_size=hidden_size,
            inter_size=inter_size,
            num_experts=num_experts,
            top_k=top_k,
            onnx_dtype=onnx_dtype,
            fc1_weights=fc1_weights,
            fc2_weights=fc2_weights,
            fc1_block_scales=fc1_block_scales,
            fc1_global_scale=fc1_global_scale,
            fc2_block_scales=fc2_block_scales,
            fc2_global_scale=fc2_global_scale,
            use_swiglu=use_swiglu,
            fc1_act_scale=fc1_act_scale,
            fc2_act_scale=fc2_act_scale,
        )

        opts = onnxruntime.SessionOptions()
        opts.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_DISABLE_ALL
        try:
            session = onnxruntime.InferenceSession(
                onnx_model, opts, providers=[resolve_cuda_plugin_ep("CUDAExecutionProvider")]
            )
        except Exception as e:
            msg = str(e)
            if "FP4" in msg or "ENABLE_FP4" in msg or "wfp4afp8" in msg or "SM" in msg:
                self.skipTest(f"WFP4AFP8 not supported in this build: {e}")
            raise

        input_tensor = torch.randn(num_tokens, hidden_size, device=device, dtype=torch_dtype)
        router_logits = torch.randn(num_tokens, num_experts, device=device, dtype=torch_dtype)
        output_tensor = torch.zeros(num_tokens, hidden_size, device=device, dtype=torch_dtype)

        iobinding = session.io_binding()
        iobinding.bind_input("input", "cuda", 0, onnx_dtype, input_tensor.shape, input_tensor.data_ptr())
        iobinding.bind_input("router_probs", "cuda", 0, onnx_dtype, router_logits.shape, router_logits.data_ptr())
        iobinding.bind_output("output", "cuda", 0, onnx_dtype, output_tensor.shape, output_tensor.data_ptr())
        iobinding.synchronize_inputs()
        try:
            session.run_with_iobinding(iobinding)
        except Exception as e:
            msg = str(e)
            if (
                "FP4" in msg
                or "MXFP4" in msg
                or "ENABLE_FP4" in msg
                or "wfp4afp8" in msg
                or "stubbed out" in msg
                or "not supported in this build" in msg
            ):
                self.skipTest(f"WFP4AFP8 kernel not available in this build: {e}")
            raise
        iobinding.synchronize_outputs()

        ort_output = output_tensor.clone()

        # Reference: dequantize MXFP4 and run BF16/FP16 MoE — matches the operator's
        # current dequant fallback path exactly.
        ref_output = self._reference(
            input_tensor, router_logits, fc1_deq_all, fc2_deq_all, num_experts, top_k, use_swiglu, torch_dtype
        )

        max_diff = (ort_output.float() - ref_output.float()).abs().max().item()
        atol = self._atol(torch_dtype)
        self.assertLess(max_diff, atol, f"WFP4AFP8 parity check failed: max_diff={max_diff}")

    @staticmethod
    def _reference(input_tensor, router_logits, fc1_deq, fc2_deq, num_experts, top_k, use_swiglu, torch_dtype):
        num_tokens, hidden_size = input_tensor.shape
        x = input_tensor.float()
        logits = router_logits.float()

        topk_vals, topk_idx = torch.topk(logits, top_k, dim=-1)
        routing_weights = F.softmax(topk_vals, dim=1)

        output = torch.zeros(num_tokens, hidden_size, device=x.device, dtype=torch.float32)
        expert_mask = F.one_hot(topk_idx, num_classes=num_experts).permute(2, 1, 0)

        for e in range(num_experts):
            idx, top_x = torch.where(expert_mask[e])
            if top_x.shape[0] == 0:
                continue
            tokens = x[top_x]
            w1 = fc1_deq[e].float()
            w2 = fc2_deq[e].float()
            h = tokens @ w1.T
            h = swiglu_ref(h) if use_swiglu else F.silu(h)
            h = h @ w2.T
            h = h * routing_weights[top_x, idx, None]
            output.index_add_(0, top_x, h)

        return output.to(torch_dtype)

    def test_wfp4afp8_fp16_silu_basic(self):
        self._run(
            hidden_size=256,
            inter_size=256,
            num_experts=4,
            top_k=2,
            num_tokens=32,
            onnx_dtype=TensorProto.FLOAT16,
        )

    def test_wfp4afp8_bf16_silu_basic(self):
        self._run(
            hidden_size=256,
            inter_size=256,
            num_experts=4,
            top_k=2,
            num_tokens=32,
            onnx_dtype=TensorProto.BFLOAT16,
        )

    def test_wfp4afp8_fp16_swiglu(self):
        self._run(
            hidden_size=256,
            inter_size=256,
            num_experts=4,
            top_k=2,
            num_tokens=32,
            onnx_dtype=TensorProto.FLOAT16,
            use_swiglu=True,
        )

    def test_wfp4afp8_fp16_with_per_tensor_act_scale(self):
        """Variant A activation scale provided as (1,)."""
        self._run(
            hidden_size=256,
            inter_size=256,
            num_experts=4,
            top_k=2,
            num_tokens=32,
            onnx_dtype=TensorProto.FLOAT16,
            with_act_scale=True,
            per_expert_act_scale=False,
        )

    def test_wfp4afp8_fp16_with_per_expert_act_scale(self):
        """Variant A activation scale provided as (num_experts,)."""
        self._run(
            hidden_size=256,
            inter_size=256,
            num_experts=4,
            top_k=2,
            num_tokens=32,
            onnx_dtype=TensorProto.FLOAT16,
            with_act_scale=True,
            per_expert_act_scale=True,
        )


@unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
@unittest.skipIf(not has_onnx, "ONNX not available")
@unittest.skipIf(not (has_fp4_qmoe and has_fp8_qmoe), "CUDA QMoE WFP4AFP8 kernels not enabled in this build")
@unittest.skipIf(_cuda_sm() < 100, f"Native WFP4AFP8 requires SM100+, got SM{_cuda_sm()}")
class TestQMoEWFP4AFP8Native(TestQMoEWFP4AFP8):
    """Tests that explicitly exercise the native FP8 x MXFP4 block-scaled path.

    These tests are skipped on SM<100 where the operator falls back to the
    dequant-then-A16 path. They reuse the parity-check infrastructure from
    TestQMoEWFP4AFP8 with native-path-appropriate tolerances and a couple of
    additional larger / token-count / SwiGLU configurations to cover tile
    selection on Blackwell.
    """

    def _skip_if_no_fp4(self):
        # Class-level skip already guards SM<100; nothing else to check here.
        return

    def test_wfp4afp8_native_fp16_silu_basic(self):
        self._run(
            hidden_size=256,
            inter_size=256,
            num_experts=4,
            top_k=2,
            num_tokens=32,
            onnx_dtype=TensorProto.FLOAT16,
        )

    def test_wfp4afp8_native_bf16_silu_basic(self):
        self._run(
            hidden_size=256,
            inter_size=256,
            num_experts=4,
            top_k=2,
            num_tokens=32,
            onnx_dtype=TensorProto.BFLOAT16,
        )

    def test_wfp4afp8_native_fp16_swiglu(self):
        self._run(
            hidden_size=256,
            inter_size=256,
            num_experts=4,
            top_k=2,
            num_tokens=32,
            onnx_dtype=TensorProto.FLOAT16,
            use_swiglu=True,
        )

    def test_wfp4afp8_native_bf16_swiglu(self):
        self._run(
            hidden_size=256,
            inter_size=256,
            num_experts=4,
            top_k=2,
            num_tokens=32,
            onnx_dtype=TensorProto.BFLOAT16,
            use_swiglu=True,
        )

    def test_wfp4afp8_native_fp16_more_tokens(self):
        """Larger token count to exercise grouped-GEMM tile selection."""
        self._run(
            hidden_size=256,
            inter_size=256,
            num_experts=4,
            top_k=2,
            num_tokens=128,
            onnx_dtype=TensorProto.FLOAT16,
        )

    def test_wfp4afp8_native_fp16_more_experts(self):
        """Top-4 over 8 experts."""
        self._run(
            hidden_size=256,
            inter_size=256,
            num_experts=8,
            top_k=4,
            num_tokens=32,
            onnx_dtype=TensorProto.FLOAT16,
        )

    def test_wfp4afp8_native_fp16_larger_dims(self):
        """Hidden/inter sizes large enough to cross the MinKDimAlignmentMXFPX threshold."""
        self._run(
            hidden_size=512,
            inter_size=512,
            num_experts=4,
            top_k=2,
            num_tokens=32,
            onnx_dtype=TensorProto.FLOAT16,
        )


if __name__ == "__main__":
    unittest.main()
