# --------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.  See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------
#
# Tests for QMoE NVFP4 quantization on CUDA — W4A16 mode.
#
# NVFP4 format: E2M1 4-bit weights (2 values per byte, same as MXFP4), block
# size 16, per-block scales stored as Float8E4M3FN, per-expert float32 global
# scale (weight_scale_2). Dequant:
#   w = DecodeFp4E2M1(code) * DecodeE4M3(block_scale) * global_scale[expert]
#
# Two decode paths are exercised: the dequant-to-A16 fallback (native block-scaled
# CUTLASS GEMM is Blackwell-only, so this is the general path) and, for small-decode
# SwiGLU shapes, the fused FP4 GEMV kernel (forced on via ORT_ENABLE_FP4_GEMV=1; a
# gemv_mode="0" companion checks the fallback on the same shape). Both paths are
# SM-agnostic; the tests require SM80+ / CUDA / an ENABLE_FP4 + USE_FP4_QMOE build.
# --------------------------------------------------------------------------

import os
import unittest

import numpy
import torch
import torch.nn.functional as F
from cuda_plugin_ep_helper import resolve_cuda_plugin_ep
from onnx import helper
from parameterized import parameterized

import onnxruntime

try:
    from onnx import TensorProto

    has_onnx = True
except ImportError:
    has_onnx = False

onnxruntime.preload_dlls()

build_info = onnxruntime.get_build_info()
has_fp4_qmoe = ", fp4-qmoe=" in build_info

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

torch.manual_seed(42)
numpy.random.seed(42)

# ============================================================================
# NVFP4 (E2M1 + E4M3 block scale) quantization utilities
# ============================================================================

# Positive FP4 e2m1 representable values (codes 0-7). Negative uses codes 8-15.
FP4_POS_VALUES = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])
FP4_MAX = 6.0
E4M3_MAX = 448.0
NVFP4_BLOCK_SIZE = 16


def fp4_e2m1_quantize(values):
    """Quantize float values to nearest FP4 e2m1 value; return (quantized, 4-bit codes)."""
    dev = values.device
    pos_vals = FP4_POS_VALUES.to(device=dev, dtype=torch.float32)

    flat = values.float().reshape(-1)
    sign = flat.sign()
    abs_val = flat.abs().clamp(max=FP4_MAX)

    diffs = (abs_val.unsqueeze(-1) - pos_vals.unsqueeze(0)).abs()
    nearest_idx = diffs.argmin(dim=-1)  # code 0-7

    quantized = sign * pos_vals[nearest_idx]

    codes = nearest_idx.to(torch.uint8)
    codes[sign < 0] += 8
    codes[flat == 0] = 0

    return quantized.reshape(values.shape), codes.reshape(values.shape)


def quantize_weight_to_nvfp4(weight, block_size=NVFP4_BLOCK_SIZE):
    """
    Quantize a per-expert weight matrix [N, K] to NVFP4.

    Two-level scaling: a per-expert float32 global scale (weight_scale_2) plus a
    per-block Float8E4M3FN scale. The kernel reconstructs each element as
        fp4_value * decode_e4m3(block_scale) * global_scale.

    Returns:
        packed_col_major: [K, N//2] uint8 — column-major packed FP4 codes
        block_scale_bytes: [N, K//block_size] uint8 — raw Float8E4M3FN bytes
        global_scale:      float scalar (per expert)
        dequantized:       [N, K] float — reference dequantized weights
    """
    n, k = weight.shape
    assert k % block_size == 0, f"K={k} must be divisible by block_size={block_size}"
    assert n % 2 == 0, f"N={n} must be even for FP4 packing"

    w = weight.float()
    num_blocks = k // block_size
    blocks = w.reshape(n, num_blocks, block_size)

    block_amax = blocks.abs().amax(dim=-1)  # [N, num_blocks]
    global_amax = w.abs().amax().item()

    # Per-expert global scale (weight_scale_2). Chosen so that per-block E4M3
    # scales land within [0, E4M3_MAX].
    global_scale = global_amax / (FP4_MAX * E4M3_MAX)
    if global_scale <= 0:
        global_scale = 1.0

    # Ideal per-block scale before E4M3 quantization.
    block_scale_f = block_amax / FP4_MAX / global_scale  # [N, num_blocks], all <= E4M3_MAX
    block_scale_e4m3 = block_scale_f.to(torch.float8_e4m3fn)
    block_scale_deq = block_scale_e4m3.float()  # DecodeE4M3(byte)

    # Effective per-element scale (matches the kernel): block_scale_deq * global_scale.
    eff_scale = (block_scale_deq * global_scale).unsqueeze(-1)  # [N, num_blocks, 1]
    eff_scale_safe = torch.where(eff_scale > 0, eff_scale, torch.ones_like(eff_scale))

    scaled = blocks / eff_scale_safe
    quantized_vals, fp4_codes = fp4_e2m1_quantize(scaled)
    quantized_vals = quantized_vals.reshape(n, num_blocks, block_size)
    fp4_codes = fp4_codes.reshape(n, num_blocks, block_size)

    dequantized = (quantized_vals * eff_scale).reshape(n, k)

    # Pack codes [N, K] -> transpose -> [K, N] -> pack pairs along N -> [K, N//2].
    codes_nk = fp4_codes.reshape(n, k)
    codes_kn = codes_nk.T.contiguous()  # [K, N]
    low = codes_kn[:, 0::2].to(torch.uint8)  # even N-index -> low nibble
    high = codes_kn[:, 1::2].to(torch.uint8)  # odd N-index  -> high nibble
    packed = (high << 4) | low  # [K, N//2]

    block_scale_bytes = block_scale_e4m3.view(torch.uint8).contiguous()  # [N, K//block_size]

    return packed, block_scale_bytes, float(global_scale), dequantized


# ============================================================================
# SwiGLU activation reference
# ============================================================================


def swiglu_ref(x, alpha=1.702, limit=7.0):
    """SwiGLU activation matching the QMoE kernel implementation."""
    dim = x.shape[-1]
    x = x.view(-1, dim // 2, 2)
    g, l_val = x[..., 0], x[..., 1]
    if limit is not None:
        g = g.clamp(max=limit)
        l_val = l_val.clamp(min=-limit, max=limit)
    return g * torch.sigmoid(alpha * g) * (l_val + 1)


# ============================================================================
# ONNX graph builder for NVFP4 QMoE
# ============================================================================


def create_nvfp4_moe_onnx_graph(
    num_tokens,
    hidden_size,
    inter_size,
    num_experts,
    top_k,
    onnx_dtype,
    fc1_weights,  # [E, K1, N1/2] uint8 packed FP4 (column-major)
    fc2_weights,  # [E, K2, N2/2] uint8 packed FP4 (column-major)
    fc1_block_scales,  # [E, N1, K1//16] uint8 (Float8E4M3FN bytes)
    fc1_global_scale,  # [E] float32
    fc2_block_scales,  # [E, N2, K2//16] uint8 (Float8E4M3FN bytes)
    fc2_global_scale,  # [E] float32
    block_size=NVFP4_BLOCK_SIZE,
    use_swiglu=False,
):
    """Build ONNX model with QMoE operator in NVFP4 mode."""
    inputs = [
        "input",  # 0
        "router_probs",  # 1
        "fc1_weights",  # 2: uint8 packed FP4
        "fc1_scales",  # 3: Float8E4M3FN NVFP4 block scales
        "",  # 4: fc1_bias
        "fc2_weights",  # 5: uint8 packed FP4
        "fc2_scales",  # 6: Float8E4M3FN NVFP4 block scales
        "",  # 7: fc2_bias
        "",  # 8:  fc3_weights
        "",  # 9:  fc3_scales
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
            "QMoE_NVFP4",
            k=top_k,
            normalize_routing_weights=1,
            activation_type=activation,
            expert_weight_bits=4,
            quant_type="nvfp4",
            block_size=block_size,
            swiglu_fusion=1 if use_swiglu else 0,
            swiglu_limit=7.0,
            activation_alpha=1.702,
            activation_beta=1.0,
            domain="com.microsoft",
        ),
    ]

    initializers = []

    # FC1 / FC2 packed weights [E, K, N/2] uint8
    for name, tensor in [("fc1_weights", fc1_weights), ("fc2_weights", fc2_weights)]:
        arr = numpy.ascontiguousarray(tensor.cpu().numpy().astype(numpy.uint8))
        initializers.append(helper.make_tensor(name, TensorProto.UINT8, list(tensor.shape), arr.tobytes(), raw=True))

    # NVFP4 block scales [E, N, K//16] float8e4m3fn (stored as raw bytes)
    for name, tensor in [("fc1_scales", fc1_block_scales), ("fc2_scales", fc2_block_scales)]:
        arr = numpy.ascontiguousarray(tensor.cpu().numpy().astype(numpy.uint8))
        initializers.append(
            helper.make_tensor(name, TensorProto.FLOAT8E4M3FN, list(tensor.shape), arr.tobytes(), raw=True)
        )

    # Per-expert global scales [E] float32 (T4)
    for name, tensor in [("fc1_global_scale", fc1_global_scale), ("fc2_global_scale", fc2_global_scale)]:
        vals = tensor.cpu().float().flatten().tolist()
        initializers.append(helper.make_tensor(name, TensorProto.FLOAT, list(tensor.shape), vals, raw=False))

    graph_inputs = [
        helper.make_tensor_value_info("input", onnx_dtype, [num_tokens, hidden_size]),
        helper.make_tensor_value_info("router_probs", onnx_dtype, [num_tokens, num_experts]),
    ]
    graph_outputs = [
        helper.make_tensor_value_info("output", onnx_dtype, [num_tokens, hidden_size]),
    ]

    graph = helper.make_graph(nodes, "QMoE_NVFP4_Test", graph_inputs, graph_outputs, initializers)
    model = helper.make_model(
        graph,
        opset_imports=[helper.make_opsetid("", 22), helper.make_opsetid("com.microsoft", 1)],
    )
    return model.SerializeToString()


# ============================================================================
# Test class
# ============================================================================


def _cuda_sm():
    """Return SM version (e.g. 90 for Hopper)."""
    if not torch.cuda.is_available():
        return 0
    cc = torch.cuda.get_device_capability()
    return cc[0] * 10 + cc[1]


@unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
@unittest.skipIf(not has_onnx, "ONNX not available")
@unittest.skipIf(not has_fp4_qmoe, "CUDA QMoE FP4 kernels not enabled in this build")
@unittest.skipIf(not hasattr(torch, "float8_e4m3fn"), "PyTorch build does not expose torch.float8_e4m3fn")
class TestQMoENVFP4(unittest.TestCase):
    """Tests for W4A16 NVFP4 MoE quantization (dequant fallback)."""

    def _skip_if_no_fp4(self):
        # NVFP4 always uses the dequant-to-A16 fallback (native block-scaled CUTLASS GEMM is
        # Blackwell-only). That path is SM-agnostic and runs on any CUDA GEMM-capable GPU, so
        # SM80 (A100) is sufficient here even though the production target is H200/SM90.
        sm = _cuda_sm()
        if sm < 80:
            self.skipTest(f"NVFP4 QMoE requires SM80+, got SM{sm}")

    def _run_nvfp4_moe_test(
        self,
        hidden_size,
        inter_size,
        num_experts,
        top_k,
        num_tokens,
        onnx_dtype,
        use_swiglu=False,
        block_size=NVFP4_BLOCK_SIZE,
        gemv_mode=None,
    ):
        self._skip_if_no_fp4()

        torch.manual_seed(42)
        numpy.random.seed(42)

        torch_dtype = torch.float16 if onnx_dtype == TensorProto.FLOAT16 else torch.bfloat16
        onnx_elem = TensorProto.FLOAT16 if torch_dtype == torch.float16 else TensorProto.BFLOAT16

        fc1_n = 2 * inter_size if use_swiglu else inter_size
        fc1_k = hidden_size
        fc2_n = hidden_size
        fc2_k = inter_size

        fc1_packed, fc1_bs, fc1_gs, fc1_deq = [], [], [], []
        fc2_packed, fc2_bs, fc2_gs, fc2_deq = [], [], [], []

        for _ in range(num_experts):
            w1 = torch.randn(fc1_n, fc1_k, device=device) * 0.1
            p1, b1, g1, d1 = quantize_weight_to_nvfp4(w1, block_size)
            fc1_packed.append(p1)
            fc1_bs.append(b1)
            fc1_gs.append(torch.tensor(g1, dtype=torch.float32))
            fc1_deq.append(d1)

            w2 = torch.randn(fc2_n, fc2_k, device=device) * 0.1
            p2, b2, g2, d2 = quantize_weight_to_nvfp4(w2, block_size)
            fc2_packed.append(p2)
            fc2_bs.append(b2)
            fc2_gs.append(torch.tensor(g2, dtype=torch.float32))
            fc2_deq.append(d2)

        fc1_weights = torch.stack(fc1_packed, dim=0)  # [E, K, N/2]
        fc2_weights = torch.stack(fc2_packed, dim=0)  # [E, K, N/2]
        fc1_block_scales = torch.stack(fc1_bs, dim=0)  # [E, N, K//16]
        fc2_block_scales = torch.stack(fc2_bs, dim=0)  # [E, N, K//16]
        fc1_global_scale = torch.stack(fc1_gs)  # [E]
        fc2_global_scale = torch.stack(fc2_gs)  # [E]
        fc1_deq_all = torch.stack(fc1_deq, dim=0)  # [E, N, K]
        fc2_deq_all = torch.stack(fc2_deq, dim=0)  # [E, N, K]

        onnx_model = create_nvfp4_moe_onnx_graph(
            num_tokens=num_tokens,
            hidden_size=hidden_size,
            inter_size=inter_size,
            num_experts=num_experts,
            top_k=top_k,
            onnx_dtype=onnx_elem,
            fc1_weights=fc1_weights,
            fc2_weights=fc2_weights,
            fc1_block_scales=fc1_block_scales,
            fc1_global_scale=fc1_global_scale,
            fc2_block_scales=fc2_block_scales,
            fc2_global_scale=fc2_global_scale,
            block_size=block_size,
            use_swiglu=use_swiglu,
        )

        opts = onnxruntime.SessionOptions()
        opts.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_DISABLE_ALL
        # gemv_mode toggles the fused FP4 GEMV decode path (read once in the QMoE op ctor during
        # session creation): "1" forces it on, "0" forces the dequant fallback, None leaves the
        # default. Restore the previous value right after the session is built.
        prev_gemv_env = os.environ.get("ORT_ENABLE_FP4_GEMV")
        if gemv_mode is not None:
            os.environ["ORT_ENABLE_FP4_GEMV"] = gemv_mode
        try:
            session = onnxruntime.InferenceSession(
                onnx_model, opts, providers=[resolve_cuda_plugin_ep("CUDAExecutionProvider")]
            )
        except Exception as e:
            if "ENABLE_FP4" in str(e) or "requires USE_FP4_QMOE" in str(e):
                self.skipTest(f"NVFP4 not supported in this build: {e}")
            raise
        finally:
            if gemv_mode is not None:
                if prev_gemv_env is None:
                    os.environ.pop("ORT_ENABLE_FP4_GEMV", None)
                else:
                    os.environ["ORT_ENABLE_FP4_GEMV"] = prev_gemv_env

        input_tensor = torch.randn(num_tokens, hidden_size, device=device, dtype=torch_dtype)
        router_logits = torch.randn(num_tokens, num_experts, device=device, dtype=torch_dtype)
        output_tensor = torch.zeros(num_tokens, hidden_size, device=device, dtype=torch_dtype)

        iobinding = session.io_binding()
        iobinding.bind_input("input", "cuda", 0, onnx_elem, input_tensor.shape, input_tensor.data_ptr())
        iobinding.bind_input("router_probs", "cuda", 0, onnx_elem, router_logits.shape, router_logits.data_ptr())
        iobinding.bind_output("output", "cuda", 0, onnx_elem, output_tensor.shape, output_tensor.data_ptr())

        iobinding.synchronize_inputs()
        try:
            session.run_with_iobinding(iobinding)
        except Exception as e:
            msg = str(e)
            if "ENABLE_FP4" in msg or "requires USE_FP4_QMOE" in msg or "stubbed out" in msg:
                self.skipTest(f"NVFP4 kernel not available in this build: {e}")
            raise
        iobinding.synchronize_outputs()

        ort_output = output_tensor.clone()

        ref_output = self._compute_reference(
            input_tensor,
            router_logits,
            fc1_deq_all,
            fc2_deq_all,
            num_experts,
            top_k,
            use_swiglu,
            torch_dtype,
        )

        max_diff = (ort_output.float() - ref_output.float()).abs().max().item()
        dtype_tag = "FP16" if torch_dtype == torch.float16 else "BF16"
        act_tag = "SwiGLU" if use_swiglu else "SiLU"
        print(
            f"NVFP4 MoE test: {dtype_tag} {act_tag} "
            f"tokens={num_tokens} experts={num_experts} "
            f"hidden={hidden_size} inter={inter_size} "
            f"max_diff={max_diff:.6f}"
        )

        atol = 0.15 if torch_dtype == torch.bfloat16 else 0.12
        self.assertLess(
            max_diff,
            atol,
            f"NVFP4 MoE parity check failed: max_diff={max_diff:.6f} > atol={atol}",
        )

    def _assert_invalid_nvfp4_model(
        self, block_size=NVFP4_BLOCK_SIZE, truncate_fc1_scales=False, truncate_fc1_global_scale=False
    ):
        self._skip_if_no_fp4()
        num_experts = 2
        hidden_size = 64
        inter_size = 64
        fc1_weights = torch.zeros(num_experts, hidden_size, inter_size // 2, dtype=torch.uint8)
        fc2_weights = torch.zeros(num_experts, inter_size, hidden_size // 2, dtype=torch.uint8)
        fc1_scales = torch.zeros(num_experts, inter_size, hidden_size // NVFP4_BLOCK_SIZE, dtype=torch.uint8)
        fc2_scales = torch.zeros(num_experts, hidden_size, inter_size // NVFP4_BLOCK_SIZE, dtype=torch.uint8)
        if truncate_fc1_scales:
            fc1_scales = fc1_scales[:, :, :-1]
        model = create_nvfp4_moe_onnx_graph(
            num_tokens=1,
            hidden_size=hidden_size,
            inter_size=inter_size,
            num_experts=num_experts,
            top_k=1,
            onnx_dtype=TensorProto.FLOAT16,
            fc1_weights=fc1_weights,
            fc2_weights=fc2_weights,
            fc1_block_scales=fc1_scales,
            fc1_global_scale=torch.ones(num_experts - int(truncate_fc1_global_scale), dtype=torch.float32),
            fc2_block_scales=fc2_scales,
            fc2_global_scale=torch.ones(num_experts, dtype=torch.float32),
            block_size=block_size,
        )
        opts = onnxruntime.SessionOptions()
        opts.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_DISABLE_ALL
        with self.assertRaisesRegex(Exception, "block_size|fc1_scales|fc1_global_scale"):
            session = onnxruntime.InferenceSession(
                model, opts, providers=[resolve_cuda_plugin_ep("CUDAExecutionProvider")]
            )
            session.run(
                None,
                {
                    "input": numpy.zeros((1, hidden_size), dtype=numpy.float16),
                    "router_probs": numpy.zeros((1, num_experts), dtype=numpy.float16),
                },
            )

    def test_nvfp4_rejects_wrong_block_size(self):
        self._assert_invalid_nvfp4_model(block_size=32)

    def test_nvfp4_rejects_malformed_prepacked_block_scales(self):
        self._assert_invalid_nvfp4_model(truncate_fc1_scales=True)

    def test_nvfp4_rejects_malformed_prepacked_global_scale(self):
        self._assert_invalid_nvfp4_model(truncate_fc1_global_scale=True)

    @staticmethod
    def _compute_reference(input_tensor, router_logits, fc1_deq, fc2_deq, num_experts, top_k, use_swiglu, torch_dtype):
        """Reference MoE forward pass using dequantized weights."""
        num_tokens = input_tensor.shape[0]
        hidden_size = input_tensor.shape[1]

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

    # ================================================================
    # Test cases
    # ================================================================

    def test_nvfp4_fp16_silu_basic(self):
        self._run_nvfp4_moe_test(
            hidden_size=64,
            inter_size=64,
            num_experts=4,
            top_k=2,
            num_tokens=32,
            onnx_dtype=TensorProto.FLOAT16,
        )

    def test_nvfp4_bf16_silu_basic(self):
        self._run_nvfp4_moe_test(
            hidden_size=64,
            inter_size=64,
            num_experts=4,
            top_k=2,
            num_tokens=32,
            onnx_dtype=TensorProto.BFLOAT16,
        )

    def test_nvfp4_fp16_swiglu(self):
        self._run_nvfp4_moe_test(
            hidden_size=64,
            inter_size=64,
            num_experts=4,
            top_k=2,
            num_tokens=32,
            onnx_dtype=TensorProto.FLOAT16,
            use_swiglu=True,
        )

    def test_nvfp4_bf16_swiglu(self):
        self._run_nvfp4_moe_test(
            hidden_size=64,
            inter_size=64,
            num_experts=4,
            top_k=2,
            num_tokens=32,
            onnx_dtype=TensorProto.BFLOAT16,
            use_swiglu=True,
        )

    @parameterized.expand(
        [
            (8,),
            (64,),
            (128,),
        ]
    )
    def test_nvfp4_fp16_token_counts(self, num_tokens):
        self._run_nvfp4_moe_test(
            hidden_size=64,
            inter_size=64,
            num_experts=4,
            top_k=2,
            num_tokens=num_tokens,
            onnx_dtype=TensorProto.FLOAT16,
        )

    def test_nvfp4_fp16_more_experts(self):
        self._run_nvfp4_moe_test(
            hidden_size=64,
            inter_size=64,
            num_experts=8,
            top_k=2,
            num_tokens=32,
            onnx_dtype=TensorProto.FLOAT16,
        )

    def test_nvfp4_fp16_larger_dims(self):
        self._run_nvfp4_moe_test(
            hidden_size=128,
            inter_size=256,
            num_experts=4,
            top_k=2,
            num_tokens=32,
            onnx_dtype=TensorProto.FLOAT16,
        )

    # ================================================================
    # Fused FP4 GEMV decode fast path (block size 16). The GEMV support window requires
    # n, k >= 512 and expanded rows (num_tokens * top_k) <= 8, plus SwiGLU fusion, so these
    # decode-shaped SwiGLU cases route through the NVFP4 GEMV kernel (gemv_mode="1"). The
    # gemv_mode="0" companion forces the dequant fallback on the identical shape; both must
    # match the exact dequantized reference.
    # ================================================================

    def test_nvfp4_fp16_gemv_decode_swiglu(self):
        self._run_nvfp4_moe_test(
            hidden_size=512,
            inter_size=512,
            num_experts=4,
            top_k=2,
            num_tokens=2,
            onnx_dtype=TensorProto.FLOAT16,
            use_swiglu=True,
            gemv_mode="1",
        )

    def test_nvfp4_bf16_gemv_decode_swiglu(self):
        self._run_nvfp4_moe_test(
            hidden_size=512,
            inter_size=512,
            num_experts=4,
            top_k=2,
            num_tokens=2,
            onnx_dtype=TensorProto.BFLOAT16,
            use_swiglu=True,
            gemv_mode="1",
        )

    def test_nvfp4_fp16_gemv_disabled_swiglu(self):
        self._run_nvfp4_moe_test(
            hidden_size=512,
            inter_size=512,
            num_experts=4,
            top_k=2,
            num_tokens=2,
            onnx_dtype=TensorProto.FLOAT16,
            use_swiglu=True,
            gemv_mode="0",
        )


if __name__ == "__main__":
    unittest.main()
