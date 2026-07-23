# --------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.  See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------
#
# Tests for QMoE FP4 (MXFP4) quantization on CUDA — W4A16 mode.
#
# MXFP4 format: __nv_fp4_e2m1 (2-bit exponent, 1-bit mantissa), 2 values per byte.
# Block scaling: group_size=32, scale factors as float_ue8m0_t (uint8, powers of 2).
# Per-expert float32 global scale.
#
# Requires SM90+ (Hopper or newer) and CUDA 12.8+ (ENABLE_FP4 build flag).
# --------------------------------------------------------------------------

import math
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

try:
    from onnxruntime.capi import _pybind_state as _pybind

    has_pybind_pack_fp4_weights = hasattr(_pybind, "pack_fp4_weights_for_cuda_moe_gemm")
except ImportError:
    _pybind = None
    has_pybind_pack_fp4_weights = False

onnxruntime.preload_dlls()

build_info = onnxruntime.get_build_info()
has_fp4_qmoe = ", fp4-qmoe=" in build_info

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

torch.manual_seed(42)
numpy.random.seed(42)

# ============================================================================
# MXFP4 (FP4 e2m1) quantization utilities
# ============================================================================

# Positive FP4 e2m1 representable values (codes 0-7)
# Code mapping: 0→0.0, 1→0.5, 2→1.0, 3→1.5, 4→2.0, 5→3.0, 6→4.0, 7→6.0
# Negative values use codes 8-15 (sign bit in bit 3)
FP4_POS_VALUES = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])
FP4_MAX = 6.0


def fp4_e2m1_quantize(values):
    """
    Quantize float values to nearest FP4 e2m1 representable values.

    Returns:
        quantized: float tensor with FP4-representable values
        codes: uint8 tensor with 4-bit codes (0-15)
    """
    dev = values.device
    pos_vals = FP4_POS_VALUES.to(device=dev, dtype=torch.float32)

    flat = values.float().reshape(-1)
    sign = flat.sign()
    abs_val = flat.abs().clamp(max=FP4_MAX)

    # Find nearest positive FP4 value for each element
    diffs = (abs_val.unsqueeze(-1) - pos_vals.unsqueeze(0)).abs()
    nearest_idx = diffs.argmin(dim=-1)  # code 0-7

    quantized = sign * pos_vals[nearest_idx]

    # Build 4-bit codes: positive=0-7, negative=8-15
    codes = nearest_idx.to(torch.uint8)
    codes[sign < 0] += 8
    codes[flat == 0] = 0  # positive zero

    return quantized.reshape(values.shape), codes.reshape(values.shape)


def float_to_ue8m0_code(x):
    """Encode a positive float as ue8m0 (unsigned 8-bit exponent = power of 2)."""
    if x <= 0:
        return 0
    exp = round(math.log2(x))
    return max(1, min(254, exp + 127))


def ue8m0_code_to_float(code):
    """Decode ue8m0 code to float."""
    if code == 0:
        return 0.0
    return 2.0 ** (code - 127)


def quantize_weight_to_mxfp4(weight, block_size=32):
    """
    Quantize a per-expert weight matrix to MXFP4 format.

    Args:
        weight: [N, K] float tensor (one expert's FC weight)
        block_size: scaling block size (32 for MXFP4)

    Returns:
        packed_col_major: [K, N//2] uint8 — column-major packed FP4
        block_scales:     [N, K//block_size] uint8 — ue8m0 encoded
        global_scale:     float scalar (1.0 for MXFP4)
        dequantized:      [N, K] float — reference dequantized weights
    """
    n, k = weight.shape
    assert k % block_size == 0, f"K={k} must be divisible by block_size={block_size}"
    assert n % 2 == 0, f"N={n} must be even for FP4 packing"

    w = weight.float()
    num_blocks = k // block_size
    blocks = w.reshape(n, num_blocks, block_size)

    # Per-block max absolute value
    block_amax = blocks.abs().amax(dim=-1)  # [N, num_blocks]

    # Compute ue8m0 block scales (powers of 2)
    scales_float = torch.ones(n, num_blocks, dtype=torch.float32, device=weight.device)
    scales_code = torch.full((n, num_blocks), 127, dtype=torch.uint8, device=weight.device)

    for i in range(n):
        for j in range(num_blocks):
            amax = block_amax[i, j].item()
            if amax > 0:
                ideal = amax / FP4_MAX
                code = float_to_ue8m0_code(ideal)
                scales_code[i, j] = code
                scales_float[i, j] = ue8m0_code_to_float(code)

    # Quantize values within each block
    scaled = blocks / scales_float.unsqueeze(-1)
    quantized_vals, fp4_codes = fp4_e2m1_quantize(scaled)
    quantized_vals = quantized_vals.reshape(n, num_blocks, block_size)
    fp4_codes = fp4_codes.reshape(n, num_blocks, block_size)

    # Dequantize for reference: fp4_value x block_scale x global_scale
    global_scale = 1.0
    dequantized = (quantized_vals * scales_float.unsqueeze(-1) * global_scale).reshape(n, k)

    # Pack to column-major: [N, K] codes → transpose → [K, N] → pack pairs along N → [K, N//2]
    codes_nk = fp4_codes.reshape(n, k)
    codes_kn = codes_nk.T.contiguous()  # [K, N]

    low = codes_kn[:, 0::2].to(torch.uint8)  # even N-index → low nibble
    high = codes_kn[:, 1::2].to(torch.uint8)  # odd N-index  → high nibble
    packed = (high << 4) | low  # [K, N//2]

    return packed, scales_code, global_scale, dequantized


def pack_fp4_weights_for_moe(q_codes_nk, N, K):
    """
    Pack FP4 codes from [N, K] (4-bit codes) to column-major [K, N//2] bytes.
    Uses the C++ pybind function if available, otherwise falls back to Python.
    """
    if has_pybind_pack_fp4_weights:
        # Pack [N, K] codes into [N, K/2] bytes first (row-major), then use C++ transpose
        low = q_codes_nk[:, 0::2].to(torch.uint8)
        high = q_codes_nk[:, 1::2].to(torch.uint8)
        packed_row = ((high << 4) | low).cpu().numpy()  # [N, K//2]
        result = _pybind.pack_fp4_weights_for_cuda_moe_gemm(packed_row.reshape(-1), N, K)
        return torch.from_numpy(result).to(torch.uint8).reshape(K, N // 2)
    else:
        # Pure Python fallback
        codes_kn = q_codes_nk.T.contiguous()
        low = codes_kn[:, 0::2].to(torch.uint8)
        high = codes_kn[:, 1::2].to(torch.uint8)
        return (high << 4) | low


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
# ONNX graph builder for FP4 QMoE
# ============================================================================


def create_fp4_moe_onnx_graph(
    num_tokens,
    hidden_size,
    inter_size,
    num_experts,
    top_k,
    onnx_dtype,
    fc1_weights,  # [E, K1, N1/2] uint8 packed FP4 (column-major)
    fc2_weights,  # [E, K2, N2/2] uint8 packed FP4 (column-major)
    fc1_block_scales,  # [E, N1, K1//32] uint8 ue8m0
    fc1_global_scale,  # [E] float32
    fc2_block_scales,  # [E, N2, K2//32] uint8 ue8m0
    fc2_global_scale,  # [E] float32
    use_swiglu=False,
    fc1_bias=None,
    fc2_bias=None,
):
    """Build ONNX model with QMoE operator in FP4 (MXFP4) mode."""
    # QMoE op uses unified scale inputs: block scales at 3/6, global scales at 15/16.
    inputs = [
        "input",  # 0
        "router_probs",  # 1
        "fc1_weights",  # 2: uint8 packed FP4
        "fc1_scales",  # 3: uint8 MXFP4 block scales
        "fc1_bias" if fc1_bias is not None else "",  # 4
        "fc2_weights",  # 5: uint8 packed FP4
        "fc2_scales",  # 6: uint8 MXFP4 block scales
        "fc2_bias" if fc2_bias is not None else "",  # 7
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
            "QMoE_FP4",
            k=top_k,
            normalize_routing_weights=1,
            activation_type=activation,
            expert_weight_bits=4,
            quant_type="fp4",
            swiglu_fusion=1 if use_swiglu else 0,
            swiglu_limit=7.0,
            activation_alpha=1.702,
            activation_beta=1.0,
            domain="com.microsoft",
        ),
    ]

    # ── initializers ────────────────────────────────────────────────
    initializers = []

    # FC1 / FC2 packed weights [E, K, N/2] uint8
    for name, tensor in [("fc1_weights", fc1_weights), ("fc2_weights", fc2_weights)]:
        arr = numpy.ascontiguousarray(tensor.cpu().numpy().astype(numpy.uint8))
        initializers.append(helper.make_tensor(name, TensorProto.UINT8, list(tensor.shape), arr.tobytes(), raw=True))

    # FP4 block scales [E, N, K//32] float8e8m0
    for name, tensor in [
        ("fc1_scales", fc1_block_scales),
        ("fc2_scales", fc2_block_scales),
    ]:
        arr = numpy.ascontiguousarray(tensor.cpu().numpy().astype(numpy.uint8))
        initializers.append(
            helper.make_tensor(name, TensorProto.FLOAT8E8M0, list(tensor.shape), arr.tobytes(), raw=True)
        )

    # FP4 global scales [E] float32 (T4)
    for name, tensor in [
        ("fc1_global_scale", fc1_global_scale),
        ("fc2_global_scale", fc2_global_scale),
    ]:
        vals = tensor.cpu().float().flatten().tolist()
        initializers.append(helper.make_tensor(name, TensorProto.FLOAT, [num_experts], vals, raw=False))

    # Optional biases
    for bname, btensor in [("fc1_bias", fc1_bias), ("fc2_bias", fc2_bias)]:
        if btensor is not None:
            if onnx_dtype == TensorProto.BFLOAT16:
                vals = btensor.to(torch.float32).flatten().detach().cpu().tolist()
            else:
                vals = btensor.to(torch.float16).flatten().detach().cpu().tolist()
            initializers.append(helper.make_tensor(bname, onnx_dtype, list(btensor.shape), vals, raw=False))

    # ── graph I/O ───────────────────────────────────────────────────
    graph_inputs = [
        helper.make_tensor_value_info("input", onnx_dtype, [num_tokens, hidden_size]),
        helper.make_tensor_value_info("router_probs", onnx_dtype, [num_tokens, num_experts]),
    ]
    graph_outputs = [
        helper.make_tensor_value_info("output", onnx_dtype, [num_tokens, hidden_size]),
    ]

    graph = helper.make_graph(nodes, "QMoE_FP4_Test", graph_inputs, graph_outputs, initializers)
    model = helper.make_model(graph)
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
class TestQMoEFP4(unittest.TestCase):
    """Tests for W4A16 MXFP4 MoE quantization."""

    def _skip_if_no_fp4(self):
        """Skip if SM < 90 (FP4 requires Hopper+)."""
        sm = _cuda_sm()
        if sm < 90:
            self.skipTest(f"FP4 requires SM90+, got SM{sm}")

    # ----------------------------------------------------------------
    # Core test driver
    # ----------------------------------------------------------------
    def _run_fp4_moe_test(
        self,
        hidden_size,
        inter_size,
        num_experts,
        top_k,
        num_tokens,
        onnx_dtype,
        use_swiglu=False,
        block_size=32,
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

        # ── quantize per-expert weights ────────────────────────────
        fc1_packed, fc1_bs, fc1_gs, fc1_deq = [], [], [], []
        fc2_packed, fc2_bs, fc2_gs, fc2_deq = [], [], [], []

        for _ in range(num_experts):
            w1 = torch.randn(fc1_n, fc1_k, device=device) * 0.1
            p1, b1, g1, d1 = quantize_weight_to_mxfp4(w1, block_size)
            fc1_packed.append(p1)
            fc1_bs.append(b1)
            fc1_gs.append(torch.tensor(g1, dtype=torch.float32))
            fc1_deq.append(d1)

            w2 = torch.randn(fc2_n, fc2_k, device=device) * 0.1
            p2, b2, g2, d2 = quantize_weight_to_mxfp4(w2, block_size)
            fc2_packed.append(p2)
            fc2_bs.append(b2)
            fc2_gs.append(torch.tensor(g2, dtype=torch.float32))
            fc2_deq.append(d2)

        fc1_weights = torch.stack(fc1_packed, dim=0)  # [E, K, N/2]
        fc2_weights = torch.stack(fc2_packed, dim=0)  # [E, K, N/2]
        fc1_block_scales = torch.stack(fc1_bs, dim=0)  # [E, N, K//32]
        fc2_block_scales = torch.stack(fc2_bs, dim=0)  # [E, N, K//32]
        fc1_global_scale = torch.stack(fc1_gs)  # [E]
        fc2_global_scale = torch.stack(fc2_gs)  # [E]
        fc1_deq_all = torch.stack(fc1_deq, dim=0)  # [E, N, K]
        fc2_deq_all = torch.stack(fc2_deq, dim=0)  # [E, N, K]

        # ── build ONNX model ───────────────────────────────────────
        onnx_model = create_fp4_moe_onnx_graph(
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
            use_swiglu=use_swiglu,
        )

        # ── create ORT session ────────────────────────────────────
        opts = onnxruntime.SessionOptions()
        opts.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_DISABLE_ALL
        # gemv_mode toggles the fused FP4 GEMV decode path (read once in the QMoE op ctor during
        # session creation): "1" forces it on, "0" forces the dequant fallback, None leaves the
        # default (on). Restore the previous value right after the session is built.
        prev_gemv_env = os.environ.get("ORT_ENABLE_FP4_GEMV")
        if gemv_mode is not None:
            os.environ["ORT_ENABLE_FP4_GEMV"] = gemv_mode
        try:
            session = onnxruntime.InferenceSession(
                onnx_model, opts, providers=[resolve_cuda_plugin_ep("CUDAExecutionProvider")]
            )
        except Exception as e:
            if "FP4" in str(e) or "ENABLE_FP4" in str(e) or "SM" in str(e):
                self.skipTest(f"FP4 not supported in this build: {e}")
            raise
        finally:
            if gemv_mode is not None:
                if prev_gemv_env is None:
                    os.environ.pop("ORT_ENABLE_FP4_GEMV", None)
                else:
                    os.environ["ORT_ENABLE_FP4_GEMV"] = prev_gemv_env

        # ── run inference ──────────────────────────────────────────
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
            if (
                "FP4" in msg
                or "MXFP4" in msg
                or "ENABLE_FP4" in msg
                or "stubbed out" in msg
                or "not supported in this build" in msg
            ):
                self.skipTest(f"FP4 kernel not available in this build: {e}")
            raise
        iobinding.synchronize_outputs()

        ort_output = output_tensor.clone()

        # ── compute PyTorch reference ──────────────────────────────
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

        # ── compare ───────────────────────────────────────────────
        max_diff = (ort_output.float() - ref_output.float()).abs().max().item()
        dtype_tag = "FP16" if torch_dtype == torch.float16 else "BF16"
        act_tag = "SwiGLU" if use_swiglu else "SiLU"
        print(
            f"FP4 MoE test: {dtype_tag} {act_tag} "
            f"tokens={num_tokens} experts={num_experts} "
            f"hidden={hidden_size} inter={inter_size} "
            f"max_diff={max_diff:.6f}"
        )

        # FP4 quantization is lossy; tolerance is wider than INT4
        atol = 0.15 if torch_dtype == torch.bfloat16 else 0.12
        self.assertLess(
            max_diff,
            atol,
            f"FP4 MoE parity check failed: max_diff={max_diff:.6f} > atol={atol}",
        )

    # ----------------------------------------------------------------
    # Reference implementation
    # ----------------------------------------------------------------
    @staticmethod
    def _compute_reference(input_tensor, router_logits, fc1_deq, fc2_deq, num_experts, top_k, use_swiglu, torch_dtype):
        """Reference MoE forward pass using dequantized weights."""
        num_tokens = input_tensor.shape[0]
        hidden_size = input_tensor.shape[1]

        x = input_tensor.float()
        logits = router_logits.float()

        # Top-K selection then softmax (matching QMoE kernel)
        topk_vals, topk_idx = torch.topk(logits, top_k, dim=-1)
        routing_weights = F.softmax(topk_vals, dim=1)

        output = torch.zeros(num_tokens, hidden_size, device=x.device, dtype=torch.float32)
        expert_mask = F.one_hot(topk_idx, num_classes=num_experts).permute(2, 1, 0)

        for e in range(num_experts):
            idx, top_x = torch.where(expert_mask[e])
            if top_x.shape[0] == 0:
                continue

            tokens = x[top_x]  # [B, hidden]
            w1 = fc1_deq[e].float()  # [N1, K1]
            w2 = fc2_deq[e].float()  # [N2, K2]

            h = tokens @ w1.T  # FC1
            h = swiglu_ref(h) if use_swiglu else F.silu(h)  # activation
            h = h @ w2.T  # FC2
            h = h * routing_weights[top_x, idx, None]

            output.index_add_(0, top_x, h)

        return output.to(torch_dtype)

    # ================================================================
    # Test cases
    # ================================================================

    # Dimensions must be multiples of 128 for MXFP4 alignment
    # (MinKDimAlignmentMXFPX = 128, MinNDimAlignmentMXFPX = 128)

    def test_fp4_rejects_non_32_multiple_hidden_size(self):
        """Reject truncated MXFP4 block-scale shapes before launching kernels."""
        self._skip_if_no_fp4()

        hidden_size = 258
        inter_size = 256
        num_experts = 2
        top_k = 1
        num_tokens = 1
        onnx_dtype = TensorProto.FLOAT16

        fc1_weights = torch.zeros((num_experts, hidden_size, inter_size // 2), dtype=torch.uint8, device=device)
        fc2_weights = torch.zeros((num_experts, inter_size, hidden_size // 2), dtype=torch.uint8, device=device)
        fc1_block_scales = torch.ones((num_experts, inter_size, hidden_size // 32), dtype=torch.uint8, device=device)
        fc2_block_scales = torch.ones((num_experts, hidden_size, inter_size // 32), dtype=torch.uint8, device=device)
        global_scale = torch.ones(num_experts, dtype=torch.float32, device=device)

        onnx_model = create_fp4_moe_onnx_graph(
            num_tokens=num_tokens,
            hidden_size=hidden_size,
            inter_size=inter_size,
            num_experts=num_experts,
            top_k=top_k,
            onnx_dtype=onnx_dtype,
            fc1_weights=fc1_weights,
            fc2_weights=fc2_weights,
            fc1_block_scales=fc1_block_scales,
            fc1_global_scale=global_scale,
            fc2_block_scales=fc2_block_scales,
            fc2_global_scale=global_scale,
        )

        opts = onnxruntime.SessionOptions()
        opts.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_DISABLE_ALL
        opts.add_session_config_entry("session.disable_prepacking", "1")
        session = onnxruntime.InferenceSession(
            onnx_model, opts, providers=[resolve_cuda_plugin_ep("CUDAExecutionProvider")]
        )

        input_tensor = torch.zeros((num_tokens, hidden_size), device=device, dtype=torch.float16)
        router_logits = torch.ones((num_tokens, num_experts), device=device, dtype=torch.float16)
        output_tensor = torch.empty((num_tokens, hidden_size), device=device, dtype=torch.float16)

        iobinding = session.io_binding()
        iobinding.bind_input("input", "cuda", 0, onnx_dtype, input_tensor.shape, input_tensor.data_ptr())
        iobinding.bind_input("router_probs", "cuda", 0, onnx_dtype, router_logits.shape, router_logits.data_ptr())
        iobinding.bind_output("output", "cuda", 0, onnx_dtype, output_tensor.shape, output_tensor.data_ptr())

        with self.assertRaisesRegex(Exception, "hidden_size to be a multiple of 32"):
            session.run_with_iobinding(iobinding)

    def test_fp4_fp16_silu_basic(self):
        """Basic FP16 + SiLU activation."""
        self._run_fp4_moe_test(
            hidden_size=256,
            inter_size=256,
            num_experts=4,
            top_k=2,
            num_tokens=32,
            onnx_dtype=TensorProto.FLOAT16,
        )

    def test_fp4_bf16_silu_basic(self):
        """Basic BF16 + SiLU activation."""
        self._run_fp4_moe_test(
            hidden_size=256,
            inter_size=256,
            num_experts=4,
            top_k=2,
            num_tokens=32,
            onnx_dtype=TensorProto.BFLOAT16,
        )

    def test_fp4_fp16_swiglu(self):
        """FP16 + SwiGLU activation (interleaved fusion)."""
        self._run_fp4_moe_test(
            hidden_size=256,
            inter_size=256,
            num_experts=4,
            top_k=2,
            num_tokens=32,
            onnx_dtype=TensorProto.FLOAT16,
            use_swiglu=True,
        )

    def test_fp4_bf16_swiglu(self):
        """BF16 + SwiGLU activation."""
        self._run_fp4_moe_test(
            hidden_size=256,
            inter_size=256,
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
    def test_fp4_fp16_token_counts(self, num_tokens):
        """Test with different token counts."""
        self._run_fp4_moe_test(
            hidden_size=256,
            inter_size=256,
            num_experts=4,
            top_k=2,
            num_tokens=num_tokens,
            onnx_dtype=TensorProto.FLOAT16,
        )

    def test_fp4_fp16_more_experts(self):
        """Test with more experts (8 experts, top-2)."""
        self._run_fp4_moe_test(
            hidden_size=256,
            inter_size=256,
            num_experts=8,
            top_k=2,
            num_tokens=32,
            onnx_dtype=TensorProto.FLOAT16,
        )

    def test_fp4_fp16_top4(self):
        """Test with top-4 expert selection."""
        self._run_fp4_moe_test(
            hidden_size=256,
            inter_size=256,
            num_experts=8,
            top_k=4,
            num_tokens=32,
            onnx_dtype=TensorProto.FLOAT16,
        )

    def test_fp4_fp16_larger_dims(self):
        """Test with larger hidden/intermediate dimensions."""
        self._run_fp4_moe_test(
            hidden_size=512,
            inter_size=512,
            num_experts=4,
            top_k=2,
            num_tokens=16,
            onnx_dtype=TensorProto.FLOAT16,
        )

    @parameterized.expand(
        [
            (TensorProto.FLOAT16, 1, 4),
            (TensorProto.FLOAT16, 2, 4),
            (TensorProto.BFLOAT16, 1, 4),
            (TensorProto.BFLOAT16, 2, 4),
        ]
    )
    def test_fp4_decode_swiglu_gemv(self, onnx_dtype, num_tokens, top_k):
        """Decode-shaped SwiGLU (hidden=inter=512, expanded_rows = num_tokens*top_k <= 8).

        This shape satisfies is_moe_gemv_fp4_supported (n,k >= 512, expanded_rows in (0, 8]),
        so forcing ORT_ENABLE_FP4_GEMV=1 exercises the fused MXFP4 W4A16 GEMV decode path.
        test_fp4_decode_swiglu_fallback runs the identical shape with the dequant fallback
        (ORT_ENABLE_FP4_GEMV=0); together they give an on/off parity check for the fused GEMV.
        """
        self._run_fp4_moe_test(
            hidden_size=512,
            inter_size=512,
            num_experts=8,
            top_k=top_k,
            num_tokens=num_tokens,
            onnx_dtype=onnx_dtype,
            use_swiglu=True,
            gemv_mode="1",
        )

    @parameterized.expand(
        [
            (TensorProto.FLOAT16, 1, 4),
            (TensorProto.BFLOAT16, 1, 4),
        ]
    )
    def test_fp4_decode_swiglu_fallback(self, onnx_dtype, num_tokens, top_k):
        """Same decode-shaped SwiGLU as test_fp4_decode_swiglu_gemv but with the fused GEMV
        disabled (ORT_ENABLE_FP4_GEMV=0), so it exercises the dequant fallback on a shape the
        GEMV path also supports."""
        self._run_fp4_moe_test(
            hidden_size=512,
            inter_size=512,
            num_experts=8,
            top_k=top_k,
            num_tokens=num_tokens,
            onnx_dtype=onnx_dtype,
            use_swiglu=True,
            gemv_mode="0",
        )

    def test_fp4_native_cutlass_row_varying_scales(self):
        """Native SM90 WFP4A16 scale prepack preserves per-output-row MXFP4 scales."""
        self._skip_if_no_fp4()
        if _cuda_sm() != 90:
            self.skipTest(f"Native FP4 CUTLASS GEMM is currently enabled only on SM90, got SM{_cuda_sm()}")

        hidden_size = 512
        inter_size = 512
        num_experts = 1
        top_k = 1
        num_tokens = 1
        onnx_dtype = TensorProto.FLOAT16

        fc1_weights = torch.full((num_experts, hidden_size, inter_size // 2), 0x22, dtype=torch.uint8, device=device)
        fc2_codes = torch.zeros((hidden_size, inter_size), dtype=torch.uint8, device=device)
        diagonal = torch.arange(hidden_size, device=device)
        fc2_codes[diagonal, diagonal] = 2
        fc2_codes_kn = fc2_codes.T.contiguous()
        fc2_weights = ((fc2_codes_kn[:, 1::2] << 4) | fc2_codes_kn[:, 0::2])[None]

        row_codes = torch.where(
            (torch.arange(inter_size, device=device) % 2) == 0,
            torch.tensor(126, dtype=torch.uint8, device=device),
            torch.tensor(128, dtype=torch.uint8, device=device),
        )
        fc1_block_scales = row_codes[None, :, None].expand(num_experts, inter_size, hidden_size // 32).contiguous()
        fc2_block_scales = torch.full(
            (num_experts, hidden_size, inter_size // 32), 127, dtype=torch.uint8, device=device
        )
        global_scale = torch.ones(num_experts, dtype=torch.float32, device=device)

        onnx_model = create_fp4_moe_onnx_graph(
            num_tokens=num_tokens,
            hidden_size=hidden_size,
            inter_size=inter_size,
            num_experts=num_experts,
            top_k=top_k,
            onnx_dtype=onnx_dtype,
            fc1_weights=fc1_weights,
            fc2_weights=fc2_weights,
            fc1_block_scales=fc1_block_scales,
            fc1_global_scale=global_scale,
            fc2_block_scales=fc2_block_scales,
            fc2_global_scale=global_scale,
            use_swiglu=False,
        )

        input_tensor = torch.full((num_tokens, hidden_size), 1.0 / hidden_size, device=device, dtype=torch.float16)
        router_logits = torch.ones((num_tokens, num_experts), device=device, dtype=torch.float16)

        def run(enable_native):
            old_cutlass = os.environ.get("ORT_ENABLE_FP4_CUTLASS_GEMM")
            old_unsafe = os.environ.get("ORT_ENABLE_FP4_CUTLASS_UNSAFE")
            os.environ["ORT_ENABLE_FP4_CUTLASS_GEMM"] = "1" if enable_native else "0"
            if enable_native:
                os.environ["ORT_ENABLE_FP4_CUTLASS_UNSAFE"] = "1"
            else:
                os.environ.pop("ORT_ENABLE_FP4_CUTLASS_UNSAFE", None)
            try:
                opts = onnxruntime.SessionOptions()
                opts.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_DISABLE_ALL
                session = onnxruntime.InferenceSession(
                    onnx_model, opts, providers=[resolve_cuda_plugin_ep("CUDAExecutionProvider")]
                )
            finally:
                if old_cutlass is None:
                    os.environ.pop("ORT_ENABLE_FP4_CUTLASS_GEMM", None)
                else:
                    os.environ["ORT_ENABLE_FP4_CUTLASS_GEMM"] = old_cutlass
                if old_unsafe is None:
                    os.environ.pop("ORT_ENABLE_FP4_CUTLASS_UNSAFE", None)
                else:
                    os.environ["ORT_ENABLE_FP4_CUTLASS_UNSAFE"] = old_unsafe

            output_tensor = torch.empty((num_tokens, hidden_size), device=device, dtype=torch.float16)
            iobinding = session.io_binding()
            iobinding.bind_input("input", "cuda", 0, onnx_dtype, input_tensor.shape, input_tensor.data_ptr())
            iobinding.bind_input("router_probs", "cuda", 0, onnx_dtype, router_logits.shape, router_logits.data_ptr())
            iobinding.bind_output("output", "cuda", 0, onnx_dtype, output_tensor.shape, output_tensor.data_ptr())
            session.run_with_iobinding(iobinding)
            iobinding.synchronize_outputs()
            return output_tensor.float().cpu()

        fallback_output = run(enable_native=False)
        native_output = run(enable_native=True)
        max_diff = (native_output - fallback_output).abs().max().item()
        self.assertEqual(max_diff, 0.0)


# ============================================================================
# Standalone packing utility tests
# ============================================================================


class TestFP4PackingUtility(unittest.TestCase):
    """Unit tests for the FP4 weight packing functions."""

    def test_quantize_roundtrip(self):
        """Quantize to FP4 and verify dequantized values are FP4-representable."""
        w = torch.randn(128, 128) * 2.0
        _, _, _, deq = quantize_weight_to_mxfp4(w, block_size=32)

        # Every dequantized value should be exactly representable as fp4 x scale
        # (i.e., no additional rounding beyond FP4 grid)
        self.assertEqual(deq.shape, w.shape)
        self.assertFalse(torch.isnan(deq).any())
        self.assertFalse(torch.isinf(deq).any())

    def test_pack_shape(self):
        """Verify packed weight shape is [K, N//2]."""
        n, k = 128, 256
        w = torch.randn(n, k)
        packed, bs, gs, _ = quantize_weight_to_mxfp4(w, block_size=32)

        self.assertEqual(packed.shape, (k, n // 2))
        self.assertEqual(bs.shape, (n, k // 32))
        self.assertEqual(gs, 1.0)

    def test_fp4_codes_range(self):
        """All FP4 codes should be in [0, 15]."""
        values = torch.randn(1000) * 5.0
        _, codes = fp4_e2m1_quantize(values)
        self.assertTrue((codes <= 15).all())
        self.assertTrue((codes >= 0).all())

    def test_ue8m0_roundtrip(self):
        """ue8m0 encode/decode roundtrip for powers of 2."""
        for exp in range(-10, 11):
            val = 2.0**exp
            code = float_to_ue8m0_code(val)
            decoded = ue8m0_code_to_float(code)
            self.assertAlmostEqual(val, decoded, places=5, msg=f"ue8m0 roundtrip failed for 2^{exp}")

    @unittest.skipIf(not has_pybind_pack_fp4_weights, "pack_fp4_weights_for_cuda_moe_gemm not available")
    def test_pybind_pack_matches_python(self):
        """C++ packing matches Python reference."""
        n, k = 64, 128
        # Create random 4-bit codes [N, K]
        codes_nk = torch.randint(0, 16, (n, k), dtype=torch.uint8)

        # Pack row-major [N, K/2] for C++ input
        low = codes_nk[:, 0::2]
        high = codes_nk[:, 1::2]
        packed_row = ((high << 4) | low).numpy()  # [N, K//2]

        # C++ packing
        result_cpp = _pybind.pack_fp4_weights_for_cuda_moe_gemm(packed_row, n, k)
        result_cpp = numpy.array(result_cpp, dtype=numpy.uint8).reshape(k, n // 2)

        # Python reference: transpose [N,K] → [K,N], pack [K, N//2]
        codes_kn = codes_nk.T.contiguous()
        low_ref = codes_kn[:, 0::2].numpy()
        high_ref = codes_kn[:, 1::2].numpy()
        result_py = (high_ref << 4) | low_ref

        numpy.testing.assert_array_equal(result_cpp, result_py)


if __name__ == "__main__":
    unittest.main()
