# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.  See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

"""Synthetic ONNX graph generators for Qwen3 transformer optimization tests.

Qwen3 is a decoder-only architecture with:
  - RMSNorm (SimplifiedLayerNormalization) instead of LayerNorm
  - Grouped Query Attention (GQA): fewer KV heads than Q heads
  - QK-Norm: RMSNorm applied to Q and K after projection
  - RoPE: rotary positional embeddings
  - SwiGLU activation in FFN
"""

import numpy as np
from onnx import TensorProto, helper


def _float_tensor(name, shape, random=False):
    total = 1
    for d in shape:
        total *= d
    vals = [np.random.uniform(0, 1) for _ in range(total)] if random else [1.0] * total
    return helper.make_tensor(name, TensorProto.FLOAT, shape, vals)


def _rmsnorm_nodes(prefix, input_name, weight_name, output_name, eps=1e-6):
    """Build the raw-op RMSNorm pattern: Pow -> ReduceMean -> Add(eps) -> Sqrt -> Reciprocal -> Mul -> Mul(weight).

    This is the pattern that FusionSimplifiedLayerNormalization fuses into SimplifiedLayerNormalization.
    """
    return [
        helper.make_node("Pow", [input_name, f"{prefix}_pow_exp"], [f"{prefix}_pow_out"], f"{prefix}_pow"),
        helper.make_node(
            "ReduceMean",
            [f"{prefix}_pow_out"],
            [f"{prefix}_mean_out"],
            f"{prefix}_mean",
            axes=[-1],
            keepdims=1,
        ),
        helper.make_node(
            "Add", [f"{prefix}_mean_out", f"{prefix}_eps"], [f"{prefix}_add_eps_out"], f"{prefix}_add_eps"
        ),
        helper.make_node("Sqrt", [f"{prefix}_add_eps_out"], [f"{prefix}_sqrt_out"], f"{prefix}_sqrt"),
        helper.make_node("Reciprocal", [f"{prefix}_sqrt_out"], [f"{prefix}_rsqrt_out"], f"{prefix}_rsqrt"),
        helper.make_node("Mul", [input_name, f"{prefix}_rsqrt_out"], [f"{prefix}_normed"], f"{prefix}_mul_rsqrt"),
        helper.make_node("Mul", [weight_name, f"{prefix}_normed"], [output_name], f"{prefix}_mul_weight"),
    ]


def _rmsnorm_initializers(prefix, hidden_size, eps=1e-6):
    """Initializers for a single RMSNorm block."""
    return [
        helper.make_tensor(f"{prefix}_pow_exp", TensorProto.FLOAT, [1], [2.0]),
        helper.make_tensor(f"{prefix}_eps", TensorProto.FLOAT, [], [eps]),
    ]


def create_qwen3_decoder_layer(
    hidden_size=64,
    num_heads=8,
    num_kv_heads=2,
    batch_size=1,
    seq_len=4,
):
    """Create a single Qwen3 decoder layer with RMSNorm, Q/K/V projections, QK-Norm, and residual Add.

    The generated graph exercises:
      - SimplifiedLayerNormalization fusion (pre-attn RMSNorm, Q-norm, K-norm)
      - SkipSimplifiedLayerNormalization fusion (residual Add + post-attn RMSNorm)

    Returns an onnx.ModelProto.
    """
    head_dim = hidden_size // num_heads
    kv_dim = num_kv_heads * head_dim

    nodes = []
    initializers = []

    # --- Pre-attention RMSNorm ---
    nodes.extend(_rmsnorm_nodes("pre_ln", "input_0", "pre_ln_weight", "pre_ln_out"))
    initializers.extend(_rmsnorm_initializers("pre_ln", hidden_size))
    initializers.append(_float_tensor("pre_ln_weight", [hidden_size]))

    # --- Q projection: MatMul ---
    nodes.append(helper.make_node("MatMul", ["pre_ln_out", "q_weight"], ["q_proj"], "q_matmul"))
    initializers.append(_float_tensor("q_weight", [hidden_size, hidden_size], random=True))

    # --- K projection: MatMul ---
    nodes.append(helper.make_node("MatMul", ["pre_ln_out", "k_weight"], ["k_proj"], "k_matmul"))
    initializers.append(_float_tensor("k_weight", [hidden_size, kv_dim], random=True))

    # --- V projection: MatMul ---
    nodes.append(helper.make_node("MatMul", ["pre_ln_out", "v_weight"], ["v_proj"], "v_matmul"))
    initializers.append(_float_tensor("v_weight", [hidden_size, kv_dim], random=True))

    # --- Q reshape to (batch, seq, num_heads, head_dim) ---
    q_shape = [batch_size, seq_len, num_heads, head_dim]
    nodes.append(helper.make_node("Reshape", ["q_proj", "q_shape"], ["q_reshaped"], "q_reshape"))
    initializers.append(helper.make_tensor("q_shape", TensorProto.INT64, [4], q_shape))

    # --- K reshape to (batch, seq, num_kv_heads, head_dim) ---
    k_shape = [batch_size, seq_len, num_kv_heads, head_dim]
    nodes.append(helper.make_node("Reshape", ["k_proj", "k_shape"], ["k_reshaped"], "k_reshape"))
    initializers.append(helper.make_tensor("k_shape", TensorProto.INT64, [4], k_shape))

    # --- QK-Norm: RMSNorm on Q ---
    nodes.extend(_rmsnorm_nodes("q_norm", "q_reshaped", "q_norm_weight", "q_normed"))
    initializers.extend(_rmsnorm_initializers("q_norm", head_dim))
    initializers.append(_float_tensor("q_norm_weight", [head_dim]))

    # --- QK-Norm: RMSNorm on K ---
    nodes.extend(_rmsnorm_nodes("k_norm", "k_reshaped", "k_norm_weight", "k_normed"))
    initializers.extend(_rmsnorm_initializers("k_norm", head_dim))
    initializers.append(_float_tensor("k_norm_weight", [head_dim]))

    # --- Transposes for multi-head layout ---
    nodes.append(helper.make_node("Transpose", ["q_normed"], ["q_transposed"], "q_transpose", perm=[0, 2, 1, 3]))
    nodes.append(helper.make_node("Transpose", ["k_normed"], ["k_transposed"], "k_transpose", perm=[0, 2, 1, 3]))

    # --- V reshape + transpose ---
    nodes.append(helper.make_node("Reshape", ["v_proj", "k_shape"], ["v_reshaped"], "v_reshape"))
    nodes.append(helper.make_node("Transpose", ["v_reshaped"], ["v_transposed"], "v_transpose", perm=[0, 2, 1, 3]))

    # --- Simplified attention: QK^T -> Softmax -> *V ---
    nodes.append(helper.make_node("Transpose", ["k_transposed"], ["k_T"], "k_transpose_for_matmul", perm=[0, 1, 3, 2]))
    nodes.append(helper.make_node("MatMul", ["q_transposed", "k_T"], ["qk_scores"], "qk_matmul"))
    nodes.append(helper.make_node("Mul", ["qk_scores", "scale_factor"], ["qk_scaled"], "qk_scale"))
    initializers.append(helper.make_tensor("scale_factor", TensorProto.FLOAT, [1], [1.0 / (head_dim**0.5)]))
    nodes.append(helper.make_node("Softmax", ["qk_scaled"], ["attn_weights"], "softmax", axis=-1))
    nodes.append(helper.make_node("MatMul", ["attn_weights", "v_transposed"], ["attn_out"], "attn_v_matmul"))

    # --- Transpose attention output back ---
    nodes.append(helper.make_node("Transpose", ["attn_out"], ["attn_transposed"], "attn_transpose", perm=[0, 2, 1, 3]))

    # --- Reshape to (batch, seq, hidden) ---
    out_shape = [batch_size, seq_len, hidden_size]
    nodes.append(helper.make_node("Reshape", ["attn_transposed", "out_shape"], ["attn_flat"], "attn_reshape"))
    initializers.append(helper.make_tensor("out_shape", TensorProto.INT64, [3], out_shape))

    # --- Output projection ---
    nodes.append(helper.make_node("MatMul", ["attn_flat", "o_weight"], ["o_proj"], "o_matmul"))
    initializers.append(_float_tensor("o_weight", [hidden_size, hidden_size], random=True))

    # --- Residual Add -> SkipSimplifiedLayerNormalization pattern ---
    nodes.append(helper.make_node("Add", ["input_0", "o_proj"], ["residual_add"], "residual_add"))

    # --- Post-attention RMSNorm (anchored on residual_add -> will fuse to SkipSimplifiedLN) ---
    nodes.extend(_rmsnorm_nodes("post_ln", "residual_add", "post_ln_weight", "post_ln_out"))
    initializers.extend(_rmsnorm_initializers("post_ln", hidden_size))
    initializers.append(_float_tensor("post_ln_weight", [hidden_size]))

    # --- Simplified FFN: just a MatMul + residual for testing ---
    nodes.append(helper.make_node("MatMul", ["post_ln_out", "ffn_weight"], ["ffn_out"], "ffn_matmul"))
    initializers.append(_float_tensor("ffn_weight", [hidden_size, hidden_size], random=True))

    nodes.append(helper.make_node("Add", ["residual_add", "ffn_out"], ["output_0"], "ffn_residual_add"))

    # --- Graph definition ---
    graph = helper.make_graph(
        nodes,
        "qwen3_decoder_layer",
        [
            helper.make_tensor_value_info("input_0", TensorProto.FLOAT, [batch_size, seq_len, hidden_size]),
        ],
        [
            helper.make_tensor_value_info("output_0", TensorProto.FLOAT, [batch_size, seq_len, hidden_size]),
        ],
        initializers,
    )

    model = helper.make_model(graph)
    model.ir_version = 7
    model.opset_import[0].version = 17
    return model
