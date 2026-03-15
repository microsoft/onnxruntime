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
  - RoPE: rotary positional embeddings (on-the-fly computation)
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


def _rotate_half_nodes(prefix, input_name, output_name, cos_name, sin_name):
    """Build the rotate_half + apply_rotary_pos_emb pattern with dynamic Shape→Gather→Div.

    Pattern: x_embed = (x * cos) + (rotate_half(x) * sin)
    Where rotate_half(x) = concat(-x[..., dim//2:], x[..., :dim//2])

    Uses the dynamic Shape→Gather→Div→Cast→Cast→Unsqueeze pattern for Slice indices,
    matching Qwen3's TorchScript export.
    """
    nodes = []

    # Compute dim//2 dynamically: Shape → Gather(dim=-1) → Div(2) → Cast → Cast → Unsqueeze
    nodes.append(helper.make_node("Shape", [input_name], [f"{prefix}_shape"], f"{prefix}_shape"))
    nodes.append(
        helper.make_node(
            "Gather", [f"{prefix}_shape", f"{prefix}_dim_idx"], [f"{prefix}_last_dim"], f"{prefix}_gather_dim"
        )
    )
    nodes.append(
        helper.make_node("Div", [f"{prefix}_last_dim", f"{prefix}_two"], [f"{prefix}_half_dim"], f"{prefix}_div2")
    )
    # Cast nodes (from floor division tracing in TorchScript)
    nodes.append(
        helper.make_node("Cast", [f"{prefix}_half_dim"], [f"{prefix}_half_cast1"], f"{prefix}_cast1", to=7)
    )  # to INT64
    nodes.append(helper.make_node("Cast", [f"{prefix}_half_cast1"], [f"{prefix}_half_cast2"], f"{prefix}_cast2", to=7))

    # Unsqueeze for Slice starts/ends
    nodes.append(
        helper.make_node(
            "Unsqueeze",
            [f"{prefix}_half_cast2", f"{prefix}_unsq_axis"],
            [f"{prefix}_half_unsq"],
            f"{prefix}_unsq_half",
        )
    )

    # x1 = x[..., :dim//2]  (Slice with starts=0, ends=half_dim, axes=-1)
    nodes.append(
        helper.make_node(
            "Slice",
            [input_name, f"{prefix}_zero_start", f"{prefix}_half_unsq", f"{prefix}_slice_axis", f"{prefix}_one_step"],
            [f"{prefix}_x1"],
            f"{prefix}_slice_x1",
        )
    )

    # x2 = x[..., dim//2:]  (Slice with starts=half_dim, ends=INT64_MAX, axes=-1)
    nodes.append(
        helper.make_node(
            "Slice",
            [input_name, f"{prefix}_half_unsq", f"{prefix}_large_end", f"{prefix}_slice_axis", f"{prefix}_one_step"],
            [f"{prefix}_x2"],
            f"{prefix}_slice_x2",
        )
    )

    # rotate_half = concat(-x2, x1)
    nodes.append(helper.make_node("Neg", [f"{prefix}_x2"], [f"{prefix}_neg_x2"], f"{prefix}_neg"))
    nodes.append(
        helper.make_node(
            "Concat", [f"{prefix}_neg_x2", f"{prefix}_x1"], [f"{prefix}_rotated"], f"{prefix}_concat", axis=-1
        )
    )

    # x * cos
    nodes.append(helper.make_node("Mul", [input_name, cos_name], [f"{prefix}_x_cos"], f"{prefix}_mul_cos"))

    # rotate_half(x) * sin
    nodes.append(helper.make_node("Mul", [f"{prefix}_rotated", sin_name], [f"{prefix}_rot_sin"], f"{prefix}_mul_sin"))

    # x_embed = x*cos + rotate_half(x)*sin
    nodes.append(helper.make_node("Add", [f"{prefix}_x_cos", f"{prefix}_rot_sin"], [output_name], f"{prefix}_add_rope"))

    return nodes


def _rotate_half_initializers(prefix):
    """Initializers for the dynamic Slice index computation in rotate_half."""
    return [
        helper.make_tensor(f"{prefix}_dim_idx", TensorProto.INT64, [], [3]),  # last dim index
        helper.make_tensor(f"{prefix}_two", TensorProto.INT64, [], [2]),
        helper.make_tensor(f"{prefix}_unsq_axis", TensorProto.INT64, [1], [0]),
        helper.make_tensor(f"{prefix}_zero_start", TensorProto.INT64, [1], [0]),
        helper.make_tensor(f"{prefix}_large_end", TensorProto.INT64, [1], [9223372036854775807]),  # INT64_MAX
        helper.make_tensor(f"{prefix}_slice_axis", TensorProto.INT64, [1], [-1]),
        helper.make_tensor(f"{prefix}_one_step", TensorProto.INT64, [1], [1]),
    ]


def _on_the_fly_rope_nodes(prefix, head_dim, include_expand=False):
    """Build the on-the-fly RoPE computation: MatMul(inv_freq, positions) → Cos/Sin → Mul(scaling).

    This matches Qwen3's RoPE pattern:
        inv_freq_expanded @ position_ids_expanded → Transpose → Concat(freqs, freqs) → Cos/Sin → Mul(scaling)

    When include_expand=True, adds Cast → Expand → Where nodes between inv_freq and MatMul,
    matching the pattern seen in some exports where inv_freq is explicitly expanded to batch size.
    """
    nodes = []

    # Unsqueeze position_ids: (B, S) → (B, 1, S)
    nodes.append(
        helper.make_node(
            "Unsqueeze",
            ["position_ids", f"{prefix}_unsq_axis"],
            [f"{prefix}_pos_unsq"],
            f"{prefix}_pos_unsqueeze",
        )
    )
    # Cast to float for MatMul
    nodes.append(helper.make_node("Cast", [f"{prefix}_pos_unsq"], [f"{prefix}_pos_float"], f"{prefix}_pos_cast", to=1))

    # inv_freq path: optionally add Cast → Expand → Where between inv_freq and MatMul
    if include_expand:
        matmul_inv_freq_input = f"{prefix}_inv_freq_where"
        nodes.append(
            helper.make_node(
                "Cast",
                [f"{prefix}_inv_freq"],
                [f"{prefix}_inv_freq_cast"],
                f"{prefix}_inv_freq_cast",
                to=1,
            )
        )
        nodes.append(
            helper.make_node(
                "Expand",
                [f"{prefix}_inv_freq_cast", f"{prefix}_expand_shape"],
                [f"{prefix}_inv_freq_expand"],
                f"{prefix}_inv_freq_expand",
            )
        )
        nodes.append(
            helper.make_node(
                "Where",
                [f"{prefix}_where_cond", f"{prefix}_inv_freq_expand", f"{prefix}_where_zero"],
                [matmul_inv_freq_input],
                f"{prefix}_inv_freq_where",
            )
        )
    else:
        matmul_inv_freq_input = f"{prefix}_inv_freq"

    # MatMul: inv_freq @ position_ids → freqs
    # inv_freq shape: (1, head_dim/2, 1) → expand not needed in test
    # position_ids shape: (B, 1, S)
    # Output: (B, head_dim/2, S) or (1, head_dim/2, S)
    nodes.append(
        helper.make_node(
            "MatMul",
            [matmul_inv_freq_input, f"{prefix}_pos_float"],
            [f"{prefix}_freqs_raw"],
            f"{prefix}_freq_matmul",
        )
    )

    # Transpose: (B, head_dim/2, S) → (B, S, head_dim/2)
    nodes.append(
        helper.make_node(
            "Transpose", [f"{prefix}_freqs_raw"], [f"{prefix}_freqs"], f"{prefix}_freq_transpose", perm=[0, 2, 1]
        )
    )

    # Concat(freqs, freqs): (B, S, head_dim/2) → (B, S, head_dim)
    nodes.append(
        helper.make_node(
            "Concat", [f"{prefix}_freqs", f"{prefix}_freqs"], [f"{prefix}_emb"], f"{prefix}_freq_concat", axis=-1
        )
    )

    # Cos and Sin
    nodes.append(helper.make_node("Cos", [f"{prefix}_emb"], [f"{prefix}_cos_raw"], f"{prefix}_cos"))
    nodes.append(helper.make_node("Sin", [f"{prefix}_emb"], [f"{prefix}_sin_raw"], f"{prefix}_sin"))

    # Mul by attention_scaling (1.0 for test)
    nodes.append(
        helper.make_node(
            "Mul", [f"{prefix}_cos_raw", f"{prefix}_scaling"], [f"{prefix}_cos_scaled"], f"{prefix}_cos_scale"
        )
    )
    nodes.append(
        helper.make_node(
            "Mul", [f"{prefix}_sin_raw", f"{prefix}_scaling"], [f"{prefix}_sin_scaled"], f"{prefix}_sin_scale"
        )
    )

    # Unsqueeze to add head dimension: (B, S, head_dim) → (B, 1, S, head_dim)
    nodes.append(
        helper.make_node(
            "Unsqueeze",
            [f"{prefix}_cos_scaled", f"{prefix}_head_unsq_axis"],
            [f"{prefix}_cos_out"],
            f"{prefix}_cos_unsqueeze",
        )
    )
    nodes.append(
        helper.make_node(
            "Unsqueeze",
            [f"{prefix}_sin_scaled", f"{prefix}_head_unsq_axis"],
            [f"{prefix}_sin_out"],
            f"{prefix}_sin_unsqueeze",
        )
    )

    return nodes


def _on_the_fly_rope_initializers(prefix, head_dim, batch_size=1, include_expand=False, inv_freq_as_graph_input=False):
    """Initializers for on-the-fly RoPE computation."""
    inv_freq = 1.0 / (10000.0 ** (np.arange(0, head_dim, 2, dtype=np.float32) / head_dim))
    inits = [
        helper.make_tensor(f"{prefix}_unsq_axis", TensorProto.INT64, [1], [1]),
        helper.make_tensor(f"{prefix}_scaling", TensorProto.FLOAT, [], [1.0]),
        helper.make_tensor(f"{prefix}_head_unsq_axis", TensorProto.INT64, [1], [1]),
    ]
    if not inv_freq_as_graph_input:
        inits.append(
            helper.make_tensor(f"{prefix}_inv_freq", TensorProto.FLOAT, [1, head_dim // 2, 1], inv_freq.tolist())
        )
    if include_expand:
        inits.extend(
            [
                helper.make_tensor(f"{prefix}_expand_shape", TensorProto.INT64, [3], [batch_size, head_dim // 2, 1]),
                helper.make_tensor(
                    f"{prefix}_where_cond",
                    TensorProto.BOOL,
                    [batch_size, head_dim // 2, 1],
                    [True] * (batch_size * (head_dim // 2)),
                ),
                helper.make_tensor(f"{prefix}_where_zero", TensorProto.FLOAT, [], [0.0]),
            ]
        )
    return inits


def create_qwen3_decoder_layer(
    hidden_size=64,
    num_heads=8,
    num_kv_heads=2,
    batch_size=1,
    seq_len=4,
    include_rope=False,
    include_expand_in_inv_freq=False,
    inv_freq_as_graph_input=False,
):
    """Create a single Qwen3 decoder layer with RMSNorm, Q/K/V projections, QK-Norm, and residual Add.

    The generated graph exercises:
      - SimplifiedLayerNormalization fusion (pre-attn RMSNorm, Q-norm, K-norm)
      - SkipSimplifiedLayerNormalization fusion (residual Add + post-attn RMSNorm)
      - RotaryEmbedding fusion (when include_rope=True)

    Returns an onnx.ModelProto.
    """
    head_dim = hidden_size // num_heads
    kv_dim = num_kv_heads * head_dim

    nodes = []
    initializers = []
    inputs = [
        helper.make_tensor_value_info("input_0", TensorProto.FLOAT, [batch_size, seq_len, hidden_size]),
    ]

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

    if include_rope:
        # --- On-the-fly RoPE computation (shared cos/sin) ---
        nodes.extend(_on_the_fly_rope_nodes("rope", head_dim, include_expand=include_expand_in_inv_freq))
        initializers.extend(
            _on_the_fly_rope_initializers(
                "rope",
                head_dim,
                batch_size=batch_size,
                include_expand=include_expand_in_inv_freq,
                inv_freq_as_graph_input=inv_freq_as_graph_input,
            )
        )
        inputs.append(
            helper.make_tensor_value_info("position_ids", TensorProto.INT64, [batch_size, seq_len]),
        )
        if inv_freq_as_graph_input:
            inputs.append(
                helper.make_tensor_value_info("rope_inv_freq", TensorProto.FLOAT, [1, head_dim // 2, 1]),
            )

        # --- Apply RoPE to Q ---
        nodes.extend(_rotate_half_nodes("q_rope", "q_transposed", "q_rope_out", "rope_cos_out", "rope_sin_out"))
        initializers.extend(_rotate_half_initializers("q_rope"))

        # --- Apply RoPE to K ---
        nodes.extend(_rotate_half_nodes("k_rope", "k_transposed", "k_rope_out", "rope_cos_out", "rope_sin_out"))
        initializers.extend(_rotate_half_initializers("k_rope"))

        q_for_attn = "q_rope_out"
        k_for_attn = "k_rope_out"
    else:
        q_for_attn = "q_transposed"
        k_for_attn = "k_transposed"

    # --- V reshape + transpose ---
    nodes.append(helper.make_node("Reshape", ["v_proj", "k_shape"], ["v_reshaped"], "v_reshape"))
    nodes.append(helper.make_node("Transpose", ["v_reshaped"], ["v_transposed"], "v_transpose", perm=[0, 2, 1, 3]))

    # --- Simplified attention: QK^T -> Softmax -> *V ---
    nodes.append(helper.make_node("Transpose", [k_for_attn], ["k_T"], "k_transpose_for_matmul", perm=[0, 1, 3, 2]))
    nodes.append(helper.make_node("MatMul", [q_for_attn, "k_T"], ["qk_scores"], "qk_matmul"))
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
        inputs,
        [
            helper.make_tensor_value_info("output_0", TensorProto.FLOAT, [batch_size, seq_len, hidden_size]),
        ],
        initializers,
    )

    model = helper.make_model(graph)
    model.ir_version = 7
    model.opset_import[0].version = 17
    return model
