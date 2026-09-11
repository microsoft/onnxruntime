# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.  See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

"""Synthetic ONNX graph generators for Diffusion Transformer (DiT) attention fusion tests.

DiT models (F5-TTS, etc.) use an attention pattern where:
  - Q, K, V are pre-computed (e.g., after RoPE) in BNSH format
  - K is pre-transposed to BNHS for the attention MatMul
  - A custom scalar scale (e.g., 100.0) is applied before Softmax
  - Optional Cast nodes (FP16↔FP32) wrap Softmax for mixed-precision
"""

import numpy as np
from onnx import TensorProto, helper, numpy_helper


def _float_tensor(name, shape, random=False):
    vals = np.random.uniform(0, 1, size=shape).astype(np.float32) if random else np.ones(shape, dtype=np.float32)
    return numpy_helper.from_array(vals, name)


def create_dit_attention(
    batch_size=2,
    seq_len=4,
    num_heads=4,
    head_dim=8,
    scale=100.0,
    use_fp16_casts=False,
):
    """Create a DiT attention subgraph that exercises FusionMultiHeadAttentionDiT.

    The generated graph models the F5-TTS DiT attention pattern:

        hidden_states -> Q/K/V projections -> Reshape -> Transpose -> (K pre-transpose)
        -> MatMul(Q, K^T) -> [Cast FP16->FP32] -> Mul(scale) -> Softmax
        -> [Cast FP32->FP16] -> MatMul(attn, V) -> Transpose -> Reshape -> output_projection

    Args:
        batch_size: batch size (e.g., 2 for classifier-free guidance).
        seq_len: sequence length.
        num_heads: number of attention heads.
        head_dim: dimension per head.
        scale: attention logit scale factor (e.g., 100.0).
        use_fp16_casts: if True, add Cast nodes around Softmax (simulates FP16 model).

    Returns:
        onnx.ModelProto: the generated model.
    """
    hidden_size = num_heads * head_dim

    nodes = []
    initializers = []

    inputs = [
        helper.make_tensor_value_info("input_0", TensorProto.FLOAT, [batch_size, seq_len, hidden_size]),
    ]

    # --- Q projection: MatMul -> Reshape(BSNH) -> Transpose(BNSH) ---
    nodes.append(helper.make_node("MatMul", ["input_0", "q_weight"], ["q_proj"], "q_matmul"))
    initializers.append(_float_tensor("q_weight", [hidden_size, hidden_size], random=True))

    q_shape = [batch_size, seq_len, num_heads, head_dim]
    nodes.append(helper.make_node("Reshape", ["q_proj", "q_shape"], ["q_reshaped"], "q_reshape"))
    initializers.append(helper.make_tensor("q_shape", TensorProto.INT64, [4], q_shape))

    nodes.append(helper.make_node("Transpose", ["q_reshaped"], ["q_bnsh"], "q_transpose", perm=[0, 2, 1, 3]))

    # --- K projection: MatMul -> Reshape(BSNH) -> Transpose(BNSH) -> Transpose(BNHS, pre-transpose) ---
    nodes.append(helper.make_node("MatMul", ["input_0", "k_weight"], ["k_proj"], "k_matmul"))
    initializers.append(_float_tensor("k_weight", [hidden_size, hidden_size], random=True))

    k_shape = [batch_size, seq_len, num_heads, head_dim]
    nodes.append(helper.make_node("Reshape", ["k_proj", "k_shape"], ["k_reshaped"], "k_reshape"))
    initializers.append(helper.make_tensor("k_shape", TensorProto.INT64, [4], k_shape))

    nodes.append(helper.make_node("Transpose", ["k_reshaped"], ["k_bnsh"], "k_transpose", perm=[0, 2, 1, 3]))

    # Pre-transpose K: BNSH -> BNHS (this is the optimization done in DiT models)
    nodes.append(helper.make_node("Transpose", ["k_bnsh"], ["k_bnhs"], "k_pre_transpose", perm=[0, 1, 3, 2]))

    # --- V projection: MatMul -> Reshape(BSNH) -> Transpose(BNSH) ---
    nodes.append(helper.make_node("MatMul", ["input_0", "v_weight"], ["v_proj"], "v_matmul"))
    initializers.append(_float_tensor("v_weight", [hidden_size, hidden_size], random=True))

    v_shape = [batch_size, seq_len, num_heads, head_dim]
    nodes.append(helper.make_node("Reshape", ["v_proj", "v_shape"], ["v_reshaped"], "v_reshape"))
    initializers.append(helper.make_tensor("v_shape", TensorProto.INT64, [4], v_shape))

    nodes.append(helper.make_node("Transpose", ["v_reshaped"], ["v_bnsh"], "v_transpose", perm=[0, 2, 1, 3]))

    # --- Attention: MatMul(Q, K^T) -> [Cast] -> Mul(scale) -> Softmax -> [Cast] -> MatMul(attn, V) ---
    # QK^T: [B, N, S, H] @ [B, N, H, S] -> [B, N, S, S]
    nodes.append(helper.make_node("MatMul", ["q_bnsh", "k_bnhs"], ["qk_scores"], "qk_matmul"))

    if use_fp16_casts:
        # Cast QK scores FP16 -> FP32 (simulating FP16 model needing FP32 Softmax)
        nodes.append(helper.make_node("Cast", ["qk_scores"], ["qk_scores_fp32"], "cast_to_fp32", to=1))
        mul_input = "qk_scores_fp32"
    else:
        mul_input = "qk_scores"

    # Mul by custom scale
    initializers.append(helper.make_tensor("attn_scale", TensorProto.FLOAT, [], [scale]))
    nodes.append(helper.make_node("Mul", [mul_input, "attn_scale"], ["qk_scaled"], "qk_scale"))

    # Softmax
    nodes.append(helper.make_node("Softmax", ["qk_scaled"], ["attn_weights"], "softmax", axis=-1))

    if use_fp16_casts:
        # Cast attention weights FP32 -> FP16
        nodes.append(helper.make_node("Cast", ["attn_weights"], ["attn_weights_fp16"], "cast_to_fp16", to=10))
        attn_matmul_input = "attn_weights_fp16"
        # Cast V to FP16 so MatMul inputs are type-consistent (both FP16)
        nodes.append(helper.make_node("Cast", ["v_bnsh"], ["v_bnsh_fp16"], "cast_v_to_fp16", to=10))
        v_matmul_input = "v_bnsh_fp16"
    else:
        attn_matmul_input = "attn_weights"
        v_matmul_input = "v_bnsh"

    # Attention @ V: [B, N, S, S] @ [B, N, S, H] -> [B, N, S, H]
    nodes.append(helper.make_node("MatMul", [attn_matmul_input, v_matmul_input], ["attn_out"], "attn_v_matmul"))

    # --- Output: Transpose(BNSH -> BSNH) -> Reshape(BSD) -> output projection ---
    nodes.append(helper.make_node("Transpose", ["attn_out"], ["attn_transposed"], "attn_transpose", perm=[0, 2, 1, 3]))

    out_shape = [batch_size, seq_len, hidden_size]
    nodes.append(helper.make_node("Reshape", ["attn_transposed", "out_shape"], ["attn_flat"], "attn_reshape"))
    initializers.append(helper.make_tensor("out_shape", TensorProto.INT64, [3], out_shape))

    if use_fp16_casts:
        # Cast attention output back to FP32 so the output projection MatMul
        # has type-consistent inputs with FP32 o_weight.
        nodes.append(helper.make_node("Cast", ["attn_flat"], ["attn_flat_fp32"], "cast_attn_flat_to_fp32", to=1))
        o_matmul_input = "attn_flat_fp32"
    else:
        o_matmul_input = "attn_flat"

    # Output projection
    nodes.append(helper.make_node("MatMul", [o_matmul_input, "o_weight"], ["output_0"], "o_matmul"))
    initializers.append(_float_tensor("o_weight", [hidden_size, hidden_size], random=True))

    # --- Graph definition ---
    graph = helper.make_graph(
        nodes,
        "dit_attention",
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


def create_dit_attention_no_k_transpose(
    batch_size=2,
    seq_len=4,
    num_heads=4,
    head_dim=8,
    scale=100.0,
):
    """Create a DiT attention graph where K is directly in BNHS format (no explicit Transpose).

    This tests the path where the fusion needs to add its own Transpose for K.
    Uses graph inputs for Q/K/V in the expected 4D formats.

    Returns:
        onnx.ModelProto: the generated model.
    """
    hidden_size = num_heads * head_dim

    nodes = []
    initializers = []

    # Use 4D inputs directly (as if Q/K/V come from RoPE or other external computation)
    inputs = [
        helper.make_tensor_value_info("q_bnsh", TensorProto.FLOAT, [batch_size, num_heads, seq_len, head_dim]),
        helper.make_tensor_value_info("k_bnhs", TensorProto.FLOAT, [batch_size, num_heads, head_dim, seq_len]),
        helper.make_tensor_value_info("v_bnsh", TensorProto.FLOAT, [batch_size, num_heads, seq_len, head_dim]),
    ]

    # QK^T: [B, N, S, H] @ [B, N, H, S] -> [B, N, S, S]
    nodes.append(helper.make_node("MatMul", ["q_bnsh", "k_bnhs"], ["qk_scores"], "qk_matmul"))

    # Mul by scale
    initializers.append(helper.make_tensor("attn_scale", TensorProto.FLOAT, [], [scale]))
    nodes.append(helper.make_node("Mul", ["qk_scores", "attn_scale"], ["qk_scaled"], "qk_scale"))

    # Softmax
    nodes.append(helper.make_node("Softmax", ["qk_scaled"], ["attn_weights"], "softmax", axis=-1))

    # Attention @ V
    nodes.append(helper.make_node("MatMul", ["attn_weights", "v_bnsh"], ["attn_out"], "attn_v_matmul"))

    # Transpose + Reshape
    nodes.append(helper.make_node("Transpose", ["attn_out"], ["attn_transposed"], "attn_transpose", perm=[0, 2, 1, 3]))
    out_shape = [batch_size, seq_len, hidden_size]
    nodes.append(helper.make_node("Reshape", ["attn_transposed", "out_shape"], ["output_0"], "attn_reshape"))
    initializers.append(helper.make_tensor("out_shape", TensorProto.INT64, [3], out_shape))

    graph = helper.make_graph(
        nodes,
        "dit_attention_no_k_transpose",
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
