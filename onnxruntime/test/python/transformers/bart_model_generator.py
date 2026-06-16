# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.  See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

"""Generator for synthetic BART SDPA attention ONNX graphs used in fusion tests.

This module reproduces the SDPA attention pattern emitted by HuggingFace
Transformers >= 4.49 when exporting BART models, so that the
FusionBartAttention pass can be exercised without a real model checkpoint.
"""

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


def create_bart_attention_sdpa(hidden_size: int = 16, num_heads: int = 4, with_mask: bool = True) -> onnx.ModelProto:
    """Create a minimal BART SDPA attention graph for fusion testing.

    The graph reproduces the self-attention subgraph exported by HuggingFace
    Transformers >= 4.49 for BART, including:
      - Pre-LayerNorm on the input
      - Q/K/V linear projections (MatMul + Add + Reshape + Transpose)
      - SDPA-specific K^T chain (Reshape -> Transpose(0,2,1) -> Reshape)
      - Separate Q and K scaling (Mul by 1/sqrt(head_dim))
      - QK MatMul, optional mask Add, Softmax, IsNaN/Where NaN guard
      - Attention * V MatMul
      - Output projection (MatMul + Add) with residual Add
      - Final LayerNormalization as the fusion anchor

    Args:
        hidden_size: Total hidden dimension (must be divisible by num_heads).
        num_heads: Number of attention heads.
        with_mask: If True, include an additive float attention mask input.

    Returns:
        An onnx.ModelProto representing the attention subgraph.
    """
    assert hidden_size % num_heads == 0, "hidden_size must be divisible by num_heads"

    head_dim = hidden_size // num_heads
    batch = 1
    seq = 8
    sqrt_scale = float(1.0 / (head_dim**0.5))

    # ------------------------------------------------------------------
    # Initializers (weights and shape constants)
    # ------------------------------------------------------------------
    np.random.seed(42)

    ln_weight = numpy_helper.from_array(np.ones(hidden_size, dtype=np.float32), "ln_weight")
    ln_bias = numpy_helper.from_array(np.zeros(hidden_size, dtype=np.float32), "ln_bias")

    q_weight = numpy_helper.from_array(np.random.randn(hidden_size, hidden_size).astype(np.float32), "q_weight")
    q_bias = numpy_helper.from_array(np.random.randn(hidden_size).astype(np.float32), "q_bias")
    k_weight = numpy_helper.from_array(np.random.randn(hidden_size, hidden_size).astype(np.float32), "k_weight")
    k_bias = numpy_helper.from_array(np.random.randn(hidden_size).astype(np.float32), "k_bias")
    v_weight = numpy_helper.from_array(np.random.randn(hidden_size, hidden_size).astype(np.float32), "v_weight")
    v_bias = numpy_helper.from_array(np.random.randn(hidden_size).astype(np.float32), "v_bias")

    out_weight = numpy_helper.from_array(np.random.randn(hidden_size, hidden_size).astype(np.float32), "out_weight")
    out_bias = numpy_helper.from_array(np.random.randn(hidden_size).astype(np.float32), "out_bias")

    ln2_weight = numpy_helper.from_array(np.ones(hidden_size, dtype=np.float32), "ln2_weight")
    ln2_bias = numpy_helper.from_array(np.zeros(hidden_size, dtype=np.float32), "ln2_bias")

    # Shape constants used by Reshape nodes.
    # Q/K/V projection reshape: [batch, seq, num_heads, head_dim]
    shape_qkv_4d = numpy_helper.from_array(np.array([batch, seq, num_heads, head_dim], dtype=np.int64), "shape_qkv_4d")
    # K^T chain — first Reshape merges batch and num_heads dims:
    #   [batch, num_heads, seq, head_dim] -> [batch*num_heads, seq, head_dim]
    shape_k_3d = numpy_helper.from_array(np.array([batch * num_heads, seq, head_dim], dtype=np.int64), "shape_k_3d")
    # K^T chain — second Reshape expands back to 4-D with transposed inner dims:
    #   [batch*num_heads, head_dim, seq] -> [batch, num_heads, head_dim, seq]
    shape_kt_4d = numpy_helper.from_array(np.array([batch, num_heads, head_dim, seq], dtype=np.int64), "shape_kt_4d")
    # Output reshape: [batch, seq, hidden_size]
    shape_output = numpy_helper.from_array(np.array([batch, seq, hidden_size], dtype=np.int64), "shape_output")

    # Scalar attention scale: 1 / sqrt(head_dim)
    scale_val = numpy_helper.from_array(np.array(sqrt_scale, dtype=np.float32), "sqrt_scale")

    # Constant used by the NaN guard Where node.
    zero_val = numpy_helper.from_array(np.array(0.0, dtype=np.float32), "zero_constant")

    # Large negative value for masked positions in attention mask.
    neg_inf_val = numpy_helper.from_array(np.array(-1e9, dtype=np.float32), "neg_inf_constant")

    initializers = [
        ln_weight,
        ln_bias,
        q_weight,
        q_bias,
        k_weight,
        k_bias,
        v_weight,
        v_bias,
        out_weight,
        out_bias,
        ln2_weight,
        ln2_bias,
        shape_qkv_4d,
        shape_k_3d,
        shape_kt_4d,
        shape_output,
        scale_val,
        zero_val,
        neg_inf_val,
    ]

    # ------------------------------------------------------------------
    # Graph inputs
    # ------------------------------------------------------------------
    input_1 = helper.make_tensor_value_info("input_1", TensorProto.FLOAT, [batch, seq, hidden_size])
    graph_inputs = [input_1]
    if with_mask:
        # Boolean mask input — the Where node converts it to a float mask,
        # matching the mask_nodes_bart pattern in fusion_bart_attention.py.
        attention_mask_bool = helper.make_tensor_value_info(
            "attention_mask_bool", TensorProto.BOOL, [batch, 1, seq, seq]
        )
        graph_inputs.append(attention_mask_bool)

    # ------------------------------------------------------------------
    # Graph output
    # ------------------------------------------------------------------
    graph_output = helper.make_tensor_value_info("output", TensorProto.FLOAT, [batch, seq, hidden_size])

    # ------------------------------------------------------------------
    # Nodes
    # ------------------------------------------------------------------
    nodes = []

    # 1. Pre-LayerNorm on the raw input.
    nodes.append(
        helper.make_node(
            "LayerNormalization",
            ["input_1", "ln_weight", "ln_bias"],
            ["layer_norm_out"],
            "layer_norm",
            axis=-1,
            epsilon=1e-5,
        )
    )

    # 2. Q projection: MatMul -> Add -> Reshape -> Transpose(0,2,1,3)
    nodes.append(helper.make_node("MatMul", ["layer_norm_out", "q_weight"], ["q_matmul_out"], "q_matmul"))
    nodes.append(helper.make_node("Add", ["q_matmul_out", "q_bias"], ["q_add_out"], "q_add"))
    nodes.append(helper.make_node("Reshape", ["q_add_out", "shape_qkv_4d"], ["q_reshape_out"], "q_reshape"))
    nodes.append(helper.make_node("Transpose", ["q_reshape_out"], ["q_transposed"], "q_transpose", perm=[0, 2, 1, 3]))

    # 3. K projection: MatMul -> Add -> Reshape -> Transpose(0,2,1,3)
    nodes.append(helper.make_node("MatMul", ["layer_norm_out", "k_weight"], ["k_matmul_out"], "k_matmul"))
    nodes.append(helper.make_node("Add", ["k_matmul_out", "k_bias"], ["k_add_out"], "k_add"))
    nodes.append(helper.make_node("Reshape", ["k_add_out", "shape_qkv_4d"], ["k_reshape_out"], "k_reshape"))
    nodes.append(helper.make_node("Transpose", ["k_reshape_out"], ["k_transposed"], "k_transpose", perm=[0, 2, 1, 3]))

    # 4. V projection: MatMul -> Add -> Reshape -> Transpose(0,2,1,3)
    nodes.append(helper.make_node("MatMul", ["layer_norm_out", "v_weight"], ["v_matmul_out"], "v_matmul"))
    nodes.append(helper.make_node("Add", ["v_matmul_out", "v_bias"], ["v_add_out"], "v_add"))
    nodes.append(helper.make_node("Reshape", ["v_add_out", "shape_qkv_4d"], ["v_reshape_out"], "v_reshape"))
    nodes.append(helper.make_node("Transpose", ["v_reshape_out"], ["v_transposed"], "v_transpose", perm=[0, 2, 1, 3]))

    # 5. SDPA-specific K^T chain.
    #
    #   k_transposed  [batch, num_heads, seq, head_dim]
    #     -> Reshape  [batch*num_heads, seq, head_dim]         (k_3d)
    #     -> Transpose(0,2,1)  [batch*num_heads, head_dim, seq] (k_3d_t)
    #     -> Reshape  [batch, num_heads, head_dim, seq]         (k_4d_t)
    #
    # The fusion pattern (k_nodes_sdpa) matches the data path:
    #   Mul <- Reshape <- Transpose <- Reshape <- Transpose <- Reshape <- Add <- MatMul
    # The two Reshapes here (k_3d, k_4d_t) are nodes 3 and 5 in that chain
    # (counting from the Mul), preceded by the initial K projection Reshape.
    nodes.append(helper.make_node("Reshape", ["k_transposed", "shape_k_3d"], ["k_3d"], "k_reshape_3d"))
    nodes.append(helper.make_node("Transpose", ["k_3d"], ["k_3d_t"], "k_transpose_3d", perm=[0, 2, 1]))
    nodes.append(helper.make_node("Reshape", ["k_3d_t", "shape_kt_4d"], ["k_4d_t"], "k_reshape_4d"))

    # 6. Separate Q and K scaling by 1/sqrt(head_dim).
    nodes.append(helper.make_node("Mul", ["q_transposed", "sqrt_scale"], ["q_scaled"], "q_scale"))
    nodes.append(helper.make_node("Mul", ["k_4d_t", "sqrt_scale"], ["k_scaled"], "k_scale"))

    # 7. QK attention scores.
    nodes.append(helper.make_node("MatMul", ["q_scaled", "k_scaled"], ["qk_matmul_out"], "qk_matmul"))

    # 8. Optional additive attention mask.
    #    In BART, the boolean mask is converted to a float mask via
    #    Where(condition, 0.0, -1e9) and then added to QK scores.
    #    The fusion code (mask_nodes_bart) matches: add_qk input[1] -> Where.
    if with_mask:
        nodes.append(
            helper.make_node(
                "Where",
                ["attention_mask_bool", "zero_constant", "neg_inf_constant"],
                ["attention_mask_float"],
                "mask_where",
            )
        )
        nodes.append(helper.make_node("Add", ["qk_matmul_out", "attention_mask_float"], ["qk_masked"], "qk_mask_add"))
        softmax_input = "qk_masked"
    else:
        softmax_input = "qk_matmul_out"

    # 9. Softmax over the last axis.
    nodes.append(helper.make_node("Softmax", [softmax_input], ["softmax_out"], "softmax", axis=-1))

    # 10. NaN guard: Where(IsNaN(softmax), 0, softmax).
    #     Where inputs: [condition, value_if_true, value_if_false]
    #     softmax_out is at index 2 (value_if_false), matching the fusion
    #     pattern qk_nodes_sdpa_{no,with}_mask which follows input[2].
    nodes.append(helper.make_node("IsNaN", ["softmax_out"], ["is_nan"], "isnan"))
    nodes.append(helper.make_node("Where", ["is_nan", "zero_constant", "softmax_out"], ["nan_guarded"], "nan_guard"))

    # 11. Attention output: NaN-guarded weights * V.
    nodes.append(helper.make_node("MatMul", ["nan_guarded", "v_transposed"], ["attn_v_out"], "attn_v_matmul"))

    # 12. Reshape attention output back to [batch, seq, hidden_size].
    nodes.append(
        helper.make_node("Transpose", ["attn_v_out"], ["attn_transposed"], "attn_transpose", perm=[0, 2, 1, 3])
    )
    nodes.append(helper.make_node("Reshape", ["attn_transposed", "shape_output"], ["attn_reshaped"], "attn_reshape"))

    # 13. Output projection.
    nodes.append(helper.make_node("MatMul", ["attn_reshaped", "out_weight"], ["out_matmul_out"], "out_matmul"))
    nodes.append(helper.make_node("Add", ["out_matmul_out", "out_bias"], ["out_add_out"], "out_add"))

    # 14. Residual connection.
    #
    #   We use layer_norm_out (a node output) rather than input_1 (a graph
    #   input) so that the fusion code can resolve root_input via
    #   output_name_to_node.  The first LayerNorm output also has Q/K/V
    #   MatMuls as direct children, which satisfies the fusion's heuristic
    #   for confirming the true attention root (lines 97-104 of
    #   fusion_bart_attention.py).
    nodes.append(helper.make_node("Add", ["layer_norm_out", "out_add_out"], ["residual_out"], "residual_add"))

    # 15. Final LayerNormalization — the fusion anchor node.
    nodes.append(
        helper.make_node(
            "LayerNormalization",
            ["residual_out", "ln2_weight", "ln2_bias"],
            ["output"],
            "layer_norm_2",
            axis=-1,
            epsilon=1e-5,
        )
    )

    # ------------------------------------------------------------------
    # Assemble and return the model
    # ------------------------------------------------------------------
    graph = helper.make_graph(
        nodes,
        "bart_sdpa_attention",
        graph_inputs,
        [graph_output],
        initializers,
    )
    opset = helper.make_opsetid("", 18)
    model = helper.make_model(graph, opset_imports=[opset])
    model.ir_version = 9
    return model


if __name__ == "__main__":
    import os

    output_dir = os.path.dirname(__file__)

    model_with_mask = create_bart_attention_sdpa(hidden_size=16, num_heads=4, with_mask=True)
    path_with_mask = os.path.join(output_dir, "bart_sdpa_attention_with_mask.onnx")
    onnx.save(model_with_mask, path_with_mask)
    print(f"Saved: {path_with_mask}")

    model_no_mask = create_bart_attention_sdpa(hidden_size=16, num_heads=4, with_mask=False)
    path_no_mask = os.path.join(output_dir, "bart_sdpa_attention_no_mask.onnx")
    onnx.save(model_no_mask, path_no_mask)
    print(f"Saved: {path_no_mask}")
