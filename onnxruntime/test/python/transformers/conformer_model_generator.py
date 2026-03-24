# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.  See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------


import numpy as np
import onnx
from bert_model_generator import float_tensor
from onnx import TensorProto, helper, numpy_helper

# Minimum non-zero value used for the QK attention bias initializer in test models.
# A zero bias would be eliminated by ORT's basic constant folding (it removes Add(x, 0)
# as a no-op), breaking the fusion patterns that expect an Add node before Softmax.
_NON_ZERO_QK_BIAS = 1e-4


# Adapted from bert_model_generator.py
def get_tensor_and_weight(name: str, shape: list[int], random=False, zeros=False):
    low = 0.0
    high = 1.0
    total_elements = 1
    for x in shape:
        total_elements *= x
    weights = (
        [np.random.uniform(low, high) for _ in range(total_elements)]
        if random
        else [0.0] * total_elements
        if zeros
        else [1.0] * total_elements
    )
    return helper.make_tensor(name, TensorProto.FLOAT, shape, weights), weights


def create_conformer_attention(
    hidden_size=512,
    num_heads=8,
    epsilon=0.000009999999747378752,
    add_before_layernorm=False,
    fused=False,
):
    # Get head size and ensure head size is an integer
    assert hidden_size % num_heads == 0
    head_size = hidden_size // num_heads

    # Construct input and output nodes
    inputs = [
        helper.make_tensor_value_info("input_0", TensorProto.FLOAT, ["batch_size", 8, 512]),
        helper.make_tensor_value_info("input_1", TensorProto.FLOAT, ["batch_size", 8, 512]),
        helper.make_tensor_value_info("inp_cache_k", TensorProto.FLOAT, [24, "batch_size", 8, 72, head_size]),
        helper.make_tensor_value_info("inp_cache_v", TensorProto.FLOAT, [24, "batch_size", 8, 72, head_size]),
    ]
    outputs = [
        helper.make_tensor_value_info("output_0", TensorProto.FLOAT, ["batch_size", 8, hidden_size]),
        helper.make_tensor_value_info("output_1", TensorProto.FLOAT, ["batch_size", 8, 512]),
        helper.make_tensor_value_info("oup_cache_k", TensorProto.FLOAT, ["batch_size", 8, 80, 64]),
        helper.make_tensor_value_info("oup_cache_v", TensorProto.FLOAT, ["batch_size", 8, 80, 64]),
    ]
    nodes = []

    # Create layernorm (Add + LayerNorm or SkipLayerNorm)
    if add_before_layernorm:
        nodes.extend(
            [
                helper.make_node(
                    "Add", ["input_0", "input_1"], ["layernorm_output_to_skiplayernorm"], "add_before_layernorm"
                ),
                helper.make_node(
                    "LayerNormalization",
                    ["layernorm_output_to_skiplayernorm", "layernorm_weight", "layernorm_bias"],
                    ["layernorm_add_output_to_matmul"],
                    "layernorm",
                    epsilon=epsilon,
                ),
            ]
        )
    else:
        nodes.append(
            helper.make_node(
                "SkipLayerNormalization",
                ["input_0", "input_1", "layernorm_weight", "layernorm_bias"],
                ["layernorm_add_output_to_matmul", "", "", "layernorm_add_output_to_skiplayernorm"],
                "skiplayernorm",
                domain="com.microsoft",
                epsilon=epsilon,
            )
        )

    if fused:
        fused_q_nodes = [
            helper.make_node(
                "MatMul",
                ["layernorm_add_output_to_matmul", "q_weight"],
                ["q_matmul_output"],
                "q_path_matmul",
            ),
            helper.make_node("Add", ["q_bias", "q_matmul_output"], ["q_add_output"], "q_path_add"),
            helper.make_node(
                "Reshape", ["q_add_output", "k_attn_heads_output"], ["q_4d_bsnh"], "q_reshape_to_4d", allowzero=0
            ),
            helper.make_node("Transpose", ["q_4d_bsnh"], ["q_4d_bnsh"], "q_transpose_to_bnsh", perm=[0, 2, 1, 3]),
            helper.make_node(
                "Div",
                ["q_4d_bnsh", "q_scale"],
                ["q_div_output"],
                "q_div_by_sqrt_head_size",
            ),
        ]
        nodes.extend(fused_q_nodes)
        nodes.extend(
            [
                helper.make_node(
                    "MatMul",
                    ["layernorm_add_output_to_matmul", "k_weight"],
                    ["k_matmul_output"],
                    "k_path_matmul",
                ),
                helper.make_node(
                    "MatMul",
                    ["layernorm_add_output_to_matmul", "v_weight"],
                    ["v_matmul_output"],
                    "v_path_matmul",
                ),
                helper.make_node(
                    "Reshape", ["q_div_output", "position_embed_output"], ["reshape_pos_emb"], "r_pos_emb", allowzero=0
                ),
                helper.make_node(
                    "Transpose", ["reshape_pos_emb"], ["transpose_reshape_pos_emb"], "p_transpose", perm=[1, 0, 2]
                ),
                helper.make_node(
                    "MatMul",
                    ["transpose_reshape_pos_emb", "transpose_reshape_pos_emb"],
                    ["pos_matmul"],
                    "pos_embed_matmul",
                ),
                helper.make_node(
                    "Transpose", ["pos_matmul"], ["transpose_pos_matmul"], "p_matmul_transpose", perm=[1, 0, 2]
                ),
                helper.make_node(
                    "Reshape",
                    ["transpose_pos_matmul", "position_embed_output"],
                    ["reshape_position_emb"],
                    "final_reshape_pos_emb",
                    allowzero=0,
                ),
                helper.make_node(
                    "MultiHeadAttention",
                    [
                        "q_matmul_output",
                        "k_matmul_output",
                        "v_matmul_output",
                        "Attention_0_qkv_bias",
                        "",
                        "reshape_position_emb",
                        "gather_past_k_output",
                        "gather_past_v_output",
                    ],
                    ["attn_output", "oup_cache_k", "oup_cache_v"],
                    "Attention_0",
                    domain="com.microsoft",
                    num_heads=num_heads,
                ),
            ]
        )
        # Create nodes used with qkv concats, reshapes, and transposes
        nodes.extend(
            [
                helper.make_node("Shape", ["layernorm_add_output_to_matmul"], ["shape_output"], "shape", start=0),
                helper.make_node("Gather", ["shape_output", "idx_0"], ["gather_0_output"], "gather_0", axis=0),
                helper.make_node(
                    "Mul",
                    ["gather_0_output", "num_heads_int"],
                    ["mul_attn_heads_output"],
                    "mul_num_heads",
                ),
                helper.make_node(
                    "Unsqueeze",
                    ["mul_attn_heads_output", "unsqueeze_axes_input"],
                    ["unsqueeze_position_embed"],
                    "unsqueeze_position_embed",
                ),
                helper.make_node(
                    "Concat",
                    ["unsqueeze_position_embed", "neg_one", "head_size"],
                    ["position_embed_output"],
                    "position_embed_concat_output",
                    axis=0,
                ),
                helper.make_node(
                    "Unsqueeze",
                    ["gather_0_output", "unsqueeze_axes_input"],
                    ["unsqueeze_attn_heads_output"],
                    "unsqueeze_num_heads",
                ),
                helper.make_node(
                    "Concat",
                    ["unsqueeze_attn_heads_output", "neg_one", "head_size", "q_bsnh_reshape"],
                    ["k_attn_heads_output"],
                    "k_num_heads",
                    axis=0,
                ),
            ]
        )

        nodes.extend(
            [
                helper.make_node("Gather", ["inp_cache_v", "idx_0"], ["gather_past_v_output"], "gather_past_v", axis=0),
                helper.make_node("Gather", ["inp_cache_k", "idx_0"], ["gather_past_k_output"], "gather_past_k", axis=0),
            ]
        )
    else:
        # Create nodes for Q/K/V paths
        q_nodes = [
            helper.make_node(
                "MatMul", ["layernorm_add_output_to_matmul", "q_weight"], ["q_matmul_output"], "q_path_matmul"
            ),
            helper.make_node("Add", ["q_bias", "q_matmul_output"], ["q_add_output"], "q_path_add"),
            helper.make_node("Reshape", ["q_add_output", "q_attn_heads_output"], ["q_4d_bsnh"], "q_reshape_to_4d"),
            helper.make_node("Transpose", ["q_4d_bsnh"], ["q_4d_bnsh"], "q_transpose_to_bnsh", perm=[0, 2, 1, 3]),
            helper.make_node(
                "Div",
                ["q_4d_bnsh", "q_scale"],
                ["q_div_output"],
                "q_div_by_sqrt_head_size",
            ),
        ]
        k_nodes = [
            helper.make_node(
                "MatMul",
                ["layernorm_add_output_to_matmul", "k_weight"],
                ["k_matmul_output"],
                "k_path_matmul",
            ),
            helper.make_node("Add", ["k_bias", "k_matmul_output"], ["k_add_output"], "k_path_add"),
            helper.make_node("Reshape", ["k_add_output", "k_attn_heads_output"], ["k_4d_bsnh"], "k_reshape_to_4d"),
            helper.make_node("Transpose", ["k_4d_bsnh"], ["k_4d_bnsh"], "k_transpose_to_bnsh", perm=[0, 2, 1, 3]),
            helper.make_node(
                "Concat",
                ["gather_past_k_output", "k_4d_bnsh"],
                ["oup_cache_k"],
                "concat_past_k_and_curr_k",
                axis=2,
            ),
            helper.make_node(
                "Transpose",
                ["oup_cache_k"],
                ["k_output_transpose"],
                "k_transpose_last_two_dims",
                perm=[0, 1, 3, 2],
            ),
        ]
        v_nodes = [
            helper.make_node(
                "MatMul",
                ["layernorm_add_output_to_matmul", "v_weight"],
                ["v_matmul_output"],
                "v_path_matmul",
            ),
            helper.make_node("Add", ["v_bias", "v_matmul_output"], ["v_add_output"], "v_path_add"),
            helper.make_node("Reshape", ["v_add_output", "v_attn_heads_output"], ["v_4d_bsnh"], "v_reshape_to_4d"),
            helper.make_node("Transpose", ["v_4d_bsnh"], ["v_4d_bnsh"], "v_transpose_to_bnsh", perm=[0, 2, 1, 3]),
            helper.make_node(
                "Concat",
                ["gather_past_v_output", "v_4d_bnsh"],
                ["oup_cache_v"],
                "concat_past_v_and_curr_v",
                axis=2,
            ),
        ]
        pos_embed = [
            helper.make_node("Reshape", ["q_div_output", "position_embed_output"], ["reshape_pos_emb"], "r_pos_emb"),
            helper.make_node(
                "Transpose", ["reshape_pos_emb"], ["transpose_reshape_pos_emb"], "p_transpose", perm=[1, 0, 2]
            ),
            helper.make_node(
                "MatMul",
                ["transpose_reshape_pos_emb", "transpose_reshape_pos_emb"],
                ["pos_matmul"],
                "pos_embed_matmul",
            ),
            helper.make_node(
                "Transpose", ["pos_matmul"], ["transpose_pos_matmul"], "p_matmul_transpose", perm=[1, 0, 2]
            ),
            helper.make_node(
                "Reshape",
                ["transpose_pos_matmul", "position_embed_output"],
                ["reshape_position_emb"],
                "final_reshape_pos_emb",
            ),
        ]
        nodes.extend(q_nodes)
        nodes.extend(k_nodes)
        nodes.extend(v_nodes)
        nodes.extend(pos_embed)

        # Create nodes used with qkv concats, reshapes, and transposes
        nodes.extend(
            [
                helper.make_node("Shape", ["layernorm_add_output_to_matmul"], ["shape_output"], "shape", start=0),
                helper.make_node("Gather", ["shape_output", "idx_0"], ["gather_0_output"], "gather_0", axis=0),
                helper.make_node(
                    "Mul",
                    ["gather_0_output", "num_heads_int"],
                    ["mul_attn_heads_output"],
                    "mul_num_heads",
                ),
                helper.make_node(
                    "Unsqueeze",
                    ["mul_attn_heads_output", "unsqueeze_axes_input"],
                    ["unsqueeze_position_embed"],
                    "unsqueeze_position_embed",
                ),
                helper.make_node(
                    "Concat",
                    ["unsqueeze_position_embed", "neg_one", "head_size"],
                    ["position_embed_output"],
                    "position_embed_concat_output",
                    axis=0,
                ),
                helper.make_node(
                    "Unsqueeze",
                    ["gather_0_output", "unsqueeze_axes_input"],
                    ["unsqueeze_attn_heads_output"],
                    "unsqueeze_num_heads",
                ),
                helper.make_node(
                    "Concat",
                    ["unsqueeze_attn_heads_output", "neg_one", "head_size", "q_bsnh_reshape"],
                    ["q_attn_heads_output"],
                    "q_num_heads",
                    axis=0,
                ),
                helper.make_node(
                    "Concat",
                    ["unsqueeze_attn_heads_output", "neg_one", "head_size", "q_bsnh_reshape"],
                    ["k_attn_heads_output"],
                    "k_num_heads",
                    axis=0,
                ),
                helper.make_node(
                    "Concat",
                    ["unsqueeze_attn_heads_output", "neg_one", "head_size", "q_bsnh_reshape"],
                    ["v_attn_heads_output"],
                    "v_num_heads",
                    axis=0,
                ),
                helper.make_node(
                    "Concat",
                    ["unsqueeze_attn_heads_output", "neg_one", "head_size"],
                    ["bsd_format"],
                    axis=0,
                ),
                helper.make_node(
                    "Constant",
                    inputs=[],
                    outputs=["q_bsnh_reshape"],
                    value=numpy_helper.from_array(
                        np.array([0, 0, num_heads, head_size], dtype="int64"), name="const_tensor"
                    ),
                ),
            ]
        )

        nodes.extend(
            [
                helper.make_node("Gather", ["inp_cache_v", "idx_0"], ["gather_past_v_output"], "gather_past_v", axis=0),
                helper.make_node("Gather", ["inp_cache_k", "idx_0"], ["gather_past_k_output"], "gather_past_k", axis=0),
            ]
        )

        # Compute Q x K'
        nodes.extend(
            [
                helper.make_node(
                    "MatMul",
                    [
                        "q_div_output",
                        "k_output_transpose",
                    ],
                    ["qk_output"],
                    "matmul_qk",
                )
            ]
        )

        # Create nodes for computing softmax(Q x K') x V
        nodes.extend(
            [
                helper.make_node(
                    "Add",
                    [
                        "qk_output",
                        "reshape_position_emb",
                    ],
                    ["add_qk_output"],
                    "add_qk",
                ),
                helper.make_node(
                    "Softmax",
                    ["add_qk_output"],
                    ["softmax_output"],
                    "softmax_qk",
                    axis=2,
                ),
                helper.make_node(
                    "MatMul",
                    ["softmax_output", "oup_cache_v"],
                    ["qkv_output_(num_heads*batch_size,seq_len,head_size)"],
                    "matmul_qkv",
                ),
                helper.make_node(
                    "Transpose",
                    ["qkv_output_(num_heads*batch_size,seq_len,head_size)"],
                    ["qkv_bsnh"],
                    "transpose_bnsh_to_bsnh",
                    perm=[0, 2, 1, 3],
                ),
                helper.make_node("Reshape", ["qkv_bsnh", "bsd_format"], ["attn_output"], "qkv_bsd"),
            ]
        )

    # Create final nodes to conclude attention
    nodes.append(
        helper.make_node(
            "MatMul",
            ["attn_output", "matmul_after_attn_initializer"],
            ["matmul_after_attn_output"],
            "matmul_after_attn",
        ),
    )
    if not fused:
        next_sln_inputs = [
            "layernorm_add_output_to_skiplayernorm",
            "add_after_attn_output",
            "layernorm_weight",
            "layernorm_bias",
        ]
        nodes.extend(
            [
                helper.make_node(
                    "Add",
                    ["add_after_attn_initializer", "matmul_after_attn_output"],
                    ["add_after_attn_output"],
                    "add_after_attn",
                ),
                helper.make_node(
                    "SkipLayerNormalization",
                    next_sln_inputs,
                    ["output_0", "", "", "output_1"],
                    "next_skiplayernorm",
                    domain="com.microsoft",
                    epsilon=epsilon,
                ),
            ]
        )
    else:
        next_sln_inputs = [
            "matmul_after_attn_output",
            "layernorm_add_output_to_skiplayernorm",
            "layernorm_weight",
            "layernorm_bias",
            "add_after_attn_initializer",
        ]
        nodes.append(
            helper.make_node(
                "SkipLayerNormalization",
                next_sln_inputs,
                ["output_0", "", "", "output_1"],
                "SkipLayerNorm_AddBias_0",
                domain="com.microsoft",
                epsilon=epsilon,
            )
        )

    # Create initializers
    v_weight, v_weight_data = get_tensor_and_weight("v_weight", [hidden_size, hidden_size])
    v_bias, v_bias_data = get_tensor_and_weight("v_bias", [hidden_size])
    q_weight, q_weight_data = get_tensor_and_weight("q_weight", [hidden_size, hidden_size])
    q_bias, q_bias_data = get_tensor_and_weight("q_bias", [hidden_size])
    k_weight, k_weight_data = get_tensor_and_weight("k_weight", [hidden_size, hidden_size])
    k_bias, k_bias_data = get_tensor_and_weight("k_bias", [hidden_size])

    qkv_bias = helper.make_tensor(
        "Attention_0_qkv_bias",
        TensorProto.FLOAT,
        [3 * hidden_size],
        q_bias_data + k_bias_data + v_bias_data,
    )
    initializers = [
        float_tensor("layernorm_weight", [hidden_size]),
        float_tensor("layernorm_bias", [hidden_size]),
        float_tensor("matmul_after_attn_initializer", [hidden_size, hidden_size]),
        float_tensor("add_after_attn_initializer", [hidden_size]),
    ]

    # Add Q/K/V weight tensors as initializers
    if fused:
        initializers.extend([q_weight, k_weight, v_weight])
        initializers.extend([q_bias])
        initializers.append(qkv_bias)
        initializers.extend(
            [
                numpy_helper.from_array(np.array(num_heads, dtype="int64"), name="num_heads_int"),
                numpy_helper.from_array(np.array([head_size], dtype="int64"), name="head_size"),
                numpy_helper.from_array(np.array(1 / np.sqrt(head_size), dtype="float32"), name="q_scale"),
                numpy_helper.from_array(np.array(0, dtype="int64"), name="idx_0"),
                numpy_helper.from_array(np.array([-1], dtype="int64"), name="neg_one"),
                numpy_helper.from_array(np.array([0], dtype="int64"), name="unsqueeze_axes_input"),
                numpy_helper.from_array(np.array([0, 0, num_heads, head_size], dtype="int64"), name="q_bsnh_reshape"),
            ]
        )
    else:
        initializers.extend([q_weight, k_weight, v_weight])

        initializers.extend([q_bias, k_bias, v_bias])

        initializers.extend(
            [
                numpy_helper.from_array(np.array(num_heads, dtype="int64"), name="num_heads_int"),
                numpy_helper.from_array(np.array([num_heads], dtype="int64"), name="num_heads"),
                numpy_helper.from_array(np.array([head_size], dtype="int64"), name="head_size"),
                numpy_helper.from_array(np.array([hidden_size], dtype="int64"), name="hidden_size"),
                numpy_helper.from_array(np.array(1 / np.sqrt(head_size), dtype="float32"), name="q_scale"),
                numpy_helper.from_array(np.array(0, dtype="int64"), name="idx_0"),
                numpy_helper.from_array(np.array(1, dtype="int64"), name="idx_1"),
                numpy_helper.from_array(np.array([-1], dtype="int64"), name="neg_one"),
                numpy_helper.from_array(np.array([0], dtype="int64"), name="unsqueeze_axes_input"),
            ]
        )

    # Construct graph
    graph = helper.make_graph(nodes, "conformer_self_mha_graph", inputs, outputs, initializers, doc_string="conformer")
    opsetid = helper.make_opsetid("ai.onnx", min(onnx.defs.onnx_opset_version(), 16))
    return helper.make_model(graph, opset_imports=(opsetid,))


def create_conformer_attention_simple_bias(
    hidden_size=64,
    num_heads=4,
    epsilon=0.000009999999747378752,
):
    """
    Standard conformer attention where the QK add_bias is a plain initializer (no positional
    embedding computation). The extra_q_nodes match_parent_path will return None for both the
    conformer-transducer and Nemotron patterns, so fusion proceeds with extra_q_nodes=None.

    This is a regression test to verify that the fix restoring optional extra_q_nodes semantics
    works correctly: graphs that never had an auxiliary Q branch must still fuse.

    Q path:  MatMul -> Add(bias, matmul_out) -> Reshape -> Transpose([0,2,1,3]) -> Div -> matmul_qk
    K path:  MatMul -> Add(matmul_out, bias) -> Reshape -> Transpose([0,2,3,1]) -> matmul_qk
    V path:  MatMul -> Add(matmul_out, bias) -> Reshape -> Transpose([0,2,1,3]) -> matmul_qkv
    QK:      MatMul -> Add(qk_out, qk_bias_init) -> Softmax -> MatMul
    Output:  Transpose -> Reshape -> MatMul -> Add(bias, matmul) -> SkipLayerNorm
    """
    assert hidden_size % num_heads == 0
    head_size = hidden_size // num_heads

    inputs = [
        helper.make_tensor_value_info("input_0", TensorProto.FLOAT, ["batch_size", "seq_len", hidden_size]),
        helper.make_tensor_value_info("input_1", TensorProto.FLOAT, ["batch_size", "seq_len", hidden_size]),
    ]
    outputs = [
        helper.make_tensor_value_info("output_0", TensorProto.FLOAT, ["batch_size", "seq_len", hidden_size]),
        helper.make_tensor_value_info("output_1", TensorProto.FLOAT, ["batch_size", "seq_len", hidden_size]),
    ]
    nodes = []

    # SkipLayerNorm
    nodes.append(
        helper.make_node(
            "SkipLayerNormalization",
            ["input_0", "input_1", "ln_weight", "ln_bias"],
            ["ln_out", "", "", "ln_skip_out"],
            "skiplayernorm",
            domain="com.microsoft",
            epsilon=epsilon,
        )
    )

    # Q path: MatMul -> Add(bias[0], matmul[1]) -> Reshape -> Transpose -> Div
    nodes.extend(
        [
            helper.make_node("MatMul", ["ln_out", "q_weight"], ["q_matmul_out"], "q_matmul"),
            helper.make_node("Add", ["q_bias", "q_matmul_out"], ["q_add_out"], "q_add"),
            helper.make_node("Reshape", ["q_add_out", "qkv_reshape_shape"], ["q_4d_bsnh"], "q_reshape"),
            helper.make_node("Transpose", ["q_4d_bsnh"], ["q_4d_bnsh"], "q_transpose", perm=[0, 2, 1, 3]),
            helper.make_node("Div", ["q_4d_bnsh", "q_scale"], ["q_scaled"], "q_div"),
        ]
    )

    # K path: MatMul -> Add(matmul[0], bias[1]) -> Reshape -> Transpose (single, for K^T)
    nodes.extend(
        [
            helper.make_node("MatMul", ["ln_out", "k_weight"], ["k_matmul_out"], "k_matmul"),
            helper.make_node("Add", ["k_matmul_out", "k_bias"], ["k_add_out"], "k_add"),
            helper.make_node("Reshape", ["k_add_out", "qkv_reshape_shape"], ["k_4d_bsnh"], "k_reshape"),
            # perm=[0,2,3,1]: [B,S,H,D] -> [B,H,D,S] giving K^T for attention dot product
            helper.make_node("Transpose", ["k_4d_bsnh"], ["k_transposed"], "k_transpose", perm=[0, 2, 3, 1]),
        ]
    )

    # V path: MatMul -> Add(matmul[0], bias[1]) -> Reshape -> Transpose (BNSH)
    nodes.extend(
        [
            helper.make_node("MatMul", ["ln_out", "v_weight"], ["v_matmul_out"], "v_matmul"),
            helper.make_node("Add", ["v_matmul_out", "v_bias"], ["v_add_out"], "v_add"),
            helper.make_node("Reshape", ["v_add_out", "qkv_reshape_shape"], ["v_4d_bsnh"], "v_reshape"),
            helper.make_node("Transpose", ["v_4d_bsnh"], ["v_4d_bnsh"], "v_transpose", perm=[0, 2, 1, 3]),
        ]
    )

    # QK: MatMul -> Add(qk_out, simple_bias_init) -> Softmax -> MatMul
    # qk_bias is a plain initializer, so extra_q_nodes will be None.
    nodes.extend(
        [
            helper.make_node("MatMul", ["q_scaled", "k_transposed"], ["qk_out"], "matmul_qk"),
            helper.make_node("Add", ["qk_out", "qk_bias"], ["qk_add_out"], "add_qk"),
            helper.make_node("Softmax", ["qk_add_out"], ["softmax_out"], "softmax_qk", axis=3),
            helper.make_node("MatMul", ["softmax_out", "v_4d_bnsh"], ["qkv_bnsh"], "matmul_qkv"),
        ]
    )

    # Output: Transpose -> Reshape -> MatMul -> Add -> SkipLayerNorm
    nodes.extend(
        [
            helper.make_node("Transpose", ["qkv_bnsh"], ["qkv_bsnh"], "qkv_transpose", perm=[0, 2, 1, 3]),
            helper.make_node("Reshape", ["qkv_bsnh", "out_reshape_shape"], ["attn_out"], "out_reshape"),
            helper.make_node("MatMul", ["attn_out", "out_weight"], ["out_matmul"], "out_matmul"),
            helper.make_node("Add", ["out_bias", "out_matmul"], ["out_add"], "out_add"),
            helper.make_node(
                "SkipLayerNormalization",
                ["ln_skip_out", "out_add", "ln_weight", "ln_bias"],
                ["output_0", "", "", "output_1"],
                "next_skiplayernorm",
                domain="com.microsoft",
                epsilon=epsilon,
            ),
        ]
    )

    q_weight, _ = get_tensor_and_weight("q_weight", [hidden_size, hidden_size])
    k_weight, _ = get_tensor_and_weight("k_weight", [hidden_size, hidden_size])
    v_weight, _ = get_tensor_and_weight("v_weight", [hidden_size, hidden_size])

    initializers = [
        float_tensor("ln_weight", [hidden_size]),
        float_tensor("ln_bias", [hidden_size]),
        float_tensor("out_weight", [hidden_size, hidden_size]),
        float_tensor("out_bias", [hidden_size]),
        q_weight,
        k_weight,
        v_weight,
        numpy_helper.from_array(np.array([1.0] * hidden_size, dtype="float32"), name="q_bias"),
        numpy_helper.from_array(np.array([1.0] * hidden_size, dtype="float32"), name="k_bias"),
        numpy_helper.from_array(np.array([1.0] * hidden_size, dtype="float32"), name="v_bias"),
        # QK bias: a simple non-zero initializer so extra_q_nodes won't match any positional-embed pattern.
        # Non-zero so ORT's constant folding (which removes Add(x, 0)) doesn't eliminate this node.
        numpy_helper.from_array(np.array([_NON_ZERO_QK_BIAS], dtype="float32"), name="qk_bias"),
        numpy_helper.from_array(np.array(1.0 / np.sqrt(head_size), dtype="float32"), name="q_scale"),
        # Reshape shape [0, 0, num_heads, head_size] for Q/K/V
        numpy_helper.from_array(np.array([0, 0, num_heads, head_size], dtype="int64"), name="qkv_reshape_shape"),
        # Reshape shape [0, 0, hidden_size] for output
        numpy_helper.from_array(np.array([0, 0, hidden_size], dtype="int64"), name="out_reshape_shape"),
    ]

    graph = helper.make_graph(
        nodes, "conformer_simple_bias_graph", inputs, outputs, initializers, doc_string="conformer"
    )
    opsetid = helper.make_opsetid("ai.onnx", min(onnx.defs.onnx_opset_version(), 16))
    return helper.make_model(graph, opset_imports=(opsetid,))


def create_conformer_attention_no_add_kv(
    hidden_size=64,
    num_heads=4,
    epsilon=0.000009999999747378752,
):
    """
    Nemotron-like conformer attention model with no Add-bias nodes in the K and V paths,
    and a Q path that begins with Transpose→Add→Reshape→MatMul (no leading Div/Mul).
    The QKV output path also omits the trailing Add before the SkipLayerNorm.

    This exercises the following new fallback patterns:
      - QKV output: ["MatMul", "Reshape", "Transpose", "MatMul"] with [1, 0, 0, 0]
      - Q path:     ["Transpose", "Add", "Reshape", "MatMul"]     with [0, 0, 0, 0]
      - K path:     ["Transpose", "Reshape", "MatMul"]            with [1, 0, 0]
      - V path:     ["Transpose", "Reshape", "MatMul"]            with [1, 0, 0]
    """
    assert hidden_size % num_heads == 0
    head_size = hidden_size // num_heads

    inputs = [
        helper.make_tensor_value_info("input_0", TensorProto.FLOAT, ["batch_size", "seq_len", hidden_size]),
        helper.make_tensor_value_info("input_1", TensorProto.FLOAT, ["batch_size", "seq_len", hidden_size]),
    ]
    outputs = [
        helper.make_tensor_value_info("output_0", TensorProto.FLOAT, ["batch_size", "seq_len", hidden_size]),
        helper.make_tensor_value_info("output_1", TensorProto.FLOAT, ["batch_size", "seq_len", hidden_size]),
    ]
    nodes = []

    # SkipLayerNorm
    nodes.append(
        helper.make_node(
            "SkipLayerNormalization",
            ["input_0", "input_1", "ln_weight", "ln_bias"],
            ["ln_out", "", "", "ln_skip_out"],
            "skiplayernorm",
            domain="com.microsoft",
            epsilon=epsilon,
        )
    )

    # Q path: MatMul -> Reshape -> Add(reshape[0], bias[1]) -> Transpose -> matmul_qk
    # Matches: ["Transpose", "Add", "Reshape", "MatMul"] with [0, 0, 0, 0]
    nodes.extend(
        [
            helper.make_node("MatMul", ["ln_out", "q_weight"], ["q_matmul_out"], "q_matmul"),
            helper.make_node("Reshape", ["q_matmul_out", "qkv_reshape_shape"], ["q_4d_bsnh"], "q_reshape"),
            helper.make_node("Add", ["q_4d_bsnh", "q_bias_4d"], ["q_4d_biased"], "q_add"),
            helper.make_node("Transpose", ["q_4d_biased"], ["q_4d_bnsh"], "q_transpose", perm=[0, 2, 1, 3]),
        ]
    )

    # K path: MatMul -> Reshape -> Transpose (no Add)
    # Matches: ["Transpose", "Reshape", "MatMul"] with [1, 0, 0]
    nodes.extend(
        [
            helper.make_node("MatMul", ["ln_out", "k_weight"], ["k_matmul_out"], "k_matmul"),
            helper.make_node("Reshape", ["k_matmul_out", "qkv_reshape_shape"], ["k_4d_bsnh"], "k_reshape"),
            # perm=[0,2,3,1]: [B,S,H,D] -> [B,H,D,S] for K^T
            helper.make_node("Transpose", ["k_4d_bsnh"], ["k_transposed"], "k_transpose", perm=[0, 2, 3, 1]),
        ]
    )

    # V path: MatMul -> Reshape -> Transpose (no Add)
    # Matches: ["Transpose", "Reshape", "MatMul"] with [1, 0, 0]
    nodes.extend(
        [
            helper.make_node("MatMul", ["ln_out", "v_weight"], ["v_matmul_out"], "v_matmul"),
            helper.make_node("Reshape", ["v_matmul_out", "qkv_reshape_shape"], ["v_4d_bsnh"], "v_reshape"),
            helper.make_node("Transpose", ["v_4d_bsnh"], ["v_4d_bnsh"], "v_transpose", perm=[0, 2, 1, 3]),
        ]
    )

    # QK: MatMul -> Add(qk_out, bias) -> Softmax -> MatMul
    nodes.extend(
        [
            helper.make_node("MatMul", ["q_4d_bnsh", "k_transposed"], ["qk_out"], "matmul_qk"),
            helper.make_node("Add", ["qk_out", "qk_bias"], ["qk_add_out"], "add_qk"),
            helper.make_node("Softmax", ["qk_add_out"], ["softmax_out"], "softmax_qk", axis=3),
            helper.make_node("MatMul", ["softmax_out", "v_4d_bnsh"], ["qkv_bnsh"], "matmul_qkv"),
        ]
    )

    # Output: Transpose -> Reshape -> MatMul (no trailing Add before SkipLayerNorm)
    # Matches QKV path: ["MatMul", "Reshape", "Transpose", "MatMul"] with [1, 0, 0, 0]
    nodes.extend(
        [
            helper.make_node("Transpose", ["qkv_bnsh"], ["qkv_bsnh"], "qkv_transpose", perm=[0, 2, 1, 3]),
            helper.make_node("Reshape", ["qkv_bsnh", "out_reshape_shape"], ["attn_out"], "out_reshape"),
            helper.make_node("MatMul", ["attn_out", "out_weight"], ["out_matmul"], "out_matmul"),
            helper.make_node(
                "SkipLayerNormalization",
                ["ln_skip_out", "out_matmul", "ln_weight", "ln_bias"],
                ["output_0", "", "", "output_1"],
                "next_skiplayernorm",
                domain="com.microsoft",
                epsilon=epsilon,
            ),
        ]
    )

    q_weight, _ = get_tensor_and_weight("q_weight", [hidden_size, hidden_size])
    k_weight, _ = get_tensor_and_weight("k_weight", [hidden_size, hidden_size])
    v_weight, _ = get_tensor_and_weight("v_weight", [hidden_size, hidden_size])

    initializers = [
        float_tensor("ln_weight", [hidden_size]),
        float_tensor("ln_bias", [hidden_size]),
        float_tensor("out_weight", [hidden_size, hidden_size]),
        q_weight,
        k_weight,
        v_weight,
        # Q bias in 4D shape [1, 1, num_heads, head_size] for broadcasting after Reshape
        numpy_helper.from_array(np.ones([1, 1, num_heads, head_size], dtype="float32"), name="q_bias_4d"),
        # Non-zero qk_bias so ORT's constant folding (which removes Add(x, 0)) doesn't eliminate this node.
        numpy_helper.from_array(np.array([_NON_ZERO_QK_BIAS], dtype="float32"), name="qk_bias"),
        numpy_helper.from_array(np.array([0, 0, num_heads, head_size], dtype="int64"), name="qkv_reshape_shape"),
        numpy_helper.from_array(np.array([0, 0, hidden_size], dtype="int64"), name="out_reshape_shape"),
    ]

    graph = helper.make_graph(nodes, "conformer_no_add_kv_graph", inputs, outputs, initializers, doc_string="conformer")
    opsetid = helper.make_opsetid("ai.onnx", min(onnx.defs.onnx_opset_version(), 16))
    return helper.make_model(graph, opset_imports=(opsetid,))


def create_conformer_attention_qk_div_masking(
    hidden_size=64,
    num_heads=4,
    epsilon=0.000009999999747378752,
):
    """
    Conformer attention with QK masking using Where→Softmax→Where→Div→Add→MatMul.

    This exercises the new QK path:
      ["Where", "Softmax", "Where", "Div", "Add", "MatMul"] with [0, 2, 0, 2, 0, 0]

    The graph structure for the masked QK computation is:
      MatMul(Q,K^T) → Add(qk_bias) → Div(scale) → inner_Where → Softmax → outer_Where → MatMul(V)
    """
    assert hidden_size % num_heads == 0
    head_size = hidden_size // num_heads

    inputs = [
        helper.make_tensor_value_info("input_0", TensorProto.FLOAT, ["batch_size", "seq_len", hidden_size]),
        helper.make_tensor_value_info("input_1", TensorProto.FLOAT, ["batch_size", "seq_len", hidden_size]),
    ]
    outputs = [
        helper.make_tensor_value_info("output_0", TensorProto.FLOAT, ["batch_size", "seq_len", hidden_size]),
        helper.make_tensor_value_info("output_1", TensorProto.FLOAT, ["batch_size", "seq_len", hidden_size]),
    ]
    nodes = []

    # SkipLayerNorm
    nodes.append(
        helper.make_node(
            "SkipLayerNormalization",
            ["input_0", "input_1", "ln_weight", "ln_bias"],
            ["ln_out", "", "", "ln_skip_out"],
            "skiplayernorm",
            domain="com.microsoft",
            epsilon=epsilon,
        )
    )

    # Q path: MatMul -> Add(bias, matmul_out) -> Reshape -> Transpose -> Div
    # Matches: ["Div", "Transpose", "Reshape", "Add", "MatMul"] with [0, 0, 0, 0, 1]
    nodes.extend(
        [
            helper.make_node("MatMul", ["ln_out", "q_weight"], ["q_matmul_out"], "q_matmul"),
            helper.make_node("Add", ["q_bias", "q_matmul_out"], ["q_add_out"], "q_add"),
            helper.make_node("Reshape", ["q_add_out", "qkv_reshape_shape"], ["q_4d_bsnh"], "q_reshape"),
            helper.make_node("Transpose", ["q_4d_bsnh"], ["q_4d_bnsh"], "q_transpose", perm=[0, 2, 1, 3]),
            helper.make_node("Div", ["q_4d_bnsh", "q_scale"], ["q_scaled"], "q_div"),
        ]
    )

    # K path: MatMul -> Add(matmul_out, bias) -> Reshape -> Transpose
    # Matches: ["Transpose", "Reshape", "Add", "MatMul"] with [1, 0, 0, 0]
    nodes.extend(
        [
            helper.make_node("MatMul", ["ln_out", "k_weight"], ["k_matmul_out"], "k_matmul"),
            helper.make_node("Add", ["k_matmul_out", "k_bias"], ["k_add_out"], "k_add"),
            helper.make_node("Reshape", ["k_add_out", "qkv_reshape_shape"], ["k_4d_bsnh"], "k_reshape"),
            helper.make_node("Transpose", ["k_4d_bsnh"], ["k_transposed"], "k_transpose", perm=[0, 2, 3, 1]),
        ]
    )

    # V path: MatMul -> Add(matmul_out, bias) -> Reshape -> Transpose
    # Matches: ["Transpose", "Reshape", "Add", "MatMul"] with [1, 0, 0, 0]
    nodes.extend(
        [
            helper.make_node("MatMul", ["ln_out", "v_weight"], ["v_matmul_out"], "v_matmul"),
            helper.make_node("Add", ["v_matmul_out", "v_bias"], ["v_add_out"], "v_add"),
            helper.make_node("Reshape", ["v_add_out", "qkv_reshape_shape"], ["v_4d_bsnh"], "v_reshape"),
            helper.make_node("Transpose", ["v_4d_bsnh"], ["v_4d_bnsh"], "v_transpose", perm=[0, 2, 1, 3]),
        ]
    )

    # QK computation with Div masking:
    #   MatMul(QK) -> Add(qk_bias) -> Div(scale) -> inner_Where -> Softmax -> outer_Where -> MatMul(V)
    #
    # Matches: ["Where", "Softmax", "Where", "Div", "Add", "MatMul"] with [0, 2, 0, 2, 0, 0]
    # where_qk = inner_Where
    nodes.extend(
        [
            helper.make_node("MatMul", ["q_scaled", "k_transposed"], ["qk_out"], "matmul_qk"),
            helper.make_node("Add", ["qk_out", "qk_bias"], ["qk_add_out"], "add_qk"),
            helper.make_node("Div", ["qk_add_out", "qk_div_scale"], ["qk_div_out"], "div_qk"),
            # inner_Where: condition ? qk_div_out : mask_value  → input[0]=cond, [1]=mask, [2]=qk_div_out
            helper.make_node(
                "Where",
                ["mask_condition", "mask_value", "qk_div_out"],
                ["inner_where_out"],
                "inner_where",
            ),
            helper.make_node("Softmax", ["inner_where_out"], ["softmax_out"], "softmax_qk", axis=3),
            # outer_Where: condition ? zeros : softmax_out → input[0]=cond, [1]=zeros, [2]=softmax_out
            helper.make_node(
                "Where",
                ["mask_condition", "zeros_val", "softmax_out"],
                ["outer_where_out"],
                "outer_where",
            ),
            helper.make_node("MatMul", ["outer_where_out", "v_4d_bnsh"], ["qkv_bnsh"], "matmul_qkv"),
        ]
    )

    # Output: Transpose -> Reshape -> MatMul -> Add -> SkipLayerNorm
    nodes.extend(
        [
            helper.make_node("Transpose", ["qkv_bnsh"], ["qkv_bsnh"], "qkv_transpose", perm=[0, 2, 1, 3]),
            helper.make_node("Reshape", ["qkv_bsnh", "out_reshape_shape"], ["attn_out"], "out_reshape"),
            helper.make_node("MatMul", ["attn_out", "out_weight"], ["out_matmul"], "out_matmul"),
            helper.make_node("Add", ["out_bias", "out_matmul"], ["out_add"], "out_add"),
            helper.make_node(
                "SkipLayerNormalization",
                ["ln_skip_out", "out_add", "ln_weight", "ln_bias"],
                ["output_0", "", "", "output_1"],
                "next_skiplayernorm",
                domain="com.microsoft",
                epsilon=epsilon,
            ),
        ]
    )

    q_weight, _ = get_tensor_and_weight("q_weight", [hidden_size, hidden_size])
    k_weight, _ = get_tensor_and_weight("k_weight", [hidden_size, hidden_size])
    v_weight, _ = get_tensor_and_weight("v_weight", [hidden_size, hidden_size])

    initializers = [
        float_tensor("ln_weight", [hidden_size]),
        float_tensor("ln_bias", [hidden_size]),
        float_tensor("out_weight", [hidden_size, hidden_size]),
        float_tensor("out_bias", [hidden_size]),
        q_weight,
        k_weight,
        v_weight,
        numpy_helper.from_array(np.array([1.0] * hidden_size, dtype="float32"), name="q_bias"),
        numpy_helper.from_array(np.array([1.0] * hidden_size, dtype="float32"), name="k_bias"),
        numpy_helper.from_array(np.array([1.0] * hidden_size, dtype="float32"), name="v_bias"),
        # Non-zero qk_bias so ORT's constant folding (which removes Add(x, 0)) doesn't eliminate this node.
        numpy_helper.from_array(np.array([_NON_ZERO_QK_BIAS], dtype="float32"), name="qk_bias"),
        numpy_helper.from_array(np.array(1.0 / np.sqrt(head_size), dtype="float32"), name="q_scale"),
        numpy_helper.from_array(np.array(float(head_size), dtype="float32"), name="qk_div_scale"),
        # Boolean mask condition (all True = no masking, for test purposes)
        helper.make_tensor("mask_condition", TensorProto.BOOL, [1, 1, 1, 1], [True]),
        numpy_helper.from_array(np.array([-1e9], dtype="float32"), name="mask_value"),
        numpy_helper.from_array(np.array([0.0], dtype="float32"), name="zeros_val"),
        numpy_helper.from_array(np.array([0, 0, num_heads, head_size], dtype="int64"), name="qkv_reshape_shape"),
        numpy_helper.from_array(np.array([0, 0, hidden_size], dtype="int64"), name="out_reshape_shape"),
    ]

    graph = helper.make_graph(
        nodes, "conformer_qk_div_masking_graph", inputs, outputs, initializers, doc_string="conformer"
    )
    opsetid = helper.make_opsetid("ai.onnx", min(onnx.defs.onnx_opset_version(), 16))
    return helper.make_model(graph, opset_imports=(opsetid,))


if __name__ == "__main__":
    np.random.seed(2)
    num_heads = 8
    hidden_size = 512

    model = create_conformer_attention(num_heads=num_heads, hidden_size=hidden_size)
    onnx.save(model, "conformer_self_mha.onnx")

    model = create_conformer_attention(num_heads=num_heads, hidden_size=hidden_size, fused=True)
    onnx.save(model, "./test_data/models/conformer/conformer_self_mha_fused.onnx")
