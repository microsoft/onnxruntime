# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.  See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

from typing import List

import numpy as np
import onnx
from bert_model_generator import float_tensor
from onnx import TensorProto, helper, numpy_helper


# Adapted from bert_model_generator.py
def get_tensor_and_weight(name: str, shape: List[int], random=False, zeros=False):
    low = 0.0
    high = 1.0
    total_elements = 1
    for x in shape:
        total_elements *= x
    weights = (
        [np.random.uniform(low, high) for _ in range(total_elements)]
        if random
        else [0.0] * total_elements if zeros else [1.0] * total_elements
    )
    return helper.make_tensor(name, TensorProto.FLOAT, shape, weights), weights


def create_whisper_encoder_attention(
    hidden_size=768,
    num_heads=12,
    epsilon=0.000009999999747378752,
    add_before_layernorm=False,
    add_k=False,
    fused=False,
):
    # Get head size and ensure head size is an integer
    assert hidden_size % num_heads == 0
    head_size = hidden_size // num_heads

    # Construct input and output nodes
    inputs = [
        helper.make_tensor_value_info("input_0", TensorProto.FLOAT, ["batch_size", 1500, hidden_size]),
    ]
    outputs = [
        helper.make_tensor_value_info("output_0", TensorProto.FLOAT, ["batch_size", 1500, hidden_size]),
        helper.make_tensor_value_info("output_1", TensorProto.FLOAT, ["batch_size", "sequence_length", hidden_size]),
    ]

    nodes = []
    # Create layernorm (Add + LayerNorm or SkipLayerNorm)
    if add_before_layernorm:
        nodes.extend(
            [
                helper.make_node(
                    "Add", ["input_0", "input_0"], ["layernorm_output_to_skiplayernorm"], "add_before_layernorm"
                ),
                helper.make_node(
                    "LayerNormalization",
                    ["layernorm_output_to_skiplayernorm", "layernorm_weight", "layernorm_bias"],
                    ["layernorm_output_to_matmul"],
                    "layernorm",
                    epsilon=epsilon,
                ),
            ]
        )
    else:
        nodes.append(
            helper.make_node(
                "SkipLayerNormalization",
                ["input_0", "input_0", "layernorm_weight", "layernorm_bias"],
                ["layernorm_output_to_matmul", "", "", "layernorm_output_to_skiplayernorm"],
                "skiplayernorm",
                domain="com.microsoft",
                epsilon=epsilon,
            )
        )

    if fused:
        nodes.append(
            helper.make_node(
                "Attention",
                ["layernorm_output_to_matmul", "Attention_0_qkv_weight", "Attention_0_qkv_bias", ""],
                ["attn_output"],
                "Attention_0",
                domain="com.microsoft",
                num_heads=num_heads,
            ),
        )
    else:
        # Create nodes for Q/K/V paths
        q_nodes = [
            helper.make_node(
                "MatMul", ["layernorm_output_to_matmul", "q_weight"], ["q_matmul_output"], "q_path_matmul"
            ),
            helper.make_node("Add", ["q_bias", "q_matmul_output"], ["q_add_output"], "q_path_add"),
            helper.make_node("Mul", ["q_add_output", "q_scale"], ["q_mul_output"], "q_path_mul"),
            helper.make_node("Reshape", ["q_mul_output", "q_bsnh_reshape"], ["q_4d_bsnh"], "q_reshape_to_4d"),
            helper.make_node("Transpose", ["q_4d_bsnh"], ["q_4d_bnsh"], "q_transpose_to_bnsh", perm=[0, 2, 1, 3]),
            helper.make_node(
                "Reshape",
                ["q_4d_bnsh", "q_attn_heads_output"],
                ["q_output_(num_heads*batch_size,seq_len,head_size)"],
                "q_reshape_to_3d",
            ),
        ]
        k_nodes = [
            helper.make_node(
                "MatMul", ["layernorm_output_to_matmul", "k_weight"], ["k_matmul_output"], "k_path_matmul"
            ),
        ]
        if add_k:
            k_nodes.extend(
                [
                    helper.make_node("Add", ["k_bias", "k_matmul_output"], ["k_add_output"], "k_path_add"),
                    helper.make_node("Reshape", ["k_add_output", "bsnh_reshape"], ["k_4d_bsnh"], "k_reshape_to_4d"),
                ]
            )
        else:
            k_nodes.append(
                helper.make_node("Reshape", ["k_matmul_output", "kv_bsnh_reshape"], ["k_4d_bsnh"], "k_reshape_to_4d"),
            )
        k_nodes.extend(
            [
                helper.make_node("Transpose", ["k_4d_bsnh"], ["k_4d_bnsh"], "k_transpose_to_bnsh", perm=[0, 2, 1, 3]),
                helper.make_node(
                    "Reshape",
                    ["k_4d_bnsh", "k_attn_heads_output"],
                    ["k_output_(num_heads*batch_size,seq_len,head_size)"],
                    "k_reshape_to_3d",
                ),
                helper.make_node(
                    "Transpose",
                    ["k_output_(num_heads*batch_size,seq_len,head_size)"],
                    ["k_output_(num_heads*batch_size,head_size,seq_len)"],
                    "k_transpose_last_two_dims",
                    perm=[0, 2, 1],
                ),
            ]
        )
        v_nodes = [
            helper.make_node(
                "MatMul", ["layernorm_output_to_matmul", "v_weight"], ["v_matmul_output"], "v_path_matmul"
            ),
            helper.make_node("Add", ["v_bias", "v_matmul_output"], ["v_add_output"], "v_path_add"),
            helper.make_node("Reshape", ["v_add_output", "kv_bsnh_reshape"], ["v_4d_bsnh"], "v_reshape_to_4d"),
            helper.make_node("Transpose", ["v_4d_bsnh"], ["v_4d_bnsh"], "v_transpose_to_bnsh", perm=[0, 2, 1, 3]),
            helper.make_node(
                "Reshape",
                ["v_4d_bnsh", "v_attn_heads_output"],
                ["v_output_(num_heads*batch_size,seq_len,head_size)"],
                "v_reshape_to_3d",
            ),
        ]
        nodes.extend(q_nodes)
        nodes.extend(k_nodes)
        nodes.extend(v_nodes)

        # Create nodes used with qkv concats, reshapes, and transposes
        nodes.extend(
            [
                helper.make_node("Shape", ["layernorm_output_to_matmul"], ["shape_output"], "shape"),
                helper.make_node("Gather", ["shape_output", "idx_0"], ["gather_0_output"], "gather_0", axis=0),
                helper.make_node(
                    "Mul", ["gather_0_output", "num_heads_int"], ["mul_attn_heads_output"], "mul_num_heads"
                ),
                helper.make_node(
                    "Unsqueeze",
                    ["mul_attn_heads_output", "unsqueeze_axes_input"],
                    ["unsqueeze_attn_heads_output"],
                    "unsqueeze_num_heads",
                ),
                helper.make_node(
                    "Concat",
                    ["unsqueeze_attn_heads_output", "neg_one", "head_size"],
                    ["q_attn_heads_output"],
                    "q_num_heads",
                    axis=0,
                ),
                helper.make_node(
                    "Concat",
                    ["unsqueeze_attn_heads_output", "neg_one", "head_size"],
                    ["k_attn_heads_output"],
                    "k_num_heads",
                    axis=0,
                ),
                helper.make_node(
                    "Concat",
                    ["unsqueeze_attn_heads_output", "neg_one", "head_size"],
                    ["v_attn_heads_output"],
                    "v_num_heads",
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
                helper.make_node(
                    "Constant",
                    inputs=[],
                    outputs=["kv_bsnh_reshape"],
                    value=numpy_helper.from_array(
                        np.array([0, -1, num_heads, head_size], dtype="int64"), name="const_tensor"
                    ),
                ),
            ]
        )

        # Create nodes used with Q x K' and softmax(Q x K') x V
        nodes.extend(
            [
                helper.make_node("Gather", ["shape_output", "idx_1"], ["gather_1_output"], "gather_1", axis=0),
                helper.make_node(
                    "Unsqueeze", ["gather_0_output", "unsqueeze_axes_input"], ["unsqueeze_0_output"], "unsqueeze_0"
                ),
                helper.make_node(
                    "Unsqueeze", ["gather_1_output", "unsqueeze_axes_input"], ["unsqueeze_1_output"], "unsqueeze_1"
                ),
                helper.make_node(
                    "Concat",
                    ["unsqueeze_0_output", "num_heads", "unsqueeze_1_output", "head_size"],
                    ["bnsh_format"],
                    axis=0,
                ),
                helper.make_node(
                    "Concat", ["unsqueeze_0_output", "unsqueeze_1_output", "hidden_size"], ["bsd_format"], axis=0
                ),
            ]
        )

        # Create nodes for computing softmax(Q x K') x V
        nodes.extend(
            [
                helper.make_node(
                    "MatMul",
                    [
                        "q_output_(num_heads*batch_size,seq_len,head_size)",
                        "k_output_(num_heads*batch_size,head_size,seq_len)",
                    ],
                    ["qk_output_(num_heads*batch_size,seq_len,seq_len)"],
                    "matmul_qk",
                ),
                helper.make_node(
                    "Softmax",
                    ["qk_output_(num_heads*batch_size,seq_len,seq_len)"],
                    ["softmax_output"],
                    "softmax_qk",
                    axis=2,
                ),
                helper.make_node(
                    "MatMul",
                    ["softmax_output", "v_output_(num_heads*batch_size,seq_len,head_size)"],
                    ["qkv_output_(num_heads*batch_size,seq_len,head_size)"],
                    "matmul_qkv",
                ),
                helper.make_node(
                    "Reshape",
                    ["qkv_output_(num_heads*batch_size,seq_len,head_size)", "bnsh_format"],
                    ["qkv_bnsh"],
                    "reshape_qkv_to_bnsh",
                ),
                helper.make_node("Transpose", ["qkv_bnsh"], ["qkv_bsnh"], "transpose_bnsh_to_bsnh", perm=[0, 2, 1, 3]),
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
            "layernorm_output_to_skiplayernorm",
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
            "layernorm_output_to_skiplayernorm",
            "matmul_after_attn_output",
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
    q_weight, q_weight_data = get_tensor_and_weight("q_weight", [hidden_size, hidden_size])
    q_bias, q_bias_data = get_tensor_and_weight("q_bias", [hidden_size])
    k_weight, k_weight_data = get_tensor_and_weight("k_weight", [hidden_size, hidden_size])
    k_bias, k_bias_data = get_tensor_and_weight("k_bias", [hidden_size], zeros=(not add_k))
    v_weight, v_weight_data = get_tensor_and_weight("v_weight", [hidden_size, hidden_size])
    v_bias, v_bias_data = get_tensor_and_weight("v_bias", [hidden_size])
    qkv_weight = helper.make_tensor(
        "Attention_0_qkv_weight",
        TensorProto.FLOAT,
        [hidden_size, 3 * hidden_size],
        q_weight_data + k_weight_data + v_weight_data,
    )
    qkv_bias = helper.make_tensor(
        "Attention_0_qkv_bias", TensorProto.FLOAT, [3 * hidden_size], q_bias_data + k_bias_data + v_bias_data
    )
    initializers = [
        float_tensor("layernorm_weight", [hidden_size]),
        float_tensor("layernorm_bias", [hidden_size]),
        float_tensor("matmul_after_attn_initializer", [hidden_size, hidden_size]),
        float_tensor("add_after_attn_initializer", [hidden_size]),
    ]
    if fused:
        initializers.extend([qkv_weight, qkv_bias])
    else:
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
        if add_k:
            initializers.extend([q_weight, q_bias, k_weight, k_bias, v_weight, v_bias])
        else:
            initializers.extend([q_weight, q_bias, k_weight, v_weight, v_bias])

    # Construct graph
    graph = helper.make_graph(
        nodes, "whisper_encoder_attention_graph", inputs, outputs, initializers, doc_string="whisper"
    )
    opsetid = helper.make_opsetid("ai.onnx", min(onnx.defs.onnx_opset_version(), 16))
    return helper.make_model(graph, opset_imports=(opsetid,))


def create_whisper_decoder_attention(
    hidden_size=768,
    num_heads=12,
    epsilon=0.000009999999747378752,
    add_before_layernorm=False,
    add_k=False,
    fused=False,
):
    # Get head size and ensure head size is an integer
    assert hidden_size % num_heads == 0
    head_size = hidden_size // num_heads

    # Construct input and output nodes
    # Dummy inputs are used to prevent the nodes in the path for the decoder attention mask to be fused together
    # before attention is fused
    inputs = [
        helper.make_tensor_value_info("input_0", TensorProto.FLOAT, ["batch_size", 1500, hidden_size]),
        helper.make_tensor_value_info("dummy_input_int64", TensorProto.INT64, ["dummy_input_1d_int64"]),
        helper.make_tensor_value_info("dummy_input_fp32", TensorProto.FLOAT, ["dummy_input_1d_fp32"]),
    ]
    outputs = [
        helper.make_tensor_value_info(
            "present.0.decoder.key", TensorProto.FLOAT, ["batch_size", num_heads, 1500, head_size]
        ),
        helper.make_tensor_value_info(
            "present.0.decoder.value", TensorProto.FLOAT, ["batch_size", num_heads, 1500, head_size]
        ),
        helper.make_tensor_value_info("output_0", TensorProto.FLOAT, ["batch_size", 1500, hidden_size]),
        helper.make_tensor_value_info("output_1", TensorProto.FLOAT, ["batch_size", "sequence_length", hidden_size]),
    ]

    nodes = []
    # Create layernorm (Add + LayerNorm or SkipLayerNorm)
    if add_before_layernorm:
        nodes.extend(
            [
                helper.make_node(
                    "Add", ["input_0", "input_0"], ["layernorm_output_to_skiplayernorm"], "add_before_layernorm"
                ),
                helper.make_node(
                    "LayerNormalization",
                    ["layernorm_output_to_skiplayernorm", "layernorm_weight", "layernorm_bias"],
                    ["layernorm_output_to_matmul"],
                    "layernorm",
                    epsilon=epsilon,
                ),
            ]
        )
    else:
        nodes.append(
            helper.make_node(
                "SkipLayerNormalization",
                ["input_0", "input_0", "layernorm_weight", "layernorm_bias"],
                ["layernorm_output_to_matmul", "", "", "layernorm_output_to_skiplayernorm"],
                "skiplayernorm",
                domain="com.microsoft",
                epsilon=epsilon,
            )
        )

    if fused:
        nodes.extend(
            [
                helper.make_node(
                    "Attention",
                    [
                        "layernorm_output_to_matmul",
                        "Attention_0_qkv_weight",
                        "Attention_0_qkv_bias",
                        "",
                        "",
                        "attention_add_qk_mask",
                    ],
                    ["attn_output", "present_0_decoder"],
                    "Attention_0",
                    domain="com.microsoft",
                    num_heads=num_heads,
                ),
                helper.make_node(
                    "Gather",
                    ["present_0_decoder", "index_0"],
                    ["present.0.decoder.key"],
                    "Gather_0",
                    axis=0,
                ),
                helper.make_node(
                    "Gather",
                    ["present_0_decoder", "index_1"],
                    ["present.0.decoder.value"],
                    "Gather_1",
                    axis=0,
                ),
            ]
        )
    else:
        # Create nodes for Q/K/V paths
        q_nodes = [
            helper.make_node(
                "MatMul", ["layernorm_output_to_matmul", "q_weight"], ["q_matmul_output"], "q_path_matmul"
            ),
            helper.make_node("Add", ["q_bias", "q_matmul_output"], ["q_add_output"], "q_path_add"),
            helper.make_node("Mul", ["q_add_output", "q_scale"], ["q_mul_output"], "q_path_mul"),
            helper.make_node("Reshape", ["q_mul_output", "q_bsnh_reshape"], ["q_4d_bsnh"], "q_reshape_to_4d"),
            helper.make_node("Transpose", ["q_4d_bsnh"], ["q_4d_bnsh"], "q_transpose_to_bnsh", perm=[0, 2, 1, 3]),
            helper.make_node(
                "Reshape",
                ["q_4d_bnsh", "q_attn_heads_output"],
                ["q_output_(num_heads*batch_size,seq_len,head_size)"],
                "q_reshape_to_3d",
            ),
        ]
        k_nodes = [
            helper.make_node(
                "MatMul", ["layernorm_output_to_matmul", "k_weight"], ["k_matmul_output"], "k_path_matmul"
            ),
        ]
        if add_k:
            k_nodes.extend(
                [
                    helper.make_node("Add", ["k_bias", "k_matmul_output"], ["k_add_output"], "k_path_add"),
                    helper.make_node("Reshape", ["k_add_output", "bsnh_reshape"], ["k_4d_bsnh"], "k_reshape_to_4d"),
                ]
            )
        else:
            k_nodes.append(
                helper.make_node("Reshape", ["k_matmul_output", "kv_bsnh_reshape"], ["k_4d_bsnh"], "k_reshape_to_4d"),
            )
        k_nodes.extend(
            [
                helper.make_node(
                    "Transpose",
                    ["k_4d_bsnh"],
                    ["present.0.decoder.key"],
                    "k_transpose_to_bnsh",
                    perm=[0, 2, 1, 3],
                ),
                helper.make_node(
                    "Reshape",
                    ["present.0.decoder.key", "k_attn_heads_output"],
                    ["k_output_(num_heads*batch_size,seq_len,head_size)"],
                    "k_reshape_to_3d",
                ),
                helper.make_node(
                    "Transpose",
                    ["k_output_(num_heads*batch_size,seq_len,head_size)"],
                    ["k_output_(num_heads*batch_size,head_size,seq_len)"],
                    "k_transpose_last_two_dims",
                    perm=[0, 2, 1],
                ),
            ]
        )
        v_nodes = [
            helper.make_node(
                "MatMul", ["layernorm_output_to_matmul", "v_weight"], ["v_matmul_output"], "v_path_matmul"
            ),
            helper.make_node("Add", ["v_bias", "v_matmul_output"], ["v_add_output"], "v_path_add"),
            helper.make_node("Reshape", ["v_add_output", "kv_bsnh_reshape"], ["v_4d_bsnh"], "v_reshape_to_4d"),
            helper.make_node(
                "Transpose", ["v_4d_bsnh"], ["present.0.decoder.value"], "v_transpose_to_bnsh", perm=[0, 2, 1, 3]
            ),
            helper.make_node(
                "Reshape",
                ["present.0.decoder.value", "v_attn_heads_output"],
                ["v_output_(num_heads*batch_size,seq_len,head_size)"],
                "v_reshape_to_3d",
            ),
        ]
        nodes.extend(q_nodes)
        nodes.extend(k_nodes)
        nodes.extend(v_nodes)

        # Create nodes used with qkv concats, reshapes, and transposes
        nodes.extend(
            [
                helper.make_node("Shape", ["layernorm_output_to_matmul"], ["shape_output"], "shape"),
                helper.make_node("Gather", ["shape_output", "idx_0"], ["gather_0_output"], "gather_0", axis=0),
                helper.make_node(
                    "Mul", ["gather_0_output", "num_heads_int"], ["mul_attn_heads_output"], "mul_num_heads"
                ),
                helper.make_node(
                    "Unsqueeze",
                    ["mul_attn_heads_output", "unsqueeze_axes_input"],
                    ["unsqueeze_attn_heads_output"],
                    "unsqueeze_num_heads",
                ),
                helper.make_node(
                    "Concat",
                    ["unsqueeze_attn_heads_output", "neg_one", "head_size"],
                    ["q_attn_heads_output"],
                    "q_num_heads",
                    axis=0,
                ),
                helper.make_node(
                    "Concat",
                    ["unsqueeze_attn_heads_output", "neg_one", "head_size"],
                    ["k_attn_heads_output"],
                    "k_num_heads",
                    axis=0,
                ),
                helper.make_node(
                    "Concat",
                    ["unsqueeze_attn_heads_output", "neg_one", "head_size"],
                    ["v_attn_heads_output"],
                    "v_num_heads",
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
                helper.make_node(
                    "Constant",
                    inputs=[],
                    outputs=["kv_bsnh_reshape"],
                    value=numpy_helper.from_array(
                        np.array([0, -1, num_heads, head_size], dtype="int64"), name="const_tensor"
                    ),
                ),
            ]
        )

        # Create nodes used with mask
        nodes.extend(
            [
                helper.make_node(
                    "Shape", ["k_output_(num_heads*batch_size,seq_len,head_size)"], ["mask_shape_output"], "mask_shape"
                ),
                helper.make_node(
                    "Gather", ["mask_shape_output", "idx_1"], ["mask_gather_1_output"], "mask_gather_1", axis=0
                ),
                helper.make_node(
                    "Unsqueeze",
                    ["mask_gather_1_output", "unsqueeze_axes_input"],
                    ["mask_unsqueeze_1_output"],
                    "mask_unsqueeze_1",
                ),
                helper.make_node(
                    "Concat",
                    ["unsqueeze_0_output", "num_heads", "unsqueeze_1_output", "mask_unsqueeze_1_output"],
                    ["mask_concat_output"],
                    "mask_concat",
                    axis=0,
                ),
                helper.make_node(
                    "Mul", ["gather_0_output", "num_heads_int"], ["mul_mask_heads_output"], "mul_mask_heads"
                ),
                helper.make_node(
                    "Unsqueeze",
                    ["mul_mask_heads_output", "unsqueeze_axes_input"],
                    ["unsqueeze_mask_heads_output"],
                    "unsqueeze_mask_heads",
                ),
                helper.make_node(
                    "Concat",
                    ["unsqueeze_mask_heads_output", "unsqueeze_1_output", "mask_unsqueeze_1_output"],
                    ["concat_input_for_reshape_after_add"],
                    "concat_for_reshape_after_add",
                    axis=0,
                ),
            ]
        )

        # Create nodes used with Q x K' + mask and softmax(Q x K' + mask) x V
        nodes.extend(
            [
                helper.make_node("Gather", ["shape_output", "idx_1"], ["gather_1_output"], "gather_1", axis=0),
                helper.make_node(
                    "Unsqueeze", ["gather_0_output", "unsqueeze_axes_input"], ["unsqueeze_0_output"], "unsqueeze_0"
                ),
                helper.make_node(
                    "Unsqueeze", ["gather_1_output", "unsqueeze_axes_input"], ["unsqueeze_1_output"], "unsqueeze_1"
                ),
                helper.make_node(
                    "Concat",
                    ["unsqueeze_0_output", "num_heads", "unsqueeze_1_output", "head_size"],
                    ["bnsh_format"],
                    axis=0,
                ),
                helper.make_node(
                    "Concat", ["unsqueeze_0_output", "unsqueeze_1_output", "hidden_size"], ["bsd_format"], axis=0
                ),
            ]
        )

        # Create nodes for computing softmax(Q x K' + mask) x V
        nodes.extend(
            [
                helper.make_node(
                    "MatMul",
                    [
                        "q_output_(num_heads*batch_size,seq_len,head_size)",
                        "k_output_(num_heads*batch_size,head_size,seq_len)",
                    ],
                    ["qk_output_(num_heads*batch_size,seq_len,seq_len)"],
                    "matmul_qk",
                ),
                helper.make_node(
                    "Reshape",
                    ["qk_output_(num_heads*batch_size,seq_len,seq_len)", "mask_concat_output"],
                    ["qk_output_(batch_size,num_heads,seq_len,seq_len)"],
                    "reshape_qk_to_bnsh",
                ),
                helper.make_node(
                    "Add",
                    ["qk_output_(batch_size,num_heads,seq_len,seq_len)", "attention_add_qk"],
                    ["add_qk_output_(batch_size,num_heads_seq_len,seq_len)"],
                    "add_qk",
                ),
                helper.make_node(
                    "Reshape",
                    ["add_qk_output_(batch_size,num_heads_seq_len,seq_len)", "concat_input_for_reshape_after_add"],
                    ["add_qk_output_(num_heads*batch_size,seq_len,seq_len)"],
                    "reshape_add_qk_before_softmax",
                ),
                helper.make_node(
                    "Softmax",
                    ["add_qk_output_(num_heads*batch_size,seq_len,seq_len)"],
                    ["softmax_output"],
                    "softmax_qk",
                    axis=2,
                ),
                helper.make_node(
                    "MatMul",
                    ["softmax_output", "v_output_(num_heads*batch_size,seq_len,head_size)"],
                    ["qkv_output_(num_heads*batch_size,seq_len,head_size)"],
                    "matmul_qkv",
                ),
                helper.make_node(
                    "Reshape",
                    ["qkv_output_(num_heads*batch_size,seq_len,head_size)", "bnsh_format"],
                    ["qkv_bnsh"],
                    "reshape_qkv_to_bnsh",
                ),
                helper.make_node("Transpose", ["qkv_bnsh"], ["qkv_bsnh"], "transpose_bnsh_to_bsnh", perm=[0, 2, 1, 3]),
                helper.make_node("Reshape", ["qkv_bsnh", "bsd_format"], ["attn_output"], "qkv_bsd"),
            ]
        )

    # Create nodes that make attention mask
    nodes.extend(
        [
            # "attention_mask" is (decoder_seq_len, decoder_seq_len) but is assumed to be (1, 1) for this test.
            # There are other nodes that automatically set the attention mask size correctly but those nodes do not
            # impact the attention fusion. Hence, this assumption is made in order to simplify the inputs for the
            # following nodes.
            helper.make_node(
                "Where",
                ["all_ones", "where_filter_constant", "dummy_input_fp32"],
                ["where_output"],
                "mask_filter_where",
            ),
            helper.make_node(
                "Unsqueeze",
                ["where_output", "dummy_input_int64"],
                ["unsqueeze_mask_output_1"],
                "unsqueeze_attn_mask_1",
            ),
            helper.make_node(
                "Unsqueeze",
                ["unsqueeze_mask_output_1", "dummy_input_int64"],
                ["unsqueeze_mask_output_2"],
                "unsqueeze_attn_mask_2",
            ),
            helper.make_node(
                "Expand",
                inputs=["unsqueeze_mask_output_2", "dummy_input_int64"],
                outputs=["attention_add_qk"],
                name="expand_mask_from_(b,1,m,m)_to_(b,n,m,m)",
            ),
        ]
    )
    if fused:
        nodes.append(
            helper.make_node(
                "Concat",
                inputs=["attention_add_qk" for _ in range(num_heads)],
                outputs=["attention_add_qk_mask"],
                name="Concat_0",
                axis=1,
            ),
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
            "layernorm_output_to_skiplayernorm",
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
            "layernorm_output_to_skiplayernorm",
            "matmul_after_attn_output",
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
    q_weight, q_weight_data = get_tensor_and_weight("q_weight", [hidden_size, hidden_size])
    q_bias, q_bias_data = get_tensor_and_weight("q_bias", [hidden_size])
    k_weight, k_weight_data = get_tensor_and_weight("k_weight", [hidden_size, hidden_size])
    k_bias, k_bias_data = get_tensor_and_weight("k_bias", [hidden_size], zeros=(not add_k))
    v_weight, v_weight_data = get_tensor_and_weight("v_weight", [hidden_size, hidden_size])
    v_bias, v_bias_data = get_tensor_and_weight("v_bias", [hidden_size])
    qkv_weight = helper.make_tensor(
        "Attention_0_qkv_weight",
        TensorProto.FLOAT,
        [hidden_size, 3 * hidden_size],
        q_weight_data + k_weight_data + v_weight_data,
    )
    qkv_bias = helper.make_tensor(
        "Attention_0_qkv_bias", TensorProto.FLOAT, [3 * hidden_size], q_bias_data + k_bias_data + v_bias_data
    )
    initializers = [
        float_tensor("layernorm_weight", [hidden_size]),
        float_tensor("layernorm_bias", [hidden_size]),
        float_tensor("matmul_after_attn_initializer", [hidden_size, hidden_size]),
        float_tensor("add_after_attn_initializer", [hidden_size]),
    ]
    # Add initializers for attention mask
    initializers.extend(
        [
            numpy_helper.from_array(np.array([[1]], dtype=bool), name="all_ones"),
            numpy_helper.from_array(np.array([1], dtype="float32"), name="where_filter_constant"),
        ]
    )

    if fused:
        initializers.extend(
            [
                qkv_weight,
                qkv_bias,
                numpy_helper.from_array(np.array(0, dtype="int64"), name="index_0"),
                numpy_helper.from_array(np.array(1, dtype="int64"), name="index_1"),
            ]
        )
    else:
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

        if add_k:
            initializers.extend([q_weight, q_bias, k_weight, k_bias, v_weight, v_bias])
        else:
            initializers.extend([q_weight, q_bias, k_weight, v_weight, v_bias])

    # Construct graph
    graph = helper.make_graph(
        nodes, "whisper_decoder_attention_graph", inputs, outputs, initializers, doc_string="whisper"
    )
    opsetid = helper.make_opsetid("ai.onnx", min(onnx.defs.onnx_opset_version(), 16))
    return helper.make_model(graph, opset_imports=(opsetid,))


def create_whisper_decoder_multihead_attention(
    hidden_size=768, num_heads=12, epsilon=0.000009999999747378752, add_k=False, fused=False
):
    # Get head size and ensure head size is an integer
    assert hidden_size % num_heads == 0
    head_size = hidden_size // num_heads

    # Construct input and output nodes
    inputs = [
        helper.make_tensor_value_info("input_0", TensorProto.FLOAT, ["batch_size", 1500, hidden_size]),
        helper.make_tensor_value_info("encoder_hidden_states", TensorProto.FLOAT, ["batch_size", 1500, hidden_size]),
    ]
    outputs = [
        helper.make_tensor_value_info("output_0", TensorProto.FLOAT, ["batch_size", 1500, hidden_size]),
        helper.make_tensor_value_info("output_1", TensorProto.FLOAT, ["batch_size", "sequence_length", hidden_size]),
        helper.make_tensor_value_info(
            "present.0.encoder.key", TensorProto.FLOAT, ["batch_size", num_heads, 1500, head_size]
        ),
        helper.make_tensor_value_info(
            "present.0.encoder.value", TensorProto.FLOAT, ["batch_size", num_heads, 1500, head_size]
        ),
    ]

    # Create SkipLayerNorm (since there's no Add + LayerNorm variant for this attention subgraph)
    nodes = [
        helper.make_node(
            "SkipLayerNormalization",
            ["input_0", "input_0", "layernorm_weight", "layernorm_bias"],
            ["layernorm_output_to_matmul", "", "", "layernorm_output_to_skiplayernorm"],
            "skiplayernorm",
            domain="com.microsoft",
            epsilon=epsilon,
        )
    ]

    if fused:
        nodes.extend(
            [
                helper.make_node(
                    "MatMul", ["layernorm_output_to_matmul", "q_weight"], ["q_matmul_output"], "q_path_matmul"
                ),
                helper.make_node("MatMul", ["encoder_hidden_states", "k_weight"], ["k_matmul_output"], "k_path_matmul"),
                helper.make_node("MatMul", ["encoder_hidden_states", "v_weight"], ["v_matmul_output"], "v_path_matmul"),
                helper.make_node(
                    "MultiHeadAttention",
                    ["q_matmul_output", "k_matmul_output", "v_matmul_output", "Attention_0_qkv_bias"],
                    ["attn_output", "present.0.encoder.key", "present.0.encoder.value"],
                    "Attention_0",
                    domain="com.microsoft",
                    num_heads=num_heads,
                ),
            ]
        )
    else:
        # Create nodes for Q/K/V paths
        q_nodes = [
            helper.make_node(
                "MatMul",
                ["layernorm_output_to_matmul", "q_weight"],
                ["q_matmul_output"],
                "q_path_matmul",
            ),
            helper.make_node("Add", ["q_bias", "q_matmul_output"], ["q_add_output"], "q_path_add"),
            helper.make_node("Mul", ["q_add_output", "q_scale"], ["q_mul_output"], "q_path_mul"),
            helper.make_node("Reshape", ["q_mul_output", "q_bsnh_reshape"], ["q_4d_bsnh"], "q_reshape_to_4d"),
            helper.make_node("Transpose", ["q_4d_bsnh"], ["q_4d_bnsh"], "q_transpose_to_bnsh", perm=[0, 2, 1, 3]),
            helper.make_node(
                "Reshape",
                ["q_4d_bnsh", "q_attn_heads_output"],
                ["q_output_(num_heads*batch_size,seq_len,head_size)"],
                "q_reshape_to_3d",
            ),
        ]
        k_nodes = [
            helper.make_node("MatMul", ["encoder_hidden_states", "k_weight"], ["k_matmul_output"], "k_path_matmul"),
        ]
        if add_k:
            k_nodes.extend(
                [
                    helper.make_node("Add", ["k_bias", "k_matmul_output"], ["k_add_output"], "k_path_add"),
                    helper.make_node("Reshape", ["k_add_output", "bsnh_reshape"], ["k_4d_bsnh"], "k_reshape_to_4d"),
                ]
            )
        else:
            k_nodes.append(
                helper.make_node("Reshape", ["k_matmul_output", "kv_bsnh_reshape"], ["k_4d_bsnh"], "k_reshape_to_4d"),
            )
        k_nodes.extend(
            [
                helper.make_node(
                    "Transpose", ["k_4d_bsnh"], ["present.0.encoder.key"], "k_transpose_to_bnsh", perm=[0, 2, 1, 3]
                ),
                helper.make_node(
                    "Reshape",
                    ["present.0.encoder.key", "k_attn_heads_output"],
                    ["k_output_(num_heads*batch_size,seq_len,head_size)"],
                    "k_reshape_to_3d",
                ),
                helper.make_node(
                    "Transpose",
                    ["k_output_(num_heads*batch_size,seq_len,head_size)"],
                    ["k_output_(num_heads*batch_size,head_size,seq_len)"],
                    "k_transpose_last_two_dims",
                    perm=[0, 2, 1],
                ),
            ]
        )
        v_nodes = [
            helper.make_node("MatMul", ["encoder_hidden_states", "v_weight"], ["v_matmul_output"], "v_path_matmul"),
            helper.make_node("Add", ["v_bias", "v_matmul_output"], ["v_add_output"], "v_path_add"),
            helper.make_node("Reshape", ["v_add_output", "kv_bsnh_reshape"], ["v_4d_bsnh"], "v_reshape_to_4d"),
            helper.make_node(
                "Transpose", ["v_4d_bsnh"], ["present.0.encoder.value"], "v_transpose_to_bnsh", perm=[0, 2, 1, 3]
            ),
            helper.make_node(
                "Reshape",
                ["present.0.encoder.value", "v_attn_heads_output"],
                ["v_output_(num_heads*batch_size,seq_len,head_size)"],
                "v_reshape_to_3d",
            ),
        ]
        nodes.extend(q_nodes)
        nodes.extend(k_nodes)
        nodes.extend(v_nodes)

        # Create nodes used with qkv concats, reshapes, and transposes
        nodes.extend(
            [
                helper.make_node("Shape", ["layernorm_output_to_matmul"], ["shape_output"], "shape"),
                helper.make_node("Gather", ["shape_output", "idx_0"], ["gather_0_output"], "gather_0", axis=0),
                helper.make_node(
                    "Mul", ["gather_0_output", "num_heads_int"], ["mul_attn_heads_output"], "mul_num_heads"
                ),
                helper.make_node(
                    "Unsqueeze",
                    ["mul_attn_heads_output", "unsqueeze_axes_input"],
                    ["unsqueeze_attn_heads_output"],
                    "unsqueeze_num_heads",
                ),
                helper.make_node(
                    "Concat",
                    ["unsqueeze_attn_heads_output", "neg_one", "head_size"],
                    ["q_attn_heads_output"],
                    "q_num_heads",
                    axis=0,
                ),
                helper.make_node(
                    "Concat",
                    ["unsqueeze_attn_heads_output", "neg_one", "head_size"],
                    ["k_attn_heads_output"],
                    "k_num_heads",
                    axis=0,
                ),
                helper.make_node(
                    "Concat",
                    ["unsqueeze_attn_heads_output", "neg_one", "head_size"],
                    ["v_attn_heads_output"],
                    "v_num_heads",
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
                helper.make_node(
                    "Constant",
                    inputs=[],
                    outputs=["kv_bsnh_reshape"],
                    value=numpy_helper.from_array(
                        np.array([0, -1, num_heads, head_size], dtype="int64"), name="const_tensor"
                    ),
                ),
            ]
        )

        # Create nodes used with Q x K' and softmax(Q x K') x V
        nodes.extend(
            [
                helper.make_node("Gather", ["shape_output", "idx_1"], ["gather_1_output"], "gather_1", axis=0),
                helper.make_node(
                    "Unsqueeze",
                    ["gather_0_output", "unsqueeze_axes_input"],
                    ["unsqueeze_0_output"],
                    "unsqueeze_0",
                ),
                helper.make_node(
                    "Unsqueeze",
                    ["gather_1_output", "unsqueeze_axes_input"],
                    ["unsqueeze_1_output"],
                    "unsqueeze_1",
                ),
                helper.make_node(
                    "Concat",
                    ["unsqueeze_0_output", "num_heads", "unsqueeze_1_output", "head_size"],
                    ["bnsh_format"],
                    axis=0,
                ),
                helper.make_node(
                    "Concat",
                    ["unsqueeze_0_output", "unsqueeze_1_output", "hidden_size"],
                    ["bsd_format"],
                    axis=0,
                ),
            ]
        )

        # Create nodes for computing softmax(Q x K') x V
        nodes.extend(
            [
                helper.make_node(
                    "MatMul",
                    [
                        "q_output_(num_heads*batch_size,seq_len,head_size)",
                        "k_output_(num_heads*batch_size,head_size,seq_len)",
                    ],
                    ["qk_output_(num_heads*batch_size,seq_len,seq_len)"],
                    "matmul_qk",
                ),
                helper.make_node(
                    "Softmax",
                    ["qk_output_(num_heads*batch_size,seq_len,seq_len)"],
                    ["softmax_output"],
                    "softmax_qk",
                    axis=2,
                ),
                helper.make_node(
                    "MatMul",
                    ["softmax_output", "v_output_(num_heads*batch_size,seq_len,head_size)"],
                    ["qkv_output_(num_heads*batch_size,seq_len,head_size)"],
                    "matmul_qkv",
                ),
                helper.make_node(
                    "Reshape",
                    ["qkv_output_(num_heads*batch_size,seq_len,head_size)", "bnsh_format"],
                    ["qkv_bnsh"],
                    "reshape_qkv_to_bnsh",
                ),
                helper.make_node("Transpose", ["qkv_bnsh"], ["qkv_bsnh"], "transpose_bnsh_to_bsnh", perm=[0, 2, 1, 3]),
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
            "layernorm_output_to_skiplayernorm",
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
            "layernorm_output_to_skiplayernorm",
            "matmul_after_attn_output",
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
    q_weight, q_weight_data = get_tensor_and_weight("q_weight", [hidden_size, hidden_size])
    q_bias, q_bias_data = get_tensor_and_weight("q_bias", [hidden_size])
    k_weight, k_weight_data = get_tensor_and_weight("k_weight", [hidden_size, hidden_size])
    k_bias, k_bias_data = get_tensor_and_weight("k_bias", [hidden_size], zeros=(not add_k))
    v_weight, v_weight_data = get_tensor_and_weight("v_weight", [hidden_size, hidden_size])
    v_bias, v_bias_data = get_tensor_and_weight("v_bias", [hidden_size])
    qkv_bias = helper.make_tensor(
        "Attention_0_qkv_bias", TensorProto.FLOAT, [3 * hidden_size], q_bias_data + k_bias_data + v_bias_data
    )
    initializers = [
        float_tensor("layernorm_weight", [hidden_size]),
        float_tensor("layernorm_bias", [hidden_size]),
        float_tensor("matmul_after_attn_initializer", [hidden_size, hidden_size]),
        float_tensor("add_after_attn_initializer", [hidden_size]),
    ]

    # Add Q/K/V weight tensors as initializers
    initializers.extend([q_weight, k_weight, v_weight])

    if fused:
        initializers.append(qkv_bias)
    else:
        if add_k:
            initializers.extend([q_bias, k_bias, v_bias])
        else:
            initializers.extend([q_bias, v_bias])

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
    graph = helper.make_graph(nodes, "whisper_decoder_mha_graph", inputs, outputs, initializers, doc_string="whisper")
    opsetid = helper.make_opsetid("ai.onnx", min(onnx.defs.onnx_opset_version(), 16))
    return helper.make_model(graph, opset_imports=(opsetid,))


def create_whisper_decoder_with_past_multihead_self_attention(
    hidden_size=768,
    num_heads=12,
    epsilon=0.000009999999747378752,
    add_before_layernorm=False,
    add_k=False,
    fused=False,
):
    # Get head size and ensure head size is an integer
    assert hidden_size % num_heads == 0
    head_size = hidden_size // num_heads

    # Construct input and output nodes
    inputs = [
        helper.make_tensor_value_info("input_0", TensorProto.FLOAT, ["batch_size", 1500, hidden_size]),
        helper.make_tensor_value_info(
            "past_key_values.0.decoder.key", TensorProto.FLOAT, ["batch_size", num_heads, "past_seq_len", head_size]
        ),
        helper.make_tensor_value_info(
            "past_key_values.0.decoder.value", TensorProto.FLOAT, ["batch_size", num_heads, "past_seq_len", head_size]
        ),
    ]
    outputs = [
        helper.make_tensor_value_info("output_0", TensorProto.FLOAT, ["batch_size", 1500, hidden_size]),
        helper.make_tensor_value_info("output_1", TensorProto.FLOAT, ["batch_size", "sequence_length", hidden_size]),
        helper.make_tensor_value_info(
            "present.0.decoder.key", TensorProto.FLOAT, ["batch_size", num_heads, "past_seq_len + 1", head_size]
        ),
        helper.make_tensor_value_info(
            "present.0.decoder.value", TensorProto.FLOAT, ["batch_size", num_heads, "past_seq_len + 1", head_size]
        ),
    ]
    nodes = []

    # Create layernorm (Add + LayerNorm or SkipLayerNorm)
    if add_before_layernorm:
        nodes.extend(
            [
                helper.make_node(
                    "Add", ["input_0", "input_0"], ["layernorm_output_to_skiplayernorm"], "add_before_layernorm"
                ),
                helper.make_node(
                    "LayerNormalization",
                    ["layernorm_output_to_skiplayernorm", "layernorm_weight", "layernorm_bias"],
                    ["layernorm_output_to_matmul"],
                    "layernorm",
                    epsilon=epsilon,
                ),
            ]
        )
    else:
        nodes.append(
            helper.make_node(
                "SkipLayerNormalization",
                ["input_0", "input_0", "layernorm_weight", "layernorm_bias"],
                ["layernorm_output_to_matmul", "", "", "layernorm_output_to_skiplayernorm"],
                "skiplayernorm",
                domain="com.microsoft",
                epsilon=epsilon,
            )
        )

    if fused:
        nodes.extend(
            [
                helper.make_node(
                    "MatMul",
                    ["layernorm_output_to_matmul", "MatMul_0_qkv_weight"],
                    ["MatMul_0_qkv_out"],
                    "MatMul_0",
                ),
                helper.make_node(
                    "Slice",
                    ["MatMul_0_qkv_out", "MatMul_0_q_start_index", "MatMul_0_k_start_index", "MatMul_0_qkv_last_axis"],
                    ["MatMul_0_q_out"],
                    "Slice_0",
                ),
                helper.make_node(
                    "Slice",
                    ["MatMul_0_qkv_out", "MatMul_0_k_start_index", "MatMul_0_v_start_index", "MatMul_0_qkv_last_axis"],
                    ["MatMul_0_k_out"],
                    "Slice_1",
                ),
                helper.make_node(
                    "Slice",
                    [
                        "MatMul_0_qkv_out",
                        "MatMul_0_v_start_index",
                        "MatMul_0_end_of_qkv_index",
                        "MatMul_0_qkv_last_axis",
                    ],
                    ["MatMul_0_v_out"],
                    "Slice_2",
                ),
                helper.make_node(
                    "MultiHeadAttention",
                    [
                        "MatMul_0_q_out",
                        "MatMul_0_k_out",
                        "MatMul_0_v_out",
                        "Attention_0_qkv_bias",
                        "",
                        "",
                        "past_key_values.0.decoder.key",
                        "past_key_values.0.decoder.value",
                    ],
                    ["attn_output", "present.0.decoder.key", "present.0.decoder.value"],
                    "Attention_0",
                    domain="com.microsoft",
                    num_heads=num_heads,
                ),
            ]
        )
    else:
        # Create nodes for Q/K/V paths
        q_nodes = [
            helper.make_node(
                "MatMul", ["layernorm_output_to_matmul", "q_weight"], ["q_matmul_output"], "q_path_matmul"
            ),
            helper.make_node("Add", ["q_bias", "q_matmul_output"], ["q_add_output"], "q_path_add"),
            helper.make_node("Mul", ["q_add_output", "q_scale"], ["q_mul_output"], "q_path_mul"),
            helper.make_node("Reshape", ["q_mul_output", "q_bsnh_reshape"], ["q_4d_bsnh"], "q_reshape_to_4d"),
            helper.make_node("Transpose", ["q_4d_bsnh"], ["q_4d_bnsh"], "q_transpose_to_bnsh", perm=[0, 2, 1, 3]),
            helper.make_node(
                "Reshape",
                ["q_4d_bnsh", "q_attn_heads_output"],
                ["q_output_(num_heads*batch_size,seq_len,head_size)"],
                "q_reshape_to_3d",
            ),
        ]
        k_nodes = [
            helper.make_node(
                "MatMul",
                ["layernorm_output_to_matmul", "k_weight"],
                ["k_matmul_output"],
                "k_path_matmul",
            ),
        ]
        if add_k:
            k_nodes.extend(
                [
                    helper.make_node("Add", ["k_bias", "k_matmul_output"], ["k_add_output"], "k_path_add"),
                    helper.make_node("Reshape", ["k_add_output", "bsnh_reshape"], ["k_4d_bsnh"], "k_reshape_to_4d"),
                ]
            )
        else:
            k_nodes.append(
                helper.make_node("Reshape", ["k_matmul_output", "kv_bsnh_reshape"], ["k_4d_bsnh"], "k_reshape_to_4d"),
            )
        k_nodes.extend(
            [
                helper.make_node("Transpose", ["k_4d_bsnh"], ["k_4d_bnsh"], "k_transpose_to_bnsh", perm=[0, 2, 1, 3]),
                helper.make_node(
                    "Concat",
                    ["past_key_values.0.decoder.key", "k_4d_bnsh"],
                    ["present.0.decoder.key"],
                    "concat_past_k_and_curr_k",
                    axis=2,
                ),
                helper.make_node(
                    "Reshape",
                    ["present.0.decoder.key", "k_attn_heads_output"],
                    ["k_output_(num_heads*batch_size,seq_len,head_size)"],
                    "k_reshape_to_3d",
                ),
                helper.make_node(
                    "Transpose",
                    ["k_output_(num_heads*batch_size,seq_len,head_size)"],
                    ["k_output_(num_heads*batch_size,head_size,seq_len)"],
                    "k_transpose_last_two_dims",
                    perm=[0, 2, 1],
                ),
            ]
        )
        v_nodes = [
            helper.make_node(
                "MatMul",
                ["layernorm_output_to_matmul", "v_weight"],
                ["v_matmul_output"],
                "v_path_matmul",
            ),
            helper.make_node("Add", ["v_bias", "v_matmul_output"], ["v_add_output"], "v_path_add"),
            helper.make_node("Reshape", ["v_add_output", "kv_bsnh_reshape"], ["v_4d_bsnh"], "v_reshape_to_4d"),
            helper.make_node("Transpose", ["v_4d_bsnh"], ["v_4d_bnsh"], "v_transpose_to_bnsh", perm=[0, 2, 1, 3]),
            helper.make_node(
                "Concat",
                ["past_key_values.0.decoder.value", "v_4d_bnsh"],
                ["present.0.decoder.value"],
                "concat_past_v_and_curr_v",
                axis=2,
            ),
            helper.make_node(
                "Reshape",
                ["present.0.decoder.value", "v_attn_heads_output"],
                ["v_output_(num_heads*batch_size,seq_len,head_size)"],
                "v_reshape_to_3d",
            ),
        ]
        nodes.extend(q_nodes)
        nodes.extend(k_nodes)
        nodes.extend(v_nodes)

        # Create nodes used with qkv concats, reshapes, and transposes
        nodes.extend(
            [
                helper.make_node("Shape", ["layernorm_output_to_matmul"], ["shape_output"], "shape"),
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
                    ["unsqueeze_attn_heads_output"],
                    "unsqueeze_num_heads",
                ),
                helper.make_node(
                    "Concat",
                    ["unsqueeze_attn_heads_output", "neg_one", "head_size"],
                    ["q_attn_heads_output"],
                    "q_num_heads",
                    axis=0,
                ),
                helper.make_node(
                    "Concat",
                    ["unsqueeze_attn_heads_output", "neg_one", "head_size"],
                    ["k_attn_heads_output"],
                    "k_num_heads",
                    axis=0,
                ),
                helper.make_node(
                    "Concat",
                    ["unsqueeze_attn_heads_output", "neg_one", "head_size"],
                    ["v_attn_heads_output"],
                    "v_num_heads",
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
                helper.make_node(
                    "Constant",
                    inputs=[],
                    outputs=["kv_bsnh_reshape"],
                    value=numpy_helper.from_array(
                        np.array([0, -1, num_heads, head_size], dtype="int64"), name="const_tensor"
                    ),
                ),
            ]
        )

        # Create nodes used with Q x K' and softmax(Q x K') x V
        nodes.extend(
            [
                helper.make_node("Gather", ["shape_output", "idx_1"], ["gather_1_output"], "gather_1", axis=0),
                helper.make_node(
                    "Unsqueeze",
                    ["gather_0_output", "unsqueeze_axes_input"],
                    ["unsqueeze_0_output"],
                    "unsqueeze_0",
                ),
                helper.make_node(
                    "Unsqueeze",
                    ["gather_1_output", "unsqueeze_axes_input"],
                    ["unsqueeze_1_output"],
                    "unsqueeze_1",
                ),
                helper.make_node(
                    "Concat",
                    ["unsqueeze_0_output", "num_heads", "unsqueeze_1_output", "head_size"],
                    ["bnsh_format"],
                    axis=0,
                ),
                helper.make_node(
                    "Concat",
                    ["unsqueeze_0_output", "unsqueeze_1_output", "hidden_size"],
                    ["bsd_format"],
                    axis=0,
                ),
            ]
        )

        # Create nodes for computing softmax(Q x K') x V
        nodes.extend(
            [
                helper.make_node(
                    "MatMul",
                    [
                        "q_output_(num_heads*batch_size,seq_len,head_size)",
                        "k_output_(num_heads*batch_size,head_size,seq_len)",
                    ],
                    ["qk_output_(num_heads*batch_size,seq_len,seq_len)"],
                    "matmul_qk",
                ),
                helper.make_node(
                    "Softmax",
                    ["qk_output_(num_heads*batch_size,seq_len,seq_len)"],
                    ["softmax_output"],
                    "softmax_qk",
                    axis=2,
                ),
                helper.make_node(
                    "MatMul",
                    ["softmax_output", "v_output_(num_heads*batch_size,seq_len,head_size)"],
                    ["qkv_output_(num_heads*batch_size,seq_len,head_size)"],
                    "matmul_qkv",
                ),
                helper.make_node(
                    "Reshape",
                    ["qkv_output_(num_heads*batch_size,seq_len,head_size)", "bnsh_format"],
                    ["qkv_bnsh"],
                    "reshape_qkv_to_bnsh",
                ),
                helper.make_node("Transpose", ["qkv_bnsh"], ["qkv_bsnh"], "transpose_bnsh_to_bsnh", perm=[0, 2, 1, 3]),
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
            "layernorm_output_to_skiplayernorm",
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
            "layernorm_output_to_skiplayernorm",
            "matmul_after_attn_output",
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
    q_weight, q_weight_data = get_tensor_and_weight("q_weight", [hidden_size, hidden_size])
    q_bias, q_bias_data = get_tensor_and_weight("q_bias", [hidden_size])
    k_weight, k_weight_data = get_tensor_and_weight("k_weight", [hidden_size, hidden_size])
    k_bias, k_bias_data = get_tensor_and_weight("k_bias", [hidden_size], zeros=(not add_k))
    v_weight, v_weight_data = get_tensor_and_weight("v_weight", [hidden_size, hidden_size])
    v_bias, v_bias_data = get_tensor_and_weight("v_bias", [hidden_size])
    qkv_weight = helper.make_tensor(
        "MatMul_0_qkv_weight",
        TensorProto.FLOAT,
        [hidden_size, 3 * hidden_size],
        q_weight_data + k_weight_data + v_weight_data,
    )
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

    if fused:
        # Add packed QKV weight tensor as initializer
        initializers.append(qkv_weight)

        # Add Slice indices as initializers
        initializers.extend(
            [
                helper.make_tensor(name="MatMul_0_q_start_index", data_type=TensorProto.INT64, dims=[1], vals=[0]),
                helper.make_tensor(
                    name="MatMul_0_k_start_index", data_type=TensorProto.INT64, dims=[1], vals=[hidden_size]
                ),
                helper.make_tensor(
                    name="MatMul_0_v_start_index", data_type=TensorProto.INT64, dims=[1], vals=[2 * hidden_size]
                ),
                helper.make_tensor(
                    name="MatMul_0_end_of_qkv_index", data_type=TensorProto.INT64, dims=[1], vals=[3 * hidden_size]
                ),
                helper.make_tensor(name="MatMul_0_qkv_last_axis", data_type=TensorProto.INT64, dims=[1], vals=[-1]),
            ]
        )

        # Add packed QKV bias tensor as initializer
        initializers.append(qkv_bias)
    else:
        # Add Q/K/V weight tensors as initializers
        initializers.extend([q_weight, k_weight, v_weight])

        if add_k:
            initializers.extend([q_bias, k_bias, v_bias])
        else:
            initializers.extend([q_bias, v_bias])

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
    graph = helper.make_graph(
        nodes, "whisper_decoder_with_past_self_mha_graph", inputs, outputs, initializers, doc_string="whisper"
    )
    opsetid = helper.make_opsetid("ai.onnx", min(onnx.defs.onnx_opset_version(), 16))
    return helper.make_model(graph, opset_imports=(opsetid,))


def create_whisper_decoder_with_past_multihead_cross_attention(
    hidden_size=768, num_heads=12, epsilon=0.000009999999747378752, fused=False
):
    # Get head size and ensure head size is an integer
    assert hidden_size % num_heads == 0
    head_size = hidden_size // num_heads

    # Construct input and output nodes
    inputs = [
        helper.make_tensor_value_info("input_0", TensorProto.FLOAT, ["batch_size", 1500, hidden_size]),
        helper.make_tensor_value_info(
            "past_key_values.0.encoder.key", TensorProto.FLOAT, ["batch_size", num_heads, "past_seq_len", head_size]
        ),
        helper.make_tensor_value_info(
            "past_key_values.0.encoder.value", TensorProto.FLOAT, ["batch_size", num_heads, "past_seq_len", head_size]
        ),
    ]
    outputs = [
        helper.make_tensor_value_info("output_0", TensorProto.FLOAT, ["batch_size", 1500, hidden_size]),
        helper.make_tensor_value_info("output_1", TensorProto.FLOAT, ["batch_size", "sequence_length", hidden_size]),
    ]

    # Create SkipLayerNorm (since there's no Add + LayerNorm variant for this attention subgraph)
    nodes = [
        helper.make_node(
            "SkipLayerNormalization",
            ["input_0", "input_0", "layernorm_weight", "layernorm_bias"],
            ["layernorm_output_to_matmul", "", "", "layernorm_output_to_skiplayernorm"],
            "skiplayernorm",
            domain="com.microsoft",
            epsilon=epsilon,
        )
    ]

    if fused:
        nodes.extend(
            [
                helper.make_node(
                    "MatMul", ["layernorm_output_to_matmul", "q_weight"], ["q_matmul_output"], "q_path_matmul"
                ),
                helper.make_node(
                    "MultiHeadAttention",
                    [
                        "q_matmul_output",
                        "past_key_values.0.encoder.key",
                        "past_key_values.0.encoder.value",
                        "Attention_0_qkv_bias",
                    ],
                    ["attn_output"],
                    "Attention_0",
                    domain="com.microsoft",
                    num_heads=num_heads,
                ),
            ]
        )
    else:
        # Create nodes for Q/K/V paths
        q_nodes = [
            helper.make_node(
                "MatMul", ["layernorm_output_to_matmul", "q_weight"], ["q_matmul_output"], "q_path_matmul"
            ),
            helper.make_node("Add", ["q_bias", "q_matmul_output"], ["q_add_output"], "q_path_add"),
            helper.make_node("Mul", ["q_add_output", "q_scale"], ["q_mul_output"], "q_path_mul"),
            helper.make_node("Reshape", ["q_mul_output", "q_bsnh_reshape"], ["q_4d_bsnh"], "q_reshape_to_4d"),
            helper.make_node("Transpose", ["q_4d_bsnh"], ["q_4d_bnsh"], "q_transpose_to_bnsh", perm=[0, 2, 1, 3]),
            helper.make_node(
                "Reshape",
                ["q_4d_bnsh", "q_attn_heads_output"],
                ["q_output_(num_heads*batch_size,seq_len,head_size)"],
                "q_reshape_to_3d",
            ),
        ]
        k_nodes = [
            helper.make_node(
                "Reshape",
                ["past_key_values.0.encoder.key", "k_attn_heads_output"],
                ["k_output_(num_heads*batch_size,seq_len,head_size)"],
                "k_reshape_to_3d",
            ),
            helper.make_node(
                "Transpose",
                ["k_output_(num_heads*batch_size,seq_len,head_size)"],
                ["k_output_(num_heads*batch_size,head_size,seq_len)"],
                "k_transpose_last_two_dims",
                perm=[0, 2, 1],
            ),
        ]
        v_nodes = [
            helper.make_node(
                "Reshape",
                ["past_key_values.0.encoder.value", "v_attn_heads_output"],
                ["v_output_(num_heads*batch_size,seq_len,head_size)"],
                "v_reshape_to_3d",
            ),
        ]
        nodes.extend(q_nodes)
        nodes.extend(k_nodes)
        nodes.extend(v_nodes)

        # Create nodes used with qkv concats, reshapes, and transposes
        nodes.extend(
            [
                helper.make_node("Shape", ["layernorm_output_to_matmul"], ["shape_output"], "shape"),
                helper.make_node("Gather", ["shape_output", "idx_0"], ["gather_0_output"], "gather_0", axis=0),
                helper.make_node(
                    "Mul", ["gather_0_output", "num_heads_int"], ["mul_attn_heads_output"], "mul_num_heads"
                ),
                helper.make_node(
                    "Unsqueeze",
                    ["mul_attn_heads_output", "unsqueeze_axes_input"],
                    ["unsqueeze_attn_heads_output"],
                    "unsqueeze_num_heads",
                ),
                helper.make_node(
                    "Concat",
                    ["unsqueeze_attn_heads_output", "neg_one", "head_size"],
                    ["q_attn_heads_output"],
                    "q_num_heads",
                    axis=0,
                ),
                helper.make_node(
                    "Concat",
                    ["unsqueeze_attn_heads_output", "neg_one", "head_size"],
                    ["k_attn_heads_output"],
                    "k_num_heads",
                    axis=0,
                ),
                helper.make_node(
                    "Concat",
                    ["unsqueeze_attn_heads_output", "neg_one", "head_size"],
                    ["v_attn_heads_output"],
                    "v_num_heads",
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

        # Create nodes used with Q x K' and softmax(Q x K') x V
        nodes.extend(
            [
                helper.make_node("Gather", ["shape_output", "idx_1"], ["gather_1_output"], "gather_1", axis=0),
                helper.make_node(
                    "Unsqueeze", ["gather_0_output", "unsqueeze_axes_input"], ["unsqueeze_0_output"], "unsqueeze_0"
                ),
                helper.make_node(
                    "Unsqueeze", ["gather_1_output", "unsqueeze_axes_input"], ["unsqueeze_1_output"], "unsqueeze_1"
                ),
                helper.make_node(
                    "Concat",
                    ["unsqueeze_0_output", "num_heads", "unsqueeze_1_output", "head_size"],
                    ["bnsh_format"],
                    axis=0,
                ),
                helper.make_node(
                    "Concat",
                    ["unsqueeze_0_output", "unsqueeze_1_output", "hidden_size"],
                    ["bsd_format"],
                    axis=0,
                ),
            ]
        )

        # Create nodes for computing softmax(Q x K') x V
        nodes.extend(
            [
                helper.make_node(
                    "MatMul",
                    [
                        "q_output_(num_heads*batch_size,seq_len,head_size)",
                        "k_output_(num_heads*batch_size,head_size,seq_len)",
                    ],
                    ["qk_output_(num_heads*batch_size,seq_len,seq_len)"],
                    "matmul_qk",
                ),
                helper.make_node(
                    "Softmax",
                    ["qk_output_(num_heads*batch_size,seq_len,seq_len)"],
                    ["softmax_output"],
                    "softmax_qk",
                    axis=2,
                ),
                helper.make_node(
                    "MatMul",
                    ["softmax_output", "v_output_(num_heads*batch_size,seq_len,head_size)"],
                    ["qkv_output_(num_heads*batch_size,seq_len,head_size)"],
                    "matmul_qkv",
                ),
                helper.make_node(
                    "Reshape",
                    ["qkv_output_(num_heads*batch_size,seq_len,head_size)", "bnsh_format"],
                    ["qkv_bnsh"],
                    "reshape_qkv_to_bnsh",
                ),
                helper.make_node("Transpose", ["qkv_bnsh"], ["qkv_bsnh"], "transpose_bnsh_to_bsnh", perm=[0, 2, 1, 3]),
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
            "layernorm_output_to_skiplayernorm",
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
            "layernorm_output_to_skiplayernorm",
            "matmul_after_attn_output",
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
    q_weight, q_weight_data = get_tensor_and_weight("q_weight", [hidden_size, hidden_size])
    q_bias, q_bias_data = get_tensor_and_weight("q_bias", [hidden_size])
    k_bias, k_bias_data = get_tensor_and_weight("k_bias", [hidden_size], zeros=True)
    v_bias, v_bias_data = get_tensor_and_weight("v_bias", [hidden_size], zeros=True)
    qkv_bias = helper.make_tensor(
        "Attention_0_qkv_bias", TensorProto.FLOAT, [3 * hidden_size], q_bias_data + k_bias_data + v_bias_data
    )
    initializers = [
        float_tensor("layernorm_weight", [hidden_size]),
        float_tensor("layernorm_bias", [hidden_size]),
        float_tensor("matmul_after_attn_initializer", [hidden_size, hidden_size]),
        float_tensor("add_after_attn_initializer", [hidden_size]),
        q_weight,
    ]

    if fused:
        # Add packed QKV bias tensor as initializer
        initializers.append(qkv_bias)
    else:
        # Add Q bias tensor as initializer
        initializers.append(q_bias)

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
    graph = helper.make_graph(
        nodes, "whisper_decoder_with_past_cross_mha_graph", inputs, outputs, initializers, doc_string="whisper"
    )
    opsetid = helper.make_opsetid("ai.onnx", min(onnx.defs.onnx_opset_version(), 16))
    return helper.make_model(graph, opset_imports=(opsetid,))


if __name__ == "__main__":
    np.random.seed(2)
    num_heads = 4
    hidden_size = 64

    model = create_whisper_encoder_attention(num_heads=num_heads, hidden_size=hidden_size)
    onnx.save(model, "whisper_encoder_attention_sln.onnx")

    model = create_whisper_encoder_attention(num_heads=num_heads, hidden_size=hidden_size, fused=True)
    onnx.save(model, "./test_data/models/whisper/encoder_attention_with_sln_fused.onnx")

    model = create_whisper_decoder_attention(num_heads=num_heads, hidden_size=hidden_size)
    onnx.save(model, "whisper_decoder_attention_sln.onnx")

    model = create_whisper_decoder_attention(num_heads=num_heads, hidden_size=hidden_size, fused=True)
    onnx.save(model, "./test_data/models/whisper/decoder_attention_with_sln_fused.onnx")

    model = create_whisper_decoder_multihead_attention(num_heads=num_heads, hidden_size=hidden_size)
    onnx.save(model, "whisper_decoder_mha.onnx")

    model = create_whisper_decoder_multihead_attention(num_heads=num_heads, hidden_size=hidden_size, fused=True)
    onnx.save(model, "./test_data/models/whisper/decoder_mha_fused.onnx")

    model = create_whisper_decoder_with_past_multihead_self_attention(num_heads=num_heads, hidden_size=hidden_size)
    onnx.save(model, "whisper_decoder_with_past_self_mha.onnx")

    model = create_whisper_decoder_with_past_multihead_self_attention(
        num_heads=num_heads, hidden_size=hidden_size, fused=True
    )
    onnx.save(model, "./test_data/models/whisper/decoder_with_past_self_mha_fused.onnx")

    model = create_whisper_decoder_with_past_multihead_cross_attention(num_heads=num_heads, hidden_size=hidden_size)
    onnx.save(model, "whisper_decoder_with_past_cross_mha.onnx")

    model = create_whisper_decoder_with_past_multihead_cross_attention(
        num_heads=num_heads, hidden_size=hidden_size, fused=True
    )
    onnx.save(model, "./test_data/models/whisper/decoder_with_past_cross_mha_fused.onnx")
