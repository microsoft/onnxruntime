# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.  See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

import math
from typing import List  # noqa: F401

import numpy
import onnx
from bert_model_generator import float_tensor, reverse_if
from onnx import TensorProto, helper
from packaging import version


def create_gpt2_attention(hidden_size=64, num_heads=4, max_seq_len=32, switch_add_inputs=False):
    # unsqueeze in opset version 13 has two inputs (axis is moved from attribute to input).
    is_opset_13_or_newer = version.parse(onnx.__version__) >= version.parse("1.8.0")

    # nodes in attention subgraph
    nodes = [
        helper.make_node("Add", ["input_1", "input_2"], ["layernorm_input"], "add_layernorm"),
        helper.make_node(
            "LayerNormalization",
            ["layernorm_input", "layer_norm_weight", "layer_norm_bias"],
            ["layernorm_out"],
            "layernorm",
            epsion=0.000009999999747378752,
        ),
        # fully connection nodes
        helper.make_node(
            "MatMul",
            ["layernorm_out", "matmul_fc_weight"],
            ["matmul_fc_out"],
            "matmul_fc",
        ),
        helper.make_node(
            "Add",
            reverse_if(["matmul_fc_out", "add_fc_weight"], switch_add_inputs),
            ["fc_out"],
            "add_fc",
        ),
        (
            helper.make_node("Split", ["fc_out", "split_q_k_v"], ["q", "k", "v"], "split_qkv", axis=2)
            if is_opset_13_or_newer
            else helper.make_node(
                "Split",
                ["fc_out"],
                ["q", "k", "v"],
                "split_qkv",
                axis=2,
                split=[hidden_size, hidden_size, hidden_size],
            )
        ),
        # q nodes
        helper.make_node("Reshape", ["q", "reshape_x_shape"], ["reshape_q_out"], "reshape_q"),
        helper.make_node(
            "Transpose",
            ["reshape_q_out"],
            ["transpose_q_out"],
            "transpose_q",
            perm=[0, 2, 1, 3],
        ),
        # k nodes
        helper.make_node("Reshape", ["k", "reshape_x_shape"], ["reshape_k_out"], "reshape_k"),
        helper.make_node(
            "Transpose",
            ["reshape_k_out"],
            ["transpose_k_out"],
            "transpose_k",
            perm=[0, 2, 1, 3],
        ),
        # v nodes
        helper.make_node("Reshape", ["v", "reshape_x_shape"], ["reshape_v_out"], "reshape_v"),
        helper.make_node(
            "Transpose",
            ["reshape_v_out"],
            ["transpose_v_out"],
            "transpose_v",
            perm=[0, 2, 1, 3],
        ),
        # past
        (
            helper.make_node("Split", ["past", "split_1_1"], ["split_k", "split_v"], "split_past", axis=0)
            if is_opset_13_or_newer
            else helper.make_node(
                "Split",
                ["past"],
                ["split_k", "split_v"],
                "split_past",
                axis=0,
                split=[1, 1],
            )
        ),
        (
            helper.make_node("Squeeze", ["split_k", "axes_0"], ["past_k"], "squeeze_past_k")
            if is_opset_13_or_newer
            else helper.make_node("Squeeze", ["split_k"], ["past_k"], "squeeze_past_k", axes=[0])
        ),
        helper.make_node(
            "Concat",
            ["past_k", "transpose_k_out"],
            ["concat_k_out"],
            "concat_k",
            axis=-2,
        ),
        helper.make_node(
            "Transpose",
            ["concat_k_out"],
            ["concat_k_transpose_out"],
            "transpose_concat_k",
            perm=[0, 1, 3, 2],
        ),
        (
            helper.make_node("Squeeze", ["split_v", "axes_0"], ["past_v"], "squeeze_past_v")
            if is_opset_13_or_newer
            else helper.make_node("Squeeze", ["split_v"], ["past_v"], "squeeze_past_v", axes=[0])
        ),
        helper.make_node(
            "Concat",
            ["past_v", "transpose_v_out"],
            ["concat_v_out"],
            "concat_v",
            axis=-2,
        ),
        # present
        (
            helper.make_node(
                "Unsqueeze",
                ["concat_k_out", "axes_0"],
                ["concat_k_unsqueeze_out"],
                "concat_k_unsqueeze",
            )
            if is_opset_13_or_newer
            else helper.make_node(
                "Unsqueeze",
                ["concat_k_out"],
                ["concat_k_unsqueeze_out"],
                "concat_k_unsqueeze",
                axes=[0],
            )
        ),
        (
            helper.make_node(
                "Unsqueeze",
                ["concat_v_out", "axes_0"],
                ["concat_v_unsqueeze_out"],
                "concat_v_unsqueeze",
            )
            if is_opset_13_or_newer
            else helper.make_node(
                "Unsqueeze",
                ["concat_v_out"],
                ["concat_v_unsqueeze_out"],
                "concat_v_unsqueeze",
                axes=[0],
            )
        ),
        helper.make_node(
            "Concat",
            ["concat_k_unsqueeze_out", "concat_v_unsqueeze_out"],
            ["present"],
            "concat_present",
            axis=0,
        ),
        helper.make_node("Shape", ["transpose_q_out"], ["transpose_q_shape_out"], "transpose_q_shape"),
        helper.make_node(
            "Slice",
            ["transpose_q_shape_out", "starts_n2", "ends_n1", "axes_0"],
            ["transpose_q_shape_slice_out"],
            "transpose_q_shape_slice",
        ),
        (
            helper.make_node(
                "Squeeze",
                ["transpose_q_shape_slice_out", "axes_0"],
                ["transpose_q_shape_slice_squeeze_out"],
                "transpose_q_shape_slice_squeeze",
            )
            if is_opset_13_or_newer
            else helper.make_node(
                "Squeeze",
                ["transpose_q_shape_slice_out"],
                ["transpose_q_shape_slice_squeeze_out"],
                "transpose_q_shape_slice_squeeze",
                axes=[0],
            )
        ),
        helper.make_node("Shape", ["concat_k_out"], ["concat_k_shape_out"], "concat_k_shape"),
        helper.make_node(
            "Slice",
            ["concat_k_shape_out", "starts_n2", "ends_n1", "axes_0"],
            ["concat_k_shape_slice_out"],
            "concat_k_shape_slice",
        ),
        (
            helper.make_node(
                "Squeeze",
                ["concat_k_shape_slice_out", "axes_0"],
                ["concat_k_shape_slice_squeeze_out"],
                "concat_k_shape_slice_squeeze",
            )
            if is_opset_13_or_newer
            else helper.make_node(
                "Squeeze",
                ["concat_k_shape_slice_out"],
                ["concat_k_shape_slice_squeeze_out"],
                "concat_k_shape_slice_squeeze",
                axes=[0],
            )
        ),
        helper.make_node(
            "Sub",
            ["concat_k_shape_slice_squeeze_out", "transpose_q_shape_slice_squeeze_out"],
            ["sub_out"],
            "sub",
        ),
        (
            helper.make_node("Unsqueeze", ["sub_out", "axes_0"], ["sub_unsqueeze_out"], "sub_unsqueeze")
            if is_opset_13_or_newer
            else helper.make_node("Unsqueeze", ["sub_out"], ["sub_unsqueeze_out"], "sub_unsqueeze", axes=[0])
        ),
        (
            helper.make_node(
                "Unsqueeze",
                ["concat_k_shape_slice_squeeze_out", "axes_0"],
                ["concat_k_shape_slice_squeeze_unsqueeze_out"],
                "concat_k_shape_slice_squeeze_unsqueeze",
            )
            if is_opset_13_or_newer
            else helper.make_node(
                "Unsqueeze",
                ["concat_k_shape_slice_squeeze_out"],
                ["concat_k_shape_slice_squeeze_unsqueeze_out"],
                "concat_k_shape_slice_squeeze_unsqueeze",
                axes=[0],
            )
        ),
        helper.make_node(
            "Slice",
            [
                "undir_mask",
                "sub_unsqueeze_out",
                "concat_k_shape_slice_squeeze_unsqueeze_out",
                "axes_2",
                "steps_1",
            ],
            ["undir_mask_slice_out"],
            "undir_mask_slice",
        ),
        helper.make_node(
            "Slice",
            [
                "undir_mask_slice_out",
                "starts_0",
                "concat_k_shape_slice_squeeze_unsqueeze_out",
                "axes_3",
                "steps_1",
            ],
            ["mask_slice_slice_out"],
            "mask_slice_slice",
        ),
        helper.make_node(
            "Cast",
            ["mask_slice_slice_out"],
            ["undir_mask_out"],
            "undir_mask_cast",
            to=9,
        ),
        # mask nodes
        helper.make_node(
            "Reshape",
            ["input_mask", "input_mask_shape"],
            ["input_mask_reshape_out"],
            "input_mask_reshape",
        ),
        (
            helper.make_node(
                "Unsqueeze",
                ["input_mask_reshape_out", "axes_1"],
                ["unsqueeze0_out"],
                "unsqueeze0",
            )
            if is_opset_13_or_newer
            else helper.make_node(
                "Unsqueeze",
                ["input_mask_reshape_out"],
                ["unsqueeze0_out"],
                "unsqueeze0",
                axes=[1],
            )
        ),
        (
            helper.make_node("Unsqueeze", ["unsqueeze0_out", "axes_2"], ["unsqueeze1_out"], "unsqueeze1")
            if is_opset_13_or_newer
            else helper.make_node("Unsqueeze", ["unsqueeze0_out"], ["unsqueeze1_out"], "unsqueeze1", axes=[2])
        ),
        helper.make_node("Sub", ["sub_weight", "unsqueeze1_out"], ["mask_sub_out"], "sub_mask"),
        helper.make_node("Mul", ["mask_sub_out", "mul_weight"], ["mul_mask_out"], "mul_mask"),
        # qk nodes
        helper.make_node(
            "MatMul",
            ["transpose_q_out", "concat_k_transpose_out"],
            ["qk_out"],
            "matmul_qk",
        ),
        helper.make_node("Div", ["qk_out", "div_weight"], ["qk_norm_out"], "qk_norm"),
        helper.make_node(
            "Where",
            ["undir_mask_out", "qk_norm_out", "where_weight"],
            ["where_out"],
            "where",
        ),
        helper.make_node(
            "Add",
            reverse_if(["where_out", "mul_mask_out"], switch_add_inputs),
            ["add_mask_out"],
            "add_mask",
        ),
        helper.make_node("Softmax", ["add_mask_out"], ["softmax_out"], "softmax", axis=3),
        # qkv nodes
        helper.make_node(
            "MatMul",
            ["softmax_out", "concat_v_out"],
            ["matmul_qkv_1_out"],
            "matmul_qk_v",
        ),
        helper.make_node(
            "Transpose",
            ["matmul_qkv_1_out"],
            ["transpose_qkv_out"],
            "transpose_qkv",
            perm=[0, 2, 1, 3],
        ),
        helper.make_node(
            "Reshape",
            ["transpose_qkv_out", "reshape_weight_qkv"],
            ["reshape_qkv_out"],
            "reshape_qkv",
        ),
        helper.make_node("Shape", ["reshape_qkv_out"], ["qkv_shape"], "shape_qkv"),
        helper.make_node(
            "Slice",
            ["qkv_shape", "starts_n1", "ends_inf", "axes_0"],
            ["qkv_shape_slice_out"],
            "qkv_shape_slice",
        ),
        (
            helper.make_node(
                "Squeeze",
                ["qkv_shape_slice_out", "axes_0"],
                ["qkv_shape_slice_squeeze_out"],
                "qkv_shape_slice_squeeze",
            )
            if is_opset_13_or_newer
            else helper.make_node(
                "Squeeze",
                ["qkv_shape_slice_out"],
                ["qkv_shape_slice_squeeze_out"],
                "qkv_shape_slice_squeeze",
                axes=[0],
            )
        ),
        (
            helper.make_node(
                "Unsqueeze",
                ["qkv_shape_slice_squeeze_out", "axes_0"],
                ["qkv_shape_slice_squeeze_unsqueeze_out"],
                "qkv_shape_slice_squeeze_unsqueeze",
            )
            if is_opset_13_or_newer
            else helper.make_node(
                "Unsqueeze",
                ["qkv_shape_slice_squeeze_out"],
                ["qkv_shape_slice_squeeze_unsqueeze_out"],
                "qkv_shape_slice_squeeze_unsqueeze",
                axes=[0],
            )
        ),
        helper.make_node(
            "Concat",
            ["concat_n1", "qkv_shape_slice_squeeze_unsqueeze_out"],
            ["qkv_shape_slice_squeeze_unsqueeze_concat_out"],
            "qkv_shape_slice_squeeze_unsqueeze_concat",
            axis=0,
        ),
        helper.make_node(
            "Reshape",
            ["reshape_qkv_out", "qkv_shape_slice_squeeze_unsqueeze_concat_out"],
            ["qkv_reshape_out"],
            "qkv_reshape",
        ),
        helper.make_node(
            "Gemm",
            ["qkv_reshape_out", "gemm_weight", "gemm_bias"],
            ["gemm_out"],
            "gemm",
            alpha=1.0,
            beta=1.0,
            transA=0,
            transB=0,
        ),
        helper.make_node(
            "Gather",
            ["qkv_shape", "indices_1"],
            ["qkv_shape_1"],
            "shape_qkv_gather_1",
            axis=0,
        ),
        helper.make_node(
            "Gather",
            ["qkv_shape", "indices_0"],
            ["qkv_shape_0"],
            "shape_qkv_gather_0",
            axis=0,
        ),
        (
            helper.make_node(
                "Unsqueeze",
                ["qkv_shape_1", "axes_0"],
                ["qkv_shape_1_unsqueeze_out"],
                "qkv_shape_1_unsqueeze",
            )
            if is_opset_13_or_newer
            else helper.make_node(
                "Unsqueeze",
                ["qkv_shape_1"],
                ["qkv_shape_1_unsqueeze_out"],
                "qkv_shape_1_unsqueeze",
                axes=[0],
            )
        ),
        (
            helper.make_node(
                "Unsqueeze",
                ["qkv_shape_0", "axes_0"],
                ["qkv_shape_0_unsqueeze_out"],
                "qkv_shape_0_unsqueeze",
            )
            if is_opset_13_or_newer
            else helper.make_node(
                "Unsqueeze",
                ["qkv_shape_0"],
                ["qkv_shape_0_unsqueeze_out"],
                "qkv_shape_0_unsqueeze",
                axes=[0],
            )
        ),
        helper.make_node(
            "Concat",
            ["qkv_shape_0_unsqueeze_out", "qkv_shape_1_unsqueeze_out", "qkv_hidden"],
            ["shape_qkv_concat_out"],
            "shape_qkv_concat",
            axis=0,
        ),
        helper.make_node(
            "Reshape",
            ["gemm_out", "shape_qkv_concat_out"],
            ["gemm_reshape_out"],
            "gemm_reshape",
        ),
        helper.make_node(
            "Add",
            reverse_if(["gemm_reshape_out", "layernorm_input"], switch_add_inputs),
            ["skip_output"],
            "add_skip",
        ),
        helper.make_node(
            "LayerNormalization",
            ["skip_output", "layer_norm_weight", "layer_norm_bias"],
            ["output"],
            "layernorm2",
            epsion=0.000009999999747378752,
        ),
    ]

    head_size = int(hidden_size // num_heads)
    unidir_mask = (
        numpy.tril(numpy.ones((max_seq_len, max_seq_len))).reshape([max_seq_len * max_seq_len]).astype(numpy.uint8)
    )
    initializers = [  # initializers
        float_tensor("layer_norm_weight", [hidden_size]),
        float_tensor("layer_norm_bias", [hidden_size]),
        float_tensor("matmul_fc_weight", [hidden_size, 3 * hidden_size]),
        float_tensor("add_fc_weight", [3 * hidden_size]),
        float_tensor("gemm_weight", [hidden_size, hidden_size]),
        float_tensor("gemm_bias", [hidden_size]),
        helper.make_tensor(
            "undir_mask",
            TensorProto.UINT8,
            [1, 1, max_seq_len, max_seq_len],
            unidir_mask.tolist(),
        ),
        helper.make_tensor("div_weight", TensorProto.FLOAT, [], [math.sqrt(head_size)]),
        helper.make_tensor("sub_weight", TensorProto.FLOAT, [], [1.0]),
        helper.make_tensor("where_weight", TensorProto.FLOAT, [], [-10000.0]),
        helper.make_tensor("mul_weight", TensorProto.FLOAT, [], [-10000]),
        helper.make_tensor("input_mask_shape", TensorProto.INT64, [2], [0, -1]),
        helper.make_tensor("starts_0", TensorProto.INT64, [1], [0]),
        helper.make_tensor("concat_n1", TensorProto.INT64, [1], [-1]),
        helper.make_tensor("starts_n1", TensorProto.INT64, [1], [-1]),
        helper.make_tensor("ends_inf", TensorProto.INT64, [1], [9223372036854775807]),
        helper.make_tensor("starts_n2", TensorProto.INT64, [1], [-2]),
        helper.make_tensor("ends_n1", TensorProto.INT64, [1], [-1]),
        helper.make_tensor("axes_0", TensorProto.INT64, [1], [0]),
        helper.make_tensor("axes_2", TensorProto.INT64, [1], [2]),
        helper.make_tensor("axes_3", TensorProto.INT64, [1], [3]),
        helper.make_tensor("steps_1", TensorProto.INT64, [1], [1]),
        helper.make_tensor("indices_0", TensorProto.INT64, [], [0]),
        helper.make_tensor("indices_1", TensorProto.INT64, [], [1]),
        helper.make_tensor("qkv_hidden", TensorProto.INT64, [1], [hidden_size]),
        helper.make_tensor("reshape_x_shape", TensorProto.INT64, [4], [0, 0, num_heads, head_size]),
        helper.make_tensor("reshape_weight_qkv", TensorProto.INT64, [3], [0, 0, hidden_size]),
    ]

    if is_opset_13_or_newer:
        initializers.append(helper.make_tensor("split_1_1", TensorProto.INT64, [2], [1, 1]))
        initializers.append(
            helper.make_tensor(
                "split_q_k_v",
                TensorProto.INT64,
                [3],
                [hidden_size, hidden_size, hidden_size],
            )
        )
        initializers.append(helper.make_tensor("axes_1", TensorProto.INT64, [1], [1]))

    graph = helper.make_graph(
        [node for node in nodes if node],
        "GPT2",  # name
        [  # inputs
            helper.make_tensor_value_info(
                "input_1",
                TensorProto.FLOAT,
                ["batch_size", "sequence_length", hidden_size],
            ),
            helper.make_tensor_value_info(
                "input_2",
                TensorProto.FLOAT,
                ["batch_size", "sequence_length", hidden_size],
            ),
            helper.make_tensor_value_info(
                "input_mask",
                TensorProto.FLOAT,
                ["batch_size", "past_sequence_length + sequence_length"],
            ),
            helper.make_tensor_value_info(
                "past",
                TensorProto.FLOAT,
                [2, "batch_size", num_heads, "past_sequence_length", head_size],
            ),
        ],
        [  # outputs
            helper.make_tensor_value_info(
                "output",
                TensorProto.FLOAT,
                ["batch_size", "sequence_length", hidden_size],
            ),
            helper.make_tensor_value_info(
                "present",
                TensorProto.FLOAT,
                [
                    2,
                    "batch_size",
                    num_heads,
                    "past_sequence_length + sequence_length",
                    head_size,
                ],
            ),
        ],
        initializers,
    )

    # Needed so that we don't see the new LayerNormalization function added in version 17.
    # TODO(https://github.com/microsoft/onnxruntime/issues/11916): Remove once fixed.
    opsetid = helper.make_opsetid("ai.onnx", min(onnx.defs.onnx_opset_version(), 16))
    return helper.make_model(graph, opset_imports=(opsetid,))


def create_gpt2_embedlayer(
    pos_embed=100,
    word_embed=101,
    token_embed=10,
    hidden_size=768,
    attn_hidden_dim=256,
    num_heads=4,
    epsilon=0.1,
    one_attention_node=False,
    has_skip_layer_norm=True,
    output_embedding_sum=False,
):
    # Construct input and output nodes
    inputs = [
        helper.make_tensor_value_info("ids", TensorProto.INT32, ["batch_size", "sequence_length"]),
    ]
    outputs = [
        helper.make_tensor_value_info("output_0", TensorProto.FLOAT, ["batch_size", "sequence_length", hidden_size])
    ]

    if output_embedding_sum:
        outputs.append(
            helper.make_tensor_value_info(
                "embedding_sum", TensorProto.FLOAT, ["batch_size", "sequence_length", hidden_size]
            )
        )

    # Construct graph nodes
    embed_layernorm_nodes = [
        helper.make_node("Gather", ["word_embeddings_weight", "ids"], ["gather_0_out"], "gather_word_embeddings"),
        helper.make_node("Gather", ["pos_embeddings_weight", "ids"], ["gather_1_out"], "gather_position_embeddings"),
        helper.make_node("Add", ["gather_0_out", "gather_1_out"], ["add_0_out"], "add_before_layernorm"),
        helper.make_node("Gather", ["token_embeddings_weight", "ids"], ["gather_2_out"], "gather_token_embeddings"),
    ]

    if has_skip_layer_norm:
        embed_layernorm_nodes.append(
            helper.make_node(
                "SkipLayerNormalization",
                ["add_0_out", "gather_2_out", "layernorm_weight", "layernorm_bias"],
                ["skip_layernorm_out"] if not output_embedding_sum else ["skip_layernorm_out", "", "", "embedding_sum"],
                "skip_layernorm",
                domain="com.microsoft",
                epsilon=epsilon,
            )
        )
    else:
        embed_layernorm_nodes.append(
            helper.make_node("Add", ["add_0_out", "gather_2_out"], ["embedding_sum"], "embedding_sum")
        )

        embed_layernorm_nodes.append(
            helper.make_node(
                "LayerNormalization",
                ["embedding_sum", "layernorm_weight", "layernorm_bias"],
                ["skip_layernorm_out"],
                "layernorm",
                epsilon=epsilon,
            )
        )

    attention_nodes = (
        [
            helper.make_node("MatMul", ["skip_layernorm_out", "q_weight"], ["q_out"], "q_attn"),
            helper.make_node("MatMul", ["skip_layernorm_out", "k_weight"], ["k_out"], "k_attn"),
            helper.make_node("MatMul", ["skip_layernorm_out", "v_weight"], ["v_out"], "v_attn"),
            helper.make_node("Add", ["q_out", "k_out"], ["qk_out"], "qk_attn"),
            helper.make_node("Add", ["qk_out", "v_out"], ["qkv_out"], "qkv_attn"),
        ]
        if not one_attention_node
        else [
            helper.make_node(
                "Attention",
                ["skip_layernorm_out", "qkv_weight", "qkv_bias", ""],
                ["attn_out"],
                "qkv_attn",
                domain="com.microsoft",
                num_heads=num_heads,
            ),
            helper.make_node(
                "MatMul",
                ["attn_out", "fix_hidden_size"],
                ["qkv_out"],
                "matmul_after_attn",
            ),
        ]
    )
    nodes = [
        helper.make_node(
            "SkipLayerNormalization",
            ["skip_layernorm_out", "qkv_out", "layernorm_weight", "layernorm_bias", "dense_bias"],
            ["output_0"],
            "attn_skip_layernorm",
            domain="com.microsoft",
            epsilon=epsilon,
        ),
    ]
    nodes.extend(embed_layernorm_nodes)
    nodes.extend(attention_nodes)

    # Construct data initializers for graph nodes
    embed_layernorm_initializers = [
        helper.make_tensor(
            "word_embeddings_weight",
            TensorProto.FLOAT,
            [word_embed, hidden_size],
            [(i + 1) / (word_embed * hidden_size) for i in range(word_embed * hidden_size)],
        ),
        helper.make_tensor(
            "pos_embeddings_weight",
            TensorProto.FLOAT,
            [pos_embed, hidden_size],
            [(i + 2) / (pos_embed * hidden_size) for i in range(pos_embed * hidden_size)],
        ),
        helper.make_tensor(
            "token_embeddings_weight",
            TensorProto.FLOAT,
            [token_embed, hidden_size],
            [(i + 3) / (token_embed * hidden_size) for i in range(token_embed * hidden_size)],
        ),
        helper.make_tensor(
            "layernorm_weight", TensorProto.FLOAT, [hidden_size], [(i + 4) / hidden_size for i in range(hidden_size)]
        ),
        helper.make_tensor(
            "layernorm_bias", TensorProto.FLOAT, [hidden_size], [(i + 5) / hidden_size for i in range(hidden_size)]
        ),
    ]
    attention_initializers = (
        [
            helper.make_tensor(
                "q_weight",
                TensorProto.FLOAT,
                [hidden_size, hidden_size],
                [(i + 6) / (hidden_size * hidden_size) for i in range(hidden_size * hidden_size)],
            ),
            helper.make_tensor(
                "k_weight",
                TensorProto.FLOAT,
                [hidden_size, hidden_size],
                [(i + 7) / (hidden_size * hidden_size) for i in range(hidden_size * hidden_size)],
            ),
            helper.make_tensor(
                "v_weight",
                TensorProto.FLOAT,
                [hidden_size, hidden_size],
                [(i + 8) / (hidden_size * hidden_size) for i in range(hidden_size * hidden_size)],
            ),
        ]
        if not one_attention_node
        else [
            helper.make_tensor(
                "qkv_weight",
                TensorProto.FLOAT,
                [hidden_size, hidden_size],
                [(i + 9) / (hidden_size * hidden_size) for i in range(hidden_size * hidden_size)],
            ),
            helper.make_tensor(
                "qkv_bias", TensorProto.FLOAT, [hidden_size], [(i + 10) / hidden_size for i in range(hidden_size)]
            ),
            helper.make_tensor(
                "fix_hidden_size",
                TensorProto.FLOAT,
                [attn_hidden_dim, hidden_size],
                [(i + 11) / (attn_hidden_dim * hidden_size) for i in range(attn_hidden_dim * hidden_size)],
            ),
        ]
    )
    initializers = [
        helper.make_tensor(
            "dense_bias", TensorProto.FLOAT, [hidden_size], [(i + 12) / hidden_size for i in range(hidden_size)]
        ),
    ]
    initializers.extend(embed_layernorm_initializers)
    initializers.extend(attention_initializers)

    # Construct graph
    graph = helper.make_graph(nodes, "GPT2_embedlayer_graph", inputs, outputs, initializers)
    opsetid = helper.make_opsetid("ai.onnx", min(onnx.defs.onnx_opset_version(), 16))
    return helper.make_model(graph, opset_imports=(opsetid,))


def create_gpt2_fused_embedlayer(
    pos_embed=100,
    word_embed=101,
    token_embed=10,
    hidden_size=768,
    attn_hidden_dim=256,
    num_heads=4,
    epsilon=0.1,
    one_attention_node=False,
    output_embedding_sum=False,
):
    # Construct input and output nodes
    inputs = [
        helper.make_tensor_value_info("ids", TensorProto.INT32, ["batch_size", "sequence_length"]),
    ]
    outputs = [
        helper.make_tensor_value_info("output_0", TensorProto.FLOAT, ["batch_size", "sequence_length", hidden_size])
    ]
    if output_embedding_sum:
        outputs.append(
            helper.make_tensor_value_info(
                "embedding_sum", TensorProto.FLOAT, ["batch_size", "sequence_length", hidden_size]
            )
        )

    # Construct graph nodes
    embed_layernorm_nodes = [
        helper.make_node(
            "EmbedLayerNormalization",
            [
                "ids",
                "ids",
                "word_embeddings_weight",
                "pos_embeddings_weight",
                "token_embeddings_weight",
                "layernorm_weight",
                "layernorm_bias",
                "",
                "ids",
            ],
            (
                ["EmbedLayerNormalization_0_output", "EmbedLayerNormalization_0_dummy_mask_index", "embedding_sum"]
                if output_embedding_sum
                else ["EmbedLayerNormalization_0_output", "EmbedLayerNormalization_0_dummy_mask_index"]
            ),
            "EmbedLayerNormalization_0",
            domain="com.microsoft",
            epsilon=epsilon,
        ),
    ]
    attention_nodes = (
        [
            helper.make_node("MatMul", ["EmbedLayerNormalization_0_output", "q_weight"], ["q_out"], "q_attn"),
            helper.make_node("MatMul", ["EmbedLayerNormalization_0_output", "k_weight"], ["k_out"], "k_attn"),
            helper.make_node("MatMul", ["EmbedLayerNormalization_0_output", "v_weight"], ["v_out"], "v_attn"),
            helper.make_node("Add", ["q_out", "k_out"], ["qk_out"], "qk_attn"),
            helper.make_node("Add", ["qk_out", "v_out"], ["qkv_out"], "qkv_attn"),
        ]
        if not one_attention_node
        else [
            helper.make_node(
                "Attention",
                ["EmbedLayerNormalization_0_output", "qkv_weight", "qkv_bias", ""],
                ["attn_out"],
                "qkv_attn",
                domain="com.microsoft",
                num_heads=num_heads,
            ),
            helper.make_node(
                "MatMul",
                ["attn_out", "fix_hidden_size"],
                ["qkv_out"],
                "matmul_after_attn",
            ),
        ]
    )
    nodes = [
        helper.make_node(
            "SkipLayerNormalization",
            ["EmbedLayerNormalization_0_output", "qkv_out", "layernorm_weight", "layernorm_bias", "dense_bias"],
            ["output_0"],
            "attn_skip_layernorm",
            domain="com.microsoft",
            epsilon=epsilon,
        ),
    ]
    nodes.extend(embed_layernorm_nodes)
    nodes.extend(attention_nodes)

    # Construct data initializers for graph nodes
    embed_layernorm_initializers = [
        helper.make_tensor(
            "word_embeddings_weight",
            TensorProto.FLOAT,
            [word_embed, hidden_size],
            [(i + 1) / (word_embed * hidden_size) for i in range(word_embed * hidden_size)],
        ),
        helper.make_tensor(
            "pos_embeddings_weight",
            TensorProto.FLOAT,
            [pos_embed, hidden_size],
            [(i + 2) / (pos_embed * hidden_size) for i in range(pos_embed * hidden_size)],
        ),
        helper.make_tensor(
            "token_embeddings_weight",
            TensorProto.FLOAT,
            [token_embed, hidden_size],
            [(i + 3) / (token_embed * hidden_size) for i in range(token_embed * hidden_size)],
        ),
        helper.make_tensor(
            "layernorm_weight", TensorProto.FLOAT, [hidden_size], [(i + 4) / hidden_size for i in range(hidden_size)]
        ),
        helper.make_tensor(
            "layernorm_bias", TensorProto.FLOAT, [hidden_size], [(i + 5) / hidden_size for i in range(hidden_size)]
        ),
    ]
    attention_initializers = (
        [
            helper.make_tensor(
                "q_weight",
                TensorProto.FLOAT,
                [hidden_size, hidden_size],
                [(i + 6) / (hidden_size * hidden_size) for i in range(hidden_size * hidden_size)],
            ),
            helper.make_tensor(
                "k_weight",
                TensorProto.FLOAT,
                [hidden_size, hidden_size],
                [(i + 7) / (hidden_size * hidden_size) for i in range(hidden_size * hidden_size)],
            ),
            helper.make_tensor(
                "v_weight",
                TensorProto.FLOAT,
                [hidden_size, hidden_size],
                [(i + 8) / (hidden_size * hidden_size) for i in range(hidden_size * hidden_size)],
            ),
        ]
        if not one_attention_node
        else [
            helper.make_tensor(
                "qkv_weight",
                TensorProto.FLOAT,
                [hidden_size, hidden_size],
                [(i + 9) / (hidden_size * hidden_size) for i in range(hidden_size * hidden_size)],
            ),
            helper.make_tensor(
                "qkv_bias", TensorProto.FLOAT, [hidden_size], [(i + 10) / hidden_size for i in range(hidden_size)]
            ),
            helper.make_tensor(
                "fix_hidden_size",
                TensorProto.FLOAT,
                [attn_hidden_dim, hidden_size],
                [(i + 11) / (attn_hidden_dim * hidden_size) for i in range(attn_hidden_dim * hidden_size)],
            ),
        ]
    )
    initializers = [
        helper.make_tensor(
            "dense_bias", TensorProto.FLOAT, [hidden_size], [(i + 12) / hidden_size for i in range(hidden_size)]
        ),
    ]
    initializers.extend(embed_layernorm_initializers)
    initializers.extend(attention_initializers)

    # Construct graph
    graph = helper.make_graph(nodes, "GPT2_embedlayer_graph", inputs, outputs, initializers)
    opsetid = helper.make_opsetid("ai.onnx", min(onnx.defs.onnx_opset_version(), 16))
    return helper.make_model(graph, opset_imports=(opsetid,))


if __name__ == "__main__":
    model = create_gpt2_attention()
    onnx.save(model, "gpt2_attention.onnx")

    model = create_gpt2_attention(switch_add_inputs=True)
    onnx.save(model, "gpt2_attention_add.onnx")

    model = create_gpt2_embedlayer()
    onnx.save(model, "gpt2_embedlayer.onnx")

    model = create_gpt2_fused_embedlayer()
    onnx.save(model, "./test_data/models/gpt2_embedlayer_exp.onnx")

    model = create_gpt2_embedlayer(one_attention_node=True)
    onnx.save(model, "gpt2_embedlayer_one_attn.onnx")

    model = create_gpt2_fused_embedlayer(one_attention_node=True)
    onnx.save(model, "./test_data/models/gpt2_embedlayer_one_attn_exp.onnx")

    model = create_gpt2_embedlayer(one_attention_node=True, output_embedding_sum=True)
    onnx.save(model, "gpt2_embedlayer_one_attn_output_sum.onnx")

    model = create_gpt2_fused_embedlayer(one_attention_node=True, output_embedding_sum=True)
    onnx.save(model, "./test_data/models/gpt2_embedlayer_one_attn_output_sum_exp.onnx")
