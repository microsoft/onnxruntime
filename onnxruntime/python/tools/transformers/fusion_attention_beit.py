# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
from logging import getLogger
from typing import Tuple, Union

import numpy as np
from fusion_attention import AttentionMask, FusionAttention
from fusion_options import AttentionMaskFormat
from fusion_utils import NumpyHelper
from onnx import NodeProto, TensorProto, helper
from onnx_model import OnnxModel

logger = getLogger(__name__)


class FusionAttentionBeit(FusionAttention):
    """
    Fuse Attention subgraph of Beit (from huggingface) into one Attention node.
    """

    def __init__(
        self,
        model: OnnxModel,
        hidden_size: int,
        num_heads: int,
    ):
        attention_mask = AttentionMask(model)
        attention_mask.mask_format = AttentionMaskFormat.NoMask

        super().__init__(
            model,
            hidden_size,
            num_heads,
            attention_mask,
            use_multi_head_attention=False,
            search_op_types=["SkipLayerNormalization"],
        )

    def fuse(self, normalize_node, input_name_to_nodes, output_name_to_node):
        skip_input_index = 1
        node_before_layernorm = self.model.match_parent(normalize_node, "SkipLayerNormalization", skip_input_index)
        if node_before_layernorm is not None:
            root_input = node_before_layernorm.output[0]
        else:
            node_before_layernorm = self.model.match_parent(normalize_node, "Concat", skip_input_index)
            if node_before_layernorm is None:
                return
            child = self.model.find_first_child_by_type(
                node_before_layernorm, "LayerNormalization", input_name_to_nodes, False
            )
            if child is None:
                return
            root_input = child.output[0]

        qkv_nodes = self.model.match_parent_path(
            normalize_node,
            ["Mul", "Add", "MatMul", "Reshape", "Transpose", "MatMul"],
            [1 - skip_input_index, None, None, 0, 0, 0],
        )
        if qkv_nodes is None:
            return

        (_, _, _, reshape_qkv, transpose_qkv, matmul_qkv) = qkv_nodes

        v_nodes = self.model.match_parent_path(matmul_qkv, ["Transpose", "Reshape", "Add", "MatMul"], [1, 0, 0, None])
        if v_nodes is None:
            logger.debug("fuse_attention: failed to match v path")
            return
        (_, reshape_v, add_v, matmul_v) = v_nodes

        qk_nodes = self.model.match_parent_path(matmul_qkv, ["Softmax", "Add", "Div", "MatMul"], [0, 0, None, 0])
        if qk_nodes is None:
            logger.debug("fuse_attention: failed to match qk path")
            return

        (_softmax_qk, add_bias_qk, _div_sqrt_head_size, matmul_qk) = qk_nodes

        q_nodes = self.model.match_parent_path(matmul_qk, ["Transpose", "Reshape", "Add", "MatMul"], [0, 0, 0, None])
        if q_nodes is None:
            logger.debug("fuse_attention: failed to match q path")
            return
        (_transpose_q, reshape_q, add_q, matmul_q) = q_nodes

        k_nodes = self.model.match_parent_path(matmul_qk, ["Transpose", "Reshape", "MatMul"], [1, 0, 0])
        if k_nodes is None:
            logger.debug("fuse_attention: failed to match k path")
            return

        (_transpose_k, _reshape_k, matmul_k) = k_nodes
        if matmul_q.input[0] != root_input or matmul_k.input[0] != root_input or matmul_v.input[0] != root_input:
            logger.debug("fuse_attention: expect to have same input to q, k and v matmul")
            return

        num_heads, hidden_size = self.get_num_heads_and_hidden_size(reshape_q)
        if num_heads <= 0:
            logger.debug("fuse_attention: failed to detect num_heads")
            return

        attention_last_node = reshape_qkv

        # number of heads are same for all the paths, hence to create attention node, we pass the q_num_heads
        new_node = self.create_attention_node(
            mask_index=None,
            q_matmul=matmul_q,
            k_matmul=matmul_k,
            v_matmul=matmul_v,
            q_add=add_q,
            k_add=None,
            v_add=add_v,
            num_heads=num_heads,
            hidden_size=hidden_size,
            input=root_input,
            output=attention_last_node.output[0],
            add_qk_str=add_bias_qk.input[1],
            scale=None,
        )
        if new_node is None:
            return

        self.nodes_to_add.append(new_node)
        self.node_name_to_graph_name[new_node.name] = self.this_graph_name

        self.nodes_to_remove.extend([attention_last_node, transpose_qkv])

        # Use prune graph to remove nodes since they are shared by all attention nodes.
        self.prune_graph = True
