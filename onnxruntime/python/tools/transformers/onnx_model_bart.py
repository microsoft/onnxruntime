#-------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.
#--------------------------------------------------------------------------
import logging
import onnx
import sys
import argparse
import numpy as np
from typing import Tuple, Union
from collections import deque
from onnx import helper, ModelProto, NodeProto, TensorProto, numpy_helper
from onnx_model import OnnxModel
from onnx_model_bert import BertOnnxModel
from fusion_attention import FusionAttention, AttentionMask, AttentionMaskFormat
from fusion_utils import FusionUtils, NumpyHelper

logger = logging.getLogger(__name__)

class FusionBartEncoderAttention(FusionAttention):
    """
    Fuse Bart Attention subgraph into one Attention node.
    """
    def __init__(self, model: OnnxModel, hidden_size: int, num_heads: int, attention_mask: AttentionMask):
        super().__init__(model, hidden_size, num_heads, attention_mask)

    def fuse(self, normalize_node, input_name_to_nodes, output_name_to_node):
        # Sometimes we can not fuse skiplayernormalization since the add before layernorm has an output that used by nodes outside skiplayernorm
        # Conceptually we treat add before layernorm as skiplayernorm node since they share the same pattern
        start_node = normalize_node
        if normalize_node.op_type == 'LayerNormalization':
            add_before_layernorm = self.model.match_parent(normalize_node, 'Add', 0)
            if add_before_layernorm is not None:
                start_node = add_before_layernorm
            else:
                return

        # SkipLayerNormalization has two inputs, and one of them is the root input for attention.
        qkv_nodes = self.model.match_parent_path(start_node, ['Add', 'MatMul', 'Reshape', 'Transpose', 'Reshape', 'MatMul'],
                                                 [None, 1, 0, 0, 0, 0])
        if qkv_nodes is not None:
            (add_out, matmul_out, reshape_qkv_2, transpose_qkv, reshape_qkv_1, matmul_qkv) = qkv_nodes
        else:
            return

        other_inputs = []
        for i, input in enumerate(start_node.input):
            if input not in output_name_to_node:
                continue
            if input == qkv_nodes[0].output[0]:
                continue
            other_inputs.append(input)
        if len(other_inputs) != 1:
            return

        root_input = other_inputs[0]
        children = input_name_to_nodes[root_input]
        children_types = [child.op_type for child in children]
        if children_types.count('MatMul') != 3:
            return

        v_nodes = self.model.match_parent_path(matmul_qkv, ['Reshape', 'Transpose', 'Reshape', 'Add', 'MatMul'], [1, 0, 0, 0, None])
        if v_nodes is None:
            logger.debug("fuse_attention: failed to match v path")
            return
        (reshape_v_2, transpose_v, reshape_v_1, add_v, matmul_v) = v_nodes
       
        qk_nodes = self.model.match_parent_path(matmul_qkv, ['Softmax', 'MatMul'], [0, 0])
        if qk_nodes is not None:
            _, matmul_qk = qk_nodes
        else:
            return

        q_nodes = self.model.match_parent_path(matmul_qk, ['Reshape', 'Transpose', 'Reshape', 'Mul', 'Add', 'MatMul'], [0, 0, 0, 0, 0, 1])
        if q_nodes is not None:
            reshape_q_2, _, reshape_q_1, _, add_q, matmul_q = q_nodes
        else:
            return

        k_nodes = self.model.match_parent_path(matmul_qk, ['Transpose', 'Reshape', 'Transpose', 'Reshape', 'Add', 'MatMul'], [1, 0, 0, 0, 0, 1])
        if k_nodes is not None:
            _, reshape_k_2, _, reshape_k_1, add_k, matmul_k = k_nodes
        else:
            return

        if matmul_v.input[0] == root_input and matmul_q.input[0] == root_input and matmul_v.input[0] == root_input:

            mask_nodes = []
            mask_index = None
            attention_last_node = reshape_qkv_2

            num_heads, hidden_size = self.get_num_heads_and_hidden_size(reshape_q_1)

            if num_heads <= 0 or hidden_size <= 0 or (hidden_size % num_heads) != 0:
                logger.debug("fuse_attention: failed to detect num_heads or hidden_size")
                return

            new_node = self.create_attention_node(mask_index, matmul_q, matmul_k, matmul_v, add_q, add_k, add_v,
                                                  num_heads, hidden_size, root_input, attention_last_node.output[0], None)
            if new_node is None:
                return

            self.nodes_to_add.append(new_node)
            self.node_name_to_graph_name[new_node.name] = self.this_graph_name

            self.nodes_to_remove.extend([attention_last_node, transpose_qkv, matmul_qkv])
            self.nodes_to_remove.extend(qk_nodes)
            self.nodes_to_remove.extend(q_nodes)
            self.nodes_to_remove.extend(k_nodes)
            self.nodes_to_remove.extend(v_nodes)

            # Use prune graph to remove mask nodes since they are shared by all attention nodes.
            self.nodes_to_remove.extend(mask_nodes)
            self.prune_graph = True

class BartOnnxModel(BertOnnxModel):
    def __init__(self, model, num_heads, hidden_size):
        super().__init__(model, num_heads, hidden_size)
        self.attention_mask = AttentionMask(self)
        self.attention_fusion = FusionBartEncoderAttention(self, self.hidden_size, self.num_heads, self.attention_mask)

    def fuse_attention(self):
        self.attention_fusion.apply()

    def get_num_heads_and_hidden_size(self, node: NodeProto) -> Tuple[int, int]:

        reshape_nodes = self.model.match_parent_path(node, ['Concat'], [1])
        if reshape_nodes is not None:
            concat_q = reshape_nodes[0]
        else:
            return self.num_heads, self.hidden_size

        num_heads_proto = self.model.get_initializer(concat_q.input[2])
        head_size_proto = self.model.get_initializer(concat_q.input[3])

        if num_heads_proto is None or head_size_proto is None:
            return self.num_heads, self.hidden_size
            
        num_heads = NumpyHelper.to_array(num_heads_proto)[0]
        head_size = NumpyHelper.to_array(head_size_proto)[0]
        hidden_size = num_heads * head_size

        if self.num_heads > 0 and num_heads != self.num_heads:
            logger.warn(f"--num_heads is {self.num_heads}. Detected value is {num_heads}. Using detected value.")

        if self.hidden_size > 0 and hidden_size != self.hidden_size:
            logger.warn(f"--hidden_size is {self.hidden_size}. Detected value is {hidden_size}. Using detected value.")

        return num_heads, hidden_size       


