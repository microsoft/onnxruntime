# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import logging
from typing import Union

from fusion_attention import AttentionMask
from onnx import NodeProto, TensorProto, helper, numpy_helper
from onnx_model import OnnxModel
from onnx_model_bert import BertOnnxModel
from fusion_base import Fusion
from fusion_utils import FusionUtils
import numpy as np
logger = logging.getLogger(__name__)


class FusionTNLGV4Attention(Fusion):
    """
    Fuse GPT-2 Attention with past state subgraph into one Attention node.
    """

    def __init__(self, model: OnnxModel, num_heads: int):
        super().__init__(model, "Attention", ["MatMul"])
        self.num_heads = num_heads

    def create_attention_node(
        self,
        fc_weight,
        fc_bias,
        input,
        output,
        mask,
        past = None,
        present_key = None,
        present_value = None,
    ):
        attention_node_name = self.model.create_node_name("GptNeoXAttention")
        attention_node = helper.make_node(
            "Attention",
            inputs=[input, fc_weight, fc_bias, mask],
            outputs=[output, present_key, present_value],
            name=attention_node_name,
        )
        attention_node.domain = "com.microsoft"
        attention_node.attribute.extend(
            [
                helper.make_attribute("num_heads", self.num_heads),
                helper.make_attribute("unidirectional", 1),
                helper.make_attribute("do_rotary", 1),
                helper.make_attribute("scale", 0.0078125),
            ]
        )

        # attention_node.attribute.extend([helper.make_attribute("mask_filter_value", float(self.mask_filter_value))])

        self.nodes_to_add.extend([attention_node])
        self.node_name_to_graph_name[attention_node.name] = self.this_graph_name

    def fuse(self, node, input_name_to_nodes, output_name_to_node):
        # note: this is not an exact match. experiment purpose only.
        stem_nodes = self.model.match_parent_path(
            node,
            ['Reshape', 'Transpose', 'Reshape', 'MatMul', 'Transpose', 'Reshape', 'Split', 'Reshape', 'Add', 'MatMul']
        )
        if stem_nodes is None:
            return

        kv_matmul_node = stem_nodes[3]
        split_node = stem_nodes[-4]

        attn_mask_nodes = self.model.match_parent_path(
            kv_matmul_node,
            ['Reshape', 'Softmax', 'Where', 'Slice', 'Slice']
        )
        if attn_mask_nodes is None:
            return

        where_node = attn_mask_nodes[2]
        present_nodes = self.model.match_parent_path(
            where_node,
            ['Reshape', 'Concat', 'Unsqueeze', 'Gather', 'Shape', 'Add'],
            [2, 1, 3, 0, 0, 0]
        )
        if present_nodes is None:
            return

        present_key = present_nodes[-1].output[0]
        print('present_key:', present_key)
        present_value = split_node.output[2]

        input = stem_nodes[-1].input[0]
        attn_mask = attn_mask_nodes[-1].input[0]
        fc_weights = stem_nodes[-1].input[1]
        fc_bias = stem_nodes[-2].input[0]
        output = node.input[0]

        self.create_attention_node(fc_weights, fc_bias, input, output, attn_mask, None, present_key, present_value)

        self.nodes_to_remove.extend(stem_nodes)
        self.nodes_to_remove.extend(attn_mask_nodes)
        self.nodes_to_remove.extend(present_nodes)

        add_node = present_nodes[-1]
        add_node.output[0] = "null"
        mul_0_nodes = self.model.match_parent_path(
            add_node,
            ['Mul'],
            [0]
        )
        if mul_0_nodes is None:
            return
        mul_1_nodes = self.model.match_parent_path(
            add_node,
            ['Mul'],
            [1]
        )
        if mul_1_nodes is None:
            return
        mul_0_node = mul_0_nodes[0]
        mul_1_node = mul_1_nodes[0]
        cos_path_nodes = self.model.match_parent_path(
            mul_0_node,
            ['Slice', 'Reshape', 'Cos', 'Concat', 'Einsum', 'Range', 'Cast', 'Gather', 'Shape']
        )
        if cos_path_nodes is None:
            return
        shape_path_nodes = self.model.match_parent_path(
            mul_1_node,
            ['Slice', 'Unsqueeze', 'Gather', 'Shape']
        )
        if shape_path_nodes is None:
            return

        self.nodes_to_remove.extend(mul_0_nodes)
        self.nodes_to_remove.extend(mul_1_nodes)
        self.nodes_to_remove.extend(cos_path_nodes)
        self.nodes_to_remove.extend(shape_path_nodes)

        # todo_1: append transpose before and after the attention node
        # todo_2: support past and present(combine key and value)
        self.prune_graph = True
        #print("create attention node")


class Tnlgv4OnnxModel(BertOnnxModel):
    def __init__(self, model, num_heads, hidden_size):
        super().__init__(model, num_heads, hidden_size)
        self.attention_mask = AttentionMask(self)
        self.attention_fusion = FusionTNLGV4Attention(self, self.num_heads)

    def fuse_attention(self):
        self.attention_fusion.apply()
